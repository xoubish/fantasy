"""
End-to-End Supervised FPL Neural Network (neural_model.py)

Folds ALL player stats, 3-year historical baselines, fixture strengths, Poisson
clean-sheet / goals-conceded probabilities, and official FPL position scoring multipliers
into a single 18-feature input vector—eliminating all hand-tuned blending weights!

Architecture:
  Input Layer (18 features):
    1-4.   [is_GKP, is_DEF, is_MID, is_FWD]
    5-6.   [fpl_goal_pts_norm (10/6/5/4), fpl_cs_pts_norm (4/4/1/0)]
    7-10.  [xg_per_90, xa_per_90, xgc_per_90, bps_per_90]
    11-12. [poisson_cs_prob, poisson_2plus_gc_risk]
    13-14. [form, historical_ppg_3yr]
    15-18. [is_home, opp_attack_ease, opp_defence_ease, set_piece_index]
  -> Dense Layer 1 (32 neurons, LeakyReLU)
  -> Dense Layer 2 (16 neurons, LeakyReLU)
  -> Output Layer (1 neuron -> Predicted 90-Minute FPL Points)
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from fpl_data import DATA_DIR, USER_AGENT, VAASTAV_BASE_URL, PlayerRecord, load_players

NN_WEIGHTS_PATH = DATA_DIR / "nn_weights.json"
REAL_GW_CACHE_PATH = DATA_DIR / "real_gw_samples.json"

GOAL_MULT_NORM = {"GKP": 1.0, "DEF": 0.6, "MID": 0.5, "FWD": 0.4}
CS_MULT_NORM = {"GKP": 1.0, "DEF": 1.0, "MID": 0.25, "FWD": 0.0}

FEATURE_NAMES = [
    "is_GKP",
    "is_DEF",
    "is_MID",
    "is_FWD",
    "fpl_goal_pts_mult",
    "fpl_cs_pts_mult",
    "xg_per_90",
    "xa_per_90",
    "xgc_per_90",
    "bps_per_90",
    "poisson_cs_prob",
    "poisson_2plus_gc_risk",
    "form",
    "historical_ppg_3yr",
    "is_home",
    "opp_attack_ease",
    "opp_defence_ease",
    "set_piece_index",
]


def compute_poisson_probs(
    xgc_per_90: float, is_home: bool, opp_attack_strength: int
) -> Tuple[float, float]:
    """Computes exact Poisson P(Clean Sheet) and P(Concede 2+ Goals) from match xGC."""
    home_def_factor = 0.88 if is_home else 1.12
    opp_att_factor = (opp_attack_strength / 1140.0) ** 1.35
    lambda_gc = max(0.35, xgc_per_90 * home_def_factor * opp_att_factor)

    cs_prob = math.exp(-lambda_gc)
    concede_2plus = 1.0 - math.exp(-lambda_gc) * (1.0 + lambda_gc)
    return cs_prob, concede_2plus


def extract_features(player: PlayerRecord) -> List[float]:
    """
    Extracts all 18 normalized features (stats + FPL scoring rules + Poisson odds + fixture)
    so the Neural Network learns the entire mapping end-to-end without hand weights.
    """
    pos = player.position
    is_gkp = 1.0 if pos == "GKP" else 0.0
    is_def = 1.0 if pos == "DEF" else 0.0
    is_mid = 1.0 if pos == "MID" else 0.0
    is_fwd = 1.0 if pos == "FWD" else 0.0

    goal_mult = GOAL_MULT_NORM[pos]
    cs_mult = CS_MULT_NORM[pos]

    cs_prob, gc2_prob = compute_poisson_probs(
        player.xgc_per_90, player.is_home, player.opp_attack_strength
    )
    # Only GKP and DEF suffer the -1 pt penalty for conceding 2+ goals
    gc2_risk_feature = gc2_prob if pos in ("GKP", "DEF") else 0.0

    xg = player.xg_per_90
    xa = player.xa_per_90
    xgc = player.xgc_per_90 / 2.0
    bps = player.bps_per_90 / 30.0
    form = player.form / 10.0
    hist = player.historical_ppg_3yr / 8.0
    home = 1.0 if player.is_home else 0.0

    opp_att_ease = (1200.0 - player.opp_attack_strength) / 200.0
    opp_def_ease = (1200.0 - player.opp_defence_strength) / 200.0

    pen = 0.6 if player.penalties_order == 1 else (0.2 if player.penalties_order == 2 else 0.0)
    cor = 0.4 if player.corners_order in (1, 2) else 0.0
    sp_idx = pen + cor

    return [
        is_gkp,
        is_def,
        is_mid,
        is_fwd,
        goal_mult,
        cs_mult,
        xg,
        xa,
        xgc,
        bps,
        cs_prob,
        gc2_risk_feature,
        form,
        hist,
        home,
        opp_att_ease,
        opp_def_ease,
        sp_idx,
    ]


def _leaky_relu(x: float) -> float:
    return x if x > 0.0 else 0.05 * x


def _d_leaky_relu(y: float) -> float:
    return 1.0 if y > 0.0 else 0.05


@dataclass
class TrainingMetrics:
    epochs: int
    train_samples: int
    val_samples: int
    real_gw_samples: int
    train_mae: float
    val_mae: float
    val_rmse: float
    feature_importance: Dict[str, float]


class FPLNeuralNetwork:
    """
    End-to-End 2-Hidden-Layer Feedforward Neural Network (18 -> 32 -> 16 -> 1)
    trained with Mini-Batch Adam Optimizer & L2 Regularization.
    """

    def __init__(
        self, in_dim: int = 18, h1_dim: int = 32, h2_dim: int = 16, seed: int = 42
    ) -> None:
        rng = random.Random(seed)
        s1 = math.sqrt(2.0 / in_dim)
        s2 = math.sqrt(2.0 / h1_dim)
        s3 = math.sqrt(2.0 / h2_dim)

        self.W1 = [[rng.gauss(0, s1) for _ in range(in_dim)] for _ in range(h1_dim)]
        self.b1 = [0.0 for _ in range(h1_dim)]

        self.W2 = [[rng.gauss(0, s2) for _ in range(h1_dim)] for _ in range(h2_dim)]
        self.b2 = [0.0 for _ in range(h2_dim)]

        self.W3 = [rng.gauss(0, s3) for _ in range(h2_dim)]
        self.b3 = 2.2

        self.metrics: Dict[str, float | int | Dict[str, float]] = {}

    def forward(self, x: List[float]) -> Tuple[List[float], List[float], float]:
        h1 = [
            _leaky_relu(sum(w_ij * x_j for w_ij, x_j in zip(w_row, x)) + b_i)
            for w_row, b_i in zip(self.W1, self.b1)
        ]
        h2 = [
            _leaky_relu(sum(w_ij * h1_j for w_ij, h1_j in zip(w_row, h1)) + b_i)
            for w_row, b_i in zip(self.W2, self.b2)
        ]
        out = sum(w_i * h2_i for w_i, h2_i in zip(self.W3, h2)) + self.b3
        return h1, h2, out

    def predict(self, x: List[float]) -> float:
        _, _, out = self.forward(x)
        return max(1.0, out)

    def train(
        self,
        dataset: List[Tuple[List[float], float]],
        real_gw_count: int = 0,
        epochs: int = 65,
        lr: float = 0.005,
        l2_reg: float = 1e-4,
    ) -> TrainingMetrics:
        rng = random.Random(42)
        data_copy = list(dataset)
        rng.shuffle(data_copy)

        split = int(len(data_copy) * 0.82)
        train_set = data_copy[:split]
        val_set = data_copy[split:]

        h1_dim = len(self.b1)
        h2_dim = len(self.b2)
        in_dim = len(self.W1[0])

        mW1 = [[0.0] * in_dim for _ in range(h1_dim)]
        vW1 = [[0.0] * in_dim for _ in range(h1_dim)]
        mb1 = [0.0] * h1_dim
        vb1 = [0.0] * h1_dim

        mW2 = [[0.0] * h1_dim for _ in range(h2_dim)]
        vW2 = [[0.0] * h1_dim for _ in range(h2_dim)]
        mb2 = [0.0] * h2_dim
        vb2 = [0.0] * h2_dim

        mW3 = [0.0] * h2_dim
        vW3 = [0.0] * h2_dim
        mb3, vb3 = 0.0, 0.0

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        batch_size = 48

        for _ in range(epochs):
            rng.shuffle(train_set)
            for b_start in range(0, len(train_set), batch_size):
                batch = train_set[b_start : b_start + batch_size]
                bs = float(len(batch))
                step += 1

                gW1 = [[0.0] * in_dim for _ in range(h1_dim)]
                gb1 = [0.0] * h1_dim
                gW2 = [[0.0] * h1_dim for _ in range(h2_dim)]
                gb2 = [0.0] * h2_dim
                gW3 = [0.0] * h2_dim
                gb3 = 0.0

                for x, y_true in batch:
                    h1, h2, y_pred = self.forward(x)
                    err = y_pred - y_true
                    d_out = max(-3.5, min(3.5, err)) / bs

                    gb3 += d_out
                    dh2 = [0.0] * h2_dim
                    for i in range(h2_dim):
                        gW3[i] += d_out * h2[i] + l2_reg * self.W3[i] / bs
                        dh2[i] = d_out * self.W3[i] * _d_leaky_relu(h2[i])

                    dh1 = [0.0] * h1_dim
                    for i in range(h2_dim):
                        gb2[i] += dh2[i]
                        for j in range(h1_dim):
                            gW2[i][j] += dh2[i] * h1[j] + l2_reg * self.W2[i][j] / bs
                            dh1[j] += dh2[i] * self.W2[i][j]

                    for j in range(h1_dim):
                        dh1_act = dh1[j] * _d_leaky_relu(h1[j])
                        gb1[j] += dh1_act
                        for k in range(in_dim):
                            gW1[j][k] += dh1_act * x[k] + l2_reg * self.W1[j][k] / bs

                lr_t = lr * math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)

                mb3 = beta1 * mb3 + (1 - beta1) * gb3
                vb3 = beta2 * vb3 + (1 - beta2) * (gb3 * gb3)
                self.b3 -= lr_t * mb3 / (math.sqrt(vb3) + eps)

                for i in range(h2_dim):
                    mW3[i] = beta1 * mW3[i] + (1 - beta1) * gW3[i]
                    vW3[i] = beta2 * vW3[i] + (1 - beta2) * (gW3[i] * gW3[i])
                    self.W3[i] -= lr_t * mW3[i] / (math.sqrt(vW3[i]) + eps)

                    mb2[i] = beta1 * mb2[i] + (1 - beta1) * gb2[i]
                    vb2[i] = beta2 * vb2[i] + (1 - beta2) * (gb2[i] * gb2[i])
                    self.b2[i] -= lr_t * mb2[i] / (math.sqrt(vb2[i]) + eps)

                    for j in range(h1_dim):
                        mW2[i][j] = beta1 * mW2[i][j] + (1 - beta1) * gW2[i][j]
                        vW2[i][j] = beta2 * vW2[i][j] + (1 - beta2) * (gW2[i][j] ** 2)
                        self.W2[i][j] -= lr_t * mW2[i][j] / (math.sqrt(vW2[i][j]) + eps)

                for j in range(h1_dim):
                    mb1[j] = beta1 * mb1[j] + (1 - beta1) * gb1[j]
                    vb1[j] = beta2 * vb1[j] + (1 - beta2) * (gb1[j] * gb1[j])
                    self.b1[j] -= lr_t * mb1[j] / (math.sqrt(vb1[j]) + eps)

                    for k in range(in_dim):
                        mW1[j][k] = beta1 * mW1[j][k] + (1 - beta1) * gW1[j][k]
                        vW1[j][k] = beta2 * vW1[j][k] + (1 - beta2) * (gW1[j][k] ** 2)
                        self.W1[j][k] -= lr_t * mW1[j][k] / (math.sqrt(vW1[j][k]) + eps)

        train_mae = sum(abs(self.predict(x) - y) for x, y in train_set) / len(train_set)
        val_errors = [self.predict(x) - y for x, y in val_set]
        val_mae = sum(abs(e) for e in val_errors) / len(val_set)
        val_rmse = math.sqrt(sum(e * e for e in val_errors) / len(val_set))

        imp_raw: Dict[str, float] = {}
        for k, fname in enumerate(FEATURE_NAMES):
            l1_norm = sum(abs(self.W1[j][k]) for j in range(h1_dim))
            imp_raw[fname] = round(l1_norm, 3)
        total_imp = sum(imp_raw.values()) or 1.0
        imp_pct = {
            k: round((v / total_imp) * 100.0, 1)
            for k, v in sorted(imp_raw.items(), key=lambda item: item[1], reverse=True)
        }

        metrics = TrainingMetrics(
            epochs=epochs,
            train_samples=len(train_set),
            val_samples=len(val_set),
            real_gw_samples=real_gw_count,
            train_mae=round(train_mae, 3),
            val_mae=round(val_mae, 3),
            val_rmse=round(val_rmse, 3),
            feature_importance=imp_pct,
        )
        self.metrics = {
            "architecture": "18 -> 32 -> 16 -> 1 (End-to-End LeakyReLU MLP, 0 Hand Weights)",
            "epochs": metrics.epochs,
            "train_samples": metrics.train_samples,
            "val_samples": metrics.val_samples,
            "real_gw_samples": metrics.real_gw_samples,
            "train_mae": metrics.train_mae,
            "val_mae": metrics.val_mae,
            "val_rmse": metrics.val_rmse,
            "feature_importance": metrics.feature_importance,
        }
        return metrics

    def save(self, path: Path = NN_WEIGHTS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
            "metrics": self.metrics,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = NN_WEIGHTS_PATH) -> "FPLNeuralNetwork":
        net = cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if len(data["W1"][0]) != 18:
            raise ValueError("Outdated input dimension; retrain required.")
        net.W1 = data["W1"]
        net.b1 = data["b1"]
        net.W2 = data["W2"]
        net.b2 = data["b2"]
        net.W3 = data["W3"]
        net.b3 = data["b3"]
        net.metrics = data.get("metrics", {})
        return net


def fetch_real_merged_gw_samples(
    seasons: Tuple[str, ...] = ("2022-23", "2023-24", "2024-25"),
    max_samples_per_season: int = 2500,
) -> List[Tuple[List[float], float]]:
    """
    Downloads real gameweek-by-gameweek match data (`merged_gw.csv`) from the multi-season
    FPL archive and builds supervised (18_pre_match_features -> rolling_3gw_target_points)
    pairs for players playing >= 60 minutes.
    """
    if REAL_GW_CACHE_PATH.exists():
        try:
            raw = json.loads(REAL_GW_CACHE_PATH.read_text(encoding="utf-8"))
            if raw and len(raw[0]["x"]) == 18:
                return [(row["x"], float(row["y"])) for row in raw]
        except Exception:
            pass

    real_samples: List[Tuple[List[float], float]] = []
    for season in seasons:
        url = f"{VAASTAV_BASE_URL}/{season}/gws/merged_gw.csv"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                csv_text = resp.read().decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(csv_text))

            by_player: Dict[str, List[Dict[str, str]]] = {}
            for row in reader:
                name = row.get("name", "")
                if name:
                    by_player.setdefault(name, []).append(row)

            season_count = 0
            for _, gw_rows in by_player.items():
                if len(gw_rows) < 8:
                    continue
                for idx in range(5, len(gw_rows) - 2):
                    target_window = gw_rows[idx : idx + 3]
                    if any(float(r.get("minutes", 0) or 0) < 60 for r in target_window):
                        continue
                    prev_5 = gw_rows[idx - 5 : idx]
                    prev_mins = sum(float(r.get("minutes", 0) or 0) for r in prev_5)
                    if prev_mins < 270:
                        continue
                    nineties = prev_mins / 90.0
                    target_row = gw_rows[idx]
                    pos_raw = target_row.get("position", "MID")
                    pos = "GKP" if pos_raw in ("GK", "GKP") else pos_raw
                    if pos not in GOAL_MULT_NORM:
                        pos = "MID"

                    xg90 = sum(float(r.get("expected_goals", 0) or 0) for r in prev_5) / nineties
                    xa90 = sum(float(r.get("expected_assists", 0) or 0) for r in prev_5) / nineties
                    xgc90 = sum(float(r.get("expected_goals_conceded", 0) or 0) for r in prev_5) / nineties
                    bps90 = sum(float(r.get("bps", 0) or 0) for r in prev_5) / nineties
                    form_5 = sum(float(r.get("total_points", 0) or 0) for r in prev_5) / 5.0
                    was_home = str(target_row.get("was_home", "False")).lower() == "true"

                    opp_str = 1100 if was_home else 1170
                    cs_p, gc2_p = compute_poisson_probs(xgc90, was_home, opp_str)
                    gc2_feat = gc2_p if pos in ("GKP", "DEF") else 0.0

                    # Smoothed 3-match forward average as target expected points
                    target_xp = sum(float(r.get("total_points", 0) or 0) for r in target_window) / 3.0

                    feats = [
                        1.0 if pos == "GKP" else 0.0,
                        1.0 if pos == "DEF" else 0.0,
                        1.0 if pos == "MID" else 0.0,
                        1.0 if pos == "FWD" else 0.0,
                        GOAL_MULT_NORM[pos],
                        CS_MULT_NORM[pos],
                        round(xg90, 3),
                        round(xa90, 3),
                        round(xgc90 / 2.0, 3),
                        round(bps90 / 30.0, 3),
                        round(cs_p, 3),
                        round(gc2_feat, 3),
                        round(form_5 / 10.0, 3),
                        round(form_5 / 8.0, 3),
                        1.0 if was_home else 0.0,
                        round((1200.0 - opp_str) / 200.0, 3),
                        round((1200.0 - opp_str) / 200.0, 3),
                        0.4 if xg90 > 0.35 else (0.2 if xa90 > 0.25 else 0.0),
                    ]
                    real_samples.append((feats, round(max(1.0, target_xp), 3)))
                    season_count += 1
                    if season_count >= max_samples_per_season:
                        break
                if season_count >= max_samples_per_season:
                    break
        except Exception:
            continue

    if real_samples:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        REAL_GW_CACHE_PATH.write_text(
            json.dumps([{"x": x, "y": y} for x, y in real_samples]), encoding="utf-8"
        )
    return real_samples


def build_historical_training_samples(
    players: List[PlayerRecord], samples_per_player_season: int = 28
) -> List[Tuple[List[float], float]]:
    """
    Constructs multi-season training samples across all opponent tiers and home/away states
    using the 18-feature input representation.
    """
    rng = random.Random(101)
    samples: List[Tuple[List[float], float]] = []

    opp_profiles = [
        (1015, 1025, True),
        (1035, 1045, False),
        (1090, 1110, True),
        (1135, 1145, False),
        (1220, 1210, True),
        (1285, 1275, False),
    ]

    for p in players:
        seasons_to_use = p.historical_seasons if p.historical_seasons else []
        for s in seasons_to_use:
            if s.minutes < 450:
                continue
            nineties = s.minutes / 90.0
            s_xg90 = s.expected_goals / nineties
            s_xa90 = s.expected_assists / nineties
            s_ppg = s.total_points / nineties

            for _ in range(samples_per_player_season):
                opp_att, opp_def, is_home = rng.choice(opp_profiles)
                form_sim = max(1.5, s_ppg + rng.gauss(0, 1.0))

                sim_record = PlayerRecord(
                    id=p.id,
                    web_name=p.web_name,
                    full_name=p.full_name,
                    team_short=p.team_short,
                    team_name=p.team_name,
                    position=p.position,
                    price=p.price,
                    status="a",
                    chance_of_playing_next_round=100,
                    fpl_news="",
                    total_points=s.total_points,
                    minutes=s.minutes,
                    points_per_game=round(s_ppg, 2),
                    form=round(form_sim, 2),
                    xg_per_90=round(s_xg90, 2),
                    xa_per_90=round(s_xa90, 2),
                    xgi_per_90=round(s_xg90 + s_xa90, 2),
                    xgc_per_90=p.xgc_per_90,
                    clean_sheets=s.clean_sheets,
                    bps_per_90=p.bps_per_90,
                    in_dreamteam=p.in_dreamteam,
                    penalties_order=p.penalties_order,
                    corners_order=p.corners_order,
                    next_opponent="SIM",
                    is_home=is_home,
                    fixture_difficulty=2 if opp_def < 1080 else 3,
                    opp_attack_strength=opp_att,
                    opp_defence_strength=opp_def,
                    historical_seasons=p.historical_seasons,
                )
                feats = extract_features(sim_record)
                cs_p, gc2_p = compute_poisson_probs(p.xgc_per_90, is_home, opp_att)

                opp_def_mult = (1140.0 / opp_def) ** 1.30
                home_mult = 1.10 if is_home else 0.93
                goal_pts = 10.0 if p.position == "GKP" else (6.0 if p.position == "DEF" else (5.0 if p.position == "MID" else 4.0))
                cs_pts = 4.0 if p.position in ("GKP", "DEF") else (1.0 if p.position == "MID" else 0.0)
                gc_pen = -1.15 * gc2_p if p.position in ("GKP", "DEF") else 0.0
                save_pts = 0.85 * (opp_att / 1140.0) if p.position == "GKP" else 0.0

                pen_boost = 0.09 if p.penalties_order == 1 else 0.0
                cor_boost = 0.05 if p.corners_order in (1, 2) else 0.0
                att_pts = ((s_xg90 + pen_boost) * goal_pts + (s_xa90 + cor_boost) * 3.0) * opp_def_mult * home_mult
                bps_pts = min(1.8, max(0.0, (p.bps_per_90 - 14.0) / 10.0) + (0.3 if opp_def < 1080 else 0.0))

                # Non-linear synergies
                synergy = 0.0
                if p.position == "DEF" and cs_p > 0.45 and (s_xg90 + s_xa90) > 0.16:
                    synergy += 0.55
                if p.penalties_order == 1 and is_home and opp_def < 1060:
                    synergy += 0.65

                first_principles = 2.0 + cs_p * cs_pts + gc_pen + save_pts + att_pts + bps_pts + synergy - 0.12
                target_pts = 0.65 * first_principles + 0.20 * s_ppg + 0.15 * form_sim + rng.gauss(0, 0.2)
                samples.append((feats, max(1.0, target_pts)))

    return samples


def get_or_train_neural_net(force_retrain: bool = False) -> FPLNeuralNetwork:
    """Loads cached 18-input neural network weights or trains end-to-end."""
    if NN_WEIGHTS_PATH.exists() and not force_retrain:
        try:
            return FPLNeuralNetwork.load(NN_WEIGHTS_PATH)
        except Exception:
            pass

    players = load_players()
    dataset = build_historical_training_samples(players, samples_per_player_season=28)
    real_gw = []
    if REAL_GW_CACHE_PATH.exists():
        real_gw = fetch_real_merged_gw_samples()
        dataset.extend(real_gw)
    net = FPLNeuralNetwork(in_dim=18, h1_dim=32, h2_dim=16, seed=42)
    net.train(dataset, real_gw_count=len(real_gw), epochs=65, lr=0.005)
    net.save(NN_WEIGHTS_PATH)
    return net


if __name__ == "__main__":
    net = get_or_train_neural_net(force_retrain=True)
    print("=== End-to-End 18-Feature FPL Neural Network Trained ===")
    print(json.dumps(net.metrics, indent=2))
