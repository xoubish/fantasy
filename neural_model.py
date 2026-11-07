"""
Time-Aware End-to-End Supervised FPL Neural Network (neural_model.py)

Trained exclusively on active 2026/27 Premier League players (`data/bootstrap_static.json`)
and their real match-by-match gameweek logs (`data/player_summaries.json`).

Why Time & Recency Matter (And Why 2-Year-Old History Is Dropped):
  1. Current FPL Price (`price_tier`) ALREADY encodes how good a player was 1–2 years ago.
     Feeding multi-year historical stats into the network double-counts old reputation and
     overvalues expensive players who were good 2 years ago but are out of form now.
  2. Within the current season, every player stat (`ewma_ppg`, `ewma_xg_per_90`, `ewma_xa_per_90`,
     `ewma_def_contrib_per_90`, `ewma_ict_per_90`, `ewma_bps_per_90`, `recent_minutes_security`)
     is computed using an Exponentially Weighted Moving Average (EWMA, decay = 0.78 per GW back)
     plus explicit `form_momentum` (recent EWMA minus season average).
  3. During backpropagation, each training match from Gameweek `g` is weighted by
     `sample_weight = 0.85 ** (latest_gw - g)` so last week's games drive gradient updates
     far more strongly than early-August matches.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fpl_data import (
    CACHE_BOOTSTRAP,
    CACHE_FIXTURES,
    DATA_DIR,
    PlayerRecord,
    compute_team_strengths,
    load_players,
)

NN_WEIGHTS_PATH = DATA_DIR / "nn_weights.json"

GOAL_MULT_NORM = {"GKP": 1.0, "DEF": 0.6, "MID": 0.5, "FWD": 0.4}
CS_MULT_NORM = {"GKP": 1.0, "DEF": 1.0, "MID": 0.25, "FWD": 0.0}

FEATURE_NAMES = [
    "is_GKP",
    "is_DEF",
    "is_MID",
    "is_FWD",
    "fpl_goal_pts_mult",
    "fpl_cs_pts_mult",
    "ewma_recent_ppg",
    "form_momentum_trend",
    "ewma_xg_per_90",
    "ewma_xa_per_90",
    "ewma_def_contrib_per_90",
    "ewma_ict_per_90",
    "ewma_bps_per_90",
    "recent_minutes_security",
    "is_home",
    "attack_vs_weaker_def",
    "defence_vs_weaker_att",
    "poisson_cs_prob",
    "poisson_2plus_gc_risk",
    "current_price_tier",
    "current_value_per_m",
    "set_piece_duty",
]


def compute_poisson_probs(
    xgc_per_90: float,
    is_home: bool,
    team_def_strength: int,
    opp_att_strength: int,
) -> Tuple[float, float]:
    """
    Computes exact Poisson P(Clean Sheet) and P(Concede 2+ Goals) incorporating:
    - Player's team time-decayed xGC/90
    - Home vs Away factor
    - Opponent attack strength relative to player's team defence strength
    """
    home_def_factor = 0.88 if is_home else 1.12
    mismatch_ratio = (opp_att_strength / max(team_def_strength, 950)) ** 1.35
    base_xgc = max(0.55, min(2.2, xgc_per_90 if xgc_per_90 > 0.1 else 1.25))
    lambda_gc = max(0.35, base_xgc * home_def_factor * mismatch_ratio)

    cs_prob = math.exp(-lambda_gc)
    concede_2plus = 1.0 - math.exp(-lambda_gc) * (1.0 + lambda_gc)
    return cs_prob, concede_2plus


def compute_recency_weighted_metrics(
    gw_rows: List[Dict[str, Any]],
    fallback_player: PlayerRecord,
    decay: float = 0.78,
) -> Dict[str, float]:
    """
    Computes Exponentially Weighted Moving Average (EWMA) per-90 performance stats
    from a player's completed 2026/27 match logs (`gw_rows`).
    Most recent GW has weight 1.0, k weeks ago has weight `decay ** k`.
    """
    completed = [
        r for r in gw_rows if r.get("team_h_score") is not None
    ]
    if not completed:
        return {
            "ewma_ppg": fallback_player.points_per_game,
            "momentum": (fallback_player.form - fallback_player.points_per_game),
            "ewma_xg90": fallback_player.xg_per_90,
            "ewma_xa90": fallback_player.xa_per_90,
            "ewma_xgc90": fallback_player.xgc_per_90,
            "ewma_def90": fallback_player.defensive_contribution_per_90,
            "ewma_ict90": fallback_player.ict_per_90,
            "ewma_bps90": fallback_player.bps_per_90,
            "recent_mins_sec": min(1.0, fallback_player.minutes / 720.0),
        }

    n = len(completed)
    w_pts = 0.0
    w_xg = 0.0
    w_xa = 0.0
    w_xgc = 0.0
    w_def = 0.0
    w_ict = 0.0
    w_bps = 0.0
    w_90s = 0.0

    raw_pts = 0.0
    raw_90s = 0.0

    # Also track recent 4 matches minutes security with exponential decay
    recent_slice = completed[-4:]
    rec_mins_w = 0.0
    rec_weight_sum = 0.0
    for idx, r in enumerate(recent_slice):
        age = len(recent_slice) - 1 - idx
        w = 0.75 ** age
        mins = float(r.get("minutes", 0) or 0)
        rec_mins_w += w * min(1.0, mins / 85.0)
        rec_weight_sum += w

    for idx, r in enumerate(completed):
        mins = float(r.get("minutes", 0) or 0)
        if mins <= 0:
            continue
        age = n - 1 - idx  # 0 for the most recent match, 1 for the one before, etc.
        w = decay ** age
        m90 = mins / 90.0

        pts = float(r.get("total_points", 0) or 0)
        w_pts += w * pts
        w_xg += w * float(r.get("expected_goals", 0) or 0)
        w_xa += w * float(r.get("expected_assists", 0) or 0)
        w_xgc += w * float(r.get("expected_goals_conceded", 0) or 0)
        w_def += w * float(r.get("defensive_contribution", 0) or 0)
        w_ict += w * float(r.get("ict_index", 0) or 0)
        w_bps += w * float(r.get("bps", 0) or 0)
        w_90s += w * m90

        raw_pts += pts
        raw_90s += m90

    if w_90s < 0.35:
        return {
            "ewma_ppg": fallback_player.points_per_game,
            "momentum": 0.0,
            "ewma_xg90": fallback_player.xg_per_90,
            "ewma_xa90": fallback_player.xa_per_90,
            "ewma_xgc90": fallback_player.xgc_per_90,
            "ewma_def90": fallback_player.defensive_contribution_per_90,
            "ewma_ict90": fallback_player.ict_per_90,
            "ewma_bps90": fallback_player.bps_per_90,
            "recent_mins_sec": rec_mins_w / max(rec_weight_sum, 1.0),
        }

    ewma_ppg = w_pts / w_90s
    season_ppg = raw_pts / max(raw_90s, 0.5)
    # Positive momentum = heating up in recent weeks vs earlier in the season
    momentum = ewma_ppg - season_ppg

    return {
        "ewma_ppg": ewma_ppg,
        "momentum": momentum,
        "ewma_xg90": w_xg / w_90s,
        "ewma_xa90": w_xa / w_90s,
        "ewma_xgc90": w_xgc / w_90s,
        "ewma_def90": w_def / w_90s,
        "ewma_ict90": w_ict / w_90s,
        "ewma_bps90": w_bps / w_90s,
        "recent_mins_sec": rec_mins_w / max(rec_weight_sum, 1.0),
    }


def extract_features(player: PlayerRecord) -> List[float]:
    """
    Extracts all 22 time-decayed features for a real Premier League player and their
    upcoming fixture opponent. No 2-year-old stats are used because `current_price_tier`
    already prices in past-seasons reputation!
    """
    pos = player.position
    is_gkp = 1.0 if pos == "GKP" else 0.0
    is_def = 1.0 if pos == "DEF" else 0.0
    is_mid = 1.0 if pos == "MID" else 0.0
    is_fwd = 1.0 if pos == "FWD" else 0.0

    goal_mult = GOAL_MULT_NORM[pos]
    cs_mult = CS_MULT_NORM[pos]

    rec = compute_recency_weighted_metrics(player.gw_history, player, decay=0.78)

    cs_prob, gc2_prob = compute_poisson_probs(
        rec["ewma_xgc90"],
        player.is_home,
        player.team_defence_strength,
        player.opp_attack_strength,
    )
    gc2_risk_feat = gc2_prob if pos in ("GKP", "DEF") else 0.0

    ewma_ppg_norm = rec["ewma_ppg"] / 8.0
    momentum_norm = max(-0.6, min(0.6, rec["momentum"] / 5.0))
    xg = rec["ewma_xg90"]
    xa = rec["ewma_xa90"]
    def_contrib = rec["ewma_def90"] / 12.0
    ict = rec["ewma_ict90"] / 10.0
    bps = rec["ewma_bps90"] / 30.0
    mins_sec = rec["recent_mins_sec"]

    home = 1.0 if player.is_home else 0.0
    att_vs_weak_def = (player.team_attack_strength - player.opp_defence_strength) / 200.0
    def_vs_weak_att = (player.team_defence_strength - player.opp_attack_strength) / 200.0

    # Current price already encodes multi-season reputation; value_per_m measures current ROI!
    price_norm = player.price / 15.0
    current_val_per_m = (rec["ewma_ppg"] / max(player.price, 4.0))

    pen = 0.55 if player.penalties_order == 1 else (0.20 if player.penalties_order == 2 else 0.0)
    cor = 0.30 if player.corners_order in (1, 2) else 0.0
    fk = 0.15 if player.direct_freekicks_order == 1 else 0.0
    set_piece_duty = pen + cor + fk

    return [
        is_gkp,
        is_def,
        is_mid,
        is_fwd,
        goal_mult,
        cs_mult,
        ewma_ppg_norm,
        momentum_norm,
        xg,
        xa,
        def_contrib,
        ict,
        bps,
        mins_sec,
        home,
        att_vs_weak_def,
        def_vs_weak_att,
        cs_prob,
        gc2_risk_feat,
        price_norm,
        current_val_per_m,
        set_piece_duty,
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
    latest_trained_gw: int
    train_mae: float
    val_mae: float
    val_rmse: float
    feature_importance: Dict[str, float]


class FPLNeuralNetwork:
    """
    Time-Aware End-to-End 2-Hidden-Layer Feedforward Neural Network (22 -> 36 -> 18 -> 1)
    trained on real Premier League player-gameweek match logs via Temporally-Weighted Mini-Batch Adam.
    """

    def __init__(
        self, in_dim: int = 22, h1_dim: int = 36, h2_dim: int = 18, seed: int = 42
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
        self.b3 = 2.4

        self.metrics: Dict[str, Any] = {}

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
        dataset: List[Tuple[List[float], float, float]],
        real_gw_count: int = 0,
        latest_trained_gw: int = 9,
        epochs: int = 70,
        lr: float = 0.005,
        l2_reg: float = 1.5e-4,
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
        batch_size = 32

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

                for x, y_true, sample_w in batch:
                    h1, h2, y_pred = self.forward(x)
                    err = y_pred - y_true
                    # Multiply gradient by temporal sample_w so recent GWs dominate learning
                    d_out = (max(-3.5, min(3.5, err)) * sample_w) / bs

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

        train_mae = sum(abs(self.predict(x) - y) for x, y, _ in train_set) / len(train_set)
        val_errors = [self.predict(x) - y for x, y, _ in val_set]
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
            latest_trained_gw=latest_trained_gw,
            train_mae=round(train_mae, 3),
            val_mae=round(val_mae, 3),
            val_rmse=round(val_rmse, 3),
            feature_importance=imp_pct,
        )
        self.metrics = {
            "architecture": "22 -> 36 -> 18 -> 1 (Time-Decayed EWMA + Temporal Adam)",
            "epochs": metrics.epochs,
            "train_samples": metrics.train_samples,
            "val_samples": metrics.val_samples,
            "real_gw_samples": metrics.real_gw_samples,
            "latest_trained_gw": metrics.latest_trained_gw,
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
        if len(data["W1"][0]) != 22 or "ewma_recent_ppg" not in data.get("metrics", {}).get("feature_importance", {}):
            raise ValueError("Outdated neural feature schema; retrain required.")
        net.W1 = data["W1"]
        net.b1 = data["b1"]
        net.W2 = data["W2"]
        net.b2 = data["b2"]
        net.W3 = data["W3"]
        net.b3 = data["b3"]
        net.metrics = data.get("metrics", {})
        return net


def build_real_premier_league_training_dataset(
    players: List[PlayerRecord],
) -> Tuple[List[Tuple[List[float], float, float]], int, int]:
    """
    Constructs time-decayed supervised training examples directly from the real 2026/27
    Premier League gameweek-by-gameweek match logs (`gw_history`).
    Returns `(samples_with_temporal_weight, real_gw_match_count, latest_finished_gw)`.
    """
    bootstrap = json.loads(CACHE_BOOTSTRAP.read_text(encoding="utf-8"))
    fixtures = json.loads(CACHE_FIXTURES.read_text(encoding="utf-8"))
    latest_finished_gw = max(
        (
            int(f.get("event", 0) or 0)
            for f in fixtures
            if f.get("finished") and f.get("team_h_score") is not None
        ),
        default=9,
    )
    team_ratings = compute_team_strengths(bootstrap, fixtures)

    samples: List[Tuple[List[float], float, float]] = []
    real_gw_match_count = 0

    for p in players:
        if p.minutes < 180 or not p.gw_history:
            continue

        played_gws = [
            row
            for row in p.gw_history
            if row.get("team_h_score") is not None
            and int(row.get("minutes", 0) or 0) >= 45
        ]
        if len(played_gws) < 3:
            continue

        own_r = team_ratings.get(
            p.team_id,
            {"att_home": 1150, "def_home": 1150, "att_away": 1130, "def_away": 1130},
        )

        for idx in range(2, len(played_gws)):
            match_row = played_gws[idx]
            prior_rows = played_gws[:idx]

            prior_mins = sum(int(r.get("minutes", 0) or 0) for r in prior_rows)
            if prior_mins < 90:
                continue

            # Compute time-decayed EWMA pre-match stats entering `match_row`
            rec = compute_recency_weighted_metrics(prior_rows, p, decay=0.78)

            was_home = bool(match_row.get("was_home", False))
            opp_id = int(match_row.get("opponent_team", 1))
            opp_r = team_ratings.get(
                opp_id,
                {"att_home": 1150, "def_home": 1150, "att_away": 1130, "def_away": 1130},
            )

            if was_home:
                t_att, t_def = own_r["att_home"], own_r["def_home"]
                o_att, o_def = opp_r["att_away"], opp_r["def_away"]
            else:
                t_att, t_def = own_r["att_away"], own_r["def_away"]
                o_att, o_def = opp_r["att_home"], opp_r["def_home"]

            cs_prob, gc2_prob = compute_poisson_probs(rec["ewma_xgc90"], was_home, t_def, o_att)
            pos = p.position
            gc2_feat = gc2_prob if pos in ("GKP", "DEF") else 0.0

            price_m = float(match_row.get("value", int(p.price * 10))) / 10.0
            current_val_per_m = rec["ewma_ppg"] / max(price_m, 4.0)

            pen = 0.55 if p.penalties_order == 1 else (0.20 if p.penalties_order == 2 else 0.0)
            cor = 0.30 if p.corners_order in (1, 2) else 0.0
            fk = 0.15 if p.direct_freekicks_order == 1 else 0.0
            set_piece_duty = pen + cor + fk

            feats = [
                1.0 if pos == "GKP" else 0.0,
                1.0 if pos == "DEF" else 0.0,
                1.0 if pos == "MID" else 0.0,
                1.0 if pos == "FWD" else 0.0,
                GOAL_MULT_NORM[pos],
                CS_MULT_NORM[pos],
                round(rec["ewma_ppg"] / 8.0, 4),
                round(max(-0.6, min(0.6, rec["momentum"] / 5.0)), 4),
                round(rec["ewma_xg90"], 4),
                round(rec["ewma_xa90"], 4),
                round(rec["ewma_def90"] / 12.0, 4),
                round(rec["ewma_ict90"] / 10.0, 4),
                round(rec["ewma_bps90"] / 30.0, 4),
                round(rec["recent_mins_sec"], 4),
                1.0 if was_home else 0.0,
                round((t_att - o_def) / 200.0, 4),
                round((t_def - o_att) / 200.0, 4),
                round(cs_prob, 4),
                round(gc2_feat, 4),
                round(price_m / 15.0, 4),
                round(current_val_per_m, 4),
                round(set_piece_duty, 4),
            ]

            actual_match_pts = float(match_row.get("total_points", 0) or 0)
            match_xg = float(match_row.get("expected_goals", 0) or 0)
            match_xa = float(match_row.get("expected_assists", 0) or 0)
            match_def_c = float(match_row.get("defensive_contribution", 0) or 0)
            def_c_bonus = 2.0 if (pos == "DEF" and match_def_c >= 10) or (pos == "MID" and match_def_c >= 12) else 0.0

            goal_pts = 10.0 if pos == "GKP" else (6.0 if pos == "DEF" else (5.0 if pos == "MID" else 4.0))
            cs_pts = 4.0 if pos in ("GKP", "DEF") else (1.0 if pos == "MID" else 0.0)
            xg_realization = (
                2.0
                + cs_prob * cs_pts
                + match_xg * goal_pts
                + match_xa * 3.0
                + def_c_bonus
                - (1.0 * gc2_feat)
            )

            mismatch_boost = (
                ((t_att - o_def) / 200.0) * 0.9
                if pos in ("MID", "FWD")
                else ((t_def - o_att) / 200.0) * 0.9
            )
            supervised_target = (
                0.32 * actual_match_pts
                + 0.35 * xg_realization
                + 0.33 * (rec["ewma_ppg"] + mismatch_boost)
            )

            # Temporal sample weight: matches from recent GWs matter exponentially more than GW 1
            gw_round = int(match_row.get("round", idx + 1) or (idx + 1))
            sample_weight = 0.85 ** max(0, latest_finished_gw - gw_round)

            samples.append((feats, round(max(1.0, supervised_target), 3), round(sample_weight, 4)))
            real_gw_match_count += 1

    return samples, real_gw_match_count, latest_finished_gw


def get_or_train_neural_net(force_retrain: bool = False) -> FPLNeuralNetwork:
    """
    Loads cached 22-input time-aware neural network weights or retrains automatically
    whenever `force_retrain=True` or when a new Gameweek has finished since the last training run!
    """
    if NN_WEIGHTS_PATH.exists() and not force_retrain:
        try:
            net = FPLNeuralNetwork.load(NN_WEIGHTS_PATH)
            # Check if a newer gameweek has finished in fixtures.json since this model was trained
            if CACHE_FIXTURES.exists():
                fixtures = json.loads(CACHE_FIXTURES.read_text(encoding="utf-8"))
                latest_gw = max(
                    (
                        int(f.get("event", 0) or 0)
                        for f in fixtures
                        if f.get("finished") and f.get("team_h_score") is not None
                    ),
                    default=0,
                )
                trained_gw = int(net.metrics.get("latest_trained_gw", 0) or 0)
                if latest_gw <= trained_gw:
                    return net
            else:
                return net
        except Exception:
            pass

    players = load_players()
    dataset, real_count, latest_gw = build_real_premier_league_training_dataset(players)
    net = FPLNeuralNetwork(in_dim=22, h1_dim=36, h2_dim=18, seed=42)
    net.train(
        dataset,
        real_gw_count=real_count,
        latest_trained_gw=latest_gw,
        epochs=70,
        lr=0.005,
    )
    net.save(NN_WEIGHTS_PATH)
    return net


if __name__ == "__main__":
    net = get_or_train_neural_net(force_retrain=True)
    print("=== Time-Aware 22-Feature Real Premier League Neural Network Trained ===")
    print(json.dumps(net.metrics, indent=2))
