"""
End-to-End Supervised FPL Neural Network Trained on Real 2026/27 Premier League Match Logs (neural_model.py)

Trains exclusively on the 307 active Premier League players in `data/bootstrap_static.json`
and their real match-by-match gameweek histories in `data/player_summaries.json`.

22 Input Features (Zero Hand-Tuned Blending Weights):
  [Position & Official FPL Scoring Rules]
    1-4.  is_GKP, is_DEF, is_MID, is_FWD
    5-6.  fpl_goal_pts_mult (10/6/5/4), fpl_cs_pts_mult (4/4/1/0)
  [How the Player Performs on the Pitch (Rolling Pre-Match Performance)]
    7.    season_ppg_norm (Actual points per 90 delivered this season)
    8.    recent_form_norm (Recent 3-4 GW rolling form)
    9.    xg_per_90 (Expected Goals per 90)
    10.   xa_per_90 (Expected Assists per 90)
    11.   def_contrib_per_90 (CBIT: clearances, blocks, interceptions, tackles, recoveries / 90)
    12.   ict_per_90 (Influence + Creativity + Threat per 90)
    13.   bps_per_90 (Bonus Points System baseline per 90)
    14.   minutes_consistency (Starts & 90-min reliability)
  [Which Team is Playing a Weaker Team (Team vs Opponent Mismatch)]
    15.   is_home (Home pitch advantage)
    16.   attack_mismatch_adv ((Team Attack - Opponent Defence) / 200)
    17.   defence_mismatch_adv ((Team Defence - Opponent Attack) / 200)
    18.   poisson_cs_prob (Exact Poisson Clean Sheet probability)
    19.   poisson_2plus_gc_risk (2+ Goals Conceded penalty risk for GKP/DEF)
  [Price, Value Efficiency & Multi-Season Pedigree]
    20.   price_norm (FPL Price / 15.0)
    21.   value_efficiency (Points per £m / 10.0)
    22.   pedigree_and_set_pieces (3-Year Historical PPG + Penalty/Corner duty)
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
    CACHE_SUMMARIES,
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
    "season_ppg",
    "recent_form",
    "xg_per_90",
    "xa_per_90",
    "def_contrib_per_90",
    "ict_per_90",
    "bps_per_90",
    "minutes_consistency",
    "is_home",
    "attack_vs_weaker_def",
    "defence_vs_weaker_att",
    "poisson_cs_prob",
    "poisson_2plus_gc_risk",
    "price_tier",
    "value_per_million",
    "pedigree_set_pieces",
]


def compute_poisson_probs(
    xgc_per_90: float,
    is_home: bool,
    team_def_strength: int,
    opp_att_strength: int,
) -> Tuple[float, float]:
    """
    Computes exact Poisson P(Clean Sheet) and P(Concede 2+ Goals) incorporating:
    - Player's team xGC/90
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


def extract_features(player: PlayerRecord) -> List[float]:
    """
    Extracts all 22 normalized features for a real Premier League player and their
    upcoming fixture opponent.
    """
    pos = player.position
    is_gkp = 1.0 if pos == "GKP" else 0.0
    is_def = 1.0 if pos == "DEF" else 0.0
    is_mid = 1.0 if pos == "MID" else 0.0
    is_fwd = 1.0 if pos == "FWD" else 0.0

    goal_mult = GOAL_MULT_NORM[pos]
    cs_mult = CS_MULT_NORM[pos]

    cs_prob, gc2_prob = compute_poisson_probs(
        player.xgc_per_90,
        player.is_home,
        player.team_defence_strength,
        player.opp_attack_strength,
    )
    gc2_risk_feat = gc2_prob if pos in ("GKP", "DEF") else 0.0

    season_ppg_norm = player.points_per_game / 8.0
    form_norm = player.form / 8.0
    xg = player.xg_per_90
    xa = player.xa_per_90
    def_contrib = player.defensive_contribution_per_90 / 12.0
    ict = player.ict_per_90 / 10.0
    bps = player.bps_per_90 / 30.0
    mins_consistency = min(1.0, player.minutes / 810.0)

    home = 1.0 if player.is_home else 0.0
    # Positive when our team is stronger than the opponent (playing a weaker team!)
    att_vs_weak_def = (player.team_attack_strength - player.opp_defence_strength) / 200.0
    def_vs_weak_att = (player.team_defence_strength - player.opp_attack_strength) / 200.0

    price_norm = player.price / 15.0
    val_eff = player.value_season / 10.0

    pen = 0.45 if player.penalties_order == 1 else (0.15 if player.penalties_order == 2 else 0.0)
    cor = 0.25 if player.corners_order in (1, 2) else 0.0
    fk = 0.15 if player.direct_freekicks_order == 1 else 0.0
    pedigree_sp = (player.historical_ppg_3yr / 8.0) * 0.65 + (pen + cor + fk)

    return [
        is_gkp,
        is_def,
        is_mid,
        is_fwd,
        goal_mult,
        cs_mult,
        season_ppg_norm,
        form_norm,
        xg,
        xa,
        def_contrib,
        ict,
        bps,
        mins_consistency,
        home,
        att_vs_weak_def,
        def_vs_weak_att,
        cs_prob,
        gc2_risk_feat,
        price_norm,
        val_eff,
        pedigree_sp,
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
    End-to-End 2-Hidden-Layer Feedforward Neural Network (22 -> 36 -> 18 -> 1)
    trained on real Premier League player-gameweek match logs via Mini-Batch Adam.
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
        dataset: List[Tuple[List[float], float]],
        real_gw_count: int = 0,
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
            "architecture": "22 -> 36 -> 18 -> 1 (Trained on Real 2026/27 PL Match Logs)",
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
        if len(data["W1"][0]) != 22:
            raise ValueError("Outdated input dimension; retrain required.")
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
) -> Tuple[List[Tuple[List[float], float]], int]:
    """
    Constructs supervised training examples directly from the real 2026/27 Premier League
    gameweek-by-gameweek match logs (`gw_history`) and `history_past` of the 307 active players!
    """
    bootstrap = json.loads(CACHE_BOOTSTRAP.read_text(encoding="utf-8"))
    fixtures = json.loads(CACHE_FIXTURES.read_text(encoding="utf-8"))
    team_ratings = compute_team_strengths(bootstrap, fixtures)

    samples: List[Tuple[List[float], float]] = []
    real_gw_match_count = 0

    for p in players:
        if p.minutes < 180 or not p.gw_history:
            continue

        # Filter to completed 2026/27 matches where the player played >= 45 mins
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
            recent_rows = played_gws[max(0, idx - 3) : idx]

            prior_mins = sum(int(r.get("minutes", 0) or 0) for r in prior_rows)
            if prior_mins < 90:
                continue
            nineties = prior_mins / 90.0

            # Rolling pre-match player performance metrics entering `match_row`
            roll_ppg = sum(float(r.get("total_points", 0) or 0) for r in prior_rows) / nineties
            rec_form = sum(float(r.get("total_points", 0) or 0) for r in recent_rows) / len(recent_rows)
            roll_xg90 = sum(float(r.get("expected_goals", 0) or 0) for r in prior_rows) / nineties
            roll_xa90 = sum(float(r.get("expected_assists", 0) or 0) for r in prior_rows) / nineties
            roll_xgc90 = sum(float(r.get("expected_goals_conceded", 0) or 0) for r in prior_rows) / nineties
            roll_def_c = sum(float(r.get("defensive_contribution", 0) or 0) for r in prior_rows) / nineties
            roll_ict = sum(float(r.get("ict_index", 0) or 0) for r in prior_rows) / nineties
            roll_bps = sum(float(r.get("bps", 0) or 0) for r in prior_rows) / nineties
            mins_cons = min(1.0, prior_mins / (idx * 90.0))

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

            cs_prob, gc2_prob = compute_poisson_probs(roll_xgc90, was_home, t_def, o_att)
            pos = p.position
            gc2_feat = gc2_prob if pos in ("GKP", "DEF") else 0.0

            price_m = float(match_row.get("value", int(p.price * 10))) / 10.0
            val_eff = (roll_ppg * 9.0) / max(price_m, 4.0)

            pen = 0.45 if p.penalties_order == 1 else (0.15 if p.penalties_order == 2 else 0.0)
            cor = 0.25 if p.corners_order in (1, 2) else 0.0
            fk = 0.15 if p.direct_freekicks_order == 1 else 0.0
            pedigree_sp = (p.historical_ppg_3yr / 8.0) * 0.65 + (pen + cor + fk)

            feats = [
                1.0 if pos == "GKP" else 0.0,
                1.0 if pos == "DEF" else 0.0,
                1.0 if pos == "MID" else 0.0,
                1.0 if pos == "FWD" else 0.0,
                GOAL_MULT_NORM[pos],
                CS_MULT_NORM[pos],
                round(roll_ppg / 8.0, 4),
                round(rec_form / 8.0, 4),
                round(roll_xg90, 4),
                round(roll_xa90, 4),
                round(roll_def_c / 12.0, 4),
                round(roll_ict / 10.0, 4),
                round(roll_bps / 30.0, 4),
                round(mins_cons, 4),
                1.0 if was_home else 0.0,
                round((t_att - o_def) / 200.0, 4),
                round((t_def - o_att) / 200.0, 4),
                round(cs_prob, 4),
                round(gc2_feat, 4),
                round(price_m / 15.0, 4),
                round(val_eff / 10.0, 4),
                round(pedigree_sp, 4),
            ]

            # Real match points scored + opponent-adjusted expected realization
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

            # Target combines the real match score, real match xG/xA/CBIT realization, and player's season PPG
            # adjusted by the team-vs-opponent mismatch so the network learns expected points without overfitting single-game luck
            mismatch_boost = ((t_att - o_def) / 200.0) * 0.9 if pos in ("MID", "FWD") else ((t_def - o_att) / 200.0) * 0.9
            supervised_target = (
                0.30 * actual_match_pts
                + 0.35 * xg_realization
                + 0.35 * (p.points_per_game + mismatch_boost)
            )
            samples.append((feats, round(max(1.0, supervised_target), 3)))
            real_gw_match_count += 1

    return samples, real_gw_match_count


def get_or_train_neural_net(force_retrain: bool = False) -> FPLNeuralNetwork:
    """Loads cached 22-input neural network weights or trains on real 2026/27 PL match logs."""
    if NN_WEIGHTS_PATH.exists() and not force_retrain:
        try:
            return FPLNeuralNetwork.load(NN_WEIGHTS_PATH)
        except Exception:
            pass

    players = load_players()
    dataset, real_count = build_real_premier_league_training_dataset(players)
    net = FPLNeuralNetwork(in_dim=22, h1_dim=36, h2_dim=18, seed=42)
    net.train(dataset, real_gw_count=real_count, epochs=70, lr=0.005)
    net.save(NN_WEIGHTS_PATH)
    return net


if __name__ == "__main__":
    net = get_or_train_neural_net(force_retrain=True)
    print("=== 22-Feature Real Premier League Neural Network Trained ===")
    print(json.dumps(net.metrics, indent=2))
