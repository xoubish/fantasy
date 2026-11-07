"""
Multi-Head FPL Neural Network (neural_model.py)

A small numpy feed-forward network trained on REAL per-gameweek FPL outcomes:

  shared trunk  (F -> 48 -> 24)
    head "minutes": 3-class softmax  P(0 min) / P(1-59 min) / P(60+ min)
    head "points":  expected FPL points GIVEN the player plays 60+ minutes
    head "haul":    P(points >= 9)   GIVEN the player plays 60+ minutes

Unconditional expected points = P(60+) * xP_full + P(1-59) * cameo_points.

Evaluation is TIME-ORDERED: epochs are chosen on validation weeks, then the selected
model is scored once on later test weeks. Baseline constants use training data only.
The deployed weights are then refit on all rows.

Match-log features use rows BEFORE the gameweek being predicted. Roster and fixture
metadata come from the current snapshot, so historical evaluation has that limitation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "numpy is required: python3 -m pip install --user --break-system-packages numpy scipy"
    ) from exc

from fpl_data import (
    CACHE_BOOTSTRAP,
    CACHE_FIXTURES,
    DATA_DIR,
    PlayerRecord,
    compute_team_strengths,
    load_players,
)

NN_WEIGHTS_PATH = DATA_DIR / "nn_weights.json"
MODEL_VERSION = "mh-v2"

# Average FPL points of a 1-59 minute appearance (re-estimated from data at train time)
DEFAULT_CAMEO_POINTS = 1.15
HAUL_THRESHOLD = 9.0
EWMA_DECAY = 0.80
SHRINK_PSEUDO_90S = 1.0  # pseudo-minutes (in 90s) added to every per-90 denominator
XP_FULL_CLIP = (0.5, 14.0)  # sane range for points given 60+ minutes

FEATURE_NAMES: List[str] = [
    "is_GKP",
    "is_DEF",
    "is_MID",
    "is_FWD",
    # --- availability / role ---
    "start_rate_last3",
    "start_rate_last6",
    "mins_share_last3",
    "played_last_gw",
    "full_last_gw",
    "season_start_rate",
    "season_games_seen",
    # --- recency-weighted per-90 output ---
    "ewma_pts_per90",
    "season_pts_per90",
    "momentum",
    "ewma_xg90",
    "ewma_xa90",
    "ewma_xgc90",
    "ewma_bps90",
    "ewma_ict90",
    "ewma_defcon90",
    "ewma_saves90",
    "ewma_bonus90",
    # --- fixture context ---
    "is_home",
    "att_vs_def",
    "def_vs_att",
    "poisson_cs_prob",
    "fdr",
    # --- static / market ---
    "price",
    "past_season_pts90",
    "past_season_mins_share",
    "pen_taker",
    "corner_taker",
]
N_FEATURES = len(FEATURE_NAMES)

DEFAULT_RATING = {"att_home": 1150, "def_home": 1150, "att_away": 1130, "def_away": 1130}


# --------------------------------------------------------------------------------------
# Feature engineering (shared by training, live prediction and backtests)
# --------------------------------------------------------------------------------------
def compute_poisson_probs(
    xgc_per_90: float, is_home: bool, team_def_strength: int, opp_att_strength: int
) -> Tuple[float, float]:
    """Poisson P(clean sheet) and P(concede 2+) from team xGC/90 and venue/strength mismatch."""
    home_def_factor = 0.88 if is_home else 1.12
    mismatch_ratio = (opp_att_strength / max(team_def_strength, 950)) ** 1.35
    base_xgc = max(0.55, min(2.2, xgc_per_90 if xgc_per_90 > 0.1 else 1.25))
    lambda_gc = max(0.35, base_xgc * home_def_factor * mismatch_ratio)
    cs_prob = math.exp(-lambda_gc)
    concede_2plus = 1.0 - math.exp(-lambda_gc) * (1.0 + lambda_gc)
    return cs_prob, concede_2plus


def _f(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def compute_recency_weighted_metrics(
    gw_rows: List[Dict[str, Any]],
    fallback_player: Optional[PlayerRecord] = None,
    decay: float = EWMA_DECAY,
) -> Dict[str, float]:
    """
    Recency-weighted per-90 stats from a player's completed match rows.
    Weight of a row = decay ** (gameweeks ago) * (minutes / 90).
    Only rows with minutes > 0 contribute to per-90 rates; all rows feed availability stats.
    Returned keys are also consumed by `ai_coach.py` (`ewma_ppg`, `momentum`, `ewma_def90`, ...).
    """
    completed = [r for r in gw_rows if r.get("team_h_score") is not None]
    completed.sort(key=lambda r: (int(r.get("round", 0) or 0), str(r.get("kickoff_time", ""))))
    out: Dict[str, float] = {
        "ewma_ppg": 0.0,
        "season_ppg90": 0.0,
        "momentum": 0.0,
        "ewma_xg90": 0.0,
        "ewma_xa90": 0.0,
        "ewma_xgc90": 0.0,
        "ewma_def90": 0.0,
        "ewma_ict90": 0.0,
        "ewma_bps90": 0.0,
        "ewma_saves90": 0.0,
        "ewma_bonus90": 0.0,
        "start_rate_last3": 0.0,
        "start_rate_last6": 0.0,
        "mins_share_last3": 0.0,
        "played_last_gw": 0.0,
        "full_last_gw": 0.0,
        "season_start_rate": 0.0,
        "season_games_seen": 0.0,
        "recent_mins_sec": 0.0,
    }
    if not completed:
        if fallback_player is not None and fallback_player.minutes > 0:
            n90 = max(fallback_player.minutes / 90.0, 0.5)
            out["ewma_ppg"] = fallback_player.total_points / n90
            out["season_ppg90"] = out["ewma_ppg"]
            out["ewma_xg90"] = fallback_player.xg_per_90
            out["ewma_xa90"] = fallback_player.xa_per_90
            out["ewma_xgc90"] = fallback_player.xgc_per_90
            out["ewma_def90"] = fallback_player.defensive_contribution_per_90
            out["ewma_ict90"] = fallback_player.ict_per_90
            out["ewma_bps90"] = fallback_player.bps_per_90
        return out

    n = len(completed)
    latest_round = int(completed[-1].get("round", n) or n)

    def _started(r: Dict[str, Any]) -> float:
        if "starts" in r and r.get("starts") is not None:
            return 1.0 if int(r.get("starts") or 0) > 0 else 0.0
        return 1.0 if _f(r, "minutes") >= 60 else 0.0

    last3 = completed[-3:]
    last6 = completed[-6:]
    out["start_rate_last3"] = sum(_started(r) for r in last3) / len(last3)
    out["start_rate_last6"] = sum(_started(r) for r in last6) / len(last6)
    out["mins_share_last3"] = sum(min(1.0, _f(r, "minutes") / 90.0) for r in last3) / len(last3)
    out["played_last_gw"] = 1.0 if _f(completed[-1], "minutes") > 0 else 0.0
    out["full_last_gw"] = 1.0 if _f(completed[-1], "minutes") >= 60 else 0.0
    out["season_start_rate"] = sum(_started(r) for r in completed) / n
    out["season_games_seen"] = min(1.0, n / 12.0)
    out["recent_mins_sec"] = out["mins_share_last3"]

    acc = {k: 0.0 for k in ("pts", "xg", "xa", "xgc", "def", "ict", "bps", "saves", "bonus")}
    w_90 = 0.0
    raw_pts = 0.0
    raw_90 = 0.0
    for r in completed:
        mins = _f(r, "minutes")
        if mins <= 0:
            continue
        age = max(0, latest_round - int(r.get("round", latest_round) or latest_round))
        m90 = mins / 90.0
        w = (decay**age) * m90
        acc["pts"] += w * _f(r, "total_points") / m90
        acc["xg"] += w * _f(r, "expected_goals") / m90
        acc["xa"] += w * _f(r, "expected_assists") / m90
        acc["xgc"] += w * _f(r, "expected_goals_conceded") / m90
        acc["def"] += w * _f(r, "defensive_contribution") / m90
        acc["ict"] += w * _f(r, "ict_index") / m90
        acc["bps"] += w * _f(r, "bps") / m90
        acc["saves"] += w * _f(r, "saves") / m90
        acc["bonus"] += w * _f(r, "bonus") / m90
        w_90 += w
        raw_pts += _f(r, "total_points")
        raw_90 += m90

    if w_90 > 1e-9:
        # Shrink per-90 rates toward zero with one pseudo-match of 90 minutes so a 2-minute
        # cameo with 2 points does not become "90 points per 90".
        den = w_90 + SHRINK_PSEUDO_90S
        out["ewma_ppg"] = acc["pts"] / den
        out["ewma_xg90"] = acc["xg"] / den
        out["ewma_xa90"] = acc["xa"] / den
        out["ewma_xgc90"] = acc["xgc"] / den
        out["ewma_def90"] = acc["def"] / den
        out["ewma_ict90"] = acc["ict"] / den
        out["ewma_bps90"] = acc["bps"] / den
        out["ewma_saves90"] = acc["saves"] / den
        out["ewma_bonus90"] = acc["bonus"] / den
        out["season_ppg90"] = raw_pts / (raw_90 + SHRINK_PSEUDO_90S)
        out["momentum"] = out["ewma_ppg"] - out["season_ppg90"]
    return out


def _past_season_features(player: PlayerRecord) -> Tuple[float, float]:
    """Last season's points per 90 and minutes share (0 when the player has no record)."""
    if not player.historical_seasons:
        return 0.0, 0.0
    last = player.historical_seasons[-1]
    if last.minutes < 450:
        return 0.0, last.minutes / 3420.0
    return last.total_points / (last.minutes / 90.0), min(1.0, last.minutes / 3420.0)


def features_from_history(
    position: str,
    prior_rows: List[Dict[str, Any]],
    fixture: Dict[str, Any],
    price: float,
    past_pts90: float,
    past_mins_share: float,
    pen_taker: bool,
    corner_taker: bool,
    fallback_player: Optional[PlayerRecord] = None,
) -> List[float]:
    """Builds the feature vector for ONE fixture from match rows strictly before it."""
    rec = compute_recency_weighted_metrics(prior_rows, fallback_player)
    is_home = bool(fixture.get("is_home", True))
    t_att = int(fixture.get("team_att", 1150))
    t_def = int(fixture.get("team_def", 1150))
    o_att = int(fixture.get("opp_att", 1150))
    o_def = int(fixture.get("opp_def", 1150))
    cs_prob, _ = compute_poisson_probs(rec["ewma_xgc90"], is_home, t_def, o_att)

    return [
        1.0 if position == "GKP" else 0.0,
        1.0 if position == "DEF" else 0.0,
        1.0 if position == "MID" else 0.0,
        1.0 if position == "FWD" else 0.0,
        rec["start_rate_last3"],
        rec["start_rate_last6"],
        rec["mins_share_last3"],
        rec["played_last_gw"],
        rec["full_last_gw"],
        rec["season_start_rate"],
        rec["season_games_seen"],
        rec["ewma_ppg"] / 8.0,
        rec["season_ppg90"] / 8.0,
        max(-1.0, min(1.0, rec["momentum"] / 5.0)),
        rec["ewma_xg90"],
        rec["ewma_xa90"],
        rec["ewma_xgc90"] / 2.0,
        rec["ewma_bps90"] / 30.0,
        rec["ewma_ict90"] / 10.0,
        rec["ewma_def90"] / 12.0,
        rec["ewma_saves90"] / 4.0,
        rec["ewma_bonus90"] / 2.0,
        1.0 if is_home else 0.0,
        (t_att - o_def) / 200.0,
        (t_def - o_att) / 200.0,
        cs_prob,
        float(fixture.get("difficulty", 3)) / 5.0,
        price / 15.0,
        past_pts90 / 8.0,
        past_mins_share,
        1.0 if pen_taker else 0.0,
        1.0 if corner_taker else 0.0,
    ]


_LATEST_FINISHED_GW_CACHE: Optional[int] = None
_LATEST_FINISHED_GW_SIGNATURE = None


def latest_finished_gw() -> int:
    """Last gameweek in which EVERY fixture is finished (from fixtures.json), cached per process."""
    global _LATEST_FINISHED_GW_CACHE, _LATEST_FINISHED_GW_SIGNATURE
    signature = (CACHE_FIXTURES.stat().st_mtime_ns, CACHE_FIXTURES.stat().st_size) if CACHE_FIXTURES.exists() else None
    if _LATEST_FINISHED_GW_CACHE is None or signature != _LATEST_FINISHED_GW_SIGNATURE:
        _LATEST_FINISHED_GW_SIGNATURE = signature
        if not CACHE_FIXTURES.exists():
            _LATEST_FINISHED_GW_CACHE = 0
        else:
            fixtures = json.loads(CACHE_FIXTURES.read_text(encoding="utf-8"))
            by_gw: Dict[int, List[bool]] = {}
            for f in fixtures:
                gw = int(f.get("event", 0) or 0)
                if gw:
                    by_gw.setdefault(gw, []).append(bool(f.get("finished")) and f.get("team_h_score") is not None)
            done = [gw for gw, flags in by_gw.items() if flags and all(flags)]
            _LATEST_FINISHED_GW_CACHE = max(done, default=0)
    return _LATEST_FINISHED_GW_CACHE


def _finished_rows(player: PlayerRecord, max_gw: Optional[int] = None) -> List[Dict[str, Any]]:
    """Completed match rows, capped at the last FULLY finished gameweek so a round that is
    in progress (one match played, bonus not final) never leaks into features or targets."""
    cap = latest_finished_gw() if max_gw is None else max_gw
    return [
        r
        for r in player.gw_history
        if r.get("team_h_score") is not None and int(r.get("round", 0) or 0) <= cap
    ]


def extract_features(
    player: PlayerRecord, fixture_override: Optional[Dict[str, Any]] = None
) -> List[float]:
    """Live feature vector for `player`'s next fixture (or any dict from `upcoming_3gw_fixtures`)."""
    fixture = fixture_override or {
        "is_home": player.is_home,
        "team_att": player.team_attack_strength,
        "team_def": player.team_defence_strength,
        "opp_att": player.opp_attack_strength,
        "opp_def": player.opp_defence_strength,
        "difficulty": player.fixture_difficulty,
    }
    past_pts90, past_share = _past_season_features(player)
    return features_from_history(
        player.position,
        _finished_rows(player),
        fixture,
        player.price,
        past_pts90,
        past_share,
        player.penalties_order == 1,
        player.corners_order in (1, 2),
        fallback_player=player,
    )


# --------------------------------------------------------------------------------------
# Training dataset: one sample per (player, finished fixture) with strictly-prior features
# --------------------------------------------------------------------------------------
@dataclass
class Sample:
    features: List[float]
    minutes: int
    points: float
    gw: int
    player_id: int
    naive_xp: float  # baseline: prior played-rate x prior points per appearance


def build_training_samples(
    players: List[PlayerRecord],
    bootstrap: Optional[Dict[str, Any]] = None,
    fixtures: Optional[List[Dict[str, Any]]] = None,
    max_gw: Optional[int] = None,
) -> List[Sample]:
    """
    Every finished fixture row (including 0-minute rows) of every player becomes a sample.
    Features use only rows from EARLIER gameweeks; team strengths are computed from matches
    finished before that gameweek, so nothing from the target fixture leaks into its features.
    """
    bootstrap = bootstrap or json.loads(CACHE_BOOTSTRAP.read_text(encoding="utf-8"))
    fixtures = fixtures or json.loads(CACHE_FIXTURES.read_text(encoding="utf-8"))

    rounds_present = sorted(
        {
            int(r.get("round", 0) or 0)
            for p in players
            for r in p.gw_history
            if r.get("team_h_score") is not None
        }
    )
    if max_gw is not None:
        rounds_present = [g for g in rounds_present if g <= max_gw]
    ratings_before: Dict[int, Dict[int, Dict[str, int]]] = {
        g: compute_team_strengths(bootstrap, fixtures, max_gw=g - 1) for g in rounds_present
    }
    fdr_by_fixture: Dict[int, Tuple[int, int]] = {
        int(f["id"]): (int(f.get("team_h_difficulty", 3) or 3), int(f.get("team_a_difficulty", 3) or 3))
        for f in fixtures
        if f.get("id") is not None
    }

    samples: List[Sample] = []
    for p in players:
        rows = _finished_rows(p, max_gw=max_gw)
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: (int(r.get("round", 0) or 0), str(r.get("kickoff_time", ""))))
        past_pts90, past_share = _past_season_features(p)
        pen = p.penalties_order == 1
        cor = p.corners_order in (1, 2)

        for idx, row in enumerate(rows):
            gw = int(row.get("round", 0) or 0)
            prior = [r for r in rows[:idx] if int(r.get("round", 0) or 0) < gw]
            if not prior:
                continue
            was_home = bool(row.get("was_home", False))
            opp_id = int(row.get("opponent_team", 0) or 0)
            tr = ratings_before.get(gw, {})
            own = tr.get(p.team_id, DEFAULT_RATING)
            opp = tr.get(opp_id, DEFAULT_RATING)
            fdr_h, fdr_a = fdr_by_fixture.get(int(row.get("fixture", -1) or -1), (3, 3))
            fixture = {
                "is_home": was_home,
                "team_att": own["att_home"] if was_home else own["att_away"],
                "team_def": own["def_home"] if was_home else own["def_away"],
                "opp_att": opp["att_away"] if was_home else opp["att_home"],
                "opp_def": opp["def_away"] if was_home else opp["def_home"],
                "difficulty": fdr_h if was_home else fdr_a,
            }
            price = _f(row, "value") / 10.0 if _f(row, "value") > 0 else p.price
            feats = features_from_history(
                p.position, prior, fixture, price, past_pts90, past_share, pen, cor
            )
            played = [r for r in prior if _f(r, "minutes") > 0]
            play_rate = len(played) / len(prior)
            ppa = (sum(_f(r, "total_points") for r in played) / len(played)) if played else 0.0
            samples.append(
                Sample(
                    features=feats,
                    minutes=int(_f(row, "minutes")),
                    points=_f(row, "total_points"),
                    gw=gw,
                    player_id=p.id,
                    naive_xp=play_rate * ppa,
                )
            )
    return samples


# --------------------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------------------
def _leaky(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0.05 * x)


def _dleaky(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, 0.05)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class FPLNeuralNetwork:
    """Multi-head MLP: trunk F->h1->h2, heads minutes(3), points(1), haul(1)."""

    def __init__(self, in_dim: int = N_FEATURES, h1: int = 48, h2: int = 24, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.in_dim, self.h1, self.h2 = in_dim, h1, h2
        self.params: Dict[str, np.ndarray] = {
            "W1": rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, h1)),
            "b1": np.zeros(h1),
            "W2": rng.normal(0, math.sqrt(2.0 / h1), (h1, h2)),
            "b2": np.zeros(h2),
            "Wm": rng.normal(0, math.sqrt(1.0 / h2), (h2, 3)),
            "bm": np.zeros(3),
            "Wp": rng.normal(0, math.sqrt(1.0 / h2), (h2, 1)),
            "bp": np.array([3.0]),
            "Wh": rng.normal(0, math.sqrt(1.0 / h2), (h2, 1)),
            "bh": np.array([-2.2]),
        }
        self.mu = np.zeros(in_dim)
        self.sigma = np.ones(in_dim)
        self.cameo_points = DEFAULT_CAMEO_POINTS
        self.metrics: Dict[str, Any] = {}
        self.baselines = {"points": 0.0, "minutes": [1/3, 1/3, 1/3], "haul": 0.1}

    # ---- forward ----
    def _forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        P = self.params
        Xn = np.clip((X - self.mu) / self.sigma, -6.0, 6.0)
        z1 = Xn @ P["W1"] + P["b1"]
        a1 = _leaky(z1)
        z2 = a1 @ P["W2"] + P["b2"]
        a2 = _leaky(z2)
        pm = _softmax(a2 @ P["Wm"] + P["bm"])
        xp = (a2 @ P["Wp"] + P["bp"])[:, 0]
        ph = _sigmoid((a2 @ P["Wh"] + P["bh"])[:, 0])
        return {"Xn": Xn, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "pm": pm, "xp": xp, "ph": ph}

    def predict_batch(self, X: Sequence[Sequence[float]]) -> Dict[str, np.ndarray]:
        """Vectorised prediction. Returns p_none, p_cameo, p_full, xp_full, haul, expected_points."""
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[None, :]
        f = self._forward(Xa)
        xp_full = np.clip(f["xp"], XP_FULL_CLIP[0], XP_FULL_CLIP[1])
        p_full = f["pm"][:, 2]
        p_cameo = f["pm"][:, 1]
        return {
            "p_none": f["pm"][:, 0],
            "p_cameo": p_cameo,
            "p_full": p_full,
            "xp_full": xp_full,
            "haul": np.clip(f["ph"], 0.01, 0.9),
            "expected_points": p_full * xp_full + p_cameo * self.cameo_points,
        }

    def predict_full(self, x: Sequence[float]) -> Dict[str, float]:
        r = self.predict_batch([list(x)])
        return {k: float(v[0]) for k, v in r.items()}

    def predict_both(self, x: Sequence[float]) -> Tuple[float, float]:
        """Backwards-compatible: (expected points given 60+ minutes, haul probability)."""
        r = self.predict_full(x)
        return r["xp_full"], r["haul"]

    def predict(self, x: Sequence[float]) -> float:
        return self.predict_full(x)["expected_points"]

    # ---- training ----
    def train(
        self,
        samples: List[Sample],
        epochs: int = 40,
        lr: float = 1e-3,
        l2: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
        verbose: bool = False,
        on_epoch_end: Optional[Callable[[int, "FPLNeuralNetwork"], None]] = None,
    ) -> None:
        if not samples:
            raise ValueError("No completed training samples. Run `python3 fpl_assistant.py sync`; at least two gameweeks of histories are needed.")
        rng = np.random.default_rng(seed)
        X = np.asarray([s.features for s in samples], dtype=float)
        mins = np.asarray([s.minutes for s in samples], dtype=float)
        pts = np.asarray([s.points for s in samples], dtype=float)
        y_m = np.where(mins <= 0, 0, np.where(mins < 60, 1, 2))
        full = (mins >= 60).astype(float)
        y_h = ((pts >= HAUL_THRESHOLD) & (mins >= 60)).astype(float)
        self.baselines = {
            "points": float(pts.mean()),
            "minutes": (np.bincount(y_m, minlength=3) / len(samples)).tolist(),
            "haul": float(y_h[mins >= 60].mean()) if (mins >= 60).any() else 0.0,
        }
        cameo_rows = (mins > 0) & (mins < 60)
        if cameo_rows.sum() >= 20:
            self.cameo_points = float(pts[cameo_rows].mean())

        self.mu = X.mean(axis=0)
        # Floor the scale: a feature that is nearly constant in a small early-season training
        # set must not turn a different live value into a huge standardised input.
        self.sigma = np.maximum(X.std(axis=0), 0.05)
        # Recency weight: the most recent gameweeks matter slightly more
        gws = np.asarray([s.gw for s in samples], dtype=float)
        w_time = 0.93 ** (gws.max() - gws)

        P = self.params
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(val) for k, val in P.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        n = len(samples)
        huber_delta = 4.0

        for ep in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                bs = len(idx)
                f = self._forward(X[idx])
                wt = w_time[idx]
                fm = full[idx]
                n_full = max(1.0, fm.sum())

                # minutes head: cross-entropy
                onehot = np.zeros((bs, 3))
                onehot[np.arange(bs), y_m[idx]] = 1.0
                d_m = (f["pm"] - onehot) * wt[:, None] / bs

                # points head: Huber on rows with 60+ minutes
                err = f["xp"] - pts[idx]
                d_xp = np.clip(err, -huber_delta, huber_delta) * fm * wt / n_full * 0.5

                # haul head: BCE on rows with 60+ minutes
                d_h = (f["ph"] - y_h[idx]) * fm * wt / n_full * 0.6

                g: Dict[str, np.ndarray] = {}
                a2 = f["a2"]
                g["Wm"] = a2.T @ d_m + l2 * P["Wm"]
                g["bm"] = d_m.sum(axis=0)
                g["Wp"] = a2.T @ d_xp[:, None] + l2 * P["Wp"]
                g["bp"] = np.array([d_xp.sum()])
                g["Wh"] = a2.T @ d_h[:, None] + l2 * P["Wh"]
                g["bh"] = np.array([d_h.sum()])

                da2 = d_m @ P["Wm"].T + d_xp[:, None] @ P["Wp"].T + d_h[:, None] @ P["Wh"].T
                dz2 = da2 * _dleaky(f["z2"])
                g["W2"] = f["a1"].T @ dz2 + l2 * P["W2"]
                g["b2"] = dz2.sum(axis=0)
                da1 = dz2 @ P["W2"].T
                dz1 = da1 * _dleaky(f["z1"])
                g["W1"] = f["Xn"].T @ dz1 + l2 * P["W1"]
                g["b1"] = dz1.sum(axis=0)

                step += 1
                lr_t = lr * math.sqrt(1 - b2**step) / (1 - b1**step)
                for k in P:
                    m[k] = b1 * m[k] + (1 - b1) * g[k]
                    v[k] = b2 * v[k] + (1 - b2) * (g[k] * g[k])
                    P[k] -= lr_t * m[k] / (np.sqrt(v[k]) + eps)
            if on_epoch_end is not None:
                on_epoch_end(ep + 1, self)
            if verbose and (ep + 1) % 10 == 0:
                print(f"  epoch {ep + 1}: {self.evaluate(samples)['mae_points']:.3f} MAE (train)")

    def clone_params(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    # ---- evaluation ----
    def evaluate(self, samples: List[Sample]) -> Dict[str, float]:
        """Scores predictions against ACTUAL points and minutes on `samples`."""
        if not samples:
            return {}
        X = [s.features for s in samples]
        r = self.predict_batch(X)
        actual = np.asarray([s.points for s in samples])
        mins = np.asarray([s.minutes for s in samples])
        naive = np.asarray([s.naive_xp for s in samples])
        pred = r["expected_points"]
        y_m = np.where(mins <= 0, 0, np.where(mins < 60, 1, 2))
        pm = np.stack([r["p_none"], r["p_cameo"], r["p_full"]], axis=1)
        ll = -np.log(np.clip(pm[np.arange(len(samples)), y_m], 1e-6, 1)).mean()
        base_rates = np.asarray(self.baselines["minutes"])
        ll_base = -np.log(np.clip(base_rates[y_m], 1e-6, 1)).mean()
        full = mins >= 60
        haul_y = (actual >= HAUL_THRESHOLD)[full].astype(float)
        haul_p = r["haul"][full]
        out = {
            "n": float(len(samples)),
            "mae_points": float(np.abs(pred - actual).mean()),
            "mae_naive_baseline": float(np.abs(naive - actual).mean()),
            "mae_constant_baseline": float(np.abs(self.baselines["points"] - actual).mean()),
            "rmse_points": float(np.sqrt(((pred - actual) ** 2).mean())),
            "bias_points": float((pred - actual).mean()),
            "corr_points": float(np.corrcoef(pred, actual)[0, 1]) if pred.std() > 1e-12 and actual.std() > 1e-12 else 0.0,
            "corr_naive_baseline": float(np.corrcoef(naive, actual)[0, 1]) if naive.std() > 1e-12 and actual.std() > 1e-12 else 0.0,
            "minutes_logloss": float(ll),
            "minutes_logloss_baseline": float(ll_base),
            "minutes_accuracy": float((pm.argmax(axis=1) == y_m).mean()),
            "haul_brier": float(((haul_p - haul_y) ** 2).mean()) if full.any() else 0.0,
            "haul_brier_baseline": float(((self.baselines["haul"] - haul_y) ** 2).mean()) if full.any() else 0.0,
            "haul_base_rate": float(haul_y.mean()) if full.any() else 0.0,
        }
        # Error among the players you would actually pick (model's top 15% by xP)
        if len(samples) >= 40:
            cut = np.quantile(pred, 0.85)
            top = pred >= cut
            out["mae_points_top15pct"] = float(np.abs(pred[top] - actual[top]).mean())
            out["bias_points_top15pct"] = float((pred[top] - actual[top]).mean())
            out["mae_naive_top15pct"] = float(np.abs(naive[top] - actual[top]).mean())
        return {k: round(v, 4) for k, v in out.items()}

    def feature_importance(self) -> Dict[str, float]:
        """Permutation-free proxy: L1 norm of first-layer weights per standardised input."""
        w = np.abs(self.params["W1"]).sum(axis=1)
        w = w / w.sum() * 100.0
        order = np.argsort(-w)
        return {FEATURE_NAMES[i]: round(float(w[i]), 1) for i in order}

    # ---- persistence ----
    def save(self, path: Path = NN_WEIGHTS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MODEL_VERSION,
            "feature_names": FEATURE_NAMES,
            "dims": [self.in_dim, self.h1, self.h2],
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "cameo_points": self.cameo_points,
            "baselines": self.baselines,
            "params": {k: v.tolist() for k, v in self.params.items()},
            "metrics": self.metrics,
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = NN_WEIGHTS_PATH) -> "FPLNeuralNetwork":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != MODEL_VERSION or data.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Outdated neural model schema; retrain required.")
        in_dim, h1, h2 = data["dims"]
        net = cls(in_dim=in_dim, h1=h1, h2=h2)
        net.params = {k: np.asarray(v, dtype=float) for k, v in data["params"].items()}
        net.mu = np.asarray(data["mu"], dtype=float)
        net.sigma = np.asarray(data["sigma"], dtype=float)
        net.cameo_points = float(data.get("cameo_points", DEFAULT_CAMEO_POINTS))
        net.baselines = data["baselines"]
        net.metrics = data.get("metrics", {})
        return net


# --------------------------------------------------------------------------------------
# Train with an honest time-ordered holdout, then refit on everything for deployment
# --------------------------------------------------------------------------------------
def train_with_time_holdout(
    samples: List[Sample],
    holdout_gws: int = 2,
    max_epochs: int = 60,
    seed: int = 42,
    verbose: bool = False,
    test_gws: int = 1,
) -> Tuple[FPLNeuralNetwork, Dict[str, Any]]:
    """
    1. Select epochs on chronological validation weeks preceding the test window.
    2. Evaluate the chosen model once on the later test weeks, never used for selection.
    3. Refit on ALL rows with that epoch count for deployment.
    """
    if not samples:
        raise ValueError("No completed training samples. Run `python3 fpl_assistant.py sync`; at least two gameweeks of histories are needed.")
    if holdout_gws < 1 or test_gws < 0 or max_epochs < 1:
        raise ValueError("Validation weeks and epochs must be positive; test weeks cannot be negative.")
    gws = sorted({s.gw for s in samples})
    latest = gws[-1]
    # Count observed rounds, so postponed/blank weeks do not empty a split.
    n_test = min(test_gws, max(0, len(gws) - holdout_gws - 1))
    test_rounds = set(gws[-n_test:]) if n_test else set()
    development_rounds = gws[:-n_test] if n_test else gws
    n_val = min(holdout_gws, max(0, len(development_rounds) - 1))
    val_rounds = set(development_rounds[-n_val:]) if n_val else set()
    train = [s for s in samples if s.gw not in test_rounds | val_rounds]
    val = [s for s in samples if s.gw in val_rounds]
    test = [s for s in samples if s.gw in test_rounds]
    best_epochs = min(25, max_epochs)

    metrics: Dict[str, Any] = {
        "version": MODEL_VERSION,
        "architecture": f"{N_FEATURES} -> 48 -> 24 -> [minutes(3) | points | haul]",
        "latest_trained_gw": latest,
        "real_gw_samples": len(samples),
        "train_samples": len(train),
        "val_samples": len(val),
        "holdout_gws": f"GW {min(val_rounds)}-{max(val_rounds)}" if val else "none",
        "test_gws": f"GW {min(test_rounds)}-{max(test_rounds)}" if test else "none",
        "test_samples": len(test),
        "evaluation_protocol": "epochs selected on validation; later test weeks evaluated once; deployed weights refit on all samples",
        "targets": "actual FPL points and actual minutes (no synthetic targets)",
    }
    if train and val and len(val) >= 30:
        probe = FPLNeuralNetwork(seed=seed)
        curve: List[Tuple[int, float]] = []
        best: Dict[str, Any] = {"epoch": 0, "mae": float("inf"), "params": None}

        def _checkpoint(ep: int, net: FPLNeuralNetwork) -> None:
            mae = net.evaluate(val)["mae_points"]
            curve.append((ep, mae))
            if mae < best["mae"] - 1e-4:
                best.update(epoch=ep, mae=mae, params=net.clone_params())

        probe.train(train, epochs=max_epochs, seed=seed, verbose=False, on_epoch_end=_checkpoint)
        if best["params"] is not None:
            probe.params = best["params"]
            best_epochs = int(best["epoch"])
        ev_val = probe.evaluate(val)
        ev_train = probe.evaluate(train)
        metrics["holdout"] = ev_val
        metrics["validation"] = ev_val
        if test:
            metrics["test"] = probe.evaluate(test)
        metrics["early_stopping"] = {
            "best_epoch": best_epochs,
            "curve": [(e, round(m, 3)) for e, m in curve if e % 5 == 0 or e == best_epochs],
        }
        metrics["train_fit"] = {k: ev_train[k] for k in ("mae_points", "corr_points") if k in ev_train}
        if verbose:
            print(f"  early stopping: best epoch {best_epochs} (validation MAE {best['mae']:.3f})")
        # Legacy keys read by the web page / CLI, now with honest meaning
        metrics["val_mae"] = ev_val["mae_points"]
        metrics["val_mae_naive_baseline"] = ev_val["mae_naive_baseline"]
        metrics["val_rmse"] = ev_val["rmse_points"]
        metrics["haul_val_brier"] = ev_val["haul_brier"]
        metrics["haul_val_brier_baseline"] = ev_val["haul_brier_baseline"]
        metrics["train_mae"] = ev_train["mae_points"]

    final = FPLNeuralNetwork(seed=seed)
    final.train(samples, epochs=best_epochs, seed=seed, verbose=verbose)
    metrics["feature_importance"] = final.feature_importance()
    metrics["cameo_points"] = round(final.cameo_points, 3)
    final.metrics = metrics
    return final, metrics


def get_or_train_neural_net(force_retrain: bool = False, verbose: bool = False) -> FPLNeuralNetwork:
    """Loads cached weights, retraining when a new gameweek has finished or the schema changed."""
    if NN_WEIGHTS_PATH.exists() and not force_retrain:
        try:
            net = FPLNeuralNetwork.load(NN_WEIGHTS_PATH)
            if latest_finished_gw() <= int(net.metrics.get("latest_trained_gw", 0) or 0):
                return net
        except Exception:
            pass
    players = load_players()
    samples = build_training_samples(players)
    net, _ = train_with_time_holdout(samples, verbose=verbose)
    net.save(NN_WEIGHTS_PATH)
    return net


if __name__ == "__main__":
    import time

    t0 = time.time()
    net = get_or_train_neural_net(force_retrain=True, verbose=True)
    print(f"=== Multi-head FPL model trained in {time.time() - t0:.1f}s ===")
    print(json.dumps(net.metrics, indent=2))
