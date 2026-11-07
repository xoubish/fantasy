"""
Multi-Head FPL Neural Network (neural_model.py)

A small numpy feed-forward network trained on REAL per-gameweek FPL outcomes:

  shared trunk  (F -> 48 -> 24)
    head "minutes":    3-class softmax  P(0 min) / P(1-59 min) / P(60+ min)
    head "points":     expected FPL points GIVEN the player plays 60+ minutes (direct)
    head "haul":       P(points >= 9)   GIVEN the player plays 60+ minutes
    component heads (all GIVEN 60+ minutes): goals, assists, clean sheet, goals conceded,
                       saves, bonus, defensive-contribution threshold, yellow card

Component expected points apply the official scoring rules to the component heads:
  2 + goals x goal points(position) + 3 x assists + P(clean sheet) x clean-sheet points
  - E[floor(goals conceded / 2)] (GKP/DEF) + E[bonus] + E[floor(saves / 3)] (GKP)
  + 2 x P(defensive contribution threshold) (2025-26 rules onward) - P(yellow card)
  + small position constants for red cards, own goals and penalties learned from training.

The deployed "points given 60+" output is the direct head, the component sum, or their
average, chosen on validation weeks by `historical_training.py`; it is never chosen on
test rows. Unconditional expected points = P(60+) * xP_full + P(1-59) * cameo_points.

Match-log features use rows BEFORE the gameweek being predicted. Per-90 rates are shrunk
toward the player's previous-season rates (or position averages when there is no previous
Premier League season). Fixture context comes from the cross-season Poisson team model in
`team_ratings.py`, fitted on matches before the gameweek deadline.
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

from fpl_data import CACHE_FIXTURES, DATA_DIR, PlayerRecord

NN_WEIGHTS_PATH = DATA_DIR / "nn_weights.json"
MODEL_VERSION = "mh-v4.0-components"

# Average FPL points of a 1-59 minute appearance (re-estimated from data at train time)
DEFAULT_CAMEO_POINTS = 1.15
HAUL_THRESHOLD = 9.0
EWMA_DECAY = 0.80
PRIOR_PSEUDO_90S = 3.0  # weight (in 90s) of a player's previous-season per-90 rates
POSITION_PSEUDO_90S = 1.0  # weight of the position average when there is no previous season
PRIOR_MIN_MINUTES = 450  # previous-season minutes needed before its rates are used as a prior
XP_FULL_CLIP = (0.5, 14.0)  # sane range for points given 60+ minutes
DEFAULT_LAMBDA = 1.35  # league-average goals per team per match (fallback fixture context)

# Position averages of per-90 rates, players with >= 450 minutes in 2022-23..2024-25
# (`historical_data.py position-priors` recomputes them). Defensive contribution has no
# pre-2025 history, so its prior is zero.
POSITION_PRIOR_RATES: Dict[str, Dict[str, float]] = {
    "GKP": {"pts": 3.29, "xg": 0.00, "xa": 0.00, "xgc": 1.51, "def": 0.0, "ict": 2.53, "bps": 16.0, "saves": 3.12,
            "bonus": 0.25, "goals": 0.00, "assists": 0.01, "cs": 0.22, "yellow": 0.07},
    "DEF": {"pts": 3.06, "xg": 0.05, "xa": 0.06, "xgc": 1.52, "def": 0.0, "ict": 3.51, "bps": 15.5, "saves": 0.0,
            "bonus": 0.20, "goals": 0.04, "assists": 0.07, "cs": 0.23, "yellow": 0.19},
    "MID": {"pts": 4.09, "xg": 0.16, "xa": 0.12, "xgc": 1.49, "def": 0.0, "ict": 5.58, "bps": 16.2, "saves": 0.0,
            "bonus": 0.25, "goals": 0.16, "assists": 0.17, "cs": 0.26, "yellow": 0.21},
    "FWD": {"pts": 5.10, "xg": 0.42, "xa": 0.08, "xgc": 1.51, "def": 0.0, "ict": 6.80, "bps": 16.9, "saves": 0.0,
            "bonus": 0.58, "goals": 0.40, "assists": 0.17, "cs": 0.28, "yellow": 0.18},
}
RATE_KEYS = ("pts", "xg", "xa", "xgc", "def", "ict", "bps", "saves", "bonus", "goals", "assists", "cs", "yellow")

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
    # --- recency-weighted per-90 output, shrunk toward previous-season rates ---
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
    "ewma_goals90",
    "ewma_assists90",
    "cs_rate",
    "yellow90",
    # --- fixture context from the cross-season Poisson team model ---
    "is_home",
    "log_lambda_for",
    "log_lambda_against",
    "team_cs_prob",
    # --- static / market ---
    "price",
    "past_season_pts90",
    "past_season_mins_share",
    "prior_known",
    "defcon_observed",
    "scoring_2025_onward",
]
N_FEATURES = len(FEATURE_NAMES)
_ERA_INDEX = FEATURE_NAMES.index("scoring_2025_onward")

# Component targets carried with every training sample, in this order.
COMPONENT_NAMES = ("goals", "assists", "clean_sheet", "goals_conceded", "saves", "bonus", "defcon",
                   "yellow", "red", "own_goals", "pen_missed", "pen_saved", "has_defcon")
COMPONENT_ROW_KEYS = {"goals": "goals_scored", "assists": "assists", "clean_sheet": "clean_sheets",
                      "goals_conceded": "goals_conceded", "saves": "saves", "bonus": "bonus",
                      "defcon": "defensive_contribution", "yellow": "yellow_cards", "red": "red_cards",
                      "own_goals": "own_goals", "pen_missed": "penalties_missed", "pen_saved": "penalties_saved"}
GOAL_POINTS = np.array([10.0, 6.0, 5.0, 4.0])  # GKP, DEF, MID, FWD
CS_POINTS = np.array([4.0, 4.0, 1.0, 0.0])
DEFCON_THRESHOLD = np.array([999.0, 10.0, 12.0, 12.0])  # goalkeepers never score it
POS_ORDER = ("GKP", "DEF", "MID", "FWD")


def components_from_row(row: Dict[str, Any]) -> Tuple[float, ...]:
    """Component outcomes of one match row (archive or live API fields)."""
    values = [float(row.get(COMPONENT_ROW_KEYS[name], 0) or 0) for name in COMPONENT_NAMES[:-1]]
    has_defcon = row.get("has_defcon")
    if has_defcon is None:
        has_defcon = "defensive_contribution" in row
    values.append(1.0 if has_defcon else 0.0)
    return tuple(values)


# --------------------------------------------------------------------------------------
# Feature engineering (shared by training, live prediction and backtests)
# --------------------------------------------------------------------------------------
def _f(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def prior_rates_from_totals(totals: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Previous-season per-90 rates from season totals (API `history_past` or archive `players_raw`).

    `valid` is false when the player had fewer than PRIOR_MIN_MINUTES minutes; the rates are
    then ignored and only the minutes share remains informative.
    """
    totals = totals or {}
    minutes = _f(totals, "minutes")
    n90 = minutes / 90.0
    out: Dict[str, float] = {"valid": 0.0, "n90": n90, "mins_share": min(1.0, minutes / 3420.0)}
    for key in RATE_KEYS:
        out[key] = 0.0
    if minutes < PRIOR_MIN_MINUTES:
        return out
    source = {"pts": "total_points", "xg": "expected_goals", "xa": "expected_assists",
              "xgc": "expected_goals_conceded", "def": "defensive_contribution", "ict": "ict_index",
              "bps": "bps", "saves": "saves", "bonus": "bonus", "goals": "goals_scored",
              "assists": "assists", "cs": "clean_sheets", "yellow": "yellow_cards"}
    for key, column in source.items():
        out[key] = _f(totals, column) / n90
    out["valid"] = 1.0
    return out


def compute_recency_weighted_metrics(
    gw_rows: List[Dict[str, Any]],
    fallback_player: Optional[PlayerRecord] = None,
    decay: float = EWMA_DECAY,
    prior: Optional[Dict[str, float]] = None,
    position: Optional[str] = None,
) -> Dict[str, float]:
    """
    Recency-weighted per-90 stats from a player's completed match rows.
    Weight of a row = decay ** (gameweeks ago) * (minutes / 90).
    Rates are shrunk toward the previous-season rates in `prior` (PRIOR_PSEUDO_90S) or, without a
    usable previous season, toward the position average (POSITION_PSEUDO_90S).
    Returned keys are also consumed by `ai_coach.py` (`ewma_ppg`, `momentum`, `ewma_def90`, ...).
    """
    position = position or (fallback_player.position if fallback_player is not None else "MID")
    if prior is None and fallback_player is not None:
        prior = prior_rates_from_player(fallback_player)
    if prior and prior.get("valid"):
        centre, k = {key: float(prior.get(key, 0.0)) for key in RATE_KEYS}, PRIOR_PSEUDO_90S
    else:
        centre, k = dict(POSITION_PRIOR_RATES.get(position, POSITION_PRIOR_RATES["MID"])), POSITION_PSEUDO_90S
    completed = [r for r in gw_rows if r.get("team_h_score") is not None]
    completed.sort(key=lambda r: (int(r.get("round", 0) or 0), str(r.get("kickoff_time", ""))))
    out: Dict[str, float] = {
        "ewma_ppg": centre["pts"], "season_ppg90": centre["pts"], "momentum": 0.0,
        "ewma_xg90": centre["xg"], "ewma_xa90": centre["xa"], "ewma_xgc90": centre["xgc"],
        "ewma_def90": centre["def"], "ewma_ict90": centre["ict"], "ewma_bps90": centre["bps"],
        "ewma_saves90": centre["saves"], "ewma_bonus90": centre["bonus"],
        "ewma_goals90": centre["goals"], "ewma_assists90": centre["assists"],
        "cs_rate": centre["cs"], "yellow90": centre["yellow"],
        "start_rate_last3": 0.0, "start_rate_last6": 0.0, "mins_share_last3": 0.0,
        "played_last_gw": 0.0, "full_last_gw": 0.0, "season_start_rate": 0.0,
        "season_games_seen": 0.0, "recent_mins_sec": 0.0,
    }
    if not completed:
        return out

    n = len(completed)
    latest_round = int(completed[-1].get("round", n) or n)

    def _started(r: Dict[str, Any]) -> float:
        if "starts" in r and r.get("starts") is not None:
            return 1.0 if int(_f(r, "starts")) > 0 else 0.0
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

    columns = {"pts": "total_points", "xg": "expected_goals", "xa": "expected_assists",
               "xgc": "expected_goals_conceded", "def": "defensive_contribution", "ict": "ict_index",
               "bps": "bps", "saves": "saves", "bonus": "bonus", "goals": "goals_scored",
               "assists": "assists", "yellow": "yellow_cards"}
    acc = {key: 0.0 for key in columns}
    w_90 = 0.0
    raw_pts = 0.0
    raw_90 = 0.0
    cs_acc = 0.0
    cs_w = 0.0
    for r in completed:
        mins = _f(r, "minutes")
        if mins <= 0:
            continue
        age = max(0, latest_round - int(r.get("round", latest_round) or latest_round))
        m90 = mins / 90.0
        w = (decay**age) * m90
        for key, column in columns.items():
            acc[key] += w * _f(r, column) / m90
        w_90 += w
        raw_pts += _f(r, "total_points")
        raw_90 += m90
        if mins >= 60:
            cs_acc += (decay**age) * (1.0 if _f(r, "clean_sheets") > 0 else 0.0)
            cs_w += decay**age

    den = w_90 + k
    names = {"pts": "ewma_ppg", "xg": "ewma_xg90", "xa": "ewma_xa90", "xgc": "ewma_xgc90", "def": "ewma_def90",
             "ict": "ewma_ict90", "bps": "ewma_bps90", "saves": "ewma_saves90", "bonus": "ewma_bonus90",
             "goals": "ewma_goals90", "assists": "ewma_assists90", "yellow": "yellow90"}
    for key, name in names.items():
        out[name] = (acc[key] + k * centre[key]) / den
    out["season_ppg90"] = (raw_pts + k * centre["pts"]) / (raw_90 + k)
    out["cs_rate"] = (cs_acc + k * centre["cs"]) / (cs_w + k)
    out["momentum"] = out["ewma_ppg"] - out["season_ppg90"]
    return out


def prior_rates_from_player(player: PlayerRecord) -> Dict[str, float]:
    """Previous-season rates for a live player (only the immediately preceding season counts)."""
    from historical_data import current_season_start, season_name
    previous = season_name(current_season_start() - 1).replace("-", "/")
    for season in player.historical_seasons:
        if str(season.season_name).replace("-", "/") == previous:
            return prior_rates_from_totals({
                "minutes": season.minutes, "total_points": season.total_points,
                "expected_goals": season.expected_goals, "expected_assists": season.expected_assists,
                "expected_goals_conceded": season.expected_goals_conceded,
                "defensive_contribution": season.defensive_contribution, "ict_index": season.ict_index,
                "bps": season.bps, "saves": season.saves, "bonus": season.bonus,
                "goals_scored": season.goals_scored, "assists": season.assists,
                "clean_sheets": season.clean_sheets, "yellow_cards": season.yellow_cards,
            })
    return prior_rates_from_totals(None)


def features_from_history(
    position: str,
    prior_rows: List[Dict[str, Any]],
    fixture: Dict[str, Any],
    price: float,
    prior: Optional[Dict[str, float]] = None,
    season_start: int = 2026,
) -> List[float]:
    """Builds the feature vector for ONE fixture from match rows strictly before it.

    `fixture` carries `is_home` and the Poisson team model's `lambda_for` / `lambda_against`.
    `prior` is the previous-season rate dict from `prior_rates_from_totals`.
    """
    prior = prior or prior_rates_from_totals(None)
    rec = compute_recency_weighted_metrics(prior_rows, decay=EWMA_DECAY, prior=prior, position=position)
    is_home = bool(fixture.get("is_home", True))
    lam_for = max(0.05, float(fixture.get("lambda_for", DEFAULT_LAMBDA) or DEFAULT_LAMBDA))
    lam_against = max(0.05, float(fixture.get("lambda_against", DEFAULT_LAMBDA) or DEFAULT_LAMBDA))

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
        rec["ewma_goals90"],
        rec["ewma_assists90"],
        rec["cs_rate"],
        rec["yellow90"],
        1.0 if is_home else 0.0,
        math.log(lam_for),
        math.log(lam_against),
        math.exp(-lam_against),
        price / 15.0,
        float(prior.get("pts", 0.0)) / 8.0 if prior.get("valid") else 0.0,
        float(prior.get("mins_share", 0.0)),
        1.0 if prior.get("valid") else 0.0,
        float(any("defensive_contribution" in r for r in prior_rows)),
        float(season_start >= 2025),
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
    from historical_data import current_season_start
    fixture = fixture_override or {
        "is_home": player.is_home,
        "lambda_for": player.lambda_for,
        "lambda_against": player.lambda_against,
    }
    return features_from_history(
        player.position,
        _finished_rows(player),
        fixture,
        player.price,
        prior=prior_rates_from_player(player),
        season_start=current_season_start(),
    )


# --------------------------------------------------------------------------------------
# Training sample: one per (player, finished fixture) with strictly-prior features
# --------------------------------------------------------------------------------------
@dataclass
class Sample:
    features: List[float]
    minutes: int
    points: float
    gw: int
    player_id: int
    naive_xp: float  # baseline: prior played-rate x prior points per appearance
    season: str = ""
    fixture_id: int = 0
    components: Tuple[float, ...] = ()
    naive_ppa: float = 0.0  # baseline given the player plays: prior points per appearance

    @property
    def period(self) -> Tuple[int, int]:
        return (int(self.season[:4]) if self.season else 0, self.gw)


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


def _poisson_pmf_matrix(lam: np.ndarray, k_max: int) -> np.ndarray:
    ks = np.arange(k_max + 1)
    log_pmf = ks[None, :] * np.log(lam[:, None]) - lam[:, None] - np.array([math.lgamma(k + 1) for k in ks])[None, :]
    return np.exp(log_pmf)


def expected_floor_division(lam: np.ndarray, divisor: int, k_max: int = 12) -> np.ndarray:
    """E[floor(K / divisor)] for K ~ Poisson(lam), truncated at k_max."""
    pmf = _poisson_pmf_matrix(np.maximum(lam, 1e-6), k_max)
    return pmf @ np.floor(np.arange(k_max + 1) / divisor)


# Component heads: name -> (link, bias init). Poisson heads use a log link.
COMPONENT_HEADS: Dict[str, Tuple[str, float]] = {
    "goals": ("poisson", math.log(0.15)),
    "assists": ("poisson", math.log(0.12)),
    "cs": ("logistic", -1.1),
    "gc": ("poisson", math.log(1.4)),
    "saves": ("poisson", math.log(3.0)),
    "bonus": ("poisson", math.log(0.3)),
    "defcon": ("logistic", -0.8),
    "yellow": ("logistic", -1.7),
}
POINTS_OUTPUTS = ("direct", "component", "blend")


class FPLNeuralNetwork:
    """Multi-head MLP: trunk F->h1->h2, heads minutes(3), points(1), haul(1) and component heads."""

    def __init__(self, in_dim: int = N_FEATURES, h1: int = 48, h2: int = 24, seed: int = 42, points_loss: str = "mse",
                 points_output: str = "direct"):
        if points_loss not in ("mse", "huber"):
            raise ValueError("points_loss must be mse or huber")
        if points_output not in POINTS_OUTPUTS:
            raise ValueError(f"points_output must be one of {POINTS_OUTPUTS}")
        self.points_loss = points_loss
        self.points_output = points_output
        self.calibration = {"minutes_temperature": 1.0, "points": [[0.0, 1.0]] * 4,
                            "residual_std": [3.0] * 4, "fitted": False}
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
        for name, (_, bias) in COMPONENT_HEADS.items():
            self.params["W_" + name] = rng.normal(0, math.sqrt(0.5 / h2), (h2, 1))
            self.params["b_" + name] = np.array([bias])
        self.mu = np.zeros(in_dim)
        self.sigma = np.ones(in_dim)
        self.cameo_points = DEFAULT_CAMEO_POINTS
        self.component_constants = [0.0, 0.0, 0.0, 0.0]  # red/own-goal/penalty points by position, given 60+
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
        xp_direct = (a2 @ P["Wp"] + P["bp"])[:, 0]
        ph = _sigmoid((a2 @ P["Wh"] + P["bh"])[:, 0])
        heads: Dict[str, np.ndarray] = {}
        raw: Dict[str, np.ndarray] = {}
        for name, (link, _) in COMPONENT_HEADS.items():
            z = (a2 @ P["W_" + name] + P["b_" + name])[:, 0]
            raw[name] = z
            heads[name] = np.exp(np.clip(z, -8, 4)) if link == "poisson" else _sigmoid(z)
        pos = X[:, :4].argmax(axis=1)
        era = (X[:, _ERA_INDEX] > 0.5).astype(float)
        is_gk = (pos == 0).astype(float)
        is_gk_def = (pos <= 1).astype(float)
        xp_component = (
            2.0
            + heads["goals"] * GOAL_POINTS[pos]
            + heads["assists"] * 3.0
            + heads["cs"] * CS_POINTS[pos]
            - expected_floor_division(heads["gc"], 2) * is_gk_def
            + heads["bonus"]
            + expected_floor_division(heads["saves"], 3, k_max=15) * is_gk
            + 2.0 * heads["defcon"] * era * (1.0 - is_gk)
            - heads["yellow"]
            + np.asarray(self.component_constants)[pos]
        )
        if self.points_output == "direct":
            xp = xp_direct
        elif self.points_output == "component":
            xp = xp_component
        else:
            xp = 0.5 * (xp_direct + xp_component)
        return {"Xn": Xn, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "pm": pm, "xp": xp, "ph": ph,
                "xp_direct": xp_direct, "xp_component": xp_component, "heads": heads, "head_raw": raw,
                "pos": pos}

    def predict_batch(self, X: Sequence[Sequence[float]]) -> Dict[str, np.ndarray]:
        """Vectorised prediction. Returns p_none, p_cameo, p_full, xp_full, haul, expected_points, components."""
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[None, :]
        f = self._forward(Xa)
        pos = f["pos"]
        calibration = np.asarray(self.calibration["points"])[pos]
        xp_full = np.clip(calibration[:, 0] + calibration[:, 1] * f["xp"], XP_FULL_CLIP[0], XP_FULL_CLIP[1])
        pm = _softmax(np.log(np.maximum(f["pm"], 1e-12)) / self.calibration["minutes_temperature"])
        p_full = pm[:, 2]
        p_cameo = pm[:, 1]
        out = {
            "p_none": pm[:, 0],
            "p_cameo": p_cameo,
            "p_full": p_full,
            "xp_full": xp_full,
            "xp_direct": f["xp_direct"],
            "xp_component": f["xp_component"],
            "haul": np.clip(f["ph"], 0.01, 0.9),
            "expected_points": p_full * xp_full + p_cameo * self.cameo_points,
            "residual_std": np.asarray(self.calibration["residual_std"])[pos],
        }
        for name, values in f["heads"].items():
            out["c_" + name] = values
        return out

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
    @staticmethod
    def _component_matrix(samples: List[Sample]) -> np.ndarray:
        rows = np.zeros((len(samples), len(COMPONENT_NAMES)))
        for i, s in enumerate(samples):
            if s.components:
                rows[i, : len(s.components)] = s.components
        return rows

    def train(
        self,
        samples: List[Sample],
        epochs: int = 40,
        lr: float = 1e-3,
        l2: float = 1e-3,
        batch_size: int = 256,
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
        C = self._component_matrix(samples)
        ci = {name: i for i, name in enumerate(COMPONENT_NAMES)}
        y_m = np.where(mins <= 0, 0, np.where(mins < 60, 1, 2))
        full = (mins >= 60).astype(float)
        y_h = ((pts >= HAUL_THRESHOLD) & (mins >= 60)).astype(float)
        pos_all = X[:, :4].argmax(axis=1)
        self.baselines = {
            "points": float(pts.mean()),
            "minutes": (np.bincount(y_m, minlength=3) / len(samples)).tolist(),
            "haul": float(y_h[mins >= 60].mean()) if (mins >= 60).any() else 0.0,
        }
        cameo_rows = (mins > 0) & (mins < 60)
        if cameo_rows.sum() >= 20:
            self.cameo_points = float(pts[cameo_rows].mean())
        # Rare events are position constants: red cards, own goals, missed and saved penalties.
        rare = -3.0 * C[:, ci["red"]] - 2.0 * C[:, ci["own_goals"]] - 2.0 * C[:, ci["pen_missed"]] + 5.0 * C[:, ci["pen_saved"]]
        self.component_constants = [
            float(rare[(pos_all == p) & (full > 0)].mean()) if ((pos_all == p) & (full > 0)).sum() >= 50 else 0.0
            for p in range(4)
        ]
        targets = {
            "goals": C[:, ci["goals"]], "assists": C[:, ci["assists"]], "cs": (C[:, ci["clean_sheet"]] > 0).astype(float),
            "gc": C[:, ci["goals_conceded"]], "saves": C[:, ci["saves"]], "bonus": C[:, ci["bonus"]],
            "defcon": (C[:, ci["defcon"]] >= DEFCON_THRESHOLD[pos_all]).astype(float),
            "yellow": (C[:, ci["yellow"]] > 0).astype(float),
        }
        head_mask = {
            name: full.copy() for name in COMPONENT_HEADS
        }
        head_mask["saves"] = full * (pos_all == 0)
        head_mask["defcon"] = full * (C[:, ci["has_defcon"]] > 0) * (pos_all != 0)
        head_weight = {"goals": 0.5, "assists": 0.5, "cs": 0.4, "gc": 0.4, "saves": 0.3, "bonus": 0.5, "defcon": 0.4, "yellow": 0.2}

        self.mu = X.mean(axis=0)
        # Floor the scale: a feature that is nearly constant in a small early-season training
        # set must not turn a different live value into a huge standardised input.
        self.sigma = np.maximum(X.std(axis=0), 0.05)
        gws = np.asarray([s.gw for s in samples], dtype=float)
        years = np.asarray([s.period[0] for s in samples])
        # Season IDs never collide. Older seasons retain meaningful weight rather than
        # being nearly erased by a weekly decay applied across multiple years.
        w_time = (0.65 ** (years.max() - years)) * (0.995 ** (38 - gws))
        w_time /= w_time.mean()

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

                # Direct points head: MSE targets the conditional mean required by expected-points decisions.
                err = f["xp_direct"] - pts[idx]
                point_grad = err if self.points_loss == "mse" else np.clip(err, -huber_delta, huber_delta)
                d_xp = point_grad * fm * wt / n_full * 0.5

                # haul head: BCE on rows with 60+ minutes
                d_h = (f["ph"] - y_h[idx]) * fm * wt / n_full * 0.6

                # component heads: Poisson NLL (log link) or BCE, each on its own rows
                d_c: Dict[str, np.ndarray] = {}
                for name in COMPONENT_HEADS:
                    mask = head_mask[name][idx]
                    n_mask = max(1.0, mask.sum())
                    d_c[name] = (f["heads"][name] - targets[name][idx]) * mask * wt / n_mask * head_weight[name]

                g: Dict[str, np.ndarray] = {}
                a2 = f["a2"]
                g["Wm"] = a2.T @ d_m + l2 * P["Wm"]
                g["bm"] = d_m.sum(axis=0)
                g["Wp"] = a2.T @ d_xp[:, None] + l2 * P["Wp"]
                g["bp"] = np.array([d_xp.sum()])
                g["Wh"] = a2.T @ d_h[:, None] + l2 * P["Wh"]
                g["bh"] = np.array([d_h.sum()])
                da2 = d_m @ P["Wm"].T + d_xp[:, None] @ P["Wp"].T + d_h[:, None] @ P["Wh"].T
                for name, d in d_c.items():
                    g["W_" + name] = a2.T @ d[:, None] + l2 * P["W_" + name]
                    g["b_" + name] = np.array([d.sum()])
                    da2 = da2 + d[:, None] @ P["W_" + name].T

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
    def evaluate(self, samples: List[Sample]) -> Dict[str, Any]:
        """Scores predictions against ACTUAL points and minutes on `samples`.

        Overall metrics are dominated by zero-minute rows, so the decision-relevant slice
        (rows where the player played 60+ minutes) and per-position slices are reported too.
        """
        if not samples:
            return {}
        X = np.asarray([s.features for s in samples], dtype=float)
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

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-12 and b.std() > 1e-12 else 0.0

        out: Dict[str, Any] = {
            "n": float(len(samples)),
            "mae_points": float(np.abs(pred - actual).mean()),
            "mae_naive_baseline": float(np.abs(naive - actual).mean()),
            "mae_constant_baseline": float(np.abs(self.baselines["points"] - actual).mean()),
            "rmse_points": float(np.sqrt(((pred - actual) ** 2).mean())),
            "bias_points": float((pred - actual).mean()),
            "corr_points": corr(pred, actual),
            "corr_naive_baseline": corr(naive, actual),
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
        # Decision-relevant slice: the player actually played 60+ minutes. Conditional
        # predictions (xp_full) are compared with the naive points-per-appearance.
        if full.sum() >= 20:
            naive_full = np.asarray([s.naive_ppa if s.naive_ppa else s.naive_xp for s in samples])
            xp_full = r["xp_full"]
            a = actual[full]
            out["played60"] = {
                "n": int(full.sum()),
                "mae_points": float(np.abs(xp_full[full] - a).mean()),
                "mae_naive_baseline": float(np.abs(naive_full[full] - a).mean()),
                "bias_points": float((xp_full[full] - a).mean()),
                "corr_points": corr(xp_full[full], a),
                "corr_naive_baseline": corr(naive_full[full], a),
                "corr_direct_head": corr(r["xp_direct"][full], a),
                "corr_component_sum": corr(r["xp_component"][full], a),
            }
            pos = X[:, :4].argmax(axis=1)
            by_pos: Dict[str, Dict[str, float]] = {}
            for p, name in enumerate(POS_ORDER):
                mask = full & (pos == p)
                if mask.sum() >= 20:
                    by_pos[name] = {"n": int(mask.sum()),
                                    "mae_points": float(np.abs(xp_full[mask] - actual[mask]).mean()),
                                    "mae_naive_baseline": float(np.abs(naive_full[mask] - actual[mask]).mean()),
                                    "corr_points": corr(xp_full[mask], actual[mask]),
                                    "corr_naive_baseline": corr(naive_full[mask], actual[mask]),
                                    "bias_points": float((xp_full[mask] - actual[mask]).mean())}
            out["played60_by_position"] = by_pos
            C = self._component_matrix(samples)
            if C.any():
                ci = {name: i for i, name in enumerate(COMPONENT_NAMES)}
                comp = {}
                for head, column in (("goals", "goals"), ("assists", "assists"), ("bonus", "bonus"), ("gc", "goals_conceded")):
                    comp[head] = {"mean_predicted": float(r["c_" + head][full].mean()),
                                  "mean_actual": float(C[full, ci[column]].mean())}
                comp["cs"] = {"mean_predicted": float(r["c_cs"][full].mean()),
                              "mean_actual": float((C[full, ci["clean_sheet"]] > 0).mean())}
                out["component_calibration_played60"] = comp
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}

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
            "component_constants": self.component_constants,
            "baselines": self.baselines,
            "points_loss": self.points_loss,
            "points_output": self.points_output,
            "calibration": self.calibration,
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
        net = cls(in_dim=in_dim, h1=h1, h2=h2, points_loss=data["points_loss"],
                  points_output=data.get("points_output", "direct"))
        net.params = {k: np.asarray(v, dtype=float) for k, v in data["params"].items()}
        net.mu = np.asarray(data["mu"], dtype=float)
        net.sigma = np.asarray(data["sigma"], dtype=float)
        net.cameo_points = float(data.get("cameo_points", DEFAULT_CAMEO_POINTS))
        net.component_constants = list(data.get("component_constants", [0.0] * 4))
        net.baselines = data["baselines"]
        net.calibration = data["calibration"]
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
        "architecture": f"{N_FEATURES} -> 48 -> 24 -> [minutes(3) | points | haul | components]",
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
    """Fit completed historical seasons only; live weeks are inference/evaluation inputs."""
    from historical_training import archive_signature, train_historical_model
    signature = archive_signature()
    if NN_WEIGHTS_PATH.exists() and not force_retrain:
        try:
            net = FPLNeuralNetwork.load(NN_WEIGHTS_PATH)
            if net.metrics.get("training_signature") == signature:
                return net
        except Exception:
            pass
    net, _ = train_historical_model(verbose=verbose)
    net.save(NN_WEIGHTS_PATH)
    return net


if __name__ == "__main__":
    import time

    t0 = time.time()
    net = get_or_train_neural_net(force_retrain=True, verbose=True)
    print(f"=== Multi-head FPL model trained in {time.time() - t0:.1f}s ===")
    print(json.dumps(net.metrics, indent=2))
