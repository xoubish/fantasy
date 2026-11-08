"""Cross-season, point-in-time team strength model (team_ratings.py).

A time-decayed Poisson attack/defence model (Dixon-Coles style, independent Poisson):

    log E[home goals] = mu + home_adv + attack[home] - defence[away]
    log E[away goals] = mu +            attack[away] - defence[home]

Fitted on every finished Premier League match whose kickoff precedes a cutoff, across
seasons, so ratings do not restart from scratch each August. Match weights decay with a
fixed half-life. Team parameters are ridge-shrunk toward zero, or toward a weaker
"promoted" prior for teams that played no Premier League match in the previous season.
Stable FPL team codes identify clubs across seasons.

Nothing after the cutoff is used; callers pass the deadline of the gameweek they are
predicting so backtests stay point-in-time.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
ARCHIVE_DIR = DATA_DIR / "historical"

HALF_LIFE_DAYS = 240.0  # selected on 2022-23..2024-25 one-step-ahead match log-likelihood (see select_half_life)
RIDGE = 2.5  # pseudo-information pulling each team parameter toward its prior mean
PROMOTED_PRIOR = -0.30  # log-scale prior for attack and defence of a newly promoted club
DEFAULT_LAMBDA = 1.35  # league-average goals per team per match, used when no data exists
RATING_SCALE_CENTRE = 1150
RATING_SCALE_SLOPE = 200


@dataclass(frozen=True)
class Match:
    kickoff: datetime
    home: int  # stable team code
    away: int
    home_goals: int
    away_goals: int


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def season_of(moment: datetime) -> int:
    """Football season start year of a kickoff (July onward belongs to the new season)."""
    return moment.year if moment.month >= 7 else moment.year - 1


def _is_finished(f: Dict[str, Any]) -> bool:
    for key in ("finished", "finished_provisional"):
        val = f.get(key)
        if isinstance(val, str):
            if val.lower() == "true":
                return True
        elif val:
            return True
    return False


def matches_from_fixtures(fixtures: Iterable[Dict[str, Any]], code_of_team: Dict[int, int]) -> List[Match]:
    """Finished fixtures (API or archive rows) to Match records keyed by stable team code."""
    out: List[Match] = []
    for f in fixtures:
        if not _is_finished(f) or f.get("team_h_score") in (None, "") or not f.get("kickoff_time"):
            continue
        try:
            home, away = code_of_team[int(f["team_h"])], code_of_team[int(f["team_a"])]
        except (KeyError, ValueError):
            continue
        out.append(Match(_parse_time(f["kickoff_time"]), home, away, int(float(f["team_h_score"])), int(float(f["team_a_score"]))))
    return out


def _archive_signature() -> Tuple[Tuple[str, int, int], ...]:
    rows = []
    for path in sorted(ARCHIVE_DIR.glob("*/fixtures.csv")):
        stat = path.stat()
        rows.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(rows)


@lru_cache(maxsize=4)
def _archive_matches(signature: Tuple[Tuple[str, int, int], ...]) -> Tuple[Match, ...]:
    matches: List[Match] = []
    for path, _, _ in signature:
        season_dir = Path(path).parent
        teams_path = season_dir / "teams.csv"
        if not teams_path.exists():
            continue
        with teams_path.open(encoding="utf-8-sig", newline="") as handle:
            codes = {int(float(r["id"])): int(float(r["code"])) for r in csv.DictReader(handle) if r.get("code")}
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            matches.extend(matches_from_fixtures(csv.DictReader(handle), codes))
    return tuple(matches)


def archived_matches() -> List[Match]:
    """Every finished match in the pinned historical archive (all seasons, including context-only ones)."""
    return list(_archive_matches(_archive_signature()))


def combined_matches(live_fixtures: Optional[Sequence[Dict[str, Any]]] = None,
                     live_teams: Optional[Sequence[Dict[str, Any]]] = None) -> List[Match]:
    """Archive matches plus finished live-season fixtures, de-duplicated by kickoff and teams."""
    matches = {(m.kickoff, m.home, m.away): m for m in archived_matches()}
    if live_fixtures and live_teams:
        codes = {int(t["id"]): int(t["code"]) for t in live_teams if t.get("code") is not None}
        for m in matches_from_fixtures(live_fixtures, codes):
            matches[(m.kickoff, m.home, m.away)] = m
    return sorted(matches.values(), key=lambda m: m.kickoff)


class PoissonRatings:
    """Attack/defence ratings fitted on matches strictly before `cutoff`."""

    def __init__(self, matches: Sequence[Match], cutoff: datetime, half_life_days: float = HALF_LIFE_DAYS,
                 ridge: float = RIDGE, promoted_prior: float = PROMOTED_PRIOR):
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        self.cutoff = cutoff
        self.half_life_days = half_life_days
        used = [m for m in matches if m.kickoff < cutoff]
        self.n_matches = len(used)
        teams = sorted({m.home for m in used} | {m.away for m in used})
        self.index = {code: i for i, code in enumerate(teams)}
        self.mu = math.log(DEFAULT_LAMBDA)
        self.home_adv = 0.0
        self.attack = np.zeros(len(teams))
        self.defence = np.zeros(len(teams))
        self.prior_mean = np.zeros(len(teams))
        self.weight_by_team = np.zeros(len(teams))
        if not used:
            return
        current = season_of(cutoff - timedelta(seconds=1))
        played_last_season = {m.home for m in used if season_of(m.kickoff) == current - 1} | \
                             {m.away for m in used if season_of(m.kickoff) == current - 1}
        any_last_season = any(season_of(m.kickoff) == current - 1 for m in used)
        for code, i in self.index.items():
            if any_last_season and code not in played_last_season:
                self.prior_mean[i] = promoted_prior
        h = np.array([self.index[m.home] for m in used])
        a = np.array([self.index[m.away] for m in used])
        hg = np.array([m.home_goals for m in used], dtype=float)
        ag = np.array([m.away_goals for m in used], dtype=float)
        days = np.array([(cutoff - m.kickoff).total_seconds() / 86400.0 for m in used])
        w = 0.5 ** (days / half_life_days)
        self.weight_by_team = np.bincount(h, weights=w, minlength=len(teams)) + np.bincount(a, weights=w, minlength=len(teams))
        self._fit(h, a, hg, ag, w, ridge)

    def _fit(self, h, a, hg, ag, w, ridge) -> None:
        n = len(self.index)
        from scipy.optimize import minimize

        def unpack(theta):
            return theta[0], theta[1], theta[2:2 + n], theta[2 + n:]

        def objective(theta):
            mu, adv, att, dfn = unpack(theta)
            lh = np.exp(mu + adv + att[h] - dfn[a])
            la = np.exp(mu + att[a] - dfn[h])
            nll = np.sum(w * (lh - hg * np.log(lh))) + np.sum(w * (la - ag * np.log(la)))
            pen = ridge * (np.sum((att - self.prior_mean) ** 2) + np.sum((dfn - self.prior_mean) ** 2))
            rh, ra = w * (lh - hg), w * (la - ag)
            g = np.zeros_like(theta)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            g_att = np.bincount(h, weights=rh, minlength=n) + np.bincount(a, weights=ra, minlength=n)
            g_def = -np.bincount(a, weights=rh, minlength=n) - np.bincount(h, weights=ra, minlength=n)
            g[2:2 + n] = g_att + 2 * ridge * (att - self.prior_mean)
            g[2 + n:] = g_def + 2 * ridge * (dfn - self.prior_mean)
            return nll + pen, g

        theta0 = np.concatenate([[self.mu, 0.2], self.prior_mean, self.prior_mean])
        result = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.mu, self.home_adv, att, dfn = unpack(result.x)
        self.attack, self.defence = np.array(att), np.array(dfn)

    # ---- queries ----
    def _params(self, code: Optional[int]) -> Tuple[float, float]:
        if code is None or code not in self.index:
            prior = PROMOTED_PRIOR if self.n_matches else 0.0
            return prior, prior
        i = self.index[code]
        return float(self.attack[i]), float(self.defence[i])

    def expected_goals(self, home_code: Optional[int], away_code: Optional[int]) -> Tuple[float, float]:
        """(home goals, away goals) expected in a match between the two clubs."""
        ah, dh = self._params(home_code)
        aa, da = self._params(away_code)
        return math.exp(self.mu + self.home_adv + ah - da), math.exp(self.mu + aa - dh)

    def fixture_context(self, team_code: Optional[int], opponent_code: Optional[int], is_home: bool) -> Dict[str, Any]:
        """Feature inputs for one side of a fixture, plus legacy display ratings."""
        if is_home:
            lam_for, lam_against = self.expected_goals(team_code, opponent_code)
        else:
            lam_against, lam_for = self.expected_goals(opponent_code, team_code)
        own, opp = self.display_ratings(team_code), self.display_ratings(opponent_code)
        return {
            "is_home": bool(is_home),
            "lambda_for": round(lam_for, 4),
            "lambda_against": round(lam_against, 4),
            "team_att": own["att_home" if is_home else "att_away"],
            "team_def": own["def_home" if is_home else "def_away"],
            "opp_att": opp["att_away" if is_home else "att_home"],
            "opp_def": opp["def_away" if is_home else "def_home"],
        }

    def display_ratings(self, code: Optional[int]) -> Dict[str, int]:
        """Legacy 950-1350 style ratings derived from the log-scale parameters (display only)."""
        att, dfn = self._params(code)
        half = self.home_adv / 2.0
        scale = RATING_SCALE_SLOPE
        return {
            "att_home": int(round(RATING_SCALE_CENTRE + scale * (att + half))),
            "def_home": int(round(RATING_SCALE_CENTRE + scale * (dfn + half))),
            "att_away": int(round(RATING_SCALE_CENTRE + scale * (att - half))),
            "def_away": int(round(RATING_SCALE_CENTRE + scale * (dfn - half))),
        }

    def table(self, names: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
        rows = []
        for code, i in self.index.items():
            rows.append({"code": code, "name": (names or {}).get(code, str(code)),
                         "attack": round(float(self.attack[i]), 3), "defence": round(float(self.defence[i]), 3),
                         "weighted_matches": round(float(self.weight_by_team[i]), 1)})
        return sorted(rows, key=lambda r: -(r["attack"] + r["defence"]))


_FIT_CACHE: Dict[Tuple[Any, ...], PoissonRatings] = {}


def ratings_at(cutoff: datetime, matches: Optional[Sequence[Match]] = None,
               half_life_days: float = HALF_LIFE_DAYS) -> PoissonRatings:
    """Cached fit on all known matches before `cutoff`."""
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    pool = list(matches) if matches is not None else archived_matches()
    key = (cutoff, half_life_days, len(pool), pool[-1].kickoff if pool else None, id(matches) if matches is not None else 0)
    if key not in _FIT_CACHE:
        if len(_FIT_CACHE) > 512:
            _FIT_CACHE.clear()
        _FIT_CACHE[key] = PoissonRatings(pool, cutoff, half_life_days=half_life_days)
    return _FIT_CACHE[key]


def select_half_life(candidates: Sequence[float] = (120.0, 240.0, 365.0), seasons: Sequence[int] = (2022, 2023, 2024),
                     step_days: int = 7) -> Dict[str, Any]:
    """One-step-ahead Poisson log-likelihood of each half-life on development seasons only.

    The 2025-26 evaluation season and the live season are deliberately excluded, so this
    choice never sees any row from a held-out test window.
    """
    matches = archived_matches()
    report: Dict[str, Any] = {"candidates": {}, "seasons": list(seasons)}
    for half_life in candidates:
        total, count = 0.0, 0
        for season in seasons:
            season_matches = [m for m in matches if season_of(m.kickoff) == season]
            if not season_matches:
                continue
            cutoff = min(m.kickoff for m in season_matches)
            end = max(m.kickoff for m in season_matches)
            while cutoff <= end:
                nxt = cutoff + timedelta(days=step_days)
                block = [m for m in season_matches if cutoff <= m.kickoff < nxt]
                if block:
                    model = PoissonRatings(matches, cutoff, half_life_days=half_life)
                    for m in block:
                        lh, la = model.expected_goals(m.home, m.away)
                        total += (m.home_goals * math.log(lh) - lh - math.lgamma(m.home_goals + 1)
                                  + m.away_goals * math.log(la) - la - math.lgamma(m.away_goals + 1))
                        count += 1
                cutoff = nxt
        report["candidates"][str(half_life)] = {"matches": count, "mean_loglik": round(total / max(1, count), 5)}
    best = max(report["candidates"], key=lambda k: report["candidates"][k]["mean_loglik"])
    report["selected_half_life_days"] = float(best)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["select", "table"])
    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(select_half_life(), indent=2))
    else:
        matches = archived_matches()
        now = datetime.now(timezone.utc)
        model = ratings_at(now, matches)
        print(json.dumps(model.table(), indent=1))
