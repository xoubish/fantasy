"""Bookmaker market context for fixtures (market_odds.py).

Source: football-data.co.uk Premier League CSVs (one per season) and its upcoming
`fixtures.csv`. The average pre-match 1X2 and over/under 2.5 prices across bookmakers
are converted to implied expected goals for each side by matching two Poisson
quantities: P(total goals >= 3) and P(home win). Prices are the ones football-data
collects before the weekend, not closing prices, so they do not contain line-up news
published after the FPL deadline.

Files live in data/odds/ and are listed in data/odds/odds_manifest.json. Completed
seasons are kept as downloaded; the current season and the fixtures file are refreshed
by every sync. A fixture without a matching market row yields None, and the feature
builder then falls back to the Poisson team model with a flag.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import urllib.request
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"
ODDS_DIR = DATA_DIR / "odds"
ODDS_MANIFEST = ODDS_DIR / "odds_manifest.json"
BASE_URL = "https://www.football-data.co.uk"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"

# football-data club names -> FPL club names (others are identical)
NAME_MAP = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Sheffield United": "Sheffield Utd",
    "Nottingham Forest": "Nott'm Forest",
    "Luton": "Luton Town",
    "Leicester": "Leicester",
    "Wolverhampton": "Wolves",
}


def fpl_name(name: str) -> str:
    return NAME_MAP.get(name.strip(), name.strip())


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def sync_odds(season_years: Iterable[int], current_year: Optional[int] = None, refresh_completed: bool = False) -> Dict[str, Any]:
    """Download season files (kept once present unless refresh_completed) and the live fixtures file."""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(ODDS_MANIFEST.read_text()) if ODDS_MANIFEST.exists() else {"files": {}}
    manifest.setdefault("files", {})
    manifest["source"] = BASE_URL
    errors: List[str] = []
    jobs: List[Tuple[str, str, bool]] = []
    for year in sorted(set(season_years)):
        always = current_year is not None and year >= current_year
        jobs.append((f"E0_{year}.csv", f"{BASE_URL}/mmz4281/{season_code(year)}/E0.csv", always or refresh_completed))
    jobs.append(("fixtures.csv", f"{BASE_URL}/fixtures.csv", True))
    for name, url, force in jobs:
        path = ODDS_DIR / name
        if path.exists() and not force:
            continue
        try:
            content = _download(url)
            rows = list(csv.DictReader(content.decode("utf-8-sig").splitlines()))
            if not rows or "HomeTeam" not in rows[0]:
                raise ValueError("unexpected file layout")
            path.write_bytes(content)
            manifest["files"][name] = {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "rows": len(rows),
                                       "downloaded_at": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:  # keep the previous file when the site is unavailable
            errors.append(f"{name}: {exc}")
    manifest["errors"] = errors
    ODDS_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    _load_market.cache_clear()
    return manifest


# --------------------------------------------------------------------------------------
# Implied expected goals
# --------------------------------------------------------------------------------------
def _implied(prices: List[Optional[float]]) -> Optional[List[float]]:
    if any(p is None or p <= 1.0 for p in prices):
        return None
    raw = [1.0 / p for p in prices]
    total = sum(raw)
    return [r / total for r in raw]


def _poisson_pmf(lam: float, k_max: int = 12) -> List[float]:
    out, term = [], math.exp(-lam)
    for k in range(k_max + 1):
        out.append(term)
        term *= lam / (k + 1)
    return out


def _p_over(total: float, line_goals: int = 3) -> float:
    pmf = _poisson_pmf(total)
    return 1.0 - sum(pmf[:line_goals])


def _p_home_win(lh: float, la: float) -> float:
    ph, pa = _poisson_pmf(lh), _poisson_pmf(la)
    return sum(ph[i] * pa[j] for i in range(len(ph)) for j in range(i))


def implied_lambdas(p_home: float, p_over25: float) -> Tuple[float, float]:
    """Expected goals (home, away) whose independent Poissons reproduce P(home win) and P(3+ goals)."""
    lo, hi = 0.3, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _p_over(mid) < p_over25:
            lo = mid
        else:
            hi = mid
    total = (lo + hi) / 2
    lo, hi = 0.05, total - 0.05
    for _ in range(50):
        mid = (lo + hi) / 2
        if _p_home_win(mid, total - mid) < p_home:
            lo = mid
        else:
            hi = mid
    lh = (lo + hi) / 2
    return lh, total - lh


def _number(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "", "None") else None
    except ValueError:
        return None


def _market_row(row: Dict[str, Any]) -> Optional[Dict[str, float]]:
    result = _implied([_number(row.get("AvgH")), _number(row.get("AvgD")), _number(row.get("AvgA"))]) \
        or _implied([_number(row.get("B365H")), _number(row.get("B365D")), _number(row.get("B365A"))])
    if result is None:
        return None
    over = _implied([_number(row.get("Avg>2.5")), _number(row.get("Avg<2.5"))]) \
        or _implied([_number(row.get("B365>2.5")), _number(row.get("B365<2.5"))])
    p_over = over[0] if over else 0.52  # league-average share of 3+ goal matches when no total line exists
    lh, la = implied_lambdas(result[0], p_over)
    return {"lambda_home": round(lh, 4), "lambda_away": round(la, 4), "p_home": round(result[0], 4),
            "p_draw": round(result[1], 4), "p_away": round(result[2], 4), "total_line_available": 1.0 if over else 0.0}


def _parse_date(value: str) -> Optional[datetime]:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _signature() -> Tuple[Tuple[str, int, int], ...]:
    if not ODDS_DIR.exists():
        return ()
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(ODDS_DIR.glob("*.csv")))


@lru_cache(maxsize=2)
def _load_market(signature: Tuple[Tuple[str, int, int], ...]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    table: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for name, _, _ in signature:
        with (ODDS_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Div") not in (None, "", "E0"):
                    continue
                date = _parse_date(row.get("Date", ""))
                if date is None or not row.get("HomeTeam"):
                    continue
                market = _market_row(row)
                if market is None:
                    continue
                key = (date.strftime("%Y-%m-%d"), fpl_name(row["HomeTeam"]), fpl_name(row["AwayTeam"]))
                # Season files take precedence over the fixtures file for the same match.
                if key not in table or name.startswith("E0_"):
                    table[key] = market
    return table


def market_for_match(kickoff_iso: str, home_name: str, away_name: str) -> Optional[Dict[str, float]]:
    """Market expected goals for a match identified by kickoff date (UTC) and FPL club names."""
    if not kickoff_iso:
        return None
    try:
        kickoff = datetime.fromisoformat(str(kickoff_iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    table = _load_market(_signature())
    home, away = fpl_name(home_name), fpl_name(away_name)
    for offset in (0, 1, -1):
        key = ((kickoff + timedelta(days=offset)).strftime("%Y-%m-%d"), home, away)
        if key in table:
            return table[key]
    return None


def market_context(kickoff_iso: str, home_name: str, away_name: str, is_home: bool) -> Dict[str, Any]:
    """Feature inputs for one side of a fixture; `market_available` is 0 when no odds were found."""
    market = market_for_match(kickoff_iso, home_name, away_name)
    if market is None:
        return {"market_available": 0.0}
    lam_for = market["lambda_home"] if is_home else market["lambda_away"]
    lam_against = market["lambda_away"] if is_home else market["lambda_home"]
    p_win = market["p_home"] if is_home else market["p_away"]
    return {"market_available": 1.0, "market_lambda_for": lam_for, "market_lambda_against": lam_against,
            "market_p_win": p_win, "market_p_draw": market["p_draw"]}


def coverage(matches: Iterable[Tuple[str, str, str]]) -> Dict[str, Any]:
    """Share of (kickoff, home, away) tuples with a market row; useful for checking name mapping."""
    found, missing = 0, []
    for kickoff, home, away in matches:
        if market_for_match(kickoff, home, away) is None:
            missing.append((kickoff, home, away))
        else:
            found += 1
    return {"found": found, "missing": len(missing), "examples": missing[:10]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sync", "check"])
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    args = parser.parse_args()
    from historical_data import current_season_start
    current = current_season_start()
    years = args.seasons or list(range(current - 5, current + 1))
    if args.command == "sync":
        result = sync_odds(years, current_year=current)
        print(json.dumps({k: v for k, v in result.items() if k != "files"} | {"files": list(result["files"])}, indent=2))
    else:
        from team_ratings import archived_matches
        from historical_data import load_season
        import json as _json
        bootstrap = _json.loads((DATA_DIR / "bootstrap_static.json").read_text())
        live_names = {int(t["id"]): t["name"] for t in bootstrap["teams"]}
        fixtures = _json.loads((DATA_DIR / "fixtures.json").read_text())
        live = [(f["kickoff_time"], live_names[int(f["team_h"])], live_names[int(f["team_a"])]) for f in fixtures if f.get("kickoff_time")]
        print("live season:", coverage(live))
        for season in ("2022-23", "2023-24", "2024-25", "2025-26"):
            archive = load_season(season)
            names = {tid: t["name"] for tid, t in archive["teams"].items()}
            rows = [(f["kickoff_time"], names[f["team_h"]], names[f["team_a"]]) for f in archive["fixtures"]]
            print(season, coverage(rows))
