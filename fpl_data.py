"""
FPL Live Data Collector & Real Premier League Roster Loader (fpl_data.py)

Strictly loads ONLY active players belonging to the 20 current Premier League clubs
from the official Fantasy Premier League API (`bootstrap-static/`, `fixtures/`,
and per-player `element-summary/{id}/` match logs).

Filters out any player who has left the Premier League (`status == 'u'`,
`removed == True`, or `can_select == False`).
"""

from __future__ import annotations

import concurrent.futures
import csv
import io
import gzip
import json
import sqlite3
import time
import urllib.request
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "fpl.db"
CACHE_BOOTSTRAP = DATA_DIR / "bootstrap_static.json"
CACHE_FIXTURES = DATA_DIR / "fixtures.json"
CACHE_HISTORICAL = DATA_DIR / "historical_3yr.json"
CACHE_SUMMARIES = DATA_DIR / "player_summaries.json"

FPL_BASE_URL = "https://fantasy.premierleague.com/api"
VAASTAV_BASE_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)
HISTORICAL_SEASONS = ["2023-24", "2024-25"]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Number of upcoming gameweeks attached to every player (used by the transfer planner)
HORIZON_GWS = 5


@dataclass
class HistoricalSeasonStat:
    season_name: str
    total_points: int
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    expected_goals: float
    expected_assists: float
    bonus: int
    expected_goals_conceded: float = 0.0
    defensive_contribution: float = 0.0
    ict_index: float = 0.0
    bps: int = 0
    saves: int = 0
    yellow_cards: int = 0

    @property
    def points_per_90(self) -> float:
        if self.minutes < 270:
            return 0.0
        return round((self.total_points / self.minutes) * 90.0, 2)


@dataclass
class PlayerRecord:
    id: int
    web_name: str
    full_name: str
    team_id: int
    team_short: str
    team_name: str
    position: str  # GKP, DEF, MID, FWD
    price: float  # in millions, e.g. 8.1
    status: str  # 'a' available, 'd' doubtful, 'i' injured, 's' suspended
    chance_of_playing_next_round: Optional[int]
    fpl_news: str
    total_points: int
    minutes: int
    starts: int
    points_per_game: float
    form: float
    value_form: float  # form points per £m
    value_season: float  # total points per £m
    selected_by_percent: float
    xg_per_90: float
    xa_per_90: float
    xgi_per_90: float
    xgc_per_90: float
    defensive_contribution_per_90: float  # CBIT + tackles + recoveries per 90
    ict_per_90: float
    threat_per_90: float
    creativity_per_90: float
    saves_per_90: float
    clean_sheets: int
    bps_per_90: float
    bonus: int
    in_dreamteam: bool
    penalties_order: Optional[int]
    corners_order: Optional[int]
    direct_freekicks_order: Optional[int]
    next_opponent: str  # e.g., "HUL (H)" or "CRY (A)"
    is_home: bool
    fixture_difficulty: int  # 1 (easiest) to 5 (hardest)
    team_attack_strength: int  # Own team attack strength in this venue
    team_defence_strength: int  # Own team defence strength in this venue
    opp_attack_strength: int  # Opponent attack strength in this venue
    opp_defence_strength: int  # Opponent defence strength in this venue
    historical_seasons: List[HistoricalSeasonStat] = field(default_factory=list)
    gw_history: List[Dict[str, Any]] = field(default_factory=list)
    upcoming_3gw_fixtures: List[Dict[str, Any]] = field(default_factory=list)  # next HORIZON_GWS gameweeks
    fixture_run_3gw: str = ""
    target_gw: int = 0
    lambda_for: float = 1.35  # Poisson expected goals for the player's team in the next fixture
    lambda_against: float = 1.35  # Poisson expected goals against
    market_available: float = 0.0  # 1 when bookmaker prices exist for the next fixture
    market_lambda_for: float = 1.35
    market_lambda_against: float = 1.35

    @property
    def historical_ppg_3yr(self) -> float:
        """Weighted points-per-90 across past seasons + current season."""
        total_mins = sum(s.minutes for s in self.historical_seasons) + self.minutes
        total_pts = (
            sum(s.total_points for s in self.historical_seasons) + self.total_points
        )
        if total_mins < 270:
            return self.points_per_game
        return round((total_pts / total_mins) * 90.0, 2)

    @property
    def attack_mismatch_advantage(self) -> float:
        """Positive when player's team attack is stronger than opponent's defence."""
        return (self.team_attack_strength - self.opp_defence_strength) / 100.0

    @property
    def defence_mismatch_advantage(self) -> float:
        """Positive when player's team defence is stronger than opponent's attack."""
        return (self.team_defence_strength - self.opp_attack_strength) / 100.0


def _http_get_json(url: str, timeout: int = 12) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_text(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_player_summaries_concurrent(
    player_ids: List[int],
    latest_finished_gw: int = 0,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetches `element-summary/{id}/` (which contains `history` for every GW this season
    and `history_past` for prior seasons) concurrently for active Premier League players.
    Refresh every requested player on each sync. A gameweek number cannot certify a
    complete double gameweek or finalized bonus points. Old entries remain a fallback
    if an individual request fails. The legacy freshness arguments remain compatible.
    """
    existing: Dict[str, Any] = {}
    if CACHE_SUMMARIES.exists():
        try:
            existing = json.loads(CACHE_SUMMARIES.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    to_fetch = list(dict.fromkeys(player_ids))
    if not to_fetch:
        return existing

    def _fetch_one(pid: int) -> tuple[str, Any]:
        data = _http_get_json(f"{FPL_BASE_URL}/element-summary/{pid}/", timeout=10)
        return str(pid), {
            "history": data.get("history", []),
            "history_past": data.get("history_past", []),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_one, pid): pid for pid in to_fetch}
        for fut in concurrent.futures.as_completed(futures):
            try:
                k, val = fut.result()
                existing[k] = val
            except Exception:
                failed.append(futures[fut])

    if failed:
        warnings.warn(f"Could not refresh {len(failed)} player histories; retained any cached entries.")

    CACHE_SUMMARIES.write_text(json.dumps(existing), encoding="utf-8")
    return existing


def sync_live_fpl_data(
    fetch_player_histories: bool = True, force_refresh_summaries: bool = False
) -> Dict[str, Any]:
    """
    Downloads live FPL bootstrap-static, fixtures, and per-player match histories (`element-summary`).
    Automatically detects newly finished gameweeks and appends last week's match data.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "live_fpl_synced": False,
        "historical_synced": False,
        "summaries_synced": 0,
        "latest_finished_gw": 0,
        "errors": [],
    }

    try:
        bootstrap = _http_get_json(f"{FPL_BASE_URL}/bootstrap-static/")
        CACHE_BOOTSTRAP.write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")

        fixtures = _http_get_json(f"{FPL_BASE_URL}/fixtures/")
        CACHE_FIXTURES.write_text(json.dumps(fixtures, indent=2), encoding="utf-8")
        status["live_fpl_synced"] = True
    except Exception as exc:
        status["errors"].append(f"FPL API fetch failed: {exc}")

    # Include unused players: zero-minute rows are needed by the availability model.
    if (status["live_fpl_synced"] or CACHE_BOOTSTRAP.exists()) and fetch_player_histories:
        try:
            bootstrap = json.loads(CACHE_BOOTSTRAP.read_text(encoding="utf-8"))
            fixtures = (
                json.loads(CACHE_FIXTURES.read_text(encoding="utf-8"))
                if CACHE_FIXTURES.exists()
                else []
            )
            latest_finished_gw = max(
                (
                    int(f.get("event", 0) or 0)
                    for f in fixtures
                    if (f.get("finished") or f.get("finished_provisional"))
                    and f.get("team_h_score") is not None
                ),
                default=0,
            )
            status["latest_finished_gw"] = latest_finished_gw

            active_ids = [
                int(el["id"])
                for el in bootstrap.get("elements", [])
                if el.get("status") != "u"
                and not el.get("removed")
            ]
            summaries = fetch_player_summaries_concurrent(
                active_ids,
                latest_finished_gw=latest_finished_gw,
                force_refresh=force_refresh_summaries,
            )
            status["summaries_synced"] = len(summaries)
            status["historical_synced"] = True
        except Exception as exc:
            status["errors"].append(f"Element summaries fetch warning: {exc}")

    if status["live_fpl_synced"]:
        status["snapshot"] = str(archive_live_snapshot())
    try:
        from historical_data import current_season_start
        from market_odds import sync_odds
        current = current_season_start()
        odds = sync_odds(range(current - 5, current + 1), current_year=current)
        status["odds_errors"] = odds.get("errors", [])
    except Exception as exc:  # odds are optional inputs; the model falls back to team ratings
        status["errors"].append(f"Odds sync warning: {exc}")
    return status


def archive_live_snapshot() -> Path:
    """Preserve timestamped public API inputs for future deadline-aware evaluation.

    Per-player fetched_at distinguishes refreshed histories from a failed-request
    fallback. These files are evidence for future evaluation, not training inputs.
    """
    stamp = datetime.now(timezone.utc)
    directory = DATA_DIR / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (stamp.strftime("%Y%m%dT%H%M%S%fZ") + ".json.gz")
    payload = {"observed_at": stamp.isoformat(),
               "bootstrap": json.loads(CACHE_BOOTSTRAP.read_text()),
               "fixtures": json.loads(CACHE_FIXTURES.read_text()),
               "player_summaries": json.loads(CACHE_SUMMARIES.read_text()) if CACHE_SUMMARIES.exists() else {}}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


NEUTRAL_FIXTURE = {"opponent": "TBD", "is_home": True, "difficulty": 3, "team_att": 1150, "team_def": 1150,
                   "opp_att": 1150, "opp_def": 1150, "lambda_for": 1.35, "lambda_against": 1.35}


def team_model(
    bootstrap: Dict[str, Any],
    fixtures: List[Dict[str, Any]],
    max_gw: Optional[int] = None,
    cutoff: Optional[datetime] = None,
):
    """Cross-season Poisson team ratings fitted on matches before `cutoff`.

    With `max_gw`, the cutoff is the official deadline of gameweek `max_gw + 1`, so a
    walk-forward backtest sees exactly the results a manager had at that deadline.
    Returns (PoissonRatings, {team id: stable team code}).
    """
    from team_ratings import combined_matches, ratings_at

    teams = [t for t in bootstrap.get("teams", [])]
    codes = {int(t["id"]): int(t["code"]) for t in teams if t.get("code") is not None}
    if cutoff is None:
        cutoff = datetime.now(timezone.utc)
        if max_gw is not None:
            deadline = next((e.get("deadline_time") for e in bootstrap.get("events", []) if int(e.get("id", 0)) == max_gw + 1), None)
            if deadline:
                cutoff = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
            else:
                kickoffs = [f["kickoff_time"] for f in fixtures if int(f.get("event", 0) or 0) == max_gw + 1 and f.get("kickoff_time")]
                if kickoffs:
                    cutoff = datetime.fromisoformat(min(kickoffs).replace("Z", "+00:00"))
    matches = combined_matches(fixtures, teams)
    return ratings_at(cutoff, matches), codes


def fixture_context(model, codes: Dict[int, int], team_id: int, opponent_id: int, is_home: bool,
                    kickoff: str = "", names: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """Feature inputs for `team_id` against `opponent_id`: Poisson team model plus bookmaker market
    context when `kickoff` and club `names` are supplied and prices exist."""
    from market_odds import market_context
    out = model.fixture_context(codes.get(int(team_id)), codes.get(int(opponent_id)), is_home)
    out["kickoff"] = kickoff
    if kickoff and names:
        home_id, away_id = (team_id, opponent_id) if is_home else (opponent_id, team_id)
        out |= market_context(kickoff, names.get(int(home_id), ""), names.get(int(away_id), ""), is_home)
    else:
        out["market_available"] = 0.0
    return out


def compute_team_strengths(
    bootstrap: Dict[str, Any],
    fixtures: List[Dict[str, Any]],
    max_gw: Optional[int] = None,
) -> Dict[int, Dict[str, int]]:
    """Display ratings (950-1350 style) for every current team, derived from `team_model`."""
    model, codes = team_model(bootstrap, fixtures, max_gw=max_gw)
    return {tid: model.display_ratings(codes.get(tid)) for tid in (int(t["id"]) for t in bootstrap.get("teams", []))}


def next_editable_gameweek(bootstrap: Dict[str, Any], now: Optional[datetime] = None) -> int:
    """Choose the first deadline still open, irrespective of fixtures in progress."""
    now = now or datetime.now(timezone.utc)
    future = []
    for event in bootstrap.get("events", []):
        deadline = event.get("deadline_time")
        if deadline and datetime.fromisoformat(deadline.replace("Z", "+00:00")) > now:
            future.append((deadline, int(event["id"])))
    if not future:
        raise RuntimeError("No future gameweek deadline in the cache. Sync data or specify target_gw for historical analysis.")
    return min(future)[1]


def load_players(target_gw: Optional[int] = None) -> List[PlayerRecord]:
    """
    Loads ONLY active players from the 20 current Premier League teams in `bootstrap_static.json`.
    Picks the next editable gameweek unless `target_gw` is explicitly supplied.
    """
    if not CACHE_BOOTSTRAP.exists() or not CACHE_FIXTURES.exists():
        raise RuntimeError(
            "Live FPL cache (`data/bootstrap_static.json`) not found. Run `python3 fpl_assistant.py sync` first."
        )

    bootstrap = json.loads(CACHE_BOOTSTRAP.read_text(encoding="utf-8"))
    fixtures = json.loads(CACHE_FIXTURES.read_text(encoding="utf-8"))

    summaries_map: Dict[str, Any] = {}
    if CACHE_SUMMARIES.exists():
        try:
            summaries_map = json.loads(CACHE_SUMMARIES.read_text(encoding="utf-8"))
        except Exception:
            summaries_map = {}

    historical_map: Dict[str, List[Dict[str, Any]]] = {}
    if CACHE_HISTORICAL.exists():
        try:
            historical_map = json.loads(CACHE_HISTORICAL.read_text(encoding="utf-8"))
        except Exception:
            historical_map = {}

    # Strictly index the 20 current Premier League teams
    teams_by_id: Dict[int, Dict[str, Any]] = {
        int(t["id"]): t
        for t in bootstrap.get("teams", [])
        if not t.get("unavailable", False)
    }
    model, codes = team_model(bootstrap, fixtures)
    team_names = {tid: t.get("name", "") for tid, t in teams_by_id.items()}

    if target_gw is None:
        target_gw = next_editable_gameweek(bootstrap)

    # Map team_id -> list of upcoming fixture dicts for the next HORIZON_GWS gameweeks
    # (each dict carries its "gw", so blank and double gameweeks are represented naturally)
    horizon_gws = [target_gw + k for k in range(HORIZON_GWS)]
    upcoming_3gw_by_team: Dict[int, List[Dict[str, Any]]] = {tid: [] for tid in teams_by_id}
    for gw_ev in horizon_gws:
        for fix in fixtures:
            if fix.get("event") == gw_ev:
                th = int(fix["team_h"])
                ta = int(fix["team_a"])
                th_obj = teams_by_id.get(th, {})
                ta_obj = teams_by_id.get(ta, {})
                if th in upcoming_3gw_by_team:
                    upcoming_3gw_by_team[th].append(
                        {"gw": gw_ev, "opponent": f"{ta_obj.get('short_name', 'OPP')} (H)",
                         "difficulty": int(fix.get("team_h_difficulty", 3) or 3)}
                        | fixture_context(model, codes, th, ta, True, fix.get("kickoff_time") or "", team_names)
                    )
                if ta in upcoming_3gw_by_team:
                    upcoming_3gw_by_team[ta].append(
                        {"gw": gw_ev, "opponent": f"{th_obj.get('short_name', 'OPP')} (A)",
                         "difficulty": int(fix.get("team_a_difficulty", 3) or 3)}
                        | fixture_context(model, codes, ta, th, False, fix.get("kickoff_time") or "", team_names)
                    )

    next_fix_by_team: Dict[int, Dict[str, Any]] = {
        tid: fixes[0]
        for tid, fixes in upcoming_3gw_by_team.items()
        if fixes
    }

    records: List[PlayerRecord] = []
    for el in bootstrap.get("elements", []):
        # Strictly exclude players who have left the Premier League or cannot be selected
        if (
            el.get("status") == "u"
            or el.get("removed", False)
            or not el.get("can_select", True)
        ):
            continue

        tid = int(el["team"])
        if tid not in teams_by_id:
            continue

        team_obj = teams_by_id[tid]
        fix_info = next_fix_by_team.get(tid, dict(NEUTRAL_FIXTURE))

        pid = int(el["id"])
        mins = int(el.get("minutes", 0) or 0)
        nineties = max(mins / 90.0, 1.0)
        web_name = el.get("web_name", "")
        full_name = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()

        # Past seasons from element-summary or historical_map
        sum_entry = summaries_map.get(str(pid), {})
        past_list = sum_entry.get("history_past") or (
            historical_map.get(web_name.lower())
            or historical_map.get(full_name.lower())
            or []
        )
        hist_objs = [
            HistoricalSeasonStat(
                season_name=str(p.get("season_name", "")),
                total_points=int(p.get("total_points", 0) or 0),
                minutes=int(p.get("minutes", 0) or 0),
                goals_scored=int(p.get("goals_scored", 0) or 0),
                assists=int(p.get("assists", 0) or 0),
                clean_sheets=int(p.get("clean_sheets", 0) or 0),
                expected_goals=float(p.get("expected_goals", 0.0) or 0.0),
                expected_assists=float(p.get("expected_assists", 0.0) or 0.0),
                bonus=int(p.get("bonus", 0) or 0),
                expected_goals_conceded=float(p.get("expected_goals_conceded", 0.0) or 0.0),
                defensive_contribution=float(p.get("defensive_contribution", 0.0) or 0.0),
                ict_index=float(p.get("ict_index", 0.0) or 0.0),
                bps=int(p.get("bps", 0) or 0),
                saves=int(p.get("saves", 0) or 0),
                yellow_cards=int(p.get("yellow_cards", 0) or 0),
            )
            for p in past_list[-3:]
        ]

        gw_hist = sum_entry.get("history", [])

        records.append(
            PlayerRecord(
                id=pid,
                web_name=web_name,
                full_name=full_name,
                team_id=tid,
                team_short=team_obj.get("short_name", "UNK"),
                team_name=team_obj.get("name", "Unknown"),
                position=POSITION_MAP.get(int(el.get("element_type", 3)), "MID"),
                price=round(float(el.get("now_cost", 50)) / 10.0, 1),
                status=el.get("status", "a"),
                chance_of_playing_next_round=el.get("chance_of_playing_next_round"),
                fpl_news=el.get("news", "") or "",
                total_points=int(el.get("total_points", 0) or 0),
                minutes=mins,
                starts=int(el.get("starts", 0) or 0),
                points_per_game=float(el.get("points_per_game", 0.0) or 0.0),
                form=float(el.get("form", 0.0) or 0.0),
                value_form=float(el.get("value_form", 0.0) or 0.0),
                value_season=float(el.get("value_season", 0.0) or 0.0),
                selected_by_percent=float(el.get("selected_by_percent", 0.0) or 0.0),
                xg_per_90=float(
                    el.get("expected_goals_per_90")
                    or round(float(el.get("expected_goals", 0.0) or 0.0) / nineties, 2)
                ),
                xa_per_90=float(
                    el.get("expected_assists_per_90")
                    or round(float(el.get("expected_assists", 0.0) or 0.0) / nineties, 2)
                ),
                xgi_per_90=float(
                    el.get("expected_goal_involvements_per_90")
                    or round(
                        float(el.get("expected_goal_involvements", 0.0) or 0.0) / nineties,
                        2,
                    )
                ),
                xgc_per_90=float(
                    el.get("expected_goals_conceded_per_90")
                    or round(
                        float(el.get("expected_goals_conceded", 0.0) or 0.0) / nineties, 2
                    )
                ),
                defensive_contribution_per_90=float(
                    el.get("defensive_contribution_per_90", 0.0) or 0.0
                ),
                ict_per_90=round(float(el.get("ict_index", 0.0) or 0.0) / nineties, 2),
                threat_per_90=round(float(el.get("threat", 0.0) or 0.0) / nineties, 1),
                creativity_per_90=round(
                    float(el.get("creativity", 0.0) or 0.0) / nineties, 1
                ),
                saves_per_90=float(el.get("saves_per_90", 0.0) or 0.0),
                clean_sheets=int(el.get("clean_sheets", 0) or 0),
                bps_per_90=round(float(el.get("bps", 0) or 0) / nineties, 1),
                bonus=int(el.get("bonus", 0) or 0),
                in_dreamteam=bool(el.get("in_dreamteam", False)),
                penalties_order=el.get("penalties_order"),
                corners_order=el.get("corners_and_indirect_freekicks_order"),
                direct_freekicks_order=el.get("direct_freekicks_order"),
                next_opponent=fix_info["opponent"],
                is_home=fix_info["is_home"],
                fixture_difficulty=fix_info["difficulty"],
                team_attack_strength=fix_info["team_att"],
                team_defence_strength=fix_info["team_def"],
                opp_attack_strength=fix_info["opp_att"],
                opp_defence_strength=fix_info["opp_def"],
                historical_seasons=hist_objs,
                gw_history=gw_hist,
                upcoming_3gw_fixtures=upcoming_3gw_by_team.get(tid, [fix_info]),
                fixture_run_3gw=" → ".join(
                    f["opponent"] for f in upcoming_3gw_by_team.get(tid, [fix_info])[:3]
                ),
                target_gw=int(target_gw),
                lambda_for=float(fix_info.get("lambda_for", 1.35)),
                lambda_against=float(fix_info.get("lambda_against", 1.35)),
                market_available=float(fix_info.get("market_available", 0.0)),
                market_lambda_for=float(fix_info.get("market_lambda_for", fix_info.get("lambda_for", 1.35))),
                market_lambda_against=float(fix_info.get("market_lambda_against", fix_info.get("lambda_against", 1.35))),
            )
        )
    return records
