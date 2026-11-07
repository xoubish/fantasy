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
import json
import sqlite3
import time
import urllib.request
from dataclasses import asdict, dataclass, field
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
    Automatically refreshes cached players whenever a new Gameweek (`latest_finished_gw`)
    has finished so last week's match logs are appended every week.
    """
    existing: Dict[str, Any] = {}
    if CACHE_SUMMARIES.exists() and not force_refresh:
        try:
            existing = json.loads(CACHE_SUMMARIES.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    def _needs_update(pid: int) -> bool:
        entry = existing.get(str(pid))
        if not entry:
            return True
        if latest_finished_gw <= 0:
            return False
        hist = entry.get("history", [])
        if not hist:
            return True
        max_cached_finished_gw = max(
            (
                int(r.get("round", 0) or 0)
                for r in hist
                if r.get("team_h_score") is not None
            ),
            default=0,
        )
        return max_cached_finished_gw < latest_finished_gw

    to_fetch = [pid for pid in player_ids if _needs_update(pid)]
    if not to_fetch:
        return existing

    def _fetch_one(pid: int) -> tuple[str, Any]:
        data = _http_get_json(f"{FPL_BASE_URL}/element-summary/{pid}/", timeout=10)
        return str(pid), {
            "history": data.get("history", []),
            "history_past": data.get("history_past", []),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_one, pid): pid for pid in to_fetch}
        for fut in concurrent.futures.as_completed(futures):
            try:
                k, val = fut.result()
                existing[k] = val
            except Exception:
                continue

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

    # Fetch per-player element-summary for all active players with >= 90 mins
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
                    if f.get("finished") and f.get("team_h_score") is not None
                ),
                default=0,
            )
            status["latest_finished_gw"] = latest_finished_gw

            active_ids = [
                int(el["id"])
                for el in bootstrap.get("elements", [])
                if el.get("status") != "u"
                and not el.get("removed")
                and int(el.get("minutes", 0) or 0) >= 90
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

    return status


def compute_team_strengths(
    bootstrap: Dict[str, Any],
    fixtures: List[Dict[str, Any]],
    max_gw: Optional[int] = None,
) -> Dict[int, Dict[str, int]]:
    """
    Computes time-decayed real Attack and Defence strength ratings (on the 950-1350 scale)
    for all 20 Premier League teams using:
      1. Exponentially time-weighted goals scored (GF/g) and conceded (GA/g) from finished matches
         (decay = 0.86 per GW back, so the last 3-4 weeks matter much more than GW 1)
      2. Official `strength_overall_home` / `strength_overall_away` (2 to 5)
      3. Current Premier League table `position` (1 to 20)
    """
    teams = {int(t["id"]): t for t in bootstrap.get("teams", [])}
    gf_w = {tid: 0.0 for tid in teams}
    ga_w = {tid: 0.0 for tid in teams}
    gp_w = {tid: 0.0 for tid in teams}

    finished_fixes = [
        f
        for f in fixtures
        if f.get("finished")
        and f.get("team_h_score") is not None
        and (max_gw is None or int(f.get("event", 0) or 0) <= max_gw)
    ]
    latest_gw = max(
        (int(f.get("event", 1) or 1) for f in finished_fixes),
        default=1,
    )

    for fix in finished_fixes:
        gw = int(fix.get("event", 1) or 1)
        # Exponential recency weight: last week = 1.0, 4 weeks ago = 0.55, 8 weeks ago = 0.30
        recency_w = 0.86 ** max(0, latest_gw - gw)
        th, ta = int(fix["team_h"]), int(fix["team_a"])
        sh, sa = int(fix["team_h_score"]), int(fix["team_a_score"])
        if th in gf_w:
            gf_w[th] += sh * recency_w
            ga_w[th] += sa * recency_w
            gp_w[th] += recency_w
        if ta in gf_w:
            gf_w[ta] += sa * recency_w
            ga_w[ta] += sh * recency_w
            gp_w[ta] += recency_w

    ratings: Dict[int, Dict[str, int]] = {}
    for tid, t in teams.items():
        n = max(0.5, gp_w[tid])
        gf_pg = gf_w[tid] / n  # time-weighted goals scored per game
        ga_pg = ga_w[tid] / n  # time-weighted goals conceded per game
        pos_rank = int(t.get("position", 10) or 10)
        str_h = int(t.get("strength_overall_home", 3) or 3)
        str_a = int(t.get("strength_overall_away", 3) or 3)

        # Higher attack rating = team scores more goals recently & has higher overall strength
        att_base = 1000 + int((gf_pg - 0.65) * 140) + (20 - pos_rank) * 4
        # Higher defence rating = team concedes FEWER goals recently & has higher overall strength
        def_base = 1000 + int((2.35 - ga_pg) * 150) + (20 - pos_rank) * 4

        ratings[tid] = {
            "att_home": max(960, min(1360, att_base + str_h * 18)),
            "def_home": max(960, min(1360, def_base + str_h * 18)),
            "att_away": max(950, min(1350, att_base + str_a * 12 - 20)),
            "def_away": max(950, min(1350, def_base + str_a * 12 - 20)),
        }
    return ratings


def load_players(target_gw: Optional[int] = None) -> List[PlayerRecord]:
    """
    Loads ONLY active players from the 20 current Premier League teams in `bootstrap_static.json`.
    Automatically picks the active/upcoming unfinished Gameweek (e.g. GW 10) unless `target_gw` is specified.
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
    team_ratings = compute_team_strengths(bootstrap, fixtures)

    # Determine target Gameweek: pick the earliest GW that still has unfinished matches (GW 10)
    if target_gw is None:
        events = bootstrap.get("events", [])
        current_ev = next((e for e in events if e.get("is_current")), None)
        next_ev = next((e for e in events if e.get("is_next")), None)
        if current_ev:
            gw_id = int(current_ev["id"])
            gw_fixes = [f for f in fixtures if f.get("event") == gw_id]
            # If the majority of matches in current_ev are still unfinished, use current_ev (GW 10)
            unfinished = sum(1 for f in gw_fixes if not f.get("finished"))
            target_gw = gw_id if unfinished >= 5 else (int(next_ev["id"]) if next_ev else gw_id)
        elif next_ev:
            target_gw = int(next_ev["id"])
        else:
            target_gw = 10

    # Map team_id -> fixture info for target_gw
    next_fix_by_team: Dict[int, Dict[str, Any]] = {}
    for fix in fixtures:
        if fix.get("event") == target_gw:
            th = int(fix["team_h"])
            ta = int(fix["team_a"])
            th_obj = teams_by_id.get(th, {})
            ta_obj = teams_by_id.get(ta, {})
            r_h = team_ratings.get(th, {"att_home": 1150, "def_home": 1150, "att_away": 1130, "def_away": 1130})
            r_a = team_ratings.get(ta, {"att_home": 1150, "def_home": 1150, "att_away": 1130, "def_away": 1130})
            next_fix_by_team[th] = {
                "opponent": f"{ta_obj.get('short_name', 'OPP')} (H)",
                "is_home": True,
                "difficulty": int(fix.get("team_h_difficulty", 3)),
                "team_att": r_h["att_home"],
                "team_def": r_h["def_home"],
                "opp_att": r_a["att_away"],
                "opp_def": r_a["def_away"],
            }
            next_fix_by_team[ta] = {
                "opponent": f"{th_obj.get('short_name', 'OPP')} (A)",
                "is_home": False,
                "difficulty": int(fix.get("team_a_difficulty", 3)),
                "team_att": r_a["att_away"],
                "team_def": r_a["def_away"],
                "opp_att": r_h["att_home"],
                "opp_def": r_h["def_home"],
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
        fix_info = next_fix_by_team.get(
            tid,
            {
                "opponent": "TBD",
                "is_home": True,
                "difficulty": 3,
                "team_att": 1150,
                "team_def": 1150,
                "opp_att": 1150,
                "opp_def": 1150,
            },
        )

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
            )
        )
    return records
