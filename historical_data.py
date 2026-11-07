"""Versioned public FPL match archive, independent of today's selectable roster.

Source: https://github.com/vaastav/Fantasy-Premier-League (MIT).
Only completed seasons are imported; raw files are pinned by commit and SHA256.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from fpl_data import CACHE_BOOTSTRAP, DATA_DIR, POSITION_MAP, USER_AGENT
from team_ratings import archived_matches, ratings_at

ARCHIVE_DIR = DATA_DIR / "historical"
MANIFEST_PATH = DATA_DIR / "historical_manifest.json"
REPOSITORY = "vaastav/Fantasy-Premier-League"
FILES = ("gws/merged_gw.csv", "fixtures.csv", "players_raw.csv", "teams.csv")
# The season before the first training season supplies prior-season player rates and
# match results for cross-season team ratings. Its match logs are never training rows.
CONTEXT_FILES = ("fixtures.csv", "players_raw.csv", "teams.csv")


@lru_cache(maxsize=1)
def current_season_start() -> int:
    if CACHE_BOOTSTRAP.exists():
        bootstrap = json.loads(CACHE_BOOTSTRAP.read_text())
        first = min(e["deadline_time"] for e in bootstrap["events"] if e.get("deadline_time"))
        date = datetime.fromisoformat(first.replace("Z", "+00:00"))
    else:
        date = datetime.now(timezone.utc)
    return date.year if date.month >= 7 else date.year - 1


def season_name(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def sync_historical_data(seasons=None, refresh=False) -> dict:
    latest = current_season_start()
    seasons = seasons or [season_name(y) for y in range(max(2022, latest - 4), latest)]
    if any(int(s[:4]) >= latest for s in seasons):
        raise ValueError("Historical training accepts completed seasons only.")
    context_season = season_name(min(int(s[:4]) for s in seasons) - 1)
    old = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    jobs = [f"{season}/{file}" for season in seasons for file in FILES] + [f"{context_season}/{file}" for file in CONTEXT_FILES]
    if not refresh and old.get("seasons") == seasons and old.get("context_season") == context_season:
        if all((ARCHIVE_DIR / name).exists() and hashlib.sha256((ARCHIVE_DIR / name).read_bytes()).hexdigest() == meta["sha256"]
               for name, meta in old.get("files", {}).items()) and set(old.get("files", {})) == set(jobs):
            return old
    commit = old.get("commit") if old.get("seasons") == seasons and not refresh else None
    commit = commit or json.loads(_download(f"https://api.github.com/repos/{REPOSITORY}/commits/master"))["sha"]

    def fetch(name):
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/data/{name}"
        content = _download(url)
        # Validate before replacing a previously usable archive.
        rows = list(csv.DictReader(content.decode("utf-8-sig").splitlines()))
        if not rows:
            raise ValueError(f"Empty historical file: {name}")
        return name, content, {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "rows": len(rows)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = list(pool.map(fetch, jobs))
    manifest = {"repository": f"https://github.com/{REPOSITORY}", "license": "MIT", "commit": commit,
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "seasons": seasons,
                "context_season": context_season, "files": {}}
    for name, content, meta in downloaded:
        path = ARCHIVE_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        manifest["files"][name] = meta
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def number(value, default=0.0):
    try:
        return float(value) if value not in (None, "", "None") else default
    except ValueError:
        return default


def csv_rows(season, filename):
    path = ARCHIVE_DIR / season / filename
    if not path.exists():
        raise RuntimeError("Historical archive missing. Run `python3 historical_data.py sync`.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=8)
def load_season(season: str) -> dict:
    """Normalize rows; derive a player's club from that match, never final-season club."""
    fixtures = []
    for row in csv_rows(season, "fixtures.csv"):
        if row.get("finished", "").lower() != "true" or not row.get("kickoff_time"):
            continue
        fixtures.append({k: (row[k] if k == "kickoff_time" else int(number(row[k])))
                         for k in ("id", "event", "team_h", "team_a", "team_h_score", "team_a_score", "kickoff_time")}
                       | {"finished": True})
    fixture_by_id = {f["id"]: f for f in fixtures}
    raw = {int(r["id"]): r for r in csv_rows(season, "players_raw.csv")}
    teams = {int(r["id"]): r for r in csv_rows(season, "teams.csv")}
    by_player = defaultdict(list)
    seen = {}
    duplicate_rows = 0
    manager_rows = 0
    for source in csv_rows(season, "gws/merged_gw.csv"):
        position = source.get("position") or POSITION_MAP.get(int(raw[int(source["element"])]["element_type"]))
        if position == "AM":
            manager_rows += 1
            continue
        position = "GKP" if position == "GK" else position
        if position not in POSITION_MAP.values():
            raise ValueError(f"Unsupported historical position {position!r}")
        pid, fid = int(source["element"]), int(source["fixture"])
        if (pid, fid) in seen:
            if seen[pid, fid] != source:
                raise ValueError(f"Conflicting player/fixture in {season}: {pid}/{fid}")
            duplicate_rows += 1
            continue
        seen[pid, fid] = source
        fixture = fixture_by_id.get(fid)
        if not fixture:
            continue
        home = source["was_home"].lower() == "true"
        row = {k: number(v) for k, v in source.items() if k not in ("name", "position", "team", "kickoff_time", "was_home")}
        row.update(round=int(source.get("round") or source["GW"]), fixture=fid, was_home=home,
                   kickoff_time=source["kickoff_time"], position=position,
                   team_id=fixture["team_h"] if home else fixture["team_a"],
                   name=source["name"], code=int(number(raw.get(pid, {}).get("code"))),
                   has_defcon="defensive_contribution" in source)
        by_player[pid].append(row)
    for rows in by_player.values():
        rows.sort(key=lambda r: (r["kickoff_time"], r["fixture"]))
    # Reconstruct a pre-match deadline; no target-week rows are ever used in features.
    deadlines = {}
    for f in fixtures:
        kickoff = datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))
        deadline = (kickoff - timedelta(minutes=90)).isoformat().replace("+00:00", "Z")
        deadlines[f["event"]] = min(deadlines.get(f["event"], deadline), deadline)
    # End-of-season strength/FDR fields are deliberately excluded. Team ratings come from the
    # cross-season Poisson model fitted on every archived match before each deadline.
    codes = {tid: int(number(row.get("code"))) for tid, row in teams.items()}
    matches = archived_matches()
    ratings = {gw: ratings_at(datetime.fromisoformat(cutoff.replace("Z", "+00:00")), matches)
               for gw, cutoff in deadlines.items()}
    return {"season": season, "players": dict(by_player), "teams": teams, "codes": codes,
            "fixtures": fixtures, "deadlines": deadlines, "ratings": ratings,
            "duplicate_rows_removed": duplicate_rows, "manager_rows_excluded": manager_rows}


PRIOR_COLUMNS = ("minutes", "total_points", "expected_goals", "expected_assists", "expected_goals_conceded",
                 "defensive_contribution", "ict_index", "bps", "saves", "bonus", "goals_scored", "assists",
                 "clean_sheets", "yellow_cards")


def prior_season_stats(season: str) -> dict:
    """Previous-season totals by stable FPL code (inputs to `neural_model.prior_rates_from_totals`)."""
    previous = season_name(int(season[:4]) - 1)
    if not (ARCHIVE_DIR / previous / "players_raw.csv").exists():
        return {}
    # Stable FPL code is used across seasons; element IDs are never joined across years.
    return {int(r["code"]): {column: number(r.get(column)) for column in PRIOR_COLUMNS}
            for r in csv_rows(previous, "players_raw.csv") if number(r.get("code"))}


def fixture_features(archive: dict, row: dict) -> dict:
    """Point-in-time fixture context for a match row: Poisson expected goals and display ratings."""
    ratings = archive["ratings"][row["round"]]
    own = archive["codes"].get(row["team_id"])
    opp = archive["codes"].get(int(row["opponent_team"]))
    return ratings.fixture_context(own, opp, row["was_home"]) | {"difficulty": 3}


def build_historical_samples():
    from neural_model import Sample, components_from_row, features_from_history, prior_rates_from_totals
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Historical training is required. Run `python3 historical_data.py sync` first.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    samples = []
    for season in manifest["seasons"]:
        if int(season[:4]) >= current_season_start():
            raise ValueError("Current-season outcomes cannot enter historical training.")
        archive = load_season(season)
        previous = prior_season_stats(season)
        for pid, rows in archive["players"].items():
            for row in rows:
                gw = row["round"]
                prior = [r for r in rows if r["round"] < gw and r["kickoff_time"] < archive["deadlines"][gw]]
                feats = features_from_history(row["position"], prior, fixture_features(archive, row),
                                              row["value"] / 10, prior_rates_from_totals(previous.get(row["code"])),
                                              season_start=int(season[:4]))
                played = [r for r in prior if r["minutes"] > 0]
                ppa = sum(r["total_points"] for r in played) / len(played) if played else 0.0
                naive = sum(r["total_points"] for r in prior) / len(prior) if prior else 0.0
                samples.append(Sample(feats, int(row["minutes"]), row["total_points"], gw, pid, naive,
                                      season=season, fixture_id=row["fixture"],
                                      components=components_from_row(row), naive_ppa=ppa))
    return samples


def position_prior_rates(seasons=("2022-23", "2023-24", "2024-25"), min_minutes=450) -> dict:
    """Recompute `neural_model.POSITION_PRIOR_RATES` from development seasons (players >= min_minutes)."""
    from statistics import mean
    source = {"pts": "total_points", "xg": "expected_goals", "xa": "expected_assists", "xgc": "expected_goals_conceded",
              "def": "defensive_contribution", "ict": "ict_index", "bps": "bps", "saves": "saves", "bonus": "bonus",
              "goals": "goals_scored", "assists": "assists", "cs": "clean_sheets", "yellow": "yellow_cards"}
    rates = {pos: {k: [] for k in source} for pos in POSITION_MAP.values()}
    for season in seasons:
        for r in csv_rows(season, "players_raw.csv"):
            minutes = number(r.get("minutes"))
            if minutes < min_minutes:
                continue
            pos = POSITION_MAP.get(int(number(r.get("element_type"))))
            if pos is None:
                continue
            for key, column in source.items():
                rates[pos][key].append(number(r.get(column)) / (minutes / 90))
    return {pos: {k: round(mean(v), 2) if v else 0.0 for k, v in cols.items()} for pos, cols in rates.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sync", "position-priors"])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.command == "position-priors":
        print(json.dumps(position_prior_rates(), indent=1))
        raise SystemExit(0)
    result = sync_historical_data(refresh=args.refresh)
    print(json.dumps({"seasons": result["seasons"], "commit": result["commit"],
                      "match_rows": sum(v["rows"] for k, v in result["files"].items() if k.endswith("merged_gw.csv"))}, indent=2))
