"""Weekly minutes-prediction ledger (minutes_ledger.py).

Every page build before a deadline records, per player, the minutes head's own
probabilities and the final availability after official flags, return dates and curated
notes. Once the gameweek is complete, `score` compares each column with actual minutes
(log-loss and Brier for P(60+) and P(played)). This is the measurable loop for the
availability layer: a news rule earns its place only if the "final" column beats the
"model" column on later weeks. Records made after a deadline are never stored.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fpl_data import CACHE_BOOTSTRAP, DATA_DIR

LEDGER_PATH = DATA_DIR / "minutes_ledger.json"


def _load() -> Dict[str, Any]:
    if LEDGER_PATH.exists():
        try:
            return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {"gameweeks": {}}


def deadline_of(gw: int, bootstrap: Optional[Dict[str, Any]] = None) -> Optional[datetime]:
    bootstrap = bootstrap or (json.loads(CACHE_BOOTSTRAP.read_text(encoding="utf-8")) if CACHE_BOOTSTRAP.exists() else {})
    value = next((e.get("deadline_time") for e in bootstrap.get("events", []) if int(e.get("id", 0)) == gw), None)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None


def record(predictions: Iterable[Any], target_gw: int, now: Optional[datetime] = None,
           bootstrap: Optional[Dict[str, Any]] = None, path: Path = LEDGER_PATH) -> Dict[str, Any]:
    """Store pre-deadline probabilities for `target_gw`; later records before the deadline replace earlier ones."""
    now = now or datetime.now(timezone.utc)
    deadline = deadline_of(target_gw, bootstrap)
    if deadline is None or now >= deadline:
        return {"recorded": 0, "reason": "deadline passed or unknown"}
    ledger = _load() if path == LEDGER_PATH else (json.loads(path.read_text()) if path.exists() else {"gameweeks": {}})
    entry = {"deadline": deadline.isoformat(), "observed_at": now.isoformat(), "players": {}}
    for p in predictions:
        fixture_this_week = any(int(f.get("gw", 0) or 0) == target_gw for f in p.player.upcoming_3gw_fixtures)
        if not fixture_this_week:
            continue
        entry["players"][str(p.player.id)] = {
            "name": p.player.web_name,
            "model_p_full": p.model_p_full, "model_p_cameo": p.model_p_cameo,
            "final_p_full": p.p_full, "final_p_cameo": p.p_cameo,
            "status": p.player.status, "chance": p.player.chance_of_playing_next_round,
            "return_date": getattr(p.news, "return_date", None), "risk": p.news.risk_level,
        }
    ledger["gameweeks"][str(target_gw)] = entry
    path.write_text(json.dumps(ledger, indent=1) + "\n", encoding="utf-8")
    return {"recorded": len(entry["players"]), "gw": target_gw, "observed_at": entry["observed_at"]}


def _logloss(p: float, y: int) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -math.log(p if y else 1 - p)


def score(actual_minutes: Dict[int, Dict[int, int]], path: Path = LEDGER_PATH) -> Dict[str, Any]:
    """`actual_minutes[gw][player_id]` -> minutes. Returns per-week and pooled metrics for each column."""
    ledger = json.loads(path.read_text()) if path.exists() else {"gameweeks": {}}
    out: Dict[str, Any] = {"weeks": {}, "pooled": {}}
    pooled: Dict[str, List[float]] = {}
    for gw_text, entry in sorted(ledger["gameweeks"].items(), key=lambda kv: int(kv[0])):
        gw = int(gw_text)
        if gw not in actual_minutes:
            continue
        rows = []
        for pid, rec in entry["players"].items():
            minutes = actual_minutes[gw].get(int(pid))
            if minutes is None:
                continue
            rows.append((rec, minutes))
        if not rows:
            continue
        week: Dict[str, Any] = {"n": len(rows)}
        base_full = sum(1 for _, m in rows if m >= 60) / len(rows)
        base_played = sum(1 for _, m in rows if m > 0) / len(rows)
        for column in ("model", "final"):
            ll_full = [_logloss(rec[f"{column}_p_full"], int(m >= 60)) for rec, m in rows]
            ll_played = [_logloss(min(1.0, rec[f"{column}_p_full"] + rec[f"{column}_p_cameo"]), int(m > 0)) for rec, m in rows]
            brier = [(rec[f"{column}_p_full"] - int(m >= 60)) ** 2 for rec, m in rows]
            week[column] = {"logloss_60plus": round(sum(ll_full) / len(rows), 4),
                            "logloss_played": round(sum(ll_played) / len(rows), 4),
                            "brier_60plus": round(sum(brier) / len(rows), 4)}
            pooled.setdefault(column + "_ll60", []).extend(ll_full)
            pooled.setdefault(column + "_llplayed", []).extend(ll_played)
            pooled.setdefault(column + "_brier", []).extend(brier)
        week["base_rate"] = {"logloss_60plus": round(sum(_logloss(base_full, int(m >= 60)) for _, m in rows) / len(rows), 4),
                             "logloss_played": round(sum(_logloss(base_played, int(m > 0)) for _, m in rows) / len(rows), 4)}
        pooled.setdefault("base_ll60", []).extend(_logloss(base_full, int(m >= 60)) for _, m in rows)
        pooled.setdefault("base_llplayed", []).extend(_logloss(base_played, int(m > 0)) for _, m in rows)
        out["weeks"][gw] = week
    if pooled:
        out["pooled"] = {k: round(sum(v) / len(v), 4) for k, v in pooled.items()} | {"n": len(pooled["model_ll60"])}
    return out


def actual_minutes_from_players(players: Iterable[Any], finished_gw: int) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {}
    for p in players:
        for row in p.gw_history:
            gw = int(row.get("round", 0) or 0)
            if 0 < gw <= finished_gw and row.get("team_h_score") is not None:
                out.setdefault(gw, {})[p.id] = out.get(gw, {}).get(p.id, 0) + int(row.get("minutes", 0) or 0)
    return out


if __name__ == "__main__":
    from fpl_data import load_players
    from neural_model import latest_finished_gw

    players = load_players()
    result = score(actual_minutes_from_players(players, latest_finished_gw()))
    print(json.dumps(result, indent=1))
