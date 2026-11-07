"""
AI Tactical Head Coach Layer (ai_coach.py)

Acts as the contextual football-intelligence layer on top of the multi-head neural network:
  1. Analyzes Teammate Creative/Defensive Synergy (club-level xG/xA supply & key injuries)
  2. Evaluates 3-Gameweek Fixture Swings (`GW t -> GW t+1 -> GW t+2`) to flag 1-week traps
     vs. 3-week green-run breakouts
  3. Classifies each player's Tactical Archetype (`EXPLOSIVE_TALISMAN`, `CBIT_FLOOR_ANCHOR`,
     `CS_FORTRESS`, `MOMENTUM_BREAKOUT`, `FIXTURE_TRAP`, `CAMEO_RISK`)
  4. Optionally queries Gemini (`gemini-2.5-flash`) when `GEMINI_API_KEY` is set in the
     environment to incorporate free-text press conference nuances, caching results in
     `data/ai_coach_decisions.json`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpl_data import DATA_DIR, PlayerRecord
from neural_model import compute_recency_weighted_metrics
from news_scraper import NewsArticle, analyze_player_news

COACH_CACHE_PATH = DATA_DIR / "ai_coach_decisions.json"


@dataclass
class CoachDecision:
    player_id: int
    web_name: str
    team_short: str
    role_tag: str
    tactical_xp_delta: float  # Contextual adjustment to GW xP (-0.6 to +0.7)
    horizon_3gw_bonus: float  # Bonus/penalty for 3-GW fixture run (-0.8 to +0.9)
    haul_prob_boost: float  # Contextual boost to P(10+ pt haul) (-0.05 to +0.10)
    coach_note: str


def evaluate_squad_with_ai_coach(
    players: List[PlayerRecord],
    articles: List[NewsArticle],
    use_gemini_if_available: bool = True,
    save_snapshot: bool = True,
) -> Dict[int, CoachDecision]:
    """
    Produces a contextual `CoachDecision` for every active Premier League player by combining:
      - Club-level creative supply & teammate availability
      - 3-Gameweek fixture swing analysis (GW 10, GW 11, GW 12)
      - CBIT defensive floor vs. Set-Piece / xGI explosive ceiling
      - Live press conference intelligence (+ optional Gemini LLM refinement)
    """
    # 1. Compute club-level healthy creative supply (top creators' xA/90 + xG/90)
    club_creativity: Dict[int, float] = {}
    club_injured_stars: Dict[int, List[str]] = {}
    for p in players:
        if p.minutes >= 270 and p.status == "a":
            club_creativity[p.team_id] = club_creativity.get(p.team_id, 0.0) + p.xa_per_90
        elif p.minutes >= 270 and p.status in ("i", "s", "d") and p.price >= 6.5:
            club_injured_stars.setdefault(p.team_id, []).append(p.web_name)

    decisions: Dict[int, CoachDecision] = {}

    for p in players:
        rec = compute_recency_weighted_metrics(p.gw_history, p, decay=0.78)
        news_imp = analyze_player_news(p, articles)

        # Analyze 3-GW fixture horizon
        up3 = p.upcoming_3gw_fixtures[:3]
        if up3:
            avg_fdr_3gw = sum(float(f.get("difficulty", 3)) for f in up3) / len(up3)
            future_fdrs = [float(f.get("difficulty", 3)) for f in up3[1:]]
            avg_future_fdr = sum(future_fdrs) / len(future_fdrs) if future_fdrs else avg_fdr_3gw
            home_count_3gw = sum(1 for f in up3 if f.get("is_home"))
        else:
            avg_fdr_3gw = float(p.fixture_difficulty)
            avg_future_fdr = float(p.fixture_difficulty)
            home_count_3gw = 1 if p.is_home else 0

        # Detect 1-Week Fixture Trap vs 3-Week Green Run
        is_1wk_trap = p.fixture_difficulty <= 2 and avg_future_fdr >= 3.8
        is_green_run = avg_fdr_3gw <= 2.75 or (avg_fdr_3gw <= 3.0 and home_count_3gw >= 2)

        tactical_delta = 0.0
        horizon_bonus = 0.0
        haul_boost = 0.0
        reasons: List[str] = []

        # Club creative supply synergy for attackers
        club_xa = club_creativity.get(p.team_id, 0.8)
        if p.position in ("MID", "FWD") and club_xa >= 1.35 and p.xg_per_90 >= 0.28:
            tactical_delta += 0.22
            haul_boost += 0.04
            reasons.append("high club chance-creation supply")

        # Set-piece monopoly + weak opponent defence synergy
        has_pens = p.penalties_order == 1
        has_corners = p.corners_order in (1, 2)
        att_adv = p.attack_mismatch_advantage
        def_adv = p.defence_mismatch_advantage

        if has_pens and att_adv > 0.2:
            tactical_delta += 0.28
            haul_boost += 0.06
            reasons.append("1st-choice penalties vs vulnerable defence")
        elif has_corners and p.position in ("DEF", "MID"):
            tactical_delta += 0.15
            reasons.append("set-piece delivery/aerial threat")

        # CBIT defensive contribution floor
        cbit = rec["ewma_def90"]
        if (p.position == "DEF" and cbit >= 9.2) or (p.position == "MID" and cbit >= 10.5):
            tactical_delta += 0.25
            reasons.append(f"elite {cbit:.1f} CBIT/90 (+2pt floor magnet)")

        # Momentum surge (EWMA form significantly above season average)
        if rec["momentum"] >= 0.65:
            tactical_delta += 0.22
            haul_boost += 0.03
            reasons.append(f"hot streak (+{rec['momentum']:.1f} EWMA momentum)")
        elif rec["momentum"] <= -0.75:
            tactical_delta -= 0.20
            reasons.append(f"cooling form ({rec['momentum']:.1f} trend)")

        # 3-GW Horizon adjustments
        if is_green_run:
            horizon_bonus += 0.45
            reasons.append(f"strong 3-GW run ({p.fixture_run_3gw})")
        elif is_1wk_trap:
            horizon_bonus -= 0.40
            reasons.append(f"1-GW trap (tough GW11-12: {p.fixture_run_3gw})")

        # Determine Role Tag
        if news_imp.risk_level in ("CAMEO_ROTATION_RISK", "INJURED_OR_OUT", "BACKUP_LOW_MINS"):
            role_tag = "ROTATION / INJURY RISK"
            tactical_delta -= 0.35
        elif (has_pens or p.xgi_per_90 >= 0.55) and p.price >= 7.0:
            role_tag = "EXPLOSIVE TALISMAN"
        elif p.position in ("GKP", "DEF") and def_adv >= 0.45:
            role_tag = "CS FORTRESS"
            tactical_delta += 0.18
        elif (p.position == "DEF" and cbit >= 9.2) or (p.position == "MID" and cbit >= 10.5):
            role_tag = "CBIT FLOOR ANCHOR"
        elif rec["momentum"] >= 0.55 and p.value_season >= 4.8:
            role_tag = "MOMENTUM BREAKOUT"
        elif is_green_run:
            role_tag = "3-GW GREEN RUN TARGET"
        else:
            role_tag = "STEADY STARTER"

        if not reasons:
            reasons.append(f"3-GW schedule: {p.fixture_run_3gw}")

        coach_note = f"[{role_tag}] " + "; ".join(reasons[:2]).capitalize() + "."

        decisions[p.id] = CoachDecision(
            player_id=p.id,
            web_name=p.web_name,
            team_short=p.team_short,
            role_tag=role_tag,
            tactical_xp_delta=round(max(-0.6, min(0.65, tactical_delta)), 2),
            horizon_3gw_bonus=round(max(-0.6, min(0.75, horizon_bonus)), 2),
            haul_prob_boost=round(max(-0.05, min(0.10, haul_boost)), 3),
            coach_note=coach_note,
        )

    # Optional Gemini LLM refinement if GEMINI_API_KEY is configured locally
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if use_gemini_if_available and api_key:
        _refine_top_candidates_with_gemini(players, decisions, articles, api_key)

    if not save_snapshot:
        return decisions
    # Save decisions snapshot for transparency
    try:
        COACH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            str(pid): asdict(dec)
            for pid, dec in sorted(
                decisions.items(),
                key=lambda item: item[1].tactical_xp_delta + item[1].horizon_3gw_bonus,
                reverse=True,
            )[:60]
        }
        COACH_CACHE_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    except Exception:
        pass

    return decisions


def _refine_top_candidates_with_gemini(
    players: List[PlayerRecord],
    decisions: Dict[int, CoachDecision],
    articles: List[NewsArticle],
    api_key: str,
) -> None:
    """
    When `GEMINI_API_KEY` is set locally, sends the top 25 FPL candidates + recent
    press conference headlines to `gemini-2.5-flash` to refine `tactical_xp_delta` and `coach_note`.
    """
    top_players = sorted(players, key=lambda x: x.total_points, reverse=True)[:25]
    headlines = [f"- [{a.source}] {a.headline}" for a in articles[:18]]
    player_lines = [
        f"ID={p.id} | {p.web_name} ({p.team_short}, {p.position}, £{p.price}m) | "
        f"3GW={p.fixture_run_3gw} | PPG={p.points_per_game} | Form={p.form} | News={p.fpl_news or 'Healthy'}"
        for p in top_players
    ]
    prompt = (
        "You are an elite Fantasy Premier League AI Head Coach. Review these 25 players and "
        "recent headlines. Return a JSON object mapping player ID (as string) to "
        '{"delta": float between -0.5 and 0.5, "note": concise 12-word tactical insight}.\n\n'
        "Headlines:\n" + "\n".join(headlines) + "\n\nPlayers:\n" + "\n".join(player_lines)
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        overrides = json.loads(text)
        for pid_str, info in overrides.items():
            pid = int(pid_str)
            if pid in decisions and isinstance(info, dict):
                d = decisions[pid]
                llm_delta = float(info.get("delta", 0.0) or 0.0)
                d.tactical_xp_delta = round(
                    max(-0.6, min(0.7, d.tactical_xp_delta + 0.5 * llm_delta)), 2
                )
                if info.get("note"):
                    d.coach_note = f"[{d.role_tag}] {info['note']}"
    except Exception:
        pass
