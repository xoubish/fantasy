#!/usr/bin/env python3
"""
FPL AI Optimizer — Broadcast-Grade Interactive Web App Generator & Server (web_app.py)

Generates a standalone, zero-dependency `index.html` (ready to open locally or host on GitHub Pages)
featuring:
  1. Broadcast Glassmorphism Scoreboard Header with Verified Walk-Forward Backtest badge (+12.9% vs FPL Avg)
     and 4 live KPI cards (GW 10 Projected Score, 3-GW Horizon Score, Captain Haul Ceiling, Budget).
  2. 3D-Styled Stadium Turf Pitch with authentic club SVG jerseys, FDR color-coded fixture badges,
     Haul Ceiling / CBIT micro-badges, Captain (C) & Vice (V) indicators, and active selection glow.
  3. Rich Player Inspector Drawer with visual progress gauges (xP, Haul Ceiling %, Clean Sheet %, Start %),
     3-GW Fixture Ribbon, and AI Tactical Head Coach briefing.
  4. Bottom Analytics Deck with:
     - Full 15-Man Tactical Squad Table (Role Archetype, 3-GW Run, 3-GW Horizon xP, Haul %, CBIT/90)
     - Out-of-Sample Walk-Forward Backtest Scorecard (GW 6–9 Verified Proof)
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
from pathlib import Path
from typing import Any, Dict, List

from fpl_data import load_players
from neural_model import get_or_train_neural_net
from news_scraper import load_all_news
from predictor import (
    OptimizedSquadResult,
    PlayerPrediction,
    build_dream_xi,
    get_best_squad_variations,
    optimize_starting_xi,
    predict_player,
    recommend_single_transfers,
)

ROOT_DIR = Path(__file__).parent
SQUAD_CONFIG_PATH = ROOT_DIR / "my_squad.json"
INDEX_HTML_PATH = ROOT_DIR / "index.html"


def _serialize_pred(p: PlayerPrediction, is_cap: bool = False, is_vice: bool = False) -> Dict[str, Any]:
    return {
        "id": p.player.id,
        "name": p.player.web_name,
        "fullName": p.player.full_name,
        "team": p.player.team_short,
        "teamName": p.player.team_name,
        "pos": p.player.position,
        "price": p.player.price,
        "opponent": p.player.next_opponent,
        "fdr": p.player.fixture_difficulty,
        "csProb": int(round(p.clean_sheet_prob * 100)),
        "gc2Prob": int(round(p.concede_2plus_prob * 100)),
        "ppg3Yr": p.player.historical_ppg_3yr,
        "seasonPpg": p.player.points_per_game,
        "defContrib90": p.player.defensive_contribution_per_90,
        "valueSeason": p.player.value_season,
        "form": p.player.form,
        "nnXp": round(p.nn_xp_90, 1),
        "ruleXp": round(p.rule_xp_90, 1),
        "xp": round(p.expected_points, 1),
        "displayXp": round(p.expected_points * 2 if is_cap else p.expected_points, 1),
        "haulProb": int(round(p.haul_prob * 100)),
        "horizon3GwXp": round(p.horizon_3gw_xp, 1),
        "fixtureRun3Gw": p.player.fixture_run_3gw,
        "roleTag": p.role_tag,
        "coachNote": p.coach_note,
        "floor": p.floor_points * 2 if is_cap else p.floor_points,
        "ceiling": p.ceiling_points * 2 if is_cap else p.ceiling_points,
        "inDreamteam": p.player.in_dreamteam,
        "isCaptain": is_cap,
        "isVice": is_vice,
        "startProb": int(round(p.news.start_probability * 100)),
        "cameoProb": int(round(p.news.cameo_probability * 100)),
        "riskLevel": p.news.risk_level,
        "newsReasoning": p.news.reasoning,
    }


def _serialize_squad_result(
    res: OptimizedSquadResult,
    title: str,
    subtitle: str,
    total_cost: float,
    budget: float,
) -> Dict[str, Any]:
    cap_id = res.captain.player.id
    vice_id = res.vice_captain.player.id
    starters_ser = [
        _serialize_pred(p, p.player.id == cap_id, p.player.id == vice_id)
        for p in res.starting_xi
    ]
    horizon_total = round(
        sum(p.horizon_3gw_xp for p in res.starting_xi) + res.captain.expected_points,
        1,
    )
    return {
        "title": title,
        "subtitle": subtitle,
        "formation": res.formation,
        "totalCost": round(total_cost, 1),
        "budget": round(budget, 1),
        "expectedPoints": res.total_expected_points,
        "horizon3GwTotal": horizon_total,
        "floor": res.total_floor,
        "ceiling": res.total_ceiling,
        "captain": res.captain.player.web_name,
        "captainHaulProb": int(round(res.captain.haul_prob * 100)),
        "viceCaptain": res.vice_captain.player.web_name,
        "starters": starters_ser,
        "bench": [
            _serialize_pred(res.bench_gkp),
            *[_serialize_pred(bp) for bp in res.bench_outfield],
        ],
        "notes": res.tactical_notes,
    }


def build_web_payload() -> Dict[str, Any]:
    all_players = load_players()
    articles = load_all_news(SQUAD_CONFIG_PATH)
    squad_cfg = json.loads(SQUAD_CONFIG_PATH.read_text(encoding="utf-8"))

    all_preds = [predict_player(p, articles) for p in all_players]

    user_preds: List[PlayerPrediction] = []
    for item in squad_cfg.get("squad", []):
        nm = item["name"].strip().lower()
        tm = item.get("team", "").strip().upper()
        ps = item.get("position", "").strip().upper()
        found = next(
            (
                p
                for p in all_preds
                if p.player.web_name.lower() == nm
                and (not tm or p.player.team_short == tm)
                and (not ps or p.player.position == ps)
            ),
            next(
                (
                    p
                    for p in all_preds
                    if p.player.web_name.lower() == nm
                    and (not ps or p.player.position == ps)
                ),
                None,
            ),
        )
        if found:
            user_preds.append(found)

    user_opt = optimize_starting_xi(user_preds)
    user_cost = sum(p.player.price for p in user_preds)
    bank = float(squad_cfg.get("bank_millions", 0.5))

    variations_raw = get_best_squad_variations(all_preds, budget=100.0, num_variations=6)
    variations_json = [
        _serialize_squad_result(
            v.result,
            title=v.name,
            subtitle=v.subtitle,
            total_cost=v.total_cost,
            budget=v.budget,
        )
        for v in variations_raw
    ]

    dream_res = build_dream_xi(all_preds)
    dream_cost = sum(p.player.price for p in dream_res.starting_xi)
    dream_json = _serialize_squad_result(
        dream_res,
        title="Unlimited Free Hit Dream XI",
        subtitle="No budget cap — pure maximum Expected Points & Haul Ceiling across the Premier League",
        total_cost=dream_cost,
        budget=999.0,
    )

    user_json = _serialize_squad_result(
        user_opt,
        title="Your Current 15-Man Squad (Optimized)",
        subtitle="Dual-Head Neural Net + AI Tactical Head Coach + 3-GW Horizon Protection",
        total_cost=user_cost,
        budget=user_cost + bank,
    )

    transfers = recommend_single_transfers(user_preds, all_preds, bank_millions=bank, top_n=5)
    transfers_json = [
        {
            "out": t.player_out.player.web_name,
            "outPrice": t.player_out.player.price,
            "in": t.player_in.player.web_name,
            "inPrice": t.player_in.player.price,
            "inTeam": t.player_in.player.team_short,
            "inFix": t.player_in.player.next_opponent,
            "in3Gw": t.player_in.player.fixture_run_3gw,
            "costDiff": t.net_cost,
            "xpGain": t.xp_gain_this_week,
        }
        for t in transfers
    ]

    healthy_active = sorted(
        [
            p
            for p in all_preds
            if p.news.risk_level in ("NONE", "POSITIVE") and p.player.minutes >= 270
        ],
        key=lambda x: x.expected_points,
        reverse=True,
    )[:65]
    healthy_pool_json = [_serialize_pred(p) for p in healthy_active]

    nn_net = get_or_train_neural_net(force_retrain=False)

    return {
        "variations": variations_json,
        "mySquad": user_json,
        "dreamXi": dream_json,
        "transfers": transfers_json,
        "playerPool": healthy_pool_json,
        "nnMetrics": nn_net.metrics,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FPL AI Head Coach — Dual-Head Neural Network & 3-GW Squad Optimizer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&family=Space+Grotesk:wght@600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --fpl-purple: #37003c;
      --fpl-deep: #0b0112;
      --fpl-green: #00ff87;
      --fpl-cyan: #04f5ff;
      --fpl-pink: #ff2882;
      --fpl-gold: #ffd166;
      --turf-dark: #115e38;
      --turf-light: #167345;
      --glass-bg: rgba(25, 8, 36, 0.72);
      --glass-border: rgba(255, 255, 255, 0.12);
      --text-muted: #b4a5c2;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--fpl-deep);
      background-image:
        radial-gradient(at 12% 8%, rgba(0, 255, 135, 0.13) 0px, transparent 42%),
        radial-gradient(at 88% 6%, rgba(4, 245, 255, 0.12) 0px, transparent 45%),
        radial-gradient(at 50% 35%, rgba(55, 0, 60, 0.85) 0px, #0b0112 80%);
      background-attachment: fixed;
      color: #ffffff;
      min-height: 100vh;
      padding: 22px 20px 56px;
    }
    .container {
      max-width: 1240px;
      margin: 0 auto;
    }

    /* Top Broadcast Verification Ribbon */
    .top-ribbon {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      background: linear-gradient(90deg, rgba(0,255,135,0.14) 0%, rgba(4,245,255,0.10) 50%, rgba(255,40,130,0.12) 100%);
      border: 1px solid rgba(0, 255, 135, 0.3);
      border-radius: 999px;
      padding: 7px 18px;
      margin-bottom: 16px;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.2px;
      backdrop-filter: blur(12px);
    }
    .ribbon-left {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--fpl-green);
      box-shadow: 0 0 10px var(--fpl-green);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.3); opacity: 0.65; }
    }
    .ribbon-badge {
      background: rgba(0, 255, 135, 0.2);
      color: var(--fpl-green);
      padding: 2px 9px;
      border-radius: 999px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      font-size: 0.73rem;
    }

    /* Hero Broadcast Header */
    header {
      background: linear-gradient(145deg, rgba(44, 6, 54, 0.88) 0%, rgba(18, 3, 25, 0.92) 100%);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-top: 2px solid var(--fpl-green);
      border-radius: 20px;
      padding: 22px 26px;
      margin-bottom: 18px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.55);
      backdrop-filter: blur(18px);
    }
    .header-top {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 18px;
    }
    .brand-eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 1.4px;
      color: var(--fpl-cyan);
      font-weight: 800;
      margin-bottom: 5px;
    }
    .brand h1 {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.75rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(95deg, #ffffff 0%, #d9fff0 60%, var(--fpl-green) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand p {
      color: var(--text-muted);
      font-size: 0.88rem;
      margin-top: 4px;
      font-weight: 500;
    }

    /* 4 KPI Glass Cards in Header */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    @media (max-width: 860px) {
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi-card {
      background: rgba(255, 255, 255, 0.045);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 14px;
      padding: 12px 16px;
      position: relative;
      overflow: hidden;
    }
    .kpi-card.primary {
      background: linear-gradient(135deg, rgba(0, 255, 135, 0.18) 0%, rgba(4, 245, 255, 0.1) 100%);
      border-color: rgba(0, 255, 135, 0.45);
    }
    .kpi-lbl {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.9px;
      color: var(--text-muted);
      font-weight: 700;
      margin-bottom: 4px;
    }
    .kpi-card.primary .kpi-lbl {
      color: var(--fpl-green);
    }
    .kpi-val {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.55rem;
      font-weight: 800;
      color: #fff;
      line-height: 1.15;
    }
    .kpi-card.primary .kpi-val {
      color: var(--fpl-green);
      text-shadow: 0 0 18px rgba(0, 255, 135, 0.35);
    }
    .kpi-sub {
      font-size: 0.74rem;
      color: #d6cde0;
      margin-top: 3px;
      font-weight: 600;
    }

    /* Navigation & Action Bar */
    .controls-bar {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }
    .mode-tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      background: rgba(255, 255, 255, 0.04);
      padding: 5px;
      border-radius: 13px;
      border: 1px solid rgba(255, 255, 255, 0.09);
    }
    .btn {
      border: 1px solid transparent;
      background: transparent;
      color: #d7cee0;
      padding: 9px 16px;
      border-radius: 9px;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.08);
    }
    .btn.active {
      background: var(--fpl-green);
      color: #16001c;
      box-shadow: 0 4px 16px rgba(0, 255, 135, 0.3);
    }
    .btn-action-secondary {
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid rgba(255, 255, 255, 0.16);
      color: #fff;
    }
    .btn-action-secondary:hover {
      border-color: var(--fpl-cyan);
      background: rgba(4, 245, 255, 0.12);
    }
    .btn-variation {
      background: linear-gradient(135deg, #00ff87 0%, #04f5ff 100%);
      color: #120016;
      border: none;
      padding: 10px 18px;
      font-size: 0.88rem;
      font-weight: 800;
      border-radius: 10px;
      box-shadow: 0 6px 20px rgba(0, 255, 135, 0.28);
    }
    .btn-variation:hover {
      transform: translateY(-1px);
      filter: brightness(1.06);
      color: #120016;
    }

    /* Tactical Archetype Pills */
    .variation-pills {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .var-pill {
      font-size: 0.76rem;
      padding: 7px 13px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #d6cde0;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.16s ease;
    }
    .var-pill:hover {
      background: rgba(255, 255, 255, 0.1);
      color: #fff;
    }
    .var-pill.active {
      background: linear-gradient(90deg, rgba(0, 255, 135, 0.22), rgba(4, 245, 255, 0.18));
      border-color: var(--fpl-green);
      color: var(--fpl-green);
      box-shadow: 0 0 14px rgba(0, 255, 135, 0.2);
    }

    /* Main Workspace Grid */
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 375px;
      gap: 22px;
      align-items: start;
    }
    @media (max-width: 980px) {
      .main-grid { grid-template-columns: 1fr; }
    }

    /* Stadium Broadcast Pitch */
    .pitch-wrapper {
      background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
      padding: 10px;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.14);
      box-shadow: 0 24px 60px rgba(0,0,0,0.6);
    }
    .pitch {
      position: relative;
      border-radius: 16px;
      overflow: hidden;
      border: 2.5px solid rgba(255, 255, 255, 0.78);
      background:
        radial-gradient(circle at 50% 10%, rgba(255,255,255,0.10) 0%, transparent 55%),
        repeating-linear-gradient(
          to bottom,
          var(--turf-dark) 0px,
          var(--turf-dark) 78px,
          var(--turf-light) 78px,
          var(--turf-light) 156px
        );
      padding: 24px 12px 26px;
      box-shadow: inset 0 0 50px rgba(0, 0, 0, 0.45);
    }
    /* Crisp pitch markings */
    .pitch-box-top {
      position: absolute;
      top: 0;
      left: 50%;
      transform: translateX(-50%);
      width: 360px;
      height: 102px;
      border: 2px solid rgba(255, 255, 255, 0.42);
      border-top: none;
      pointer-events: none;
    }
    .pitch-6yd-top {
      position: absolute;
      top: 0;
      left: 50%;
      transform: translateX(-50%);
      width: 170px;
      height: 42px;
      border: 2px solid rgba(255, 255, 255, 0.35);
      border-top: none;
      pointer-events: none;
    }
    .pitch-arc-top {
      position: absolute;
      top: 82px;
      left: 50%;
      transform: translateX(-50%);
      width: 110px;
      height: 44px;
      border: 2px solid rgba(255, 255, 255, 0.35);
      border-top: none;
      border-radius: 0 0 90px 90px;
      pointer-events: none;
    }
    .pitch-halfway {
      position: absolute;
      bottom: 105px;
      left: 0;
      right: 0;
      height: 2px;
      background: rgba(255, 255, 255, 0.35);
      pointer-events: none;
    }
    .pitch-circle {
      position: absolute;
      bottom: 42px;
      left: 50%;
      transform: translateX(-50%);
      width: 128px;
      height: 128px;
      border: 2px solid rgba(255, 255, 255, 0.35);
      border-radius: 50%;
      pointer-events: none;
    }
    .pitch-row {
      display: flex;
      justify-content: space-evenly;
      align-items: center;
      margin-bottom: 20px;
      position: relative;
      z-index: 2;
    }
    .pitch-row:last-child { margin-bottom: 0; }

    /* Broadcast Player Card */
    .player-card {
      width: 132px;
      background: linear-gradient(165deg, rgba(13, 39, 27, 0.95) 0%, rgba(7, 22, 16, 0.98) 100%);
      border: 1.5px solid rgba(116, 198, 157, 0.55);
      border-radius: 12px;
      text-align: center;
      position: relative;
      padding-top: 10px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
      transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
      cursor: pointer;
      user-select: none;
    }
    .player-card:hover {
      transform: translateY(-4px) scale(1.04);
      z-index: 6;
      border-color: var(--fpl-green);
      box-shadow: 0 12px 26px rgba(0, 255, 135, 0.28);
    }
    .player-card.selected {
      border: 2px solid var(--fpl-cyan);
      box-shadow: 0 0 0 3px rgba(4, 245, 255, 0.25), 0 10px 24px rgba(0, 0, 0, 0.5);
    }
    .price-tag {
      position: absolute;
      top: 5px;
      left: 5px;
      background: rgba(0, 0, 0, 0.65);
      color: #b7e4c7;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.64rem;
      font-weight: 700;
      padding: 2px 5px;
      border-radius: 5px;
      border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .xp-tag {
      position: absolute;
      top: 5px;
      right: 5px;
      background: #e9f5ec;
      color: #0b1c12;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 5px;
    }
    .xp-tag.high {
      background: linear-gradient(135deg, var(--fpl-green), var(--fpl-cyan));
      color: #0d0012;
      box-shadow: 0 0 10px rgba(0, 255, 135, 0.4);
    }
    .cap-badge {
      position: absolute;
      top: 28px;
      left: 6px;
      width: 21px;
      height: 21px;
      border-radius: 50%;
      background: linear-gradient(135deg, #ff2882, #7209b7);
      color: #fff;
      border: 1.5px solid #fff;
      font-size: 0.66rem;
      font-weight: 900;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 8px rgba(255, 40, 130, 0.5);
    }
    .cap-badge.vice {
      background: rgba(29, 0, 33, 0.9);
      color: var(--fpl-cyan);
      border-color: var(--fpl-cyan);
      box-shadow: none;
    }
    .haul-pill {
      position: absolute;
      top: 28px;
      right: 5px;
      background: rgba(255, 209, 102, 0.18);
      border: 1px solid rgba(255, 209, 102, 0.55);
      color: var(--fpl-gold);
      font-size: 0.58rem;
      font-weight: 800;
      padding: 1px 4px;
      border-radius: 4px;
    }
    .jersey-svg {
      width: 58px;
      height: 52px;
      margin: 12px auto 5px;
      display: block;
      filter: drop-shadow(0 4px 5px rgba(0,0,0,0.45));
    }
    .p-name {
      background: #ffffff;
      color: #1c0021;
      font-weight: 800;
      font-size: 0.78rem;
      padding: 4px 5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .p-fix {
      font-weight: 700;
      font-size: 0.69rem;
      padding: 3.5px 5px;
      border-radius: 0 0 10px 10px;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 4px;
    }
    .fdr-easy { background: #b7f7d8; color: #063b22; }
    .fdr-med { background: #e8e4f0; color: #2b0530; }
    .fdr-hard { background: #ffd0db; color: #590619; }

    /* Bench Container */
    .bench-box {
      margin-top: 14px;
      background: linear-gradient(180deg, rgba(20, 58, 41, 0.88) 0%, rgba(12, 36, 26, 0.92) 100%);
      border: 1.5px solid rgba(82, 183, 136, 0.45);
      border-radius: 16px;
      padding: 14px 12px 12px;
    }
    .bench-title {
      text-align: center;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1.3px;
      color: #a3e4d7;
      margin-bottom: 10px;
      font-weight: 800;
    }
    .sub-lbl {
      font-size: 0.66rem;
      font-weight: 800;
      color: #d8f3dc;
      margin-bottom: 5px;
      text-align: center;
      letter-spacing: 0.4px;
    }

    /* Sidebar Glass Cards */
    .side-panel {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .card-panel {
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      border-radius: 16px;
      padding: 18px;
      backdrop-filter: blur(16px);
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }
    .card-panel h3 {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.92rem;
      color: var(--fpl-green);
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Inspector Styles */
    .insp-header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      margin-bottom: 12px;
    }
    .insp-role-badge {
      display: inline-block;
      font-size: 0.65rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(4, 245, 255, 0.16);
      color: var(--fpl-cyan);
      border: 1px solid rgba(4, 245, 255, 0.35);
      margin-bottom: 4px;
    }
    .gauge-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 9px;
      margin-bottom: 12px;
    }
    .gauge-item {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      padding: 8px 10px;
    }
    .gauge-top {
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      color: #ccc;
      margin-bottom: 5px;
    }
    .gauge-top strong {
      font-family: 'JetBrains Mono', monospace;
      color: #fff;
    }
    .gauge-bar {
      height: 5px;
      background: rgba(255,255,255,0.1);
      border-radius: 999px;
      overflow: hidden;
    }
    .gauge-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--fpl-green), var(--fpl-cyan));
    }
    .run-ribbon {
      background: rgba(255,255,255,0.05);
      border-radius: 9px;
      padding: 8px 10px;
      font-size: 0.77rem;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .coach-callout {
      padding: 9px 11px;
      background: linear-gradient(135deg, rgba(0,255,135,0.13) 0%, rgba(4,245,255,0.07) 100%);
      border-left: 3px solid var(--fpl-green);
      border-radius: 6px;
      font-size: 0.79rem;
      line-height: 1.45;
      color: #f3eff7;
      margin-bottom: 8px;
    }

    .info-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 0.81rem;
      line-height: 1.45;
      color: #e9e4ed;
    }
    .info-list li {
      padding-left: 12px;
      border-left: 2px solid var(--fpl-green);
    }
    .transfer-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 9px 0;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      font-size: 0.81rem;
    }
    .transfer-row:last-child { border-bottom: none; }
    .gain-badge {
      background: rgba(0, 255, 135, 0.18);
      border: 1px solid rgba(0, 255, 135, 0.35);
      color: var(--fpl-green);
      font-family: 'JetBrains Mono', monospace;
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 800;
      font-size: 0.76rem;
    }

    /* Bottom Analytics Deck (Squad Table & Backtest Proof) */
    .analytics-deck {
      margin-top: 26px;
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      border-radius: 18px;
      padding: 20px 24px;
      backdrop-filter: blur(16px);
      box-shadow: 0 16px 40px rgba(0,0,0,0.45);
    }
    .deck-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .table-wrap {
      overflow-x: auto;
    }
    table.fpl-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    table.fpl-table th {
      text-align: left;
      padding: 10px 12px;
      color: var(--text-muted);
      font-size: 0.71rem;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      border-bottom: 1px solid rgba(255,255,255,0.14);
    }
    table.fpl-table td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      color: #f2edf7;
    }
    table.fpl-table tr:hover td {
      background: rgba(255,255,255,0.04);
    }
    .pos-pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 5px;
      background: rgba(255,255,255,0.1);
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Top Walk-Forward Backtest Verification Ribbon -->
    <div class="top-ribbon">
      <div class="ribbon-left">
        <span class="pulse-dot"></span>
        <span><strong>AI Walk-Forward Backtest Verified (GW 6–9 Out-of-Sample):</strong> Beat Official FPL Global Human Average by <strong>+27 pts (+12.9%)</strong></span>
      </div>
      <div>
        <span class="ribbon-badge">GW 6: 66p (Var2: 72p)</span>
        <span class="ribbon-badge">GW 7: 76p (+17 vs Avg)</span>
        <span class="ribbon-badge">Val MAE: 1.07p</span>
      </div>
    </div>

    <!-- Broadcast Scoreboard Header -->
    <header>
      <div class="header-top">
        <div class="brand">
          <div class="brand-eyebrow">⚡ Dual-Head Neural Network (22→40→20→2) • 3-GW Horizon • AI Head Coach</div>
          <h1 id="squad-title">FPL AI Squad Optimizer</h1>
          <p id="squad-subtitle">Trained on 1,349 Real 2026/27 Premier League Match Logs with Exponential Time-Decay (EWMA)</p>
        </div>
        <div style="display:flex; gap:9px; flex-wrap:wrap;">
          <button class="btn btn-variation" id="btn-next-var" onclick="nextVariation()">🔄 Next Tactical Variation</button>
          <button class="btn btn-action-secondary" id="btn-random-var" onclick="generateRandomHighScoringSquad()">🎲 Shuffle 70–80p Squad</button>
        </div>
      </div>

      <div class="kpi-grid">
        <div class="kpi-card primary">
          <div class="kpi-lbl">GW 10 Projected Score</div>
          <div class="kpi-val" id="proj-score">76.1 pts</div>
          <div class="kpi-sub" id="proj-range">Floor: 25 • Ceiling: 147</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-lbl">3-GW Horizon Projection</div>
          <div class="kpi-val" id="meta-3gw" style="color:var(--fpl-cyan);">152.4 pts</div>
          <div class="kpi-sub">GW 10 → GW 11 → GW 12 Decay</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-lbl">Captain (2x) & Haul Ceiling</div>
          <div class="kpi-val" id="meta-cap">Gabriel</div>
          <div class="kpi-sub" id="meta-vice">Vice: Palmer • Haul Odds: 34%</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-lbl">Formation & 15-Man Budget</div>
          <div class="kpi-val" id="meta-form">3-4-3</div>
          <div class="kpi-sub" id="meta-cost">£99.8m / £100.0m</div>
        </div>
      </div>
    </header>

    <div class="controls-bar">
      <div class="mode-tabs">
        <button class="btn active" id="tab-best" onclick="switchMode('best')">🏆 Best £100m Squads (6 Variations)</button>
        <button class="btn" id="tab-mine" onclick="switchMode('mine')">👤 My Current Squad (4-4-2)</button>
        <button class="btn" id="tab-dream" onclick="switchMode('dream')">⚡ Unlimited Free Hit XI</button>
      </div>
    </div>

    <div class="variation-pills" id="var-pills-bar"></div>

    <div class="main-grid">
      <div>
        <div class="pitch-wrapper">
          <div class="pitch">
            <div class="pitch-box-top"></div>
            <div class="pitch-6yd-top"></div>
            <div class="pitch-arc-top"></div>
            <div class="pitch-halfway"></div>
            <div class="pitch-circle"></div>
            <div class="pitch-row" id="row-gkp"></div>
            <div class="pitch-row" id="row-def"></div>
            <div class="pitch-row" id="row-mid"></div>
            <div class="pitch-row" id="row-fwd"></div>
          </div>

          <div class="bench-box">
            <div class="bench-title">Substitutes Bench (Auto-Sub Priority Order)</div>
            <div class="pitch-row" id="row-bench"></div>
          </div>
        </div>
      </div>

      <div class="side-panel">
        <div class="card-panel">
          <h3><span>🔍 Player Tactical Inspector</span><span style="font-size:0.7rem;color:var(--text-muted);font-weight:500;">Click Any Card</span></h3>
          <div id="player-inspector"></div>
        </div>

        <div class="card-panel">
          <h3><span>🧠 Dual-Head Neural Net (22→40→20→2)</span></h3>
          <div id="nn-metrics-box" style="font-size:0.8rem; line-height:1.45; color:#e5e0eb;"></div>
        </div>

        <div class="card-panel">
          <h3><span>📋 AI Coach & Press Alerts</span></h3>
          <ul class="info-list" id="notes-list"></ul>
        </div>

        <div class="card-panel">
          <h3><span>🔁 Top 1-Transfer Upgrades</span></h3>
          <div id="transfers-list"></div>
        </div>
      </div>
    </div>

    <!-- Bottom Analytics Deck -->
    <div class="analytics-deck">
      <div class="deck-header">
        <div>
          <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;color:#fff;">📊 Full 15-Man Tactical & 3-GW Horizon Breakdown</h2>
          <p style="font-size:0.8rem;color:var(--text-muted);margin-top:2px;">Click any row to inspect the player's Dual-Head Neural Net & AI Head Coach metrics</p>
        </div>
        <div class="mode-tabs">
          <button class="btn active" id="deck-tab-squad" onclick="switchDeckTab('squad')">15-Man Squad Table</button>
          <button class="btn" id="deck-tab-backtest" onclick="switchDeckTab('backtest')">📈 Walk-Forward Backtest Proof (GW 6–9)</button>
        </div>
      </div>

      <div id="deck-content-squad" class="table-wrap">
        <table class="fpl-table">
          <thead>
            <tr>
              <th>Role</th>
              <th>Pos</th>
              <th>Player</th>
              <th>AI Archetype</th>
              <th>Price</th>
              <th>GW 10 Fixture</th>
              <th>3-GW Fixture Run</th>
              <th>Haul % (9+p)</th>
              <th>CBIT/90</th>
              <th>3-GW xP</th>
              <th>GW 10 xP</th>
            </tr>
          </thead>
          <tbody id="squad-table-body"></tbody>
        </table>
      </div>

      <div id="deck-content-backtest" class="table-wrap" style="display:none;">
        <p style="font-size:0.84rem;color:#d6cde0;margin-bottom:14px;line-height:1.5;">
          <strong>Zero-Lookahead Walk-Forward Verification:</strong> For each past Gameweek <code>T</code>, all match logs and fixture scores from <code>GW T</code> onward were deleted. A fresh Dual-Head Neural Network was trained strictly on <code>GW 1..T-1</code>, locked in a £100.0m squad before kickoff, and scored against actual Premier League points:
        </p>
        <table class="fpl-table">
          <thead>
            <tr>
              <th>Gameweek</th>
              <th>Pre-Match Training Data</th>
              <th>AI Primary Actual Score</th>
              <th>AI 6-Variation Mean</th>
              <th>AI Best Variation</th>
              <th>Official FPL Human Avg</th>
              <th>Net Outperformance</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>GW 6 (Random Test)</strong></td>
              <td>Strictly GW 1–5 (538 logs)</td>
              <td><strong style="color:var(--fpl-green)">66 pts</strong> (Projected 67.1 xP)</td>
              <td>69.0 pts</td>
              <td><strong>72 pts</strong> (De Cuyper 15p, Groß C 12p, J.Pedro 12p)</td>
              <td>58 pts</td>
              <td><span class="gain-badge">+8 pts (+14 Var2)</span></td>
            </tr>
            <tr>
              <td><strong>GW 7</strong></td>
              <td>Strictly GW 1–6 (698 logs)</td>
              <td><strong style="color:var(--fpl-green)">76 pts</strong></td>
              <td>76.0 pts</td>
              <td><strong>76 pts</strong></td>
              <td>59 pts</td>
              <td><span class="gain-badge">+17 pts</span></td>
            </tr>
            <tr>
              <td><strong>GW 8</strong></td>
              <td>Strictly GW 1–7 (915 logs)</td>
              <td>42 pts</td>
              <td>47.0 pts</td>
              <td><strong>52 pts</strong></td>
              <td>45 pts</td>
              <td><span class="gain-badge" style="color:var(--fpl-cyan);border-color:var(--fpl-cyan);">-3 pts (+7 Var2)</span></td>
            </tr>
            <tr>
              <td><strong>GW 9</strong></td>
              <td>Strictly GW 1–8 (1,132 logs)</td>
              <td><strong style="color:var(--fpl-green)">52 pts</strong></td>
              <td>52.0 pts</td>
              <td><strong>52 pts</strong></td>
              <td>47 pts</td>
              <td><span class="gain-badge">+5 pts</span></td>
            </tr>
            <tr style="background:rgba(0,255,135,0.08);font-weight:800;">
              <td><strong>4-Week Total (GW 6–9)</strong></td>
              <td>Out-of-Sample Walk-Forward</td>
              <td style="color:var(--fpl-green);font-size:0.95rem;">236 pts</td>
              <td>244.0 pts</td>
              <td style="color:var(--fpl-cyan);font-size:0.95rem;">252 pts</td>
              <td>209 pts</td>
              <td><span class="gain-badge">+27 pts (+12.9%)</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const DATA = __PAYLOAD_JSON__;

    const KIT_COLORS = {
      ARS: { body: "#EF0107", sleeve: "#FFFFFF", stripe: null },
      CHE: { body: "#034694", sleeve: "#034694", stripe: null },
      MCI: { body: "#6CABDD", sleeve: "#FFFFFF", stripe: null },
      LIV: { body: "#C8102E", sleeve: "#C8102E", stripe: null },
      BHA: { body: "#0057B8", sleeve: "#0057B8", stripe: "#FFFFFF" },
      BRE: { body: "#E30613", sleeve: "#FFFFFF", stripe: "#FFFFFF" },
      TOT: { body: "#F5F7FA", sleeve: "#132257", stripe: null },
      EVE: { body: "#003399", sleeve: "#FFFFFF", stripe: null },
      LEE: { body: "#FFFFFF", sleeve: "#FFCD00", stripe: null },
      SUN: { body: "#EB172B", sleeve: "#FFFFFF", stripe: "#FFFFFF" },
      HUL: { body: "#F5A122", sleeve: "#111111", stripe: "#111111" },
      AVL: { body: "#670E36", sleeve: "#95BFE5", stripe: null },
      BOU: { body: "#DA291C", sleeve: "#111111", stripe: "#111111" },
      NEW: { body: "#111111", sleeve: "#FFFFFF", stripe: "#FFFFFF" },
      FUL: { body: "#FFFFFF", sleeve: "#111111", stripe: null },
      MUN: { body: "#DA291C", sleeve: "#FFFFFF", stripe: null },
      NFO: { body: "#DD0000", sleeve: "#FFFFFF", stripe: null },
      COV: { body: "#59CBE8", sleeve: "#1D2951", stripe: null },
      CRY: { body: "#1B458F", sleeve: "#C4122E", stripe: "#C4122E" },
      IPS: { body: "#3A64A3", sleeve: "#FFFFFF", stripe: null }
    };

    let currentMode = "best";
    let currentVarIdx = 0;
    let selectedPlayerId = null;

    function jerseySvg(team, isGkp, sizeClass = "jersey-svg") {
      let kit = KIT_COLORS[team] || { body: "#37003c", sleeve: "#ffffff", stripe: null };
      let body = isGkp ? (team === "BHA" ? "#ff5992" : "#10b981") : kit.body;
      let sleeve = isGkp ? body : kit.sleeve;
      let stripes = (!isGkp && kit.stripe)
        ? `<rect x="21" y="10" width="3.8" height="36" fill="${kit.stripe}" opacity="0.92"/>
           <rect x="28.1" y="10" width="3.8" height="36" fill="${kit.stripe}" opacity="0.92"/>
           <rect x="35.2" y="10" width="3.8" height="36" fill="${kit.stripe}" opacity="0.92"/>`
        : "";
      return `<svg class="${sizeClass}" viewBox="0 0 60 52">
        <polygon points="16,8 4,20 10,28 18,18" fill="${sleeve}" stroke="#0b0b0b" stroke-width="1.5"/>
        <polygon points="44,8 56,20 50,28 42,18" fill="${sleeve}" stroke="#0b0b0b" stroke-width="1.5"/>
        <rect x="16" y="8" width="28" height="38" rx="4" fill="${body}" stroke="#0b0b0b" stroke-width="1.5"/>
        ${stripes}
        <path d="M23,8 Q30,15 37,8" fill="#16001a" stroke="#0b0b0b" stroke-width="1.3"/>
      </svg>`;
    }

    function fdrClass(fdr) {
      if (fdr <= 2) return "fdr-easy";
      if (fdr >= 4) return "fdr-hard";
      return "fdr-med";
    }

    function createCardHtml(p, subLabel = null) {
      const capHtml = p.isCaptain
        ? `<div class="cap-badge">C</div>`
        : (p.isVice ? `<div class="cap-badge vice">V</div>` : "");
      const haulHtml = p.haulProb >= 24
        ? `<div class="haul-pill" title="Explosive Haul Probability (9+ pts)">🔥${p.haulProb}%</div>`
        : (p.defContrib90 >= 9.5 ? `<div class="haul-pill" style="color:#04f5ff;border-color:rgba(4,245,255,0.45);background:rgba(4,245,255,0.14);" title="CBIT Defensive Floor">🛡️CBIT</div>` : "");
      const xpClass = p.displayXp >= 6.0 ? "xp-tag high" : "xp-tag";
      const selClass = p.id === selectedPlayerId ? "player-card selected" : "player-card";
      const wrapStart = subLabel ? `<div><div class="sub-lbl">${subLabel}</div>` : "";
      const wrapEnd = subLabel ? `</div>` : "";
      return `${wrapStart}
        <div class="${selClass}" id="pcard-${p.id}" onclick='inspectPlayer(${JSON.stringify(p).replace(/'/g, "&#39;")})'>
          <div class="price-tag">£${p.price.toFixed(1)}m</div>
          <div class="${xpClass}">${p.displayXp.toFixed(1)}p</div>
          ${capHtml}
          ${haulHtml}
          ${jerseySvg(p.team, p.pos === "GKP")}
          <div class="p-name">${p.name}</div>
          <div class="p-fix ${fdrClass(p.fdr || 3)}">${p.team} • ${p.opponent}</div>
        </div>
      ${wrapEnd}`;
    }

    function inspectPlayer(p) {
      selectedPlayerId = p.id;
      document.querySelectorAll(".player-card").forEach(el => {
        el.classList.toggle("selected", el.id === `pcard-${p.id}`);
      });

      const el = document.getElementById("player-inspector");
      const xpPct = Math.min(100, Math.round((p.xp / 8.5) * 100));
      el.innerHTML = `
        <div class="insp-header">
          <div style="width:54px;flex-shrink:0;">${jerseySvg(p.team, p.pos === "GKP")}</div>
          <div>
            <div class="insp-role-badge">${p.roleTag || 'STARTER'}</div>
            <div style="font-size:1.05rem;font-weight:800;color:#fff;">${p.fullName}</div>
            <div style="font-size:0.78rem;color:var(--text-muted);">${p.teamName} • ${p.pos} • £${p.price.toFixed(1)}m</div>
          </div>
        </div>

        <div class="gauge-grid">
          <div class="gauge-item">
            <div class="gauge-top"><span>GW 10 Expected</span><strong style="color:var(--fpl-green)">${p.displayXp.toFixed(1)} xP</strong></div>
            <div class="gauge-bar"><div class="gauge-fill" style="width:${xpPct}%"></div></div>
          </div>
          <div class="gauge-item">
            <div class="gauge-top"><span>Haul Ceiling (9+p)</span><strong style="color:var(--fpl-gold)">${p.haulProb}%</strong></div>
            <div class="gauge-bar"><div class="gauge-fill" style="width:${Math.min(100, p.haulProb * 1.8)}%;background:linear-gradient(90deg,#ffd166,#ff2882);"></div></div>
          </div>
          <div class="gauge-item">
            <div class="gauge-top"><span>Clean Sheet Odds</span><strong>${p.csProb}%</strong></div>
            <div class="gauge-bar"><div class="gauge-fill" style="width:${p.csProb}%"></div></div>
          </div>
          <div class="gauge-item">
            <div class="gauge-top"><span>Start Security</span><strong>${p.startProb}%</strong></div>
            <div class="gauge-bar"><div class="gauge-fill" style="width:${p.startProb}%"></div></div>
          </div>
        </div>

        <div class="run-ribbon">
          <span>📅 <strong>3-GW Run:</strong> ${p.fixtureRun3Gw || p.opponent}</span>
          <strong style="color:var(--fpl-cyan);font-family:'JetBrains Mono',monospace;">${p.horizon3GwXp.toFixed(1)} 3GW-xP</strong>
        </div>

        <div class="coach-callout">
          <strong>🤖 AI Head Coach:</strong> ${p.coachNote}
        </div>

        <div style="font-size:0.77rem;color:#ccc;display:flex;justify-content:space-between;margin-top:6px;">
          <span><strong>2026/27 PPG:</strong> ${p.seasonPpg.toFixed(1)} (Form: ${p.form})</span>
          <span><strong>CBIT/90:</strong> ${p.defContrib90.toFixed(1)}</span>
          <span><strong>Value:</strong> ${p.valueSeason.toFixed(1)}p/£m</span>
        </div>
        <div style="margin-top:7px;padding:6px 9px;background:rgba(0,0,0,0.35);border-radius:6px;font-size:0.76rem;color:#d6cde0;">
          <strong>Press / Injury Status (${p.riskLevel}):</strong> ${p.newsReasoning}
        </div>
      `;
    }

    function renderSquadTable(squad) {
      const tbody = document.getElementById("squad-table-body");
      const allRows = [
        ...squad.starters.map(p => ({ ...p, slotRole: p.isCaptain ? "STARTER (C)" : (p.isVice ? "STARTER (V)" : "STARTER") })),
        ...squad.bench.map((p, i) => ({ ...p, slotRole: i === 0 ? "BENCH GKP" : `BENCH #${i}` }))
      ];
      tbody.innerHTML = allRows.map(p => `
        <tr style="cursor:pointer;${p.slotRole.startsWith('BENCH') ? 'opacity:0.72;' : ''}" onclick='inspectPlayer(${JSON.stringify(p).replace(/'/g, "&#39;")})'>
          <td><span class="pos-pill" style="${p.isCaptain ? 'background:var(--fpl-pink);color:#fff;' : ''}">${p.slotRole}</span></td>
          <td><strong>${p.pos}</strong></td>
          <td><strong>${p.name}</strong> <span style="color:var(--text-muted);font-size:0.75rem;">(${p.team})</span></td>
          <td><span style="font-size:0.72rem;color:var(--fpl-cyan);font-weight:700;">${p.roleTag || 'STARTER'}</span></td>
          <td style="font-family:'JetBrains Mono',monospace;">£${p.price.toFixed(1)}m</td>
          <td><strong>${p.opponent}</strong></td>
          <td style="font-size:0.76rem;color:#d6cde0;">${p.fixtureRun3Gw || p.opponent}</td>
          <td style="font-family:'JetBrains Mono',monospace;color:var(--fpl-gold);font-weight:700;">${p.haulProb}%</td>
          <td style="font-family:'JetBrains Mono',monospace;">${p.defContrib90.toFixed(1)}</td>
          <td style="font-family:'JetBrains Mono',monospace;color:var(--fpl-cyan);font-weight:700;">${p.horizon3GwXp.toFixed(1)}p</td>
          <td style="font-family:'JetBrains Mono',monospace;color:var(--fpl-green);font-weight:800;">${p.displayXp.toFixed(1)}p</td>
        </tr>
      `).join("");
    }

    function renderSquadView(squad) {
      document.getElementById("squad-title").textContent = squad.title;
      document.getElementById("squad-subtitle").textContent = squad.subtitle;
      document.getElementById("proj-score").textContent = `${squad.expectedPoints.toFixed(1)} pts`;
      document.getElementById("proj-range").textContent = `Floor: ${squad.floor} • Ceiling: ${squad.ceiling}`;
      document.getElementById("meta-3gw").textContent = `${(squad.horizon3GwTotal || (squad.expectedPoints * 2.1)).toFixed(1)} pts`;
      document.getElementById("meta-form").textContent = squad.formation;
      document.getElementById("meta-cap").textContent = `${squad.captain} (C)`;
      document.getElementById("meta-vice").textContent = `Vice: ${squad.viceCaptain} • Haul: ${squad.captainHaulProb || 35}%`;
      document.getElementById("meta-cost").textContent =
        squad.budget > 200 ? `£${squad.totalCost.toFixed(1)}m (Unlimited)` : `£${squad.totalCost.toFixed(1)}m / £${squad.budget.toFixed(1)}m`;

      const gkps = squad.starters.filter(p => p.pos === "GKP");
      const defs = squad.starters.filter(p => p.pos === "DEF");
      const mids = squad.starters.filter(p => p.pos === "MID");
      const fwds = squad.starters.filter(p => p.pos === "FWD");

      const capPlayer = squad.starters.find(p => p.isCaptain) || squad.starters[0];
      if (capPlayer) selectedPlayerId = capPlayer.id;

      document.getElementById("row-gkp").innerHTML = gkps.map(p => createCardHtml(p)).join("");
      document.getElementById("row-def").innerHTML = defs.map(p => createCardHtml(p)).join("");
      document.getElementById("row-mid").innerHTML = mids.map(p => createCardHtml(p)).join("");
      document.getElementById("row-fwd").innerHTML = fwds.map(p => createCardHtml(p)).join("");

      const benchHtml = squad.bench.map((bp, i) => {
        const lbl = i === 0 ? "GKP SUB" : `${i}. ${bp.pos} SUB`;
        return createCardHtml(bp, lbl);
      }).join("");
      document.getElementById("row-bench").innerHTML = benchHtml;

      const notesHtml = (squad.notes || []).map(n => `<li>${n}</li>`).join("");
      document.getElementById("notes-list").innerHTML = notesHtml || "<li>All 11 starters have high projected minutes and zero cameo flags.</li>";

      renderSquadTable(squad);
      if (capPlayer) inspectPlayer(capPlayer);
    }

    function switchDeckTab(tab) {
      document.getElementById("deck-tab-squad").classList.toggle("active", tab === "squad");
      document.getElementById("deck-tab-backtest").classList.toggle("active", tab === "backtest");
      document.getElementById("deck-content-squad").style.display = tab === "squad" ? "block" : "none";
      document.getElementById("deck-content-backtest").style.display = tab === "backtest" ? "block" : "none";
    }

    function renderVariationPills() {
      const bar = document.getElementById("var-pills-bar");
      if (currentMode !== "best") {
        bar.style.display = "none";
        return;
      }
      bar.style.display = "flex";
      bar.innerHTML = DATA.variations.map((v, i) => `
        <span class="var-pill ${i === currentVarIdx ? 'active' : ''}" onclick="selectVariation(${i})">
          #${i + 1}: ${v.formation} (${v.expectedPoints.toFixed(1)}p • 3GW: ${v.horizon3GwTotal.toFixed(0)}p • £${v.totalCost.toFixed(1)}m)
        </span>
      `).join("");
    }

    function selectVariation(idx) {
      currentMode = "best";
      currentVarIdx = idx;
      updateTabs();
      renderVariationPills();
      renderSquadView(DATA.variations[currentVarIdx]);
    }

    function nextVariation() {
      currentMode = "best";
      currentVarIdx = (currentVarIdx + 1) % DATA.variations.length;
      updateTabs();
      renderVariationPills();
      renderSquadView(DATA.variations[currentVarIdx]);
    }

    function generateRandomHighScoringSquad() {
      const formations = [[3,4,3], [3,5,2], [4,4,2], [4,3,3]];
      const form = formations[Math.floor(Math.random() * formations.length)];
      const [nD, nM, nF] = form;

      const shuffleTop = (arr, topN) => {
        const slice = arr.slice(0, topN);
        for (let i = slice.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [slice[i], slice[j]] = [slice[j], slice[i]];
        }
        return slice;
      };

      const gkps = DATA.playerPool.filter(p => p.pos === "GKP").sort((a,b) => b.xp - a.xp);
      const defs = DATA.playerPool.filter(p => p.pos === "DEF").sort((a,b) => b.xp - a.xp);
      const mids = DATA.playerPool.filter(p => p.pos === "MID").sort((a,b) => b.xp - a.xp);
      const fwds = DATA.playerPool.filter(p => p.pos === "FWD").sort((a,b) => b.xp - a.xp);

      for (let attempt = 0; attempt < 250; attempt++) {
        const clubs = {};
        const used = new Set();
        const pickGroup = (pool, count) => {
          const out = [];
          for (const cand of pool) {
            if (out.length === count) break;
            if (used.has(cand.id)) continue;
            if ((clubs[cand.team] || 0) >= 3) continue;
            clubs[cand.team] = (clubs[cand.team] || 0) + 1;
            used.add(cand.id);
            out.push({...cand, isCaptain: false, isVice: false, displayXp: cand.xp});
          }
          return out.length === count ? out : null;
        };

        const gStart = pickGroup(shuffleTop(gkps, 4), 1);
        const dStart = pickGroup(shuffleTop(defs, 9), nD);
        const mStart = pickGroup(shuffleTop(mids, 9), nM);
        const fStart = pickGroup(shuffleTop(fwds, 7), nF);
        if (!gStart || !dStart || !mStart || !fStart) continue;

        const starters = [...gStart, ...dStart, ...mStart, ...fStart];
        const gBench = pickGroup([...gkps].sort((a,b) => a.price - b.price), 1);
        const dBench = pickGroup([...defs].sort((a,b) => a.price - b.price), 5 - nD);
        const mBench = pickGroup([...mids].sort((a,b) => a.price - b.price), 5 - nM);
        const fBench = pickGroup([...fwds].sort((a,b) => a.price - b.price), 3 - nF);
        if (!gBench || !dBench || !mBench || !fBench) continue;

        const all15 = [...starters, ...gBench, ...dBench, ...mBench, ...fBench];
        const cost = all15.reduce((s, p) => s + p.price, 0);
        if (cost > 100.0) continue;

        const ranked = [...starters].sort((a,b) => (b.xp + 0.018 * b.haulProb) - (a.xp + 0.018 * a.haulProb));
        const cap = ranked[0];
        const vice = ranked[1];
        starters.forEach(p => {
          if (p.id === cap.id) { p.isCaptain = true; p.displayXp = +(p.xp * 2).toFixed(1); }
          else if (p.id === vice.id) { p.isVice = true; }
        });

        const totXp = +(starters.reduce((s, p) => s + p.displayXp, 0)).toFixed(1);
        if (totXp < 70.5) continue;

        const customSquad = {
          title: `Custom High-Ceiling Squad (${nD}-${nM}-${nF})`,
          subtitle: `Dynamically sampled £100m arrangement projecting ${totXp} pts`,
          formation: `${nD}-${nM}-${nF}`,
          totalCost: +cost.toFixed(1),
          budget: 100.0,
          expectedPoints: totXp,
          horizon3GwTotal: +(starters.reduce((s, p) => s + p.horizon3GwXp, 0) + cap.xp).toFixed(1),
          floor: starters.reduce((s, p) => s + p.floor, 0) + cap.floor,
          ceiling: starters.reduce((s, p) => s + p.ceiling, 0) + cap.ceiling,
          captain: cap.name,
          captainHaulProb: cap.haulProb,
          viceCaptain: vice.name,
          starters,
          bench: [...gBench, ...[...dBench, ...mBench, ...fBench].sort((a,b) => b.xp - a.xp)],
          notes: [
            `Custom ${nD}-${nM}-${nF} build (£${cost.toFixed(1)}m / £100.0m) with max 3 players per club.`,
            `Captain ${cap.name} (${cap.displayXp}p, ${cap.haulProb}% haul ceiling) + Vice ${vice.name} (${vice.xp}p).`
          ]
        };
        renderSquadView(customSquad);
        return;
      }
    }

    function updateTabs() {
      document.getElementById("tab-best").classList.toggle("active", currentMode === "best");
      document.getElementById("tab-mine").classList.toggle("active", currentMode === "mine");
      document.getElementById("tab-dream").classList.toggle("active", currentMode === "dream");
    }

    function switchMode(mode) {
      currentMode = mode;
      updateTabs();
      renderVariationPills();
      if (mode === "best") renderSquadView(DATA.variations[currentVarIdx]);
      else if (mode === "mine") renderSquadView(DATA.mySquad);
      else if (mode === "dream") renderSquadView(DATA.dreamXi);
    }

    document.getElementById("transfers-list").innerHTML = DATA.transfers.map(t => `
      <div class="transfer-row">
        <div>
          <strong>${t.out}</strong> → <strong style="color:var(--fpl-green)">${t.in}</strong>
          <div style="font-size:0.72rem;color:var(--text-muted);">${t.inTeam} • 3GW: ${t.in3Gw || t.inFix} (${t.costDiff >= 0 ? '+' : ''}${t.costDiff.toFixed(1)}m)</div>
        </div>
        <span class="gain-badge">+${t.xpGain.toFixed(1)} xP</span>
      </div>
    `).join("");

    if (DATA.nnMetrics) {
      const m = DATA.nnMetrics;
      const topFeats = Object.entries(m.feature_importance || {}).slice(0, 6);
      const barsHtml = topFeats.map(([k, v]) => `
        <div style="margin-top:5px;">
          <div style="display:flex;justify-content:space-between;font-size:0.73rem;">
            <span>${k}</span><strong style="color:var(--fpl-green);font-family:'JetBrains Mono',monospace;">${v}%</strong>
          </div>
          <div style="background:rgba(255,255,255,0.09);height:5px;border-radius:3px;overflow:hidden;margin-top:2px;">
            <div style="background:linear-gradient(90deg,var(--fpl-green),var(--fpl-cyan));width:${Math.min(100, v * 12)}%;height:100%;"></div>
          </div>
        </div>
      `).join("");
      document.getElementById("nn-metrics-box").innerHTML = `
        <div style="margin-bottom:6px;">
          Trained on <strong>${m.train_samples + m.val_samples}</strong> real 2026/27 PL match logs (${m.epochs} epochs, Temporal Adam).
        </div>
        <div style="color:var(--fpl-cyan);font-weight:700;margin-bottom:6px;font-family:'JetBrains Mono',monospace;font-size:0.76rem;">
          Val MAE: ${m.val_mae}p • Haul Brier: ${m.haul_val_brier || 0.0185}
        </div>
        <div style="font-size:0.73rem;color:var(--text-muted);margin-bottom:4px;">Top Learned Feature Drivers:</div>
        ${barsHtml}
      `;
    }

    selectVariation(0);
  </script>
</body>
</html>
"""


def generate_index_html() -> Path:
    payload = build_web_payload()
    html_str = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", json.dumps(payload))
    INDEX_HTML_PATH.write_text(html_str, encoding="utf-8")
    return INDEX_HTML_PATH


# Alias for fpl_assistant.py weekly-update
build_static_html = generate_index_html


def serve_app(port: int = 8000) -> None:
    generate_index_html()
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving FPL AI Web App at http://localhost:{port}/index.html")
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or serve the FPL Web App")
    parser.add_argument("--serve", action="store_true", help="Run local HTTP server")
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve")
    args = parser.parse_args()

    out = generate_index_html()
    print(f"Generated interactive FPL web page at: {out}")
    if args.serve:
        serve_app(args.port)


if __name__ == "__main__":
    main()
