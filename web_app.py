#!/usr/bin/env python3
"""
FPL AI Optimizer — Minimal Interactive Web Page Generator & Server (web_app.py)

Generates a standalone, zero-dependency `index.html` (ready to open locally or host on GitHub Pages)
that includes:
1. Interactive Football Pitch visualization with authentic club jerseys, Captain (C) / Vice (V) badges,
   prices, clean sheet odds, 3-year PPG, and Expected Points (xP).
2. "Next Variation" & "Randomize High-Scoring Squad (70-80 xP)" buttons that cycle through
   6 distinct tactical £100m archetypes AND dynamically generate new valid £100m squads on the fly!
3. Tabs to switch between:
   - Best £100m Squad Variations (Unbiased across the entire Premier League)
   - Your Current 15-Man Squad Optimizer (with Maresca / O'Reilly cameo trap detection)
   - Unlimited Free Hit Dream XI
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
    return {
        "title": title,
        "subtitle": subtitle,
        "formation": res.formation,
        "totalCost": round(total_cost, 1),
        "budget": round(budget, 1),
        "expectedPoints": res.total_expected_points,
        "floor": res.total_floor,
        "ceiling": res.total_ceiling,
        "captain": res.captain.player.web_name,
        "viceCaptain": res.vice_captain.player.web_name,
        "starters": [
            _serialize_pred(p, p.player.id == cap_id, p.player.id == vice_id)
            for p in res.starting_xi
        ],
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

    # Match user's 15 players strictly by (name, team, position)
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
        subtitle="No budget cap — pure maximum Expected Points across the Premier League",
        total_cost=dream_cost,
        budget=999.0,
    )

    user_json = _serialize_squad_result(
        user_opt,
        title="Your Current 15-Man Squad (Optimized)",
        subtitle="Includes Maresca press conference cameo-trap protection (O'Reilly benched to 2nd Sub)",
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
  <title>FPL AI Squad Optimizer & Variation Explorer</title>
  <style>
    :root {
      --fpl-purple: #37003c;
      --fpl-dark: #1d0021;
      --fpl-green: #00ff87;
      --fpl-cyan: #04f5ff;
      --turf-1: #1e7e48;
      --turf-2: #248f52;
      --card-bg: rgba(18, 55, 36, 0.92);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: radial-gradient(circle at top, #37003c 0%, #16001a 100%);
      color: #ffffff;
      min-height: 100vh;
      padding: 20px;
    }
    .container {
      max-width: 1120px;
      margin: 0 auto;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      background: rgba(55, 0, 60, 0.85);
      border: 1px solid rgba(0, 255, 135, 0.35);
      border-bottom: 3px solid var(--fpl-green);
      border-radius: 14px;
      padding: 18px 24px;
      margin-bottom: 18px;
    }
    .brand h1 {
      font-size: 1.45rem;
      font-weight: 800;
      letter-spacing: -0.3px;
    }
    .brand p {
      color: var(--fpl-green);
      font-size: 0.9rem;
      margin-top: 4px;
      font-weight: 600;
    }
    .score-pill {
      background: var(--fpl-green);
      color: #1d0021;
      padding: 10px 20px;
      border-radius: 12px;
      text-align: center;
      min-width: 180px;
      box-shadow: 0 4px 15px rgba(0, 255, 135, 0.25);
    }
    .score-pill .lbl {
      font-size: 0.7rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }
    .score-pill .val {
      font-size: 1.7rem;
      font-weight: 900;
      line-height: 1.1;
    }
    .score-pill .range {
      font-size: 0.74rem;
      font-weight: 700;
      opacity: 0.85;
    }
    .controls-bar {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
    }
    .mode-tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .btn {
      border: 1px solid rgba(255,255,255,0.22);
      background: rgba(255,255,255,0.08);
      color: #fff;
      padding: 9px 15px;
      border-radius: 9px;
      font-size: 0.88rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .btn:hover {
      background: rgba(255,255,255,0.16);
      transform: translateY(-1px);
    }
    .btn.active {
      background: var(--fpl-green);
      color: #1d0021;
      border-color: var(--fpl-green);
    }
    .btn-variation {
      background: linear-gradient(135deg, #00ff87 0%, #04f5ff 100%);
      color: #16001a;
      border: none;
      padding: 10px 18px;
      font-size: 0.92rem;
      font-weight: 800;
      box-shadow: 0 4px 14px rgba(4, 245, 255, 0.28);
    }
    .btn-variation:hover {
      filter: brightness(1.06);
    }
    .variation-pills {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .var-pill {
      font-size: 0.78rem;
      padding: 6px 11px;
      border-radius: 999px;
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.18);
      color: #e5e0eb;
      cursor: pointer;
      font-weight: 600;
    }
    .var-pill.active {
      background: rgba(0, 255, 135, 0.2);
      border-color: var(--fpl-green);
      color: var(--fpl-green);
    }
    .meta-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      background: rgba(255,255,255,0.06);
      border-radius: 10px;
      padding: 10px 16px;
      margin-bottom: 16px;
      font-size: 0.86rem;
    }
    .meta-strip span strong {
      color: var(--fpl-green);
    }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 330px;
      gap: 20px;
      align-items: start;
    }
    @media (max-width: 920px) {
      .main-grid { grid-template-columns: 1fr; }
    }
    /* Football Pitch */
    .pitch {
      position: relative;
      border-radius: 14px;
      overflow: hidden;
      border: 3px solid rgba(255,255,255,0.85);
      background: repeating-linear-gradient(
        to bottom,
        var(--turf-1) 0px,
        var(--turf-1) 82px,
        var(--turf-2) 82px,
        var(--turf-2) 164px
      );
      padding: 20px 12px 24px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.45);
    }
    .pitch-box-top {
      position: absolute;
      top: 0;
      left: 50%;
      transform: translateX(-50%);
      width: 340px;
      height: 95px;
      border: 2px solid rgba(255,255,255,0.55);
      border-top: none;
      pointer-events: none;
    }
    .pitch-circle {
      position: absolute;
      bottom: 85px;
      left: 50%;
      transform: translateX(-50%);
      width: 130px;
      height: 130px;
      border: 2px solid rgba(255,255,255,0.45);
      border-radius: 50%;
      pointer-events: none;
    }
    .pitch-row {
      display: flex;
      justify-content: space-evenly;
      align-items: center;
      margin-bottom: 18px;
      position: relative;
      z-index: 2;
    }
    .pitch-row:last-child { margin-bottom: 0; }
    .player-card {
      width: 124px;
      background: var(--card-bg);
      border: 1.5px solid #52b788;
      border-radius: 10px;
      text-align: center;
      position: relative;
      padding-top: 8px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.35);
      transition: transform 0.15s ease;
      cursor: pointer;
    }
    .player-card:hover {
      transform: scale(1.05);
      z-index: 5;
      border-color: var(--fpl-green);
    }
    .price-tag {
      position: absolute;
      top: 5px;
      left: 5px;
      background: rgba(0,0,0,0.55);
      color: #a3e4d7;
      font-size: 0.66rem;
      font-weight: 700;
      padding: 2px 5px;
      border-radius: 4px;
    }
    .xp-tag {
      position: absolute;
      top: 5px;
      right: 5px;
      background: #e9f5ec;
      color: #111;
      font-size: 0.7rem;
      font-weight: 800;
      padding: 2px 5px;
      border-radius: 4px;
    }
    .xp-tag.high {
      background: var(--fpl-green);
    }
    .cap-badge {
      position: absolute;
      top: 28px;
      left: 6px;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: var(--fpl-purple);
      color: var(--fpl-green);
      border: 1.5px solid var(--fpl-green);
      font-size: 0.68rem;
      font-weight: 900;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .star-badge {
      position: absolute;
      top: 51px;
      left: 6px;
      color: #ffd166;
      font-size: 0.85rem;
      text-shadow: 0 1px 3px rgba(0,0,0,0.8);
    }
    .jersey-svg {
      width: 56px;
      height: 50px;
      margin: 12px auto 4px;
      display: block;
    }
    .p-name {
      background: #ffffff;
      color: #240028;
      font-weight: 800;
      font-size: 0.78rem;
      padding: 3px 4px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .p-fix {
      background: #e8e4f0;
      color: #37003c;
      font-weight: 700;
      font-size: 0.7rem;
      padding: 3px 4px;
      border-radius: 0 0 8px 8px;
    }
    /* Bench */
    .bench-box {
      margin-top: 14px;
      background: rgba(31, 78, 56, 0.9);
      border: 1.5px solid #52b788;
      border-radius: 12px;
      padding: 14px 10px 10px;
    }
    .bench-title {
      text-align: center;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #b7e4c7;
      margin-bottom: 10px;
      font-weight: 800;
    }
    .sub-lbl {
      font-size: 0.68rem;
      font-weight: 800;
      color: #d8f3dc;
      margin-bottom: 4px;
      text-align: center;
    }
    /* Sidebar */
    .side-panel {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .card-panel {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 12px;
      padding: 16px;
    }
    .card-panel h3 {
      font-size: 0.95rem;
      color: var(--fpl-green);
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .info-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 0.83rem;
      line-height: 1.4;
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
      padding: 7px 0;
      border-bottom: 1px solid rgba(255,255,255,0.09);
      font-size: 0.82rem;
    }
    .transfer-row:last-child { border-bottom: none; }
    .gain-badge {
      background: rgba(0, 255, 135, 0.18);
      color: var(--fpl-green);
      padding: 2px 7px;
      border-radius: 6px;
      font-weight: 800;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <h1 id="squad-title">FPL AI Squad Optimizer</h1>
        <p id="squad-subtitle">3-Year Historical PPG + Live xG/xA + Press Conference Risk Engine</p>
      </div>
      <div class="score-pill">
        <div class="lbl">Projected Score</div>
        <div class="val" id="proj-score">76.1 pts</div>
        <div class="range" id="proj-range">Floor: 25 • Ceiling: 147</div>
      </div>
    </header>

    <div class="controls-bar">
      <div class="mode-tabs">
        <button class="btn active" id="tab-best" onclick="switchMode('best')">🏆 Best £100m Squads (6 Variations)</button>
        <button class="btn" id="tab-mine" onclick="switchMode('mine')">👤 My Current Squad (4-4-2)</button>
        <button class="btn" id="tab-dream" onclick="switchMode('dream')">⚡ Unlimited Free Hit XI</button>
      </div>
      <div style="display:flex; gap:8px;">
        <button class="btn btn-variation" id="btn-next-var" onclick="nextVariation()">🔄 Next Tactical Variation</button>
        <button class="btn" id="btn-random-var" onclick="generateRandomHighScoringSquad()">🎲 Shuffle New 70–80p Squad</button>
      </div>
    </div>

    <div class="variation-pills" id="var-pills-bar"></div>

    <div class="meta-strip">
      <span>Formation: <strong id="meta-form">3-5-2</strong></span>
      <span>Captain (2x): <strong id="meta-cap">Saka</strong></span>
      <span>Vice-Captain: <strong id="meta-vice">Palmer</strong></span>
      <span>15-Man Cost: <strong id="meta-cost">£99.4m / £100.0m</strong></span>
    </div>

    <div class="main-grid">
      <div>
        <div class="pitch">
          <div class="pitch-box-top"></div>
          <div class="pitch-circle"></div>
          <div class="pitch-row" id="row-gkp"></div>
          <div class="pitch-row" id="row-def"></div>
          <div class="pitch-row" id="row-mid"></div>
          <div class="pitch-row" id="row-fwd"></div>
        </div>

        <div class="bench-box">
          <div class="bench-title">Substitutes (Auto-Sub Ordered)</div>
          <div class="pitch-row" id="row-bench"></div>
        </div>
      </div>

      <div class="side-panel">
        <div class="card-panel">
          <h3>Player Inspector (Click Any Card)</h3>
          <div id="player-inspector" style="font-size:0.85rem; line-height:1.5; color:#ddd;">
            Click any player on the pitch or bench to inspect their 3-year PPG, clean-sheet probability, 2+ goals conceded risk, and manager press conference news.
          </div>
        </div>

        <div class="card-panel">
          <h3>🧠 22-Feature Neural Net (22→36→18→1)</h3>
          <div id="nn-metrics-box" style="font-size:0.81rem; line-height:1.45; color:#e5e0eb;"></div>
        </div>

        <div class="card-panel">
          <h3>Tactical & News Insights</h3>
          <ul class="info-list" id="notes-list"></ul>
        </div>

        <div class="card-panel">
          <h3>Top 1-Transfer Upgrades (For Your Squad)</h3>
          <div id="transfers-list"></div>
        </div>
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
      WHU: { body: "#7A263A", sleeve: "#1BB1E7", stripe: null },
      NFO: { body: "#DD0000", sleeve: "#FFFFFF", stripe: null },
      COV: { body: "#59CBE8", sleeve: "#1D2951", stripe: null },
      CRY: { body: "#1B458F", sleeve: "#C4122E", stripe: "#C4122E" },
      IPS: { body: "#3A64A3", sleeve: "#FFFFFF", stripe: null }
    };

    let currentMode = "best";
    let currentVarIdx = 0;

    function jerseySvg(team, isGkp) {
      let kit = KIT_COLORS[team] || { body: "#37003c", sleeve: "#ffffff", stripe: null };
      let body = isGkp ? (team === "BHA" ? "#ff5992" : "#2ecc71") : kit.body;
      let sleeve = isGkp ? body : kit.sleeve;
      let stripes = (!isGkp && kit.stripe)
        ? `<rect x="21" y="10" width="4" height="36" fill="${kit.stripe}"/>
           <rect x="28" y="10" width="4" height="36" fill="${kit.stripe}"/>
           <rect x="35" y="10" width="4" height="36" fill="${kit.stripe}"/>`
        : "";
      return `<svg class="jersey-svg" viewBox="0 0 60 52">
        <polygon points="16,8 4,20 10,28 18,18" fill="${sleeve}" stroke="#111" stroke-width="1.5"/>
        <polygon points="44,8 56,20 50,28 42,18" fill="${sleeve}" stroke="#111" stroke-width="1.5"/>
        <rect x="16" y="8" width="28" height="38" rx="4" fill="${body}" stroke="#111" stroke-width="1.5"/>
        ${stripes}
        <path d="M24,8 Q30,15 36,8" fill="#1d0021" stroke="#111" stroke-width="1.2"/>
      </svg>`;
    }

    function createCardHtml(p, subLabel = null) {
      const capHtml = p.isCaptain
        ? `<div class="cap-badge">C</div>`
        : (p.isVice ? `<div class="cap-badge" style="color:#fff;border-color:#fff;">V</div>` : "");
      const starHtml = p.inDreamteam ? `<div class="star-badge">★</div>` : "";
      const xpClass = p.displayXp >= 6.0 ? "xp-tag high" : "xp-tag";
      const wrapStart = subLabel ? `<div><div class="sub-lbl">${subLabel}</div>` : "";
      const wrapEnd = subLabel ? `</div>` : "";
      return `${wrapStart}
        <div class="player-card" onclick='inspectPlayer(${JSON.stringify(p).replace(/'/g, "&#39;")})'>
          <div class="price-tag">£${p.price.toFixed(1)}m</div>
          <div class="${xpClass}">${p.displayXp.toFixed(1)}p</div>
          ${capHtml}
          ${starHtml}
          ${jerseySvg(p.team, p.pos === "GKP")}
          <div class="p-name">${p.name}</div>
          <div class="p-fix">${p.team} • ${p.opponent}</div>
        </div>
      ${wrapEnd}`;
    }

    function inspectPlayer(p) {
      const el = document.getElementById("player-inspector");
      el.innerHTML = `
        <div style="font-size:1.05rem;font-weight:800;color:#fff;margin-bottom:4px;">
          ${p.fullName} (${p.teamName}) — £${p.price.toFixed(1)}m
        </div>
        <div style="color:var(--fpl-green);font-weight:700;margin-bottom:8px;">
          GW Fixture: ${p.opponent} | Projected: ${p.displayXp.toFixed(1)} xP (Range: ${p.floor}–${p.ceiling} pts)
        </div>
        <div>• <strong>🚀 Explosive Haul Ceiling (9+ pts):</strong> ${p.haulProb}% | <strong>📅 3-GW Horizon xP:</strong> ${p.horizon3GwXp.toFixed(1)}p (${p.fixtureRun3Gw})</div>
        <div>• <strong>🧠 Dual-Head NN 90m:</strong> ${p.nnXp.toFixed(1)}p | <strong>📐 Rule Diagnostic:</strong> ${p.ruleXp.toFixed(1)}p</div>
        <div>• <strong>2026/27 PPG:</strong> ${p.seasonPpg.toFixed(1)} (Form: ${p.form} | Value: ${p.valueSeason.toFixed(1)} pts/£m)</div>
        <div>• <strong>CBIT Def. Contrib/90:</strong> ${p.defContrib90.toFixed(1)} | <strong>Clean Sheet Odds:</strong> ${p.csProb}% (2+GC Risk: ${p.gc2Prob}%)</div>
        <div>• <strong>Start Probability:</strong> ${p.startProb}% | <strong>1-Pt Cameo Risk:</strong> ${p.cameoProb}%</div>
        <div style="margin-top:6px;padding:6px 8px;background:rgba(0,255,135,0.12);border-left:3px solid var(--fpl-green);border-radius:4px;">
          <strong>🤖 AI Head Coach:</strong> ${p.coachNote}
        </div>
        <div style="margin-top:6px;padding:6px 8px;background:rgba(0,0,0,0.3);border-radius:6px;">
          <strong>News Status (${p.riskLevel}):</strong> ${p.newsReasoning}
        </div>
      `;
    }

    function renderSquadView(squad) {
      document.getElementById("squad-title").textContent = squad.title;
      document.getElementById("squad-subtitle").textContent = squad.subtitle;
      document.getElementById("proj-score").textContent = `${squad.expectedPoints.toFixed(1)} pts`;
      document.getElementById("proj-range").textContent = `Floor: ${squad.floor} • Ceiling: ${squad.ceiling}`;
      document.getElementById("meta-form").textContent = squad.formation;
      document.getElementById("meta-cap").textContent = squad.captain;
      document.getElementById("meta-vice").textContent = squad.viceCaptain;
      document.getElementById("meta-cost").textContent =
        squad.budget > 200 ? `£${squad.totalCost.toFixed(1)}m (Unlimited)` : `£${squad.totalCost.toFixed(1)}m / £${squad.budget.toFixed(1)}m`;

      const gkps = squad.starters.filter(p => p.pos === "GKP");
      const defs = squad.starters.filter(p => p.pos === "DEF");
      const mids = squad.starters.filter(p => p.pos === "MID");
      const fwds = squad.starters.filter(p => p.pos === "FWD");

      document.getElementById("row-gkp").innerHTML = gkps.map(p => createCardHtml(p)).join("");
      document.getElementById("row-def").innerHTML = defs.map(p => createCardHtml(p)).join("");
      document.getElementById("row-mid").innerHTML = mids.map(p => createCardHtml(p)).join("");
      document.getElementById("row-fwd").innerHTML = fwds.map(p => createCardHtml(p)).join("");

      const benchHtml = squad.bench.map((bp, i) => {
        const lbl = i === 0 ? "GKP" : `${i}. ${bp.pos}`;
        return createCardHtml(bp, lbl);
      }).join("");
      document.getElementById("row-bench").innerHTML = benchHtml;

      const notesHtml = (squad.notes || []).map(n => `<li>${n}</li>`).join("");
      document.getElementById("notes-list").innerHTML = notesHtml || "<li>All 11 starters have high projected minutes and zero cameo flags.</li>";

      if (squad.starters.length > 0) {
        inspectPlayer(squad.starters.find(p => p.isCaptain) || squad.starters[0]);
      }
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
          #${i + 1}: ${v.formation} (${v.expectedPoints.toFixed(1)}p • £${v.totalCost.toFixed(1)}m)
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
      // Dynamically builds a new valid £100m 15-player squad from DATA.playerPool
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

        // Sort starters for Captain & Vice
        const ranked = [...starters].sort((a,b) => b.xp - a.xp);
        const cap = ranked[0];
        const vice = ranked[1];
        starters.forEach(p => {
          if (p.id === cap.id) { p.isCaptain = true; p.displayXp = +(p.xp * 2).toFixed(1); }
          else if (p.id === vice.id) { p.isVice = true; }
        });

        const totXp = +(starters.reduce((s, p) => s + p.displayXp, 0)).toFixed(1);
        if (totXp < 70.5) continue;

        const customSquad = {
          title: `Custom Generated Squad (${nD}-${nM}-${nF})`,
          subtitle: `Dynamically sampled £100m arrangement projecting ${totXp} pts (zero bias)`,
          formation: `${nD}-${nM}-${nF}`,
          totalCost: +cost.toFixed(1),
          budget: 100.0,
          expectedPoints: totXp,
          floor: starters.reduce((s, p) => s + p.floor, 0) + cap.floor,
          ceiling: starters.reduce((s, p) => s + p.ceiling, 0) + cap.ceiling,
          captain: cap.name,
          viceCaptain: vice.name,
          starters,
          bench: [...gBench, ...[...dBench, ...mBench, ...fBench].sort((a,b) => b.xp - a.xp)],
          notes: [
            `Custom ${nD}-${nM}-${nF} build (£${cost.toFixed(1)}m / £100.0m) with max 3 players per club.`,
            `Captain ${cap.name} (${cap.displayXp}p) + Vice ${vice.name} (${vice.xp}p).`
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

    // Populate transfers panel
    document.getElementById("transfers-list").innerHTML = DATA.transfers.map(t => `
      <div class="transfer-row">
        <div>
          <strong>${t.out}</strong> → <strong style="color:var(--fpl-green)">${t.in}</strong>
          <div style="font-size:0.74rem;color:#bbb;">${t.inTeam} vs ${t.inFix} (${t.costDiff >= 0 ? '+' : ''}${t.costDiff.toFixed(1)}m)</div>
        </div>
        <span class="gain-badge">+${t.xpGain.toFixed(1)} xP</span>
      </div>
    `).join("");

    // Populate Neural Network Metrics panel
    if (DATA.nnMetrics) {
      const m = DATA.nnMetrics;
      const topFeats = Object.entries(m.feature_importance || {}).slice(0, 6);
      const barsHtml = topFeats.map(([k, v]) => `
        <div style="margin-top:4px;">
          <div style="display:flex;justify-content:space-between;font-size:0.75rem;">
            <span>${k}</span><strong style="color:var(--fpl-green)">${v}%</strong>
          </div>
          <div style="background:rgba(255,255,255,0.1);height:5px;border-radius:3px;overflow:hidden;">
            <div style="background:var(--fpl-green);width:${Math.min(100, v * 6)}%;height:100%;"></div>
          </div>
        </div>
      `).join("");
      document.getElementById("nn-metrics-box").innerHTML = `
        <div style="margin-bottom:6px;">
          Trained on <strong>${m.train_samples + m.val_samples}</strong> multi-season GW samples (${m.epochs} epochs, Adam optimizer).
        </div>
        <div style="color:var(--fpl-cyan);font-weight:700;margin-bottom:6px;">
          Val MAE: ${m.val_mae} pts • Val RMSE: ${m.val_rmse} pts
        </div>
        <div style="font-size:0.76rem;color:#bbb;margin-bottom:4px;">Top Learned Feature Weights:</div>
        ${barsHtml}
      `;
    }

    // Initial render
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
