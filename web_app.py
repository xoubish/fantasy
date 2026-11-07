#!/usr/bin/env python3
"""
FPL AI Optimizer — Minimal, Sleek Web App Generator (web_app.py)

Focused strictly on the Best £100m Squad of the Week before games start,
with 1-click Variations and a clean Player Inspector when you click a card.
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
    get_best_squad_variations,
    predict_player,
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
        "seasonPpg": p.player.points_per_game,
        "form": p.player.form,
        "xp": round(p.expected_points, 1),
        "displayXp": round(p.expected_points * 2 if is_cap else p.expected_points, 1),
        "haulProb": int(round(p.haul_prob * 100)),
        "fixtureRun3Gw": p.player.fixture_run_3gw,
        "roleTag": p.role_tag,
        "coachNote": p.coach_note,
        "floor": p.floor_points * 2 if is_cap else p.floor_points,
        "ceiling": p.ceiling_points * 2 if is_cap else p.ceiling_points,
        "isCaptain": is_cap,
        "isVice": is_vice,
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
    cap = res.captain
    vice = res.vice_captain
    coach_verdict = (
        f"AI Head Coach selects {res.formation} (£{total_cost:.1f}m): "
        f"Captains {cap.player.web_name} ({cap.expected_points * 2:.1f}p, {int(round(cap.haul_prob * 100))}% haul odds vs {cap.player.next_opponent}) "
        f"backed by {vice.player.web_name} (V, {vice.expected_points:.1f}p vs {vice.player.next_opponent}), "
        f"balancing 3-GW fixture swings and CBIT defensive floor."
    )
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
        "coachVerdict": coach_verdict,
        "starters": [
            _serialize_pred(p, p.player.id == cap_id, p.player.id == vice_id)
            for p in res.starting_xi
        ],
        "bench": [
            _serialize_pred(res.bench_gkp),
            *[_serialize_pred(bp) for bp in res.bench_outfield],
        ],
    }


def build_web_payload() -> Dict[str, Any]:
    all_players = load_players()
    articles = load_all_news(SQUAD_CONFIG_PATH)
    all_preds = [predict_player(p, articles) for p in all_players]

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
        "playerPool": healthy_pool_json,
        "nnMetrics": nn_net.metrics,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FPL AI — Best Squad of the Week</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@600;700&family=Space+Grotesk:wght@700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --fpl-deep: #0d0214;
      --fpl-green: #00ff87;
      --fpl-cyan: #04f5ff;
      --fpl-pink: #ff2882;
      --turf-dark: #12613a;
      --turf-light: #177547;
      --glass-bg: rgba(26, 9, 37, 0.72);
      --glass-border: rgba(255, 255, 255, 0.11);
      --text-muted: #b0a2bf;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--fpl-deep);
      background-image:
        radial-gradient(at 15% 10%, rgba(0, 255, 135, 0.11) 0px, transparent 45%),
        radial-gradient(at 85% 10%, rgba(4, 245, 255, 0.10) 0px, transparent 45%),
        radial-gradient(at 50% 40%, rgba(55, 0, 60, 0.75) 0px, #0d0214 80%);
      background-attachment: fixed;
      color: #ffffff;
      min-height: 100vh;
      padding: 28px 18px 48px;
    }
    .container {
      max-width: 1060px;
      margin: 0 auto;
    }

    /* Minimal Glass Header */
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      border-radius: 20px;
      padding: 20px 26px;
      margin-bottom: 16px;
      backdrop-filter: blur(18px);
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45);
    }
    .brand h1 {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.55rem;
      font-weight: 800;
      letter-spacing: -0.4px;
      color: #ffffff;
    }
    .brand-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 6px;
      font-size: 0.84rem;
      color: var(--text-muted);
    }
    .brand-meta span strong {
      color: var(--fpl-green);
      font-weight: 700;
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .score-badge {
      background: rgba(0, 255, 135, 0.14);
      border: 1px solid rgba(0, 255, 135, 0.4);
      padding: 8px 16px;
      border-radius: 12px;
      text-align: right;
    }
    .score-badge .lbl {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.9px;
      color: var(--fpl-green);
      font-weight: 700;
    }
    .score-badge .val {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.45rem;
      font-weight: 800;
      color: #fff;
      line-height: 1.1;
    }

    .btn {
      border: 1px solid rgba(255, 255, 255, 0.16);
      background: rgba(255, 255, 255, 0.07);
      color: #fff;
      padding: 10px 16px;
      border-radius: 11px;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.18s ease;
    }
    .btn:hover {
      background: rgba(255, 255, 255, 0.13);
      transform: translateY(-1px);
    }
    .btn-primary {
      background: linear-gradient(135deg, #00ff87 0%, #04f5ff 100%);
      color: #120016;
      border: none;
      font-weight: 800;
      box-shadow: 0 6px 18px rgba(0, 255, 135, 0.25);
    }
    .btn-primary:hover {
      filter: brightness(1.06);
      color: #120016;
    }

    /* Minimal Variation Dots/Pills */
    .variation-bar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .var-pill {
      font-size: 0.76rem;
      padding: 6px 13px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text-muted);
      cursor: pointer;
      font-weight: 600;
      transition: all 0.15s ease;
    }
    .var-pill:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.1);
    }
    .var-pill.active {
      background: rgba(0, 255, 135, 0.18);
      border-color: var(--fpl-green);
      color: var(--fpl-green);
    }

    /* Main 2-Column Layout */
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 310px;
      gap: 20px;
      align-items: start;
    }
    @media (max-width: 880px) {
      .main-grid { grid-template-columns: 1fr; }
    }

    /* Stadium Pitch */
    .pitch-wrapper {
      background: rgba(255, 255, 255, 0.03);
      padding: 10px;
      border-radius: 22px;
      border: 1px solid var(--glass-border);
      box-shadow: 0 22px 55px rgba(0, 0, 0, 0.55);
    }
    .pitch {
      position: relative;
      border-radius: 16px;
      overflow: hidden;
      border: 2px solid rgba(255, 255, 255, 0.7);
      background:
        radial-gradient(circle at 50% 10%, rgba(255,255,255,0.08) 0%, transparent 55%),
        repeating-linear-gradient(
          to bottom,
          var(--turf-dark) 0px,
          var(--turf-dark) 78px,
          var(--turf-light) 78px,
          var(--turf-light) 156px
        );
      padding: 24px 10px 26px;
    }
    .pitch-box-top {
      position: absolute;
      top: 0;
      left: 50%;
      transform: translateX(-50%);
      width: 320px;
      height: 92px;
      border: 2px solid rgba(255, 255, 255, 0.35);
      border-top: none;
      pointer-events: none;
    }
    .pitch-circle {
      position: absolute;
      bottom: 50px;
      left: 50%;
      transform: translateX(-50%);
      width: 120px;
      height: 120px;
      border: 2px solid rgba(255, 255, 255, 0.3);
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

    /* Clean Player Card */
    .player-card {
      width: 122px;
      background: rgba(11, 35, 24, 0.94);
      border: 1.5px solid rgba(116, 198, 157, 0.45);
      border-radius: 12px;
      text-align: center;
      position: relative;
      padding-top: 9px;
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.4);
      transition: all 0.16s ease;
      cursor: pointer;
      user-select: none;
    }
    .player-card:hover {
      transform: translateY(-3px);
      border-color: var(--fpl-green);
    }
    .player-card.selected {
      border: 2px solid var(--fpl-cyan);
      box-shadow: 0 0 0 3px rgba(4, 245, 255, 0.22), 0 8px 20px rgba(0, 0, 0, 0.5);
    }
    .price-tag {
      position: absolute;
      top: 5px;
      left: 5px;
      color: #a3e4d7;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.63rem;
      font-weight: 700;
      padding: 1px 4px;
    }
    .xp-tag {
      position: absolute;
      top: 5px;
      right: 5px;
      background: rgba(255, 255, 255, 0.9);
      color: #0b1c12;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.66rem;
      font-weight: 800;
      padding: 1px 5px;
      border-radius: 4px;
    }
    .xp-tag.high {
      background: var(--fpl-green);
      color: #0d0012;
    }
    .cap-badge {
      position: absolute;
      top: 26px;
      left: 6px;
      width: 19px;
      height: 19px;
      border-radius: 50%;
      background: var(--fpl-pink);
      color: #fff;
      font-size: 0.62rem;
      font-weight: 900;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .cap-badge.vice {
      background: rgba(255, 255, 255, 0.18);
      color: #fff;
    }
    .jersey-svg {
      width: 54px;
      height: 48px;
      margin: 10px auto 5px;
      display: block;
      filter: drop-shadow(0 3px 4px rgba(0,0,0,0.4));
    }
    .p-name {
      background: #ffffff;
      color: #1c0021;
      font-weight: 800;
      font-size: 0.76rem;
      padding: 3.5px 4px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .p-fix {
      background: #e8e4f0;
      color: #2b0530;
      font-weight: 700;
      font-size: 0.67rem;
      padding: 3px 4px;
      border-radius: 0 0 10px 10px;
    }

    /* Bench */
    .bench-box {
      margin-top: 12px;
      background: rgba(16, 48, 34, 0.85);
      border: 1px solid rgba(82, 183, 136, 0.35);
      border-radius: 14px;
      padding: 12px 10px 10px;
    }
    .bench-title {
      text-align: center;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: #a3e4d7;
      margin-bottom: 8px;
      font-weight: 700;
    }
    .sub-lbl {
      font-size: 0.63rem;
      font-weight: 700;
      color: #b7e4c7;
      margin-bottom: 4px;
      text-align: center;
    }

    /* Minimal Sidebar */
    .side-panel {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .card-panel {
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      border-radius: 16px;
      padding: 18px;
      backdrop-filter: blur(16px);
    }
    .card-panel h3 {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.9px;
    }
    .stat-row {
      display: flex;
      justify-content: space-between;
      padding: 7px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.07);
      font-size: 0.82rem;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-row span { color: var(--text-muted); }
    .stat-row strong { color: #fff; font-family: 'JetBrains Mono', monospace; }
    .coach-note {
      margin-top: 12px;
      padding: 10px 12px;
      background: rgba(0, 255, 135, 0.09);
      border-left: 3px solid var(--fpl-green);
      border-radius: 6px;
      font-size: 0.79rem;
      line-height: 1.45;
      color: #ece6f2;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <h1 id="squad-title">Best £100m Squad of the Week</h1>
        <div class="brand-meta">
          <span>Formation: <strong id="meta-form">3-4-3</strong></span>
          <span>Captain (2x): <strong id="meta-cap">Gabriel</strong></span>
          <span>Vice: <strong id="meta-vice">Haaland</strong></span>
          <span>Cost: <strong id="meta-cost">£99.8m / £100.0m</strong></span>
        </div>
      </div>

      <div class="header-right">
        <div class="score-badge">
          <div class="lbl">Projected Score</div>
          <div class="val" id="proj-score">74.2 pts</div>
        </div>
        <button class="btn btn-primary" onclick="nextVariation()">🔄 Next Variation</button>
        <button class="btn" onclick="generateRandomHighScoringSquad()">🎲 Shuffle</button>
      </div>
    </header>

    <div class="variation-bar" id="var-pills-bar"></div>

    <div class="main-grid">
      <div class="pitch-wrapper">
        <div class="pitch">
          <div class="pitch-box-top"></div>
          <div class="pitch-circle"></div>
          <div class="pitch-row" id="row-gkp"></div>
          <div class="pitch-row" id="row-def"></div>
          <div class="pitch-row" id="row-mid"></div>
          <div class="pitch-row" id="row-fwd"></div>
        </div>

        <div class="bench-box">
          <div class="bench-title">Substitutes</div>
          <div class="pitch-row" id="row-bench"></div>
        </div>
      </div>

      <div class="side-panel">
        <div class="card-panel">
          <h3>Player Details</h3>
          <div id="player-inspector"></div>
        </div>

        <div class="card-panel" style="font-size:0.79rem; color:var(--text-muted); line-height:1.5;">
          <h3>AI Coach & Model Summary</h3>
          <div id="coach-verdict-box" style="margin-bottom:10px;padding:9px 11px;background:rgba(4,245,255,0.08);border-left:3px solid var(--fpl-cyan);border-radius:6px;color:#ece6f2;"></div>
          <div id="nn-summary"></div>
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
      NFO: { body: "#DD0000", sleeve: "#FFFFFF", stripe: null },
      COV: { body: "#59CBE8", sleeve: "#1D2951", stripe: null },
      CRY: { body: "#1B458F", sleeve: "#C4122E", stripe: "#C4122E" },
      IPS: { body: "#3A64A3", sleeve: "#FFFFFF", stripe: null }
    };

    let currentVarIdx = 0;
    let selectedPlayerId = null;

    function jerseySvg(team, isGkp) {
      let kit = KIT_COLORS[team] || { body: "#37003c", sleeve: "#ffffff", stripe: null };
      let body = isGkp ? (team === "BHA" ? "#ff5992" : "#10b981") : kit.body;
      let sleeve = isGkp ? body : kit.sleeve;
      let stripes = (!isGkp && kit.stripe)
        ? `<rect x="21" y="10" width="3.8" height="36" fill="${kit.stripe}" opacity="0.92"/>
           <rect x="28.1" y="10" width="3.8" height="36" fill="${kit.stripe}" opacity="0.92"/>
           <rect x="35.2" y="10" width="3.8" height="36" fill="${kit.stripe}" opacity="0.92"/>`
        : "";
      return `<svg class="jersey-svg" viewBox="0 0 60 52">
        <polygon points="16,8 4,20 10,28 18,18" fill="${sleeve}" stroke="#0b0b0b" stroke-width="1.5"/>
        <polygon points="44,8 56,20 50,28 42,18" fill="${sleeve}" stroke="#0b0b0b" stroke-width="1.5"/>
        <rect x="16" y="8" width="28" height="38" rx="4" fill="${body}" stroke="#0b0b0b" stroke-width="1.5"/>
        ${stripes}
        <path d="M23,8 Q30,15 37,8" fill="#16001a" stroke="#0b0b0b" stroke-width="1.3"/>
      </svg>`;
    }

    function createCardHtml(p, subLabel = null) {
      const capHtml = p.isCaptain
        ? `<div class="cap-badge">C</div>`
        : (p.isVice ? `<div class="cap-badge vice">V</div>` : "");
      const xpClass = p.displayXp >= 6.0 ? "xp-tag high" : "xp-tag";
      const selClass = p.id === selectedPlayerId ? "player-card selected" : "player-card";
      const wrapStart = subLabel ? `<div><div class="sub-lbl">${subLabel}</div>` : "";
      const wrapEnd = subLabel ? `</div>` : "";
      return `${wrapStart}
        <div class="${selClass}" id="pcard-${p.id}" onclick='inspectPlayer(${JSON.stringify(p).replace(/'/g, "&#39;")})'>
          <div class="price-tag">£${p.price.toFixed(1)}m</div>
          <div class="${xpClass}">${p.displayXp.toFixed(1)}p</div>
          ${capHtml}
          ${jerseySvg(p.team, p.pos === "GKP")}
          <div class="p-name">${p.name}</div>
          <div class="p-fix">${p.team} • ${p.opponent}</div>
        </div>
      ${wrapEnd}`;
    }

    function inspectPlayer(p) {
      selectedPlayerId = p.id;
      document.querySelectorAll(".player-card").forEach(el => {
        el.classList.toggle("selected", el.id === `pcard-${p.id}`);
      });

      document.getElementById("player-inspector").innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
          <div style="width:50px;flex-shrink:0;">${jerseySvg(p.team, p.pos === "GKP")}</div>
          <div>
            <div style="font-size:1.05rem;font-weight:800;color:#fff;">${p.fullName}</div>
            <div style="font-size:0.78rem;color:var(--text-muted);">${p.teamName} • ${p.pos} • £${p.price.toFixed(1)}m</div>
          </div>
        </div>
        <div class="stat-row"><span>Expected Points</span><strong style="color:var(--fpl-green)">${p.displayXp.toFixed(1)} pts</strong></div>
        <div class="stat-row"><span>This Week</span><strong>${p.opponent}</strong></div>
        <div class="stat-row"><span>Next 3 Games</span><strong style="font-size:0.75rem;">${p.fixtureRun3Gw || p.opponent}</strong></div>
        <div class="stat-row"><span>9+ Pt Haul Odds</span><strong style="color:var(--fpl-cyan)">${p.haulProb}%</strong></div>
        <div class="stat-row"><span>Clean Sheet Odds</span><strong>${p.csProb}%</strong></div>
        <div class="stat-row"><span>Season PPG / Form</span><strong>${p.seasonPpg.toFixed(1)} / ${p.form}</strong></div>
        <div class="coach-note">${p.coachNote}</div>
      `;
    }

    function renderSquadView(squad) {
      document.getElementById("proj-score").textContent = `${squad.expectedPoints.toFixed(1)} pts`;
      document.getElementById("meta-form").textContent = squad.formation;
      document.getElementById("meta-cap").textContent = squad.captain;
      document.getElementById("meta-vice").textContent = squad.viceCaptain;
      document.getElementById("meta-cost").textContent = `£${squad.totalCost.toFixed(1)}m / £${squad.budget.toFixed(1)}m`;
      document.getElementById("coach-verdict-box").innerHTML = `<strong>🤖 AI Decision:</strong> ${squad.coachVerdict || ('Selected ' + squad.formation + ' led by Captain ' + squad.captain + ' & Vice ' + squad.viceCaptain + '.')}`;

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

      document.getElementById("row-bench").innerHTML = squad.bench.map((bp, i) => {
        const lbl = i === 0 ? "GKP" : `${i}. ${bp.pos}`;
        return createCardHtml(bp, lbl);
      }).join("");

      if (capPlayer) inspectPlayer(capPlayer);
    }

    function renderVariationPills() {
      document.getElementById("var-pills-bar").innerHTML = DATA.variations.map((v, i) => `
        <span class="var-pill ${i === currentVarIdx ? 'active' : ''}" onclick="selectVariation(${i})">
          Variation ${i + 1} (${v.formation} • ${v.expectedPoints.toFixed(1)}p)
        </span>
      `).join("");
    }

    function selectVariation(idx) {
      currentVarIdx = idx;
      renderVariationPills();
      renderSquadView(DATA.variations[currentVarIdx]);
    }

    function nextVariation() {
      currentVarIdx = (currentVarIdx + 1) % DATA.variations.length;
      renderVariationPills();
      renderSquadView(DATA.variations[currentVarIdx]);
    }

    function generateRandomHighScoringSquad() {
      const formations = [[3,4,3], [3,5,2], [4,4,2], [4,3,3]];
      const [nD, nM, nF] = formations[Math.floor(Math.random() * formations.length)];

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

        renderSquadView({
          formation: `${nD}-${nM}-${nF}`,
          totalCost: +cost.toFixed(1),
          budget: 100.0,
          expectedPoints: totXp,
          captain: cap.name,
          viceCaptain: vice.name,
          starters,
          bench: [...gBench, ...[...dBench, ...mBench, ...fBench].sort((a,b) => b.xp - a.xp)]
        });
        return;
      }
    }

    if (DATA.nnMetrics) {
      const m = DATA.nnMetrics;
      const top3 = Object.entries(m.feature_importance || {})
        .slice(0, 3)
        .map(([k, v]) => `${k} (${v}%)`)
        .join(", ");
      document.getElementById("nn-summary").innerHTML = `
        <div>• <strong>AI Head Coach:</strong> Adjusts xP & captaincy using press conferences, club creative supply, set-piece roles & 3-GW fixture traps</div>
        <div>• <strong>Neural Network:</strong> ${m.architecture || '22→40→20→2 Dual-Head (xP + Haul Ceiling)'}</div>
        <div>• <strong>Training Data:</strong> ${m.real_gw_samples || (m.train_samples + m.val_samples)} real 2026/27 PL match logs (through GW ${m.latest_trained_gw || 9}, ${m.val_mae}p MAE)</div>
        <div>• <strong>Top Learned Signals:</strong> ${top3}</div>
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
