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
from typing import Any, Dict, List, Optional

from fpl_data import load_players
from neural_model import get_or_train_neural_net
from news_scraper import load_all_news
from predictor import (
    OptimizedSquadResult,
    PlayerPrediction,
    build_transfer_plan,
    get_best_squad_variations,
    is_selectable,
    predict_player,
)


ROOT_DIR = Path(__file__).parent
SQUAD_CONFIG_PATH = ROOT_DIR / "my_squad.json"
INDEX_HTML_PATH = ROOT_DIR / "index.html"
BACKTEST_RESULTS_PATH = ROOT_DIR / "data" / "backtest_results.json"


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
    xi_xp = sum(p.expected_points for p in res.starting_xi)
    coach_verdict = (
        f"Captain {cap.player.web_name}: {cap.expected_points:.1f} xP doubled, {int(round(cap.haul_prob * 100))}% haul odds vs {cap.player.next_opponent}; "
        f"vice {vice.player.web_name} ({vice.expected_points:.1f} xP vs {vice.player.next_opponent}). "
        f"The eleven add up to {xi_xp:.1f} xP before the armband. "
        f"Every player's xP is minutes-weighted, so rotation risks are priced in rather than filtered out."
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
        [p for p in all_preds if is_selectable(p, 0.6)],
        key=lambda x: x.expected_points,
        reverse=True,
    )[:65]
    healthy_pool_json = [_serialize_pred(p) for p in healthy_active]

    nn_net = get_or_train_neural_net(force_retrain=False)

    return {
        "variations": variations_json,
        "playerPool": healthy_pool_json,
        "nnMetrics": nn_net.metrics,
        "transferPlan": _build_my_squad_plan(all_players, all_preds),
        "backtest": _load_backtest_summary(),
        "meta": _gameweek_meta(all_players),
    }


def _build_my_squad_plan(all_players, all_preds) -> Optional[Dict[str, Any]]:
    """Multi-gameweek MILP transfer plan for the squad in my_squad.json (None if not 15 players)."""
    try:
        from fpl_assistant import _match_squad_players

        cfg = json.loads(SQUAD_CONFIG_PATH.read_text(encoding="utf-8"))
        squad_players = _match_squad_players(all_players, cfg)
        if len(squad_players) != 15:
            return None
        by_id = {p.player.id: p for p in all_preds}
        squad_preds = [by_id[p.id] for p in squad_players]
        ft = int(cfg.get("free_transfers", 1))
        bank = float(cfg.get("bank_millions", 0.0))
        view = build_transfer_plan(squad_preds, all_preds, bank, ft, horizon=4)
        if view is None:
            return None
        weeks = []
        for w in view.plan.weeks:
            weeks.append({
                "gw": w.gw,
                "ft": w.free_transfers_before,
                "out": [view.by_id[i].player.web_name for i in w.transfers_out],
                "in": [f"{view.by_id[i].player.web_name} (£{view.by_id[i].player.price:.1f}m)" for i in w.transfers_in],
                "hits": w.hits,
                "captain": view.by_id[w.captain_id].player.web_name,
                "xp": w.expected_points,
            })
        return {"name": cfg.get("manager_name", "My squad"), "freeTransfers": ft, "bank": bank,
                "objective": view.plan.objective, "totalHits": view.plan.total_hits, "weeks": weeks}
    except Exception as exc:  # the page must still build without a valid my_squad.json
        return {"error": str(exc)}


def _load_backtest_summary() -> Optional[Dict[str, Any]]:
    if not BACKTEST_RESULTS_PATH.exists():
        return None
    try:
        res = json.loads(BACKTEST_RESULTS_PATH.read_text(encoding="utf-8"))
        s = res["summary"]
        return {
            "weeks": [w["gw"] for w in res["weeks"]],
            "totals": s["totals"],
            "modelRollingVsAvg": s["model_rolling_vs_fpl_average"],
            "naiveRollingVsAvg": s["naive_rolling_vs_fpl_average"],
            "modelFhVsNaiveFh": s["model_freehit_vs_naive_freehit"],
            "playerLevel": s["player_level"],
            "bias": s["freehit_prediction_bias"],
            "perWeek": [
                {k: w[k] for k in ("gw", "model_rolling", "naive_rolling", "fpl_average", "model_freehit",
                                   "naive_freehit", "model_rolling_hits", "player_mae_model", "player_mae_naive")}
                for w in res["weeks"]
            ],
        }
    except Exception:
        return None


def _gameweek_meta(all_players) -> Dict[str, Any]:
    """Target gameweek number, its deadline and the last fully finished gameweek."""
    from neural_model import latest_finished_gw

    gw = next((p.target_gw for p in all_players if p.target_gw), 0)
    deadline = ""
    try:
        bootstrap = json.loads((ROOT_DIR / "data" / "bootstrap_static.json").read_text(encoding="utf-8"))
        ev = next((e for e in bootstrap.get("events", []) if int(e.get("id", 0)) == gw), {})
        deadline = ev.get("deadline_time", "") or ""
    except Exception:
        pass
    return {"gw": gw, "deadline": deadline, "trainedThroughGw": latest_finished_gw()}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FPL AI Optimizer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f0a16;
      --surface: #171022;
      --surface-2: #1f1630;
      --line: rgba(255, 255, 255, 0.09);
      --line-strong: rgba(255, 255, 255, 0.16);
      --text: #f3eef9;
      --text-2: #c9bfd8;
      --muted: #8f86a3;
      --accent: #37f0a0;
      --accent-ink: #0b1a12;
      --cap: #ff3d8a;
      --turf-a: #146a3c;
      --turf-b: #187a46;
      --s-model: #2fae72;
      --s-avg: #5f78e3;
      --s-naive: #b8862a;
      --radius: 16px;
      --shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      background-image:
        radial-gradient(900px 500px at 10% -10%, rgba(55, 240, 160, 0.10), transparent 60%),
        radial-gradient(800px 500px at 95% 0%, rgba(95, 120, 227, 0.12), transparent 60%);
      color: var(--text);
      min-height: 100vh;
      padding: 22px 16px 56px;
      line-height: 1.45;
      -webkit-font-smoothing: antialiased;
    }
    .container { max-width: 1180px; margin: 0 auto; min-width: 0; }
    .tabs, .topbar { min-width: 0; }
    h1, h2, h3 { font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -0.01em; }

    /* ---------- top bar ---------- */
    .topbar {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px 20px;
      padding: 6px 4px 18px;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
      width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
      background: linear-gradient(145deg, var(--accent), #5f78e3); color: #0b0812; font-weight: 800; font-size: 0.95rem;
      font-family: 'Space Grotesk', sans-serif;
    }
    .brand h1 { font-size: 1.3rem; font-weight: 700; line-height: 1.1; }
    .brand .sub { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }
    .gw-chip {
      display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px;
      background: var(--surface); border: 1px solid var(--line-strong); font-size: 0.82rem; color: var(--text-2);
    }
    .gw-chip strong { color: var(--text); font-weight: 700; }
    .gw-chip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 4px rgba(55, 240, 160, 0.18); }

    /* ---------- variation tabs ---------- */
    .tabs {
      display: flex; gap: 6px; flex-wrap: wrap; padding: 6px; border-radius: 14px;
      background: var(--surface); border: 1px solid var(--line); margin-bottom: 16px;
    }
    .tab {
      flex: 1 1 150px; border: 0; background: transparent; color: var(--muted); cursor: pointer;
      padding: 10px 12px; border-radius: 10px; text-align: left; transition: background 0.15s, color 0.15s;
      font-family: inherit;
    }
    .tab:hover { background: rgba(255, 255, 255, 0.05); color: var(--text-2); }
    .tab.active { background: var(--surface-2); color: var(--text); box-shadow: inset 0 0 0 1px var(--line-strong); }
    .tab .t-name { display: block; font-weight: 700; font-size: 0.84rem; }
    .tab .t-meta { display: block; font-size: 0.72rem; color: var(--muted); margin-top: 2px; }
    .tab.active .t-meta { color: var(--accent); }

    /* ---------- KPI strip ---------- */
    .kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
    .kpi {
      background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 14px 16px;
      min-width: 0;
    }
    .kpi .k-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 600; }
    .kpi .k-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; line-height: 1.15; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .kpi .k-value.accent { color: var(--accent); }
    .kpi .k-sub { font-size: 0.76rem; color: var(--text-2); margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    /* ---------- main grid ---------- */
    .main-grid { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 16px; align-items: start; margin-bottom: 16px; }

    .pitch-card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 12px; box-shadow: var(--shadow); }
    .pitch {
      position: relative; border-radius: 12px; overflow: hidden;
      background:
        radial-gradient(circle at 50% 0%, rgba(255,255,255,0.07) 0%, transparent 45%),
        repeating-linear-gradient(to bottom, var(--turf-a) 0 72px, var(--turf-b) 72px 144px);
      padding: 22px 8px 24px;
    }
    .pitch::before, .pitch::after { content: ""; position: absolute; left: 50%; transform: translateX(-50%); border: 2px solid rgba(255,255,255,0.28); pointer-events: none; }
    .pitch::before { top: -2px; width: 42%; height: 86px; border-top: 0; border-radius: 0 0 4px 4px; }
    .pitch::after { bottom: 46px; width: 118px; height: 118px; border-radius: 50%; }
    .pitch-row { display: flex; justify-content: space-evenly; align-items: flex-start; gap: 6px; margin-bottom: 18px; position: relative; z-index: 2; }
    .pitch-row:last-child { margin-bottom: 0; }

    .player-card {
      width: 112px; border-radius: 12px; text-align: center; position: relative; cursor: pointer; user-select: none;
      background: rgba(12, 32, 22, 0.92); border: 1.5px solid rgba(255, 255, 255, 0.22); padding-top: 10px;
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35); transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
    }
    .player-card:hover { transform: translateY(-3px); border-color: rgba(255, 255, 255, 0.55); }
    .player-card.selected { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(55, 240, 160, 0.25), 0 8px 20px rgba(0, 0, 0, 0.5); }
    .player-card.out { opacity: 0.55; }
    .price-tag { position: absolute; top: 6px; left: 7px; font-size: 0.64rem; font-weight: 600; color: rgba(255,255,255,0.75); font-variant-numeric: tabular-nums; }
    .xp-tag {
      position: absolute; top: 5px; right: 6px; font-size: 0.68rem; font-weight: 800; padding: 2px 6px; border-radius: 6px;
      background: rgba(255, 255, 255, 0.92); color: #0b1a12; font-variant-numeric: tabular-nums;
    }
    .xp-tag.high { background: var(--accent); }
    .cap-badge {
      position: absolute; top: 27px; left: 7px; width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center;
      background: var(--cap); color: #fff; font-size: 0.64rem; font-weight: 800; box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }
    .cap-badge.vice { background: rgba(255, 255, 255, 0.25); }
    .jersey-svg { width: 50px; height: 44px; margin: 10px auto 6px; display: block; filter: drop-shadow(0 3px 4px rgba(0,0,0,0.4)); }
    .p-name { background: #fff; color: #140a1c; font-weight: 700; font-size: 0.74rem; padding: 4px 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .p-fix { background: #e9e4f1; color: #3a2347; font-weight: 600; font-size: 0.66rem; padding: 3px 4px; border-radius: 0 0 10px 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    .bench { margin-top: 12px; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--line); padding: 10px 8px 10px; }
    .bench-title { text-align: center; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 600; margin-bottom: 8px; }
    .sub-lbl { font-size: 0.62rem; font-weight: 600; color: var(--muted); margin-bottom: 4px; text-align: center; }

    /* ---------- side panels ---------- */
    .side { display: flex; flex-direction: column; gap: 14px; }
    .panel { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 16px 18px; min-width: 0; }
    .panel h3 { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 12px; font-weight: 600; }
    .panel p { font-size: 0.82rem; color: var(--text-2); }
    .stat-row { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-bottom: 1px solid var(--line); font-size: 0.82rem; }
    .stat-row:last-child { border-bottom: 0; }
    .stat-row span { color: var(--muted); }
    .stat-row strong { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }
    .note { margin-top: 12px; padding: 10px 12px; background: rgba(55, 240, 160, 0.08); border-left: 3px solid var(--accent); border-radius: 6px; font-size: 0.8rem; color: var(--text-2); }
    .verdict { font-size: 0.84rem; color: var(--text-2); }
    .verdict strong { color: var(--text); }

    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.68rem; font-weight: 600; background: rgba(255,255,255,0.08); color: var(--text-2); }
    .pill.ok { background: rgba(47, 174, 114, 0.18); color: #7ee0b4; }
    .pill.warn { background: rgba(184, 134, 42, 0.2); color: #f0c56a; }
    .pill.bad { background: rgba(255, 61, 138, 0.18); color: #ff8fbd; }

    /* ---------- lower grid ---------- */
    .lower { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    .lower .panel { height: 100%; }

    .plan-week { padding: 10px 0; border-top: 1px solid var(--line); font-size: 0.8rem; color: var(--text-2); }
    .plan-week:first-of-type { border-top: 0; padding-top: 0; }
    .plan-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .plan-head strong { color: var(--text); font-size: 0.86rem; }
    .plan-move { display: flex; gap: 8px; align-items: baseline; }
    .plan-move .arrow { color: var(--muted); }
    .plan-move .in { color: var(--accent); font-weight: 600; }
    .plan-foot { margin-top: 10px; font-size: 0.74rem; color: var(--muted); }

    .chart { width: 100%; height: auto; display: block; margin-top: 6px; }
    .chart text { font-family: 'Inter', sans-serif; fill: var(--muted); font-size: 10.5px; }
    .chart .lbl { fill: var(--text-2); font-weight: 600; }
    .chart .grid { stroke: rgba(255,255,255,0.08); stroke-width: 1; }
    .chart .axis { stroke: rgba(255,255,255,0.22); stroke-width: 1; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px 14px; font-size: 0.74rem; color: var(--text-2); margin-top: 8px; }
    .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; background: var(--c); }
    .tooltip {
      position: fixed; pointer-events: none; z-index: 50; padding: 8px 10px; border-radius: 8px; font-size: 0.76rem;
      background: #0b0712; border: 1px solid var(--line-strong); color: var(--text); box-shadow: var(--shadow); display: none; white-space: nowrap;
    }
    .tooltip b { color: var(--accent); }
    .metric-bars { margin-top: 4px; }
    .metric { padding: 8px 0; border-top: 1px solid var(--line); font-size: 0.78rem; }
    .metric:first-child { border-top: 0; }
    .metric .m-head { display: flex; justify-content: space-between; color: var(--text-2); margin-bottom: 5px; }
    .metric .m-head b { color: var(--text); font-weight: 600; }
    .bar-pair { display: grid; grid-template-columns: 44px 1fr 44px; gap: 8px; align-items: center; font-size: 0.72rem; color: var(--muted); margin: 3px 0; }
    .bar-pair .track { height: 8px; border-radius: 4px; background: rgba(255,255,255,0.07); overflow: hidden; }
    .bar-pair .fill { height: 100%; border-radius: 4px; }
    .bar-pair .val { text-align: right; color: var(--text); font-variant-numeric: tabular-nums; }
    .summary-line { display: flex; justify-content: space-between; gap: 10px; font-size: 0.78rem; color: var(--text-2); padding: 6px 0; border-top: 1px solid var(--line); }
    .summary-line:first-child { border-top: 0; }
    .summary-line b { color: var(--text); font-variant-numeric: tabular-nums; white-space: nowrap; }
    .caveat { font-size: 0.74rem; color: var(--muted); margin-top: 10px; }
    footer { margin-top: 20px; font-size: 0.74rem; color: var(--muted); text-align: center; }

    @media (max-width: 980px) {
      .main-grid { grid-template-columns: 1fr; }
      .lower { grid-template-columns: 1fr; }
      .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
      body { padding: 14px 12px 40px; }
      .pitch-card { padding: 8px; }
      .pitch-row { gap: 3px; }
      .player-card { width: calc((100vw - 64px) / 5 - 3px); max-width: 86px; min-width: 56px; border-radius: 9px; padding-top: 8px; }
      .price-tag { display: none; }
      .xp-tag { font-size: 0.6rem; padding: 1px 4px; top: 4px; right: 4px; }
      .cap-badge { width: 16px; height: 16px; font-size: 0.56rem; top: 4px; left: 4px; }
      .jersey-svg { width: 34px; height: 30px; margin: 8px auto 4px; }
      .p-name { font-size: 0.6rem; padding: 3px 3px; }
      .p-fix { font-size: 0.56rem; padding: 2px 2px; border-radius: 0 0 7px 7px; }
      .p-team { display: none; }
      .bench .pitch-row { flex-wrap: wrap; row-gap: 8px; }
      .pitch { padding: 16px 4px 18px; }
      .pitch::before { width: 55%; height: 60px; }
      .pitch::after { width: 80px; height: 80px; bottom: 34px; }
      .kpi .k-value { font-size: 1.3rem; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark">xP</div>
        <div>
          <h1>FPL AI Optimizer</h1>
          <div class="sub" id="brand-sub">Multi-head model · exact MILP squads · walk-forward backtest</div>
        </div>
      </div>
      <div class="gw-chip"><span class="dot"></span><span id="gw-chip-text">Gameweek</span></div>
    </div>

    <div class="tabs" id="tabs"></div>

    <div class="kpis">
      <div class="kpi"><div class="k-label">Projected XI points</div><div class="k-value accent" id="kpi-xp">–</div><div class="k-sub" id="kpi-xp-sub">incl. captain, floor–ceiling</div></div>
      <div class="kpi"><div class="k-label">Captain</div><div class="k-value" id="kpi-cap">–</div><div class="k-sub" id="kpi-cap-sub"></div></div>
      <div class="kpi"><div class="k-label">Formation</div><div class="k-value" id="kpi-form">–</div><div class="k-sub" id="kpi-form-sub"></div></div>
      <div class="kpi"><div class="k-label">Squad cost</div><div class="k-value" id="kpi-cost">–</div><div class="k-sub" id="kpi-cost-sub"></div></div>
    </div>

    <div class="main-grid">
      <div class="pitch-card">
        <div class="pitch">
          <div class="pitch-row" id="row-gkp"></div>
          <div class="pitch-row" id="row-def"></div>
          <div class="pitch-row" id="row-mid"></div>
          <div class="pitch-row" id="row-fwd"></div>
        </div>
        <div class="bench">
          <div class="bench-title">Bench · in auto-sub order</div>
          <div class="pitch-row" id="row-bench"></div>
        </div>
      </div>

      <div class="side">
        <div class="panel">
          <h3>Player</h3>
          <div id="player-inspector"></div>
        </div>
        <div class="panel">
          <h3>Why this squad</h3>
          <div class="verdict" id="verdict"></div>
        </div>
      </div>
    </div>

    <div class="lower">
      <div class="panel">
        <h3>Your squad · transfer plan</h3>
        <div id="plan-box"></div>
      </div>
      <div class="panel">
        <h3>Model quality · holdout gameweeks</h3>
        <div id="model-box"></div>
      </div>
      <div class="panel">
        <h3>Walk-forward backtest</h3>
        <div id="backtest-box"></div>
      </div>
    </div>

    <footer id="footer"></footer>
  </div>
  <div class="tooltip" id="tooltip"></div>

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
    const TAB_NAMES = ["This week", "5-GW run", "No premiums", "Fortress", "Differentials", "Strong bench"];
    const COLORS = { model: "#2fae72", avg: "#5f78e3", naive: "#b8862a" };

    let currentVarIdx = 0;
    let selectedPlayerId = null;
    const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    const f1 = (x) => (x === null || x === undefined) ? "–" : Number(x).toFixed(1);
    const f2 = (x) => (x === null || x === undefined) ? "–" : Number(x).toFixed(2);
    const signed = (x, d = 1) => (x >= 0 ? "+" : "") + Number(x).toFixed(d);

    function jerseySvg(team, isGkp) {
      const kit = KIT_COLORS[team] || { body: "#37003c", sleeve: "#ffffff", stripe: null };
      const body = isGkp ? (team === "BHA" ? "#ff5992" : "#10b981") : kit.body;
      const sleeve = isGkp ? body : kit.sleeve;
      const stripes = (!isGkp && kit.stripe)
        ? `<rect x="21" y="10" width="3.8" height="36" fill="${kit.stripe}" opacity="0.92"/>
           <rect x="28.1" y="10" width="3.8" height="36" fill="${kit.stripe}" opacity="0.92"/>
           <rect x="35.2" y="10" width="3.8" height="36" fill="${kit.stripe}" opacity="0.92"/>`
        : "";
      return `<svg class="jersey-svg" viewBox="0 0 60 52" aria-hidden="true">
        <polygon points="16,8 4,20 10,28 18,18" fill="${sleeve}" stroke="#0b0b0b" stroke-width="1.5"/>
        <polygon points="44,8 56,20 50,28 42,18" fill="${sleeve}" stroke="#0b0b0b" stroke-width="1.5"/>
        <rect x="16" y="8" width="28" height="38" rx="4" fill="${body}" stroke="#0b0b0b" stroke-width="1.5"/>
        ${stripes}
        <path d="M23,8 Q30,15 37,8" fill="#16001a" stroke="#0b0b0b" stroke-width="1.3"/>
      </svg>`;
    }

    const PLAYER_INDEX = {};
    function registerPlayers(list) { list.forEach(p => { PLAYER_INDEX[p.id] = p; }); }
    DATA.variations.forEach(v => { registerPlayers(v.starters); registerPlayers(v.bench); });

    function createCardHtml(p, subLabel = null) {
      const capHtml = p.isCaptain ? `<div class="cap-badge" title="Captain">C</div>` : (p.isVice ? `<div class="cap-badge vice" title="Vice-captain">V</div>` : "");
      const xpClass = p.displayXp >= 6.0 ? "xp-tag high" : "xp-tag";
      const cls = ["player-card", p.id === selectedPlayerId ? "selected" : ""].join(" ");
      const wrapStart = subLabel ? `<div><div class="sub-lbl">${esc(subLabel)}</div>` : "";
      const wrapEnd = subLabel ? `</div>` : "";
      return `${wrapStart}
        <div class="${cls}" id="pcard-${p.id}" onclick="inspectPlayer(${p.id})" title="${esc(p.fullName)}">
          <div class="price-tag">£${p.price.toFixed(1)}m</div>
          <div class="${xpClass}">${p.displayXp.toFixed(1)}</div>
          ${capHtml}
          ${jerseySvg(p.team, p.pos === "GKP")}
          <div class="p-name">${esc(p.name)}</div>
          <div class="p-fix"><span class="p-team">${esc(p.team)} · </span>${esc(p.opponent)}</div>
        </div>
      ${wrapEnd}`;
    }

    function inspectPlayer(id) {
      const p = PLAYER_INDEX[id];
      if (!p) return;
      selectedPlayerId = id;
      document.querySelectorAll(".player-card").forEach(el => el.classList.toggle("selected", el.id === `pcard-${id}`));
      const role = p.isCaptain ? '<span class="pill" style="background:rgba(255,61,138,.2);color:#ff8fbd">Captain</span>' : (p.isVice ? '<span class="pill">Vice-captain</span>' : "");
      document.getElementById("player-inspector").innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
          <div style="width:46px;flex-shrink:0;">${jerseySvg(p.team, p.pos === "GKP")}</div>
          <div style="min-width:0;">
            <div style="font-size:1rem;font-weight:700;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(p.fullName)}</div>
            <div style="font-size:0.76rem;color:var(--muted);">${esc(p.teamName)} · ${esc(p.pos)} · £${p.price.toFixed(1)}m ${role}</div>
          </div>
        </div>
        <div class="stat-row"><span>Expected points</span><strong style="color:var(--accent)">${p.displayXp.toFixed(1)}${p.isCaptain ? " (×2)" : ""}</strong></div>
        <div class="stat-row"><span>This week</span><strong>${esc(p.opponent)}</strong></div>
        <div class="stat-row"><span>Next three</span><strong style="font-size:0.74rem;">${esc(p.fixtureRun3Gw || p.opponent)}</strong></div>
        <div class="stat-row"><span>Range (floor–ceiling)</span><strong>${p.floor}–${p.ceiling}</strong></div>
        <div class="stat-row"><span>Haul odds (9+)</span><strong>${p.haulProb}%</strong></div>
        <div class="stat-row"><span>Clean-sheet odds</span><strong>${p.csProb}%</strong></div>
        <div class="stat-row"><span>Season PPG · form</span><strong>${p.seasonPpg.toFixed(1)} · ${p.form}</strong></div>
        <div class="note">${esc(p.coachNote)}</div>
      `;
    }

    function renderSquadView(squad) {
      const starters = squad.starters;
      const cap = starters.find(p => p.isCaptain) || starters[0];
      const vice = starters.find(p => p.isVice);
      document.getElementById("kpi-xp").textContent = squad.expectedPoints.toFixed(1);
      document.getElementById("kpi-xp-sub").textContent = `range ${squad.floor}–${squad.ceiling} · captain included`;
      document.getElementById("kpi-cap").textContent = cap.name;
      document.getElementById("kpi-cap-sub").textContent = `${cap.displayXp.toFixed(1)} doubled vs ${cap.opponent}${vice ? " · vice " + vice.name : ""}`;
      document.getElementById("kpi-form").textContent = squad.formation;
      const benchXp = squad.bench.reduce((s, p) => s + p.xp, 0);
      document.getElementById("kpi-form-sub").textContent = `bench ${benchXp.toFixed(1)} xP`;
      document.getElementById("kpi-cost").textContent = `£${squad.totalCost.toFixed(1)}m`;
      document.getElementById("kpi-cost-sub").textContent = `of £${squad.budget.toFixed(1)}m · £${(squad.budget - squad.totalCost).toFixed(1)}m in bank`;
      document.getElementById("verdict").innerHTML = `<p style="margin-bottom:8px;"><strong>${esc(squad.title.replace(/^Variation \d+: /, ""))}.</strong> ${esc(squad.subtitle)}</p><p>${esc(squad.coachVerdict)}</p>`;

      const byPos = (pos) => starters.filter(p => p.pos === pos).map(p => createCardHtml(p)).join("");
      document.getElementById("row-gkp").innerHTML = byPos("GKP");
      document.getElementById("row-def").innerHTML = byPos("DEF");
      document.getElementById("row-mid").innerHTML = byPos("MID");
      document.getElementById("row-fwd").innerHTML = byPos("FWD");
      document.getElementById("row-bench").innerHTML = squad.bench.map((bp, i) => createCardHtml(bp, i === 0 ? "GK" : `${i}. ${bp.pos}`)).join("");
      inspectPlayer(cap.id);
    }

    function renderTabs() {
      document.getElementById("tabs").innerHTML = DATA.variations.map((v, i) => `
        <button class="tab ${i === currentVarIdx ? "active" : ""}" onclick="selectVariation(${i})" title="${esc(v.subtitle)}">
          <span class="t-name">${esc(TAB_NAMES[i] || v.title)}</span>
          <span class="t-meta">${v.formation} · ${v.expectedPoints.toFixed(1)} xP · C ${esc(v.captain)}</span>
        </button>`).join("");
    }
    function selectVariation(i) { currentVarIdx = i; renderTabs(); renderSquadView(DATA.variations[i]); }

    /* ---------- meta ---------- */
    (function renderMeta() {
      const m = DATA.meta || {};
      let dl = "";
      if (m.deadline) {
        const d = new Date(m.deadline);
        dl = " · deadline " + d.toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
      }
      document.getElementById("gw-chip-text").innerHTML = `<strong>Gameweek ${m.gw || "?"}</strong>${esc(dl)}`;
      document.getElementById("brand-sub").textContent = `Model trained on real outcomes through GW ${m.trainedThroughGw || "?"} · squads are exact mixed-integer optima`;
    })();

    /* ---------- tooltip helpers ---------- */
    const tip = document.getElementById("tooltip");
    function showTip(evt, html) { tip.innerHTML = html; tip.style.display = "block"; moveTip(evt); }
    function moveTip(evt) { tip.style.left = (evt.clientX + 14) + "px"; tip.style.top = (evt.clientY - 10) + "px"; }
    function hideTip() { tip.style.display = "none"; }

    /* ---------- transfer plan ---------- */
    (function renderPlan() {
      const plan = DATA.transferPlan;
      const box = document.getElementById("plan-box");
      if (!plan) { box.innerHTML = `<p>Add your 15 players to <code>my_squad.json</code> and rebuild to see a multi-gameweek plan.</p>`; return; }
      if (plan.error) { box.innerHTML = `<p>Plan unavailable: ${esc(plan.error)}</p>`; return; }
      const head = `<div class="summary-line"><span>${esc(plan.name)}</span><b>FT ${plan.freeTransfers} · bank £${plan.bank.toFixed(1)}m</b></div>
                    <div class="summary-line"><span>Plan value (xP − hits)</span><b>${plan.objective.toFixed(1)} · ${plan.totalHits} hit${plan.totalHits === 1 ? "" : "s"}</b></div>`;
      const weeks = plan.weeks.map(w => {
        const hit = w.hits ? `<span class="pill bad">hit −${4 * w.hits}</span>` : (w.out.length ? `<span class="pill ok">free</span>` : `<span class="pill">hold</span>`);
        const moves = w.out.length
          ? w.out.map((o, k) => `<div class="plan-move"><span>${esc(o)}</span><span class="arrow">→</span><span class="in">${esc(w.in[k] || "")}</span></div>`).join("")
          : `<div class="plan-move"><span>No transfer</span></div>`;
        return `<div class="plan-week">
          <div class="plan-head"><strong>GW ${w.gw}</strong><span>${hit} <span class="pill">FT ${w.ft}</span></span></div>
          ${moves}
          <div style="margin-top:4px;color:var(--muted)">Captain ${esc(w.captain)} · XI ${w.xp.toFixed(1)} xP</div>
        </div>`;
      }).join("");
      box.innerHTML = head + `<div style="margin-top:8px">${weeks}</div><div class="plan-foot">Free transfers bank to five. Extra transfers cost 4 points plus a 1.5-point friction margin, so marginal swaps are skipped. Selling prices assumed equal to current prices.</div>`;
    })();

    /* ---------- model quality ---------- */
    function pairBars(label, modelVal, naiveVal, lowerIsBetter, fmt) {
      const max = Math.max(modelVal, naiveVal) * 1.15 || 1;
      const better = lowerIsBetter ? modelVal < naiveVal : modelVal > naiveVal;
      const pill = better ? `<span class="pill ok">model better</span>` : `<span class="pill warn">baseline better</span>`;
      const row = (name, v, c) => `<div class="bar-pair"><span>${name}</span><div class="track"><div class="fill" style="width:${(v / max * 100).toFixed(1)}%;background:${c}"></div></div><span class="val">${fmt(v)}</span></div>`;
      return `<div class="metric"><div class="m-head"><b>${label}</b>${pill}</div>${row("model", modelVal, COLORS.model)}${row("naive", naiveVal, COLORS.naive)}</div>`;
    }
    (function renderModel() {
      const m = DATA.nnMetrics || {};
      const h = m.holdout || {};
      const box = document.getElementById("model-box");
      if (!h.mae_points) { box.innerHTML = `<p>Run <code>python3 fpl_assistant.py train-nn</code> to populate.</p>`; return; }
      const top = Object.entries(m.feature_importance || {}).slice(0, 4).map(([k]) => k.replace(/_/g, " ")).join(", ");
      box.innerHTML = `
        <div class="summary-line"><span>Trained on</span><b>${m.real_gw_samples} player-weeks · GW ≤ ${m.latest_trained_gw}</b></div>
        <div class="summary-line"><span>Scored on</span><b>${esc(m.holdout_gws || "")} (${h.n | 0} rows, unseen)</b></div>
        <div class="metric-bars">
          ${pairBars("Points error (MAE, lower is better)", h.mae_points, h.mae_naive_baseline, true, f2)}
          ${pairBars("Correlation with actual points", h.corr_points, h.corr_naive_baseline, false, f2)}
          ${pairBars("Minutes log-loss (lower is better)", h.minutes_logloss, h.minutes_logloss_baseline, true, f2)}
          ${pairBars("Haul odds Brier (lower is better)", h.haul_brier, h.haul_brier_baseline, true, (x) => Number(x).toFixed(3))}
        </div>
        <div class="caveat">Naive baseline: prior play-rate × points per appearance. Constant-mean MAE ${f2(h.mae_constant_baseline)}. Strongest inputs: ${esc(top)}.</div>`;
    })();

    /* ---------- backtest chart ---------- */
    (function renderBacktest() {
      const bt = DATA.backtest;
      const box = document.getElementById("backtest-box");
      if (!bt || !bt.perWeek) { box.innerHTML = `<p>Run <code>python3 backtest.py</code> to populate out-of-sample results.</p>`; return; }
      const rows = bt.perWeek;
      const series = [
        { key: "model_rolling", name: "Model, 1 FT/week", c: COLORS.model },
        { key: "fpl_average", name: "FPL average", c: COLORS.avg },
        { key: "naive_rolling", name: "Naive, 1 FT/week", c: COLORS.naive },
      ];
      const W = 420, H = 190, padL = 30, padR = 8, padT = 10, padB = 24;
      const maxV = Math.max(...rows.flatMap(r => series.map(s => r[s.key]))) * 1.1;
      const groupW = (W - padL - padR) / rows.length;
      const barW = Math.min(14, (groupW - 10) / series.length - 2);
      const y = (v) => padT + (H - padT - padB) * (1 - v / maxV);
      const ticks = [0, 20, 40, 60, 80].filter(t => t <= maxV);
      let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Weekly points by strategy">`;
      ticks.forEach(t => { svg += `<line class="grid" x1="${padL}" x2="${W - padR}" y1="${y(t)}" y2="${y(t)}"/><text x="${padL - 6}" y="${y(t) + 3.5}" text-anchor="end">${t}</text>`; });
      svg += `<line class="axis" x1="${padL}" x2="${W - padR}" y1="${y(0)}" y2="${y(0)}"/>`;
      rows.forEach((r, i) => {
        const gx = padL + i * groupW + (groupW - (barW + 2) * series.length) / 2;
        series.forEach((s, k) => {
          const v = r[s.key];
          const x = gx + k * (barW + 2);
          const top = y(Math.max(v, 0));
          const hgt = Math.max(0, y(0) - top);
          const tipHtml = `GW ${r.gw} · ${s.name}<br><b>${v} pts</b>${s.key === "model_rolling" && r.model_rolling_hits ? ` (after ${r.model_rolling_hits} hit${r.model_rolling_hits > 1 ? "s" : ""})` : ""}`;
          svg += `<rect x="${x}" y="${top}" width="${barW}" height="${hgt}" rx="3" fill="${s.c}" onmousemove='moveTip(event)' onmouseenter='showTip(event, ${JSON.stringify(tipHtml)})' onmouseleave='hideTip()'/>`;
          if (s.key === "model_rolling") svg += `<text class="lbl" x="${x + barW / 2}" y="${top - 4}" text-anchor="middle">${v}</text>`;
        });
        svg += `<text x="${padL + i * groupW + groupW / 2}" y="${H - 8}" text-anchor="middle">GW ${r.gw}</text>`;
      });
      svg += `</svg>`;
      const legend = `<div class="legend">${series.map(s => `<span style="--c:${s.c}">${s.name}</span>`).join("")}</div>`;
      const m1 = bt.modelRollingVsAvg, m2 = bt.modelFhVsNaiveFh, pl = bt.playerLevel;
      const t = bt.totals;
      box.innerHTML = `
        <div class="summary-line"><span>Weeks ${bt.weeks[0]}–${bt.weeks[bt.weeks.length - 1]}, trained only on earlier weeks</span><b>${bt.weeks.length} GWs</b></div>
        ${svg}${legend}
        <div style="margin-top:10px">
          <div class="summary-line"><span>Totals · model / FPL avg / naive</span><b>${t.model_rolling} / ${t.fpl_average} / ${t.naive_rolling}</b></div>
          <div class="summary-line"><span>Model vs FPL average, per week</span><b>${signed(m1.mean)} ± ${f1(m1.sd)} · won ${m1.weeks_won}/${m1.weeks}</b></div>
          <div class="summary-line"><span>Model vs naive, weekly rebuild</span><b>${signed(m2.mean)} ± ${f1(m2.sd)} · won ${m2.weeks_won}/${m2.weeks}</b></div>
          <div class="summary-line"><span>Player-level MAE · correlation</span><b>${f2(pl.model_mae)} vs ${f2(pl.naive_mae)} · ${f2(pl.model_corr)} vs ${f2(pl.naive_corr)}</b></div>
        </div>
        <div class="caveat">Real points with auto-subs and captaincy. The rolling manager starts cold in GW 3 with two weeks of data, while humans built squads with full prior-season knowledge. One gameweek has a standard deviation of roughly 15 points.</div>`;
    })();

    document.getElementById("footer").textContent = "Expected points = P(60+ minutes) × points given a start + P(cameo) × cameo points. Availability uses the learned minutes model and official FPL flags; headlines are informational only.";

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
