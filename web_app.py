#!/usr/bin/env python3
"""
Touchline — Fantasy Football Field Notes (web_app.py)

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
        "optimizationNotes": res.tactical_notes,
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
        "historicalBacktest": _load_historical_summary(),
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


def _load_historical_summary() -> List[Dict[str, Any]]:
    path = ROOT_DIR / "data" / "historical_backtest_results.json"
    if not path.exists():
        return []
    result = json.loads(path.read_text(encoding="utf-8"))
    return [{"season": row["season"], "weeks": len(row["weeks"]), "totals": row["totals"],
             "margin": row["rolling_margin_mean"], "interval": row["rolling_margin_block_bootstrap_95pct"]}
            for row in result["seasons"]]


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
    return {"gw": gw, "deadline": deadline, "formThroughGw": latest_finished_gw()}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Touchline — Fantasy football field notes</title>
  <meta name="description" content="Your fantasy football field notes: six squad ideas, player projections, transfer plans, and an honest look at the results." />
  <meta name="theme-color" content="#f4f0e7" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Serif+4:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f4f0e7; --surface: #fbf8f1; --surface-2: #eae5d9;
      --text: #292d27; --text-2: #4d5349; --muted: #65695e;
      --line: #d6d3c6; --line-strong: #b7bbab;
      --accent: #6f293f; --cap: #6f293f; --lime: #e5e6a3;
      --turf-a: #294f43; --turf-b: #2d5447;
      --s-model: #385e4c; --s-avg: #783b50; --s-naive: #a57d3a;
      --serif: 'Source Serif 4', Georgia, serif;
      --sans: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; scroll-padding-top: 24px; }
    body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.5; -webkit-font-smoothing: antialiased; }
    button, a, summary { -webkit-tap-highlight-color: transparent; }
    button { font: inherit; }
    a { color: inherit; text-underline-offset: 4px; }
    button:focus-visible, a:focus-visible, summary:focus-visible { outline: 3px solid var(--accent); outline-offset: 4px; }
    .skip-link { position: absolute; top: -80px; left: 20px; background: var(--surface); padding: 12px; z-index: 20; }
    .skip-link:focus { top: 12px; }
    .container { max-width: 1280px; margin: 0 auto; padding: 0 40px; }
    .topbar { display: flex; justify-content: space-between; align-items: center; gap: 24px; padding: 24px 0 20px; border-bottom: 1px solid var(--text); }
    .brand { display: flex; gap: 12px; align-items: center; text-decoration: none; }
    .brand-mark { width: 32px; height: 40px; color: var(--accent); }
    .wordmark { font: 600 35px/1 var(--serif); letter-spacing: -1.8px; color: var(--accent); }
    .brand-caption, .eyebrow, .section-label, .k-label, .bench-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1.7px; font-weight: 600; }
    .brand-caption { margin-top: 7px; color: var(--muted); font-size: 9px; letter-spacing: 1.2px; }
    .top-nav { display: flex; gap: 28px; font-size: 12px; }
    .top-nav a { text-decoration: none; border-bottom: 1px solid transparent; padding: 8px 0; }
    .top-nav a:hover { border-color: var(--accent); color: var(--accent); }
    .edition { font-size: 11px; color: var(--muted); }
    .intro { display: flex; align-items: center; justify-content: space-between; gap: 32px; padding: 32px 0; }
    .eyebrow { color: var(--accent); margin-bottom: 9px; }
    h1 { font: 500 clamp(38px, 4.5vw, 60px)/1.08 var(--serif); letter-spacing: -2px; }
    h1 em { font-weight: 400; color: var(--accent); }
    .intro-copy { color: var(--text-2); font-size: 13px; margin-top: 12px; max-width: 470px; }
    .fixture-ticket { display: flex; align-items: center; gap: 22px; padding: 15px 24px; border: 1px solid var(--line-strong); background: var(--lime); transform: rotate(-1.5deg); flex-shrink: 0; }
    .ticket-number { padding-right: 22px; border-right: 1px dashed #969c70; text-align: center; }
    .ticket-number span { display: block; font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; }
    .ticket-number strong { display: block; font: 500 46px/1.05 var(--serif); }
    .ticket-info { max-width: 180px; font-size: 12px; }
    .ticket-info .eyebrow { font-size: 9px; color: #4b573a; margin-bottom: 4px; }
    .ticket-info time { display: block; font-weight: 600; }
    .ticket-info small { display: block; margin-top: 5px; font-size: 10px; color: #525b43; }
    .section-head { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 14px 0; border-top: 1px solid var(--text); }
    .section-label { display: flex; align-items: center; gap: 14px; }
    .section-number { font: italic 18px var(--serif); color: var(--accent); letter-spacing: 0; }
    .section-hint { font-size: 11px; color: var(--muted); }
    .tabs { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); border: 1px solid var(--line); margin-bottom: 24px; }
    .tab { position: relative; border: 0; border-right: 1px solid var(--line); background: transparent; color: var(--text-2); cursor: pointer; padding: 13px 14px; text-align: left; transition: background .15s; min-width: 0; }
    .tab:last-child { border-right: 0; }
    .tab:hover { background: var(--surface-2); }
    .tab.active { background: var(--accent); color: var(--surface); }
    .t-number { font: italic 12px var(--serif); opacity: .7; display: block; margin-bottom: 6px; }
    .t-name { display: block; font-weight: 600; font-size: 12px; }
    .t-meta { display: block; font-size: 10px; margin-top: 3px; opacity: .85; }
    .kpis { display: grid; grid-template-columns: 1.05fr 1.2fr .8fr 1fr; margin-bottom: 25px; }
    .kpi { min-width: 0; padding: 0 24px; border-left: 1px solid var(--line); }
    .kpi:first-child { padding-left: 0; border-left: 0; }
    .k-label { color: var(--muted); font-size: 9px; letter-spacing: 1.1px; }
    .k-value { font: 500 34px/1.25 var(--serif); letter-spacing: -1px; margin: 3px 0; overflow-wrap: anywhere; }
    .k-value.accent { color: var(--accent); }
    .k-sub { font-size: 11px; color: var(--muted); }
    .main-grid { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 28px; align-items: start; }
    .pitch-card { background: var(--turf-a); padding: 16px; color: var(--surface); }
    .pitch-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 1px 4px 15px; }
    .pitch-heading h2 { font: 500 23px var(--serif); }
    .pitch-heading span { font-size: 10px; color: #d1dbce; }
    .pitch { position: relative; overflow: hidden; background: repeating-linear-gradient(to bottom, var(--turf-a) 0 78px, var(--turf-b) 78px 156px); padding: 18px 6px; border: 1px solid #708877; }
    .pitch-lines { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; color: #8da08c; opacity: .55; }
    .pitch-row { display: flex; justify-content: space-evenly; align-items: flex-start; gap: 6px; margin-bottom: 22px; position: relative; z-index: 2; }
    .pitch-row:last-child { margin-bottom: 0; }
    .player-card { display: block; flex: 0 1 105px; width: 105px; min-width: 0; padding: 10px 0 0; border-radius: 3px; position: relative; cursor: pointer; text-align: center; background: #203e34; border: 1px solid #78907d; color: var(--surface); transition: transform .15s, outline-color .15s; }
    .player-card:hover { transform: translateY(-3px); }
    .player-card.selected { outline: 2px solid var(--lime); outline-offset: 3px; }
    .player-card:focus-visible { outline: 3px solid white; outline-offset: 4px; }
    .price-tag { position: absolute; top: 6px; left: 6px; font-size: 9px; color: #d5decc; font-variant-numeric: tabular-nums; }
    .xp-tag { position: absolute; top: 5px; right: 5px; font-size: 10px; font-weight: 700; padding: 1px 4px; border-radius: 2px; background: #f4f0e7; color: #243d30; font-variant-numeric: tabular-nums; }
    .xp-tag.high { background: var(--lime); }
    .cap-badge { position: absolute; top: 28px; left: 6px; width: 18px; height: 18px; display: grid; place-items: center; border-radius: 50%; background: var(--cap); color: white; font-size: 10px; font-weight: 700; border: 1px solid #e6c9ca; }
    .cap-badge.vice { background: #dfdccb; color: #3d4a3e; border-color: #dfdccb; }
    .jersey-svg { width: 50px; height: 44px; margin: 11px auto 7px; display: block; }
    .p-name { display: block; background: var(--surface); color: #292d27; font-weight: 700; font-size: 11px; padding: 4px 3px 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .p-fix { display: block; background: #e9e7d9; color: #465044; font-size: 9px; padding: 3px 2px; border-radius: 0 0 2px 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bench { margin-top: 15px; padding: 13px 6px 4px; border-top: 1px solid #708877; }
    .bench-title { font-size: 9px; text-align: center; color: #d1dbce; margin-bottom: 10px; letter-spacing: 1.3px; }
    .sub-lbl { color: #d1dbce; font-size: 9px; text-align: center; margin-bottom: 5px; }
    .side { display: flex; flex-direction: column; gap: 22px; min-width: 0; }
    .panel { min-width: 0; }
    .panel h3 { font: 500 25px/1.2 var(--serif); letter-spacing: -.5px; margin-bottom: 16px; }
    .panel-kicker { display: block; color: var(--muted); text-transform: uppercase; letter-spacing: 1.4px; font: 600 9px var(--sans); margin-bottom: 7px; }
    .scout-panel { background: var(--surface); padding: 23px; border: 1px solid var(--line); border-top: 3px solid var(--accent); }
    .player-profile { display: flex; align-items: center; gap: 12px; padding-bottom: 16px; }
    .profile-kit { width: 52px; flex-shrink: 0; }
    .profile-copy { min-width: 0; }
    .profile-name { font: 600 23px/1.15 var(--serif); letter-spacing: -.5px; overflow-wrap: anywhere; }
    .profile-meta { font-size: 11px; color: var(--muted); margin-top: 6px; }
    .stat-row { display: flex; justify-content: space-between; gap: 14px; padding: 8px 0; border-bottom: 1px solid var(--line); font-size: 11px; }
    .stat-row span { color: var(--muted); }
    .stat-row strong { font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; max-width: 60%; }
    .note { margin-top: 15px; font-size: 11px; line-height: 1.65; color: var(--text-2); border-left: 2px solid var(--line-strong); padding-left: 12px; }
    .verdict-panel { padding: 0 4px; }
    .verdict-panel h3 { font-style: italic; margin-bottom: 12px; }
    .panel p, .verdict { font-size: 12px; line-height: 1.7; color: var(--text-2); }
    .verdict p + p { margin-top: 10px; }
    details { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 10px; }
    summary { font-size: 11px; font-weight: 600; cursor: pointer; color: var(--accent); }
    .pill { display: inline-block; padding: 2px 6px; font-size: 9px; font-weight: 600; background: var(--surface-2); color: var(--text-2); border-radius: 2px; }
    .pill.ok { background: #e2ebdf; color: #36543a; }
    .pill.warn { background: #f0e3c7; color: #76551f; }
    .pill.bad, .pill.captain { background: #eddce1; color: var(--accent); }
    .moves-section, .evidence-section { margin-top: 36px; }
    .plan-summary { display: flex; justify-content: space-between; gap: 20px; padding: 6px 0 18px; }
    .plan-summary h3 { font: 500 27px var(--serif); }
    .plan-summary p { color: var(--muted); font-size: 11px; margin-top: 4px; }
    .plan-value { text-align: right; }
    .plan-value strong { display: block; font: 500 28px var(--serif); color: var(--accent); }
    .plan-value span { font-size: 10px; color: var(--muted); }
    .plan-weeks { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); border: 1px solid var(--line); background: var(--surface); }
    .plan-week { padding: 18px; border-right: 1px solid var(--line); font-size: 12px; color: var(--text-2); }
    .plan-week:last-child { border-right: 0; }
    .plan-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 14px; }
    .plan-head strong { font: 500 22px var(--serif); color: var(--text); }
    .plan-move { display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; margin: 5px 0; }
    .plan-move .arrow { color: var(--muted); }
    .plan-move .in { color: var(--accent); font-weight: 600; }
    .plan-foot { margin-top: 12px; font-size: 10px; color: var(--muted); max-width: 800px; }
    .lower { display: grid; grid-template-columns: .85fr 1.15fr; gap: 36px; }
    .lower .panel + .panel { border-left: 1px solid var(--line); padding-left: 36px; }
    .evidence-intro { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 10px; margin: 5px 0 25px; }
    .evidence-intro h2 { font: 500 32px var(--serif); letter-spacing: -.7px; }
    .evidence-intro p { font-size: 12px; color: var(--muted); }
    .chart { width: 100%; height: auto; display: block; margin-top: 18px; }
    .chart text { font-family: var(--sans); fill: var(--muted); font-size: 10.5px; }
    .chart .lbl { fill: var(--text); font-weight: 600; }
    .chart .grid { stroke: var(--line); stroke-width: 1; }
    .chart .axis { stroke: var(--line-strong); stroke-width: 1; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 16px; font-size: 10px; color: var(--text-2); margin-top: 10px; }
    .legend span::before { content: ''; display: inline-block; width: 8px; height: 8px; margin-right: 6px; background: var(--c); }
    .tooltip { position: fixed; pointer-events: none; z-index: 50; padding: 9px 12px; font-size: 11px; background: var(--text); color: var(--surface); display: none; max-width: calc(100vw - 24px); }
    .tooltip b { color: var(--lime); }
    .metric-bars { margin-top: 16px; }
    .metric { padding: 13px 0; border-top: 1px solid var(--line); font-size: 11px; }
    .m-head { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .m-head b { font-weight: 600; }
    .bar-pair { display: grid; grid-template-columns: 44px 1fr 44px; gap: 8px; align-items: center; font-size: 10px; color: var(--muted); margin: 5px 0; }
    .track { height: 6px; background: var(--surface-2); overflow: hidden; }
    .fill { height: 100%; }
    .val { text-align: right; color: var(--text); font-variant-numeric: tabular-nums; }
    .summary-line { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px 16px; font-size: 11px; color: var(--text-2); padding: 9px 0; border-top: 1px solid var(--line); }
    .summary-line:first-child { border-top: 0; }
    .summary-line b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }
    .caveat { font-size: 10px; line-height: 1.7; color: var(--muted); margin-top: 10px; }
    .historical-heading { font: 500 21px var(--serif); margin-top: 24px; margin-bottom: 8px; }
    .site-footer { display: flex; justify-content: space-between; gap: 40px; border-top: 1px solid var(--text); margin-top: 40px; padding: 24px 0 32px; }
    .footer-brand { font: 600 22px var(--serif); color: var(--accent); white-space: nowrap; }
    #footer { max-width: 720px; font-size: 10px; line-height: 1.7; color: var(--muted); }
    @media (max-width: 1050px) {
      .container { padding: 0 24px; }
      .main-grid { grid-template-columns: minmax(0, 1fr) 290px; gap: 20px; }
      .scout-panel { padding: 18px; }
      .tab { padding: 12px 10px; }
      .kpi { padding: 0 16px; }
      .k-value { font-size: 29px; }
      .p-team { display: none; }
      .price-tag { display: none; }
      .plan-weeks { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .plan-week:nth-child(2n) { border-right: 0; }
      .plan-week:nth-child(n+3) { border-top: 1px solid var(--line); }
    }
    @media (max-width: 800px) {
      .edition { display: none; }
      .intro { gap: 20px; }
      .fixture-ticket { padding: 12px 15px; gap: 14px; }
      .ticket-number { padding-right: 14px; }
      .ticket-info { max-width: 140px; }
      .main-grid { grid-template-columns: minmax(0, 1fr); }
      .side { display: grid; grid-template-columns: 1.1fr 1fr; gap: 24px; }
      .verdict-panel { padding-top: 12px; }
      .tabs { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .tab:nth-child(3) { border-right: 0; }
      .tab:nth-child(n+4) { border-top: 1px solid var(--line); }
      .t-number { display: inline; margin-right: 6px; }
      .t-name { display: inline; }
      .lower { gap: 24px; }
      .lower .panel + .panel { padding-left: 24px; }
      .price-tag, .p-team { display: block; }
      .p-team { display: inline; }
    }
    @media (max-width: 600px) {
      .container { padding: 0 16px; }
      .topbar { padding: 20px 0 16px; gap: 15px; }
      .wordmark { font-size: 30px; }
      .brand-mark { width: 25px; height: 34px; }
      .brand { gap: 8px; }
      .brand-caption { font-size: 8px; letter-spacing: .7px; }
      .top-nav { gap: 14px; font-size: 10px; }
      .top-nav a:first-child { display: none; }
      .intro { display: block; padding: 26px 0 24px; }
      h1 { font-size: 43px; letter-spacing: -1.7px; }
      .intro-copy { font-size: 12px; max-width: 320px; }
      .fixture-ticket { margin-top: 22px; padding: 10px 15px; transform: none; width: 100%; gap: 16px; }
      .ticket-number { display: flex; align-items: center; gap: 8px; padding-right: 16px; }
      .ticket-number span { font-size: 8px; max-width: 45px; }
      .ticket-number strong { font-size: 34px; }
      .ticket-info { max-width: none; }
      .ticket-info time { font-size: 11px; }
      .ticket-info small { display: none; }
      .ticket-info .eyebrow { margin-bottom: 2px; font-size: 8px; }
      .section-label { font-size: 9px; gap: 9px; letter-spacing: 1.3px; }
      .section-hint { font-size: 9px; }
      .tabs { margin-bottom: 20px; }
      .tab { padding: 12px 9px; }
      .t-name { font-size: 11px; }
      .t-number { font-size: 10px; }
      .t-meta { font-size: 9px; }
      .kpis { grid-template-columns: 1fr 1fr; gap: 16px 0; }
      .kpi { padding: 0 14px; }
      .kpi:nth-child(3) { border-left: 0; padding-left: 0; }
      .k-label { font-size: 8px; }
      .k-value { font-size: 30px; }
      .k-sub { font-size: 10px; }
      .pitch-card { padding: 11px 9px; }
      .pitch-heading { padding: 2px 3px 12px; }
      .pitch-heading h2 { font-size: 21px; }
      .pitch-heading span { font-size: 9px; }
      .pitch { padding: 16px 4px; }
      .pitch-row { gap: 5px; margin-bottom: 25px; }
      .player-card { width: min(82px, calc((100vw - 84px) / 5)); flex-basis: min(82px, calc((100vw - 84px) / 5)); padding-top: 8px; }
      .price-tag, .p-team { display: none; }
      .xp-tag { font-size: 9px; padding: 0 3px; top: 4px; right: 3px; }
      .cap-badge { width: 14px; height: 14px; font-size: 8px; top: 4px; left: 3px; }
      .jersey-svg { width: 34px; height: 30px; margin: 12px auto 6px; }
      .p-name { font-size: 9px; padding: 4px 2px 2px; }
      .p-fix { font-size: 8px; padding: 3px 1px; }
      .profile-kit .jersey-svg { width: 50px; height: 44px; }
      .bench { padding: 12px 2px 4px; }
      .side { display: block; }
      .verdict-panel { padding: 24px 4px 0; }
      .scout-panel { padding: 20px; }
      .stat-row { font-size: 12px; }
      .lower { grid-template-columns: minmax(0, 1fr); gap: 28px; }
      .lower .panel + .panel { padding: 24px 0 0; border-left: 0; border-top: 1px solid var(--line); }
      .plan-week { padding: 14px 12px; font-size: 11px; }
      .plan-head { align-items: flex-start; flex-direction: column; gap: 6px; }
      .plan-summary h3 { font-size: 24px; }
      .evidence-intro h2 { font-size: 29px; }
      .site-footer { gap: 14px; flex-direction: column; }
    }
    @media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } *, *::before, *::after { transition: none !important; } }
  </style>
</head>
<body>
  <a class="skip-link" href="#selection">Skip to the squad</a>
  <div class="container">
    <header class="topbar">
      <a class="brand" href="#" aria-label="Touchline home">
        <svg class="brand-mark" viewBox="0 0 32 40" fill="none" aria-hidden="true"><path d="M2 2h28v25L16 38 2 27Z" stroke="currentColor" stroke-width="1.5"/><path d="M2 16h28M16 2v36" stroke="currentColor" stroke-width="1"/><circle cx="16" cy="16" r="7" fill="var(--bg)" stroke="currentColor"/><path d="m16 11 4 3-1.5 4.5h-5L12 14Z" fill="currentColor"/></svg>
        <div><div class="wordmark">Touchline.</div><div class="brand-caption">Fantasy football field notes</div></div>
      </a>
      <nav class="top-nav" aria-label="Page sections"><a href="#selection">The lineup</a><a href="#moves">Your moves</a><a href="#evidence">The evidence</a></nav>
      <span class="edition" id="edition">The gameweek edition</span>
    </header>

    <main>
      <div class="intro">
        <div>
          <p class="eyebrow">A head start for your gameweek</p>
          <h1>Your next <em>eleven.</em></h1>
          <p class="intro-copy">Six ways to set up. The numbers behind every pick.<br>A little homework before the whistle.</p>
        </div>
        <div class="fixture-ticket" aria-label="Upcoming gameweek deadline">
          <div class="ticket-number"><span>Game<br>week</span><strong id="gw-number">–</strong></div>
          <div class="ticket-info"><p class="eyebrow">The team sheet closes</p><time id="gw-deadline">Deadline to be confirmed</time><small id="gw-form"></small></div>
        </div>
      </div>

      <section id="selection" aria-labelledby="selection-label">
        <div class="section-head"><h2 class="section-label" id="selection-label"><span class="section-number">01</span> The selection</h2><span class="section-hint">Six approaches · £100m to work with</span></div>
        <div class="tabs" id="tabs" role="group" aria-label="Squad approaches"></div>
        <div id="squad-view">
          <div class="kpis" aria-live="polite" aria-atomic="true">
            <div class="kpi"><div class="k-label">Projected XI points</div><div class="k-value accent" id="kpi-xp">–</div><div class="k-sub" id="kpi-xp-sub"></div></div>
            <div class="kpi"><div class="k-label">The armband</div><div class="k-value" id="kpi-cap">–</div><div class="k-sub" id="kpi-cap-sub"></div></div>
            <div class="kpi"><div class="k-label">The shape</div><div class="k-value" id="kpi-form">–</div><div class="k-sub" id="kpi-form-sub"></div></div>
            <div class="kpi"><div class="k-label">Squad spend</div><div class="k-value" id="kpi-cost">–</div><div class="k-sub" id="kpi-cost-sub"></div></div>
          </div>
          <div class="main-grid">
            <div class="pitch-card">
              <div class="pitch-heading"><h2 id="squad-title">The starting XI</h2><span>Tap a shirt for the scouting notes ↗</span></div>
              <div class="pitch" role="group" aria-label="Starting eleven">
                <svg class="pitch-lines" viewBox="0 0 600 540" preserveAspectRatio="none" fill="none" aria-hidden="true"><path d="M12 12H588V528H12ZM12 270H588M190 12V88H410V12M250 12V42H350V12M190 528V452H410V528M250 528V498H350V528" stroke="currentColor"/><circle cx="300" cy="270" r="60" stroke="currentColor"/><circle cx="300" cy="270" r="3" fill="currentColor"/><path d="M263 88Q300 130 337 88M263 452Q300 410 337 452" stroke="currentColor"/></svg>
                <div class="pitch-row" id="row-gkp"></div><div class="pitch-row" id="row-def"></div><div class="pitch-row" id="row-mid"></div><div class="pitch-row" id="row-fwd"></div>
              </div>
              <div class="bench"><div class="bench-title">The bench · in auto-sub order</div><div class="pitch-row" id="row-bench" role="group" aria-label="Substitutes"></div></div>
            </div>
            <aside class="side" aria-label="Scouting notes">
              <div class="panel scout-panel"><h3><span class="panel-kicker">Under the microscope</span>Player notes</h3><div id="player-inspector" aria-live="polite"></div></div>
              <div class="panel verdict-panel"><h3>From the touchline</h3><div class="verdict" id="verdict"></div></div>
            </aside>
          </div>
        </div>
      </section>

      <section class="moves-section" id="moves" aria-labelledby="moves-label">
        <div class="section-head"><h2 class="section-label" id="moves-label"><span class="section-number">02</span> Your next moves</h2><span class="section-hint">A plan for the weeks ahead</span></div>
        <div id="plan-box"></div>
      </section>

      <section class="evidence-section" id="evidence" aria-labelledby="evidence-label">
        <div class="section-head"><h2 class="section-label" id="evidence-label"><span class="section-number">03</span> The evidence</h2><span class="section-hint">Keep the predictions honest</span></div>
        <div class="evidence-intro"><h2>Show your working.</h2><p id="brand-sub">Historical training. Out-of-sample results.</p></div>
        <div class="lower">
          <div class="panel"><h3><span class="panel-kicker">The individual picks</span>How well do we predict?</h3><div id="model-box"></div></div>
          <div class="panel"><h3><span class="panel-kicker">The whole team</span>How did the strategy do?</h3><div id="backtest-box"></div></div>
        </div>
      </section>
    </main>
    <footer class="site-footer"><span class="footer-brand">Touchline.</span><p id="footer"></p></footer>
  </div>
  <div class="tooltip" id="tooltip" role="tooltip"></div>

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
    const COLORS = { model: "#385e4c", avg: "#783b50", naive: "#a57d3a" };

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
      const capHtml = p.isCaptain ? `<span class="cap-badge" title="Captain">C</span>` : (p.isVice ? `<span class="cap-badge vice" title="Vice-captain">V</span>` : "");
      const xpClass = p.displayXp >= 6.0 ? "xp-tag high" : "xp-tag";
      const cls = ["player-card", p.id === selectedPlayerId ? "selected" : ""].join(" ");
      const wrapStart = subLabel ? `<div><div class="sub-lbl">${esc(subLabel)}</div>` : "";
      const wrapEnd = subLabel ? `</div>` : "";
      return `${wrapStart}
        <button type="button" class="${cls}" id="pcard-${p.id}" onclick="inspectPlayer(${p.id})" aria-pressed="${p.id === selectedPlayerId}" aria-controls="player-inspector" aria-label="${esc(p.fullName)}, ${esc(p.pos)}, ${p.displayXp.toFixed(1)} expected points${p.isCaptain ? ', captain, points doubled' : p.isVice ? ', vice-captain' : ''}" title="${esc(p.fullName)}">
          <span class="price-tag">£${p.price.toFixed(1)}m</span>
          <span class="${xpClass}">${p.displayXp.toFixed(1)}</span>
          ${capHtml}
          ${jerseySvg(p.team, p.pos === "GKP")}
          <span class="p-name">${esc(p.name)}</span>
          <span class="p-fix"><span class="p-team">${esc(p.team)} · </span>${esc(p.opponent)}</span>
        </button>
      ${wrapEnd}`;
    }

    function inspectPlayer(id) {
      const p = PLAYER_INDEX[id];
      if (!p) return;
      selectedPlayerId = id;
      document.querySelectorAll(".player-card").forEach(el => {
        const selected = el.id === `pcard-${id}`;
        el.classList.toggle("selected", selected);
        el.setAttribute("aria-pressed", String(selected));
      });
      const role = p.isCaptain ? '<span class="pill captain">Captain</span>' : (p.isVice ? '<span class="pill">Vice-captain</span>' : "");
      document.getElementById("player-inspector").innerHTML = `
        <div class="player-profile">
          <div class="profile-kit">${jerseySvg(p.team, p.pos === "GKP")}</div>
          <div class="profile-copy">
            <div class="profile-name">${esc(p.fullName)}</div>
            <div class="profile-meta">${esc(p.teamName)} · ${esc(p.pos)} · £${p.price.toFixed(1)}m ${role}</div>
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
      // Captain, vice-captain and doubled points belong to this variation.
      registerPlayers(squad.starters);
      registerPlayers(squad.bench);
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
      document.getElementById("squad-title").textContent = TAB_NAMES[currentVarIdx] || "The starting XI";
      document.getElementById("verdict").innerHTML = `<p><strong>${esc(squad.title.replace(/^Variation \d+: /, ""))}.</strong> ${esc(squad.subtitle)}</p><p>${esc(squad.coachVerdict)}</p><details><summary>How this squad was selected</summary><div class="caveat">${(squad.optimizationNotes || []).map(esc).join("<br>")}</div></details>`;

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
        <button type="button" class="tab ${i === currentVarIdx ? "active" : ""}" aria-pressed="${i === currentVarIdx}" aria-controls="squad-view" onclick="selectVariation(${i})" title="${esc(v.subtitle)}">
          <span class="t-number">0${i + 1}</span>
          <span class="t-name">${esc(TAB_NAMES[i] || v.title)}</span>
          <span class="t-meta">${v.formation} · ${v.expectedPoints.toFixed(1)} xP</span>
        </button>`).join("");
    }
    function selectVariation(i) {
      currentVarIdx = i;
      // Preserve keyboard focus instead of replacing the clicked button.
      if (!document.getElementById("tabs").children.length) renderTabs();
      document.querySelectorAll(".tab").forEach((el, n) => {
        el.classList.toggle("active", n === i);
        el.setAttribute("aria-pressed", String(n === i));
      });
      renderSquadView(DATA.variations[i]);
    }

    /* ---------- meta ---------- */
    (function renderMeta() {
      const m = DATA.meta || {};
      document.getElementById("gw-number").textContent = m.gw || "–";
      if (m.deadline) {
        const d = new Date(m.deadline);
        const deadline = document.getElementById("gw-deadline");
        deadline.dateTime = m.deadline;
        deadline.textContent = d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
        const year = d.getUTCFullYear() - (d.getUTCMonth() < 6 ? 1 : 0);
        document.getElementById("edition").textContent = `${year}/${String(year + 1).slice(2)} · Gameweek ${m.gw || "–"}`;
      }
      document.getElementById("gw-form").textContent = m.formThroughGw ? `Form through GW ${m.formThroughGw} · your local time` : "Shown in your local time";
      document.getElementById("brand-sub").textContent = `${(DATA.nnMetrics.real_gw_samples || 0).toLocaleString()} player-matches · ${(DATA.nnMetrics.training_seasons || []).length} completed seasons`;
    })();

    /* ---------- tooltip helpers ---------- */
    const tip = document.getElementById("tooltip");
    function showTip(evt, html) { tip.innerHTML = html; tip.style.display = "block"; moveTip(evt); }
    function moveTip(evt) {
      tip.style.left = Math.max(12, Math.min(evt.clientX + 14, innerWidth - tip.offsetWidth - 12)) + "px";
      tip.style.top = Math.max(12, Math.min(evt.clientY - 10, innerHeight - tip.offsetHeight - 12)) + "px";
    }
    function hideTip() { tip.style.display = "none"; }

    /* ---------- transfer plan ---------- */
    (function renderPlan() {
      const plan = DATA.transferPlan;
      const box = document.getElementById("plan-box");
      if (!plan) { box.innerHTML = `<p>Add your 15 players to <code>my_squad.json</code> and rebuild to see a multi-gameweek plan.</p>`; return; }
      if (plan.error) { box.innerHTML = `<p>Plan unavailable: ${esc(plan.error)}</p>`; return; }
      const head = `<div class="plan-summary"><div><h3>${esc(plan.name)}</h3><p>${plan.freeTransfers} free transfer${plan.freeTransfers === 1 ? "" : "s"} · £${plan.bank.toFixed(1)}m in the bank</p></div><div class="plan-value"><strong>${plan.objective.toFixed(1)}</strong><span>plan xP after ${plan.totalHits} hit${plan.totalHits === 1 ? "" : "s"}</span></div></div>`;
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
      box.innerHTML = head + `<div class="plan-weeks">${weeks}</div><div class="plan-foot">Free transfers bank to five. Extra transfers cost 4 points plus a 1.5-point friction margin, so marginal swaps are skipped. Selling prices assumed equal to current prices.</div>`;
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
      const h = m.test || m.validation || m.holdout || {};
      const box = document.getElementById("model-box");
      if (!h.mae_points) { box.innerHTML = `<p>Run <code>python3 fpl_assistant.py train-nn</code> to populate.</p>`; return; }
      const top = Object.entries(m.feature_importance || {}).slice(0, 4).map(([k]) => k.replace(/_/g, " ")).join(", ");
      box.innerHTML = `
        <div class="summary-line"><span>Historical training</span><b>${Number(m.real_gw_samples).toLocaleString()} player-matches · ${esc(m.training_end || "")}</b></div>
        <div class="summary-line"><span>${m.test ? "Held-out test" : "Validation"}</span><b>${esc((m.test ? m.test_gws : m.holdout_gws) || "")} (${(h.n | 0).toLocaleString()} rows)</b></div>
        <div class="metric-bars">
          ${pairBars("Points error (MAE, lower is better)", h.mae_points, h.mae_naive_baseline, true, f2)}
          ${pairBars("Correlation with actual points", h.corr_points, h.corr_naive_baseline, false, f2)}
          ${pairBars("Minutes log-loss (lower is better)", h.minutes_logloss, h.minutes_logloss_baseline, true, f2)}
          ${pairBars("Haul odds Brier (lower is better)", h.haul_brier, h.haul_brier_baseline, true, (x) => Number(x).toFixed(3))}
        </div>
        <div class="caveat">Current-season outcomes are excluded from training and tuning. Naive baseline: prior points per fixture.</div><details><summary>Training & evaluation notes</summary><div class="caveat">Trained on ${esc((m.training_seasons || []).join(", "))}. ${m.test ? "Historical test rows are excluded from model selection and calibration." : "Validation is used for model selection."} Deployment is then refit on completed seasons. Constant-mean MAE ${f2(h.mae_constant_baseline)}. Largest input-weight norms: ${esc(top)}.</div></details>`;
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
        <div class="summary-line"><span>This season · GW ${bt.weeks[0]}–${bt.weeks[bt.weeks.length - 1]}</span><b>${bt.weeks.length} gameweeks</b></div>
        ${svg}${legend}
        <div style="margin-top:10px">
          <div class="summary-line"><span>Totals · model / FPL avg / naive</span><b>${t.model_rolling} / ${t.fpl_average} / ${t.naive_rolling}</b></div>
          <div class="summary-line"><span>Model vs FPL average, per week</span><b>${signed(m1.mean)} ± ${f1(m1.sd)} · won ${m1.weeks_won}/${m1.weeks}</b></div>
          <div class="summary-line"><span>Model vs naive, weekly rebuild</span><b>${signed(m2.mean)} ± ${f1(m2.sd)} · won ${m2.weeks_won}/${m2.weeks}</b></div>
          <div class="summary-line"><span>Player-level MAE · correlation</span><b>${f2(pl.model_mae)} vs ${f2(pl.naive_mae)} · ${f2(pl.model_corr)} vs ${f2(pl.naive_corr)}</b></div>
        </div>
        <div class="caveat">Current-season outcomes are evaluated with fixed historical weights. Form inputs use earlier weeks; roster and club membership use today's snapshot. Short tests cannot establish a reliable edge.</div>`;
      const historical = DATA.historicalBacktest || [];
      if (historical.length) box.innerHTML += `<h4 class="historical-heading">Further back · ${historical.reduce((n, r) => n + r.weeks, 0)} historical weeks</h4>
        ${historical.map(r => `<div class="summary-line"><span>${esc(r.season)} · model / naive</span><b>${r.totals.model_rolling} / ${r.totals.naive_rolling}</b></div>
        <div class="caveat">Margin ${signed(r.margin)} pts/week; 95% block-bootstrap interval ${f1(r.interval[0])} to ${f1(r.interval[1])}.</div>`).join("")}
        <div class="caveat">GW19–38 in each season, excluded from fitting, model selection and calibration. Official historical human averages are unavailable. Archived fixture schedules may reflect later postponements.</div>`;
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
        print(f"Serving Touchline at http://localhost:{port}/index.html")
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
