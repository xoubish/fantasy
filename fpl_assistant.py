#!/usr/bin/env python3
"""
FPL AI Helper & Prediction Pipeline (fpl_assistant.py)

Usage:
  python3 fpl_assistant.py analyze          # Analyze your 15-man squad + multi-week transfer plan
  python3 fpl_assistant.py plan             # Multi-gameweek MILP transfer plan for my_squad.json
  python3 fpl_assistant.py best-squad       # Exact best £100m squad (MILP) for this gameweek
  python3 fpl_assistant.py dream-team       # Optimal Free Hit XI from scratch
  python3 fpl_assistant.py sync             # Sync live FPL API + match logs + news
  python3 fpl_assistant.py train-nn         # Retrain the multi-head model (time-ordered holdout report)
  python3 backtest.py                       # Walk-forward backtest vs naive baseline and FPL average
  python3 fpl_assistant.py add-news "Player" "Headline" "Manager quote or article snippet"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from fpl_data import DATA_DIR, PlayerRecord, load_players, sync_live_fpl_data
from neural_model import get_or_train_neural_net
from news_scraper import load_all_news, scrape_sports_websites
from pitch_renderer import render_pitch_figure
from predictor import (
    OptimizedSquadResult,
    PlayerPrediction,
    TransferPlanView,
    TransferSuggestion,
    build_dream_xi,
    build_transfer_plan,
    get_best_squad_of_the_week,
    optimize_starting_xi,
    predict_player,
    recommend_single_transfers,
)

SQUAD_CONFIG_PATH = Path(__file__).parent / "my_squad.json"
REPORT_PATH = DATA_DIR / "latest_report.md"


def _match_squad_players(
    all_players: List[PlayerRecord], squad_cfg: dict
) -> List[PlayerRecord]:
    matched: List[PlayerRecord] = []
    for item in squad_cfg.get("squad", []):
        name = item["name"].strip().lower()
        team = item.get("team", "").strip().upper()
        pos = item.get("position", "").strip().upper()

        exact = next(
            (
                p
                for p in all_players
                if p.web_name.lower() == name
                and (not team or p.team_short == team)
                and (not pos or p.position == pos)
            ),
            None,
        )
        if exact:
            matched.append(exact)
            continue

        by_pos = next(
            (
                p
                for p in all_players
                if p.web_name.lower() == name and (not pos or p.position == pos)
            ),
            None,
        )
        if by_pos:
            matched.append(by_pos)
    return matched


def format_markdown_report(
    squad_result: OptimizedSquadResult,
    dream_result: OptimizedSquadResult,
    all_squad_preds: List[PlayerPrediction],
    transfers: List[TransferSuggestion],
    free_transfers: int,
    bank_millions: float,
    plan_view: Optional[TransferPlanView] = None,
) -> str:
    lines: List[str] = []
    lines.append("# FPL Weekly Data & News Prediction Report\n")
    lines.append(
        f"- **Recommended Formation:** `{squad_result.formation}`\n"
        f"- **Captain (2x):** **{squad_result.captain.player.web_name}** "
        f"({squad_result.captain.player.next_opponent}) — "
        f"`{squad_result.captain.expected_points * 2:.1f} xP`\n"
        f"- **Vice-Captain:** **{squad_result.vice_captain.player.web_name}** "
        f"({squad_result.vice_captain.player.next_opponent}) — "
        f"`{squad_result.vice_captain.expected_points:.1f} xP`\n"
        f"- **Projected Total Score:** **{squad_result.total_expected_points:.1f} pts** "
        f"*(Floor: {squad_result.total_floor} pts | Ceiling: {squad_result.total_ceiling} pts)*\n"
    )

    lines.append("## 1. Optimal Starting XI\n")
    lines.append(
        "| Pos | Player | Club | Fixture | 3Yr PPG | CS% | 2+GC% | Start% | Cameo% | News Risk | xP (Range) |"
    )
    lines.append(
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |"
    )

    for p in squad_result.starting_xi:
        role = ""
        if p.player.id == squad_result.captain.player.id:
            role = " **(C)**"
        elif p.player.id == squad_result.vice_captain.player.id:
            role = " **(V)**"
        star = " ★" if p.player.in_dreamteam else ""
        xp_disp = (
            p.expected_points * 2
            if p.player.id == squad_result.captain.player.id
            else p.expected_points
        )
        fl_disp = (
            p.floor_points * 2
            if p.player.id == squad_result.captain.player.id
            else p.floor_points
        )
        cl_disp = (
            p.ceiling_points * 2
            if p.player.id == squad_result.captain.player.id
            else p.ceiling_points
        )
        lines.append(
            f"| **{p.player.position}** | **{p.player.web_name}**{star}{role} | {p.player.team_short} | "
            f"`{p.player.next_opponent}` | {p.player.historical_ppg_3yr:.1f} | "
            f"{int(p.clean_sheet_prob * 100)}% | {int(p.concede_2plus_prob * 100)}% | "
            f"{int(p.news.start_probability * 100)}% | {int(p.news.cameo_probability * 100)}% | "
            f"`{p.news.risk_level}` | **{xp_disp:.1f}** *({fl_disp}–{cl_disp})* |"
        )

    lines.append("\n## 2. Recommended Bench Order\n")
    lines.append(
        "| Slot | Pos | Player | Club | Fixture | Start% | Cameo% | News / Press Conference Status | xP |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: |")

    bg = squad_result.bench_gkp
    lines.append(
        f"| **GKP** | GKP | **{bg.player.web_name}** | {bg.player.team_short} | `{bg.player.next_opponent}` | "
        f"{int(bg.news.start_probability * 100)}% | {int(bg.news.cameo_probability * 100)}% | "
        f"{bg.news.reasoning} | **{bg.expected_points:.1f}** |"
    )
    for idx, bp in enumerate(squad_result.bench_outfield, start=1):
        lines.append(
            f"| **{idx}. Sub** | {bp.player.position} | **{bp.player.web_name}** | {bp.player.team_short} | "
            f"`{bp.player.next_opponent}` | {int(bp.news.start_probability * 100)}% | "
            f"{int(bp.news.cameo_probability * 100)}% | {bp.news.reasoning} | **{bp.expected_points:.1f}** |"
        )

    lines.append("\n## 3. News & Tactical Alerts\n")
    for note in squad_result.tactical_notes:
        lines.append(f"- {note}")
    for p in all_squad_preds:
        for hl in p.news.matched_headlines:
            lines.append(f"- **{p.player.web_name} News:** {hl}")

    lines.append(
        f"\n## 4. Top 1-Transfer Upgrades (FT: {free_transfers} | Bank: £{bank_millions:.1f}m)\n"
    )
    if free_transfers == 0:
        lines.append(
            "> **Note:** You have `0` Free Transfers (`-4 pt` hit required). Only make a transfer if net gain exceeds `+4.0 xP`.\n"
        )
    if transfers:
        lines.append("| Out | In | Club & Fixture | Cost Diff | Starting XI Gain | Hit-Adjusted Gain |")
        lines.append("| :--- | :--- | :--- | :---: | :---: | :---: |")
        hit_cost = 0.0 if free_transfers >= 1 else 4.0
        for tr in transfers:
            net_after_hit = tr.xp_gain_this_week - hit_cost
            lines.append(
                f"| {tr.player_out.player.web_name} (£{tr.player_out.player.price:.1f}m) "
                f"| **{tr.player_in.player.web_name}** (£{tr.player_in.player.price:.1f}m) "
                f"| {tr.player_in.player.team_short} `{tr.player_in.player.next_opponent}` "
                f"| {tr.net_cost:+.1f}m | **+{tr.xp_gain_this_week:.1f} xP** "
                f"| `{net_after_hit:+.1f} xP` |"
            )

    if plan_view is not None:
        lines.append(
            f"\n## 4b. Multi-Gameweek Transfer Plan (exact MILP, FT banked to 5, -4 per extra transfer)\n"
        )
        lines.append("| GW | Free transfers | Out | In | Hit | Captain | XI xP |")
        lines.append("| :---: | :---: | :--- | :--- | :---: | :--- | :---: |")
        for w in plan_view.plan.weeks:
            outs = ", ".join(plan_view.by_id[i].player.web_name for i in w.transfers_out) or "—"
            ins = ", ".join(
                f"**{plan_view.by_id[i].player.web_name}** (£{plan_view.by_id[i].player.price:.1f}m)"
                for i in w.transfers_in
            ) or "—"
            lines.append(
                f"| {w.gw} | {w.free_transfers_before} | {outs} | {ins} | {('-' + str(4 * w.hits)) if w.hits else '0'} "
                f"| {plan_view.by_id[w.captain_id].player.web_name} | {w.expected_points:.1f} |"
            )
        lines.append(
            f"\n> Plan value (xP minus hits over {len(plan_view.plan.weeks)} GWs): **{plan_view.plan.objective:.1f}** | "
            f"hits taken: {plan_view.plan.total_hits}. Selling prices are assumed equal to current prices."
        )

    lines.append(
        f"\n## 5. Unlimited Free Hit / Dream XI Comparison (`{dream_result.formation}` — {dream_result.total_expected_points:.1f} xP)\n"
    )
    lines.append("| Pos | Player | Club | Price | Fixture | 3Yr PPG | xP |")
    lines.append("| :--- | :--- | :--- | :---: | :--- | :---: | :---: |")
    for dp in dream_result.starting_xi:
        role = ""
        if dp.player.id == dream_result.captain.player.id:
            role = " **(C)**"
        elif dp.player.id == dream_result.vice_captain.player.id:
            role = " **(V)**"
        xp_val = (
            dp.expected_points * 2
            if dp.player.id == dream_result.captain.player.id
            else dp.expected_points
        )
        lines.append(
            f"| **{dp.player.position}** | **{dp.player.web_name}**{role} | {dp.player.team_short} | "
            f"£{dp.player.price:.1f}m | `{dp.player.next_opponent}` | "
            f"{dp.player.historical_ppg_3yr:.1f} | **{xp_val:.1f}** |"
        )

    return "\n".join(lines) + "\n"


def cmd_sync(args: argparse.Namespace) -> None:
    print("1/2 Syncing FPL API & 3-Year Historical Dataset...")
    fpl_status = sync_live_fpl_data(fetch_player_histories=args.deep)
    print(f"    FPL Live Synced: {fpl_status['live_fpl_synced']}")
    print(f"    3-Year History Synced: {fpl_status['historical_synced']}")
    if fpl_status["errors"]:
        for err in fpl_status["errors"]:
            print(f"    [Note] {err} (Using built-in 2026/27 + 3yr database)")

    print("2/2 Scraping Sporting Life, BBC Sport, Guardian & Sky Sports News...")
    news_status = scrape_sports_websites()
    print(f"    Articles fetched: {news_status['articles_fetched']}")
    if news_status["errors"]:
        for err in news_status["errors"]:
            print(f"    [Note] {err}")


def cmd_analyze(args: argparse.Namespace) -> None:
    all_players = load_players()
    articles = load_all_news(SQUAD_CONFIG_PATH)
    squad_cfg = json.loads(SQUAD_CONFIG_PATH.read_text(encoding="utf-8"))

    squad_players = _match_squad_players(all_players, squad_cfg)
    squad_preds = [predict_player(p, articles) for p in squad_players]
    all_preds = [predict_player(p, articles) for p in all_players]

    opt_squad = optimize_starting_xi(squad_preds)
    dream_xi = build_dream_xi(all_preds)
    free_transfers = int(squad_cfg.get("free_transfers", 0))
    bank_millions = float(squad_cfg.get("bank_millions", 0.5))
    transfers = recommend_single_transfers(
        squad_preds, all_preds, bank_millions=bank_millions
    )
    plan_view = None
    if len(squad_preds) == 15:
        plan_view = build_transfer_plan(
            squad_preds, all_preds, bank_millions, free_transfers, horizon=4
        )

    report_md = format_markdown_report(
        opt_squad, dream_xi, squad_preds, transfers, free_transfers, bank_millions, plan_view
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    my_pitch_path = render_pitch_figure(
        opt_squad,
        total_cost=sum(p.player.price for p in squad_preds),
        budget=sum(p.player.price for p in squad_preds) + bank_millions,
        output_path=DATA_DIR / "my_squad_pitch.png",
        title="FPL AI Optimizer — Your Optimal Starting XI",
    )

    print(report_md)
    print(f"[Saved full report to {REPORT_PATH}]")
    print(f"[Saved pitch figure to {my_pitch_path}]")


def cmd_dream_team(args: argparse.Namespace) -> None:
    all_players = load_players()
    articles = load_all_news(SQUAD_CONFIG_PATH)
    all_preds = [predict_player(p, articles) for p in all_players]
    dream = build_dream_xi(all_preds)

    print(
        f"=== ULTIMATE FREE HIT / DREAM XI ({dream.formation}) | "
        f"Projected: {dream.total_expected_points:.1f} pts (Range: {dream.total_floor}-{dream.total_ceiling}) ==="
    )
    for p in dream.starting_xi:
        tag = (
            "(C)"
            if p.player.id == dream.captain.player.id
            else ("(V)" if p.player.id == dream.vice_captain.player.id else "   ")
        )
        pts = (
            p.expected_points * 2
            if p.player.id == dream.captain.player.id
            else p.expected_points
        )
        print(
            f"  {p.player.position:3s} | {p.player.web_name:12s} {tag} | {p.player.team_short} vs {p.player.next_opponent:7s} "
            f"| £{p.player.price:4.1f}m | 3Yr PPG: {p.player.historical_ppg_3yr:4.1f} | xP: {pts:4.1f}"
        )


def cmd_best_squad(args: argparse.Namespace) -> None:
    all_players = load_players()
    articles = load_all_news(SQUAD_CONFIG_PATH)
    all_preds = [predict_player(p, articles) for p in all_players]
    best_res, total_cost = get_best_squad_of_the_week(
        all_preds, budget=args.budget, max_per_club=3
    )

    print(
        f"# Best 15-Player FPL Squad of the Week (£{total_cost:.1f}m / £{args.budget:.1f}m Budget)\n"
    )
    print(
        f"- **Formation:** `{best_res.formation}`\n"
        f"- **Captain (2x):** **{best_res.captain.player.web_name}** ({best_res.captain.player.next_opponent})\n"
        f"- **Vice-Captain:** **{best_res.vice_captain.player.web_name}** ({best_res.vice_captain.player.next_opponent})\n"
        f"- **Projected Score:** **{best_res.total_expected_points:.1f} pts** "
        f"*(Floor: {best_res.total_floor} | Ceiling: {best_res.total_ceiling})*\n"
    )
    print("## Starting XI\n")
    print("| Pos | Player | Club | Price | Fixture | CS% | 3Yr PPG | xP |")
    print("| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |")
    for p in best_res.starting_xi:
        role = (
            " **(C)**"
            if p.player.id == best_res.captain.player.id
            else (" **(V)**" if p.player.id == best_res.vice_captain.player.id else "")
        )
        pts = (
            p.expected_points * 2
            if p.player.id == best_res.captain.player.id
            else p.expected_points
        )
        print(
            f"| **{p.player.position}** | **{p.player.web_name}**{role} | {p.player.team_short} | "
            f"£{p.player.price:.1f}m | `{p.player.next_opponent}` | "
            f"{int(p.clean_sheet_prob * 100)}% | {p.player.historical_ppg_3yr:.1f} | **{pts:.1f}** |"
        )

    print("\n## Substitutes (4)\n")
    print("| Slot | Pos | Player | Club | Price | Fixture | xP |")
    print("| :--- | :--- | :--- | :--- | :---: | :--- | :---: |")
    bg = best_res.bench_gkp
    print(
        f"| **GKP** | GKP | **{bg.player.web_name}** | {bg.player.team_short} | "
        f"£{bg.player.price:.1f}m | `{bg.player.next_opponent}` | **{bg.expected_points:.1f}** |"
    )
    for idx, bp in enumerate(best_res.bench_outfield, start=1):
        print(
            f"| **{idx}. Sub** | {bp.player.position} | **{bp.player.web_name}** | {bp.player.team_short} | "
            f"£{bp.player.price:.1f}m | `{bp.player.next_opponent}` | **{bp.expected_points:.1f}** |"
        )

    best_pitch_path = render_pitch_figure(
        best_res,
        total_cost=total_cost,
        budget=args.budget,
        output_path=DATA_DIR / "best_squad_pitch.png",
        title="FPL AI Optimizer — Best £100m Squad of the Week",
    )
    print(f"\n[Saved formation figure to {best_pitch_path}]")


def cmd_plan(args: argparse.Namespace) -> None:
    """Multi-gameweek transfer plan for the squad in my_squad.json (exact MILP)."""
    all_players = load_players()
    articles = load_all_news(SQUAD_CONFIG_PATH)
    squad_cfg = json.loads(SQUAD_CONFIG_PATH.read_text(encoding="utf-8"))
    squad_players = _match_squad_players(all_players, squad_cfg)
    if len(squad_players) != 15:
        raise SystemExit(f"my_squad.json matched {len(squad_players)} players, need 15")
    all_preds = [predict_player(p, articles) for p in all_players]
    by_id = {p.player.id: p for p in all_preds}
    squad_preds = [by_id[p.id] for p in squad_players]
    free_transfers = int(squad_cfg.get("free_transfers", 1))
    bank = float(squad_cfg.get("bank_millions", 0.0))
    view = build_transfer_plan(
        squad_preds, all_preds, bank, free_transfers, horizon=args.horizon, wildcard_now=args.wildcard
    )
    if view is None:
        raise SystemExit("Transfer plan infeasible (check budget / club limits in my_squad.json)")
    print(f"# Transfer plan from GW {view.target_gw} ({args.horizon} gameweeks, FT={free_transfers}, bank £{bank:.1f}m)\n")
    for line in view.describe():
        print("- " + line)
    w0 = view.plan.weeks[0]
    print("\nThis week's XI:")
    for pid in w0.lineup_ids:
        p = view.by_id[pid]
        tag = " (C)" if pid == w0.captain_id else (" (V)" if pid == w0.vice_id else "")
        print(f"  {p.player.position} {p.player.web_name}{tag} {p.player.team_short} {p.player.next_opponent} xP {p.expected_points:.1f}")
    print("Bench order: " + ", ".join(view.by_id[i].player.web_name for i in w0.bench_order))


def cmd_add_news(args: argparse.Namespace) -> None:
    cfg = json.loads(SQUAD_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg.setdefault("custom_news", []).append(
        {
            "player": args.player,
            "source": args.source,
            "headline": args.headline,
            "text": args.text,
        }
    )
    SQUAD_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Added press conference / news note for {args.player} to {SQUAD_CONFIG_PATH.name}!")


def _print_model_metrics(net) -> None:
    m = net.metrics
    h = m.get("holdout", {})
    print(
        f"    {m.get('architecture')} | {m.get('real_gw_samples')} player-gameweek rows through GW {m.get('latest_trained_gw')}\n"
        f"    Holdout {m.get('holdout_gws')}: MAE vs actual points {h.get('mae_points')} "
        f"(naive baseline {h.get('mae_naive_baseline')}, constant {h.get('mae_constant_baseline')}) | "
        f"corr {h.get('corr_points')} (naive {h.get('corr_naive_baseline')})\n"
        f"    Minutes log-loss {h.get('minutes_logloss')} (base rate {h.get('minutes_logloss_baseline')}) | "
        f"haul Brier {h.get('haul_brier')} (base rate {h.get('haul_brier_baseline')})"
    )


def cmd_train_nn(args: argparse.Namespace) -> None:
    print("Training multi-head FPL model (minutes | points given 60+ | haul) on real gameweek outcomes...")
    net = get_or_train_neural_net(force_retrain=True, verbose=True)
    _print_model_metrics(net)
    print(json.dumps(net.metrics, indent=2))


def cmd_weekly_update(args: argparse.Namespace) -> None:
    """
    End-to-end weekly pipeline:
      1. Syncs live FPL API + appends newly finished GW match logs (`element-summary`)
      2. Scrapes latest Premier League injury & press conference news
      3. Retrains the multi-head model (minutes | points | haul) with a time-ordered holdout
      4. Runs the AI Tactical Coach, the MILP squad/transfer optimizer & rebuilds `index.html`
      5. Optionally commits & pushes to GitHub Pages (`--push`)
    """
    import subprocess

    print("=== Step 1/4: Syncing Live FPL API & Appending Last Week's Match Logs ===")
    fpl_status = sync_live_fpl_data(fetch_player_histories=True)
    print(
        f"    FPL Synced: {fpl_status['live_fpl_synced']} | "
        f"Latest Finished GW: {fpl_status.get('latest_finished_gw')} | "
        f"Player Histories Cached: {fpl_status['summaries_synced']}"
    )

    print("=== Step 2/4: Scraping Premier League Team & Injury News ===")
    news_status = scrape_sports_websites()
    print(f"    Articles fetched: {news_status['articles_fetched']}")

    print("=== Step 3/4: Retraining multi-head model (time-ordered holdout) ===")
    net = get_or_train_neural_net(force_retrain=True)
    _print_model_metrics(net)

    print("=== Step 4/4: Running AI Tactical Head Coach & Rebuilding Web App (index.html) ===")
    from web_app import build_static_html

    build_static_html()
    cmd_analyze(args)
    print("=== Weekly Update Complete! ===")

    if getattr(args, "push", False):
        print("=== Committing & Pushing Updated Page to GitHub Pages ===")
        subprocess.run(
            [
                "git",
                "add",
                "index.html",
                "data/nn_weights.json",
                "data/ai_coach_decisions.json",
                "data/latest_report.md",
                "data/my_squad_pitch.png",
            ],
            check=False,
        )
        gw_num = net.metrics.get("latest_trained_gw", 9) + 1
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"Weekly FPL AI Update for GW {gw_num} (multi-head model + MILP planner)",
            ],
            check=False,
        )
        subprocess.run(["git", "push"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FPL Multi-Season Data + News Scraper & Squad Optimizer"
    )
    sub = parser.add_subparsers(dest="command")

    p_sync = sub.add_parser("sync", help="Fetch live FPL API, player match logs, and sports news")
    p_sync.add_argument("--deep", action="store_true", help="Fetch per-player element-summary")
    p_sync.set_defaults(func=cmd_sync)

    p_wk = sub.add_parser(
        "weekly-update",
        help="Full weekly refresh: pull last week's GW data, retrain the model, run the AI Coach, and rebuild index.html",
    )
    p_wk.add_argument(
        "--push",
        action="store_true",
        help="Automatically git commit and push the updated index.html and weights to GitHub Pages",
    )
    p_wk.set_defaults(func=cmd_weekly_update)

    p_an = sub.add_parser("analyze", help="Analyze your squad in my_squad.json")
    p_an.set_defaults(func=cmd_analyze)

    p_dt = sub.add_parser("dream-team", help="Build the optimal Free Hit Dream XI")
    p_dt.set_defaults(func=cmd_dream_team)

    p_bs = sub.add_parser(
        "best-squad",
        help="Build the best full 15-player FPL squad of the week under £100.0m budget",
    )
    p_bs.add_argument(
        "--budget", type=float, default=100.0, help="Total 15-man budget in £m (default: 100.0)"
    )
    p_bs.set_defaults(func=cmd_best_squad)

    p_plan = sub.add_parser("plan", help="Multi-gameweek MILP transfer plan for my_squad.json")
    p_plan.add_argument("--horizon", type=int, default=4, help="Gameweeks to plan (2-5, default 4)")
    p_plan.add_argument("--wildcard", action="store_true", help="Treat this week as a wildcard (unlimited transfers)")
    p_plan.set_defaults(func=cmd_plan)

    p_news = sub.add_parser("add-news", help="Add a custom manager quote or injury update")
    p_news.add_argument("player", help="Player web_name, e.g. \"O'Reilly\"")
    p_news.add_argument("headline", help="Short headline")
    p_news.add_argument("text", help="Full press conference quote or news snippet")
    p_news.add_argument("--source", default="Press Conference / Sporting Life", help="Source name")
    p_news.set_defaults(func=cmd_add_news)

    p_nn = sub.add_parser(
        "train-nn",
        help="Retrain the multi-head model and print the time-ordered holdout report",
    )
    p_nn.add_argument(
        "--fetch-real",
        action="store_true",
        help="Download real merged_gw.csv rows from multi-season GitHub archive first",
    )
    p_nn.set_defaults(func=cmd_train_nn)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        cmd_analyze(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
