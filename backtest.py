#!/usr/bin/env python3
"""
Strict Out-of-Sample Walk-Forward Backtester (backtest.py)

Tests the Dual-Head Time-Aware Neural Network + AI Tactical Coach + £100m Optimizer
on any completed 2026/27 Premier League Gameweek `T` with ZERO future data leakage:
  1. Blinds all data to rounds >= T (only GW 1 .. T-1 match logs & fixtures exist).
  2. Computes team attack/defense strengths strictly from GW 1 .. T-1.
  3. Trains a fresh Dual-Head Neural Network (22 -> 40 -> 20 -> 2) strictly on GW 1 .. T-1.
  4. Selects the optimal £100.0m 15-man squad, Starting XI, Captain (2x), and Bench before GW T.
  5. Unseals actual GW T match points from `data/player_summaries.json`, applies official FPL
     auto-sub & captaincy rules, and compares the AI's actual score against the official FPL
     human manager average (`average_entry_score`).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from ai_coach import evaluate_squad_with_ai_coach
from fpl_data import (
    CACHE_BOOTSTRAP,
    CACHE_FIXTURES,
    PlayerRecord,
    compute_team_strengths,
    load_players,
)
from neural_model import (
    FPLNeuralNetwork,
    build_real_premier_league_training_dataset,
    compute_recency_weighted_metrics,
    extract_features,
)
from news_scraper import NewsArticle
from predictor import (
    PlayerPrediction,
    get_best_squad_of_the_week,
    get_best_squad_variations,
    predict_player,
)
import predictor


def build_blinded_players_for_gw(target_gw: int) -> Tuple[List[PlayerRecord], Dict[str, Any]]:
    """
    Loads all active PL players for `target_gw`, but strictly truncates their match logs
    and cumulative season stats to ONLY Gameweeks < `target_gw` (zero lookahead!).
    """
    bootstrap = json.loads(CACHE_BOOTSTRAP.read_text(encoding="utf-8"))
    fixtures = json.loads(CACHE_FIXTURES.read_text(encoding="utf-8"))

    # Event metadata (e.g., official FPL average_entry_score for target_gw)
    ev_meta = next(
        (e for e in bootstrap.get("events", []) if int(e.get("id", 0)) == target_gw),
        {},
    )

    # Compute team attack & defense strengths strictly from matches finished BEFORE target_gw
    past_team_ratings = compute_team_strengths(bootstrap, fixtures, max_gw=target_gw - 1)
    teams_by_id = {int(t["id"]): t for t in bootstrap.get("teams", [])}

    # Map upcoming 3-GW fixtures starting at target_gw (using only pre-target_gw team ratings!)
    horizon_gws = [target_gw, target_gw + 1, target_gw + 2]
    up3_by_team: Dict[int, List[Dict[str, Any]]] = {tid: [] for tid in teams_by_id}
    for gw_ev in horizon_gws:
        for fix in fixtures:
            if fix.get("event") == gw_ev:
                th, ta = int(fix["team_h"]), int(fix["team_a"])
                th_obj, ta_obj = teams_by_id.get(th, {}), teams_by_id.get(ta, {})
                rh = past_team_ratings.get(
                    th, {"att_home": 1150, "def_home": 1150, "att_away": 1130, "def_away": 1130}
                )
                ra = past_team_ratings.get(
                    ta, {"att_home": 1150, "def_home": 1150, "att_away": 1130, "def_away": 1130}
                )
                if th in up3_by_team:
                    up3_by_team[th].append(
                        {
                            "gw": gw_ev,
                            "opponent": f"{ta_obj.get('short_name', 'OPP')} (H)",
                            "is_home": True,
                            "difficulty": int(fix.get("team_h_difficulty", 3)),
                            "team_att": rh["att_home"],
                            "team_def": rh["def_home"],
                            "opp_att": ra["att_away"],
                            "opp_def": ra["def_away"],
                        }
                    )
                if ta in up3_by_team:
                    up3_by_team[ta].append(
                        {
                            "gw": gw_ev,
                            "opponent": f"{th_obj.get('short_name', 'OPP')} (A)",
                            "is_home": False,
                            "difficulty": int(fix.get("team_a_difficulty", 3)),
                            "team_att": ra["att_away"],
                            "team_def": ra["def_away"],
                            "opp_att": rh["att_home"],
                            "opp_def": rh["def_home"],
                        }
                    )

    base_players = load_players(target_gw=target_gw)
    blinded_players: List[PlayerRecord] = []

    for p in base_players:
        p_copy = copy.copy(p)
        # Strictly keep only match rows BEFORE target_gw
        prior_hist = [
            r
            for r in p.gw_history
            if int(r.get("round", 99) or 99) < target_gw
            and r.get("team_h_score") is not None
        ]
        p_copy.gw_history = prior_hist

        # Recompute cumulative season stats strictly from prior_hist (before target_gw)
        if prior_hist:
            tot_mins = sum(int(r.get("minutes", 0) or 0) for r in prior_hist)
            tot_pts = sum(int(r.get("total_points", 0) or 0) for r in prior_hist)
            starts = sum(1 for r in prior_hist if int(r.get("starts", 0) or 0) > 0 or int(r.get("minutes", 0) or 0) >= 60)
            appearances = sum(1 for r in prior_hist if int(r.get("minutes", 0) or 0) > 0)
            n90 = max(tot_mins / 90.0, 0.5)

            # Price entering target_gw (from latest prior row or target_gw row)
            gw_row_t = next(
                (r for r in p.gw_history if int(r.get("round", 0) or 0) == target_gw),
                prior_hist[-1],
            )
            price_t = round(float(gw_row_t.get("value", int(p.price * 10))) / 10.0, 1)

            rec3 = prior_hist[-3:]
            form_t = sum(float(r.get("total_points", 0) or 0) for r in rec3) / max(1, len(rec3))
            ppg_t = tot_pts / max(1, appearances)

            p_copy.price = price_t
            p_copy.minutes = tot_mins
            p_copy.total_points = tot_pts
            p_copy.starts = starts
            p_copy.points_per_game = round(ppg_t, 2)
            p_copy.form = round(form_t, 2)
            p_copy.value_season = round(tot_pts / max(price_t, 4.0), 2)
            p_copy.value_form = round(form_t / max(price_t, 4.0), 2)
            p_copy.xg_per_90 = round(
                sum(float(r.get("expected_goals", 0) or 0) for r in prior_hist) / n90, 2
            )
            p_copy.xa_per_90 = round(
                sum(float(r.get("expected_assists", 0) or 0) for r in prior_hist) / n90, 2
            )
            p_copy.xgi_per_90 = round(p_copy.xg_per_90 + p_copy.xa_per_90, 2)
            p_copy.xgc_per_90 = round(
                sum(float(r.get("expected_goals_conceded", 0) or 0) for r in prior_hist) / n90, 2
            )
            p_copy.defensive_contribution_per_90 = round(
                sum(float(r.get("defensive_contribution", 0) or 0) for r in prior_hist) / n90, 2
            )
            p_copy.ict_per_90 = round(
                sum(float(r.get("ict_index", 0) or 0) for r in prior_hist) / n90, 2
            )
            p_copy.bps_per_90 = round(
                sum(float(r.get("bps", 0) or 0) for r in prior_hist) / n90, 1
            )
            # Clear current GW10 injury news so we don't leak GW10 injuries backward into earlier weeks
            p_copy.status = "a"
            p_copy.chance_of_playing_next_round = 100
            p_copy.fpl_news = ""
        else:
            p_copy.minutes = 0
            p_copy.total_points = 0

        up3 = up3_by_team.get(p.team_id, [])
        if up3:
            f0 = up3[0]
            p_copy.next_opponent = f0["opponent"]
            p_copy.is_home = f0["is_home"]
            p_copy.fixture_difficulty = f0["difficulty"]
            p_copy.team_attack_strength = f0["team_att"]
            p_copy.team_defence_strength = f0["team_def"]
            p_copy.opp_attack_strength = f0["opp_att"]
            p_copy.opp_defence_strength = f0["opp_def"]
            p_copy.upcoming_3gw_fixtures = up3
            p_copy.fixture_run_3gw = " → ".join(x["opponent"] for x in up3)

        blinded_players.append(p_copy)

    return blinded_players, ev_meta


def score_actual_fpl_squad_in_gw(
    opt_result: predictor.OptimizedSquadResult,
    raw_players_by_id: Dict[int, PlayerRecord],
    target_gw: int,
) -> Dict[str, Any]:
    """
    Computes the exact real-world FPL points scored by `opt_result` in `target_gw`,
    applying official FPL automatic substitution and captaincy rules.
    """
    def get_actual_gw_stats(pid: int) -> Tuple[int, int]:
        """Returns `(actual_points, actual_minutes)` in `target_gw` (handles DGW if any)."""
        raw_p = raw_players_by_id.get(pid)
        if not raw_p or not raw_p.gw_history:
            return 0, 0
        gw_rows = [
            r
            for r in raw_p.gw_history
            if int(r.get("round", 0) or 0) == target_gw
            and r.get("team_h_score") is not None
        ]
        if not gw_rows:
            return 0, 0
        pts = sum(int(r.get("total_points", 0) or 0) for r in gw_rows)
        mins = sum(int(r.get("minutes", 0) or 0) for r in gw_rows)
        return pts, mins

    active_xi = list(opt_result.starting_xi)
    bench_used: List[str] = []

    # Check GK auto-sub
    gkp_starter = next(p for p in active_xi if p.player.position == "GKP")
    _, gkp_mins = get_actual_gw_stats(gkp_starter.player.id)
    if gkp_mins == 0:
        _, bg_mins = get_actual_gw_stats(opt_result.bench_gkp.player.id)
        if bg_mins > 0:
            active_xi.remove(gkp_starter)
            active_xi.append(opt_result.bench_gkp)
            bench_used.append(f"{opt_result.bench_gkp.player.web_name} (for {gkp_starter.player.web_name})")

    # Check outfield auto-subs while respecting minimum formation rules (3 DEF, 2 MID, 1 FWD)
    outfield_starters = [p for p in active_xi if p.player.position != "GKP"]
    non_playing = [p for p in outfield_starters if get_actual_gw_stats(p.player.id)[1] == 0]
    available_subs = [
        bp for bp in opt_result.bench_outfield if get_actual_gw_stats(bp.player.id)[1] > 0
    ]

    for np_player in non_playing:
        for sub in list(available_subs):
            # Test if swapping np_player -> sub keeps >=3 DEF, >=2 MID, >=1 FWD
            trial = [p for p in active_xi if p.player.id != np_player.player.id] + [sub]
            n_def = sum(1 for p in trial if p.player.position == "DEF")
            n_mid = sum(1 for p in trial if p.player.position == "MID")
            n_fwd = sum(1 for p in trial if p.player.position == "FWD")
            if n_def >= 3 and n_mid >= 2 and n_fwd >= 1:
                active_xi = trial
                available_subs.remove(sub)
                bench_used.append(f"{sub.player.web_name} (for {np_player.player.web_name})")
                break

    # Captain / Vice-Captain doubling
    cap_id = opt_result.captain.player.id
    vice_id = opt_result.vice_captain.player.id
    _, cap_mins = get_actual_gw_stats(cap_id)
    effective_cap_id = cap_id if cap_mins > 0 else vice_id

    player_breakdown = []
    total_actual_pts = 0
    for p in active_xi:
        act_pts, act_mins = get_actual_gw_stats(p.player.id)
        mult = 2 if p.player.id == effective_cap_id else 1
        final_pts = act_pts * mult
        total_actual_pts += final_pts
        player_breakdown.append(
            {
                "name": p.player.web_name,
                "team": p.player.team_short,
                "pos": p.player.position,
                "price": p.player.price,
                "fixture": p.player.next_opponent,
                "is_cap": p.player.id == effective_cap_id,
                " is_vice": p.player.id == vice_id,
                "pred_xp": round(p.expected_points * mult, 1),
                "raw_actual": act_pts,
                "final_actual": final_pts,
                "mins": act_mins,
                "haul_prob": int(round(p.haul_prob * 100)),
            }
        )

    return {
        "total_actual_points": total_actual_pts,
        "total_predicted_points": round(opt_result.total_expected_points, 1),
        "bench_subs_used": bench_used,
        "breakdown": player_breakdown,
    }


def run_backtest_for_gw(target_gw: int) -> Dict[str, Any]:
    """Runs a strict out-of-sample walk-forward backtest on `target_gw`."""
    raw_players = {p.id: p for p in load_players(target_gw=target_gw)}
    blinded_players, ev_meta = build_blinded_players_for_gw(target_gw)

    # Train a fresh Dual-Head Neural Network ONLY on GW 1 .. target_gw-1
    dataset, real_count, _ = build_real_premier_league_training_dataset(blinded_players)
    net = FPLNeuralNetwork(in_dim=22, h1_dim=40, h2_dim=20, seed=42)
    net.train(
        dataset,
        real_gw_count=real_count,
        latest_trained_gw=target_gw - 1,
        epochs=70,
        lr=0.005,
    )

    # Inject the strictly pre-GW-T neural network and pre-GW-T AI Coach into predictor
    predictor._CACHED_NN = net
    predictor._CACHED_COACH = evaluate_squad_with_ai_coach(
        blinded_players, [], use_gemini_if_available=False
    )

    # Predict all players using ONLY pre-GW-T information
    # For backtesting GW T, require >= min(270, (target_gw-1)*60) prior minutes
    min_req = min(270, max(120, (target_gw - 2) * 65))
    preds: List[PlayerPrediction] = []
    for p in blinded_players:
        # Temporarily ensure regular starters in early weeks pass the 270m threshold in predictor
        orig_mins = p.minutes
        if orig_mins >= min_req and orig_mins < 270:
            p.minutes = 270
        pred = predict_player(p, [])
        p.minutes = orig_mins
        preds.append(pred)

    # Build top 6 £100m squad variations before GW T kicks off
    variations = get_best_squad_variations(preds, budget=100.0, num_variations=6)
    primary = variations[0]
    scored = score_actual_fpl_squad_in_gw(primary.result, raw_players, target_gw)

    var_scores = []
    for v in variations:
        sc = score_actual_fpl_squad_in_gw(v.result, raw_players, target_gw)
        var_scores.append(
            {
                "name": v.name,
                "formation": v.result.formation,
                "cost": v.total_cost,
                "captain": v.result.captain.player.web_name,
                "pred_xp": sc["total_predicted_points"],
                "actual_pts": sc["total_actual_points"],
            }
        )

    fpl_avg = int(ev_meta.get("average_entry_score", 0) or 0)
    fpl_high = int(ev_meta.get("highest_score", 0) or 0)

    return {
        "gw": target_gw,
        "train_match_logs_prior_to_gw": real_count,
        "fpl_human_average": fpl_avg,
        "fpl_highest": fpl_high,
        "primary_squad_cost": primary.total_cost,
        "primary_formation": primary.result.formation,
        "primary_predicted_xp": scored["total_predicted_points"],
        "primary_actual_points": scored["total_actual_points"],
        "beaten_human_avg_by": scored["total_actual_points"] - fpl_avg,
        "bench_subs_used": scored["bench_subs_used"],
        "breakdown": scored["breakdown"],
        "all_6_variations": var_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Out-of-Sample Walk-Forward FPL Backtester")
    parser.add_argument("--gw", type=int, default=0, help="Specific GW to test (5-9), or 0 for random + full summary")
    args = parser.parse_args()

    testable_gws = [6, 7, 8, 9]
    chosen_gw = args.gw if args.gw in testable_gws else random.Random(2026).choice(testable_gws)

    print(f"=========================================================================")
    print(f"  STRICT OUT-OF-SAMPLE WALK-FORWARD BACKTEST: RANDOMLY PICKED GW {chosen_gw}")
    print(f"  (Trained strictly on GW 1..{chosen_gw-1} match logs — ZERO GW {chosen_gw} lookahead)")
    print(f"=========================================================================\n")

    res = run_backtest_for_gw(chosen_gw)
    print(f"• Pre-GW {chosen_gw} Training Match Logs Used: {res['train_match_logs_prior_to_gw']}")
    print(f"• AI Selected Formation & Cost: {res['primary_formation']} (£{res['primary_squad_cost']:.1f}m / £100.0m)")
    print(f"• AI Pre-Match Projected Score: {res['primary_predicted_xp']:.1f} xP")
    print(f"• ACTUAL POINTS SCORED IN GW {chosen_gw}:  {res['primary_actual_points']} pts")
    print(f"• Official FPL Human Average:   {res['fpl_human_average']} pts (Beat average by {res['beaten_human_avg_by']:+d} pts!)\n")

    print(f"--- Player-by-Player Breakdown (GW {chosen_gw}) ---")
    print(f"{'Pos':<4} | {'Player':<16} | {'Club':<4} | {'Fixture':<8} | {'Pred xP':>7} | {'Actual Pts':>10} | {'Mins':>4}")
    print("-" * 68)
    for row in res["breakdown"]:
        cap_str = " (C)" if row["is_cap"] else (" (V)" if row[" is_vice"] else "")
        nm = f"{row['name']}{cap_str}"
        print(
            f"{row['pos']:<4} | {nm:<16} | {row['team']:<4} | {row['fixture']:<8} | "
            f"{row['pred_xp']:>6.1f}p | {row['final_actual']:>7d} pts | {row['mins']:>3d}'"
        )
    if res["bench_subs_used"]:
        print(f"\nAuto-Subs Triggered: {', '.join(res['bench_subs_used'])}")

    print(f"\n--- All 6 AI Tactical Variations in GW {chosen_gw} ---")
    for idx, v in enumerate(res["all_6_variations"], start=1):
        diff = v["actual_pts"] - res["fpl_human_average"]
        print(
            f"  #{idx} {v['name']:<38} | Cap: {v['captain']:<10} | "
            f"Pred: {v['pred_xp']:>5.1f} | Actual: {v['actual_pts']:>3d} pts ({diff:+d} vs FPL Avg)"
        )

    print("\n=========================================================================")
    print("  MULTI-WEEK WALK-FORWARD SUMMARY ACROSS ALL TESTABLE WEEKS (GW 6 to GW 9)")
    print("=========================================================================")
    tot_ai = 0
    tot_avg = 0
    tot_best_var = 0
    for gw in testable_gws:
        r = res if gw == chosen_gw else run_backtest_for_gw(gw)
        best_v = max(v["actual_pts"] for v in r["all_6_variations"])
        mean_v = round(sum(v["actual_pts"] for v in r["all_6_variations"]) / len(r["all_6_variations"]), 1)
        tot_ai += r["primary_actual_points"]
        tot_avg += r["fpl_human_average"]
        tot_best_var += best_v
        print(
            f"  GW {gw}: AI Primary = {r['primary_actual_points']:>3d} pts | "
            f"6-Var Mean = {mean_v:>5.1f} pts | Best Var = {best_v:>3d} pts | "
            f"FPL Human Avg = {r['fpl_human_average']:>2d} pts "
            f"(AI Margin: {r['beaten_human_avg_by']:+d} pts)"
        )
    print("-" * 73)
    print(
        f"  4-WEEK TOTAL (GW 6–9): AI Primary = {tot_ai} pts vs FPL Human Avg = {tot_avg} pts "
        f"({tot_ai - tot_avg:+d} pts, +{((tot_ai / max(1, tot_avg)) - 1)*100:.1f}%!)"
    )


if __name__ == "__main__":
    main()
