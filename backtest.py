#!/usr/bin/env python3
"""
Strict walk-forward backtester (backtest.py)

For every completed gameweek T in the test range:
  1. Blind all data to gameweeks < T (match logs, prices, team strengths, availability).
  2. Train the multi-head network on those rows only (early-stopped on the last prior GW).
  3. Predict every player for GW T (and the following horizon).
  4. Run several STRATEGIES, score them with real GW T points, official auto-subs and captaincy:

       model_freehit   fresh £100m squad every week from model xP   (unlimited transfers)
       naive_freehit   fresh £100m squad every week from a naive xP (prior play-rate x points
                       per appearance) — same optimizer, so the gap is the model's value
       model_rolling   realistic manager: squad carried week to week, 1 free transfer/week
                       banked to 5, -4 per extra transfer, 3-GW MILP transfer plan
       naive_rolling   the same rolling manager using naive xP
       fpl_average     the official average human score (1 free transfer, no hindsight)

  5. Report per-week scores plus mean and standard deviation of the margins, and player-level
     prediction accuracy (MAE / correlation vs naive) on the weeks tested.

Results are also written to data/backtest_results.json for the web page.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import news_scraper
import predictor
from ai_coach import evaluate_squad_with_ai_coach
from fpl_data import CACHE_BOOTSTRAP, CACHE_FIXTURES, DATA_DIR, HORIZON_GWS, PlayerRecord, compute_team_strengths, load_players
from neural_model import (
    FPLNeuralNetwork,
    Sample,
    build_training_samples,
    train_with_time_holdout,
)
from optimizer import PlanPlayer, plan_transfers, select_candidate_pool, solve_squad
from predictor import PlayerPrediction, predict_player, to_plan_players

RESULTS_PATH = DATA_DIR / "backtest_results.json"
DEFAULT_RATING = {"att_home": 1150, "def_home": 1150, "att_away": 1130, "def_away": 1130}


# --------------------------------------------------------------------------------------
# Blinding
# --------------------------------------------------------------------------------------
def build_blinded_players_for_gw(target_gw: int) -> Tuple[List[PlayerRecord], Dict[str, Any]]:
    bootstrap = json.loads(CACHE_BOOTSTRAP.read_text(encoding="utf-8"))
    fixtures = json.loads(CACHE_FIXTURES.read_text(encoding="utf-8"))
    ev_meta = next((e for e in bootstrap.get("events", []) if int(e.get("id", 0)) == target_gw), {})
    past_ratings = compute_team_strengths(bootstrap, fixtures, max_gw=target_gw - 1)
    teams_by_id = {int(t["id"]): t for t in bootstrap.get("teams", [])}

    up_by_team: Dict[int, List[Dict[str, Any]]] = {tid: [] for tid in teams_by_id}
    for gw_ev in [target_gw + k for k in range(HORIZON_GWS)]:
        for fix in fixtures:
            if fix.get("event") != gw_ev:
                continue
            th, ta = int(fix["team_h"]), int(fix["team_a"])
            rh, ra = past_ratings.get(th, DEFAULT_RATING), past_ratings.get(ta, DEFAULT_RATING)
            if th in up_by_team:
                up_by_team[th].append({"gw": gw_ev, "opponent": f"{teams_by_id.get(ta, {}).get('short_name', 'OPP')} (H)", "is_home": True,
                                       "difficulty": int(fix.get("team_h_difficulty", 3)), "team_att": rh["att_home"], "team_def": rh["def_home"],
                                       "opp_att": ra["att_away"], "opp_def": ra["def_away"]})
            if ta in up_by_team:
                up_by_team[ta].append({"gw": gw_ev, "opponent": f"{teams_by_id.get(th, {}).get('short_name', 'OPP')} (A)", "is_home": False,
                                       "difficulty": int(fix.get("team_a_difficulty", 3)), "team_att": ra["att_away"], "team_def": ra["def_away"],
                                       "opp_att": rh["att_home"], "opp_def": rh["def_home"]})

    blinded: List[PlayerRecord] = []
    for p in load_players(target_gw=target_gw):
        q = copy.copy(p)
        prior = [r for r in p.gw_history if int(r.get("round", 99) or 99) < target_gw and r.get("team_h_score") is not None]
        q.gw_history = prior
        q.status, q.chance_of_playing_next_round, q.fpl_news = "a", None, ""
        if prior:
            tot_mins = sum(int(r.get("minutes", 0) or 0) for r in prior)
            tot_pts = sum(int(r.get("total_points", 0) or 0) for r in prior)
            apps = sum(1 for r in prior if int(r.get("minutes", 0) or 0) > 0)
            n90 = max(tot_mins / 90.0, 0.5)
            row_t = next((r for r in p.gw_history if int(r.get("round", 0) or 0) == target_gw), prior[-1])
            q.price = round(float(row_t.get("value", int(p.price * 10))) / 10.0, 1)
            q.minutes, q.total_points = tot_mins, tot_pts
            q.starts = sum(1 for r in prior if int(r.get("starts", 0) or 0) > 0)
            q.points_per_game = round(tot_pts / max(1, apps), 2)
            q.form = round(sum(float(r.get("total_points", 0) or 0) for r in prior[-3:]) / max(1, len(prior[-3:])), 2)
            q.value_season = round(tot_pts / max(q.price, 4.0), 2)
            q.xg_per_90 = round(sum(float(r.get("expected_goals", 0) or 0) for r in prior) / n90, 2)
            q.xa_per_90 = round(sum(float(r.get("expected_assists", 0) or 0) for r in prior) / n90, 2)
            q.xgi_per_90 = round(q.xg_per_90 + q.xa_per_90, 2)
            q.xgc_per_90 = round(sum(float(r.get("expected_goals_conceded", 0) or 0) for r in prior) / n90, 2)
            q.defensive_contribution_per_90 = round(sum(float(r.get("defensive_contribution", 0) or 0) for r in prior) / n90, 2)
            q.ict_per_90 = round(sum(float(r.get("ict_index", 0) or 0) for r in prior) / n90, 2)
            q.bps_per_90 = round(sum(float(r.get("bps", 0) or 0) for r in prior) / n90, 1)
        else:
            q.minutes, q.total_points, q.starts = 0, 0, 0
            q.points_per_game, q.form = 0.0, 0.0
        up = up_by_team.get(p.team_id, [])
        q.upcoming_3gw_fixtures = up
        q.fixture_run_3gw = " → ".join(x["opponent"] for x in up[:3])
        q.target_gw = target_gw
        if up and up[0]["gw"] == target_gw:
            f0 = up[0]
            q.next_opponent, q.is_home, q.fixture_difficulty = f0["opponent"], f0["is_home"], f0["difficulty"]
            q.team_attack_strength, q.team_defence_strength = f0["team_att"], f0["team_def"]
            q.opp_attack_strength, q.opp_defence_strength = f0["opp_att"], f0["opp_def"]
        else:
            q.next_opponent = "BLANK"
        blinded.append(q)
    return blinded, ev_meta


# --------------------------------------------------------------------------------------
# Real scoring with official auto-subs & captaincy
# --------------------------------------------------------------------------------------
def actual_gw_stats(raw_by_id: Dict[int, PlayerRecord], pid: int, gw: int) -> Tuple[int, int]:
    p = raw_by_id.get(pid)
    if not p:
        return 0, 0
    rows = [r for r in p.gw_history if int(r.get("round", 0) or 0) == gw and r.get("team_h_score") is not None]
    return sum(int(r.get("total_points", 0) or 0) for r in rows), sum(int(r.get("minutes", 0) or 0) for r in rows)


def score_week(
    lineup: List[int], bench_order: List[int], captain: int, vice: int,
    pos_of: Dict[int, str], raw_by_id: Dict[int, PlayerRecord], gw: int,
) -> Tuple[int, List[str]]:
    """Applies FPL auto-sub rules (GK for GK; outfield in bench order keeping >=3 DEF, >=2 MID, >=1 FWD)."""
    active = list(lineup)
    subs_used: List[str] = []
    gk = [pid for pid in active if pos_of.get(pid) == "GKP"]
    bench_gk = [pid for pid in bench_order if pos_of.get(pid) == "GKP"]
    if gk and actual_gw_stats(raw_by_id, gk[0], gw)[1] == 0 and bench_gk and actual_gw_stats(raw_by_id, bench_gk[0], gw)[1] > 0:
        active.remove(gk[0])
        active.append(bench_gk[0])
        subs_used.append("GK")
    outfield_bench = [pid for pid in bench_order if pos_of.get(pid) != "GKP"]
    for pid in [x for x in lineup if pos_of.get(x) != "GKP"]:
        if actual_gw_stats(raw_by_id, pid, gw)[1] > 0:
            continue
        for sub in list(outfield_bench):
            if actual_gw_stats(raw_by_id, sub, gw)[1] == 0:
                continue
            trial = [x for x in active if x != pid] + [sub]
            n = {k: sum(1 for x in trial if pos_of.get(x) == k) for k in ("DEF", "MID", "FWD")}
            if n["DEF"] >= 3 and n["MID"] >= 2 and n["FWD"] >= 1:
                active = trial
                outfield_bench.remove(sub)
                subs_used.append(f"{sub}->{pid}")
                break
    eff_cap = captain if actual_gw_stats(raw_by_id, captain, gw)[1] > 0 else vice
    total = 0
    for pid in active:
        pts, _ = actual_gw_stats(raw_by_id, pid, gw)
        total += pts * (2 if pid == eff_cap else 1)
    return total, subs_used


# --------------------------------------------------------------------------------------
# Per-gameweek model fitting and prediction
# --------------------------------------------------------------------------------------
def fit_model_for_gw(blinded: List[PlayerRecord], target_gw: int) -> Tuple[FPLNeuralNetwork, int]:
    samples = build_training_samples(blinded, max_gw=target_gw - 1)
    if not samples:
        raise RuntimeError(f"No training rows before GW {target_gw}")
    latest = max(s.gw for s in samples)
    if latest >= 4:
        net, _ = train_with_time_holdout(samples, holdout_gws=1)
    else:
        net = FPLNeuralNetwork()
        net.train(samples, epochs=15)
    return net, len(samples)


def naive_plan_players(preds: List[PlayerPrediction], gws: List[int]) -> List[PlanPlayer]:
    """Naive xP = prior play-rate x prior points per appearance, 0 on blank weeks."""
    out: List[PlanPlayer] = []
    for p in preds:
        rows = p.player.gw_history
        played = [r for r in rows if int(r.get("minutes", 0) or 0) > 0]
        rate = len(played) / len(rows) if rows else 0.0
        ppa = (sum(float(r.get("total_points", 0) or 0) for r in played) / len(played)) if played else 0.0
        fixtures_per_gw = {g: sum(1 for f in p.player.upcoming_3gw_fixtures if int(f.get("gw", 0)) == g) for g in gws}
        out.append(PlanPlayer(p.player.id, p.player.web_name, p.player.team_short, p.player.position, p.player.price, p.player.price,
                              {g: rate * ppa * fixtures_per_gw.get(g, 0) for g in gws}, p.player.selected_by_percent, 0.0))
    return out


def predict_all_for_gw(target_gw: int, use_coach: bool) -> Tuple[List[PlayerPrediction], Dict[str, Any], int]:
    blinded, ev_meta = build_blinded_players_for_gw(target_gw)
    net, n_rows = fit_model_for_gw(blinded, target_gw)
    predictor._CACHED_NN = net
    news_scraper.MINUTES_MODEL_OVERRIDE = net
    predictor._CACHED_COACH = evaluate_squad_with_ai_coach(blinded, [], use_gemini_if_available=False) if use_coach else {}
    preds = [predict_player(p, []) for p in blinded]
    return preds, ev_meta, n_rows


# --------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------
class RollingManager:
    """Carries a squad across gameweeks with FPL transfer rules and a MILP transfer plan."""

    def __init__(self, label: str, horizon: int = 3, friction: float = 1.5, log: bool = False):
        self.label = label
        self.horizon = horizon
        self.friction = friction
        self.log = log
        self.squad: Optional[List[int]] = None
        self.bank = 0.0
        self.free_transfers = 1
        self.hits_total = 0

    def step(self, plan_players: List[PlanPlayer], by_id: Dict[int, PlayerPrediction], gw: int) -> Tuple[List[int], List[int], int, int, int, float]:
        gws = [gw + k for k in range(self.horizon)]
        price_of = {pl.id: pl.price for pl in plan_players}
        if self.squad is None:
            pool = select_candidate_pool(plan_players, gws)
            week = solve_squad(pool, gw, budget=100.0, horizon_weights={g: 0.85**k for k, g in enumerate(gws)})
            assert week is not None
            self.squad = list(week.squad_ids)
            self.bank = round(100.0 - sum(price_of[i] for i in self.squad), 1)
            hits = 0
            lineup, bench, cap, vice, xp = week.lineup_ids, week.bench_order, week.captain_id, week.vice_id, week.expected_points
        else:
            # players who vanished from the pool keep their last price and 0 xP
            known = {pl.id for pl in plan_players}
            extra = [PlanPlayer(i, str(i), "???", by_id[i].player.position if i in by_id else "MID", 4.0, 4.0, {}, 0.0, 0.0)
                     for i in self.squad if i not in known]
            pool = select_candidate_pool(plan_players + extra, gws, keep_ids=self.squad)
            budget = round(sum(price_of.get(i, 4.0) for i in self.squad) + self.bank, 1)
            plan = plan_transfers(pool, self.squad, gws, budget=budget, free_transfers=self.free_transfers, time_limit=30.0,
                                  transfer_friction=self.friction)
            assert plan is not None
            w0 = plan.weeks[0]
            if self.log and (w0.transfers_in or w0.hits):
                names = {pl.id: pl.name for pl in pool}
                print(f"    [{self.label}] GW{gw} FT={self.free_transfers} bank={self.bank} OUT {[names.get(i, i) for i in w0.transfers_out]} "
                      f"IN {[names.get(i, i) for i in w0.transfers_in]} hits={w0.hits}")
            hits = w0.hits
            n_in = len(w0.transfers_in)
            self.hits_total += hits
            self.bank = round(budget - sum(price_of.get(i, 4.0) for i in w0.squad_ids), 1)
            self.free_transfers = max(1, min(5, self.free_transfers - n_in + hits + 1))
            self.squad = list(w0.squad_ids)
            lineup, bench, cap, vice, xp = w0.lineup_ids, w0.bench_order, w0.captain_id, w0.vice_id, w0.expected_points
        return lineup, bench, cap, vice, hits, xp


def run_backtest(gws: List[int], use_coach: bool = True, verbose: bool = True, friction: float = 1.5) -> Dict[str, Any]:
    raw_by_id = {p.id: p for p in load_players()}
    rolling_model = RollingManager("model_rolling", friction=friction, log=verbose)
    rolling_naive = RollingManager("naive_rolling", friction=friction, log=verbose)
    weeks: List[Dict[str, Any]] = []
    pred_rows: List[Tuple[float, float, float]] = []  # (model xP, naive xP, actual) per player-week

    for gw in gws:
        preds, ev_meta, n_rows = predict_all_for_gw(gw, use_coach)
        by_id = {p.player.id: p for p in preds}
        pos_of = {p.player.id: p.player.position for p in preds}
        pool_preds = [p for p in preds if predictor.is_selectable(p, 0.2)]
        model_pp = to_plan_players(pool_preds)
        naive_pp = naive_plan_players(pool_preds, [gw + k for k in range(HORIZON_GWS)])

        # player-level accuracy on everyone with a fixture this week
        naive_xp_of = {pl.id: pl.xp_by_gw.get(gw, 0.0) for pl in naive_plan_players(preds, [gw])}
        week_rows: List[Tuple[float, float, float]] = []
        for p in preds:
            if p.player.next_opponent == "BLANK" or not p.player.gw_history:
                continue
            week_rows.append((p.expected_points, naive_xp_of.get(p.player.id, 0.0), float(actual_gw_stats(raw_by_id, p.player.id, gw)[0])))
        pred_rows.extend(week_rows)
        wa = np.asarray(week_rows)

        row: Dict[str, Any] = {"gw": gw, "train_rows": n_rows, "fpl_average": int(ev_meta.get("average_entry_score", 0) or 0),
                               "fpl_highest": int(ev_meta.get("highest_score", 0) or 0),
                               "player_mae_model": round(float(np.abs(wa[:, 0] - wa[:, 2]).mean()), 2),
                               "player_mae_naive": round(float(np.abs(wa[:, 1] - wa[:, 2]).mean()), 2),
                               "player_corr_model": round(float(np.corrcoef(wa[:, 0], wa[:, 2])[0, 1]), 2),
                               "player_corr_naive": round(float(np.corrcoef(wa[:, 1], wa[:, 2])[0, 1]), 2),
                               "max_player_xp": round(float(wa[:, 0].max()), 1)}

        # Free-hit style weekly rebuilds
        for label, pp in (("model_freehit", model_pp), ("naive_freehit", naive_pp)):
            pool = select_candidate_pool(pp, [gw])
            week = solve_squad(pool, gw, budget=100.0)
            assert week is not None
            pts, subs = score_week(week.lineup_ids, week.bench_order, week.captain_id, week.vice_id, pos_of, raw_by_id, gw)
            row[label] = pts
            row[label + "_pred"] = round(week.expected_points, 1)
            row[label + "_captain"] = by_id[week.captain_id].player.web_name
            if label == "model_freehit":
                row["model_freehit_xi"] = [f"{by_id[i].player.web_name}{' (C)' if i == week.captain_id else ''}:{actual_gw_stats(raw_by_id, i, gw)[0]}" for i in week.lineup_ids]
                row["model_freehit_subs"] = subs

        # Realistic rolling managers
        for mgr, pp in ((rolling_model, model_pp), (rolling_naive, naive_pp)):
            lineup, bench, cap, vice, hits, xp = mgr.step(pp, by_id, gw)
            pts, _ = score_week(lineup, bench, cap, vice, pos_of, raw_by_id, gw)
            row[mgr.label] = pts - 4 * hits
            row[mgr.label + "_hits"] = hits
            row[mgr.label + "_pred"] = round(xp, 1)
            row[mgr.label + "_captain"] = by_id[cap].player.web_name if cap in by_id else str(cap)

        weeks.append(row)
        if verbose:
            print(f"GW {gw:2d} | rows {n_rows:5d} | model FH {row['model_freehit']:3d} (pred {row['model_freehit_pred']:5.1f}, C {row['model_freehit_captain']}) "
                  f"| naive FH {row['naive_freehit']:3d} | model roll {row['model_rolling']:3d} (hits {row['model_rolling_hits']}) "
                  f"| naive roll {row['naive_rolling']:3d} (hits {row['naive_rolling_hits']}) | FPL avg {row['fpl_average']:2d} "
                  f"| player MAE {row['player_mae_model']:.2f} vs naive {row['player_mae_naive']:.2f}, corr {row['player_corr_model']:.2f} vs {row['player_corr_naive']:.2f}, max xP {row['max_player_xp']}")

    # ---- summary ----
    def margin_stats(key: str, ref: str = "fpl_average") -> Dict[str, float]:
        d = [w[key] - w[ref] for w in weeks]
        return {"mean": round(statistics.mean(d), 1), "sd": round(statistics.pstdev(d), 1) if len(d) > 1 else 0.0,
                "total": int(sum(d)), "weeks_won": sum(1 for x in d if x > 0), "weeks": len(d)}

    arr = np.asarray(pred_rows) if pred_rows else np.zeros((0, 3))
    summary = {
        "weeks_tested": gws,
        "use_coach": use_coach,
        "transfer_friction": friction,
        "totals": {k: int(sum(w[k] for w in weeks)) for k in ("model_freehit", "naive_freehit", "model_rolling", "naive_rolling", "fpl_average")},
        "model_rolling_vs_fpl_average": margin_stats("model_rolling"),
        "naive_rolling_vs_fpl_average": margin_stats("naive_rolling"),
        "model_rolling_vs_naive_rolling": margin_stats("model_rolling", "naive_rolling"),
        "model_freehit_vs_naive_freehit": margin_stats("model_freehit", "naive_freehit"),
        "model_freehit_vs_fpl_average": margin_stats("model_freehit"),
        "freehit_prediction_bias": round(statistics.mean(w["model_freehit_pred"] - w["model_freehit"] for w in weeks), 1),
        "player_level": {
            "n": int(len(arr)),
            "model_mae": round(float(np.abs(arr[:, 0] - arr[:, 2]).mean()), 3) if len(arr) else None,
            "naive_mae": round(float(np.abs(arr[:, 1] - arr[:, 2]).mean()), 3) if len(arr) else None,
            "model_corr": round(float(np.corrcoef(arr[:, 0], arr[:, 2])[0, 1]), 3) if len(arr) > 2 else None,
            "naive_corr": round(float(np.corrcoef(arr[:, 1], arr[:, 2])[0, 1]), 3) if len(arr) > 2 else None,
        },
        "total_hits_model_rolling": rolling_model.hits_total,
        "total_hits_naive_rolling": rolling_naive.hits_total,
    }
    return {"weeks": weeks, "summary": summary}


def print_summary(res: Dict[str, Any]) -> None:
    s = res["summary"]
    weeks = res["weeks"]
    print("\n" + "=" * 96)
    print("  WALK-FORWARD BACKTEST  (every number below is out-of-sample: trained on GW < T, scored on GW T)")
    print("=" * 96)
    print(f"{'GW':>3} | {'model FH':>8} {'(pred)':>7} | {'naive FH':>8} | {'model roll':>10} {'hits':>4} | {'naive roll':>10} {'hits':>4} | {'FPL avg':>7} | {'MAE m/n':>11} | {'corr m/n':>11}")
    for w in weeks:
        print(f"{w['gw']:>3} | {w['model_freehit']:>8} {w['model_freehit_pred']:>7} | {w['naive_freehit']:>8} | {w['model_rolling']:>10} {w['model_rolling_hits']:>4} | {w['naive_rolling']:>10} {w['naive_rolling_hits']:>4} | {w['fpl_average']:>7} | {w['player_mae_model']:>5.2f}/{w['player_mae_naive']:<5.2f} | {w['player_corr_model']:>5.2f}/{w['player_corr_naive']:<5.2f}")
    t = s["totals"]
    print("-" * 96)
    print(f"TOT | {t['model_freehit']:>8} {'':>7} | {t['naive_freehit']:>8} | {t['model_rolling']:>10} {s['total_hits_model_rolling']:>4} | {t['naive_rolling']:>10} {s['total_hits_naive_rolling']:>4} | {t['fpl_average']:>7}")
    print()
    for key, label in (
        ("model_rolling_vs_fpl_average", "Model (realistic rolling) vs FPL human average"),
        ("naive_rolling_vs_fpl_average", "Naive (realistic rolling) vs FPL human average"),
        ("model_rolling_vs_naive_rolling", "Model rolling vs naive rolling  (value of the model)"),
        ("model_freehit_vs_naive_freehit", "Model weekly rebuild vs naive weekly rebuild"),
        ("model_freehit_vs_fpl_average", "Model weekly rebuild vs FPL average  (UNFAIR: unlimited transfers)"),
    ):
        m = s[key]
        print(f"  {label:<68} {m['mean']:+5.1f} ± {m['sd']:4.1f} per week | won {m['weeks_won']}/{m['weeks']}")
    pl = s["player_level"]
    print(f"\n  Player-level (n={pl['n']}): model MAE {pl['model_mae']} vs naive {pl['naive_mae']} | corr {pl['model_corr']} vs {pl['naive_corr']}")
    print(f"  Weekly-rebuild projection bias (predicted - actual): {s['freehit_prediction_bias']:+.1f} pts/week")
    if len(weeks) < 10:
        print(f"\n  Caution: {len(weeks)} weeks is a small sample; one GW has a standard deviation of ~15 pts.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward FPL backtester")
    ap.add_argument("--from-gw", type=int, default=3)
    ap.add_argument("--to-gw", type=int, default=0, help="last GW to test (default: last fully finished GW)")
    ap.add_argument("--no-coach", action="store_true", help="disable the AI coach deltas")
    ap.add_argument("--friction", type=float, default=1.5, help="xP friction charged per transfer in the rolling planner")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    from neural_model import latest_finished_gw

    last = args.to_gw or latest_finished_gw()
    gws = list(range(max(2, args.from_gw), last + 1))
    res = run_backtest(gws, use_coach=not args.no_coach, verbose=not args.quiet, friction=args.friction)
    print_summary(res)
    if not args.no_save:
        RESULTS_PATH.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\n[saved {RESULTS_PATH}]")


if __name__ == "__main__":
    main()
