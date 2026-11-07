"""Fixed-protocol tests on archived seasons, including realistic rolling managers."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from backtest import RollingManager, score_week
from fpl_data import DATA_DIR
from historical_data import load_season, prior_season_stats, fixture_features
from historical_training import load_samples, fit_for_season, EVALUATION_MODEL
from neural_model import FPLNeuralNetwork, features_from_history, get_or_train_neural_net
from optimizer import PlanPlayer, select_candidate_pool, solve_squad

RESULTS_PATH = DATA_DIR / "historical_backtest_results.json"


def week_players(archive, net, gw, horizon=3):
    previous = prior_season_stats(archive["season"])
    cutoff = archive["deadlines"][gw]
    model, naive, positions = [], [], {}
    for pid, rows in archive["players"].items():
        prior = [r for r in rows if r["round"] < gw and r["kickoff_time"] < cutoff]
        current = [r for r in rows if r["round"] == gw]
        if not prior and not current:
            continue
        # Match-row price/position/team are archived metadata, never final-season values.
        snapshot = current[0] if current else prior[-1]
        pts, mins = previous.get(snapshot["code"], (0, 0))
        past90 = pts / (mins / 90) if mins >= 450 else 0.0
        xp, nxp = {}, {}
        n_effective = max(3, sum(.9 ** max(0, gw - 1 - r["round"]) for r in prior if r["minutes"] > 0))
        uncertainty = 0.0
        for target in range(gw, min(39, gw + horizon)):
            xp[target] = nxp[target] = 0.0
            for fixture in archive["fixtures"]:
                if fixture["event"] != target or snapshot["team_id"] not in (fixture["team_h"], fixture["team_a"]):
                    continue
                home = snapshot["team_id"] == fixture["team_h"]
                context = dict(snapshot, round=gw, was_home=home,
                               opponent_team=fixture["team_a"] if home else fixture["team_h"])
                features = features_from_history(snapshot["position"], prior, fixture_features(archive, context),
                                                 snapshot["value"] / 10, past90, min(1, mins / 3420), False, False,
                                                 season_start=int(archive["season"][:4]))
                out = net.predict_full(features)
                xp[target] += out["expected_points"]
                nxp[target] += sum(r["total_points"] for r in prior) / len(prior) if prior else 0.0
                uncertainty = out["residual_std"] / math.sqrt(n_effective)
        team = archive["teams"][snapshot["team_id"]]["short_name"]
        p = PlanPlayer(pid, snapshot["name"], team, snapshot["position"], snapshot["value"] / 10,
                       snapshot["value"] / 10, xp, xp_uncertainty=uncertainty)
        model.append(p)
        naive.append(replace(p, xp_by_gw=nxp))
        positions[pid] = snapshot["position"]
    return model, naive, positions


def season_strategy_test(net, season, start=19, end=38, verbose=True):
    archive = load_season(season)
    raw = {pid: SimpleNamespace(gw_history=rows) for pid, rows in archive["players"].items()}
    managers = {label: RollingManager(label) for label in ("model_rolling", "naive_rolling")}
    weeks = []
    for gw in range(start, end + 1):
        if gw not in archive["deadlines"]:
            continue
        model, naive, positions = week_players(archive, net, gw, min(3, 39-gw))
        row = {"gw": gw}
        for label, candidates in (("model", model), ("naive", naive)):
            week = solve_squad(select_candidate_pool(candidates, [gw]), gw)
            if week is None:
                raise RuntimeError(f"No feasible {season} GW{gw} {label} squad")
            actual, _ = score_week(week.lineup_ids, week.bench_order, week.captain_id, week.vice_id, positions, raw, gw)
            row[label + "_freehit"] = actual
            row[label + "_freehit_pred"] = week.expected_points
            manager = managers[label + "_rolling"]
            manager.horizon = min(3, 39-gw)
            lineup, bench, captain, vice, hits, predicted = manager.step(candidates, {}, gw)
            actual, _ = score_week(lineup, bench, captain, vice, positions, raw, gw)
            row[label + "_rolling"] = actual - 4 * hits
            row[label + "_rolling_hits"] = hits
            row[label + "_rolling_pred"] = predicted - 4 * hits
        weeks.append(row)
        if verbose:
            print(f"{season} GW{gw}: rolling model {row['model_rolling']} / naive {row['naive_rolling']}; "
                  f"hits {row['model_rolling_hits']}/{row['naive_rolling_hits']}", flush=True)
    totals = {key: sum(w[key] for w in weeks) for key in ("model_rolling", "naive_rolling", "model_freehit", "naive_freehit",
                                                        "model_rolling_hits", "naive_rolling_hits")}
    differences = [w["model_rolling"] - w["naive_rolling"] for w in weeks]
    # Resample contiguous four-week blocks, since a rolling manager's weeks are dependent.
    rng = np.random.default_rng(42)
    bootstrap = []
    for _ in range(2000):
        sampled = []
        while len(sampled) < len(differences):
            offset = int(rng.integers(max(1, len(differences) - 3)))
            sampled.extend(differences[offset:offset+4])
        bootstrap.append(np.mean(sampled[:len(differences)]))
    return {"season": season, "weeks": weeks, "totals": totals,
            "rolling_margin_mean": round(statistics.mean(differences), 3),
            "rolling_margin_block_bootstrap_95pct": np.quantile(bootstrap, [.025, .975]).round(3).tolist(),
            "rolling_prediction_bias": round(statistics.mean(w["model_rolling_pred"] - w["model_rolling"] for w in weeks), 3),
            "human_average": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    args = parser.parse_args()
    samples = load_samples()
    deployed = get_or_train_neural_net()
    results = []
    for season in args.seasons:
        if season == deployed.metrics["evaluation"]["evaluation_season"]:
            net, evaluation = FPLNeuralNetwork.load(EVALUATION_MODEL), deployed.metrics["evaluation"]
        else:
            net, evaluation = fit_for_season(samples, season, verbose=True)
        result = season_strategy_test(net, season)
        result["prediction_evaluation"] = evaluation
        results.append(result)
        RESULTS_PATH.write_text(json.dumps({"protocol": "train through GW10, select on GW11-14, calibrate GW15-18, test GW19-38; no test-row fitting",
                                           "limitations": "Archived final fixture assignments may include later postponements; historical official average scores unavailable. Managers start at GW19 with £100m, no chips, and current-price selling values.",
                                           "seasons": results}, indent=2) + "\n")
    print(json.dumps({r["season"]: r["totals"] for r in results}, indent=2))


if __name__ == "__main__":
    main()
