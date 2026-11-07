"""Offline regression tests for data integrity, FPL accounting and evaluation."""
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ai_coach
import backtest
import fpl_assistant
import fpl_data
import neural_model
import optimizer
import predictor
from news_scraper import NewsArticle


def players():
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    scores = [3, 2] + [2] * 5 + [8] * 5 + [9] * 3
    return [optimizer.PlanPlayer(i, str(i), f"club{i // 2}", pos, 5, 5,
                                {gw: xp for gw in range(1, 5)})
            for i, (pos, xp) in enumerate(zip(positions, scores), 1)]


class DataTests(unittest.TestCase):
    def test_deadline_not_fixture_count_controls_target(self):
        bootstrap = {"events": [
            {"id": 10, "deadline_time": "2026-11-06T18:30:00Z", "is_current": True},
            {"id": 11, "deadline_time": "2026-11-21T11:00:00Z", "is_next": True},
        ]}
        self.assertEqual(fpl_data.next_editable_gameweek(bootstrap, datetime(2026, 11, 6, 18, 29, tzinfo=timezone.utc)), 10)
        self.assertEqual(fpl_data.next_editable_gameweek(bootstrap, datetime(2026, 11, 6, 18, 30, tzinfo=timezone.utc)), 11)
        with self.assertRaisesRegex(RuntimeError, "No future"):
            fpl_data.next_editable_gameweek(bootstrap, datetime(2027, 1, 1, tzinfo=timezone.utc))

    def test_double_gameweek_and_bonus_corrections_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "summaries.json"
            old = {"round": 10, "fixture": 101, "team_h_score": 1, "total_points": 5}
            cache.write_text(json.dumps({"1": {"history": [old]}}))
            updated = {"history": [dict(old, total_points=7), dict(old, fixture=102)], "history_past": []}
            with patch.object(fpl_data, "CACHE_SUMMARIES", cache), patch.object(fpl_data, "_http_get_json", return_value=updated) as fetch:
                result = fpl_data.fetch_player_summaries_concurrent([1], latest_finished_gw=10)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(len(result["1"]["history"]), 2)
            self.assertEqual(result["1"]["history"][0]["total_points"], 7)

    def test_failed_refresh_retains_cached_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "summaries.json"
            old = {"1": {"history": [{"round": 10}]}}
            cache.write_text(json.dumps(old))
            with patch.object(fpl_data, "CACHE_SUMMARIES", cache), patch.object(fpl_data, "_http_get_json", side_effect=OSError("offline")):
                with self.assertWarns(UserWarning):
                    result = fpl_data.fetch_player_summaries_concurrent([1], force_refresh=True)
            self.assertEqual(result, old)

    def test_sync_fetches_histories_by_default(self):
        with patch("sys.argv", ["fpl_assistant.py", "sync"]), patch.object(fpl_assistant, "cmd_sync") as sync:
            fpl_assistant.main()
        self.assertTrue(sync.call_args.args[0].deep)

    def test_no_finished_round_excludes_partial_rows(self):
        player = SimpleNamespace(gw_history=[{"round": 1, "team_h_score": 1}])
        with patch.object(neural_model, "latest_finished_gw", return_value=0):
            self.assertEqual(neural_model._finished_rows(player), [])


class PlanningTests(unittest.TestCase):
    def test_solver_lineup_and_captain_survive_rendering(self):
        pool = players()
        week = optimizer.solve_squad(pool, 1, budget=100, min_def_in_xi=4)
        preds = {p.id: SimpleNamespace(player=SimpleNamespace(id=p.id, position=p.position),
                                      floor_points=0, ceiling_points=15) for p in pool}
        result = predictor._result_from_week_plan(week, preds)
        self.assertEqual([p.player.id for p in result.starting_xi], week.lineup_ids)
        self.assertEqual(sum(p.player.position == "DEF" for p in result.starting_xi), 4)
        self.assertEqual(result.captain.player.id, week.captain_id)
        self.assertEqual(result.total_expected_points, round(week.expected_points, 1))
        self.assertIsNotNone(week.solver_gap)
        self.assertEqual(week.candidate_count, len(pool))

    def test_owned_player_is_not_devalued_when_ineligible(self):
        pool = players()
        pool[7] = replace(pool[7], price=10, sell_price=10)
        manager = backtest.RollingManager("test", horizon=2)
        manager.squad = [p.id for p in pool]
        manager.bank = 5
        original = backtest.plan_transfers
        captured = {}

        def observe(candidates, *args, **kwargs):
            captured["player"] = next(p for p in candidates if p.id == 8)
            captured["budget"] = kwargs["budget"]
            return original(candidates, *args, **kwargs)

        with patch.object(backtest, "plan_transfers", side_effect=observe):
            manager.step(pool, {}, 1, eligible_ids=[p.id for p in pool if p.id != 8])
        self.assertEqual(captured["player"], pool[7])
        self.assertEqual(captured["budget"], 85)

    def test_missing_owned_player_retains_last_known_metadata(self):
        pool = players()
        pool[7] = replace(pool[7], price=10, sell_price=10)
        manager = backtest.RollingManager("test", horizon=2)
        manager.squad = [p.id for p in pool]
        manager.known_players = {p.id: p for p in pool}
        manager.bank = 5
        original = backtest.plan_transfers
        with patch.object(backtest, "plan_transfers", wraps=original) as solve:
            manager.step([p for p in pool if p.id != 8], {}, 1)
        restored = next(p for p in solve.call_args.args[0] if p.id == 8)
        self.assertEqual((restored.price, restored.position, restored.team), (10, "MID", pool[7].team))
        self.assertEqual(restored.xp_by_gw, {})
        self.assertEqual(solve.call_args.kwargs["budget"], 85)

    def test_wildcard_preserves_bank(self):
        pool = players()
        for ft in (1, 4, 5):
            with self.subTest(ft=ft):
                plan = optimizer.plan_transfers(pool, [p.id for p in pool], [1, 2], 100, ft, wildcard_week_index=0)
                self.assertEqual(plan.weeks[0].free_transfers_after, ft)
                self.assertEqual(plan.weeks[1].free_transfers_before, ft)
                self.assertEqual(plan.total_hits, 0)

    def test_rolling_and_cap_accounting(self):
        pool = players()
        plan = optimizer.plan_transfers(pool, [p.id for p in pool], [1, 2, 3], 100, 4)
        self.assertEqual([w.free_transfers_before for w in plan.weeks], [4, 5, 5])
        self.assertEqual([w.free_transfers_after for w in plan.weeks], [5, 5, 5])
        self.assertEqual(plan.total_hits, 0)

    def test_terminal_value_changes_last_week_transfer(self):
        pool = [replace(p, xp_by_gw={1: 5}) for p in players()]
        pool.append(optimizer.PlanPlayer(16, "Upgrade", "new", "MID", 5, 5, {1: 5.5}))
        arguments = dict(players=pool, initial_squad=list(range(1, 16)), gws=[1], budget=100,
                         free_transfers=1, transfer_friction=0, bench_weight=0)
        spend = optimizer.plan_transfers(**arguments, ft_end_value=0)
        roll = optimizer.plan_transfers(**arguments, ft_end_value=1.2)
        self.assertEqual(spend.weeks[0].transfers_in, [16])
        self.assertEqual(roll.weeks[0].transfers_in, [])
        self.assertEqual(roll.weeks[0].free_transfers_after, 2)

    def test_hits_are_charged_and_cannot_manufacture_free_transfers(self):
        pool = [replace(p, xp_by_gw={1: 5}) for p in players()]
        pool.append(optimizer.PlanPlayer(16, "Upgrade", "new", "MID", 5, 5, {1: 15}))
        plan = optimizer.plan_transfers(pool, list(range(1, 16)), [1], 100, 0, transfer_friction=0)
        self.assertEqual(plan.weeks[0].transfers_in, [16])
        self.assertEqual(plan.weeks[0].hits, 1)
        self.assertEqual(plan.weeks[0].free_transfers_after, 1)


class EvaluationTests(unittest.TestCase):
    def test_empty_training_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "sync"):
            neural_model.train_with_time_holdout([])

    def test_baselines_are_learned_from_training_only(self):
        net = neural_model.FPLNeuralNetwork()
        features = [0.0] * neural_model.N_FEATURES
        train = [neural_model.Sample(features, 90, 2, 1, 1, 0)] * 30
        test = [neural_model.Sample(features, 90, 9, 2, 1, 0)] * 30
        net.train(train, epochs=1)
        scores = net.evaluate(test)
        self.assertEqual(scores["mae_constant_baseline"], 7)
        self.assertEqual(scores["haul_brier_baseline"], 1)

    def test_test_round_evaluated_once_after_epoch_selection(self):
        features = [0.0] * neural_model.N_FEATURES
        samples = [neural_model.Sample(features, 90, 2, gw, i, 2)
                   for gw in range(2, 10) for i in range(30)]
        evaluated = []
        original = neural_model.FPLNeuralNetwork.evaluate

        def observe(net, rows):
            evaluated.append({s.gw for s in rows})
            return original(net, rows)

        with patch.object(neural_model.FPLNeuralNetwork, "evaluate", observe):
            _, metrics = neural_model.train_with_time_holdout(samples, max_epochs=3)
        self.assertEqual(metrics["holdout_gws"], "GW 7-8")
        self.assertEqual(metrics["test_gws"], "GW 9-9")
        self.assertEqual(sum(9 in rounds for rounds in evaluated), 1)
        self.assertIn("test", metrics)


class CoachTests(unittest.TestCase):
    def test_news_headline_reaches_optional_coach_request(self):
        article = NewsArticle(source="test", headline="Team news", text="")
        with patch.object(ai_coach.urllib.request, "urlopen", side_effect=OSError("offline")) as request:
            ai_coach._refine_top_candidates_with_gemini([], {}, [article], "test-key")
        self.assertEqual(request.call_count, 1)
        self.assertIn("Team news", request.call_args.args[0].data.decode())


if __name__ == "__main__":
    unittest.main()
