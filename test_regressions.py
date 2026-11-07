"""Offline regression tests for data integrity, FPL accounting and evaluation."""
import json
import math
import gzip
import tempfile
import unittest
from collections import Counter
from itertools import combinations
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
import historical_data
import historical_training
import squad_shuffle
from news_scraper import NewsArticle


def players():
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    scores = [3, 2] + [2] * 5 + [8] * 5 + [9] * 3
    return [optimizer.PlanPlayer(i, str(i), f"club{i // 2}", pos, 5, 5,
                                {gw: xp for gw in range(1, 5)})
            for i, (pos, xp) in enumerate(zip(positions, scores), 1)]


class ShuffleTests(unittest.TestCase):
    def test_fixed_squad_score_matches_exhaustive_lineup_and_captain_search(self):
        pool = players()
        pool[4] = replace(pool[4], xp_by_gw={1: 12.25})
        pool[8] = replace(pool[8], xp_by_gw={1: 7.13})
        result = squad_shuffle.score_squad(pool, 1)
        possible = []
        for xi in combinations(pool, 11):
            counts = Counter(p.position for p in xi)
            if all(optimizer.XI_MIN[pos] <= counts[pos] <= optimizer.XI_MAX[pos] for pos in optimizer.POS_ORDER):
                possible.append(sum(p.xp_by_gw[1] for p in xi) + max(p.xp_by_gw[1] for p in xi))
        self.assertEqual(result.expected_points, round(max(possible), 2))
        self.assertIn(result.captain_id, result.lineup_ids)
        self.assertNotEqual(result.captain_id, result.vice_id)
        self.assertEqual(len(result.bench_order), 4)
        self.assertEqual(next(p.position for p in pool if p.id == result.bench_order[-1]), "GKP")

    def test_sampled_squads_are_legal_distinct_and_use_original_scores(self):
        pool = players()
        pool += [replace(p, id=p.id + 20, team="extras" if p.id % 2 else "other",
                         price=6, xp_by_gw={1: p.xp_by_gw[1] - .4}) for p in list(pool)]
        before = [dict(p.xp_by_gw) for p in pool]
        results = squad_shuffle.generate_shuffle_pool(pool, 1, budget=80, count=16, seed=19)
        self.assertEqual(len(results), 16)
        self.assertEqual(len({tuple(sorted(w.lineup_ids)) for w in results}), 16)
        by_id = {p.id: p for p in pool}
        for week in results:
            squad = [by_id[i] for i in week.squad_ids]
            self.assertEqual(Counter(p.position for p in squad), optimizer.SQUAD_SLOTS)
            self.assertLessEqual(sum(p.price for p in squad), 80)
            self.assertLessEqual(max(Counter(p.team for p in squad).values()), 3)
            xi = [by_id[i] for i in week.lineup_ids]
            self.assertEqual(len(xi), 11)
            for pos in optimizer.POS_ORDER:
                n = sum(p.position == pos for p in xi)
                self.assertTrue(optimizer.XI_MIN[pos] <= n <= optimizer.XI_MAX[pos])
            self.assertEqual(week.expected_points, round(sum(p.xp_by_gw[1] for p in xi) + by_id[week.captain_id].xp_by_gw[1], 2))
        self.assertEqual(before, [p.xp_by_gw for p in pool])
        repeated = squad_shuffle.generate_shuffle_pool(pool, 1, budget=80, count=16, seed=19)
        self.assertEqual([w.lineup_ids for w in results], [w.lineup_ids for w in repeated])

    def test_small_pool_exhaustion_and_impossible_budget_are_bounded(self):
        result = squad_shuffle.generate_shuffle_pool(players(), 1, count=12, max_attempts=20)
        self.assertEqual(len(result), 1)
        with self.assertRaisesRegex(RuntimeError, "No valid squad"):
            squad_shuffle.generate_shuffle_pool(players(), 1, budget=1, max_attempts=0)
        with self.assertRaisesRegex(ValueError, "unique"):
            squad_shuffle.generate_shuffle_pool(players() + [players()[0]], 1)


class DataTests(unittest.TestCase):
    def test_public_snapshot_preserves_fetch_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bootstrap, fixtures, summaries = (root / name for name in ("bootstrap.json", "fixtures.json", "summaries.json"))
            bootstrap.write_text('{"events": []}')
            fixtures.write_text('[]')
            summaries.write_text('{"1": {"fetched_at": "2026-11-07T10:00:00Z", "history": []}}')
            with patch.object(fpl_data, "DATA_DIR", root), patch.object(fpl_data, "CACHE_BOOTSTRAP", bootstrap), patch.object(fpl_data, "CACHE_FIXTURES", fixtures), patch.object(fpl_data, "CACHE_SUMMARIES", summaries):
                path = fpl_data.archive_live_snapshot()
            with gzip.open(path, "rt") as handle:
                saved = json.load(handle)
            self.assertIn("observed_at", saved)
            self.assertEqual(saved["player_summaries"]["1"]["fetched_at"], "2026-11-07T10:00:00Z")
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


class HistoricalTests(unittest.TestCase):
    def test_target_match_outcomes_do_not_change_its_features(self):
        row = dict(round=1, fixture=1, code=123, position="MID", value=70, minutes=90,
                   total_points=2, kickoff_time="2025-08-15T19:00:00Z", expected_goals=0)
        archive = {"players": {7: [row]}, "deadlines": {1: "2025-08-15T17:30:00Z"}}
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"seasons": ["2025-26"]}))
            with patch.object(historical_data, "MANIFEST_PATH", manifest), patch.object(historical_data, "current_season_start", return_value=2026), patch.object(historical_data, "load_season", return_value=archive), patch.object(historical_data, "prior_season_stats", return_value={}), patch.object(historical_data, "fixture_features", return_value={}):
                before = historical_data.build_historical_samples()[0]
                row.update(total_points=25, expected_goals=4, minutes=30)
                after = historical_data.build_historical_samples()[0]
        self.assertEqual(before.features, after.features)
        self.assertEqual(len(before.features), neural_model.N_FEATURES)
        self.assertNotEqual(before.points, after.points)

    def test_seasons_do_not_collide_and_test_never_enters_development(self):
        feats = [0.] * neural_model.N_FEATURES
        samples = [neural_model.Sample(feats, 90, 2, gw, 1, 0, season=season)
                   for season in ("2023-24", "2024-25", "2025-26", "2026-27")
                   for gw in range(1, 39)]
        train, val, calibration, test = historical_training.split_historical(samples, "2025-26")
        self.assertIn((2024, 38), {s.period for s in train})
        self.assertEqual({s.period for s in val}, {(2025, g) for g in range(11, 15)})
        self.assertEqual({s.period for s in calibration}, {(2025, g) for g in range(15, 19)})
        self.assertEqual({s.period for s in test}, {(2025, g) for g in range(19, 39)})
        self.assertFalse(any(s.season == "2026-27" for split in (train, val, calibration, test) for s in split))

    def test_archive_normalizes_goalkeepers_and_excludes_managers(self):
        fixture = dict(id="1", event="1", team_h="1", team_a="2", team_h_score="1", team_a_score="0",
                       kickoff_time="2025-08-15T19:00:00Z", finished="True")
        player = dict(element="7", fixture="1", round="1", GW="1", was_home="True", position="GK",
                      kickoff_time=fixture["kickoff_time"], opponent_team="2", name="Keeper", minutes="90", total_points="6", value="50")
        tables = {"fixtures.csv": [fixture], "players_raw.csv": [{"id": "7", "code": "123"}],
                  "teams.csv": [{"id": "1"}, {"id": "2"}],
                  "gws/merged_gw.csv": [player, dict(player), dict(player, element="99", position="AM")]}
        historical_data.load_season.cache_clear()
        with patch.object(historical_data, "csv_rows", side_effect=lambda s, f: tables[f]):
            result = historical_data.load_season("test-season")
        historical_data.load_season.cache_clear()
        self.assertEqual(result["players"][7][0]["position"], "GKP")
        self.assertEqual(result["players"][7][0]["team_id"], 1)
        self.assertEqual(result["duplicate_rows_removed"], 1)
        self.assertEqual(result["manager_rows_excluded"], 1)

    def test_current_season_outcomes_rejected_by_importer(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"seasons": ["2026-27"]}))
            with patch.object(historical_data, "MANIFEST_PATH", manifest), patch.object(historical_data, "current_season_start", return_value=2026):
                with self.assertRaisesRegex(ValueError, "Current-season"):
                    historical_data.build_historical_samples()

    def test_risk_margin_blocks_noisy_marginal_swap(self):
        pool = [replace(p, xp_by_gw={1: 5}, xp_uncertainty=1.5) for p in players()]
        pool.append(optimizer.PlanPlayer(16, "Uncertain", "new", "MID", 5, 5, {1: 5.6}, xp_uncertainty=1.5))
        args = dict(players=pool, initial_squad=list(range(1, 16)), gws=[1], budget=100,
                    free_transfers=1, transfer_friction=0, ft_end_value=0, bench_weight=0)
        risky = optimizer.plan_transfers(**args, risk_margin=0)
        guarded = optimizer.plan_transfers(**args, risk_margin=.5)
        self.assertEqual(risky.weeks[0].transfers_in, [16])
        self.assertEqual(guarded.weeks[0].transfers_in, [])


class TeamRatingTests(unittest.TestCase):
    def _matches(self):
        import team_ratings
        base = datetime(2024, 8, 10, tzinfo=timezone.utc)
        rows = []
        for week in range(12):
            day = base.replace(day=1) if False else base
            kickoff = datetime(2024, 8, 10, tzinfo=timezone.utc)
            from datetime import timedelta
            kickoff = kickoff + timedelta(days=7 * week)
            rows.append(team_ratings.Match(kickoff, 1, 2, 3, 0))  # club 1 keeps beating club 2 at home
            rows.append(team_ratings.Match(kickoff, 2, 1, 0, 2) if week % 2 else team_ratings.Match(kickoff, 3, 4, 1, 1))
        return rows

    def test_ratings_are_point_in_time_and_rank_strength(self):
        import team_ratings
        from datetime import timedelta
        matches = self._matches()
        cutoff = datetime(2024, 9, 20, tzinfo=timezone.utc)
        early = team_ratings.PoissonRatings(matches, cutoff)
        later_only = [m for m in matches if m.kickoff < cutoff]
        self.assertEqual(early.n_matches, len(later_only))
        same = team_ratings.PoissonRatings(later_only, cutoff)
        self.assertAlmostEqual(float(early.expected_goals(1, 2)[0]), float(same.expected_goals(1, 2)[0]))
        home_for, home_against = early.expected_goals(1, 2)
        self.assertGreater(home_for, home_against)
        context = early.fixture_context(2, 1, False)
        self.assertAlmostEqual(context["lambda_for"], round(home_against, 4))
        self.assertAlmostEqual(context["lambda_against"], round(home_for, 4))
        self.assertIn("team_att", context)

    def test_unknown_club_gets_promoted_prior_and_features_tolerate_missing_context(self):
        import team_ratings
        model = team_ratings.PoissonRatings(self._matches(), datetime(2024, 10, 1, tzinfo=timezone.utc))
        self.assertEqual(model._params(999), (team_ratings.PROMOTED_PRIOR, team_ratings.PROMOTED_PRIOR))
        feats = neural_model.features_from_history("MID", [], {}, 5.0)
        self.assertEqual(len(feats), neural_model.N_FEATURES)
        self.assertAlmostEqual(feats[neural_model.FEATURE_NAMES.index("log_lambda_for")], __import__("math").log(neural_model.DEFAULT_LAMBDA))


class ComponentModelTests(unittest.TestCase):
    def test_component_points_follow_official_scoring_rules(self):
        import math
        net = neural_model.FPLNeuralNetwork()
        for key in list(net.params):
            if key.startswith("W_"):
                net.params[key][:] = 0.0
        net.params["b_goals"][:] = math.log(0.5)
        net.params["b_assists"][:] = math.log(0.25)
        net.params["b_cs"][:] = 0.0  # P(clean sheet) = 0.5
        net.params["b_gc"][:] = math.log(1e-6)
        net.params["b_saves"][:] = math.log(1e-6)
        net.params["b_bonus"][:] = math.log(0.4)
        net.params["b_defcon"][:] = 30.0  # certain threshold
        net.params["b_yellow"][:] = -30.0
        net.component_constants = [0.0, 0.0, 0.0, 0.0]
        net.points_output = "component"
        feats = [0.0] * neural_model.N_FEATURES
        feats[neural_model.FEATURE_NAMES.index("is_DEF")] = 1.0
        feats[neural_model.FEATURE_NAMES.index("scoring_2025_onward")] = 1.0
        xp = net._forward(__import__("numpy").asarray([feats]))["xp_component"][0]
        expected = 2 + 0.5 * 6 + 0.25 * 3 + 0.5 * 4 + 0.4 + 2
        self.assertAlmostEqual(float(xp), expected, places=4)
        feats[neural_model.FEATURE_NAMES.index("scoring_2025_onward")] = 0.0
        xp_old = net._forward(__import__("numpy").asarray([feats]))["xp_component"][0]
        self.assertAlmostEqual(float(xp_old), expected - 2, places=4)

    def test_expected_floor_division_matches_enumeration(self):
        import numpy as np
        lam = np.array([1.7])
        value = neural_model.expected_floor_division(lam, 2, k_max=30)[0]
        enumerated = sum((k // 2) * math.exp(-1.7) * 1.7 ** k / math.factorial(k) for k in range(31))
        self.assertAlmostEqual(float(value), enumerated, places=6)

    def test_prior_shrinkage_uses_previous_season_rates(self):
        prior = neural_model.prior_rates_from_totals({"minutes": 2700, "total_points": 180, "expected_goals": 9})
        self.assertEqual(prior["valid"], 1.0)
        self.assertAlmostEqual(prior["pts"], 6.0)
        rec = neural_model.compute_recency_weighted_metrics([], prior=prior, position="MID")
        self.assertAlmostEqual(rec["ewma_ppg"], 6.0)
        one_row = [{"round": 1, "minutes": 90, "total_points": 2, "team_h_score": 1, "kickoff_time": "k"}]
        rec = neural_model.compute_recency_weighted_metrics(one_row, prior=prior, position="MID")
        self.assertAlmostEqual(rec["ewma_ppg"], (2 + 3 * 6.0) / 4.0)
        thin = neural_model.prior_rates_from_totals({"minutes": 100, "total_points": 20})
        self.assertEqual(thin["valid"], 0.0)
        rec = neural_model.compute_recency_weighted_metrics([], prior=thin, position="FWD")
        self.assertAlmostEqual(rec["ewma_ppg"], neural_model.POSITION_PRIOR_RATES["FWD"]["pts"])


class BacktestBlindingTests(unittest.TestCase):
    def test_players_without_a_target_week_row_are_excluded(self):
        fixtures = [{"id": 1, "event": 3, "team_h": 1, "team_a": 2, "kickoff_time": "2026-09-01T14:00:00Z", "finished": True,
                     "team_h_score": 1, "team_a_score": 0},
                    {"id": 2, "event": 4, "team_h": 2, "team_a": 1, "kickoff_time": "2026-09-08T14:00:00Z", "finished": True,
                     "team_h_score": 0, "team_a_score": 0}]
        bootstrap = {"teams": [{"id": 1, "code": 11, "short_name": "AAA", "name": "A"}, {"id": 2, "code": 22, "short_name": "BBB", "name": "B"}],
                     "events": [{"id": 4, "deadline_time": "2026-09-07T10:00:00Z"}]}
        row3 = {"round": 3, "fixture": 1, "was_home": True, "minutes": 90, "total_points": 6, "team_h_score": 1, "team_a_score": 0, "value": 50}
        row4 = {"round": 4, "fixture": 2, "was_home": False, "minutes": 90, "total_points": 2, "team_h_score": 0, "team_a_score": 0, "value": 50}
        base = dict(id=0, web_name="x", full_name="x", team_id=2, team_short="BBB", team_name="B", position="MID", price=5.0, status="a",
                    chance_of_playing_next_round=None, fpl_news="", total_points=0, minutes=0, starts=0, points_per_game=0, form=0,
                    value_form=0, value_season=0, selected_by_percent=0, xg_per_90=0, xa_per_90=0, xgi_per_90=0, xgc_per_90=0,
                    defensive_contribution_per_90=0, ict_per_90=0, threat_per_90=0, creativity_per_90=0, saves_per_90=0, clean_sheets=0,
                    bps_per_90=0, bonus=0, in_dreamteam=False, penalties_order=None, corners_order=None, direct_freekicks_order=None,
                    next_opponent="", is_home=True, fixture_difficulty=3, team_attack_strength=1150, team_defence_strength=1150,
                    opp_attack_strength=1150, opp_defence_strength=1150)
        registered = fpl_data.PlayerRecord(**dict(base, id=1, gw_history=[row3, row4]))
        later_signing = fpl_data.PlayerRecord(**dict(base, id=2, gw_history=[]))
        moved_club = fpl_data.PlayerRecord(**dict(base, id=3, team_id=2, gw_history=[row3]))  # GW3 row says home side = team 1
        with tempfile.TemporaryDirectory() as tmp:
            bootstrap_path, fixtures_path = Path(tmp) / "bootstrap.json", Path(tmp) / "fixtures.json"
            bootstrap_path.write_text(json.dumps(bootstrap))
            fixtures_path.write_text(json.dumps(fixtures))
            with patch.object(backtest, "load_players", return_value=[registered, later_signing, moved_club]), \
                 patch.object(backtest, "snapshot_before", return_value=None), \
                 patch.object(backtest, "CACHE_BOOTSTRAP", bootstrap_path), \
                 patch.object(backtest, "CACHE_FIXTURES", fixtures_path):
                blinded, meta = backtest.build_blinded_players_for_gw(4)
        ids = {p.id for p in blinded}
        self.assertIn(1, ids)
        self.assertNotIn(2, ids)
        self.assertNotIn(3, ids)  # had no GW4 row although its club played: not registered that week
        self.assertEqual(meta["excluded_unregistered"], 2)
        self.assertEqual([p.team_id for p in blinded if p.id == 1], [1])
        self.assertEqual([len(p.gw_history) for p in blinded if p.id == 1], [1])

    def test_snapshot_after_deadline_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payload = {"observed_at": "2026-09-07T12:00:00+00:00", "bootstrap": {"elements": []}}
            with gzip.open(directory / "20260907T120000000000Z.json.gz", "wt") as handle:
                json.dump(payload, handle)
            with patch.object(backtest, "SNAPSHOT_DIR", directory):
                self.assertIsNone(backtest.snapshot_before("2026-09-07T10:00:00Z"))
                self.assertIsNotNone(backtest.snapshot_before("2026-09-08T10:00:00Z"))


class PlannerParameterTests(unittest.TestCase):
    def test_defaults_when_no_selection_file(self):
        with patch.object(optimizer, "PLANNER_PARAMS_PATH", Path("/nonexistent/planner_params.json")):
            params = optimizer.load_planner_params()
        self.assertEqual(params, optimizer.DEFAULT_PLANNER_PARAMS)
        manager = backtest.RollingManager("x", friction=0.25, risk_margin=0.75)
        self.assertEqual((manager.friction, manager.risk_margin), (0.25, 0.75))


if __name__ == "__main__":
    unittest.main()
