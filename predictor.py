"""
FPL Expected Points (xP) Predictor & Squad Optimizer (predictor.py)

Pipeline per player:
  1. Multi-head neural network (neural_model.py): P(0 / 1-59 / 60+ minutes), points given
     60+ minutes, and P(9+ haul) — all trained on real gameweek outcomes.
  2. Availability (news_scraper.analyze_player_news): minutes model x official FPL flags
     x your own tagged notes.
  3. AI Tactical Coach contextual delta (ai_coach.py) — small, bounded adjustments.
  4. Per-gameweek expected points for the next HORIZON_GWS gameweeks (blanks = 0, doubles summed).

Squad selection and transfer planning are exact mixed-integer programs (optimizer.py).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ai_coach import CoachDecision, evaluate_squad_with_ai_coach
from fpl_data import HORIZON_GWS, PlayerRecord, load_players
from neural_model import extract_features, get_or_train_neural_net
from news_scraper import NewsArticle, PlayerNewsAssessment, analyze_player_news
from optimizer import (
    PlanPlayer,
    TransferPlan,
    WeekPlan,
    plan_transfers,
    select_candidate_pool,
    solve_squad,
)

_CACHED_NN = None
_CACHED_COACH: Optional[Dict[int, CoachDecision]] = None


def _get_nn():
    global _CACHED_NN
    if _CACHED_NN is None:
        _CACHED_NN = get_or_train_neural_net(force_retrain=False)
    return _CACHED_NN


def _get_coach_decisions(articles: List[NewsArticle]) -> Dict[int, CoachDecision]:
    global _CACHED_COACH
    if _CACHED_COACH is None:
        all_p = load_players()
        _CACHED_COACH = evaluate_squad_with_ai_coach(all_p, articles)
    return _CACHED_COACH


def reset_caches() -> None:
    global _CACHED_NN, _CACHED_COACH
    _CACHED_NN = None
    _CACHED_COACH = None


# Official FPL Scoring Rules by Position (used for the diagnostic rule-based xP only)
GOAL_POINTS = {"GKP": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
ASSIST_POINTS = 3.0
CS_POINTS = {"GKP": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
APPEARANCE_60PLUS_PTS = 2.0
APPEARANCE_CAMEO_PTS = 1.0
YELLOW_CARD_PTS = -1.0
PENALTY_MISS_PTS = -2.0

VALID_FORMATIONS: List[Tuple[int, int, int]] = [
    (3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (4, 5, 1), (5, 3, 2), (5, 4, 1), (5, 2, 3),
]

HORIZON_WEIGHTS_3GW = [1.0, 0.75, 0.55]


def _poisson_prob(lmbda: float, k: int) -> float:
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)


@dataclass
class PlayerPrediction:
    player: PlayerRecord
    news: PlayerNewsAssessment
    clean_sheet_prob: float
    concede_2plus_prob: float
    expected_goals: float
    expected_assists: float
    rule_xp_90: float  # Official FPL rule engine 90-min reference (diagnostic only)
    nn_xp_90: float  # Neural network points GIVEN 60+ minutes
    xp_if_90: float  # nn_xp_90 + AI coach delta
    expected_points: float  # Unconditional xP for the target gameweek (minutes-weighted)
    haul_prob: float  # P(points >= 9) given 60+ minutes, scaled by P(60+)
    horizon_3gw_xp: float  # Weighted next-3-gameweek xP
    role_tag: str
    coach_note: str
    floor_points: int
    ceiling_points: int
    lineup_score: float
    gw_xps: Dict[int, float] = field(default_factory=dict)  # gameweek -> xP (next HORIZON_GWS)
    p_full: float = 0.0  # P(60+ minutes) this gameweek
    p_cameo: float = 0.0  # P(1-59 minutes) this gameweek
    model_p_full: float = 0.0  # minutes head alone, before official flags and notes
    model_p_cameo: float = 0.0
    xp_uncertainty: float = 0.0


def _availability_for_future_gw(news: PlayerNewsAssessment, model_full: float, model_cameo: float, k: int,
                                kickoff: Optional[str] = None) -> Tuple[float, float]:
    """Availability k gameweeks ahead: official flags fade, the learned minutes model remains.
    An official return date is applied exactly: out before it, model-based (halved on the
    first eligible week) from it."""
    if k == 0:
        return news.start_probability, news.cameo_probability
    if news.return_date and kickoff:
        from news_scraper import kickoff_datetime
        kick = kickoff_datetime(kickoff)
        if kick is not None:
            if kick.date().isoformat() < news.return_date:
                return 0.0, 0.02
            first_back = news.risk_level == "INJURED_OR_OUT"
            return (model_full * (0.5 if first_back else 1.0), min(1.0, model_cameo + (model_full * 0.25 if first_back else 0.0)))
    if news.risk_level == "INJURED_OR_OUT":
        fade = 0.0 if k == 1 else (0.5 if k == 2 else 0.8)
        return model_full * fade, model_cameo * fade
    blend = 0.5 if k == 1 else 0.0
    return (
        blend * news.start_probability + (1 - blend) * model_full,
        blend * news.cameo_probability + (1 - blend) * model_cameo,
    )


def predict_player(player: PlayerRecord, articles: List[NewsArticle]) -> PlayerPrediction:
    nn_model = _get_nn()
    coach_map = _get_coach_decisions(articles)
    coach = coach_map.get(player.id)

    base_feats = extract_features(player)
    base = nn_model.predict_full(base_feats)
    news = analyze_player_news(player, articles, minutes_probs=(base["p_full"], base["p_cameo"]))

    adjust = coach is not None and nn_model.metrics.get("coach_adjustments", False)
    tactical_delta = coach.tactical_xp_delta if adjust else 0.0
    haul_boost = coach.haul_prob_boost if adjust else 0.0
    horizon_bonus = coach.horizon_3gw_bonus if adjust else 0.0
    role_tag = coach.role_tag if coach else "STARTER"
    coach_note = coach.coach_note if coach else f"Next fixtures: {player.fixture_run_3gw}"

    # ---- diagnostic rule-based xP (not used for selection) ----
    home_def_factor = 0.88 if player.is_home else 1.12
    mismatch_ratio = (player.opp_attack_strength / max(player.team_defence_strength, 950)) ** 1.35
    base_xgc = max(0.55, min(2.2, player.xgc_per_90 if player.xgc_per_90 > 0.1 else 1.25))
    lambda_gc = max(0.35, base_xgc * home_def_factor * mismatch_ratio)
    cs_prob = _poisson_prob(lambda_gc, 0)
    concede_2plus_prob = 1.0 - _poisson_prob(lambda_gc, 0) - _poisson_prob(lambda_gc, 1)
    home_att_factor = 1.10 if player.is_home else 0.93
    opp_def_factor = (player.team_attack_strength / max(player.opp_defence_strength, 950)) ** 1.30
    match_att_mult = home_att_factor * opp_def_factor
    exp_g = (player.xg_per_90 + (0.10 if player.penalties_order == 1 else 0.0)) * match_att_mult
    exp_a = (player.xa_per_90 + (0.05 if player.corners_order in (1, 2) else 0.0)) * match_att_mult
    gc_penalty = -concede_2plus_prob if player.position in ("GKP", "DEF") else 0.0
    rule_90_xp = (
        APPEARANCE_60PLUS_PTS
        + cs_prob * CS_POINTS[player.position]
        + gc_penalty
        + exp_g * GOAL_POINTS[player.position]
        + exp_a * ASSIST_POINTS
        + min(1.5, max(0.0, (player.bps_per_90 - 14.0) / 10.0))
    )

    # ---- per-gameweek expected points over the horizon ----
    cameo_pts = nn_model.cameo_points
    fixtures_by_gw: Dict[int, List[dict]] = {}
    for fx in player.upcoming_3gw_fixtures:
        fixtures_by_gw.setdefault(int(fx.get("gw", player.target_gw)), []).append(fx)
    target_gw = player.target_gw or (min(fixtures_by_gw) if fixtures_by_gw else 0)
    gw_xps: Dict[int, float] = {}
    for k in range(HORIZON_GWS):
        gw = target_gw + k
        total = 0.0
        for fx in fixtures_by_gw.get(gw, []):
            out = base if (k == 0 and len(fixtures_by_gw.get(gw, [])) == 1) else nn_model.predict_full(
                extract_features(player, fixture_override=fx)
            )
            p_full, p_cameo = _availability_for_future_gw(news, out["p_full"], out["p_cameo"], k, fx.get("kickoff"))
            total += p_full * max(0.5, out["xp_full"] + tactical_delta * (1.0 if k == 0 else 0.6)) + p_cameo * cameo_pts
        gw_xps[gw] = round(total, 3)

    expected_pts = gw_xps.get(target_gw, 0.0)
    xp_if_90 = max(0.5, base["xp_full"] + tactical_delta)
    final_haul_prob = round(max(0.0, min(0.85, (base["haul"] + haul_boost) * news.start_probability)), 3) if fixtures_by_gw.get(target_gw) else 0.0

    horizon_3gw_xp = sum(
        w * gw_xps.get(target_gw + k, 0.0) for k, w in enumerate(HORIZON_WEIGHTS_3GW)
    ) + horizon_bonus

    if news.risk_level == "INJURED_OR_OUT":
        floor_pts, ceiling_pts = 0, 0
    elif news.start_probability < 0.5:
        floor_pts = 0
        ceiling_pts = max(2, int(round(xp_if_90 + 2)))
    else:
        floor_pts = 1 if (player.position in ("GKP", "DEF") and concede_2plus_prob > 0.45) else 2
        ceiling_pts = int(round(xp_if_90 + 7.5 * final_haul_prob + (5.0 if player.position in ("MID", "FWD") else 3.8)))

    return PlayerPrediction(
        player=player,
        news=news,
        clean_sheet_prob=round(cs_prob, 2),
        concede_2plus_prob=round(concede_2plus_prob, 2),
        expected_goals=round(exp_g, 2),
        expected_assists=round(exp_a, 2),
        rule_xp_90=round(rule_90_xp, 2),
        nn_xp_90=round(base["xp_full"], 2),
        xp_if_90=round(xp_if_90, 2),
        expected_points=round(expected_pts, 2),
        haul_prob=final_haul_prob,
        horizon_3gw_xp=round(max(0.0, horizon_3gw_xp), 1),
        role_tag=role_tag,
        coach_note=coach_note,
        floor_points=floor_pts,
        ceiling_points=ceiling_pts,
        lineup_score=round(expected_pts, 2),
        gw_xps=gw_xps,
        p_full=round(news.start_probability, 3),
        p_cameo=round(news.cameo_probability, 3),
        model_p_full=round(base["p_full"], 3),
        model_p_cameo=round(base["p_cameo"], 3),
        xp_uncertainty=round(base["residual_std"] / math.sqrt(max(3, sum(
            0.9 ** max(0, target_gw - 1 - int(r.get("round", 0)))
            for r in player.gw_history if int(r.get("round", 0)) < target_gw and float(r.get("minutes", 0)) > 0
        ))), 3),
    )


def is_selectable(p: PlayerPrediction, min_start_prob: float = 0.5) -> bool:
    """Players worth considering for a squad: not ruled out and likely to play."""
    return p.news.risk_level != "INJURED_OR_OUT" and p.news.start_probability >= min_start_prob


# --------------------------------------------------------------------------------------
# Starting XI / captaincy from a fixed 15 (exact enumeration, small)
# --------------------------------------------------------------------------------------
@dataclass
class OptimizedSquadResult:
    formation: str
    starting_xi: List[PlayerPrediction]
    bench_gkp: PlayerPrediction
    bench_outfield: List[PlayerPrediction]
    captain: PlayerPrediction
    vice_captain: PlayerPrediction
    total_expected_points: float
    total_floor: int
    total_ceiling: int
    tactical_notes: List[str]


def _captain_key(p: PlayerPrediction) -> float:
    return p.expected_points + 0.5 * p.haul_prob


def optimize_starting_xi(squad_predictions: List[PlayerPrediction]) -> OptimizedSquadResult:
    gkps = sorted([p for p in squad_predictions if p.player.position == "GKP"], key=lambda x: x.expected_points, reverse=True)
    defs = [p for p in squad_predictions if p.player.position == "DEF"]
    mids = [p for p in squad_predictions if p.player.position == "MID"]
    fwds = [p for p in squad_predictions if p.player.position == "FWD"]
    starting_gkp = gkps[0]
    bench_gkp = gkps[1] if len(gkps) > 1 else gkps[0]

    best_score, best_formation, best_outfield = -999.0, (4, 4, 2), []
    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        if len(defs) < n_def or len(mids) < n_mid or len(fwds) < n_fwd:
            continue
        for d_combo in itertools.combinations(defs, n_def):
            for m_combo in itertools.combinations(mids, n_mid):
                for f_combo in itertools.combinations(fwds, n_fwd):
                    outfield = list(d_combo) + list(m_combo) + list(f_combo)
                    score = sum(p.expected_points for p in outfield)
                    if score > best_score:
                        best_score, best_formation, best_outfield = score, (n_def, n_mid, n_fwd), outfield

    starting_xi = [starting_gkp] + best_outfield
    starting_ids = {p.player.id for p in starting_xi}
    remaining = [p for p in squad_predictions if p.player.position != "GKP" and p.player.id not in starting_ids]
    bench_outfield = sorted(remaining, key=lambda p: p.expected_points, reverse=True)

    ranked_cap = sorted(starting_xi, key=_captain_key, reverse=True)
    captain, vice_captain = ranked_cap[0], ranked_cap[1]
    cap_zero = max(0.0, 1.0 - captain.news.start_probability - captain.news.cameo_probability)
    captaincy_bonus_xp = captain.expected_points + cap_zero * vice_captain.expected_points

    first_sub_xp = bench_outfield[0].expected_points if bench_outfield else 0.0
    auto_sub_boost = sum(
        max(0.0, 1.0 - s.news.start_probability - s.news.cameo_probability) * first_sub_xp * 0.25
        for s in best_outfield
    )
    total_xp = sum(p.expected_points for p in starting_xi) + captaincy_bonus_xp + auto_sub_boost

    notes: List[str] = []
    for p in squad_predictions:
        if p.news.risk_level == "CAMEO_ROTATION_RISK":
            notes.append(f"Cameo risk ({p.player.web_name} - {'STARTED' if p in starting_xi else 'BENCHED'}): {p.news.reasoning}")
        elif p.news.risk_level == "INJURED_OR_OUT":
            notes.append(f"Unavailable ({p.player.web_name}): {p.news.reasoning}")

    return OptimizedSquadResult(
        formation=f"{best_formation[0]}-{best_formation[1]}-{best_formation[2]}",
        starting_xi=starting_xi,
        bench_gkp=bench_gkp,
        bench_outfield=bench_outfield,
        captain=captain,
        vice_captain=vice_captain,
        total_expected_points=round(total_xp, 1),
        total_floor=sum(p.floor_points for p in starting_xi) + captain.floor_points,
        total_ceiling=sum(p.ceiling_points for p in starting_xi) + captain.ceiling_points,
        tactical_notes=notes,
    )


def build_dream_xi(all_predictions: List[PlayerPrediction], max_per_club: int = 3) -> OptimizedSquadResult:
    """Best Free-Hit style XI from the whole league (MILP with an unlimited budget)."""
    res, _ = get_best_squad_of_the_week(all_predictions, budget=1000.0, max_per_club=max_per_club)
    res.tactical_notes.append("Free Hit-style XI respecting max 3 players per club (unlimited budget).")
    return res


@dataclass
class TransferSuggestion:
    player_out: PlayerPrediction
    player_in: PlayerPrediction
    net_cost: float
    xp_gain_this_week: float
    new_starting_xp: float


def recommend_single_transfers(
    squad_predictions: List[PlayerPrediction],
    all_predictions: List[PlayerPrediction],
    bank_millions: float = 0.5,
    top_n: int = 5,
) -> List[TransferSuggestion]:
    base_opt = optimize_starting_xi(squad_predictions)
    base_xp = base_opt.total_expected_points
    squad_ids = {p.player.id for p in squad_predictions}
    club_counts: Dict[str, int] = {}
    for p in squad_predictions:
        club_counts[p.player.team_short] = club_counts.get(p.player.team_short, 0) + 1

    suggestions: List[TransferSuggestion] = []
    for idx, p_out in enumerate(squad_predictions):
        max_price = round(p_out.player.price + bank_millions, 1)
        for p_in in all_predictions:
            if p_in.player.id in squad_ids or p_in.player.position != p_out.player.position:
                continue
            if p_in.player.price > max_price or not is_selectable(p_in):
                continue
            new_cnt = club_counts.get(p_in.player.team_short, 0) - (1 if p_in.player.team_short == p_out.player.team_short else 0)
            if new_cnt >= 3:
                continue
            new_squad = list(squad_predictions)
            new_squad[idx] = p_in
            new_opt = optimize_starting_xi(new_squad)
            gain = round(new_opt.total_expected_points - base_xp, 2)
            if gain > 0.2:
                suggestions.append(TransferSuggestion(p_out, p_in, round(p_in.player.price - p_out.player.price, 1), gain, new_opt.total_expected_points))
    suggestions.sort(key=lambda s: s.xp_gain_this_week, reverse=True)
    return suggestions[:top_n]


# --------------------------------------------------------------------------------------
# MILP-backed squad building & transfer planning
# --------------------------------------------------------------------------------------
def to_plan_players(preds: Sequence[PlayerPrediction], sell_prices: Optional[Dict[int, float]] = None) -> List[PlanPlayer]:
    sell_prices = sell_prices or {}
    return [
        PlanPlayer(
            id=p.player.id,
            name=p.player.web_name,
            team=p.player.team_short,
            position=p.player.position,
            price=p.player.price,
            sell_price=sell_prices.get(p.player.id, p.player.price),
            xp_by_gw=dict(p.gw_xps),
            ownership=p.player.selected_by_percent,
            haul_prob=p.haul_prob,
            xp_uncertainty=p.xp_uncertainty,
        )
        for p in preds
    ]


def _target_gw(preds: Sequence[PlayerPrediction]) -> int:
    for p in preds:
        if p.gw_xps:
            return min(p.gw_xps)
    return 0


def _result_from_week_plan(week: WeekPlan, by_id: Dict[int, PlayerPrediction]) -> OptimizedSquadResult:
    starters = [by_id[pid] for pid in week.lineup_ids]
    bench = [by_id[pid] for pid in week.bench_order]
    captain, vice = by_id[week.captain_id], by_id[week.vice_id]
    counts = [sum(p.player.position == pos for p in starters) for pos in ("DEF", "MID", "FWD")]
    gap = f"{week.solver_gap:.2%}" if week.solver_gap is not None else "unknown"
    return OptimizedSquadResult(
        formation="-".join(map(str, counts)), starting_xi=starters,
        bench_gkp=next(p for p in bench if p.player.position == "GKP"),
        bench_outfield=[p for p in bench if p.player.position != "GKP"],
        captain=captain, vice_captain=vice,
        total_expected_points=round(week.expected_points, 1),
        total_floor=sum(p.floor_points for p in starters) + captain.floor_points,
        total_ceiling=sum(p.ceiling_points for p in starters) + captain.ceiling_points,
        tactical_notes=[f"Solver: {week.solver_status} | gap {gap} | {week.candidate_count} candidates. "
                        "Optimality applies only to this candidate pool."],
    )


def get_best_squad_of_the_week(
    all_predictions: Optional[List[PlayerPrediction]] = None,
    budget: float = 100.0,
    max_per_club: int = 3,
    horizon_gws: int = 1,
) -> Tuple[OptimizedSquadResult, float]:
    """Exact best 15-man squad under the budget (MILP). `horizon_gws` > 1 scores the same
    squad against a weighted run of upcoming gameweeks."""
    if all_predictions is None:
        from news_scraper import load_all_news

        articles = load_all_news()
        all_predictions = [predict_player(p, articles) for p in load_players()]
    pool_preds = [p for p in all_predictions if is_selectable(p, 0.2)]
    gw = _target_gw(pool_preds)
    gws = [gw + k for k in range(horizon_gws)]
    plan_players = select_candidate_pool(to_plan_players(pool_preds), gws)
    hw = {gw + k: 0.85**k for k in range(horizon_gws)} if horizon_gws > 1 else None
    week = solve_squad(plan_players, gw, budget=budget, horizon_weights=hw, max_per_club=max_per_club)
    if week is None:
        raise RuntimeError("Squad MILP infeasible")
    by_id = {p.player.id: p for p in all_predictions}
    res = _result_from_week_plan(week, by_id)
    cost = round(sum(by_id[pid].player.price for pid in week.squad_ids), 1)
    res.tactical_notes.insert(0, f"Full 15-man squad cost: £{cost:.1f}m / £{budget:.1f}m | max {max_per_club} per club.")
    return res, cost


@dataclass
class SquadVariation:
    variation_id: int
    name: str
    subtitle: str
    result: OptimizedSquadResult
    total_cost: float
    budget: float


def get_shuffle_squads(all_predictions: List[PlayerPrediction], budget: float = 100.0,
                       num_squads: int = 120) -> List[SquadVariation]:
    """A sampled library of distinct, high-scoring XIs for the web shuffler."""
    from squad_shuffle import generate_shuffle_pool

    eligible = [p for p in all_predictions if is_selectable(p, .2)]
    by_id = {p.player.id: p for p in eligible}
    weeks = generate_shuffle_pool(to_plan_players(eligible), _target_gw(eligible), budget, num_squads)
    squads = []
    for i, week in enumerate(weeks):
        result = _result_from_week_plan(week, by_id)
        result.tactical_notes = [
            f"15 players, £{budget:.1f}m budget, at most three per club; legal XI and captain.",
            "Random player swaps generate alternatives; the original model scores are never randomized. "
            "For each squad, the best legal XI and captain are selected using this gameweek's projections.",
            f"This is a bounded sample from {week.candidate_count} shortlisted players, not every possible team. "
            "The initial squad maximizes projected XI plus captain points within that shortlist; alternatives stay within five points of it.",
            "Points are XI plus the extra captain score, with no bench, automatic-substitution or vice-captain bonus added. "
            "These are fresh £100m squads, not transfers from your existing team.",
        ]
        cost = round(sum(by_id[pid].player.price for pid in week.squad_ids), 1)
        squads.append(SquadVariation(i + 1, "Top projected squad" if i == 0 else "Shuffled squad",
                                     "A fresh £100m squad for this gameweek. Captain points included.",
                                     result, cost, budget))
    return squads


def get_best_squad_variations(
    all_predictions: Optional[List[PlayerPrediction]] = None,
    budget: float = 100.0,
    max_per_club: int = 3,
    num_variations: int = 6,
) -> List[SquadVariation]:
    """Six data-driven squad archetypes, each an exact MILP optimum under its own constraint."""
    if all_predictions is None:
        from news_scraper import load_all_news

        articles = load_all_news()
        all_predictions = [predict_player(p, articles) for p in load_players()]
    pool_preds = [p for p in all_predictions if is_selectable(p, 0.2)]
    by_id = {p.player.id: p for p in all_predictions}
    gw = _target_gw(pool_preds)
    gws5 = [gw + k for k in range(HORIZON_GWS)]
    base_pool = to_plan_players(pool_preds)

    premium_ids = [p.player.id for p in pool_preds if p.player.price >= 11.0]
    popular_ids = [p.player.id for p in pool_preds if p.player.selected_by_percent >= 15.0]
    hw3 = {gw: 1.0, gw + 1: 0.8, gw + 2: 0.65}
    hw5 = {gw + k: 0.85**k for k in range(HORIZON_GWS)}

    archetypes = [
        dict(name="Variation 1: This Gameweek", subtitle=f"Single-gameweek squad under £{budget:.1f}m, optimized within the candidate pool", kw=dict()),
        dict(name="Variation 2: 5-Gameweek Fixture Run", subtitle="Same 15 scored over the next five gameweeks (decaying weights) — a squad to hold, not a Free Hit", kw=dict(horizon_weights=hw5)),
        dict(name="Variation 3: No Premiums", subtitle="Every player under £11.0m — spreads the budget instead of one superstar", kw=dict(exclude=premium_ids)),
        dict(name="Variation 4: Clean-Sheet Fortress", subtitle="At least four defenders start; scored over the next three gameweeks", kw=dict(min_def_in_xi=4, horizon_weights=hw3)),
        dict(name="Variation 5: Differentials", subtitle="Nobody owned by more than 15% of managers — rank-climbing upside", kw=dict(exclude=popular_ids)),
        dict(name="Variation 6: Strong Bench", subtitle="Bench points weighted at 50% — Bench Boost ready and rotation-proof", kw=dict(bench_weight=0.5, horizon_weights=hw3)),
    ]

    variations: List[SquadVariation] = []
    for idx, arch in enumerate(archetypes[:num_variations], start=1):
        kw = dict(arch["kw"])
        excluded = set(kw.get("exclude", []))
        pool = select_candidate_pool([p for p in base_pool if p.id not in excluded], gws5 if kw.get("horizon_weights") else [gw])
        week = solve_squad(pool, gw, budget=budget, max_per_club=max_per_club, **kw)
        if week is None:
            continue
        res = _result_from_week_plan(week, by_id)
        cost = round(sum(by_id[pid].player.price for pid in week.squad_ids), 1)
        res.tactical_notes.insert(0, f"£{cost:.1f}m / £{budget:.1f}m | {arch['subtitle']}")
        variations.append(SquadVariation(idx, arch["name"], arch["subtitle"], res, cost, budget))
    return variations


@dataclass
class TransferPlanView:
    plan: TransferPlan
    by_id: Dict[int, PlayerPrediction]
    target_gw: int

    def describe(self) -> List[str]:
        lines: List[str] = []
        for w in self.plan.weeks:
            ins = ", ".join(f"{self.by_id[i].player.web_name} (£{self.by_id[i].player.price:.1f}m)" for i in w.transfers_in) or "none"
            outs = ", ".join(f"{self.by_id[i].player.web_name}" for i in w.transfers_out) or "none"
            cap = self.by_id[w.captain_id].player.web_name
            hit = f" | hit -{4 * w.hits}" if w.hits else ""
            lines.append(
                f"GW {w.gw}: FT={w.free_transfers_before} | OUT: {outs} | IN: {ins}{hit} | C: {cap} | XI xP {w.expected_points:.1f}"
            )
        lines.append(f"Plan objective (xP minus hits): {self.plan.objective:.1f} | total hits: {self.plan.total_hits}")
        gap = f"{self.plan.solver_gap:.2%}" if self.plan.solver_gap is not None else "unknown"
        lines.append(f"Solver: {self.plan.solver_status} | gap {gap} | {self.plan.candidate_count} candidates")
        return lines


def build_transfer_plan(
    squad_predictions: List[PlayerPrediction],
    all_predictions: List[PlayerPrediction],
    bank_millions: float,
    free_transfers: int,
    horizon: int = 4,
    wildcard_now: bool = False,
    sell_prices: Optional[Dict[int, float]] = None,
) -> Optional[TransferPlanView]:
    """Multi-gameweek transfer plan for your actual squad (exact MILP)."""
    squad_ids = [p.player.id for p in squad_predictions]
    sell_prices = sell_prices or {}
    budget = sum(sell_prices.get(p.player.id, p.player.price) for p in squad_predictions) + bank_millions
    gw = _target_gw(all_predictions)
    gws = [gw + k for k in range(horizon)]
    pool_preds = [p for p in all_predictions if is_selectable(p, 0.2) or p.player.id in squad_ids]
    plan_players = select_candidate_pool(to_plan_players(pool_preds, sell_prices), gws, keep_ids=squad_ids)
    plan = plan_transfers(
        plan_players,
        squad_ids,
        gws,
        budget=round(budget, 1),
        free_transfers=max(0, min(5, free_transfers)),
        wildcard_week_index=0 if wildcard_now else None,
    )
    if plan is None:
        return None
    by_id = {p.player.id: p for p in all_predictions}
    return TransferPlanView(plan=plan, by_id=by_id, target_gw=gw)
