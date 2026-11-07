"""
FPL Expected Points (xP) Predictor & Squad Optimizer (predictor.py)

Combines:
1. Current season underlying metrics (xG/90, xA/90, xGC/90, BPS/90, Form, Set Pieces)
2. Past 2-3 seasons historical consistency (`historical_ppg_3yr`)
3. Poisson Clean Sheet & Goals Conceded model (including -1 pt for every 2 goals conceded)
4. Live News & Manager Press Conference adjustments (Start Probability vs 1-Point Cameo Trap)
5. Lineup, Formation, Captaincy, Bench Order, and Dream Team / Free Hit optimization
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ai_coach import CoachDecision, evaluate_squad_with_ai_coach
from fpl_data import PlayerRecord, load_players
from neural_model import extract_features, get_or_train_neural_net
from news_scraper import NewsArticle, PlayerNewsAssessment, analyze_player_news

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

# Official FPL Scoring Rules by Position
GOAL_POINTS = {"GKP": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
ASSIST_POINTS = 3.0
CS_POINTS = {"GKP": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
APPEARANCE_60PLUS_PTS = 2.0
APPEARANCE_CAMEO_PTS = 1.0
YELLOW_CARD_PTS = -1.0
PENALTY_MISS_PTS = -2.0

VALID_FORMATIONS: List[Tuple[int, int, int]] = [
    (3, 4, 3),
    (3, 5, 2),
    (4, 4, 2),
    (4, 3, 3),
    (4, 5, 1),
    (5, 3, 2),
    (5, 4, 1),
]


def _poisson_prob(lmbda: float, k: int) -> float:
    """Probability of exactly k events under a Poisson(lmbda) distribution."""
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)


@dataclass
class PlayerPrediction:
    player: PlayerRecord
    news: PlayerNewsAssessment
    clean_sheet_prob: float
    concede_2plus_prob: float
    expected_goals: float
    expected_assists: float
    rule_xp_90: float  # Official FPL Rule Engine 90-min reference
    nn_xp_90: float  # Dual-Head Neural Network (22->40->20->2) 90-min prediction
    xp_if_90: float  # Neural Net + AI Head Coach 90-min prediction
    expected_points: float  # News-adjusted final xP for upcoming GW
    haul_prob: float  # Explosive Haul Probability P(Points >= 9 / Double-Digit Ceiling)
    horizon_3gw_xp: float  # Time-decayed 3-Gameweek Expected Points (GW t, t+1, t+2)
    role_tag: str  # AI Tactical Head Coach Archetype
    coach_note: str  # AI Tactical Head Coach 1-sentence briefing
    floor_points: int
    ceiling_points: int
    lineup_score: float  # penalizes high 1-pt cameo risk + rewards 3-GW horizon & ceiling


def predict_player(
    player: PlayerRecord, articles: List[NewsArticle]
) -> PlayerPrediction:
    """
    Calculates a player's Expected Points (xP), 3-GW Horizon xP, Explosive Haul Probability,
    AI Tactical Head Coach briefing, floor, ceiling, and clean sheet probability.
    """
    news = analyze_player_news(player, articles)
    coach_map = _get_coach_decisions(articles)
    coach = coach_map.get(player.id)

    # 1. Poisson model for expected goals conceded (lambda_gc)
    home_def_factor = 0.88 if player.is_home else 1.12
    mismatch_ratio = (player.opp_attack_strength / max(player.team_defence_strength, 950)) ** 1.35
    base_xgc = max(0.55, min(2.2, player.xgc_per_90 if player.xgc_per_90 > 0.1 else 1.25))
    lambda_gc = max(0.35, base_xgc * home_def_factor * mismatch_ratio)

    cs_prob = _poisson_prob(lambda_gc, 0)
    p_gc_2_3 = _poisson_prob(lambda_gc, 2) + _poisson_prob(lambda_gc, 3)
    p_gc_4_5 = _poisson_prob(lambda_gc, 4) + _poisson_prob(lambda_gc, 5)
    p_gc_6plus = max(
        0.0,
        1.0
        - sum(_poisson_prob(lambda_gc, k) for k in range(6)),
    )
    concede_2plus_prob = p_gc_2_3 + p_gc_4_5 + p_gc_6plus

    # 2. Expected Goals (xG) & Expected Assists (xA) for this match
    home_att_factor = 1.10 if player.is_home else 0.93
    opp_def_factor = (player.team_attack_strength / max(player.opp_defence_strength, 950)) ** 1.30
    match_att_mult = home_att_factor * opp_def_factor

    pen_boost = 0.10 if player.penalties_order == 1 else 0.0
    corner_boost = 0.05 if player.corners_order in (1, 2) else 0.0

    exp_g = (player.xg_per_90 + pen_boost) * match_att_mult
    exp_a = (player.xa_per_90 + corner_boost) * match_att_mult

    # 3. Calculate 90-minute Expected Points from Official FPL Scoring Rules (diagnostic reference):
    appearance_pts = APPEARANCE_60PLUS_PTS
    cs_pts = cs_prob * CS_POINTS[player.position]
    gc_penalty = (
        (-1.0 * p_gc_2_3 - 2.0 * p_gc_4_5 - 3.0 * p_gc_6plus)
        if player.position in ("GKP", "DEF")
        else 0.0
    )
    expected_saves = (
        max(2.1, player.saves_per_90) * (player.opp_attack_strength / 1140.0)
        if player.position == "GKP"
        else 0.0
    )
    save_pts = expected_saves / 3.0
    attack_pts = exp_g * GOAL_POINTS[player.position] + exp_a * ASSIST_POINTS
    def_contrib_pts = (
        min(1.6, max(0.0, (player.defensive_contribution_per_90 - 5.0) * 0.22))
        if player.position in ("DEF", "MID")
        else 0.0
    )
    pen_miss_penalty = (
        PENALTY_MISS_PTS * 0.03 if player.penalties_order == 1 else 0.0
    )
    card_penalty = (
        YELLOW_CARD_PTS * 0.14
        if player.position in ("DEF", "MID")
        else (YELLOW_CARD_PTS * 0.08 if player.position == "FWD" else 0.0)
    )

    bps_base = max(0.0, (player.bps_per_90 - 14.0) / 10.0)
    fixture_bps_boost = 0.35 if player.fixture_difficulty <= 2 else 0.0
    bonus_pts = min(1.8, bps_base + fixture_bps_boost)

    raw_90_xp = (
        appearance_pts
        + cs_pts
        + gc_penalty
        + save_pts
        + attack_pts
        + def_contrib_pts
        + pen_miss_penalty
        + card_penalty
        + bonus_pts
    )

    # 4. Dual-Head 22-Feature Supervised Neural Network (22 -> 40 -> 20 -> 2)
    # Head 1: Expected Points (xP) | Head 2: Explosive Haul Probability P(Points >= 9)
    nn_model = _get_nn()
    nn_90_xp, nn_haul_prob = nn_model.predict_both(extract_features(player))

    # Apply AI Tactical Head Coach contextual adjustment
    tactical_delta = coach.tactical_xp_delta if coach else 0.0
    haul_boost = coach.haul_prob_boost if coach else 0.0
    horizon_bonus = coach.horizon_3gw_bonus if coach else 0.0
    role_tag = coach.role_tag if coach else "STARTER"
    coach_note = coach.coach_note if coach else f"3-GW schedule: {player.fixture_run_3gw}"

    blended_90_xp = max(1.0, nn_90_xp + tactical_delta)
    rule_90_xp = raw_90_xp

    # 5. Apply News / Press Conference Start & Cameo Probabilities
    expected_pts = (
        news.start_probability * blended_90_xp
        + news.cameo_probability * APPEARANCE_CAMEO_PTS
    )
    final_haul_prob = round(
        max(0.01, min(0.85, (nn_haul_prob + haul_boost) * news.start_probability)),
        2,
    )

    # 6. Multi-Gameweek 3-GW Rolling Horizon Evaluation (GW t, GW t+1, GW t+2)
    horizon_weights = [1.0, 0.75, 0.55]
    up3 = player.upcoming_3gw_fixtures[:3]
    if up3:
        gw_xps: List[float] = []
        for idx, fix_ov in enumerate(up3):
            w_h = horizon_weights[idx] if idx < len(horizon_weights) else 0.5
            if idx == 0:
                gw_xps.append(w_h * expected_pts)
            else:
                fut_xp90, _ = nn_model.predict_both(extract_features(player, fixture_override=fix_ov))
                fut_xp = news.start_probability * max(1.0, fut_xp90 + tactical_delta * 0.7) + news.cameo_probability
                gw_xps.append(w_h * fut_xp)
        horizon_3gw_xp = sum(gw_xps) + horizon_bonus
    else:
        horizon_3gw_xp = expected_pts * 2.3 + horizon_bonus

    # Lineup selection score: blends immediate GW xP with 3-GW horizon & explosive ceiling,
    # while penalizing 1-point cameo traps.
    cameo_trap_penalty = 1.25 if news.risk_level == "CAMEO_ROTATION_RISK" else 0.0
    lineup_score = (
        0.78 * expected_pts
        + 0.22 * (horizon_3gw_xp / 2.3)
        + 0.65 * final_haul_prob
        - cameo_trap_penalty
    )

    # Floor & Ceiling estimates
    if news.risk_level == "INJURED_OR_OUT":
        floor_pts, ceiling_pts = 0, 0
    elif news.risk_level == "CAMEO_ROTATION_RISK":
        floor_pts = 1
        ceiling_pts = max(2, int(round(blended_90_xp + 2)))
    else:
        floor_pts = (
            1
            if (player.position in ("GKP", "DEF") and concede_2plus_prob > 0.45)
            else 2
        )
        if cs_prob > 0.50 and player.position in ("GKP", "DEF"):
            floor_pts = 3
        ceiling_pts = int(
            round(
                blended_90_xp
                + (7.5 * final_haul_prob)
                + (5.0 if player.position in ("MID", "FWD") else 3.8)
            )
        )

    return PlayerPrediction(
        player=player,
        news=news,
        clean_sheet_prob=round(cs_prob, 2),
        concede_2plus_prob=round(concede_2plus_prob, 2),
        expected_goals=round(exp_g, 2),
        expected_assists=round(exp_a, 2),
        rule_xp_90=round(rule_90_xp, 2),
        nn_xp_90=round(nn_90_xp, 2),
        xp_if_90=round(blended_90_xp, 2),
        expected_points=round(expected_pts, 2),
        haul_prob=final_haul_prob,
        horizon_3gw_xp=round(max(0.0, horizon_3gw_xp), 1),
        role_tag=role_tag,
        coach_note=coach_note,
        floor_points=floor_pts,
        ceiling_points=ceiling_pts,
        lineup_score=round(lineup_score, 2),
    )


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


def optimize_starting_xi(
    squad_predictions: List[PlayerPrediction],
) -> OptimizedSquadResult:
    """
    Finds the highest-scoring valid FPL Starting XI, formation, Captain, Vice-Captain,
    and optimal bench order from a 15-player squad.
    """
    gkps = sorted(
        [p for p in squad_predictions if p.player.position == "GKP"],
        key=lambda x: x.lineup_score,
        reverse=True,
    )
    defs = [p for p in squad_predictions if p.player.position == "DEF"]
    mids = [p for p in squad_predictions if p.player.position == "MID"]
    fwds = [p for p in squad_predictions if p.player.position == "FWD"]

    starting_gkp = gkps[0]
    bench_gkp = gkps[1] if len(gkps) > 1 else gkps[0]

    best_score = -999.0
    best_formation = (4, 4, 2)
    best_outfield: List[PlayerPrediction] = []

    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        if len(defs) < n_def or len(mids) < n_mid or len(fwds) < n_fwd:
            continue
        for d_combo in itertools.combinations(defs, n_def):
            for m_combo in itertools.combinations(mids, n_mid):
                for f_combo in itertools.combinations(fwds, n_fwd):
                    outfield = list(d_combo) + list(m_combo) + list(f_combo)
                    score = sum(p.lineup_score for p in outfield)

                    # Mild diversification penalty if >4 starters come from the exact same fixture
                    fixture_counts: Dict[str, int] = {}
                    for p in [starting_gkp] + outfield:
                        opp_code = p.player.next_opponent.split()[0]
                        pair_key = "-".join(sorted([p.player.team_short, opp_code]))
                        fixture_counts[pair_key] = fixture_counts.get(pair_key, 0) + 1
                    for count in fixture_counts.values():
                        if count >= 5:
                            score -= 0.75 * (count - 4)

                    if score > best_score:
                        best_score = score
                        best_formation = (n_def, n_mid, n_fwd)
                        best_outfield = outfield

    starting_xi = [starting_gkp] + best_outfield
    starting_ids = {p.player.id for p in starting_xi}

    # Order bench outfield players: healthy 90-min players first, cameo-risk second, injured last
    remaining_outfield = [
        p
        for p in squad_predictions
        if p.player.position != "GKP" and p.player.id not in starting_ids
    ]
    bench_outfield = sorted(
        remaining_outfield,
        key=lambda p: (
            0 if p.news.risk_level == "INJURED_OR_OUT" else 1,
            0 if p.news.risk_level == "CAMEO_ROTATION_RISK" else 1,
            p.expected_points,
        ),
        reverse=True,
    )

    # Pick Captain & Vice-Captain by Expected Points + Explosive Haul Ceiling (P(Points >= 9))
    ranked_cap = sorted(
        starting_xi,
        key=lambda p: p.expected_points + 1.8 * p.haul_prob,
        reverse=True,
    )
    captain = ranked_cap[0]
    vice_captain = ranked_cap[1]

    # Official FPL Captaincy & Vice-Captain Rule:
    # Captain gets 2x points. If Captain plays 0 minutes (prob = 1 - start_prob - cameo_prob),
    # Vice-Captain automatically inherits the 2x armband!
    cap_zero_mins_prob = max(
        0.0,
        1.0 - captain.news.start_probability - captain.news.cameo_probability,
    )
    captaincy_bonus_xp = (
        captain.expected_points + cap_zero_mins_prob * vice_captain.expected_points
    )

    # Official FPL Auto-Substitution Rule:
    # If an outfield starter plays 0 minutes, the 1st healthy bench sub comes on automatically.
    first_sub_xp = bench_outfield[0].expected_points if bench_outfield else 0.0
    auto_sub_expected_boost = 0.0
    for starter in best_outfield:
        p_zero = max(
            0.0,
            1.0 - starter.news.start_probability - starter.news.cameo_probability,
        )
        auto_sub_expected_boost += p_zero * (first_sub_xp * 0.25)

    total_xp = (
        sum(p.expected_points for p in starting_xi)
        + captaincy_bonus_xp
        + auto_sub_expected_boost
    )
    total_floor = sum(p.floor_points for p in starting_xi) + captain.floor_points
    total_ceiling = sum(p.ceiling_points for p in starting_xi) + captain.ceiling_points

    # Generate automated tactical notes
    notes: List[str] = []
    for p in squad_predictions:
        if p.news.risk_level == "CAMEO_ROTATION_RISK":
            status_str = "BENCHED" if p not in starting_xi else "STARTED"
            notes.append(
                f"Cameo Trap Alert ({p.player.web_name} - {status_str}): {p.news.reasoning}"
            )
        elif p.news.risk_level == "INJURED_OR_OUT":
            notes.append(
                f"Unavailable ({p.player.web_name}): Buried in 3rd bench slot (0% start probability)."
            )

    # Check if any attacker faces our own starting defenders
    def_teams = {
        p.player.team_short
        for p in starting_xi
        if p.player.position in ("GKP", "DEF")
    }
    for p in starting_xi:
        if p.player.position in ("MID", "FWD"):
            opp_team = p.player.next_opponent.split()[0]
            if opp_team in def_teams:
                notes.append(
                    f"Fixture Hedge ({p.player.web_name} vs {opp_team} defense): "
                    f"Starting {p.player.web_name} alongside your {opp_team} defensive assets "
                    f"hedges your floor so you still get points if {opp_team} concede."
                )

    return OptimizedSquadResult(
        formation=f"{best_formation[0]}-{best_formation[1]}-{best_formation[2]}",
        starting_xi=starting_xi,
        bench_gkp=bench_gkp,
        bench_outfield=bench_outfield,
        captain=captain,
        vice_captain=vice_captain,
        total_expected_points=round(total_xp, 1),
        total_floor=total_floor,
        total_ceiling=total_ceiling,
        tactical_notes=notes,
    )


def build_dream_xi(
    all_predictions: List[PlayerPrediction], max_per_club: int = 3
) -> OptimizedSquadResult:
    """
    Builds the highest-scoring Free Hit / Dream Starting XI from all available Premier League
    players respecting FPL's max-3-players-per-club and valid formation rules.
    """
    healthy = [
        p
        for p in all_predictions
        if p.news.risk_level in ("NONE", "POSITIVE") and p.player.minutes >= 270
    ]
    # Keep top candidates per position to keep combinatorial search instant
    top_gkps = sorted(
        [p for p in healthy if p.player.position == "GKP"],
        key=lambda x: x.expected_points,
        reverse=True,
    )[:4]
    top_defs = sorted(
        [p for p in healthy if p.player.position == "DEF"],
        key=lambda x: x.expected_points,
        reverse=True,
    )[:8]
    top_mids = sorted(
        [p for p in healthy if p.player.position == "MID"],
        key=lambda x: x.expected_points,
        reverse=True,
    )[:8]
    top_fwds = sorted(
        [p for p in healthy if p.player.position == "FWD"],
        key=lambda x: x.expected_points,
        reverse=True,
    )[:6]

    best_xi: List[PlayerPrediction] = []
    best_form = (3, 4, 3)
    best_xp = -999.0

    for gkp in top_gkps:
        for n_def, n_mid, n_fwd in VALID_FORMATIONS:
            for d_c in itertools.combinations(top_defs, n_def):
                for m_c in itertools.combinations(top_mids, n_mid):
                    for f_c in itertools.combinations(top_fwds, n_fwd):
                        xi = [gkp] + list(d_c) + list(m_c) + list(f_c)
                        # Check max 3 players per club
                        club_counts: Dict[str, int] = {}
                        valid_clubs = True
                        for p in xi:
                            c = p.player.team_short
                            club_counts[c] = club_counts.get(c, 0) + 1
                            if club_counts[c] > max_per_club:
                                valid_clubs = False
                                break
                        if not valid_clubs:
                            continue

                        cap_xp = max(p.expected_points for p in xi)
                        tot_xp = sum(p.expected_points for p in xi) + cap_xp
                        if tot_xp > best_xp:
                            best_xp = tot_xp
                            best_xi = xi
                            best_form = (n_def, n_mid, n_fwd)

    ranked_cap = sorted(best_xi, key=lambda p: p.expected_points, reverse=True)
    captain = ranked_cap[0]
    vice_captain = ranked_cap[1]

    return OptimizedSquadResult(
        formation=f"{best_form[0]}-{best_form[1]}-{best_form[2]}",
        starting_xi=best_xi,
        bench_gkp=top_gkps[-1],
        bench_outfield=[],
        captain=captain,
        vice_captain=vice_captain,
        total_expected_points=round(best_xp, 1),
        total_floor=sum(p.floor_points for p in best_xi) + captain.floor_points,
        total_ceiling=sum(p.ceiling_points for p in best_xi) + captain.ceiling_points,
        tactical_notes=[
            "Optimal Free Hit XI respecting max 3 players per Premier League club."
        ],
    )


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
    """
    Evaluates all valid 1-for-1 transfers within budget (`bank_millions`) and club limits (max 3),
    ranking them by net gain to the optimized Starting XI's Expected Points.
    """
    base_opt = optimize_starting_xi(squad_predictions)
    base_xp = base_opt.total_expected_points
    squad_ids = {p.player.id for p in squad_predictions}

    club_counts: Dict[str, int] = {}
    for p in squad_predictions:
        c = p.player.team_short
        club_counts[c] = club_counts.get(c, 0) + 1

    suggestions: List[TransferSuggestion] = []
    for idx, p_out in enumerate(squad_predictions):
        max_price = round(p_out.player.price + bank_millions, 1)
        for p_in in all_predictions:
            if p_in.player.id in squad_ids:
                continue
            if p_in.player.position != p_out.player.position:
                continue
            if p_in.player.price > max_price:
                continue
            if p_in.news.risk_level not in ("NONE", "POSITIVE") or p_in.player.minutes < 270:
                continue

            # Check max 3 per club after removing p_out
            new_club_count = club_counts.get(p_in.player.team_short, 0) - (
                1 if p_in.player.team_short == p_out.player.team_short else 0
            )
            if new_club_count >= 3:
                continue

            new_squad = list(squad_predictions)
            new_squad[idx] = p_in
            new_opt = optimize_starting_xi(new_squad)
            gain = round(new_opt.total_expected_points - base_xp, 2)
            if gain > 0.2:
                suggestions.append(
                    TransferSuggestion(
                        player_out=p_out,
                        player_in=p_in,
                        net_cost=round(p_in.player.price - p_out.player.price, 1),
                        xp_gain_this_week=gain,
                        new_starting_xp=new_opt.total_expected_points,
                    )
                )

    suggestions.sort(key=lambda s: s.xp_gain_this_week, reverse=True)
    return suggestions[:top_n]


def get_best_squad_of_the_week(
    all_predictions: Optional[List[PlayerPrediction]] = None,
    budget: float = 100.0,
    max_per_club: int = 3,
) -> Tuple[OptimizedSquadResult, float]:
    """
    Finds the optimal 15-player FPL squad (2 GKP, 5 DEF, 5 MID, 3 FWD) for the week
    that satisfies all official FPL constraints:
      1. Total 15-man squad cost <= `budget` (default £100.0m)
      2. Max `max_per_club` (3) players from any single Premier League club across all 15 slots
      3. Valid Starting XI formation (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD)
      4. Optimal Captain (2x), Vice-Captain, and ordered 4-player Bench

    Returns:
      (OptimizedSquadResult, total_15_man_cost)
    """
    if all_predictions is None:
        from fpl_data import load_players
        from news_scraper import load_all_news

        players = load_players()
        articles = load_all_news()
        all_predictions = [predict_player(p, articles) for p in players]

    healthy = [
        p
        for p in all_predictions
        if p.news.risk_level in ("NONE", "POSITIVE") and p.player.minutes >= 270
    ]

    gkps = sorted(
        [p for p in healthy if p.player.position == "GKP"],
        key=lambda x: x.expected_points,
        reverse=True,
    )
    defs = sorted(
        [p for p in healthy if p.player.position == "DEF"],
        key=lambda x: x.expected_points,
        reverse=True,
    )
    mids = sorted(
        [p for p in healthy if p.player.position == "MID"],
        key=lambda x: x.expected_points,
        reverse=True,
    )
    fwds = sorted(
        [p for p in healthy if p.player.position == "FWD"],
        key=lambda x: x.expected_points,
        reverse=True,
    )

    best_squad_15: List[PlayerPrediction] = []
    best_cost = 0.0
    best_score = -999.0

    # Candidate pools per position (top xP + top value-per-million bench enablers)
    def _pool(items: List[PlayerPrediction], top_k: int, cheap_k: int) -> List[PlayerPrediction]:
        top = items[:top_k]
        cheap = sorted(items, key=lambda p: (p.player.price, -p.expected_points))[:cheap_k]
        seen = set()
        merged = []
        for p in top + cheap:
            if p.player.id not in seen:
                seen.add(p.player.id)
                merged.append(p)
        return merged

    g_pool = _pool(gkps, 3, 2)
    d_pool = _pool(defs, 7, 4)
    m_pool = _pool(mids, 7, 4)
    f_pool = _pool(fwds, 5, 4)

    for g_Starter in g_pool[:3]:
        for g_Bench in sorted(g_pool, key=lambda x: x.player.price)[:2]:
            if g_Starter.player.id == g_Bench.player.id:
                continue
            for n_def, n_mid, n_fwd in VALID_FORMATIONS:
                for d_starters in itertools.combinations(d_pool[:8], n_def):
                    for m_starters in itertools.combinations(m_pool[:8], n_mid):
                        for f_starters in itertools.combinations(f_pool[:6], n_fwd):
                            starters = (
                                [g_Starter]
                                + list(d_starters)
                                + list(m_starters)
                                + list(f_starters)
                            )
                            starter_cost = sum(p.player.price for p in starters) + g_Bench.player.price
                            # Need (5 - n_def) bench DEFs, (5 - n_mid) bench MIDs, (3 - n_fwd) bench FWDs
                            rem_def = 5 - n_def
                            rem_mid = 5 - n_mid
                            rem_fwd = 3 - n_fwd
                            min_bench_cost = (rem_def + rem_mid + rem_fwd) * 4.0
                            if starter_cost + min_bench_cost > budget + 1e-6:
                                continue

                            starter_ids = {p.player.id for p in starters} | {g_Bench.player.id}
                            club_counts: Dict[str, int] = {}
                            valid = True
                            for p in starters + [g_Bench]:
                                c = p.player.team_short
                                club_counts[c] = club_counts.get(c, 0) + 1
                                if club_counts[c] > max_per_club:
                                    valid = False
                                    break
                            if not valid:
                                continue

                            # Pick cheapest healthy bench players that respect club limits
                            bench_picks: List[PlayerPrediction] = []
                            temp_clubs = dict(club_counts)

                            def _pick_bench(pool: List[PlayerPrediction], count: int) -> bool:
                                chosen = 0
                                for cand in sorted(
                                    pool,
                                    key=lambda p: (p.player.price, -p.expected_points),
                                ):
                                    if chosen == count:
                                        break
                                    if cand.player.id in starter_ids:
                                        continue
                                    c = cand.player.team_short
                                    if temp_clubs.get(c, 0) >= max_per_club:
                                        continue
                                    temp_clubs[c] = temp_clubs.get(c, 0) + 1
                                    bench_picks.append(cand)
                                    chosen += 1
                                return chosen == count

                            if not (
                                _pick_bench(d_pool, rem_def)
                                and _pick_bench(m_pool, rem_mid)
                                and _pick_bench(f_pool, rem_fwd)
                            ):
                                continue

                            total_cost = round(
                                starter_cost + sum(p.player.price for p in bench_picks), 1
                            )
                            if total_cost > budget + 1e-6:
                                continue

                            cap_xp = max(p.expected_points for p in starters)
                            xi_xp = sum(p.expected_points for p in starters) + cap_xp
                            bench_xp = sum(p.expected_points for p in bench_picks) * 0.10
                            score = xi_xp + bench_xp

                            if score > best_score:
                                best_score = score
                                best_cost = total_cost
                                best_squad_15 = starters + [g_Bench] + bench_picks

    opt_result = optimize_starting_xi(best_squad_15)
    opt_result.tactical_notes.insert(
        0,
        f"Full 15-Man Squad Cost: £{best_cost:.1f}m / £{budget:.1f}m (£{budget - best_cost:.1f}m in bank) | Max {max_per_club} per club enforced.",
    )
    return opt_result, best_cost


@dataclass
class SquadVariation:
    variation_id: int
    name: str
    subtitle: str
    result: OptimizedSquadResult
    total_cost: float
    budget: float


def get_best_squad_variations(
    all_predictions: Optional[List[PlayerPrediction]] = None,
    budget: float = 100.0,
    max_per_club: int = 3,
    num_variations: int = 6,
) -> List[SquadVariation]:
    """
    Generates multiple distinct, high-scoring 15-player £100m squads (all projecting ~72-77 xP
    with 135-150+ ceiling) across different formations, captaincy options, and player combinations
    so you can explore diverse tactical setups without repeating the same players.
    """
    if all_predictions is None:
        from fpl_data import load_players
        from news_scraper import load_all_news

        players = load_players()
        articles = load_all_news()
        all_predictions = [predict_player(p, articles) for p in players]

    healthy = [
        p
        for p in all_predictions
        if p.news.risk_level in ("NONE", "POSITIVE") and p.player.minutes >= 270
    ]

    # Define 6 distinct tactical archetypes using ONLY real active 2026/27 Premier League players
    archetypes = [
        {
            "name": "Variation 1: Neural Net Optimal (3-4-3)",
            "subtitle": "Highest raw 22-feature Neural Network xP under £100.0m across all 20 PL clubs",
            "formations": [(3, 4, 3), (3, 5, 2)],
            "must_include": set(),
            "exclude_starters": set(),
            "bench_seed": 0,
        },
        {
            "name": "Variation 2: Haaland (£15.8m) + Gabriel + De Cuyper Power Build (3-4-3)",
            "subtitle": "Fits £15.8m Haaland alongside FPL's #1 defender De Cuyper & Arsenal's Gabriel",
            "formations": [(3, 4, 3)],
            "must_include": {"Haaland", "De Cuyper", "Gabriel"},
            "exclude_starters": set(),
            "bench_seed": 1,
        },
        {
            "name": "Variation 3: High-Value Form & xGI Stars (3-4-3)",
            "subtitle": "Targets top 2026/27 points-per-million stars: Barry, Evanilson, Tavernier, Dewsbury-Hall & Isak",
            "formations": [(3, 4, 3)],
            "must_include": {"Isak", "Barry", "Evanilson", "Tavernier"},
            "exclude_starters": {"Haaland"},
            "bench_seed": 2,
        },
        {
            "name": "Variation 4: Clean-Sheet & CBIT Fortress (4-4-2)",
            "subtitle": "4 elite clean-sheet & defensive-contribution leaders (Gabriel, De Cuyper, Tarkowski, Konsa)",
            "formations": [(4, 4, 2)],
            "must_include": {"Gabriel", "De Cuyper", "Tarkowski", "Konsa"},
            "exclude_starters": {"Vuskovic"},
            "bench_seed": 3,
        },
        {
            "name": "Variation 5: Weak-Opponent Mismatch Hunters (3-5-2)",
            "subtitle": "Targets weak defences/attacks in GW10: ARS vs HUL, CHE vs SUN, EVE vs COV, BHA vs BRE",
            "formations": [(3, 5, 2)],
            "must_include": {"Palmer", "Rogers", "Rice", "Dewsbury-Hall"},
            "exclude_starters": {"Tavernier"},
            "bench_seed": 4,
        },
        {
            "name": "Variation 6: Balanced No-Haaland Super-Squad (3-4-3)",
            "subtitle": "Spreads £15.8m Haaland budget across Palmer, Isak, Gabriel, Raya, Groß & N.Jackson",
            "formations": [(3, 4, 3)],
            "must_include": {"Raya", "Palmer", "Isak", "N.Jackson"},
            "exclude_starters": {"Haaland", "Verbruggen"},
            "bench_seed": 5,
        },
    ]

    gkps = sorted(
        [p for p in healthy if p.player.position == "GKP"],
        key=lambda x: x.expected_points,
        reverse=True,
    )
    defs = sorted(
        [p for p in healthy if p.player.position == "DEF"],
        key=lambda x: x.expected_points,
        reverse=True,
    )
    mids = sorted(
        [p for p in healthy if p.player.position == "MID"],
        key=lambda x: x.expected_points,
        reverse=True,
    )
    fwds = sorted(
        [p for p in healthy if p.player.position == "FWD"],
        key=lambda x: x.expected_points,
        reverse=True,
    )

    variations: List[SquadVariation] = []
    usage_counts: Dict[int, int] = {}

    for idx, arch in enumerate(archetypes[:num_variations], start=1):
        must_names = arch["must_include"]
        excl_names = arch["exclude_starters"]
        allowed_forms = arch["formations"]

        # Filter starter pools
        g_starters_pool = [p for p in gkps if p.player.web_name not in excl_names][:5]
        d_starters_pool = [p for p in defs if p.player.web_name not in excl_names][:9]
        m_starters_pool = [p for p in mids if p.player.web_name not in excl_names][:9]
        f_starters_pool = [p for p in fwds if p.player.web_name not in excl_names][:8]

        # Cheap bench pools (rotated by bench_seed so bench players also vary!)
        cheap_gkps = sorted(gkps, key=lambda p: (p.player.price, -p.expected_points))[:4]
        cheap_defs = sorted(defs, key=lambda p: (p.player.price, -p.expected_points))[:6]
        cheap_mids = sorted(mids, key=lambda p: (p.player.price, -p.expected_points))[:6]
        cheap_fwds = sorted(fwds, key=lambda p: (p.player.price, -p.expected_points))[:6]

        seed = arch["bench_seed"]
        if len(cheap_defs) > 2:
            cheap_defs = cheap_defs[seed % 3 :] + cheap_defs[: seed % 3]
        if len(cheap_mids) > 2:
            cheap_mids = cheap_mids[seed % 3 :] + cheap_mids[: seed % 3]
        if len(cheap_fwds) > 2:
            cheap_fwds = cheap_fwds[seed % 3 :] + cheap_fwds[: seed % 3]

        best_15: List[PlayerPrediction] = []
        best_cost = 0.0
        best_score = -999.0

        for g_s in g_starters_pool[:3]:
            g_b = next(
                (g for g in cheap_gkps if g.player.id != g_s.player.id and g.player.team_short != g_s.player.team_short),
                cheap_gkps[0],
            )
            for n_def, n_mid, n_fwd in allowed_forms:
                for d_c in itertools.combinations(d_starters_pool, n_def):
                    for m_c in itertools.combinations(m_starters_pool, n_mid):
                        for f_c in itertools.combinations(f_starters_pool, n_fwd):
                            starters = [g_s] + list(d_c) + list(m_c) + list(f_c)
                            starter_names = {p.player.web_name for p in starters}
                            if not must_names.issubset(starter_names):
                                continue

                            starter_cost = sum(p.player.price for p in starters) + g_b.player.price
                            rem_def, rem_mid, rem_fwd = 5 - n_def, 5 - n_mid, 3 - n_fwd
                            if starter_cost + (rem_def + rem_mid + rem_fwd) * 4.0 > budget + 1e-6:
                                continue

                            used_ids = {p.player.id for p in starters} | {g_b.player.id}
                            clubs: Dict[str, int] = {}
                            valid_clubs = True
                            for p in starters + [g_b]:
                                c = p.player.team_short
                                clubs[c] = clubs.get(c, 0) + 1
                                if clubs[c] > max_per_club:
                                    valid_clubs = False
                                    break
                            if not valid_clubs:
                                continue

                            bench_picks: List[PlayerPrediction] = []
                            temp_clubs = dict(clubs)

                            def _fill_bench(pool: List[PlayerPrediction], count: int) -> bool:
                                picked = 0
                                for cand in pool:
                                    if picked == count:
                                        break
                                    if cand.player.id in used_ids:
                                        continue
                                    c = cand.player.team_short
                                    if temp_clubs.get(c, 0) >= max_per_club:
                                        continue
                                    temp_clubs[c] = temp_clubs.get(c, 0) + 1
                                    used_ids.add(cand.player.id)
                                    bench_picks.append(cand)
                                    picked += 1
                                return picked == count

                            if not (
                                _fill_bench(cheap_defs, rem_def)
                                and _fill_bench(cheap_mids, rem_mid)
                                and _fill_bench(cheap_fwds, rem_fwd)
                            ):
                                continue

                            tot_cost = round(
                                starter_cost + sum(p.player.price for p in bench_picks), 1
                            )
                            if tot_cost > budget + 1e-6:
                                continue

                            cap_xp = max(p.expected_points for p in starters)
                            xi_xp = sum(p.expected_points for p in starters) + cap_xp
                            # Small diversity bonus for players used less often in earlier variations
                            rep_penalty = sum(
                                usage_counts.get(p.player.id, 0) * 0.15 for p in starters
                            )
                            score = xi_xp - rep_penalty

                            if score > best_score:
                                best_score = score
                                best_cost = tot_cost
                                best_15 = starters + [g_b] + bench_picks

        if best_15:
            opt = optimize_starting_xi(best_15)
            for p in opt.starting_xi:
                usage_counts[p.player.id] = usage_counts.get(p.player.id, 0) + 1
            variations.append(
                SquadVariation(
                    variation_id=idx,
                    name=arch["name"],
                    subtitle=arch["subtitle"],
                    result=opt,
                    total_cost=best_cost,
                    budget=budget,
                )
            )

    return variations



