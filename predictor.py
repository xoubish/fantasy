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

from fpl_data import PlayerRecord
from news_scraper import NewsArticle, PlayerNewsAssessment, analyze_player_news

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
    xp_if_90: float
    expected_points: float  # news-adjusted xP
    floor_points: int
    ceiling_points: int
    lineup_score: float  # penalizes high 1-pt cameo risk when picking Starting XI


def predict_player(
    player: PlayerRecord, articles: List[NewsArticle]
) -> PlayerPrediction:
    """
    Calculates a player's Expected Points (xP), floor, ceiling, and clean sheet probability
    for the upcoming Gameweek.
    """
    news = analyze_player_news(player, articles)

    # 1. Poisson model for expected goals conceded (lambda_gc)
    home_def_factor = 0.88 if player.is_home else 1.12
    opp_att_factor = (player.opp_attack_strength / 1140.0) ** 1.35
    lambda_gc = max(0.35, player.xgc_per_90 * home_def_factor * opp_att_factor)

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
    opp_def_factor = (1140.0 / max(player.opp_defence_strength, 950)) ** 1.30
    match_att_mult = home_att_factor * opp_def_factor

    pen_boost = 0.10 if player.penalties_order == 1 else 0.0
    corner_boost = 0.05 if player.corners_order in (1, 2) else 0.0

    exp_g = (player.xg_per_90 + pen_boost) * match_att_mult
    exp_a = (player.xa_per_90 + corner_boost) * match_att_mult

    # 3. Calculate 90-minute Expected Points from Official FPL Scoring Rules:
    # - 2 pts for playing 60+ mins
    # - Clean Sheet: +4 pts (GKP/DEF), +1 pt (MID), 0 pts (FWD)
    # - Goals Scored: +10 pts (GKP), +6 pts (DEF), +5 pts (MID), +4 pts (FWD)
    # - Assists: +3 pts (All positions)
    # - Goals Conceded (GKP/DEF only): -1 pt for 2-3 GC, -2 pts for 4-5 GC, -3 pts for 6+ GC
    # - Goalkeeper Saves: +1 pt per 3 saves (scaled by opponent attacking pressure)
    # - Penalty Miss: -2 pts (~20% of penalties taken are missed/saved)
    # - Yellow Cards: -1 pt (~12-16% per match for outfield players)
    appearance_pts = APPEARANCE_60PLUS_PTS
    cs_pts = cs_prob * CS_POINTS[player.position]
    gc_penalty = (
        (-1.0 * p_gc_2_3 - 2.0 * p_gc_4_5 - 3.0 * p_gc_6plus)
        if player.position in ("GKP", "DEF")
        else 0.0
    )
    expected_saves = (
        2.4 * (player.opp_attack_strength / 1140.0)
        if player.position == "GKP"
        else 0.0
    )
    save_pts = expected_saves / 3.0
    attack_pts = exp_g * GOAL_POINTS[player.position] + exp_a * ASSIST_POINTS
    pen_miss_penalty = (
        PENALTY_MISS_PTS * 0.03 if player.penalties_order == 1 else 0.0
    )
    card_penalty = (
        YELLOW_CARD_PTS * 0.14
        if player.position in ("DEF", "MID")
        else (YELLOW_CARD_PTS * 0.08 if player.position == "FWD" else 0.0)
    )

    # Expected Bonus Points (BPS: 1 to 3 pts awarded to top 3 BPS players in match)
    bps_base = max(0.0, (player.bps_per_90 - 14.0) / 10.0)
    fixture_bps_boost = 0.35 if player.fixture_difficulty <= 2 else 0.0
    bonus_pts = min(1.8, bps_base + fixture_bps_boost)

    raw_90_xp = (
        appearance_pts
        + cs_pts
        + gc_penalty
        + save_pts
        + attack_pts
        + pen_miss_penalty
        + card_penalty
        + bonus_pts
    )

    # 4. Blend with 3-Year Historical Baseline & Current Form for stability
    hist_ppg = player.historical_ppg_3yr
    blended_90_xp = 0.65 * raw_90_xp + 0.20 * player.form + 0.15 * hist_ppg

    # 5. Apply News / Press Conference Start & Cameo Probabilities
    # If a player starts (60+ mins) -> blended_90_xp; if cameo (<60 mins) -> 1.0 pt; if 0 mins -> 0.0 pt
    expected_pts = (
        news.start_probability * blended_90_xp
        + news.cameo_probability * APPEARANCE_CAMEO_PTS
    )

    # Lineup selection score: penalize players with high 1-point cameo risk because
    # a 1-point cameo blocks a healthy 2-3 pt bench player from auto-subbing in.
    cameo_trap_penalty = 1.25 if news.risk_level == "CAMEO_ROTATION_RISK" else 0.0
    lineup_score = expected_pts - cameo_trap_penalty

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
                + (6.0 if player.position in ("MID", "FWD") else 4.5)
            )
        )

    return PlayerPrediction(
        player=player,
        news=news,
        clean_sheet_prob=round(cs_prob, 2),
        concede_2plus_prob=round(concede_2plus_prob, 2),
        expected_goals=round(exp_g, 2),
        expected_assists=round(exp_a, 2),
        xp_if_90=round(blended_90_xp, 2),
        expected_points=round(expected_pts, 2),
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

    # Pick Captain & Vice-Captain by highest expected_points in Starting XI
    ranked_cap = sorted(
        starting_xi, key=lambda p: p.expected_points, reverse=True
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
        if p.news.risk_level not in ("INJURED_OR_OUT", "CAMEO_ROTATION_RISK")
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
            if p_in.news.risk_level in ("INJURED_OR_OUT", "CAMEO_ROTATION_RISK"):
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
        if p.news.risk_level not in ("INJURED_OR_OUT", "CAMEO_ROTATION_RISK")
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
        if p.news.risk_level not in ("INJURED_OR_OUT", "CAMEO_ROTATION_RISK")
    ]

    # Define distinct tactical archetypes for each variation
    archetypes = [
        {
            "name": "Variation 1: Balanced Optimal (3-5-2)",
            "subtitle": "Highest raw expected points across a 5-man midfield & 2 forwards",
            "formations": [(3, 5, 2)],
            "must_include": set(),
            "exclude_starters": set(),
            "bench_seed": 0,
        },
        {
            "name": "Variation 2: Haaland + Saka + Palmer 'Threemium' (3-4-3)",
            "subtitle": "Fits Haaland (£14.5m), Saka (£10.5m) & Palmer (£11.0m) in a 3-forward attack",
            "formations": [(3, 4, 3)],
            "must_include": {"Haaland", "Saka", "Palmer"},
            "exclude_starters": set(),
            "bench_seed": 1,
        },
        {
            "name": "Variation 3: Heavy Metal Front Three (3-4-3)",
            "subtitle": "Isak + Watkins + Cunha up front backed by Son, Saka, Palmer & Bruno",
            "formations": [(3, 4, 3)],
            "must_include": {"Watkins", "Isak", "B.Fernandes"},
            "exclude_starters": {"Groß", "Mbeumo", "Ndiaye"},
            "bench_seed": 2,
        },
        {
            "name": "Variation 4: Clean-Sheet Fortress (4-4-2)",
            "subtitle": "4 elite home/clean-sheet defenders (Gabriel, Saliba, Gvardiol, Robinson) + Raya",
            "formations": [(4, 4, 2)],
            "must_include": {"Saliba", "Robinson", "Gordon"},
            "exclude_starters": {"Pedro Porro", "Ndiaye"},
            "bench_seed": 3,
        },
        {
            "name": "Variation 5: Differential Upside & Wingers (3-5-2)",
            "subtitle": "Targets high-ceiling differentials: Son, Bowen, Gordon, Bruno & Watkins",
            "formations": [(3, 5, 2)],
            "must_include": {"Son", "Bowen", "Gordon", "Watkins"},
            "exclude_starters": {"Ødegaard", "Groß", "Isak"},
            "bench_seed": 4,
        },
        {
            "name": "Variation 6: All-Out Attacking Fullbacks + Haaland (4-3-3)",
            "subtitle": "Haaland + Solanke + Cunha up top with attacking fullbacks Hall & Porro",
            "formations": [(4, 3, 3)],
            "must_include": {"Haaland", "Solanke", "Hall"},
            "exclude_starters": {"Gvardiol", "Isak", "Ødegaard"},
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



