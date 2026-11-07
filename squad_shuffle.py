"""Build a reproducible library of valid squads for the static-page shuffle.

Random swaps explore alternatives; every displayed score uses the original model
predictions. This is a bounded sample, not an exhaustive enumeration or a uniform
draw from all possible FPL squads.
"""
from collections import Counter
from itertools import product
import random
from typing import List, Optional, Sequence

from optimizer import PlanPlayer, SQUAD_SLOTS, WeekPlan, select_candidate_pool, solve_squad


def score_squad(players: Sequence[PlanPlayer], gw: int) -> WeekPlan:
    """Best legal XI and captain for a fixed 15, scored without bench bonuses."""
    if len({p.id for p in players}) != 15 or Counter(p.position for p in players) != SQUAD_SLOTS:
        raise ValueError("A squad must contain 15 unique players: 2 GK, 5 DEF, 5 MID, 3 FWD.")
    xp = lambda p: p.xp_by_gw.get(gw, 0.0)
    ranked = {pos: sorted((p for p in players if p.position == pos), key=lambda p: (-xp(p), p.id))
              for pos in SQUAD_SLOTS}
    best_score, best_xi = float("-inf"), []
    for defenders, mids, forwards in product(range(3, 6), range(2, 6), range(1, 4)):
        if defenders + mids + forwards != 10:
            continue
        xi = ranked["GKP"][:1] + ranked["DEF"][:defenders] + ranked["MID"][:mids] + ranked["FWD"][:forwards]
        score = sum(xp(p) for p in xi) + max(xp(p) for p in xi)
        if score > best_score:
            best_score, best_xi = score, xi
    leaders = sorted(best_xi, key=lambda p: (-xp(p), p.id))
    starting_ids = {p.id for p in best_xi}
    bench = sorted((p for p in players if p.id not in starting_ids),
                   key=lambda p: (p.position == "GKP", -xp(p), p.id))
    return WeekPlan(gw=gw, squad_ids=[p.id for p in players], lineup_ids=[p.id for p in best_xi],
                    captain_id=leaders[0].id, vice_id=leaders[1].id, bench_order=[p.id for p in bench],
                    transfers_in=[], transfers_out=[], free_transfers_before=0, hits=0,
                    expected_points=round(best_score, 2))


def generate_shuffle_pool(players: List[PlanPlayer], gw: int, budget: float = 100.0,
                          count: int = 120, seed: Optional[int] = None,
                          max_attempts: Optional[int] = None) -> List[WeekPlan]:
    """Return distinct starting XIs, ordered by original projected points.

    The first solve maximizes this week's XI plus captain, within the shortlisted
    player pool. Bounded random swaps then explore teams within five points of
    that result. Each squad satisfies the same budget, club and position rules.
    Fewer than `count` results are returned if the search budget is exhausted.
    """
    if count < 1 or budget <= 0 or (max_attempts is not None and max_attempts < 0):
        raise ValueError("Count and budget must be positive; attempts must be nonnegative.")
    if len({p.id for p in players}) != len(players):
        raise ValueError("Candidate player IDs must be unique.")
    candidates = sorted(select_candidate_pool(players, [gw]), key=lambda p: p.id)
    base = solve_squad(candidates, gw, budget=budget, bench_weight=0, time_limit=10)
    if base is None:
        raise RuntimeError("No valid squad found within the budget.")
    by_id = {p.id: p for p in candidates}
    base_players = [by_id[i] for i in base.squad_ids]
    solver_result = base
    base = score_squad(base_players, gw)
    base.solver_status = solver_result.solver_status
    base.solver_gap = solver_result.solver_gap
    # Deliberately do not claim an exact global optimum for the sampled library.
    rng = random.Random(gw if seed is None else seed)
    by_position = {pos: [p for p in candidates if p.position == pos] for pos in SQUAD_SLOTS}
    frontier = [base_players]
    seen_squads = {tuple(sorted(base.squad_ids))}
    results = {tuple(sorted(base.lineup_ids)): base}
    floor = max(0.0, base.expected_points - 5.0)
    attempts = count * 100 if max_attempts is None else max_attempts
    for _ in range(attempts):
        if len(results) >= count:
            break
        trial = list(rng.choice(frontier) if rng.random() < .7 else base_players)
        for slot in rng.sample(range(15), rng.randint(1, 5)):
            used = {p.id for p in trial}
            options = [p for p in by_position[trial[slot].position] if p.id not in used]
            if options:
                weights = [max(.1, p.xp_by_gw.get(gw, 0.0)) ** 2 for p in options]
                trial[slot] = rng.choices(options, weights=weights, k=1)[0]
        signature = tuple(sorted(p.id for p in trial))
        if signature in seen_squads:
            continue
        if sum(round(p.price * 10) for p in trial) > round(budget * 10):
            continue
        if max(Counter(p.team for p in trial).values()) > 3:
            continue
        seen_squads.add(signature)
        week = score_squad(trial, gw)
        if week.expected_points < floor:
            continue
        if len(frontier) < 400:
            frontier.append(trial)
        xi_signature = tuple(sorted(week.lineup_ids))
        if xi_signature not in results:
            week.solver_status = "Random squad swaps; best legal XI and captain within this squad"
            results[xi_signature] = week
    output = sorted(results.values(), key=lambda w: (-w.expected_points, tuple(sorted(w.lineup_ids))))
    for week in output:
        week.candidate_count = len(candidates)
    return output
