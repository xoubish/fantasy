"""
Mixed-Integer squad & transfer optimizer (optimizer.py)

Solves the real FPL decision problem with scipy's HiGHS MILP solver:

  * `solve_squad`     — best 15-man squad for one objective (single GW or a weighted horizon),
                        with optional must-include / exclude / price / ownership constraints.
  * `plan_transfers`  — multi-gameweek transfer plan from an EXISTING squad: free transfers
                        bank up to 5, extra transfers cost -4, lineup + captain per week,
                        bench value, and a chip-style "unlimited transfers this week" switch.

Both share one formulation.  Variables per (player p, week t):
    s[p,t] in squad, l[p,t] in starting XI, c[p,t] captain, i[p,t]/o[p,t] transferred in/out
plus per week: f[t] free transfers available (0-5), h[t] paid transfers (hits).

Objective (maximised):
    sum_t w_t * ( sum_p xp[p,t]*(l + c) + bench_w * sum_p xp[p,t]*(s - l) )
    - 4 * sum_t h[t]  + ft_value * f[T]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

POS_ORDER = ["GKP", "DEF", "MID", "FWD"]
SQUAD_SLOTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
HIT_COST = 4.0
MAX_FREE_TRANSFERS = 5


@dataclass
class PlanPlayer:
    """Minimal player view the optimizer needs (decoupled from PlayerPrediction)."""
    id: int
    name: str
    team: str
    position: str
    price: float  # buy price (£m)
    sell_price: float  # what you get back if sold (£m)
    xp_by_gw: Dict[int, float]  # expected points per gameweek (0 for blanks)
    ownership: float = 0.0
    haul_prob: float = 0.0
    xp_uncertainty: float = 0.0  # conservative mean-estimate error proxy, not a confidence interval


@dataclass
class WeekPlan:
    gw: int
    squad_ids: List[int]
    lineup_ids: List[int]
    captain_id: int
    vice_id: int
    bench_order: List[int]  # outfield bench in order, then bench GK last
    transfers_in: List[int]
    transfers_out: List[int]
    free_transfers_before: int
    hits: int
    expected_points: float  # XI + captain for this week (before hits)
    free_transfers_after: int = 0  # available at the following deadline
    solver_status: str = ""
    solver_gap: Optional[float] = None
    candidate_count: int = 0


@dataclass
class TransferPlan:
    weeks: List[WeekPlan]
    objective: float
    total_hits: int
    solver_status: str
    notes: List[str] = field(default_factory=list)
    solver_gap: Optional[float] = None
    candidate_count: int = 0


class _VarIndex:
    """Dense index for (kind, p, t) -> column, plus per-week scalar vars."""

    def __init__(self, n_players: int, n_weeks: int):
        self.P, self.T = n_players, n_weeks
        self.kinds = ["s", "l", "c", "i", "o"]
        self.block = n_players * n_weeks
        self.n_pt = len(self.kinds) * self.block
        self.f0 = self.n_pt  # f[t]
        self.h0 = self.f0 + n_weeks + 1  # include terminal f[T]
        self.hit_active0 = self.h0 + n_weeks
        self.cap_active0 = self.hit_active0 + n_weeks
        self.n = self.cap_active0 + n_weeks

    def pt(self, kind: str, p: int, t: int) -> int:
        return self.kinds.index(kind) * self.block + t * self.P + p

    def f(self, t: int) -> int:
        return self.f0 + t

    def h(self, t: int) -> int:
        return self.h0 + t


def _solve(
    players: List[PlanPlayer],
    gws: List[int],
    initial_squad: Optional[Sequence[int]],
    budget: float,
    free_transfers: int,
    week_weights: Sequence[float],
    bench_weight: float,
    ft_end_value: float,
    unlimited_weeks: Sequence[int],
    max_per_club: int,
    must_include: Sequence[int],
    exclude: Sequence[int],
    lock_lineup_min_def: Optional[int],
    time_limit: float,
    freeze_after_first: bool = False,
    transfer_friction: float = 0.0,
    risk_margin: float = 0.5,
) -> Tuple[Optional[np.ndarray], _VarIndex, Dict[str, Any]]:
    P, T = len(players), len(gws)
    if not P or not T:
        raise ValueError("The optimizer needs players and at least one gameweek.")
    if initial_squad is not None and not 0 <= free_transfers <= MAX_FREE_TRANSFERS:
        raise ValueError("Free transfers must be between 0 and 5.")
    if ft_end_value < 0:
        raise ValueError("Terminal free-transfer value must be nonnegative.")
    if risk_margin < 0 or transfer_friction < 0:
        raise ValueError("Transfer risk and friction margins must be nonnegative.")
    V = _VarIndex(P, T)
    pid_to_idx = {pl.id: k for k, pl in enumerate(players)}

    # ---------------- objective ----------------
    cvec = np.zeros(V.n)
    for t, gw in enumerate(gws):
        w = week_weights[t]
        for p, pl in enumerate(players):
            xp = pl.xp_by_gw.get(gw, 0.0)
            cvec[V.pt("l", p, t)] -= w * xp * (1.0 - bench_weight)
            cvec[V.pt("s", p, t)] -= w * xp * bench_weight
            cvec[V.pt("c", p, t)] -= w * xp
        if t not in unlimited_weeks:
            cvec[V.h(t)] += HIT_COST * w
    cvec[V.f(T)] -= ft_end_value  # value AFTER the final week's transfers
    # Friction per transfer (free or paid): expected-point gains between two players are
    # noisy estimates, so a swap must clear a margin before it is worth making.
    for t in range(T):
        if initial_squad is None and t == 0:
            continue
        for p in range(P):
            # A transfer must clear both its hit/friction cost and an estimation-risk
            # margin for the incoming and outgoing players across the remaining horizon.
            uncertainty_cost = risk_margin * players[p].xp_uncertainty * float(np.sqrt(sum(w*w for w in week_weights[t:])))
            cvec[V.pt("i", p, t)] += 1e-3 + transfer_friction * week_weights[t] + uncertainty_cost
            cvec[V.pt("o", p, t)] += uncertainty_cost

    # ---------------- constraints ----------------
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    r = 0

    def add(entries: List[Tuple[int, float]], lb: float, ub: float) -> None:
        nonlocal r
        for col, val in entries:
            rows.append(r)
            cols.append(col)
            vals.append(val)
        lo.append(lb)
        hi.append(ub)
        r += 1

    clubs = sorted({pl.team for pl in players})
    for t in range(T):
        # squad composition
        for pos in POS_ORDER:
            add([(V.pt("s", p, t), 1.0) for p, pl in enumerate(players) if pl.position == pos],
                SQUAD_SLOTS[pos], SQUAD_SLOTS[pos])
            n_min = XI_MIN[pos]
            if pos == "DEF" and lock_lineup_min_def:
                n_min = max(n_min, lock_lineup_min_def)
            add([(V.pt("l", p, t), 1.0) for p, pl in enumerate(players) if pl.position == pos],
                n_min, XI_MAX[pos])
        add([(V.pt("l", p, t), 1.0) for p in range(P)], 11, 11)
        add([(V.pt("c", p, t), 1.0) for p in range(P)], 1, 1)
        for club in clubs:
            add([(V.pt("s", p, t), 1.0) for p, pl in enumerate(players) if pl.team == club], 0, max_per_club)
        for p in range(P):
            add([(V.pt("l", p, t), 1.0), (V.pt("s", p, t), -1.0)], -np.inf, 0)  # l <= s
            add([(V.pt("c", p, t), 1.0), (V.pt("l", p, t), -1.0)], -np.inf, 0)  # c <= l
        # budget: players already owned count at their SELL price (budget = squad sell value + bank),
        # everyone else at their buy price
        owned = set(initial_squad or [])
        entries = [
            (V.pt("s", p, t), pl.sell_price if pl.id in owned else pl.price)
            for p, pl in enumerate(players)
        ]
        add(entries, -np.inf, budget)
        if freeze_after_first and t >= 1:
            for p in range(P):
                add([(V.pt("i", p, t), 1.0), (V.pt("o", p, t), 1.0)], 0, 0)

        # continuity
        for p in range(P):
            if t == 0:
                init = 1.0 if (initial_squad and players[p].id in initial_squad) else 0.0
                # s = init + i - o
                add([(V.pt("s", p, 0), 1.0), (V.pt("i", p, 0), -1.0), (V.pt("o", p, 0), 1.0)], init, init)
            else:
                add(
                    [(V.pt("s", p, t), 1.0), (V.pt("s", p, t - 1), -1.0), (V.pt("i", p, t), -1.0), (V.pt("o", p, t), 1.0)],
                    0, 0,
                )
            add([(V.pt("i", p, t), 1.0), (V.pt("o", p, t), 1.0)], 0, 1)

        # transfer accounting
        n_in = [(V.pt("i", p, t), 1.0) for p in range(P)]
        if initial_squad is None and t == 0:
            # fresh squad / wildcard: unlimited free transfers in week 0
            add([(V.f(0), 1.0)], 15, 15)
            add([(V.h(0), 1.0)], 0, 0)
        elif t in unlimited_weeks:
            add([(V.h(t), 1.0)], 0, 0)
            if t == 0:
                add([(V.f(0), 1.0)], free_transfers, free_transfers)
        else:
            if t == 0:
                add([(V.f(0), 1.0)], free_transfers, free_transfers)
            # h[t] >= n_in[t] - f[t]
            add([(V.h(t), 1.0), (V.f(t), 1.0)] + [(col, -1.0) for col, _ in n_in], 0, np.inf)
        if initial_squad is None and t == 0:
            add([(V.f(t + 1), 1.0)], 1, 1)
        elif t in unlimited_weeks:
            # Wildcards preserve the bank; no extra FT is earned that week.
            add([(V.f(t + 1), 1.0), (V.f(t), -1.0)], 0, 0)
        else:
            # Exact accounting: h=max(0, transfers-f), f_next=min(5, f-transfers+h+1).
            # Binary branches prevent inventing hits or silently discarding banked FTs.
            big_m = 20.0
            hit_active = V.hit_active0 + t
            capped = V.cap_active0 + t
            add([(V.h(t), 1.0), (hit_active, -big_m)], -np.inf, 0)
            add([(V.h(t), 1.0), (V.f(t), 1.0), (hit_active, big_m)]
                + [(col, -1.0) for col, _ in n_in], -np.inf, big_m)
            balance = [(V.f(t + 1), 1.0), (V.f(t), -1.0), (V.h(t), -1.0)] + n_in
            add(balance, -np.inf, 1)
            add(balance + [(capped, big_m)], 1, np.inf)
            add([(V.f(t + 1), 1.0), (capped, -big_m)], MAX_FREE_TRANSFERS - big_m, np.inf)

    # must include / exclude (all weeks)
    for pid in must_include:
        if pid in pid_to_idx:
            for t in range(T):
                add([(V.pt("s", pid_to_idx[pid], t), 1.0)], 1, 1)
    for pid in exclude:
        if pid in pid_to_idx:
            for t in range(T):
                add([(V.pt("s", pid_to_idx[pid], t), 1.0)], 0, 0)

    A = coo_matrix((vals, (rows, cols)), shape=(r, V.n)).tocsr()
    constraints = LinearConstraint(A, np.asarray(lo), np.asarray(hi))

    lb = np.zeros(V.n)
    ub = np.ones(V.n)
    integrality = np.ones(V.n)
    for t in range(T):
        ub[V.f(t)] = 15 if t == 0 and initial_squad is None else MAX_FREE_TRANSFERS
        ub[V.h(t)] = 15
    ub[V.f(T)] = MAX_FREE_TRANSFERS
    bounds = Bounds(lb, ub)

    res = milp(
        c=cvec,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 1e-4},
    )
    info = {"status": str(res.message), "gap": getattr(res, "mip_gap", None), "candidates": P}
    if res.x is None:
        return None, V, info
    return np.round(res.x), V, info


def _bench_order(players: List[PlanPlayer], squad_ids: List[int], lineup_ids: List[int], gw: int) -> List[int]:
    by_id = {pl.id: pl for pl in players}
    bench = [pid for pid in squad_ids if pid not in lineup_ids]
    outfield = sorted(
        [pid for pid in bench if by_id[pid].position != "GKP"],
        key=lambda pid: -by_id[pid].xp_by_gw.get(gw, 0.0),
    )
    gk = [pid for pid in bench if by_id[pid].position == "GKP"]
    return outfield + gk


def _extract_plan(
    x: np.ndarray, V: _VarIndex, players: List[PlanPlayer], gws: List[int], status: Dict[str, Any]
) -> TransferPlan:
    weeks: List[WeekPlan] = []
    total_hits = 0
    for t, gw in enumerate(gws):
        squad = [players[p].id for p in range(V.P) if x[V.pt("s", p, t)] > 0.5]
        lineup = [players[p].id for p in range(V.P) if x[V.pt("l", p, t)] > 0.5]
        cap = next((players[p].id for p in range(V.P) if x[V.pt("c", p, t)] > 0.5), lineup[0])
        by_id = {pl.id: pl for pl in players}
        vice = max((pid for pid in lineup if pid != cap), key=lambda pid: by_id[pid].xp_by_gw.get(gw, 0.0))
        t_in = [players[p].id for p in range(V.P) if x[V.pt("i", p, t)] > 0.5]
        t_out = [players[p].id for p in range(V.P) if x[V.pt("o", p, t)] > 0.5]
        hits = int(x[V.h(t)])
        total_hits += hits
        xi_xp = sum(by_id[pid].xp_by_gw.get(gw, 0.0) for pid in lineup) + by_id[cap].xp_by_gw.get(gw, 0.0)
        weeks.append(
            WeekPlan(
                gw=gw,
                squad_ids=squad,
                lineup_ids=lineup,
                captain_id=cap,
                vice_id=vice,
                bench_order=_bench_order(players, squad, lineup, gw),
                transfers_in=t_in,
                transfers_out=t_out,
                free_transfers_before=int(x[V.f(t)]),
                hits=hits,
                expected_points=round(xi_xp, 2),
                free_transfers_after=int(x[V.f(t + 1)]),
                solver_status=status["status"],
                solver_gap=status["gap"],
                candidate_count=status["candidates"],
            )
        )
    objective = sum(w.expected_points for w in weeks) - HIT_COST * total_hits
    return TransferPlan(weeks=weeks, objective=round(objective, 2), total_hits=total_hits,
                        solver_status=status["status"], solver_gap=status["gap"], candidate_count=status["candidates"])


def solve_squad(
    players: List[PlanPlayer],
    gw: int,
    budget: float = 100.0,
    horizon_weights: Optional[Dict[int, float]] = None,
    bench_weight: float = 0.08,
    max_per_club: int = 3,
    must_include: Sequence[int] = (),
    exclude: Sequence[int] = (),
    min_def_in_xi: Optional[int] = None,
    time_limit: float = 20.0,
) -> Optional[WeekPlan]:
    """
    Best 15-man squad built from scratch for gameweek `gw`.
    `horizon_weights` = {gw: weight, gw+1: weight, ...} scores the SAME squad against several
    future gameweeks (lineup chosen per week) — a squad that is good for a run of fixtures,
    not only this week.  Returns the plan for `gw` (first week).
    """
    if horizon_weights:
        gws = sorted(horizon_weights)
        weights = [horizon_weights[g] for g in gws]
    else:
        gws, weights = [gw], [1.0]
    x, V, status = _solve(
        players, gws, None, budget, 15, weights, bench_weight, 0.0, [], max_per_club,
        must_include, exclude, min_def_in_xi, time_limit, freeze_after_first=True,
    )
    if x is None:
        return None
    plan = _extract_plan(x, V, players, gws, status)
    return plan.weeks[0]


def plan_transfers(
    players: List[PlanPlayer],
    initial_squad: Sequence[int],
    gws: List[int],
    budget: float,
    free_transfers: int,
    week_decay: float = 0.84,
    bench_weight: float = 0.08,
    ft_end_value: float = 1.2,
    wildcard_week_index: Optional[int] = None,
    max_per_club: int = 3,
    time_limit: float = 45.0,
    transfer_friction: float = 1.5,
    risk_margin: float = 0.5,
) -> Optional[TransferPlan]:
    """
    Multi-week transfer plan from `initial_squad` (list of player ids).
    `budget` = current squad selling value + bank.  Free transfers bank up to 5; every extra
    transfer costs 4 points.  Set `wildcard_week_index` to make that week's transfers free.
    `transfer_friction` (xP) is charged on every transfer so marginal swaps are not made.
    """
    weights = [week_decay**k for k in range(len(gws))]
    unlimited = [wildcard_week_index] if wildcard_week_index is not None else []
    x, V, status = _solve(
        players, gws, list(initial_squad), budget, free_transfers, weights, bench_weight,
        ft_end_value, unlimited, max_per_club, (), (), None, time_limit,
        transfer_friction=transfer_friction,
        risk_margin=risk_margin,
    )
    if x is None:
        return None
    plan = _extract_plan(x, V, players, gws, status)
    if wildcard_week_index is not None:
        plan.notes.append(f"Week index {wildcard_week_index} treated as unlimited free transfers (wildcard).")
    return plan


def select_candidate_pool(
    players: List[PlanPlayer],
    gws: Sequence[int],
    keep_ids: Sequence[int] = (),
    per_position: Optional[Dict[str, int]] = None,
    cheap_per_position: int = 4,
) -> List[PlanPlayer]:
    """Shrinks the pool to the strongest candidates by horizon xP plus cheap enablers and
    anything in `keep_ids` (e.g. the current squad). Keeps the MILP small and fast."""
    per_position = per_position or {"GKP": 10, "DEF": 28, "MID": 32, "FWD": 18}
    keep = set(keep_ids)
    out: Dict[int, PlanPlayer] = {}

    def horizon_xp(pl: PlanPlayer) -> float:
        return sum(pl.xp_by_gw.get(g, 0.0) * (0.85**k) for k, g in enumerate(gws))

    for pos in POS_ORDER:
        pool = [pl for pl in players if pl.position == pos]
        pool.sort(key=horizon_xp, reverse=True)
        for pl in pool[: per_position.get(pos, 20)]:
            out[pl.id] = pl
        cheap = sorted(pool, key=lambda pl: (pl.price, -horizon_xp(pl)))
        taken = 0
        for pl in cheap:
            if taken >= cheap_per_position:
                break
            if pl.id not in out and horizon_xp(pl) > 0.3 * len(gws):
                out[pl.id] = pl
                taken += 1
    for pl in players:
        if pl.id in keep:
            out[pl.id] = pl
    return list(out.values())
