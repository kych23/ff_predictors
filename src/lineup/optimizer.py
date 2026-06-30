"""ILP lineup optimizer using scipy.optimize.milp.

Given weekly projections for a roster of players, solve for the optimal slot
assignment (QB:1, RB:2, WR:2, TE:1, FLEX:1, K:1, DEF:1) that maximizes the
selected quantile's total projected points.

Strategy modes map to which quantile column drives the objective:
  safe     -> P10 (maximize floor)
  balanced -> P50 (default)
  upside   -> P90 (maximize ceiling)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import eye as speye

from src.config import LeagueConfig, load_config

STRATEGY_MAP = {"safe": "p10", "balanced": "p50", "upside": "p90"}
EXCLUDED_STATUSES = {"OUT", "IR", "O"}


@dataclass
class LineupResult:
    starters: pd.DataFrame
    bench: pd.DataFrame
    total_points: float
    strategy: str
    slot_assignments: Dict[str, List[str]]


def optimize_lineup(
    players: pd.DataFrame,
    *,
    cfg: Optional[LeagueConfig] = None,
    strategy: str = "balanced",
    excluded_ids: Optional[set] = None,
) -> LineupResult:
    """Solve for the optimal starting lineup via ILP.

    `players` must have: player_id, position, p10, p50, p90.
    Optional: name, team, status.

    Returns LineupResult with starters, bench, and slot assignments.
    """
    cfg = cfg or load_config()
    obj_col = STRATEGY_MAP.get(strategy, "p50")

    df = players.copy()
    df["player_id"] = df["player_id"].astype(str)

    if excluded_ids:
        df = df[~df["player_id"].isin(excluded_ids)].copy()
    if "status" in df.columns:
        df = df[~df["status"].str.upper().isin(EXCLUDED_STATUSES)].copy()

    df = df.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return LineupResult(
            starters=pd.DataFrame(), bench=pd.DataFrame(),
            total_points=0.0, strategy=strategy, slot_assignments={},
        )

    positions = df["position"].values
    values = df[obj_col].fillna(0).values

    starter_slots = {}
    for slot, count in cfg.roster.slots.items():
        if slot in ("BENCH",):
            continue
        starter_slots[slot] = count
    flex_eligible = set(cfg.roster.flex_eligible)

    slot_names = list(starter_slots.keys())
    n_slots = len(slot_names)
    # Decision variables: x[i, j] = 1 if player i assigned to slot j
    n_vars = n * n_slots

    # Objective: maximize sum of values for assigned players
    # milp minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n):
        for j, slot in enumerate(slot_names):
            c[i * n_slots + j] = -values[i]

    # Integrality: all binary
    integrality = np.ones(n_vars, dtype=int)

    # Bounds: 0 <= x <= 1
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)
    for i in range(n):
        pos = positions[i]
        for j, slot in enumerate(slot_names):
            if _can_fill(pos, slot, flex_eligible):
                ub[i * n_slots + j] = 1.0

    # Constraint 1: each player assigned to at most 1 slot
    # sum_j x[i,j] <= 1 for each player i
    A_player = np.zeros((n, n_vars))
    for i in range(n):
        for j in range(n_slots):
            A_player[i, i * n_slots + j] = 1.0
    player_constraint = LinearConstraint(A_player, 0, 1)

    # Constraint 2: each slot filled by exactly its count
    # sum_i x[i,j] == slots[j] for each slot j
    A_slot = np.zeros((n_slots, n_vars))
    for j in range(n_slots):
        for i in range(n):
            A_slot[j, i * n_slots + j] = 1.0
    slot_counts = np.array([starter_slots[s] for s in slot_names], dtype=float)

    # Use <= instead of == to handle cases where roster is too small
    total_starters = int(slot_counts.sum())
    if n < total_starters:
        slot_constraint = LinearConstraint(A_slot, 0, slot_counts)
    else:
        slot_constraint = LinearConstraint(A_slot, slot_counts, slot_counts)

    result = milp(
        c=c,
        constraints=[player_constraint, slot_constraint],
        integrality=integrality,
        bounds=scipy_bounds(lb, ub),
    )

    if not result.success:
        return _greedy_fallback(df, obj_col, cfg, strategy, flex_eligible, starter_slots)

    x = np.round(result.x).astype(int).reshape(n, n_slots)
    starter_mask = x.sum(axis=1) > 0
    starter_indices = np.where(starter_mask)[0]

    slot_assignments: Dict[str, List[str]] = {s: [] for s in slot_names}
    for i in starter_indices:
        for j, slot in enumerate(slot_names):
            if x[i, j] == 1:
                slot_assignments[slot].append(df.iloc[i]["player_id"])

    starters = df.iloc[starter_indices].copy()
    starters["slot"] = ""
    for i in starter_indices:
        for j, slot in enumerate(slot_names):
            if x[i, j] == 1:
                starters.loc[i, "slot"] = slot
                break

    bench = df[~starter_mask].copy()
    bench["slot"] = "BENCH"

    return LineupResult(
        starters=starters.reset_index(drop=True),
        bench=bench.reset_index(drop=True),
        total_points=float(-result.fun),
        strategy=strategy,
        slot_assignments=slot_assignments,
    )


def scipy_bounds(lb, ub):
    from scipy.optimize import Bounds
    return Bounds(lb=lb, ub=ub)


def _can_fill(position: str, slot: str, flex_eligible: set) -> bool:
    """Can a player of `position` fill `slot`?"""
    if slot == "FLEX":
        return position in flex_eligible
    return position == slot


def _greedy_fallback(df: pd.DataFrame, obj_col: str, cfg: LeagueConfig,
                     strategy: str, flex_eligible: set,
                     starter_slots: dict) -> LineupResult:
    """Greedy fallback if ILP solver fails."""
    df = df.sort_values(obj_col, ascending=False).reset_index(drop=True)
    assigned = set()
    slot_assignments: Dict[str, List[str]] = {s: [] for s in starter_slots}

    for slot, count in starter_slots.items():
        if slot == "FLEX":
            continue
        candidates = df[(df["position"] == slot) & (~df.index.isin(assigned))]
        for idx in candidates.index[:count]:
            assigned.add(idx)
            slot_assignments[slot].append(df.loc[idx, "player_id"])

    flex_count = starter_slots.get("FLEX", 0)
    flex_candidates = df[
        (df["position"].isin(flex_eligible)) & (~df.index.isin(assigned))
    ].head(flex_count)
    for idx in flex_candidates.index:
        assigned.add(idx)
        slot_assignments["FLEX"].append(df.loc[idx, "player_id"])

    starters = df.loc[list(assigned)].copy()
    starters["slot"] = ""
    for slot, pids in slot_assignments.items():
        for pid in pids:
            starters.loc[starters["player_id"] == pid, "slot"] = slot

    bench = df[~df.index.isin(assigned)].copy()
    bench["slot"] = "BENCH"

    total = starters[obj_col].fillna(0).sum()
    return LineupResult(
        starters=starters.reset_index(drop=True),
        bench=bench.reset_index(drop=True),
        total_points=float(total),
        strategy=strategy,
        slot_assignments=slot_assignments,
    )


def optimal_lineup_points(players: pd.DataFrame, *,
                          cfg: Optional[LeagueConfig] = None,
                          quantile_col: str = "p50") -> float:
    """Convenience: return just the total points of the optimal lineup."""
    strategy = {v: k for k, v in STRATEGY_MAP.items()}.get(quantile_col, "balanced")
    result = optimize_lineup(players, cfg=cfg, strategy=strategy)
    return result.total_points
