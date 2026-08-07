"""Season evaluation (§15.3, §15.4).

Two things make this cheap enough to run inside a 60-second pick clock:

1. **The starter mask is chosen from projections, not draws.** Non-clairvoyance
   is a modelling requirement (§15.3) and it also means the mask depends only
   on (roster, week) — so it is computed once per roster-week and reused across
   every replication and parameter draw, rather than once per replication.

2. **Scoring a season is then a matmul per week.** ``masks @ S`` over
   (T x P)(P x R).

The mask is ``(T, W, P)`` and not ``(T, P)``: bye weeks make the optimal starter
set week-dependent, and a week-invariant mask leaves a bye player's slot
unfilled, costing every team a full starter on every bye week — hitting exactly
the clustered-bye rosters the tool exists to warn about.
"""
from __future__ import annotations

import numpy as np

from src.core.config.slots import SlotPlan
from src.domain.roster.lineup import greedy_lineup


def selection_values(bundle, roster_idx: np.ndarray, week: int) -> np.ndarray:
    """Pre-week projection used to CHOOSE the lineup.

    Availability and byes enter here rather than in the draw, so a player known
    to be out is not started — while the *realized* points still come from the
    drawn tensor.
    """
    values = bundle.rate_p50[roster_idx] * bundle.games_hazard[roster_idx, week]
    on_bye = bundle.bye_weeks[roster_idx] == week + 1
    return np.where(on_bye, 0.0, values)


def build_masks(bundle, rosters: list[np.ndarray], plan: SlotPlan,
                weeks: int) -> np.ndarray:
    """(T, W, P) float32 starter indicators.

    Computed once per (roster, week) — NOT per replication, which is what keeps
    this off the hot path.
    """
    n_players = bundle.n_players
    masks = np.zeros((len(rosters), weeks, n_players), dtype="float32")
    for t, roster_idx in enumerate(rosters):
        if len(roster_idx) == 0:
            continue
        positions = bundle.positions[roster_idx]
        for w in range(weeks):
            chosen = greedy_lineup(selection_values(bundle, roster_idx, w),
                                   positions, plan)
            masks[t, w, roster_idx[chosen]] = 1.0
    return masks


def free_agent_pool(bundle, rosters: list[np.ndarray], size: int) -> np.ndarray:
    """The waiver pool: the best `size` players nobody drafted.

    Taken by projection rather than at random because a waiver claim is not a
    lottery — the wire is picked over, and the players who matter are the ones
    at the top of what is left. `size` comes from `strategy.simulation.waiver`
    and bounds the claim search; the whole pool is shared across teams, so a
    claimed player leaves it for everyone.
    """
    rostered: set[int] = set()
    for roster in rosters:
        rostered.update(int(i) for i in roster)
    free = np.array([i for i in range(bundle.n_players) if i not in rostered],
                    dtype=int)
    if free.size <= size:
        return free
    order = np.argsort(-bundle.rate_p50[free], kind="stable")
    return np.sort(free[order[:size]])


def apply_waiver_floor(points: np.ndarray, masks: np.ndarray,
                       team_week: np.ndarray, rosters: list[np.ndarray],
                       bundle, cfg, strategy, *,
                       in_decision_path: bool = False) -> np.ndarray:
    """Correct `team_week` for in-season waiver replacement (§15.6, §8.6).

    Without this a drafted bust is carried at his realized production for all
    fourteen weeks, which nobody does — you drop him. Leaving it out prices
    every late-round flier as though it could never be cut, understating late
    picks and overstating the cost of a miss.

    Wired here rather than inside `evaluate_rosters` because the drop decision
    depends on *observed* points and so is per-replication, while the starter
    mask deliberately is not (§15.3). Returns a corrected `(T, W, R)`.

    Two switches, because the correction is expensive enough to change what the
    engine can do inside a pick clock (see the cost note in `strategy.yaml`):
    `enabled` governs it at all, and `in_decision_path` governs whether it also
    runs during a live recommendation. Callers on the clock pass
    `in_decision_path=True` and are refused unless BOTH are set. When either is
    off the input is returned unchanged, so pre-waiver behaviour stays exactly
    reproducible.
    """
    waiver_cfg = getattr(strategy.simulation, "waiver", None) or {}
    if not waiver_cfg.get("enabled", False):
        return team_week
    if in_decision_path and not waiver_cfg.get("in_decision_path", False):
        return team_week

    from src.engine.sim.waiver import apply_waiver

    pool = free_agent_pool(bundle, rosters, int(waiver_cfg.get("pool_size", 40)))
    if pool.size == 0:
        return team_week
    return apply_waiver(
        points, masks, team_week, rosters, pool, bundle,
        detection_lag_weeks=int(waiver_cfg.get("detection_lag_weeks", 2)),
        shrinkage_k_games=int(waiver_cfg.get("shrinkage_k_games", 6)),
        regular_season_weeks=cfg.schedule.regular_season_weeks,
    ).team_week


def evaluate_rosters(points: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """(T, W, R) team-week totals from (P, W, R) points and (T, W, P) masks."""
    n_teams, weeks, _ = masks.shape
    reps = points.shape[2]
    out = np.empty((n_teams, weeks, reps), dtype="float32")
    for w in range(weeks):
        out[:, w, :] = masks[:, w, :] @ points[:, w, :]
    return out


def season_outcome(team_week: np.ndarray, cfg, *, my_team_index: int = 0):
    """Bundle team-week totals into what the objective consumes (§16.1)."""
    from src.domain.payout.reducers import SeasonOutcome
    from src.domain.schedule.bracket import final_ranks, regular_season_ranks

    regular, _ = regular_season_ranks(team_week, cfg.schedule)
    return SeasonOutcome(
        team_week=team_week,
        final_rank=final_ranks(team_week, regular, cfg.schedule),
        regular_rank=regular,
        season_points=team_week[:, : cfg.schedule.regular_season_weeks, :].sum(axis=1),
        my_team_index=my_team_index,
    )
