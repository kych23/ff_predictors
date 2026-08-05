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
