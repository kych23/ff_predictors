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

    Byes are exact here: a player on bye is worth zero this week and drops out
    of the lineup, which is why the mask is week-dependent at all.

    **Availability is only a probability, and that is a known limitation.**
    This multiplies by ``games_hazard``, the fitted *chance* of playing, so a
    70%-available starter is chosen on 70% of his rate. The drawn ``active``
    indicator from `draws.draw_points` never reaches `build_masks`, so when a
    replication draws him inactive he contributes an exact zero and no healthy
    bench player is substituted. The docstring here used to claim "a player
    known to be out is not started", which describes the intent rather than
    the behaviour.

    Measured on the shipped bundle (12 ADP-drafted rosters, 200 replications,
    17 weeks): the fitted hazard averages 0.724 for skill players, and
    **22.5% of chosen-starter player-weeks are drawn at exactly zero**. Mean
    team-week comes out near 73 where letting the lineup see realized
    availability gives about 85 — a systematic understatement of roughly 15%.

    Fixing it is not cheap, which is the honest reason it stands. Availability
    is knowable before kickoff, so selecting on it would NOT breach
    non-clairvoyance (§15.3 forbids selecting on realized POINTS, not on who
    is active). But the mask would become replication-dependent — (T, W, P, R)
    rather than (T, W, P) — and computing it once per roster-week is precisely
    what keeps this off the hot path inside a 60-second pick clock.

    The bias is common across teams, so it largely cancels in rank-type
    prizes. It does NOT cancel for bench depth, which is what the late rounds
    are priced on, so this is the largest known fidelity gap in the objective.
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


#: Which starters a bench player may cover. Substitution is bounded by what
#: could legally take the slot: a quarterback cannot cover a receiver, but the
#: flex positions cover each other because FLEX already lets them.
def _substitution_groups(cfg) -> list[frozenset[str]]:
    flex = frozenset(cfg.roster.flex_eligible)
    singles = [frozenset({p}) for p in cfg.roster.slots
               if p not in flex and p != "BENCH"]
    return [flex, *singles]


def apply_starter_substitution(points: np.ndarray, masks: np.ndarray,
                               team_week: np.ndarray,
                               rosters: list[np.ndarray], bundle,
                               cfg, strategy=None) -> np.ndarray:
    """Start the next man up when a chosen starter is drawn inactive.

    **The largest measured error in the objective.** `build_masks` picks the
    lineup from projections times the availability HAZARD, so a 70%-available
    starter is chosen on 70% of his rate — but the drawn `active` indicator
    never reaches the mask. When a replication draws him out he contributes an
    exact zero and the roster plays a man short, which no manager does.

    Measured on the shipped bundle (12 ADP rosters, 200 replications, 17
    weeks): **22.5% of chosen-starter player-weeks draw at zero**, and mean
    team-week came out near 73 against about 85 if the lineup may see realized
    availability. Roughly a 15% understatement, and it does not cancel for
    bench depth — which is exactly what the late rounds are priced on.

    **This is not clairvoyance.** §15.3 forbids selecting on realized POINTS;
    who is active is known before kickoff from the injury report, and starting
    your backup when the starter is out is the most ordinary thing in fantasy.
    The substitute's SCORE is still whatever he drew.

    Vectorized rather than looped over replications, which is what makes it
    affordable on a pick clock — the waiver correction is off in the decision
    path precisely because its claim loop is sequential.
    """
    if strategy is not None:
        gate = getattr(strategy.simulation, "starter_substitution", None) or {}
        if not gate.get("enabled", False):
            return team_week

    n_teams, weeks, _ = masks.shape
    positions = np.asarray(bundle.positions)
    groups = _substitution_groups(cfg)
    corrected = team_week.copy()

    for team in range(n_teams):
        roster = np.asarray(rosters[team], dtype=int)
        if roster.size == 0:
            continue
        for week in range(weeks):
            starting = masks[team, week, roster] > 0
            starters = roster[starting]
            bench = roster[~starting]
            if starters.size == 0 or bench.size == 0:
                continue

            for group in groups:
                in_group = np.isin(positions[starters], list(group))
                bench_in_group = np.isin(positions[bench], list(group))
                if not in_group.any() or not bench_in_group.any():
                    continue

                # A starter drawn at exactly zero is out — inactive or on bye.
                # Both are cases where the manager starts someone else.
                out = points[starters[in_group], week, :] == 0.0
                missing = out.sum(axis=0)
                if not missing.any():
                    continue

                # The substitute is chosen by PROJECTION, never by what he
                # went on to score. Ranking the bench by drawn points would be
                # clairvoyance in its purest form — picking the man who
                # happened to have a good week — and §15.3 forbids exactly
                # that. Who is ACTIVE is fair game; it is on the injury report
                # before kickoff. What he scores is not.
                candidates = bench[bench_in_group]
                order = np.argsort(
                    -selection_values(bundle, candidates, week), kind="stable")
                ranked = candidates[order]

                drawn = points[ranked, week, :]
                active = drawn != 0.0

                # Walk the projection order and take the first `missing`
                # ACTIVE players. `rank_among_active` is 1-based position in
                # that walk, so the test is simply "am I within the number of
                # slots that opened".
                rank_among_active = np.cumsum(active, axis=0)
                used = active & (rank_among_active <= missing[None, :])
                gained = (drawn * used).sum(axis=0)
                corrected[team, week, :] += gained.astype(corrected.dtype)
    return corrected


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
