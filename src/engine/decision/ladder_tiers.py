"""Concrete rungs of the §7.3 degradation ladder, plus §17.4 stage timings.

`app.cockpit.ladder.recommend` owns the *policy* — try tiers in order, demote
on timeout or exception. This module owns the rungs themselves, and one thing
about them that the policy cannot express:

**Tier 1 reuses tier 0's completed parameter draws. It never restarts.**

That is the entire reason tier 1 exists. If tier 0 burns eighteen seconds and
finishes thirty of fifty draws before the wall-clock check demotes it, throwing
those thirty away and starting a fresh single-level run would spend the
remaining budget rediscovering what is already in memory — and would produce a
*worse* answer than the one already computed. The shared `DrawCache` is what
makes the reuse structural rather than a thing someone has to remember.

Timings (§17.4) are recorded per stage into `stage_timings_ms` on every path,
including the failing ones. A demotion whose cause is invisible is a demotion
nobody can fix.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from src.core.errors import DataError
from src.engine.decision.allocate import allocate
from src.engine.decision.recommendation import Recommendation
from src.engine.decision.tiers import tier3_recommendation


@dataclass
class StageTimings:
    """§17.4. Mutable by design — stages report as they complete."""

    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            # recorded even when the stage RAISES: the timing of a failure is
            # what tells you whether it timed out or died instantly
            self.stage_timings_ms[name] = round(
                (time.perf_counter() - start) * 1000, 2)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stage_timings_ms.values()), 2)

    def describe(self) -> str:
        rows = sorted(self.stage_timings_ms.items(), key=lambda kv: -kv[1])
        return "  ".join(f"{name} {ms:.0f}ms" for name, ms in rows)


class DrawCache:
    """(candidate, draw_index) -> (R,) dollars, shared across tiers.

    Every evaluation is memoized here, so a draw computed under tier 0 is free
    to tier 1. Draw indices are consecutive from zero — the property §15.1's
    RNG addressing exists to provide — so a cached prefix is exactly the
    prefix a re-run would recompute.
    """

    def __init__(self, evaluate: Callable[[str, int], np.ndarray]) -> None:
        self._evaluate = evaluate
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self.calls = 0
        self.hits = 0

    def __call__(self, candidate: str, draw: int) -> np.ndarray:
        key = (candidate, draw)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        value = self._evaluate(candidate, draw)
        self._cache[key] = value
        return value

    def completed(self) -> dict[str, np.ndarray]:
        """(K, R) per candidate over the draws that actually finished."""
        by_candidate: dict[str, list[np.ndarray]] = {}
        for (candidate, draw), value in sorted(self._cache.items()):
            by_candidate.setdefault(candidate, []).append((draw, value))
        return {c: np.array([v for _, v in sorted(rows)])
                for c, rows in by_candidate.items()}

    @property
    def n_draws(self) -> int:
        return len(self._cache)


def make_tier0(cache: DrawCache, candidates: list[str], names: dict[str, str],
               timings: StageTimings, *, indifference_zone: float,
               initial_draws: int, max_draws: int, seed: int = 0) -> Callable:
    """Two-level sim + successive halving — the full path."""

    def run(allowance_s: float) -> Recommendation:
        with timings.stage("tier0_allocate"):
            result = allocate(
                candidates, cache, initial_draws=initial_draws,
                max_draws=max_draws, indifference_zone=indifference_zone,
                deadline=time.perf_counter() + allowance_s, seed=seed,
            )
        return _from_allocation(result, names, tier=0, timings=timings)

    return run


def make_tier1(cache: DrawCache, names: dict[str, str],
               timings: StageTimings, *, indifference_zone: float
               ) -> Callable:
    """Mean over tier 0's COMPLETED draws. No new simulation at all.

    Deliberately does not call `allocate`: halving would want to spend budget
    that, by the time this rung runs, does not exist. Every candidate is scored
    on whatever it already has, and candidates with unequal draw counts are
    reported as such rather than silently compared.
    """

    def run(allowance_s: float) -> Recommendation:
        with timings.stage("tier1_reuse"):
            completed = cache.completed()
            if not completed:
                raise DataError(
                    "tier 1 has nothing to reuse: tier 0 completed zero draws"
                )
            means = {c: float(v.mean()) for c, v in completed.items()}
            leader = max(means, key=means.get)

            aleatory, epistemic = {}, {}
            for c, draws in completed.items():
                k, r = draws.shape
                # same two estimators as allocate: CRN means R independent
                # aleatory samples, not K*R
                aleatory[c] = (float(draws.mean(axis=0).std(ddof=1) / np.sqrt(r))
                               if r > 1 else 0.0)
                epistemic[c] = (float(draws.mean(axis=1).std(ddof=1) / np.sqrt(k))
                                if k > 1 else 0.0)

            indifferent = [leader]
            for c in means:
                if c == leader:
                    continue
                gap = means[leader] - means[c]
                se = np.sqrt(aleatory[leader] ** 2 + epistemic[leader] ** 2
                             + aleatory[c] ** 2 + epistemic[c] ** 2)
                if abs(gap) < indifference_zone or abs(gap) < 1.645 * se:
                    indifferent.append(c)

            ranked = pd.DataFrame(
                [{"candidate": c, "name": names.get(c, c), "dollars": m,
                  "draws_used": completed[c].shape[0]}
                 for c, m in sorted(means.items(), key=lambda kv: -kv[1])]
            )

        return Recommendation(
            tier=1, leader=leader, leader_name=names.get(leader, leader),
            ranked=ranked, indifference_set=indifferent, dollars=means,
            aleatory_se=aleatory, epistemic_se=epistemic,
            draws_used=min(v.shape[0] for v in completed.values()),
            stopped_because="reused_tier0_draws",
            stale_flags=(["unequal_draw_counts"]
                         if len({v.shape[0] for v in completed.values()}) > 1
                         else []),
        )

    return run


def make_tier3(tier_table: pd.DataFrame, needed_positions: list[str],
               drafted: set[str], timings: StageTimings) -> Callable:
    """Static board. No simulator, no model objects, one parquet file."""

    def run(allowance_s: float) -> Recommendation:
        with timings.stage("tier3_static"):
            pick = tier3_recommendation(tier_table, needed_positions, drafted)
            if pick is None:
                raise DataError(
                    "tier 3 exhausted: every needed position is drafted out"
                )
            pool = tier_table[
                tier_table["position"].isin(needed_positions)
                & ~tier_table["player_id"].astype(str).isin(drafted)
            ].sort_values(["tier", "rank"]).head(6)
        return Recommendation(
            tier=3, leader=str(pick["player_id"]),
            leader_name=str(pick["player_name"]),
            ranked=pool.reset_index(drop=True),
            # everyone in the same tier is, by construction, interchangeable
            indifference_set=[str(p) for p in
                              pool[pool["tier"] == pick["tier"]]["player_id"]],
            stopped_because="static_tier_list",
            stale_flags=["no_simulation", "no_dollars"],
        )

    return run


def _from_allocation(result, names: dict[str, str], *, tier: int,
                     timings: StageTimings) -> Recommendation:
    ranked = pd.DataFrame(
        [{"candidate": e.candidate, "name": names.get(e.candidate, e.candidate),
          "dollars": e.mean, "total_se": e.total_se,
          "draws_used": e.draws_used}
         for e in sorted(result.estimates.values(), key=lambda e: -e.mean)]
    )
    return Recommendation(
        tier=tier, leader=result.leader,
        leader_name=names.get(result.leader, result.leader),
        ranked=ranked, indifference_set=result.indifference_set,
        dollars={c: e.mean for c, e in result.estimates.items()},
        aleatory_se={c: e.aleatory_se for c, e in result.estimates.items()},
        epistemic_se={c: e.epistemic_se for c, e in result.estimates.items()},
        p_best=result.p_best, draws_used=result.estimates[result.leader].draws_used,
        stopped_because=result.stopped_because,
        separating_axis=result.separating_axis,
    )
