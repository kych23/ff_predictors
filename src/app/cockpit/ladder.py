"""Degradation ladder (§7.3, §19).

Draft night is n=1 and unforgiving: the probability-weighted cost of an
operations failure at pick 4 exceeds the expected gain from any modelling
layer. So the recommender always returns something, and *demotion is automatic
on a wall-clock check, never a judgment call at the table*.

====  =========================================  ==============================
tier  path                                       demote when
====  =========================================  ==============================
0     two-level sim (theta draws) + allocator    t > demote_after, or it raises
1     single-level sim at the posterior mean     kernel raises, bundle missing
2     VONA + log-normal survival (the v1 path)   sim unavailable
3     static tier list from the bundle           anything else
====  =========================================  ==============================

Tier 1 reuses tier 0's completed parameter draws rather than restarting: if
even one draw finished, its mean is a better answer than throwing it away.

Every recommendation reports the tier that produced it and the stale flags of
its inputs. A number whose provenance is invisible is worse than no number.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Recommendation:
    tier: int
    leader: str
    leader_name: str
    ranked: pd.DataFrame
    indifference_set: list[str]
    dollars: dict[str, float] = field(default_factory=dict)
    aleatory_se: dict[str, float] = field(default_factory=dict)
    epistemic_se: dict[str, float] = field(default_factory=dict)
    p_best: float = float("nan")
    draws_used: int = 0
    stopped_because: str = ""
    separating_axis: str = ""
    stale_flags: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def describe(self) -> str:
        head = (f"tier {self.tier}  {self.leader_name}  "
                f"({self.elapsed_s:.1f}s)")
        if self.tier <= 1 and self.leader in self.dollars:
            head += (f"   E[$] {self.dollars[self.leader]:+.2f} "
                     f"+/- {self.total_se(self.leader):.2f}")
        if len(self.indifference_set) > 1:
            head += f"\n  indifference set: {', '.join(self.indifference_set)}"
        if self.stale_flags:
            head += f"\n  stale: {', '.join(self.stale_flags)}"
        return head

    def total_se(self, candidate: str) -> float:
        a = self.aleatory_se.get(candidate, 0.0)
        e = self.epistemic_se.get(candidate, 0.0)
        return (a ** 2 + e ** 2) ** 0.5


#: A tier callable takes a remaining-seconds budget and returns a
#: Recommendation, or raises to demote.
TierFn = Callable[[float], Recommendation]


def recommend(tiers: dict[int, TierFn], *, budget_s: float,
              demote_after_s: float) -> Recommendation:
    """Try tiers in order; demote on timeout or exception; always return.

    ``demote_after_s`` bounds the simulated tiers only. Tiers 2 and 3 are
    milliseconds and always get the remaining budget, because returning a
    stale-but-real answer beats returning nothing.
    """
    start = time.perf_counter()
    for tier in sorted(tiers):
        remaining = budget_s - (time.perf_counter() - start)
        if remaining <= 0 and tier < 2:
            logger.warning("tier %d skipped: no budget left", tier)
            continue
        allowance = min(remaining, demote_after_s) if tier <= 1 else remaining
        try:
            rec = tiers[tier](allowance)
        except Exception as exc:  # noqa: BLE001 - demotion is the point
            logger.warning("tier %d failed (%s); demoting", tier, exc)
            continue
        elapsed = time.perf_counter() - start
        return Recommendation(**{**rec.__dict__, "elapsed_s": elapsed})

    raise RuntimeError(
        "every tier failed, including the static list. The bundle is "
        "unreadable — there is no data source left (§19)."
    )
