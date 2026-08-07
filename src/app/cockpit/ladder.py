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
from collections.abc import Callable

# Re-exported: the record is produced by the engine and merely presented here
# (§9.0 — app may import engine, never the reverse).
from src.engine.decision.recommendation import Recommendation

logger = logging.getLogger(__name__)

__all__ = ["Recommendation", "TierFn", "recommend"]


#: A tier callable takes a remaining-seconds budget and returns a
#: Recommendation, or raises to demote.
TierFn = Callable[[float], Recommendation]


def recommend(tiers: dict[int, TierFn], *, budget_s: float,
              demote_after_s: float,
              timings: dict[str, float] | None = None) -> Recommendation:
    """Try tiers in order; demote on timeout or exception; always return.

    ``demote_after_s`` bounds the simulated tiers only. Tiers 2 and 3 are
    milliseconds and always get the remaining budget, because returning a
    stale-but-real answer beats returning nothing.

    ``timings`` is the §17.4 stage map, shared with the tier callables. It is
    attached to the returned recommendation INCLUDING the stages of tiers that
    failed on the way down — a demotion whose cause is invisible is a demotion
    nobody can fix.
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
        return Recommendation(**{**rec.__dict__, "elapsed_s": elapsed,
                                 "stage_timings_ms": dict(timings or {})})

    raise RuntimeError(
        "every tier failed, including the static list. The bundle is "
        "unreadable — there is no data source left (§19)."
    )
