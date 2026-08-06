"""The Recommendation record (§17.3).

This lives in `engine` rather than in `app` because the engine PRODUCES it:
`ladder_tiers` builds one on every rung. Keeping it in `app.cockpit` forced
`engine` to import `app`, which inverts the §9.0 dependency rule — and
`tests/test_layer_deps.py` is right to reject that. The cockpit imports this
record; the record knows nothing about a cockpit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Recommendation:
    tier: int
    leader: str
    leader_name: str
    ranked: pd.DataFrame
    indifference_set: list[str]
    dollars: dict[str, float] = field(default_factory=dict)
    aleatory_se: dict[str, float] = field(default_factory=dict)
    #: correlation parameters only, per §15.5 — NOT total model uncertainty
    epistemic_se: dict[str, float] = field(default_factory=dict)
    p_best: float = float("nan")
    draws_used: int = 0
    stopped_because: str = ""
    separating_axis: str = ""
    stale_flags: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    #: §17.4 — per-stage wall clock, populated even on demoted paths
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        head = f"tier {self.tier}  {self.leader_name}  ({self.elapsed_s:.1f}s)"
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
