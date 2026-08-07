"""The attribution record (§18) — the only thing narration is allowed to know.

Instrumented from `decompose` rather than reconstructed afterwards. §18 is
explicit about why: retrofitting it late yields an unenforced prompt
instruction, and an unenforced instruction is worse than no gate because it
launders unverified claims as verified.

Everything the narrator may say has to be a lookup on this object. If a fact is
not here, no sentence can assert it and pass the gate — which is the design,
not a limitation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

#: The quantity kinds a claim may carry. Fixed, because the gate needs one
#: entailment rule per kind and an unknown kind has no rule.
QUANTITIES = ("dollars", "probability", "week", "slot")


@dataclass(frozen=True)
class AttributionRecord:
    """Why the leader beats the runner-up, decomposed."""

    pair: tuple[str, str]
    delta_by_prize: dict[str, float]
    delta_weeks: list[tuple[int, float]]
    roster_slot_affected: str
    aleatory_se: float
    epistemic_se: float
    survival_probabilities: dict[str, float] = field(default_factory=dict)
    bye_conflicts: list[int] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)

    @property
    def total_delta(self) -> float:
        return float(sum(self.delta_by_prize.values()))

    @property
    def total_se(self) -> float:
        return float(np.hypot(self.aleatory_se, self.epistemic_se))

    def display(self, candidate: str) -> str:
        return self.names.get(candidate, candidate)

    # ------------------------------------------------------------- lookups
    def dollars_for(self, subject: str) -> float | None:
        """Prize id, `total`, or a standard-error name."""
        key = subject.strip().lower()
        if key in ("total", "total_dollars", "overall"):
            return self.total_delta
        if key in ("aleatory_se", "aleatory"):
            return self.aleatory_se
        if key in ("epistemic_se", "epistemic"):
            return self.epistemic_se
        if key in ("total_se", "se", "standard_error"):
            return self.total_se
        for prize, value in self.delta_by_prize.items():
            if prize.lower() == key:
                return float(value)
        return None

    def probability_for(self, subject: str) -> float | None:
        """Survival probability, by player id or by display name."""
        key = subject.strip().lower()
        for player, p in self.survival_probabilities.items():
            if player.lower() == key or self.display(player).lower() == key:
                return float(p)
        return None

    def weeks(self) -> set[int]:
        return {int(w) for w, _ in self.delta_weeks} | set(self.bye_conflicts)

    def week_delta(self, week: int) -> float | None:
        for w, value in self.delta_weeks:
            if int(w) == int(week):
                return float(value)
        return None

    def top_weeks(self, n: int = 3) -> list[tuple[int, float]]:
        return sorted(self.delta_weeks, key=lambda wv: -abs(wv[1]))[:n]


def build_record(leader: str, runner_up: str, *,
                 leader_parts: Mapping[str, np.ndarray],
                 runner_parts: Mapping[str, np.ndarray],
                 seat: int,
                 leader_team_week: np.ndarray,
                 runner_team_week: np.ndarray,
                 aleatory_se: float, epistemic_se: float,
                 roster_slot_affected: str = "",
                 survival_probabilities: Mapping[str, float] | None = None,
                 bye_conflicts: list[int] | None = None,
                 names: Mapping[str, str] | None = None) -> AttributionRecord:
    """Assemble from two `decompose` outputs and their team-week tensors.

    `*_parts` map prize id -> (T, R) dollars, exactly what
    `CompiledPayout.decompose` returns. The subtraction happens here so the
    narration layer never sees a raw simulation array and cannot compute a
    number the gate has no way to check.
    """
    prizes = sorted(set(leader_parts) | set(runner_parts))
    delta_by_prize = {}
    for prize in prizes:
        a = leader_parts.get(prize)
        b = runner_parts.get(prize)
        av = float(a[seat].mean()) if a is not None else 0.0
        bv = float(b[seat].mean()) if b is not None else 0.0
        delta_by_prize[prize] = av - bv

    weekly = (leader_team_week[seat].mean(axis=-1)
              - runner_team_week[seat].mean(axis=-1))
    delta_weeks = [(w + 1, float(v)) for w, v in enumerate(weekly)]

    return AttributionRecord(
        pair=(leader, runner_up),
        delta_by_prize=delta_by_prize,
        delta_weeks=delta_weeks,
        roster_slot_affected=roster_slot_affected,
        aleatory_se=float(aleatory_se), epistemic_se=float(epistemic_se),
        survival_probabilities=dict(survival_probabilities or {}),
        bye_conflicts=list(bye_conflicts or []),
        names=dict(names or {}),
    )
