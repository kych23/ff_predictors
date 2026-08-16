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
#: `player_fact` covers the football facts below. The gate refuses any
#: quantity not named here, which is why adding a new kind means adding
#: an entailment rule for it rather than just a prompt line.
QUANTITIES = ("dollars", "probability", "player_fact", "week", "slot")


#: The fixed vocabulary of football facts a narration may cite. Each one is
#: derivable from the board or a fitted model, so the gate can verify it.
#:
#: Deliberately ABSENT, and worth naming: team changes, target share, depth
#: chart moves, holdouts, coaching changes. The bundle carries a player's
#: CURRENT team and no history, so "moved to a new team" is not checkable —
#: and an unverifiable clause is exactly what the gate exists to stop. Adding
#: those means adding the data first, not loosening the prompt.
PLAYER_FACTS = ("games", "consistency", "adp", "ceiling", "floor")


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
    #: Verifiable football facts per player id, for the narration to reason
    #: from. Keys are a fixed vocabulary — see `PLAYER_FACTS`. Everything here
    #: comes off the board or the fitted models, so the gate can check it.
    #: Anything NOT derivable that way (a trade, a holdout, a target share)
    #: is deliberately absent: the model cannot cite what it cannot be
    #: checked on, which is the whole reason the narration is trustworthy.
    player_facts: dict[str, dict[str, float]] = field(default_factory=dict)

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

    def fact_for(self, subject: str) -> float | None:
        """`<player name> <fact>` -> the number, or None if not on record."""
        key = subject.strip().lower()
        for player, facts in self.player_facts.items():
            display = self.display(player).lower()
            for fact, value in facts.items():
                if key in (f"{display} {fact}", f"{player.lower()} {fact}"):
                    return float(value)
        return None

    def fact_subjects(self) -> list[str]:
        return sorted(f"{self.display(p)} {fact}"
                      for p, facts in self.player_facts.items()
                      for fact in facts)

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
                 names: Mapping[str, str] | None = None,
                 player_facts: Mapping[str, Mapping[str, float]] | None = None,
                 ) -> AttributionRecord:
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
        player_facts={p: dict(f) for p, f in (player_facts or {}).items()},
    )
