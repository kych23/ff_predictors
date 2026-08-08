"""Compile the payout DSL into a vectorized objective (§16.1).

All validation already ran in ``core.config.strategy.load_strategy``; this
raises only if a type slipped through with no reducer, which would mean the
core vocabulary and this registry disagreed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.config.league import LeagueConfig
from src.core.config.strategy import PayoutConfig
from src.core.errors import UnknownPrizeTypeError
from src.domain.payout.reducers import REDUCERS, SeasonOutcome


@dataclass(frozen=True)
class CompiledObjective:
    """Dollars per team per replication, plus a per-prize decomposition.

    ``decompose`` exists from day one because the narration layer's attribution
    record is built from it (§18). Retrofitting it late yields an unenforced
    prompt instruction, which is worse than no gate — it launders unverified
    claims as verified.
    """

    prize_ids: tuple[str, ...]
    _plan: tuple[tuple[str, object, float], ...]  # (id, PrizeRule, amount)
    _league: LeagueConfig

    def __call__(self, sim: SeasonOutcome) -> np.ndarray:
        total = np.zeros(sim.shape, dtype="float64")
        for _, prize, amount in self._plan:
            total += REDUCERS[prize.type](sim, prize, self._league, amount)
        return total

    def decompose(self, sim: SeasonOutcome) -> dict[str, np.ndarray]:
        return {
            pid: REDUCERS[prize.type](sim, prize, self._league, amount)
            for pid, prize, amount in self._plan
        }

    def my_dollars(self, sim: SeasonOutcome) -> np.ndarray:
        """(R,) dollars for my seat — what the decision layer actually reads."""
        return self(sim)[sim.my_team_index]

    @property
    def needs_bracket(self) -> bool:
        """True when some prize reads a rank.

        §16.4 describes this as letting the kernel skip the bracket and the
        schedule entirely. It does not: `kernel.season_outcome` computes both
        unconditionally and nothing in `src/` calls this. It is a correct
        predicate that no caller consults — kept because the optimisation is
        still the right one to make, but do not read the docstring as a
        description of what runs today.
        """
        return any(
            prize.type in ("final_rank", "regular_season_rank")
            for _, prize, _ in self._plan
        )


def compile_payout(payout: PayoutConfig, league: LeagueConfig) -> CompiledObjective:
    pot = payout.pot(league.teams)
    plan = []
    for prize in payout.prizes:
        if prize.type not in REDUCERS:
            raise UnknownPrizeTypeError(
                f"prize {prize.id!r} has type {prize.type!r} with no reducer; "
                f"registered: {sorted(REDUCERS)}"
            )
        plan.append((prize.id, prize, prize.amount.resolve(pot)))
    return CompiledObjective(
        prize_ids=tuple(p.id for p in payout.prizes),
        _plan=tuple(plan),
        _league=league,
    )
