"""Payout DSL vocabulary and shapes (§10.3, §16).

The *names* live in core so `strategy.py` can validate a prize `type` without
importing the reducers, which take a `LeagueConfig` and live in `domain.payout`
(rank 1). Validating against them from core would be an upward import and the
layer test rejects it — so core owns the vocabulary and domain owns the
implementations, with `domain.payout.reducers` asserting its registry matches
`PRIZE_TYPES` exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from src.core.errors import ConfigError

#: Every prize type that must have a registered reducer (§16.2).
PRIZE_TYPES: Final[frozenset[str]] = frozenset({
    "final_rank",
    "regular_season_rank",
    "season_points_rank",
    "weekly_high_score",
    "weekly_rank",
})

#: Types whose payout occurs once per season rather than once per week.
RANK_TYPES: Final[frozenset[str]] = frozenset({
    "final_rank", "regular_season_rank", "season_points_rank",
})

#: Types that pay per week within a week range.
WEEKLY_TYPES: Final[frozenset[str]] = frozenset({
    "weekly_high_score", "weekly_rank",
})

TiePolicy = Literal["split", "first", "all"]
TIE_POLICIES: Final[frozenset[str]] = frozenset({"split", "first", "all"})


@dataclass(frozen=True)
class Amount:
    """Either an absolute figure or a fraction of the pot. Exactly one."""

    dollars: float | None = None
    pct_of_pot: float | None = None

    def __post_init__(self) -> None:
        if (self.dollars is None) == (self.pct_of_pot is None):
            raise ConfigError(
                "amount must set exactly one of `dollars` or `pct_of_pot`, "
                f"got dollars={self.dollars!r} pct_of_pot={self.pct_of_pot!r}"
            )

    def resolve(self, pot: float) -> float:
        return self.dollars if self.dollars is not None else self.pct_of_pot * pot

    @classmethod
    def from_raw(cls, raw: object, *, where: str) -> Amount:
        if not isinstance(raw, dict):
            raise ConfigError(f"{where}: amount must be a mapping, got {type(raw).__name__}")
        unknown = set(raw) - {"dollars", "pct_of_pot"}
        if unknown:
            raise ConfigError(f"{where}: unknown amount keys {sorted(unknown)}")
        try:
            return cls(
                dollars=float(raw["dollars"]) if "dollars" in raw else None,
                pct_of_pot=float(raw["pct_of_pot"]) if "pct_of_pot" in raw else None,
            )
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}") from None


@dataclass(frozen=True)
class WeekRange:
    """Inclusive at both ends (§10.5 rule 9)."""

    start: int
    end: int

    @property
    def n_weeks(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class PrizeRule:
    id: str
    type: str
    amount: Amount
    ties: TiePolicy = "split"
    rank: int | None = None
    weeks: WeekRange | None = None
    #: Money from outside the pot. Excluded from conservation, and the only
    #: setting under which `ties: all` or a negative amount is legal (§16.2).
    funded_external: bool = False

    def n_occurrences(self) -> int:
        """How many times this prize pays over a season (§10.5 rule 9)."""
        if self.type in WEEKLY_TYPES:
            assert self.weeks is not None  # enforced at parse
            return self.weeks.n_weeks
        return 1
