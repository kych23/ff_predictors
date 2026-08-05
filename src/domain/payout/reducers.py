"""Prize reducers (§16.1, §16.2).

Adding a prize structure is adding one reducer and nothing else. Each returns
dollars shaped ``(T, R)``.

The registry is asserted against ``core.config.payout_schema.PRIZE_TYPES`` at
import, so the vocabulary the config validates against and the set actually
implemented cannot drift. Core owns the names because ``core.config.strategy``
must validate a prize ``type`` without importing this module, which takes a
``LeagueConfig`` (§9.0).

Amounts arrive already resolved: ``pct_of_pot`` needs the pot, the pot needs
``buy_in``, and ``buy_in`` lives in strategy — so the compiler resolves once and
passes a float, rather than reducers reaching back into config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from src.core.config.league import LeagueConfig
from src.core.config.payout_schema import PRIZE_TYPES, PrizeRule
from src.domain.schedule.bracket import resolve_rank


@dataclass(frozen=True)
class SeasonOutcome:
    """What the kernel hands the objective (§16.1)."""

    team_week: np.ndarray      # (T, W, R) float32
    final_rank: np.ndarray     # (T, R) int, 1..T
    regular_rank: np.ndarray   # (T, R) int, 1..T
    season_points: np.ndarray  # (T, R) float
    my_team_index: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.final_rank.shape  # type: ignore[return-value]


class Reducer(Protocol):
    def __call__(
        self, sim: SeasonOutcome, prize: PrizeRule, league: LeagueConfig, amount: float
    ) -> np.ndarray: ...


REDUCERS: dict[str, Reducer] = {}


def register(name: str) -> Callable[[Reducer], Reducer]:
    def wrap(fn: Reducer) -> Reducer:
        if name in REDUCERS:
            raise RuntimeError(f"reducer {name!r} already registered")
        REDUCERS[name] = fn
        return fn
    return wrap


def _pay(winners: np.ndarray, amount: float, ties: str) -> np.ndarray:
    """Distribute ``amount`` over a boolean winner mask, shaped (T, R).

    ``split`` divides among tied teams — the only policy that conserves the
    pot, which is why §10.5 rule 12 restricts the others to externally funded
    prizes. ``first`` awards the lowest team index. ``all`` pays each in full.
    """
    if ties == "all":
        return winners * amount
    if ties == "first":
        out = np.zeros(winners.shape, dtype="float64")
        any_winner = winners.any(axis=0)
        first = np.argmax(winners, axis=0)
        out[first[any_winner], np.flatnonzero(any_winner)] = amount
        return out
    if ties != "split":
        raise ValueError(f"unknown ties policy {ties!r}")
    n = winners.sum(axis=0, keepdims=True).astype("float64")
    per = np.divide(amount, n, out=np.zeros_like(n), where=n > 0)
    return winners * per


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Dense 1..T ranking of a (T, R) score array, highest first."""
    order = (-scores).argsort(axis=0, kind="stable")
    ranks = np.empty(order.shape, dtype="int32")
    ladder = np.tile(np.arange(1, scores.shape[0] + 1)[:, None], (1, scores.shape[1]))
    np.put_along_axis(ranks, order, ladder, axis=0)
    return ranks


@register("final_rank")
def final_rank(sim, prize, league, amount):  # noqa: ANN001
    target = resolve_rank(int(prize.rank), league.teams)
    return _pay(sim.final_rank == target, amount, prize.ties)


@register("regular_season_rank")
def regular_season_rank(sim, prize, league, amount):  # noqa: ANN001
    target = resolve_rank(int(prize.rank), league.teams)
    return _pay(sim.regular_rank == target, amount, prize.ties)


@register("season_points_rank")
def season_points_rank(sim, prize, league, amount):  # noqa: ANN001
    target = resolve_rank(int(prize.rank), league.teams)
    return _pay(_ranks_from_scores(sim.season_points) == target, amount, prize.ties)


@register("weekly_high_score")
def weekly_high_score(sim, prize, league, amount):  # noqa: ANN001
    """Needs only ``team_week`` — no bracket, no waiver, no trajectory.

    The cheapest term to compute and, per §5.3, the most sigma-elastic, which
    is why degraded tiers can still produce a useful answer.
    """
    out = np.zeros(sim.shape, dtype="float64")
    for week in range(prize.weeks.start, prize.weeks.end + 1):
        col = sim.team_week[:, week - 1, :]
        out += _pay(col == col.max(axis=0, keepdims=True), amount, prize.ties)
    return out


@register("weekly_rank")
def weekly_rank(sim, prize, league, amount):  # noqa: ANN001
    target = resolve_rank(int(prize.rank), league.teams)
    out = np.zeros(sim.shape, dtype="float64")
    for week in range(prize.weeks.start, prize.weeks.end + 1):
        ranks = _ranks_from_scores(sim.team_week[:, week - 1, :])
        out += _pay(ranks == target, amount, prize.ties)
    return out


_missing = PRIZE_TYPES - set(REDUCERS)
_extra = set(REDUCERS) - PRIZE_TYPES
if _missing or _extra:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        f"reducer registry disagrees with core PRIZE_TYPES: "
        f"missing={sorted(_missing)} unexpected={sorted(_extra)}"
    )
