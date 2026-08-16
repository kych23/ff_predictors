"""A 0-100 confidence score that means something specific.

**What it claims:** the chance this recommendation is the better pick than the
next candidate — not "how good is the player", not a star rating.

The temptation with a number like this is to invent one, and an invented score
is worse than none: it reads as authority on a screen where every other figure
is measured. So this is built out of quantities the engine already computes,
and every adjustment below traces to something measured rather than felt.

**The anchor is `p_best`** — the paired bootstrap over the CRN difference
between the leader and the runner-up. That is a genuine probability and it is
already computed on every tier-0 run.

**Everything else shrinks it toward 50, never toward 0.** That direction is the
whole design. A pick the engine cannot support is not *wrong*, it is a coin
flip, and 50 is what a coin flip should read as. Pushing uncertain picks toward
0 would say "take the other guy", which the evidence does not support either.

The shrink factors, and where each number comes from:

* **Reach.** Backtested on 2021-2024 with real outcomes: the engine's ranking
  beats ADP by +0.185 Spearman every year (p=0.0013), but its most aggressive
  disagreements won only **50 of 96** head-to-head pairs — 52.1%, p=0.76.
  Indistinguishable from chance. So confidence decays with how far past ADP the
  pick is, and a 25-pick reach lands near 50 no matter how sure the simulation
  is of itself.
* **Tier.** Tier 2 is the VONA heuristic with no simulation behind it; tier 3
  is a static ADP list. Neither has a `p_best` to anchor on.
* **Truncation.** `stopped_because == "deadline"` means the allocator ran out
  of clock and answered from its initial draws rather than its full budget.
* **No projection.** K, DST and floored players share one fitted value apiece,
  so the model has no opinion to be confident about.
* **Indifference.** A large indifference set is the engine saying, in its own
  words, that it cannot separate these candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: A coin flip. Every shrink pulls toward this, not toward zero.
NEUTRAL = 50.0

#: Reach, in picks past the player's ADP, at which the measured edge is gone.
#: The backtest's aggressive-disagreement bucket averaged ~20 picks and won
#: 52.1% of pairs, so that is where confidence should be neutral.
REACH_NEUTRAL_PICKS = 20.0

#: Reach small enough to carry no penalty — inside a round, the engine's
#: ordering edge is the thing being expressed and it is well supported.
REACH_FREE_PICKS = 6.0

_TIER_FACTOR = {0: 1.0, 1: 0.85, 2: 0.45, 3: 0.20}


@dataclass(frozen=True)
class Confidence:
    """The score plus the reasons, because a bare number is not auditable."""

    score: int
    drivers: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.score >= 75:
            return "strong"
        if self.score >= 60:
            return "moderate"
        if self.score > 52:
            return "slight"
        return "coin flip"


def _reach_factor(adp: float | None, current_pick: int | None) -> float:
    """1.0 when taking a player at or after his ADP, decaying as you reach.

    Linear between `REACH_FREE_PICKS` and `REACH_NEUTRAL_PICKS`, floored at 0
    so a very large reach is exactly neutral rather than negative.
    """
    if adp is None or current_pick is None:
        return 1.0
    reach = float(adp) - float(current_pick)
    if reach <= REACH_FREE_PICKS:
        return 1.0
    span = REACH_NEUTRAL_PICKS - REACH_FREE_PICKS
    return max(0.0, 1.0 - (reach - REACH_FREE_PICKS) / span)


def score(*, tier: int, p_best: float | None, stopped_because: str = "",
          adp: float | None = None, current_pick: int | None = None,
          indifference_size: int = 1, has_projection: bool = True,
          stale_flags: tuple[str, ...] = ()) -> Confidence:
    """Combine what the engine knows into 0-100, with the reasons."""
    drivers: list[str] = []

    if p_best is None or p_best != p_best:          # NaN on tiers 2/3
        base = 55.0
        drivers.append("no simulation behind this pick")
    else:
        base = 100.0 * float(p_best)

    factor = 1.0

    tier_factor = _TIER_FACTOR.get(int(tier), 0.2)
    if tier_factor < 1.0:
        factor *= tier_factor
        drivers.append(f"tier {tier} fallback, not the full simulation")

    if stopped_because == "deadline":
        # Harsh on purpose. A truncated run computes `p_best` from the two
        # initial draws, so the anchor itself is noise — discounting gently
        # would launder a meaningless probability into a confident-looking
        # number.
        factor *= 0.15
        drivers.append("ran out of clock before separating the candidates")

    reach_factor = _reach_factor(adp, current_pick)
    if reach_factor < 1.0 and adp is not None and current_pick is not None:
        factor *= reach_factor
        picks = float(adp) - float(current_pick)
        drivers.append(
            f"{picks:.0f} picks earlier than the market — backtested reaches "
            f"this large won 52% of the time")

    if not has_projection:
        # Every defence shares one fitted value and every kicker another, so a
        # dollar gap between two of them is simulation noise, not a read.
        factor *= 0.3
        drivers.append("no individual projection (K/DST or replacement level)")

    if indifference_size > 1:
        # Each additional indistinguishable candidate is the engine saying it
        # could not separate one more option.
        factor *= max(0.35, 1.0 - 0.15 * (indifference_size - 1))
        drivers.append(
            f"{indifference_size} candidates statistically indistinguishable")

    if stale_flags:
        factor *= 0.85
        drivers.append(f"board flags: {', '.join(stale_flags)}")

    final = NEUTRAL + (base - NEUTRAL) * factor
    return Confidence(score=int(round(min(100.0, max(0.0, final)))),
                      drivers=drivers)
