"""Objective backtest (§21.6) — is the simulator's *distribution* right?

Every other check asks whether the engine ranks players well. This asks
something the ranking cannot answer: when the simulator says a roster will
score 118 points with an 80% interval of 92–147, does that interval contain
the truth 80% of the time?

Two statistics, because the objective needs both to be right:

**PIT (probability integral transform).** For each realized roster-week total
`y`, the predicted CDF evaluated at `y` — in practice the fraction of
simulated draws below it. If the predictive distribution is correct, PIT is
Uniform(0,1); a KS test against uniform is the gate. The SHAPE of the failure
is the diagnostic and is reported rather than collapsed into the p-value:

    ∪-shaped (mass at 0 and 1)  intervals TOO NARROW — overconfident
    ∩-shaped (mass at 0.5)      intervals TOO WIDE  — underconfident
    sloped                      biased high or low

**Weekly-high frequency.** The payout pays $6 for a weekly high (§16), so the
objective depends on the top of the twelve-roster distribution, not the middle.
PIT can look perfect while the tail is wrong, because PIT averages over the
whole distribution and the weekly-high prize lives entirely in its corner.

**This gate is `descriptive_only` and the §21.5 audit says why**: 8 seasons is
n=8 for a season-level statistic, and the power audit puts the sample needed
for 80% power at 113 seasons. Read the numbers, do not claim them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kstest

#: Nominal coverage of the weekly-high predictive interval (§21.6).
INTERVAL_MASS = 0.90


@dataclass(frozen=True)
class PITResult:
    values: np.ndarray
    ks_statistic: float
    p_value: float
    alpha: float

    @property
    def uniform(self) -> bool:
        return self.p_value >= self.alpha

    @property
    def mean(self) -> float:
        """Uniform has mean 0.5. Below means predictions run HIGH."""
        return float(self.values.mean())

    @property
    def slope(self) -> float:
        """OLS slope of the ten histogram bins, in counts-per-unit-PIT.

        The mean alone is a weak detector of a tilt: a clean monotone decline
        across the bins can sit at mean 0.45 and pass every threshold written
        in terms of edge and centre mass. Measured on the real backtest, the
        histogram fell from ratio 1.28 to 0.75 — obviously sloped, and the
        first version of this classifier called it 'no simple shape'.
        """
        counts, edges = np.histogram(self.values, bins=10, range=(0.0, 1.0))
        centres = (edges[:-1] + edges[1:]) / 2.0
        expected = max(len(self.values) / 10, 1)
        return float(np.polyfit(centres, counts / expected, 1)[0])

    @property
    def shape(self) -> str:
        """Which way the intervals are wrong, in words.

        A p-value says 'not uniform'. It does not say whether to widen the
        intervals, narrow them, or move them — and that is the only actionable
        part. Checked in order of how much the diagnosis changes what you do.
        """
        if self.uniform:
            return "uniform"
        # BOTH edges, checked separately. Summing them cannot tell a U from a
        # one-sided pile-up, and the two call for opposite remedies: widen the
        # intervals versus move them. A forecast shifted a full SD puts ~40% of
        # its PIT mass in one outer bin and was diagnosed "too narrow" by the
        # summed version — which would have sent someone off to inflate sigma
        # when the level was the problem.
        low = float(np.mean(self.values < 0.1))
        high = float(np.mean(self.values > 0.9))
        middle = float(np.mean((self.values > 0.4) & (self.values < 0.6)))
        if low > 0.14 and high > 0.14:
            return (f"U-shaped ({low:.0%} low, {high:.0%} high): intervals TOO "
                    f"NARROW — overconfident")
        if middle > 0.28:
            return (f"inverted-U ({middle:.0%} in the centre): intervals TOO "
                    f"WIDE — underconfident")
        if self.slope < -0.15:
            return (f"sloped down (slope {self.slope:+.2f}, mean PIT "
                    f"{self.mean:.3f}): predictions biased HIGH — realized "
                    f"totals land below the middle of the forecast")
        if self.slope > 0.15:
            return (f"sloped up (slope {self.slope:+.2f}, mean PIT "
                    f"{self.mean:.3f}): predictions biased LOW — realized "
                    f"totals land above the middle of the forecast")
        return f"non-uniform, no simple shape (mean PIT {self.mean:.3f})"

    def histogram(self, bins: int = 10) -> pd.DataFrame:
        counts, edges = np.histogram(self.values, bins=bins, range=(0.0, 1.0))
        expected = len(self.values) / bins
        return pd.DataFrame({
            "bin_lo": edges[:-1], "bin_hi": edges[1:], "count": counts,
            "expected": expected,
            "ratio": counts / expected if expected else np.nan,
        })


def pit_values(simulated: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Fraction of draws strictly below each realized value.

    `simulated` is (N, R); `realized` is (N,). Ties are broken by adding half
    the equal mass — the standard randomized-PIT correction. Without it a
    discrete predictive distribution (K/DST draws come from an empirical CDF)
    piles PIT mass on exact grid points and fails a KS test for a reason that
    has nothing to do with calibration.
    """
    simulated = np.asarray(simulated, dtype=float)
    realized = np.asarray(realized, dtype=float)
    if simulated.ndim != 2:
        raise ValueError(f"simulated must be (N, R), got {simulated.shape}")
    if len(realized) != simulated.shape[0]:
        raise ValueError(
            f"{len(realized)} realized values against {simulated.shape[0]} "
            f"simulated rows"
        )
    below = (simulated < realized[:, None]).mean(axis=1)
    equal = (simulated == realized[:, None]).mean(axis=1)
    return below + 0.5 * equal


def pit_test(simulated: np.ndarray, realized: np.ndarray, *,
             alpha: float = 0.05) -> PITResult:
    values = pit_values(simulated, realized)
    stat, p = kstest(values, "uniform")
    return PITResult(values=values, ks_statistic=float(stat),
                     p_value=float(p), alpha=alpha)


@dataclass(frozen=True)
class WeeklyHighResult:
    season: int
    realized_rate: float
    predicted_lo: float
    predicted_hi: float
    predicted_mean: float
    n_weeks: int

    @property
    def inside(self) -> bool:
        return self.predicted_lo <= self.realized_rate <= self.predicted_hi


def weekly_high_coverage(team_week_sim: np.ndarray, realized: np.ndarray,
                         team: int, season: int, *,
                         mass: float = INTERVAL_MASS) -> WeeklyHighResult:
    """Did `team` top the league as often as the simulator said it would?

    `team_week_sim` is (T, W, R); `realized` is (T, W). The predictive interval
    is over the SEASON RATE, taken across replications — not over individual
    weeks. A per-week interval would be a Bernoulli and its 90% interval is
    {0, 1}, which contains everything and tests nothing.
    """
    if team_week_sim.ndim != 3:
        raise ValueError(f"expected (T, W, R), got {team_week_sim.shape}")
    if realized.shape != team_week_sim.shape[:2]:
        raise ValueError(
            f"realized {realized.shape} does not match simulated "
            f"{team_week_sim.shape[:2]}"
        )
    weeks = team_week_sim.shape[1]
    won_sim = (team_week_sim.argmax(axis=0) == team)      # (W, R)
    rate_by_rep = won_sim.mean(axis=0)                    # (R,)
    tail = (1.0 - mass) / 2.0
    return WeeklyHighResult(
        season=season,
        realized_rate=float((realized.argmax(axis=0) == team).mean()),
        predicted_lo=float(np.quantile(rate_by_rep, tail)),
        predicted_hi=float(np.quantile(rate_by_rep, 1.0 - tail)),
        predicted_mean=float(rate_by_rep.mean()),
        n_weeks=weeks,
    )


@dataclass(frozen=True)
class BacktestReport:
    pit: PITResult
    weekly_high: list[WeeklyHighResult]
    seasons: list[int]
    instrument: str
    n_roster_weeks: int
    #: §21.5 rates this gate descriptive_only. Carried so a reader of the
    #: artifact cannot separate the number from its licence.
    verdict: str = "descriptive_only"
    notes: list[str] = field(default_factory=list)

    @property
    def seasons_inside(self) -> int:
        return sum(w.inside for w in self.weekly_high)

    @property
    def passes(self) -> bool:
        """§21.6's stated criterion. Reported, NOT claimed — see `verdict`."""
        required = max(len(self.weekly_high) - 1, 0)
        return self.pit.uniform and self.seasons_inside >= required

    def describe(self) -> str:
        lines = [
            f"instrument: {self.instrument}",
            f"{self.n_roster_weeks:,} roster-weeks over {len(self.seasons)} seasons",
            f"PIT: KS={self.pit.ks_statistic:.4f}  p={self.pit.p_value:.4f}  "
            f"-> {self.pit.shape}",
            f"weekly high: {self.seasons_inside}/{len(self.weekly_high)} seasons "
            f"inside the {INTERVAL_MASS:.0%} interval",
            f"criterion: {'MET' if self.passes else 'NOT MET'}  "
            f"(verdict: {self.verdict})",
        ]
        return "\n".join(lines)
