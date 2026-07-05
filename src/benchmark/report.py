"""Benchmark aggregation + significance (DrafterSpec.md §4.7 / §4.7.1).

Aggregates results across seasons into distributions with significance, and reports
metrics BOTH overall and on the data-complete era (>= 2016, where depth-chart/role
coverage is solid). The complete-era result is the honest forward number — pre-2015
role-change data is sparse and understates the deployed edge (§4.7). The ~8-9
evaluable-season ceiling (§4.6.1) is stated, never overclaimed as "many".

This package may read ADP/market data (benchmark side of the wall, §4.0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover
    wilcoxon = None

DATA_COMPLETE_ERA_START = 2016

# Exact two-sided sign/Wilcoxon tests cannot reach p < 0.05 below n = 6
# (minimum achievable p = 2 * 0.5^n): running them there makes the gate
# mathematically impossible to pass, silently. Below that we gate on the
# bootstrap CI alone and say so; below MIN_N_FOR_GATE nothing can pass.
MIN_N_FOR_P_TESTS = 6
MIN_N_FOR_GATE = 5


@dataclass
class PairedTest:
    n: int
    mean_diff: float
    wilcoxon_p: Optional[float]
    sign_test_p: Optional[float]
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        """PASS = positive mean diff, bootstrap CI excludes 0, and — when a
        powered p-test exists (n >= MIN_N_FOR_P_TESTS) — p < 0.05 (§4.7.1)."""
        if self.n < MIN_N_FOR_GATE:
            return False
        if not (self.mean_diff > 0 and self.ci_low > 0):
            return False
        p = self.wilcoxon_p if self.wilcoxon_p is not None else self.sign_test_p
        if p is not None:
            return p < 0.05
        return True


def _sign_test_p(diffs: np.ndarray) -> Optional[float]:
    """Two-sided sign test via the binomial distribution."""
    from scipy.stats import binomtest
    pos = int((diffs > 0).sum())
    neg = int((diffs < 0).sum())
    n = pos + neg
    if n == 0:
        return None
    return float(binomtest(pos, n, 0.5, alternative="two-sided").pvalue)


def bootstrap_ci(diffs: Sequence[float], *, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    diffs = np.asarray([d for d in diffs if not np.isnan(d)], dtype=float)
    if diffs.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, diffs.size), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_test(engine: Sequence[float], baseline: Sequence[float]) -> PairedTest:
    """Paired per-season test of (engine - baseline), per §4.7.1 gate logic."""
    e = np.asarray(engine, dtype=float)
    b = np.asarray(baseline, dtype=float)
    diffs = e - b
    mask = ~np.isnan(diffs)
    diffs = diffs[mask]
    n = diffs.size
    wp = None
    sp = None
    if n >= MIN_N_FOR_P_TESTS:
        if wilcoxon is not None and np.any(diffs != 0):
            try:
                wp = float(wilcoxon(diffs).pvalue)
            except ValueError:
                wp = None
        sp = _sign_test_p(diffs)
    lo, hi = bootstrap_ci(diffs)
    return PairedTest(n=n, mean_diff=float(np.mean(diffs)) if n else float("nan"),
                      wilcoxon_p=wp, sign_test_p=sp,
                      ci_low=lo, ci_high=hi)


def format_gate(name: str, test: PairedTest, ceiling_note: bool = True) -> str:
    lines = [
        f"[{name}] paired seasons n={test.n}",
        f"  mean(engine - baseline) = {test.mean_diff:+.4f}",
        f"  wilcoxon p = {test.wilcoxon_p if test.wilcoxon_p is not None else 'n/a'}",
        f"  sign-test p = {test.sign_test_p if test.sign_test_p is not None else 'n/a'}",
        f"  bootstrap 95% CI = [{test.ci_low:+.4f}, {test.ci_high:+.4f}]",
        f"  GATE {'PASS' if test.significant else 'FAIL'}",
    ]
    if test.n < MIN_N_FOR_GATE:
        lines.append(f"  note: n < {MIN_N_FOR_GATE} — too few seasons for any gate to pass.")
    elif test.n < MIN_N_FOR_P_TESTS:
        lines.append(f"  note: n < {MIN_N_FOR_P_TESTS} — exact p-tests are unpowered "
                     "at 0.05; gate uses the bootstrap CI only.")
    if ceiling_note:
        lines.append("  note: ~8-9 evaluable seasons is the honest ceiling, not 'many'.")
    return "\n".join(lines)
