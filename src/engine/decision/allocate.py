"""Budget allocation and the indifference set (§15.5, §17.2, §17.3, AD-15).

One knob, not three. A *budget unit* is one (candidate, parameter-draw)
evaluation at fixed ``inner_seasons``. Successive halving runs over
**candidates**, not replications:

    round 1: every candidate gets ``initial_draws_per_candidate`` draws
    round j: keep ceil(n/2) by mean, DOUBLE each survivor's draw count
    then:    keep doubling the sole survivor until the deadline or the ceiling

Draw indices are consecutive from 0, so doubling REUSES earlier draws — the
property §15.1's addressing exists to provide.

The two standard errors mean different things and are estimated differently,
because §15.1 pins ``c = 0`` on every aleatory stream: the same ``R`` uniforms
serve all ``K`` parameter draws, so there are only ``R`` independent aleatory
samples, not ``K*R``.

    aleatory_se  = sd_r(mean_k D[r, :]) / sqrt(R)
    epistemic_se = sd_k(mean_r D[:, k]) / sqrt(K)

Dividing the aleatory term by ``K*R`` — as an earlier revision did — assumes an
independence the CRN design deliberately removes, and understates it by up to
sqrt(K) ~ 7 at 50 draws. That error propagates straight into the indifference
CI and the early stop, separating candidates that have not separated.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CandidateEstimate:
    candidate: str
    mean: float
    aleatory_se: float
    epistemic_se: float
    draws_used: int
    reps: int

    @property
    def total_se(self) -> float:
        return math.sqrt(self.aleatory_se ** 2 + self.epistemic_se ** 2)


@dataclass(frozen=True)
class AllocationResult:
    estimates: dict[str, CandidateEstimate]
    leader: str
    indifference_set: list[str]
    p_best: float
    stopped_because: str
    units_spent: int
    separating_axis: str = ""
    samples: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def _summarize(candidate: str, draws: np.ndarray) -> CandidateEstimate:
    """``draws`` is (K, R): one row per parameter draw."""
    k, r = draws.shape
    per_draw = draws.mean(axis=1)               # (K,)
    per_rep = draws.mean(axis=0)                # (R,)
    aleatory = float(per_rep.std(ddof=1) / math.sqrt(r)) if r > 1 else 0.0
    epistemic = float(per_draw.std(ddof=1) / math.sqrt(k)) if k > 1 else 0.0
    return CandidateEstimate(candidate, float(draws.mean()), aleatory,
                             epistemic, k, r)


def _p_best(samples: dict[str, np.ndarray], leader: str, runner: str,
            rng: np.random.Generator, n_boot: int = 2000) -> float:
    """Paired bootstrap on the CRN difference.

    Resamples the crossed (r, k) indices over the COMMON draw prefix — the only
    draws over which the pairing is valid, since halving leaves survivors with
    more draws than eliminated candidates.
    """
    a, b = samples[leader], samples[runner]
    k = min(a.shape[0], b.shape[0])
    r = min(a.shape[1], b.shape[1])
    diff = a[:k, :r] - b[:k, :r]
    wins = 0
    for _ in range(n_boot):
        ki = rng.integers(0, k, k)
        ri = rng.integers(0, r, r)
        if diff[np.ix_(ki, ri)].mean() > 0:
            wins += 1
    return wins / n_boot


def allocate(
    candidates: list[str],
    evaluate: Callable[[str, int], np.ndarray],
    *,
    initial_draws: int = 2,
    max_draws: int = 50,
    indifference_zone: float = 2.0,
    deadline: float | None = None,
    seed: int = 0,
) -> AllocationResult:
    """``evaluate(candidate, draw_index) -> (R,)`` dollars for my seat."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[np.ndarray]] = {c: [] for c in candidates}
    alive = list(candidates)
    units = 0
    stopped = "budget"

    def out_of_time() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    def spend(cands: list[str], upto: int) -> None:
        nonlocal units
        for c in cands:
            while len(samples[c]) < upto:
                samples[c].append(evaluate(c, len(samples[c])))
                units += 1
                if out_of_time():
                    return

    target = initial_draws
    spend(alive, target)

    while len(alive) > 1 and not out_of_time():
        ranked = sorted(alive, key=lambda c: -np.mean(samples[c]))
        keep = math.ceil(len(alive) / 2)
        alive = ranked[:keep]
        if len(alive) == 1:
            break
        target = min(target * 2, max_draws)
        spend(alive, target)

        if len(alive) == 2 and not out_of_time():
            est = {c: _summarize(c, np.array(samples[c])) for c in alive}
            hi, lo = sorted(alive, key=lambda c: -est[c].mean)
            gap = est[hi].mean - est[lo].mean
            se = math.sqrt(est[hi].total_se ** 2 + est[lo].total_se ** 2)
            if gap > indifference_zone and gap > 1.645 * se:
                alive = [hi]
                stopped = "separated"
                break
        if target >= max_draws:
            break

    # keep deepening the survivor: extra draws tighten the answer actually given
    while len(alive) == 1 and len(samples[alive[0]]) < max_draws and not out_of_time():
        samples[alive[0]].append(evaluate(alive[0], len(samples[alive[0]])))
        units += 1

    if out_of_time() and stopped == "budget":
        stopped = "deadline"

    estimates = {c: _summarize(c, np.array(v))
                 for c, v in samples.items() if v}

    # THE LEADER IS THE SURVIVOR OF HALVING, not whoever has the highest raw
    # mean. Eliminated arms hold few draws, so their means are noisy; letting
    # one of them win defeats the entire procedure and produces the incoherent
    # signature of a leader with a low p(best). Rank by draws first, mean
    # second, so a candidate the allocator spent budget on wins ties.
    ranked = sorted(estimates.values(), key=lambda e: (-e.draws_used, -e.mean))
    leader = ranked[0].candidate
    # present the rest by mean, which is what a reader expects
    ranked = [estimates[leader]] + sorted(
        (e for e in estimates.values() if e.candidate != leader),
        key=lambda e: -e.mean,
    )

    # A candidate is indistinguishable if the gap is inside the zone OR the
    # paired 90% CI of the difference contains zero. Both, because a large gap
    # with a huge interval is not a decision either.
    indifferent = [leader]
    for est in ranked[1:]:
        gap = estimates[leader].mean - est.mean
        se = math.sqrt(estimates[leader].total_se ** 2 + est.total_se ** 2)
        # abs(): a shallow-sampled arm can sit ABOVE the survivor's mean, and
        # that is precisely a case where they are not separated
        if abs(gap) < indifference_zone or abs(gap) < 1.645 * se:
            indifferent.append(est.candidate)

    arrays = {c: np.array(v) for c, v in samples.items() if v}
    p_best = (_p_best(arrays, leader, ranked[1].candidate, rng)
              if len(ranked) > 1 else 1.0)

    return AllocationResult(
        estimates=estimates, leader=leader, indifference_set=indifferent,
        p_best=p_best, stopped_because=stopped, units_spent=units,
        samples=arrays,
    )
