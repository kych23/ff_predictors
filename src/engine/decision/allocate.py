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

from src.core.constants import SEPARATION_Z


@dataclass(frozen=True)
class CandidateEstimate:
    """One candidate's dollar estimate and its two standard-error components.

    ``epistemic_se`` is the spread across OUTER DRAWS, and the name overstates
    what that is. In the live cockpit an outer draw varies the correlation
    matrix *and* the opponent-draft realization together, and the draft
    dominates: pinning it collapses this term by roughly 4x on a
    fully-measured arm. It is honest as "uncertainty from everything held
    constant within a draw and varied between them"; it is not parameter
    uncertainty on its own. `total_se` remains a valid standard error of the
    mean, because the draws do sample the joint distribution E[$] averages
    over. See the OUTER DRAWS note in `app/cockpit/build.py`.
    """

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


def paired_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean and standard error of ``a - b`` over their common CRN prefix.

    **The difference must be estimated from the pairing, not from the two
    marginals.** Draw index d addresses the same parameter draw and the same
    aleatory uniforms for every candidate (§15.1), so `a` and `b` are strongly
    positively correlated and

        Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)

    is far below the ``sqrt(se_a^2 + se_b^2)`` this module used to compute.
    Dropping the covariance term throws away the entire point of running CRN:
    the variance reduction lives in the difference, which is the only quantity
    the ranking decision depends on.

    Measured on a live tier-0 run, leader against runner-up at 50 draws each:
    gap $4.29, unpaired SE 3.940, paired SE 2.222 — **1.77x**. At 90% the
    unpaired test called them indistinguishable and the paired test separated
    them. The cost was paid twice: an indifference set padded with candidates
    the evidence already excludes, and an early stop that never fired, so every
    recommendation burned the whole clock re-measuring a settled question.

    `_p_best` below has always resampled the paired difference, so the module
    already knew the pairing existed; it just was not used where it decided
    anything.

    Only the common prefix is paired — halving leaves survivors deeper than
    eliminated arms, and draws beyond the shared prefix have no counterpart.

    **Known bias, conservative and bounded.** Combining the two components with
    `hypot` is exact when the difference decomposes into a draw effect plus a
    rep effect, and double-counts when a residual term survives: pure i.i.d.
    residual noise contributes ``sigma^2 / (K*R)`` to BOTH components, so the
    combination overstates by up to sqrt(2). Measured against the empirical SE
    of the mean difference over 3,000 regenerated datasets at K=50, R=256:

        pure residual (pairing removes all structure)   1.41x
        shared shock dominant (the CRN regime)          1.02x
        mixed                                           1.02x

    So it is 2% in the regime this engine runs in, and it errs toward calling
    candidates indistinguishable. The same `hypot` combination predates this
    function in `_summarize`, so changing it would move every reported
    `total_se`, not just this test.

    The one direction it DOES understate is not structural and not specific to
    this estimator: any standard error built from a sample standard deviation
    carries the Jensen factor ``c4(K) = E[s]/sigma < 1``. Measured against
    ground truth at R=96, dividing out c4 leaves the ratio at 1.00 in every
    shared-shock regime, so the shortfall is exactly c4 and nothing more —
    0.8% at K=50, 1.1% at K=24, shrinking as the allocator deepens an arm.
    That is far too small to turn a genuine tie into a separation, but it is
    not zero, so this is stated as a bound rather than as "conservative only".
    """
    k = min(a.shape[0], b.shape[0])
    r = min(a.shape[1], b.shape[1])
    diff = a[:k, :r] - b[:k, :r]
    aleatory = (float(diff.mean(axis=0).std(ddof=1) / math.sqrt(r))
                if r > 1 else 0.0)
    epistemic = (float(diff.mean(axis=1).std(ddof=1) / math.sqrt(k))
                 if k > 1 else 0.0)
    return float(diff.mean()), math.hypot(aleatory, epistemic)


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

    # Halving stops at TWO, not one. Cutting to a single arm here decided the
    # leader on whatever the 8-draw rung happened to say, and then deepened
    # only that arm — so a winner whose estimate drifted DOWN under deeper
    # sampling still outranked a runner-up nobody re-measured. Observed live: a
    # leader at $70.20 on 50 draws sitting above a candidate at $71.74 on 8.
    while len(alive) > 2 and not out_of_time():
        ranked = sorted(alive, key=lambda c: -np.mean(samples[c]))
        alive = ranked[:max(2, math.ceil(len(alive) / 2))]
        target = min(target * 2, max_draws)
        spend(alive, target)

        if len(alive) == 2 and not out_of_time():
            est = {c: _summarize(c, np.array(samples[c])) for c in alive}
            hi, lo = sorted(alive, key=lambda c: -est[c].mean)
            gap, se = paired_difference(np.array(samples[hi]),
                                        np.array(samples[lo]))
            if gap > indifference_zone and gap > SEPARATION_Z * se:
                alive = [hi]
                stopped = "separated"
                break
        if target >= max_draws:
            break

    # Deepen EVERY finalist, not just a nominal winner. If the pair separated
    # statistically, `alive` is already one arm and this behaves as before. If
    # it did not, both get the full budget and the comparison that decides the
    # recommendation is finally like-for-like.
    while (not out_of_time()
           and any(len(samples[c]) < max_draws for c in alive)):
        for c in alive:
            if out_of_time():
                break
            if len(samples[c]) < max_draws:
                samples[c].append(evaluate(c, len(samples[c])))
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
    # Present the rest by DEPTH first, then mean. Sorting purely by mean floats
    # arms the allocator eliminated after two draws above the one it spent
    # fifty on — a $47.64 estimate with a 3.42 standard error sitting over a
    # $46.37 estimate with 1.47, which reads as the engine contradicting
    # itself. Depth ordering puts the arms that were actually measured first.
    ranked = [estimates[leader]] + sorted(
        (e for e in estimates.values() if e.candidate != leader),
        key=lambda e: (-e.draws_used, -e.mean),
    )

    arrays = {c: np.array(v) for c, v in samples.items() if v}

    # A candidate is indistinguishable if the gap is inside the zone OR the
    # paired 90% CI of the difference contains zero. Both, because a large gap
    # with a huge interval is not a decision either.
    #
    # Gap AND standard error both come off the COMMON PREFIX. Comparing a
    # full-depth difference against a paired standard error would mix two
    # samples and is not a coherent test; `_p_best` resamples the same prefix
    # for the same reason. The per-candidate means reported in `estimates`
    # still use every draw that arm was given — that is the better estimate of
    # each candidate on its own, just not of the difference between two.
    indifferent = [leader]
    for est in ranked[1:]:
        # abs(): a shallow-sampled arm can sit ABOVE the survivor's mean, and
        # that is precisely a case where they are not separated
        gap, se = paired_difference(arrays[leader], arrays[est.candidate])
        if abs(gap) < indifference_zone or abs(gap) < SEPARATION_Z * se:
            indifferent.append(est.candidate)
    p_best = (_p_best(arrays, leader, ranked[1].candidate, rng)
              if len(ranked) > 1 else 1.0)

    return AllocationResult(
        estimates=estimates, leader=leader, indifference_set=indifferent,
        p_best=p_best, stopped_because=stopped, units_spent=units,
        samples=arrays,
    )
