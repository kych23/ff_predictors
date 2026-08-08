"""The difference between two candidates is estimated from the PAIRING.

Common random numbers make draw index d address the same parameter draw and
the same aleatory uniforms for every candidate, so two candidates' dollar
arrays are strongly positively correlated. The variance reduction CRN buys
lives entirely in the difference:

    Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)

`allocate` used to compute the difference's standard error as
``sqrt(se_a^2 + se_b^2)``, dropping the covariance term and therefore the
entire benefit. Measured on a live tier-0 run at 50 draws per arm: gap $4.29,
unpaired SE 3.940, paired SE 2.222. The unpaired test called the top two
indistinguishable; the paired test separated them at 90%.

It cost twice over — an indifference set padded with candidates the evidence
already excludes, and an early stop that could never fire, so every
recommendation spent its whole clock re-measuring a settled question.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

from src.engine.decision.allocate import allocate, paired_difference


def _correlated(k: int, r: int, *, shift: float, rho: float, seed: int):
    """Two (K, R) arrays sharing `rho` of their noise — a stand-in for CRN."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 1, (k, r))
    a = shared * rho + rng.normal(0, 1, (k, r)) * math.sqrt(1 - rho ** 2)
    b = shared * rho + rng.normal(0, 1, (k, r)) * math.sqrt(1 - rho ** 2)
    return a + shift, b


def _seed_of(candidate: str) -> int:
    """Stable per-candidate seed.

    NOT `hash()`. Python randomizes string hashing per process, so seeding an
    RNG from it makes the fixture different on every run — these tests passed
    or failed depending on PYTHONHASHSEED, which is the worst possible property
    for the gate that guards the allocator.
    """
    return zlib.crc32(candidate.encode())


def _unpaired_se(a: np.ndarray, b: np.ndarray) -> float:
    """What the module used to do, kept here as the thing being improved on."""
    def se(d):
        k, r = d.shape
        al = d.mean(axis=0).std(ddof=1) / math.sqrt(r) if r > 1 else 0.0
        ep = d.mean(axis=1).std(ddof=1) / math.sqrt(k) if k > 1 else 0.0
        return math.hypot(al, ep)
    return math.hypot(se(a), se(b))


# ------------------------------------------------------- the estimator
def test_pairing_beats_the_marginals_when_draws_are_shared():
    a, b = _correlated(50, 200, shift=1.0, rho=0.9, seed=0)
    _, paired = paired_difference(a, b)
    assert paired < _unpaired_se(a, b) / 1.5, (
        "highly correlated arms must yield a much tighter difference")


def test_pairing_is_no_worse_than_the_marginals_when_draws_are_independent():
    """The honest boundary case: with rho=0 there is nothing to gain, and the
    paired estimator must not somehow claim there is."""
    a, b = _correlated(50, 200, shift=1.0, rho=0.0, seed=1)
    _, paired = paired_difference(a, b)
    assert paired == pytest.approx(_unpaired_se(a, b), rel=0.25)


def test_the_reported_gap_is_the_mean_difference():
    a, b = _correlated(20, 50, shift=2.5, rho=0.8, seed=2)
    k = min(a.shape[0], b.shape[0])
    r = min(a.shape[1], b.shape[1])
    gap, _ = paired_difference(a, b)
    assert gap == pytest.approx((a[:k, :r] - b[:k, :r]).mean())


def test_only_the_common_prefix_is_paired():
    """Halving leaves survivors deeper than eliminated arms. Draws past the
    shared prefix have no counterpart to difference against."""
    a, b = _correlated(50, 100, shift=1.0, rho=0.9, seed=3)
    gap_full, se_full = paired_difference(a, b)
    gap_cut, se_cut = paired_difference(a, b[:8])
    assert (gap_full, se_full) != (gap_cut, se_cut)
    assert paired_difference(a[:8], b[:8]) == (gap_cut, se_cut)


def test_a_single_draw_yields_a_finite_standard_error():
    """K=1 leaves no epistemic spread to estimate. It must degrade to the
    aleatory term, not to nan — a nan here silences the separation test."""
    a, b = _correlated(1, 200, shift=1.0, rho=0.9, seed=4)
    gap, se = paired_difference(a, b)
    assert math.isfinite(gap) and math.isfinite(se) and se > 0


def test_a_single_rep_yields_a_finite_standard_error():
    a, b = _correlated(30, 1, shift=1.0, rho=0.9, seed=5)
    gap, se = paired_difference(a, b)
    assert math.isfinite(gap) and math.isfinite(se) and se > 0


def test_identical_arms_have_zero_gap_and_zero_error():
    a, _ = _correlated(20, 50, shift=0.0, rho=0.9, seed=6)
    assert paired_difference(a, a) == (pytest.approx(0.0), pytest.approx(0.0))


# ------------------------------------------------- effect on the decision
def _evaluate_factory(edge: dict[str, float], *, shock: float = 8.0,
                      idiosyncratic: float = 0.5, reps: int = 400):
    """CRN by construction.

    Draw index d fixes a single parameter-level shock that EVERY candidate
    sees — which is what an epistemic draw is in this engine: one sampled
    correlation matrix and hazard, shared across the candidates being compared.
    It moves both arms together, so it dominates each candidate's own standard
    error and cancels almost entirely in the difference. That asymmetry is the
    whole reason CRN is used.
    """
    def evaluate(candidate: str, draw: int) -> np.ndarray:
        shared = float(np.random.default_rng(draw).normal(0, shock))
        own = np.random.default_rng(
            [_seed_of(candidate), draw]).normal(0, idiosyncratic, reps)
        return edge[candidate] + shared + own
    return evaluate


def test_the_indifference_set_excludes_a_candidate_crn_separates():
    """The headline. A $3 edge under a large shared shock is invisible to the
    marginals and obvious to the pairing."""
    edge = {"good": 3.0, "bad": 0.0}
    result = allocate(["good", "bad"],
                      _evaluate_factory(edge),
                      initial_draws=2, max_draws=32, indifference_zone=2.0)

    assert result.leader == "good"
    assert result.indifference_set == ["good"], (
        "the pairing resolves this comparison; the marginals do not")

    a, b = result.samples["good"], result.samples["bad"]
    gap, paired = paired_difference(a, b)
    assert abs(gap) > 1.645 * paired
    assert abs(gap) < 1.645 * _unpaired_se(a, b), (
        "fixture no longer reproduces the disagreement this test exists for")


def test_genuinely_tied_candidates_stay_in_the_indifference_set():
    """The fix must sharpen the test, not simply make it always separate."""
    edge = {"a": 0.0, "b": 0.0}
    result = allocate(["a", "b"], _evaluate_factory(edge),
                      initial_draws=2, max_draws=32, indifference_zone=2.0)
    assert set(result.indifference_set) == {"a", "b"}


def test_a_shallow_arm_above_the_leader_is_not_declared_separated():
    """Two draws is not evidence. The `abs()` in the indifference test exists
    so an under-measured arm sitting ABOVE the survivor still counts as
    unresolved rather than as a decided loss.

    Asserted as a RATE over many seeds rather than a single outcome. At two
    draws the sampled gap is itself noisy, so one seed proves very little —
    and an earlier version of this test seeded from `hash()`, which Python
    randomizes per process, so it silently passed or failed depending on
    PYTHONHASHSEED.
    """
    edge = {"lead": 0.0, "noisy": 0.5}
    kept = 0
    trials = 150
    for salt in range(trials):
        result = allocate(
            ["lead", "noisy"],
            _four_arm_factory(edge, salt=salt, shock=1.0, idiosyncratic=4.0,
                              reps=20),
            initial_draws=2, max_draws=2, indifference_zone=2.0)
        kept += len(result.indifference_set) == 2
    assert kept / trials > 0.85, (
        f"only {kept}/{trials} runs kept both arms; two draws should almost "
        f"never resolve a gap this far inside the indifference zone")


# --------------------------------------------- the test must stay honest
def _four_arm_factory(edge, *, salt, shock=8.0, idiosyncratic=3.0, reps=200):
    def evaluate(candidate: str, draw: int) -> np.ndarray:
        shared = float(np.random.default_rng([salt, draw]).normal(0, shock))
        own = np.random.default_rng(
            [salt, _seed_of(candidate), draw]).normal(
                0, idiosyncratic, reps)
        return edge[candidate] + shared + own
    return evaluate


def _sweep(edge, trials=120):
    separated = wrong = 0
    for salt in range(trials):
        result = allocate(["a", "b", "c", "d"],
                          _four_arm_factory(edge, salt=salt),
                          initial_draws=2, max_draws=32, indifference_zone=2.0)
        if result.stopped_because == "separated":
            separated += 1
        if edge[result.leader] < max(edge.values()):
            wrong += 1
    return separated / trials, wrong / trials


def test_tied_arms_essentially_never_stop_early():
    """Guards the direction the fix could have gone wrong in. A tighter
    standard error makes the early stop fire more readily, so the thing to
    prove is that it does NOT fire on candidates that are actually equal."""
    separated, wrong = _sweep({"a": 0.0, "b": 0.0, "c": -3.0, "d": -4.0})
    assert separated <= 0.05, f"false separation rate {separated:.1%}"
    assert wrong == 0.0


def test_a_real_gap_now_stops_early_and_picks_the_right_arm():
    """The benefit side. Under the unpaired standard error this comparison
    could not resolve and every recommendation ran to the clock."""
    separated, wrong = _sweep({"a": 5.0, "b": 0.0, "c": -3.0, "d": -4.0})
    assert separated >= 0.90, f"only separated {separated:.1%} of the time"
    assert wrong == 0.0


def test_the_hypot_combination_stays_within_its_documented_bounds():
    """The known bias, pinned as a two-sided bound.

    Residual noise contributes sigma^2/(K*R) to BOTH components, so `hypot`
    double-counts it — up to sqrt(2) when pairing removes all structure, 1.02x
    in the regime this engine runs in.

    The other side matters too, and an earlier version of this test asserted
    the estimator could ONLY overstate. It cannot quite: every standard error
    built from a sample standard deviation carries the Jensen factor
    ``c4(K) = E[s]/sigma < 1``. Dividing that out leaves the ratio at 1.00 in
    the shared-shock regimes, so the shortfall is exactly c4 — about 1% here —
    and not a structural understatement. Small enough that it cannot turn a
    tie into a separation; large enough that "conservative only" was an
    overclaim.
    """
    K, R, trials = 30, 128, 400
    rng = np.random.default_rng(19)

    def sample(sig_row, sig_col, sig_resid, seed):
        gen = np.random.default_rng(seed)
        row = gen.normal(0, sig_row, (K, 1))
        col = gen.normal(0, sig_col, (1, R))
        return row + col + gen.normal(0, sig_resid, (K, R))

    for sig_row, sig_col, sig_resid in [(0.0, 0.0, 6.0), (3.0, 1.0, 0.5)]:
        truth = np.std([sample(sig_row, sig_col, sig_resid, s).mean()
                        for s in range(trials)], ddof=1)
        estimated = np.mean([
            paired_difference(sample(sig_row, sig_col, sig_resid, s),
                              np.zeros((K, R)))[1]
            for s in range(60)])
        # c4(K) is the only licensed shortfall; allow a little sampling slack.
        c4 = math.sqrt(2 / (K - 1)) * math.gamma(K / 2) / math.gamma((K - 1) / 2)
        assert estimated >= truth * c4 * 0.93, (
            f"understates by more than the Jensen factor: "
            f"{estimated / truth:.3f} against c4={c4:.3f}")
        assert estimated <= truth * 1.5, (
            f"overstatement {estimated / truth:.2f}x exceeds the sqrt(2) bound")
    assert rng is not None
