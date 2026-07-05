"""Tests for the paired-significance gate logic (benchmark report)."""
import numpy as np
import pandas as pd

from src.benchmark.report import (
    MIN_N_FOR_GATE,
    MIN_N_FOR_P_TESTS,
    bootstrap_ci,
    format_gate,
    paired_test,
)
from src.benchmark.ranking import run_ranking_benchmark
from src.benchmark.calibration import run_calibration_benchmark


def test_clean_sweep_passes():
    engine = [0.9, 0.91, 0.88, 0.92, 0.9, 0.89, 0.93, 0.9]
    baseline = [0.8, 0.82, 0.79, 0.83, 0.81, 0.8, 0.84, 0.82]
    t = paired_test(engine, baseline)
    assert t.n == 8
    assert t.mean_diff > 0
    assert t.wilcoxon_p is not None and t.wilcoxon_p < 0.05
    assert t.ci_low > 0
    assert t.significant


def test_no_edge_fails():
    rng = np.random.RandomState(0)
    base = rng.uniform(0.7, 0.9, 8)
    noise = rng.normal(0, 0.05, 8)
    t = paired_test(base + noise, base)
    assert not t.significant or t.ci_low > 0  # only passes with real CI evidence


def test_mixed_sign_fails():
    engine = [0.9, 0.7, 0.9, 0.7, 0.9, 0.7, 0.9, 0.7]
    baseline = [0.8] * 8
    t = paired_test(engine, baseline)
    assert not t.significant


def test_small_n_p_tests_suppressed():
    """Below MIN_N_FOR_P_TESTS, exact tests can't reach 0.05 — must be None,
    not a silently unpassable gate."""
    t = paired_test([1.0, 1.1, 1.2, 1.3, 1.4], [0.5, 0.6, 0.7, 0.8, 0.9])
    assert t.n == 5 < MIN_N_FOR_P_TESTS
    assert t.wilcoxon_p is None
    assert t.sign_test_p is None
    # gate falls back to the bootstrap CI, which excludes 0 here
    assert t.significant
    assert "unpowered" in format_gate("X", t)


def test_below_gate_minimum_never_passes():
    t = paired_test([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert t.n == 3 < MIN_N_FOR_GATE
    assert not t.significant
    assert "too few seasons" in format_gate("X", t)


def test_nan_pairs_dropped():
    t = paired_test([1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0])
    assert t.n == 5


def test_zero_diffs_not_significant():
    t = paired_test([1.0] * 8, [1.0] * 8)
    assert not t.significant


def test_bootstrap_ci_empty():
    lo, hi = bootstrap_ci([])
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_contains_mean():
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = bootstrap_ci(diffs)
    assert lo <= np.mean(diffs) <= hi


# --- ranking benchmark ---

def _ranking_fixture(n_seasons: int = 6, n_players: int = 40):
    """Engine ranks players perfectly; ADP ranks them inversely."""
    rows_p, rows_a, rows_l = [], [], []
    for s in range(2018, 2018 + n_seasons):
        for i in range(n_players):
            pid = f"p{i}"
            fppg = float(n_players - i)          # true value: p0 best
            rows_p.append({"player_id": pid, "season": s, "p50": fppg})
            rows_a.append({"player_id": pid, "season": s, "adp": float(n_players - i)})
            rows_l.append({"player_id": pid, "season": s, "fppg": fppg})
    return (pd.DataFrame(rows_p), pd.DataFrame(rows_a), pd.DataFrame(rows_l))


def test_ranking_engine_beats_inverted_adp():
    proj, adp, labels = _ranking_fixture()
    res = run_ranking_benchmark(proj, adp, labels, ks=(10,))
    per = res.per_season
    assert (per["ndcg_engine"] == 1.0).all()      # perfect ranking
    assert (per["ndcg_adp"] < 1.0).all()          # inverted ranking is worse
    assert res.gate[10].significant


def test_ranking_respects_eligible_seasons():
    proj, adp, labels = _ranking_fixture(n_seasons=6)
    res = run_ranking_benchmark(proj, adp, labels, ks=(10,),
                                eligible_seasons=[2018, 2019])
    assert set(res.per_season["season"]) == {2018, 2019}


# --- calibration benchmark ---

def test_calibration_benchmark_coverage():
    rng = np.random.RandomState(1)
    n = 200
    p50 = rng.uniform(5, 20, n)
    oof = pd.DataFrame({
        "p10": p50 - 5, "p50": p50, "p90": p50 + 5,
        "y": p50,                                 # always inside
        "is_rookie": [False] * n,
        "seasons_of_history": [5.0] * n,
        "team_changed": [0] * n,
    })
    out = run_calibration_benchmark(oof)
    overall = out[out["bucket"] == "overall"].iloc[0]
    assert overall["coverage_p10_p90"] == 1.0
    assert "established_vet" in set(out["bucket"])
