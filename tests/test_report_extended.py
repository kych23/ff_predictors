"""Black-box extensions for the benchmark significance gate + benchmarks."""
import numpy as np
import pandas as pd

from src.benchmark.calibration import run_calibration_benchmark
from src.benchmark.ranking import run_ranking_benchmark
from src.benchmark.report import bootstrap_ci, paired_test


def test_engine_worse_never_significant():
    engine = [0.7, 0.72, 0.69, 0.71, 0.7, 0.68, 0.73, 0.7]
    baseline = [0.8, 0.82, 0.79, 0.83, 0.81, 0.8, 0.84, 0.82]
    t = paired_test(engine, baseline)
    assert t.mean_diff < 0
    assert not t.significant


def test_mean_diff_is_engine_minus_baseline():
    t = paired_test([1.0] * 8, [0.4] * 8)
    assert abs(t.mean_diff - 0.6) < 1e-9


def test_bootstrap_ci_ordering():
    rng = np.random.RandomState(2)
    diffs = list(rng.normal(1.0, 0.5, 30))
    lo, hi = bootstrap_ci(diffs)
    assert lo <= hi


def test_bootstrap_ci_constant_diffs_degenerate():
    lo, hi = bootstrap_ci([2.0] * 10)
    assert lo == hi == 2.0


def test_paired_test_symmetry_of_sign():
    """Swapping engine and baseline flips the mean diff's sign."""
    a = [0.9, 0.91, 0.88, 0.92, 0.9, 0.89, 0.93, 0.9]
    b = [0.8, 0.82, 0.79, 0.83, 0.81, 0.8, 0.84, 0.82]
    t_fwd = paired_test(a, b)
    t_rev = paired_test(b, a)
    assert abs(t_fwd.mean_diff + t_rev.mean_diff) < 1e-12
    assert not t_rev.significant


def test_all_nan_pairs_not_significant():
    t = paired_test([np.nan] * 6, [np.nan] * 6)
    assert t.n == 0
    assert not t.significant


# --- ranking benchmark ---

def _fixture(engine_perfect=True, n_seasons=6, n_players=30):
    rows_p, rows_a, rows_l = [], [], []
    for s in range(2018, 2018 + n_seasons):
        for i in range(n_players):
            pid = f"p{i}"
            fppg = float(n_players - i)
            p50 = fppg if engine_perfect else float(i)
            rows_p.append({"player_id": pid, "season": s, "p50": p50})
            # ADP ranks identically to true value (best player = ADP 1)
            rows_a.append({"player_id": pid, "season": s, "adp": float(i + 1)})
            rows_l.append({"player_id": pid, "season": s, "fppg": fppg})
    return (pd.DataFrame(rows_p), pd.DataFrame(rows_a), pd.DataFrame(rows_l))


def test_engine_equal_to_adp_not_significant():
    """Engine and ADP produce identical rankings -> zero diffs -> gate fails."""
    proj, adp, labels = _fixture(engine_perfect=True)
    res = run_ranking_benchmark(proj, adp, labels, ks=(10,))
    per = res.per_season
    assert np.allclose(per["ndcg_engine"], per["ndcg_adp"])
    assert not res.gate[10].significant


def test_engine_worse_than_adp_not_significant():
    proj, adp, labels = _fixture(engine_perfect=False)
    res = run_ranking_benchmark(proj, adp, labels, ks=(10,))
    per = res.per_season
    assert (per["ndcg_engine"] <= per["ndcg_adp"]).all()
    assert not res.gate[10].significant


def test_per_season_rows_one_per_season():
    proj, adp, labels = _fixture(n_seasons=5)
    res = run_ranking_benchmark(proj, adp, labels, ks=(10,))
    assert len(res.per_season) == 5
    assert set(res.per_season["season"]) == set(range(2018, 2023))


# --- calibration benchmark ---

def test_calibration_zero_coverage():
    rng = np.random.RandomState(4)
    n = 150
    p50 = rng.uniform(5, 20, n)
    oof = pd.DataFrame({
        "p10": p50 - 2, "p50": p50, "p90": p50 + 2,
        "y": p50 + 100.0,  # always far outside the interval
        "is_rookie": [False] * n,
        "seasons_of_history": [5.0] * n,
        "team_changed": [0] * n,
    })
    out = run_calibration_benchmark(oof)
    overall = out[out["bucket"] == "overall"].iloc[0]
    assert overall["coverage_p10_p90"] == 0.0


def test_calibration_coverage_bounds():
    rng = np.random.RandomState(6)
    n = 300
    p50 = rng.uniform(5, 20, n)
    oof = pd.DataFrame({
        "p10": p50 - 3, "p50": p50, "p90": p50 + 3,
        "y": p50 + rng.normal(0, 4, n),
        "is_rookie": rng.choice([True, False], n),
        "seasons_of_history": rng.choice([0.0, 1.0, 5.0], n),
        "team_changed": rng.choice([0, 1], n),
    })
    out = run_calibration_benchmark(oof)
    assert (out["coverage_p10_p90"] >= 0.0).all()
    assert (out["coverage_p10_p90"] <= 1.0).all()
    assert "overall" in set(out["bucket"])
