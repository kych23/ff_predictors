"""Black-box extensions for the weekly lineup benchmark."""
import pandas as pd

from src.benchmark.weekly_bench import bench_one_week, run_weekly_benchmark

PLAYERS = [
    ("qb1", "QB", 20), ("rb1", "RB", 18), ("rb2", "RB", 15),
    ("wr1", "WR", 17), ("wr2", "WR", 14), ("te1", "TE", 13),
    ("rb3", "RB", 12), ("k1", "K", 8), ("def1", "DEF", 7),
    ("wr3", "WR", 4),
]


def _proj(players):
    return pd.DataFrame([
        {"player_id": pid, "position": pos,
         "p10": p50 * 0.6, "p50": p50, "p90": p50 * 1.5}
        for pid, pos, p50 in players
    ])


def _act(points):
    return pd.DataFrame([
        {"player_id": pid, "fantasy_points": fp} for pid, fp in points
    ])


def test_recommended_never_exceeds_optimal():
    proj = _proj(PLAYERS)
    act = _act([("qb1", 5), ("rb1", 30), ("rb2", 1), ("wr1", 2), ("wr2", 25),
                ("te1", 0), ("rb3", 19), ("k1", 3), ("def1", 12), ("wr3", 22)])
    result = bench_one_week(proj, act)
    assert result["recommended_total"] <= result["optimal_total"]
    assert result["pts_left_on_bench"] >= 0


def test_n_correct_bounded_by_starters():
    proj = _proj(PLAYERS)
    act = _act([("qb1", 5), ("rb1", 30), ("rb2", 1), ("wr1", 2), ("wr2", 25),
                ("te1", 0), ("rb3", 19), ("k1", 3), ("def1", 12), ("wr3", 22)])
    result = bench_one_week(proj, act)
    assert 0 <= result["n_correct"] <= result["n_starters"]


def test_inverted_projections_leave_points():
    """Projections perfectly anti-correlated with reality within a position
    must leave points on the bench."""
    proj = _proj(PLAYERS)
    # invert RB actuals: rb3 (proj 12) scores 25, rb1 (proj 18) scores 2
    act = _act([("qb1", 20), ("rb1", 2), ("rb2", 15), ("wr1", 17), ("wr2", 14),
                ("te1", 13), ("rb3", 25), ("k1", 8), ("def1", 7), ("wr3", 4)])
    result = bench_one_week(proj, act)
    assert result["pts_left_on_bench"] > 0


def test_single_week_benchmark_counts_one_week():
    proj_rows, label_rows = [], []
    for pid, pos, p50 in PLAYERS:
        proj_rows.append({"player_id": pid, "season": 2024, "week": 7,
                          "position": pos, "p10": p50 * 0.6, "p50": p50,
                          "p90": p50 * 1.5})
        label_rows.append({"player_id": pid, "season": 2024, "week": 7,
                           "fantasy_points": p50})
    result = run_weekly_benchmark(pd.DataFrame(proj_rows),
                                  pd.DataFrame(label_rows), seasons=[2024])
    assert result.n_weeks == 1
    assert result.mean_pts_left == 0.0


def test_seasons_filter_restricts_weeks():
    proj_rows, label_rows = [], []
    for season in (2023, 2024):
        for week in (1, 2):
            for pid, pos, p50 in PLAYERS:
                proj_rows.append({"player_id": pid, "season": season,
                                  "week": week, "position": pos,
                                  "p10": p50 * 0.6, "p50": p50,
                                  "p90": p50 * 1.5})
                label_rows.append({"player_id": pid, "season": season,
                                   "week": week, "fantasy_points": p50})
    result = run_weekly_benchmark(pd.DataFrame(proj_rows),
                                  pd.DataFrame(label_rows), seasons=[2024])
    assert result.n_weeks == 2  # only 2024's two weeks


def test_hit_rate_bounds():
    proj_rows, label_rows = [], []
    for pid, pos, p50 in PLAYERS:
        proj_rows.append({"player_id": pid, "season": 2024, "week": 1,
                          "position": pos, "p10": p50 * 0.6, "p50": p50,
                          "p90": p50 * 1.5})
        # scramble actuals
        label_rows.append({"player_id": pid, "season": 2024, "week": 1,
                           "fantasy_points": 30.0 - p50})
    result = run_weekly_benchmark(pd.DataFrame(proj_rows),
                                  pd.DataFrame(label_rows), seasons=[2024])
    assert 0.0 <= result.optimal_hit_rate <= 1.0
    assert result.mean_pts_left >= 0.0
