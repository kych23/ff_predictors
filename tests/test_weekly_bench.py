"""Weekly lineup benchmark tests."""
import numpy as np
import pandas as pd

from src.benchmark.weekly_bench import bench_one_week, run_weekly_benchmark


def _make_projections(players, week_p50s=None):
    """Build projections DataFrame from a list of (player_id, position, p50) tuples."""
    rows = []
    for pid, pos, p50 in players:
        rows.append({
            "player_id": pid, "position": pos,
            "p10": p50 * 0.6, "p50": p50, "p90": p50 * 1.5,
        })
    return pd.DataFrame(rows)


def _make_actuals(player_points):
    """Build actuals DataFrame from a list of (player_id, fantasy_points) tuples."""
    return pd.DataFrame([
        {"player_id": pid, "fantasy_points": fp} for pid, fp in player_points
    ])


def test_bench_one_week_perfect_projection():
    """When projections perfectly match actuals, pts left on bench = 0."""
    players = [
        ("qb1", "QB", 20), ("rb1", "RB", 18), ("rb2", "RB", 15),
        ("wr1", "WR", 17), ("wr2", "WR", 14), ("te1", "TE", 13),
        ("rb3", "RB", 12),  # flex candidate
        ("k1", "K", 8), ("def1", "DEF", 7),
        ("rb4", "RB", 5), ("wr3", "WR", 4),  # bench
    ]
    actuals = [(pid, p50) for pid, _, p50 in players]
    proj = _make_projections(players)
    act = _make_actuals(actuals)
    result = bench_one_week(proj, act)
    assert result["pts_left_on_bench"] == 0.0
    assert result["n_correct"] == result["n_starters"]


def test_bench_one_week_wrong_starter():
    """When projection picks wrong starter, pts left on bench > 0."""
    players = [
        ("qb1", "QB", 20), ("rb1", "RB", 18), ("rb2", "RB", 15),
        ("wr1", "WR", 17), ("wr2", "WR", 14), ("te1", "TE", 13),
        ("rb3", "RB", 12),
        ("k1", "K", 8), ("def1", "DEF", 7),
        ("wr3", "WR", 4),  # projected low
    ]
    # wr3 actually outscores wr2 — model missed it
    actuals = [
        ("qb1", 20), ("rb1", 18), ("rb2", 15), ("wr1", 17),
        ("wr2", 2), ("te1", 13), ("rb3", 12),
        ("k1", 8), ("def1", 7), ("wr3", 25),
    ]
    proj = _make_projections(players)
    act = _make_actuals(actuals)
    result = bench_one_week(proj, act)
    assert result["pts_left_on_bench"] > 0
    assert result["optimal_total"] > result["recommended_total"]


def test_bench_one_week_empty():
    proj = pd.DataFrame(columns=["player_id", "position", "p10", "p50", "p90"])
    act = pd.DataFrame(columns=["player_id", "fantasy_points"])
    result = bench_one_week(proj, act)
    assert result["pts_left_on_bench"] == 0
    assert result["n_starters"] == 0


def test_run_weekly_benchmark_multi_week():
    """Run across 2 weeks, verify aggregation."""
    proj_rows = []
    label_rows = []
    players = [
        ("qb1", "QB", 20), ("rb1", "RB", 18), ("rb2", "RB", 15),
        ("wr1", "WR", 17), ("wr2", "WR", 14), ("te1", "TE", 13),
        ("rb3", "RB", 12), ("k1", "K", 8), ("def1", "DEF", 7),
        ("wr3", "WR", 4),
    ]
    for week in [1, 2]:
        for pid, pos, p50 in players:
            proj_rows.append({
                "player_id": pid, "season": 2024, "week": week,
                "position": pos, "p10": p50 * 0.6, "p50": p50, "p90": p50 * 1.5,
            })
            label_rows.append({
                "player_id": pid, "season": 2024, "week": week,
                "fantasy_points": p50,  # perfect projection
            })

    projections = pd.DataFrame(proj_rows)
    labels = pd.DataFrame(label_rows)
    result = run_weekly_benchmark(projections, labels, seasons=[2024])
    assert result.n_weeks == 2
    assert result.mean_pts_left == 0.0
    assert result.optimal_hit_rate == 1.0


def test_run_weekly_benchmark_empty():
    projections = pd.DataFrame(columns=["player_id", "season", "week",
                                        "position", "p10", "p50", "p90"])
    labels = pd.DataFrame(columns=["player_id", "season", "week", "fantasy_points"])
    result = run_weekly_benchmark(projections, labels)
    assert result.n_weeks == 0
    assert np.isnan(result.mean_pts_left)


def test_multi_slot_averages_over_draft_positions():
    """With a large pool, different draft slots get different rosters; the
    week metric is the average over slots, not a single team's number."""
    import itertools
    rng_vals = itertools.count(1)
    players = []
    for pos, count in [("QB", 24), ("RB", 60), ("WR", 60), ("TE", 24),
                       ("K", 12), ("DEF", 12)]:
        for i in range(count):
            players.append((f"{pos.lower()}{i}", pos, 30.0 - (i * 0.4)))
    proj = _make_projections(players)
    act = _make_actuals([(pid, p50) for pid, _, p50 in players])

    per_slot = [bench_one_week(proj, act, team_slots=(s,)) for s in (0, 5, 11)]
    avg = bench_one_week(proj, act, team_slots=(0, 5, 11))
    expected = sum(r["pts_left_on_bench"] for r in per_slot) / 3
    assert abs(avg["pts_left_on_bench"] - expected) < 1e-9
    # early slot drafts a stronger roster than the late slot
    assert per_slot[0]["optimal_total"] >= per_slot[2]["optimal_total"]


def test_hit_rate_below_one_when_wrong():
    """Hit rate < 1 when model picks wrong starters."""
    players = [
        ("qb1", "QB", 20), ("rb1", "RB", 18), ("rb2", "RB", 15),
        ("wr1", "WR", 17), ("wr2", "WR", 14), ("te1", "TE", 13),
        ("rb3", "RB", 12), ("k1", "K", 8), ("def1", "DEF", 7),
        ("wr3", "WR", 4),
    ]
    proj_rows = []
    label_rows = []
    for pid, pos, p50 in players:
        proj_rows.append({
            "player_id": pid, "season": 2024, "week": 1,
            "position": pos, "p10": p50 * 0.6, "p50": p50, "p90": p50 * 1.5,
        })
    # wr3 outscores wr2 in actuals
    actuals = {"qb1": 20, "rb1": 18, "rb2": 15, "wr1": 17,
               "wr2": 2, "te1": 13, "rb3": 12, "k1": 8, "def1": 7, "wr3": 25}
    for pid, fp in actuals.items():
        label_rows.append({
            "player_id": pid, "season": 2024, "week": 1, "fantasy_points": fp,
        })

    projections = pd.DataFrame(proj_rows)
    labels = pd.DataFrame(label_rows)
    result = run_weekly_benchmark(projections, labels, seasons=[2024])
    assert result.optimal_hit_rate < 1.0
    assert result.mean_pts_left > 0
