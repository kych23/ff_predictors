"""Black-box extensions for weekly label computation.

Contract (notes/weekly-pipeline-design.md): score each week individually,
REG-only filter (keep all rows when season_type column absent), carry
position/team, stamp snapshot_id.
"""
import pandas as pd

from src.labels.build_labels import compute_weekly_labels
from src.labels.scoring import score_stat_line


def _raw(rows):
    recs = []
    for r in rows:
        rec = {"player_id": r["player_id"], "season": r["season"], "week": r["week"]}
        rec.update(r.get("stats", {}))
        rec["position"] = r.get("position", "RB")
        rec["team"] = r.get("team", "SF")
        if "season_type" in r:
            rec["season_type"] = r["season_type"]
        else:
            rec["season_type"] = "REG"
        recs.append(rec)
    return pd.DataFrame(recs)


def test_multiple_players_multiple_weeks_row_count():
    rows = []
    for pid in ("00-a", "00-b", "00-c"):
        for wk in (1, 2, 3, 4):
            rows.append({"player_id": pid, "season": 2024, "week": wk,
                         "stats": {"rushing_yards": 50}})
    out = compute_weekly_labels(_raw(rows), "snap")
    assert len(out) == 12
    assert set(out["player_id"]) == {"00-a", "00-b", "00-c"}


def test_zero_stat_line_scores_zero():
    out = compute_weekly_labels(_raw([
        {"player_id": "00-a", "season": 2024, "week": 1, "stats": {}},
    ]), "snap")
    assert out.iloc[0]["fantasy_points"] == 0.0


def test_negative_week_possible():
    out = compute_weekly_labels(_raw([
        {"player_id": "00-a", "season": 2024, "week": 1,
         "stats": {"passing_interceptions": 2, "rushing_fumbles_lost": 1}},
    ]), "snap")
    assert out.iloc[0]["fantasy_points"] == -4.0


def test_each_week_scored_independently():
    stats_by_week = {
        1: {"receptions": 5, "receiving_yards": 50},
        2: {"rushing_yards": 100, "rushing_tds": 1},
        3: {"passing_yards": 250, "passing_tds": 2},
    }
    rows = [{"player_id": "00-a", "season": 2024, "week": wk, "stats": st}
            for wk, st in stats_by_week.items()]
    out = compute_weekly_labels(_raw(rows), "snap").set_index("week")
    for wk, st in stats_by_week.items():
        assert out.loc[wk, "fantasy_points"] == score_stat_line(st)


def test_missing_season_type_column_keeps_all_rows():
    """Design doc: when season_type column is absent, no filter applies."""
    df = pd.DataFrame([
        {"player_id": "00-a", "season": 2024, "week": 1, "rushing_yards": 50,
         "position": "RB", "team": "SF"},
        {"player_id": "00-a", "season": 2024, "week": 2, "rushing_yards": 60,
         "position": "RB", "team": "SF"},
    ])
    out = compute_weekly_labels(df, "snap")
    assert len(out) == 2


def test_only_post_rows_yields_empty():
    out = compute_weekly_labels(_raw([
        {"player_id": "00-a", "season": 2024, "week": 19, "season_type": "POST",
         "stats": {"rushing_yards": 100}},
    ]), "snap")
    assert len(out) == 0


def test_output_schema():
    out = compute_weekly_labels(_raw([
        {"player_id": "00-a", "season": 2024, "week": 1,
         "position": "WR", "team": "KC", "stats": {"receptions": 4}},
    ]), "snap-42")
    row = out.iloc[0]
    assert row["player_id"] == "00-a"
    assert row["season"] == 2024
    assert row["week"] == 1
    assert row["position"] == "WR"
    assert row["team"] == "KC"
    assert row["snapshot_id"] == "snap-42"
