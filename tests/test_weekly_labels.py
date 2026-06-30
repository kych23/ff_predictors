"""Weekly label scoring correctness tests."""
import pandas as pd

from src.labels.build_labels import compute_weekly_labels
from src.labels.scoring import score_stat_line


def _make_weekly_raw(rows):
    recs = []
    for r in rows:
        rec = {"player_id": r["player_id"], "season": r["season"], "week": r["week"]}
        rec.update(r.get("stats", {}))
        rec["position"] = r.get("position", "RB")
        rec["team"] = r.get("team", "SF")
        rec["season_type"] = r.get("season_type", "REG")
        recs.append(rec)
    return pd.DataFrame(recs)


def test_weekly_scoring_matches_stat_line():
    stats = {"receptions": 5, "receiving_yards": 80, "receiving_tds": 1}
    raw = _make_weekly_raw([
        {"player_id": "00-aaa", "season": 2024, "week": 1, "stats": stats},
    ])
    result = compute_weekly_labels(raw, "test-snap")
    expected = score_stat_line(stats)
    assert len(result) == 1
    assert result.iloc[0]["fantasy_points"] == expected


def test_postseason_excluded():
    raw = _make_weekly_raw([
        {"player_id": "00-aaa", "season": 2024, "week": 1, "season_type": "REG",
         "stats": {"rushing_yards": 100}},
        {"player_id": "00-aaa", "season": 2024, "week": 18, "season_type": "POST",
         "stats": {"rushing_yards": 150}},
    ])
    result = compute_weekly_labels(raw, "test-snap")
    assert len(result) == 1
    assert result.iloc[0]["week"] == 1


def test_position_team_carried():
    raw = _make_weekly_raw([
        {"player_id": "00-aaa", "season": 2024, "week": 1,
         "position": "WR", "team": "BUF", "stats": {"receptions": 3}},
    ])
    result = compute_weekly_labels(raw, "test-snap")
    assert result.iloc[0]["position"] == "WR"
    assert result.iloc[0]["team"] == "BUF"


def test_snapshot_id_set():
    raw = _make_weekly_raw([
        {"player_id": "00-aaa", "season": 2024, "week": 1,
         "stats": {"rushing_yards": 50}},
    ])
    result = compute_weekly_labels(raw, "my-snapshot-123")
    assert (result["snapshot_id"] == "my-snapshot-123").all()
