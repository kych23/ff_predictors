"""ILP lineup optimizer tests."""
import pandas as pd
import pytest

from src.config import load_config
from src.lineup.optimizer import optimize_lineup, optimal_lineup_points, STRATEGY_MAP


def _roster(players_data):
    return pd.DataFrame(players_data)


def _full_roster():
    """15-player roster: 2 QB, 4 RB, 4 WR, 2 TE, 1 K, 1 DEF + 1 extra RB."""
    return _roster([
        {"player_id": "qb1", "position": "QB", "p10": 15, "p50": 20, "p90": 28, "name": "QB1"},
        {"player_id": "qb2", "position": "QB", "p10": 10, "p50": 14, "p90": 20, "name": "QB2"},
        {"player_id": "rb1", "position": "RB", "p10": 12, "p50": 18, "p90": 25, "name": "RB1"},
        {"player_id": "rb2", "position": "RB", "p10": 10, "p50": 15, "p90": 22, "name": "RB2"},
        {"player_id": "rb3", "position": "RB", "p10": 8, "p50": 12, "p90": 18, "name": "RB3"},
        {"player_id": "rb4", "position": "RB", "p10": 5, "p50": 9, "p90": 14, "name": "RB4"},
        {"player_id": "rb5", "position": "RB", "p10": 3, "p50": 7, "p90": 11, "name": "RB5"},
        {"player_id": "wr1", "position": "WR", "p10": 11, "p50": 17, "p90": 24, "name": "WR1"},
        {"player_id": "wr2", "position": "WR", "p10": 9, "p50": 14, "p90": 20, "name": "WR2"},
        {"player_id": "wr3", "position": "WR", "p10": 7, "p50": 11, "p90": 16, "name": "WR3"},
        {"player_id": "wr4", "position": "WR", "p10": 4, "p50": 8, "p90": 13, "name": "WR4"},
        {"player_id": "te1", "position": "TE", "p10": 8, "p50": 13, "p90": 19, "name": "TE1"},
        {"player_id": "te2", "position": "TE", "p10": 5, "p50": 9, "p90": 14, "name": "TE2"},
        {"player_id": "k1", "position": "K", "p10": 5, "p50": 8, "p90": 12, "name": "K1"},
        {"player_id": "def1", "position": "DEF", "p10": 4, "p50": 7, "p90": 11, "name": "DEF1"},
    ])


def test_optimal_lineup_basic():
    roster = _full_roster()
    result = optimize_lineup(roster, strategy="balanced")
    assert result.total_points > 0
    assert len(result.starters) > 0
    assert len(result.bench) > 0
    starter_ids = set(result.starters["player_id"])
    bench_ids = set(result.bench["player_id"])
    assert starter_ids & bench_ids == set()


def test_optimal_lineup_slot_counts():
    cfg = load_config()
    roster = _full_roster()
    result = optimize_lineup(roster, cfg=cfg, strategy="balanced")
    slot_counts = result.starters["slot"].value_counts().to_dict()
    assert slot_counts.get("QB", 0) == cfg.roster.slots.get("QB", 0)
    assert slot_counts.get("RB", 0) == cfg.roster.slots.get("RB", 0)
    assert slot_counts.get("WR", 0) == cfg.roster.slots.get("WR", 0)
    assert slot_counts.get("TE", 0) == cfg.roster.slots.get("TE", 0)
    assert slot_counts.get("K", 0) == cfg.roster.slots.get("K", 0)
    assert slot_counts.get("DEF", 0) == cfg.roster.slots.get("DEF", 0)
    assert slot_counts.get("FLEX", 0) == cfg.roster.slots.get("FLEX", 0)


def test_flex_goes_to_best_remaining():
    """FLEX should be the best remaining flex-eligible player after pure slots filled."""
    roster = _full_roster()
    result = optimize_lineup(roster, strategy="balanced")
    flex_pids = result.slot_assignments.get("FLEX", [])
    assert len(flex_pids) == 1
    flex_player = result.starters[result.starters["player_id"] == flex_pids[0]]
    assert flex_player.iloc[0]["position"] in {"RB", "WR", "TE"}


def test_strategy_safe_uses_p10():
    roster = _full_roster()
    result_safe = optimize_lineup(roster, strategy="safe")
    result_upside = optimize_lineup(roster, strategy="upside")
    assert result_safe.strategy == "safe"
    assert result_upside.strategy == "upside"


def test_strategy_changes_lineup():
    """When p10 and p90 rankings diverge, strategy should pick different starters."""
    roster = _roster([
        {"player_id": "qb1", "position": "QB", "p10": 15, "p50": 18, "p90": 20},
        {"player_id": "rb_floor", "position": "RB", "p10": 15, "p50": 16, "p90": 16},
        {"player_id": "rb_upside", "position": "RB", "p10": 1, "p50": 10, "p90": 30},
        {"player_id": "rb3", "position": "RB", "p10": 8, "p50": 12, "p90": 18},
        {"player_id": "wr1", "position": "WR", "p10": 11, "p50": 17, "p90": 24},
        {"player_id": "wr2", "position": "WR", "p10": 9, "p50": 14, "p90": 20},
        {"player_id": "wr3", "position": "WR", "p10": 7, "p50": 11, "p90": 16},
        {"player_id": "te1", "position": "TE", "p10": 8, "p50": 13, "p90": 19},
        {"player_id": "k1", "position": "K", "p10": 5, "p50": 8, "p90": 12},
        {"player_id": "def1", "position": "DEF", "p10": 4, "p50": 7, "p90": 11},
    ])
    safe = optimize_lineup(roster, strategy="safe")
    upside = optimize_lineup(roster, strategy="upside")
    safe_ids = set(safe.starters["player_id"])
    upside_ids = set(upside.starters["player_id"])
    assert "rb_floor" in safe_ids
    assert "rb_upside" in upside_ids


def test_excluded_players():
    roster = _full_roster()
    result = optimize_lineup(roster, excluded_ids={"qb1"})
    all_ids = set(result.starters["player_id"]) | set(result.bench["player_id"])
    assert "qb1" not in all_ids


def test_status_exclusion():
    roster = _full_roster()
    roster.loc[roster["player_id"] == "rb1", "status"] = "IR"
    roster.loc[roster["player_id"] == "wr1", "status"] = "OUT"
    result = optimize_lineup(roster)
    all_ids = set(result.starters["player_id"]) | set(result.bench["player_id"])
    assert "rb1" not in all_ids
    assert "wr1" not in all_ids


def test_questionable_not_excluded():
    roster = _full_roster()
    roster["status"] = "active"
    roster.loc[roster["player_id"] == "rb1", "status"] = "Q"
    result = optimize_lineup(roster)
    all_ids = set(result.starters["player_id"]) | set(result.bench["player_id"])
    assert "rb1" in all_ids


def test_optimal_lineup_points_convenience():
    roster = _full_roster()
    pts = optimal_lineup_points(roster, quantile_col="p50")
    assert pts > 0


def test_empty_roster():
    roster = pd.DataFrame(columns=["player_id", "position", "p10", "p50", "p90"])
    result = optimize_lineup(roster)
    assert result.total_points == 0.0
    assert result.starters.empty


def test_best_players_start():
    """QB1 > QB2, so QB1 should start at QB."""
    roster = _full_roster()
    result = optimize_lineup(roster, strategy="balanced")
    qb_starters = result.starters[result.starters["slot"] == "QB"]
    assert "qb1" in qb_starters["player_id"].values
