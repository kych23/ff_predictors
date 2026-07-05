"""Black-box extensions for the ILP lineup optimizer.

Slot contract (PRD F4): QB1 RB2 WR2 TE1 FLEX1 K1 DEF1; FLEX eligible RB/WR/TE;
OUT/IR excluded. Optimality: each strategy maximizes its own quantile column.
"""
import pandas as pd

from src.config import load_config
from src.lineup.optimizer import optimize_lineup


def _full_roster():
    return pd.DataFrame([
        {"player_id": "qb1", "position": "QB", "p10": 15, "p50": 20, "p90": 28},
        {"player_id": "qb2", "position": "QB", "p10": 10, "p50": 14, "p90": 20},
        {"player_id": "rb1", "position": "RB", "p10": 12, "p50": 18, "p90": 25},
        {"player_id": "rb2", "position": "RB", "p10": 10, "p50": 15, "p90": 22},
        {"player_id": "rb3", "position": "RB", "p10": 8, "p50": 12, "p90": 18},
        {"player_id": "rb4", "position": "RB", "p10": 1, "p50": 9, "p90": 30},
        {"player_id": "wr1", "position": "WR", "p10": 11, "p50": 17, "p90": 24},
        {"player_id": "wr2", "position": "WR", "p10": 9, "p50": 14, "p90": 20},
        {"player_id": "wr3", "position": "WR", "p10": 7, "p50": 11, "p90": 16},
        {"player_id": "wr4", "position": "WR", "p10": 8.5, "p50": 10, "p90": 12},
        {"player_id": "te1", "position": "TE", "p10": 8, "p50": 13, "p90": 19},
        {"player_id": "te2", "position": "TE", "p10": 5, "p50": 9, "p90": 14},
        {"player_id": "k1", "position": "K", "p10": 5, "p50": 8, "p90": 12},
        {"player_id": "def1", "position": "DEF", "p10": 4, "p50": 7, "p90": 11},
    ])


def test_starters_and_bench_partition_roster():
    roster = _full_roster()
    result = optimize_lineup(roster, strategy="balanced")
    starter_ids = set(result.starters["player_id"])
    bench_ids = set(result.bench["player_id"])
    assert starter_ids & bench_ids == set()
    assert starter_ids | bench_ids == set(roster["player_id"])


def test_total_starter_count_matches_config():
    cfg = load_config()
    result = optimize_lineup(_full_roster(), cfg=cfg, strategy="balanced")
    # starting slots only — cfg.roster.slots also carries BENCH
    starter_slots = ("QB", "RB", "WR", "TE", "FLEX", "K", "DEF")
    expected = sum(cfg.roster.slots.get(s, 0) for s in starter_slots)
    assert len(result.starters) == expected


def test_each_strategy_maximizes_its_own_quantile():
    """The lineup optimized for column Q must score >= any other feasible
    lineup when both are evaluated on column Q."""
    roster = _full_roster()
    results = {s: optimize_lineup(roster, strategy=s)
               for s in ("safe", "balanced", "upside")}

    def col_total(res, col):
        ids = set(res.starters["player_id"])
        return roster[roster["player_id"].isin(ids)][col].sum()

    for strategy, col in (("safe", "p10"), ("balanced", "p50"), ("upside", "p90")):
        best = col_total(results[strategy], col)
        for other in results.values():
            assert best >= col_total(other, col) - 1e-9


def test_backup_qb_never_starts_two():
    """Only one QB slot and QB is not FLEX-eligible — even a great backup QB
    cannot crack the lineup."""
    roster = _full_roster()
    roster.loc[roster["player_id"] == "qb2", ["p10", "p50", "p90"]] = [40, 50, 60]
    result = optimize_lineup(roster, strategy="balanced")
    qb_starters = result.starters[result.starters["position"] == "QB"]
    assert len(qb_starters) == 1
    assert qb_starters.iloc[0]["player_id"] == "qb2"  # the better QB starts


def test_kicker_never_takes_flex():
    roster = _full_roster()
    extra_k = pd.DataFrame([
        {"player_id": "k2", "position": "K", "p10": 90, "p50": 95, "p90": 99},
    ])
    roster = pd.concat([roster, extra_k], ignore_index=True)
    result = optimize_lineup(roster, strategy="balanced")
    k_starters = result.starters[result.starters["position"] == "K"]
    assert len(k_starters) == 1  # second kicker can't sneak into FLEX


def test_def_never_takes_flex():
    roster = _full_roster()
    extra = pd.DataFrame([
        {"player_id": "def2", "position": "DEF", "p10": 90, "p50": 95, "p90": 99},
    ])
    roster = pd.concat([roster, extra], ignore_index=True)
    result = optimize_lineup(roster, strategy="balanced")
    def_starters = result.starters[result.starters["position"] == "DEF"]
    assert len(def_starters) == 1


def test_flex_only_rb_wr_te():
    result = optimize_lineup(_full_roster(), strategy="balanced")
    flex_ids = result.slot_assignments.get("FLEX", [])
    positions = result.starters.set_index("player_id").loc[flex_ids, "position"]
    assert set(positions) <= {"RB", "WR", "TE"}


def test_exclusion_and_status_combined():
    roster = _full_roster()
    roster.loc[roster["player_id"] == "wr1", "status"] = "OUT"
    result = optimize_lineup(roster, excluded_ids={"rb1"})
    all_ids = set(result.starters["player_id"]) | set(result.bench["player_id"])
    assert "rb1" not in all_ids
    assert "wr1" not in all_ids


def test_ir_players_never_start_even_if_best():
    roster = _full_roster()
    roster.loc[roster["player_id"] == "rb1", "status"] = "IR"
    result = optimize_lineup(roster, strategy="balanced")
    assert "rb1" not in set(result.starters["player_id"])


def test_total_points_nonnegative_and_finite():
    result = optimize_lineup(_full_roster(), strategy="upside")
    assert result.total_points >= 0
    assert result.total_points == result.total_points  # not NaN


def test_deterministic_given_same_input():
    roster = _full_roster()
    a = optimize_lineup(roster, strategy="balanced")
    b = optimize_lineup(roster, strategy="balanced")
    assert set(a.starters["player_id"]) == set(b.starters["player_id"])
    assert a.total_points == b.total_points
