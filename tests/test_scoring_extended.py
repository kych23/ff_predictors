"""Black-box scoring contract extensions.

Grounded in notes/overall-plan.md scoring table:
  passing_yards 1/25, passing_td 4, interception -1, rushing_yards 1/10,
  rushing_td 6, reception 1, receiving_yards 1/10, receiving_td 6,
  two_pt 2, fumble_lost -2.
Uses exact yardage multiples so per-N rounding conventions cannot matter.
"""
import pandas as pd

from src.config import load_config
from src.labels.scoring import score_dataframe, score_stat_line

SC = load_config().scoring


def test_empty_stat_line_scores_zero():
    assert score_stat_line({}) == 0.0


def test_all_zero_stats_score_zero():
    line = {
        "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
        "rushing_yards": 0, "rushing_tds": 0,
        "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
    }
    assert score_stat_line(line) == 0.0


def test_passing_yards_rate():
    # 1 point per 25 passing yards
    assert score_stat_line({"passing_yards": 25}) == 1.0
    assert score_stat_line({"passing_yards": 250}) == 10.0


def test_rushing_yards_rate():
    # 1 point per 10 rushing yards
    assert score_stat_line({"rushing_yards": 10}) == 1.0
    assert score_stat_line({"rushing_yards": 130}) == 13.0


def test_receiving_yards_rate():
    assert score_stat_line({"receiving_yards": 10}) == 1.0
    assert score_stat_line({"receiving_yards": 90}) == 9.0


def test_reception_ppr_value():
    assert score_stat_line({"receptions": 1}) == 1.0
    assert score_stat_line({"receptions": 7}) == 7.0


def test_passing_td_value():
    assert score_stat_line({"passing_tds": 1}) == 4.0
    assert score_stat_line({"passing_tds": 3}) == 12.0


def test_rushing_td_value():
    assert score_stat_line({"rushing_tds": 2}) == 12.0


def test_receiving_td_value():
    assert score_stat_line({"receiving_tds": 2}) == 12.0


def test_negative_total_possible():
    # a pick-six machine with no yards can go negative
    assert score_stat_line({"passing_interceptions": 3}) == -3.0
    assert score_stat_line({"rushing_fumbles_lost": 2}) == -4.0


def test_additivity_of_components():
    a = {"passing_yards": 200, "passing_tds": 1}
    b = {"rushing_yards": 50, "rushing_tds": 1}
    combined = {**a, **b}
    assert score_stat_line(combined) == score_stat_line(a) + score_stat_line(b)


def test_scaling_linearity():
    line = {"receptions": 4, "receiving_yards": 40, "receiving_tds": 1}
    doubled = {k: v * 2 for k, v in line.items()}
    assert score_stat_line(doubled) == 2 * score_stat_line(line)


def test_kitchen_sink_stat_line():
    # every offensive component at once, from the documented scoring table
    line = {
        "passing_yards": 250,          # 10
        "passing_tds": 2,              # 8
        "passing_interceptions": 2,    # -2
        "rushing_yards": 40,           # 4
        "rushing_tds": 1,              # 6
        "receptions": 3,               # 3
        "receiving_yards": 30,         # 3
        "receiving_tds": 1,            # 6
        "rushing_fumbles_lost": 1,     # -2
        "passing_2pt_conversions": 1,  # 2
    }
    assert score_stat_line(line) == 38.0


def test_score_dataframe_length_and_order():
    df = pd.DataFrame([
        {"receptions": 1},
        {"receptions": 2},
        {"receptions": 3},
    ])
    out = score_dataframe(df)
    assert len(out) == 3
    assert list(out) == [1.0, 2.0, 3.0]


def test_score_dataframe_disjoint_columns_nan_as_missing():
    """Rows contributing different stat columns leave NaN cells; NaN must be
    treated as absent (0), matching row-wise dict scoring."""
    df = pd.DataFrame([
        {"passing_yards": 250},
        {"receptions": 5},
        {"rushing_tds": 1},
    ])
    vec = list(score_dataframe(df))
    assert vec == [10.0, 5.0, 6.0]


def test_score_dataframe_single_negative_column():
    df = pd.DataFrame({"passing_interceptions": [0, 1, 4]})
    assert list(score_dataframe(df)) == [0.0, -1.0, -4.0]


def test_config_scoring_matches_documented_league():
    """league.yaml drives the scoring; the documented league values must hold."""
    assert SC.two_pt == 2
    assert SC.fumble_lost == -2
