"""Scoring contract tests (DrafterSpec.md §4.4.1 acceptance).

Protects the custom-scoring edge: asserts the scoring function reproduces known
totals AND each component — especially that fumbles and 2-pt are summed across ALL
THREE of their columns (a missing column would silently under-count).
"""
import pandas as pd

from src.config import load_config
from src.labels.scoring import score_dataframe, score_stat_line

SC = load_config().scoring


def test_full_stat_line_total():
    line = {
        "passing_yards": 300, "passing_tds": 3, "passing_interceptions": 1,
        "rushing_yards": 20, "rushing_tds": 1,
        "receptions": 5, "receiving_yards": 60, "receiving_tds": 1,
    }
    # 12 + 12 - 1 + 2 + 6 + 5 + 6 + 6 = 48
    assert score_stat_line(line) == 48.0


def test_two_point_summed_across_three_columns():
    base = {"passing_2pt_conversions": 1, "rushing_2pt_conversions": 1,
            "receiving_2pt_conversions": 1}
    # 3 conversions * 2 pts each = 6
    assert score_stat_line(base) == 6.0
    # each column individually contributes 2
    for col in base:
        assert score_stat_line({col: 1}) == SC.two_pt


def test_fumbles_summed_across_three_columns():
    base = {"rushing_fumbles_lost": 1, "receiving_fumbles_lost": 1, "sack_fumbles_lost": 1}
    # 3 fumbles * -2 = -6
    assert score_stat_line(base) == -6.0
    for col in base:
        assert score_stat_line({col: 1}) == SC.fumble_lost


def test_return_td_uses_special_teams_tds():
    assert score_stat_line({"special_teams_tds": 2}) == 2 * SC.return_td


def test_interception_value_is_league_specific():
    # League uses int = -1 (NOT the nflverse default -2).
    assert score_stat_line({"passing_interceptions": 1}) == -1.0


def test_vectorized_matches_rowwise():
    df = pd.DataFrame([
        {"receptions": 5, "receiving_yards": 50, "receiving_tds": 1},
        {"rushing_yards": 100, "rushing_tds": 2},
        {"passing_yards": 250, "passing_tds": 2, "passing_interceptions": 2},
    ])
    vec = score_dataframe(df)
    row = [score_stat_line(r) for r in df.to_dict("records")]
    assert list(vec) == row


def test_empty_frame():
    assert score_dataframe(pd.DataFrame()).empty
