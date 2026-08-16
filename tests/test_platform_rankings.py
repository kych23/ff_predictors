"""The market the engine models is the board the LEAGUE drafts on.

Opponents pick from one platform's board. FFC is a composite of mock drafts
across the industry — a good consensus and the wrong board. Compared
rank-to-rank on the 2026 export: Spearman 0.92 against FFC, mean 6.6 ranks
apart over the first five rounds, 12.3 over ten. Close, and not the same:
Tucker Kraft is FFC 113 and Yahoo 71, so an engine reading FFC expects him to
last 42 picks longer than he will.

**Ranks are not draft positions.** The export gives Gibbs `1`; FFC gives `1.6`,
an average pick with real decimals. Ranks are dense and evenly spaced, picks
are not, and the gaps are what a survival curve is made of. Order comes from
the platform, spacing from the market.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.platform.sources import rankings_csv as rc


def _configured_export() -> Path:
    """The export the ENGINE reads, not a filename copied into this file.

    This constant used to be a literal. The literal was the same fact written
    twice, and it failed in the one direction that matters: re-exporting the
    board and repointing `strategy.yaml` left this pointing at the previous
    file, so `skipif(not EXPORT.exists())` turned both real-export tests below
    into silent skips — losing the rank-vs-pick guard exactly when the input it
    guards had just changed.
    """
    from src.core.config import load_league, load_strategy

    return Path(load_strategy(load_league()).adp.rankings_path)


EXPORT = _configured_export()


def _export(tmp_path, rows=None) -> Path:
    rows = rows or [
        {"Rank": i, "Player": f"Player {i}", "POS": f"WR{i}", "Team": "DET",
         "AVG": float(i), "Expert": i, "Sleeper": i, "ESPN": i,
         "Yahoo": i, "Underdog": i, "CBS": i, "FFPC": i}
        for i in range(1, 21)]
    path = tmp_path / "ranks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------- loading
def test_the_expert_column_is_never_loaded():
    """It is analyst RANKINGS — somebody's projections. The ADP wall lets the
    market price availability and never value; reading it would make every
    'beats the market' claim circular."""
    assert "Expert" not in rc.PLATFORMS


def test_platform_columns_become_rank_fields(tmp_path):
    frame = rc.load(_export(tmp_path)).frame
    for platform in ("yahoo", "espn", "cbs", "sleeper", "underdog", "ffpc"):
        assert f"rank_{platform}" in frame.columns
    assert "rank_expert" not in frame.columns


def test_a_missing_platform_column_is_rejected(tmp_path):
    rows = [{"Rank": 1, "Player": "A", "POS": "WR1", "Team": "DET",
             "AVG": 1.0, "Expert": 1}]
    path = tmp_path / "thin.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="platform"):
        rc.load(path)


def test_a_missing_required_column_is_rejected(tmp_path):
    rows = [{"Rank": 1, "Player": "A", "Yahoo": 1}]
    path = tmp_path / "broken.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        rc.load(path)


def test_dispersion_measures_disagreement_between_platforms(tmp_path):
    rows = [{"Rank": 1, "Player": "Agreed", "POS": "WR1", "Team": "DET",
             "AVG": 10.0, "Expert": 10, "Sleeper": 10, "ESPN": 10,
             "Yahoo": 10, "Underdog": 10, "CBS": 10, "FFPC": 10},
            {"Rank": 2, "Player": "Contested", "POS": "WR2", "Team": "LAR",
             "AVG": 30.0, "Expert": 30, "Sleeper": 10, "ESPN": 50,
             "Yahoo": 20, "Underdog": 45, "CBS": 15, "FFPC": 40}]
    path = tmp_path / "d.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame = rc.load(path).frame.set_index("player_name")
    assert frame.loc["Agreed", "rank_spread"] == 0
    assert frame.loc["Contested", "rank_spread"] > 30


# ------------------------------------------------- ranks -> pick positions
def test_ranks_are_mapped_onto_the_observed_pick_scale():
    """The output must live in PICK units, or `survival_probability` — which
    solves a lognormal on pick number — silently changes meaning."""
    rank = pd.Series([1.0, 2.0, 3.0, 4.0])
    scale = pd.Series([1.6, 12.4, 40.0, 95.0])
    picks = rc.to_pick_positions(rank, scale)
    assert picks.min() >= scale.min() and picks.max() <= scale.max()
    assert picks.is_monotonic_increasing


def test_order_comes_from_the_platform_not_the_scale():
    """Reversing the ranking must reverse the prices."""
    scale = pd.Series([1.0, 10.0, 50.0, 120.0])
    ascending = rc.to_pick_positions(pd.Series([1.0, 2, 3, 4]), scale)
    descending = rc.to_pick_positions(pd.Series([4.0, 3, 2, 1]), scale)
    assert list(descending) == list(ascending)[::-1]


def test_spacing_comes_from_the_market_not_the_ranking():
    """Ranks are evenly spaced and picks are not. If the gaps came from the
    ranking, the top of the board would spread out and the middle compress —
    exactly the tail a survival curve reads."""
    picks = rc.to_pick_positions(pd.Series([1.0, 2, 3, 4]),
                                 pd.Series([1.0, 2.0, 3.0, 100.0]))
    gaps = np.diff(sorted(picks.dropna()))
    assert gaps[-1] > 10 * gaps[0], "market spacing was flattened"


def test_an_unranked_player_keeps_no_price():
    """Absent is not last. `survival_probability` already reads a missing ADP
    as 'the market has no opinion', which is the honest encoding."""
    picks = rc.to_pick_positions(pd.Series([1.0, np.nan, 3.0]),
                                 pd.Series([2.0, 20.0, 60.0]))
    assert pd.isna(picks.iloc[1])
    assert picks.notna().sum() == 2


def test_an_empty_scale_yields_no_prices():
    picks = rc.to_pick_positions(pd.Series([1.0, 2.0]), pd.Series(dtype=float))
    assert picks.isna().all()


# --------------------------------------------------------- the real export
@pytest.mark.skipif(not EXPORT.exists(), reason="no rankings export on disk")
def test_the_real_export_covers_most_of_the_board():
    from src.core.names import normalize_name

    frame = rc.load(EXPORT).frame
    board = pd.read_parquet("data/bundles/draft_night_bundle.parquet")
    board["match_key"] = board["player_name"].map(normalize_name)
    merged = board.merge(frame[["match_key", "rank_yahoo"]], on="match_key",
                         how="left")
    assert merged["rank_yahoo"].notna().mean() > 0.80


@pytest.mark.skipif(not EXPORT.exists(), reason="no rankings export on disk")
def test_the_shipped_board_is_priced_in_pick_units():
    """A board accidentally priced in ranks would look plausible — dense
    integers from 1 — and quietly wreck every wait term."""
    board = pd.read_parquet("data/bundles/draft_night_bundle.parquet")
    adp = board["adp"].dropna()
    assert adp.min() < 3.0
    assert adp.max() > 100, "top-heavy range suggests ranks, not picks"
    # Real ADP is not dense integers.
    assert (adp % 1 != 0).mean() > 0.5
