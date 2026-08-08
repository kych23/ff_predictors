"""Prior-production decay is measured in SEASONS, not in rows.

`build_prior_production` documents `halflife` as a number of seasons and used
`DataFrame.ewm(halflife=...)` to apply it. Without `times=`, pandas weights the
i-th row back as ``0.5 ** (i / halflife)`` where i counts ROWS — identical to
season distance only when a player appears every year.

18.7% of multi-season players in 2011-2025 have a gap. For them the two
readings diverge, always in the same direction: a stale pre-gap season is
carried forward as though it were recent. That cohort is players returning from
a lost year, which is precisely where a projection wants to be MORE cautious,
not less.

These tests pin both halves: unchanged behavior on contiguous histories (the
81% majority, so the fix cannot be a silent regression), and correct decay
across a gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_league
from src.models.features.prior_production import build_prior_production

#: Minimal scoring map — the function refuses to reach for global config.
OFFENSE = load_league().scoring.offense


def _stats(rows: list[dict]) -> pd.DataFrame:
    """Weekly REG rows shaped like `player_stats`, one game per season row."""
    out = []
    for row in rows:
        out.append({
            "player_id": row["player_id"], "season": row["season"],
            "week": row.get("week", 1), "season_type": "REG",
            "position": "WR", "team": row.get("team", "DET"),
            "receptions": row.get("receptions", 0),
            "targets": row.get("targets", 0),
            "receiving_yards": row.get("receiving_yards", 0.0),
            "receiving_tds": 0, "rushing_yards": 0.0, "rushing_tds": 0,
            "carries": 0, "completions": 0, "attempts": 0,
            "passing_yards": 0.0, "passing_tds": 0,
            "passing_interceptions": 0, "sacks_suffered": 0,
        })
    return pd.DataFrame(out)


def _targets_per_game(frame: pd.DataFrame, player_id: str) -> float:
    row = frame.loc[frame["player_id"] == player_id]
    assert len(row) == 1
    return float(row["prior_targets_per_game"].iloc[0])


# ------------------------------------------------- contiguous: no regression
@pytest.mark.parametrize("halflife", [0.5, 1.0, 2.0])
def test_contiguous_seasons_match_the_pandas_formula(halflife):
    """The 81% majority. Season distance and row distance coincide here, so
    the fix must reproduce `ewm` exactly — otherwise it is a regression
    dressed up as a bug fix."""
    seasons, values = [2021, 2022, 2023], [4.0, 10.0, 7.0]
    stats = _stats([{"player_id": "p", "season": s, "targets": v}
                    for s, v in zip(seasons, values, strict=True)])

    got = _targets_per_game(
        build_prior_production(stats, 2024, halflife=halflife,
                               offense=OFFENSE), "p")
    expected = float(pd.Series(values).ewm(halflife=halflife).mean().iloc[-1])
    assert got == pytest.approx(expected, rel=1e-9)


# -------------------------------------------------------- gapped: the bug
def test_a_gap_decays_by_season_distance_not_row_distance():
    """Seasons 2019, 2020, 2024. Row-distance weights are (0.25, 0.5, 1.0);
    season-distance weights are (0.031, 0.062, 1.0). The stale 2019 season is
    otherwise counted eight times too heavily."""
    stats = _stats([{"player_id": "p", "season": s, "targets": v}
                    for s, v in [(2019, 10.0), (2020, 20.0), (2024, 5.0)]])

    got = _targets_per_game(
        build_prior_production(stats, 2025, halflife=1.0, offense=OFFENSE), "p")

    seasons = np.array([2019, 2020, 2024])
    values = np.array([10.0, 20.0, 5.0])
    weights = 0.5 ** (seasons.max() - seasons)
    assert got == pytest.approx(float((weights * values).sum() / weights.sum()))

    row_distance = float(pd.Series(values).ewm(halflife=1.0).mean().iloc[-1])
    assert abs(got - row_distance) > 1.0, (
        "fixture no longer reproduces the divergence this test exists for")


def test_a_gap_pulls_the_estimate_toward_the_recent_season():
    """Direction matters more than the exact number: after a lost year, the
    pre-gap production must count for LESS, never more."""
    gapped = _stats([{"player_id": "p", "season": s, "targets": v}
                     for s, v in [(2019, 20.0), (2024, 5.0)]])
    contiguous = _stats([{"player_id": "p", "season": s, "targets": v}
                         for s, v in [(2023, 20.0), (2024, 5.0)]])

    after_gap = _targets_per_game(
        build_prior_production(gapped, 2025, halflife=1.0, offense=OFFENSE), "p")
    no_gap = _targets_per_game(
        build_prior_production(contiguous, 2025, halflife=1.0,
                               offense=OFFENSE), "p")
    assert after_gap < no_gap
    assert after_gap == pytest.approx(5.0, abs=0.5), (
        "five seasons of distance should leave almost only the recent year")


def test_a_longer_halflife_still_remembers_across_a_gap():
    """The fix must not become "ignore anything before a gap" — it is a decay,
    and a patient halflife should still carry weight across one."""
    stats = _stats([{"player_id": "p", "season": s, "targets": v}
                    for s, v in [(2020, 20.0), (2024, 4.0)]])
    patient = _targets_per_game(
        build_prior_production(stats, 2025, halflife=4.0, offense=OFFENSE), "p")
    impatient = _targets_per_game(
        build_prior_production(stats, 2025, halflife=0.5, offense=OFFENSE), "p")
    assert patient > impatient
    assert patient > 4.0


# ------------------------------------------------------------- NaN handling
def test_a_missing_column_value_renormalizes_rather_than_poisoning():
    """`ewm().mean()` skips NaN and renormalizes the surviving weights. The
    replacement must too, or a single missing season would drag a rate toward
    zero instead of simply not informing it."""
    stats = _stats([{"player_id": "p", "season": s, "targets": t,
                     "receptions": r}
                    for s, t, r in [(2022, 0, 0), (2023, 10, 5), (2024, 8, 4)]])
    out = build_prior_production(stats, 2025, halflife=1.0, offense=OFFENSE)

    # catch_rate is receptions/targets and is NaN in 2022 (zero targets),
    # so it must be the weighted mean of 2023 and 2024 alone.
    got = float(out.loc[out.player_id == "p", "prior_catch_rate"].iloc[0])
    weights = np.array([0.5, 1.0])
    assert got == pytest.approx(
        float((weights * np.array([0.5, 0.5])).sum() / weights.sum()))


def test_a_column_that_is_never_observed_stays_nan():
    stats = _stats([{"player_id": "p", "season": 2024, "targets": 5}])
    out = build_prior_production(stats, 2025, halflife=1.0, offense=OFFENSE)
    assert pd.isna(out.loc[out.player_id == "p", "prior_yards_per_carry"].iloc[0])


# ------------------------------------------------------------- bookkeeping
def test_history_depth_counts_seasons_played_not_seasons_spanned():
    """`seasons_of_history` drives the calibration bucket, so it has to mean
    "how much do we know", not "how long ago did he start"."""
    stats = _stats([{"player_id": "p", "season": s, "targets": 5}
                    for s in (2019, 2024)])
    out = build_prior_production(stats, 2025, halflife=1.0, offense=OFFENSE)
    assert int(out.loc[out.player_id == "p", "seasons_of_history"].iloc[0]) == 2
    assert int(out.loc[out.player_id == "p", "last_season"].iloc[0]) == 2024


def test_players_are_decayed_independently():
    stats = _stats(
        [{"player_id": "gap", "season": s, "targets": v}
         for s, v in [(2019, 20.0), (2024, 5.0)]]
        + [{"player_id": "steady", "season": s, "targets": v}
           for s, v in [(2023, 20.0), (2024, 5.0)]])
    out = build_prior_production(stats, 2025, halflife=1.0, offense=OFFENSE)
    assert _targets_per_game(out, "gap") < _targets_per_game(out, "steady")
