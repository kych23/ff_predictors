"""Reliability study (§21.4).

Two things here are easy to get wrong and both silently inflate the reported
stability, which is the direction that matters: resampling whole seasons
instead of players within them, and scoring agreement on the intersection of
two top-k sets instead of their union.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.audit.reliability import (
    _top_k_agreement,
    reliability,
    resample_players_within_season,
)


def _matrix(seasons=(2019, 2020, 2021, 2022, 2023), per_season=40):
    rows = []
    for s in seasons:
        for p in range(per_season):
            rows.append({"season": s, "player_id": f"p{p:03d}", "x": p * 1.0})
    return pd.DataFrame(rows)


# ------------------------------------------------------ resampling unit
def test_every_season_survives_resampling():
    """A season-block bootstrap breaks make_expanding_folds: it needs strictly
    ordered distinct seasons and silently dedupes duplicates, and a four-season
    half emits ZERO folds against min_train_seasons=5, leaving the calibrator
    nothing to fit."""
    frame = _matrix()
    out = resample_players_within_season(frame, np.random.default_rng(0))
    assert set(out["season"]) == set(frame["season"])
    assert len(out) == len(frame)


def test_each_season_keeps_its_own_row_count():
    frame = pd.concat([_matrix(seasons=(2020,), per_season=10),
                       _matrix(seasons=(2021,), per_season=90)])
    out = resample_players_within_season(frame, np.random.default_rng(1))
    counts = out.groupby("season").size()
    assert counts[2020] == 10 and counts[2021] == 90


def test_resampling_actually_resamples():
    """Duplicates must appear WITHIN a season — the same player appears once
    per season in the source, so counting duplicates across the whole frame
    measures the panel structure rather than the bootstrap."""
    frame = _matrix()
    out = resample_players_within_season(frame, np.random.default_rng(2))
    dupes = out.groupby("season")["player_id"].apply(
        lambda s: s.duplicated().sum())
    assert dupes.sum() > 0
    assert (frame.groupby("season")["player_id"]
                 .apply(lambda s: s.duplicated().sum()).sum() == 0)


def test_missing_season_column_is_named():
    with pytest.raises(KeyError, match="season"):
        resample_players_within_season(pd.DataFrame({"x": [1]}),
                                       np.random.default_rng(0))


# ------------------------------------------------------ top-k agreement
def test_identical_orderings_score_one():
    order = [f"p{i}" for i in range(50)]
    assert _top_k_agreement(order, order, 12) == pytest.approx(1.0)


def test_membership_disagreement_is_penalized():
    """Ranking only the INTERSECTION would score two orderings sharing three
    players as highly as two sharing eighty. The union, with absentees ranked
    last, penalizes disagreement about membership too — which is the
    disagreement that actually changes a pick."""
    a = [f"p{i}" for i in range(50)]
    b = [f"q{i}" for i in range(12)] + a
    shared = _top_k_agreement(a, a, 12)
    disjoint = _top_k_agreement(a, b, 12)
    assert disjoint < shared


def test_reversed_ordering_scores_negative():
    order = [f"p{i}" for i in range(50)]
    assert _top_k_agreement(order, order[::-1], 50) < 0


def test_degenerate_inputs_do_not_raise():
    assert _top_k_agreement(["a"], ["a"], 5) == 1.0
    assert _top_k_agreement(["a"], ["b"], 5) in (0.0, 1.0)
    assert _top_k_agreement([], [], 5) == 1.0


# ------------------------------------------------------------ the study
def _states(n=6):
    return [{"state_id": f"s{i}", "round": i % 3 + 1, "drafted": "[]"}
            for i in range(n)]


def test_a_deterministic_model_is_perfectly_reliable():
    order = [f"p{i}" for i in range(100)]
    report = reliability(_states(), lambda state, refit: order, n_refits=3)
    assert (report.overall["spearman"] > 0.999).all()
    assert (report.overall["exact_match_at_1"] == 1.0).all()


def test_a_random_model_is_not():
    def rank_fn(state, refit):
        rng = np.random.default_rng(abs(hash((state["state_id"], refit))) % 2**32)
        order = [f"p{i}" for i in range(100)]
        rng.shuffle(order)
        return order

    report = reliability(_states(20), rank_fn, n_refits=3)
    assert report.overall["spearman"].max() < 0.5
    assert report.overall["exact_match_at_1"].max() < 0.2


def test_the_upper_bound_flag_travels_with_the_report():
    """A stability figure quoted without its scope is the kind of number that
    gets believed for years."""
    report = reliability(_states(), lambda s, r: ["a", "b"], n_refits=2)
    assert report.is_upper_bound is True
    assert "UPPER BOUND" in report.describe()


def test_by_round_is_reported_separately():
    """Round 1 is nearly deterministic while the middle rounds are where the
    ordering is genuinely uncertain. One average would hide that."""
    report = reliability(_states(9), lambda s, r: ["a", "b", "c"], n_refits=2)
    assert set(report.by_round["round"]) == {1, 2, 3}


def test_one_refit_is_rejected():
    with pytest.raises(ValueError, match="at least two refits"):
        reliability(_states(), lambda s, r: ["a"], n_refits=1)


def test_no_states_is_rejected():
    with pytest.raises(ValueError, match="no recorded draft states"):
        reliability([], lambda s, r: ["a"], n_refits=2)
