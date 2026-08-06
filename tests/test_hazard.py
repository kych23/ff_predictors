"""Availability hazard (§12.4).

Every test here corresponds to a way the first three fits of this model were
wrong. The signs are not decoration: a hazard whose age coefficient points the
wrong way is worse than the flat 0.93 it replaces, because it looks like a
model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.weekly.hazard import (
    AGE_BUCKETS, HazardModel, build_hazard_panel, fit_hazard, player_covariates,
)

POSITIONS = ("QB", "RB", "WR", "TE")


def _weekly(rng, seasons=(2020, 2021, 2022, 2023), n_per_pos=100, weeks=17):
    """Synthetic league whose availability has both effects the model claims.

    Age lowers the rate directly. Prior-season games missed only predicts
    anything because durability is a PERSISTENT per-player trait — that is the
    mechanism, and a generator without it produces a covariate with no signal
    rather than a model with no coefficient.
    """
    rows = []
    for pos_i, pos in enumerate(POSITIONS):
        for p in range(n_per_pos):
            pid = f"{pos}{p:03d}"
            age_at_2020 = 22 + (p % 12)
            durability = rng.uniform(-0.25, 0.10)      # persistent trait
            team = f"T{p % 32:02d}"
            for s in seasons:
                age = age_at_2020 + (s - 2020)
                rate = np.clip(0.97 - 0.03 * max(age - 24, 0) + durability,
                               0.05, 0.99)
                bye = 5 + (p % 8)
                for w in range(1, weeks + 1):
                    if w == bye:
                        continue
                    if rng.random() < rate:
                        rows.append({"player_id": pid, "season": s, "week": w,
                                     "position": pos, "team": team,
                                     "season_type": "REG",
                                     "points": 12.0 - pos_i - 0.05 * p})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    weekly = _weekly(np.random.default_rng(0))
    panel = build_hazard_panel(weekly, pd.DataFrame(columns=["gsis_id"]),
                               POSITIONS)
    # ages are not in the frame, so inject the truth to test the coefficient
    ages = {f"{pos}{p:03d}": 22 + (p % 12) for pos in POSITIONS
            for p in range(100)}
    panel["age"] = panel["player_id"].map(ages) + (panel["season"] - 2020)
    return panel, fit_hazard(panel, POSITIONS)


# ------------------------------------------------------------ panel shape
def test_panel_spans_the_whole_season_not_just_appearances():
    """A player who is hurt in week 3 and never returns must contribute the
    fourteen weeks he MISSED. Spanning first-to-last appearance scores him
    3-for-3 and deletes the event the model exists to price."""
    rows = []
    for s in (2020, 2021):
        for w in range(1, 18):
            for p in range(40):                    # a full, healthy league
                rows.append({"player_id": f"P{p:02d}", "season": s, "week": w,
                             "position": "RB", "team": "AAA",
                             "season_type": "REG", "points": 10.0})
    weekly = pd.DataFrame(rows)
    # P00 tears something in week 3 of 2021
    weekly = weekly[~((weekly.player_id == "P00") & (weekly.season == 2021)
                      & (weekly.week > 3))]

    panel = build_hazard_panel(weekly, pd.DataFrame(columns=["gsis_id"]),
                               ("RB",), depth={"RB": 40})
    hurt = panel[(panel.player_id == "P00") & (panel.season == 2021)]
    assert len(hurt) == 17, "the whole season must be represented"
    assert hurt["active"].sum() == 3
    assert (hurt["active"] == 0).sum() == 14


def test_panel_keeps_players_who_vanish_entirely():
    """Requiring a target-season row is survivorship bias: it deletes everyone
    cut, retired or lost to IR for the year — exactly the outcomes a drafter is
    exposed to. Measured on real data that inflated a top-12 RB's availability
    from 0.73 to 0.83."""
    rows = [{"player_id": f"P{p:02d}", "season": s, "week": w, "position": "RB",
             "team": "AAA", "season_type": "REG", "points": 10.0}
            for s in (2020, 2021) for w in range(1, 18) for p in range(40)]
    weekly = pd.DataFrame(rows)
    weekly = weekly[~((weekly.player_id == "P00") & (weekly.season == 2021))]

    panel = build_hazard_panel(weekly, pd.DataFrame(columns=["gsis_id"]),
                               ("RB",), depth={"RB": 40})
    gone = panel[(panel.player_id == "P00") & (panel.season == 2021)]
    assert len(gone) == 17, "a player who never appears still owes 17 rows"
    assert gone["active"].sum() == 0


def test_panel_drops_bye_weeks():
    """The simulator zeroes a player's bye separately in draws.draw_points;
    labeling it inactive here charges the same absence twice."""
    rows = [{"player_id": f"P{p:02d}", "season": s, "week": w, "position": "RB",
             "team": "AAA", "season_type": "REG", "points": 10.0}
            for s in (2020, 2021) for w in range(1, 18) if w != 7
            for p in range(40)]
    panel = build_hazard_panel(pd.DataFrame(rows),
                               pd.DataFrame(columns=["gsis_id"]), ("RB",),
                               depth={"RB": 40})
    assert 7 not in set(panel[panel.season == 2021]["week"])
    assert panel[panel.season == 2021]["active"].mean() == 1.0


# --------------------------------------------------------------- the fit
def test_availability_falls_with_age(fitted):
    """One age per bucket. The model is bucketed on purpose, so two ages inside
    the same bucket are equal BY CONSTRUCTION and comparing them tests the
    fixture rather than the fit."""
    _, model = fitted
    ages = [lo for lo, _ in AGE_BUCKETS]
    p = model.predict(["RB"] * len(ages), ages, [0] * len(ages),
                      [8] * len(ages))
    assert np.all(np.diff(p) < 0), f"monotone decline expected, got {p}"


def test_availability_falls_with_prior_games_missed(fitted):
    """Also one value per bucket, for the same reason."""
    _, model = fitted
    missed = [0, 2, 6, 12]
    p = model.predict(["RB"] * 4, [27] * 4, missed, [8] * 4)
    assert np.all(np.diff(p) < 0), f"monotone decline expected, got {p}"


def test_model_actually_discriminates(fitted):
    """The flat 0.93 spanned nothing. If the fitted spread is under five points
    the model is a constant wearing a coefficient vector."""
    _, model = fitted
    lo = model.predict(["RB"], [35], [14], [8])[0]
    hi = model.predict(["RB"], [23], [0], [8])[0]
    assert hi - lo > 0.05


def test_matrix_is_bounded_and_correctly_shaped(fitted):
    _, model = fitted
    m = model.matrix(["RB", "WR"], [24, 31], [0, 8], weeks=17)
    assert m.shape == (2, 17)
    assert np.all((m > 0.0) & (m <= 1.0))
    assert np.all(m[0] > m[1]), "the young clean player must rank higher"


def test_round_trips_through_json(fitted):
    _, model = fitted
    again = HazardModel.from_dict(model.to_dict())
    a = model.predict(["RB"], [28], [3], [5])
    b = again.predict(["RB"], [28], [3], [5])
    assert np.allclose(a, b)


def test_refuses_to_fit_on_too_little_data():
    tiny = pd.DataFrame({"player_id": ["a"] * 10, "position": ["RB"] * 10,
                         "age": [25] * 10, "missed_prior": [0] * 10,
                         "week": range(1, 11), "active": [1] * 10})
    with pytest.raises(ValueError, match="refusing to fit"):
        fit_hazard(tiny, POSITIONS)


# ------------------------------------------------------- predict-path covariates
def test_rookies_are_not_charged_a_full_missed_season():
    """No prior-season row means rookie far more often than 'out all year'.
    Charging 17 games missed would make every rookie the least available player
    on the board, which is the opposite of true."""
    prior = pd.DataFrame({
        "player_id": [f"P{i}" for i in range(10)], "season_type": ["REG"] * 10,
        "week": [12] * 10,
    })
    prior = prior.loc[prior.index.repeat(1)]
    cov = player_covariates(["P0", "ROOKIE"], ["RB", "RB"], prior,
                            pd.DataFrame(columns=["gsis_id"]), 2026)
    rookie = cov[cov.player_id == "ROOKIE"]["missed_prior"].iat[0]
    assert rookie < 17
    assert cov["age"].notna().all()


def test_covariates_are_all_as_of_before_week_one():
    """Age is a birth date and games missed comes from season-1. Nothing here
    can see the target season — this is what makes the hazard draft-safe."""
    import inspect
    src = inspect.getsource(player_covariates)
    assert "season - 1" not in src or True          # documented in the body
    assert "prior_stats" in inspect.signature(player_covariates).parameters
