"""When a starter is drawn out, the next man up plays.

`build_masks` picks the lineup from projections times the availability HAZARD,
so a 70%-available starter is chosen on 70% of his rate — but the drawn
`active` indicator never reached the mask. A replication that drew him out left
the roster playing a man short, which no manager does.

Measured on the shipped bundle: **22.5% of chosen-starter player-weeks draw at
exactly zero**, and mean team-week ran about 20% low.

**Not clairvoyance.** §15.3 forbids selecting on realized POINTS. Who is active
is on the injury report before kickoff, and the substitute is chosen by
PROJECTION — an earlier version ranked the bench by drawn points, which is
picking the man who happened to have a good week, and it inflated the
correction from +20.6% to +27.2%.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.config import load_league
from src.engine.sim import kernel


@pytest.fixture(scope="module")
def league():
    return load_league()


class _Bundle:
    """Two RB, two WR, one QB — enough for a flex group and a lone position."""

    positions = np.array(["RB", "RB", "WR", "WR", "QB", "QB"])
    rate_p50 = np.array([20.0, 10.0, 18.0, 9.0, 22.0, 8.0])
    bye_weeks = np.zeros(6, dtype=int)
    n_players = 6

    def __init__(self, weeks=1):
        self.games_hazard = np.ones((6, weeks))


def _setup(starters, bench, weeks=1, reps=4):
    """masks mark `starters`; everyone else on the roster is bench."""
    roster = np.array(sorted(starters + bench))
    masks = np.zeros((1, weeks, 6), dtype="float32")
    masks[0, :, starters] = 1.0
    return roster, masks


def _run(points, roster, masks, league, weeks=1):
    bundle = _Bundle(weeks)
    team_week = kernel.evaluate_rosters(points, masks)
    fixed = kernel.apply_starter_substitution(
        points, masks, team_week, [roster], bundle, league)
    return team_week, fixed


# ------------------------------------------------------------- the fix
def test_an_inactive_starter_is_replaced_by_his_backup(league):
    roster, masks = _setup(starters=[0, 2], bench=[1])
    points = np.zeros((6, 1, 2))
    points[0, 0, :] = [15.0, 0.0]      # RB1 plays rep 0, out rep 1
    points[2, 0, :] = [12.0, 12.0]     # WR1 plays both
    points[1, 0, :] = [7.0, 7.0]       # RB2 on the bench, available

    before, after = _run(points, roster, masks, league)
    assert after[0, 0, 0] == pytest.approx(before[0, 0, 0]), "nobody was out"
    assert after[0, 0, 1] == pytest.approx(before[0, 0, 1] + 7.0)


def test_a_healthy_lineup_is_untouched(league):
    roster, masks = _setup(starters=[0, 2], bench=[1])
    points = np.zeros((6, 1, 3))
    points[0, 0, :] = 15.0
    points[2, 0, :] = 12.0
    points[1, 0, :] = 9.0
    before, after = _run(points, roster, masks, league)
    assert np.allclose(before, after)


def test_two_starters_out_pull_two_different_backups(league):
    """Each substitute is used once. Counting one bench player twice would
    invent points that no roster could have scored."""
    roster, masks = _setup(starters=[0, 2], bench=[1, 3])
    points = np.zeros((6, 1, 1))
    points[1, 0, :] = 7.0
    points[3, 0, :] = 5.0
    before, after = _run(points, roster, masks, league)
    assert after[0, 0, 0] == pytest.approx(12.0)      # 7 + 5, each once


def test_an_empty_bench_leaves_the_slot_empty(league):
    roster, masks = _setup(starters=[0, 2], bench=[])
    points = np.zeros((6, 1, 1))
    points[2, 0, :] = 12.0
    before, after = _run(points, roster, masks, league)
    assert np.allclose(before, after)


def test_an_inactive_backup_cannot_cover(league):
    """A bench player drawn at zero is out too. Substituting him would score
    the same as leaving the slot empty, and pretending otherwise is invention."""
    roster, masks = _setup(starters=[0], bench=[1])
    points = np.zeros((6, 1, 1))     # everyone out
    before, after = _run(points, roster, masks, league)
    assert np.allclose(before, after)


# ------------------------------------------------- position eligibility
def test_a_quarterback_cannot_cover_a_receiver(league):
    """Substitution is bounded by what could legally take the slot."""
    roster, masks = _setup(starters=[2], bench=[4])   # WR starting, QB benched
    points = np.zeros((6, 1, 1))
    points[4, 0, :] = 25.0                            # the QB had a huge week
    before, after = _run(points, roster, masks, league)
    assert np.allclose(before, after), "a QB covered a WR slot"


def test_flex_positions_cover_each_other(league):
    """RB/WR/TE are flex-eligible in this league, so they interchange."""
    roster, masks = _setup(starters=[0], bench=[2])   # RB out, WR benched
    points = np.zeros((6, 1, 1))
    points[2, 0, :] = 11.0
    before, after = _run(points, roster, masks, league)
    assert after[0, 0, 0] == pytest.approx(11.0)


def test_a_backup_quarterback_covers_a_quarterback(league):
    roster, masks = _setup(starters=[4], bench=[5])
    points = np.zeros((6, 1, 1))
    points[5, 0, :] = 14.0
    before, after = _run(points, roster, masks, league)
    assert after[0, 0, 0] == pytest.approx(14.0)


# ------------------------------------------------------ non-clairvoyance
def test_the_substitute_is_chosen_by_projection_not_by_what_he_scored(league):
    """The rule §15.3 exists for. Bench RB2 projects higher than the deep WR
    but scores less this week; the manager starts RB2 because that is what he
    knew before kickoff."""
    roster, masks = _setup(starters=[0], bench=[1, 3])
    points = np.zeros((6, 1, 1))
    points[1, 0, :] = 4.0     # rate_p50 10.0 — the higher projection
    points[3, 0, :] = 30.0    # rate_p50  9.0 — the lucky one
    before, after = _run(points, roster, masks, league)
    assert after[0, 0, 0] == pytest.approx(4.0), (
        "the bench was ranked by hindsight, not by projection")


# ----------------------------------------------------------- the switch
def test_the_correction_can_be_turned_off(league):
    class _Off:
        class simulation:
            starter_substitution = {"enabled": False}

    roster, masks = _setup(starters=[0], bench=[1])
    points = np.zeros((6, 1, 1))
    points[1, 0, :] = 7.0
    team_week = kernel.evaluate_rosters(points, masks)
    same = kernel.apply_starter_substitution(
        points, masks, team_week, [roster], _Bundle(), league, _Off())
    assert np.allclose(same, team_week)


def test_the_live_path_applies_it():
    import inspect

    from src.app.cockpit import build

    source = inspect.getsource(build.build_tiers)
    assert "apply_starter_substitution" in source
