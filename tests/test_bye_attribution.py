"""Bye weeks reach the EXPLANATION, not just the objective.

Byes were already priced: `sim.kernel.starter_values` zeroes a player in his
bye week off the bundle, so a roster with four players idle in week 9 loses
real dollars in the simulation. What was missing is that nothing ever said so.
`AttributionRecord.bye_conflicts` existed, `render_table` printed it and the
prompt carried a `has_bye_conflict` flag — but `build_tiers` never passed the
argument, so the field was empty on every production recommendation and the
flag was False forever. The narration would dock a player for a bye pile-up
and then explain the pick with an unrelated fact.

These tests pin the wiring at both ends: the board columns survive into
`RosterState`, and a conflict is only reported when my roster is ALREADY on
that week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app.cockpit.build import _bye_conflicts, my_roster_rows
from src.app.narration.attribution import AttributionRecord
from src.app.narration.render import render_table
from src.core.config import load_league
from src.engine.decision.roster_state import RosterState


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture
def frame():
    """Four players; two share a week-9 bye, one has none on record."""
    return pd.DataFrame({
        "player_id": ["p0", "p1", "p2", "p3"],
        "player_name": ["Alpha", "Bravo", "Charlie", "Delta"],
        "position": ["RB", "WR", "TE", "QB"],
        "team": ["DET", "DET", "KC", "PHI"],
        "bye_week": [9, 9, 6, 0],
    })


@pytest.fixture
def row_of(frame):
    return {pid: i for i, pid in enumerate(frame["player_id"])}


def _roster(league, frame, row_of, held):
    return RosterState(cfg=league, draft_position=1,
                       my_roster=my_roster_rows(frame, row_of, held))


# --------------------------------------------------------- board -> roster
def test_roster_rows_carry_team_and_bye(frame, row_of):
    rows = my_roster_rows(frame, row_of, ["p0"])
    assert rows == [{"player_id": "p0", "position": "RB",
                     "team": "DET", "bye_week": 9}]


def test_a_zero_bye_week_reads_as_no_bye_on_record(frame, row_of):
    """0 is the bundle's sentinel for "unknown", not week zero. Passing it
    through would let two players with NO bye data look like a conflict."""
    assert my_roster_rows(frame, row_of, ["p3"])[0]["bye_week"] is None


def test_unknown_player_ids_are_dropped_not_crashed(frame, row_of):
    assert my_roster_rows(frame, row_of, ["p0", "ghost"]) == \
        my_roster_rows(frame, row_of, ["p0"])


def test_missing_board_columns_degrade_to_none(frame, row_of):
    """Tier-3 and replay boards do not always carry team or bye_week."""
    thin = frame.drop(columns=["team", "bye_week"])
    row = my_roster_rows(thin, row_of, ["p0"])[0]
    assert row["team"] is None and row["bye_week"] is None


def test_nan_cells_degrade_to_none(frame, row_of):
    frame.loc[0, "bye_week"] = np.nan
    assert my_roster_rows(frame, row_of, ["p0"])[0]["bye_week"] is None


# ------------------------------------------------------------- the conflict
def test_an_empty_roster_has_no_conflicts(league, frame, row_of):
    rs = _roster(league, frame, row_of, [])
    assert _bye_conflicts(rs, frame, row_of["p1"]) == []


def test_a_shared_bye_week_is_a_conflict(league, frame, row_of):
    rs = _roster(league, frame, row_of, ["p0"])          # DET, bye 9
    assert _bye_conflicts(rs, frame, row_of["p1"]) == [9]   # DET, bye 9


def test_a_different_bye_week_is_not_a_conflict(league, frame, row_of):
    rs = _roster(league, frame, row_of, ["p0"])          # bye 9
    assert _bye_conflicts(rs, frame, row_of["p2"]) == []    # bye 6


def test_a_players_own_bye_alone_is_not_a_conflict(league, frame, row_of):
    """Every player has a bye. Reporting all of them would make the flag
    meaningless — only the pile-up is worth a sentence."""
    rs = _roster(league, frame, row_of, ["p2"])          # bye 6
    assert _bye_conflicts(rs, frame, row_of["p1"]) == []    # bye 9, alone


def test_a_candidate_with_no_bye_on_record_never_conflicts(league, frame, row_of):
    rs = _roster(league, frame, row_of, ["p0", "p1"])
    assert _bye_conflicts(rs, frame, row_of["p3"]) == []


def test_bye_counts_ignore_players_with_no_bye(league, frame, row_of):
    rs = _roster(league, frame, row_of, ["p0", "p1", "p3"])
    assert rs.my_bye_week_counts() == {9: 2}


# ------------------------------------------------- it reaches the narration
def _record(bye_conflicts):
    return AttributionRecord(
        pair=("p1", "p2"),
        delta_by_prize={"champion": 3.0},
        delta_weeks=[(1, 0.5), (9, -0.2)],
        roster_slot_affected="WR",
        aleatory_se=1.0, epistemic_se=0.5,
        names={"p1": "Bravo", "p2": "Charlie"},
        bye_conflicts=bye_conflicts,
    )


def test_the_table_fallback_names_the_conflicting_week():
    assert "bye conflicts" in render_table(_record([9]))
    assert "w9" in render_table(_record([9]))


def test_the_table_stays_silent_without_a_conflict():
    assert "bye conflicts" not in render_table(_record([]))


def test_a_bye_week_becomes_a_verifiable_week_claim():
    """The gate checks `week` claims against `record.weeks()`. A bye week that
    never enters the record is a week the narration may not mention."""
    assert 9 in _record([9]).weeks()
    assert 13 not in _record([9]).weeks()


def test_the_prompt_flag_follows_the_record():
    from src.app.narration.backends import prompt_payload

    assert '"has_bye_conflict": true' in prompt_payload(
        _record([9]), indifference_zone=1.0).lower()
    assert '"has_bye_conflict": false' in prompt_payload(
        _record([]), indifference_zone=1.0).lower()


# ------------------------------------------------------------- handcuffs
def test_there_is_no_handcuff_helper_pretending_to_be_wired(league):
    """`my_rb_teams()` sat unused and labelled "for handcuff detection",
    which reads as a built feature. Handcuff value is already in the objective
    via the fitted RB1 x RB2 slot correlation (-0.139); an explicit bonus on
    top would double-count it. If this attribute comes back, something is
    scoring handcuffs twice."""
    assert not hasattr(RosterState(cfg=league, draft_position=1), "my_rb_teams")


# ------------------------------- the market input, wired the same way
def test_survival_probabilities_are_computed_for_both_candidates(frame, row_of):
    from src.app.cockpit.build import _survival_probabilities

    pid_of = {str(i): p for p, i in row_of.items()}
    frame["adp"] = [3.0, 40.0, 80.0, 150.0]
    frame["adp_stdev"] = [2.0, 10.0, 20.0, 30.0]
    out = _survival_probabilities(frame, ["0", "2"], pid_of, next_pick=60)

    assert set(out) == {"p0", "p2"}
    assert 0.0 <= out["p0"] <= 1.0 and 0.0 <= out["p2"] <= 1.0
    assert out["p0"] < out["p2"], "an early-ADP player is less likely to last"


def test_no_next_pick_means_no_survival_claim(frame, row_of):
    """At my last turn nothing survives to a turn that does not exist. 1.0
    would read as 'certain to last', which is the opposite of the truth."""
    from src.app.cockpit.build import _survival_probabilities

    pid_of = {str(i): p for p, i in row_of.items()}
    frame["adp"] = [3.0, 40.0, 80.0, 150.0]
    frame["adp_stdev"] = [2.0, 10.0, 20.0, 30.0]
    assert _survival_probabilities(frame, ["0"], pid_of, next_pick=None) == {}


def test_a_player_with_no_adp_makes_no_claim(frame, row_of):
    from src.app.cockpit.build import _survival_probabilities

    pid_of = {str(i): p for p, i in row_of.items()}
    frame["adp"] = [np.nan, 40.0, 80.0, 150.0]
    frame["adp_stdev"] = [np.nan, 10.0, 20.0, 30.0]
    out = _survival_probabilities(frame, ["0", "1"], pid_of, next_pick=60)
    assert "p0" not in out and "p1" in out


def test_the_probability_quantity_is_reachable_once_populated():
    """`allowed_subjects` builds the generation enum FROM the record, so an
    empty dict makes every probability clause unexpressible and leaves
    `verify`'s probability rule dead code."""
    from src.app.narration.backends import allowed_subjects

    record = _record([])
    assert allowed_subjects(record)["probability"] == []

    with_survival = AttributionRecord(
        pair=("p1", "p2"), delta_by_prize={"champion": 3.0},
        delta_weeks=[(1, 0.5)], roster_slot_affected="WR",
        aleatory_se=1.0, epistemic_se=0.5,
        names={"p1": "Bravo", "p2": "Charlie"},
        survival_probabilities={"p1": 0.12, "p2": 0.64},
    )
    assert allowed_subjects(with_survival)["probability"] == ["Bravo", "Charlie"]
    assert with_survival.probability_for("Bravo") == pytest.approx(0.12)


def test_the_live_cockpit_passes_survival_probabilities():
    import inspect

    from src.app.cockpit import build

    source = inspect.getsource(build.build_tiers)
    assert "survival_probabilities=_survival_probabilities(" in source
