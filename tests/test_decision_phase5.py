"""Ported tier-2 decision path on the v2 config (§9.2, §7.3).

Behavior must be unchanged from v1; these tests pin the v2 config API
(``flex_eligibility``, ``load_league``) and the DEF -> DST rename.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.config import load_league
from src.engine.decision import (
    quantile_schedule, replacement, roster_state, snake, survival, vona,
)


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture()
def board():
    rows = [
        ("RB", 300.0, 1.5), ("RB", 280.0, 3.0), ("WR", 275.0, 4.0),
        ("WR", 260.0, 6.0), ("QB", 250.0, 30.0), ("TE", 200.0, 20.0),
        ("QB", 240.0, 45.0), ("RB", 190.0, 50.0), ("WR", 185.0, 55.0),
    ]
    return pd.DataFrame([
        {"player_id": f"p{i}", "position": p, "value": v, "adp": a,
         "adp_stdev": 1.5}
        for i, (p, v, a) in enumerate(rows)
    ])


# ------------------------------------------------------------- config API
def test_flex_eligible_flattens_the_v2_mapping(league):
    """v1 stored a flat list; v2 keys by slot name so multiple flex slots are
    expressible. The flattened view answers the tier-2 question."""
    assert league.roster.flex_eligible == frozenset({"RB", "WR", "TE"})
    assert league.roster.flex_slot_names == ("FLEX",)
    assert league.roster.total_flex_slots == 1


def test_dst_replaces_def_everywhere():
    """The DEF -> DST rename is a data rename too (§10.7); no literal survives."""
    import pathlib
    for path in pathlib.Path("src/engine/decision").glob("*.py"):
        assert '"DEF"' not in path.read_text(), f"{path} still spells DEF"


# ------------------------------------------------------------------ snake
def test_snake_pick_sequence_for_slot_four(league):
    picks = roster_state.my_pick_sequence(4, league.teams, league.roster.rounds)
    assert picks[:5] == [4, 21, 28, 45, 52]
    assert len(picks) == 15


def test_snake_reverses_on_even_rounds(league):
    assert roster_state.overall_pick(1, 1, 12) == 1
    assert roster_state.overall_pick(2, 1, 12) == 24
    assert roster_state.overall_pick(2, 12, 12) == 13


def test_has_open_slot_uses_flex(league):
    fill = {"RB": 2, "WR": 2, "TE": 1, "QB": 1, "FLEX": 0, "K": 0, "DST": 0,
            "BENCH": 0}
    # pure RB slots full, but FLEX accepts RB
    assert snake.has_open_slot("RB", dict(fill), league)


# ------------------------------------------------------------ replacement
def test_replacement_levels_are_per_position(league, board):
    levels = replacement.compute_replacement(board, cfg=league)
    for pos in league.roster.modeled_positions:
        assert levels.get(pos) >= 0


# ------------------------------------------------------------- roster state
def _draft(rs, positions):
    """Record picks as MINE. record_pick also tracks opponents' picks (which
    only enter `drafted`), so `mine=True` is what fills my slots."""
    for i, pos in enumerate(positions):
        rs.record_pick(f"{pos}{i}", pos, mine=True)


def test_flex_absorbs_a_third_running_back(league):
    rs = roster_state.RosterState(cfg=league, draft_position=4)
    _draft(rs, ["RB", "RB", "RB"])
    assert rs.slot_fill["RB"] == 2
    assert rs.slot_fill["FLEX"] == 1, "third RB should take the flex slot"


def test_fourth_running_back_goes_to_the_bench(league):
    rs = roster_state.RosterState(cfg=league, draft_position=4)
    _draft(rs, ["RB", "RB", "RB", "RB"])
    assert rs.slot_fill["BENCH"] == 1


def test_opponent_picks_do_not_fill_my_slots(league):
    rs = roster_state.RosterState(cfg=league, draft_position=4)
    rs.record_pick("someone_elses_rb", "RB")     # mine=False by default
    assert rs.slot_fill["RB"] == 0
    assert "someone_elses_rb" in rs.drafted


def test_needs_starter_accounts_for_flex(league):
    rs = roster_state.RosterState(cfg=league, draft_position=1)
    _draft(rs, ["RB", "RB"])
    assert rs.needs_starter("RB"), "flex is still open, so RB still starts"
    _draft(rs, ["WR", "WR", "TE", "TE"])   # the second TE takes the flex
    assert not rs.needs_starter("RB")


# ---------------------------------------------------------------- survival
def test_survival_is_monotonic_in_adp(league):
    early = survival.survival_probability(1.5, 1.0, 21)
    late = survival.survival_probability(55.0, 8.0, 21)
    assert early < 0.01, "an ADP-1.5 player never lasts to pick 21"
    assert late > 0.9, "an ADP-55 player almost always does"


def test_survival_handles_missing_adp():
    """No public ADP means opponents drafting by ADP will not take him, so he
    survives — without this a single NaN zeroes the whole VONA wait term."""
    assert survival.survival_probability(float("nan"), 1.0, 20) == 1.0
    assert survival.survival_probability(None, 1.0, 20) == 1.0


def test_survival_at_last_pick_is_certain():
    assert survival.survival_probability(10.0, 2.0, None) == 1.0


# -------------------------------------------------------------------- vona
def test_vona_prefers_the_scarce_position(league, board):
    """The QB tier is deep, so waiting is cheap and VONA discounts QBs even
    though their raw value is high — the behavior VONA exists to produce."""
    levels = replacement.compute_replacement(board, cfg=league)
    rs = roster_state.RosterState(cfg=league, draft_position=4)
    scored = vona.score_board(board, rs, levels, next_pick=21)

    top = scored.iloc[0]
    assert top["position"] in {"RB", "WR"}

    qb = scored[scored.position == "QB"].iloc[0]
    assert qb["wait_term"] > 5.0, "a second QB survives, so waiting is cheap"
    assert qb["vona_score"] < top["vona_score"]


def test_vona_last_pick_has_no_wait_term(league, board):
    levels = replacement.compute_replacement(board, cfg=league)
    rs = roster_state.RosterState(cfg=league, draft_position=4)
    scored = vona.score_board(board, rs, levels, next_pick=None)
    assert (scored["wait_term"] == 0).all()
    assert (scored["vona_score"] == scored["marginal"]).all()


def test_vona_handles_an_empty_board(league):
    levels = replacement.compute_replacement(
        pd.DataFrame(columns=["player_id", "position", "value"]), cfg=league)
    rs = roster_state.RosterState(cfg=league, draft_position=1)
    empty = pd.DataFrame(columns=["player_id", "position", "value", "adp",
                                  "adp_stdev"])
    assert vona.score_board(empty, rs, levels, next_pick=10).empty


# ------------------------------------------------------- quantile schedule
def test_quantile_schedule_shifts_with_round():
    """The interim heuristic (§8.10): floor-leaning early, ceiling-leaning
    late. It must remain removable without touching the simulator."""
    rounds = 15
    early = quantile_schedule.round_to_alpha(1, rounds)
    late = quantile_schedule.round_to_alpha(rounds, rounds)
    assert early < late, "floor-leaning early, ceiling-leaning late"
    assert 0.0 < early < 1.0 and 0.0 < late < 1.0
