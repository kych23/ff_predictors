"""Replay is a pure function of (history, board): picks land on the right roster,
skips consume a pick slot, and replay order matches incremental application."""

import pytest

# §9.4: this module is pinned to the v1 config shape. api/, web/ and the
# weekly start/sit surface are FROZEN for the build window, so v1 stays
# alive underneath them and these tests are deselected from the default
# run rather than migrated. Thaw is a §22.2 follow-up.
pytestmark = pytest.mark.v1_frozen
import pandas as pd

from api.replay import apply_event, replay_history
from src.config import load_config


def make_board():
    rows = []
    pid = 0
    for pos in ["QB", "RB", "WR", "TE"]:
        for i in range(30):
            pid += 1
            p50 = 250.0 - i * 6
            rows.append({"player_id": f"P{pid:04d}", "name": f"{pos} {i+1}",
                         "position": pos, "team": f"T{(pid % 32) + 1}",
                         "p10": p50 * 0.7, "p50": p50, "p90": p50 * 1.3,
                         "adp": float(pid), "adp_stdev": 6.0,
                         "bye_week": (pid % 14) + 1})
    return pd.DataFrame(rows)


def test_replay_records_my_pick_with_metadata():
    cfg = load_config()
    board = make_board()
    history = [[["pick", "P0031", True]]]  # first RB row
    state = replay_history(history, board, cfg, draft_position=1)
    assert "P0031" in state.drafted
    assert len(state.my_roster) == 1
    mine = state.my_roster[0]
    assert mine["position"] == "RB"
    assert mine["team"] == board.loc[board.player_id == "P0031", "team"].iloc[0]
    assert mine["bye_week"] == int(board.loc[board.player_id == "P0031", "bye_week"].iloc[0])


def test_replay_opponent_pick_not_on_my_roster():
    cfg = load_config()
    state = replay_history([[["pick", "P0031", False]]], make_board(), cfg, draft_position=1)
    assert "P0031" in state.drafted
    assert state.my_roster == []


def test_skip_consumes_a_pick_slot():
    cfg = load_config()
    state = replay_history([[["skip", "_skip_1"]]], make_board(), cfg, draft_position=1)
    assert state.current_overall_pick() == 2
    assert state.my_roster == []


def test_unknown_player_id_still_recorded():
    cfg = load_config()
    state = replay_history([[["pick", "GHOST", False]]], make_board(), cfg, draft_position=1)
    assert "GHOST" in state.drafted


def test_replay_equals_incremental_application():
    cfg = load_config()
    board = make_board()
    history = [[["pick", "P0031", False]], [["pick", "P0061", True]], [["skip", "_skip_3"]]]
    replayed = replay_history(history, board, cfg, draft_position=2)
    from src.recommender.roster_state import RosterState
    incremental = RosterState(cfg=cfg, draft_position=2)
    for command in history:
        for ev in command:
            apply_event(incremental, board, ev)
    assert replayed.drafted == incremental.drafted
    assert replayed.my_roster == incremental.my_roster
    assert replayed.slot_fill == incremental.slot_fill
