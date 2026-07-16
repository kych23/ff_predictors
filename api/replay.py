"""Rebuild draft state by replaying an event history.

The history format is shared with the scripts/draft.py save file:
history = list[command]; command = list[event];
event = ["pick", player_id, mine] | ["skip", token].
State is always a pure function of (history, board, draft_position) — undo is
"pop the last command and replay", never per-action reversal.
"""
from __future__ import annotations

import pandas as pd

from src.config import LeagueConfig
from src.recommender.roster_state import RosterState


def apply_event(state: RosterState, board: pd.DataFrame, ev: list) -> None:
    kind = ev[0]
    if kind == "skip":
        state.drafted.add(ev[1])
        return
    pid, mine = ev[1], ev[2]
    prow = board.loc[board["player_id"] == pid]
    if prow.empty:
        state.record_pick(pid, None, mine=mine)
        return
    prow = prow.iloc[0]
    team_val = prow["team"] if "team" in board.columns and pd.notna(prow.get("team")) else None
    bye_val = (int(prow["bye_week"]) if "bye_week" in board.columns
               and pd.notna(prow.get("bye_week")) else None)
    state.record_pick(pid, prow["position"], mine=mine, team=team_val, bye_week=bye_val)


def replay_history(history: list, board: pd.DataFrame, cfg: LeagueConfig,
                   draft_position: int) -> RosterState:
    state = RosterState(cfg=cfg, draft_position=draft_position)
    for command in history:
        for ev in command:
            apply_event(state, board, ev)
    return state
