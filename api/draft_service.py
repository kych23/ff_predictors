"""Draft session orchestration: the API-side equivalent of scripts/draft.py's
event loop. State is never stored — only history is; every read replays.

Board/replacement are cached per season at module level: projections change at
most daily (pipeline runs), while a live draft polls every few seconds.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from api.db_models import DraftSession
from api.replay import replay_history
from src.config import LeagueConfig
from src.recommender.board import PROJ_COLS, load_board
from src.recommender.recommend import build_replacement_from_projections, recommend
from src.recommender.roster_state import RosterState


class DraftNotFound(Exception):
    pass


class InvalidPick(Exception):
    pass


_BOARD_CACHE: dict[int, pd.DataFrame] = {}


def get_cached_board(season: int, cfg: Optional[LeagueConfig] = None) -> pd.DataFrame:
    from src.config import load_config
    if season not in _BOARD_CACHE:
        _BOARD_CACHE[season] = load_board(season, cfg or load_config())
    return _BOARD_CACHE[season]


class DraftService:
    def __init__(self, db, cfg: LeagueConfig,
                 board_for: Callable[[int], pd.DataFrame]):
        self.db = db
        self.cfg = cfg
        self.board_for = board_for
        self._replacement_cache: dict[int, object] = {}

    # --- session lifecycle ---

    def create_session(self, season: int, draft_position: int,
                       snapshot_id: Optional[str] = None) -> DraftSession:
        if not (1 <= draft_position <= self.cfg.teams):
            raise InvalidPick(
                f"draft_position must be in 1..{self.cfg.teams}, got {draft_position}")
        sess = DraftSession(season=season, draft_position=draft_position,
                            snapshot_id=snapshot_id)
        self.db.add(sess)
        self.db.commit()
        return sess

    def _get(self, session_id: str) -> DraftSession:
        sess = self.db.get(DraftSession, session_id)
        if sess is None:
            raise DraftNotFound(session_id)
        return sess

    # --- state ---

    def _rebuild(self, sess: DraftSession) -> tuple[RosterState, pd.DataFrame]:
        board = self.board_for(sess.season)
        state = replay_history(sess.history, board, self.cfg, sess.draft_position)
        return state, board

    def state(self, session_id: str) -> dict:
        sess = self._get(session_id)
        state, board = self._rebuild(sess)
        names = dict(zip(board["player_id"], board["name"]))
        picks = []
        n = 0
        for command in sess.history:
            for ev in command:
                n += 1
                if ev[0] == "skip":
                    picks.append({"pick_number": n, "player_id": None, "name": None,
                                  "mine": False, "skipped": True})
                else:
                    picks.append({"pick_number": n, "player_id": ev[1],
                                  "name": names.get(ev[1]), "mine": bool(ev[2]),
                                  "skipped": False})
        cur = state.current_overall_pick()
        return {
            "session_id": sess.session_id,
            "season": sess.season,
            "draft_position": sess.draft_position,
            "platform": sess.platform,
            "status": sess.status,
            "teams": self.cfg.teams,
            "rounds": self.cfg.roster.rounds,
            "my_picks": state.my_picks,
            "current_overall_pick": cur,
            "is_my_turn": cur in state.my_picks,
            "next_my_pick": state.next_my_pick(),
            "remaining_picks": state.remaining_picks(),
            "picks": picks,
            "my_roster": [dict(p, name=names.get(p["player_id"])) for p in state.my_roster],
            "open_starters": state.unfilled_mandatory_slots(),
        }

    # --- mutations (history append + full replay; mirrors the CLI's undo stack) ---

    def record_pick(self, session_id: str, player_id: Optional[str] = None,
                    skip: bool = False, mine: Optional[bool] = None) -> dict:
        sess = self._get(session_id)
        state, board = self._rebuild(sess)
        cur = state.current_overall_pick()
        if skip:
            command = [["skip", f"_skip_{cur}"]]
        else:
            if player_id is None:
                raise InvalidPick("player_id required unless skip=true")
            if player_id not in set(board["player_id"]):
                raise InvalidPick(f"unknown player_id {player_id!r}")
            if player_id in state.drafted:
                raise InvalidPick(f"{player_id!r} already drafted")
            if mine is None:
                mine = cur in state.my_picks
            command = [["pick", player_id, bool(mine)]]
        sess.history = sess.history + [command]   # reassign: JSON col, no mutation tracking
        self.db.commit()
        return self.state(session_id)

    def undo(self, session_id: str) -> dict:
        sess = self._get(session_id)
        if sess.history:
            sess.history = sess.history[:-1]
            self.db.commit()
        return self.state(session_id)

    # --- recommendations ---

    def recommendations(self, session_id: str, top_n: int = 10) -> list[dict]:
        sess = self._get(session_id)
        state, board = self._rebuild(sess)
        if sess.season not in self._replacement_cache:
            self._replacement_cache[sess.season] = \
                build_replacement_from_projections(board, cfg=self.cfg)
        replacement = self._replacement_cache[sess.season]
        proj_cols = [c for c in PROJ_COLS if c in board.columns]
        avail = board[proj_cols][~board["player_id"].isin(state.drafted)].copy()
        recs = recommend(avail, state, replacement, cfg=self.cfg, top_n=top_n)
        if recs.empty:
            return []
        names = dict(zip(board["player_id"], board["name"]))
        out = recs.assign(name=recs["player_id"].map(names))
        out = out.where(pd.notna(out), None)     # NaN -> None for JSON
        return out.to_dict(orient="records")
