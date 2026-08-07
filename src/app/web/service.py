"""Cockpit orchestration: session, sources, recommendation worker (§19).

**The single-writer rule.** Every mutation of `Session` and `DecisionLedger`
happens on the asyncio event-loop thread. `Session._append` mutates a list,
rebuilds state by replay and rewrites a file; with two writers an `undo` racing
an append drops the wrong event. `DecisionLedger` wraps `sqlite3.connect`
without `check_same_thread`, so it is thread-affine by default. Confining writes
to one thread means neither needs a lock protocol that can be got wrong.

The corollary, which is easy to lose: **the apply path contains no `await`**.
The duplicate check -> `Board.take` -> `Session._append` sequence is
check-then-act; an `await` landing inside it would let the poller interleave and
break the invariant that the check protects.

**The worker is pure compute.** It receives an immutable description of the
board state and returns a `Recommendation`. It never touches the session.

**Supersede, don't cancel.** `allocate` has no cancellation token, so a run made
obsolete by a pick or an undo is abandoned on completion rather than stopped —
worst case one wasted `budget_seconds`. A monotonic `generation` counter, bumped
on every state change, is what makes that detectable.

**Narration is off the pick clock.** `OllamaBackend` defaults to a 30 s timeout
plus two 2 s probes; bundling that into the run would put worst-case latency
near 55 s against a 25 s budget. `rec_ready` fires as soon as the ladder
returns, and narration follows as its own event.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.cockpit.ledger import DecisionLedger
from src.app.cockpit.session import Session
from src.app.web import schemas
from src.app.web.resolve import (
    AMBIGUOUS, RESOLVED, Resolution, build_spine, resolve_name,
)
from src.app.web.sources.base import DraftEvent, DraftEventSource, SourceStatus
from src.app.web.sources.manual import ManualSource
from src.app.web.sources.replay import ReplaySource
from src.app.web.sources.yahoo import YahooSource
from src.core.errors import DataError
from src.engine.decision.board import Board

logger = logging.getLogger(__name__)

IDLE, RUNNING, READY, ERROR = "idle", "running", "ready", "error"


class SessionExists(Exception):
    """A session file is already on disk and `resume` was not requested."""


class RunInFlight(Exception):
    """One recommendation at a time."""


@dataclass
class RecommendationState:
    status: str = IDLE
    payload: dict | None = None
    detail: str = ""
    started_at: float = 0.0
    generation: int = -1

    def snapshot(self) -> dict:
        if self.status == RUNNING:
            return {"status": RUNNING,
                    "elapsed_s": round(time.monotonic() - self.started_at, 1)}
        if self.status == READY and self.payload is not None:
            return {"status": READY, **self.payload}
        if self.status == ERROR:
            return {"status": ERROR, "detail": self.detail,
                    "previous": self.payload}
        return {"status": IDLE}


def make_source(name: str, *, web_cfg, strategy) -> DraftEventSource:
    if name == "manual":
        return ManualSource()
    if name == "replay":
        if not web_cfg.replay_path:
            raise DataError("source 'replay' requires web.replay_path")
        return ReplaySource(path=web_cfg.resolved(web_cfg.replay_path))
    if name == "yahoo":
        yahoo = getattr(strategy, "yahoo", {}) or {}
        return YahooSource(
            league_key=yahoo.get("league_key"),
            token_path=web_cfg.resolved(web_cfg.yahoo.token_path),
            interval_s=web_cfg.poll_interval_seconds,
            manager_map=yahoo.get("manager_map", {}) or {},
        )
    raise DataError(f"unknown source {name!r}")


@dataclass
class CockpitService:
    cfg: Any
    strategy: Any
    web_cfg: Any
    board: Board
    session: Session | None = None
    ledger: DecisionLedger | None = None
    source: DraftEventSource | None = None
    generation: int = 0
    recommendation: RecommendationState = field(default_factory=RecommendationState)
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    _spine: Any = field(default=None, repr=False)
    _running: bool = False

    # --------------------------------------------------------------- setup
    def __post_init__(self) -> None:
        self._spine = build_spine(self.board.players, self.board.snapshot_id)

    @property
    def shortlist(self) -> int:
        """From strategy.yaml, NOT web.yaml — it changes which candidate wins,
        so it belongs inside `strategy_hash`."""
        return int(self.strategy.simulation.decision.get("shortlist_size", 6))

    @property
    def players(self) -> pd.DataFrame:
        return self.board.players

    def start_session(self, *, seat: int, source: str, resume: bool) -> Session:
        # Release whatever the previous session held FIRST. Restarting or
        # resuming in the same process otherwise leaks a sqlite3 connection per
        # call, and for YahooSource an open OAuth2Client, because `close()` only
        # ever reaches the last one at shutdown.
        self.close()
        path = self.web_cfg.resolved(self.web_cfg.session_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not resume:
            raise SessionExists(
                f"a session already exists at {path}; pass resume=true to "
                f"continue it or DELETE /api/session to archive it")

        session = Session(my_seat=seat, teams=self.cfg.teams,
                          rounds=self.cfg.roster.rounds,
                          snapshot_id=self.board.snapshot_id, path=path)
        self.session = session.resume_or_new() if resume else session
        # Persist immediately. A session is durable from the moment it is
        # created, not from the first pick — otherwise a crash between "start"
        # and "someone picks" loses the seat, which is the one thing the
        # operator cannot reconstruct from the room.
        self.session.save()
        self.ledger = DecisionLedger(
            self.web_cfg.resolved(self.web_cfg.ledger_path))
        self.source = make_source(source, web_cfg=self.web_cfg,
                                  strategy=self.strategy)
        self.source.start()
        self.generation += 1
        self.recommendation = RecommendationState()
        return self.session

    def archive_session(self) -> Path | None:
        """Never truncates the ledger — it is append-only and hash-chained by
        design, and a draft you deleted the record of is a draft you cannot
        review in November."""
        path = self.web_cfg.resolved(self.web_cfg.session_path)
        archived = None
        if path.exists():
            archived = path.with_suffix(f".{int(time.time())}.bak")
            path.replace(archived)
        self.close()
        self.source = None
        self.ledger = None
        self.session = None
        self.generation += 1
        self.recommendation = RecommendationState()
        return archived

    def close(self) -> None:
        if self.source is not None:
            self.source.stop()
        if self.ledger is not None:
            self.ledger.close()

    # ------------------------------------------------------------ resolving
    def resolve(self, text: str) -> Resolution:
        drafted = set(self.session.state.drafted) if self.session else set()
        return resolve_name(text, self._spine, self.players, exclude=drafted)

    # -------------------------------------------------------------- writing
    def apply_event(self, event: DraftEvent) -> str:
        """Apply one pick. SYNCHRONOUS end to end — see the module docstring.

        Returns what happened: "recorded", "unresolved", or "ambiguous".
        """
        session = self._require_session()
        if event.seat is not None and not 0 <= event.seat < self.cfg.teams:
            # `rollout` indexes a fixed-length per-seat list, so an out-of-range
            # seat raises IndexError mid-recommendation AND persists into
            # by_seat, corrupting every later pick. Refuse it at the door.
            raise DataError(
                f"seat {event.seat} is outside 0..{self.cfg.teams - 1}")
        player_id = event.player_id
        if player_id is None:
            resolution = self.resolve(event.raw_name)
            if resolution.status == RESOLVED:
                player_id = resolution.player_id
            elif resolution.status == AMBIGUOUS:
                return AMBIGUOUS
        if player_id is None:
            session.record_unresolved(event.raw_name)
            self._bump()
            return "unresolved"

        if player_id in session.state.drafted:
            raise DataError(f"{player_id} has already been drafted")

        seat = event.seat
        if seat is not None and seat == session.my_seat or seat is None and session.is_my_turn():
            session.record_my_pick(player_id, raw_input=event.raw_name)
        else:
            session.record_pick(player_id, seat=seat, raw_input=event.raw_name)

        self._ledger_my_pick(player_id)
        self._bump()
        return "recorded"

    def undo(self) -> None:
        self._require_session().undo()
        self._bump()

    def _ledger_my_pick(self, player_id: str) -> None:
        """Recorded alongside the pick so `followed` stays derivable rather
        than remembered. A ledger fault must never cost a pick."""
        state = self.recommendation
        if (self.ledger is None or self.session is None
                or state.status != READY or state.payload is None):
            return
        if player_id not in set(self.session.state.my_roster):
            return
        try:
            self.ledger.append(
                self.board.snapshot_id,
                pick_no=self.session.state.pick_number - 1,
                tier=int(state.payload.get("tier", 3)),
                recommendation=schemas.ledger_recommendation(state.payload),
                actual_pick=player_id, snapshot_id=self.board.snapshot_id)
        except Exception as exc:                      # noqa: BLE001
            logger.warning("ledger write failed: %s", exc)

    def _bump(self) -> None:
        self.generation += 1

    def _require_session(self) -> Session:
        if self.session is None:
            raise LookupError("no active session")
        return self.session

    # ------------------------------------------------------------- reading
    def session_payload(self) -> dict:
        session = self._require_session()
        state = session.state
        names = {str(r.player_id): str(r.player_name)
                 for r in self.players.itertuples(index=False)}
        positions = {str(r.player_id): str(r.position)
                     for r in self.players.itertuples(index=False)}
        # `is not None`, NOT truthiness: ManualSource defines __len__, so an
        # empty queue makes the adapter itself falsy and the fallback fires
        # on a perfectly healthy source. Caught by a live run reporting
        # source "none" for a session created with "manual".
        status = (self.source.status if self.source is not None
                  else SourceStatus())
        return {
            "seat": session.my_seat, "teams": self.cfg.teams,
            "rounds": self.cfg.roster.rounds,
            "snapshot_id": self.board.snapshot_id,
            "pick_number": state.pick_number,
            "round": (state.pick_number - 1) // self.cfg.teams + 1,
            "on_the_clock": session.on_the_clock(),
            "is_my_turn": session.is_my_turn(),
            "picks_until_my_turn": session.picks_until_my_turn(),
            "is_complete": session.is_complete,
            "generation": self.generation,
            "drafted_count": len(state.drafted),
            "unresolved": list(state.unresolved),
            "pick_clock_seconds": self.web_cfg.pick_clock_seconds,
            "my_roster": [{"player_id": p, "name": names.get(p, p),
                           "position": positions.get(p, "")}
                          for p in state.my_roster],
            "source": {"name": (self.source.name if self.source is not None
                                else "none"),
                       "state": status.state, "detail": status.detail},
        }

    def board_payload(self, limit: int = 50) -> dict:
        drafted = set(self.session.state.drafted) if self.session else set()
        live = self.players[~self.players["player_id"].astype(str).isin(drafted)]
        # ADP ascending, unpriced players last rather than dropped.
        live = live.sort_values("adp", na_position="last").head(limit)
        return {"players": [{
            "player_id": str(r.player_id), "name": str(r.player_name),
            "position": str(r.position),
            "team": None if pd.isna(r.team) else str(r.team),
            "adp": schemas._json_float(r.adp),
            "bye_week": schemas._json_float(getattr(r, "bye_week", None)),
        } for r in live.itertuples(index=False)]}

    # ----------------------------------------------------- recommendation
    def build_run_args(self) -> dict:
        session = self._require_session()
        state = session.state
        return {
            "seat": session.my_seat + 1,          # build_tiers is 1-indexed
            "drafted": list(state.drafted),
            "my_roster": list(state.my_roster),
            "by_seat": {k: list(v) for k, v in state.by_seat.items()},
            "unresolved_count": len(state.unresolved),
            "reps": self.web_cfg.engine.reps,
            "shortlist": self.shortlist,
            "budget_s": self.web_cfg.engine.budget_seconds,
        }

    def run_recommendation(self, args: dict):
        """Pure compute. Runs in an executor; touches no shared state."""
        from src.app.cockpit.build import build_tiers
        from src.app.cockpit.ladder import recommend

        board = Board.from_bundle()
        for pid in args["drafted"]:
            try:
                board.take(pid)
            except Exception:                     # noqa: BLE001 — a stale id must not cost a pick
                pass
        tiers, _ = build_tiers(
            board, self.cfg, self.strategy, args["seat"], args["reps"],
            args["shortlist"], my_roster=args["my_roster"],
            unresolved_count=args["unresolved_count"], by_seat=args["by_seat"])
        demote = float(
            self.strategy.simulation.decision["demote_to_tier1_after_seconds"])
        rec = recommend(tiers, budget_s=args["budget_s"], demote_after_s=demote)
        return rec, getattr(tiers.get(0), "record", None)

    def payload_for(self, rec, generation: int) -> dict:
        return schemas.recommendation_payload(
            rec, self.players, snapshot_id=self.board.snapshot_id,
            generation=generation, reps=self.web_cfg.engine.reps,
            shortlist=self.shortlist,
            budget_seconds=self.web_cfg.engine.budget_seconds)

    def narrate(self, record):
        from src.app.narration import NarrationConfig, narrate as _narrate
        from src.app.narration.backends import OllamaBackend

        cfg = NarrationConfig.from_strategy(self.strategy)
        backend = OllamaBackend(timeout=self.web_cfg.narration_timeout_seconds)
        try:
            return _narrate(record, cfg, backend=backend)
        except TypeError:                          # backend kwarg unsupported
            return _narrate(record, cfg)

    # ------------------------------------------------------------- SSE fan-out
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def publish(self, event: str, data: dict) -> None:
        """Drop the oldest frame for a slow client rather than blocking the
        loop. A browser that cannot keep up must not stall the draft."""
        for queue in list(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait({"event": event, "data": data})
            except asyncio.QueueFull:              # pragma: no cover
                logger.warning("dropping %s for a stalled subscriber", event)
