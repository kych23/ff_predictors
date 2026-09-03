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
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.cockpit.ledger import DecisionLedger
from src.app.cockpit.session import (
    MY_PICK,
    PICK,
    UNRESOLVED,
    Session,
    snake_seat,
)
from src.app.web import schemas
from src.app.web.resolve import (
    AMBIGUOUS,
    RESOLVED,
    Resolution,
    build_spine,
    resolve_name,
)
from src.app.web.sources.base import DraftEvent, DraftEventSource, SourceStatus
from src.app.web.sources.manual import ManualSource
from src.app.web.sources.replay import PICK_KINDS, ReplaySource
from src.app.web.sources.yahoo import YahooSource
from src.core.errors import DataError
from src.engine.decision.board import Board
from src.engine.decision.roster_state import RosterState
from src.engine.sim.bundle_build import build_projection_bundle

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


#: Suffix `archive_session` gives a set-aside draft log.
ARCHIVE_SUFFIX = ".bak"


def archive_paths(web_cfg) -> list[Path]:
    """Archived draft logs, newest first."""
    session_path = web_cfg.resolved(web_cfg.session_path)
    pattern = f"{session_path.stem}.*{ARCHIVE_SUFFIX}"
    return sorted(session_path.parent.glob(pattern),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def describe_archive(path: Path) -> dict:
    """One archived draft, summarized for the replay picker.

    `started_at` comes from the LOG when it has one, falling back to the file
    mtime. The two differ for a draft that ran for an hour, and the question
    the picker answers is "which draft was this", so when it started beats
    when it was filed away.
    """
    stat = path.stat()
    out = {
        "id": path.name,
        "archived_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(
            timespec="seconds"),
        "started_at": None, "picks": 0, "seat": None,
        "snapshot_id": None, "session_id": None, "readable": False,
    }
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return out
    events = raw.get("events", []) if isinstance(raw, dict) else []
    out.update(
        readable=True,
        started_at=raw.get("started_at") or out["archived_at"],
        session_id=raw.get("session_id"),
        snapshot_id=raw.get("snapshot_id"),
        seat=raw.get("my_seat"),
        picks=sum(1 for e in events
                  if str(e.get("kind", "")).lower() in PICK_KINDS),
    )
    return out


def list_archives(web_cfg) -> list[dict]:
    return [describe_archive(p) for p in archive_paths(web_cfg)]


def resolve_archive(web_cfg, archive_id: str) -> Path:
    """`archive_id` -> path, refusing anything outside the session directory.

    The id arrives from a client, so it is matched against the DIRECTORY
    LISTING rather than joined onto a path. Joining would let `../../etc`
    through, and this endpoint reads whatever it is handed.
    """
    for path in archive_paths(web_cfg):
        if path.name == archive_id:
            return path
    raise DataError(f"no archived draft named {archive_id!r}")


def make_source(name: str, *, web_cfg, strategy,
                archive_id: str | None = None) -> DraftEventSource:
    if name == "manual":
        return ManualSource()
    if name == "replay":
        if archive_id:
            return ReplaySource(path=resolve_archive(web_cfg, archive_id))
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


def _is_operator(event) -> bool:
    """Did a human enter this, or did a feed report it?

    `POST /api/picks` tags every event `source="manual"`. Anything else came
    from a poller that cannot disambiguate a name or judge whether a pick
    really happened.
    """
    return (event.source or "manual") == "manual"


class DuplicatePick(DataError):
    """The same pick reported twice.

    A subclass of `DataError` deliberately: the poll loop already drops
    `DataError` from a source without backing off, and that behaviour is
    exactly right for a feed echoing a pick the operator typed.

    What it adds is a way for the HTTP layer to tell "this pick is already
    recorded" apart from "this request is wrong". Both directions of the race
    are benign — operator first then feed, or feed first then operator
    clicking a row the UI has not yet dropped — and neither should surface as
    an error on a pick clock.
    """

    def __init__(self, message: str, player_id: str) -> None:
        super().__init__(message)
        self.player_id = player_id


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

    def start_session(self, *, seat: int, source: str, resume: bool,
                      archive_id: str | None = None) -> Session:
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
                                  strategy=self.strategy,
                                  archive_id=archive_id)
        self.source.start()
        self.generation += 1
        self.recommendation = RecommendationState()
        self._warm_projection_bundle()
        return self.session

    def _warm_projection_bundle(self) -> None:
        """Pay the network cost at SETUP, never on the pick clock.

        `build_projection_bundle` reaches nflverse for hazard covariates and
        took 61.85 s on the live board — against a 25 s allocator budget, so
        the first tier-0 recommendation blew its deadline before evaluating a
        single candidate and answered from two draws instead of fifty. It was
        not merely slow: the cold run returned a DIFFERENT leader than the
        warm one, because it had no simulation behind it.

        The bundle is memoized on the full board, which does not change during
        a draft, so doing it once here makes every pick warm. Failures are
        swallowed on purpose — this is a head start, and a draft must still
        start if nflverse is unreachable.
        """
        try:
            weeks = (self.cfg.schedule.regular_season_weeks
                     + len(self.cfg.schedule.playoff_weeks))
            frame = self.board.players.reset_index(drop=True).copy()
            frame["player_id"] = frame["player_id"].astype(str)
            started = time.perf_counter()
            build_projection_bundle(frame, self.cfg, weeks)
            logger.info("projection bundle warmed in %.1fs",
                        time.perf_counter() - started)
        except Exception as exc:                  # noqa: BLE001
            logger.warning("bundle warm-up skipped (%s); the first "
                           "recommendation will pay for it", exc)

    def archive_session(self, *, purge: bool = False) -> Path | None:
        """Clear the active draft. Archives by default, deletes on `purge`.

        Two paths because they are not the same act. Archiving renames the log
        aside and is recoverable; purging is not, so it is a separate request
        the UI has to ask for deliberately.

        **Archiving never touches the ledger; purging does.** Archiving is
        "put this aside", so the record of what the engine said must survive
        it — that is the whole reason the ledger is append-only and
        hash-chained. Purging is "this draft did not happen", and leaving its
        decisions behind would mean a ledger describing a draft with no log,
        which is worse than either keeping both or removing both.

        Only the ACTIVE draft can be purged, and it is by construction the
        newest in the ledger, so its entries are a contiguous suffix and the
        chain that remains still verifies from genesis. `delete_session`
        refuses anything else.
        """
        path = self.web_cfg.resolved(self.web_cfg.session_path)
        archived = None
        if purge and self.ledger is not None and self.session is not None:
            try:
                removed = self.ledger.delete_session(self.session.session_id)
                if removed:
                    logger.info("purged %d ledger entries for session %s",
                                removed, self.session.session_id)
            except Exception as exc:              # noqa: BLE001
                # A ledger that refuses to lose its chain is doing its job;
                # never let that block clearing the screen.
                logger.warning("ledger purge skipped: %s", exc)
        if path.exists():
            if purge:
                path.unlink()
            else:
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
    def _dup(self, player_id: str, who: str) -> DuplicatePick:
        return DuplicatePick(
            f"{who} is already drafted; ignoring duplicate", player_id)

    def resolve(self, text: str) -> Resolution:
        drafted = set(self.session.state.drafted) if self.session else set()
        return resolve_name(text, self._spine, self.players, exclude=drafted)

    def _names_someone_already_drafted(self, text: str) -> str | None:
        """The player id this text names, IF he is already off the board.

        `resolve` excludes drafted players on purpose — late in a draft
        "Jones" should not offer three men who are gone. But that exclusion
        turns a duplicate report into an UNRESOLVED one, and an unresolved
        pick still advances the clock. With a live feed echoing picks the
        operator has already typed, every echo would burn a pick slot: the
        draft runs out of picks at half distance and every seat attribution
        after the first is wrong.

        So duplicates are identified BEFORE unresolved is recorded, by
        resolving against the full board with nothing excluded.
        """
        if not self.session or not (text or "").strip():
            return None
        hit = resolve_name(text, self._spine, self.players, exclude=set())
        if hit.status == RESOLVED and hit.player_id in self.session.state.drafted:
            return str(hit.player_id)
        return None

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
                # A FEED cannot choose between candidates, and the poll loop
                # discards this return value — so an ambiguous name from Yahoo
                # was recorded nowhere while its pick number went into the
                # source's `_seen`. The pick vanished, permanently, with no
                # symptom. Refuse it loudly so the supervisor logs it and the
                # operator (who can tell two Robinsons apart) enters it.
                if not _is_operator(event):
                    raise DataError(
                        f"{event.raw_name!r} from {event.source or 'feed'} is "
                        f"ambiguous; enter this pick by hand")
                return AMBIGUOUS
        if player_id is None:
            # A feed echoing a pick already entered by hand. Refuse it as the
            # duplicate it is; the poll loop drops DataError without stopping.
            duplicate = self._names_someone_already_drafted(event.raw_name)
            if duplicate is not None:
                raise self._dup(duplicate, repr(event.raw_name))
            # An UNRESOLVED entry consumes a pick slot — `pick_number` is
            # `len(drafted) + len(unresolved) + 1` — and there is no targeted
            # way to repair one; only `undo`. So when a feed reports a name
            # this board cannot match, the operator clicks the right player a
            # moment later and ONE real pick has consumed TWO slots, silently
            # desynchronising the draft from that point on.
            #
            # The operator is the authoritative writer; the feed is a helper.
            # A name the feed cannot land is the feed's problem to report, not
            # a pick to invent.
            if not _is_operator(event):
                raise DataError(
                    f"{event.raw_name!r} from {event.source or 'feed'} matches "
                    f"nobody on the board; enter this pick by hand")
            session.record_unresolved(event.raw_name)
            self._bump()
            return "unresolved"

        if player_id in session.state.drafted:
            # The SAME pick reported twice — either the feed echoing what was
            # typed, or the operator clicking a row the feed already recorded
            # and the UI has not dropped yet. The desired state already holds,
            # so this is not a failure in either direction.
            raise self._dup(player_id, player_id)

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
        # The source dedupes on what it has already emitted, so an undone
        # pick would otherwise never be re-reported and the feed would stay
        # permanently one pick behind the board.
        source = getattr(self, "source", None)
        if source is not None:
            try:
                source.forget()
            except Exception:                          # noqa: BLE001
                logger.warning("source.forget() failed", exc_info=True)
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
                self.session.session_id,
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
    def roster_slots(self, positions: dict[str, str],
                     names: dict[str, str]) -> list[dict]:
        """Every starting slot in league order, filled or empty.

        The UI shows the shape of the roster, not just what is on it — an
        empty TE slot in round 12 is the single most useful thing on the
        screen, and a list of drafted players cannot show it.

        Assignment reuses `RosterState._assign_slot`'s rule rather than
        reimplementing it: pure position first, then FLEX if eligible, then
        bench. Two different answers to "which slot is this player in" is the
        same class of split-brain that produced the K/DST bug.
        """
        state = self._require_session().state
        rs = RosterState(cfg=self.cfg, draft_position=self.session.my_seat + 1)
        placed: dict[str, list[str]] = {}
        bench: list[str] = []

        for pid in state.my_roster:
            position = positions.get(pid, "")
            before = dict(rs.slot_fill)
            rs._assign_slot(position)
            moved = next((s for s, n in rs.slot_fill.items()
                          if n > before.get(s, 0)), "BENCH")
            if moved == "BENCH":
                bench.append(pid)
            else:
                placed.setdefault(moved, []).append(pid)

        def entry(slot: str, pid: str | None) -> dict:
            return {"slot": slot, "player_id": pid,
                    "name": names.get(pid) if pid else None,
                    "position": positions.get(pid) if pid else None}

        out: list[dict] = []
        for slot, count in self.cfg.roster.slots.items():
            if slot == "BENCH":
                continue
            filled = placed.get(slot, [])
            for i in range(int(count)):
                out.append(entry(slot, filled[i] if i < len(filled) else None))

        # Bench slots ALWAYS render, filled or not — same reason the starters
        # do. "Four bench spots left" is a real constraint late in a draft, and
        # a rail that only shows what you already have cannot express it.
        # Overflow lands here: a third WR fills FLEX (a starting slot that
        # takes RB/WR/TE), the fourth benches.
        bench_size = int(self.cfg.roster.slots.get("BENCH", 0))
        for i in range(max(bench_size, len(bench))):
            out.append(entry("BENCH", bench[i] if i < len(bench) else None))
        return out

    def draft_grid(self, positions: dict[str, str],
                   names: dict[str, str]) -> list[dict]:
        """Every pick made so far, with the seat and round it landed in.

        Rebuilt from the EVENT LOG rather than from `state.drafted`, because an
        unresolved pick consumes a slot without adding a drafted id. Deriving
        the grid from `drafted` alone would shift every later pick one column
        left the moment a name failed to resolve — the same off-by-one the
        clock fix exists to prevent, showing up again in the display.
        """
        session = self._require_session()
        teams = self.cfg.teams
        out: list[dict] = []
        n = 0
        for event in session.events:
            if event.kind not in (PICK, MY_PICK, UNRESOLVED):
                continue
            n += 1
            seat = (event.seat if event.seat is not None
                    else snake_seat(n, teams))
            pid = event.player_id or None
            out.append({
                "pick_number": n,
                "round": (n - 1) // teams + 1,
                "seat": seat,
                "player_id": pid,
                "name": (names.get(pid, pid) if pid
                         else (event.raw_input or "unresolved")),
                "position": positions.get(pid) if pid else None,
                "resolved": pid is not None,
                "is_mine": seat == session.my_seat,
            })
        return out

    def team_names(self) -> list[str]:
        """Configured names, padded or truncated to the league size.

        Generic until the real draft order is known — a placeholder that is
        obviously a placeholder beats a wrong name that looks right.
        """
        configured = list(getattr(self.web_cfg, "team_names", []) or [])
        return [configured[i] if i < len(configured) else f"Team {i + 1}"
                for i in range(self.cfg.teams)]

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
            # Identifies THIS draft, not the bundle. The client keys per-draft
            # local state on it (the notes pad), so starting a new draft does
            # not inherit the last one's scribbles.
            "session_id": session.session_id,
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
            "roster_slots": self.roster_slots(positions, names),
            "picks": self.draft_grid(positions, names),
            "team_names": self.team_names(),
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
            # What the HUMAN reads. `adp` above is the single-platform board
            # the engine models; this is every platform averaged. Falls back
            # to `adp` for the players no export matched.
            "adp_consensus": schemas._json_float(
                getattr(r, "adp_consensus", None)) or
                schemas._json_float(r.adp),
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
        # `current_pick` drives the confidence score's reach term: a
        # recommendation 20 picks before the market takes him is a coin flip
        # by measurement, however sure the simulation is of itself.
        pick = (self.session.state.pick_number
                if self.session is not None else None)
        return schemas.recommendation_payload(
            rec, self.players, snapshot_id=self.board.snapshot_id,
            generation=generation, reps=self.web_cfg.engine.reps,
            shortlist=self.shortlist,
            budget_seconds=self.web_cfg.engine.budget_seconds,
            current_pick=pick)

    def narrate(self, record):
        from src.app.narration import NarrationConfig
        from src.app.narration import narrate as _narrate
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
