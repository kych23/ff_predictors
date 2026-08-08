"""Event-sourced draft session (§19) — the draft-night loop.

**State is a pure function of the event history.** Every command appends an
event and the board is rebuilt by replay. That is not architectural taste: it
makes `undo` a one-line pop instead of a per-command inverse, and per-command
inverses are exactly the thing that breaks under pressure. At pick 47, with
eleven people waiting and someone having just said a name wrong, the ability to
undo without thinking is worth more than any modelling layer.

The same property makes the session crash-safe. The history is written after
every command, so recovery is replay, and replay is the code path already
exercised by every `undo` in every rehearsal.

Distinct from `api/draft_service.py`'s `DraftSession` (§19) — they share no
storage and no code. That one is a frozen v1 web surface; this is the CLI.

Nothing here touches the network. Name resolution is the local `Spine`, the
board is a parquet bundle, the ledger is local SQLite, and narration is a local
model. §19's offline requirement is a property of the imports, not a promise.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.errors import DataError

#: Event kinds. Additive only — a history written by an older build must
#: replay under a newer one, because the recovery path IS replay.
PICK = "pick"
MY_PICK = "my_pick"
ZERO = "zero"
ADP_OVERRIDE = "adp_override"
UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Event:
    kind: str
    player_id: str = ""
    seat: int | None = None
    value: float | None = None
    raw_input: str = ""
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in ("", None)} \
            | {"kind": self.kind}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Event:
        return cls(
            kind=str(raw["kind"]), player_id=str(raw.get("player_id", "")),
            seat=raw.get("seat"), value=raw.get("value"),
            raw_input=str(raw.get("raw_input", "")), at=str(raw.get("at", "")),
        )


@dataclass
class DraftState:
    """Rebuilt from scratch on every replay. Never mutated in place by a
    command — only by `_replay`."""

    drafted: list[str] = field(default_factory=list)
    my_roster: list[str] = field(default_factory=list)
    by_seat: dict[int, list[str]] = field(default_factory=dict)
    zeroed: set[str] = field(default_factory=set)
    adp_overrides: dict[str, float] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)

    @property
    def pick_number(self) -> int:
        """1-indexed pick currently on the clock.

        Unresolved picks count. A name nobody could resolve is still a pick
        that happened: the room moved on, and the next seat is up. Counting
        only `drafted` left the clock — and therefore `on_the_clock`,
        `is_my_turn`, `next_my_pick` and `describe_clock` — short by one for
        the REST OF THE DRAFT after a single unresolvable name. A human typist
        hits that rarely; a live feed hits it often.

        The unresolved player is deliberately NOT given a placeholder id in
        `drafted`: `Board.take` rejects ids that are not on the board, and a
        sentinel would leak into `available()`, `by_seat` and the ledger.
        """
        return len(self.drafted) + len(self.unresolved) + 1


def snake_seat(pick_number: int, teams: int) -> int:
    """0-indexed seat on the clock at a 1-indexed overall pick."""
    if pick_number < 1:
        raise ValueError(f"pick_number is 1-indexed, got {pick_number}")
    rnd, index = divmod(pick_number - 1, teams)
    return index if rnd % 2 == 0 else teams - 1 - index


def my_pick_numbers(seat: int, teams: int, rounds: int) -> list[int]:
    """Every overall pick belonging to `seat` (0-indexed)."""
    return [p for p in range(1, teams * rounds + 1)
            if snake_seat(p, teams) == seat]


class Session:
    """Append-only event log plus the state it replays to."""

    def __init__(self, *, my_seat: int, teams: int, rounds: int,
                 snapshot_id: str = "", path: Path | None = None,
                 session_id: str = "", started_at: str = "") -> None:
        if not 0 <= my_seat < teams:
            raise ValueError(
                f"seat {my_seat} is outside a {teams}-team league (0-indexed)")
        #: Identifies THIS draft. Distinct from `snapshot_id`, which identifies
        #: the bundle and is shared by every draft run against it — the ledger
        #: was being keyed on the snapshot, so two drafts on the same bundle
        #: were indistinguishable in the record and deleting one would take
        #: the other with it.
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.started_at = started_at or _now()
        self.my_seat = my_seat
        self.teams = teams
        self.rounds = rounds
        self.snapshot_id = snapshot_id
        self.path = Path(path) if path else None
        self.events: list[Event] = []
        self.state = DraftState()

    # ----------------------------------------------------------- replay
    def _replay(self) -> DraftState:
        state = DraftState()
        for event in self.events:
            if event.kind in (PICK, MY_PICK):
                if event.player_id in state.drafted:
                    # A duplicate name is a mis-entry, not a second pick. Two
                    # seats cannot hold the same player, and silently allowing
                    # it desynchronises every downstream count.
                    raise DataError(
                        f"{event.player_id} was already drafted; the history "
                        f"is inconsistent")
                state.drafted.append(event.player_id)
                seat = event.seat if event.seat is not None else 0
                state.by_seat.setdefault(seat, []).append(event.player_id)
                if event.kind == MY_PICK:
                    state.my_roster.append(event.player_id)
            elif event.kind == ZERO:
                state.zeroed.add(event.player_id)
            elif event.kind == ADP_OVERRIDE:
                state.adp_overrides[event.player_id] = float(event.value or 0.0)
            elif event.kind == UNRESOLVED:
                state.unresolved.append(event.raw_input)
        return state

    def _append(self, event: Event) -> DraftState:
        self.events.append(event)
        try:
            self.state = self._replay()
        except DataError:
            self.events.pop()          # never leave the log in a bad state
            raise
        self.save()
        return self.state

    # --------------------------------------------------------- commands
    def record_pick(self, player_id: str, *, seat: int | None = None,
                    raw_input: str = "") -> DraftState:
        """A pick by whoever is on the clock, unless a seat is given."""
        on_clock = self.state.pick_number
        resolved = snake_seat(on_clock, self.teams) if seat is None else seat
        kind = MY_PICK if resolved == self.my_seat else PICK
        return self._append(Event(kind=kind, player_id=player_id,
                                  seat=resolved, raw_input=raw_input,
                                  at=_now()))

    def record_my_pick(self, player_id: str, raw_input: str = "") -> DraftState:
        return self._append(Event(kind=MY_PICK, player_id=player_id,
                                  seat=self.my_seat, raw_input=raw_input,
                                  at=_now()))

    def zero_player(self, player_id: str) -> DraftState:
        """Injury/news override: the player is worthless, without a restart."""
        return self._append(Event(kind=ZERO, player_id=player_id, at=_now()))

    def override_adp(self, player_id: str, adp: float) -> DraftState:
        return self._append(Event(kind=ADP_OVERRIDE, player_id=player_id,
                                  value=float(adp), at=_now()))

    def record_unresolved(self, raw_input: str) -> DraftState:
        """A name the spine could not resolve. Recorded rather than dropped —
        the pick still happened, and a board that thinks a player is available
        when he is gone will keep recommending him."""
        return self._append(Event(kind=UNRESOLVED, raw_input=raw_input,
                                  at=_now()))

    def undo(self) -> DraftState:
        """Pop the last event and replay. The whole reason for event sourcing.

        Per-command inverses are what break under pressure — an inverse for
        `zero` has to remember whether the player was already zeroed, and
        nobody gets that right at pick 47.
        """
        if not self.events:
            raise DataError("nothing to undo")
        self.events.pop()
        self.state = self._replay()
        self.save()
        return self.state

    # ------------------------------------------------------------ clock
    @property
    def my_picks(self) -> list[int]:
        return my_pick_numbers(self.my_seat, self.teams, self.rounds)

    def on_the_clock(self) -> int:
        return snake_seat(self.state.pick_number, self.teams)

    def is_my_turn(self) -> bool:
        return self.on_the_clock() == self.my_seat

    def next_my_pick(self) -> int | None:
        return next((p for p in self.my_picks if p >= self.state.pick_number),
                    None)

    def picks_until_my_turn(self) -> int | None:
        nxt = self.next_my_pick()
        return None if nxt is None else nxt - self.state.pick_number

    @property
    def is_complete(self) -> bool:
        return self.state.pick_number > self.teams * self.rounds

    def describe_clock(self) -> str:
        if self.is_complete:
            return "draft complete"
        rnd = (self.state.pick_number - 1) // self.teams + 1
        away = self.picks_until_my_turn()
        who = "YOU" if self.is_my_turn() else f"seat {self.on_the_clock() + 1}"
        tail = "" if away is None else (
            " — you are up" if away == 0 else f" — {away} until your pick")
        return (f"round {rnd}, pick {self.state.pick_number} of "
                f"{self.teams * self.rounds}, {who} on the clock{tail}")

    # ------------------------------------------------------- durability
    def save(self) -> None:
        """Written after EVERY command. Recovery is replay, and replay is the
        path every `undo` already exercises."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id, "started_at": self.started_at,
            "my_seat": self.my_seat, "teams": self.teams,
            "rounds": self.rounds, "snapshot_id": self.snapshot_id,
            "events": [e.to_dict() for e in self.events],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)      # atomic; a crash mid-write cannot truncate

    @classmethod
    def load(cls, path: Path) -> Session:
        path = Path(path)
        if not path.exists():
            raise DataError(f"no session at {path}")
        raw = json.loads(path.read_text())
        session = cls(my_seat=int(raw["my_seat"]), teams=int(raw["teams"]),
                      rounds=int(raw["rounds"]),
                      snapshot_id=str(raw.get("snapshot_id", "")), path=path,
                      # Logs written before session_id existed get one on load
                      # so a resumed old draft still has a stable identity.
                      session_id=str(raw.get("session_id", "")),
                      started_at=str(raw.get("started_at", "")))
        session.events = [Event.from_dict(e) for e in raw.get("events", [])]
        session.state = session._replay()
        return session

    def resume_or_new(self) -> Session:
        if self.path is not None and self.path.exists():
            return Session.load(self.path)
        return self


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def available(board_ids: Iterable[str], state: DraftState) -> list[str]:
    """Board minus everyone drafted. Zeroed players stay AVAILABLE — a player
    ruled out by news is still draftable by someone else, and removing him
    would make the opponent model think he is gone."""
    gone = set(state.drafted)
    return [p for p in board_ids if p not in gone]
