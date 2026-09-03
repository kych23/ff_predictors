"""Where picks come from (§11.5, §19).

The draft cockpit needs to learn that a pick happened. It can learn it three
ways — the operator types it, a recorded log replays it, or a league API
reports it — and the recommendation path should not care which. That is the
whole reason this protocol exists: Yahoo has no credentials and no granted
Fantasy Sports permission, so the adapter that matters most cannot be built or
validated yet. Putting it behind an interface means the cockpit is complete and
testable now, and Yahoo becomes a config flip rather than a rewrite.

**`external_pick_number` is advisory and is never trusted.** `Session` derives
ordering from its own event log; a source's numbering is recorded so a
disagreement can be diagnosed after the fact, not so it can place a pick. A
feed that batches or reorders must not be able to corrupt the snake.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

OK = "ok"
DEGRADED = "degraded"
FAILED = "failed"


@dataclass(frozen=True)
class DraftEvent:
    """One pick, as reported by a source.

    ``player_id`` is None when the name could not be resolved. That is a normal
    outcome, not an error: an unidentified pick still consumed a slot, and the
    session records it so the clock stays correct.
    """

    raw_name: str
    seat: int | None = None
    player_id: str | None = None
    source: str = ""
    observed_at: str = ""
    external_pick_number: int | None = None

    @property
    def resolved(self) -> bool:
        return self.player_id is not None


@dataclass(frozen=True)
class SourceStatus:
    """What the UI's connection chip shows.

    A source reports trouble rather than raising it. A draft does not stop
    because a feed hiccuped — the operator types the pick instead — so an
    adapter that throws on the poll path would take down a tool that had a
    perfectly good fallback.
    """

    state: str = OK
    detail: str = ""

    @property
    def healthy(self) -> bool:
        return self.state == OK

    @classmethod
    def failed(cls, detail: str) -> SourceStatus:
        return cls(state=FAILED, detail=detail)

    @classmethod
    def degraded(cls, detail: str) -> SourceStatus:
        return cls(state=DEGRADED, detail=detail)


@runtime_checkable
class DraftEventSource(Protocol):
    """Every adapter implements exactly this."""

    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def forget(self) -> None:
        """Drop the record of what has already been emitted.

        Called after an undo. A polling source dedupes by pick number, so
        without this an undone pick is never re-reported and the feed is
        permanently one pick behind the truth.
        """

    def poll(self) -> list[DraftEvent]:
        """Events not previously returned. Must never raise."""

    @property
    def status(self) -> SourceStatus: ...


@dataclass
class _QueueSource:
    """Shared base for adapters that hand back a buffered list.

    Kept concrete rather than abstract because `poll()` draining exactly once
    is the part every adapter gets wrong, and there is no reason for three
    copies of it.
    """

    name: str = "queue"
    _pending: list[DraftEvent] = field(default_factory=list, repr=False)
    _status: SourceStatus = field(default_factory=SourceStatus, repr=False)

    def start(self) -> None:
        self._status = SourceStatus()

    def stop(self) -> None:
        self._pending.clear()

    def forget(self) -> None:
        """No-op: a queue source hands over its buffer and keeps no cursor, so
        there is nothing an undo could make it re-report."""

    def poll(self) -> list[DraftEvent]:
        out, self._pending = self._pending, []
        return out

    @property
    def status(self) -> SourceStatus:
        return self._status
