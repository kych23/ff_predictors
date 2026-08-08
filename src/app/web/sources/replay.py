"""Replay a recorded draft. Drives the tests and the browser walkthrough.

The log format is exactly what ``Session.save`` writes, so any real or
rehearsed draft is replayable with no conversion step.

**It emits `raw_name` and leaves `player_id` unset on purpose.** Replaying
already-resolved ids would skip the identity cascade entirely — and name
resolution is the subsystem most likely to break under a live feed, since Yahoo
reports whatever string the league office typed. A replay that never exercises
it would be a test that passes for the wrong reason.

One event per `poll()`. `speed` is accepted and stored for a future wall-clock
mode but is not used to sleep: a replay that paced itself in real time would
make a 180-pick test take an hour.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.app.cockpit.session import MY_PICK, PICK
from src.app.web.sources.base import DraftEvent, SourceStatus, _QueueSource
from src.core.errors import DataError

#: Event kinds in a Session log that represent a pick, taken FROM the session
#: module rather than spelled again here.
#:
#: This was written out as ``("PICK", "MY_PICK")`` — uppercase — while
#: `Session` emits ``"pick"`` and ``"my_pick"``. Nothing raised: every event in
#: a real saved draft simply failed the membership test, so replaying an actual
#: session produced an EMPTY queue and a draft that never advanced. The
#: checked-in fixture happened to be written in uppercase, so the tests passed
#: while the documented feature ("any real or rehearsed draft is replayable
#: with no conversion step") did not work at all.
PICK_KINDS = (PICK, MY_PICK)


@dataclass
class ReplaySource(_QueueSource):
    path: Path = Path("")
    speed: float = 1.0
    name: str = "replay"
    _queue: list[DraftEvent] = field(default_factory=list, repr=False)
    _cursor: int = 0

    def start(self) -> None:
        self._queue = list(self._load(self.path))
        self._cursor = 0
        self._status = SourceStatus()

    def stop(self) -> None:
        self._cursor = len(self._queue)

    def poll(self) -> list[DraftEvent]:
        """One event per call. Returning the whole log at once would apply 180
        picks between two recommendations and make the replay useless as a
        walkthrough."""
        if self._cursor >= len(self._queue):
            return []
        event = self._queue[self._cursor]
        self._cursor += 1
        return [event]

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._queue)

    @property
    def remaining(self) -> int:
        return max(0, len(self._queue) - self._cursor)

    @staticmethod
    def _load(path: Path) -> list[DraftEvent]:
        p = Path(path)
        if not p.exists():
            raise DataError(f"no replay log at {p}")
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise DataError(f"replay log at {p} is not valid JSON: {exc}") from exc

        events = raw.get("events", []) if isinstance(raw, dict) else raw
        out: list[DraftEvent] = []
        for i, ev in enumerate(events):
            # Case-insensitive so the checked-in fixture (written uppercase)
            # and a real `Session.save` log (lowercase) both replay.
            kind = str(ev.get("kind", "")).strip().lower()
            if kind not in PICK_KINDS:
                continue
            # `raw_input` is dropped from the log when empty (Event.to_dict
            # omits falsy fields), so fall back to the id. The id is used only
            # as a NAME here — the service still resolves it.
            name = str(ev.get("raw_input") or ev.get("player_id") or "").strip()
            if not name:
                continue
            out.append(DraftEvent(
                raw_name=name,
                seat=ev.get("seat"),
                player_id=None,           # deliberately unresolved; see module docstring
                source="replay",
                observed_at=str(ev.get("at", "")),
                external_pick_number=i + 1,
            ))
        return out
