"""The operator types the pick. Always available.

This adapter is the default source, and the reason the Yahoo failure modes are
survivable: `POST /api/picks` applies a pick DIRECTLY through
`CockpitService.apply_event`, whatever source the session was created with, so
"the feed died, type it yourself" is a real path and never waits for a poll
interval.

`submit()`/`poll()` therefore exist for the source protocol and for tests, not
for the HTTP route. Anything built on top of this queue expecting it to mirror
picks made over HTTP will see an empty queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.app.web.sources.base import DraftEvent, SourceStatus, _QueueSource
from src.platform.store.manifest import utc_now


@dataclass
class ManualSource(_QueueSource):
    name: str = "manual"
    _status: SourceStatus = field(default_factory=SourceStatus, repr=False)

    def submit(self, raw_name: str, *, seat: int | None = None,
               player_id: str | None = None) -> DraftEvent:
        """Queue one operator-entered pick.

        Resolution happens in the service, not here: the adapter's job is to
        report what was said, and `player_id` is passed through only when the
        caller already resolved it (the ambiguity flow, where the operator
        picked a specific candidate).
        """
        event = DraftEvent(
            raw_name=raw_name.strip(), seat=seat, player_id=player_id,
            source=self.name, observed_at=utc_now(),
        )
        self._pending.append(event)
        return event

    def __len__(self) -> int:
        return len(self._pending)
