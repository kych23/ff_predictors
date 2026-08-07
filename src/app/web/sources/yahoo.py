"""Poll a live Yahoo draft (§11.5). **Ships disabled.**

There are no Yahoo credentials in this project and no granted Fantasy Sports
API permission: `.env` carries no `YAHOO_CLIENT_ID`/`YAHOO_CLIENT_SECRET`,
there is no `.yahoo_token.json`, and `config/strategy.yaml` still has
`yahoo.league_key: null` with `game_keys: {}` marked "populated once Fantasy
Sports permission is granted".

So this adapter is written to the interface and **not validated against the
live API**. Its shipped behaviour is to report `failed` with the reason. That
is deliberate: an adapter that raised on the poll path would take down a
cockpit whose manual fallback works perfectly well.

Two things are known-unfinished and are reported rather than guessed:

* **Seat mapping.** `strategy.yaml` documents `yahoo.manager_map` as
  ``team_key -> stable manager id``, and nothing in the repo maps a manager id
  to a draft seat (`ManagerConfig` has no seat field). Until a real mapping
  exists, values are treated as seat indices and validated at `start()`, which
  fails loudly instead of silently drafting to the wrong team.
* **Response shape.** Community docs say `draftresults` is unpaginated, but that
  is unverified here.

`authlib` is imported lazily inside `start()` so a missing or broken install
degrades this adapter alone.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from src.app.web.sources.base import DraftEvent, SourceStatus, _QueueSource

BASE_URL = "https://fantasysports.yahooapis.com/fantasy/v2"
MAX_BACKOFF_SECONDS = 30.0


@dataclass
class YahooSource(_QueueSource):
    league_key: str | None = None
    token_path: Path = Path(".yahoo_token.json")
    interval_s: float = 3.0
    manager_map: Mapping[str, Any] = field(default_factory=dict)
    name: str = "yahoo"

    _client: Any = field(default=None, repr=False)
    _seen: set[int] = field(default_factory=set, repr=False)
    _backoff: float = 0.0
    _next_poll: float = 0.0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Report why it cannot run rather than raising."""
        if not self.league_key:
            self._status = SourceStatus.failed(
                "no league_key; run scripts/spikes/probe_yahoo.py once Fantasy "
                "Sports permission is granted")
            return
        if not Path(self.token_path).exists():
            self._status = SourceStatus.failed(
                f"no OAuth token at {self.token_path}; "
                f"run scripts/spikes/probe_yahoo.py")
            return

        seat_error = self._validate_seat_map()
        if seat_error:
            self._status = SourceStatus.failed(seat_error)
            return

        try:
            import json

            from authlib.integrations.httpx_client import OAuth2Client
        except ImportError as exc:                     # noqa: BLE001
            self._status = SourceStatus.failed(f"authlib unavailable: {exc}")
            return

        try:
            token = json.loads(Path(self.token_path).read_text())
            self._client = OAuth2Client(token=token)
            self._status = SourceStatus()
        except Exception as exc:                       # noqa: BLE001
            self._status = SourceStatus.failed(f"token load failed: {exc}")

    def stop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:                          # noqa: BLE001
                pass

    # ----------------------------------------------------------- seat map
    def _validate_seat_map(self) -> str | None:
        """`manager_map` is documented as team_key -> manager id, and no
        manager -> seat table exists anywhere in the repo. Treat the values as
        seat indices and say so plainly if they are not."""
        if not self.manager_map:
            return ("yahoo.manager_map is empty; cannot map team_key -> seat. "
                    "Populate it before enabling the yahoo source.")
        for key, value in self.manager_map.items():
            try:
                seat = int(value)
            except (TypeError, ValueError):
                return (f"yahoo.manager_map[{key!r}] = {value!r} is not a seat "
                        f"index. It is documented as a manager id, and no "
                        f"manager->seat mapping exists yet.")
            if seat < 0:
                return f"yahoo.manager_map[{key!r}] = {seat} is not a valid seat"
        return None

    # --------------------------------------------------------------- poll
    def poll(self) -> list[DraftEvent]:
        """Never raises. Network trouble becomes `degraded` plus backoff."""
        if self._client is None or self._status.state == "failed":
            return []
        now = time.monotonic()
        if now < self._next_poll:
            return []

        try:
            payload = self._fetch_draft_results()
        except Exception as exc:                       # noqa: BLE001
            self._backoff = min(
                MAX_BACKOFF_SECONDS, max(self.interval_s, self._backoff * 2 or 1.0))
            self._next_poll = now + self._backoff
            self._status = SourceStatus.degraded(
                f"{type(exc).__name__}: {exc}; backoff {self._backoff:.0f}s")
            return []

        self._backoff = 0.0
        self._next_poll = now + self.interval_s
        self._status = SourceStatus()
        return self._to_events(payload)

    def _fetch_draft_results(self) -> Any:
        url = f"{BASE_URL}/league/{self.league_key}/draftresults?format=json"
        response = self._client.get(url, timeout=10.0)
        response.raise_for_status()
        return response.json()

    def _to_events(self, payload: Any) -> list[DraftEvent]:
        """Emit only picks not previously returned, in pick order."""
        out: list[DraftEvent] = []
        for pick in _iter_draft_results(payload):
            number = pick.get("pick")
            if number is None or number in self._seen:
                continue
            self._seen.add(int(number))
            team_key = str(pick.get("team_key", ""))
            seat = self.manager_map.get(team_key)
            out.append(DraftEvent(
                raw_name=str(pick.get("player_key", "")),
                seat=int(seat) if seat is not None else None,
                player_id=None,        # resolved by the service, like every source
                source=self.name,
                observed_at="",
                external_pick_number=int(number),
            ))
        return sorted(out, key=lambda e: e.external_pick_number or 0)


def _iter_draft_results(payload: Any):
    """Yahoo's JSON is deeply nested and numerically keyed. Walk it defensively
    rather than indexing a shape nobody here has seen a real response from."""
    if not isinstance(payload, Mapping):
        return
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            if "draft_result" in node and isinstance(node["draft_result"], Mapping):
                yield node["draft_result"]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
