"""Board persistence: one JSON document per board, written atomically.

Three properties this file exists to guarantee.

**Containment.** `board_id` arrives from the client in the URL path on four
routes. `path_for` is the single choke point: an allowlist `fullmatch` plus a
positive assertion that the resolved parent IS the rankings directory. Both,
not either — the regex alone would be enough today, and the resolve check is
what keeps it true if the regex is ever loosened.

**Atomicity.** Writes go to a temp file in the same directory and land via
`os.replace`, which is atomic on POSIX. A crash mid-write therefore leaves the
previous board intact rather than a truncated one. The user's research is the
product here; a half-written board is worse than a stale one.

**Conflict detection that actually detects.** `rev` is a monotonic integer
compared under the same lock that performs the write. Comparing `updated_at`
instead cannot work: the client that just saved also just advanced the
timestamp, so it has no way to notice that another tab's edit went missing.

The lock is a `threading.Lock`, not an `asyncio.Lock`, because every call runs
inside `asyncio.to_thread` — the routes are `async def` (enforced by
`tests/test_web_api.py::test_every_route_is_async`) and must not do file I/O on
the loop that serves the pick clock. It assumes one uvicorn worker, which is
what this deployment is.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .schema import (
    BOARD_ID,
    Board,
    RankingsError,
    Scope,
    parse,
    slugify,
    to_json,
    utc_now,
)

logger = logging.getLogger(__name__)

#: `list()` reads every board on disk. A cap keeps a directory someone dumped
#: files into from turning a page load into a directory walk.
MAX_BOARDS = 100


class SlugExists(RankingsError):
    """A board with this slug is already on disk.

    Distinct from a generic validation error because it maps to 409, not 400:
    the request was well-formed, it just lost a race with an existing name.
    """


class RevConflict(RankingsError):
    """Someone else wrote this board since the client last read it.

    Carries the current board so the route can hand it back — the client cannot
    resolve the conflict without seeing what it lost to.
    """

    def __init__(self, current: Board, payload: dict | None = None) -> None:
        super().__init__(
            f"board {current.board_id!r} is at rev {current.rev}")
        self.current = current
        #: Rendered by whoever raises, on the worker thread. The route cannot
        #: build it: `board_payload` scans the bundle for stale ids, and doing
        #: that in the exception handler would put it back on the event loop.
        self.payload = payload if payload is not None else to_json(current)


class RankingsStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        #: Injected by the app so a rev conflict can render its response —
        #: including stale ids, which need the bundle — on the worker thread
        #: that raised, rather than in a route handler on the event loop.
        self.render: Callable[[Board], dict] | None = None

    # ------------------------------------------------------------- paths
    def path_for(self, board_id: str) -> Path:
        if not isinstance(board_id, str) or not BOARD_ID.fullmatch(board_id):
            raise RankingsError(f"bad board id {board_id!r}")
        path = (self.root / f"{board_id}.json").resolve()
        if path.parent != self.root.resolve():
            raise RankingsError("board id escapes the rankings directory")
        return path

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- read
    def load(self, board_id: str) -> Board:
        path = self.path_for(board_id)
        if not path.exists():
            raise FileNotFoundError(board_id)
        return parse(json.loads(path.read_text()))

    def list(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        out: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json"))[:MAX_BOARDS]:
            try:
                board = parse(json.loads(path.read_text()))
            except (RankingsError, json.JSONDecodeError, OSError):
                # One unreadable board must not hide the others. It stays on
                # disk and surfaces its real error when opened directly.
                logger.warning("skipping unreadable board %s", path.name)
                continue
            out.append({
                "board_id": board.board_id,
                "name": board.name,
                "updated_at": board.updated_at,
                "rev": board.rev,
                "counts": {scope: sum(len(t.player_ids) for t in s.tiers)
                           for scope, s in board.scopes.items()},
            })
        return out

    # ------------------------------------------------------------- write
    def _write(self, board: Board) -> None:
        """Temp file in the SAME directory, then `os.replace`.

        Same directory because `os.replace` is only atomic within a filesystem,
        and /tmp is often a different one.
        """
        self._ensure_root()
        # Validate on the way OUT as well as in. `_write_board` parses first,
        # but `/seed` builds its board straight from `seed_scope`, so without
        # this a duplicate id in the tier artifact would fan out through the
        # left-merge and persist a board that every later GET refuses to open.
        board = parse(to_json(board))
        path = self.path_for(board.board_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(to_json(board), indent=2) + "\n")
        os.replace(tmp, path)

    def create(self, name: str, scopes: Mapping[str, Scope],
               seeded_from: Mapping[str, Any]) -> Board:
        board_id = slugify(name)
        now = utc_now()
        board = Board(board_id=board_id, name=name, created_at=now,
                      updated_at=now, rev=1, seeded_from=dict(seeded_from),
                      scopes=dict(scopes))
        with self._lock:
            # Check and write under one lock, or two creates race into one
            # file and the loser's board silently becomes the winner's.
            if self.path_for(board_id).exists():
                raise SlugExists(
                    f"a board named {name!r} already exists (id {board_id!r})")
            self._write(board)
        return board

    def save(self, board: Board, *, expected_rev: int) -> Board:
        """Write `board` if nobody else has written since `expected_rev`.

        There is deliberately no force-overwrite path. A save that ignores the
        revision is a save that can erase an edit made in another tab, and this
        store holds the one thing in the app the user typed by hand.
        """
        with self._lock:
            path = self.path_for(board.board_id)
            if not path.exists():
                raise FileNotFoundError(board.board_id)
            current = parse(json.loads(path.read_text()))
            if current.rev != expected_rev:
                raise RevConflict(current, self._render(current))
            saved = board.with_rev(current.rev + 1)
            self._write(saved)
        return saved

    def _render(self, board: Board) -> dict:
        return self.render(board) if self.render else to_json(board)

    def delete(self, board_id: str) -> None:
        path = self.path_for(board_id)
        if not path.exists():
            raise FileNotFoundError(board_id)
        path.unlink()
