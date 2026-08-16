"""The board document: its shape, its validation, and its migration seam.

A board is one user's tier rankings across seven scopes. The invariants below
are enforced at `parse`, which is the only way a document enters the process —
so a malformed file on disk and a malformed request body fail identically.

**Scopes are independent.** Reordering `overall` does not touch `RB`. Two-way
sync between an overall list and seven position lists is the classic bug farm
in this kind of tool: a player who is RB3 in the position list and RB7 in
overall is a legitimate state during research, and forcing consistency destroys
the thing being reasoned about. The one bridge is an explicit, one-way
`from_overall` seed.

**A player appears at most once per scope, and a duplicate is refused rather
than de-duplicated.** Silently dropping one of two ranks loses the user's work
without telling them; refusing the save loses nothing.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = 1

SCOPES = ("overall", "QB", "RB", "WR", "TE", "K", "DST")
POSITION_SCOPES = SCOPES[1:]

#: An intensity ramp, NOT a hue set. `web/src/positions.ts` already spends
#: green/orange/blue/pink/slate on positions, and `PlayerRow` renders a position
#: chip inside a tier — a colour that means "running back" in one place and
#: "tier 2" two components over is worse than no colour at all.
TIER_COLORS = ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8")

#: Allowlist, applied with `fullmatch` to every client-supplied board id. A
#: denylist of "/", ".." and leading dots admits backslashes, NULs, over-length
#: names and case-collisions on macOS; this admits none of them.
BOARD_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")

#: Bounds. A board is a research document, not a database; these exist so a
#: malformed or hostile payload cannot make the store do unbounded work.
MAX_TIERS_PER_SCOPE = 200
MAX_PLAYERS_PER_TIER = 500
MAX_PLAYER_ID = 64
MAX_NAME = 80


class RankingsError(ValueError):
    """A board document is malformed. Always the client's fault or the disk's,
    never an internal error — the routes map it to 400 or 500 accordingly."""


@dataclass(frozen=True)
class Tier:
    id: str
    label: str
    color: str
    player_ids: tuple[str, ...]


@dataclass(frozen=True)
class Scope:
    tiers: tuple[Tier, ...] = ()


@dataclass(frozen=True)
class Board:
    board_id: str
    name: str
    created_at: str
    updated_at: str
    rev: int
    seeded_from: Mapping[str, Any]
    scopes: Mapping[str, Scope]
    schema_version: int = SCHEMA_VERSION

    def with_rev(self, rev: int) -> Board:
        return replace(self, rev=rev, updated_at=utc_now())

    def player_ids(self) -> set[str]:
        return {pid
                for scope in self.scopes.values()
                for tier in scope.tiers
                for pid in tier.player_ids}


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(name: str) -> str:
    """A file-safe board id derived from the display name.

    Truncation happens BEFORE the trailing-hyphen strip, so a cut landing in
    the middle of a separator cannot leave `my-very-long-board-` behind.
    """
    if not isinstance(name, str):
        raise RankingsError("board name must be a string")
    folded = unicodedata.normalize("NFKD", name)
    folded = folded.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", folded)[:64].strip("-")
    if not slug or not BOARD_ID.fullmatch(slug):
        raise RankingsError(f"board name {name!r} has no usable slug")
    return slug


def empty_scopes() -> dict[str, Scope]:
    return {scope: Scope() for scope in SCOPES}


# ----------------------------------------------------------------- parsing
def _str(raw: Mapping[str, Any], key: str, *, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise RankingsError(f"{where}: {key} must be a non-empty string")
    return value


def _parse_tier(raw: Any, *, where: str, seen: set[str]) -> Tier:
    if not isinstance(raw, Mapping):
        raise RankingsError(f"{where}: a tier must be a mapping")
    tier_id = _str(raw, "id", where=where)
    if tier_id in seen:
        raise RankingsError(f"{where}: duplicate tier id {tier_id!r}")
    seen.add(tier_id)

    color = raw.get("color", TIER_COLORS[0])
    if color not in TIER_COLORS:
        raise RankingsError(
            f"{where}: unknown colour {color!r}; expected one of {TIER_COLORS}")

    ids = raw.get("player_ids", [])
    if not isinstance(ids, (list, tuple)):
        raise RankingsError(f"{where}: player_ids must be a list")
    if len(ids) > MAX_PLAYERS_PER_TIER:
        raise RankingsError(
            f"{where}: {len(ids)} players in one tier exceeds "
            f"{MAX_PLAYERS_PER_TIER}")
    players: list[str] = []
    for pid in ids:
        if not isinstance(pid, str) or not pid:
            raise RankingsError(f"{where}: player ids must be non-empty strings")
        if len(pid) > MAX_PLAYER_ID:
            raise RankingsError(f"{where}: player id {pid[:20]!r}… is too long")
        players.append(pid)

    label = raw.get("label", "")
    if not isinstance(label, str):
        raise RankingsError(f"{where}: label must be a string")
    return Tier(id=tier_id, label=label, color=color,
                player_ids=tuple(players))


def _parse_scope(raw: Any, *, where: str) -> Scope:
    if raw is None:
        return Scope()
    if not isinstance(raw, Mapping):
        raise RankingsError(f"{where}: a scope must be a mapping")
    tiers_raw = raw.get("tiers", [])
    if not isinstance(tiers_raw, (list, tuple)):
        raise RankingsError(f"{where}: tiers must be a list")
    if len(tiers_raw) > MAX_TIERS_PER_SCOPE:
        raise RankingsError(
            f"{where}: {len(tiers_raw)} tiers exceeds {MAX_TIERS_PER_SCOPE}")

    seen_tier_ids: set[str] = set()
    tiers = tuple(_parse_tier(t, where=where, seen=seen_tier_ids)
                  for t in tiers_raw)

    # One rank per player per scope. Checked here rather than at write time so
    # a hand-edited file cannot smuggle a double entry past the API.
    seen_players: set[str] = set()
    for tier in tiers:
        for pid in tier.player_ids:
            if pid in seen_players:
                raise RankingsError(
                    f"{where}: player {pid!r} appears in more than one tier")
            seen_players.add(pid)
    return Scope(tiers=tiers)


def migrate(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Bring a stored document up to `SCHEMA_VERSION`.

    A missing `schema_version` is treated as 1, because hand-editing a board is
    a supported thing to do and the key is the least memorable part. A version
    from the FUTURE is refused rather than guessed at — a newer build may have
    given an existing field a new meaning, and silently reinterpreting the
    user's research is worse than declining to open it.
    """
    if not isinstance(raw, Mapping):
        raise RankingsError("a board must be a mapping")
    version = raw.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise RankingsError(f"schema_version must be an int, got {version!r}")
    if version > SCHEMA_VERSION:
        raise RankingsError(
            f"board is schema_version {version}; this build understands "
            f"{SCHEMA_VERSION}. Upgrade rather than opening it.")
    return raw


def parse(raw: Mapping[str, Any]) -> Board:
    """The only door into a `Board`. Used for stored files and request bodies
    alike, so both fail the same way."""
    raw = migrate(raw)

    board_id = _str(raw, "board_id", where="board")
    if not BOARD_ID.fullmatch(board_id):
        raise RankingsError(f"board: bad board_id {board_id!r}")
    name = _str(raw, "name", where="board")
    if len(name) > MAX_NAME:
        raise RankingsError(f"board: name exceeds {MAX_NAME} characters")

    rev = raw.get("rev", 1)
    if not isinstance(rev, int) or isinstance(rev, bool) or rev < 1:
        raise RankingsError(f"board: rev must be a positive int, got {rev!r}")

    scopes_raw = raw.get("scopes", {})
    if not isinstance(scopes_raw, Mapping):
        raise RankingsError("board: scopes must be a mapping")
    unknown = set(scopes_raw) - set(SCOPES)
    if unknown:
        raise RankingsError(f"board: unknown scope(s) {sorted(unknown)}")

    # A MISSING scope is filled, an UNKNOWN one is refused. Filling keeps older
    # and hand-edited files loadable; refusing catches a typo before it becomes
    # a silently-ignored section of someone's board.
    scopes = {name_: _parse_scope(scopes_raw.get(name_), where=f"scopes.{name_}")
              for name_ in SCOPES}

    seeded = raw.get("seeded_from", {})
    if not isinstance(seeded, Mapping):
        raise RankingsError("board: seeded_from must be a mapping")

    now = utc_now()
    return Board(
        board_id=board_id, name=name,
        created_at=str(raw.get("created_at") or now),
        updated_at=str(raw.get("updated_at") or now),
        rev=rev, seeded_from=dict(seeded), scopes=scopes,
    )


def to_json(board: Board) -> dict[str, Any]:
    return {
        "schema_version": board.schema_version,
        "rev": board.rev,
        "board_id": board.board_id,
        "name": board.name,
        "created_at": board.created_at,
        "updated_at": board.updated_at,
        "seeded_from": dict(board.seeded_from),
        "scopes": {
            name: {"tiers": [{"id": t.id, "label": t.label, "color": t.color,
                              "player_ids": list(t.player_ids)}
                             for t in scope.tiers]}
            for name, scope in board.scopes.items()
        },
    }
