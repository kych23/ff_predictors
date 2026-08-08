"""Turn a reported name into a player id (§11.4).

Ported out of ``scripts/draft_night.py``, which is a script and therefore not
importable by a backend. The cascade is unchanged: the identity spine first,
then a substring scan over the board.

**Why the substring pass exists.** The spine is built from the bundle, and at a
live draft a typo'd or shortened surname is far more common than a genuinely
unknown player. `Spine.resolve` deliberately returns `None` rather than guessing
between two players who share a normalized name — a coin flip that returns a
`SpineRow` is worse than no answer, because the caller cannot tell it apart from
a real hit and drafts the wrong person. The substring pass turns that `None`
into a *choice* for the operator instead of a dead end.

**Ambiguity is not an error.** Multiple matches come back as candidates so a
human can pick. A status code cannot carry options.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.platform.identity.spine import Spine

RESOLVED = "resolved"
AMBIGUOUS = "ambiguous"
UNRESOLVED = "unresolved"

#: Cap on candidates offered for an ambiguous name. More than this on a pick
#: clock is not a choice, it is a wall of text.
MAX_CANDIDATES = 8


@dataclass(frozen=True)
class Candidate:
    player_id: str
    name: str
    position: str
    team: str | None = None
    adp: float | None = None

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "name": self.name,
                "position": self.position, "team": self.team, "adp": self.adp}


@dataclass(frozen=True)
class Resolution:
    status: str
    player_id: str | None = None
    name: str | None = None
    tier: str | None = None
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == RESOLVED


def _row_to_candidate(row: pd.Series) -> Candidate:
    adp = row.get("adp")
    return Candidate(
        player_id=str(row["player_id"]),
        name=str(row["player_name"]),
        position=str(row.get("position", "")),
        team=None if pd.isna(row.get("team")) else str(row.get("team")),
        adp=None if adp is None or pd.isna(adp) else float(adp),
    )


def build_spine(players: pd.DataFrame, snapshot_id: str) -> Spine:
    """The spine wants a `name` column; the board calls it `player_name`."""
    frame = players.rename(columns={"player_name": "name"})
    return Spine(frame[["player_id", "name", "position", "team"]],
                 snapshot_id=snapshot_id)


def resolve_name(text: str, spine: Spine, board: pd.DataFrame, *,
                 exclude: set[str] | None = None) -> Resolution:
    """Resolve one reported name against the board.

    ``exclude`` is the already-drafted set. Excluding it matters for the
    substring pass specifically: late in a draft "Jones" may match four players
    of whom three are gone, and offering those makes the operator do the
    filtering the tool should have done.
    """
    query = (text or "").strip()
    if not query:
        return Resolution(UNRESOLVED)

    taken = exclude or set()
    hit = spine.resolve(query)
    if hit is not None and str(hit.player_id) not in taken:
        return Resolution(RESOLVED, player_id=str(hit.player_id),
                          name=str(hit.name), tier=hit.tier)

    lowered = query.lower()
    names = board["player_name"].astype(str).str.lower()
    matches = board[names.str.contains(lowered, regex=False)]
    if taken:
        matches = matches[~matches["player_id"].astype(str).isin(taken)]

    if len(matches) == 1:
        row = matches.iloc[0]
        return Resolution(RESOLVED, player_id=str(row["player_id"]),
                          name=str(row["player_name"]), tier="substring")
    if len(matches) > 1:
        candidates = [_row_to_candidate(r) for _, r in
                      matches.head(MAX_CANDIDATES).iterrows()]
        return Resolution(AMBIGUOUS, candidates=candidates)

    return Resolution(UNRESOLVED)
