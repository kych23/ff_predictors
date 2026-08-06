"""Identity spine (§11.4) — one player, one row, one stated provenance.

Draft night resolves a name typed into a terminal against a board built weeks
earlier. It has to be right, and when it cannot be right it has to say so in
under five seconds. The cascade, first hit wins:

    exact_id       yahoo_id present in both sides
    manual         config/id_overrides.yaml, typed by a human
    deterministic  normalized name + position (+ team when known)
    fuzzy          rapidfuzz token_sort_ratio >= 92, unique winner >= 6 clear
    unresolved     confirm card

Every resolution carries the tier that produced it. A `fuzzy` hit and an
`exact_id` hit are not the same fact, and collapsing them into "matched"
throws away the only information that tells you which ones to check.

**`exact_id` is currently unreachable** — it needs the Yahoo player list, and
Fantasy Sports API access is pending (§11.5). The tier is implemented and
tested against a synthetic list so that access is a config change rather than
a code change. Everything below it works now, which is what makes the draft
safe without Yahoo.

**Rookies resolve to `rookie_declared`, never to a silent NaN → 0.0.** A rookie
has no prior-season row by construction; treating that as "unknown player"
routes him to replacement level, and treating it as zero makes him unstartable.
The coverage tier is what sends him to the rookie calibration bucket instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import yaml

from src.core.errors import ConfigError
from src.platform.identity.match import (
    FUZZY_MIN_MARGIN, FUZZY_MIN_SCORE, normalize_name,
)

DEFAULT_OVERRIDES: Final = Path("config/id_overrides.yaml")

#: Ordered best-first. `Spine.resolve` returns the first hit, and the caller
#: is expected to treat later tiers with more suspicion.
TIERS: Final = ("exact_id", "manual", "deterministic", "fuzzy")

#: Coverage labels a resolved row can carry (§11.4, §19).
COVERAGE = ("full", "rookie_declared", "no_projection", "unresolved")


@dataclass(frozen=True)
class SpineRow:
    player_id: str
    name: str
    position: str
    team: str | None
    tier: str
    coverage: str = "full"
    score: float = 100.0
    yahoo_id: str | None = None
    #: Carried alongside `coverage` so the rookie calibration bucket survives
    #: even when `no_projection` takes the coverage slot.
    is_rookie: bool = False

    @property
    def is_confident(self) -> bool:
        """Whether this may be used without a human glance."""
        return self.tier in ("exact_id", "manual", "deterministic")


def load_overrides(path: Path | None = None) -> dict[str, str]:
    """`normalized name|POSITION -> player_id`, from a human-typed YAML.

    This file is the escape hatch for draft night: when the room drafts someone
    the cascade cannot resolve, a one-line edit fixes it without a rebuild.
    Keys are normalized on load so the file can be written the natural way
    ("Marvin Harrison Jr.") rather than in the matcher's internal form.
    """
    path = path or DEFAULT_OVERRIDES
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("overrides", raw) or {}
    if not isinstance(entries, dict):
        raise ConfigError(
            f"{path}: `overrides` must be a mapping of 'Name|POS' -> player_id"
        )
    out: dict[str, str] = {}
    for key, player_id in entries.items():
        if "|" not in str(key):
            raise ConfigError(
                f"{path}: override key {key!r} must be 'Name|POSITION' — "
                f"position is part of the key because name collisions across "
                f"positions produce wrong joins that never get reported"
            )
        name, position = str(key).rsplit("|", 1)
        out[f"{normalize_name(name)}|{position.strip().upper()}"] = str(player_id)
    return out


class Spine:
    """Constructed from a snapshot; immutable. No module-level mutable state."""

    def __init__(self, rows: pd.DataFrame, *, snapshot_id: str,
                 overrides: dict[str, str] | None = None) -> None:
        required = {"player_id", "name", "position"}
        missing = required - set(rows.columns)
        if missing:
            raise ConfigError(f"spine rows are missing {sorted(missing)}")

        self.snapshot_id = snapshot_id
        self._overrides = dict(overrides or {})
        frame = rows.copy()
        frame["player_id"] = frame["player_id"].astype(str)
        frame["_norm"] = frame["name"].map(normalize_name)
        frame["_key"] = frame["_norm"] + "|" + frame["position"].astype(str).str.upper()
        self._rows = frame

        self._by_id = {r["player_id"]: r for _, r in frame.iterrows()}
        self._by_yahoo = ({str(r["yahoo_id"]): r for _, r in frame.iterrows()
                           if pd.notna(r.get("yahoo_id"))}
                          if "yahoo_id" in frame.columns else {})
        # keep(first) rather than raising: duplicate (name, position) pairs are
        # real (two players share a name at the same position often enough) and
        # a hard failure here would take the whole draft down.
        self._by_key: dict[str, pd.Series] = {}
        self._ambiguous: set[str] = set()
        for _, r in frame.iterrows():
            if r["_key"] in self._by_key:
                self._ambiguous.add(r["_key"])
            else:
                self._by_key[r["_key"]] = r

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def ambiguous_keys(self) -> set[str]:
        """(name, position) pairs held by more than one player.

        Surfaced rather than hidden: these are precisely the resolutions that
        need a team to disambiguate, and a caller that ignores them will
        silently draft the wrong person.
        """
        return set(self._ambiguous)

    def _coverage(self, row: pd.Series) -> str:
        """`no_projection` outranks `rookie_declared`.

        The two are not mutually exclusive — most rookies without a projection
        are both — but they drive different machinery, and only one of them is
        operationally decisive. `no_projection` means §19 must assign
        replacement level and drop the player from the shortlist; a
        `rookie_declared` row with a projection is a normal, startable player
        who happens to use the rookie calibration bucket. Losing the first to
        report the second would silently field a short lineup.

        `is_rookie` is carried separately so the bucket signal is not lost.
        """
        if "has_projection" in row and not bool(row["has_projection"]):
            return "no_projection"
        if bool(row.get("is_rookie", False)):
            return "rookie_declared"
        return "full"

    def _row(self, row: pd.Series, tier: str, score: float = 100.0) -> SpineRow:
        return SpineRow(
            player_id=str(row["player_id"]), name=str(row["name"]),
            position=str(row["position"]).upper(),
            team=(str(row["team"]) if "team" in row and pd.notna(row["team"])
                  else None),
            tier=tier, coverage=self._coverage(row), score=score,
            yahoo_id=(str(row["yahoo_id"]) if "yahoo_id" in row
                      and pd.notna(row["yahoo_id"]) else None),
            is_rookie=bool(row.get("is_rookie", False)),
        )

    def resolve(self, name: str, position: str | None = None,
                team: str | None = None, *, yahoo_id: str | None = None
                ) -> SpineRow | None:
        """First hit in the cascade, or None for the confirm card."""
        # 1. exact_id — needs the Yahoo player list (§11.5, pending)
        if yahoo_id is not None and str(yahoo_id) in self._by_yahoo:
            return self._row(self._by_yahoo[str(yahoo_id)], "exact_id")

        norm = normalize_name(name)
        pos = str(position).upper() if position else None
        key = f"{norm}|{pos}" if pos else None

        # 2. manual — a human already decided this one
        if key and key in self._overrides:
            target = self._overrides[key]
            if target in self._by_id:
                return self._row(self._by_id[target], "manual")

        # 3. deterministic — normalized name + position, team as tiebreak
        if key and key in self._ambiguous:
            # Two players share this (name, position). Without a team this is a
            # coin flip, and a coin flip that returns a SpineRow is worse than
            # returning None: the caller cannot tell it apart from a real hit
            # and drafts the wrong person. Fall through to the confirm card.
            if team is not None:
                candidates = self._rows[(self._rows["_key"] == key)
                                        & (self._rows["team"] == team)]
                if len(candidates) == 1:
                    return self._row(candidates.iloc[0], "deterministic")
            return None
        if key and key in self._by_key:
            return self._row(self._by_key[key], "deterministic")
        if not key and norm:
            # no position given: only accept a globally unique name
            hits = self._rows[self._rows["_norm"] == norm]
            if len(hits) == 1:
                return self._row(hits.iloc[0], "deterministic")

        # 4. fuzzy — a unique winner, well clear of the runner-up
        hit = self._fuzzy(norm, pos)
        if hit is not None:
            return hit
        return None

    def _fuzzy(self, norm: str, pos: str | None) -> SpineRow | None:
        try:
            from rapidfuzz import fuzz, process
        except ImportError:      # optional; the tiers above still work
            return None
        if not norm:
            return None

        pool = self._rows
        if pos:
            pool = pool[pool["position"].astype(str).str.upper() == pos]
        if pool.empty:
            return None

        choices = pool["_norm"].tolist()
        # token_sort_ratio per §11.4, stated explicitly: rapidfuzz defaults to
        # WRatio, which applies its own partial-match heuristics and scores
        # substrings far higher. "Chase" against "Ja'Marr Chase" clears 92 on
        # WRatio and does not on token_sort_ratio — and a bare surname
        # resolving to a first-round receiver is exactly the wrong-join class
        # the threshold exists to prevent.
        results = process.extract(norm, choices, limit=2,
                                  scorer=fuzz.token_sort_ratio)
        if not results:
            return None
        best_name, best_score, best_i = results[0]
        runner = results[1][1] if len(results) > 1 else 0.0
        # The margin matters more than the score. Two brothers on the board
        # both score in the high 90s against either name, and picking the
        # higher one is a coin flip dressed as a match.
        if best_score < FUZZY_MIN_SCORE or (best_score - runner) < FUZZY_MIN_MARGIN:
            return None
        return self._row(pool.iloc[best_i], "fuzzy", score=float(best_score))


def build_spine(players: pd.DataFrame, *, snapshot_id: str,
                yahoo_player_list: pd.DataFrame | None = None,
                overrides_path: Path | None = None) -> Spine:
    """Assemble the spine from the players table plus optional Yahoo ids."""
    rows = players.copy()
    if yahoo_player_list is not None and len(yahoo_player_list):
        rows = rows.merge(
            yahoo_player_list[["player_id", "yahoo_id"]].astype(str),
            on="player_id", how="left")
    return Spine(rows, snapshot_id=snapshot_id,
                 overrides=load_overrides(overrides_path))
