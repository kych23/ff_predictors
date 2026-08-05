"""Draft board, read from the offline bundle (§9.2 note 6, §19).

REWRITTEN, not ported. The v1 `src/recommender/board.py` read the Supabase
serving DB via `serving_session_scope()` and fetched nflverse schedules over
the network for bye weeks. §19 forbids both on draft night: the recommendation
path touches no Postgres, no Supabase and no network, so a dropped pooler
connection or a slow CDN cannot cost a pick.

Everything the board needs is in `draft_night_bundle.parquet`, which is why
that file is a superset of `ProjectionBundle` rather than just the arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.core.config.league import LeagueConfig
from src.platform import bundle as bundle_mod

DEFAULT_BUNDLE = Path("data/bundles/draft_night_bundle.parquet")


@dataclass
class Board:
    """The draftable universe plus who has been taken."""

    players: pd.DataFrame
    snapshot_id: str
    value_source: str
    built_at: str
    drafted: set[str] = field(default_factory=set)

    @classmethod
    def from_bundle(cls, path: Path | None = None) -> "Board":
        b = bundle_mod.read(path or DEFAULT_BUNDLE)
        return cls(players=b.board(), snapshot_id=b.snapshot_id,
                   value_source=b.value_source, built_at=b.built_at)

    # ------------------------------------------------------------- queries
    def available(self) -> pd.DataFrame:
        return self.players[~self.players["player_id"].isin(self.drafted)].copy()

    def take(self, player_id: str) -> None:
        if player_id in self.drafted:
            raise ValueError(f"{player_id} has already been drafted")
        if player_id not in set(self.players["player_id"]):
            raise KeyError(f"{player_id} is not on the board")
        self.drafted.add(player_id)

    def find(self, query: str, *, limit: int = 8) -> pd.DataFrame:
        """Fuzzy-ish lookup for live entry — substring on a normalized name.

        Manual entry is the primary input path (§19), so this has to tolerate
        the way a name is actually typed under a 60-second clock.
        """
        from src.platform.identity.match import normalize_name

        key = normalize_name(query)
        if not key:
            return self.available().head(0)
        avail = self.available()
        norm = avail["player_name"].map(normalize_name)
        starts = avail[norm.str.startswith(key)]
        contains = avail[norm.str.contains(key, regex=False) & ~norm.str.startswith(key)]
        return pd.concat([starts, contains]).head(limit)

    # -------------------------------------------------------------- staleness
    def stale_flags(self, cfg: LeagueConfig) -> list[str]:
        """What the recommendation must disclose about its own inputs."""
        flags: list[str] = []
        if self.value_source != "quantile_model":
            flags.append(f"value_source={self.value_source}")
        n_no_value = int((self.players["coverage"] != "full").sum())
        if n_no_value:
            flags.append(f"no_prior_season={n_no_value}")
        if self.players["adp"].isna().any():
            flags.append("missing_adp")
        return flags
