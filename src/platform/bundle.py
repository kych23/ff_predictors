"""Draft-night bundle — the only file the cockpit reads (§12.6, §19).

A superset of ``ProjectionBundle``: the arrays the simulator needs, plus the
market and display fields §14.1/§17.1 need, because §19 forbids any other data
source on draft night. No Postgres, no Supabase, no network in the
recommendation path.

The bundle carries its own provenance. ``build_bundle`` asserts every input
artifact shares the bundle's ``snapshot_id`` (§20.3), so a projection fitted on
one snapshot cannot be silently paired with ADP from another.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from src.core.errors import ArtifactSnapshotMismatch, DataError

#: Columns every bundle must carry. `value` is the projection the recommender
#: ranks on; `value_source` records what produced it so a placeholder can never
#: masquerade as a fitted model.
REQUIRED_COLUMNS: Final = (
    "player_id", "player_name", "position", "team", "bye_week",
    "value", "value_source", "coverage", "adp", "adp_stdev",
)

#: The CALIBRATED interval around `value`, carried so the simulator can draw
#: from the distribution the quantile model actually fitted. Optional because
#: older bundles predate them and because K/DST never had a quantile model —
#: NaN means "no calibrated band", and `sim.bundle_build` falls back per row.
OPTIONAL_COLUMNS: Final = ("value_p10", "value_p90", "adp_consensus")

#: Declared coverage tiers (§11.4). A row is never silently valueless.
COVERAGE_TIERS: Final = ("full", "no_prior_season", "unresolved")

#: Recognized `value_source` values, worst to best. The two placeholder tiers
#: exist so a filled-in value can never be mistaken for a fitted projection:
#: `replacement_floor` is a positional 10th percentile standing in for a player
#: the identity join could not value, and `kdst_empirical` is the fitted K/DST
#: distribution's mean (K and DST are outside `modeled_positions`, so the
#: quantile model never covers them). Both are deliberately *rankable* — a zero
#: is not a low value, it is an unpickable one.
VALUE_SOURCES: Final = (
    "none", "replacement_floor", "kdst_empirical",
    "prior_season_ppg", "quantile_model",
)


@dataclass(frozen=True)
class Bundle:
    players: pd.DataFrame
    snapshot_id: str
    model_version: str
    decision_version: str
    value_source: str
    built_at: str

    def __post_init__(self) -> None:
        missing = set(REQUIRED_COLUMNS) - set(self.players.columns)
        if missing:
            raise DataError(f"bundle missing columns: {sorted(missing)}")

    @property
    def n_players(self) -> int:
        return len(self.players)

    def with_adp_only(self) -> pd.DataFrame:
        """Rows carrying a market price — the draftable universe."""
        return self.players[self.players["adp"].notna()].copy()

    def board(self) -> pd.DataFrame:
        """Frame shaped for the tier-2 recommender.

        Optional columns ride along when present. `value_p10`/`value_p90` are
        what the simulator draws the season rate from, and dropping them here
        silently sent tier 0 back to a synthesized band.
        """
        columns = list(REQUIRED_COLUMNS) + [
            c for c in OPTIONAL_COLUMNS if c in self.players.columns]
        return self.players[columns].copy()


def write(bundle: Bundle, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = bundle.players.copy()
    frame.attrs.clear()
    frame.to_parquet(path, index=False)
    meta = {
        "snapshot_id": bundle.snapshot_id,
        "model_version": bundle.model_version,
        "decision_version": bundle.decision_version,
        "value_source": bundle.value_source,
        "built_at": bundle.built_at,
        "n_players": bundle.n_players,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return path


def read(path: Path) -> Bundle:
    """Load a bundle. No network, no DB — this is the draft-night path."""
    if not path.exists():
        raise DataError(
            f"no draft bundle at {path}. Run scripts/build_bundle.py before "
            f"draft night; the cockpit has no other data source (§19)."
        )
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise DataError(f"bundle at {path} has no sidecar metadata at {meta_path}")
    meta = json.loads(meta_path.read_text())
    return Bundle(
        players=pd.read_parquet(path),
        snapshot_id=meta["snapshot_id"],
        model_version=meta["model_version"],
        decision_version=meta["decision_version"],
        value_source=meta["value_source"],
        built_at=meta["built_at"],
    )


def assert_artifact_snapshots_match(bundle_snapshot: str, **artifacts: str) -> None:
    """Every fitted artifact must come from the bundle's snapshot (§20.3).

    Stamping provenance is not enough on its own — r4 recorded snapshot ids on
    artifacts and never compared them, which would let a correlation posterior
    fitted on one snapshot pair silently with projections from another.
    """
    bad = {name: sid for name, sid in artifacts.items()
           if sid is not None and sid != bundle_snapshot}
    if bad:
        raise ArtifactSnapshotMismatch(
            f"artifacts disagree with bundle snapshot {bundle_snapshot}: {bad}"
        )
