"""Fitted-artifact persistence with checked provenance (§20.3).

Every artifact is written to ``data/artifacts/<name>_<snapshot_id>.<ext>`` and
carries ``snapshot_id`` and ``model_version`` as *fields*, not merely in the
filename — a filename is a convention, a field is data.

``build_bundle`` then asserts every artifact's snapshot matches the bundle's.
Stamping provenance without comparing it is what r4 did, which would let a
correlation matrix fitted on one snapshot pair silently with projections from
another.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from src.core.errors import ArtifactSnapshotMismatch, DataError
from src.models.correlation.slot_matrix import SlotCorrelation
from src.models.weekly.kdst import KDSTModel
from src.models.weekly.variance import WeeklySigmaModel

ARTIFACT_ROOT: Final = Path("data/artifacts")


@dataclass(frozen=True)
class ArtifactSet:
    """Everything the simulator needs from the modeling layer."""

    weekly_sigma: WeeklySigmaModel
    correlation: SlotCorrelation
    kdst: KDSTModel
    snapshot_id: str

    def assert_consistent(self) -> None:
        bad = {
            name: sid
            for name, sid in (
                ("weekly_sigma", self.weekly_sigma.snapshot_id),
                ("correlation", self.correlation.snapshot_id),
                ("kdst", self.kdst.snapshot_id),
            )
            if sid and sid != self.snapshot_id
        }
        if bad:
            raise ArtifactSnapshotMismatch(
                f"artifacts disagree with snapshot {self.snapshot_id}: {bad}"
            )


def _path(name: str, snapshot_id: str, ext: str, root: Path | None = None) -> Path:
    base = root or ARTIFACT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}_{snapshot_id}.{ext}"


def save_weekly_sigma(model: WeeklySigmaModel, *, root: Path | None = None) -> Path:
    path = _path("weekly_sigma", model.snapshot_id, "json", root)
    path.write_text(json.dumps(model.to_dict(), indent=2))
    return path


def load_weekly_sigma(snapshot_id: str, *, root: Path | None = None
                      ) -> WeeklySigmaModel:
    path = _path("weekly_sigma", snapshot_id, "json", root)
    if not path.exists():
        raise DataError(f"no weekly-sigma artifact at {path}")
    return WeeklySigmaModel.from_dict(json.loads(path.read_text()))


def save_correlation(corr: SlotCorrelation, *, root: Path | None = None) -> Path:
    path = _path("correlation", corr.snapshot_id, "npz", root)
    np.savez(
        path, matrix=corr.matrix, cholesky=corr.cholesky,
        pair_counts=corr.pair_counts,
        meta=json.dumps({
            "slots": list(corr.slots), "min_eigenvalue": corr.min_eigenvalue,
            "was_projected": corr.was_projected, "source": corr.source,
            "snapshot_id": corr.snapshot_id, "model_version": corr.model_version,
        }),
    )
    return path


def load_correlation(snapshot_id: str, *, root: Path | None = None
                     ) -> SlotCorrelation:
    path = _path("correlation", snapshot_id, "npz", root)
    if not path.exists():
        raise DataError(f"no correlation artifact at {path}")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return SlotCorrelation(
            slots=tuple(meta["slots"]), matrix=data["matrix"],
            cholesky=data["cholesky"], pair_counts=data["pair_counts"],
            min_eigenvalue=float(meta["min_eigenvalue"]),
            was_projected=bool(meta["was_projected"]), source=meta["source"],
            snapshot_id=meta["snapshot_id"], model_version=meta["model_version"],
        )


def save_kdst(model: KDSTModel, *, root: Path | None = None) -> Path:
    path = _path("kdst", model.snapshot_id, "json", root)
    path.write_text(json.dumps(model.to_dict(), indent=2))
    return path


def load_kdst(snapshot_id: str, *, root: Path | None = None) -> KDSTModel:
    path = _path("kdst", snapshot_id, "json", root)
    if not path.exists():
        raise DataError(f"no K/DST artifact at {path}")
    return KDSTModel.from_dict(json.loads(path.read_text()))


def load_all(snapshot_id: str, *, root: Path | None = None) -> ArtifactSet:
    artifacts = ArtifactSet(
        weekly_sigma=load_weekly_sigma(snapshot_id, root=root),
        correlation=load_correlation(snapshot_id, root=root),
        kdst=load_kdst(snapshot_id, root=root),
        snapshot_id=snapshot_id,
    )
    artifacts.assert_consistent()
    return artifacts
