"""Snapshot manifest and merkle root (§11.1).

``snapshot_id`` is a hash over the sorted manifest, so it names the exact set of
bytes an artifact was built from. Two consequences the design leans on:

* the reconciliation gate becomes a pure function — ``reconcile --snapshot X``
  replays from blobs with no network;
* nflverse's retroactive stat corrections stop being an existential "do I
  retrain?" and become a mechanical ``diff()``: a new digest for the same
  logical asset produces a drift report naming what changed, and the only
  question left is whether the pick order moves.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Sequence

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
SNAPSHOT_ROOT: Final = _REPO_ROOT / "data" / "snapshots"

SNAPSHOT_PREFIX: Final = "sn2_"
SNAPSHOT_HEX_LEN: Final = 24


@dataclass(frozen=True, order=True)
class ManifestEntry:
    """One fetched logical asset. Ordering is the manifest's sort order."""

    source: str
    logical_asset: str
    params_key: str
    digest: str
    fetched_at: str          # ISO-8601 UTC
    canonicalizer: str

    def identity(self) -> tuple[str, str, str]:
        return (self.source, self.logical_asset, self.params_key)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_snapshot(entries: Iterable[ManifestEntry]) -> str:
    """Merkle root over the sorted entry identities and digests.

    ``fetched_at`` is deliberately excluded: refetching identical bytes must
    yield the same snapshot id, or nothing downstream is reproducible.
    """
    rows = sorted(
        (e.source, e.logical_asset, e.params_key, e.digest) for e in entries
    )
    blob = json.dumps(rows, separators=(",", ":")).encode()
    return SNAPSHOT_PREFIX + hashlib.sha256(blob).hexdigest()[:SNAPSHOT_HEX_LEN]


def write_manifest(
    snapshot_id: str, entries: Sequence[ManifestEntry], *, root: Path | None = None
) -> Path:
    base = root or SNAPSHOT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{snapshot_id}.ndjson"
    with path.open("w", encoding="utf-8") as fh:
        for entry in sorted(entries):
            fh.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
    return path


def read_manifest(snapshot_id: str, *, root: Path | None = None) -> list[ManifestEntry]:
    base = root or SNAPSHOT_ROOT
    path = base / f"{snapshot_id}.ndjson"
    if not path.exists():
        raise FileNotFoundError(f"manifest for {snapshot_id} not found at {path}")
    return [
        ManifestEntry(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass(frozen=True)
class DriftReport:
    added: tuple[tuple[str, str, str], ...]
    removed: tuple[tuple[str, str, str], ...]
    changed: tuple[tuple[str, str, str], ...]

    @property
    def is_clean(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def describe(self) -> str:
        if self.is_clean:
            return "no drift"
        parts = []
        for label, items in (("added", self.added), ("removed", self.removed),
                             ("changed", self.changed)):
            if items:
                parts.append(f"{label}: " + ", ".join(f"{s}/{a}" for s, a, _ in items))
        return "; ".join(parts)


def diff(a: str, b: str, *, root: Path | None = None) -> DriftReport:
    """Compare two snapshots. Defined only between ``sn2_`` ids (§20.1)."""
    for sid in (a, b):
        if not sid.startswith(SNAPSHOT_PREFIX):
            raise ValueError(
                f"{sid!r} is not a v2 snapshot id; diff does not cross the "
                f"v1/v2 format boundary (the research DB is rebuilt instead)"
            )
    left = {e.identity(): e.digest for e in read_manifest(a, root=root)}
    right = {e.identity(): e.digest for e in read_manifest(b, root=root)}
    return DriftReport(
        added=tuple(sorted(k for k in right if k not in left)),
        removed=tuple(sorted(k for k in left if k not in right)),
        changed=tuple(sorted(k for k in left if k in right and left[k] != right[k])),
    )
