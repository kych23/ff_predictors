"""Content-addressed store and snapshot manifest (§11.1)."""
from src.platform.store import blobs, canonicalize
from src.platform.store.manifest import (
    DriftReport, ManifestEntry, build_snapshot, diff, read_manifest, write_manifest,
)

__all__ = ["blobs", "canonicalize", "ManifestEntry", "DriftReport",
           "build_snapshot", "diff", "read_manifest", "write_manifest"]
