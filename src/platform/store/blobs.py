"""Content-addressed blob store (§11.1).

Every external response body is canonicalized, hashed, and written here. The
point is that ``snapshot_id`` stops being a label stamped on writes and becomes
a *claim about exactly which bytes produced an artifact* — which is what makes
the reconciliation gate replayable offline, in CI, and in December.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final, Mapping

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]

#: Overridable so tests never touch the real store.
BLOB_ROOT: Final = Path(os.environ.get("FF_BLOB_ROOT", _REPO_ROOT / "data" / "blobs"))


def params_key(params: Mapping[str, object]) -> str:
    """Canonical serialization of fetch parameters.

    Pinned because it is part of the manifest's composite key and therefore of
    the merkle root — two spellings would produce two snapshot ids for one
    fetch.
    """
    return json.dumps(
        {str(k): params[k] for k in sorted(params)},
        sort_keys=True, separators=(",", ":"), default=str,
    )


def digest_of(body: bytes) -> str:
    """Full 64-char sha256 hex. Blob digests are never truncated."""
    return hashlib.sha256(body).hexdigest()


def _path_for(digest: str, root: Path | None = None) -> Path:
    base = root or BLOB_ROOT
    return base / digest[:2] / digest


def put(body: bytes, *, root: Path | None = None) -> str:
    """Write ``body`` and return its digest. Idempotent by construction."""
    digest = digest_of(body)
    path = _path_for(digest, root)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # write-then-rename so a crash cannot leave a truncated blob at a
        # digest that claims to describe complete content
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
    return digest


def get(digest: str, *, root: Path | None = None) -> bytes:
    path = _path_for(digest, root)
    if not path.exists():
        raise FileNotFoundError(f"blob {digest} not in store at {path.parent.parent}")
    return path.read_bytes()


def exists(digest: str, *, root: Path | None = None) -> bool:
    return _path_for(digest, root).exists()
