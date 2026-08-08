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
from src.models.weekly.hazard import HazardModel
from src.models.weekly.kdst import KDSTModel
from src.models.weekly.variance import WeeklySigmaModel

ARTIFACT_ROOT: Final = Path("data/artifacts")

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]

#: The source that decides what a recommendation is. `src/models` and
#: `src/domain` produce the artifact; `src/engine` turns it into an answer.
#:
#: `src/engine` is included because of what the golden actually asserts.
#: `tests/test_refactor_parity.py` pins the LEADER, which comes out of
#: `build_tiers -> allocate -> rollout -> kernel`. Covering only the model
#: surfaces would leave a change to the allocator or the kernel with both an
#: unchanged snapshot_id and an unchanged code_digest, so the golden would go
#: on comparing against numbers a different engine produced — precisely the
#: failure this digest exists to catch, on the code the test exercises most.
#:
#: `src/app` is deliberately OUT: the cockpit is orchestration and its churn
#: (web schemas, narration wording) would invalidate the golden constantly
#: without changing an answer. The one piece of `app` that does move numbers,
#: `cockpit/build.py`, is the acknowledged gap — a UI-shaped module doing
#: engine-shaped work, which is a structural problem to fix by moving code,
#: not by widening a hash.
CODE_SURFACES: Final = ("src/models", "src/domain", "src/engine")


def code_digest(surfaces: tuple[str, ...] = CODE_SURFACES, *,
                root: Path | None = None) -> str:
    """Fingerprint of the code that produced an artifact.

    ``snapshot_id`` is a Merkle root over the SOURCE DATA only — source,
    asset, params, digest — and deliberately so: refetching identical bytes
    must yield the same id. But that leaves a hole, because it means a change
    to feature or model code produces a materially different artifact under an
    unchanged id, which then overwrites its predecessor with nothing recording
    that anything moved.

    Observed here: fixing the prior-production decay changed Josh Allen's p50
    from 20.30 to 20.88 and reshuffled the top ten, and both artifacts carried
    ``sn2_58ced77e1b3cfec91e9fadd6`` and the same ``model_version``. No field
    anywhere distinguished them.

    The consequence that bites is `tests/test_refactor_parity.py`. It skips
    when the bundle's snapshot no longer matches the golden's, which is how a
    stale baseline is meant to announce itself — but a code-only change keeps
    the snapshot identical, so the golden goes on comparing against numbers a
    different engine produced and passes. The one guard against a silent
    change in recommendations was blind to the most likely way one happens.

    This is the code axis of that identity. It is NOT a substitute for
    `snapshot_id`: data and code are independent reasons for an artifact to
    differ, and provenance needs both.
    """
    import hashlib

    base = root or _REPO_ROOT
    digest = hashlib.sha256()
    for surface in sorted(surfaces):
        directory = base / surface
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            digest.update(path.relative_to(base).as_posix().encode())
            digest.update(_semantic_source(path).encode())
    return digest.hexdigest()[:12]


def _semantic_source(path: Path) -> str:
    """The file's code with comments and docstrings removed.

    Hashing raw bytes was the first version and it was too eager: a
    documentation pass over `src/engine` invalidated the parity golden, which
    trains whoever hits it to regenerate reflexively. A guard that cries wolf
    stops being read, and this one exists to be believed.

    Round-tripping through `ast` drops comments and formatting; stripping
    docstring expressions drops the rest of the prose. What survives is every
    statement and constant that can change an answer. Falls back to raw bytes
    on a syntax error rather than silently hashing nothing.
    """
    import ast

    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            # A lone docstring is the whole body for a stub; keep it valid.
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


@dataclass(frozen=True)
class ArtifactSet:
    """Everything the simulator needs from the modeling layer."""

    weekly_sigma: WeeklySigmaModel
    correlation: SlotCorrelation
    kdst: KDSTModel
    snapshot_id: str
    #: Optional so an ArtifactSet fitted before §12.4 still loads. Absent means
    #: the simulator falls back to a flat hazard and says so.
    hazard: HazardModel | None = None

    def assert_consistent(self) -> None:
        bad = {
            name: sid
            for name, sid in (
                ("weekly_sigma", self.weekly_sigma.snapshot_id),
                ("correlation", self.correlation.snapshot_id),
                ("kdst", self.kdst.snapshot_id),
                ("hazard", self.hazard.snapshot_id if self.hazard else None),
            )
            if sid and sid != self.snapshot_id
        }
        if bad:
            raise ArtifactSnapshotMismatch(
                f"artifacts disagree with snapshot {self.snapshot_id}: {bad}"
            )


def newest(pattern: str, *, root: Path | None = None) -> Path | None:
    """The most recently WRITTEN artifact matching `pattern`, or None.

    Every caller that wants "the latest artifact" must come through here.
    ``sorted(glob(...))[-1]`` looks like it does this and does not: artifact
    filenames end in a snapshot_id, which is a content hash, so lexicographic
    order is effectively random. Measured on a real directory, that idiom
    selected a two-day-old projections file over one written minutes earlier —
    in `build_bundle`, which means a freshly fitted model gets built, stamped,
    and then silently ignored on draft night.

    mtime is the right key precisely because snapshot ids carry no ordering.
    """
    hits = list((root or ARTIFACT_ROOT).glob(pattern))
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def newest_all(pattern: str, *, root: Path | None = None) -> list[Path]:
    """Matches newest-first. Same ordering rule as `newest`."""
    return sorted((root or ARTIFACT_ROOT).glob(pattern),
                  key=lambda p: p.stat().st_mtime, reverse=True)


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


def save_hazard(model: HazardModel, *, root: Path | None = None) -> Path:
    path = _path("hazard", model.snapshot_id, "json", root)
    path.write_text(json.dumps(model.to_dict(), indent=2))
    return path


def load_hazard(snapshot_id: str, *, root: Path | None = None) -> HazardModel:
    path = _path("hazard", snapshot_id, "json", root)
    if not path.exists():
        raise DataError(f"no hazard artifact at {path}")
    return HazardModel.from_dict(json.loads(path.read_text()))


def load_all(snapshot_id: str, *, root: Path | None = None) -> ArtifactSet:
    try:
        hazard = load_hazard(snapshot_id, root=root)
    except DataError:
        hazard = None
    artifacts = ArtifactSet(
        weekly_sigma=load_weekly_sigma(snapshot_id, root=root),
        correlation=load_correlation(snapshot_id, root=root),
        kdst=load_kdst(snapshot_id, root=root),
        snapshot_id=snapshot_id,
        hazard=hazard,
    )
    artifacts.assert_consistent()
    return artifacts
