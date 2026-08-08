"""Delete fitted artifacts no longer reachable by the engine.

Every training run writes a new generation under `data/artifacts`, keyed by
snapshot id, and nothing has ever removed the old ones — a single day of
rebuilds left sixteen generations and 6.1 MB, most of it superseded training
matrices. That is fine on a laptop and not fine once the directory is tracked,
because the repo would grow by ~1 MB per rebuild forever.

**Reachability, not age.** `models.artifacts.newest` selects by mtime, per
pattern, so the live set is exactly "what `newest` would return for each
artifact family" plus whatever the current bundle references. Anything else is
unreachable by definition: no code path can load it.

Dry run by default. `--apply` is a separate, deliberate flag because these
files cost minutes of CPU and a CFBD pull to regenerate.

    venv/bin/python scripts/prune_artifacts.py           # show
    venv/bin/python scripts/prune_artifacts.py --apply   # delete
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "data" / "artifacts"

#: One family per artifact kind. The engine loads each through
#: `models.artifacts.newest`, so the newest of each is live.
FAMILIES = (
    "projections_*.parquet", "projections_*.meta.json",
    "training_matrix_*.parquet",
    "weekly_sigma_*.json", "hazard_*.json", "kdst_*.json",
    "correlation_*.npz", "correlation_posterior_*.npz",
    "tiers_*.parquet",
    # Audit outputs. Not loaded by the draft path, but they are the recorded
    # result of a study that took real time to run, and some tests read them.
    # Keeping the newest of each costs kilobytes.
    "reliability_*.parquet", "states_*.parquet", "vs_adp_*.json",
    "objective_backtest*.parquet",
)

_SNAPSHOT = re.compile(r"(sn2_[0-9a-f]+)")


def live_paths() -> set[Path]:
    """Everything the engine can still reach."""
    from src.models.artifacts import newest

    keep: set[Path] = set()
    for pattern in FAMILIES:
        # `correlation_*` also matches `correlation_posterior_*`; keep both by
        # asking for each family separately and letting the union settle it.
        hit = newest(pattern, root=ARTIFACTS)
        if hit is not None:
            keep.add(hit)

    # The fitted models must agree with each other, so keep the whole
    # generation that the newest hazard belongs to rather than a mix.
    from src.models.artifacts import newest as _newest
    hazard = _newest("hazard_*.json", root=ARTIFACTS)
    if hazard is not None:
        match = _SNAPSHOT.search(hazard.name)
        if match:
            keep |= set(ARTIFACTS.glob(f"*{match.group(1)}*"))

    # And the generation the shipped bundle points at.
    meta = _newest("projections_*.meta.json", root=ARTIFACTS)
    if meta is not None:
        snapshot = json.loads(meta.read_text()).get("snapshot_id", "")
        if snapshot:
            keep |= set(ARTIFACTS.glob(f"*{snapshot}*"))
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; omitted, this only reports")
    args = ap.parse_args()

    if not ARTIFACTS.exists():
        raise SystemExit(f"no artifact directory at {ARTIFACTS}")

    keep = live_paths()
    everything = {p for p in ARTIFACTS.iterdir() if p.is_file()}
    stale = sorted(everything - keep)

    by_snapshot: dict[str, list[Path]] = defaultdict(list)
    for path in stale:
        match = _SNAPSHOT.search(path.name)
        by_snapshot[match.group(1) if match else "(unversioned)"].append(path)

    kept_mb = sum(p.stat().st_size for p in keep) / 1e6
    stale_mb = sum(p.stat().st_size for p in stale) / 1e6
    print(f"  live   {len(keep):3d} files  {kept_mb:6.2f} MB")
    print(f"  stale  {len(stale):3d} files  {stale_mb:6.2f} MB")
    for snapshot, paths in sorted(by_snapshot.items()):
        print(f"    {snapshot}  {len(paths)} file(s)")

    if not stale:
        print("\n  nothing to prune")
        return
    if not args.apply:
        print("\n  dry run — pass --apply to delete")
        return
    for path in stale:
        path.unlink()
    print(f"\n  deleted {len(stale)} files ({stale_mb:.2f} MB)")


if __name__ == "__main__":
    main()
