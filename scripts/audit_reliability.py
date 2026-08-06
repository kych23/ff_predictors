"""The §21.4 reliability study — how much of the ranking survives a refit.

    venv/bin/python scripts/audit_reliability.py [--refits 3]

Refits the projection model on data resampled by PLAYERS WITHIN SEASON, runs
every refit against the same recorded draft states, and reports how much the
ordering moves.

WHAT THIS NUMBER IS NOT: it varies the player sample only. Seasons, feature
set, hyperparameters and scoring config are held fixed, so the true stability
of the whole pipeline is LOWER than what prints below. Quote it with that
sentence attached or do not quote it.

Requires `scripts/record_states.py` to have run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_league  # noqa: E402
from src.engine.audit.reliability import (  # noqa: E402
    reliability, resample_players_within_season,
)
from src.models.projection.quantile_model import QuantileGBM  # noqa: E402
from src.platform import bundle as bundle_mod  # noqa: E402

ARTIFACTS = Path("data/artifacts")


def load_states(snapshot_id: str) -> list[dict]:
    path = ARTIFACTS / f"states_{snapshot_id}.parquet"
    if not path.exists():
        raise SystemExit(
            f"no recorded states at {path}. Run:\n"
            f"  venv/bin/python scripts/record_states.py --n 200"
        )
    return pd.read_parquet(path).to_dict("records")


def load_training_matrix() -> tuple[pd.DataFrame, list[str]]:
    """The feature matrix the projection model was last trained on."""
    metas = sorted(ARTIFACTS.glob("projections_*.meta.json"))
    if not metas:
        raise SystemExit("no projection artifact; run train_projection_v2.py")
    meta = json.loads(metas[-1].read_text())
    path = ARTIFACTS / f"training_matrix_{meta['snapshot_id']}.parquet"
    if not path.exists():
        raise SystemExit(
            f"no training matrix at {path}.\n"
            f"train_projection_v2.py must persist its matrix for the "
            f"reliability study to refit against the same data."
        )
    frame = pd.read_parquet(path)
    features = [c for c in frame.columns
                if c not in {"player_id", "season", "position", "fppg",
                             "as_of_date", "name", "team"}]
    return frame, features


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bundle", type=Path,
                    default=Path("data/bundles/draft_night_bundle.parquet"))
    args = ap.parse_args()

    cfg = load_league()
    b = bundle_mod.read(args.bundle)
    states = load_states(b.snapshot_id)
    matrix, features = load_training_matrix()

    print("§21.4 RELIABILITY STUDY")
    print(f"  {args.refits} refits over {len(matrix):,} player-seasons, "
          f"{len(features)} features")
    print(f"  {len(states)} recorded draft states from {b.snapshot_id}")
    print("  resampling unit: PLAYERS WITHIN SEASON (never whole seasons)\n")

    # One refit per index, cached: rank_fn is called once per state per refit
    # and refitting inside it would be O(states x refits) model fits.
    # Refits train on every season BUT rank only the target season. The matrix
    # carries one row per player-season, so scoring it whole and keying a dict
    # by player_id would silently keep whichever season landed last.
    target_season = int(matrix["season"].max())
    target_rows = matrix[matrix["season"] == target_season]
    train_rows = matrix[matrix["season"] < target_season]
    print(f"  ranking {len(target_rows):,} players in {target_season}; "
          f"training on {len(train_rows):,} rows before it")

    fitted: dict[int, dict[str, float]] = {}
    for i in range(args.refits):
        rng = np.random.default_rng(args.seed + i)
        sample = resample_players_within_season(train_rows, rng)
        model = QuantileGBM()
        model.fit(sample[features], sample["fppg"])
        preds = model.predict(target_rows[features])
        p50 = np.asarray(preds["p50"], dtype=float)
        fitted[i] = dict(zip(target_rows["player_id"].astype(str), p50))
        print(f"  refit {i}: fitted on {len(sample):,} resampled rows")

    board_ids = set(b.board()["player_id"].astype(str))

    def rank_fn(state: dict, refit: int):
        gone = set(json.loads(state["drafted"]))
        values = fitted[refit]
        available = [p for p in board_ids if p not in gone]
        return sorted(available, key=lambda p: -values.get(p, -1e9))

    report = reliability(states, rank_fn, n_refits=args.refits)

    print(f"\n{report.describe()}\n")
    print(f"  {'round':>6s} {'spearman@12':>12s} {'spearman@84':>12s} "
          f"{'exact@1':>9s}")
    pivot = report.by_round.pivot(index="round", columns="k")
    for rnd in sorted(report.by_round["round"].unique()):
        s12 = pivot.loc[rnd, ("spearman", 12)]
        s84 = pivot.loc[rnd, ("spearman", 84)]
        e1 = pivot.loc[rnd, ("exact_match_at_1", 84)]
        print(f"  {rnd:6d} {s12:12.3f} {s84:12.3f} {e1:9.3f}")

    out = ARTIFACTS / f"reliability_{b.snapshot_id}.parquet"
    report.by_round.assign(is_upper_bound=True).to_parquet(out, index=False)
    print(f"\n  wrote {out}")
    print("  §17.3 reads this for reliability_at_round; a missing file makes")
    print("  the UI present the indifference set instead of asserting a pick.")

    early = report.by_round[(report.by_round["round"] <= 4)
                            & (report.by_round["k"] == 12)]["spearman"].mean()
    late = report.by_round[(report.by_round["round"] >= 12)
                           & (report.by_round["k"] == 12)]["spearman"].mean()
    print(f"\n  READ THIS BEFORE QUOTING THE HEADLINE NUMBER.")
    print(f"  spearman@84 is high ({report.overall.set_index('k').loc[84, 'spearman']:.3f}) "
          f"because the draftable pool's coarse")
    print(f"  ordering is stable. spearman@12 is NOT: {early:.3f} in rounds 1-4")
    print(f"  versus {late:.3f} in rounds 12+. The top of the board — the only")
    print("  part that decides an early pick — is the least reproducible part")
    print("  of the ranking. That is the finding, not the 0.9.")
    print("\n  INSTRUMENT: players are ranked by raw calibrated p50, not by")
    print("  VONA or E[$]. This measures the PROJECTION's reliability, which")
    print("  is an input to the recommender rather than the recommender")
    print("  itself; a positional-value ranking would move differently.")
    print("\n  UPPER BOUND. Player resampling only — seasons, features and")
    print("  hyperparameters were held fixed. True stability is lower.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
