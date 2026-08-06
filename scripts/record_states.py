"""Record draft states for the §21.4 reliability study and the §14.2 gate.

    venv/bin/python scripts/record_states.py [--n 200] [--seed 0]

Samples board states from mock drafts and writes
`data/artifacts/states_<snapshot_id>.parquet`. A "state" is everything a
recommender needs to make one pick: whose turn it is, what is gone, and what
the roster already holds.

States are sampled across ROUNDS on purpose. Reliability is not one number —
round 1 is nearly deterministic (the engine and ADP agree) while the middle
rounds are where the ordering is genuinely uncertain, and an average over a
badly spread sample would hide exactly that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_league, load_strategy  # noqa: E402
from src.platform import bundle as bundle_mod  # noqa: E402

ARTIFACTS = Path("data/artifacts")


def simulate_draft(board: pd.DataFrame, cfg, rng: np.random.Generator,
                   *, noise: float) -> list[dict]:
    """One ADP-following mock with Gumbel noise, recording every pick.

    Gumbel noise on the ADP score is a Plackett-Luce sampler: it produces a
    different plausible draft each run rather than the same snake order, which
    is what makes 200 states 200 states instead of 15 repeated fifteen times.
    """
    adp = pd.to_numeric(board["adp"], errors="coerce").fillna(9999.0).to_numpy()
    score = -adp + rng.gumbel(0.0, noise, len(adp))
    order = np.argsort(-score)

    rosters: list[list[int]] = [[] for _ in range(cfg.teams)]
    drafted: list[int] = []
    states = []
    cursor = 0
    for rnd in range(cfg.roster.rounds):
        seats = range(cfg.teams) if rnd % 2 == 0 else reversed(range(cfg.teams))
        for seat in seats:
            if cursor >= len(order):
                break
            states.append({
                "round": rnd + 1,
                "seat": int(seat),
                "pick_overall": len(drafted) + 1,
                "drafted": json.dumps([str(board["player_id"].iloc[i])
                                       for i in drafted]),
                "roster": json.dumps([str(board["player_id"].iloc[i])
                                      for i in rosters[seat]]),
            })
            pick = int(order[cursor])
            cursor += 1
            rosters[seat].append(pick)
            drafted.append(pick)
    return states


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=6.0,
                    help="Gumbel scale on the ADP score, in ADP ranks")
    ap.add_argument("--bundle", type=Path,
                    default=Path("data/bundles/draft_night_bundle.parquet"))
    args = ap.parse_args()

    cfg = load_league()
    load_strategy(cfg)                       # validates the pair before writing
    b = bundle_mod.read(args.bundle)
    board = b.board()
    rng = np.random.default_rng(args.seed)

    print(f"recording {args.n} draft states from board {b.snapshot_id}")
    pool: list[dict] = []
    drafts = 0
    while len(pool) < args.n * 3:
        pool.extend(simulate_draft(board, cfg, rng, noise=args.noise))
        drafts += 1

    frame = pd.DataFrame(pool)
    # Stratify by round so every round is represented rather than whichever
    # rounds a uniform sample happened to hit.
    per_round = max(1, round(args.n / cfg.roster.rounds))
    sampled = pd.concat(
        [g.sample(min(len(g), per_round), random_state=args.seed)
         for _, g in frame.groupby("round", sort=True)],
        ignore_index=True,
    ).head(args.n).reset_index(drop=True)
    sampled["state_id"] = [f"s{i:04d}" for i in range(len(sampled))]
    sampled["snapshot_id"] = b.snapshot_id

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"states_{b.snapshot_id}.parquet"
    sampled.to_parquet(out, index=False)

    counts = sampled.groupby("round").size()
    print(f"  {drafts} mock drafts -> {len(frame):,} states, sampled "
          f"{len(sampled)}")
    print(f"  per round: {', '.join(f'r{r}:{n}' for r, n in counts.items())}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
