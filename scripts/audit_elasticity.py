"""The §13.4 elasticity table — which knobs move E[$] enough to argue about.

    venv/bin/python scripts/audit_elasticity.py [--reps 1024] [--perturb 0.10]

Every arm re-simulates the same board with the same seed root, so the
differences below are paired (common random numbers) and the shared Monte-Carlo
noise cancels. Without that pairing a $0.30 effect would need roughly a hundred
times the replications to see.

Read the output as a research budget: `nail_it` knobs deserve modeling work,
`hardcode_it` knobs deserve a defensible constant and no further discussion,
and `irrelevant` knobs deserve neither.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_league, load_strategy  # noqa: E402
from src.core.config.slots import build_slot_plan  # noqa: E402
from src.domain.payout.compile import compile_payout  # noqa: E402
from src.engine.audit.elasticity import KNOBS, elasticity_table  # noqa: E402
from src.engine.audit.sobol import sobol_indices  # noqa: E402
from src.engine.sim import kernel  # noqa: E402
from src.engine.sim import rng as rng_mod
from src.engine.sim.draws import draw_points  # noqa: E402
from src.platform import bundle as bundle_mod  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import (  # noqa: E402
    build_projection_bundle,
    draft_rosters,
    load_correlation_matrix,
)

MY_SEAT = 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1024)
    ap.add_argument("--perturb", type=float, default=0.10)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--sobol", type=int, default=0, metavar="N",
                    help="also run a §15.5 Sobol decomposition at base sample "
                         "N (costs N*(k+2) simulations; 0 disables)")
    ap.add_argument("--bundle", type=Path,
                    default=Path("data/bundles/draft_night_bundle.parquet"))
    args = ap.parse_args()

    cfg = load_league()
    strategy = load_strategy(cfg)
    weeks = cfg.schedule.regular_season_weeks + len(cfg.schedule.playoff_weeks)
    zone = float(strategy.simulation.decision["indifference_zone_dollars"])

    b = bundle_mod.read(args.bundle)
    board = b.board()
    corr = load_correlation_matrix()
    proj = build_projection_bundle(board, cfg, weeks, args.season)
    plan = build_slot_plan(cfg.roster.slots, cfg.roster.flex_eligibility)
    rosters = draft_rosters(board, cfg, np.random.default_rng(0))
    objective = compile_payout(strategy.payout, cfg)

    # ONE seed root, reused by every arm. This is the CRN contract.
    root = rng_mod.seed_root(b.snapshot_id, cfg.model_version,
                             strategy.strategy_hash)
    masks = kernel.build_masks(proj, rosters, plan, weeks)

    calls = {"n": 0}

    def simulate(bundle, cholesky) -> float:
        calls["n"] += 1
        points = draw_points(bundle, cholesky, root, reps=args.reps)
        team_week = kernel.evaluate_rosters(points, masks)
        outcome = kernel.season_outcome(team_week, cfg, my_team_index=MY_SEAT)
        return float(objective(outcome)[MY_SEAT].mean())

    print("§13.4 ELASTICITY TABLE")
    print(f"  board {b.snapshot_id}  {len(board)} players  seat {MY_SEAT + 1}")
    print(f"  {args.reps} reps/arm, +/-{args.perturb:.0%} perturbation, CRN "
          f"root {root}")
    print(f"  indifference zone ${zone:.2f} (strategy.yaml)\n")

    t0 = time.perf_counter()
    table = elasticity_table(proj, corr.cholesky, simulate,
                             indifference_zone=zone, perturb=args.perturb)
    elapsed = time.perf_counter() - t0

    print(f"  {'knob':22s} {'-10%':>9s} {'base':>9s} {'+10%':>9s} "
          f"{'|move|':>8s}  verdict")
    print(f"  {'-' * 22} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 8}  {'-' * 12}")
    for _, r in table.iterrows():
        print(f"  {r['knob']:22s} {r['minus_dollars']:9.2f} "
              f"{r['base_dollars']:9.2f} {r['plus_dollars']:9.2f} "
              f"{r['max_abs_move']:8.2f}  {r['verdict']}")

    print(f"\n  {calls['n']} simulations in {elapsed:.1f}s")
    for verdict in ("nail_it", "hardcode_it", "irrelevant"):
        hits = table[table["verdict"] == verdict]["knob"].tolist()
        print(f"  {verdict:12s} {', '.join(hits) if hits else '(none)'}")

    ref = table[table["knob"] == "rate_level_scale"]
    if len(ref):
        move = float(ref["max_abs_move"].iat[0])
        print(f"\n  Reference: a 10% shift in the projection itself moves E[$] "
              f"by ${move:.2f}.")
        print("  Any knob far below that is not where the remaining error is.")
    if args.sobol:
        print(f"\n§15.5 SOBOL DECOMPOSITION (base sample {args.sobol})")
        print("  One-at-a-time elasticities report a local slope and cannot")
        print("  see interaction. These decompose the VARIANCE instead.")

        span = args.perturb
        bounds = [(1.0 - span, 1.0 + span)] * len(KNOBS)

        def evaluate_vector(x) -> float:
            bundle, chol = proj, corr.cholesky
            for knob, factor in zip(KNOBS, x, strict=False):
                bundle, chol = knob.apply(bundle, chol, float(factor))
            return simulate(bundle, chol)

        t0 = time.perf_counter()
        sob = sobol_indices([k.name for k in KNOBS], bounds, evaluate_vector,
                            n=args.sobol, seed=0)
        print(f"\n{sob.describe()}")
        print(f"  {sob.n_evaluations} simulations in "
              f"{time.perf_counter() - t0:.1f}s")
        if sob.converged:
            worst = sob.table.iloc[sob.table["interaction"].argmax()]
            print(f"\n  Largest interaction: {worst['knob']} "
                  f"(S={worst['first_order']:+.3f}, S_T={worst['total']:+.3f}).")
            print("  Where that gap is wide, read the elasticity table above")
            print("  with suspicion — its one-at-a-time slope is not the whole")
            print("  effect.")
        else:
            print("\n  Not interpreting the table above.")
            print("  Sobol takes no differences against a common base, so it")
            print("  cannot use CRN the way the elasticity table does — the")
            print("  simulator's own MC error enters the variance directly and")
            print(f"  is charged to 'interaction'. At --reps {args.reps} that")
            print("  noise is comparable to the knob effects themselves.")
            print(f"  Raise BOTH: --reps 4096 --sobol {max(args.sobol * 4, 1024)}.")

    print("\n  Note: the design's original lambda_{QB,RB,WR,TE} knobs do not")
    print("  exist. AD-2 replaced the one-factor loading model with a direct")
    print("  empirical slot matrix; correlation_scale asks the same question")
    print("  of the model that actually shipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
