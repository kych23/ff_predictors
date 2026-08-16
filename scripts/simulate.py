"""End-to-end season simulation and E[$] (§15, §16).

    venv/bin/python scripts/simulate.py [--reps 2048]

Assembles a ProjectionBundle from the draft bundle plus the fitted artifacts,
draws a season, scores it through the payout DSL, and reports expected dollars
decomposed by prize.
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
from src.engine.sim import kernel  # noqa: E402
from src.engine.sim import rng as rng_mod

# MOVED to src/engine/sim/bundle_build.py so a web backend can import them —
# `scripts/` is not a package. Re-exported here so every existing importer
# (draft_recommend, audit_elasticity, audit_objective) keeps working unchanged.
from src.engine.sim.bundle_build import (  # noqa: E402,F401
    ARTIFACTS,
    SLOT_FOR,
    build_projection_bundle,
    draft_rosters,
    hazard_matrix,
    load_correlation_matrix,
    nflverse_covariates,
)
from src.engine.sim.draws import draw_points  # noqa: E402
from src.platform import bundle as bundle_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2048)
    ap.add_argument("--bundle", type=Path,
                    default=Path("data/bundles/draft_night_bundle.parquet"))
    args = ap.parse_args()

    cfg = load_league()
    strategy = load_strategy(cfg)
    weeks = cfg.schedule.regular_season_weeks + len(cfg.schedule.playoff_weeks)

    b = bundle_mod.read(args.bundle)
    board = b.board()
    print(f"bundle {b.snapshot_id}  value_source={b.value_source}  "
          f"{len(board)} players")

    corr = load_correlation_matrix()
    print(f"correlation: {corr.source}, min eigenvalue {corr.min_eigenvalue:.3f}")

    proj = build_projection_bundle(board, cfg, weeks)
    plan = build_slot_plan(cfg.roster.slots, cfg.roster.flex_eligibility)
    rosters = draft_rosters(board, cfg, np.random.default_rng(0))
    print(f"rosters: {cfg.teams} teams x {len(rosters[0])} picks")

    root = rng_mod.seed_root(b.snapshot_id, cfg.model_version,
                             strategy.strategy_hash)
    print(f"seed root {root}")

    t0 = time.perf_counter()
    points = draw_points(proj, corr.cholesky, root, reps=args.reps)
    t_draw = time.perf_counter() - t0

    t0 = time.perf_counter()
    masks = kernel.build_masks(proj, rosters, plan, weeks)
    t_mask = time.perf_counter() - t0

    t0 = time.perf_counter()
    team_week = kernel.evaluate_rosters(points, masks)
    team_week = kernel.apply_starter_substitution(
        points, masks, team_week, rosters, proj, cfg, strategy)
    team_week = kernel.apply_waiver_floor(points, masks, team_week, rosters,
                                          proj, cfg, strategy)
    t_eval = time.perf_counter() - t0

    t0 = time.perf_counter()
    outcome = kernel.season_outcome(team_week, cfg, my_team_index=3)
    objective = compile_payout(strategy.payout, cfg)
    dollars = objective(outcome)
    parts = objective.decompose(outcome)
    t_obj = time.perf_counter() - t0

    print(f"\ntiming @ {args.reps} reps, {proj.n_players} players, {weeks} weeks")
    print(f"  draws          {t_draw*1000:8.0f} ms")
    print(f"  lineup masks   {t_mask*1000:8.0f} ms   (once per roster-week)")
    print(f"  evaluate       {t_eval*1000:8.0f} ms")
    print(f"  objective      {t_obj*1000:8.0f} ms")
    print(f"  tensor         {points.nbytes/1e6:8.1f} MB")

    pot = strategy.payout.pot(cfg.teams)
    total = dollars.sum(axis=0)
    print(f"\npot conservation: paid {total.mean():.2f} of {pot:.2f} "
          f"(min {total.min():.2f}, max {total.max():.2f})")

    print("\nE[$] by team (seat 4 = mine):")
    for t in range(cfg.teams):
        mark = " <-- me" if t == 3 else ""
        print(f"  seat {t+1:2d}: {dollars[t].mean():7.2f} "
              f"+/- {dollars[t].std()/np.sqrt(args.reps):.2f}{mark}")

    print("\nmy E[$] by prize:")
    for pid, arr in parts.items():
        print(f"  {pid:14s} {arr[3].mean():7.2f}")
    print(f"  {'TOTAL':14s} {dollars[3].mean():7.2f}   "
          f"(baseline {pot/cfg.teams:.2f})")


if __name__ == "__main__":
    main()
