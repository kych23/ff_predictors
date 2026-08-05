"""End-to-end season simulation and E[$] (§15, §16).

    venv/bin/python scripts/simulate.py [--reps 2048]

Assembles a ProjectionBundle from the draft bundle plus the fitted artifacts,
draws a season, scores it through the payout DSL, and reports expected dollars
decomposed by prize.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_league, load_strategy  # noqa: E402
from src.core.config.slots import build_slot_plan  # noqa: E402
from src.domain.payout.compile import compile_payout  # noqa: E402
from src.engine.sim import kernel, rng as rng_mod  # noqa: E402
from src.engine.sim.draws import ProjectionBundle, draw_points  # noqa: E402
from src.models.correlation.slot_matrix import DEFAULT_SLOTS, from_prior  # noqa: E402
from src.models.weekly.variance import WeeklySigmaModel  # noqa: E402
from src.platform import bundle as bundle_mod  # noqa: E402

ARTIFACTS = Path("data/artifacts")
#: depth-chart rank -> correlation slot, by position
SLOT_FOR = {"QB": ["QB1"], "RB": ["RB1", "RB2"],
            "WR": ["WR1", "WR2", "WR3"], "TE": ["TE1"]}


def _latest(pattern: str):
    hits = sorted(ARTIFACTS.glob(pattern))
    return hits[-1] if hits else None


def build_projection_bundle(board: pd.DataFrame, cfg, weeks: int) -> ProjectionBundle:
    """Map the draft board onto the simulator's array contract."""
    sigma_path = _latest("weekly_sigma_*.json")
    if sigma_path is None:
        raise SystemExit("no weekly-sigma artifact; run scripts/fit_models.py")
    sigma_model = WeeklySigmaModel.from_dict(json.loads(sigma_path.read_text()))

    df = board.reset_index(drop=True).copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    # slot identity within NFL team, by projected value — the unit the
    # correlation matrix is estimated over
    df["rank_in_team"] = (df.groupby(["team", "position"])["value"]
                            .rank(ascending=False, method="first"))
    slot_index = {s: i for i, s in enumerate(DEFAULT_SLOTS)}
    slots = []
    for _, row in df.iterrows():
        names = SLOT_FOR.get(row["position"], [])
        k = int(row["rank_in_team"]) - 1
        slots.append(slot_index[names[k]] if 0 <= k < len(names) else -1)

    is_kdst = df["position"].isin(["K", "DST"]).to_numpy()
    value = df["value"].to_numpy()
    sigma = sigma_model.predict(df["position"].to_numpy(), value)
    sigma = np.where(is_kdst, 0.0, sigma)

    n = len(df)
    grid = np.linspace(0.0, 1.0, 101)
    kdst_q = np.tile(np.full(101, np.nan), (n, 1))
    kdst_path = _latest("kdst_*.json")
    if kdst_path is not None:
        kdst = json.loads(kdst_path.read_text())["distributions"]
        for i, pos in enumerate(df["position"]):
            if pos in kdst:
                kdst_q[i] = np.asarray(kdst[pos]["quantiles"])

    return ProjectionBundle(
        player_ids=df["player_id"].to_numpy(),
        positions=df["position"].to_numpy(),
        nfl_teams=df["team"].fillna("FA").to_numpy(),
        slot_idx=np.array(slots),
        bye_weeks=pd.to_numeric(df["bye_week"], errors="coerce").fillna(0)
                    .astype(int).to_numpy(),
        is_kdst=is_kdst,
        rate_p10=np.maximum(value - 1.28 * sigma / np.sqrt(17), 0.0),
        rate_p50=value,
        rate_p90=value + 1.28 * sigma / np.sqrt(17),
        weekly_sigma=sigma,
        games_hazard=np.full((n, weeks), 0.93),
        kdst_quantiles=kdst_q,
    )


def draft_rosters(board: pd.DataFrame, cfg, rng) -> list[np.ndarray]:
    """A plausible 12-team board: snake by ADP, which is the tier-2 opponent
    model the §14.2 gate left us with."""
    order = board.sort_values("adp").index.to_numpy()
    rosters = [[] for _ in range(cfg.teams)]
    pick = 0
    for rnd in range(cfg.roster.rounds):
        seats = range(cfg.teams) if rnd % 2 == 0 else reversed(range(cfg.teams))
        for seat in seats:
            if pick < len(order):
                rosters[seat].append(order[pick])
                pick += 1
    return [np.array(r) for r in rosters]


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

    corr = from_prior(__import__("yaml").safe_load(
        Path("config/correlation_prior.yaml").read_text()))
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
