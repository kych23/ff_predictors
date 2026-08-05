"""Draft-night recommendation across the full degradation ladder (§7.3, §19).

    venv/bin/python scripts/draft_recommend.py --slot 4
    venv/bin/python scripts/draft_recommend.py --slot 4 --taken "Jahmyr Gibbs,Bijan Robinson"

Offline: reads the bundle and the fitted artifacts, nothing else.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.cockpit.ladder import Recommendation, recommend  # noqa: E402
from src.core.config import load_league, load_strategy  # noqa: E402
from src.core.config.slots import build_slot_plan  # noqa: E402
from src.domain.payout.compile import compile_payout  # noqa: E402
from src.engine.decision import recommend as tier2  # noqa: E402
from src.engine.decision import roster_state as rs_mod  # noqa: E402
from src.engine.decision.allocate import allocate  # noqa: E402
from src.engine.decision.board import Board  # noqa: E402
from src.engine.sim import kernel, rng as rng_mod  # noqa: E402
from src.engine.sim.draws import draw_points  # noqa: E402
from src.engine.sim.rollout import rollout  # noqa: E402
from src.models.correlation.slot_matrix import from_prior  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from simulate import build_projection_bundle  # noqa: E402


def build_tiers(board: Board, cfg, strategy, slot: int, reps: int, shortlist: int):
    """Construct the four tiers as closures over shared state."""
    weeks = cfg.schedule.regular_season_weeks + len(cfg.schedule.playoff_weeks)
    frame = board.available().reset_index(drop=True)
    taken_rows = {}
    rs = rs_mod.RosterState(cfg=cfg, draft_position=slot)
    position = len(board.drafted) + 1
    my_pick = next((p for p in rs.my_picks if p >= position), rs.my_picks[-1])
    flags = board.stale_flags(cfg)

    def tier2_fn(_budget: float) -> Recommendation:
        result = tier2.score(frame, rs, cfg, strategy, current_pick=my_pick,
                             stale_flags=tuple(flags))
        live = result.ranked[result.ranked["sink_reason"] == ""]
        leader = live.iloc[0]
        return Recommendation(
            tier=2, leader=str(leader["player_id"]),
            leader_name=str(leader["player_name"]),
            ranked=live.head(10), indifference_set=[str(leader["player_id"])],
            stale_flags=flags + ["tier2_vona_only"],
        )

    def tier3_fn(_budget: float) -> Recommendation:
        ranked = frame.sort_values("adp").head(10)
        leader = ranked.iloc[0]
        return Recommendation(
            tier=3, leader=str(leader["player_id"]),
            leader_name=str(leader["player_name"]), ranked=ranked,
            indifference_set=[str(leader["player_id"])],
            stale_flags=flags + ["static_adp_list"],
        )

    def tier0_fn(budget: float) -> Recommendation:
        deadline = time.perf_counter() + budget
        corr = from_prior(yaml.safe_load(
            Path("config/correlation_prior.yaml").read_text()))
        proj = build_projection_bundle(frame, cfg, weeks)
        plan = build_slot_plan(cfg.roster.slots, cfg.roster.flex_eligibility)
        objective = compile_payout(strategy.payout, cfg)
        root = rng_mod.seed_root(board.snapshot_id, cfg.model_version,
                                 strategy.strategy_hash)

        shortlist_rows = tier2.score(
            frame, rs, cfg, strategy, current_pick=my_pick,
        ).ranked
        live = shortlist_rows[shortlist_rows["sink_reason"] == ""]
        candidates = [str(r) for r in live.head(shortlist).index.tolist()]

        cache: dict[int, np.ndarray] = {}

        def points_for(draw: int) -> np.ndarray:
            if draw not in cache:
                cache[draw] = draw_points(proj, corr.cholesky, root, reps=reps)
            return cache[draw]

        def evaluate(candidate: str, draw: int) -> np.ndarray:
            row = int(candidate)
            res = rollout(frame, cfg, strategy, my_seat=slot - 1,
                          already_taken=taken_rows,
                          forced=(slot - 1, row), root=root, rep=draw)
            masks = kernel.build_masks(proj, res.rosters, plan, weeks)
            team_week = kernel.evaluate_rosters(points_for(draw), masks)
            outcome = kernel.season_outcome(team_week, cfg,
                                            my_team_index=slot - 1)
            return objective(outcome)[slot - 1]

        result = allocate(
            candidates, evaluate,
            initial_draws=strategy.simulation.initial_draws_per_candidate,
            max_draws=strategy.simulation.outer_parameter_draws,
            indifference_zone=float(
                strategy.simulation.decision["indifference_zone_dollars"]),
            deadline=deadline,
        )
        names = {str(i): frame.at[i, "player_name"] for i in frame.index}
        ranked = pd.DataFrame([
            {"player_name": names[c], "position": frame.at[int(c), "position"],
             "adp": frame.at[int(c), "adp"], "E_dollars": e.mean,
             "aleatory_se": e.aleatory_se, "epistemic_se": e.epistemic_se,
             "draws": e.draws_used}
            for c, e in sorted(result.estimates.items(), key=lambda kv: -kv[1].mean)
        ])
        return Recommendation(
            tier=0, leader=result.leader, leader_name=names[result.leader],
            ranked=ranked,
            indifference_set=[names[c] for c in result.indifference_set],
            dollars={c: e.mean for c, e in result.estimates.items()},
            aleatory_se={c: e.aleatory_se for c, e in result.estimates.items()},
            epistemic_se={c: e.epistemic_se for c, e in result.estimates.items()},
            p_best=result.p_best,
            draws_used=max(e.draws_used for e in result.estimates.values()),
            stopped_because=result.stopped_because, stale_flags=flags,
        )

    return {0: tier0_fn, 2: tier2_fn, 3: tier3_fn}, my_pick


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--taken", default="")
    ap.add_argument("--reps", type=int, default=512)
    ap.add_argument("--shortlist", type=int, default=4)
    ap.add_argument("--budget", type=float, default=25.0)
    args = ap.parse_args()

    cfg = load_league()
    strategy = load_strategy(cfg)
    board = Board.from_bundle()
    for name in [x.strip() for x in args.taken.split(",") if x.strip()]:
        hits = board.find(name, limit=1)
        if not hits.empty:
            board.take(hits.iloc[0]["player_id"])

    tiers, my_pick = build_tiers(board, cfg, strategy, args.slot,
                                 args.reps, args.shortlist)
    print(f"seat {args.slot}, on the clock at pick {my_pick}, "
          f"{len(board.available())} available")
    print(f"budget {args.budget:.0f}s, demote after "
          f"{strategy.simulation.decision['demote_to_tier1_after_seconds']}s\n")

    rec = recommend(
        tiers, budget_s=args.budget,
        demote_after_s=float(
            strategy.simulation.decision["demote_to_tier1_after_seconds"]),
    )
    print(rec.describe())
    print()
    cols = [c for c in ("player_name", "position", "adp", "E_dollars",
                        "aleatory_se", "epistemic_se", "draws", "vona_score")
            if c in rec.ranked.columns]
    print(rec.ranked[cols].round(3).to_string(index=False))
    if rec.tier == 0:
        print(f"\n  p(best) {rec.p_best:.2f}   stopped: {rec.stopped_because}   "
              f"draws {rec.draws_used}")


if __name__ == "__main__":
    main()
