"""Draft-night recommendation across the full degradation ladder (§7.3, §19).

    venv/bin/python scripts/draft_recommend.py --slot 4
    venv/bin/python scripts/draft_recommend.py --slot 4 --taken "Jahmyr Gibbs,Bijan Robinson"

Offline: reads the bundle and the fitted artifacts, nothing else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.cockpit.ladder import recommend  # noqa: E402
from src.app.narration import NarrationConfig, narrate  # noqa: E402
from src.core.config import load_league, load_strategy  # noqa: E402
from src.engine.decision.board import Board  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# MOVED: build_tiers -> src/app/cockpit/build.py and _load_posterior ->
# src/engine/sim/bundle_build.py, so the web backend can import them
# (`scripts/` is not a package). Re-exported here so draft_night.py and
# mock_draft.py keep working unchanged.
from src.app.cockpit.build import build_tiers  # noqa: E402,F401
from src.engine.sim.bundle_build import (
    load_posterior as _load_posterior,  # noqa: E402,F401
)


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
        if rec.separating_axis:
            print(f"  separating axis: {rec.separating_axis}")

    # §18 narration. Every quantitative statement is parsed back out and
    # checked against the attribution record before it is printed; anything
    # unentailed falls back to the table, which states the same content and
    # needs no model at all.
    record = getattr(tiers.get(0), "record", None)
    if record is not None:
        narration = narrate(record, NarrationConfig.from_strategy(strategy))
        print(f"\nWHY ({narration.source}"
              f"{'' if narration.verified else ', UNVERIFIED'}):")
        print("  " + narration.text.replace("\n", "\n  "))
        if narration.reason:
            print(f"  [{narration.reason}]")


if __name__ == "__main__":
    main()
