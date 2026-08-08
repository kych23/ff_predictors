"""Regenerate tests/fixtures/parity_golden.json against the current bundle.

The golden pins one recommendation — the leader, its tier, and the dollar
estimates of the shortlist — so that a refactor which silently changes an
answer fails loudly. It carries the `snapshot_id` it was produced against, so
rebuilding the bundle makes `test_refactor_parity.py` SKIP rather than pass by
luck. That skip is the signal to run this.

Regenerate only when the projections legitimately changed and you have decided
the new answer is the correct one. Regenerating to make a red test green
destroys the entire value of the fixture — read the diff this prints before
accepting it.

    venv/bin/python scripts/regen_parity_golden.py --note "<why>"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "tests" / "fixtures" / "parity_golden.json"

# The scenario. These mirror the fixture in tests/test_refactor_parity.py and
# must not drift from it — the golden is only meaningful if the replay runs
# the same draft.
SEAT = 4
REPS = 512
SHORTLIST = 6
BUDGET_S = 60.0
DEMOTE_AFTER_S = 55.0
PICKS = 27


def seat_of(i: int, teams: int) -> int:
    """Snake seat for the i-th overall pick (0-indexed)."""
    rnd, idx = divmod(i, teams)
    return idx if rnd % 2 == 0 else teams - 1 - idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True,
                    help="why the golden is being regenerated")
    ap.add_argument("--picks", type=int, default=PICKS)
    args = ap.parse_args()

    from src.app.cockpit.build import build_tiers
    from src.app.cockpit.ladder import recommend
    from src.core.config import load_league, load_strategy
    from src.engine.decision.board import Board
    from src.models.artifacts import code_digest

    cfg = load_league()
    strategy = load_strategy(cfg)
    board = Board.from_bundle()

    # REUSE the existing scenario. Regenerating the opening as well as the
    # answer would change two things at once, and the printed diff would no
    # longer isolate what the projections did. Only fall back to an ADP
    # opening when there is no golden to inherit from.
    previous = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else None
    if previous and previous.get("drafted"):
        opening = [str(p) for p in previous["drafted"]]
        missing = [p for p in opening
                   if p not in set(board.players["player_id"].astype(str))]
        if missing:
            raise SystemExit(
                f"{len(missing)} of the golden's drafted players are absent "
                f"from the rebuilt bundle ({missing[:3]}...). The scenario "
                f"cannot be replayed; pick a new one deliberately.")
    else:
        opening = (board.players.sort_values("adp")
                   .head(args.picks)["player_id"].astype(str).tolist())

    by_seat: dict[int, list[str]] = {}
    for i, pid in enumerate(opening):
        by_seat.setdefault(seat_of(i, cfg.teams), []).append(pid)
        board.take(pid)

    tiers, _ = build_tiers(board, cfg, strategy, SEAT, REPS, SHORTLIST,
                           my_roster=list(by_seat.get(SEAT - 1, [])),
                           by_seat=by_seat)
    result = recommend(tiers, budget_s=BUDGET_S, demote_after_s=DEMOTE_AFTER_S)

    if result.stopped_because == "deadline":
        raise SystemExit(
            "run hit the budget — a truncated run makes the leader arbitrary "
            "and would pin noise. Re-run on a quieter machine.")

    fresh = {
        "_note": args.note,
        "snapshot_id": board.snapshot_id,
        "code_digest": code_digest(),
        "seat": SEAT, "reps": REPS, "shortlist": SHORTLIST,
        "budget_s": BUDGET_S,
        "drafted": opening,
        "tier": result.tier,
        "leader": result.leader,
        "leader_name": result.leader_name,
        "stopped_because": result.stopped_because,
        "dollars": {k: float(v) for k, v in sorted(result.dollars.items())},
    }

    if previous is not None:
        old = previous
        print(f"  snapshot  {old['snapshot_id']} -> {fresh['snapshot_id']}")
        print(f"  code      {old.get('code_digest')} -> {fresh['code_digest']}")
        print(f"  tier      {old['tier']} -> {fresh['tier']}")
        print(f"  leader    {old['leader_name']!r} -> {fresh['leader_name']!r}")
        if old["leader"] != fresh["leader"]:
            print("  ** THE RECOMMENDATION CHANGED. Confirm that is intended "
                  "before committing this fixture. **")
        for pid, value in fresh["dollars"].items():
            was = old["dollars"].get(pid)
            print(f"    {pid}  {was if was is None else f'{was:8.3f}'} -> "
                  f"{value:8.3f}")

    GOLDEN.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
    print(f"\n  wrote {GOLDEN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
