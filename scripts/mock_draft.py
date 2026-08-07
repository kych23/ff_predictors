"""Full-draft rehearsal and chaos harness (§22.1 day 20).

    venv/bin/python scripts/mock_draft.py --seat 4
    venv/bin/python scripts/mock_draft.py --seat 4 --chaos

Plays all 180 picks against the real bundle: eleven ADP bots with Gumbel noise,
and the engine on my seat, driven through the same `Session` the live cockpit
uses. Nothing is stubbed — the same ladder, ledger, spine and narration path.

`--chaos` injects the failures that actually happen at a draft, rather than the
ones that are easy to simulate:

    a name nobody can resolve      typo'd surname mid-draft
    a player taken twice           someone calls a name already gone
    an undo under time pressure    the classic
    an injury scratch              `zero` on a rostered player
    a mid-draft restart            process dies, resume from disk

**The pass criterion is not "no exception".** It is that every one of my picks
came back inside the latency budget and the ledger chain verifies at the end.
A tool that survives by silently doing nothing has not survived.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from draft_recommend import build_tiers  # noqa: E402

from src.app.cockpit.ladder import recommend  # noqa: E402
from src.app.cockpit.ledger import DecisionLedger  # noqa: E402
from src.app.cockpit.session import Session  # noqa: E402
from src.core.config import load_league, load_strategy  # noqa: E402
from src.core.errors import DataError  # noqa: E402
from src.engine.decision.board import Board  # noqa: E402


def bot_order(frame, rng, noise: float) -> list[str]:
    """ADP with Gumbel noise — a Plackett-Luce sampler, so each rehearsal is a
    different plausible draft rather than the same snake every time."""
    adp = frame["adp"].fillna(9999.0).to_numpy(dtype=float)
    score = -adp + rng.gumbel(0.0, noise, len(adp))
    ids = frame["player_id"].astype(str).to_numpy()
    return list(ids[np.argsort(-score)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seat", type=int, default=4)
    ap.add_argument("--reps", type=int, default=256)
    ap.add_argument("--shortlist", type=int, default=3)
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--noise", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chaos", action="store_true")
    ap.add_argument("--workdir", type=Path, default=Path("data/rehearsal"))
    args = ap.parse_args()

    cfg = load_league()
    strategy = load_strategy(cfg)
    board = Board.from_bundle()
    frame = board.players.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    rng = np.random.default_rng(args.seed)

    args.workdir.mkdir(parents=True, exist_ok=True)
    session_path = args.workdir / "session.json"
    ledger_path = args.workdir / "ledger.sqlite"
    session_path.unlink(missing_ok=True)
    ledger_path.unlink(missing_ok=True)

    session = Session(my_seat=args.seat - 1, teams=cfg.teams,
                      rounds=cfg.roster.rounds,
                      snapshot_id=board.snapshot_id, path=session_path)
    ledger = DecisionLedger(ledger_path)
    order = bot_order(frame, rng, args.noise)
    budget = float(strategy.simulation.decision["latency_budget_seconds"])

    print(f"MOCK DRAFT — seat {args.seat}, {cfg.teams}x{cfg.roster.rounds}, "
          f"{'CHAOS' if args.chaos else 'clean'}")
    print(f"  bundle {board.snapshot_id}, {len(frame)} players, "
          f"latency budget {budget:.0f}s\n")

    my_latencies: list[float] = []
    tiers_used: dict[int, int] = {}
    injected: list[str] = []
    restarts = 0

    while not session.is_complete:
        pick_no = session.state.pick_number

        # ---- chaos injections, at picks nobody would choose to be unlucky on
        if args.chaos and pick_no == 7 and "unresolved" not in injected:
            session.record_unresolved("Jauan Jennngz")
            injected.append("unresolved")
            print(f"  [chaos] pick {pick_no}: unresolvable name recorded")
        if args.chaos and pick_no == 15 and "duplicate" not in injected:
            try:
                session.record_pick(session.state.drafted[0])
                print("  [chaos] FAILED: a duplicate pick was accepted")
                return 1
            except DataError:
                injected.append("duplicate")
                print(f"  [chaos] pick {pick_no}: duplicate refused, log clean")
        if args.chaos and pick_no == 23 and "restart" not in injected:
            session = Session.load(session_path)
            restarts += 1
            injected.append("restart")
            print(f"  [chaos] pick {pick_no}: reloaded from disk "
                  f"({len(session.events)} events replayed)")
        if (args.chaos and pick_no == 40 and "zero" not in injected
                and session.state.my_roster):
                session.zero_player(session.state.my_roster[0])
                injected.append("zero")
                print(f"  [chaos] pick {pick_no}: my RB1 scratched")

        if session.is_my_turn():
            scratch = Board.from_bundle()
            for pid in session.state.drafted:
                try:
                    scratch.take(pid)
                except Exception:      # noqa: BLE001
                    pass
            built, _ = build_tiers(scratch, cfg, strategy, args.seat,
                                   args.reps, args.shortlist,
                                   my_roster=list(session.state.my_roster),
                                   by_seat={k: list(v) for k, v
                                            in session.state.by_seat.items()})
            start = time.perf_counter()
            rec = recommend(
                built, budget_s=args.budget,
                demote_after_s=float(
                    strategy.simulation.decision["demote_to_tier1_after_seconds"]),
            )
            elapsed = time.perf_counter() - start
            my_latencies.append(elapsed)
            tiers_used[rec.tier] = tiers_used.get(rec.tier, 0) + 1

            choice = rec.leader
            if choice in session.state.drafted or choice not in set(frame["player_id"]):
                choice = next(p for p in order
                              if p not in session.state.drafted)
            session.record_my_pick(choice)
            ledger.append("rehearsal", pick_no=pick_no, tier=rec.tier,
                          recommendation={"leader": rec.leader,
                                          "leader_name": rec.leader_name},
                          actual_pick=choice, snapshot_id=board.snapshot_id)
            flag = "" if elapsed <= budget else "  <-- OVER BUDGET"
            print(f"  pick {pick_no:3d} (mine)  tier {rec.tier}  "
                  f"{rec.leader_name[:22]:22s} {elapsed:5.1f}s{flag}")

            if args.chaos and pick_no > 40 and "undo" not in injected:
                session.undo()
                injected.append("undo")
                print(f"  [chaos] pick {pick_no}: undone, re-picking")
                continue
        else:
            pick = next((p for p in order if p not in session.state.drafted),
                        None)
            if pick is None:
                break
            session.record_pick(pick)

    print(f"\n  {len(session.state.drafted)} picks recorded, "
          f"{len(session.state.my_roster)} mine")
    roster = frame[frame["player_id"].isin(session.state.my_roster)]
    print("  my roster: " + ", ".join(
        f"{r.player_name} ({r.position})" for r in roster.itertuples()))

    ok = True
    print("\n  LATENCY")
    if my_latencies:
        over = [t for t in my_latencies if t > budget]
        print(f"    median {np.median(my_latencies):.1f}s   "
              f"max {max(my_latencies):.1f}s   budget {budget:.0f}s")
        print(f"    over budget: {len(over)} of {len(my_latencies)}")
        ok = ok and not over
    print("  TIERS USED: "
          + ", ".join(f"tier {t}: {n}" for t, n in sorted(tiers_used.items())))

    verdict = ledger.verify()
    print(f"  LEDGER: {verdict.detail}")
    ok = ok and verdict.ok
    if args.chaos:
        print(f"  CHAOS INJECTED: {', '.join(injected) or 'none'}")
        expected = {"unresolved", "duplicate", "restart", "zero", "undo"}
        missing = expected - set(injected)
        if missing:
            print(f"    NOT EXERCISED: {sorted(missing)} — the rehearsal did "
                  f"not actually test these")
            ok = False
        print(f"  RESTARTS SURVIVED: {restarts}")

    ledger.close()
    print(f"\n  {'REHEARSAL PASSED' if ok else 'REHEARSAL FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
