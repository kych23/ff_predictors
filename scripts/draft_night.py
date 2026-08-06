"""The draft-night cockpit (§19). Fully offline.

    venv/bin/python scripts/draft_night.py --seat 4
    venv/bin/python scripts/draft_night.py --seat 4 --resume

Type a name to record a pick by whoever is on the clock. The board rebuilds by
replay after every command, so `undo` always works and closing the terminal
costs nothing.

COMMANDS
    <name>            record the pick on the clock (fuzzy matched)
    me <name>         record MY pick
    go                recommend now (also automatic when I'm on the clock)
    board [n]         print the top n available
    roster            print my roster
    zero <name>       injury/news: this player is worthless, no restart
    adp <name> <n>    override a player's ADP
    undo              undo the last command
    log               show the decision ledger
    quit

Nothing here touches the network: names resolve through the local `Spine`, the
board is a parquet bundle, the ledger is local SQLite, and narration runs on a
local model. That is §19's requirement expressed as a set of imports.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.app.cockpit.ledger import DecisionLedger  # noqa: E402
from src.app.cockpit.session import Session, available  # noqa: E402
from src.core.config import load_league, load_strategy  # noqa: E402
from src.core.errors import DataError  # noqa: E402
from src.engine.decision.board import Board  # noqa: E402
from src.platform.identity.spine import Spine  # noqa: E402


def build_spine(board: Board) -> Spine:
    frame = board.players.rename(columns={"player_name": "name"})
    return Spine(frame[["player_id", "name", "position", "team"]],
                 snapshot_id=board.snapshot_id)


def resolve(spine: Spine, text: str, board_frame: pd.DataFrame):
    """Spine first, then a plain substring scan over the board.

    The substring pass exists because the spine is built from the bundle and a
    typo'd surname is more common at a live draft than a genuinely unknown
    player. An ambiguous substring returns the candidates so the operator picks
    rather than the tool guessing.
    """
    hit = spine.resolve(text)
    if hit is not None:
        return hit.player_id, hit.name, hit.tier, []
    lowered = text.strip().lower()
    matches = board_frame[
        board_frame["player_name"].str.lower().str.contains(lowered, regex=False)]
    if len(matches) == 1:
        row = matches.iloc[0]
        return str(row["player_id"]), str(row["player_name"]), "substring", []
    return None, None, None, [str(r) for r in matches["player_name"].head(6)]


def print_board(frame: pd.DataFrame, state, n: int = 12) -> None:
    live = frame[~frame["player_id"].astype(str).isin(set(state.drafted))]
    live = live.sort_values("adp").head(n)
    print(f"  {'#':>3s} {'player':24s} {'pos':4s} {'adp':>6s}")
    for i, (_, row) in enumerate(live.iterrows(), 1):
        mark = "  (zeroed)" if str(row["player_id"]) in state.zeroed else ""
        print(f"  {i:3d} {str(row['player_name'])[:24]:24s} "
              f"{str(row['position']):4s} {row['adp']:6.1f}{mark}")


def recommend_now(session: Session, board: Board, cfg, strategy, args):
    """Run the ladder for my current pick. Imported lazily so a slow model
    import never delays the first keystroke of a draft."""
    from draft_recommend import build_tiers
    from src.app.cockpit.ladder import recommend

    scratch = Board.from_bundle()
    for pid in session.state.drafted:
        try:
            scratch.take(pid)
        except Exception:      # noqa: BLE001 — a stale id must not stop a pick
            pass

    tiers, my_pick = build_tiers(scratch, cfg, strategy, args.seat,
                                 args.reps, args.shortlist,
                                 my_roster=list(session.state.my_roster),
                                 by_seat={k: list(v) for k, v
                                          in session.state.by_seat.items()})
    start = time.perf_counter()
    rec = recommend(
        tiers, budget_s=args.budget,
        demote_after_s=float(
            strategy.simulation.decision["demote_to_tier1_after_seconds"]),
    )
    print(f"\n{rec.describe()}")
    cols = [c for c in ("player_name", "position", "adp", "E_dollars",
                        "aleatory_se", "epistemic_se", "draws", "vona_score")
            if c in rec.ranked.columns]
    if cols:
        print(rec.ranked[cols].round(2).head(6).to_string(index=False))
    if rec.separating_axis:
        print(f"  separating axis: {rec.separating_axis}")

    record = getattr(tiers.get(0), "record", None)
    if record is not None:
        from src.app.narration import NarrationConfig, narrate
        narration = narrate(record, NarrationConfig.from_strategy(strategy))
        tag = narration.source + ("" if narration.verified else ", UNVERIFIED")
        print(f"\nWHY ({tag}):")
        print("  " + narration.text.replace("\n", "\n  "))
    print(f"  [{time.perf_counter() - start:.1f}s]")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seat", type=int, required=True, help="1-indexed")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reps", type=int, default=512)
    ap.add_argument("--shortlist", type=int, default=4)
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--session", type=Path, default=Path("data/draft.json"))
    ap.add_argument("--ledger", type=Path, default=Path("data/draft.sqlite"))
    args = ap.parse_args()

    cfg = load_league()
    strategy = load_strategy(cfg)
    board = Board.from_bundle()
    frame = board.players.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    spine = build_spine(board)
    name_of = dict(zip(frame["player_id"], frame["player_name"].astype(str)))

    session = Session(my_seat=args.seat - 1, teams=cfg.teams,
                      rounds=cfg.roster.rounds,
                      snapshot_id=board.snapshot_id, path=args.session)
    if args.resume:
        session = session.resume_or_new()
    elif args.session.exists():
        print(f"a session already exists at {args.session}. Pass --resume to "
              f"continue it, or delete it to start over.")
        return 1

    ledger = DecisionLedger(args.ledger)
    last_rec = None

    print(f"DRAFT NIGHT — seat {args.seat}, {cfg.teams} teams x "
          f"{cfg.roster.rounds} rounds")
    print(f"  bundle {board.snapshot_id}, {len(frame)} players")
    print(f"  session {args.session}   ledger {args.ledger}")
    print("  type a name to record a pick; `quit` to stop\n")

    while True:
        if session.is_complete:
            print("\ndraft complete.")
            break
        print(f"\n[{session.describe_clock()}]")
        if session.is_my_turn() and last_rec is None:
            last_rec = recommend_now(session, board, cfg, strategy, args)

        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nsaved. resume with --resume")
            break
        if not raw:
            continue

        head, _, rest = raw.partition(" ")
        head = head.lower()

        try:
            if head in ("quit", "exit"):
                print("saved. resume with --resume")
                break
            if head == "undo":
                session.undo()
                last_rec = None
                print(f"  undone. {session.describe_clock()}")
                continue
            if head == "board":
                print_board(frame, session.state,
                            int(rest) if rest.strip().isdigit() else 12)
                continue
            if head == "roster":
                mine = frame[frame["player_id"].isin(session.state.my_roster)]
                print(mine[["player_name", "position", "adp"]]
                      .to_string(index=False) if len(mine) else "  (empty)")
                continue
            if head == "go":
                last_rec = recommend_now(session, board, cfg, strategy, args)
                continue
            if head == "log":
                # Resolve ids back to names: the ledger stores player_ids so it
                # stays valid across bundle rebuilds, but an operator reading
                # the log mid-draft needs names.
                for entry in ledger:
                    took = (name_of.get(entry.actual_pick, entry.actual_pick)
                            if entry.actual_pick else "(pending)")
                    print(f"  pick {entry.pick_no:3d}  tier {entry.tier}  "
                          f"{entry.recommendation.get('leader_name', '?')} -> "
                          f"{took}")
                verdict = ledger.verify()
                print(f"  {verdict.detail}")
                continue

            if head in ("me", "zero", "adp"):
                target, _, tail = rest.partition(" ") if head == "adp" else (rest, "", "")
                name = rest if head != "adp" else rest.rsplit(" ", 1)[0]
                pid, display, tier, options = resolve(spine, name, frame)
            else:
                pid, display, tier, options = resolve(spine, raw, frame)

            if pid is None:
                if options:
                    print(f"  ambiguous — did you mean: {', '.join(options)}")
                else:
                    print(f"  no match for {raw!r}. Recorded as unresolved so "
                          f"the board does not keep offering him; add him to "
                          f"config/id_overrides.yaml if this recurs.")
                    session.record_unresolved(raw)
                continue
            if tier == "fuzzy":
                print(f"  (fuzzy match -> {display})")

            if head == "zero":
                session.zero_player(pid)
                print(f"  {display} zeroed")
            elif head == "adp":
                session.override_adp(pid, float(raw.rsplit(" ", 1)[1]))
                print(f"  {display} ADP overridden")
            elif head == "me":
                _record_my_pick(session, ledger, pid, display, last_rec, board)
                last_rec = None
            else:
                on_clock = session.on_the_clock()
                if on_clock == session.my_seat:
                    _record_my_pick(session, ledger, pid, display, last_rec,
                                    board)
                else:
                    session.record_pick(pid, raw_input=raw)
                    print(f"  seat {on_clock + 1}: {display}")
                last_rec = None

        except DataError as exc:
            print(f"  {exc}")
        except ValueError as exc:
            print(f"  bad input: {exc}")

    ledger.close()
    return 0


def _record_my_pick(session, ledger, pid, display, last_rec, board) -> None:
    pick_no = session.state.pick_number
    session.record_my_pick(pid, raw_input=display)
    print(f"  YOU: {display}")
    # Ledger the recommendation ALONGSIDE the pick, so `followed` is derived
    # rather than remembered (§20.2).
    payload = ({"leader": last_rec.leader, "leader_name": last_rec.leader_name,
                "dollars": {k: round(v, 3)
                            for k, v in last_rec.dollars.items()},
                "indifference_set": list(last_rec.indifference_set)}
               if last_rec is not None else {"leader": None})
    try:
        ledger.append(session.snapshot_id or "session", pick_no=pick_no,
                      tier=last_rec.tier if last_rec else 3,
                      recommendation=payload, actual_pick=pid,
                      snapshot_id=board.snapshot_id)
    except Exception as exc:      # noqa: BLE001 — a ledger fault must not cost a pick
        print(f"  (ledger write failed: {exc})")


if __name__ == "__main__":
    sys.exit(main())
