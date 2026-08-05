"""Tier-2 recommendation from the offline bundle (§7.3, §19).

The working tool. No network, no database — reads
`draft_night_bundle.parquet` and nothing else.

    venv/bin/python scripts/recommend.py --slot 4
    venv/bin/python scripts/recommend.py --slot 4 --taken "Jahmyr Gibbs,Bijan Robinson"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_league, load_strategy  # noqa: E402
from src.engine.decision import recommend as rec_mod  # noqa: E402
from src.engine.decision import roster_state  # noqa: E402
from src.engine.decision.board import DEFAULT_BUNDLE, Board  # noqa: E402


def recommend(slot: int, taken: list[str], mine: list[str], bundle_path: Path,
              top: int = 10):
    league = load_league()
    strategy = load_strategy(league)
    board = Board.from_bundle(bundle_path)

    for name in taken + mine:
        hits = board.find(name, limit=1)
        if hits.empty:
            print(f"  ! could not resolve {name!r} — skipped")
            continue
        board.take(hits.iloc[0]["player_id"])

    rs = roster_state.RosterState(cfg=league, draft_position=slot)
    for name in mine:
        hits = board.players[board.players["player_name"].str.lower() == name.lower()]
        if not hits.empty:
            row = hits.iloc[0]
            rs.record_pick(row["player_id"], row["position"], mine=True,
                           team=row.get("team"), bye_week=row.get("bye_week"))

    avail = board.available()
    # The board is at `position`; MY pick is the first of my picks at or after
    # it. Using the board position directly would ask "what should I take at
    # pick 1" from seat 4, and collapse the wait horizon onto the same pick.
    position = len(board.drafted) + 1
    current = next((p for p in rs.my_picks if p >= position), rs.my_picks[-1])
    result = rec_mod.score(avail, rs, league, strategy, current_pick=current,
                           stale_flags=tuple(board.stale_flags(league)))
    scored, levels, nxt = result.ranked, result.replacement, result.next_pick

    print(f"\n  snapshot {board.snapshot_id}   built {board.built_at}")
    print(f"  value source: {board.value_source}")
    flags = board.stale_flags(league)
    if flags:
        print(f"  stale flags : {', '.join(flags)}")
    print(f"  tier 2 (VONA + log-normal survival) — the simulator is not built yet")
    print(f"\n  board: {len(avail)} available of {len(board.players)}")
    print(f"  my seat {slot}, picks {rs.my_picks[:4]}...  on the clock at {current}, next turn {nxt}")
    print(f"  replacement: " +
          "  ".join(f"{p}={levels.get(p):.1f}" for p in league.roster.modeled_positions))
    print()
    cols = ["player_name", "position", "team", "adp", "value", "marginal",
            "wait_term", "vona_score"]
    scored = scored[scored["sink_reason"] == ""] if "sink_reason" in scored else scored
    out = scored.head(top)[cols].copy()
    out["adp"] = out["adp"].round(1)
    for c in ("value", "marginal", "wait_term", "vona_score"):
        out[c] = out[c].round(2)
    print(out.to_string(index=False))
    return scored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", type=int, required=True, help="my draft slot, 1-indexed")
    ap.add_argument("--taken", default="", help="comma-separated names already drafted")
    ap.add_argument("--mine", default="", help="comma-separated names on MY roster")
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    recommend(args.slot, split(args.taken), split(args.mine), args.bundle, args.top)


if __name__ == "__main__":
    main()
