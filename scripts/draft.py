#!/usr/bin/env python
"""Live draft-day CLI (DrafterSpec.md §4.8.1).

Reads draft state via ``draft_state_source.py`` (manual fast entry for Yahoo, or
Sleeper poll for testing) and prints the recommended board for MY next pick. Manual
mode: type the player taken (fuzzy matched); type ``go`` to ADP-auto-advance to your
next pick; ``me <name>`` to record your own pick; ``board`` to reprint.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import select

from src.config import load_config
from src.db.models import Adp, Player, Projection
from src.db.session import SessionLocal
from src.recommender.draft_state_source import ManualDraftSource
from src.recommender.recommend import build_replacement_from_projections, recommend
from src.recommender.roster_state import RosterState


def _load_board(season: int, cfg) -> pd.DataFrame:
    session = SessionLocal()
    try:
        proj = pd.DataFrame([
            {"player_id": r.player_id, "position": r.position, "p10": r.p10,
             "p50": r.p50, "p90": r.p90}
            for r in session.execute(
                select(Projection).where(Projection.season == season)).scalars()
        ])
        adp = pd.DataFrame([
            {"player_id": r.player_id, "adp": r.adp, "adp_stdev": r.adp_stdev}
            for r in session.execute(
                select(Adp).where(Adp.season == season,
                                  Adp.format == cfg.adp.format,
                                  Adp.teams == cfg.adp.teams)).scalars()
        ])
        names = pd.DataFrame([
            {"player_id": r.player_id, "name": r.name, "team": r.team_current}
            for r in session.execute(select(Player)).scalars()
        ])
    finally:
        session.close()
    board = proj.merge(adp, on="player_id", how="left").merge(names, on="player_id", how="left")
    return board


def _print_recs(recs: pd.DataFrame, names: dict) -> None:
    if recs.empty:
        print("  (no modeled recommendation — fill K/DEF or bench by ADP)")
        return
    print(f"  Round {int(recs.iloc[0]['draft_round'])}  "
          f"(target quantile {recs.iloc[0]['target_quantile']:.2f})")
    for i, r in recs.iterrows():
        flag = " [MUST-FILL]" if r.get("forced_completion") else ""
        print(f"  {i+1:2d}. {names.get(r['player_id'], r['player_id']):24s} "
              f"{r['position']:3s}  VONA={r['vona_score']:6.2f}  "
              f"val={r['value']:6.2f} wait={r['wait_term']:5.2f}{flag}")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Live draft CLI (M6).")
    parser.add_argument("--season", type=int, required=True, help="Projection season to draft")
    parser.add_argument("--position", type=int, required=True,
                        help=f"My draft slot 1..{cfg.teams}")
    args = parser.parse_args()
    if not (1 <= args.position <= cfg.teams):
        raise SystemExit(f"draft position must be in 1..{cfg.teams}")

    board = _load_board(args.season, cfg)
    if board.empty:
        raise SystemExit("No projections/ADP found. Run the pipeline (M0-M3) first.")
    names = dict(zip(board["player_id"], board["name"]))
    replacement = build_replacement_from_projections(board, cfg=cfg)
    src = ManualDraftSource(board=board[["player_id", "name", "adp", "position", "team"]].copy())
    state = RosterState(cfg=cfg, draft_position=args.position)

    print(f"Draft loaded: season {args.season}, slot {args.position}/{cfg.teams}, "
          f"{cfg.roster.rounds} rounds. My picks: {state.my_picks}")
    proj_cols = ["player_id", "position", "p10", "p50", "p90", "adp", "adp_stdev"]
    while state.remaining_picks() > 0:
        avail_board = board[proj_cols][~board["player_id"].isin(state.drafted)]
        recs = recommend(avail_board, state, replacement, cfg=cfg)
        print(f"\n--- pick #{state.current_overall_pick()} | my next: {state.next_my_pick()} ---")
        _print_recs(recs, names)
        cmd = input("> ").strip()
        if not cmd:
            continue
        if cmd.lower() == "quit":
            break
        if cmd.lower() == "board":
            continue
        if cmd.lower() == "go":
            target = state.next_my_pick() or state.current_overall_pick()
            n = target - state.current_overall_pick()
            taken = src.auto_advance(n)
            for pid in taken:
                state.record_pick(pid, mine=False)
            print(f"  auto-advanced {len(taken)} ADP picks")
            continue
        mine = cmd.lower().startswith("me ")
        name = cmd[3:] if mine else cmd
        cands = src.candidates(name)
        if not cands:
            print("  ?? no match — try again")
            continue
        if len(cands) > 1:
            # Two distinct same-name players: never silently pick one — disambiguate.
            print(f"  ambiguous — {len(cands)} players match '{name.strip()}':")
            for j, c in enumerate(cands, 1):
                row = board.loc[board["player_id"] == c].iloc[0]
                print(f"    {j}) {names.get(c)}  {row['position']}  {row.get('team', '')}")
            sel = input("  pick #> ").strip()
            if not (sel.isdigit() and 1 <= int(sel) <= len(cands)):
                print("  cancelled")
                continue
            pid = src.record_pid(cands[int(sel) - 1])
        else:
            pid = src.record_pid(cands[0])
        pos = board.loc[board["player_id"] == pid, "position"].iloc[0]
        state.record_pick(pid, pos, mine=mine)
        print(f"  recorded {'MY PICK ' if mine else ''}{names.get(pid)} ({pos})")

    print("\nFinal roster:")
    for slot, r in state.slot_fill.items():
        print(f"  {slot}: {r}")
    for p in state.my_roster:
        print(f"   - {names.get(p['player_id'])} ({p['position']})")


if __name__ == "__main__":
    main()
