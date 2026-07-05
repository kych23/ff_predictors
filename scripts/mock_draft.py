#!/usr/bin/env python
"""Replay a COMPLETED season as a mock draft and score it against actual FPPG.

Runs the VONA recommender from a chosen slot against 11 ADP bots on a past season's
projections + ADP (the same validated Tier-2 engine in ``benchmark/draft_sim.py``),
then scores every team's best legal starting lineup by that season's ACTUAL per-game
fantasy points (the labels). Answers: "if I'd drafted with this tool last year, how
would my roster actually have done?"

  python scripts/mock_draft.py --season 2025 --position 3     # one slot, full detail
  python scripts/mock_draft.py --season 2025 --all-slots      # all 12 slots, summary

The draft is deterministic (bots take lowest ADP; recommender takes its #1), so one
run per slot is exact — no Monte Carlo needed.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.benchmark.draft_sim import optimal_lineup, optimal_lineup_value, simulate_draft
from src.config import load_config
from src.db.loaders import (load_adp_df, load_player_names, load_projections_df,
                            load_season_labels_df)
from src.db.models import Adp
from src.db.session import session_scope
from src.recommender.recommend import build_replacement_from_projections


def _load(season: int, cfg):
    with session_scope() as session:
        proj = load_projections_df(session, season)[
            ["player_id", "position", "p10", "p50", "p90"]]
        adp = load_adp_df(session, cfg, season)[
            ["player_id", "adp", "adp_stdev", "position"]]
        labels = load_season_labels_df(session, season)[["player_id", "fppg"]]
        names = load_player_names(session)
    return proj, adp, labels, names


def _seasons_with_adp(cfg) -> list:
    """Seasons that actually have ADP rows for this league's format/teams (for hints)."""
    with session_scope() as session:
        rows = session.execute(
            select(Adp.season).where(Adp.format == cfg.adp.format,
                                     Adp.teams == cfg.adp.teams).distinct()
        ).scalars().all()
    return sorted(set(rows))


def _build_board(proj: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    """ADP universe (incl. K/DEF) left-joined to projections; position from proj first."""
    board = adp.merge(proj[["player_id", "p10", "p50", "p90"]],
                      on="player_id", how="left")
    proj_pos = dict(zip(proj["player_id"], proj["position"]))
    board["position"] = board["player_id"].map(proj_pos).fillna(board["position"])
    return board


def _draft_one(board, cfg, slot, replacement, pos_map, fppg_map):
    rosters = simulate_draft(board, cfg, slot, replacement)
    our_val = optimal_lineup_value(rosters[slot], pos_map, fppg_map, cfg)
    bot_vals = [optimal_lineup_value(rosters[t], pos_map, fppg_map, cfg)
                for t in rosters if t != slot]
    rank = 1 + sum(1 for v in bot_vals if v > our_val)
    return rosters, our_val, bot_vals, rank


def _run_single(board, cfg, slot, replacement, pos_map, fppg_map, names, season):
    rosters, our_val, bot_vals, rank = _draft_one(
        board, cfg, slot, replacement, pos_map, fppg_map)
    my_ids = rosters[slot]

    print(f"\n=== {season} MOCK DRAFT — your slot {slot}/{cfg.teams} (recommender vs ADP bots) ===")
    print("  your picks, round by round  (proj P50 -> actual FPPG):")
    p50_map = dict(zip(board["player_id"], board["p50"]))
    for rnd, pid in enumerate(my_ids, 1):
        pos = pos_map.get(pid, "?")
        p50 = p50_map.get(pid)
        proj_str = f"{p50:5.1f}" if pd.notna(p50) else "  —  "
        actual = fppg_map.get(pid)
        act_str = f"{actual:5.1f}" if actual is not None else "  —  "
        print(f"   R{rnd:<2} {names.get(pid, pid):24s} {pos:3s}  {proj_str} -> {act_str}")

    rows, total = optimal_lineup(my_ids, pos_map, fppg_map, cfg)
    print(f"\n  best legal starting lineup (by actual {season} FPPG):")
    for slot_label, pid, val in rows:
        print(f"   {slot_label:4s} {names.get(pid, pid):24s} {val:5.1f}")
    print(f"   {'':4s} {'TOTAL starting FPPG':24s} {total:5.1f}")

    bot_mean = float(np.mean(bot_vals))
    bot_best = float(np.max(bot_vals))
    print(f"\n  vs field: you {our_val:.1f} | ADP-bot mean {bot_mean:.1f} | "
          f"bot best {bot_best:.1f} | rank {rank} of {cfg.teams}  "
          f"({our_val - bot_mean:+.1f} vs mean)")


def _run_all_slots(board, cfg, replacement, pos_map, fppg_map, season):
    print(f"\n=== {season} MOCK DRAFT — all {cfg.teams} slots (recommender vs ADP bots) ===")
    print(f"  {'slot':>4}  {'you':>7}  {'bot_mean':>8}  {'diff':>6}  {'rank':>4}")
    diffs, wins = [], 0
    for slot in range(1, cfg.teams + 1):
        _, our_val, bot_vals, rank = _draft_one(
            board, cfg, slot, replacement, pos_map, fppg_map)
        bot_mean = float(np.mean(bot_vals))
        diff = our_val - bot_mean
        diffs.append(diff)
        wins += int(diff > 0)
        print(f"  {slot:>4}  {our_val:>7.1f}  {bot_mean:>8.1f}  {diff:>+6.1f}  {rank:>4}")
    print(f"\n  mean diff vs field: {np.mean(diffs):+.1f} | "
          f"beat the field in {wins}/{cfg.teams} slots")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Replay a past season as a scored mock draft.")
    parser.add_argument("--season", type=int, required=True,
                        help="A COMPLETED season (must have projections, ADP, and labels)")
    parser.add_argument("--position", type=int, default=None,
                        help=f"Your draft slot 1..{cfg.teams} (single-slot detail view)")
    parser.add_argument("--all-slots", action="store_true",
                        help="Run every slot and print a summary table")
    args = parser.parse_args()
    if not args.all_slots and args.position is None:
        raise SystemExit("Pass --position N for a single slot, or --all-slots for the summary.")

    proj, adp, labels, names = _load(args.season, cfg)
    print(f"{args.season}: projections={len(proj)} | adp={len(adp)} | labels={len(labels)}")
    if proj.empty or adp.empty or labels.empty:
        missing = [n for n, df in (("projections", proj), ("ADP", adp), ("labels", labels))
                   if df.empty]
        hint = ""
        if missing == ["ADP"]:
            avail = _seasons_with_adp(cfg)
            hint = (" — the Fantasy Football Calculator API has no ADP for this season "
                    "(it drops the just-finished season before re-archiving it). "
                    + (f"Seasons with ADP available: {avail}." if avail else ""))
        raise SystemExit(f"Missing {', '.join(missing)} for {args.season}.{hint}")

    board = _build_board(proj, adp)
    pos_map = dict(zip(board["player_id"], board["position"]))
    fppg_map = dict(zip(labels["player_id"], labels["fppg"]))
    replacement = build_replacement_from_projections(proj, cfg=cfg)

    if args.position is not None:
        if not (1 <= args.position <= cfg.teams):
            raise SystemExit(f"--position must be in 1..{cfg.teams}")
        _run_single(board, cfg, args.position, replacement, pos_map, fppg_map, names, args.season)
    if args.all_slots:
        _run_all_slots(board, cfg, replacement, pos_map, fppg_map, args.season)


if __name__ == "__main__":
    main()
