#!/usr/bin/env python
"""Live draft-day CLI (DrafterSpec.md §4.8.1).

Reads draft state via ``draft_state_source.py`` (manual fast entry for Yahoo, or
Sleeper poll for testing) and prints the recommended board for MY next pick. Manual
mode: type the player taken (fuzzy matched); type ``go`` to ADP-auto-advance to your
next pick; ``me <name>`` to record your own pick; ``board`` to reprint.

Every pick is logged to an event history that is (a) the crash-safe save file —
written after each pick, reloaded with ``--resume`` — and (b) the undo stack: ``undo``
pops the last command and rebuilds state by full replay, so state is always a pure
function of the history (no fragile per-action reversal).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import select

from src.config import load_config
from src.db.models import Adp, Player, Projection
from src.db.session import SessionLocal
from src.recommender.draft_state_source import ManualDraftSource, adp_auto_advance
from src.recommender.recommend import build_replacement_from_projections, recommend
from src.recommender.roster_state import RosterState


def _compute_bye_weeks(season: int) -> dict:
    """Return {team_abbr: bye_week_number} from nflreadpy schedules. Empty dict on failure."""
    try:
        from src.ingest.sources import load_schedules
        sched = load_schedules([season])
        if sched.empty:
            return {}
        if "game_type" in sched.columns:
            sched = sched[sched["game_type"] == "REG"]
        all_weeks = set(pd.to_numeric(sched["week"], errors="coerce").dropna().astype(int))
        teams = set(sched["home_team"].dropna()) | set(sched["away_team"].dropna())
        bye: dict = {}
        for team in teams:
            played = (
                set(pd.to_numeric(sched.loc[sched["home_team"] == team, "week"],
                                  errors="coerce").dropna().astype(int))
                | set(pd.to_numeric(sched.loc[sched["away_team"] == team, "week"],
                                    errors="coerce").dropna().astype(int))
            )
            missing = sorted(all_weeks - played)
            if missing:
                bye[team] = missing[0]
        return bye
    except Exception:
        return {}


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
            {"player_id": r.player_id, "name": r.name, "team": r.latest_team}
            for r in session.execute(select(Player)).scalars()
        ])
    finally:
        session.close()
    board = proj.merge(adp, on="player_id", how="left").merge(names, on="player_id", how="left")
    bye_map = _compute_bye_weeks(season)
    if bye_map:
        board["bye_week"] = board["team"].map(bye_map).where(board["team"].notna())
    return board


def _print_recs(recs: pd.DataFrame, names: dict) -> None:
    if recs.empty:
        print("  (no modeled recommendation — fill K/DEF or bench by ADP)")
        return
    top = recs.head(10).reset_index(drop=True)
    fits = list(top["vona_score"])
    best = fits[0]
    has_adp = "adp" in top.columns
    print(f"  Round {int(top.iloc[0]['draft_round'])}  "
          f"(risk quantile {top.iloc[0]['target_quantile']:.2f})")
    hdr_adp = f"  {'ADP':>5}" if has_adp else ""
    print(f"   {'#':>1}  {'PLAYER':24s} {'POS':3s}  {'FIT':>6}  {'Δ#1':>6}  {'proj':>5}{hdr_adp}")
    for i, r in top.iterrows():
        gap = "  —  " if i == 0 else f"{fits[i] - best:+.2f}"
        flag = " [MUST-FILL]" if r.get("forced_completion") else ""
        adp_str = f"  {r['adp']:5.1f}" if has_adp and pd.notna(r.get("adp")) else ("   n/a" if has_adp else "")
        print(f"   {i+1:>1}  {names.get(r['player_id'], '?'):24s} "
              f"{r['position']:3s}  {r['vona_score']:6.2f}  {gap:>6}  "
              f"{r['value']:5.1f}{adp_str}{flag}")
    if len(fits) >= 2:
        gap = best - fits[1]
        rng = best - fits[-1]
        if gap >= 1.0:
            read = f"CLEAR #1 — leads #2 by {gap:.2f}"
        elif rng <= 0.5:
            read = f"TOSS-UP — top {len(fits)} within {rng:.2f}; pick your preference"
        else:
            read = f"#1 slight edge ({gap:.2f}); #2–{len(fits)} clustered"
        print(f"   read: {read}")


def _print_run_warning(avail_board: pd.DataFrame, cur_pick: int, next_pick, cfg) -> None:
    """Positional depth: count available players with ADP between now and my next pick."""
    if avail_board.empty or "adp" not in avail_board.columns:
        return
    horizon = next_pick if next_pick else cur_pick + cfg.teams
    window = horizon - cur_pick
    if window <= 0:
        return
    parts = []
    for pos in cfg.roster.modeled_positions:
        pos_df = avail_board[avail_board["position"] == pos].dropna(subset=["adp"])
        n = int(((pos_df["adp"] >= cur_pick) & (pos_df["adp"] <= horizon)).sum())
        parts.append(f"{pos} {n}{'!' if n <= 1 else ''}")
    print(f"  runs (picks {cur_pick}→{horizon}, {window} picks): {' | '.join(parts)}")


def _print_roster(state: RosterState, names: dict) -> None:
    """Current team, open starter slots, and any bye-week stacks."""
    print("  MY ROSTER:")
    if not state.my_roster:
        print("    (empty)")
    else:
        for p in state.my_roster:
            bw = p.get("bye_week")
            tm = p.get("team") or "?"
            bye_tag = f"  bye {bw}" if bw else ""
            print(f"    {p['position']:3s} {names.get(p['player_id'], '?'):22s} {tm:3s}{bye_tag}")
    unfilled = state.unfilled_mandatory_slots()
    if unfilled:
        print("  open starters: " + ", ".join(f"{k}×{v}" for k, v in unfilled.items()))
    else:
        print("  open starters: none (all starters filled)")
    clashes = {bw: c for bw, c in state.my_bye_week_counts().items() if c >= 2}
    if clashes:
        print("  bye stacks: " + ", ".join(f"wk{bw}×{c}" for bw, c in sorted(clashes.items())))


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Live draft CLI (M6).")
    parser.add_argument("--season", type=int, required=True, help="Projection season to draft")
    parser.add_argument("--position", type=int, required=True,
                        help=f"My draft slot 1..{cfg.teams}")
    parser.add_argument("--resume", action="store_true",
                        help="Reload an in-progress draft from its save file")
    parser.add_argument("--state-file", type=str, default=None,
                        help="Path to the draft save file (default: draft_state_<season>_slot<pos>.json)")
    args = parser.parse_args()
    if not (1 <= args.position <= cfg.teams):
        raise SystemExit(f"draft position must be in 1..{cfg.teams}")

    state_path = pathlib.Path(
        args.state_file or f"draft_state_{args.season}_slot{args.position}.json")

    board = _load_board(args.season, cfg)
    if board.empty:
        raise SystemExit("No projections/ADP found. Run the pipeline (M0-M3) first.")
    names = dict(zip(board["player_id"], board["name"]))
    replacement = build_replacement_from_projections(board, cfg=cfg)
    src = ManualDraftSource(board=board[["player_id", "name", "adp", "p50", "position", "team"]].copy())
    state = RosterState(cfg=cfg, draft_position=args.position)

    # --- event-sourced draft log: the save file AND the undo stack ---
    # Each entry is one command = a list of events. Event = ["pick", pid, mine] or
    # ["skip", token]. State is rebuilt by replaying events, so it's always a pure
    # function of `history` — undo just pops and replays.
    history: list = []

    def _apply_event(ev: list) -> None:
        kind = ev[0]
        if kind == "skip":
            token = ev[1]
            state.drafted.add(token)
            src.drafted.add(token)
            return
        pid, mine = ev[1], ev[2]
        src.drafted.add(pid)
        prow = board.loc[board["player_id"] == pid]
        if prow.empty:
            state.record_pick(pid, None, mine=mine)
            return
        prow = prow.iloc[0]
        team_val = prow["team"] if "team" in board.columns and pd.notna(prow.get("team")) else None
        bye_val = (int(prow["bye_week"]) if "bye_week" in board.columns
                   and pd.notna(prow.get("bye_week")) else None)
        state.record_pick(pid, prow["position"], mine=mine, team=team_val, bye_week=bye_val)

    def _save() -> None:
        data = {"season": args.season, "position": args.position, "history": history}
        try:
            state_path.write_text(json.dumps(data))
        except OSError as e:
            print(f"  WARNING: could not save draft state to {state_path}: {e}")

    def _rebuild() -> None:
        state.reset()
        src.drafted.clear()
        for command in history:
            for ev in command:
                _apply_event(ev)

    def _commit(command: list) -> None:
        history.append(command)
        for ev in command:
            _apply_event(ev)
        _save()

    def _undo() -> None:
        if not history:
            print("  nothing to undo")
            return
        last = history.pop()
        _rebuild()
        _save()
        labels = []
        for ev in last:
            if ev[0] == "skip":
                labels.append("a skip")
            else:
                tag = "MY PICK " if ev[2] else ""
                labels.append(f"{tag}{names.get(ev[1], ev[1])}")
        print(f"  undid: {', '.join(labels)}  (now at pick #{state.current_overall_pick()})")

    if args.resume and state_path.exists():
        loaded = json.loads(state_path.read_text())
        history.extend(loaded.get("history", []))
        _rebuild()
        print(f"Resumed from {state_path}: {len(state.drafted)} picks already recorded.")
    elif state_path.exists():
        print(f"NOTE: a saved draft exists at {state_path} — pass --resume to continue it "
              f"(starting fresh will overwrite it).")

    print(f"Draft loaded: season {args.season}, slot {args.position}/{cfg.teams}, "
          f"{cfg.roster.rounds} rounds. My picks: {state.my_picks}")
    print("Commands: <name> | me <name> | go | skip | undo | roster | board | quit")
    _ALL_PROJ_COLS = ["player_id", "position", "p10", "p50", "p90",
                      "adp", "adp_stdev", "team", "bye_week"]
    proj_cols = [c for c in _ALL_PROJ_COLS if c in board.columns]
    while state.remaining_picks() > 0:
        avail_board = board[proj_cols][~board["player_id"].isin(state.drafted)].copy()
        recs = recommend(avail_board, state, replacement, cfg=cfg, top_n=10)
        cur = state.current_overall_pick()
        is_my_turn = cur in state.my_picks
        if is_my_turn:
            print(f"\n--- pick #{cur} | ** YOUR PICK (round {state.my_current_round()}) ** ---")
        else:
            print(f"\n--- pick #{cur} | my next: {state.next_my_pick()} ---")
        _print_recs(recs, names)
        _print_run_warning(avail_board, cur, state.next_my_pick(), cfg)
        cmd = input("> ").strip()
        if not cmd:
            continue
        if cmd.lower() == "quit":
            break
        if cmd.lower() == "board":
            continue
        if cmd.lower() == "roster":
            _print_roster(state, names)
            continue
        if cmd.lower() == "undo":
            _undo()
            continue
        if cmd.lower() == "go":
            if is_my_turn:
                print("  it's your pick — use 'me <player name>' to record it")
                continue
            target = state.next_my_pick() or state.current_overall_pick()
            n = target - state.current_overall_pick()
            taken = adp_auto_advance(src.board, src.drafted, n)
            if taken:
                _commit([["pick", pid, False] for pid in taken])
            skipped = n - len(taken)
            if skipped > 0:
                print(f"  auto-advanced {len(taken)} picks ({skipped} remaining — ADP exhausted; enter opponent names or 'skip')")
            else:
                print(f"  auto-advanced {len(taken)} picks")
            continue
        if cmd.lower() == "skip":
            _commit([["skip", f"_skip_{cur}"]])
            print(f"  skipped pick #{cur} (opponent)")
            continue
        mine = cmd.lower().startswith("me ")
        name = cmd[3:] if mine else cmd
        # At user's own pick: bare name entry = their pick (no need to prefix "me")
        if is_my_turn and not mine:
            mine = True
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
            pid = cands[int(sel) - 1]
        else:
            pid = cands[0]
        _commit([["pick", pid, mine]])
        prow = board.loc[board["player_id"] == pid].iloc[0]
        bye_val = (int(prow["bye_week"]) if "bye_week" in board.columns
                   and pd.notna(prow.get("bye_week")) else None)
        bye_tag = f" bye {bye_val}" if mine and bye_val else ""
        print(f"  recorded {'MY PICK ' if mine else ''}{names.get(pid)} ({prow['position']}{bye_tag})")

    print("\nFinal roster:")
    for slot, r in state.slot_fill.items():
        print(f"  {slot}: {r}")
    for p in state.my_roster:
        print(f"   - {names.get(p['player_id'])} ({p['position']})")


if __name__ == "__main__":
    main()
