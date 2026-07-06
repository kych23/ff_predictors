#!/usr/bin/env python
"""Weekly start/sit CLI — "Who Should I Start?"

Reads a YAML roster file, loads weekly projections (or season P50 as fallback),
runs the ILP optimizer, and prints the optimal starting lineup.

Usage:
  python scripts/start.py --season 2026 --week 5 --roster roster.yaml
  python scripts/start.py --season 2026 --week 5 --roster roster.yaml --strategy upside
  python scripts/start.py --season 2026 --week 5 --roster roster.yaml compare "Josh Allen" vs "Jalen Hurts"
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from sqlalchemy import select

from src.config import load_config
from src.db.models import Player, Projection, WeeklyProjection
from src.db.session import SessionLocal
from src.lineup.optimizer import optimize_lineup, STRATEGY_MAP

# ANSI color codes
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _load_roster(path: str) -> dict:
    """Load YAML roster file. Returns dict with 'season' and 'roster' list."""
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f"roster file not found: {path}")
    data = yaml.safe_load(p.read_text())
    if "roster" not in data:
        raise SystemExit(f"roster file missing 'roster' key: {path}")
    return data


def _match_player(name: str, names_df: pd.DataFrame) -> str | None:
    """Fuzzy match a roster name to a player_id. Exact first, then substring."""
    name_lower = name.strip().lower()
    exact = names_df[names_df["name_lower"] == name_lower]
    if len(exact) == 1:
        return exact.iloc[0]["player_id"]
    if len(exact) > 1:
        return exact.iloc[0]["player_id"]
    partial = names_df[names_df["name_lower"].str.contains(name_lower, na=False)]
    if len(partial) == 1:
        return partial.iloc[0]["player_id"]
    if len(partial) > 1:
        return partial.iloc[0]["player_id"]
    return None


def _grade_color(dvp_rank: float | None) -> tuple[str, str]:
    """Matchup grade from DvP rank (1 = most points allowed = best matchup)."""
    if dvp_rank is None or pd.isna(dvp_rank):
        return "—", _DIM
    r = int(dvp_rank)
    if r <= 10:
        return "A", _GREEN
    if r <= 22:
        return "B", _YELLOW
    return "C", _RED


def _load_projections(session, season: int, week: int, player_ids: set):
    """Load weekly projections; fall back to season projections if weekly unavailable."""
    # Try weekly projections first
    weekly = pd.DataFrame([
        {"player_id": r.player_id, "position": r.position,
         "model_version": r.model_version,
         "p10": r.p10, "p50": r.p50, "p90": r.p90}
        for r in session.execute(
            select(WeeklyProjection).where(
                WeeklyProjection.season == season,
                WeeklyProjection.week == week,
                WeeklyProjection.player_id.in_(player_ids),
            )).scalars()
    ])
    if not weekly.empty:
        # forward (weekly_fwd_*, live-serving) rows win over OOF (weekly_v1.*)
        weekly["_fwd"] = weekly["model_version"].str.startswith("weekly_fwd")
        weekly = (weekly.sort_values("_fwd")
                  .drop_duplicates("player_id", keep="last")
                  .drop(columns=["_fwd", "model_version"]))
        return weekly, "weekly"

    # Fallback: season projections (scale to per-game)
    season_proj = pd.DataFrame([
        {"player_id": r.player_id, "position": r.position,
         "p10": r.p10, "p50": r.p50, "p90": r.p90}
        for r in session.execute(
            select(Projection).where(
                Projection.season == season,
                Projection.player_id.in_(player_ids),
            )).scalars()
    ])
    if not season_proj.empty:
        season_proj = season_proj.sort_values("p50", ascending=False).drop_duplicates(
            "player_id", keep="first")
        return season_proj, "season (per-game fallback)"
    return pd.DataFrame(), "none"


def _load_dvp_ranks(session, season: int, week: int) -> dict:
    """Load DvP ranks from weekly features if available. Returns {(opponent, position): rank}."""
    try:
        from src.db.models import WeeklyFeature
        rows = session.execute(
            select(WeeklyFeature).where(
                WeeklyFeature.season == season,
                WeeklyFeature.week == week,
            )).scalars().all()
        if not rows:
            return {}
        dvp_vals = {}
        for r in rows:
            feats = r.features or {}
            dvp = feats.get("opp_dvp_fpa")
            opp = feats.get("opponent")
            if dvp is not None and r.position:
                key = (r.player_id, r.position)
                dvp_vals[key] = dvp
        if not dvp_vals:
            return {}
        # Rank by position: higher FPA = easier matchup = lower rank number
        by_pos: dict[str, list] = {}
        for (pid, pos), fpa in dvp_vals.items():
            by_pos.setdefault(pos, []).append((pid, fpa))
        ranks = {}
        for pos, entries in by_pos.items():
            entries.sort(key=lambda x: -x[1])
            for rank, (pid, _) in enumerate(entries, 1):
                ranks[pid] = rank
        return ranks
    except Exception as exc:
        print(f"warning: DvP ranks unavailable ({exc})", file=sys.stderr)
        return {}


def _load_schedules_for_week(season: int, week: int) -> dict:
    """Return {team: opponent} for the given week from nflreadpy."""
    try:
        from src.ingest.sources import load_schedules
        sched = load_schedules([season])
        if sched.empty:
            return {}
        wk = sched[(sched["week"] == week)]
        if "game_type" in wk.columns:
            wk = wk[wk["game_type"] == "REG"]
        opp_map = {}
        for _, g in wk.iterrows():
            h, a = str(g["home_team"]), str(g["away_team"])
            opp_map[h] = f"vs {a}"
            opp_map[a] = f"@ {h}"
        return opp_map
    except Exception as exc:
        print(f"warning: schedule/opponent info unavailable ({exc})", file=sys.stderr)
        return {}


def _print_lineup(result, names: dict, opp_map: dict, dvp_ranks: dict,
                  strategy: str, proj_source: str):
    """Print slot-based lineup table with matchup grades."""
    obj_col = STRATEGY_MAP.get(strategy, "p50")
    print(f"\n{_BOLD}OPTIMAL LINEUP{_RESET}  (strategy: {strategy}, projections: {proj_source})")
    print(f"  {'SLOT':>4}  {'PLAYER':22s} {'OPP':>7} {'GRD':>3}  {'P10':>5} {'P50':>5} {'P90':>5}  VERDICT")
    print("  " + "─" * 72)

    slot_order = ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF"]

    for slot in slot_order:
        slot_players = result.starters[result.starters["slot"] == slot]
        for _, p in slot_players.iterrows():
            pid = p["player_id"]
            name = names.get(pid, pid)[:22]
            team = p.get("team", "")
            opp = opp_map.get(team, "") if team else ""
            rank = dvp_ranks.get(pid)
            grade, color = _grade_color(rank)
            p10 = p.get("p10", 0)
            p50 = p.get("p50", 0)
            p90 = p.get("p90", 0)
            print(f"  {slot:>4}  {name:22s} {opp:>7} {color}{grade:>3}{_RESET}  "
                  f"{p10:5.1f} {p50:5.1f} {p90:5.1f}  START")

    if not result.bench.empty:
        print("  " + "─" * 72)
        bench = result.bench.sort_values(obj_col, ascending=False)
        for _, p in bench.iterrows():
            pid = p["player_id"]
            name = names.get(pid, pid)[:22]
            team = p.get("team", "")
            opp = opp_map.get(team, "") if team else ""
            rank = dvp_ranks.get(pid)
            grade, color = _grade_color(rank)
            p10 = p.get("p10", 0)
            p50 = p.get("p50", 0)
            p90 = p.get("p90", 0)
            status = p.get("status", "")
            if status and str(status).upper() in ("IR", "OUT", "O"):
                verdict = f"{_DIM}OUT{_RESET}"
            elif opp == "":
                verdict = f"{_DIM}BYE{_RESET}"
            else:
                verdict = "BENCH"
            print(f"  {'':>4}  {name:22s} {opp:>7} {color}{grade:>3}{_RESET}  "
                  f"{p10:5.1f} {p50:5.1f} {p90:5.1f}  {verdict}")

    print(f"\n  {_BOLD}Projected total ({strategy}): {result.total_points:.1f}{_RESET}")

    # Interpretation line
    starters = result.starters
    if len(starters) >= 2:
        vals = starters[obj_col].sort_values(ascending=False)
        top = vals.iloc[0]
        bot = vals.iloc[-1]
        if not result.bench.empty:
            best_bench = result.bench[obj_col].max()
            margin = bot - best_bench if pd.notna(best_bench) else float("inf")
            if margin < 1.0:
                marginal = result.bench.loc[result.bench[obj_col].idxmax()]
                mname = names.get(marginal["player_id"], "?")
                print(f"  {_YELLOW}close call:{_RESET} {mname} ({marginal['position']}) "
                      f"within {margin:.1f} pts of the last starter")


def _print_compare(player_a: dict, player_b: dict, names: dict,
                   opp_map: dict, dvp_ranks: dict):
    """Head-to-head comparison of two players."""
    def _col(p):
        pid = p["player_id"]
        name = names.get(pid, pid)
        team = p.get("team", "")
        opp = opp_map.get(team, "")
        rank = dvp_ranks.get(pid)
        grade, color = _grade_color(rank)
        return {
            "name": name, "position": p.get("position", "?"),
            "opp": opp, "grade": f"{color}{grade}{_RESET}",
            "p10": p.get("p10", 0), "p50": p.get("p50", 0), "p90": p.get("p90", 0),
        }

    a, b = _col(player_a), _col(player_b)
    print(f"\n{_BOLD}HEAD-TO-HEAD{_RESET}")
    print(f"  {'':12s} {a['name']:>22s}   vs   {b['name']:<22s}")
    print(f"  {'Position':12s} {a['position']:>22s}        {b['position']:<22s}")
    print(f"  {'Matchup':12s} {a['opp']:>22s}        {b['opp']:<22s}")
    print(f"  {'Grade':12s} {a['grade']:>31s}        {b['grade']:<22s}")
    print(f"  {'P10 (floor)':12s} {a['p10']:>22.1f}        {b['p10']:<22.1f}")
    print(f"  {'P50 (median)':12s} {a['p50']:>22.1f}        {b['p50']:<22.1f}")
    print(f"  {'P90 (ceil)':12s} {a['p90']:>22.1f}        {b['p90']:<22.1f}")

    diff = a["p50"] - b["p50"]
    if abs(diff) < 1.0:
        verdict = "TOSS-UP — within 1 point; prefer the better matchup grade"
    elif diff > 0:
        verdict = f"START {a['name']} — +{diff:.1f} pts median edge"
    else:
        verdict = f"START {b['name']} — +{-diff:.1f} pts median edge"
    print(f"\n  {_BOLD}verdict:{_RESET} {verdict}")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Weekly start/sit optimizer.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--roster", type=str, required=True, help="Path to YAML roster file")
    parser.add_argument("--strategy", type=str, default="balanced",
                        choices=["safe", "balanced", "upside"])
    parser.add_argument("command", nargs="*", help="compare 'Player A' vs 'Player B'")
    args = parser.parse_args()

    roster_data = _load_roster(args.roster)
    roster_season = roster_data.get("season", args.season)
    roster_entries = roster_data["roster"]

    session = SessionLocal()
    try:
        # Load player names for matching
        all_players = pd.DataFrame([
            {"player_id": r.player_id, "name": r.name, "latest_team": r.latest_team}
            for r in session.execute(select(Player)).scalars()
        ])
        if all_players.empty:
            raise SystemExit("Player table empty. Run seed_db.py first.")
        all_players["name_lower"] = all_players["name"].str.lower()
        names = dict(zip(all_players["player_id"], all_players["name"]))

        # Resolve roster names to player_ids
        resolved = []
        unmatched = []
        for entry in roster_entries:
            pid = _match_player(entry["name"], all_players)
            if pid is None:
                unmatched.append(entry["name"])
                continue
            resolved.append({
                "player_id": pid,
                "position": entry.get("position", ""),
                "team": entry.get("team", ""),
                "status": entry.get("status", "active"),
            })
        if unmatched:
            print(f"{_YELLOW}WARNING: could not match:{_RESET} {', '.join(unmatched)}")

        if not resolved:
            raise SystemExit("No roster players matched. Check roster file names.")

        player_ids = {r["player_id"] for r in resolved}
        projections, proj_source = _load_projections(session, args.season, args.week, player_ids)

        if projections.empty:
            raise SystemExit(f"No projections found for season {args.season}. "
                             "Run train_projection.py or train_weekly_projection.py first.")

        # Merge roster info into projections
        roster_df = pd.DataFrame(resolved)
        players = projections.merge(
            roster_df[["player_id", "team", "status"]],
            on="player_id", how="left",
        )
        # Add names
        players["name"] = players["player_id"].map(names)

        dvp_ranks = _load_dvp_ranks(session, args.season, args.week)
        opp_map = _load_schedules_for_week(args.season, args.week)

        # Handle compare command
        if args.command and len(args.command) >= 3 and args.command[0].lower() == "compare":
            compare_parts = " ".join(args.command[1:])
            if " vs " in compare_parts.lower():
                idx = compare_parts.lower().index(" vs ")
                name_a = compare_parts[:idx].strip().strip("'\"")
                name_b = compare_parts[idx + 4:].strip().strip("'\"")
                pid_a = _match_player(name_a, all_players)
                pid_b = _match_player(name_b, all_players)
                if not pid_a:
                    raise SystemExit(f"Could not match player: {name_a}")
                if not pid_b:
                    raise SystemExit(f"Could not match player: {name_b}")
                pa = players[players["player_id"] == pid_a]
                pb = players[players["player_id"] == pid_b]
                if pa.empty:
                    raise SystemExit(f"No projection for {name_a}")
                if pb.empty:
                    raise SystemExit(f"No projection for {name_b}")
                _print_compare(pa.iloc[0].to_dict(), pb.iloc[0].to_dict(),
                               names, opp_map, dvp_ranks)
                return

        # Run optimizer
        excluded = {r["player_id"] for r in resolved
                    if str(r.get("status", "")).upper() in ("IR", "OUT", "O")}
        result = optimize_lineup(players, cfg=cfg, strategy=args.strategy,
                                 excluded_ids=excluded)
        _print_lineup(result, names, opp_map, dvp_ranks, args.strategy, proj_source)

    finally:
        session.close()


if __name__ == "__main__":
    main()
