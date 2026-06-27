"""Tier-2 draft simulation: recommender vs ADP-bots (DrafterSpec.md §4.8.1 / §4.7.1).

A snake draft of ADP-bots against our recommender; rosters scored
**availability-neutral** (each starter contributes actual-season FPPG, not raw
totals — else opponents' injury luck would falsely fail the per-game model, §4.7.1).
Runs only on SIM_ELIGIBLE seasons (ADP coverage >= teams*rounds = 180); thinner
seasons are excluded, NOT tail-backfilled (backfill would leak our signal into
opponent behavior, §4.0). Doubles as the future Monte-Carlo lookahead engine.

This package may read ADP (benchmark side of the wall).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import LeagueConfig, load_config
from src.recommender.recommend import build_replacement_from_projections, recommend
from src.recommender.roster_state import RosterState, overall_pick

from .report import PairedTest, paired_test

logger = logging.getLogger(__name__)


def _bot_pick(board: pd.DataFrame, drafted: set, slot_fill: Dict[str, int],
              cfg: LeagueConfig) -> Optional[str]:
    """ADP bot: lowest-ADP available player that fits any open slot (pure/flex/bench)."""
    avail = board[~board["player_id"].isin(drafted)].dropna(subset=["adp"]).sort_values("adp")
    for _, row in avail.iterrows():
        pos = row["position"]
        if _has_open_slot(pos, slot_fill, cfg):
            return row["player_id"]
    return avail.iloc[0]["player_id"] if not avail.empty else None


def _has_open_slot(position: str, slot_fill: Dict[str, int], cfg: LeagueConfig) -> bool:
    if slot_fill.get(position, 0) < cfg.roster.slots.get(position, 0):
        return True
    if position in cfg.roster.flex_eligible and \
            slot_fill.get("FLEX", 0) < cfg.roster.slots.get("FLEX", 0):
        return True
    return slot_fill.get("BENCH", 0) < cfg.roster.slots.get("BENCH", 0)


def _fill_slot(position: str, slot_fill: Dict[str, int], cfg: LeagueConfig) -> None:
    if slot_fill.get(position, 0) < cfg.roster.slots.get(position, 0):
        slot_fill[position] = slot_fill.get(position, 0) + 1
    elif position in cfg.roster.flex_eligible and \
            slot_fill.get("FLEX", 0) < cfg.roster.slots.get("FLEX", 0):
        slot_fill["FLEX"] = slot_fill.get("FLEX", 0) + 1
    else:
        slot_fill["BENCH"] = slot_fill.get("BENCH", 0) + 1


def simulate_draft(board: pd.DataFrame, cfg: LeagueConfig, my_position: int,
                   replacement) -> Dict[int, List[str]]:
    """Run one snake draft. Team ``my_position`` uses the recommender; rest are bots.

    ``board``: player_id, position, adp, adp_stdev, p10, p50, p90 (projections NaN
    for K/DEF). Returns team_seat -> list of drafted player_ids.
    """
    teams = cfg.teams
    rounds = cfg.roster.rounds
    rosters: Dict[int, List[str]] = {t: [] for t in range(1, teams + 1)}
    slot_fills: Dict[int, Dict[str, int]] = {t: {s: 0 for s in cfg.roster.slots}
                                             for t in range(1, teams + 1)}
    drafted: set = set()
    my_state = RosterState(cfg=cfg, draft_position=my_position)

    proj_board = board.dropna(subset=["p50"]).copy()

    for rnd in range(1, rounds + 1):
        for seat in range(1, teams + 1):
            on_clock = _seat_on_clock(rnd, seat, teams)
            if on_clock == my_position:
                pid = _our_pick(proj_board, board, my_state, replacement, cfg)
            else:
                pid = _bot_pick(board, drafted, slot_fills[on_clock], cfg)
            if pid is None:
                continue
            pos = board.loc[board["player_id"] == pid, "position"]
            pos = pos.iloc[0] if not pos.empty else None
            drafted.add(pid)
            rosters[on_clock].append(pid)
            _fill_slot(pos, slot_fills[on_clock], cfg)
            if on_clock == my_position:
                my_state.record_pick(pid, pos, mine=True)
            else:
                my_state.drafted.add(pid)
    return rosters


def _seat_on_clock(rnd: int, seat_index: int, teams: int) -> int:
    """Which draft_position is picking at this (round, seat). Snake order."""
    if rnd % 2 == 1:
        return seat_index
    return teams - seat_index + 1


def _our_pick(proj_board: pd.DataFrame, full_board: pd.DataFrame, state: RosterState,
              replacement, cfg: LeagueConfig) -> Optional[str]:
    """Our pick via the recommender; fall back to best-ADP K/DEF when only those remain."""
    rec = recommend(proj_board, state, replacement, cfg=cfg, top_n=1)
    if not rec.empty:
        return rec.iloc[0]["player_id"]
    # only non-modeled (K/DEF) slots left -> stream best available by ADP
    avail = full_board[~full_board["player_id"].isin(state.drafted)].dropna(subset=["adp"])
    avail = avail.sort_values("adp")
    return avail.iloc[0]["player_id"] if not avail.empty else None


def optimal_lineup_value(roster_ids: List[str], pos_map: Dict[str, str],
                         fppg_map: Dict[str, float], cfg: LeagueConfig) -> float:
    """Availability-neutral starting-lineup value: best actual-FPPG legal lineup.

    K/DEF contribute a constant 0 (unmodeled, identical across teams -> cancels).
    """
    players = [(pid, pos_map.get(pid), fppg_map.get(pid, 0.0)) for pid in roster_ids]
    # only modeled positions score; sort each position desc by actual fppg
    by_pos: Dict[str, List[float]] = {}
    for pid, pos, val in players:
        if pos in cfg.roster.modeled_positions:
            by_pos.setdefault(pos, []).append(val)
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    total = 0.0
    used = {pos: 0 for pos in cfg.roster.modeled_positions}
    # pure starters
    for pos in cfg.roster.modeled_positions:
        need = cfg.roster.slots.get(pos, 0)
        avail = by_pos.get(pos, [])
        take = min(need, len(avail))
        total += sum(avail[:take])
        used[pos] = take
    # flex from leftover flex-eligible
    flex_need = cfg.roster.slots.get("FLEX", 0)
    leftovers = []
    for pos in cfg.roster.flex_eligible:
        leftovers += by_pos.get(pos, [])[used[pos]:]
    leftovers.sort(reverse=True)
    total += sum(leftovers[:flex_need])
    return total


@dataclass
class SimResult:
    per_season_slot: pd.DataFrame   # season, slot, our_value, mean_bot_value, diff
    gate: PairedTest


def run_draft_sim(
    projections: pd.DataFrame,
    adp: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    sim_eligible_seasons: Sequence[int],
    slots: Optional[Sequence[int]] = None,
    cfg: Optional[LeagueConfig] = None,
) -> SimResult:
    """Run Tier-2 over SIM_ELIGIBLE seasons and draft slots; paired gate test."""
    cfg = cfg or load_config()
    slots = list(slots) if slots is not None else list(range(1, cfg.teams + 1))
    rows = []
    for season in sim_eligible_seasons:
        proj_s = projections[projections["season"] == season]
        adp_s = adp[adp["season"] == season]
        lbl_s = labels[labels["season"] == season]
        if proj_s.empty or adp_s.empty:
            continue
        board = adp_s.merge(
            proj_s[["player_id", "p10", "p50", "p90"]], on="player_id", how="left")
        if "position" not in board.columns:
            board = board.merge(proj_s[["player_id", "position"]], on="player_id", how="left")
        pos_map = dict(zip(board["player_id"], board["position"]))
        fppg_map = dict(zip(lbl_s["player_id"], lbl_s["fppg"]))
        replacement = build_replacement_from_projections(
            proj_s.rename(columns={"position": "position"}), cfg=cfg)

        for slot in slots:
            rosters = simulate_draft(board, cfg, slot, replacement)
            our_val = optimal_lineup_value(rosters[slot], pos_map, fppg_map, cfg)
            bot_vals = [optimal_lineup_value(rosters[t], pos_map, fppg_map, cfg)
                        for t in rosters if t != slot]
            rows.append({"season": int(season), "slot": int(slot),
                         "our_value": our_val, "mean_bot_value": float(np.mean(bot_vals)),
                         "diff": our_val - float(np.mean(bot_vals))})
    per = pd.DataFrame(rows)
    gate = paired_test(per["our_value"], per["mean_bot_value"]) if not per.empty \
        else paired_test([], [])
    return SimResult(per_season_slot=per, gate=gate)
