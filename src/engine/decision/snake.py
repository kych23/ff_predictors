"""Snake-draft mechanics: turn order, slot filling, and the ADP-bot heuristic.

Extracted from ``src/benchmark/draft_sim.py`` (deleted in the monte-carlo
strip — its deterministic single-draft simulation is superseded by the
Monte Carlo layer) because these pieces are reused elsewhere: the turn/slot
math is generic snake-draft bookkeeping, and ``_bot_pick`` backs
``api/draft_service.py``'s bot-pick endpoint (demo/opponent-drafting).


PORTED from src/recommender/snake.py (DraftEngineDesign.md §9.2).
Tier-2 fallback path (§7.3). Behavior unchanged from v1; the only edits
are the v2 config API (load_league, flex_eligibility) and the DEF -> DST
position rename (§10.7).
"""
from __future__ import annotations

import pandas as pd

from src.core.config.league import LeagueConfig


def seat_on_clock(rnd: int, seat_index: int, teams: int) -> int:
    """Which draft_position is picking at this (round, seat). Snake order."""
    if rnd % 2 == 1:
        return seat_index
    return teams - seat_index + 1


def has_open_slot(position: str, slot_fill: dict[str, int], cfg: LeagueConfig) -> bool:
    if slot_fill.get(position, 0) < cfg.roster.slots.get(position, 0):
        return True
    if position in cfg.roster.flex_eligible and \
            slot_fill.get("FLEX", 0) < cfg.roster.slots.get("FLEX", 0):
        return True
    return slot_fill.get("BENCH", 0) < cfg.roster.slots.get("BENCH", 0)


def fill_slot(position: str, slot_fill: dict[str, int], cfg: LeagueConfig) -> None:
    if slot_fill.get(position, 0) < cfg.roster.slots.get(position, 0):
        slot_fill[position] = slot_fill.get(position, 0) + 1
    elif position in cfg.roster.flex_eligible and \
            slot_fill.get("FLEX", 0) < cfg.roster.slots.get("FLEX", 0):
        slot_fill["FLEX"] = slot_fill.get("FLEX", 0) + 1
    else:
        slot_fill["BENCH"] = slot_fill.get("BENCH", 0) + 1


def bot_pick(board: pd.DataFrame, drafted: set, slot_fill: dict[str, int],
            cfg: LeagueConfig) -> str | None:
    """ADP bot: lowest-ADP available player that fits any open slot (pure/flex/bench)."""
    undrafted = board[~board["player_id"].isin(drafted)]
    avail = undrafted.dropna(subset=["adp"]).sort_values("adp")
    for _, row in avail.iterrows():
        pos = row["position"]
        if has_open_slot(pos, slot_fill, cfg):
            return row["player_id"]
    if not avail.empty:
        return avail.iloc[0]["player_id"]
    # ADP pool exhausted (thin ADP coverage): fall back to best-projected
    # undrafted player so the bot's roster never silently comes up short.
    if not undrafted.empty:
        return undrafted.sort_values("p50", ascending=False).iloc[0]["player_id"]
    return None
