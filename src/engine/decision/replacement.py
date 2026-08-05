"""Fixed preseason replacement level via greedy lineup-fill (DrafterSpec.md §4.8.1).

The recommender's most important calculation — it owns STRUCTURAL scarcity (RB cliff
vs QB depth, a preseason truth). Replacement is computed ONCE from preseason
projections and never recomputed mid-draft (§3): dynamic replacement + VONA would
double-count intra-draft scarcity.

Greedy lineup-fill resolves FLEX endogenously:
  1. Rank all flex-eligible skill players (RB/WR/TE) by projected value.
  2. Walk top-down: each fills an open PURE slot at his position
     (teams × slots[P]); if pure slots are full, fills an open FLEX slot
     (teams × slots[FLEX], eligibility from roster.flex_eligible); else non-starter.
  3. replacement[P] = value of the highest-ranked player at P who earned NO starting
     slot (the first non-starter at P).
QB has no flex: replacement[QB] = the (teams × slots[QB] + 1)-th QB.


PORTED from src/recommender/replacement.py (DraftEngineDesign.md §9.2).
Tier-2 fallback path (§7.3). Behavior unchanged from v1; the only edits
are the v2 config API (load_league, flex_eligibility) and the DEF -> DST
position rename (§10.7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from src.core.config.league import LeagueConfig, load_league


@dataclass
class ReplacementLevels:
    levels: Dict[str, float]                 # position -> replacement value
    starters: set = field(default_factory=set)  # player_ids that earned a starting slot

    def get(self, position: str) -> float:
        return self.levels.get(position, 0.0)


def compute_replacement(
    players: pd.DataFrame,
    *,
    cfg: LeagueConfig | None = None,
    value_col: str = "value",
) -> ReplacementLevels:
    """Compute fixed replacement levels from a projected-value board.

    ``players``: columns [player_id, position, <value_col>]. Only modeled positions
    are ranked; K/DST are filled at replacement elsewhere.
    """
    cfg = cfg or load_league()
    teams = cfg.teams
    flex_elig = set(cfg.roster.flex_eligible)
    flex_slots_total = teams * cfg.roster.slots.get("FLEX", 0)

    df = players[players["position"].isin(cfg.roster.modeled_positions)].copy()
    df = df.dropna(subset=[value_col]).sort_values(value_col, ascending=False)

    pure_capacity = {p: teams * cfg.roster.slots.get(p, 0) for p in cfg.roster.modeled_positions}
    pure_used = {p: 0 for p in cfg.roster.modeled_positions}
    flex_used = 0
    starters: set = set()
    first_non_starter: Dict[str, float] = {}

    # Flex-eligible skill players: greedy fill pure then flex.
    skill = df[df["position"].isin(flex_elig)]
    for _, row in skill.iterrows():
        pos = row["position"]
        pid = row["player_id"]
        if pure_used[pos] < pure_capacity[pos]:
            pure_used[pos] += 1
            starters.add(pid)
        elif flex_used < flex_slots_total:
            flex_used += 1
            starters.add(pid)
        else:
            first_non_starter.setdefault(pos, float(row[value_col]))

    # Non-flex modeled positions (QB): pure slots only.
    for pos in cfg.roster.modeled_positions:
        if pos in flex_elig:
            continue
        pos_players = df[df["position"] == pos]
        cap = pure_capacity[pos]
        for i, (_, row) in enumerate(pos_players.iterrows()):
            if i < cap:
                pure_used[pos] += 1
                starters.add(row["player_id"])
            else:
                first_non_starter.setdefault(pos, float(row[value_col]))
                break

    # Fallback: if a position never produced a non-starter (tiny board), use the
    # worst observed value at that position.
    levels = {}
    for pos in cfg.roster.modeled_positions:
        if pos in first_non_starter:
            levels[pos] = first_non_starter[pos]
        else:
            pos_vals = df[df["position"] == pos][value_col]
            levels[pos] = float(pos_vals.min()) if not pos_vals.empty else 0.0
    return ReplacementLevels(levels=levels, starters=starters)
