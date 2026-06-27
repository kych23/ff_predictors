"""Top-level recommender entry: draft state -> ranked board (DrafterSpec.md §4.8.1).

Ties the pieces together:
  * round-shifted quantile schedule picks each player's risk-adjusted VALUE,
  * fixed preseason replacement (passed in) is the value ruler,
  * VONA scores take-now-vs-wait using ADP survival,
  * the hard roster-completion constraint guarantees a legal lineup.

The wall: this module uses ADP only for survival/timing (recommender side, §4.0).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config import LeagueConfig, load_config

from .quantile_schedule import round_to_alpha, value_vector_at_alpha
from .replacement import ReplacementLevels, compute_replacement
from .roster_state import RosterState
from .vona import score_board


def build_replacement_from_projections(projections: pd.DataFrame,
                                       cfg: Optional[LeagueConfig] = None) -> ReplacementLevels:
    """Fixed preseason replacement from p50 projections (computed ONCE, §3)."""
    cfg = cfg or load_config()
    board = projections[["player_id", "position", "p50"]].rename(columns={"p50": "value"})
    return compute_replacement(board, cfg=cfg, value_col="value")


def recommend(
    projections: pd.DataFrame,
    roster: RosterState,
    replacement: ReplacementLevels,
    *,
    cfg: Optional[LeagueConfig] = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """Rank the best available picks for MY roster right now.

    ``projections``: player_id, position, p10, p50, p90, adp, adp_stdev.
    """
    cfg = cfg or load_config()
    rnd = roster.my_current_round()
    alpha = round_to_alpha(rnd, cfg.roster.rounds)

    board = projections.copy()
    board["value"] = value_vector_at_alpha(board["p10"], board["p50"], board["p90"], alpha)
    available = board[~board["player_id"].isin(roster.drafted)].copy()
    if available.empty:
        return available

    next_pick = roster.next_my_pick()
    scored = score_board(available, roster, replacement, next_pick=next_pick)

    # --- soft positional caps: heavily discourage drafting past a position max ---
    # (e.g. roster.max_per_position = {TE: 2}). A large penalty sinks capped
    # positions to the bottom without hard-banning them (still pickable as a last
    # resort), then re-sort by the adjusted score.
    caps = cfg.roster.max_per_position
    if caps:
        CAP_PENALTY = 1000.0
        for pos, cap in caps.items():
            if roster.position_count(pos) >= cap:
                scored.loc[scored["position"] == pos, "vona_score"] -= CAP_PENALTY
        scored = scored.sort_values("vona_score", ascending=False)

    # --- hard roster-completion constraint (§4.8.1) ---
    remaining = roster.remaining_picks()
    unfilled = roster.total_unfilled_mandatory()
    scored["forced_completion"] = False
    if remaining <= unfilled:
        # restrict to players that fill a still-needed mandatory starter slot
        needed_positions = {p for p in cfg.roster.modeled_positions if roster.needs_starter(p)}
        if needed_positions:
            mask = scored["position"].isin(needed_positions)
            scored.loc[mask, "forced_completion"] = True
            scored = scored[mask] if mask.any() else scored

    scored = scored.assign(draft_round=rnd, target_quantile=alpha)
    return scored.head(top_n).reset_index(drop=True)
