"""Tier-2 recommendation policy (§7.3, §3.4, §8.7).

Wraps the VONA scorer with the roster-shaping rules that keep its output
usable: positional caps and replacement-level treatment of K/DST.

Two corrections that matter, both visible the first time this ran against real
2026 ADP:

1. **The wait horizon is the pick AFTER the one being made**, not the current
   pick. VONA subtracts "value of the best same-position player still there at
   my next turn"; passing the current pick makes that term compare a player
   against himself, which collapses nearly every score to zero.

2. **K and DST are draft-time replacement, never recommended early** (§3.4).
   They carry no prior-season value here, so their marginal is 0 — which sorts
   *above* any skill player with a negative score and puts a kicker at the top
   of the board. Expected contribution nearly cancels across teams anyway
   (every team starts exactly one of each); only their variance matters, and
   that belongs to the simulator, not to draft-time value.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.config.league import LeagueConfig
from src.core.config.strategy import StrategyConfig
from src.core.constants import EMPIRICAL_POSITIONS
from src.engine.decision import replacement as replacement_mod
from src.engine.decision import roster_state, vona

#: Penalty applied to a position the roster has filled to its cap, and to K/DST
#: before the last rounds. Large enough to sink them below any real candidate
#: without being infinite, so the ordering among sunk players is still sane.
SINK = 1_000.0


@dataclass(frozen=True)
class Recommendation:
    ranked: pd.DataFrame
    next_pick: int | None
    current_pick: int
    replacement: replacement_mod.ReplacementLevels
    stale_flags: tuple[str, ...]

    @property
    def leader(self) -> pd.Series | None:
        return None if self.ranked.empty else self.ranked.iloc[0]


def kdst_rounds_remaining(rs: roster_state.RosterState, league: LeagueConfig) -> int:
    """How many picks I have left. K/DST become live in the last two."""
    return rs.remaining_picks()


def score(
    available: pd.DataFrame,
    rs: roster_state.RosterState,
    league: LeagueConfig,
    strategy: StrategyConfig,
    *,
    current_pick: int,
    value_col: str = "value",
    stale_flags: tuple[str, ...] = (),
) -> Recommendation:
    """Rank the board for the pick being made at ``current_pick``."""
    levels = replacement_mod.compute_replacement(
        available, cfg=league, value_col=value_col
    )
    # correction 1: the horizon is my NEXT turn, strictly after this one
    next_pick = rs.next_my_pick(after_overall=current_pick)

    scored = vona.score_board(
        available, rs, levels, next_pick=next_pick, value_col=value_col
    )
    scored = _apply_roster_shaping(scored, rs, league, strategy)
    scored = scored.sort_values("vona_score", ascending=False).reset_index(drop=True)
    return Recommendation(
        ranked=scored, next_pick=next_pick, current_pick=current_pick,
        replacement=levels, stale_flags=tuple(stale_flags),
    )


def _apply_roster_shaping(
    scored: pd.DataFrame, rs: roster_state.RosterState,
    league: LeagueConfig, strategy: StrategyConfig,
) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    out["sink_reason"] = ""

    # correction 2: K/DST are replacement-level draft slots until the endgame
    picks_left = rs.remaining_picks()
    kdst_live = picks_left <= len(EMPIRICAL_POSITIONS)
    if not kdst_live:
        mask = out["position"].isin(EMPIRICAL_POSITIONS)
        out.loc[mask, "vona_score"] -= SINK
        out.loc[mask, "sink_reason"] = "k_dst_wait_for_endgame"

    # positional caps from strategy.yaml (never model-affecting, so tuning them
    # leaves every trained model valid — §10.1)
    for position, cap in strategy.max_per_position.items():
        if rs.position_count(position) >= int(cap):
            mask = out["position"] == position
            out.loc[mask, "vona_score"] -= SINK
            out.loc[mask & (out["sink_reason"] == ""), "sink_reason"] = \
                f"at_cap_{position}_{cap}"

    # a roster slot I can no longer use is worth nothing this pick
    if picks_left <= 0:
        out["vona_score"] -= SINK
    return out
