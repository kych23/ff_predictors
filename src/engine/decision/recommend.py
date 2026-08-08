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
    """How many picks I have left. K/DST become live in the last two.

    UNUSED — no caller in `src/`, `scripts/` or `tests/`. `_apply_roster_shaping`
    calls `rs.remaining_picks()` directly. Left as the named statement of the
    rule, but it is not the thing enforcing it.
    """
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
    preseason_board: pd.DataFrame | None = None,
) -> Recommendation:
    """Rank the board for the pick being made at ``current_pick``.

    ``preseason_board`` is the FULL board including already-drafted players.
    Replacement level is structural scarcity and §3 fixes it preseason:
    recomputing it from what is still available makes it fall as a position
    drains, which is intra-draft scarcity — the effect VONA's wait term already
    owns. `replacement.py` states this invariant; this function used to break
    it by passing ``available``.

    The violation hid because it mostly cancels. With an open starter slot
    ``vona = (value - repl) - (E_best - repl) = value - E_best``, so the level
    drops out and no ranking moves — measured identical at picks 25, 49 and 97
    even though replacement[RB] had fallen from 9.92 to 4.83 across that span.
    It stops cancelling once ``marginal`` or ``wait_term`` hits its ``max(0,
    .)`` clamp, which is the late rounds: at pick 133, 51 of 124 players
    changed score and the ordering changed with them.

    Defaults to ``available`` only so existing single-board callers keep
    working; every live path passes the preseason board.
    """
    levels = replacement_mod.compute_replacement(
        available if preseason_board is None else preseason_board,
        cfg=league, value_col=value_col,
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

    # EARLY-ROUND caps, which expire. A second tight end is legal all draft and
    # wrong in round 3: he has no starting slot of his own and reaches the
    # lineup only through FLEX, competing with every back and receiver left.
    # The objective prices that thinly, but thinly is not never — a
    # high-variance TE2 can still take a shortlist place off upside alone, and
    # a shortlist place is what buys entry to the expensive simulation.
    current_round = rs.my_current_round()
    for position, rule in strategy.early_round_caps.items():
        through = int(rule.get("through_round", 0))
        limit = int(rule.get("max", 0))
        if current_round <= through and rs.position_count(position) >= limit:
            mask = out["position"] == position
            out.loc[mask, "vona_score"] -= SINK
            out.loc[mask & (out["sink_reason"] == ""), "sink_reason"] = \
                f"{position}_max_{limit}_through_round_{through}"

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
