"""VONA — Value Of Not Available, the core scoring loop (DrafterSpec.md §4.8.1).

score(p) = marginal starting-lineup value of p to MY roster
           − E[marginal value of the best same-position player still available at my
               NEXT pick]

The expectation uses ADP-survival P(player survives to next pick). This naturally
encodes "take the scarce position now, wait on the deep one." VONA owns ALL
intra-draft scarcity; fixed replacement (replacement.py) owns structural scarcity —
one effect, one place, no double-count (§3).

At the last pick (no next pick), the wait term is defined as 0, so score reduces to
pure marginal lineup value (§4.8.1).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .replacement import ReplacementLevels
from .roster_state import RosterState
from .survival import survival_probability


def marginal_value(value: float, position: str, replacement: ReplacementLevels,
                   roster: RosterState) -> float:
    """Marginal value of adding this player to MY lineup, vs replacement.

    Full VOR if the player would start (open pure/flex slot); otherwise bench upside
    = the positive VOR only (future-starter / best-available, §3). Replacement is the
    fixed preseason bar.
    """
    vor = value - replacement.get(position)
    if roster.needs_starter(position):
        return vor
    return max(0.0, vor)  # bench: keep only upside over replacement


def _expected_best_surviving(candidates: pd.DataFrame, position: str, next_pick: int,
                             replacement: ReplacementLevels, value_col: str) -> float:
    """E[value of the best same-position player available at next pick].

    Sorted by value desc; P(player i is the best survivor) =
    survival_i * Π_{j<i}(1 - survival_j). Residual prob (nobody survives) falls back
    to the replacement value (you'd take a replacement-level player).
    """
    if candidates.empty or next_pick is None:
        return replacement.get(position)
    cands = candidates.sort_values(value_col, ascending=False)
    none_survived = 1.0
    expected = 0.0
    for _, row in cands.iterrows():
        s = survival_probability(row["adp"], row.get("adp_stdev", np.nan), next_pick)
        expected += row[value_col] * s * none_survived
        none_survived *= (1.0 - s)
    expected += replacement.get(position) * none_survived
    return expected


def score_board(
    available: pd.DataFrame,
    roster: RosterState,
    replacement: ReplacementLevels,
    *,
    next_pick: Optional[int],
    value_col: str = "value",
) -> pd.DataFrame:
    """Compute the VONA score for every available player.

    ``available``: player_id, position, <value_col>, adp, adp_stdev.
    """
    if available.empty:
        return available.assign(marginal=[], wait_term=[], vona_score=[])

    df = available.copy()
    # precompute the expected best surviving VOR per position (depends only on pos)
    wait_vor_by_pos = {}
    for pos, grp in df.groupby("position"):
        if next_pick is None:
            wait_vor_by_pos[pos] = 0.0
        else:
            ebv = _expected_best_surviving(grp, pos, next_pick, replacement, value_col)
            wait_vor_by_pos[pos] = max(0.0, ebv - replacement.get(pos))

    marg = df.apply(lambda r: marginal_value(r[value_col], r["position"], replacement, roster), axis=1)
    df["marginal"] = marg.values
    df["wait_term"] = df["position"].map(wait_vor_by_pos).fillna(0.0)
    df["vona_score"] = df["marginal"] - df["wait_term"]
    return df.sort_values("vona_score", ascending=False)
