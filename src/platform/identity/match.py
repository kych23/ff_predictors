"""Name-based resolution between sources (§11.4).

FFC publishes names, not ids, so the ADP table has to be matched to nflverse
players. This is the deterministic tier of the §11.4 cascade — normalized name
plus position — with fuzzy matching behind an explicit threshold and everything
that fails landing in a reported ``unresolved`` bucket rather than silently
vanishing.

An unmatched player is not dropped. A player with a projection but no ADP is
still draftable; a player with ADP but no projection is exactly the case §19
routes to replacement level with a ``no_projection`` flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from src.core.names import normalize_name

#: rapidfuzz threshold and the margin the best match must clear (§11.4).
FUZZY_MIN_SCORE: Final = 92
FUZZY_MIN_MARGIN: Final = 6


@dataclass(frozen=True)
class MatchReport:
    matched: pd.DataFrame
    unresolved_left: pd.DataFrame
    unresolved_right: pd.DataFrame

    @property
    def coverage(self) -> float:
        total = len(self.matched) + len(self.unresolved_left)
        return len(self.matched) / total if total else 0.0

    def describe(self) -> str:
        return (
            f"matched {len(self.matched)}, "
            f"unmatched-left {len(self.unresolved_left)}, "
            f"unmatched-right {len(self.unresolved_right)}, "
            f"coverage {self.coverage:.1%}"
        )


def match_by_name(
    left: pd.DataFrame, right: pd.DataFrame, *,
    left_name: str, right_name: str,
    left_pos: str = "position", right_pos: str = "position",
    use_fuzzy: bool = True,
) -> MatchReport:
    """Resolve ``left`` against ``right`` on (normalized name, position).

    Position is part of the key deliberately: name collisions across positions
    are common enough in the NFL that name alone produces wrong joins, and a
    wrong join is worse than an unresolved row because it never gets reported.
    """
    lf = left.copy()
    rf = right.copy()
    lf["_key"] = lf[left_name].map(normalize_name) + "|" + lf[left_pos].astype(str)
    rf["_key"] = rf[right_name].map(normalize_name) + "|" + rf[right_pos].astype(str)

    rf_unique = rf.drop_duplicates("_key", keep="first")
    merged = lf.merge(rf_unique, on="_key", how="left", suffixes=("", "_r"))
    matched = merged[merged["_key"].isin(set(rf_unique["_key"]))].copy()
    missed = merged[~merged["_key"].isin(set(rf_unique["_key"]))].copy()

    if use_fuzzy and len(missed):
        matched, missed = _fuzzy_pass(matched, missed, rf_unique, right_name)

    used = set(matched["_key"])
    leftover_right = rf_unique[~rf_unique["_key"].isin(used)].copy()
    for frame in (matched, missed, leftover_right):
        frame.drop(columns=["_key"], inplace=True, errors="ignore")
    return MatchReport(matched=matched, unresolved_left=missed,
                       unresolved_right=leftover_right)


def _fuzzy_pass(matched, missed, rf_unique, right_name):
    """Second tier: rapidfuzz, requiring a unique winner clear of the field."""
    try:
        from rapidfuzz import fuzz, process
    except ImportError:  # rapidfuzz is optional; deterministic tier still works
        return matched, missed

    choices = rf_unique["_key"].tolist()
    still_missing = []
    recovered = []
    for _, row in missed.iterrows():
        # token_sort_ratio per §11.4 — rapidfuzz's default WRatio scores
        # substrings much higher and would let a bare surname clear 92.
        results = process.extract(row["_key"], choices, limit=2,
                                  scorer=fuzz.token_sort_ratio)
        if not results:
            still_missing.append(row)
            continue
        best = results[0]
        runner = results[1][1] if len(results) > 1 else 0
        if best[1] >= FUZZY_MIN_SCORE and (best[1] - runner) >= FUZZY_MIN_MARGIN:
            merged_row = row.copy()
            merged_row["_key"] = best[0]
            recovered.append(merged_row)
        else:
            still_missing.append(row)

    if recovered:
        matched = pd.concat([matched, pd.DataFrame(recovered)], ignore_index=True)
    missed = pd.DataFrame(still_missing) if still_missing else missed.iloc[0:0]
    return matched, missed
