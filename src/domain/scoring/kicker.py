"""Kicker scoring (§10.2).

MEASURED 2026-08-04: ``load_player_stats`` ships made-field-goal counts already
bucketed by distance — ``fg_made_0_19`` … ``fg_made_60_`` — at 0% null across
2012-2025, plus ``pat_made``. Probe P3 asked whether play-by-play derivation was
needed for the configured bands; it is not.

The data's buckets and the league's bands are different partitions, so this
module maps one onto the other **generically** and refuses to guess when a data
bucket straddles a configured band boundary. Hard-coding the current mapping
would silently mis-score any league whose bands differ (e.g. a 0-39/40-49/50+
league is fine, but one paying differently at 30-39 vs 20-29 is not derivable
from these columns at all, and must be told so).

No penalty for a missed field goal in this league, so ``fg_miss`` defaults to 0
and kicker value reduces to attempt volume and distance mix.
"""
from __future__ import annotations

from typing import Final, Mapping, Sequence

import pandas as pd

from src.core.errors import ConfigError

#: nflverse made-FG columns and the inclusive yard range each covers.
#: ``fg_made_60_`` is open-ended upward; 99 is the practical bound and matches
#: the configured domain in §10.5 rule 3.
FG_MADE_BUCKETS: Final[tuple[tuple[str, int, int], ...]] = (
    ("fg_made_0_19", 0, 19),
    ("fg_made_20_29", 20, 29),
    ("fg_made_30_39", 30, 39),
    ("fg_made_40_49", 40, 49),
    ("fg_made_50_59", 50, 59),
    ("fg_made_60_", 60, 99),
)
PAT_MADE_COL: Final = "pat_made"
FG_MISSED_COL: Final = "fg_missed"

REQUIRED_COLUMNS: Final = tuple(
    [c for c, _, _ in FG_MADE_BUCKETS] + [PAT_MADE_COL, FG_MISSED_COL]
)


def map_buckets_to_bands(
    bands: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Column -> points, by locating each data bucket inside a configured band.

    Raises when a bucket straddles a band boundary, because the columns carry
    counts rather than individual distances and the split is therefore not
    recoverable. Failing loudly beats scoring a straddling league wrongly.
    """
    mapping: dict[str, float] = {}
    for col, lo, hi in FG_MADE_BUCKETS:
        containing = [b for b in bands if b[0] <= lo and hi <= b[1]]
        if not containing:
            overlapping = [b for b in bands if not (b[1] < lo or b[0] > hi)]
            raise ConfigError(
                f"kicker column {col} covers {lo}-{hi}, which straddles the "
                f"configured bands {[list(b) for b in overlapping]}. "
                f"player_stats carries counts per bucket, not individual "
                f"distances, so this split cannot be recovered — either align "
                f"scoring.kicker.fg_bands to the nflverse buckets "
                f"{[(lo_, hi_) for _, lo_, hi_ in FG_MADE_BUCKETS]} or derive "
                f"kicker scoring from play-by-play."
            )
        if len(containing) > 1:  # defensive: rule 3 forbids overlap
            raise ConfigError(f"kicker column {col} matches multiple bands")
        mapping[col] = float(containing[0][2])
    return mapping


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def score_kicker(df: pd.DataFrame, kicker: Mapping[str, object]) -> pd.Series:
    """Fantasy points per kicker row."""
    if len(df) == 0:
        return pd.Series(dtype="float64")

    bands = kicker["fg_bands"]  # type: ignore[index]
    points_by_col = map_buckets_to_bands(bands)  # type: ignore[arg-type]

    total = pd.Series(0.0, index=df.index, dtype="float64")
    for col, points in points_by_col.items():
        total = total + _numeric(df, col) * points
    total = total + _numeric(df, PAT_MADE_COL) * float(kicker.get("pat", 0))

    miss = float(kicker.get("fg_miss", 0) or 0)
    if miss:
        total = total + _numeric(df, FG_MISSED_COL) * miss
    return total
