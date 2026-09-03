"""Multi-platform consensus rankings from a static CSV export.

The Yahoo Fantasy API is still blocked pending permission, so the platform the
draft actually runs on has never been the market the engine models — it has
used FFC, a composite mock-draft board. This is the bridge: a hand-exported
CSV carrying one column per platform, so the ORDER can come from Yahoo while
the API remains out of reach. `adp.source` selects between them, and
`yahoo_current` drops into the same slot when the API opens.

**These are RANKS, not ADP, and the difference is load-bearing.** Gibbs is
`1` here and `1.6` in FFC — an average pick position with real decimals.
Ranks are dense and evenly spaced; draft positions are not, and the gaps are
exactly what a survival curve is made of. Feeding ranks straight into the
lognormal would compress the tail and overstate how long everyone lasts.

So the order comes from the platform column and the SCALE comes from the
market: `to_pick_positions` maps the platform's ranking onto the observed ADP
distribution by quantile. A player Yahoo ranks 40th is priced at whatever pick
the 40th-most-drafted player actually goes at.

Measured against FFC on the 2026 board, restricted to the matched set and
compared rank-to-rank so the two depths do not confound it: Spearman 0.92, mean
absolute difference 6.6 ranks over the first five rounds and 12.3 over ten.
Close, but not the same board — Tucker Kraft is FFC 113 and Yahoo 71, and an
engine reading the wrong one expects him to last 42 picks longer than he will.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.core.names import normalize_name
from src.platform.store import blobs
from src.platform.store.manifest import ManifestEntry, utc_now

SOURCE: Final = "rankings_csv"

#: Platform columns, as spelled in the export. `Expert` is deliberately NOT
#: here: it is a set of analyst rankings, i.e. somebody's projections, and the
#: ADP wall lets the market price availability and never value. Reading it
#: would make every "beats the market" claim circular.
PLATFORMS: Final = ("Sleeper", "ESPN", "Yahoo", "Underdog", "CBS", "FFPC")

#: Columns the export must carry for any of this to mean anything.
REQUIRED: Final = ("Player", "POS", "Team", "AVG")


@dataclass(frozen=True)
class Pull:
    frame: pd.DataFrame
    entry: ManifestEntry


def _frame_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df[sorted(df.columns)].reset_index(drop=True).to_parquet(buf, index=False)
    return buf.getvalue()


def load(path: Path, *, blob_root=None) -> Pull:
    """Read the export, keeping the platform columns and a dispersion measure.

    `rank_spread` is the disagreement BETWEEN platforms, which is a different
    quantity from FFC's `adp_stdev` (variation *within* one market's drafts).
    Both are real; a player the platforms disagree about has an uncertain draft
    position however tight any single market is on him.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no rankings export at {path}")
    raw = pd.read_csv(path)

    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing {missing}; expected a Flock-style export "
            f"with {list(REQUIRED)} and one column per platform")

    present = [c for c in PLATFORMS if c in raw.columns]
    if not present:
        raise ValueError(
            f"{path.name} carries none of the platform columns {PLATFORMS}")

    out = pd.DataFrame({
        "player_name": raw["Player"].astype(str),
        "match_key": raw["Player"].map(normalize_name),
        "position": raw["POS"].astype(str).str.replace(r"\d+$", "", regex=True),
        "team": raw["Team"].astype(str),
        "consensus_rank": pd.to_numeric(raw["AVG"], errors="coerce"),
    })
    for column in present:
        out[f"rank_{column.lower()}"] = pd.to_numeric(raw[column],
                                                      errors="coerce")

    ranks = out[[f"rank_{c.lower()}" for c in present]]
    # The mean across the PLATFORMS, which is not the export's own `AVG`:
    # that column folds in `Expert`, and analyst opinion may not price
    # availability (see PLATFORMS above). Averaged over whichever platforms
    # carry the player, with `n_platforms` recording how many that was.
    out["platform_mean_rank"] = ranks.mean(axis=1)
    out["rank_spread"] = ranks.max(axis=1) - ranks.min(axis=1)
    out["rank_sd"] = ranks.std(axis=1)
    out["n_platforms"] = ranks.notna().sum(axis=1)
    out = out.dropna(subset=["consensus_rank"]).drop_duplicates("match_key")

    digest = blobs.put(_frame_bytes(out), root=blob_root)
    entry = ManifestEntry(
        source=SOURCE,
        logical_asset="overall_ppr_rankings",
        params_key=blobs.params_key({"file": path.name,
                                     "platforms": list(present)}),
        digest=digest,
        fetched_at=utc_now(),
        canonicalizer="identity",
    )
    return Pull(frame=out, entry=entry)


def to_pick_positions(rank: pd.Series, adp_scale: pd.Series) -> pd.Series:
    """Map a platform RANKING onto an observed ADP distribution, by quantile.

    The platform says who goes first; the market says how far apart picks
    actually are. Ranks are evenly spaced and draft positions are not — the top
    of a board is dense and the middle rounds spread out — so using rank as a
    pick number would compress exactly the gaps a survival curve reads.

    Rank i of n therefore receives the i-th smallest observed ADP. Order comes
    from the platform, spacing from the market, and the result is in pick units
    so `survival_probability` keeps its meaning.

    Players with no rank keep NaN rather than being pushed to the end: an
    unranked player is unknown, not undraftable, and `survival_probability`
    already treats a missing ADP as "the market has no opinion".
    """
    scale = np.sort(pd.to_numeric(adp_scale, errors="coerce").dropna().to_numpy())
    if scale.size == 0:
        return pd.Series(np.nan, index=rank.index, dtype="float64")

    numeric = pd.to_numeric(rank, errors="coerce")
    ordered = numeric.rank(method="first")            # 1..n over the ranked
    n = int(numeric.notna().sum())
    if n == 0:
        return pd.Series(np.nan, index=rank.index, dtype="float64")

    # Quantile position of each rank, mapped onto the observed distribution.
    quantile = (ordered - 0.5) / n
    index = np.clip((quantile * scale.size).astype("float64"),
                    0, scale.size - 1)
    picked = pd.Series(np.nan, index=rank.index, dtype="float64")
    valid = numeric.notna()
    picked.loc[valid] = scale[index[valid].astype(int)]
    return picked
