#!/usr/bin/env python
"""One-time ADP scoring-format gap check (DrafterSpec.md §3 / §4.5).

Bounds the gap between player value under OUR ``league.yaml`` and generic full-PPR
(the closest available ADP format) by comparing season-value rankings. Confirms the
format mismatch does not distort draft order: a high Spearman + small rank shift
means generic full-PPR ADP is a faithful input. Reads outcomes only (not ADP), so
it is wall-safe; lives as a script, not in the projection engine.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import spearmanr

from src.config import ScoringConfig, load_config
from src.ingest import sources
from src.labels.scoring import score_dataframe

# Generic full-PPR scoring (what public ADP implicitly reflects): note int = -2.
GENERIC_PPR = ScoringConfig(
    pass_yd=0.04, pass_td=4, interception=-2, rush_yd=0.1, rush_td=6,
    rec=1, rec_yd=0.1, rec_td=6, two_pt=2, fumble_lost=-2, return_td=6,
)


def _season_value(stats: pd.DataFrame, scoring: ScoringConfig) -> pd.Series:
    df = stats.copy()
    df["pts"] = score_dataframe(df, scoring)
    return df.groupby("player_id")["pts"].sum()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bound the ADP scoring-format gap (§3).")
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--top", type=int, default=200, help="Top-N by league value to inspect")
    args = parser.parse_args()

    cfg = load_config()
    stats = sources.load_player_stats([args.season])
    if "season_type" in stats.columns:
        stats = stats[stats["season_type"] == "REG"]

    league_val = _season_value(stats, cfg.scoring)
    generic_val = _season_value(stats, GENERIC_PPR)
    df = pd.DataFrame({"league": league_val, "generic": generic_val}).dropna()
    df = df.sort_values("league", ascending=False).head(args.top)
    df["rank_league"] = df["league"].rank(ascending=False, method="min")
    df["rank_generic"] = df["generic"].rank(ascending=False, method="min")
    df["rank_shift"] = (df["rank_league"] - df["rank_generic"]).abs()

    rho, _ = spearmanr(df["league"], df["generic"])
    print(f"Season {args.season}  (top {args.top} by league value)")
    print(f"  Spearman(league, generic full-PPR) = {rho:.4f}")
    print(f"  mean |rank shift| = {df['rank_shift'].mean():.2f}")
    print(f"  max  |rank shift| = {df['rank_shift'].max():.0f}")
    print(f"  pct with shift > 10 = {100 * (df['rank_shift'] > 10).mean():.1f}%")
    print("  -> generic full-PPR ADP is a faithful input if rho high & shifts small.")


if __name__ == "__main__":
    main()
