"""Cross-season prior-production features (DrafterSpec.md §4.5.1).

The stable, most-predictive block. Aggregates volume/efficiency/fppg from seasons
``< Y`` into **rates** (per-game / per-opportunity, never season totals) so 16- and
17-game eras mix without distortion (§4.4.3), then EWMA-decays across seasons with
``ewm_halflife_seasons`` (config, tuned on CV folds — §4.6.1) so recent seasons
weigh more. Output: one row per ``player_id`` with ``prior_*``-style columns.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.labels.scoring import score_dataframe

# Per-game volume columns derived from weekly sums.
_SUM_COLS = [
    "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
    "carries", "rushing_yards", "rushing_tds",
    "attempts", "completions", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered",
]


def _season_aggregates(stats: pd.DataFrame, pfr_to_gsis: Optional[Dict[str, str]],
                       snaps: Optional[pd.DataFrame], opp: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Per (player_id, season) season-level rate aggregates."""
    if stats.empty:
        return pd.DataFrame()
    df = stats.copy()
    df["week_points"] = score_dataframe(df)
    for c in _SUM_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    g = df.groupby(["player_id", "season"])
    agg = g.agg(
        games=("week", "nunique"),
        fp=("week_points", "sum"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        rec_yards=("receiving_yards", "sum"),
        rec_tds=("receiving_tds", "sum"),
        air_yards=("receiving_air_yards", "sum"),
        carries=("carries", "sum"),
        rush_yards=("rushing_yards", "sum"),
        rush_tds=("rushing_tds", "sum"),
        pass_att=("attempts", "sum"),
        completions=("completions", "sum"),
        pass_yards=("passing_yards", "sum"),
        pass_tds=("passing_tds", "sum"),
        ints=("passing_interceptions", "sum"),
        sacks=("sacks_suffered", "sum"),
        target_share=("target_share", "mean") if "target_share" in df.columns else ("targets", lambda x: np.nan),
        air_yards_share=("air_yards_share", "mean") if "air_yards_share" in df.columns else ("targets", lambda x: np.nan),
    ).reset_index()

    gp = agg["games"].clip(lower=1)
    # volume rates (bare names here; build_prior_production adds the prior_ prefix)
    agg["fppg"] = agg["fp"] / gp
    agg["targets_per_game"] = agg["targets"] / gp
    agg["carries_per_game"] = agg["carries"] / gp
    agg["touches_per_game"] = (agg["carries"] + agg["receptions"]) / gp
    agg["pass_attempts_per_game"] = agg["pass_att"] / gp
    agg["rush_attempts_per_game"] = agg["carries"] / gp
    agg["dropbacks_per_game"] = (agg["pass_att"] + agg["sacks"]) / gp
    # efficiency rates
    agg["catch_rate"] = agg["receptions"] / agg["targets"].replace(0, np.nan)
    agg["yards_per_target"] = agg["rec_yards"] / agg["targets"].replace(0, np.nan)
    agg["yards_per_reception"] = agg["rec_yards"] / agg["receptions"].replace(0, np.nan)
    agg["rec_td_rate"] = agg["rec_tds"] / agg["targets"].replace(0, np.nan)
    agg["yards_per_carry"] = agg["rush_yards"] / agg["carries"].replace(0, np.nan)
    agg["rush_td_rate"] = agg["rush_tds"] / agg["carries"].replace(0, np.nan)
    agg["completion_pct"] = agg["completions"] / agg["pass_att"].replace(0, np.nan)
    agg["yards_per_attempt"] = agg["pass_yards"] / agg["pass_att"].replace(0, np.nan)
    agg["pass_td_rate"] = agg["pass_tds"] / agg["pass_att"].replace(0, np.nan)
    agg["int_rate"] = agg["ints"] / agg["pass_att"].replace(0, np.nan)
    agg["sack_rate"] = agg["sacks"] / (agg["pass_att"] + agg["sacks"]).replace(0, np.nan)
    agg["rush_yards_per_game"] = agg["rush_yards"] / gp
    # carry_share approximated from team carries via opportunity if available; else NaN
    agg["carry_share"] = np.nan
    # red-zone opportunity not in base sources -> NaN (M7 from PBP)
    agg["rz_targets_per_game"] = np.nan
    agg["rz_carries_per_game"] = np.nan
    agg["routes_per_game"] = np.nan
    agg["yards_per_route_run"] = np.nan

    # snap share from snap_counts (pfr_player_id -> gsis), season mean of offense_pct
    if snaps is not None and not snaps.empty and pfr_to_gsis:
        s = snaps.copy()
        s["player_id"] = s["pfr_player_id"].astype("string").map(pfr_to_gsis)
        s = s[s["player_id"].notna()]
        snap_season = (s.groupby(["player_id", "season"])["offense_pct"].mean()
                       .reset_index().rename(columns={"offense_pct": "snap_share"}))
        agg = agg.merge(snap_season, on=["player_id", "season"], how="left")
    else:
        agg["snap_share"] = np.nan

    # expected fantasy points / FP-over-expected from ff_opportunity
    if opp is not None and not opp.empty:
        o = opp.copy()
        o = o.rename(columns={"posteam": "team"})
        o = o[pd.to_numeric(o["season"], errors="coerce").notna()].copy()
        o["season"] = pd.to_numeric(o["season"], errors="coerce").astype("int64")
        o["player_id"] = o["player_id"].astype(str)
        ocol = "total_fantasy_points_exp"
        if ocol in o.columns:
            os = (o.groupby(["player_id", "season"]).agg(
                xfp=(ocol, "sum"),
                opp_games=("week", "nunique"),
            ).reset_index())
            os["expected_fp_per_game"] = os["xfp"] / os["opp_games"].clip(lower=1)
            agg = agg.merge(os[["player_id", "season", "expected_fp_per_game"]],
                            on=["player_id", "season"], how="left")
            agg["fp_over_expected"] = agg["fppg"] / agg["expected_fp_per_game"].replace(0, np.nan)
        else:
            agg["expected_fp_per_game"] = np.nan
            agg["fp_over_expected"] = np.nan
    else:
        agg["expected_fp_per_game"] = np.nan
        agg["fp_over_expected"] = np.nan
    return agg


# The rate columns carried forward via EWMA (all §4.5.1 prior_production features).
RATE_FEATURES = [
    "fppg", "snap_share", "target_share", "targets_per_game",
    "carry_share", "carries_per_game", "touches_per_game",
    "rz_targets_per_game", "rz_carries_per_game", "expected_fp_per_game",
    "dropbacks_per_game", "pass_attempts_per_game", "rush_attempts_per_game",
    "routes_per_game", "yards_per_route_run", "yards_per_target", "catch_rate",
    "yards_per_reception", "rec_td_rate", "yards_per_carry", "rush_td_rate",
    "fp_over_expected", "completion_pct", "yards_per_attempt", "pass_td_rate",
    "int_rate", "sack_rate", "rush_yards_per_game", "air_yards_share",
]


def build_prior_production(
    stats_prior: pd.DataFrame,
    target_season: int,
    *,
    halflife: float = 1.0,
    pfr_to_gsis: Optional[Dict[str, str]] = None,
    snaps_prior: Optional[pd.DataFrame] = None,
    opp_prior: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """EWMA-decay per-season rates into one prior-production row per player.

    Inputs must already be filtered to seasons < ``target_season`` (caller uses
    ``leakage_guard.prior_seasons``). Most recent seasons get the highest weight.
    Re-filters defensively: target-season or future rows slipping past the
    caller would be an in-season leak, so they are dropped here too.
    """
    def _strictly_prior(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty or "season" not in df.columns:
            return df
        season = pd.to_numeric(df["season"], errors="coerce")
        return df[season < target_season]

    stats_prior = _strictly_prior(stats_prior)
    snaps_prior = _strictly_prior(snaps_prior)
    opp_prior = _strictly_prior(opp_prior)
    agg = _season_aggregates(stats_prior, pfr_to_gsis, snaps_prior, opp_prior)
    if agg.empty:
        return pd.DataFrame(columns=["player_id", *(f"prior_{c}" for c in RATE_FEATURES)])

    agg = agg.sort_values(["player_id", "season"])
    feats = [c for c in RATE_FEATURES if c in agg.columns]

    def _ewma_last(group: pd.DataFrame) -> pd.Series:
        # halflife in seasons; ewm over the ordered season series, take final value.
        ewm = group[feats].ewm(halflife=halflife).mean()
        out = ewm.iloc[-1]
        out["seasons_of_history"] = group["season"].nunique()
        out["last_season"] = int(group["season"].max())
        return out

    rolled = agg.groupby("player_id", group_keys=True).apply(_ewma_last, include_groups=False)
    if isinstance(rolled.index, pd.MultiIndex):
        rolled = rolled.reset_index(level=0)
    else:
        rolled = rolled.reset_index()
    rolled = rolled.rename(columns={c: f"prior_{c}" for c in feats})
    return rolled
