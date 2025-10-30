from __future__ import annotations

from typing import Iterable, List, Tuple, Dict

import nflreadpy as nfl
import pandas as pd


def _to_pandas(df) -> pd.DataFrame:
    return df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)

def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_player_stats(years: List[int]) -> pd.DataFrame:
    """Loads data and removes rows with NaN values, coerces numeric columns, and filters to regular season games."""
    raw = nfl.load_player_stats(seasons=years)
    df = _to_pandas(raw)

    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().eq("REG")].copy()
    if "week" in df.columns:
        df = df[df["week"].fillna(0).astype(int) >= 1].copy()

    for k in ["player_id", "team", "opponent_team", "position"]:
        if k in df.columns:
            df[k] = df[k].astype(str).str.strip()

    required_keys = ["player_id", "season", "week"]
    df = df.dropna(subset=[c for c in required_keys if c in df.columns]).copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    metric_cols = [
        "fantasy_points_ppr",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "receiving_fumbles_lost",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles_lost",
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "fg_att",
        "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
        "pat_made", "pat_att",
    ]
    df = _coerce_numeric(df, metric_cols)


    id_cols = [c for c in ["player_id", "season", "week", "team", "opponent_team", "position"] if c in df.columns]
    keep_cols = id_cols + [c for c in metric_cols if c in df.columns]
    df = df[keep_cols].copy()
    df = df.dropna(subset=keep_cols)

    return df


def split_df_by_position(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create position-specific frames (QB, SKILL = RB/WR/TE, K).

    Columns chosen to align with PPR scoring (see notes mapping):
    - QB: passing (yds/td/int/att/comp) + rushing (yds/td/carries)
    - SKILL (RB/WR/TE): receiving (targets/receptions/yds/td) + rushing (carries/yds/td)
    - K: field goals made by distance, attempts, long, PAT

    """

    def _select_and_drop(d: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        present = [c for c in cols if c in d.columns]
        out = d[present].copy()
        return out

    key = ["player_id", "season", "week", "team", "opponent_team", "position"]

    qb_feats = key + [
        "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
        "carries", "rushing_yards", "rushing_tds",
    ]
    qb_df = _select_and_drop(df[df["position"].eq("QB")], qb_feats)

    skill_mask = df["position"].isin(["RB", "WR", "TE"])
    skill_feats = key + [
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "carries", "rushing_yards", "rushing_tds",
    ]
    skill_df = _select_and_drop(df[skill_mask], skill_feats)

    k_mask = df["position"].isin(["K"])
    k_feats = key + [
        "fg_att",
        "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
        "pat_made", "pat_att",
    ]
    k_df = _select_and_drop(df[k_mask], k_feats)

    return {"QB": qb_df, "SKILL": skill_df, "K": k_df}
