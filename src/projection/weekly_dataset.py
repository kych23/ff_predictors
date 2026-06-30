"""Weekly training matrix from WeeklyFeature + WeeklyLabel.

Flattens the WeeklyFeature JSONB into one pooled (player, season, week) matrix
with position as a categorical feature. Target is per-week fantasy_points.
Season P50 is already inside the JSONB as a feature (from Phase 1 assembly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sqlalchemy import select

from src.config import LeagueConfig, load_config
from src.db.models import WeeklyFeature, WeeklyLabel

META_COLS = {"player_id", "season", "week", "position"}
CATEGORICAL = ["position"]
TARGET = "fantasy_points"


@dataclass
class WeeklyDataset:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    feature_names: List[str]

    def __len__(self) -> int:
        return len(self.X)


def _flatten_weekly_features(feat_rows: pd.DataFrame) -> pd.DataFrame:
    if feat_rows.empty:
        return feat_rows
    exploded = pd.json_normalize(feat_rows["features"]).reset_index(drop=True)
    base = feat_rows[["player_id", "season", "week", "position"]].reset_index(drop=True)
    return pd.concat([base, exploded], axis=1)


def build_weekly_matrix(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    cfg: Optional[LeagueConfig] = None,
    require_label: bool = True,
) -> WeeklyDataset:
    """Join weekly features to weekly labels into a pooled matrix."""
    cfg = cfg or load_config()
    wide = _flatten_weekly_features(features_df)
    if wide.empty:
        return WeeklyDataset(pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), [])

    how = "inner" if require_label else "left"
    lbl = labels_df[["player_id", "season", "week", TARGET]] if not labels_df.empty else \
        pd.DataFrame(columns=["player_id", "season", "week", TARGET])
    df = wide.merge(lbl, on=["player_id", "season", "week"], how=how)

    df = df[df["position"].isin(cfg.roster.modeled_positions)].copy()

    meta = df[["player_id", "season", "week", "position"]].copy()
    y = df[TARGET] if TARGET in df.columns else pd.Series(index=df.index, dtype=float)

    drop = (META_COLS - {"position"}) | {TARGET}
    feature_names = [c for c in df.columns if c not in drop]
    X = df[feature_names].copy()

    for c in feature_names:
        if c in CATEGORICAL:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return WeeklyDataset(X=X, y=y, meta=meta, feature_names=feature_names)


def load_weekly_dataset(session, snapshot_id: Optional[str] = None, *,
                        cfg: Optional[LeagueConfig] = None,
                        require_label: bool = True) -> WeeklyDataset:
    """Read WeeklyFeature + WeeklyLabel from DB and build the matrix."""
    fstmt = select(WeeklyFeature)
    if snapshot_id:
        fstmt = fstmt.where(WeeklyFeature.snapshot_id == snapshot_id)
    frows = session.execute(fstmt).scalars().all()
    feat_df = pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "week": r.week,
         "position": r.position, "features": r.features} for r in frows
    ])

    lstmt = select(WeeklyLabel)
    if snapshot_id:
        lstmt = lstmt.where(WeeklyLabel.snapshot_id == snapshot_id)
    lrows = session.execute(lstmt).scalars().all()
    lbl_df = pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "week": r.week,
         "fantasy_points": r.fantasy_points} for r in lrows
    ])
    return build_weekly_matrix(feat_df, lbl_df, cfg=cfg, require_label=require_label)
