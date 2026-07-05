"""Pooled training matrix from features + labels (DrafterSpec.md §4.6 / §4.6.1).

Flattens the ``season_features`` JSONB into ONE pooled matrix with ``position``
(as-of season Y) as a categorical feature — not four per-position matrices (§4.6.1)
— using one shared feature schema (NaN where a stat doesn't apply) so
position-changers keep correct semantics. Tags each row's position for per-position
output/eval.

NEVER reads ADP/market data — the wall (§4.0). This package must not import the
``ingest.adp`` module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sqlalchemy import select

from src.config import LeagueConfig, load_config
from src.db.models import SeasonFeature, SeasonLabel

# Columns that are identity/meta, never model inputs.
META_COLS = {"player_id", "season", "position", "as_of_date"}
CATEGORICAL = ["position"]
TARGET = "fppg"


@dataclass
class Dataset:
    X: pd.DataFrame            # feature matrix (position is a 'category' column)
    y: pd.Series               # target fppg
    meta: pd.DataFrame         # player_id, season, position, is_rookie
    feature_names: List[str]

    def __len__(self) -> int:
        return len(self.X)


def _flatten_features(feat_rows: pd.DataFrame) -> pd.DataFrame:
    """Explode the JSONB ``features`` dict column into wide columns."""
    if feat_rows.empty:
        return feat_rows
    exploded = pd.json_normalize(feat_rows["features"]).reset_index(drop=True)
    base = feat_rows[["player_id", "season", "position", "is_rookie"]].reset_index(drop=True)
    return pd.concat([base, exploded], axis=1)


def build_matrix(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    cfg: Optional[LeagueConfig] = None,
    require_label: bool = True,
) -> Dataset:
    """Join features to labels into a pooled (player, season) matrix.

    ``features_df`` has columns [player_id, season, position, is_rookie, features(dict)].
    ``labels_df`` has [player_id, season, fppg]. With ``require_label=False`` rows
    without a label are kept (serving/inference).
    """
    cfg = cfg or load_config()
    wide = _flatten_features(features_df)
    if wide.empty:
        return Dataset(pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), [])

    how = "inner" if require_label else "left"
    lbl = labels_df[["player_id", "season", TARGET]] if not labels_df.empty else \
        pd.DataFrame(columns=["player_id", "season", TARGET])
    df = wide.merge(lbl, on=["player_id", "season"], how=how)

    # restrict to modeled positions (K/DEF never modeled)
    df = df[df["position"].isin(cfg.roster.modeled_positions)].copy()

    meta = df[["player_id", "season", "position", "is_rookie"]].copy()
    y = df[TARGET] if TARGET in df.columns else pd.Series(index=df.index, dtype=float)

    drop = (META_COLS - {"position"}) | {TARGET, "is_rookie"}
    feature_names = [c for c in df.columns if c not in drop]
    X = df[feature_names].copy()

    # position as categorical for LightGBM; coerce the rest numeric, NaN preserved.
    for c in feature_names:
        if c in CATEGORICAL:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return Dataset(X=X, y=y, meta=meta, feature_names=feature_names)


def load_dataset(session, snapshot_id: Optional[str] = None, *,
                 cfg: Optional[LeagueConfig] = None, require_label: bool = True) -> Dataset:
    """Read season_features + season_labels from the DB and build the matrix."""
    fstmt = select(SeasonFeature)
    if snapshot_id:
        fstmt = fstmt.where(SeasonFeature.snapshot_id == snapshot_id)
    frows = session.execute(fstmt).scalars().all()
    feat_df = pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "position": r.position,
         "is_rookie": r.is_rookie, "features": r.features} for r in frows
    ])

    lstmt = select(SeasonLabel)
    if snapshot_id:
        lstmt = lstmt.where(SeasonLabel.snapshot_id == snapshot_id)
    lrows = session.execute(lstmt).scalars().all()
    lbl_df = pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "fppg": r.fppg} for r in lrows
    ])
    return build_matrix(feat_df, lbl_df, cfg=cfg, require_label=require_label)
