"""Weekly projection training — expanding-season CV on (player, week) observations.

Two-layer architecture: the season P50 prior is already a feature in the weekly
feature JSONB (from Phase 1 assembly). The weekly GBM learns how much to adjust
from the season baseline given in-season context (rolling stats, Vegas, DvP).

Same monotonicity enforcement (rearrangement) as the season model. Calibration
uses position-based buckets instead of the season model's player-type buckets
(is_rookie / team_changed don't vary week-to-week within a season).

Known train/serve skew: during OOF training the ``season_p50`` feature is the
OOF season projection (model trained on strictly-earlier seasons); at serving
time it comes from the full-data season model, which is trained on more data
and is typically less noisy. The weekly model treats season_p50 as an opaque
prior, so the skew is tolerated rather than corrected — but it means live
performance may differ slightly from OOF benchmarks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import LeagueConfig, load_config

from .calibrate import IntervalCalibrator, calibrate_oof_lofo, coverage_by_bucket
from .folds import make_expanding_folds
from .quantile_model import QuantileGBM
from .weekly_dataset import WeeklyDataset

logger = logging.getLogger(__name__)


def _weekly_bucket(row: pd.Series) -> str:
    """Position-based calibration bucket for weekly projections."""
    return str(row.get("position", "UNK"))


@dataclass
class WeeklyOOFResult:
    projections: pd.DataFrame
    calibrator: IntervalCalibrator


def train_weekly_oof(ds: WeeklyDataset, *, cfg: Optional[LeagueConfig] = None,
                     params: Optional[dict] = None) -> WeeklyOOFResult:
    """Expanding-season OOF for weekly projections."""
    cfg = cfg or load_config()
    seasons = sorted(ds.meta["season"].unique())
    folds = make_expanding_folds(seasons, cfg.training.min_train_seasons)
    if not folds:
        logger.warning("no evaluable weekly folds (need >= %d warmup; have %d)",
                       cfg.training.min_train_seasons, len(seasons))
        return WeeklyOOFResult(pd.DataFrame(), IntervalCalibrator(cfg.training.min_bucket_n))

    season_arr = ds.meta["season"].to_numpy()
    pieces = []
    for fold in folds:
        tr = np.isin(season_arr, fold.train_seasons)
        te = season_arr == fold.test_season
        val_season = max(fold.train_seasons)
        val_mask = season_arr == val_season
        train_mask = tr & ~val_mask
        model = QuantileGBM(params=dict(params)) if params else QuantileGBM()
        model.fit(ds.X[train_mask], ds.y[train_mask],
                  X_val=ds.X[val_mask], y_val=ds.y[val_mask])
        preds = model.predict(ds.X[te])
        block = ds.meta[te].reset_index(drop=True).copy()
        preds = preds.reset_index(drop=True)
        block[["p10", "p50", "p90"]] = preds[["p10", "p50", "p90"]]
        block["y"] = ds.y[te].reset_index(drop=True)
        pieces.append(block)
        top = model.feature_importance().head(8)
        logger.info("weekly fold test=%d: train=%d test=%d | top gain: %s",
                    fold.test_season, int(tr.sum()), int(te.sum()),
                    ", ".join(f"{k}={v:.0f}" for k, v in top.items()))

    oof = pd.concat(pieces, ignore_index=True)

    oof["bucket"] = oof.apply(_weekly_bucket, axis=1)
    # serving calibrator: fit on ALL raw OOF residuals (applied to forward weeks)
    calibrator = IntervalCalibrator(cfg.training.min_bucket_n).fit(oof)
    # reported OOF rows: leave-one-season-out so coverage isn't self-inflated
    oof = calibrate_oof_lofo(oof, cfg.training.min_bucket_n)
    for _, r in coverage_by_bucket(oof).iterrows():
        logger.info("weekly OOF coverage [%s]: %.3f (n=%d, nominal 0.80)",
                    r["bucket"], r["coverage"], int(r["n"]))

    return WeeklyOOFResult(projections=oof, calibrator=calibrator)


def fit_weekly_full(ds: WeeklyDataset, *, params: Optional[dict] = None) -> QuantileGBM:
    """Fit on ALL labeled weekly rows (for projecting a future week)."""
    model = QuantileGBM(params=dict(params)) if params else QuantileGBM()
    model.fit(ds.X, ds.y)
    return model


def train_weekly_and_write(snapshot_id: Optional[str] = None, *,
                           config_path: Optional[str] = None) -> int:
    """DB path: load weekly dataset, OOF + write projections."""
    from src.db.session import session_scope
    from src.db.upsert_data import upsert_weekly_projections
    from .weekly_dataset import load_weekly_dataset

    cfg = load_config(config_path)
    with session_scope() as session:
        ds = load_weekly_dataset(session, snapshot_id, cfg=cfg, require_label=True)
        if len(ds) == 0:
            logger.warning("empty weekly dataset — nothing to train")
            return 0
        result = train_weekly_oof(ds, cfg=cfg)
        out = result.projections
        if out.empty:
            return 0
        mv = f"weekly_v1.{cfg.version_hash}"
        out["model_version"] = mv
        out["snapshot_id"] = snapshot_id
        n = upsert_weekly_projections(
            out[["player_id", "season", "week", "model_version", "position",
                 "p10", "p50", "p90", "snapshot_id"]], session)
        session.commit()
        logger.info("weekly projections written: %d OOF (model_version=%s)", n, mv)
        return n
