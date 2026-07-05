"""Projection training orchestration (DrafterSpec.md §4.6).

dataset -> expanding folds -> fit pooled quantile GBMs -> out-of-fold predictions
-> per-player-type interval calibration -> per-position ranks -> write ``projections``,
stamping ``model_version`` with the config hash (also traces outputs to the rules
that produced them, §4.0).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import LeagueConfig, load_config

from .calibrate import (IntervalCalibrator, assign_bucket, calibrate_oof_lofo,
                        coverage_by_bucket)
from .dataset import Dataset
from .folds import make_expanding_folds
from .quantile_model import QuantileGBM

logger = logging.getLogger(__name__)


@dataclass
class OOFResult:
    projections: pd.DataFrame   # player_id, season, position, p10, p50, p90, pos_rank, y
    calibrator: IntervalCalibrator


def _add_pos_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Per (season, position) rank by p50 (1 = best)."""
    df = df.copy()
    df["pos_rank"] = (
        df.groupby(["season", "position"])["p50"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return df


def train_oof(ds: Dataset, *, cfg: Optional[LeagueConfig] = None,
              params: Optional[dict] = None) -> OOFResult:
    """Produce calibrated out-of-fold projections via expanding-season CV."""
    cfg = cfg or load_config()
    seasons = sorted(ds.meta["season"].unique())
    folds = make_expanding_folds(seasons, cfg.training.min_train_seasons)
    if not folds:
        logger.warning("no evaluable folds (need >= %d warmup seasons; have %d)",
                       cfg.training.min_train_seasons, len(seasons))
        return OOFResult(pd.DataFrame(), IntervalCalibrator(cfg.training.min_bucket_n))

    season_arr = ds.meta["season"].to_numpy()
    pieces = []
    for fold in folds:
        tr = np.isin(season_arr, fold.train_seasons)
        te = season_arr == fold.test_season
        # temporal validation: hold out most recent training season for early stopping
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
        # carry the bucket-driving features (calibrate.assign_bucket) from X so
        # rookie / 2nd-year / team-changed intervals get their own width scales.
        Xte = ds.X[te].reset_index(drop=True)
        for col in ("seasons_of_history", "team_changed"):
            if col in Xte.columns:
                block[col] = pd.to_numeric(Xte[col], errors="coerce").to_numpy()
        pieces.append(block)
        top = model.feature_importance().head(8)
        logger.info("fold test=%d: train rows=%d test rows=%d | top gain: %s",
                    fold.test_season, int(tr.sum()), int(te.sum()),
                    ", ".join(f"{k}={v:.0f}" for k, v in top.items()))

    oof = pd.concat(pieces, ignore_index=True)

    # calibration on OOF residuals, per player-type bucket
    feat_for_bucket = oof.copy()
    if "seasons_of_history" not in feat_for_bucket.columns:
        feat_for_bucket["seasons_of_history"] = np.nan
    if "team_changed" not in feat_for_bucket.columns:
        feat_for_bucket["team_changed"] = 0
    oof["bucket"] = feat_for_bucket.apply(assign_bucket, axis=1)
    # serving calibrator: fit on ALL raw OOF residuals (applied to forward rows)
    calibrator = IntervalCalibrator(cfg.training.min_bucket_n).fit(oof)
    # reported OOF rows: leave-one-season-out so coverage isn't self-inflated
    oof = calibrate_oof_lofo(oof, cfg.training.min_bucket_n)
    for _, r in coverage_by_bucket(oof).iterrows():
        logger.info("OOF coverage [%s]: %.3f (n=%d, nominal 0.80)",
                    r["bucket"], r["coverage"], int(r["n"]))

    oof = _add_pos_rank(oof)
    return OOFResult(projections=oof, calibrator=calibrator)


def fit_full(ds: Dataset, *, params: Optional[dict] = None) -> QuantileGBM:
    """Fit a single pooled model on ALL labeled rows (for serving a future season)."""
    model = QuantileGBM(params=dict(params)) if params else QuantileGBM()
    model.fit(ds.X, ds.y)
    return model


def project_forward(full_model: QuantileGBM, ds_full: Dataset,
                    calibrator: IntervalCalibrator,
                    labeled_seasons: set, *,
                    snapshot_id: Optional[str] = None,
                    cfg: Optional[LeagueConfig] = None) -> pd.DataFrame:
    """Generate projections for seasons with no labels (e.g. 2026) using the full-data model.

    ``labeled_seasons`` is the set of seasons in the training dataset; only rows
    from seasons NOT in that set are projected (the true forward/future rows).
    """
    cfg = cfg or load_config()
    future_mask = ~ds_full.meta["season"].isin(labeled_seasons)
    if not future_mask.any():
        return pd.DataFrame()
    X_future = ds_full.X[future_mask].reset_index(drop=True)
    meta_future = ds_full.meta[future_mask].reset_index(drop=True)

    preds = full_model.predict(X_future)
    # Apply per-bucket calibration widening (same as OOF path).
    X_for_bucket = X_future.copy()
    X_for_bucket["is_rookie"] = meta_future["is_rookie"].values
    for col in ("seasons_of_history", "team_changed"):
        if col not in X_for_bucket.columns:
            X_for_bucket[col] = np.nan
    bucket_col = X_for_bucket.apply(assign_bucket, axis=1)
    preds = calibrator.transform(preds, bucket_col)

    out = meta_future.copy()
    out["p10"] = preds["p10"].values
    out["p50"] = preds["p50"].values
    out["p90"] = preds["p90"].values
    out = _add_pos_rank(out)
    out["snapshot_id"] = snapshot_id
    return out


def train_and_write(snapshot_id: Optional[str] = None, *,
                    config_path: Optional[str] = None) -> int:
    """DB path: load dataset, produce OOF + forward projections, write ``projections``."""
    from src.db.session import session_scope
    from src.db.upsert_data import upsert_projections
    from .dataset import load_dataset

    cfg = load_config(config_path)
    with session_scope() as session:
        # OOF projections on labeled seasons (validation path).
        ds = load_dataset(session, snapshot_id, cfg=cfg, require_label=True)
        if len(ds) == 0:
            logger.warning("empty dataset — nothing to train")
            return 0
        result = train_oof(ds, cfg=cfg)
        out = result.projections
        if out.empty:
            return 0
        out["model_version"] = cfg.model_version
        out["snapshot_id"] = snapshot_id
        n = upsert_projections(
            out[["player_id", "season", "model_version", "position",
                 "p10", "p50", "p90", "pos_rank", "snapshot_id"]], session)
        logger.info("projections written: %d OOF (model_version=%s)", n, cfg.model_version)

        # Forward projections for unlabeled seasons (serving path, e.g. 2026).
        ds_all = load_dataset(session, snapshot_id, cfg=cfg, require_label=False)
        full_model = fit_full(ds)
        labeled_seasons = set(ds.meta["season"].unique())
        fwd = project_forward(full_model, ds_all, result.calibrator, labeled_seasons,
                              snapshot_id=snapshot_id, cfg=cfg)
        if not fwd.empty:
            # Delete stale forward projections before inserting — upsert won't remove
            # players who were dropped from the universe (e.g. retired players).
            from sqlalchemy import delete as sa_delete
            from src.db.models import Projection as ProjectionModel
            forward_seasons = sorted(fwd["season"].unique().tolist())
            # Delete ALL forward projections for these seasons (any model_version) so
            # stale rows from prior runs don't persist when the model_version changes.
            session.execute(
                sa_delete(ProjectionModel).where(
                    ProjectionModel.season.in_(forward_seasons),
                )
            )
            fwd["model_version"] = cfg.model_version
            n_fwd = upsert_projections(
                fwd[["player_id", "season", "model_version", "position",
                     "p10", "p50", "p90", "pos_rank", "snapshot_id"]], session)
            logger.info("projections written: %d forward (seasons: %s)",
                        n_fwd, sorted(fwd["season"].unique().tolist()))
            n += n_fwd

        session.commit()
        logger.info("projections total: %d (model_version=%s)", n, cfg.model_version)
        return n
