#!/usr/bin/env python
from __future__ import annotations

import argparse
from typing import List

from src.db.init_db import init_db
from src.db.session import SessionLocal
from src.ml.models.train import (
    load_flat_player_weeks,
    split_by_position,
    per_position_feature_lists,
    position_expanding_folds,
    train_xgb_per_position,
)


def main(years: List[int] | None, priors_req: int, k: float) -> None:
    init_db()
    session = SessionLocal()
    try:
        flat = load_flat_player_weeks(session, years=years)
    finally:
        session.close()

    # Split and prepare folds
    splits = split_by_position(flat)
    feats = per_position_feature_lists(flat)
    folds = position_expanding_folds(splits, min_years=3)

    # Train XGB per position and evaluate OOF predictions
    xgb_out = train_xgb_per_position(splits, feats, folds)
    for pos, out in xgb_out.items():
        oof = out["oof"]
        print(f"{pos}: XGB OOF MAE (non-NaN) = ", end="")
        oof_mask = ~splits[pos]["fantasy_points"].isna() & ~pd.isna(oof)
        if oof_mask.any():
            y = splits[pos].loc[oof_mask, "fantasy_points"].astype(float).to_numpy()
            yhat = oof[oof_mask]
            import numpy as np  # local import to avoid global dependency when imported as module
            mae = np.abs(y - yhat).mean()
            acc = (np.abs(y - yhat) <= k).mean() * 100.0
            print(f"{round(float(mae), 4)} | Within-{k} Acc = {round(float(acc), 2)}%")
        else:
            print("n/a")

if __name__ == "__main__":
    import pandas as pd
    import numpy as np  # noqa: F401
    parser = argparse.ArgumentParser(description="Test end-to-end training pipeline")
    parser.add_argument("--start", type=int, default=2012, help="Start season (inclusive)")
    parser.add_argument("--end", type=int, default=2025, help="End season (exclusive)")
    parser.add_argument("--priors-req", type=int, default=1, help="Min prior games required")
    parser.add_argument("--k", type=float, default=3.0, help="Within-k accuracy threshold in points")
    args = parser.parse_args()
    years = list(range(args.start, args.end))
    main(years, args.priors_req, args.k)

