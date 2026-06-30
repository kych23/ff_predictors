"""Weekly model smoke tests: dataset building, monotonicity, OOF pipeline."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.projection.quantile_model import rearrange_quantiles
from src.projection.weekly_dataset import build_weekly_matrix
from src.projection.weekly_train import train_weekly_oof, _weekly_bucket


def _synthetic_weekly_features(n_seasons=7, weeks_per_season=17):
    """Generate synthetic weekly features + labels for training."""
    cfg = load_config()
    positions = cfg.roster.modeled_positions
    rng = np.random.RandomState(42)
    feat_rows, label_rows = [], []
    start_season = 2015
    pid = 0
    for s_offset in range(n_seasons):
        season = start_season + s_offset
        for pos in positions:
            for _ in range(10):
                pid += 1
                player_id = f"00-{pid:04d}"
                base_fp = rng.uniform(5, 25)
                for w in range(1, weeks_per_season + 1):
                    feats = {
                        "season_p50": base_fp + rng.normal(0, 2),
                        "rolling_fppg": base_fp + rng.normal(0, 3),
                        "rolling_target_share": rng.uniform(0, 0.3),
                        "rolling_snap_share": rng.uniform(0.3, 1.0),
                        "rolling_carries_pg": rng.uniform(0, 15),
                        "wk_implied_pts": rng.uniform(18, 30),
                        "wk_game_total": rng.uniform(38, 55),
                        "wk_spread": rng.uniform(-10, 10),
                        "opp_dvp_fpa": rng.uniform(10, 35),
                    }
                    feat_rows.append({
                        "player_id": player_id, "season": season, "week": w,
                        "position": pos, "features": feats,
                    })
                    noise = rng.normal(0, 5)
                    label_rows.append({
                        "player_id": player_id, "season": season, "week": w,
                        "fantasy_points": max(0, base_fp + noise),
                    })
    return pd.DataFrame(feat_rows), pd.DataFrame(label_rows)


def test_weekly_matrix_builds():
    feats, labels = _synthetic_weekly_features(n_seasons=2, weeks_per_season=3)
    ds = build_weekly_matrix(feats, labels)
    assert len(ds) > 0
    assert "position" in ds.feature_names
    assert "season_p50" in ds.feature_names
    assert ds.y.notna().all()


def test_weekly_matrix_position_categorical():
    feats, labels = _synthetic_weekly_features(n_seasons=2, weeks_per_season=3)
    ds = build_weekly_matrix(feats, labels)
    assert str(ds.X["position"].dtype) == "category"


def test_rearrange_monotonicity():
    raw = np.array([[5.0, 3.0, 8.0], [10.0, 2.0, 7.0], [1.0, 1.0, 1.0]])
    fixed = rearrange_quantiles(raw)
    for row in fixed:
        assert row[0] <= row[1] <= row[2]


def test_weekly_bucket_uses_position():
    row = pd.Series({"position": "QB", "player_id": "x"})
    assert _weekly_bucket(row) == "QB"


def test_weekly_oof_smoke():
    feats, labels = _synthetic_weekly_features(n_seasons=7, weeks_per_season=4)
    ds = build_weekly_matrix(feats, labels)
    result = train_weekly_oof(ds)
    assert not result.projections.empty
    preds = result.projections
    assert "p10" in preds.columns
    assert "p50" in preds.columns
    assert "p90" in preds.columns
    p10 = preds["p10"].to_numpy()
    p50 = preds["p50"].to_numpy()
    p90 = preds["p90"].to_numpy()
    assert (p10 <= p50 + 1e-9).all(), "P10 > P50 violation"
    assert (p50 <= p90 + 1e-9).all(), "P50 > P90 violation"
