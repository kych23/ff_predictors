"""Black-box extensions for the weekly dataset builder + bucket assignment."""

import pytest

# §9.4: this module is pinned to the v1 config shape. api/, web/ and the
# weekly start/sit surface are FROZEN for the build window, so v1 stays
# alive underneath them and these tests are deselected from the default
# run rather than migrated. Thaw is a §22.2 follow-up.
pytestmark = pytest.mark.v1_frozen
import pandas as pd

from src.projection.weekly_dataset import build_weekly_matrix
from src.projection.weekly_train import _weekly_bucket


def _feats_labels(n_players=6, weeks=4, season=2020):
    feat_rows, label_rows = [], []
    for i in range(n_players):
        pid = f"00-{i:04d}"
        pos = ["QB", "RB", "WR", "TE"][i % 4]
        for w in range(1, weeks + 1):
            feat_rows.append({
                "player_id": pid, "season": season, "week": w, "position": pos,
                "features": {"season_p50": 10.0 + i, "rolling_fppg": 9.0 + i,
                             "wk_implied_pts": 24.0},
            })
            label_rows.append({
                "player_id": pid, "season": season, "week": w,
                "fantasy_points": 10.0 + i,
            })
    return pd.DataFrame(feat_rows), pd.DataFrame(label_rows)


def test_matrix_lengths_align():
    feats, labels = _feats_labels()
    ds = build_weekly_matrix(feats, labels)
    assert len(ds.X) == len(ds.y)


def test_matrix_feature_names_include_packed_keys():
    feats, labels = _feats_labels()
    ds = build_weekly_matrix(feats, labels)
    for k in ("season_p50", "rolling_fppg", "wk_implied_pts"):
        assert k in ds.feature_names


def test_labels_not_in_features():
    """The target must never appear as a feature column."""
    feats, labels = _feats_labels()
    ds = build_weekly_matrix(feats, labels)
    assert "fantasy_points" not in ds.feature_names


def test_rows_without_labels_dropped_or_labelled():
    """Feature rows lacking a matching label can't produce NaN targets."""
    feats, labels = _feats_labels()
    labels = labels.iloc[:-3]  # drop labels for 3 rows
    ds = build_weekly_matrix(feats, labels)
    assert ds.y.notna().all()
    assert len(ds) <= len(feats)


def test_weekly_bucket_by_position():
    for pos in ("QB", "RB", "WR", "TE"):
        row = pd.Series({"position": pos, "player_id": "x"})
        assert _weekly_bucket(row) == pos
