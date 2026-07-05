"""Black-box extensions for rearrangement + quantile model prediction shape."""
import numpy as np
import pandas as pd

from src.projection.quantile_model import QuantileGBM, rearrange_quantiles


def test_rearrange_identity_on_sorted_rows():
    raw = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-5.0, 0.0, 5.0]])
    out = rearrange_quantiles(raw)
    assert np.array_equal(out, raw)


def test_rearrange_negative_values():
    raw = np.array([[-1.0, -5.0, -3.0]])
    out = rearrange_quantiles(raw)
    assert list(out[0]) == [-5.0, -3.0, -1.0]


def test_rearrange_single_row():
    out = rearrange_quantiles(np.array([[9.0, 1.0, 5.0]]))
    assert list(out[0]) == [1.0, 5.0, 9.0]


def test_rearrange_preserves_shape():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(50, 3))
    out = rearrange_quantiles(raw)
    assert out.shape == raw.shape


def test_rearrange_fully_reversed():
    raw = np.array([[30.0, 20.0, 10.0]])
    out = rearrange_quantiles(raw)
    assert list(out[0]) == [10.0, 20.0, 30.0]


def test_predict_output_contract():
    """Predictions: one row per input row, p10/p50/p90 columns, monotone,
    all finite."""
    rng = np.random.default_rng(4)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(4 * X["a"] + rng.normal(scale=0.5, size=n))
    model = QuantileGBM().fit(X, y)
    X_new = pd.DataFrame({"a": rng.normal(size=25), "b": rng.normal(size=25)})
    preds = model.predict(X_new)
    assert len(preds) == 25
    for col in ("p10", "p50", "p90"):
        assert col in preds.columns
        assert np.isfinite(preds[col]).all()
    assert (preds["p10"] <= preds["p50"]).all()
    assert (preds["p50"] <= preds["p90"]).all()
    # intervals must have real width on noisy data (not collapsed)
    assert (preds["p90"] - preds["p10"]).mean() > 0
