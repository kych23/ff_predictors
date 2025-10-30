from __future__ import annotations

from typing import Any, Dict

import os
import joblib


_CACHE: Dict[str, Any] = {}


def get_cached_model() -> tuple[Any, list[str], str]:
    """Load model bundle from path specified by MODEL_PATH env, with caching.

    Returns (model, feature_columns, version) where version is the resolved path.
    """
    model_path = os.getenv("MODEL_PATH", "models/latest.pkl")
    mtime = None
    try:
        mtime = os.path.getmtime(model_path)
    except OSError:
        pass

    cached = _CACHE.get("bundle")
    if cached and cached.get("path") == model_path and cached.get("mtime") == mtime:
        return cached["model"], cached["feature_columns"], cached["version"]

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", [])
    version = f"{model_path}:{mtime}" if mtime is not None else model_path
    _CACHE["bundle"] = {"model": model, "feature_columns": feature_columns, "path": model_path, "mtime": mtime, "version": version}
    return model, feature_columns, version


