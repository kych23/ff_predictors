"""The projection bundle is built once per draft, not once per pick.

`build_projection_bundle` reaches the network for hazard covariates.
Rebuilding it inside every tier-0 recommendation cost **61.85 s** against a
25 s allocator budget, so the deadline was blown before the first candidate
was evaluated: the allocator got 0.18 s, reported
`stopped_because="deadline"`, and answered from the two initial draws per
candidate instead of fifty.

It was not just slow. The cold run returned a DIFFERENT leader than the warm
one, because there was no simulation behind it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_league
from src.engine.sim import bundle_build


@pytest.fixture(autouse=True)
def _clean():
    bundle_build.clear_bundle_cache()
    yield
    bundle_build.clear_bundle_cache()


@pytest.fixture(scope="module")
def league():
    return load_league()


ARTIFACTS = Path("data/artifacts")


@pytest.fixture(scope="module", autouse=True)
def _need_artifacts():
    """Only the NETWORK is stubbed here. The fitted sigma/kdst artifacts are
    read from disk exactly as in production, so the memo is exercised on the
    real build path rather than a hollow one."""
    if not list(ARTIFACTS.glob("weekly_sigma_*.json")):
        pytest.skip("no fitted artifacts; run scripts/fit_models.py")


def _board(n=6):
    return pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "player_name": [f"P {i}" for i in range(n)],
        "position": (["QB", "RB", "WR", "TE", "K", "DST"] * n)[:n],
        "team": ["DET"] * n, "bye_week": [9] * n,
        "value": np.linspace(12, 4, n), "adp": np.arange(1, n + 1) * 3.0,
        "value_p10": np.linspace(8, 2, n), "value_p90": np.linspace(18, 7, n),
    })


def _counting_loader():
    calls = {"n": 0}

    def loader(player_ids, positions, season):
        calls["n"] += 1
        return pd.DataFrame({
            "player_id": list(player_ids), "position": list(positions),
            "age": [25.0] * len(player_ids),
            "missed_prior": [0] * len(player_ids),
        })
    return loader, calls


def test_the_covariate_loader_runs_once_across_repeated_builds(league):
    """The loader is the network. Once per draft, not once per pick."""
    loader, calls = _counting_loader()
    board = _board()
    for _ in range(5):
        bundle_build.build_projection_bundle(
            board, league, 17, covariate_loader=loader, artifacts_dir=ARTIFACTS)
    assert calls["n"] <= 1, f"loader ran {calls['n']} times; it must be memoized"


def test_a_changed_board_rebuilds(league):
    """A resolved name can change a row without changing the row count, so the
    key covers contents rather than length alone."""
    loader, calls = _counting_loader()
    board = _board()
    bundle_build.build_projection_bundle(
        board, league, 17, covariate_loader=loader, artifacts_dir=ARTIFACTS)
    first = calls["n"]

    other = board.copy()
    other.loc[0, "player_id"] = "different"
    bundle_build.build_projection_bundle(
        other, league, 17, covariate_loader=loader, artifacts_dir=ARTIFACTS)
    assert calls["n"] > first, "a different board must not reuse the memo"


def test_a_changed_horizon_rebuilds(league):
    loader, calls = _counting_loader()
    board = _board()
    bundle_build.build_projection_bundle(
        board, league, 17, covariate_loader=loader, artifacts_dir=ARTIFACTS)
    first = calls["n"]
    bundle_build.build_projection_bundle(
        board, league, 15, covariate_loader=loader, artifacts_dir=ARTIFACTS)
    assert calls["n"] > first


def test_a_stubbed_loader_is_never_served_a_real_cached_bundle(league):
    """The loader is part of the key, so a test that stubs the network cannot
    silently receive a bundle built from the real one."""
    loader_a, calls_a = _counting_loader()
    loader_b, calls_b = _counting_loader()
    board = _board()
    bundle_build.build_projection_bundle(
        board, league, 17, covariate_loader=loader_a, artifacts_dir=ARTIFACTS)
    bundle_build.build_projection_bundle(
        board, league, 17, covariate_loader=loader_b, artifacts_dir=ARTIFACTS)
    assert calls_b["n"] >= 1


def test_the_cached_bundle_is_equivalent_not_merely_fast(league):
    loader, _ = _counting_loader()
    board = _board()
    a = bundle_build.build_projection_bundle(
        board, league, 17, covariate_loader=loader, artifacts_dir=ARTIFACTS)
    b = bundle_build.build_projection_bundle(
        board, league, 17, covariate_loader=loader, artifacts_dir=ARTIFACTS)
    assert np.array_equal(a.rate_p50, b.rate_p50)
    assert np.array_equal(a.games_hazard, b.games_hazard)
    assert np.array_equal(a.bye_weeks, b.bye_weeks)


def test_the_session_warms_the_bundle_at_startup():
    """The memo makes every pick after the first warm; the warm-up makes the
    FIRST one warm too, by paying the network cost during setup."""
    import inspect

    from src.app.web.service import CockpitService

    assert "_warm_projection_bundle" in inspect.getsource(
        CockpitService.start_session)
    warm = inspect.getsource(CockpitService._warm_projection_bundle)
    assert "build_projection_bundle" in warm
    assert "except Exception" in warm, (
        "a draft must still start when nflverse is unreachable")
