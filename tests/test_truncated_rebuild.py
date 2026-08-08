"""§21.2 truncated rebuild — against the REAL feature pipeline.

The property: building season Y's features from a source that *contains* Y's
in-season data must produce byte-identical output to building them from a
source where that data is physically absent. If the two differ by one column
in one row, something read the future.

Why this test and not an audit of the guards. A guard can be correct and still
be bypassed — a feature that reaches a frame before it passes through
`asof()`, a join that pulls a column across the cutoff, an aggregation whose
`groupby` silently includes the target season. None of those are visible in
the guard's own tests, and all of them are visible here, because the only
input that changes is whether the future exists at all.

`tests/test_platform_phase4.py::test_truncated_rebuild_is_identical` asserts
the same property against a synthetic three-column frame. That version tests
`guards.prior_seasons`. This one tests the pipeline: `assemble_season` with
prior production, role change and team context, over real nflverse frames.

**Verified non-vacuous by mutation**, which turned out to be worth doing: the
pipeline has THREE independent layers between raw frames and features, and the
first two mutations changed nothing observable here.

1. `guards.prior_seasons` — the filter. Mutating `<` to `<=` admits 18,128
   target-season rows, and `assert_no_future` raises before anything is built.
2. `guards.assert_no_future` — the assertion. With BOTH disabled, this file
   still passed: the leaked rows reach `build_prior_production` and are
   discarded by its own defensive re-filter (`prior_production.py:166-169`),
   so they never touch a feature value.
3. `build_prior_production`'s re-filter. Only with all three disabled does this
   file fail — and then it names all 22 contaminated columns
   (`prior_fppg`, `prior_target_share`, …) rather than reporting one opaque
   cell.

So this test is not redundant with the assertion and not stronger than it
either. The assertion catches a leak at the boundary; this catches one that
gets past the boundary and changes a number. Both are load-bearing.

Marked `slow` — it pulls from nflverse. `-m 'not slow'` skips it, and §22.1's
day-20 rehearsal runs it explicitly.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.core.config import load_league
from src.models.features.assemble import assemble_season

TARGET = 2024
HISTORY_START = 2016

#: Which key each asset is truncated on, and whether the target season is
#: legitimately available before Week 1.
#:
#: `depth`, `rosters` and `schedules` are PRESEASON artifacts: their target-
#: season rows are known before kickoff and are the only target-season rows any
#: feature may read (§11.3). Truncating them would test that the pipeline
#: ignores data it is entitled to use, which is a different and wrong claim.
BACKWARD_ASSETS = ("stats", "snaps", "opp")
PRESEASON_ASSETS = ("depth", "rosters", "schedules")


def _load_frames(seasons: list[int]) -> tuple[dict, dict]:
    """The same pull `scripts/train_projection_v2.py` performs."""
    from src.platform.sources import nflverse

    def pull(asset, **kw):
        return nflverse.fetch(asset, blob_root=None, **kw).frame

    players = pull("players").rename(columns={
        "gsis_id": "player_id", "display_name": "name",
        "college_name": "college", "rookie_season": "rookie_year"})
    id_map = pull("ff_playerids")
    pfr_to_gsis = {}
    if {"pfr_id", "gsis_id"} <= set(id_map.columns):
        pairs = id_map[["pfr_id", "gsis_id"]].dropna()
        pfr_to_gsis = dict(zip(pairs["pfr_id"].astype(str),
                               pairs["gsis_id"].astype(str), strict=False))

    frames = {
        "players": players,
        "id_map": id_map,
        "stats": pull("player_stats", seasons=seasons),
        "snaps": pull("snap_counts", seasons=[s for s in seasons if s >= 2012]),
        "opp": pull("ff_opportunity", seasons=[s for s in seasons if s >= 2006]),
        "depth": pull("depth_charts", seasons=seasons),
        "rosters": pull("rosters", seasons=seasons),
        "schedules": pull("schedules"),
    }
    return frames, pfr_to_gsis


def _truncate(frames: dict, cutoff_season: int) -> dict:
    """Physically remove every backward-looking row at or after the cutoff."""
    out = dict(frames)
    for asset in BACKWARD_ASSETS:
        df = out[asset]
        if "season" in df.columns:
            out[asset] = df[pd.to_numeric(df["season"], errors="coerce")
                            < cutoff_season].copy()
    return out


@pytest.fixture(scope="module")
def built():
    """(from_full, from_truncated) feature frames for the target season."""
    seasons = list(range(HISTORY_START, TARGET + 1))
    full, pfr_to_gsis = _load_frames(seasons)
    truncated = _truncate(full, TARGET)

    cfg = load_league()
    as_of = dt.date(TARGET, 9, 1)
    kwargs = dict(cfg=cfg, as_of_date=as_of, pfr_to_gsis=pfr_to_gsis)
    return (assemble_season(TARGET, full, **kwargs),
            assemble_season(TARGET, truncated, **kwargs))


@pytest.mark.slow
def test_the_fixture_actually_contains_the_future(built):
    """Guard on the guard. If the 'full' frames happen not to include the
    target season, this whole file passes vacuously — it would be comparing
    truncated data against itself."""
    seasons = list(range(HISTORY_START, TARGET + 1))
    full, _ = _load_frames(seasons)
    in_season = full["stats"][pd.to_numeric(full["stats"]["season"],
                                            errors="coerce") == TARGET]
    assert len(in_season) > 1000, (
        f"only {len(in_season)} target-season stat rows — the test has nothing "
        f"to catch a leak with"
    )


@pytest.mark.slow
def test_truncated_rebuild_is_identical_on_the_real_pipeline(built):
    """THE test. Same shape, same rows, same values."""
    from_full, from_trunc = built
    assert len(from_full) == len(from_trunc), (
        f"row count changed when the future was removed: "
        f"{len(from_full)} -> {len(from_trunc)}"
    )
    a = from_full.sort_values(["player_id"]).reset_index(drop=True)
    b = from_trunc.sort_values(["player_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_like=True)


@pytest.mark.slow
def test_every_packed_feature_column_agrees(built):
    """`assemble_season` packs features into an opaque dict column, so
    `assert_frame_equal` on the outer frame compares dicts by equality and
    reports the first difference as one unhelpful cell. This unpacks them and
    names the offending column — which is the difference between 'a leak
    exists' and 'a leak exists in prior_production.targets_per_game'."""
    from_full, from_trunc = built
    a = pd.json_normalize(
        from_full.sort_values("player_id")["features"]).reset_index(drop=True)
    b = pd.json_normalize(
        from_trunc.sort_values("player_id")["features"]).reset_index(drop=True)

    assert set(a.columns) == set(b.columns), (
        f"feature set changed: only-in-full={sorted(set(a) - set(b))}, "
        f"only-in-truncated={sorted(set(b) - set(a))}"
    )

    leaked = []
    for column in sorted(a.columns):
        if not a[column].equals(b[column]):
            differing = (a[column] != b[column]) & ~(a[column].isna()
                                                     & b[column].isna())
            leaked.append(f"{column} ({int(differing.sum())} rows)")
    assert not leaked, "columns that changed when the future was removed: " \
                       + ", ".join(leaked)


@pytest.mark.slow
def test_poisoning_the_target_season_changes_nothing(built):
    """The sentinel form, at pipeline scale. Replace every target-season stat
    with an absurd value: if any of it reaches a feature, the output moves."""
    seasons = list(range(HISTORY_START, TARGET + 1))
    frames, pfr_to_gsis = _load_frames(seasons)
    poisoned = dict(frames)
    stats = frames["stats"].copy()
    mask = pd.to_numeric(stats["season"], errors="coerce") == TARGET
    numeric = [c for c in stats.columns
               if pd.api.types.is_numeric_dtype(stats[c])
               and c not in {"season", "week"}]
    stats.loc[mask, numeric] = 999_999.0
    poisoned["stats"] = stats

    cfg = load_league()
    kwargs = dict(cfg=cfg, as_of_date=dt.date(TARGET, 9, 1),
                  pfr_to_gsis=pfr_to_gsis)
    from_poisoned = assemble_season(TARGET, poisoned, **kwargs)
    from_full, _ = built

    a = pd.json_normalize(
        from_full.sort_values("player_id")["features"]).reset_index(drop=True)
    b = pd.json_normalize(
        from_poisoned.sort_values("player_id")["features"]).reset_index(drop=True)
    contaminated = [c for c in sorted(set(a) & set(b)) if not a[c].equals(b[c])]
    assert not contaminated, (
        f"sentinel reached these features: {contaminated}"
    )
