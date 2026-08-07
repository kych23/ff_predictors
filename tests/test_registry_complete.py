"""§21.2 — every column feature code reads must be registered.

The registry's rule is that the default is refusal: an unregistered column
cannot be read, so a new source cannot quietly enter the feature set without
someone deciding its cutoff. That rule is only real if something checks it
against what the feature code *actually reads*.

**Runtime instrumentation, not an AST scan.** §21.2 is explicit about the
mechanism and the reason: columns are routinely built by list comprehension
(`[c for c in df.columns if c.startswith("rushing_")]`), so a scan of string
literals sees none of them. Here every accessor returns a frame whose column
access is recorded, and the recorded set is asserted to be registered.

This does not need the network — it runs against synthetic frames whose columns
are the real ones.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.errors import UnregisteredColumnError
from src.platform.asof.handle import asof
from src.platform.asof.registry import (
    REGISTRY,
    availability_of,
    is_registered,
    register,
)


class RecordingFrame(pd.DataFrame):
    """A DataFrame that records which columns are read through __getitem__.

    Subclassing rather than wrapping because the feature code indexes, slices,
    groups and merges these frames, and a proxy object would have to
    re-implement all of it. `_constructor` keeps derived frames recording too.
    """

    _metadata = ["_accessed"]

    @property
    def _constructor(self):
        return RecordingFrame

    def __getitem__(self, key):
        seen = getattr(self, "_accessed", None)
        if seen is not None:
            if isinstance(key, str):
                seen.add(key)
            elif isinstance(key, (list, tuple, pd.Index)):
                seen.update(k for k in key if isinstance(k, str))
        return super().__getitem__(key)


def _recording(frame: pd.DataFrame, seen: set) -> RecordingFrame:
    out = RecordingFrame(frame)
    out._accessed = seen
    return out


# --------------------------------------------------------- the mechanism
def test_the_recorder_actually_records():
    """Guard on the guard: an instrument that records nothing makes every
    completeness assertion below pass vacuously."""
    seen: set = set()
    frame = _recording(pd.DataFrame({"a": [1], "b": [2], "c": [3]}), seen)
    _ = frame["a"]
    _ = frame[["b", "c"]]
    assert seen == {"a", "b", "c"}


def test_derived_frames_keep_recording():
    seen: set = set()
    frame = _recording(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), seen)
    subset = frame[frame["a"] > 1]
    _ = subset["b"]
    assert "a" in seen and "b" in seen


# ------------------------------------------------------------ the claim
REAL_COLUMNS = {
    "player_stats": ["player_id", "season", "week", "season_type", "position",
                     "team", "passing_yards", "rushing_yards",
                     "receiving_yards", "receptions", "targets",
                     "target_share", "carries"],
    "rosters": ["player_id", "position", "team", "status", "season", "week"],
    "depth_charts": ["player_id", "position", "team", "depth_team", "season",
                     "week"],
    "schedules": ["season", "week", "home_team", "away_team", "gameday",
                  "spread_line", "total_line", "game_type"],
    "players": ["gsis_id", "display_name", "position", "birth_date"],
}


def _frames(seen: set) -> dict:
    built = {}
    for asset, columns in REAL_COLUMNS.items():
        data = {}
        for c in columns:
            if c == "season":
                data[c] = [2022, 2023]
            elif c == "week":
                data[c] = [1, 1]
            elif c == "season_type":
                data[c] = ["REG", "REG"]
            elif c in ("gameday",):
                data[c] = ["2022-09-08", "2023-09-07"]
            elif c in ("player_id", "gsis_id"):
                data[c] = ["00-001", "00-002"]
            elif c in ("position", "team", "status", "home_team", "away_team",
                       "display_name", "game_type", "depth_team"):
                data[c] = ["X", "Y"]
            elif c == "birth_date":
                data[c] = ["1998-01-01", "1999-01-01"]
            else:
                data[c] = [1.0, 2.0]
        built[asset] = _recording(pd.DataFrame(data), seen)
    return built


def test_every_column_of_every_shipped_asset_is_registered():
    """The columns above are the real nflverse names the pipeline reads. Any
    one of them missing from the registry means `availability_of` would raise
    at runtime — during a pipeline run, not here."""
    unregistered = [
        f"{asset}.{column}"
        for asset, columns in REAL_COLUMNS.items()
        for column in columns
        if not is_registered(asset, column)
    ]
    assert not unregistered, (
        f"columns read by the pipeline with no availability declaration: "
        f"{unregistered}"
    )


def test_accessors_read_only_registered_columns():
    """Drive the handle and assert every recorded access is registered."""
    seen: set = set()
    frames = _frames(seen)
    handle = asof(2024, snapshot_id="sn2_test",
                  loaders={k: (lambda f=v: f) for k, v in frames.items()})

    for accessor in ("player_stats", "rosters", "depth_charts", "schedules",
                     "players"):
        getattr(handle, accessor)()

    assert seen, "nothing was recorded — the instrument is not wired in"
    # every recorded column must resolve in SOME asset's registration
    unresolved = [c for c in seen
                  if not any(is_registered(a, c) for a in REAL_COLUMNS)]
    assert not unresolved, f"unregistered columns were read: {unresolved}"


def test_an_unregistered_column_is_refused_at_the_boundary():
    """The rule that makes the registry worth having. A new source column must
    fail loudly rather than flow into features with an undecided cutoff."""
    frame = pd.DataFrame({
        "player_id": ["00-001"], "season": [2022], "week": [1],
        "season_type": ["REG"], "position": ["RB"], "team": ["X"],
        "secret_in_season_metric": [42.0],
    })
    handle = asof(2024, snapshot_id="sn2_test",
                  loaders={"player_stats": lambda: frame})
    with pytest.raises(UnregisteredColumnError,
                       match="secret_in_season_metric"):
        handle.player_stats()


def test_the_refusal_message_says_what_to_do():
    """An error that names the fix is the difference between a guard and an
    obstacle."""
    with pytest.raises(UnregisteredColumnError) as exc:
        availability_of("player_stats", "not_a_real_column")
    message = str(exc.value)
    assert "registry" in message
    assert "refusal, not permission" in message


# ------------------------------------------------------ registry hygiene
def test_every_registration_carries_a_known_availability():
    allowed = {"season_end", "week_end", "preseason", "static"}
    bad = {f"{s}.{c}": spec.availability
           for (s, c), spec in REGISTRY.items()
           if spec.availability not in allowed}
    assert not bad, f"unknown availability literals: {bad}"


def test_only_preseason_columns_carry_an_offset():
    """`offset_days` shifts a cutoff earlier. On a `week_end` or `season_end`
    column that is meaningless, and a meaningless field that looks meaningful
    is how a wrong cutoff survives review."""
    misplaced = {f"{s}.{c}" for (s, c), spec in REGISTRY.items()
                 if spec.offset_days and spec.availability != "preseason"}
    assert not misplaced, misplaced


def test_scores_are_never_preseason():
    """§11.3: later-week lines live in the same frame as Week-1 lines, and
    final scores are outcomes. Registering a score as preseason would admit
    the entire season's results at draft time."""
    for column in ("home_score", "away_score", "result", "total"):
        assert availability_of("schedules", column).availability == "week_end"


def test_registration_is_per_source_not_global():
    """`position` is registered for several assets. If the registry keyed on
    column alone, registering it once would silently permit it everywhere,
    including on a source whose cutoff differs."""
    assert is_registered("player_stats", "position")
    assert not is_registered("combine", "position")


def test_register_is_idempotent_and_last_write_wins():
    key = ("test_asset", "test_column")
    try:
        register("test_asset", ("test_column",), "static")
        assert availability_of(*key).availability == "static"
        register("test_asset", ("test_column",), "week_end")
        assert availability_of(*key).availability == "week_end"
    finally:
        REGISTRY.pop(key, None)
