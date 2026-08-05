"""Platform-layer tests: store, snapshots, as-of boundary (§11, §21.2)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.core.errors import (
    CutoffMismatchError,
    DataError,
    EmptySourceError,
    LeakageError,
    UnregisteredColumnError,
)
from src.platform.asof import assert_same_cutoff, availability_of, is_registered
from src.platform.asof.handle import asof
from src.platform.sources import nflverse
from src.platform.store import blobs, canonicalize
from src.platform.store.manifest import (
    ManifestEntry, build_snapshot, diff, read_manifest, utc_now, write_manifest,
)


@pytest.fixture()
def store(tmp_path):
    return tmp_path / "blobs"


@pytest.fixture()
def snaps(tmp_path):
    return tmp_path / "snapshots"


def _entry(asset="player_stats", digest="a" * 64, source="nflverse", params="{}"):
    return ManifestEntry(source, asset, params, digest, utc_now(), "identity")


# ------------------------------------------------------------------ blobs
def test_put_is_content_addressed_and_idempotent(store):
    d1 = blobs.put(b"hello", root=store)
    d2 = blobs.put(b"hello", root=store)
    assert d1 == d2 == blobs.digest_of(b"hello")
    assert len(d1) == 64, "blob digests are never truncated"
    assert blobs.get(d1, root=store) == b"hello"


def test_missing_blob_raises(store):
    with pytest.raises(FileNotFoundError, match="not in store"):
        blobs.get("f" * 64, root=store)


def test_params_key_is_order_independent():
    assert blobs.params_key({"a": 1, "b": 2}) == blobs.params_key({"b": 2, "a": 1})


# ---------------------------------------------------------- canonicalizers
def test_ffc_canonicalizer_absorbs_the_moving_window():
    """FFC echoes a request window that moves daily even when ADP is
    unchanged; without stripping it, every pull is spurious drift."""
    a = json.dumps({"players": [{"name": "X", "adp": 1.2}],
                    "start_date": "2026-07-28", "end_date": "2026-08-04",
                    "status": "Success", "total_drafts": 4460}).encode()
    b = json.dumps({"players": [{"name": "X", "adp": 1.2}],
                    "start_date": "2026-07-29", "end_date": "2026-08-05",
                    "status": "Success", "total_drafts": 4712}).encode()
    canon = canonicalize.get("ffc")
    assert canon(a) == canon(b)


def test_canonicalizer_still_sees_real_changes():
    canon = canonicalize.get("ffc")
    a = json.dumps({"players": [{"name": "X", "adp": 1.2}], "status": "Success"}).encode()
    b = json.dumps({"players": [{"name": "X", "adp": 2.4}], "status": "Success"}).encode()
    assert canon(a) != canon(b)


def test_unknown_canonicalizer_raises():
    with pytest.raises(KeyError, match="no canonicalizer"):
        canonicalize.get("nope")


# --------------------------------------------------------------- snapshots
def test_snapshot_id_is_stable_across_refetch():
    """Identical bytes must yield the same id, so fetched_at is excluded."""
    a = ManifestEntry("nflverse", "player_stats", "{}", "d" * 64, "2026-08-04T00:00:00+00:00", "identity")
    b = ManifestEntry("nflverse", "player_stats", "{}", "d" * 64, "2026-08-05T12:00:00+00:00", "identity")
    assert build_snapshot([a]) == build_snapshot([b])


def test_snapshot_id_is_order_independent():
    e1, e2 = _entry("player_stats", "1" * 64), _entry("schedules", "2" * 64)
    assert build_snapshot([e1, e2]) == build_snapshot([e2, e1])


def test_snapshot_id_changes_when_content_changes():
    assert build_snapshot([_entry(digest="1" * 64)]) != \
        build_snapshot([_entry(digest="2" * 64)])


def test_snapshot_id_format():
    sid = build_snapshot([_entry()])
    assert sid.startswith("sn2_") and len(sid) == 28


def test_manifest_roundtrip(snaps):
    entries = [_entry("player_stats", "1" * 64), _entry("schedules", "2" * 64)]
    sid = build_snapshot(entries)
    write_manifest(sid, entries, root=snaps)
    assert sorted(read_manifest(sid, root=snaps)) == sorted(entries)


def test_diff_reports_drift(snaps):
    """A retroactive stat correction is a digest change, not a crisis."""
    before = [_entry("player_stats", "1" * 64), _entry("schedules", "2" * 64)]
    after = [_entry("player_stats", "9" * 64), _entry("schedules", "2" * 64),
             _entry("injuries", "3" * 64)]
    a, b = build_snapshot(before), build_snapshot(after)
    write_manifest(a, before, root=snaps)
    write_manifest(b, after, root=snaps)
    report = diff(a, b, root=snaps)
    assert [x[1] for x in report.changed] == ["player_stats"]
    assert [x[1] for x in report.added] == ["injuries"]
    assert not report.removed
    assert not report.is_clean


def test_diff_refuses_v1_snapshot_ids(snaps):
    with pytest.raises(ValueError, match="not a v2 snapshot"):
        diff("legacy-uuid-1234", "sn2_" + "a" * 24, root=snaps)


# ----------------------------------------------------------------- registry
def test_unregistered_column_is_refused():
    with pytest.raises(UnregisteredColumnError, match="refusal, not permission"):
        availability_of("player_stats", "some_new_column")


def test_only_week1_lines_are_preseason():
    """Betting lines are preseason-available; final scores never are."""
    assert availability_of("schedules", "total_line").availability == "preseason"
    assert availability_of("schedules", "spread_line").availability == "preseason"
    assert availability_of("schedules", "home_score").availability == "week_end"
    assert availability_of("schedules", "total").availability == "week_end"


def test_static_sources_are_always_available():
    assert availability_of("combine", "forty").availability == "static"
    assert availability_of("draft_picks", "pick").availability == "static"


# -------------------------------------------------------------------- asof
def _stats_frame():
    rows = []
    for season in (2022, 2023, 2024):
        for week in (1, 2, 3):
            rows.append({"player_id": f"p{season}{week}", "season": season,
                         "week": week, "season_type": "REG",
                         "passing_yards": 100 * week, "position": "QB",
                         "team": "KC", "player_display_name": "X",
                         "opponent_team": "DEN"})
    return pd.DataFrame(rows)


def _handle(season, week=None, frames=None):
    frames = frames or {"player_stats": _stats_frame()}
    return asof(season, week=week, snapshot_id="sn2_" + "a" * 24,
                loaders={k: (lambda v=v: v.copy()) for k, v in frames.items()})


def test_asof_excludes_the_target_season():
    got = _handle(2024).player_stats()
    assert set(got.season) == {2022, 2023}, "target season must not leak"


def test_asof_weekly_view_admits_earlier_weeks_only():
    got = _handle(2024, week=3).player_stats()
    current = got[got.season == 2024]
    assert set(current.week) == {1, 2}
    assert 3 not in set(current.week)


def test_asof_refuses_unregistered_columns():
    frame = _stats_frame()
    frame["sneaky_future_stat"] = 1.0
    with pytest.raises(UnregisteredColumnError, match="sneaky_future_stat"):
        _handle(2024, frames={"player_stats": frame}).player_stats()


def test_empty_after_filtering_is_an_error():
    """Silently-empty sources are how a pipeline produces confident garbage."""
    with pytest.raises(EmptySourceError, match="empty after cutoff"):
        _handle(2022).player_stats()  # nothing before 2022 in the fixture


def test_missing_loader_names_what_is_available():
    with pytest.raises(DataError, match="no loader registered"):
        _handle(2024).team_stats()


def test_sentinel_never_reaches_output():
    """Poison a future value and assert it cannot escape the boundary (§21.2)."""
    frame = _stats_frame()
    sentinel = 999_999.0
    frame.loc[frame.season == 2024, "passing_yards"] = sentinel
    got = _handle(2024, frames={"player_stats": frame}).player_stats()
    assert sentinel not in set(got.passing_yards), "future sentinel leaked"


def test_truncated_rebuild_is_identical():
    """Building from full history and from physically truncated source must
    agree exactly — the property that catches every leak (§21.2)."""
    full = _stats_frame()
    truncated = full[full.season < 2024].copy()
    from_full = _handle(2024, frames={"player_stats": full}).player_stats()
    from_trunc = _handle(2024, frames={"player_stats": truncated}).player_stats()
    pd.testing.assert_frame_equal(
        from_full.reset_index(drop=True), from_trunc.reset_index(drop=True)
    )


def test_joining_across_cutoffs_raises():
    a = _handle(2024).player_stats()
    b = _handle(2023).player_stats()
    with pytest.raises(CutoffMismatchError, match="different as-of cutoffs"):
        assert_same_cutoff(a, b)


def test_same_cutoff_frames_join_cleanly():
    handle = _handle(2024)
    assert_same_cutoff(handle.player_stats(), handle.player_stats())


def test_assert_no_future_is_paired_with_the_filter():
    """The filter is a convention; the assertion is what makes it a boundary."""
    from src.platform.asof import guards
    bad = _stats_frame()
    with pytest.raises(LeakageError, match="LEAKAGE"):
        guards.assert_no_future(bad, 2024, where="unit-test")


# ------------------------------------------------------------ offline replay
def test_nflverse_roundtrips_through_the_blob_store(store):
    frame = _stats_frame()

    class FakeModule:
        @staticmethod
        def load_player_stats(seasons=None):
            return frame

    pull = nflverse.fetch("player_stats", seasons=[2024], blob_root=store,
                          loader_module=FakeModule)
    rehydrated = nflverse.load_from_blob(pull.entry.digest, blob_root=store)
    pd.testing.assert_frame_equal(
        rehydrated.sort_index(axis=1), frame.sort_index(axis=1)
    )


def test_loaders_for_snapshot_reads_offline(store):
    frame = _stats_frame()

    class FakeModule:
        @staticmethod
        def load_player_stats(seasons=None):
            return frame

    pull = nflverse.fetch("player_stats", blob_root=store, loader_module=FakeModule)
    loaders = nflverse.loaders_for_snapshot([pull.entry], blob_root=store)
    handle = asof(2024, snapshot_id=build_snapshot([pull.entry]), loaders=loaders)
    got = handle.player_stats()
    assert set(got.season) == {2022, 2023}


def test_column_reordering_is_not_drift(store):
    """Sorted columns before hashing, so an upstream reorder is not a change."""
    frame = _stats_frame()

    class A:
        @staticmethod
        def load_player_stats(seasons=None):
            return frame

    class B:
        @staticmethod
        def load_player_stats(seasons=None):
            return frame[list(reversed(frame.columns))]

    d1 = nflverse.fetch("player_stats", blob_root=store, loader_module=A).entry.digest
    d2 = nflverse.fetch("player_stats", blob_root=store, loader_module=B).entry.digest
    assert d1 == d2


# ------------------------------------------------------------------- FFC
def _ffc_payload(adp=1.6, window=("2026-07-28", "2026-08-04")):
    return json.dumps({
        "status": "Success", "total_drafts": 4460,
        "start_date": window[0], "end_date": window[1],
        "meta": {"type": "PPR", "teams": 12},
        "players": [
            {"player_id": 1, "name": "Jahmyr Gibbs", "position": "RB",
             "team": "DET", "adp": adp, "stdev": 0.7, "bye": 8},
            {"player_id": 2, "name": "Bijan Robinson", "position": "RB",
             "team": "ATL", "adp": 2.0, "stdev": 0.8, "bye": 5},
        ],
    }).encode()


class _FakeResponse:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status

    def json(self):
        return json.loads(self.content)


class _FakeClient:
    def __init__(self, response):
        self._response = response

    def get(self, url, params=None):
        return self._response

    def close(self):
        pass


def test_ffc_normalizes_and_pins_column_order(store):
    from src.platform.sources import ffc
    pull = ffc.fetch_adp(season=2026, blob_root=store,
                         client=_FakeClient(_FakeResponse(_ffc_payload())))
    assert list(pull.frame.columns)[:6] == [
        "player_id", "player_name", "position", "team", "adp", "adp_stdev"
    ]
    assert pull.frame.adp_stdev.notna().all(), "adp_stdev drives the survival curve"


def test_ffc_offline_replay_is_identical(store):
    """Column order is pinned precisely so this comparison can be exact."""
    from src.platform.sources import ffc
    pull = ffc.fetch_adp(season=2026, blob_root=store,
                         client=_FakeClient(_FakeResponse(_ffc_payload())))
    pd.testing.assert_frame_equal(
        ffc.load_from_blob(pull.entry.digest, blob_root=store), pull.frame
    )


def test_ffc_moving_window_does_not_change_the_digest(store):
    from src.platform.sources import ffc
    a = ffc.fetch_adp(season=2026, blob_root=store,
                      client=_FakeClient(_FakeResponse(_ffc_payload())))
    b = ffc.fetch_adp(season=2026, blob_root=store,
                      client=_FakeClient(_FakeResponse(
                          _ffc_payload(window=("2026-07-29", "2026-08-05")))))
    assert a.entry.digest == b.entry.digest


def test_ffc_real_adp_change_does_change_the_digest(store):
    from src.platform.sources import ffc
    a = ffc.fetch_adp(season=2026, blob_root=store,
                      client=_FakeClient(_FakeResponse(_ffc_payload(adp=1.6))))
    b = ffc.fetch_adp(season=2026, blob_root=store,
                      client=_FakeClient(_FakeResponse(_ffc_payload(adp=9.9))))
    assert a.entry.digest != b.entry.digest


def test_ffc_empty_season_raises_with_the_gap_named(store):
    """2025 returns status=Error with zero players; an empty ADP table would
    silently zero every survival curve, so it must raise."""
    from src.platform.sources import ffc
    empty = json.dumps({"status": "Error", "players": []}).encode()
    with pytest.raises(DataError, match="known FFC gap"):
        ffc.fetch_adp(season=2025, blob_root=store,
                      client=_FakeClient(_FakeResponse(empty)))


def test_ffc_http_error_raises(store):
    from src.platform.sources import ffc
    with pytest.raises(DataError, match="HTTP 503"):
        ffc.fetch_adp(season=2026, blob_root=store,
                      client=_FakeClient(_FakeResponse(b"", status=503)))


@pytest.mark.slow
def test_ffc_live_2026_is_available(store):
    """Draft night depends on this being live; Yahoo is blocked (§11.5.1)."""
    from src.platform.sources import ffc
    pull = ffc.fetch_adp(season=2026, teams=12, blob_root=store)
    assert pull.n_players > 150
    assert pull.frame.adp_stdev.notna().all()
    assert pull.frame.adp.is_monotonic_increasing
