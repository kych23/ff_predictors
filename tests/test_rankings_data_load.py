"""`RankingsData.load` against a real artifacts directory.

This file exists because its absence hid a bug. Every other rankings test
builds `RankingsData(...)` directly from fixtures that are already filtered and
already indexed, so 1,077 green tests said nothing about the code that actually
assembles the object in production — and `_load_matrix` was serving an
arbitrary historical row whenever the season could not be resolved.

The concrete failure: the training matrix holds every season from 2012 (5,961
rows against 907 for one season) and its `player_id` index is NOT unique across
them, so an unfiltered frame makes `_matrix_row`'s `.iloc[0]` return the
EARLIEST row. Ja'Marr Chase's card reported his 2021 rookie age of 21.5 as
current, with no warning. A card that says "no data" is honest; one confidently
showing the wrong season is not.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.app.rankings.data import RankingsData


@pytest.fixture()
def artifacts(tmp_path):
    """A miniature artifacts root: two seasons of matrix, tiers, and a meta."""
    root = tmp_path / "artifacts"
    root.mkdir()

    rows = []
    for season in (2025, 2026):
        for i in range(3):
            rows.append({
                "player_id": f"wr-{i:02d}", "season": season,
                "position": "WR",
                "fppg": float("nan") if season == 2026 else 10.0,
                "prior_fppg": 1.0 if season == 2025 else 9.0,
                "age_at_season_start": 22.0 if season == 2025 else 23.0,
            })
    pd.DataFrame(rows).to_parquet(
        root / "training_matrix_sn2_deadbeefdeadbeefdeadbeef.parquet")

    pd.DataFrame([
        {"player_id": "wr-00", "player_name": "WR Player 0", "position": "WR",
         "tier": 1, "rank": 1, "adp": 1.0},
    ]).to_parquet(root / "tiers_sn2_cafebabecafebabecafebabe.parquet")

    (root / "projections_sn2_feedfacefeedfacefeedface.meta.json").write_text(
        json.dumps({"target_season": 2026,
                    "snapshot_id": "sn2_feedfacefeedfacefeedface"}))
    return root


@pytest.fixture()
def players():
    return pd.DataFrame([
        {"player_id": f"wr-{i:02d}", "player_name": f"WR Player {i}",
         "position": "WR", "team": "DET", "bye_week": 5, "value": 10.0 - i,
         "value_p10": 7.0, "value_p90": 13.0, "value_source": "quantile_model",
         "coverage": "full", "adp": 1.0 + i, "adp_stdev": 2.0}
        for i in range(3)
    ])


def _load(artifacts, players, cfg, strategy, web_cfg) -> RankingsData:
    return RankingsData.load(cfg, strategy, web_cfg, players, "sn2_bundle",
                             root=artifacts)


def test_load_selects_only_the_target_season(artifacts, players, web_cfg):
    from src.core.config import load_league, load_strategy

    cfg = load_league()
    data = _load(artifacts, players, cfg, load_strategy(cfg), web_cfg)

    assert data.target_season == 2026
    assert len(data.matrix) == 3, "rows from another season leaked in"
    assert set(data.matrix["season"]) == {2026}
    assert data.matrix.index.is_unique, (
        "a non-unique index makes .iloc[0] pick an arbitrary season")
    assert data.matrix.loc["wr-00", "age_at_season_start"] == 23.0


def test_a_missing_manifest_serves_no_matrix_rather_than_a_stale_one(
        artifacts, players, web_cfg):
    """The bug this file was written for.

    Without a target season the old code skipped the filter entirely and served
    whichever row sorted first — five years stale, presented as current."""
    from src.core.config import load_league, load_strategy

    for meta in artifacts.glob("projections_*.meta.json"):
        meta.unlink()

    cfg = load_league()
    data = _load(artifacts, players, cfg, load_strategy(cfg), web_cfg)

    assert data.target_season == 0
    assert data.matrix.empty, "an unresolved season must serve NO production"

    from src.app.rankings.detail import player_detail

    card = player_detail("wr-00", data)
    assert card is not None, "the player is still on the board"
    assert card["has_matrix_row"] is False
    assert card["production"] is None
    assert card["projection"]["value"] == 10.0, "the bundle still answers"


def test_a_corrupt_manifest_degrades_the_same_way(artifacts, players, web_cfg):
    from src.core.config import load_league, load_strategy

    for meta in artifacts.glob("projections_*.meta.json"):
        meta.write_text("{ not json")

    cfg = load_league()
    data = _load(artifacts, players, cfg, load_strategy(cfg), web_cfg)
    assert data.target_season == 0
    assert data.matrix.empty


def test_an_empty_artifacts_root_still_yields_a_usable_object(tmp_path,
                                                              players,
                                                              web_cfg):
    """Losing every artifact must degrade, never raise — `lifespan` catches
    broadly, but a raise there means the cockpit starts degraded for no reason.
    """
    from src.core.config import load_league, load_strategy

    empty = tmp_path / "nothing"
    empty.mkdir()
    cfg = load_league()
    data = _load(empty, players, cfg, load_strategy(cfg), web_cfg)

    assert data.matrix.empty and data.tiers.empty
    assert len(data.board) == 3, "the bundle is an argument, not an artifact"
    assert data.replacement, "replacement is computed from the board"


def test_the_snapshot_ids_are_reported_with_their_prefix(artifacts, players,
                                                         web_cfg):
    """These ids exist so a human can eyeball whether the matrix pairs with the
    bundle. Two different spellings defeat that."""
    from src.core.config import load_league, load_strategy

    cfg = load_league()
    data = _load(artifacts, players, cfg, load_strategy(cfg), web_cfg)

    assert data.snapshots["matrix"] == "sn2_deadbeefdeadbeefdeadbeef"
    assert data.snapshots["tiers"] == "sn2_cafebabecafebabecafebabe"
    assert data.snapshots["bundle"] == "sn2_bundle"
    assert all(str(v).startswith("sn2_") for v in data.snapshots.values())


def test_empty_keeps_the_board_it_is_given(players):
    """`board` is a function ARGUMENT, not a loaded artifact, so it cannot be
    what failed. Dropping it on the degraded path would make every player on a
    saved board report as stale — the user's research appearing erased."""
    data = RankingsData.empty(board=players)
    assert len(data.board) == 3
    assert data.replacement


def test_two_players_sharing_a_normalized_name_claim_no_market_row(web_cfg):
    """The market frame has no player_id, so a name collision would hand both
    players the same ranks. Neither may claim it."""
    from src.app.rankings.data import _match_keys

    twins = pd.DataFrame([
        {"player_id": "a", "player_name": "Mike Williams", "position": "WR"},
        {"player_id": "b", "player_name": "Mike Williams", "position": "WR"},
        {"player_id": "c", "player_name": "Puka Nacua", "position": "WR"},
    ])
    keys = _match_keys(twins)
    assert "a" not in keys and "b" not in keys
    assert "c" in keys
