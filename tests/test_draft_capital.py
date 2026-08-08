"""Draft capital — the only feature that separates one rookie from another.

Rookies have no prior-season production, so every `prior_*` column is null for
them. Measured on the artifact before this feature existed: rookie RBs
projected with a standard deviation of 0.74 against 4.00 for veterans, and a
maximum below the veteran median plus one SD. The model could not tell a
first-round back from an undrafted one.
"""
from __future__ import annotations

import pandas as pd

from src.models.features.draft_capital import (
    CAPITAL_COLS,
    UNDRAFTED_PICK,
    UNDRAFTED_ROUND,
    build_draft_capital,
)


def _picks():
    """Shaped like `ff_playerids`, which is the source — see the module
    docstring for why `draft_picks` cannot be used."""
    return pd.DataFrame({
        "gsis_id": ["a", "b", "c"],
        "draft_year": [2026, 2026, 2026],
        "draft_round": [1, 3, 7],
        "draft_ovr": [12, 80, 240],
    })


def test_every_player_gets_a_row():
    out = build_draft_capital(_picks(), ["a", "b", "c", "undrafted"])
    assert len(out) == 4
    assert set(CAPITAL_COLS) <= set(out.columns)


def test_draft_position_is_carried_through():
    out = build_draft_capital(_picks(), ["a", "b", "c"]).set_index("player_id")
    assert out.at["a", "draft_pick_overall"] == 12
    assert out.at["a", "draft_round"] == 1
    assert out.at["c", "draft_round"] == 7


def test_an_undrafted_player_sorts_past_the_end_of_the_draft():
    """Not null (the tree would have to learn a split around it) and not zero
    (which reads as first overall). A UDFA is worse in expectation than the
    last pick of the seventh round, so that is where he goes."""
    out = build_draft_capital(_picks(), ["nobody"]).set_index("player_id")
    assert out.at["nobody", "is_undrafted"] == 1
    assert out.at["nobody", "draft_pick_overall"] == UNDRAFTED_PICK
    assert out.at["nobody", "draft_round"] == UNDRAFTED_ROUND
    assert out.at["nobody", "draft_capital_score"] < 0.2


def test_the_score_decreases_monotonically_with_pick():
    out = build_draft_capital(_picks(), ["a", "b", "c"]).set_index("player_id")
    scores = [out.at[p, "draft_capital_score"] for p in ("a", "b", "c")]
    assert scores == sorted(scores, reverse=True)


def test_the_score_is_steep_early_and_flat_late():
    """Pick value is not linear. A raw pick number makes the tree spend splits
    recovering curvature the feature can just carry."""
    out = build_draft_capital(
        pd.DataFrame({"gsis_id": ["p1", "p33", "p200", "p232"],
                      "draft_year": [2026] * 4, "draft_round": [1, 1, 6, 7],
                      "draft_ovr": [1, 33, 200, 232]}),
        ["p1", "p33", "p200", "p232"]).set_index("player_id")
    early = out.at["p1", "draft_capital_score"] - out.at["p33", "draft_capital_score"]
    late = out.at["p200", "draft_capital_score"] - out.at["p232", "draft_capital_score"]
    assert early > late * 5


def test_a_duplicate_selection_keeps_the_earliest():
    picks = pd.DataFrame({"gsis_id": ["a", "a"], "draft_year": [2025, 2026],
                          "draft_round": [2, 5], "draft_ovr": [40, 150]})
    out = build_draft_capital(picks, ["a"]).set_index("player_id")
    assert out.at["a", "draft_pick_overall"] == 40


def test_a_missing_draft_table_degrades_rather_than_raising():
    """No draft data must not stop a bundle build."""
    out = build_draft_capital(pd.DataFrame(), ["a", "b"])
    assert (out["is_undrafted"] == 1).all()
    assert len(out) == 2


def test_rows_without_a_gsis_id_are_dropped_not_joined_blindly():
    picks = pd.DataFrame({"gsis_id": ["a", None], "draft_year": [2026, 2026],
                          "draft_round": [1, 2], "draft_ovr": [5, 40]})
    out = build_draft_capital(picks, ["a", "b"]).set_index("player_id")
    assert out.at["a", "draft_pick_overall"] == 5
    assert out.at["b", "is_undrafted"] == 1


def test_the_feature_reads_no_in_season_data():
    """As-of safety is structural here: the function is handed the draft table
    and a list of ids, and there is no parameter through which a snap, target
    or game result could enter."""
    import inspect

    params = inspect.signature(build_draft_capital).parameters
    assert set(params) == {"player_ids_table", "player_ids"}
    src = inspect.getsource(build_draft_capital)
    assert "fetch(" not in src
    for banned in ("week", "points", "target", "snap", "fppg"):
        assert banned not in src.lower(), f"{banned!r} has no business here"


def test_the_source_covers_training_and_serving_equally():
    """The failure this feature already had once: `draft_picks` keys on
    `gsis_id`, which nflverse has backfilled for 2022-2025 but NOT 2026 (0% —
    provisional ids). Training saw full coverage, serving saw none, and rookie
    projections collapsed. A draft-capital source is only usable if both sides
    of the split see the same thing."""
    import inspect

    from src.models.features.assemble import assemble_season

    src = inspect.getsource(assemble_season)
    assert 'build_draft_capital(frames.get("id_map"' in src, (
        "draft capital must come from ff_playerids (id_map), not draft_picks")
