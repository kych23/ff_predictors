"""College->NFL bridge tests (DrafterSpec.md §4.5 acceptance).

Asserts a minimum match rate AND that unmatched rookies still produce valid rows
via the NaN + has_college_stats=0 fallback (no silent bad rows, no drops).
"""
import pandas as pd

from src.features.rookies import build_rookie_features, UDFA_SYNTHETIC_PICK
from src.ingest.player_ids import normalize_name


def test_name_normalization():
    assert normalize_name("A.J. Brown") == "a j brown"
    assert normalize_name("Robert Griffin III") == "robert griffin"
    assert normalize_name("Mike Evans") == "michael evans"
    assert normalize_name(None) == ""


def test_draft_bridge_match_rate_real_data():
    """The draft_picks bridge should resolve a healthy fraction to cfb ids."""
    from src.ingest.player_ids import build_crosswalk
    cw = build_crosswalk()
    assert cw.match_rate >= 0.5  # most modern draft picks carry a cfb_player_id
    assert len(cw.id_map) > 1000


def test_unmatched_rookie_kept_with_fallback():
    """A rookie with no combine and no college match is KEPT, flags = 0."""
    players = pd.DataFrame([
        {"player_id": "00-x", "name": "Nobody", "position": "WR",
         "draft_round": 7, "draft_pick": 250, "rookie_year": 2023},
    ])
    out = build_rookie_features(
        ["00-x"], players, combine=pd.DataFrame(),
        id_map=pd.DataFrame(columns=["gsis_id", "pfr_id"]), college=None,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["has_college_stats"] == 0
    assert row["has_combine"] == 0
    assert row["is_rookie"] is True or row["is_rookie"] == True  # noqa: E712


def test_udfa_synthetic_slot_and_flag():
    players = pd.DataFrame([
        {"player_id": "00-u", "name": "Undrafted", "position": "RB",
         "draft_round": None, "draft_pick": None, "rookie_year": 2023},
    ])
    out = build_rookie_features(
        ["00-u"], players, combine=pd.DataFrame(),
        id_map=pd.DataFrame(columns=["gsis_id", "pfr_id"]), college=None,
    )
    row = out.iloc[0]
    assert row["is_udfa"] == 1
    assert row["draft_pick"] == UDFA_SYNTHETIC_PICK
