"""Black-box extensions: name normalization + rookie feature fallbacks."""
import pandas as pd

from src.features.rookies import UDFA_SYNTHETIC_PICK, build_rookie_features
from src.ingest.player_ids import normalize_name


def test_normalize_case_insensitive():
    assert normalize_name("JOSH ALLEN") == normalize_name("josh allen")


def test_normalize_strips_periods():
    assert normalize_name("D.K. Metcalf") == "d k metcalf"


def test_normalize_strips_roman_suffix():
    assert normalize_name("Kenneth Walker III") == "kenneth walker"


def test_normalize_nickname_expansion():
    # Mike -> Michael mapping documented in test_id_crosswalk
    assert normalize_name("Mike Williams") == "michael williams"


def test_normalize_empty_and_none():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""


def test_normalize_idempotent():
    for raw in ("A.J. Brown", "Robert Griffin III", "Mike Evans", "Josh Allen"):
        once = normalize_name(raw)
        assert normalize_name(once) == once


def test_drafted_rookie_not_udfa():
    players = pd.DataFrame([
        {"player_id": "00-d", "name": "First Rounder", "position": "WR",
         "draft_round": 1, "draft_pick": 10, "rookie_year": 2023},
    ])
    out = build_rookie_features(
        ["00-d"], players, combine=pd.DataFrame(),
        id_map=pd.DataFrame(columns=["gsis_id", "pfr_id"]), college=None,
    )
    row = out.iloc[0]
    assert row["is_udfa"] == 0
    assert row["draft_pick"] == 10


def test_multiple_rookies_multiple_rows():
    players = pd.DataFrame([
        {"player_id": "00-1", "name": "R One", "position": "RB",
         "draft_round": 2, "draft_pick": 40, "rookie_year": 2023},
        {"player_id": "00-2", "name": "R Two", "position": "WR",
         "draft_round": None, "draft_pick": None, "rookie_year": 2023},
    ])
    out = build_rookie_features(
        ["00-1", "00-2"], players, combine=pd.DataFrame(),
        id_map=pd.DataFrame(columns=["gsis_id", "pfr_id"]), college=None,
    )
    assert len(out) == 2
    by_id = out.set_index("player_id")
    assert by_id.loc["00-1", "is_udfa"] == 0
    assert by_id.loc["00-2", "is_udfa"] == 1
    assert by_id.loc["00-2", "draft_pick"] == UDFA_SYNTHETIC_PICK


def test_udfa_pick_worse_than_any_real_pick():
    """The synthetic UDFA slot must sort after the entire real draft (~262)."""
    assert UDFA_SYNTHETIC_PICK > 262
