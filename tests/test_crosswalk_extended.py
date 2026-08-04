"""Black-box extensions: name normalization."""
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
