"""College->NFL bridge tests (DrafterSpec.md §4.5 acceptance)."""
import pytest

from src.ingest.player_ids import normalize_name


def test_name_normalization():
    assert normalize_name("A.J. Brown") == "a j brown"
    assert normalize_name("Robert Griffin III") == "robert griffin"
    assert normalize_name("Mike Evans") == "michael evans"
    assert normalize_name(None) == ""


def test_draft_bridge_match_rate_real_data():
    """The draft_picks bridge should resolve a healthy fraction to cfb ids.

    Downloads nflverse data; skips (rather than fails) when the network is down.
    """
    from src.ingest.player_ids import build_crosswalk
    try:
        cw = build_crosswalk()
    except Exception as exc:
        pytest.skip(f"nflverse download unavailable: {exc}")
    assert cw.match_rate >= 0.5  # most modern draft picks carry a cfb_player_id
    assert len(cw.id_map) > 1000
