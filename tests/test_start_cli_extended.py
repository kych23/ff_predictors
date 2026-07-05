"""Black-box extensions for start.py CLI helpers."""
import pathlib
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from scripts.start import _grade_color, _load_roster, _match_player


def _pool():
    return pd.DataFrame([
        {"player_id": "001", "name": "Josh Allen", "name_lower": "josh allen"},
        {"player_id": "002", "name": "Patrick Mahomes", "name_lower": "patrick mahomes"},
        {"player_id": "003", "name": "Saquon Barkley", "name_lower": "saquon barkley"},
    ])


def test_match_case_insensitive_partial():
    assert _match_player("MAHOMES", _pool()) == "002"
    assert _match_player("saquon", _pool()) == "003"


def test_match_full_name_wins_over_partial():
    assert _match_player("Josh Allen", _pool()) == "001"


def test_match_empty_pool_returns_none():
    empty = pd.DataFrame(columns=["player_id", "name", "name_lower"])
    assert _match_player("Josh Allen", empty) is None


def test_grade_color_extremes():
    assert _grade_color(1)[0] == "A"
    grade_worst, _ = _grade_color(32)
    assert grade_worst not in ("A", "B")


def test_grade_returns_two_values():
    grade, color = _grade_color(5)
    assert isinstance(grade, str)


def test_load_roster_status_defaults_preserved():
    content = """
season: 2026
roster:
  - name: "Player One"
    position: WR
    team: KC
    status: Q
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        f.flush()
        data = _load_roster(f.name)
    assert data["season"] == 2026
    assert data["roster"][0]["status"] == "Q"
    assert data["roster"][0]["team"] == "KC"
