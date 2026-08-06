"""Tests for start.py CLI helpers (roster loading, matching, grades)."""

import pytest

# §9.4: this module is pinned to the v1 config shape. api/, web/ and the
# weekly start/sit surface are FROZEN for the build window, so v1 stays
# alive underneath them and these tests are deselected from the default
# run rather than migrated. Thaw is a §22.2 follow-up.
pytestmark = pytest.mark.v1_frozen
import tempfile
import pathlib

import pandas as pd
import pytest

# Import the helpers directly from the script
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from scripts.start import _load_roster, _match_player, _grade_color


def test_load_roster_valid():
    content = """
season: 2025
roster:
  - name: "Josh Allen"
    position: QB
    team: BUF
    status: active
  - name: "Saquon Barkley"
    position: RB
    team: PHI
    status: IR
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        f.flush()
        data = _load_roster(f.name)
    assert data["season"] == 2025
    assert len(data["roster"]) == 2
    assert data["roster"][0]["name"] == "Josh Allen"
    assert data["roster"][1]["status"] == "IR"


def test_load_roster_missing_file():
    with pytest.raises(SystemExit, match="not found"):
        _load_roster("/nonexistent/path.yaml")


def test_match_player_exact():
    df = pd.DataFrame([
        {"player_id": "001", "name": "Josh Allen", "name_lower": "josh allen"},
        {"player_id": "002", "name": "Patrick Mahomes", "name_lower": "patrick mahomes"},
    ])
    assert _match_player("Josh Allen", df) == "001"
    assert _match_player("josh allen", df) == "001"


def test_match_player_partial():
    df = pd.DataFrame([
        {"player_id": "001", "name": "Josh Allen", "name_lower": "josh allen"},
        {"player_id": "002", "name": "Patrick Mahomes", "name_lower": "patrick mahomes"},
    ])
    assert _match_player("Mahomes", df) == "002"


def test_match_player_no_match():
    df = pd.DataFrame([
        {"player_id": "001", "name": "Josh Allen", "name_lower": "josh allen"},
    ])
    assert _match_player("Nonexistent Player", df) is None


def test_grade_color_tiers():
    grade_a, color_a = _grade_color(5)
    assert grade_a == "A"
    grade_b, color_b = _grade_color(15)
    assert grade_b == "B"
    grade_c, color_c = _grade_color(25)
    assert grade_c == "C"


def test_grade_color_none():
    grade, color = _grade_color(None)
    assert grade == "—"


def test_grade_color_boundaries():
    assert _grade_color(10)[0] == "A"
    assert _grade_color(11)[0] == "B"
    assert _grade_color(22)[0] == "B"
    assert _grade_color(23)[0] == "C"
