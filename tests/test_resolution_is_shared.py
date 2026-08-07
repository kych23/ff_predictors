"""Both cockpits resolve names through ONE implementation.

`scripts/draft_night.py` used to carry its own copy of the spine + substring
cascade. Two implementations of identity resolution is precisely the split-brain
that produced the K/DST defect — the board and the simulator disagreeing about
the same player for weeks — and name resolution is higher-stakes, because the
failure mode is drafting the wrong person.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.app.web.resolve import build_spine, resolve_name  # noqa: E402


def test_draft_night_imports_the_shared_resolver():
    import draft_night

    from src.app.web import resolve as shared

    assert draft_night.build_spine is shared.build_spine
    assert draft_night.resolve_name is shared.resolve_name


def test_draft_night_defines_no_second_cascade():
    """A local `build_spine` would silently shadow the shared one."""
    src = Path("scripts/draft_night.py").read_text()
    assert "def build_spine(" not in src, (
        "draft_night must not redefine the cascade; import it")


@pytest.mark.parametrize("query,expected", [
    ("RB Player 0", "rb-00"),
    ("rb player 0", "rb-00"),          # case-insensitive
])
def test_both_paths_agree_on_an_exact_name(synthetic_board, query, expected):
    import draft_night

    spine = build_spine(synthetic_board.players, synthetic_board.snapshot_id)
    shared = resolve_name(query, spine, synthetic_board.players)
    pid, _name, _tier, _options = draft_night.resolve(
        spine, query, synthetic_board.players)
    assert shared.player_id == expected
    assert pid == expected


def test_both_paths_agree_that_a_name_is_ambiguous(synthetic_board):
    import draft_night

    spine = build_spine(synthetic_board.players, synthetic_board.snapshot_id)
    shared = resolve_name("Player 1", spine, synthetic_board.players)
    pid, _name, _tier, options = draft_night.resolve(
        spine, "Player 1", synthetic_board.players)
    assert shared.status == "ambiguous" and len(shared.candidates) > 1
    assert pid is None and len(options) > 1


def test_both_paths_agree_that_a_name_is_unknown(synthetic_board):
    import draft_night

    spine = build_spine(synthetic_board.players, synthetic_board.snapshot_id)
    shared = resolve_name("Zzzz Nobody", spine, synthetic_board.players)
    pid, _name, _tier, options = draft_night.resolve(
        spine, "Zzzz Nobody", synthetic_board.players)
    assert shared.status == "unresolved"
    assert pid is None and options == []
