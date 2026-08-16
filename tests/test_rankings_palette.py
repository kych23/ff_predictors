"""Tier colours have one definition, and they render.

Two separate failures this guards, neither of which shows up as a test error
anywhere else.

**Drift.** `TIER_COLORS` lives in `schema.py` and the server rejects any token
outside it. The frontend has its own copy, because TypeScript cannot import a
Python tuple. Two lists that must agree by construction is exactly the shape
`test_config_is_single_source.py` exists to catch.

**Silent purging.** Tailwind's JIT scans source for WHOLE class names and drops
anything assembled at runtime, so `bg-${color}-600` produces no CSS at all. The
page renders, the tiers are simply colourless, and no test fails. So this
asserts the classes are spelled out literally.

**Collision.** `positions.ts` already owns green/orange/blue/pink/slate. A tier
ramp reusing one of those hues would make the same colour mean two things on
one screen, inside the same row.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.app.rankings.schema import TIER_COLORS

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ROOT / "web" / "src" / "rankings" / "palette.ts"
MODEL = ROOT / "web" / "src" / "rankings" / "model.ts"
POSITIONS = ROOT / "web" / "src" / "positions.ts"

#: Hues `positions.ts` has already spent. A tier may not reuse one.
POSITION_HUES = ("green", "orange", "blue", "pink", "slate")


def _without_comments(source: str) -> str:
    """Strip block and line comments.

    The palette's own docstring names the anti-pattern it exists to prevent —
    an interpolated class — so a raw substring scan would flag the explanation
    as the offence.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"^\s*//.*$", "", source, flags=re.M)


@pytest.fixture(scope="module")
def palette() -> str:
    return _without_comments(PALETTE.read_text())


def test_every_python_token_exists_in_the_frontend_palette(palette):
    for token in TIER_COLORS:
        assert f"{token}:" in palette, (
            f"{token} is accepted by the server and unknown to the client")


def test_the_typescript_union_matches_the_python_tuple():
    source = _without_comments(MODEL.read_text())
    match = re.search(r"export type TierColor\s*=([^;]+);", source)
    assert match, "TierColor union not found"
    declared = set(re.findall(r'"(t\d)"', match.group(1)))
    assert declared == set(TIER_COLORS)


def test_the_classes_are_literal_not_interpolated(palette):
    """`bg-${c}-600` compiles to nothing — Tailwind purges what it cannot see
    spelled out. The page still renders; the tiers are just invisible."""
    assert "${" not in palette, "a class name is being built at runtime"
    for token in TIER_COLORS:
        block = re.search(rf"\b{token}:\s*\"([^\"]+)\"", palette)
        assert block, f"{token} has no literal class string"
        assert "bg-" in block.group(1) or "border-" in block.group(1)


def test_no_tier_colour_reuses_a_position_hue(palette):
    """positions.ts: 'a colour that means running back in one place and
    something else two components over is worse than no colour at all.'"""
    classes = " ".join(re.findall(r'"([^"]+)"', palette))
    for hue in POSITION_HUES:
        assert f"-{hue}-" not in classes, (
            f"the tier ramp reuses {hue}, which positions.ts already owns")


def test_the_position_palette_still_owns_those_hues():
    """Guards the guard: if positions.ts is recoloured, the check above stops
    meaning anything unless this fails first."""
    source = POSITIONS.read_text()
    for hue in POSITION_HUES:
        assert f"-{hue}-" in source


def test_every_token_has_a_header_body_and_swatch(palette):
    for name in ("TIER_HEADER", "TIER_BODY", "TIER_SWATCH"):
        block = re.search(rf"{name}[^=]*=\s*\{{(.*?)\}};", palette, re.S)
        assert block, f"{name} not found"
        for token in TIER_COLORS:
            assert f"{token}:" in block.group(1), f"{name} is missing {token}"
