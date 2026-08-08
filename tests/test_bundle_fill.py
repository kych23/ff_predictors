"""`fill_unvalued` — the board half of the "never a constant zero" rule.

The bug this covers shipped for weeks and looked like a strategy: no rehearsal
roster ever contained a kicker or a defence, which reads as sensible streaming
until you notice the engine *cannot* draft one. Every K and DST arrived with
`value = 0.0`, scored 0.0 in the VONA shortlist, and so was never among the
candidates tier 0 simulates. The simulator priced them correctly the whole
time; only the board disagreed.

The same zero hit 23 skill players, including a rookie at ADP 25.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_bundle  # noqa: E402


def _frame(rows):
    return pd.DataFrame([
        {"player_id": str(i), "player_name": f"p{i}", "position": pos,
         "team": "AAA", "bye_week": 7, "value": val,
         "value_source": "quantile_model" if val > 0 else "none",
         "coverage": "full" if val > 0 else "no_prior_season",
         "adp": 10.0 + i, "adp_stdev": 1.0}
        for i, (pos, val) in enumerate(rows)
    ])


@pytest.fixture
def kdst_artifact(tmp_path, monkeypatch):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "kdst_sn2_test.json").write_text(json.dumps({
        "distributions": {"K": {"mean": 8.0, "quantiles": [0.0] * 101},
                          "DST": {"mean": 6.25, "quantiles": [0.0] * 101}},
    }))
    monkeypatch.setattr(build_bundle, "ARTIFACTS", art)
    return art


# ------------------------------------------------------------ K and DST
def test_kdst_get_their_fitted_mean(kdst_artifact):
    out = build_bundle.fill_unvalued(_frame([("K", 0.0), ("DST", 0.0)]))
    assert out.loc[0, "value"] == pytest.approx(8.0)
    assert out.loc[1, "value"] == pytest.approx(6.25)
    assert set(out["value_source"]) == {"kdst_empirical"}


def test_every_kicker_becomes_draftable(kdst_artifact):
    """The actual regression: a positive value is what lets the shortlist —
    and therefore the simulator — ever see a kicker."""
    out = build_bundle.fill_unvalued(_frame([("K", 0.0)] * 20 + [("DST", 0.0)] * 23))
    assert (out["value"] > 0).all()


def test_a_valued_kdst_row_is_not_overwritten(kdst_artifact):
    """Only zeros are placeholders. A real number stays a real number."""
    out = build_bundle.fill_unvalued(_frame([("K", 11.5)]))
    assert out.loc[0, "value"] == pytest.approx(11.5)
    assert out.loc[0, "value_source"] == "quantile_model"


def test_missing_artifact_warns_and_leaves_kdst_at_zero(tmp_path, monkeypatch, capsys):
    """Silence here is what let the original bug live. K and DST have no valued
    peers, so the replacement floor cannot rescue them — the run must say so."""
    empty = tmp_path / "none"
    empty.mkdir()
    monkeypatch.setattr(build_bundle, "ARTIFACTS", empty)
    out = build_bundle.fill_unvalued(_frame([("K", 0.0), ("RB", 10.0)]))
    assert out.loc[0, "value"] == 0.0
    assert "CANNOT be drafted" in capsys.readouterr().out


# ------------------------------------------------- the replacement floor
def test_unvalued_skill_players_get_the_positional_floor(kdst_artifact):
    rows = [("RB", 20.0), ("RB", 10.0), ("RB", 5.0), ("RB", 0.0)]
    out = build_bundle.fill_unvalued(_frame(rows))
    expected = float(pd.Series([20.0, 10.0, 5.0]).quantile(0.10))
    assert out.loc[3, "value"] == pytest.approx(expected)
    assert out.loc[3, "value_source"] == "replacement_floor"


def test_the_floor_is_per_position(kdst_artifact):
    """A zeroed WR must not inherit the RB pool's floor."""
    out = build_bundle.fill_unvalued(_frame(
        [("RB", 30.0), ("RB", 0.0), ("WR", 3.0), ("WR", 0.0)]))
    assert out.loc[1, "value"] > out.loc[3, "value"]


def test_the_floor_is_a_low_end_value_not_a_competitive_one(kdst_artifact):
    rows = [("RB", float(v)) for v in range(4, 44)] + [("RB", 0.0)]
    out = build_bundle.fill_unvalued(_frame(rows))
    valued = out[out["value_source"] == "quantile_model"]["value"]
    floor = out.iloc[-1]["value"]
    assert 0 < floor < valued.median()


def test_the_floor_may_outrank_the_very_worst_projections_by_design(kdst_artifact):
    """Deliberate, and the reason this is a *quantile* rather than a minimum.

    On the live board the positional 10th percentile sits above 3-7 genuinely
    projected players. That is what replacement level means: a player forecast
    below a freely available pickup really is worth less than the pickup. Using
    `min(valued)` instead would make the floor hostage to the single worst row
    on the board and would stop expressing that idea at all.
    """
    rows = [("RB", float(v)) for v in range(4, 44)] + [("RB", 0.0)]
    out = build_bundle.fill_unvalued(_frame(rows))
    valued = out[out["value_source"] == "quantile_model"]["value"]
    floor = out.iloc[-1]["value"]
    assert (valued < floor).any()
    assert floor > valued.min()


def test_a_position_with_no_valued_players_is_left_alone(kdst_artifact):
    """No sample to take a quantile of. Better a loud zero than a fabricated
    number — and the print says how many survived."""
    out = build_bundle.fill_unvalued(_frame([("TE", 0.0), ("TE", 0.0)]))
    assert (out["value"] == 0.0).all()


# ----------------------------------------------------------- provenance
def test_coverage_still_records_that_there_was_no_projection(kdst_artifact):
    """The value becomes usable; the provenance must not improve with it, or
    the board's `stale:` banner would stop warning about these rows."""
    out = build_bundle.fill_unvalued(_frame([("K", 0.0), ("RB", 9.0), ("RB", 0.0)]))
    assert list(out["coverage"]) == ["no_prior_season", "full", "no_prior_season"]


def test_placeholders_are_never_labelled_as_a_fitted_model(kdst_artifact):
    out = build_bundle.fill_unvalued(_frame([("K", 0.0), ("RB", 9.0), ("RB", 0.0)]))
    filled = out[out["coverage"] == "no_prior_season"]["value_source"]
    assert set(filled) == {"kdst_empirical", "replacement_floor"}
    assert "quantile_model" not in set(filled)


def test_declared_value_sources_cover_what_is_emitted(kdst_artifact):
    from src.platform.bundle import VALUE_SOURCES

    out = build_bundle.fill_unvalued(_frame([("K", 0.0), ("RB", 9.0), ("RB", 0.0)]))
    assert set(out["value_source"]) <= set(VALUE_SOURCES)


def test_the_input_frame_is_not_mutated(kdst_artifact):
    frame = _frame([("K", 0.0)])
    build_bundle.fill_unvalued(frame)
    assert frame.loc[0, "value"] == 0.0


# ----------------------------------------- artifact selection is by TIME
def test_newest_ignores_lexicographic_order(tmp_path):
    """Artifact filenames end in a snapshot_id, which is a content HASH, so
    sorting them alphabetically orders them at random.

    This is not hypothetical: with four real projection artifacts on disk,
    `sorted(glob(...))[-1]` selected a two-day-old file over one written
    minutes earlier. `build_bundle.load_projections` used that idiom, so a
    freshly fitted model would be built, stamped and then silently ignored.
    """
    import os

    from src.models.artifacts import newest, newest_all

    stale = tmp_path / "projections_sn2_ffffffff.meta.json"
    fresh = tmp_path / "projections_sn2_00000000.meta.json"
    stale.write_text("{}")
    fresh.write_text("{}")
    os.utime(stale, (1_000_000, 1_000_000))     # older mtime, later name
    os.utime(fresh, (2_000_000, 2_000_000))

    assert newest("projections_*.meta.json", root=tmp_path) == fresh
    assert sorted(tmp_path.glob("projections_*.meta.json"))[-1] == stale, (
        "the fixture must reproduce the inversion this guards")
    assert newest_all("projections_*.meta.json", root=tmp_path) == [fresh, stale]


def test_newest_returns_none_on_an_empty_directory(tmp_path):
    from src.models.artifacts import newest, newest_all

    assert newest("projections_*.json", root=tmp_path) is None
    assert newest_all("projections_*.json", root=tmp_path) == []


# ------------------------------------- artifact identity has a CODE axis
def test_code_digest_changes_when_model_source_changes(tmp_path):
    """`snapshot_id` is a Merkle root over the SOURCE DATA, so identical
    inputs under changed feature code produce a different artifact carrying an
    identical id. Observed: fixing the prior-production decay moved Josh
    Allen's p50 from 20.30 to 20.88 and both artifacts were stamped
    `sn2_58ced77e1b3cfec91e9fadd6` with the same model_version.
    """
    from src.models.artifacts import code_digest

    surface = tmp_path / "src" / "models"
    surface.mkdir(parents=True)
    (surface / "m.py").write_text("HALFLIFE = 1.0\n")
    before = code_digest(("src/models",), root=tmp_path)

    (surface / "m.py").write_text("HALFLIFE = 2.0\n")
    assert code_digest(("src/models",), root=tmp_path) != before


def test_code_digest_is_stable_when_nothing_changes(tmp_path):
    from src.models.artifacts import code_digest

    surface = tmp_path / "src" / "models"
    surface.mkdir(parents=True)
    (surface / "m.py").write_text("X = 1\n")
    assert (code_digest(("src/models",), root=tmp_path)
            == code_digest(("src/models",), root=tmp_path))


def test_code_digest_notices_a_new_file_and_a_rename(tmp_path):
    """Filenames are hashed alongside contents, so moving logic between
    modules registers even when the bytes are unchanged."""
    from src.models.artifacts import code_digest

    surface = tmp_path / "src" / "models"
    surface.mkdir(parents=True)
    (surface / "a.py").write_text("X = 1\n")
    one = code_digest(("src/models",), root=tmp_path)

    (surface / "b.py").write_text("Y = 2\n")
    two = code_digest(("src/models",), root=tmp_path)
    assert two != one

    (surface / "b.py").unlink()
    (surface / "c.py").write_text("Y = 2\n")
    assert code_digest(("src/models",), root=tmp_path) != two


def test_code_digest_ignores_a_missing_surface(tmp_path):
    from src.models.artifacts import code_digest

    assert isinstance(code_digest(("src/nonexistent",), root=tmp_path), str)


def test_the_real_tree_has_a_nonempty_digest():
    from src.models.artifacts import CODE_SURFACES, code_digest

    assert len(code_digest()) == 12
    assert code_digest() != code_digest(CODE_SURFACES[:1])


def test_code_digest_ignores_comments_and_docstrings(tmp_path):
    """Hashing raw bytes made a documentation pass invalidate the parity
    golden, which trains people to regenerate it reflexively — and a guard
    that is regenerated on sight has stopped guarding anything."""
    from src.models.artifacts import code_digest

    surface = tmp_path / "src" / "models"
    surface.mkdir(parents=True)
    (surface / "m.py").write_text(
        '"""Original docstring."""\n\n\ndef f(x):\n    """Doc."""\n    return x + 1\n')
    before = code_digest(("src/models",), root=tmp_path)

    (surface / "m.py").write_text(
        '"""Rewritten docstring, much longer now."""\n\n\n'
        'def f(x):\n    """Different doc."""\n    # a new comment\n    return x + 1\n')
    assert code_digest(("src/models",), root=tmp_path) == before


def test_code_digest_still_catches_a_changed_constant(tmp_path):
    from src.models.artifacts import code_digest

    surface = tmp_path / "src" / "models"
    surface.mkdir(parents=True)
    (surface / "m.py").write_text("def f(x):\n    return x + 1\n")
    before = code_digest(("src/models",), root=tmp_path)
    (surface / "m.py").write_text("def f(x):\n    return x + 2\n")
    assert code_digest(("src/models",), root=tmp_path) != before


def test_code_digest_survives_a_syntax_error(tmp_path):
    """A half-saved file must not make the digest throw on every caller."""
    from src.models.artifacts import code_digest

    surface = tmp_path / "src" / "models"
    surface.mkdir(parents=True)
    (surface / "broken.py").write_text("def f(:\n")
    assert len(code_digest(("src/models",), root=tmp_path)) == 12


def test_a_docstring_only_module_stays_parseable(tmp_path):
    from src.models.artifacts import code_digest

    surface = tmp_path / "src" / "models"
    surface.mkdir(parents=True)
    (surface / "m.py").write_text('"""Only a docstring."""\n')
    assert len(code_digest(("src/models",), root=tmp_path)) == 12


_BASE_SOURCE = "def f(a, b):\n    if a < b:\n        return a * 2\n    return b + 1\n"

_SEMANTIC_MUTATIONS = {
    "flipped_comparison": _BASE_SOURCE.replace("a < b", "a > b"),
    "reordered_args": _BASE_SOURCE.replace("def f(a, b)", "def f(b, a)"),
    "changed_literal": _BASE_SOURCE.replace("* 2", "* 3"),
    "swapped_operator": _BASE_SOURCE.replace("b + 1", "b - 1"),
    "removed_branch": "def f(a, b):\n    return b + 1\n",
    "renamed_function": _BASE_SOURCE.replace("def f(", "def g("),
    "added_default": _BASE_SOURCE.replace("def f(a, b)", "def f(a, b=0)"),
}

_COSMETIC_MUTATIONS = {
    "comment": "# new comment\n" + _BASE_SOURCE,
    "docstring": _BASE_SOURCE.replace(
        "def f(a, b):\n", 'def f(a, b):\n    """Doc."""\n'),
    "whitespace": _BASE_SOURCE.replace("    return b + 1", "    return  b+1"),
}


def _digest_after(tmp_path, source):
    from src.models.artifacts import code_digest

    surface = tmp_path / "src" / "models"
    surface.mkdir(parents=True, exist_ok=True)
    (surface / "m.py").write_text(source)
    return code_digest(("src/models",), root=tmp_path)


@pytest.mark.parametrize("name", sorted(_SEMANTIC_MUTATIONS))
def test_code_digest_catches_every_semantic_change(tmp_path, name):
    """This digest is what stops the parity golden passing against a different
    engine, so a miss here silently disarms the only regression guard on the
    recommendation itself."""
    before = _digest_after(tmp_path, _BASE_SOURCE)
    assert _digest_after(tmp_path, _SEMANTIC_MUTATIONS[name]) != before


@pytest.mark.parametrize("name", sorted(_COSMETIC_MUTATIONS))
def test_code_digest_ignores_cosmetic_change(tmp_path, name):
    """The other failure mode: a guard that fires on prose gets regenerated on
    sight, and a fixture regenerated on sight has stopped guarding."""
    before = _digest_after(tmp_path, _BASE_SOURCE)
    assert _digest_after(tmp_path, _COSMETIC_MUTATIONS[name]) == before
