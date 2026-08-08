"""Every league fact has exactly ONE definition, and it lives in a yaml file.

Three separate failure shapes, all of which existed here:

* a fact written twice in two files, so an edit to one leaves the other stale;
* a constant written twice in two modules that must agree by construction;
* a function default that shadows config, right until a caller forgets to pass
  the config.

None of these break a test on the day they are introduced. They break the day
someone changes the league.
"""
from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import yaml

from src.core.config import load_league, load_strategy
from src.core.config.strategy import DEFAULT_STRATEGY_PATH
from src.core.errors import ConfigError


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture()
def strategy_raw():
    return yaml.safe_load(DEFAULT_STRATEGY_PATH.read_text())


def _load_from(tmp_path, raw, league):
    path = tmp_path / "strategy.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_strategy(league, str(path))


# ------------------------------------------- the same fact in two files
def test_adp_market_must_match_the_league_size(tmp_path, league, strategy_raw):
    """`adp.teams` picks which FFC market to pull and is the same fact as
    `league.teams`. Disagreeing prices the draft against a different league's
    board — every survival curve and wait term with it."""
    raw = copy.deepcopy(strategy_raw)
    raw["adp"]["teams"] = league.teams + 2
    with pytest.raises(ConfigError, match="adp.teams"):
        _load_from(tmp_path, raw, league)


def test_omitting_the_adp_market_inherits_the_league(tmp_path, league,
                                                     strategy_raw):
    raw = copy.deepcopy(strategy_raw)
    raw["adp"]["teams"] = None
    assert _load_from(tmp_path, raw, league).adp.teams == league.teams


def test_the_shipped_config_agrees_with_itself(league):
    assert load_strategy(league).adp.teams == league.teams


# ------------------------------------ the same constant in two modules
def test_the_replacement_floor_has_one_definition():
    """The board's value and the simulator's selection value for the same
    unvalued player. Two literals meant two answers to one question."""
    import scripts.build_bundle as build_bundle
    from src.core.constants import REPLACEMENT_FLOOR_QUANTILE

    assert build_bundle.REPLACEMENT_QUANTILE == REPLACEMENT_FLOOR_QUANTILE

    source = Path("src/engine/sim/bundle_build.py").read_text()
    assert "REPLACEMENT_FLOOR_QUANTILE" in source
    assert "np.quantile(valued, 0.10)" not in source


def test_the_separation_z_has_one_definition():
    """The dollar half of this rule (`indifference_zone_dollars`) is
    configurable; the confidence half was a bare literal in three places."""
    from src.core.constants import SEPARATION_Z

    assert pytest.approx(1.6448536269514722) == SEPARATION_Z
    for module in ("src/engine/decision/allocate.py",
                   "src/engine/decision/ladder_tiers.py"):
        source = Path(module).read_text()
        assert "1.645 *" not in source, f"{module} still has a literal z"
        assert "SEPARATION_Z" in source


# --------------------------------------- defaults that shadow config
@pytest.mark.parametrize(("module", "func", "param"), [
    ("src.engine.sim.waiver", "apply_waiver", "regular_season_weeks"),
    ("src.platform.sources.ffc", "fetch_adp", "teams"),
])
def test_config_shadowing_defaults_are_always_overridden(module, func, param):
    """These defaults are real values that duplicate config. They are legal
    only because every live caller passes the config value; this pins that,
    so a new caller cannot quietly inherit a stale league size."""
    import importlib

    signature = inspect.signature(getattr(importlib.import_module(module), func))
    assert signature.parameters[param].default is not inspect.Parameter.empty

    callers = {
        "regular_season_weeks": ("src/engine/sim/kernel.py",
                                 "regular_season_weeks=cfg.schedule."
                                 "regular_season_weeks"),
        "teams": ("scripts/build_bundle.py", "teams=strategy.adp.teams"),
    }
    path, expected = callers[param]
    assert expected in Path(path).read_text(), (
        f"{path} must pass {param} from config, not rely on the default")


# ------------------------------------------- scoring comes from config
def test_no_module_hardcodes_a_scoring_value(league):
    """`scoring/engine.py` maps config KEYS to nflverse columns. If a point
    VALUE ever appears there, two files decide what a touchdown is worth."""
    source = Path("src/domain/scoring/engine.py").read_text()
    for key, value in league.scoring.offense.items():
        assert f'"{key}": {value}' not in source
        assert f"'{key}': {value}" not in source


def test_every_scoring_caller_passes_config():
    """`score_offense` takes the coefficients as an argument precisely so no
    caller can reach for a global."""
    from src.domain.scoring.engine import score_offense

    parameters = list(inspect.signature(score_offense).parameters)
    assert parameters[1] == "offense"
    assert inspect.signature(score_offense).parameters["offense"].default \
        is inspect.Parameter.empty, "coefficients must never default"
