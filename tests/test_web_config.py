"""Web config must not touch the decision hash.

`load_strategy` hashes the ENTIRE raw strategy mapping into `strategy_hash`
(`strategy.py:205-207`), and that flows to `decision_version` and to
`rng.seed_root`. Putting the web block in `strategy.yaml` would therefore
re-seed every simulation stream and invalidate the `decision_version` stamped
into the current bundle — for settings that are pure transport.

A test that merely loads `web.yaml` twice and compares hashes would pass
vacuously: a file no loader reads cannot be an input to a hash. So this pins
the hash to a literal instead, which fails the moment anyone moves the block.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.config import load_league, load_strategy
from src.core.config.web import DEFAULT_WEB_PATH, load_web
from src.core.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_YAML = ROOT / "config" / "strategy.yaml"


def test_the_strategy_hash_is_pinned():
    """If this fails, either strategy.yaml legitimately changed — in which case
    rebuild the bundle and update the literal — or a transport setting leaked
    into the hashed config."""
    strategy = load_strategy(load_league())
    assert strategy.strategy_hash == "3f825e77493b", (
        f"strategy_hash moved to {strategy.strategy_hash}; every RNG seed and "
        f"the bundle's decision_version just changed")


def test_strategy_yaml_has_no_web_block():
    raw = yaml.safe_load(STRATEGY_YAML.read_text())
    assert "web" not in raw, (
        "web settings belong in config/web.yaml; adding them here re-seeds "
        "every simulation")


def test_the_web_config_is_a_separate_file():
    assert DEFAULT_WEB_PATH.exists()
    assert DEFAULT_WEB_PATH.name == "web.yaml"


def test_loading_the_web_config_does_not_perturb_the_strategy_hash():
    before = load_strategy(load_league()).strategy_hash
    load_web()
    after = load_strategy(load_league()).strategy_hash
    assert before == after


# ------------------------------------------------------------- the loader
def test_defaults_are_usable_without_editing_anything():
    cfg = load_web()
    assert cfg.source in ("manual", "replay", "yahoo")
    assert cfg.engine.reps > 0 and cfg.engine.budget_seconds > 0


def test_relative_paths_anchor_to_the_repo_root():
    """The entry point is a long-lived server that may be launched from any
    cwd, unlike the scripts it replaces."""
    cfg = load_web()
    resolved = cfg.resolved(cfg.session_path)
    assert resolved.is_absolute()
    assert str(resolved).startswith(str(ROOT))


def test_rankings_dir_is_read_from_the_yaml(tmp_path):
    """`load_web` builds `WebConfig` field by field and silently drops unknown
    keys, so adding the dataclass field alone ships an INERT setting: the yaml
    would be ignored and the default would always win. This pins both halves.
    """
    path = tmp_path / "web.yaml"
    path.write_text("rankings_dir: data/elsewhere/boards\n")
    assert load_web(path).rankings_dir == Path("data/elsewhere/boards")


def test_the_rankings_dir_anchors_to_the_repo_root():
    """Boards must not land wherever uvicorn happened to be launched from."""
    cfg = load_web()
    assert cfg.resolved(cfg.rankings_dir).is_absolute()


def test_an_unknown_source_is_rejected(tmp_path):
    path = tmp_path / "web.yaml"
    path.write_text("source: telepathy\n")
    with pytest.raises(ConfigError, match="source"):
        load_web(path)


def test_replay_without_a_path_is_rejected(tmp_path):
    path = tmp_path / "web.yaml"
    path.write_text("source: replay\n")
    with pytest.raises(ConfigError, match="replay_path"):
        load_web(path)


def test_a_nonpositive_budget_is_rejected(tmp_path):
    path = tmp_path / "web.yaml"
    path.write_text("engine:\n  budget_seconds: 0\n")
    with pytest.raises(ConfigError, match="budget_seconds"):
        load_web(path)


def test_a_missing_file_says_what_to_do(tmp_path):
    with pytest.raises(ConfigError, match="no web config"):
        load_web(tmp_path / "absent.yaml")


def test_shortlist_is_read_from_the_hashed_config_not_from_web_yaml():
    """It changes which candidate wins, so it must stay inside strategy_hash.
    web.yaml deliberately does not define it."""
    raw = yaml.safe_load(DEFAULT_WEB_PATH.read_text())
    assert "shortlist" not in (raw.get("engine") or {})
    strategy = load_strategy(load_league())
    assert strategy.simulation.decision["shortlist_size"] > 0
