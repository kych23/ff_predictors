"""Typed loader for ``config/web.yaml``.

Separate from ``strategy.py`` on purpose. ``load_strategy`` hashes the entire
raw mapping into ``strategy_hash``, which feeds ``decision_version`` and
``rng.seed_root``; a transport setting has no business re-seeding a simulation.
Keeping the web config in its own file makes that separation structural rather
than a rule someone has to remember.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from src.core.errors import ConfigError

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_WEB_PATH: Final = _REPO_ROOT / "config" / "web.yaml"

#: Sources a session may be created with.
SOURCES: Final = ("manual", "replay", "yahoo")


@dataclass(frozen=True)
class EngineConfig:
    """Decision-affecting knobs that cannot live in the hashed config.

    ``reps`` and ``budget_seconds`` change which candidate wins, but they are
    also machine- and time-dependent, so they cannot sit inside
    ``strategy_hash``. The compromise is that both are stamped into the ledger
    entry and the API payload, so a recommendation always carries the settings
    that produced it.
    """

    reps: int = 512
    budget_seconds: float = 25.0


@dataclass(frozen=True)
class YahooWebConfig:
    token_path: Path = Path(".yahoo_token.json")


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    source: str = "manual"
    poll_interval_seconds: float = 3.0
    auto_recommend: bool = True
    replay_path: Path | None = None
    replay_auto_recommend: bool = False
    narration_timeout_seconds: float = 12.0
    pick_clock_seconds: int = 60
    session_path: Path = Path("data/web/session.json")
    ledger_path: Path = Path("data/web/ledger.sqlite")
    #: Draft-order team names, seat 0 first. Short blanks fall back to
    #: "Team N" — a placeholder that looks like one beats a wrong name
    #: that looks right.
    team_names: tuple[str, ...] = ()
    engine: EngineConfig = field(default_factory=EngineConfig)
    yahoo: YahooWebConfig = field(default_factory=YahooWebConfig)

    def resolved(self, path: Path) -> Path:
        """Anchor a configured relative path to the repo root.

        The entry point is a long-lived server that may be launched from any
        working directory, unlike the scripts this replaces, which were always
        run from the repo root.
        """
        p = Path(path)
        return p if p.is_absolute() else _REPO_ROOT / p


def _require_mapping(raw: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(raw).__name__}")
    return raw


def load_web(path: Path | None = None) -> WebConfig:
    target = Path(path) if path is not None else DEFAULT_WEB_PATH
    if not target.exists():
        raise ConfigError(
            f"no web config at {target}. Copy config/web.yaml or pass a path.")
    raw = _require_mapping(yaml.safe_load(target.read_text()) or {}, str(target))

    source = str(raw.get("source", "manual"))
    if source not in SOURCES:
        raise ConfigError(
            f"web.source must be one of {list(SOURCES)}, got {source!r}")

    replay_path = raw.get("replay_path")
    if source == "replay" and not replay_path:
        raise ConfigError("web.source is 'replay' but replay_path is unset")

    engine_raw = _require_mapping(raw.get("engine", {}) or {}, "web.engine")
    reps = int(engine_raw.get("reps", 512))
    budget = float(engine_raw.get("budget_seconds", 25.0))
    if reps <= 0:
        raise ConfigError(f"web.engine.reps must be positive, got {reps}")
    if budget <= 0:
        raise ConfigError(
            f"web.engine.budget_seconds must be positive, got {budget}")

    yahoo_raw = _require_mapping(raw.get("yahoo", {}) or {}, "web.yahoo")

    return WebConfig(
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 8000)),
        source=source,
        poll_interval_seconds=float(raw.get("poll_interval_seconds", 3.0)),
        auto_recommend=bool(raw.get("auto_recommend", True)),
        replay_path=Path(replay_path) if replay_path else None,
        replay_auto_recommend=bool(raw.get("replay_auto_recommend", False)),
        narration_timeout_seconds=float(
            raw.get("narration_timeout_seconds", 12.0)),
        pick_clock_seconds=int(raw.get("pick_clock_seconds", 60)),
        session_path=Path(raw.get("session_path", "data/web/session.json")),
        ledger_path=Path(raw.get("ledger_path", "data/web/ledger.sqlite")),
        team_names=tuple(str(n) for n in (raw.get('team_names') or [])),
        engine=EngineConfig(reps=reps, budget_seconds=budget),
        yahoo=YahooWebConfig(
            token_path=Path(yahoo_raw.get("token_path", ".yahoo_token.json"))),
    )
