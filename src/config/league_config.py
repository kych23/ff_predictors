"""Typed loader + validator for ``config/league.yaml`` (DrafterSpec.md §4.2.1).

Central so every module reads settings exactly one way. Computes the config
version hash that is embedded in ``model_version`` on every projection and
prediction, making outputs traceable to the rules that produced them (§4.0).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml

# Repo-root-relative default path: src/config/league_config.py -> repo root is parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "league.yaml"


@dataclass(frozen=True)
class ScoringConfig:
    pass_yd: float
    pass_td: float
    interception: float   # YAML key is `int` (avoid shadowing the builtin here)
    rush_yd: float
    rush_td: float
    rec: float
    rec_yd: float
    rec_td: float
    two_pt: float
    fumble_lost: float
    return_td: float


@dataclass(frozen=True)
class RosterConfig:
    slots: Dict[str, int]
    flex_eligible: List[str]
    modeled_positions: List[str]
    # Recommender-only soft caps: heavily discourage drafting past N at a position
    # (e.g. {TE: 2} = one starter + one bench, then sink any further TE). Does NOT
    # affect the projection model, so it is excluded from the model version hash.
    max_per_position: Dict[str, int] = field(default_factory=dict)

    @property
    def rounds(self) -> int:
        """Draft rounds = total roster size (one pick per slot per round)."""
        return sum(self.slots.values())

    def starter_slots(self, position: str) -> int:
        """Pure (non-flex) starter slots for a position, league-wide is teams*this."""
        return int(self.slots.get(position, 0))


@dataclass(frozen=True)
class AdpConfig:
    format: str
    teams: int
    adp_min_ranking: int


@dataclass(frozen=True)
class TrainingConfig:
    min_games_train: int
    ewm_halflife_seasons: float
    min_train_seasons: int
    min_bucket_n: int


@dataclass(frozen=True)
class BacktestConfig:
    start_season: int
    end_season: int
    as_of_offset_days: int


@dataclass(frozen=True)
class LeagueConfig:
    teams: int
    roster: RosterConfig
    scoring: ScoringConfig
    adp: AdpConfig
    training: TrainingConfig
    backtest: BacktestConfig
    version_hash: str = field(compare=False)
    raw: dict = field(compare=False, repr=False, default_factory=dict)

    @property
    def adp_min_fullsim(self) -> int:
        """Full-draft ADP coverage needed for Tier-2 sim = teams * rounds (§4.0)."""
        return self.teams * self.roster.rounds

    @property
    def model_version(self) -> str:
        return f"draft_v1.{self.version_hash}"


def _compute_version_hash(raw: dict) -> str:
    """Deterministic short hash of the MODEL-affecting config content.

    Recommender-only preferences (e.g. ``roster.max_per_position``) don't change any
    projection, so they're excluded — the model version stays stable when you only
    tweak draft-strategy knobs.
    """
    import copy
    r = copy.deepcopy(raw)
    if isinstance(r.get("roster"), dict):
        r["roster"].pop("max_per_position", None)
    blob = json.dumps(r, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _validate(raw: dict) -> None:
    required_top = {"league", "roster", "scoring", "adp", "training", "backtest"}
    missing = required_top - set(raw)
    if missing:
        raise ValueError(f"league.yaml missing top-level sections: {sorted(missing)}")

    roster = raw["roster"]
    slots = roster["slots"]
    flex_elig = roster["flex_eligible"]
    modeled = roster["modeled_positions"]
    if "FLEX" in slots and not flex_elig:
        raise ValueError("roster.flex_eligible must be non-empty when a FLEX slot exists")
    for p in flex_elig:
        if p not in slots:
            raise ValueError(f"flex_eligible position {p!r} has no roster slot")
    for p in modeled:
        if p not in slots:
            raise ValueError(f"modeled_position {p!r} has no roster slot")

    scoring_keys = {
        "pass_yd", "pass_td", "int", "rush_yd", "rush_td", "rec", "rec_yd",
        "rec_td", "two_pt", "fumble_lost", "return_td",
    }
    missing_scoring = scoring_keys - set(raw["scoring"])
    if missing_scoring:
        raise ValueError(f"scoring section missing keys: {sorted(missing_scoring)}")
    extra_scoring = set(raw["scoring"]) - scoring_keys
    if extra_scoring:
        raise ValueError(f"scoring section has unknown keys: {sorted(extra_scoring)}")

    if "teams" not in raw["league"]:
        raise ValueError("league.yaml missing league.teams")
    training_keys = {"min_games_train", "ewm_halflife_seasons",
                     "min_train_seasons", "min_bucket_n"}
    missing_training = training_keys - set(raw["training"])
    if missing_training:
        raise ValueError(f"training section missing keys: {sorted(missing_training)}")
    backtest_keys = {"start_season", "end_season"}
    missing_backtest = backtest_keys - set(raw["backtest"])
    if missing_backtest:
        raise ValueError(f"backtest section missing keys: {sorted(missing_backtest)}")
    adp_keys = {"format", "adp_min_ranking"}
    missing_adp = adp_keys - set(raw["adp"])
    if missing_adp:
        raise ValueError(f"adp section missing keys: {sorted(missing_adp)}")


@lru_cache(maxsize=8)
def load_config(path: str | None = None) -> LeagueConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"league config not found at {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text())
    _validate(raw)

    lg = raw["league"]
    roster = RosterConfig(
        slots={str(k): int(v) for k, v in raw["roster"]["slots"].items()},
        flex_eligible=[str(p) for p in raw["roster"]["flex_eligible"]],
        modeled_positions=[str(p) for p in raw["roster"]["modeled_positions"]],
        max_per_position={str(k): int(v)
                          for k, v in raw["roster"].get("max_per_position", {}).items()},
    )
    scoring = ScoringConfig(**{("interception" if k == "int" else k): float(v)
                               for k, v in raw["scoring"].items()})
    adp = AdpConfig(
        format=str(raw["adp"]["format"]),
        teams=int(raw["adp"].get("teams", lg["teams"])),
        adp_min_ranking=int(raw["adp"]["adp_min_ranking"]),
    )
    training = TrainingConfig(
        min_games_train=int(raw["training"]["min_games_train"]),
        ewm_halflife_seasons=float(raw["training"]["ewm_halflife_seasons"]),
        min_train_seasons=int(raw["training"]["min_train_seasons"]),
        min_bucket_n=int(raw["training"]["min_bucket_n"]),
    )
    backtest = BacktestConfig(
        start_season=int(raw["backtest"]["start_season"]),
        end_season=int(raw["backtest"]["end_season"]),
        as_of_offset_days=int(raw["backtest"].get("as_of_offset_days", 1)),
    )
    return LeagueConfig(
        teams=int(lg["teams"]),
        roster=roster,
        scoring=scoring,
        adp=adp,
        training=training,
        backtest=backtest,
        version_hash=_compute_version_hash(raw),
        raw=raw,
    )
