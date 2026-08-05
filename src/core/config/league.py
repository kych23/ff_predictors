"""Typed loader for ``config/league.v2.yaml`` (§10.2, §10.4, §10.5).

Everything here is model-affecting by construction, which is why the version
hash has no exclusion list: the v1 loader excluded ``max_per_position`` so that
tuning a draft knob would not invalidate trained models, and v2 solves the same
problem better by moving that key to ``strategy.yaml`` entirely (§10.1).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import yaml

from src.core.config.slots import SlotPlan, build_slot_plan
from src.core.constants import CALIBRATION_BUCKETS, SLOT_BENCH
from src.core.errors import ConfigError

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_LEAGUE_PATH: Final = _REPO_ROOT / "config" / "league.v2.yaml"

_REQUIRED_SECTIONS: Final = frozenset({
    "league", "roster", "schedule", "scoring", "training", "weekly", "backtest",
})


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RosterConfig:
    slots: Mapping[str, int]
    flex_eligibility: Mapping[str, tuple[str, ...]]
    modeled_positions: tuple[str, ...]

    @property
    def rounds(self) -> int:
        """Draft rounds = total roster size, one pick per slot per round."""
        return sum(self.slots.values())

    def starter_slots(self, position: str) -> int:
        """Pure (non-flex) starting slots for a position."""
        return int(self.slots.get(position, 0))

    @property
    def flex_slot_names(self) -> tuple[str, ...]:
        """Slot names that accept more than their own position."""
        return tuple(self.flex_eligibility)

    @property
    def flex_eligible(self) -> frozenset[str]:
        """Positions accepted by *any* flex slot.

        v1 stored this as a flat list because it assumed a single FLEX slot;
        v2 keys eligibility by slot name so multiple (and overlapping) flex
        slots are expressible. This flattened view answers "could this position
        ever take a flex slot", which is what the tier-2 recommender asks.

        It is deliberately NOT the right question for lineup legality once more
        than one flex slot exists — that is what ``SlotPlan`` and the matroid
        oracle are for (§15.3). ``total_flex_slots`` is likewise a count, not a
        capacity check.
        """
        return frozenset(p for positions in self.flex_eligibility.values()
                         for p in positions)

    @property
    def total_flex_slots(self) -> int:
        return sum(int(self.slots.get(name, 0)) for name in self.flex_eligibility)


@dataclass(frozen=True)
class ScheduleConfig:
    regular_season_weeks: int
    playoff_weeks: tuple[int, ...]
    playoff_teams: int
    playoff_byes: int
    matchup_length_weeks: int
    seeding_tiebreakers: tuple[str, ...]

    @property
    def total_weeks(self) -> int:
        return self.regular_season_weeks + len(self.playoff_weeks)


@dataclass(frozen=True)
class ScoringConfig:
    offense: Mapping[str, float]
    kicker: Mapping[str, Any]
    dst: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingConfig:
    min_games_train: int
    ewm_halflife_seasons: float
    min_train_seasons: int
    min_bucket_n: int
    calibration_buckets: tuple[str, ...]


@dataclass(frozen=True)
class WeeklyConfig:
    min_games_for_sigma: int
    sigma_floor_ratio: float
    kdst_cdf_grid: int


@dataclass(frozen=True)
class BacktestConfig:
    start_season: int
    end_season: int
    as_of_offset_days: int


@dataclass(frozen=True)
class LeagueConfig:
    teams: int
    type: str
    kind: str
    roster: RosterConfig
    schedule: ScheduleConfig
    scoring: ScoringConfig
    scoring_exceptions: tuple[Mapping[str, Any], ...]
    training: TrainingConfig
    weekly: WeeklyConfig
    backtest: BacktestConfig
    slot_plan: SlotPlan = field(compare=False)
    league_hash: str = field(compare=False, default="")
    raw: Mapping[str, Any] = field(compare=False, repr=False, default_factory=dict)

    @property
    def model_version(self) -> str:
        return f"draft_v2.{self.league_hash}"


# --------------------------------------------------------------------------- #
# validation (§10.5 rules 1-7)
# --------------------------------------------------------------------------- #
def _tiles(bands: Sequence[Sequence[float]], lo: float, hi: float, *, where: str) -> None:
    """Assert bands cover [lo, hi] with no gap and no overlap."""
    if not bands:
        raise ConfigError(f"{where}: no bands defined")
    ordered = sorted(bands, key=lambda b: b[0])
    if ordered[0][0] != lo:
        raise ConfigError(f"{where}: first band starts at {ordered[0][0]}, expected {lo}")
    if ordered[-1][1] != hi:
        raise ConfigError(f"{where}: last band ends at {ordered[-1][1]}, expected {hi}")
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev[1] >= nxt[0]:
            raise ConfigError(f"{where}: bands {prev} and {nxt} overlap")
        if nxt[0] != prev[1] + 1:
            raise ConfigError(f"{where}: gap between {prev} and {nxt}")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _validate(raw: Mapping[str, Any]) -> None:
    missing = _REQUIRED_SECTIONS - set(raw)
    if missing:
        raise ConfigError(f"league config missing sections: {sorted(missing)}")

    roster = raw["roster"]
    slots: Mapping[str, int] = roster["slots"]
    flex: Mapping[str, Sequence[str]] = roster.get("flex_eligibility", {})
    modeled: Sequence[str] = roster["modeled_positions"]

    # rule 1 — flex targets resolve
    for slot, positions in flex.items():
        if slot not in slots:
            raise ConfigError(f"flex_eligibility names slot {slot!r} with no roster slot")
        if not positions:
            raise ConfigError(f"flex_eligibility[{slot!r}] is empty")
        for p in positions:
            if p not in slots and p not in modeled:
                raise ConfigError(
                    f"flex_eligibility[{slot!r}] accepts {p!r}, which has neither a "
                    f"roster slot nor a place in modeled_positions"
                )

    # rule 2 — slot-count sanity. `rounds` is DERIVED, so asserting
    # sum(slots) == rounds would be a tautology; assert the counts instead.
    for slot, count in slots.items():
        if slot == SLOT_BENCH:
            if count < 0:
                raise ConfigError(f"roster.slots.{SLOT_BENCH} must be >= 0, got {count}")
        elif count < 1:
            raise ConfigError(f"roster.slots.{slot} must be >= 1, got {count}")

    for p in modeled:
        if p not in slots:
            raise ConfigError(f"modeled_position {p!r} has no roster slot")

    # rules 3, 4 — scoring bands tile their domains
    _tiles(raw["scoring"]["kicker"]["fg_bands"], 0, 99, where="scoring.kicker.fg_bands")
    _tiles(raw["scoring"]["dst"]["points_allowed_tiers"], 0, 999,
           where="scoring.dst.points_allowed_tiers")

    # rule 5 — the bracket fits
    sched = raw["schedule"]
    teams = int(raw["league"]["teams"])
    pt, pb = int(sched["playoff_teams"]), int(sched["playoff_byes"])
    if pt > teams:
        raise ConfigError(f"playoff_teams ({pt}) exceeds teams ({teams})")
    if not _is_power_of_two(pt - pb):
        raise ConfigError(
            f"playoff_teams - playoff_byes = {pt - pb} is not a power of two, "
            f"so the bracket does not resolve"
        )
    depth = (pt - pb - 1).bit_length() + (1 if pb else 0)
    need = depth * int(sched["matchup_length_weeks"])
    if need > len(sched["playoff_weeks"]):
        raise ConfigError(
            f"bracket needs {need} playoff week(s) "
            f"({depth} rounds x {sched['matchup_length_weeks']}), "
            f"but playoff_weeks has {len(sched['playoff_weeks'])}"
        )
    if int(sched["regular_season_weeks"]) + len(sched["playoff_weeks"]) > 18:
        raise ConfigError("regular_season_weeks + playoff_weeks exceeds 18")

    # rule 7 — configured bucket names must match the canonical set. The
    # calibrator itself cannot be imported here (it is above `core`), so core
    # owns the names and tests/test_config_v2.py asserts the calibrator agrees.
    configured = tuple(raw["training"]["calibration_buckets"])
    if set(configured) != set(CALIBRATION_BUCKETS):
        raise ConfigError(
            f"training.calibration_buckets {sorted(configured)} does not match "
            f"the canonical {sorted(CALIBRATION_BUCKETS)} — the key would "
            f"silently do nothing while still changing model_version"
        )

    # rule 6 — declared exceptions resolve to a real scoring key
    for exc in raw.get("scoring_exceptions") or ():
        key = exc.get("key", "")
        node: Any = raw["scoring"]
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise ConfigError(f"scoring_exceptions key {key!r} does not resolve")
            node = node[part]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def compute_hash(raw: Mapping[str, Any]) -> str:
    """Hash the canonicalized parsed mapping, not the file bytes.

    Comments and whitespace therefore do not force a pipeline re-run, matching
    the v1 behavior this replaces.
    """
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _parse(raw: Mapping[str, Any]) -> LeagueConfig:
    lg, roster_raw = raw["league"], raw["roster"]
    slots = {str(k): int(v) for k, v in roster_raw["slots"].items()}
    flex = {
        str(k): tuple(str(p) for p in v)
        for k, v in (roster_raw.get("flex_eligibility") or {}).items()
    }
    roster = RosterConfig(
        slots=slots,
        flex_eligibility=flex,
        modeled_positions=tuple(str(p) for p in roster_raw["modeled_positions"]),
    )
    sched_raw = raw["schedule"]
    schedule = ScheduleConfig(
        regular_season_weeks=int(sched_raw["regular_season_weeks"]),
        playoff_weeks=tuple(int(w) for w in sched_raw["playoff_weeks"]),
        playoff_teams=int(sched_raw["playoff_teams"]),
        playoff_byes=int(sched_raw["playoff_byes"]),
        matchup_length_weeks=int(sched_raw.get("matchup_length_weeks", 1)),
        seeding_tiebreakers=tuple(str(t) for t in sched_raw["seeding_tiebreakers"]),
    )
    sc = raw["scoring"]
    training_raw, weekly_raw, bt = raw["training"], raw["weekly"], raw["backtest"]

    return LeagueConfig(
        teams=int(lg["teams"]),
        type=str(lg["type"]),
        kind=str(lg["kind"]),
        roster=roster,
        schedule=schedule,
        scoring=ScoringConfig(
            offense={str(k): float(v) for k, v in sc["offense"].items()},
            kicker=sc["kicker"],
            dst=sc["dst"],
        ),
        scoring_exceptions=tuple(raw.get("scoring_exceptions") or ()),
        training=TrainingConfig(
            min_games_train=int(training_raw["min_games_train"]),
            ewm_halflife_seasons=float(training_raw["ewm_halflife_seasons"]),
            min_train_seasons=int(training_raw["min_train_seasons"]),
            min_bucket_n=int(training_raw["min_bucket_n"]),
            calibration_buckets=tuple(str(b) for b in training_raw["calibration_buckets"]),
        ),
        weekly=WeeklyConfig(
            min_games_for_sigma=int(weekly_raw["min_games_for_sigma"]),
            sigma_floor_ratio=float(weekly_raw["sigma_floor_ratio"]),
            kdst_cdf_grid=int(weekly_raw["kdst_cdf_grid"]),
        ),
        backtest=BacktestConfig(
            start_season=int(bt["start_season"]),
            end_season=int(bt["end_season"]),
            as_of_offset_days=int(bt.get("as_of_offset_days", 1)),
        ),
        slot_plan=build_slot_plan(
            slots, flex,
            # Slot names that are ALSO positions (QB, RB, ...) — never BENCH,
            # and never a flex slot name like FLEX/SUPERFLEX, which names a
            # slot rather than a position. Including flex names would put a
            # phantom position in the universe with cap 0.
            extra_positions=[
                p for p in slots if p != SLOT_BENCH and p not in flex
            ],
        ),
        league_hash=compute_hash(raw),
        raw=raw,
    )


@lru_cache(maxsize=8)
def load_league(path: str | None = None) -> LeagueConfig:
    cfg_path = Path(path) if path else DEFAULT_LEAGUE_PATH
    if not cfg_path.exists():
        raise ConfigError(f"league config not found at {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"league config at {cfg_path} is not a mapping")
    _validate(raw)
    return _parse(raw)
