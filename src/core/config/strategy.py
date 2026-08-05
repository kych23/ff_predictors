"""Typed loader for ``config/strategy.yaml`` (§10.3, §10.4, §10.5).

Nothing here changes a projection, so editing this file re-hashes
``decision_version`` only and leaves every trained model valid — no pipeline
re-run. That boundary is the whole point of the two-file split (§10.1).

``load_strategy`` takes a ``LeagueConfig`` because rules 9, 11, 13 and 14 need
``teams``, ``playoff_teams`` and ``regular_season_weeks``. There is exactly one
validation entry point; ``domain.payout.compile`` re-checks nothing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import yaml

from src.core.config.league import LeagueConfig
from src.core.config.payout_schema import (
    PRIZE_TYPES,
    RANK_TYPES,
    TIE_POLICIES,
    WEEKLY_TYPES,
    Amount,
    PrizeRule,
    WeekRange,
)
from src.core.errors import ConfigError, PayoutImbalanceError, UnknownPrizeTypeError

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_STRATEGY_PATH: Final = _REPO_ROOT / "config" / "strategy.yaml"

POT_TOLERANCE: Final = 0.01


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PayoutConfig:
    buy_in: float
    currency: str
    prizes: tuple[PrizeRule, ...]

    def pot(self, teams: int) -> float:
        return self.buy_in * teams


@dataclass(frozen=True)
class AdpConfig:
    source: str
    format: str
    teams: int
    min_ranking: int
    min_fullsim: int


@dataclass(frozen=True)
class ManagerConfig:
    id: str
    display_name: str
    policy: str
    seasons_present: tuple[int, ...]
    fitted: Mapping[str, Any] | None


@dataclass(frozen=True)
class GateConfig:
    require_permutation_p: float
    require_max_statistic_p: float
    require_pick_change_rate: float
    on_fail: str


@dataclass(frozen=True)
class OpponentsConfig:
    defaults: Mapping[str, Any]
    shrinkage: Mapping[str, Any]
    gate: GateConfig
    managers: tuple[ManagerConfig, ...]


@dataclass(frozen=True)
class SimulationConfig:
    inner_seasons: int
    outer_parameter_draws: int
    initial_draws_per_candidate: int
    correlation: Mapping[str, Any]
    rollout: Mapping[str, Any]
    waiver: Mapping[str, Any]
    decision: Mapping[str, Any]
    narration: Mapping[str, Any]


@dataclass(frozen=True)
class StrategyConfig:
    payout: PayoutConfig
    adp: AdpConfig
    roster_preferences: Mapping[str, Mapping[str, int]]
    yahoo: Mapping[str, Any]
    opponents: OpponentsConfig
    simulation: SimulationConfig
    requires_league_hash: str
    stale_strategy: bool = False
    strategy_hash: str = field(compare=False, default="")
    raw: Mapping[str, Any] = field(compare=False, repr=False, default_factory=dict)

    @property
    def decision_version(self) -> str:
        return f"dec_v2.{self.strategy_hash}"

    @property
    def max_per_position(self) -> Mapping[str, int]:
        return self.roster_preferences.get("max_per_position", {})


# --------------------------------------------------------------------------- #
# prize parsing (§10.5 rules 9-13)
# --------------------------------------------------------------------------- #
def _parse_prize(raw: Mapping[str, Any], league: LeagueConfig) -> PrizeRule:
    pid = str(raw.get("id", "<unnamed>"))
    where = f"payout.prizes[{pid}]"

    ptype = str(raw.get("type", ""))
    if ptype not in PRIZE_TYPES:  # rule 10
        raise UnknownPrizeTypeError(
            f"{where}: unknown prize type {ptype!r}; registered types are "
            f"{sorted(PRIZE_TYPES)}"
        )

    ties = str(raw.get("ties", "split"))
    if ties not in TIE_POLICIES:
        raise ConfigError(f"{where}: unknown ties policy {ties!r}")

    funded_external = str(raw.get("funded", "")) == "external"

    # rule 12 — `ties: all` pays every tied team in full, so it cannot
    # participate in a conserved pot.
    if ties == "all" and not funded_external:
        raise ConfigError(
            f"{where}: ties: all pays each tied team the full amount, so it "
            f"cannot conserve the pot. Mark the prize `funded: external` or "
            f"use ties: split."
        )

    rank = raw.get("rank")
    if ptype in RANK_TYPES:
        if rank is None:
            raise ConfigError(f"{where}: type {ptype!r} requires `rank`")
        rank = int(rank)
        # rule 11 — 1..teams, or -teams..-1 counting from last
        if not (1 <= abs(rank) <= league.teams) or rank == 0:
            raise ConfigError(
                f"{where}: rank {rank} outside 1..{league.teams} "
                f"(or -{league.teams}..-1 counting from last)"
            )

    weeks = None
    if ptype in WEEKLY_TYPES:
        wr = raw.get("weeks")
        if not isinstance(wr, Mapping) or "from" not in wr or "to" not in wr:
            raise ConfigError(f"{where}: type {ptype!r} requires weeks: {{from, to}}")
        weeks = WeekRange(int(wr["from"]), int(wr["to"]))
        # rule 13
        rsw = league.schedule.regular_season_weeks
        if not (1 <= weeks.start <= weeks.end <= rsw):
            raise ConfigError(
                f"{where}: weeks {weeks.start}-{weeks.end} outside 1..{rsw}"
            )

    return PrizeRule(
        id=pid,
        type=ptype,
        amount=Amount.from_raw(raw.get("amount"), where=where),
        ties=ties,  # type: ignore[arg-type]
        rank=rank,
        weeks=weeks,
        funded_external=funded_external,
    )


def _assert_pot_conserved(payout: PayoutConfig, teams: int) -> None:
    """Rule 9.

    Expected outlay is ``amount x n_occurrences`` per prize, summed over prizes
    not marked ``funded: external``. Holds under ``ties: split`` only, which
    rule 12 enforces at parse time.
    """
    pot = payout.pot(teams)
    total = sum(
        p.amount.resolve(pot) * p.n_occurrences()
        for p in payout.prizes
        if not p.funded_external
    )
    if abs(total - pot) > POT_TOLERANCE:
        raise PayoutImbalanceError(total, pot, payout.currency)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def compute_hash(raw: Mapping[str, Any]) -> str:
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _parse_managers(raw: Sequence[Mapping[str, Any]], league: LeagueConfig
                    ) -> tuple[ManagerConfig, ...]:
    managers = tuple(
        ManagerConfig(
            id=str(m["id"]),
            display_name=str(m.get("display_name", m["id"])),
            policy=str(m.get("policy", "softmax_adp")),
            seasons_present=tuple(int(s) for s in m["seasons_present"]),
            fitted=m.get("fitted"),
        )
        for m in raw
    )
    # rule 14
    expected = league.teams - 1
    if len(managers) != expected:
        raise ConfigError(
            f"opponents.managers has {len(managers)} entries, expected "
            f"teams - 1 = {expected}. My own seat is not an opponent."
        )
    ids = [m.id for m in managers]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ConfigError(f"duplicate manager ids: {dupes}")
    for m in managers:
        if not m.seasons_present:
            raise ConfigError(f"manager {m.id!r} has empty seasons_present")
    return managers


def load_strategy(league: LeagueConfig, path: str | None = None) -> StrategyConfig:
    cfg_path = Path(path) if path else DEFAULT_STRATEGY_PATH
    if not cfg_path.exists():
        raise ConfigError(f"strategy config not found at {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"strategy config at {cfg_path} is not a mapping")

    # rule 8 — declared, and a mismatch warns rather than errors
    if "requires_league_hash" not in raw:
        raise ConfigError("strategy config missing `requires_league_hash`")
    declared = str(raw["requires_league_hash"])
    stale = declared not in ("PENDING", league.league_hash)

    payout_raw = raw["payout"]
    payout = PayoutConfig(
        buy_in=float(payout_raw["buy_in"]),
        currency=str(payout_raw.get("currency", "USD")),
        prizes=tuple(_parse_prize(p, league) for p in payout_raw["prizes"]),
    )
    _assert_pot_conserved(payout, league.teams)

    adp_raw = raw["adp"]
    rounds = league.roster.rounds
    adp = AdpConfig(
        source=str(adp_raw["source"]),
        format=str(adp_raw["format"]),
        teams=int(adp_raw.get("teams") or league.teams),  # rule 16
        min_ranking=int(adp_raw["min_ranking"]),
        min_fullsim=int(adp_raw["min_fullsim"] or league.teams * rounds),
    )

    sim_raw = raw["simulation"]
    reps = sim_raw["replications"]
    inner = int(reps["inner_seasons"])
    outer = int(reps["outer_parameter_draws"])
    initial = int(reps["initial_draws_per_candidate"])
    if inner < 1 or outer < 1:  # rule 15
        raise ConfigError("inner_seasons and outer_parameter_draws must be >= 1")
    if initial > outer:
        raise ConfigError(
            f"initial_draws_per_candidate ({initial}) exceeds "
            f"outer_parameter_draws ({outer})"
        )

    gate_raw = raw["opponents"]["gate"]
    opponents = OpponentsConfig(
        defaults=raw["opponents"]["defaults"],
        shrinkage=raw["opponents"]["shrinkage"],
        gate=GateConfig(
            require_permutation_p=float(gate_raw["require_permutation_p"]),
            require_max_statistic_p=float(gate_raw["require_max_statistic_p"]),
            require_pick_change_rate=float(gate_raw["require_pick_change_rate"]),
            on_fail=str(gate_raw["on_fail"]),
        ),
        managers=_parse_managers(raw["opponents"]["managers"], league),
    )

    return StrategyConfig(
        payout=payout,
        adp=adp,
        roster_preferences=raw.get("roster_preferences", {}),
        yahoo=raw.get("yahoo", {}),
        opponents=opponents,
        simulation=SimulationConfig(
            inner_seasons=inner,
            outer_parameter_draws=outer,
            initial_draws_per_candidate=initial,
            correlation=sim_raw["correlation"],
            rollout=sim_raw["rollout"],
            waiver=sim_raw["waiver"],
            decision=sim_raw["decision"],
            narration=sim_raw["narration"],
        ),
        requires_league_hash=declared,
        stale_strategy=stale,
        strategy_hash=compute_hash(raw),
        raw=raw,
    )
