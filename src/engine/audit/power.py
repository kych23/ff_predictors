"""Power audit (§21.5) — runs BEFORE any gate, and decides which may claim.

A gate that cannot detect the effect it is looking for does not "pass"; it
declines to answer. Reporting such a gate as a pass is the most common way a
research project overstates itself, so this audit runs first and labels every
gate `claim` or `descriptive_only`.

The arithmetic is standard and small:

    n_eff  = n_nominal / (1 + (m - 1) * rho)          # design effect
    se     = sd / sqrt(n_eff)
    mde    = (z_{1-a/2} + z_{power}) * se             # at 80% power
    power  = Phi(|delta| / se - z_{1-a/2})            # at the expected effect

The design effect is the part that is usually skipped. Twelve seats drawn from
one simulated draft share a board, so 200 drafts is not 2,400 independent
observations — at rho = 0.35 it is closer to 420. Every gate here declares its
cluster size and intra-cluster correlation for exactly that reason.

Surviving gates additionally get a TOST equivalence bound, because "we found no
difference" and "we can rule out a difference larger than X" are different
statements and only the second is worth printing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.core.errors import ConfigError

DEFAULT_ASSUMPTIONS: Final = Path("config/power_assumptions.yaml")

#: Power at which the MDE is quoted. Conventional, and stated so nobody has to
#: guess which convention produced the number.
MDE_POWER: Final = 0.80

REQUIRED_KEYS: Final = frozenset({
    "id", "unit", "n_nominal", "cluster_size", "intra_cluster_rho",
    "expected_effect", "sd",
})


@dataclass(frozen=True)
class Gate:
    """One row of the audit's input."""

    id: str
    unit: str
    n_nominal: int
    cluster_size: int
    intra_cluster_rho: float
    expected_effect: float
    sd: float
    description: str = ""
    sd_provenance: str = "assumed"
    effect_provenance: str = "assumed"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_nominal <= 1:
            raise ConfigError(f"gate {self.id}: n_nominal must exceed 1")
        if self.cluster_size < 1:
            raise ConfigError(f"gate {self.id}: cluster_size must be >= 1")
        if not 0.0 <= self.intra_cluster_rho < 1.0:
            raise ConfigError(
                f"gate {self.id}: intra_cluster_rho must be in [0, 1), got "
                f"{self.intra_cluster_rho}. At rho = 1 every cluster collapses "
                f"to a single observation and n_eff is undefined."
            )
        if self.sd <= 0:
            raise ConfigError(f"gate {self.id}: sd must be positive")

    @property
    def design_effect(self) -> float:
        """1 + (m - 1) * rho — the Kish factor."""
        return 1.0 + (self.cluster_size - 1) * self.intra_cluster_rho

    @property
    def n_effective(self) -> float:
        return self.n_nominal / self.design_effect

    @property
    def standard_error(self) -> float:
        return self.sd / np.sqrt(self.n_effective)


@dataclass(frozen=True)
class GateVerdict:
    gate: Gate
    mde: float
    power: float
    verdict: str                      # "claim" | "descriptive_only"
    equivalence_bound: float | None   # TOST, None when descriptive

    @property
    def fully_assumed(self) -> bool:
        """Both inputs are guesses, so the verdict is a guess about a guess."""
        return (self.gate.sd_provenance == "assumed"
                and self.gate.effect_provenance == "assumed")


def load_assumptions(path: Path | None = None) -> tuple[list[Gate], dict]:
    path = path or DEFAULT_ASSUMPTIONS
    if not path.exists():
        raise ConfigError(f"no power assumptions at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not raw.get("gates"):
        raise ConfigError(f"{path} declares no gates")

    gates = []
    seen: set[str] = set()
    for entry in raw["gates"]:
        missing = REQUIRED_KEYS - set(entry)
        if missing:
            raise ConfigError(
                f"gate {entry.get('id', '<unnamed>')} missing {sorted(missing)}"
            )
        if entry["id"] in seen:
            raise ConfigError(f"duplicate gate id {entry['id']!r}")
        seen.add(entry["id"])
        gates.append(Gate(**{k: v for k, v in entry.items()
                             if k in Gate.__dataclass_fields__}))
    settings = {
        "alpha": float(raw.get("alpha", 0.05)),
        "power_floor": float(raw.get("power_floor", 0.50)),
        "equivalence_margin_frac": float(raw.get("equivalence_margin_frac", 0.5)),
    }
    if not 0.0 < settings["alpha"] < 1.0:
        raise ConfigError(f"alpha must be in (0, 1), got {settings['alpha']}")
    return gates, settings


def evaluate(gate: Gate, *, alpha: float = 0.05, power_floor: float = 0.50,
             equivalence_margin_frac: float = 0.5) -> GateVerdict:
    se = gate.standard_error
    z_alpha = norm.ppf(1 - alpha / 2)
    mde = (z_alpha + norm.ppf(MDE_POWER)) * se
    power = float(norm.cdf(abs(gate.expected_effect) / se - z_alpha))

    verdict = "claim" if power >= power_floor else "descriptive_only"
    # TOST bound: the smallest difference this gate could rule out, quoted only
    # where the gate has the power to rule out anything. Reporting an
    # equivalence bound for an underpowered gate is the same overstatement in
    # the opposite direction.
    bound = (z_alpha * se + equivalence_margin_frac * abs(gate.expected_effect)
             if verdict == "claim" else None)
    return GateVerdict(gate=gate, mde=mde, power=power, verdict=verdict,
                       equivalence_bound=bound)


def power_table(path: Path | None = None) -> pd.DataFrame:
    """gate, n_nominal, design_effect, n_effective, mde, power, verdict."""
    gates, settings = load_assumptions(path)
    rows = []
    for gate in gates:
        v = evaluate(gate, **settings)
        rows.append({
            "gate": gate.id,
            "unit": gate.unit,
            "n_nominal": gate.n_nominal,
            "design_effect": round(gate.design_effect, 3),
            "n_effective": round(gate.n_effective, 1),
            "expected_effect": gate.expected_effect,
            "se": round(gate.standard_error, 5),
            "mde": round(v.mde, 5),
            "power_at_expected_effect": round(v.power, 3),
            "verdict": v.verdict,
            "equivalence_bound": (round(v.equivalence_bound, 5)
                                  if v.equivalence_bound is not None else None),
            "inputs_assumed": v.fully_assumed,
        })
    return pd.DataFrame(rows)


def required_n(gate: Gate, *, alpha: float = 0.05,
               target_power: float = 0.80) -> int:
    """Nominal sample size this gate would need to reach `target_power`.

    Reported for every descriptive_only gate, because "underpowered" is a
    complaint and "underpowered, needs 3,400 drafts" is a plan.
    """
    z = norm.ppf(1 - alpha / 2) + norm.ppf(target_power)
    n_eff = (z * gate.sd / abs(gate.expected_effect)) ** 2
    return int(np.ceil(n_eff * gate.design_effect))
