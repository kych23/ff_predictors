"""K and DST weekly distributions (§12.5, §3.4).

Every team starts exactly one kicker and one defense, so their **expected**
contribution nearly cancels across teams. Variance does not cancel, and the
weekly-high-score prize pays for the maximum over twelve teams — which is why
these get empirical distributions rather than a constant. A constant cancels
correctly under a lineup-points objective and stops cancelling the moment a
weekly-maximum term exists.

**No conditioning.** Earlier revisions conditioned these on Week-1 implied
opponent points. Measured 2026-08-04, that carries almost no signal:
``corr(week-1 DST points, week-1 implied opponent points) = -0.028`` (n=429),
and binning season-mean DST points by implied-opponent quintile spreads only
0.96 points/game (5.95 -> 6.91) against a weekly SD of 5.63. Five bins
differing by under a point are not worth the leakage surface.

MEASURED marginals through league scoring: DST mean 6.44, SD 5.63 (n=7,054
team-weeks); K mean 8.05, SD 4.47 (n=7,357).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmpiricalDistribution:
    """An inverse-CDF grid, sampled by interpolation on a uniform."""

    position: str
    quantiles: np.ndarray      # (Q,) values at evenly spaced probabilities
    mean: float
    sd: float
    n_observations: int

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF. Linear interpolation between grid points.

        Inverse-CDF sampling (rather than a draw method) is what keeps CRN
        intact: replication r addresses the same uniform regardless of which
        candidate is being evaluated (§15.1).
        """
        probs = np.linspace(0.0, 1.0, len(self.quantiles))
        return np.interp(np.clip(u, 0.0, 1.0), probs, self.quantiles)

    def to_dict(self) -> dict:
        return {"position": self.position, "quantiles": self.quantiles.tolist(),
                "mean": self.mean, "sd": self.sd,
                "n_observations": self.n_observations}

    @classmethod
    def from_dict(cls, raw: dict) -> EmpiricalDistribution:
        return cls(position=str(raw["position"]),
                   quantiles=np.asarray(raw["quantiles"], dtype="float64"),
                   mean=float(raw["mean"]), sd=float(raw["sd"]),
                   n_observations=int(raw["n_observations"]))


@dataclass(frozen=True)
class KDSTModel:
    distributions: dict[str, EmpiricalDistribution]
    snapshot_id: str = ""
    model_version: str = ""

    def ppf(self, position: str, u: np.ndarray) -> np.ndarray:
        if position not in self.distributions:
            raise KeyError(
                f"no empirical distribution for {position!r}; have "
                f"{sorted(self.distributions)}"
            )
        return self.distributions[position].ppf(u)

    def to_dict(self) -> dict:
        return {
            "distributions": {k: v.to_dict()
                              for k, v in self.distributions.items()},
            "snapshot_id": self.snapshot_id, "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> KDSTModel:
        return cls(
            distributions={k: EmpiricalDistribution.from_dict(v)
                           for k, v in raw["distributions"].items()},
            snapshot_id=str(raw.get("snapshot_id", "")),
            model_version=str(raw.get("model_version", "")),
        )


def fit_empirical(
    points: pd.Series, position: str, *, grid: int = 101
) -> EmpiricalDistribution:
    values = pd.to_numeric(points, errors="coerce").dropna().to_numpy()
    if len(values) < 100:
        raise ValueError(
            f"only {len(values)} observations for {position}; an empirical "
            f"distribution on that few is mostly noise"
        )
    probs = np.linspace(0.0, 1.0, grid)
    return EmpiricalDistribution(
        position=position, quantiles=np.quantile(values, probs),
        mean=float(values.mean()), sd=float(values.std(ddof=1)),
        n_observations=len(values),
    )
