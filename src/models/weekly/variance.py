"""Within-season weekly variance (§12.3, component C4).

**Nothing in v1 estimates this**, and it is the quantity both prize terms are
functions of. The projection model's P10/P90 describe uncertainty about a
season-average *rate*; weekly sigma is a different object entirely, and
conflating them made season-total intervals about sqrt(17) too narrow while
supplying no week-to-week variance at all.

Model: ``log(sigma) ~ 1 + log(mu) + C(position)``, fitted on player-seasons
with at least ``min_games_for_sigma`` games, with a Duan smearing correction
and a floor at ``sigma_floor_ratio * mu``.

MEASURED 2026-08-04 on 2012-2025, draftable universe:

=========  ==============  ===================
position   sigma (mean)    sigma/mu (median)
=========  ==============  ===================
QB         7.45            0.376
RB         7.27            0.584
WR         7.38            0.619
TE         6.04            0.696
=========  ==============  ===================

The configured floor of 0.35 therefore binds only for the lowest-variance
quarterbacks, which is what a floor is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.core.config.league import LeagueConfig

#: Player-seasons below this mean are dropped: log(mu) is undefined at zero and
#: near-zero producers carry no information about the sigma-mu relationship.
MIN_MU = 0.5


@dataclass(frozen=True)
class WeeklySigmaModel:
    """Predicts a player's weekly SD from his projected per-game mean."""

    intercept: float
    log_mu_coef: float
    position_effects: dict[str, float]
    smearing: float
    floor_ratio: float
    n_observations: int
    snapshot_id: str = ""
    model_version: str = ""
    positions: tuple[str, ...] = field(default=())

    def predict(self, position, mu) -> np.ndarray:
        """Weekly SD for each (position, mu) pair."""
        position = np.asarray(position, dtype=object)
        mu = np.asarray(mu, dtype="float64")
        safe_mu = np.clip(mu, MIN_MU, None)
        effect = np.array(
            [self.position_effects.get(str(p), 0.0) for p in position.ravel()]
        ).reshape(position.shape)
        log_sigma = self.intercept + self.log_mu_coef * np.log(safe_mu) + effect
        # Duan smearing: exp(fitted log) estimates the conditional MEDIAN, not
        # the mean, and the simulator wants the mean.
        sigma = np.exp(log_sigma) * self.smearing
        return np.maximum(sigma, self.floor_ratio * np.clip(mu, 0.0, None))

    def to_dict(self) -> dict:
        return {
            "intercept": self.intercept, "log_mu_coef": self.log_mu_coef,
            "position_effects": self.position_effects, "smearing": self.smearing,
            "floor_ratio": self.floor_ratio, "n_observations": self.n_observations,
            "snapshot_id": self.snapshot_id, "model_version": self.model_version,
            "positions": list(self.positions),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "WeeklySigmaModel":
        return cls(
            intercept=float(raw["intercept"]),
            log_mu_coef=float(raw["log_mu_coef"]),
            position_effects={str(k): float(v)
                              for k, v in raw["position_effects"].items()},
            smearing=float(raw["smearing"]),
            floor_ratio=float(raw["floor_ratio"]),
            n_observations=int(raw["n_observations"]),
            snapshot_id=str(raw.get("snapshot_id", "")),
            model_version=str(raw.get("model_version", "")),
            positions=tuple(raw.get("positions", ())),
        )


def fit_weekly_sigma(
    summary: pd.DataFrame, cfg: LeagueConfig, *, snapshot_id: str = "",
    model_version: str = "",
) -> WeeklySigmaModel:
    """OLS of log(sigma) on log(mu) plus position dummies.

    ``summary`` comes from ``panel.player_season_summary`` and carries
    ``mu``, ``sigma``, ``position``.
    """
    df = summary[(summary["mu"] > MIN_MU) & (summary["sigma"] > 0)].copy()
    if len(df) < 50:
        raise ValueError(
            f"only {len(df)} usable player-seasons for the sigma fit; "
            f"refusing to fit a relationship on that"
        )

    positions = tuple(sorted(df["position"].unique()))
    # first position is the reference level, absorbed into the intercept
    reference, others = positions[0], positions[1:]
    design = [np.ones(len(df)), np.log(df["mu"].to_numpy())]
    design += [(df["position"] == p).to_numpy(dtype="float64") for p in others]
    X = np.column_stack(design)
    y = np.log(df["sigma"].to_numpy())

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    smearing = float(np.mean(np.exp(residuals)))

    effects = {reference: 0.0}
    effects.update({p: float(beta[2 + i]) for i, p in enumerate(others)})

    return WeeklySigmaModel(
        intercept=float(beta[0]), log_mu_coef=float(beta[1]),
        position_effects=effects, smearing=smearing,
        floor_ratio=float(cfg.weekly.sigma_floor_ratio),
        n_observations=len(df), snapshot_id=snapshot_id,
        model_version=model_version, positions=positions,
    )
