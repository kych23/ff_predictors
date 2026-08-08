"""Weekly availability hazard (§12.4).

Replaces a hardcoded 0.93 applied to every player — a 34-year-old running back
and a 24-year-old receiver were treated identically, and the simulator consumed
that on every draw.

    P(active in week w) ~ logistic(position, age bucket, prior-season games
                                   missed, week index)

**Per-game rates are availability-neutral by construction** (§14), so games
missed belong here rather than in the projection. That separation is the whole
reason this model exists: `rate_p50` says how good a player is when he plays,
and the hazard says how often that happens.

The week index is the only week-varying covariate, so rows are near-flat by
design. That is intended: per-week injury news does not exist at draft time,
and pretending otherwise would leak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Age boundaries at Sept 1 of the target season.
AGE_BUCKETS = ((0, 24), (25, 27), (28, 30), (31, 99))

#: Prior-season games-missed buckets.
MISSED_BUCKETS = ((0, 0), (1, 3), (4, 8), (9, 99))


#: nflverse's players table keys on `gsis_id`; player_stats keys on
#: `player_id`. They are the same identifier under two names.
_ID_COLUMNS = ("player_id", "gsis_id")


def _ages(players: pd.DataFrame) -> pd.DataFrame | None:
    """(player_id, birth_year) from whichever id column the frame carries."""
    if "birth_date" not in players.columns:
        return None
    id_col = next((c for c in _ID_COLUMNS if c in players.columns), None)
    if id_col is None:
        return None
    out = players[[id_col, "birth_date"]].rename(columns={id_col: "player_id"})
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(str)
    out["birth_year"] = pd.to_datetime(out["birth_date"], errors="coerce").dt.year
    return out[["player_id", "birth_year"]].drop_duplicates("player_id")


def _bucket(value: float, buckets) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value <= hi:
            return i
    return len(buckets) - 1


@dataclass(frozen=True)
class HazardModel:
    """Logistic availability model. Emits a (P, W) matrix."""

    coef: np.ndarray
    positions: tuple[str, ...]
    base_rate: float
    n_observations: int
    snapshot_id: str = ""
    model_version: str = ""
    _floor: float = field(default=0.35, repr=False)
    _ceil: float = field(default=0.995, repr=False)

    def _design(self, position, age, missed, week) -> np.ndarray:
        n = len(position)
        cols = [np.ones(n)]
        for p in self.positions[1:]:                     # first is reference
            cols.append((np.asarray(position) == p).astype(float))
        for i in range(1, len(AGE_BUCKETS)):
            cols.append(np.array([_bucket(a, AGE_BUCKETS) == i for a in age],
                                 dtype=float))
        for i in range(1, len(MISSED_BUCKETS)):
            cols.append(np.array([_bucket(m, MISSED_BUCKETS) == i for m in missed],
                                 dtype=float))
        cols.append(np.asarray(week, dtype=float))
        return np.column_stack(cols)

    def predict(self, position, age, missed, week) -> np.ndarray:
        z = self._design(position, age, missed, week) @ self.coef
        p = 1.0 / (1.0 + np.exp(-z))
        return np.clip(p, self._floor, self._ceil)

    def matrix(self, position, age, missed, *, weeks: int) -> np.ndarray:
        """(P, W) availability for a whole roster across a season."""
        n = len(position)
        out = np.empty((n, weeks))
        for w in range(weeks):
            out[:, w] = self.predict(position, age, missed, np.full(n, w + 1))
        return out

    def to_dict(self) -> dict:
        return {"coef": self.coef.tolist(), "positions": list(self.positions),
                "base_rate": self.base_rate, "n_observations": self.n_observations,
                "snapshot_id": self.snapshot_id, "model_version": self.model_version}

    @classmethod
    def from_dict(cls, raw: dict) -> HazardModel:
        return cls(coef=np.asarray(raw["coef"], dtype=float),
                   positions=tuple(raw["positions"]),
                   base_rate=float(raw["base_rate"]),
                   n_observations=int(raw["n_observations"]),
                   snapshot_id=str(raw.get("snapshot_id", "")),
                   model_version=str(raw.get("model_version", "")))


#: Games in season-1 required to enter the fitting panel. See the docstring —
#: this makes the fitted population resemble the population the model is
#: applied to, which is a draft board.
MIN_PRIOR_GAMES = 6

#: Prior-season finish required at each position, roughly the depth a 12-team
#: league actually drafts. Together with MIN_PRIOR_GAMES this is the second
#: half of the population match: the board is not "everyone with a stat line",
#: it is the top few hundred, and those players are available far more often
#: than the fringe.
DRAFTABLE_DEPTH = {"QB": 32, "RB": 72, "WR": 84, "TE": 32, "K": 32, "DST": 32}


def build_hazard_panel(weekly: pd.DataFrame, players: pd.DataFrame,
                       positions: tuple[str, ...], *,
                       min_prior_games: int = MIN_PRIOR_GAMES,
                       depth: dict[str, int] | None = None,
                       value_col: str = "points") -> pd.DataFrame:
    """One row per (player, season, non-bye week): did he record a game?

    `player_stats` carries a row only when a player appeared, so absences are
    gaps and the label has to be built by reindexing. Two things about that
    reindex are load-bearing, and getting either wrong inverts the model:

    **Span the whole season, not first-to-last appearance.** A back who tears
    an ACL in week 3 appears in weeks 1-3 and never again. Spanning his
    appearances scores him 3-for-3 — perfectly available — and deletes exactly
    the event the hazard exists to price. Every included player-season is
    therefore spanned across the full schedule.

    **Fit on the population the model is applied to.** `player_stats` is mostly
    fringe: practice-squad elevations and healthy scratches who appear four
    times a year. Fitting on all of them gives a base rate near 0.75 and an age
    coefficient with the wrong sign, because a 33-year-old still in the league
    is a survivor while a 23-year-old is disproportionately a roster bubble
    player. The panel is therefore restricted to player-seasons whose prior
    season cleared `min_prior_games` AND finished inside `depth` at the
    position — which is what a draft board is. Both conditions are evaluable
    as-of before week 1, so the predict path can reproduce them.

    The depth filter selects on prior-season points, which mildly favors
    players who stayed healthy last year. That is precisely what the
    `missed_prior` covariate absorbs, so the bias is controlled rather than
    ignored.

    Byes are dropped rather than labeled inactive: the simulator zeroes a
    player's bye week separately (`draws.draw_points`), so leaving them here
    would charge the same absence twice.
    """
    df = weekly[(weekly["season_type"] == "REG")
                & weekly["position"].isin(positions)].copy()
    df["player_id"] = df["player_id"].astype(str)
    appeared = df[["player_id", "season", "week", "position", "team"]] \
        .drop_duplicates(subset=["player_id", "season", "week"])
    appeared = appeared.assign(active=1)

    # Weeks in each season, and each NFL team's bye: the week in which the team
    # fielded nobody. Derived from the same frame rather than the schedule so
    # this function needs exactly one input.
    season_weeks = appeared.groupby("season")["week"].max().to_dict()
    team_weeks = set(zip(appeared["season"], appeared["team"], appeared["week"], strict=False))

    played = (appeared.groupby(["player_id", "season"], as_index=False)["active"]
                      .sum().rename(columns={"active": "games"}))
    eligible = played[played["games"] >= min_prior_games].copy()

    if value_col in df.columns:
        depth = DRAFTABLE_DEPTH if depth is None else depth
        totals = (df.groupby(["player_id", "season", "position"], as_index=False)
                    [value_col].sum())
        totals["finish"] = (totals.groupby(["season", "position"])[value_col]
                                  .rank(ascending=False, method="first"))
        keep = totals[totals["finish"]
                      <= totals["position"].map(depth).fillna(9999)]
        eligible = eligible.merge(keep[["player_id", "season"]],
                                  on=["player_id", "season"], how="inner")

    eligible["missed_prior"] = (
        eligible["season"].map(season_weeks).fillna(17) - 1 - eligible["games"]
    ).clip(0, 17)
    eligible["season"] = eligible["season"] + 1     # applies to the NEXT season

    # A player's primary team and position, for the bye lookup. Resolved
    # against the TARGET season where he appears at all, and against the prior
    # season otherwise — because a player who never appears must still enter
    # the panel. Requiring a target-season row is survivorship bias with a
    # direct cost: it silently deletes every player who was cut, retired or
    # spent the year on IR, which are exactly the outcomes a drafter is
    # exposed to. Measured, that inflated the fitted availability of a
    # top-12 RB from the observed 0.73 to 0.83.
    primary = (appeared.groupby(["player_id", "season"])
                       .agg(team=("team", lambda s: s.mode().iat[0]),
                            position=("position", lambda s: s.mode().iat[0]))
                       .reset_index())
    fallback = primary.copy()
    fallback["season"] = fallback["season"] + 1     # last known team/position

    target = eligible[["player_id", "season", "missed_prior"]].merge(
        primary, on=["player_id", "season"], how="left")
    target = target.merge(fallback, on=["player_id", "season"], how="left",
                          suffixes=("", "_prev"))
    target["team"] = target["team"].fillna(target["team_prev"])
    target["position"] = target["position"].fillna(target["position_prev"])
    target = target.dropna(subset=["team", "position"])

    rows = []
    for r in target.itertuples(index=False):
        last = int(season_weeks.get(r.season, 17))
        for w in range(1, last + 1):
            if (r.season, r.team, w) not in team_weeks:
                continue                            # team bye — handled elsewhere
            rows.append((r.player_id, r.season, r.position, w, r.missed_prior))
    panel = pd.DataFrame(rows, columns=["player_id", "season", "position",
                                        "week", "missed_prior"])
    panel = panel.merge(appeared[["player_id", "season", "week", "active"]],
                        on=["player_id", "season", "week"], how="left")
    panel["active"] = panel["active"].fillna(0).astype(int)

    ages = _ages(players)
    if ages is None:
        panel["age"] = 26.0
    else:
        panel = panel.merge(ages, on="player_id", how="left")
        panel["age"] = (panel["season"] - panel["birth_year"]).fillna(26.0)
    return panel


def player_covariates(player_ids, positions, prior_stats: pd.DataFrame,
                      players: pd.DataFrame, season: int) -> pd.DataFrame:
    """Age and prior-season games missed for a draft board.

    Every covariate is as-of before week 1 of `season`: age is a birth date and
    games missed comes from `season - 1`. Nothing here can see the target
    season, so this is safe to call at draft time.
    """
    ids = pd.Series(list(player_ids), dtype=str)
    out = pd.DataFrame({"player_id": ids,
                        "position": list(positions)})

    prior = prior_stats[prior_stats["season_type"] == "REG"]
    last_week = int(pd.to_numeric(prior["week"], errors="coerce").max() or 17)
    played = (prior.groupby("player_id")["week"].nunique()
                   .rename("games_prior").reset_index())
    played["player_id"] = played["player_id"].astype(str)
    out = out.merge(played, on="player_id", how="left")
    # Same definition the panel uses: schedule weeks minus one bye minus games.
    missed = (last_week - 1 - out["games_prior"]).clip(0, 17)
    # A player with no prior-season row is a rookie (or was out all year).
    # Charging him 17 games missed would make every rookie the least available
    # player on the board, which is the opposite of true; he gets the median of
    # players who did appear.
    out["missed_prior"] = missed.fillna(missed.median() if missed.notna().any()
                                        else 2.0)

    ages = _ages(players)
    if ages is None:
        out["age"] = 26.0
    else:
        out = out.merge(ages, on="player_id", how="left")
        out["age"] = (season - out["birth_year"]).fillna(26.0)
    return out[["player_id", "position", "age", "missed_prior"]]


def fit_hazard(panel: pd.DataFrame, positions: tuple[str, ...], *,
               snapshot_id: str = "", model_version: str = "",
               max_iter: int = 60) -> HazardModel:
    """Newton-Raphson logistic fit. numpy only — no new dependency."""
    if len(panel) < 500:
        raise ValueError(f"only {len(panel)} player-weeks; refusing to fit")

    stub = HazardModel(coef=np.zeros(1), positions=positions, base_rate=0.0,
                       n_observations=0)
    X = stub._design(panel["position"].to_numpy(), panel["age"].to_numpy(),
                     panel["missed_prior"].to_numpy(), panel["week"].to_numpy())
    y = panel["active"].to_numpy(dtype=float)

    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = np.clip(p * (1 - p), 1e-8, None)
        grad = X.T @ (y - p)
        hess = (X * w[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break

    return HazardModel(coef=beta, positions=positions, base_rate=float(y.mean()),
                       n_observations=len(panel), snapshot_id=snapshot_id,
                       model_version=model_version)
