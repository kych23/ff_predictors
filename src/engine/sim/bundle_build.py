"""Draft board -> ``ProjectionBundle`` (§15.2).

MOVED from ``scripts/simulate.py``. It lives here because a web backend cannot
import from ``scripts/`` — that directory is not a package, and the scripts
reach each other through a ``sys.path`` insert. `engine` is the right layer:
the output type ``ProjectionBundle`` is ``engine.sim.draws``'s array contract,
and `engine` (rank 4) may import `models` (3) and `platform` (2) freely.

**Two seams, both defaulting to today's behaviour.**

``covariate_loader`` is the only code that touches the network. Moving
``hazard_matrix`` here put a ``platform.sources`` import inside `engine` for the
first time — legal, but deliberate, so the dependency is a parameter rather
than a hard-wired call. Note this is also the honest correction to a claim the
project used to make: the recommendation path has ALWAYS opened sockets here,
because the fitted-hazard branch fetches nflverse covariates on every run.

``artifacts_dir`` exists because ``covariate_loader`` alone does not decouple
this module from disk: the sigma, K/DST, hazard and correlation artifacts are
all read from a module-global path. Both are needed before a test can run
without ``data/artifacts``.

Paths are anchored to the REPO ROOT, not the cwd. The entry point is now a
long-lived server that may be launched from anywhere; the previous
cwd-relative globs would silently fall back to the assumed correlation prior —
the exact failure ``load_correlation_matrix``'s own docstring warns about.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.engine.sim.draws import ProjectionBundle
from src.models.correlation.slot_matrix import DEFAULT_SLOTS, from_prior
from src.models.weekly.hazard import HazardModel, player_covariates
from src.models.weekly.variance import WeeklySigmaModel

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default artifact directory, anchored to the repo root.
ARTIFACTS: Path = _REPO_ROOT / "data" / "artifacts"

#: Checked-in correlation prior, used only when no fitted artifact exists.
CORRELATION_PRIOR: Path = _REPO_ROOT / "config" / "correlation_prior.yaml"

#: depth-chart rank -> correlation slot, by position
SLOT_FOR = {"QB": ["QB1"], "RB": ["RB1", "RB2"],
            "WR": ["WR1", "WR2", "WR3"], "TE": ["TE1"]}


class CovariateLoader(Protocol):
    """Returns the hazard model's inputs: ``position``, ``age``, ``missed_prior``."""

    def __call__(self, player_ids: pd.Series, positions: pd.Series,
                 season: int) -> pd.DataFrame: ...


def _latest(pattern: str, artifacts_dir: Path):
    hits = sorted(Path(artifacts_dir).glob(pattern))
    return hits[-1] if hits else None


def nflverse_covariates(player_ids: pd.Series, positions: pd.Series,
                        season: int) -> pd.DataFrame:
    """The default loader. THE ONLY function here that opens a socket."""
    from src.platform.sources import nflverse

    prior = nflverse.fetch("player_stats", seasons=[season - 1]).frame
    try:
        players = nflverse.fetch("players").frame
    except Exception:                              # noqa: BLE001 — optional
        players = pd.DataFrame(columns=["player_id"])
    return player_covariates(player_ids, positions, prior, players, season)


def load_correlation_matrix(*, artifacts_dir: Path = ARTIFACTS):
    """The FITTED slot matrix, falling back to the checked-in prior.

    The YAML prior exists so the simulator runs before anything has been
    fitted. Silently preferring it once an artifact is on disk would mean
    measuring assumed correlations instead of real ones.
    """
    import yaml

    path = _latest("correlation_*.npz", artifacts_dir)
    if path is not None and "posterior" not in path.name:
        from src.models.artifacts import load_correlation
        return load_correlation(path.stem.split("_", 1)[1])
    return from_prior(yaml.safe_load(CORRELATION_PRIOR.read_text()))


def hazard_matrix(board: pd.DataFrame, cfg, weeks: int, season: int, *,
                  covariate_loader: CovariateLoader = nflverse_covariates,
                  artifacts_dir: Path = ARTIFACTS
                  ) -> tuple[np.ndarray, str]:
    """(P, W) availability, from the fitted §12.4 model where one exists.

    The fallback is a flat constant and is REPORTED as such. A flat hazard is
    not a neutral default — it prices a 34-year-old back and a 24-year-old
    receiver identically — so a run that uses it should say so out loud rather
    than let the number pass for a model output.
    """
    path = _latest("hazard_*.json", artifacts_dir)
    n = len(board)
    if path is None:
        return np.full((n, weeks), 0.93), "flat 0.93 (no hazard artifact)"

    model = HazardModel.from_dict(json.loads(path.read_text()))
    cov = covariate_loader(board["player_id"], board["position"], season)
    mat = model.matrix(cov["position"].to_numpy(), cov["age"].to_numpy(),
                       cov["missed_prior"].to_numpy(), weeks=weeks)
    # K and DST are outside the modeled positions the hazard was fitted on, and
    # both are streamed weekly rather than held through an injury. Their
    # empirical distributions already include zero weeks, so applying a hazard
    # on top would double-count the absence.
    mat[board["position"].isin(["K", "DST"]).to_numpy()] = 1.0
    return mat, f"fitted ({path.name}), range {mat.min():.3f}-{mat.max():.3f}"


def build_projection_bundle(board: pd.DataFrame, cfg, weeks: int,
                            season: int = 2026, *,
                            covariate_loader: CovariateLoader = nflverse_covariates,
                            artifacts_dir: Path = ARTIFACTS
                            ) -> ProjectionBundle:
    """Map the draft board onto the simulator's array contract."""
    sigma_path = _latest("weekly_sigma_*.json", artifacts_dir)
    if sigma_path is None:
        raise FileNotFoundError(
            f"no weekly-sigma artifact in {artifacts_dir}; "
            f"run scripts/fit_models.py")
    sigma_model = WeeklySigmaModel.from_dict(json.loads(sigma_path.read_text()))

    df = board.reset_index(drop=True).copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    # slot identity within NFL team, by projected value — the unit the
    # correlation matrix is estimated over
    df["rank_in_team"] = (df.groupby(["team", "position"])["value"]
                            .rank(ascending=False, method="first"))
    slot_index = {s: i for i, s in enumerate(DEFAULT_SLOTS)}
    slots = []
    for _, row in df.iterrows():
        names = SLOT_FOR.get(row["position"], [])
        k = int(row["rank_in_team"]) - 1
        slots.append(slot_index[names[k]] if 0 <= k < len(names) else -1)

    is_kdst = df["position"].isin(["K", "DST"]).to_numpy()
    value = df["value"].to_numpy()
    sigma = sigma_model.predict(df["position"].to_numpy(), value)
    sigma = np.where(is_kdst, 0.0, sigma)

    n = len(df)
    kdst_q = np.tile(np.full(101, np.nan), (n, 1))
    kdst_path = _latest("kdst_*.json", artifacts_dir)
    if kdst_path is not None:
        kdst = json.loads(kdst_path.read_text())["distributions"]
        for i, pos in enumerate(df["position"]):
            if pos in kdst:
                kdst_q[i] = np.asarray(kdst[pos]["quantiles"])
        # K/DST arrive from the board with value 0 when the bundle predates
        # `fill_unvalued`. Leaving that zero is the failure
        # BayesianArchitecture.md 3.4 warns about ("never assign them a constant
        # zero"): greedy_lineup requires a positive selection value, so every
        # team would field 7 starters instead of 9 and every team score would be
        # understated. Give them their fitted empirical MEAN as the selection
        # value; the draws still come from the full empirical distribution, so
        # the variance that actually matters for the weekly-high term is intact.
        for i, pos in enumerate(df["position"]):
            if pos in kdst and value[i] == 0:
                value[i] = float(kdst[pos]["mean"])

    # Same failure mode for rookies and anyone else the identity join could not
    # value: a zero selection value means greedy_lineup will never start them,
    # so a roster carrying one silently fields a short lineup. §19 specifies
    # replacement level plus a flag, not zero. Replacement here is the 10th
    # percentile of *valued* players at that position — low enough that a real
    # projection always outranks it, positive enough to be startable.
    for pos in df["position"].unique():
        at_pos = (df["position"] == pos).to_numpy()
        valued = value[at_pos & (value > 0)]
        if valued.size == 0:
            continue
        floor = float(np.quantile(valued, 0.10))
        missing = at_pos & (value <= 0)
        value[missing] = floor

    # BOTH seams forwarded. `build_tiers` never calls `hazard_matrix` directly,
    # so omitting them here would leave a stubbed loader silently bypassed.
    hazard, hazard_source = hazard_matrix(
        df, cfg, weeks, season,
        covariate_loader=covariate_loader, artifacts_dir=artifacts_dir)
    print(f"      hazard: {hazard_source}")

    return ProjectionBundle(
        player_ids=df["player_id"].to_numpy(),
        positions=df["position"].to_numpy(),
        nfl_teams=df["team"].fillna("FA").to_numpy(),
        slot_idx=np.array(slots),
        bye_weeks=pd.to_numeric(df["bye_week"], errors="coerce").fillna(0)
                    .astype(int).to_numpy(),
        is_kdst=is_kdst,
        rate_p10=np.maximum(value - 1.28 * sigma / np.sqrt(17), 0.0),
        rate_p50=value,
        rate_p90=value + 1.28 * sigma / np.sqrt(17),
        weekly_sigma=sigma,
        games_hazard=hazard,
        kdst_quantiles=kdst_q,
    )


def draft_rosters(board: pd.DataFrame, cfg, rng) -> list[np.ndarray]:
    """A plausible 12-team board: snake by ADP, which is the tier-2 opponent
    model the §14.2 gate left us with."""
    order = board.sort_values("adp").index.to_numpy()
    rosters: list[list] = [[] for _ in range(cfg.teams)]
    pick = 0
    for rnd in range(cfg.roster.rounds):
        seats = range(cfg.teams) if rnd % 2 == 0 else reversed(range(cfg.teams))
        for seat in seats:
            if pick < len(order):
                rosters[seat].append(order[pick])
                pick += 1
    return [np.array(r) for r in rosters]


def load_posterior(*, artifacts_dir: Path = ARTIFACTS):
    """Bootstrap correlation posterior, if one has been fitted.

    Absent it the kernel still runs on the point estimate — but epistemic_se
    then reflects only rollout variation, and the recommendation says so via a
    stale flag rather than quietly reporting a number that means something
    else.

    Selected by glob order, which does not necessarily match the bundle's
    snapshot. That is a pre-existing defect, preserved verbatim by this move so
    the refactor changes nothing observable; it is recorded as a follow-up in
    notes/draft-cockpit-web.md.
    """
    from src.models.correlation import slot_matrix as sm

    hits = sorted(Path(artifacts_dir).glob("correlation_posterior_*.npz"))
    if not hits:
        return None
    with np.load(hits[-1], allow_pickle=False) as data:
        slots = tuple(json.loads(str(data["slots"])))
        mats = data["matrices"]
    draws = tuple(
        sm.SlotCorrelation(
            slots=slots, matrix=m, cholesky=np.linalg.cholesky(m),
            pair_counts=np.zeros((len(slots), len(slots)), dtype=int),
            min_eigenvalue=float(np.linalg.eigvalsh(m).min()),
            was_projected=False, source="bootstrap",
        )
        for m in mats
    )
    return sm.CorrelationPosterior(point=draws[0], draws=draws)
