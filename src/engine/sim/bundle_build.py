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

from src.core.constants import REPLACEMENT_FLOOR_QUANTILE
from src.engine.sim.draws import ProjectionBundle
from src.models.artifacts import newest
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
    return newest(pattern, root=Path(artifacts_dir))


#: Fallback half-width when a row has no calibrated band, as a multiple of the
#: player's own season rate. Chosen to match the median relative width of the
#: calibrated bands on the shipped bundle rather than picked by feel.
_FALLBACK_REL_HALF_WIDTH = 0.45


def _rate_band(df: pd.DataFrame, value: np.ndarray,
               sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """The season-rate interval the split-normal draws from.

    **This used to be synthesized and it was the wrong quantity.** The old line
    was ``value +/- 1.28 * sigma_weekly / sqrt(17)``, which is the standard
    error of a season MEAN under weekly sampling noise — how precisely we could
    measure this player's average if the rate were known. What the draw needs
    is the model's uncertainty about the rate itself, and the quantile model
    fits exactly that and calibrates it conformally. `build_bundle` renamed
    ``p50 -> value`` and dropped ``p10``/``p90``, so the calibrated band never
    reached the simulator at all.

    Measured over the 206 matched players on the live bundle:

    * median calibrated width 8.817 pts/g against 4.069 synthesized — the
      simulator's rate uncertainty was **2.30x too narrow**;
    * calibrated skew ``(p90-p50)-(p50-p10)`` spans -5.24 to +5.86 and exceeds
      0.5 in magnitude for **88.3%** of players, while the synthesized band is
      symmetric to 1.8e-15 for every one. `split_normal_ppf` therefore
      degenerated to a plain normal on every draw, discarding the asymmetry
      that is the entire reason for a split normal.

    Rows with no calibrated band — K/DST, and anyone `fill_unvalued` floored —
    keep a synthesized one, but scaled off the player's own rate rather than
    off weekly sigma, so it is at least the right order of magnitude.
    """
    have = ("value_p10" in df.columns and "value_p90" in df.columns)
    if have:
        p10 = pd.to_numeric(df["value_p10"], errors="coerce").to_numpy(dtype=float)
        p90 = pd.to_numeric(df["value_p90"], errors="coerce").to_numpy(dtype=float)
    else:
        p10 = np.full(len(df), np.nan)
        p90 = np.full(len(df), np.nan)

    usable = np.isfinite(p10) & np.isfinite(p90) & (p90 > p10)
    fallback_half = _FALLBACK_REL_HALF_WIDTH * np.abs(value)
    p10 = np.where(usable, p10, value - fallback_half)
    p90 = np.where(usable, p90, value + fallback_half)

    # The band must bracket the median it is a band AROUND. A row whose p50 was
    # replaced downstream — K/DST taking their fitted mean, or a floored
    # rookie — can otherwise end up with p50 outside [p10, p90], which
    # `split_normal_ppf` cannot represent.
    p10 = np.minimum(p10, value)
    p90 = np.maximum(p90, value)
    # Non-negativity NEVER at the expense of ordering. Clamping to zero last
    # can push p10 above a negative `value`: a [-5, -1] band around
    # ``value = -2`` came out ``p10 = 0 > p50 = -2``, which `split_normal_ppf`
    # cannot represent. Unreachable from the one production caller — the floor
    # loop above makes every `value` positive first — but the invariant is
    # this function's own, so it holds it itself rather than by arrangement.
    p10 = np.minimum(np.maximum(p10, 0.0), value)

    n = int(usable.sum())
    source = (f"calibrated for {n}/{len(df)} rows, "
              f"rate-scaled fallback for {len(df) - n}")
    return p10, p90, source


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


#: Memo for `build_projection_bundle`, keyed on everything that can change it.
#: Bounded to one entry: a draft uses a single board, and holding the previous
#: one would double the memory for no hit rate.
_BUNDLE_CACHE: dict[tuple, ProjectionBundle] = {}


def clear_bundle_cache() -> None:
    """Drop the memo. Used by tests and whenever the board is rebuilt."""
    _BUNDLE_CACHE.clear()


def build_projection_bundle(board: pd.DataFrame, cfg, weeks: int,
                            season: int = 2026, *,
                            covariate_loader: CovariateLoader = nflverse_covariates,
                            artifacts_dir: Path = ARTIFACTS
                            ) -> ProjectionBundle:
    """Map the draft board onto the simulator's array contract.

    **Memoized, and that is not an optimization — it is the difference between
    a usable pick clock and an unusable one.** `hazard_matrix` calls
    `nflverse_covariates`, which opens the network, and tier 0 rebuilt this on
    every recommendation. Measured on the live board: **61.85 s**, against a
    25 s allocator budget. The deadline was therefore already blown before the
    first candidate was evaluated — the allocator got 0.18 s, returned
    `stopped_because="deadline"`, and the recommendation came from the two
    initial draws per candidate rather than the fifty it is budgeted for.
    A 60-second wait bought almost no simulation.

    The inputs genuinely do not change during a draft. Tier 0 passes the FULL
    board, not the available subset, precisely so that players already drafted
    still contribute to their team's weekly totals — so the frame is identical
    at pick 1 and pick 180. The key covers the board identity, its size, the
    horizon and the loader, so a different board or a stubbed loader still
    rebuilds.
    """
    key = (
        # The loader OBJECT, not `id(...)`. CPython reuses ids after garbage
        # collection, so a stubbed loader in one test could be handed the
        # bundle built by a real one in another — which showed up as a
        # order-dependent failure that passed in isolation. Holding the
        # reference also keeps the identity stable for as long as it is a key.
        covariate_loader, str(artifacts_dir), int(weeks), int(season),
        len(board),
        # Board contents, not just length: a resolved name can change a row
        # in place without changing the row count.
        hash(tuple(board["player_id"].astype(str))),
    )
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        return cached
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
        floor = float(np.quantile(valued, REPLACEMENT_FLOOR_QUANTILE))
        missing = at_pos & (value <= 0)
        value[missing] = floor

    rate_p10, rate_p90, band_source = _rate_band(df, value, sigma)
    print(f"      rate band: {band_source}")

    # BOTH seams forwarded. `build_tiers` never calls `hazard_matrix` directly,
    # so omitting them here would leave a stubbed loader silently bypassed.
    hazard, hazard_source = hazard_matrix(
        df, cfg, weeks, season,
        covariate_loader=covariate_loader, artifacts_dir=artifacts_dir)
    print(f"      hazard: {hazard_source}")

    bundle = ProjectionBundle(
        player_ids=df["player_id"].to_numpy(),
        positions=df["position"].to_numpy(),
        nfl_teams=df["team"].fillna("FA").to_numpy(),
        slot_idx=np.array(slots),
        bye_weeks=pd.to_numeric(df["bye_week"], errors="coerce").fillna(0)
                    .astype(int).to_numpy(),
        is_kdst=is_kdst,
        rate_p10=rate_p10,
        rate_p50=value,
        rate_p90=rate_p90,
        weekly_sigma=sigma,
        games_hazard=hazard,
        kdst_quantiles=kdst_q,
    )
    _BUNDLE_CACHE.clear()          # one entry only
    _BUNDLE_CACHE[key] = bundle
    return bundle


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

    hit = _latest("correlation_posterior_*.npz", Path(artifacts_dir))
    if hit is None:
        return None
    with np.load(hit, allow_pickle=False) as data:
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
