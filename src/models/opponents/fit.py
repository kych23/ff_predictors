"""Per-manager draft policy: conditional logit with shrinkage (§14.1, §14.2).

Score for candidate *i* when manager *m* is on the clock at overall pick *k*:

    score_i = -(adp_i - k) / tau_m
              + beta_m[position_i]
              + slot_coef_m[position_i] * slot_z

    p_i = softmax(score) over the roster-legal available set

``tau`` is in pick units and divides: larger means flatter, i.e. more reaching.
It is optimized as ``log tau`` so positivity is structural.

``beta`` and ``slot_coef`` are sum-to-zero across modeled positions, imposed
**at fit time** through free contrasts rather than by re-centering afterwards —
otherwise the likelihood carries a flat direction per manager.

``slot_coef`` is a **position interaction**, not a scalar. A scalar multiple of
``slot_z`` is constant across every candidate in a choice set and cancels
identically in the softmax, leaving its likelihood exactly flat. As an
interaction it also does the job it exists for: purging the positional-lean
estimate of the confound that a manager who picks first looks RB-heavy because
the board hands him running backs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

#: Choice sets are truncated to the best available by ADP. Picks far outside
#: the market are vanishingly unlikely and each extra candidate costs
#: likelihood time; the chosen player is always included regardless of rank.
CHOICE_SET_SIZE = 60


@dataclass(frozen=True)
class Choice:
    """One observed pick, as a discrete-choice observation."""

    manager: str
    season: int
    overall: int
    slot_z: float
    adp: np.ndarray          # (C,) candidate ADPs
    position_idx: np.ndarray  # (C,) index into `positions`
    chosen: int              # index of the taken player within the set


@dataclass(frozen=True)
class FittedManager:
    manager: str
    tau: float
    beta: dict[str, float]
    slot_coef: dict[str, float]
    n_choices: int
    slot_varies: bool

    def to_dict(self) -> dict:
        return {"tau": self.tau, "beta": self.beta, "slot_coef": self.slot_coef,
                "n_choices": self.n_choices, "slot_varies": self.slot_varies}


@dataclass(frozen=True)
class FitResult:
    managers: dict[str, FittedManager]
    log_likelihood: float
    positions: tuple[str, ...]

    def loglik_of(self, choices: list[Choice]) -> float:
        return _total_loglik(self, choices)


def _contrast_matrix(n_positions: int) -> np.ndarray:
    """(n_positions, n_positions-1): free contrasts -> sum-to-zero effects."""
    identity = np.eye(n_positions - 1)
    last = -np.ones((1, n_positions - 1))
    return np.vstack([identity, last])


def _unpack(theta: np.ndarray, n_managers: int, n_positions: int):
    per = 1 + 2 * (n_positions - 1)
    theta = theta.reshape(n_managers, per)
    log_tau = theta[:, 0]
    k = n_positions - 1
    beta_free = theta[:, 1:1 + k]
    slot_free = theta[:, 1 + k:]
    contrasts = _contrast_matrix(n_positions)
    return np.exp(log_tau), beta_free @ contrasts.T, slot_free @ contrasts.T


def _choice_loglik(tau, beta, slot_coef, choice: Choice) -> float:
    utility = (-(choice.adp - choice.overall) / tau
               + beta[choice.position_idx]
               + slot_coef[choice.position_idx] * choice.slot_z)
    utility -= utility.max()          # softmax stability
    return float(utility[choice.chosen] - np.log(np.exp(utility).sum()))


@dataclass(frozen=True)
class _Packed:
    """Choice sets padded into rectangular arrays.

    The likelihood is evaluated hundreds of times per L-BFGS fit, and the gate
    runs one fit per season per permutation. A Python loop over choices makes
    200 permutations take hours; padding once and computing the whole
    log-likelihood as array ops makes it seconds.
    """

    delta: np.ndarray       # (N, C) adp - overall, NaN-padded
    position: np.ndarray    # (N, C) int position index, 0 where padded
    valid: np.ndarray       # (N, C) bool
    chosen: np.ndarray      # (N,) index within the row
    manager: np.ndarray     # (N,) manager index
    slot_z: np.ndarray      # (N,)


def _pack(choices: list[Choice], manager_index: dict[str, int]) -> _Packed:
    n = len(choices)
    width = max(len(c.adp) for c in choices)
    delta = np.zeros((n, width))
    position = np.zeros((n, width), dtype=int)
    valid = np.zeros((n, width), dtype=bool)
    chosen = np.empty(n, dtype=int)
    manager = np.empty(n, dtype=int)
    slot_z = np.empty(n)
    for i, c in enumerate(choices):
        k = len(c.adp)
        delta[i, :k] = c.adp - c.overall
        position[i, :k] = c.position_idx
        valid[i, :k] = True
        chosen[i] = c.chosen
        manager[i] = manager_index[c.manager]
        slot_z[i] = c.slot_z
    return _Packed(delta, position, valid, chosen, manager, slot_z)


def _packed_loglik(taus, betas, slots, packed: _Packed) -> float:
    tau_row = taus[packed.manager][:, None]
    beta_row = betas[packed.manager]                       # (N, P)
    slot_row = slots[packed.manager]                       # (N, P)
    beta_sel = np.take_along_axis(beta_row, packed.position, axis=1)
    slot_sel = np.take_along_axis(slot_row, packed.position, axis=1)
    utility = -packed.delta / tau_row + beta_sel + slot_sel * packed.slot_z[:, None]
    utility = np.where(packed.valid, utility, -np.inf)
    shift = utility.max(axis=1, keepdims=True)
    exp_u = np.exp(utility - shift)
    log_denom = np.log(exp_u.sum(axis=1)) + shift[:, 0]
    picked = utility[np.arange(len(packed.chosen)), packed.chosen]
    return float((picked - log_denom).sum())


def _objective_and_grad(theta, packed, n_managers, n_positions,
                        prior_tau_sd, prior_beta_sd):
    """Negative MAP log-likelihood and its analytic gradient.

    Supplying the gradient is not an optimization nicety here: without it
    L-BFGS-B takes 2*n_params extra objective evaluations per iteration
    (77 parameters -> ~3 s per fit), and the gate needs one fit per season per
    permutation. With it a fit is milliseconds and 200 permutations is
    tractable.
    """
    taus, betas, slots = _unpack(theta, n_managers, n_positions)
    contrasts = _contrast_matrix(n_positions)          # (P, P-1)
    n_obs = len(packed.chosen)
    rows = np.arange(n_obs)

    tau_row = taus[packed.manager][:, None]
    beta_sel = np.take_along_axis(betas[packed.manager], packed.position, axis=1)
    slot_sel = np.take_along_axis(slots[packed.manager], packed.position, axis=1)
    utility = -packed.delta / tau_row + beta_sel + slot_sel * packed.slot_z[:, None]
    utility = np.where(packed.valid, utility, -np.inf)

    shift = utility.max(axis=1, keepdims=True)
    exp_u = np.exp(utility - shift)
    denom = exp_u.sum(axis=1, keepdims=True)
    prob = exp_u / denom
    loglik = float((utility[rows, packed.chosen]
                    - (np.log(denom[:, 0]) + shift[:, 0])).sum())

    # d loglik / d utility_ij  =  1{j chosen} - p_ij
    dU = -prob
    dU[rows, packed.chosen] += 1.0
    dU = np.where(packed.valid, dU, 0.0)

    # log-tau: utility = -delta/tau, so d/d(log tau) = delta/tau
    d_logtau_obs = (dU * (packed.delta / tau_row)).sum(axis=1)
    # position effects, mapped through the sum-to-zero contrasts
    onehot = (packed.position[:, :, None]
              == np.arange(n_positions)[None, None, :]) & packed.valid[:, :, None]
    dU_by_pos = (dU[:, :, None] * onehot).sum(axis=1)          # (N, P)
    d_beta_free_obs = dU_by_pos @ contrasts                     # (N, P-1)
    d_slot_free_obs = d_beta_free_obs * packed.slot_z[:, None]

    per = 1 + 2 * (n_positions - 1)
    grad_ll = np.zeros((n_managers, per))
    np.add.at(grad_ll, (packed.manager, 0), d_logtau_obs)
    for k in range(n_positions - 1):
        np.add.at(grad_ll, (packed.manager, 1 + k), d_beta_free_obs[:, k])
        np.add.at(grad_ll, (packed.manager, 1 + (n_positions - 1) + k),
                  d_slot_free_obs[:, k])

    # MAP shrinkage toward the league mean. With 11-55 observations per manager
    # this is what stops thin managers running to the bounds.
    tau_bar = taus.mean()
    beta_bar = betas.mean(axis=0)
    penalty = (((taus - tau_bar) ** 2).sum() / (2 * prior_tau_sd ** 2)
               + ((betas - beta_bar) ** 2).sum() / (2 * prior_beta_sd ** 2)
               + (slots ** 2).sum() / (2 * prior_beta_sd ** 2))

    grad_pen = np.zeros((n_managers, per))
    # sum_j (tau_j - tau_bar) == 0, so the tau_bar term drops out
    grad_pen[:, 0] = (taus - tau_bar) / prior_tau_sd ** 2 * taus
    d_beta = (betas - beta_bar) / prior_beta_sd ** 2
    d_slot = slots / prior_beta_sd ** 2
    grad_pen[:, 1:n_positions] = d_beta @ contrasts
    grad_pen[:, n_positions:] = d_slot @ contrasts

    return -(loglik - penalty), (grad_pen - grad_ll).ravel()


def fit_managers(
    choices: list[Choice], positions: tuple[str, ...], *,
    prior_tau_sd: float = 3.0, prior_beta_sd: float = 0.5,
    tau_init: float = 8.0,
) -> FitResult:
    managers = sorted({c.manager for c in choices})
    index = {m: i for i, m in enumerate(managers)}
    n_m, n_p = len(managers), len(positions)
    per = 1 + 2 * (n_p - 1)

    theta0 = np.zeros(n_m * per)
    theta0[::per] = np.log(tau_init)

    packed = _pack(choices, index)
    result = minimize(
        _objective_and_grad, theta0, jac=True,
        args=(packed, n_m, n_p, prior_tau_sd, prior_beta_sd),
        method="L-BFGS-B", options={"maxiter": 500},
    )
    taus, betas, slots = _unpack(result.x, n_m, n_p)

    counts = {m: 0 for m in managers}
    slot_values: dict[str, set] = {m: set() for m in managers}
    for c in choices:
        counts[c.manager] += 1
        slot_values[c.manager].add(round(c.slot_z, 6))

    fitted = {
        m: FittedManager(
            manager=m, tau=float(taus[i]),
            beta={p: float(betas[i, j]) for j, p in enumerate(positions)},
            slot_coef={p: float(slots[i, j]) for j, p in enumerate(positions)},
            n_choices=counts[m], slot_varies=len(slot_values[m]) > 1,
        )
        for m, i in index.items()
    }
    return FitResult(managers=fitted, log_likelihood=-result.fun,
                     positions=positions)


def _total_loglik(fit: FitResult, choices: list[Choice]) -> float:
    usable = [c for c in choices if c.manager in fit.managers]
    if not usable:
        return 0.0
    index = {m: i for i, m in enumerate(sorted(fit.managers))}
    taus = np.array([fit.managers[m].tau for m in sorted(fit.managers)])
    betas = np.array([[fit.managers[m].beta[p] for p in fit.positions]
                      for m in sorted(fit.managers)])
    slots = np.array([[fit.managers[m].slot_coef[p] for p in fit.positions]
                      for m in sorted(fit.managers)])
    return _packed_loglik(taus, betas, slots, _pack(usable, index))


def league_mean_loglik(choices: list[Choice], positions: tuple[str, ...],
                       **kwargs) -> float:
    """Log-likelihood of a single pooled policy — the null the gate tests."""
    pooled = [
        Choice("_pooled", c.season, c.overall, c.slot_z, c.adp,
               c.position_idx, c.chosen)
        for c in choices
    ]
    fit = fit_managers(pooled, positions, **kwargs)
    fm = fit.managers["_pooled"]
    beta = np.array([fm.beta[p] for p in positions])
    slot = np.array([fm.slot_coef[p] for p in positions])
    return sum(_choice_loglik(fm.tau, beta, slot, c) for c in pooled)
