"""Approximate Dynamic Programming for Blood Management — Powell's SPAR/CAVE.

The shipped :class:`OTAllocationPolicy` is *myopic*: it maximises the current
period's match contribution only. The original problem is an **ADP**: it learns a
separable, concave piecewise-linear **value of holding** blood into the future
(one PWL value function per inventory slot), so the per-period LP trades current
contribution against value-to-go.

This module reproduces that learning faithfully (the SPAR / CAVE routine):

  * per period, solve an LP that allocates blood to demand (contribution) or holds
    it (value-to-go = the current PWL slopes on parallel hold arcs);
  * read the LP **dual** of each supply (inventory) constraint — the marginal
    value ``vhat`` of a unit of that blood;
  * update the previous period's slope ``vbar`` toward ``vhat`` with a stepsize,
    then **project** neighbouring slopes back to keep the value function concave.

Parity here is *behavioural*, not numeric: the learned value functions depend on
the RNG stream / stepsize / projection, so they will not match the original
numbers — but the trained policy should beat the myopic one and the learned
value functions should be concave. The training LP uses **exact** duals
(scipy/HiGHS, faithful to the original glpk), since SPAR's update is the LP dual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog

from problems.blood_management.model import BloodManagementModel


@dataclass
class SPARConfig:
    """Hyper-parameters for SPAR ADP training."""
    horizon: int = 10          # MAX_TIME periods per sample path
    n_links: int = 40          # parallel hold arcs = PWL breakpoints per slot
    discount: float = 0.95     # DISCOUNT_FACTOR
    alpha: float = 0.2         # constant stepsize (SPAR observation weight)
    discard_penalty: float = 10.0  # cost per unit of expired (oldest-age) blood
    n_iter: int = 300          # training sample paths


@dataclass
class ValueFunction:
    """Separable concave PWL value functions: ``slopes[t, slot, link]``.

    ``slopes[t, s, :]`` is non-increasing (concave); the value of holding ``q``
    units of slot ``s`` at time ``t`` is ``sum(slopes[t, s, :q])``.
    """
    slopes: np.ndarray  # [horizon, n_slots, n_links]

    def value_to_go(self, t: int, slot: int, qty: int) -> float:
        """Total value of holding ``qty`` units of ``slot`` at time ``t``."""
        return float(np.sum(self.slopes[t, slot, :max(0, qty)]))


def _solve_period_lp(
    model: BloodManagementModel,
    inv: np.ndarray,
    demand: np.ndarray,
    slope_t: np.ndarray,         # [n_slots, n_links] value-to-go slopes for this t
    cfg: SPARConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the per-period allocate-or-hold LP exactly; return (alloc, held, dual).

    Variables: X[slot, demand] (allocation) and H[slot, link] (hold arcs, each in
    [0, 1]). Maximise  sum C*X + discount * sum slope*H  - discard on oldest age,
    s.t.  sum_d X[s,d] + sum_r H[s,r] = inv[s]  (dual = marginal value),
          sum_s X[s,d] <= demand[d],  X feasible-only, 0 <= H <= 1.
    """
    nS, nD, nL = model.n_inventory_slots, model.n_demand_types, cfg.n_links
    C = np.array(model.contribution_matrix)
    feas = np.array(model.feasible_mask)
    max_age = model.config.max_age
    nx = nS * nD
    nvar = nx + nS * nL

    # objective (minimise negative contribution / value)
    c = np.zeros(nvar)
    c[:nx] = -C.reshape(-1)
    for s in range(nS):
        age = s % max_age
        if age < max_age - 1:
            c[nx + s * nL: nx + (s + 1) * nL] = -cfg.discount * slope_t[s]
        else:  # oldest age: holding == discarding -> penalty
            c[nx + s * nL: nx + (s + 1) * nL] = cfg.discard_penalty

    # supply equalities: sum_d X[s,d] + sum_r H[s,r] = inv[s]
    A_eq = np.zeros((nS, nvar))
    for s in range(nS):
        A_eq[s, s * nD:(s + 1) * nD] = 1.0
        A_eq[s, nx + s * nL: nx + (s + 1) * nL] = 1.0
    b_eq = inv.copy()

    # demand caps: sum_s X[s,d] <= demand[d]
    A_ub = np.zeros((nD, nvar))
    for d in range(nD):
        A_ub[d, d:nx:nD] = 1.0
    b_ub = demand.copy()

    bounds = []
    for s in range(nS):
        for d in range(nD):
            bounds.append((0.0, None) if feas[s, d] else (0.0, 0.0))
    bounds += [(0.0, 1.0)] * (nS * nL)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    x = res.x
    alloc = x[:nx].reshape(nS, nD)
    held = x[nx:].reshape(nS, nL).sum(axis=1)
    # marginal value of a unit of supply at slot s = d(reward)/d(inv) = -eqlin.marginal
    dual = -np.asarray(res.eqlin.marginals)
    return alloc, held, dual


def _project_concave(arr: np.ndarray, index: int, vnew: float, vbar: float) -> None:
    """Restore non-increasing slopes after updating ``arr[index]`` (SPAR 'Avg')."""
    if vnew > vbar:  # bumped up: fix smaller-index slopes that fell below
        left = [i for i in range(index + 1) if arr[i] <= vnew]
        if left:
            arr[left] = np.mean(arr[left])
    elif vnew < vbar:  # dropped: fix larger-index slopes that rose above
        right = [i for i in range(index, len(arr)) if arr[i] >= vnew]
        if right:
            arr[right] = np.mean(arr[right])


def train_spar(model: BloodManagementModel, cfg: SPARConfig, seed: int = 0) -> ValueFunction:
    """Train separable concave PWL value functions via SPAR over sample paths."""
    import jax

    nS, nL = model.n_inventory_slots, cfg.n_links
    max_age = model.config.max_age
    slopes = np.zeros((cfg.horizon, nS, nL))
    key = jax.random.PRNGKey(seed)

    for _ in range(cfg.n_iter):
        inv = np.array(model.init_state(key)[:-1])
        prev_held: np.ndarray | None = None
        for t in range(cfg.horizon):
            key, k = jax.random.split(key)
            exog = model.sample_exogenous(k, jnp.asarray(np.concatenate([inv, [0.0]])), t)
            demand = np.array(exog.demand)
            alloc, held, dual = _solve_period_lp(model, inv, demand, slopes[t], cfg)

            # SPAR update of the PREVIOUS period's slopes with this period's dual:
            # holding slot s at t-1 ages into slot s' at t, worth dual[s'].
            if prev_held is not None and t >= 1:
                for s in range(nS):
                    age = s % max_age
                    if age >= max_age - 1:
                        continue  # oldest can't be held usefully
                    s_next = s + 1  # (blood_type, age) -> (blood_type, age+1)
                    vhat = dual[s_next]
                    idx = min(int(round(prev_held[s])), nL - 1)
                    arr = slopes[t - 1, s]
                    vbar = arr[idx]
                    vnew = cfg.alpha * vhat + (1 - cfg.alpha) * vbar
                    arr[idx] = vnew
                    _project_concave(arr, idx, vnew, vbar)

            # transition inventory: held blood ages, oldest expires, add donations
            donation = np.array(exog.donation)
            new_inv = np.zeros(nS)
            for s in range(nS):
                age = s % max_age
                if age < max_age - 1:
                    new_inv[s + 1] = held[s]
            for bt in range(model.n_blood_types):
                new_inv[bt * max_age] = donation[bt]
            inv, prev_held = new_inv, held

    return ValueFunction(slopes=slopes)


class ADPPolicy:
    """Allocation policy using trained value-to-go (the per-period ADP LP).

    Unlike the myopic OT policy, this holds blood whose learned value-to-go
    exceeds its current match contribution. Needs the realised ``demand`` and the
    current time ``t`` (read from ``state[-1]``).

    Example:
        >>> vf = train_spar(model, SPARConfig())
        >>> policy = ADPPolicy(model, SPARConfig(), vf)
        >>> allocation = policy(None, state, key, demand)
    """

    def __init__(self, model: BloodManagementModel, cfg: SPARConfig,
                 value_fn: ValueFunction) -> None:
        """Initialize with the model, SPAR config, and a trained value function."""
        self.model = model
        self.cfg = cfg
        self.value_fn = value_fn

    def __call__(self, params: Optional[Any], state: Any, key: Any, demand: Any) -> np.ndarray:
        """Return the flattened allocation maximising contribution + value-to-go."""
        inv = np.array(state[:-1])
        t = min(int(state[-1]), self.cfg.horizon - 1)
        alloc, _, _ = _solve_period_lp(
            self.model, inv, np.array(demand), self.value_fn.slopes[t], self.cfg)
        return alloc.reshape(-1)


def evaluate(model: BloodManagementModel, cfg: SPARConfig,
             value_fn: ValueFunction | None, seed: int, n_paths: int = 50) -> float:
    """Mean total contribution of the (myopic if value_fn is None) policy."""
    import jax

    nS, nL = model.n_inventory_slots, cfg.n_links
    max_age = model.config.max_age
    zero = np.zeros((nS, nL))
    key = jax.random.PRNGKey(seed)
    totals = []
    for _ in range(n_paths):
        inv = np.array(model.init_state(key)[:-1])
        total = 0.0
        for t in range(cfg.horizon):
            key, k = jax.random.split(key)
            exog = model.sample_exogenous(k, jnp.asarray(np.concatenate([inv, [0.0]])), t)
            demand = np.array(exog.demand)
            slope_t = value_fn.slopes[t] if value_fn is not None else zero
            alloc, held, _ = _solve_period_lp(model, inv, demand, slope_t, cfg)
            total += float((np.array(model.contribution_matrix) * alloc).sum())
            donation = np.array(exog.donation)
            new_inv = np.zeros(nS)
            for s in range(nS):
                if s % max_age < max_age - 1:
                    new_inv[s + 1] = held[s]
            for bt in range(model.n_blood_types):
                new_inv[bt * max_age] = donation[bt]
            inv = new_inv
        totals.append(total)
    return float(np.mean(totals))
