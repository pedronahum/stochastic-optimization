"""Policies for the Energy Storage problem (faithful to the original).

The canonical original policy is **buy-low / sell-high**: buy when the price is
at or below ``theta_buy``, sell when at or above ``theta_sell``, otherwise hold.
The original driver grid-searches ``(theta_buy, theta_sell)``; see
``grid_search`` below.
"""

from functools import partial
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

if TYPE_CHECKING:
    from problems.energy_storage.model import EnergyStorageModel

State = Float[Array, "2"]  # [energy_amount, price]
Decision = Float[Array, "2"]  # [buy, sell]
Key = PRNGKeyArray


class BuyLowSellHighPolicy:
    """Buy when price <= theta_buy, sell when price >= theta_sell, else hold.

    Returns the constrained ``[buy, sell]`` decision (a positive buy charges to
    capacity; a sell is clipped to the stored energy), matching the original
    ``build_decision`` projection.

    Example:
        >>> policy = BuyLowSellHighPolicy(model, theta_buy=20.0, theta_sell=60.0)
        >>> decision = policy(None, state, key)
    """

    def __init__(self, model: "EnergyStorageModel", theta_buy: float, theta_sell: float) -> None:
        """Initialize policy.

        Args:
            model: Energy storage model (used to apply feasibility constraints).
            theta_buy: Lower price threshold — buy at or below this.
            theta_sell: Upper price threshold — sell at or above this.
        """
        self.model = model
        self.theta_buy = theta_buy
        self.theta_sell = theta_sell

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, params: Optional[PyTree], state: State, key: Key) -> Decision:
        """Pick buy / sell / hold by price thresholds, then apply constraints."""
        price = state[1]
        raw = jnp.where(
            price <= self.theta_buy,
            jnp.array([1.0, 0.0]),  # buy
            jnp.where(
                price >= self.theta_sell,
                jnp.array([0.0, 1.0]),  # sell
                jnp.array([0.0, 0.0]),  # hold
            ),
        )
        return self.model.apply_constraints(state, raw)


class AlwaysHoldPolicy:
    """Baseline that never trades."""

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, params: Optional[PyTree], state: State, key: Key) -> Decision:
        """Always return the no-trade decision."""
        return jnp.array([0.0, 0.0])


def simulate(model: "EnergyStorageModel", policy: BuyLowSellHighPolicy, horizon: int) -> float:
    """Run ``policy`` on ``model`` for ``horizon`` steps; return total contribution.

    Mirrors the original ``run_policy``: the final period liquidates remaining
    energy (a forced sell).
    """
    state = model.init_state(jax.random.PRNGKey(0))
    total = 0.0
    for t in range(horizon):
        if t == horizon - 1:  # liquidate on the last step
            decision = model.apply_constraints(state, jnp.array([0.0, 1.0]))
        else:
            decision = policy(None, state, jax.random.PRNGKey(0))
        exog = model.sample_exogenous(jax.random.PRNGKey(0), state, t)
        total += float(model.reward(state, decision, exog))
        state = model.transition(state, decision, exog)
    return total


def grid_search(
    model: "EnergyStorageModel",
    horizon: int,
    buy_grid: Array,
    sell_grid: Array,
) -> tuple[tuple[float, float], float]:
    """Grid-search ``(theta_buy, theta_sell)`` (original ``perform_grid_search``).

    Returns ``((best_theta_buy, best_theta_sell), best_contribution)``.
    """
    best_theta = (float(buy_grid[0]), float(sell_grid[0]))
    best = -float("inf")
    for tb in buy_grid:
        for ts in sell_grid:
            if tb >= ts:
                continue
            c = simulate(model, BuyLowSellHighPolicy(model, float(tb), float(ts)), horizon)
            if c > best:
                best, best_theta = c, (float(tb), float(ts))
    return best_theta, best
