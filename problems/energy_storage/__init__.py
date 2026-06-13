"""Energy Storage problem (faithful port of the original Powell problem).

The agent charges (buys) or discharges (sells) a battery against an exogenous
price series to maximise revenue ``price * (eta*sell - buy)``.

Example:
    >>> import jax, jax.numpy as jnp
    >>> from problems.energy_storage import (
    ...     EnergyStorageModel, EnergyStorageConfig, BuyLowSellHighPolicy,
    ... )
    >>> config = EnergyStorageConfig(eta=0.9, capacity=1.0)
    >>> model = EnergyStorageModel(config, prices=jnp.array([20.0, 50.0, 15.0]))
    >>> policy = BuyLowSellHighPolicy(model, theta_buy=20.0, theta_sell=40.0)
    >>> state = model.init_state(jax.random.PRNGKey(0))
    >>> decision = policy(None, state, jax.random.PRNGKey(0))
"""

from .model import (
    Decision,
    EnergyStorageConfig,
    EnergyStorageModel,
    ExogenousInfo,
    Reward,
    State,
)
from .policy import (
    AlwaysHoldPolicy,
    BuyLowSellHighPolicy,
    grid_search,
    simulate,
)

__all__ = [
    # Model
    "EnergyStorageModel",
    "EnergyStorageConfig",
    "ExogenousInfo",
    "State",
    "Decision",
    "Reward",
    # Policies / helpers
    "BuyLowSellHighPolicy",
    "AlwaysHoldPolicy",
    "simulate",
    "grid_search",
]
