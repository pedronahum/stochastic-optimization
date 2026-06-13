"""JAX-native Energy Storage model — faithful port of the original Powell problem.

Ported from ``legacy/old_problems/EnergyStorage_I`` (Donghun Lee, 2018). The agent
charges (buys) or discharges (sells) a battery against an exogenous electricity
price series (historically the PJM RT LMP series) to maximise revenue:

  state       = [energy_amount, price]
  decision    = [buy, sell]                       (MW bought / sold this period)
  transition  : energy' = energy + eta*buy - sell ;  price' = next price
  reward      : price * (eta*sell - buy)
  constraints : a positive buy fills to capacity, (Rmax-energy)/eta; sell <= energy

There is **no** degradation term and **no** cycle counter — those were extras in
the previous reformulation and are not part of the original problem.
"""

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, PRNGKeyArray

# Type aliases
State = Float[Array, "2"]  # [energy_amount, price]
Decision = Float[Array, "2"]  # [buy, sell]
Reward = Float[Array, ""]  # scalar contribution
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information: the price prevailing in the next period.

    Attributes:
        price: Next-period electricity price ($/MWh).
    """
    price: Float[Array, ""]


@struct.dataclass
class EnergyStorageConfig:
    """Configuration for the energy storage model.

    Attributes:
        eta: Round-trip storage efficiency in (0, 1] (original default 0.9).
        capacity: Battery capacity ``Rmax`` (MWh) (original default 1.0).
        initial_energy: Initial stored energy ``R0`` (MWh) (original default 1.0).
    """
    eta: float = 0.9
    capacity: float = 1.0
    initial_energy: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (0.0 < self.eta <= 1.0):
            raise ValueError(f"eta must be in (0, 1], got {self.eta}")
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if not (0.0 <= self.initial_energy <= self.capacity):
            raise ValueError(
                f"initial_energy ({self.initial_energy}) must be in [0, capacity]"
            )


class EnergyStorageModel:
    """Energy storage optimization model (faithful to the original).

    Prices are exogenous. For a faithful run, supply the historical price series
    via ``prices`` (e.g. the PJM RT LMP column); then ``sample_exogenous`` returns
    the recorded next price. Without a series it falls back to a Gaussian
    random-walk price so the model is usable standalone.

    Example:
        >>> config = EnergyStorageConfig(eta=0.9, capacity=1.0)
        >>> model = EnergyStorageModel(config, prices=jnp.array([20.0, 25.0, 18.0]))
        >>> state = model.init_state(jax.random.PRNGKey(0))
        >>> exog = model.sample_exogenous(jax.random.PRNGKey(0), state, 0)
    """

    def __init__(self, config: EnergyStorageConfig, prices: Optional[Array] = None) -> None:
        """Initialize the model.

        Args:
            config: Model configuration.
            prices: Optional historical price series (1-D). When given,
                ``sample_exogenous(key, state, t)`` returns ``prices[t + 1]``.
        """
        self.config = config
        self.prices = None if prices is None else jnp.asarray(prices, dtype=jnp.float32)

    def init_state(self, key: Key) -> State:
        """Initial state ``[initial_energy, first price]``."""
        price0 = self.prices[0] if self.prices is not None else jnp.array(0.0)
        return jnp.array([self.config.initial_energy, price0])

    def apply_constraints(self, state: State, raw_decision: Decision) -> Decision:
        """Project a raw ``[buy, sell]`` onto the feasible set (orig ``build_decision``).

        A positive buy charges to capacity: ``buy = (Rmax - energy) / eta``.
        A sell larger than the stored energy is clipped to ``energy``.
        """
        energy = state[0]
        raw_buy, raw_sell = raw_decision[0], raw_decision[1]
        buy = jnp.where(raw_buy > 0, (self.config.capacity - energy) / self.config.eta, raw_buy)
        sell = jnp.where(raw_sell > energy, energy, raw_sell)
        return jnp.array([buy, sell])

    @partial(jax.jit, static_argnums=(0,))
    def transition(self, state: State, decision: Decision, exog: ExogenousInfo) -> State:
        """Next state: ``energy' = energy + eta*buy - sell``; ``price' = exog.price``."""
        energy = state[0]
        buy, sell = decision[0], decision[1]
        new_energy = energy + self.config.eta * buy - sell
        return jnp.array([new_energy, exog.price])

    @partial(jax.jit, static_argnums=(0,))
    def reward(self, state: State, decision: Decision, exog: ExogenousInfo) -> Reward:
        """Contribution ``price * (eta*sell - buy)`` at the current price."""
        price = state[1]
        buy, sell = decision[0], decision[1]
        return price * (self.config.eta * sell - buy)

    def sample_exogenous(self, key: Key, state: State, time: int) -> ExogenousInfo:
        """Next price: the recorded ``prices[time + 1]`` if a series was supplied,
        else a Gaussian random walk around the current price."""
        if self.prices is not None:
            idx = jnp.clip(time + 1, 0, self.prices.shape[0] - 1)
            return ExogenousInfo(price=self.prices[idx])
        change = jax.random.normal(key) * 5.0
        return ExogenousInfo(price=jnp.maximum(0.0, state[1] + change))
