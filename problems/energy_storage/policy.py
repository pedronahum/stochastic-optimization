"""JAX-native policies for Energy Storage.

This module implements various decision policies for battery energy storage:
- Threshold-based policies (buy low, sell high)
- Time-of-day policies
- Neural network policies
- Myopic/greedy policies
"""

from typing import Optional, List, Any
from functools import partial
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
import jax
import jax.numpy as jnp
import chex
from flax import nnx


# Type aliases
State = Float[Array, "3"]  # [energy, cycles, time_of_day]
Decision = Float[Array, "1"]  # [charge_power]
Key = PRNGKeyArray


@chex.dataclass(frozen=True)
class ThresholdPolicyConfig:
    """Configuration for threshold-based policies.

    Attributes:
        buy_threshold: Price below which to buy/charge.
        sell_threshold: Price above which to sell/discharge.
        charge_rate: Fraction of max rate to use when charging (0-1).
        discharge_rate: Fraction of max rate to use when discharging (0-1).
    """
    buy_threshold: float = 40.0
    sell_threshold: float = 60.0
    charge_rate: float = 0.5
    discharge_rate: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration."""
        chex.assert_scalar_positive(self.buy_threshold)
        chex.assert_scalar_positive(self.sell_threshold)
        if self.sell_threshold <= self.buy_threshold:
            raise ValueError(
                f"sell_threshold ({self.sell_threshold}) must be > "
                f"buy_threshold ({self.buy_threshold})"
            )
        if not (0.0 < self.charge_rate <= 1.0):
            raise ValueError(f"charge_rate must be in (0, 1], got {self.charge_rate}")
        if not (0.0 < self.discharge_rate <= 1.0):
            raise ValueError(f"discharge_rate must be in (0, 1], got {self.discharge_rate}")


class ThresholdPolicy:
    """Threshold-based policy: Buy low, sell high.

    Charges when price is below buy_threshold.
    Discharges when price is above sell_threshold.
    Holds when price is in between.

    Example:
        >>> from stochopt.problems.energy_storage.model import EnergyStorageModel, EnergyStorageConfig
        >>> config_model = EnergyStorageConfig()
        >>> model = EnergyStorageModel(config_model)
        >>> config_policy = ThresholdPolicyConfig(buy_threshold=40.0, sell_threshold=60.0)
        >>> policy = ThresholdPolicy(model, config_policy)
    """

    def __init__(
        self,
        model: "EnergyStorageModel",  # type: ignore[name-defined]
        config: ThresholdPolicyConfig
    ) -> None:
        """Initialize policy.

        Args:
            model: Energy storage model instance.
            config: Policy configuration.
        """
        self.model = model
        self.config = config

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Get decision based on threshold rule.

        Args:
            params: Unused (no learnable parameters).
            state: Current state [energy, cycles, time].
            key: Random key (unused for deterministic policy).

        Returns:
            Decision: charge power (+ for charge, - for discharge).
        """
        # Sample current price
        exog = self.model.sample_exogenous(key, state, 0)
        price = exog.price

        # Get feasible bounds
        min_charge, max_charge = self.model.get_feasible_bounds(state)

        # Apply threshold logic
        decision = jnp.where(
            price < self.config.buy_threshold,
            # Low price: charge at configured rate
            max_charge * self.config.charge_rate,
            jnp.where(
                price > self.config.sell_threshold,
                # High price: discharge at configured rate
                -max_charge * self.config.discharge_rate,
                # Mid price: hold
                0.0,
            )
        )

        return jnp.array([decision])


class TimeOfDayPolicy:
    """Time-of-day policy: Charge off-peak, discharge on-peak.

    Charges during off-peak hours (low demand/price).
    Discharges during peak hours (high demand/price).

    Example:
        >>> policy = TimeOfDayPolicy(model, peak_start=9, peak_end=20)
    """

    def __init__(
        self,
        model: "EnergyStorageModel",  # type: ignore[name-defined]
        peak_start: int = 9,
        peak_end: int = 20,
        charge_rate: float = 0.5,
        discharge_rate: float = 0.5,
    ) -> None:
        """Initialize policy.

        Args:
            model: Energy storage model instance.
            peak_start: Hour when peak period starts (0-23).
            peak_end: Hour when peak period ends (0-23).
            charge_rate: Fraction of max rate to use when charging.
            discharge_rate: Fraction of max rate to use when discharging.
        """
        self.model = model
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Get decision based on time-of-day.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).

        Returns:
            Decision.
        """
        hour = state[2]

        # Get feasible bounds
        min_charge, max_charge = self.model.get_feasible_bounds(state)

        # Check if in peak hours
        in_peak = (hour >= self.peak_start) & (hour <= self.peak_end)

        decision = jnp.where(
            in_peak,
            # Peak: discharge
            -max_charge * self.discharge_rate,
            # Off-peak: charge
            max_charge * self.charge_rate,
        )

        return jnp.array([decision])


class MyopicPolicy:
    """Myopic policy: Maximize immediate reward.

    Looks one step ahead to choose action with highest immediate reward.

    Example:
        >>> policy = MyopicPolicy(model)
    """

    def __init__(
        self,
        model: "EnergyStorageModel",  # type: ignore[name-defined]
        n_samples: int = 5
    ) -> None:
        """Initialize policy.

        Args:
            model: Energy storage model instance.
            n_samples: Number of actions to sample.
        """
        self.model = model
        self.n_samples = n_samples

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Get decision by maximizing immediate reward.

        Args:
            params: Unused.
            state: Current state.
            key: Random key for sampling.

        Returns:
            Decision with highest immediate reward.
        """
        # Get feasible bounds
        min_charge, max_charge = self.model.get_feasible_bounds(state)

        # Sample n_samples actions uniformly in feasible range
        key_actions, key_exog = jax.random.split(key)
        actions = jax.random.uniform(
            key_actions,
            shape=(self.n_samples,),
            minval=min_charge,
            maxval=max_charge
        )

        # Sample exogenous info
        exog = self.model.sample_exogenous(key_exog, state, 0)

        # Compute reward for each action
        def compute_reward(action: Any) -> Any:
            decision_array = jnp.array([action])
            return self.model.reward(state, decision_array, exog)

        rewards = jax.vmap(compute_reward)(actions)

        # Choose action with highest reward
        best_idx = jnp.argmax(rewards)
        best_action = actions[best_idx]

        return jnp.array([best_action])


class LinearPolicy(nnx.Module):
    """Learnable linear policy.

    Decision = w0 + w1 * energy + w2 * price + w3 * hour

    Example:
        >>> policy = LinearPolicy(rngs=nnx.Rngs(0))
    """

    def __init__(self, rngs: nnx.Rngs) -> None:
        """Initialize policy with random parameters.

        Args:
            rngs: Flax NNX random number generator state.
        """
        # Initialize learnable parameters
        # 4 features: bias, energy, price, hour
        self.weights = nnx.Param(
            jax.random.normal(rngs(), shape=(4,)) * 0.1
        )

    def __call__(self, state: State, price: float, key: Key) -> Decision:
        """Get decision from linear model.

        Args:
            state: Current state [energy, cycles, time].
            price: Current electricity price.
            key: Random key (unused for deterministic policy).

        Returns:
            Decision.
        """
        energy, cycles, hour = state[0], state[1], state[2]

        # Create feature vector
        features = jnp.array([
            1.0,  # Bias
            energy / 1000.0,  # Normalized energy
            price / 100.0,  # Normalized price
            hour / 24.0,  # Normalized hour
        ])

        # Linear combination
        decision = jnp.dot(self.weights.value, features)

        return jnp.array([decision])


class NeuralPolicy(nnx.Module):
    """Neural network policy for energy storage.

    Maps [energy, price, hour] -> charge_power

    Example:
        >>> policy = NeuralPolicy(hidden_dims=[32, 16], rngs=nnx.Rngs(0))
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        rngs: Optional[nnx.Rngs] = None
    ) -> None:
        """Initialize neural network policy.

        Args:
            hidden_dims: List of hidden layer dimensions.
            rngs: Flax NNX random number generator state.
        """
        if hidden_dims is None:
            hidden_dims = [32, 16]
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Input: [energy, price, hour] = 3 features
        input_dim = 3

        # Build network layers
        layers: List[Any] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim

        # Output layer: single value (charge power)
        layers.append(nnx.Linear(prev_dim, 1, rngs=rngs))

        self.layers = layers

    def __call__(self, state: State, price: float, key: Key) -> Decision:
        """Get decision from neural network.

        Args:
            state: Current state [energy, cycles, time].
            price: Current electricity price.
            key: Random key (unused).

        Returns:
            Decision.
        """
        energy, cycles, hour = state[0], state[1], state[2]

        # Create input features (normalized)
        x = jnp.array([
            energy / 1000.0,  # Normalized energy
            price / 100.0,  # Normalized price
            hour / 24.0,  # Normalized hour
        ])

        # Forward pass through network
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)

        # Output layer (no activation)
        decision = self.layers[-1](x)[0]

        return jnp.array([decision])


class AlwaysHoldPolicy:
    """Baseline policy that never charges or discharges."""

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Always return zero (hold).

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).

        Returns:
            Decision: Always [0.0] (hold).
        """
        return jnp.array([0.0])
