"""JAX-native Two Newsvendor Model.

This module implements a two-agent newsvendor coordination problem:
- Field agent observes demand estimate and requests quantity from Central
- Central agent observes own estimate and allocates quantity to Field
- Both agents learn biases over time using exponential smoothing
- Objective: Minimize total overage and underage costs

State: Estimates and bias tracking for both agents
Decision: Quantity request (Field) and quantity allocated (Central)
Exogenous: True demand and noisy demand estimates
"""

from typing import NamedTuple, Optional, Any
from functools import partial
from jaxtyping import Array, Float, Int, PRNGKeyArray
import jax
import jax.numpy as jnp
import chex


# Type aliases
StateField = Float[Array, "3"]  # [estimate, source_bias, central_bias]
StateCentral = Float[Array, "7"]  # [field_request, field_bias, field_weight, field_bias_hat, estimate, source_bias, source_weight]
DecisionField = Float[Array, "1"]  # [quantity_requested]
DecisionCentral = Float[Array, "1"]  # [quantity_allocated]
Reward = Float[Array, ""]
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information for Two Newsvendor.

    Attributes:
        demand: True demand realization.
        estimate_field: Noisy demand estimate observed by Field.
        estimate_central: Noisy demand estimate observed by Central.
    """
    demand: Float[Array, ""]
    estimate_field: Float[Array, ""]
    estimate_central: Float[Array, ""]


@chex.dataclass(frozen=True)
class TwoNewsvendorConfig:
    """Configuration for Two Newsvendor model.

    Attributes:
        demand_lower: Lower bound of uniform demand distribution.
        demand_upper: Upper bound of uniform demand distribution.
        est_bias_field: Bias in Field's demand estimate.
        est_std_field: Standard deviation of Field's estimate noise.
        est_bias_central: Bias in Central's demand estimate.
        est_std_central: Standard deviation of Central's estimate noise.
        overage_cost_field: Cost per unit of excess inventory for Field.
        underage_cost_field: Cost per unit of shortage for Field.
        overage_cost_central: Cost per unit of excess inventory for Central.
        underage_cost_central: Cost per unit of shortage for Central.
        alpha_bias: Smoothing parameter for bias learning (0-1).
    """
    demand_lower: float = 0.0
    demand_upper: float = 100.0
    est_bias_field: float = 0.0
    est_std_field: float = 10.0
    est_bias_central: float = 0.0
    est_std_central: float = 10.0
    overage_cost_field: float = 1.0
    underage_cost_field: float = 9.0
    overage_cost_central: float = 1.0
    underage_cost_central: float = 9.0
    alpha_bias: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.demand_upper <= self.demand_lower:
            raise ValueError(
                f"demand_upper ({self.demand_upper}) must be > "
                f"demand_lower ({self.demand_lower})"
            )
        if self.est_std_field < 0:
            raise ValueError(f"est_std_field must be non-negative, got {self.est_std_field}")
        if self.est_std_central < 0:
            raise ValueError(f"est_std_central must be non-negative, got {self.est_std_central}")
        if not (0.0 <= self.alpha_bias <= 1.0):
            raise ValueError(f"alpha_bias must be in [0, 1], got {self.alpha_bias}")
        if self.overage_cost_field < 0 or self.underage_cost_field < 0:
            raise ValueError("Costs must be non-negative")
        if self.overage_cost_central < 0 or self.underage_cost_central < 0:
            raise ValueError("Costs must be non-negative")


class TwoNewsvendorFieldModel:
    """JAX-native model for Field agent in Two Newsvendor problem.

    The Field agent:
    - Observes noisy demand estimate
    - Requests quantity from Central
    - Receives allocated quantity
    - Learns biases over time

    Example:
        >>> config = TwoNewsvendorConfig()
        >>> model = TwoNewsvendorFieldModel(config)
        >>> key = jax.random.PRNGKey(0)
        >>> state = model.init_state(key)
        >>> decision = jnp.array([50.0])  # Request 50 units
    """

    def __init__(self, config: TwoNewsvendorConfig) -> None:
        """Initialize Field model.

        Args:
            config: Model configuration.
        """
        self.config = config

    def init_state(self, key: Key) -> StateField:
        """Initialize Field state.

        Args:
            key: Random key.

        Returns:
            Initial state [estimate, source_bias, central_bias].
        """
        # Initial estimate will be set when first exogenous info is observed
        return jnp.array([
            (self.config.demand_lower + self.config.demand_upper) / 2.0,  # Mid-range estimate
            0.0,  # No source bias initially
            0.0,  # No central bias initially
        ])

    @partial(jax.jit, static_argnums=(0,))
    def transition(
        self,
        state: StateField,
        decision: DecisionField,
        exog: ExogenousInfo,
        allocated_quantity: Float[Array, ""],
    ) -> StateField:
        """Compute next state for Field.

        Args:
            state: Current state.
            decision: Field's decision (quantity requested).
            exog: Exogenous information.
            allocated_quantity: Quantity allocated by Central.

        Returns:
            Next state.
        """
        estimate, source_bias, central_bias = state[0], state[1], state[2]

        # Update biases using exponential smoothing
        # Source bias: difference between estimate and actual demand
        new_source_bias = (
            (1 - self.config.alpha_bias) * source_bias +
            self.config.alpha_bias * (estimate - exog.demand)
        )

        # Central bias: difference between allocated and requested
        requested = decision[0]
        new_central_bias = (
            (1 - self.config.alpha_bias) * central_bias +
            self.config.alpha_bias * (allocated_quantity - requested)
        )

        # Update estimate with new observation
        new_estimate = exog.estimate_field

        return jnp.array([new_estimate, new_source_bias, new_central_bias])

    @partial(jax.jit, static_argnums=(0,))
    def reward(
        self,
        state: StateField,
        decision: DecisionField,
        exog: ExogenousInfo,
        allocated_quantity: Float[Array, ""],
    ) -> Reward:
        """Compute reward for Field.

        Reward = -(overage_cost * overage + underage_cost * underage)

        Args:
            state: Current state.
            decision: Field's decision.
            exog: Exogenous information.
            allocated_quantity: Quantity allocated by Central.

        Returns:
            Negative cost (reward).
        """
        # Field's actual inventory is what was allocated
        overage = jnp.maximum(allocated_quantity - exog.demand, 0.0)
        underage = jnp.maximum(exog.demand - allocated_quantity, 0.0)

        cost = (
            self.config.overage_cost_field * overage +
            self.config.underage_cost_field * underage
        )

        return -cost

    def sample_exogenous(self, key: Key, state: StateField, time: int) -> ExogenousInfo:
        """Sample exogenous information.

        Args:
            key: Random key.
            state: Current state (unused).
            time: Current time step (unused).

        Returns:
            Sampled exogenous information.
        """
        key_demand, key_field, key_central = jax.random.split(key, 3)

        # Sample true demand from uniform distribution
        demand = jax.random.uniform(
            key_demand,
            minval=self.config.demand_lower,
            maxval=self.config.demand_upper
        )

        # Sample noisy estimates
        estimate_field = demand + jax.random.normal(key_field) * self.config.est_std_field + self.config.est_bias_field
        estimate_field = jnp.maximum(0.0, estimate_field)  # Non-negative

        estimate_central = demand + jax.random.normal(key_central) * self.config.est_std_central + self.config.est_bias_central
        estimate_central = jnp.maximum(0.0, estimate_central)  # Non-negative

        return ExogenousInfo(
            demand=demand,
            estimate_field=estimate_field,
            estimate_central=estimate_central
        )


class TwoNewsvendorCentralModel:
    """JAX-native model for Central agent in Two Newsvendor problem.

    The Central agent:
    - Receives quantity request from Field
    - Observes own noisy demand estimate
    - Allocates quantity to Field
    - Learns Field and source biases over time

    Example:
        >>> config = TwoNewsvendorConfig()
        >>> model = TwoNewsvendorCentralModel(config)
        >>> key = jax.random.PRNGKey(0)
        >>> state = model.init_state(key)
        >>> decision = jnp.array([45.0])  # Allocate 45 units
    """

    def __init__(self, config: TwoNewsvendorConfig) -> None:
        """Initialize Central model.

        Args:
            config: Model configuration.
        """
        self.config = config

    def init_state(self, key: Key) -> StateCentral:
        """Initialize Central state.

        Args:
            key: Random key.

        Returns:
            Initial state [field_request, field_bias, field_weight, field_bias_hat, estimate, source_bias, source_weight].
        """
        mid_demand = (self.config.demand_lower + self.config.demand_upper) / 2.0
        return jnp.array([
            mid_demand,  # field_request (will be updated)
            0.0,  # field_bias
            0.5,  # field_weight
            0.0,  # field_bias_hat
            mid_demand,  # estimate
            0.0,  # source_bias
            0.5,  # source_weight
        ])

    @partial(jax.jit, static_argnums=(0,))
    def transition(
        self,
        state: StateCentral,
        decision: DecisionCentral,
        exog: ExogenousInfo,
        field_request: Float[Array, ""],
    ) -> StateCentral:
        """Compute next state for Central.

        Args:
            state: Current state.
            decision: Central's decision (quantity allocated).
            exog: Exogenous information.
            field_request: Quantity requested by Field.

        Returns:
            Next state.
        """
        # Unpack state
        _, field_bias, field_weight, field_bias_hat, estimate, source_bias, source_weight = (
            state[0], state[1], state[2], state[3], state[4], state[5], state[6]
        )

        # Update biases
        # Field bias: difference between Field's request and actual demand
        new_field_bias = (
            (1 - self.config.alpha_bias) * field_bias +
            self.config.alpha_bias * (field_request - exog.demand)
        )

        # Source bias: difference between Central's estimate and actual demand
        new_source_bias = (
            (1 - self.config.alpha_bias) * source_bias +
            self.config.alpha_bias * (estimate - exog.demand)
        )

        # Update estimates and request for next round
        new_field_request = field_request  # Will be updated by Field's next decision
        new_estimate = exog.estimate_central

        return jnp.array([
            new_field_request,
            new_field_bias,
            field_weight,  # Keep same weight for now
            field_bias_hat,  # Keep same for now
            new_estimate,
            new_source_bias,
            source_weight,  # Keep same weight for now
        ])

    @partial(jax.jit, static_argnums=(0,))
    def reward(
        self,
        state: StateCentral,
        decision: DecisionCentral,
        exog: ExogenousInfo,
    ) -> Reward:
        """Compute reward for Central.

        Reward = -(overage_cost * overage + underage_cost * underage)

        Args:
            state: Current state.
            decision: Central's decision.
            exog: Exogenous information.

        Returns:
            Negative cost (reward).
        """
        allocated = decision[0]

        # Central's costs based on allocation vs demand
        overage = jnp.maximum(allocated - exog.demand, 0.0)
        underage = jnp.maximum(exog.demand - allocated, 0.0)

        cost = (
            self.config.overage_cost_central * overage +
            self.config.underage_cost_central * underage
        )

        return -cost

    def sample_exogenous(self, key: Key, state: StateCentral, time: int) -> ExogenousInfo:
        """Sample exogenous information (same as Field's).

        Args:
            key: Random key.
            state: Current state (unused).
            time: Current time step (unused).

        Returns:
            Sampled exogenous information.
        """
        # Use same sampling as Field model
        key_demand, key_field, key_central = jax.random.split(key, 3)

        demand = jax.random.uniform(
            key_demand,
            minval=self.config.demand_lower,
            maxval=self.config.demand_upper
        )

        estimate_field = demand + jax.random.normal(key_field) * self.config.est_std_field + self.config.est_bias_field
        estimate_field = jnp.maximum(0.0, estimate_field)

        estimate_central = demand + jax.random.normal(key_central) * self.config.est_std_central + self.config.est_bias_central
        estimate_central = jnp.maximum(0.0, estimate_central)

        return ExogenousInfo(
            demand=demand,
            estimate_field=estimate_field,
            estimate_central=estimate_central
        )
