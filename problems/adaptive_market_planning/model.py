"""JAX-native Adaptive Market Planning Model.

This module implements a gradient-based learning problem where an agent learns
the optimal order quantity through repeated market interactions:
- Place order and observe demand
- Compute gradient: (price - cost) if undersupply, -cost if oversupply
- Update order quantity using step size rule
- Track sign changes to adapt learning rate (Kesten's rule)

State: [order_quantity, counter] - current order and sign change count
Decision: [step_size] - learning rate for gradient update
Exogenous: demand - random demand realization
"""

from typing import NamedTuple
from functools import partial
from jaxtyping import Array, Float, Int, PRNGKeyArray
import jax
import jax.numpy as jnp
import chex


# Type aliases
State = Float[Array, "2"]  # [order_quantity, counter]
Decision = Float[Array, "1"]  # [step_size]
Reward = Float[Array, ""]  # scalar profit
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information for Adaptive Market Planning.

    Attributes:
        demand: Random demand realization.
        previous_derivative: Derivative from previous step (for sign changes).
    """
    demand: Float[Array, ""]
    previous_derivative: Float[Array, ""]


@chex.dataclass(frozen=True)
class AdaptiveMarketPlanningConfig:
    """Configuration for Adaptive Market Planning model.

    Attributes:
        price: Unit selling price (revenue per unit sold).
        cost: Unit ordering cost (cost per unit ordered).
        demand_mean: Mean of exponential demand distribution.
        initial_order_quantity: Initial order quantity.
        max_order_quantity: Maximum allowed order quantity.
    """
    price: float = 1.0
    cost: float = 0.5
    demand_mean: float = 100.0
    initial_order_quantity: float = 50.0
    max_order_quantity: float = 1000.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.cost < 0:
            raise ValueError(f"cost must be non-negative, got {self.cost}")
        if self.price <= self.cost:
            raise ValueError(
                f"price ({self.price}) must be > cost ({self.cost}) "
                f"for profitable operation"
            )
        if self.demand_mean <= 0:
            raise ValueError(f"demand_mean must be positive, got {self.demand_mean}")
        if self.initial_order_quantity < 0:
            raise ValueError(
                f"initial_order_quantity must be non-negative, "
                f"got {self.initial_order_quantity}"
            )
        if self.max_order_quantity <= 0:
            raise ValueError(
                f"max_order_quantity must be positive, got {self.max_order_quantity}"
            )


class AdaptiveMarketPlanningModel:
    """JAX-native model for Adaptive Market Planning problem.

    The agent learns optimal order quantity through gradient-based updates:
    - Observes demand and computes profit/loss
    - Computes gradient based on undersupply/oversupply
    - Updates order quantity: q_{t+1} = max(0, q_t + step_size * gradient)
    - Tracks sign changes for adaptive step size (Kesten's rule)

    Example:
        >>> config = AdaptiveMarketPlanningConfig(price=1.0, cost=0.5)
        >>> model = AdaptiveMarketPlanningModel(config)
        >>> key = jax.random.PRNGKey(0)
        >>> state = model.init_state(key)
        >>> decision = jnp.array([0.1])  # Step size
    """

    def __init__(self, config: AdaptiveMarketPlanningConfig) -> None:
        """Initialize model.

        Args:
            config: Model configuration.
        """
        self.config = config

        # Compute optimal order quantity for reference
        # For exponential demand, optimal q* = demand_mean - cost/(price-cost) * demand_mean
        # Simplified: q* ≈ demand_mean * (1 - cost/price)
        self.optimal_quantity = self.config.demand_mean * (
            1 - self.config.cost / self.config.price
        )

    def init_state(self, key: Key) -> State:
        """Initialize state.

        Args:
            key: Random key (unused, for consistency).

        Returns:
            Initial state [order_quantity, counter].
        """
        return jnp.array([
            self.config.initial_order_quantity,
            0.0,  # No sign changes initially
        ])

    @partial(jax.jit, static_argnums=(0,))
    def transition(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> State:
        """Compute next state via gradient update.

        Args:
            state: Current state [order_quantity, counter].
            decision: Decision [step_size].
            exog: Exogenous information with demand and previous derivative.

        Returns:
            Next state [new_order_quantity, new_counter].
        """
        order_quantity, counter = state[0], state[1]
        step_size = decision[0]

        # Compute gradient (derivative of profit w.r.t. order quantity)
        # If demand > order_quantity: gradient = price - cost (should order more)
        # If demand <= order_quantity: gradient = -cost (ordered too much)
        derivative = jnp.where(
            order_quantity < exog.demand,
            self.config.price - self.config.cost,  # Undersupply
            -self.config.cost  # Oversupply
        )

        # Update order quantity with gradient step (projected to [0, max])
        new_order_quantity = jnp.clip(
            order_quantity + step_size * derivative,
            0.0,
            self.config.max_order_quantity
        )

        # Update counter: increment if derivative changed sign
        sign_changed = (exog.previous_derivative * derivative) < 0
        new_counter = jnp.where(sign_changed, counter + 1, counter)

        return jnp.array([new_order_quantity, new_counter])

    @partial(jax.jit, static_argnums=(0,))
    def reward(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> Reward:
        """Compute single-period profit.

        Profit = price * min(order_quantity, demand) - cost * order_quantity

        Args:
            state: Current state.
            decision: Decision (unused).
            exog: Exogenous information.

        Returns:
            Profit (reward).
        """
        order_quantity = state[0]

        # Revenue from sales (sell min of order and demand)
        revenue = self.config.price * jnp.minimum(order_quantity, exog.demand)

        # Cost of ordering
        cost = self.config.cost * order_quantity

        profit = revenue - cost

        return profit

    def sample_exogenous(
        self,
        key: Key,
        state: State,
        time: int,
        previous_derivative: float = 0.0
    ) -> ExogenousInfo:
        """Sample exogenous information.

        Args:
            key: Random key.
            state: Current state (unused).
            time: Current time step (unused).
            previous_derivative: Derivative from previous step (for tracking sign changes).

        Returns:
            Sampled exogenous information.
        """
        # Sample demand from exponential distribution
        demand = jax.random.exponential(key) * self.config.demand_mean

        return ExogenousInfo(
            demand=demand,
            previous_derivative=jnp.array(previous_derivative)
        )

    @partial(jax.jit, static_argnums=(0,))
    def is_valid_decision(self, state: State, decision: Decision) -> jax.Array:
        """Check if decision is valid.

        Args:
            state: Current state.
            decision: Decision to validate.

        Returns:
            Boolean array: True if decision is valid (step_size >= 0).
        """
        step_size = decision[0]
        return step_size >= 0.0

    def compute_derivative(self, state: State, demand: Float[Array, ""]) -> Float[Array, ""]:
        """Compute gradient of profit w.r.t. order quantity.

        This is exposed as a separate method for use in policies.

        Args:
            state: Current state.
            demand: Observed demand.

        Returns:
            Derivative (gradient).
        """
        order_quantity = state[0]
        return jnp.where(
            order_quantity < demand,
            self.config.price - self.config.cost,
            -self.config.cost
        )
