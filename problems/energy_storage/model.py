"""JAX-native Energy Storage Model.

This model addresses battery energy storage optimization under price uncertainty.
The battery can charge (buy energy) or discharge (sell energy) to maximize profit
while accounting for efficiency losses, degradation, and operational constraints.

Key features:
- Battery state of charge tracking
- Efficiency losses (round-trip < 100%)
- Degradation from cycling
- Time-of-day pricing patterns
- Operational constraints (capacity, power limits)
"""

from typing import NamedTuple, Optional, List, Any
from functools import partial
from jaxtyping import Array, Float, PRNGKeyArray, Bool
import jax
import jax.numpy as jnp
import chex


# Type aliases
State = Float[Array, "3"]  # [energy, cycles, time_of_day]
Decision = Float[Array, "1"]  # [charge_power]  (+ = charge, - = discharge)
Reward = Float[Array, ""]  # Scalar reward
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information for energy storage.

    Attributes:
        price: Electricity price ($/MWh).
        demand: Energy demand (MW) - for context.
        renewable: Available renewable generation (MW) - for context.
    """
    price: Float[Array, ""]
    demand: Float[Array, ""]
    renewable: Float[Array, ""]


@chex.dataclass(frozen=True)
class EnergyStorageConfig:
    """Configuration for energy storage model.

    This is a pytree-registered, immutable configuration.

    Attributes:
        capacity: Maximum storage capacity (MWh).
        max_charge_rate: Maximum charging power (MW).
        max_discharge_rate: Maximum discharging power (MW).
        efficiency: Round-trip efficiency (0-1), e.g., 0.95 = 95%.
        initial_energy: Initial stored energy (MWh).
        degradation_rate: Battery degradation per full cycle.
        min_energy: Minimum allowed energy (MWh), usually 0.
    """
    capacity: float = 1000.0
    max_charge_rate: float = 100.0
    max_discharge_rate: float = 100.0
    efficiency: float = 0.95
    initial_energy: float = 500.0
    degradation_rate: float = 0.001
    min_energy: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.max_charge_rate <= 0:
            raise ValueError(f"max_charge_rate must be positive, got {self.max_charge_rate}")
        if self.max_discharge_rate <= 0:
            raise ValueError(f"max_discharge_rate must be positive, got {self.max_discharge_rate}")

        # Efficiency should be in (0, 1]
        if not (0.0 < self.efficiency <= 1.0):
            raise ValueError(f"efficiency must be in (0, 1], got {self.efficiency}")

        # Degradation rate should be in [0, 1]
        if not (0.0 <= self.degradation_rate <= 1.0):
            raise ValueError(f"degradation_rate must be in [0, 1], got {self.degradation_rate}")

        # Initial energy should be within bounds
        if not (self.min_energy <= self.initial_energy <= self.capacity):
            raise ValueError(
                f"initial_energy ({self.initial_energy}) cannot exceed capacity. "
                f"Got initial_energy={self.initial_energy}, capacity={self.capacity}"
            )


class EnergyStorageModel:
    """JAX-native energy storage optimization model.

    Models a battery energy storage system that can:
    - Charge (buy energy from grid)
    - Discharge (sell energy to grid)
    - Hold (no action)

    State representation:
        - energy: Current stored energy (MWh)
        - cycles: Cumulative cycle count (for degradation tracking)
        - time_of_day: Current hour (0-23) for time-varying prices

    Dynamics:
        - Energy evolves with efficiency losses and degradation
        - Charging: energy += power * efficiency
        - Discharging: energy -= power / efficiency
        - Degradation reduces capacity over time

    Objective:
        - Maximize profit from arbitrage (buy low, sell high)
        - Account for degradation costs

    Example:
        >>> config = EnergyStorageConfig()
        >>> model = EnergyStorageModel(config)
        >>> key = jax.random.PRNGKey(0)
        >>> state = model.init_state(key)
        >>> decision = jnp.array([50.0])  # Charge at 50 MW
        >>> exog = ExogenousInfo(
        ...     price=jnp.array(50.0),
        ...     demand=jnp.array(100.0),
        ...     renewable=jnp.array(80.0)
        ... )
        >>> next_state = model.transition(state, decision, exog)
    """

    def __init__(self, config: EnergyStorageConfig) -> None:
        """Initialize model.

        Args:
            config: Model configuration.
        """
        self.config = config

    def init_state(self, key: Key) -> State:
        """Initialize state.

        Args:
            key: Random key (unused but kept for interface consistency).

        Returns:
            Initial state [energy, cycles, time_of_day].
        """
        return jnp.array([
            self.config.initial_energy,
            0.0,  # No cycles yet
            0.0,  # Start at hour 0
        ])

    @partial(jax.jit, static_argnums=(0,))
    def transition(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> State:
        """Compute next state (JIT-compiled).

        Args:
            state: Current state [energy, cycles, time].
            decision: Charge power (+ for charge, - for discharge) in MW.
            exog: Exogenous information (unused in transition, kept for interface).

        Returns:
            Next state.
        """
        # Unpack state
        energy, cycles, time_of_day = state[0], state[1], state[2]
        charge_power = decision[0]

        # Compute energy change with efficiency losses
        # Use jnp.where for JIT compatibility (no Python if/else)
        energy_change = jnp.where(
            charge_power > 0,
            charge_power * self.config.efficiency,  # Charging loses energy
            charge_power / self.config.efficiency,  # Discharging loses energy
        )

        # Degradation from cycling
        # One full cycle = charge from 0 to capacity OR discharge from capacity to 0
        cycles_this_step = jnp.abs(charge_power) / (2.0 * self.config.capacity)
        degradation = self.config.degradation_rate * cycles_this_step * energy

        # Update energy with constraints
        new_energy = jnp.clip(
            energy + energy_change - degradation,
            self.config.min_energy,
            self.config.capacity,
        )

        # Update cycle count
        new_cycles = cycles + cycles_this_step

        # Update time (wraps at 24 hours)
        new_time = (time_of_day + 1) % 24

        return jnp.array([new_energy, new_cycles, new_time])

    @partial(jax.jit, static_argnums=(0,))
    def reward(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> Reward:
        """Compute reward (JIT-compiled).

        Reward = revenue from energy arbitrage - degradation cost

        Args:
            state: Current state.
            decision: Charge power (+ for charge, - for discharge).
            exog: Exogenous information with price.

        Returns:
            Scalar reward (profit in $).
        """
        charge_power = decision[0]
        price = exog.price

        # Revenue calculation
        # Charging: pay for energy (including losses)
        # Discharging: receive payment (accounting for losses)
        revenue = jnp.where(
            charge_power > 0,
            # Charging: pay price for energy drawn from grid (including efficiency loss)
            -(charge_power / self.config.efficiency) * price,
            # Discharging: receive price for energy sold to grid (after efficiency loss)
            (-charge_power * self.config.efficiency) * price,
        )

        # Degradation cost
        # Cycling degrades the battery, which has a replacement cost
        cycles_this_step = jnp.abs(charge_power) / (2.0 * self.config.capacity)
        degradation_cost = cycles_this_step * 1000.0  # $1000 per full cycle

        return revenue - degradation_cost

    def sample_exogenous(self, key: Key, state: State, time: int) -> ExogenousInfo:
        """Sample exogenous information.

        Simulates time-of-day price patterns with stochastic variation.

        Args:
            key: JAX random key.
            state: Current state (to get time_of_day if needed).
            time: Current time step (can use this or state time).

        Returns:
            Sampled exogenous information.
        """
        # Get hour of day from state (use JAX operations to avoid concretization)
        hour = state[2] % 24.0

        # Time-of-day effects
        # Peak hours: 9 AM to 8 PM (higher prices and demand)
        peak_hours = (9 <= hour) & (hour <= 20)
        price_mult = jnp.where(peak_hours, 1.3, 0.8)
        demand_mult = jnp.where(peak_hours, 1.2, 0.7)

        # Solar generation pattern (sine wave, peaks at noon)
        solar_mult = jnp.maximum(
            0.0,
            jnp.sin(jnp.pi * (hour - 6) / 12)  # Peaks at hour 12
        )

        # Split key for independent samples
        key_price, key_demand, key_renewable = jax.random.split(key, 3)

        # Sample price with time-of-day pattern + noise
        price = jnp.maximum(
            0.0,
            jax.random.normal(key_price) * 20.0 + 50.0 * price_mult
        )

        # Sample demand with time-of-day pattern + noise
        demand = jnp.maximum(
            0.0,
            jax.random.normal(key_demand) * 30.0 + 100.0 * demand_mult
        )

        # Sample renewable generation with solar pattern + noise
        renewable = jnp.maximum(
            0.0,
            jax.random.normal(key_renewable) * 40.0 + 80.0 * solar_mult
        )

        return ExogenousInfo(
            price=price,
            demand=demand,
            renewable=renewable,
        )

    @partial(jax.jit, static_argnums=(0,))
    def is_valid_decision(
        self,
        state: State,
        decision: Decision,
    ) -> Bool[Array, ""]:
        """Check if decision is valid (JIT-compiled).

        A decision is valid if:
        - Charging power <= max_charge_rate
        - Discharging power <= max_discharge_rate
        - Result stays within [min_energy, capacity]

        Args:
            state: Current state.
            decision: Proposed decision.

        Returns:
            Boolean indicating validity.
        """
        energy = state[0]
        charge_power = decision[0]

        # Check rate limits
        valid_charge_rate = charge_power <= self.config.max_charge_rate
        valid_discharge_rate = -charge_power <= self.config.max_discharge_rate

        # Check resulting energy constraints
        energy_change = jnp.where(
            charge_power > 0,
            charge_power * self.config.efficiency,
            charge_power / self.config.efficiency,
        )
        new_energy = energy + energy_change

        valid_capacity = new_energy <= self.config.capacity
        valid_minimum = new_energy >= self.config.min_energy

        # Combine all constraints with logical AND
        return (
            valid_charge_rate &
            valid_discharge_rate &
            valid_capacity &
            valid_minimum
        )

    def get_feasible_bounds(
        self,
        state: State,
    ) -> tuple[Any, Any]:
        """Get feasible decision bounds for current state.

        Args:
            state: Current state.

        Returns:
            Tuple of (min_charge, max_charge) in MW.
            Negative values mean discharge.
        """
        energy = state[0]

        # Maximum charge (limited by capacity and charge rate)
        energy_room = self.config.capacity - energy
        max_charge_capacity = energy_room / self.config.efficiency
        max_charge = jnp.minimum(self.config.max_charge_rate, max_charge_capacity)

        # Maximum discharge (limited by available energy and discharge rate)
        available_energy = energy - self.config.min_energy
        max_discharge_energy = available_energy * self.config.efficiency
        max_discharge = jnp.minimum(self.config.max_discharge_rate, max_discharge_energy)

        return (-max_discharge, max_charge)
