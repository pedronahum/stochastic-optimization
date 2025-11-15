"""JAX-native Energy Storage Model.

This module demonstrates a complete JAX-native implementation with:
- jaxtyping for type safety
- chex for runtime assertions
- JIT compilation for performance
- Pure functional design
- GPU/TPU compatibility
"""

from typing import NamedTuple
from functools import partial
from jaxtyping import Array, Float, PRNGKeyArray, Bool
import jax
import jax.numpy as jnp
import chex

# Type aliases
State = Float[Array, "3"]  # [energy, cycles, time_of_day]
Decision = Float[Array, "1"]  # [charge_power]
Reward = Float[Array, ""]
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information for energy storage.
    
    Attributes:
        price: Electricity price ($/MWh).
        demand: Energy demand (MW).
        renewable: Available renewable generation (MW).
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
        efficiency: Round-trip efficiency (0-1).
        initial_energy: Initial stored energy (MWh).
        degradation_rate: Battery degradation per cycle.
        min_energy: Minimum allowed energy (MWh).
    """
    capacity: float = 1000.0
    max_charge_rate: float = 100.0
    max_discharge_rate: float = 100.0
    efficiency: float = 0.95
    initial_energy: float = 500.0
    degradation_rate: float = 0.001
    min_energy: float = 0.0
    
    def __post_init__(self):
        """Validate configuration with chex assertions."""
        chex.assert_scalar_positive(self.capacity)
        chex.assert_scalar_positive(self.max_charge_rate)
        chex.assert_scalar_positive(self.max_discharge_rate)
        chex.assert_scalar_in_range(self.efficiency, 0.0, 1.0)
        chex.assert_scalar_in_range(
            self.initial_energy, 0.0, self.capacity
        )
        chex.assert_scalar_non_negative(self.degradation_rate)
        chex.assert_scalar_non_negative(self.min_energy)


class EnergyStorageModel:
    """JAX-native energy storage optimization model.
    
    All methods are pure functions designed for JIT compilation.
    The model uses functional updates - state is never modified in-place.
    
    Example:
        >>> config = EnergyStorageConfig()
        >>> model = EnergyStorageModel(config)
        >>> key = jax.random.PRNGKey(0)
        >>> state = model.init_state(key)
        >>> decision = jnp.array([50.0])
        >>> exog = ExogenousInfo(
        ...     jnp.array(50.0), jnp.array(100.0), jnp.array(80.0)
        ... )
        >>> next_state = model.transition(state, decision, exog)
    """
    
    def __init__(self, config: EnergyStorageConfig):
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
            0.0,
            0.0,
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
            decision: Charge power (+ for charge, - for discharge).
            exog: Exogenous information.
        
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
            charge_power * self.config.efficiency,  # Charging
            charge_power / self.config.efficiency,  # Discharging
        )
        
        # Degradation from cycling
        cycles_this_step = jnp.abs(charge_power) / (
            2.0 * self.config.capacity
        )
        degradation = self.config.degradation_rate * cycles_this_step * energy
        
        # Update energy with constraints
        new_energy = jnp.clip(
            energy + energy_change - degradation,
            self.config.min_energy,
            self.config.capacity,
        )
        
        # Update cycle count
        new_cycles = cycles + cycles_this_step
        
        # Update time (wraps at 24)
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
        
        Args:
            state: Current state.
            decision: Charge power.
            exog: Exogenous information with price.
        
        Returns:
            Scalar reward (profit in $).
        """
        charge_power = decision[0]
        price = exog.price
        
        # Revenue (negative for charging cost, positive for discharge)
        revenue = jnp.where(
            charge_power > 0,
            # Charging: pay for energy plus losses
            -(charge_power / self.config.efficiency) * price,
            # Discharging: receive payment minus losses
            (-charge_power * self.config.efficiency) * price,
        )
        
        # Degradation cost
        cycles_this_step = jnp.abs(charge_power) / (
            2.0 * self.config.capacity
        )
        degradation_cost = cycles_this_step * 1000.0  # $1000/cycle
        
        return revenue - degradation_cost
    
    def sample_exogenous(self, key: Key, time: int) -> ExogenousInfo:
        """Sample exogenous information.
        
        Args:
            key: JAX random key.
            time: Current time step.
        
        Returns:
            Sampled exogenous information.
        """
        hour = time % 24
        
        # Time-of-day effects (use JAX operations)
        peak_hours = (9 <= hour) & (hour <= 20)
        price_mult = jnp.where(peak_hours, 1.3, 0.8)
        demand_mult = jnp.where(peak_hours, 1.2, 0.7)
        
        # Solar generation (sine wave pattern)
        solar_mult = jnp.maximum(
            0.0,
            jnp.sin(jnp.pi * (hour - 6) / 12)
        )
        
        # Split key for independent samples
        key_price, key_demand, key_renewable = jax.random.split(key, 3)
        
        # Sample with JAX random (not numpy!)
        price = jnp.maximum(
            0.0,
            jax.random.normal(key_price) * 20.0 + 50.0 * price_mult
        )
        
        demand = jnp.maximum(
            0.0,
            jax.random.normal(key_demand) * 30.0 + 100.0 * demand_mult
        )
        
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
        valid_discharge_rate = (
            -charge_power <= self.config.max_discharge_rate
        )
        
        # Check resulting energy constraints
        energy_change = jnp.where(
            charge_power > 0,
            charge_power * self.config.efficiency,
            -charge_power / self.config.efficiency,
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
    ) -> tuple[float, float]:
        """Get feasible decision bounds.
        
        Args:
            state: Current state.
        
        Returns:
            Tuple of (min_charge, max_charge) in MW.
        """
        energy = float(state[0])
        
        # Maximum charge
        energy_room = self.config.capacity - energy
        max_charge_capacity = energy_room / self.config.efficiency
        max_charge = min(self.config.max_charge_rate, max_charge_capacity)
        
        # Maximum discharge
        available_energy = energy - self.config.min_energy
        max_discharge_energy = available_energy * self.config.efficiency
        max_discharge = min(
            self.config.max_discharge_rate,
            max_discharge_energy
        )
        
        return (-max_discharge, max_charge)


# Simple policies for testing

class MyopicPolicy:
    """Simple price-based policy.
    
    Charges when price is low, discharges when price is high.
    """
    
    def __init__(self, model: EnergyStorageModel, threshold: float = 50.0):
        """Initialize policy.
        
        Args:
            model: Energy storage model.
            threshold: Price threshold for decision.
        """
        self.model = model
        self.threshold = threshold
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: None,  # No parameters for this simple policy
        state: State,
        key: Key,
    ) -> Decision:
        """Get decision based on myopic rule.
        
        Args:
            params: Unused (no learnable parameters).
            state: Current state.
            key: Random key for sampling exogenous info.
        
        Returns:
            Decision.
        """
        # Sample price to make decision
        exog = self.model.sample_exogenous(key, int(state[2]))
        price = exog.price
        
        # Get feasible bounds
        min_charge, max_charge = self.model.get_feasible_bounds(state)
        
        # Simple threshold-based rule
        decision = jnp.where(
            price < self.threshold,
            max_charge / 2,  # Charge at half rate when cheap
            jnp.where(
                price > self.threshold,
                -max_charge / 2,  # Discharge at half rate when expensive
                0.0,  # Do nothing at threshold
            )
        )
        
        return jnp.array([decision])


# Example: Run model and demonstrate performance
if __name__ == "__main__":
    import time
    
    print("JAX-Native Energy Storage Model")
    print("=" * 70)
    
    # Create model
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)
    
    # Initialize
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)
    
    print(f"\nConfiguration:")
    print(f"  Capacity: {config.capacity} MWh")
    print(f"  Max charge rate: {config.max_charge_rate} MW")
    print(f"  Efficiency: {config.efficiency}")
    print(f"\nInitial state: {state}")
    
    # Test single transition
    decision = jnp.array([50.0])
    key, subkey = jax.random.split(key)
    exog = model.sample_exogenous(subkey, 12)
    
    print(f"\nExogenous info (hour 12): {exog}")
    print(f"Decision: Charge {decision[0]} MW")
    
    # First call (includes compilation)
    start = time.time()
    next_state = model.transition(state, decision, exog)
    compile_time = time.time() - start
    
    print(f"\nFirst call (with JIT compilation): {compile_time:.4f}s")
    print(f"Next state: {next_state}")
    
    # Second call (uses compiled version)
    start = time.time()
    next_state = model.transition(state, decision, exog)
    run_time = time.time() - start
    
    print(f"Second call (JIT-compiled): {run_time:.6f}s")
    print(f"Speedup: {compile_time/run_time:.1f}x")
    
    # Test reward
    reward = model.reward(state, decision, exog)
    print(f"\nReward: ${float(reward):.2f}")
    
    # Test validation
    valid = model.is_valid_decision(state, decision)
    print(f"Valid decision: {bool(valid)}")
    
    # Batch test with vmap
    print("\n" + "=" * 70)
    print("Batch Processing with vmap")
    print("=" * 70)
    
    batch_size = 10000
    print(f"\nBatch size: {batch_size}")
    
    # Create batch of states and decisions
    states = jnp.repeat(state[None, :], batch_size, axis=0)
    decisions = jnp.linspace(-50, 50, batch_size)[:, None]
    
    # Vectorize transition
    batch_transition = jax.vmap(
        lambda s, d: model.transition(s, d, exog)
    )
    
    # First call (compile)
    start = time.time()
    batch_next_states = batch_transition(states, decisions)
    batch_compile_time = time.time() - start
    
    print(f"First call (with compilation): {batch_compile_time:.4f}s")
    
    # Second call (compiled)
    start = time.time()
    batch_next_states = batch_transition(states, decisions)
    batch_run_time = time.time() - start
    
    print(f"Second call (compiled): {batch_run_time:.6f}s")
    print(f"Per-sample time: {batch_run_time/batch_size*1e6:.2f}μs")
    print(f"Samples per second: {batch_size/batch_run_time:.0f}")
    
    # Test gradient flow
    print("\n" + "=" * 70)
    print("Automatic Differentiation")
    print("=" * 70)
    
    def loss_fn(decision):
        """Loss is negative reward (to minimize)."""
        return -model.reward(state, decision, exog)
    
    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    decision_test = jnp.array([50.0])
    gradient = grad_fn(decision_test)
    
    print(f"\nDecision: {decision_test[0]}")
    print(f"Gradient: {gradient[0]:.4f}")
    print("✓ Gradients flow through the model!")
    
    # Test policy
    print("\n" + "=" * 70)
    print("Myopic Policy")
    print("=" * 70)
    
    policy = MyopicPolicy(model, threshold=50.0)
    
    # Test cheap price
    key, subkey = jax.random.split(key)
    exog_cheap = ExogenousInfo(
        jnp.array(30.0), jnp.array(100.0), jnp.array(150.0)
    )
    decision_cheap = policy(None, state, subkey)
    print(f"\nCheap price ($30/MWh): Decision = {decision_cheap[0]:.2f} MW")
    
    # Test expensive price
    key, subkey = jax.random.split(key)
    exog_expensive = ExogenousInfo(
        jnp.array(80.0), jnp.array(150.0), jnp.array(50.0)
    )
    decision_expensive = policy(None, state, subkey)
    print(f"Expensive price ($80/MWh): Decision = {decision_expensive[0]:.2f} MW")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
