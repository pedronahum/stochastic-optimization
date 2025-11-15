# JAX-Native Stochastic Optimization - Modernization Plan

## Overview

This plan transforms the repository into a **JAX-native** library, leveraging JAX's functional programming paradigm, JIT compilation, automatic differentiation, and GPU/TPU acceleration.

## Why JAX-Native?

### Key Advantages

1. **GPU/TPU Acceleration**: Seamless device placement and computation
2. **JIT Compilation**: 10-100x speedups via `@jax.jit`
3. **Automatic Differentiation**: Built-in `jax.grad` for policy optimization
4. **Vectorization**: `jax.vmap` for batch operations
5. **Functional Programming**: Pure functions enable reproducibility
6. **Type Safety**: `chex` and `jaxtyping` for runtime type checking
7. **Ecosystem**: Integrates with Flax, Optax, Haiku, Equinox

### Core Libraries

```toml
[project.dependencies]
jax = ">=0.4.20"
jaxlib = ">=0.4.20"
chex = ">=0.1.85"
jaxtyping = ">=0.2.25"
flax = ">=0.8.0"
optax = ">=0.1.9"
equinox = ">=0.11.3"
```

## Type System for JAX

### Primary Type Libraries

#### 1. **jaxtyping** - Comprehensive Array Typing

```python
from jaxtyping import Array, Float, Int, Bool, PyTree, PRNGKeyArray
import jax.numpy as jnp

# Shaped array types
State = Float[Array, "state_dim"]  # 1D float array
StateBatch = Float[Array, "batch state_dim"]  # 2D batch
Decision = Float[Array, "action_dim"]
Reward = Float[Array, ""]  # Scalar

# Examples
def transition(
    state: Float[Array, "n"],
    action: Float[Array, "m"],
    key: PRNGKeyArray,
) -> Float[Array, "n"]:
    """State transition with shape checking."""
    ...

# PyTree for complex structures
PolicyParams = PyTree  # Arbitrary nested structure
```

#### 2. **chex** - Runtime Assertions and Testing

```python
import chex

@chex.assert_max_traces(n=1)  # Ensure JIT compiles once
@chex.dataclass(frozen=True)  # Immutable dataclass
class Config:
    capacity: float
    efficiency: float
    
    def __post_init__(self):
        chex.assert_scalar_positive(self.capacity)
        chex.assert_scalar_in_range(self.efficiency, 0.0, 1.0)

def model_step(
    state: Float[Array, "n"],
    action: Float[Array, "m"],
) -> Float[Array, "n"]:
    # Runtime shape assertions
    chex.assert_rank(state, 1)
    chex.assert_shape(action, (m,))
    chex.assert_tree_all_finite((state, action))
    ...
```

#### 3. **Type Annotations Strategy**

```python
from typing import Protocol, Callable
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
import jax

# State as array
State = Float[Array, "state_dim"]

# Decision/Action
Decision = Float[Array, "action_dim"]

# Reward (scalar or array)
Reward = Float[Array, ""] | float

# Random key
Key = PRNGKeyArray

# Transition function type
TransitionFn = Callable[
    [State, Decision, Key],
    State
]

# Policy function type
PolicyFn = Callable[
    [PyTree, State, Key],  # params, state, key
    Decision
]
```

## JAX-Native Base Protocols

```python
"""Base protocols for JAX-native stochastic optimization."""

from typing import Protocol, Any, NamedTuple
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray
import jax
import jax.numpy as jnp
import chex

# Type aliases
State = Float[Array, "state_dim"]
Decision = Float[Array, "action_dim"]
Reward = Float[Array, ""]
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information as immutable NamedTuple.
    
    JAX prefers NamedTuples or dataclasses for pytree registration.
    """
    price: Float[Array, ""]
    demand: Float[Array, ""]
    renewable: Float[Array, ""]
    
    @classmethod
    def sample(cls, key: Key, time: int) -> 'ExogenousInfo':
        """Sample exogenous information."""
        ...


@chex.dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for models.
    
    Using chex.dataclass for automatic pytree registration
    and immutability enforcement.
    """
    capacity: float
    max_charge_rate: float
    efficiency: float
    
    def __post_init__(self):
        """Validate configuration."""
        chex.assert_scalar_positive(self.capacity)
        chex.assert_scalar_positive(self.max_charge_rate)
        chex.assert_scalar_in_range(self.efficiency, 0.0, 1.0)


class Model(Protocol):
    """Protocol for JAX-native sequential decision models.
    
    All methods should be pure functions (no side effects).
    Methods can be JIT-compiled for performance.
    """
    
    config: ModelConfig
    
    def init_state(self, key: Key) -> State:
        """Initialize state (pure function).
        
        Args:
            key: JAX random key for stochastic initialization.
        
        Returns:
            Initial state vector.
        """
        ...
    
    def transition(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> State:
        """Compute next state (pure function, JIT-compilable).
        
        Args:
            state: Current state.
            decision: Decision taken.
            exog: Exogenous information.
        
        Returns:
            Next state.
        """
        ...
    
    def reward(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> Reward:
        """Compute reward (pure function, JIT-compilable).
        
        Args:
            state: Current state.
            decision: Decision taken.
            exog: Exogenous information.
        
        Returns:
            Scalar reward.
        """
        ...
    
    def sample_exogenous(self, key: Key, time: int) -> ExogenousInfo:
        """Sample exogenous information (pure function).
        
        Args:
            key: JAX random key.
            time: Current time step.
        
        Returns:
            Sampled exogenous information.
        """
        ...
    
    def is_valid_decision(
        self,
        state: State,
        decision: Decision,
    ) -> Bool[Array, ""]:
        """Check if decision is valid (pure function).
        
        Args:
            state: Current state.
            decision: Proposed decision.
        
        Returns:
            Boolean indicating validity.
        """
        ...


# Batch operations using vmap
def batch_transition(
    model: Model,
    states: Float[Array, "batch state_dim"],
    decisions: Float[Array, "batch action_dim"],
    exogs: ExogenousInfo,  # Batched
) -> Float[Array, "batch state_dim"]:
    """Vectorized state transitions.
    
    Uses jax.vmap for efficient batch processing.
    """
    return jax.vmap(model.transition)(states, decisions, exogs)


# JIT-compiled simulation
@jax.jit
def simulate_step(
    model: Model,
    state: State,
    policy_params: PyTree,
    policy_fn: PolicyFn,
    key: Key,
    time: int,
) -> tuple[State, Reward, ExogenousInfo]:
    """Single simulation step (JIT-compiled).
    
    Args:
        model: Problem model.
        state: Current state.
        policy_params: Policy parameters.
        policy_fn: Policy function.
        key: Random key.
        time: Current time.
    
    Returns:
        Tuple of (next_state, reward, exogenous_info).
    """
    # Split key for different random operations
    key_exog, key_policy = jax.random.split(key)
    
    # Sample exogenous info
    exog = model.sample_exogenous(key_exog, time)
    
    # Get decision from policy
    decision = policy_fn(policy_params, state, key_policy)
    
    # Compute reward and transition
    reward = model.reward(state, decision, exog)
    next_state = model.transition(state, decision, exog)
    
    return next_state, reward, exog


# Scan for efficient trajectory rollout
def simulate_trajectory(
    model: Model,
    policy_params: PyTree,
    policy_fn: PolicyFn,
    key: Key,
    horizon: int,
) -> dict[str, Array]:
    """Simulate full trajectory using jax.lax.scan.
    
    Args:
        model: Problem model.
        policy_params: Policy parameters.
        policy_fn: Policy function.
        key: Initial random key.
        horizon: Number of time steps.
    
    Returns:
        Dictionary with states, decisions, rewards.
    """
    
    def step_fn(carry, t):
        state, key = carry
        key, subkey = jax.random.split(key)
        
        next_state, reward, exog = simulate_step(
            model, state, policy_params, policy_fn, subkey, t
        )
        
        outputs = {
            'state': state,
            'reward': reward,
            'exog': exog,
        }
        
        return (next_state, key), outputs
    
    # Initialize
    init_key, key = jax.random.split(key)
    init_state = model.init_state(init_key)
    
    # Scan over time steps
    _, trajectory = jax.lax.scan(
        step_fn,
        (init_state, key),
        jnp.arange(horizon)
    )
    
    return trajectory
```

## JAX-Native Energy Storage Model

```python
"""JAX-native Energy Storage Model."""

from typing import NamedTuple
from jaxtyping import Array, Float, PRNGKeyArray
import jax
import jax.numpy as jnp
import chex

# Type aliases
State = Float[Array, "3"]  # [energy, cycles, time_of_day]
Decision = Float[Array, "1"]  # [charge_power]
Reward = Float[Array, ""]
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information for energy storage."""
    price: Float[Array, ""]
    demand: Float[Array, ""]
    renewable: Float[Array, ""]


@chex.dataclass(frozen=True)
class EnergyStorageConfig:
    """Configuration for energy storage model.
    
    Immutable dataclass that's automatically a pytree.
    """
    capacity: float = 1000.0
    max_charge_rate: float = 100.0
    max_discharge_rate: float = 100.0
    efficiency: float = 0.95
    initial_energy: float = 500.0
    degradation_rate: float = 0.001
    min_energy: float = 0.0
    
    def __post_init__(self):
        """Validate configuration."""
        chex.assert_scalar_positive(self.capacity)
        chex.assert_scalar_positive(self.max_charge_rate)
        chex.assert_scalar_positive(self.max_discharge_rate)
        chex.assert_scalar_in_range(self.efficiency, 0.0, 1.0)
        chex.assert_scalar_in_range(
            self.initial_energy, 0.0, self.capacity
        )


class EnergyStorageModel:
    """JAX-native energy storage model.
    
    All methods are pure functions that can be JIT-compiled.
    State is immutable - each transition returns a new state.
    """
    
    def __init__(self, config: EnergyStorageConfig):
        """Initialize model with configuration.
        
        Args:
            config: Model configuration.
        """
        self.config = config
    
    def init_state(self, key: Key) -> State:
        """Initialize state.
        
        Args:
            key: Random key (unused but kept for interface).
        
        Returns:
            Initial state [energy, cycles, time].
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
            decision: Charge power (positive=charge, negative=discharge).
            exog: Exogenous information.
        
        Returns:
            Next state.
        """
        energy, cycles, time_of_day = state[0], state[1], state[2]
        charge_power = decision[0]
        
        # Energy change with efficiency
        energy_change = jnp.where(
            charge_power > 0,
            charge_power * self.config.efficiency,  # Charging
            charge_power / self.config.efficiency,  # Discharging
        )
        
        # Degradation
        cycles_this_step = jnp.abs(charge_power) / (2 * self.config.capacity)
        degradation = self.config.degradation_rate * cycles_this_step * energy
        
        # Update energy with clipping
        new_energy = jnp.clip(
            energy + energy_change - degradation,
            self.config.min_energy,
            self.config.capacity,
        )
        
        # Update cycles
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
            Scalar reward (profit).
        """
        charge_power = decision[0]
        price = exog.price
        
        # Revenue from discharge or cost of charge
        revenue = jnp.where(
            charge_power > 0,
            -(charge_power / self.config.efficiency) * price,  # Cost
            (-charge_power * self.config.efficiency) * price,  # Revenue
        )
        
        # Degradation cost
        cycles_this_step = jnp.abs(charge_power) / (2 * self.config.capacity)
        degradation_cost = cycles_this_step * 1000.0
        
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
        
        # Time-of-day effects
        peak_hours = (9 <= hour) & (hour <= 20)
        price_mult = jnp.where(peak_hours, 1.3, 0.8)
        demand_mult = jnp.where(peak_hours, 1.2, 0.7)
        solar_mult = jnp.maximum(0.0, jnp.sin(jnp.pi * (hour - 6) / 12))
        
        # Split key for each random sample
        key_price, key_demand, key_renewable = jax.random.split(key, 3)
        
        # Sample with JAX random
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
        valid_discharge_rate = -charge_power <= self.config.max_discharge_rate
        
        # Check energy constraints
        energy_change = jnp.where(
            charge_power > 0,
            charge_power * self.config.efficiency,
            -charge_power / self.config.efficiency,
        )
        new_energy = energy + energy_change
        
        valid_capacity = new_energy <= self.config.capacity
        valid_minimum = new_energy >= self.config.min_energy
        
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
            Tuple of (min_charge, max_charge).
        """
        energy = float(state[0])
        
        # Max charge
        energy_room = self.config.capacity - energy
        max_charge_capacity = energy_room / self.config.efficiency
        max_charge = min(self.config.max_charge_rate, max_charge_capacity)
        
        # Max discharge
        available_energy = energy - self.config.min_energy
        max_discharge_energy = available_energy * self.config.efficiency
        max_discharge = min(self.config.max_discharge_rate, max_discharge_energy)
        
        return (-max_discharge, max_charge)


# Example usage with JIT compilation
if __name__ == "__main__":
    import time
    
    # Create model
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)
    
    # Initialize
    key = jax.random.PRNGKey(0)
    state = model.init_state(key)
    
    print("Testing JAX-native Energy Storage Model")
    print("=" * 50)
    print(f"Initial state: {state}")
    
    # Sample exogenous info
    key, subkey = jax.random.split(key)
    exog = model.sample_exogenous(subkey, time=12)
    print(f"\nExogenous info: {exog}")
    
    # Test transition (first call compiles)
    decision = jnp.array([50.0])
    
    start = time.time()
    next_state = model.transition(state, decision, exog)
    compile_time = time.time() - start
    print(f"\nFirst call (with compilation): {compile_time:.4f}s")
    print(f"Next state: {next_state}")
    
    # Second call uses compiled version
    start = time.time()
    next_state = model.transition(state, decision, exog)
    run_time = time.time() - start
    print(f"Second call (compiled): {run_time:.6f}s")
    print(f"Speedup: {compile_time/run_time:.1f}x")
    
    # Test reward
    reward = model.reward(state, decision, exog)
    print(f"\nReward: ${float(reward):.2f}")
    
    # Test validation
    valid = model.is_valid_decision(state, decision)
    print(f"Valid decision: {bool(valid)}")
    
    # Batch test with vmap
    print("\n" + "=" * 50)
    print("Testing batch operations with vmap")
    
    batch_size = 1000
    states = jnp.repeat(state[None, :], batch_size, axis=0)
    decisions = jnp.linspace(-50, 50, batch_size)[:, None]
    
    # Vectorize transition function
    batch_transition = jax.vmap(
        lambda s, d: model.transition(s, d, exog)
    )
    
    start = time.time()
    batch_next_states = batch_transition(states, decisions)
    batch_time = time.time() - start
    
    print(f"Batch size: {batch_size}")
    print(f"Batch time: {batch_time:.6f}s")
    print(f"Per-sample time: {batch_time/batch_size*1e6:.2f}μs")
```

## Testing with JAX and Chex

```python
"""Tests for JAX-native energy storage model."""

import pytest
import jax
import jax.numpy as jnp
import chex
from hypothesis import given, strategies as st

from energy_storage_model import (
    EnergyStorageModel,
    EnergyStorageConfig,
    ExogenousInfo,
)


class TestEnergyStorageConfig:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test creating valid configuration."""
        config = EnergyStorageConfig(
            capacity=1000.0,
            max_charge_rate=100.0,
            efficiency=0.95,
        )
        assert config.capacity == 1000.0
        
        # chex assertions
        chex.assert_scalar_positive(config.capacity)
    
    def test_immutability(self):
        """Test that config is immutable."""
        config = EnergyStorageConfig()
        
        with pytest.raises(AttributeError):
            config.capacity = 2000.0  # Should fail - frozen


class TestEnergyStorageModel:
    """Tests for energy storage model."""
    
    @pytest.fixture
    def config(self):
        """Provide standard config."""
        return EnergyStorageConfig()
    
    @pytest.fixture
    def model(self, config):
        """Provide model instance."""
        return EnergyStorageModel(config)
    
    @pytest.fixture
    def key(self):
        """Provide JAX random key."""
        return jax.random.PRNGKey(42)
    
    def test_init_state(self, model, key):
        """Test state initialization."""
        state = model.init_state(key)
        
        # chex shape assertions
        chex.assert_rank(state, 1)
        chex.assert_shape(state, (3,))
        
        # Check values
        assert state[0] == model.config.initial_energy
        assert state[1] == 0.0  # No cycles
        assert state[2] == 0.0  # Time starts at 0
    
    def test_transition_shape(self, model, key):
        """Test that transition preserves shape."""
        state = model.init_state(key)
        decision = jnp.array([50.0])
        exog = ExogenousInfo(
            price=jnp.array(50.0),
            demand=jnp.array(100.0),
            renewable=jnp.array(80.0),
        )
        
        next_state = model.transition(state, decision, exog)
        
        # Shape should be preserved
        chex.assert_equal_shape([state, next_state])
        chex.assert_tree_all_finite(next_state)
    
    def test_transition_charging(self, model, key):
        """Test charging transition."""
        state = model.init_state(key)
        decision = jnp.array([50.0])  # Charge
        exog = ExogenousInfo(
            jnp.array(30.0), jnp.array(100.0), jnp.array(150.0)
        )
        
        next_state = model.transition(state, decision, exog)
        
        # Energy should increase
        assert next_state[0] > state[0]
        
        # Should be approximately: initial + 50 * 0.95
        expected = state[0] + 50.0 * 0.95
        chex.assert_trees_all_close(
            next_state[0], expected, rtol=0.01
        )
    
    def test_transition_is_jittable(self, model, key):
        """Test that transition can be JIT-compiled."""
        state = model.init_state(key)
        decision = jnp.array([50.0])
        exog = ExogenousInfo(
            jnp.array(50.0), jnp.array(100.0), jnp.array(80.0)
        )
        
        # Should not raise error
        jitted_transition = jax.jit(model.transition)
        next_state = jitted_transition(state, decision, exog)
        
        chex.assert_tree_all_finite(next_state)
    
    def test_reward_shape(self, model, key):
        """Test reward returns scalar."""
        state = model.init_state(key)
        decision = jnp.array([50.0])
        exog = ExogenousInfo(
            jnp.array(50.0), jnp.array(100.0), jnp.array(80.0)
        )
        
        reward = model.reward(state, decision, exog)
        
        # Should be scalar
        chex.assert_rank(reward, 0)
        chex.assert_scalar(reward)
    
    def test_is_valid_decision_returns_bool(self, model, key):
        """Test validation returns boolean."""
        state = model.init_state(key)
        decision = jnp.array([50.0])
        
        valid = model.is_valid_decision(state, decision)
        
        # Should be boolean scalar
        chex.assert_type(valid, bool)
    
    def test_batch_with_vmap(self, model, key):
        """Test batching with vmap."""
        # Create batch of states
        batch_size = 100
        key, *subkeys = jax.random.split(key, batch_size + 1)
        states = jax.vmap(model.init_state)(jnp.array(subkeys))
        
        # Batch of decisions
        decisions = jnp.linspace(-50, 50, batch_size)[:, None]
        
        # Single exogenous info (broadcast)
        exog = ExogenousInfo(
            jnp.array(50.0), jnp.array(100.0), jnp.array(80.0)
        )
        
        # Vectorize transition
        batch_transition = jax.vmap(
            lambda s, d: model.transition(s, d, exog)
        )
        
        next_states = batch_transition(states, decisions)
        
        # Check shapes
        chex.assert_shape(next_states, (batch_size, 3))
        chex.assert_tree_all_finite(next_states)


class TestEnergyStorageProperties:
    """Property-based tests with chex."""
    
    @pytest.fixture
    def model(self):
        """Provide model instance."""
        config = EnergyStorageConfig()
        return EnergyStorageModel(config)
    
    def test_energy_bounds_invariant(self, model):
        """Test energy stays within bounds."""
        key = jax.random.PRNGKey(0)
        state = model.init_state(key)
        
        # Random decision
        key, subkey = jax.random.split(key)
        decision = jax.random.uniform(subkey, (1,), minval=-100, maxval=100)
        
        exog = ExogenousInfo(
            jnp.array(50.0), jnp.array(100.0), jnp.array(80.0)
        )
        
        # Only test if decision is valid
        if model.is_valid_decision(state, decision):
            next_state = model.transition(state, decision, exog)
            
            # Energy must be in bounds
            assert 0.0 <= next_state[0] <= model.config.capacity
    
    def test_time_wraps_at_24(self, model):
        """Test time wraps around at 24."""
        key = jax.random.PRNGKey(0)
        state = jnp.array([500.0, 0.0, 23.0])  # At hour 23
        decision = jnp.array([0.0])
        exog = ExogenousInfo(
            jnp.array(50.0), jnp.array(100.0), jnp.array(80.0)
        )
        
        next_state = model.transition(state, decision, exog)
        
        # Time should wrap to 0
        chex.assert_trees_all_close(next_state[2], 0.0)
    
    def test_gradient_flow(self, model):
        """Test that gradients flow through model."""
        key = jax.random.PRNGKey(0)
        state = model.init_state(key)
        exog = ExogenousInfo(
            jnp.array(50.0), jnp.array(100.0), jnp.array(80.0)
        )
        
        # Define loss as negative reward
        def loss_fn(decision):
            return -model.reward(state, decision, exog)
        
        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        decision = jnp.array([50.0])
        grad = grad_fn(decision)
        
        # Gradient should exist and be finite
        chex.assert_tree_all_finite(grad)
        chex.assert_shape(grad, (1,))


class TestEnergyStorageIntegration:
    """Integration tests for full workflows."""
    
    def test_full_trajectory_with_scan(self):
        """Test full trajectory simulation with jax.lax.scan."""
        config = EnergyStorageConfig()
        model = EnergyStorageModel(config)
        key = jax.random.PRNGKey(0)
        
        # Simple myopic policy
        def policy_fn(params, state, key):
            # Always charge at half rate
            return jnp.array([50.0])
        
        horizon = 24
        
        def step_fn(carry, t):
            state, key = carry
            key, subkey1, subkey2 = jax.random.split(key, 3)
            
            # Get decision
            decision = policy_fn(None, state, subkey1)
            
            # Sample exogenous
            exog = model.sample_exogenous(subkey2, int(t))
            
            # Transition
            reward = model.reward(state, decision, exog)
            next_state = model.transition(state, decision, exog)
            
            outputs = {'state': state, 'reward': reward}
            return (next_state, key), outputs
        
        # Run simulation
        init_state = model.init_state(key)
        _, trajectory = jax.lax.scan(
            step_fn,
            (init_state, key),
            jnp.arange(horizon)
        )
        
        # Check outputs
        assert trajectory['state'].shape == (horizon, 3)
        assert trajectory['reward'].shape == (horizon,)
        
        # All values should be finite
        chex.assert_tree_all_finite(trajectory)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

## mypy Configuration for JAX

```ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
check_untyped_defs = True
disallow_untyped_defs = True
warn_redundant_casts = True
warn_unused_ignores = True
strict_equality = True

# JAX-specific
[mypy-jax.*]
ignore_missing_imports = False
follow_imports = skip

[mypy-jaxlib.*]
ignore_missing_imports = True

[mypy-chex.*]
ignore_missing_imports = False

[mypy-jaxtyping.*]
ignore_missing_imports = False

[mypy-optax.*]
ignore_missing_imports = False

[mypy-flax.*]
ignore_missing_imports = False

[mypy-equinox.*]
ignore_missing_imports = False
```

## pyproject.toml for JAX

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "stochastic-optimization-jax"
version = "2.0.0"
description = "JAX-Native Sequential Decision Problem Modeling Library"
requires-python = ">=3.10"
dependencies = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "chex>=0.1.85",
    "jaxtyping>=0.2.25",
    "optax>=0.1.9",
    "flax>=0.8.0",
    "equinox>=0.11.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.1",
    "hypothesis>=6.82.0",
    "mypy>=1.5.0",
    "ruff>=0.0.285",
    "black>=23.7.0",
]

cuda = [
    "jax[cuda12_pip]>=0.4.20",
]

docs = [
    "sphinx>=7.1.0",
    "sphinx-rtd-theme>=1.3.0",
]
```

## Key Advantages of JAX-Native Approach

### 1. Performance
- **JIT Compilation**: 10-100x speedup
- **GPU/TPU**: Seamless device placement
- **Vectorization**: Efficient batch operations

### 2. Gradients
- **Automatic Differentiation**: Built-in `jax.grad`
- **Policy Optimization**: Easy gradient-based learning
- **Sensitivity Analysis**: Automatic derivatives

### 3. Functional Programming
- **Pure Functions**: No side effects
- **Reproducibility**: Explicit random keys
- **Composability**: Easy to combine

### 4. Type Safety
- **jaxtyping**: Shape-aware type checking
- **chex**: Runtime assertions
- **mypy**: Static analysis

### 5. Ecosystem
- **Flax/Equinox**: Neural network libraries
- **Optax**: Optimizers
- **Haiku**: Another NN library option

## Migration from NumPy

| NumPy | JAX |
|-------|-----|
| `import numpy as np` | `import jax.numpy as jnp` |
| `np.array([1, 2, 3])` | `jnp.array([1, 2, 3])` |
| `np.random.randn()` | `jax.random.normal(key)` |
| `arr[i] = value` | `arr.at[i].set(value)` |
| `npt.NDArray` | `Float[Array, "..."]` |
| In-place ops | Functional updates |

## Summary

JAX-native provides:
✅ 10-100x performance gains
✅ GPU/TPU support
✅ Automatic differentiation
✅ Type-safe with jaxtyping + chex
✅ Functional, reproducible code
✅ Modern ML ecosystem integration

This is the future-proof choice for stochastic optimization!
