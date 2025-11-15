"""Tests for Energy Storage problem (JAX-native implementation)."""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from problems.energy_storage import (
    EnergyStorageModel,
    EnergyStorageConfig,
    ThresholdPolicy,
    ThresholdPolicyConfig,
    TimeOfDayPolicy,
    MyopicPolicy,
    LinearPolicy,
    NeuralPolicy,
    AlwaysHoldPolicy,
)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_default_values() -> None:
    """Test that config has reasonable defaults."""
    config = EnergyStorageConfig()

    assert config.capacity == 1000.0
    assert config.max_charge_rate == 100.0
    assert config.efficiency == 0.95
    assert config.degradation_rate == 0.001
    assert config.initial_energy == 500.0
    assert config.min_energy == 0.0


def test_config_validation_positive() -> None:
    """Test that config validates positive values."""
    with pytest.raises(ValueError, match="capacity must be positive"):
        EnergyStorageConfig(capacity=-100.0)

    with pytest.raises(ValueError, match="max_charge_rate must be positive"):
        EnergyStorageConfig(max_charge_rate=0.0)


def test_config_validation_bounds() -> None:
    """Test that config validates bounded values."""
    with pytest.raises(ValueError, match="efficiency must be in"):
        EnergyStorageConfig(efficiency=1.5)

    with pytest.raises(ValueError, match="efficiency must be in"):
        EnergyStorageConfig(efficiency=0.0)

    with pytest.raises(ValueError, match="degradation_rate must be in"):
        EnergyStorageConfig(degradation_rate=-0.1)

    with pytest.raises(ValueError, match="degradation_rate must be in"):
        EnergyStorageConfig(degradation_rate=1.5)


def test_config_validation_initial_energy() -> None:
    """Test that initial energy doesn't exceed capacity."""
    with pytest.raises(ValueError, match="initial_energy.*cannot exceed capacity"):
        EnergyStorageConfig(capacity=1000.0, initial_energy=1500.0)


def test_threshold_policy_config_validation() -> None:
    """Test threshold policy config validation."""
    # Valid config
    config = ThresholdPolicyConfig(buy_threshold=40.0, sell_threshold=60.0)
    assert config.buy_threshold == 40.0

    # Invalid: sell_threshold <= buy_threshold
    with pytest.raises(ValueError, match="sell_threshold.*must be >"):
        ThresholdPolicyConfig(buy_threshold=60.0, sell_threshold=40.0)

    # Invalid rates
    with pytest.raises(ValueError, match="charge_rate must be in"):
        ThresholdPolicyConfig(charge_rate=1.5)

    with pytest.raises(ValueError, match="discharge_rate must be in"):
        ThresholdPolicyConfig(discharge_rate=0.0)


# ============================================================================
# Model Tests
# ============================================================================


def test_model_initialization() -> None:
    """Test model initializes correctly."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)

    assert model.config == config
    assert isinstance(model, EnergyStorageModel)


def test_init_state() -> None:
    """Test initial state generation."""
    config = EnergyStorageConfig(initial_energy=500.0)
    model = EnergyStorageModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # Check shape
    assert state.shape == (3,)

    # Check values: [energy, cycles, time]
    assert state[0] == 500.0  # Initial energy
    assert state[1] == 0.0    # Initial cycles
    assert 0 <= state[2] < 24  # Time of day


def test_sample_exogenous() -> None:
    """Test exogenous info sampling."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    key_exog = jax.random.PRNGKey(123)
    exog = model.sample_exogenous(key_exog, state, 0)

    # Check that price is reasonable and positive
    assert exog.price >= 0.0
    assert isinstance(exog.price, jnp.ndarray) or isinstance(exog.price, float)

    # Check that demand and renewable are also present
    assert exog.demand >= 0.0
    assert exog.renewable >= 0.0


def test_get_feasible_bounds() -> None:
    """Test feasible bounds calculation."""
    config = EnergyStorageConfig(
        capacity=1000.0,
        max_charge_rate=100.0,
        efficiency=0.9
    )
    model = EnergyStorageModel(config)

    # Test when battery is empty
    state_empty = jnp.array([0.0, 0.0, 12.0])
    min_charge, max_charge = model.get_feasible_bounds(state_empty)
    assert min_charge == 0.0  # Can't discharge when empty (returns -0)
    assert max_charge > 0.0  # Can charge

    # Test when battery is full
    state_full = jnp.array([1000.0, 0.0, 12.0])
    min_charge, max_charge = model.get_feasible_bounds(state_full)
    assert min_charge < 0.0  # Can discharge (negative value)
    assert max_charge == 0.0  # Can't charge more when full

    # Test when battery is half full
    state_half = jnp.array([500.0, 0.0, 12.0])
    min_charge, max_charge = model.get_feasible_bounds(state_half)
    assert min_charge < 0.0  # Can discharge (negative value)
    assert max_charge > 0.0  # Can charge


def test_transition_charging() -> None:
    """Test state transition when charging."""
    config = EnergyStorageConfig(
        capacity=1000.0,
        max_charge_rate=100.0,
        efficiency=0.9
    )
    model = EnergyStorageModel(config)
    key = jax.random.PRNGKey(42)

    # Start with half-full battery
    state = jnp.array([500.0, 10.0, 12.0])

    # Charge at 100 MW
    decision = jnp.array([100.0])
    exog = model.sample_exogenous(key, state, 0)

    next_state = model.transition(state, decision, exog)

    # Energy should increase (with efficiency loss and small degradation)
    # Expected: 500 + 100*0.9 - small degradation
    assert next_state[0] > state[0]  # Energy increased
    assert next_state[0] < 500.0 + 100.0 * 0.9  # But less than ideal due to degradation

    # Cycles should increase
    assert next_state[1] > state[1]

    # Time should advance
    assert next_state[2] == 13.0


def test_transition_discharging() -> None:
    """Test state transition when discharging."""
    config = EnergyStorageConfig(
        capacity=1000.0,
        max_charge_rate=100.0,
        efficiency=0.9
    )
    model = EnergyStorageModel(config)
    key = jax.random.PRNGKey(42)

    # Start with full battery
    state = jnp.array([1000.0, 10.0, 12.0])

    # Discharge at -100 MW
    decision = jnp.array([-100.0])
    exog = model.sample_exogenous(key, state, 0)

    next_state = model.transition(state, decision, exog)

    # Energy should decrease (with efficiency loss and degradation)
    # Expected: 1000 - 100/0.9 - degradation
    assert next_state[0] < state[0]  # Energy decreased
    assert next_state[0] < 1000.0 - 100.0 / 0.9  # Further reduced by degradation

    # Cycles should increase
    assert next_state[1] > state[1]


def test_reward_charging() -> None:
    """Test reward calculation when charging (paying for electricity)."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)

    state = jnp.array([500.0, 10.0, 12.0])
    decision = jnp.array([100.0])  # Charge 100 MW

    # Create exogenous info with known price
    from problems.energy_storage.model import ExogenousInfo
    exog = ExogenousInfo(price=jnp.array(50.0), demand=jnp.array(100.0), renewable=jnp.array(80.0))

    reward = model.reward(state, decision, exog)

    # Charging costs money (negative reward)
    assert reward < 0.0  # Should be negative (cost)


def test_reward_discharging() -> None:
    """Test reward calculation when discharging (earning from electricity)."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)

    state = jnp.array([1000.0, 10.0, 12.0])
    decision = jnp.array([-100.0])  # Discharge 100 MW

    from problems.energy_storage.model import ExogenousInfo
    exog = ExogenousInfo(price=jnp.array(50.0), demand=jnp.array(100.0), renewable=jnp.array(80.0))

    reward = model.reward(state, decision, exog)

    # Discharging earns money
    assert reward > 0.0  # Should be positive (revenue)


def test_jit_compilation() -> None:
    """Test that model methods can be JIT compiled."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)
    key = jax.random.PRNGKey(42)

    # JIT compile the methods
    init_state_jit = jax.jit(model.init_state)

    # Should not raise error
    state = init_state_jit(key)
    assert state.shape == (3,)


def test_vmap_batching() -> None:
    """Test that model works with vmap for batching."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)

    # Create batch of states
    batch_size = 10
    batch_keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

    # Vmap over init_state
    batch_init = jax.vmap(model.init_state)
    batch_states = batch_init(batch_keys)

    assert batch_states.shape == (batch_size, 3)


# ============================================================================
# Policy Tests
# ============================================================================


def test_threshold_policy() -> None:
    """Test threshold policy logic."""
    config_model = EnergyStorageConfig()
    model = EnergyStorageModel(config_model)

    config_policy = ThresholdPolicyConfig(
        buy_threshold=40.0,
        sell_threshold=60.0,
        charge_rate=0.5,
        discharge_rate=0.5
    )
    policy = ThresholdPolicy(model, config_policy)

    key = jax.random.PRNGKey(42)
    state = jnp.array([500.0, 10.0, 12.0])

    # Should return a valid decision
    decision = policy(None, state, key)
    assert decision.shape == (1,)


def test_time_of_day_policy() -> None:
    """Test time-of-day policy logic."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)

    policy = TimeOfDayPolicy(
        model,
        peak_start=9,
        peak_end=20,
        charge_rate=0.5,
        discharge_rate=0.5
    )

    key = jax.random.PRNGKey(42)

    # Off-peak: should charge
    state_offpeak = jnp.array([500.0, 10.0, 3.0])  # 3 AM
    decision_offpeak = policy(None, state_offpeak, key)
    assert decision_offpeak[0] > 0  # Positive = charging

    # Peak: should discharge
    state_peak = jnp.array([500.0, 10.0, 15.0])  # 3 PM
    decision_peak = policy(None, state_peak, key)
    assert decision_peak[0] < 0  # Negative = discharging


def test_myopic_policy() -> None:
    """Test myopic policy."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)

    policy = MyopicPolicy(model, n_samples=5)

    key = jax.random.PRNGKey(42)
    state = jnp.array([500.0, 10.0, 12.0])

    decision = policy(None, state, key)
    assert decision.shape == (1,)


def test_linear_policy() -> None:
    """Test linear policy."""
    policy = LinearPolicy(rngs=nnx.Rngs(42))

    key = jax.random.PRNGKey(42)
    state = jnp.array([500.0, 10.0, 12.0])
    price = 50.0

    decision = policy(state, price, key)
    assert decision.shape == (1,)

    # Check that weights exist
    assert hasattr(policy, 'weights')
    assert policy.weights.value.shape == (4,)


def test_neural_policy() -> None:
    """Test neural network policy."""
    policy = NeuralPolicy(hidden_dims=[16, 8], rngs=nnx.Rngs(42))

    key = jax.random.PRNGKey(42)
    state = jnp.array([500.0, 10.0, 12.0])
    price = 50.0

    decision = policy(state, price, key)
    assert decision.shape == (1,)

    # Check that network has layers
    assert len(policy.layers) == 3  # 2 hidden + 1 output


def test_always_hold_policy() -> None:
    """Test baseline always-hold policy."""
    policy = AlwaysHoldPolicy()

    key = jax.random.PRNGKey(42)
    state = jnp.array([500.0, 10.0, 12.0])

    decision = policy(None, state, key)
    assert decision[0] == 0.0  # Should always be zero


def test_policy_jit_compilation() -> None:
    """Test that policies can be JIT compiled."""
    config_model = EnergyStorageConfig()
    model = EnergyStorageModel(config_model)
    config_policy = ThresholdPolicyConfig()
    policy = ThresholdPolicy(model, config_policy)

    key = jax.random.PRNGKey(42)
    state = jnp.array([500.0, 10.0, 12.0])

    # JIT compile
    policy_jit = jax.jit(lambda s, k: policy(None, s, k))

    decision = policy_jit(state, key)
    assert decision.shape == (1,)


def test_policy_gradient_flow() -> None:
    """Test that learnable policies allow gradient flow."""
    policy = LinearPolicy(rngs=nnx.Rngs(42))

    state = jnp.array([500.0, 10.0, 12.0])
    price = 50.0
    key = jax.random.PRNGKey(42)

    # Define a simple loss
    def loss_fn(model: LinearPolicy) -> jax.Array:
        decision = model(state, price, key)
        return jnp.sum(decision ** 2)

    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(policy)

    # Check that we got gradients
    assert 'weights' in grads
    assert grads['weights'].value.shape == policy.weights.value.shape


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_episode() -> None:
    """Test running a complete episode."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)

    config_policy = ThresholdPolicyConfig()
    policy = ThresholdPolicy(model, config_policy)

    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    total_reward = 0.0
    horizon = 24  # Run for 24 hours

    for t in range(horizon):
        # Get decision
        key, subkey = jax.random.split(key)
        decision = policy(None, state, subkey)

        # Sample exogenous info
        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, t)

        # Get reward
        reward = model.reward(state, decision, exog)
        total_reward += float(reward)

        # Transition
        state = model.transition(state, decision, exog)

    # Should have accumulated some reward (positive or negative)
    assert isinstance(total_reward, float)


def test_multiple_policies_comparison() -> None:
    """Test that we can run multiple policies and compare them."""
    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)

    # Create different policies
    policies = {
        'threshold': ThresholdPolicy(model, ThresholdPolicyConfig()),
        'time_of_day': TimeOfDayPolicy(model),
        'hold': AlwaysHoldPolicy(),
    }

    key = jax.random.PRNGKey(42)

    results = {}

    for name, policy in policies.items():
        state = model.init_state(key)
        total_reward = 0.0

        for t in range(5):  # Just run a few steps
            key, subkey = jax.random.split(key)
            decision = policy(None, state, subkey)

            key, subkey = jax.random.split(key)
            exog = model.sample_exogenous(subkey, state, t)

            reward = model.reward(state, decision, exog)
            total_reward += float(reward)

            state = model.transition(state, decision, exog)

        results[name] = total_reward

    # All policies should produce results
    assert len(results) == 3
    assert all(isinstance(r, float) for r in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
