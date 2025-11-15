"""Tests for Blood Management problem (JAX-native implementation)."""

import pytest
import jax
import jax.numpy as jnp

from problems.blood_management import (
    BloodManagementConfig,
    BloodManagementModel,
    ExogenousInfo,
    GreedyPolicy,
    FIFOPolicy,
    RandomPolicy,
)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_default_values() -> None:
    """Test that config has reasonable defaults."""
    config = BloodManagementConfig()

    assert config.max_age == 5
    assert config.max_demand_urgent == 10.0
    assert config.surge_prob == 0.1


def test_config_validation() -> None:
    """Test that config validates parameters."""
    with pytest.raises(ValueError, match="max_age"):
        BloodManagementConfig(max_age=0)

    with pytest.raises(ValueError, match="surge_prob"):
        BloodManagementConfig(surge_prob=1.5)


def test_config_immutability() -> None:
    """Test that config is immutable."""
    config = BloodManagementConfig()

    with pytest.raises((AttributeError, Exception)):
        config.max_age = 10


# ============================================================================
# Model Tests
# ============================================================================


def test_model_initialization() -> None:
    """Test model initializes correctly."""
    config = BloodManagementConfig(max_age=5)
    model = BloodManagementModel(config)

    assert model.config == config
    assert model.n_blood_types == 8
    assert model.n_surgery_types == 2
    assert model.n_demand_types == 16  # 8 blood types × 2 surgery types
    assert model.n_inventory_slots == 40  # 8 × 5


def test_init_state_structure() -> None:
    """Test that init_state creates state with correct structure."""
    config = BloodManagementConfig(max_age=5)
    model = BloodManagementModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # State: [inventory (8*5=40), time (1)] = 41
    assert state.shape == (41,)

    # Time should be 0
    assert state[-1] == 0.0


def test_substitution_matrix() -> None:
    """Test that blood substitution rules are correct."""
    config = BloodManagementConfig()
    model = BloodManagementModel(config)

    # O- is universal donor (can substitute for all)
    assert jnp.all(model.substitution_matrix[0, :])

    # AB+ is universal recipient (can only receive AB+)
    assert model.substitution_matrix[7, 7]  # AB+ → AB+

    # A+ cannot substitute for A- (Rh incompatibility)
    assert not model.substitution_matrix[3, 2]  # A+ → A-


def test_sample_exogenous() -> None:
    """Test exogenous sampling."""
    config = BloodManagementConfig(max_age=5)
    model = BloodManagementModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    exog = model.sample_exogenous(key, state, 0)

    # Should have demands for all 16 types
    assert exog.demand.shape == (16,)

    # Should have donations for all 8 blood types
    assert exog.donation.shape == (8,)

    # All values should be non-negative
    assert jnp.all(exog.demand >= 0)
    assert jnp.all(exog.donation >= 0)


def test_transition_ages_blood() -> None:
    """Test that transition correctly ages blood."""
    config = BloodManagementConfig(max_age=3)
    model = BloodManagementModel(config)
    key = jax.random.PRNGKey(42)

    # Create state with known inventory
    inventory = jnp.array([
        # Blood type 0 (O-): [age0=5, age1=3, age2=1]
        5.0, 3.0, 1.0,
        # Other blood types: all zeros
        0.0, 0.0, 0.0,  # Type 1
        0.0, 0.0, 0.0,  # Type 2
        0.0, 0.0, 0.0,  # Type 3
        0.0, 0.0, 0.0,  # Type 4
        0.0, 0.0, 0.0,  # Type 5
        0.0, 0.0, 0.0,  # Type 6
        0.0, 0.0, 0.0,  # Type 7
    ])
    state = jnp.concatenate([inventory, jnp.array([0.0])])

    # Zero allocation, 10 units donated to type 0
    decision = jnp.zeros(24 * 16)  # 24 slots × 16 demands
    exog = ExogenousInfo(
        demand=jnp.zeros(16),
        donation=jnp.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    next_state = model.transition(state, decision, exog)
    next_inv = model.get_inventory(next_state)

    # After aging: age0=10 (donation), age1=5 (was age0), age2=3 (was age1)
    # age2 (was age2=1) expires
    assert next_inv[0, 0] == 10.0  # New donation
    assert next_inv[0, 1] == 5.0   # Aged from 0
    assert next_inv[0, 2] == 3.0   # Aged from 1
    # Oldest (was age2=1) is discarded


def test_transition_allocates_blood() -> None:
    """Test that transition applies allocations correctly."""
    config = BloodManagementConfig(max_age=3)
    model = BloodManagementModel(config)

    # Create state with inventory
    inventory = jnp.array([
        10.0, 5.0, 2.0,  # Type 0
        0.0, 0.0, 0.0,   # Types 1-7...
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])
    state = jnp.concatenate([inventory, jnp.array([0.0])])

    # Allocate 3 units from slot 0 (type 0, age 0) to demand 0
    decision = jnp.zeros(24 * 16)
    decision = decision.at[0].set(3.0)  # slot 0, demand 0

    exog = ExogenousInfo(
        demand=jnp.ones(16) * 5.0,
        donation=jnp.zeros(8)
    )

    next_state = model.transition(state, decision, exog)
    next_inv = model.get_inventory(next_state)

    # After allocation and aging:
    # - slot 0 had 10, allocated 3, leaving 7
    # - After aging, age1 = 7 (from age0)
    assert next_inv[0, 1] == 7.0


def test_reward_calculation() -> None:
    """Test reward calculation."""
    config = BloodManagementConfig(max_age=3)
    model = BloodManagementModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # Create allocation and exogenous info
    decision = jnp.ones(24 * 16) * 0.5  # Small allocations
    exog = ExogenousInfo(
        demand=jnp.ones(16) * 2.0,
        donation=jnp.ones(8) * 5.0
    )

    reward = model.reward(state, decision, exog)

    # Reward should be a scalar
    assert isinstance(float(reward), float)


def test_is_valid_decision() -> None:
    """Test decision validation."""
    config = BloodManagementConfig(max_age=3)
    model = BloodManagementModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # Valid decision: zeros
    decision = jnp.zeros(24 * 16)
    assert model.is_valid_decision(state, decision)

    # Invalid decision: negative allocation
    bad_decision = jnp.ones(24 * 16) * -1.0
    assert not model.is_valid_decision(state, bad_decision)


def test_get_inventory() -> None:
    """Test extracting inventory from state."""
    config = BloodManagementConfig(max_age=5)
    model = BloodManagementModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    inventory = model.get_inventory(state)

    assert inventory.shape == (8, 5)


def test_get_time() -> None:
    """Test extracting time from state."""
    config = BloodManagementConfig(max_age=5)
    model = BloodManagementModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    time = model.get_time(state)

    assert time == 0


def test_jit_compilation() -> None:
    """Test JIT compilation."""
    config = BloodManagementConfig(max_age=3)
    model = BloodManagementModel(config)
    key = jax.random.PRNGKey(42)

    init_jit = jax.jit(model.init_state)
    state = init_jit(key)

    assert state.shape == (25,)  # 8*3 + 1


# ============================================================================
# Policy Tests
# ============================================================================


def test_greedy_policy() -> None:
    """Test greedy policy."""
    config = BloodManagementConfig(max_age=3)
    model = BloodManagementModel(config)
    policy = GreedyPolicy()
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    decision = policy(None, state, key, model)

    assert decision.shape == (24 * 16,)
    assert jnp.all(decision >= 0)


def test_fifo_policy() -> None:
    """Test FIFO policy."""
    config = BloodManagementConfig(max_age=3)
    model = BloodManagementModel(config)
    policy = FIFOPolicy()
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    decision = policy(None, state, key, model)

    assert decision.shape == (24 * 16,)
    assert jnp.all(decision >= 0)


def test_random_policy() -> None:
    """Test random policy."""
    config = BloodManagementConfig(max_age=3)
    model = BloodManagementModel(config)
    policy = RandomPolicy()
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    decision = policy(None, state, key, model)

    assert decision.shape == (24 * 16,)
    assert jnp.all(decision >= 0)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_episode() -> None:
    """Test running a complete episode."""
    config = BloodManagementConfig(max_age=3, max_donation=5.0)
    model = BloodManagementModel(config)
    policy = GreedyPolicy()

    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    total_reward = 0.0
    n_steps = 10

    for t in range(n_steps):
        key, subkey = jax.random.split(key)
        decision = policy(None, state, subkey, model)

        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, t)

        reward = model.reward(state, decision, exog)
        total_reward += float(reward)

        state = model.transition(state, decision, exog)

    # Should complete without errors
    assert model.get_time(state) == n_steps


def test_inventory_dynamics() -> None:
    """Test that inventory evolves correctly over time."""
    config = BloodManagementConfig(max_age=3, max_donation=2.0)
    model = BloodManagementModel(config)
    policy = GreedyPolicy()

    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    initial_total = jnp.sum(model.get_inventory(state))

    # Run a few steps
    for t in range(5):
        key, subkey = jax.random.split(key)
        decision = policy(None, state, subkey, model)

        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, t)

        state = model.transition(state, decision, exog)

    final_total = jnp.sum(model.get_inventory(state))

    # Inventory should change (donations, allocations, expiry)
    # Can't predict exact value, but should be different
    assert final_total != initial_total or initial_total == 0


def test_multiple_policies() -> None:
    """Test that different policies work correctly."""
    config = BloodManagementConfig(max_age=3)
    model = BloodManagementModel(config)

    policies = {
        "greedy": GreedyPolicy(),
        "fifo": FIFOPolicy(),
        "random": RandomPolicy(),
    }

    key = jax.random.PRNGKey(42)

    for name, policy in policies.items():
        state = model.init_state(key)

        # Run a few steps
        for t in range(3):
            key, subkey = jax.random.split(key)
            decision = policy(None, state, subkey, model)

            # Decision should be valid
            assert model.is_valid_decision(state, decision)

            key, subkey = jax.random.split(key)
            exog = model.sample_exogenous(subkey, state, t)
            state = model.transition(state, decision, exog)


def test_blood_expiry() -> None:
    """Test that oldest blood expires correctly."""
    config = BloodManagementConfig(max_age=2)
    model = BloodManagementModel(config)

    # Create state with blood at oldest age
    inventory = jnp.zeros(8 * 2)
    inventory = inventory.at[1].set(5.0)  # Type 0, age 1 (oldest)
    state = jnp.concatenate([inventory, jnp.array([0.0])])

    # No allocation, no donation
    decision = jnp.zeros(16 * 16)
    exog = ExogenousInfo(
        demand=jnp.zeros(16),
        donation=jnp.zeros(8)
    )

    next_state = model.transition(state, decision, exog)
    next_inv = model.get_inventory(next_state)

    # Blood at age 1 should have expired (no slot for age 2)
    assert jnp.sum(next_inv) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
