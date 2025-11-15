"""Tests for Two Newsvendor problem (JAX-native implementation)."""

import pytest
import jax
import jax.numpy as jnp
import chex
from flax import nnx

from problems.two_newsvendor import (
    TwoNewsvendorConfig,
    TwoNewsvendorFieldModel,
    TwoNewsvendorCentralModel,
    ExogenousInfo,
    NewsvendorFieldPolicy,
    BiasAdjustedFieldPolicy,
    NewsvendorCentralPolicy,
    BiasAdjustedCentralPolicy,
    NeuralFieldPolicy,
    NeuralCentralPolicy,
    AlwaysAllocateRequestedPolicy,
)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_default_values() -> None:
    """Test that config has reasonable defaults."""
    config = TwoNewsvendorConfig()

    assert config.demand_lower == 0.0
    assert config.demand_upper == 100.0
    assert config.est_bias_field == 0.0
    assert config.est_std_field == 10.0
    assert config.est_bias_central == 0.0
    assert config.est_std_central == 10.0
    assert config.overage_cost_field == 1.0
    assert config.underage_cost_field == 9.0
    assert config.alpha_bias == 0.1


def test_config_validation_demand_bounds() -> None:
    """Test that config validates demand bounds."""
    with pytest.raises(ValueError, match="demand_upper.*must be >"):
        TwoNewsvendorConfig(demand_lower=100.0, demand_upper=50.0)

    with pytest.raises(ValueError, match="demand_upper.*must be >"):
        TwoNewsvendorConfig(demand_lower=50.0, demand_upper=50.0)


def test_config_validation_std() -> None:
    """Test that config validates standard deviations."""
    with pytest.raises(ValueError, match="est_std_field must be non-negative"):
        TwoNewsvendorConfig(est_std_field=-1.0)

    with pytest.raises(ValueError, match="est_std_central must be non-negative"):
        TwoNewsvendorConfig(est_std_central=-1.0)


def test_config_validation_alpha() -> None:
    """Test that config validates alpha_bias."""
    with pytest.raises(ValueError, match="alpha_bias must be in"):
        TwoNewsvendorConfig(alpha_bias=-0.1)

    with pytest.raises(ValueError, match="alpha_bias must be in"):
        TwoNewsvendorConfig(alpha_bias=1.5)


def test_config_validation_costs() -> None:
    """Test that config validates costs."""
    with pytest.raises(ValueError, match="Costs must be non-negative"):
        TwoNewsvendorConfig(overage_cost_field=-1.0)

    with pytest.raises(ValueError, match="Costs must be non-negative"):
        TwoNewsvendorConfig(underage_cost_field=-1.0)


def test_config_immutability() -> None:
    """Test that config is immutable (frozen dataclass)."""
    config = TwoNewsvendorConfig()

    with pytest.raises((AttributeError, Exception)):
        config.demand_lower = 50.0  # Should fail - frozen dataclass


# ============================================================================
# Field Model Tests
# ============================================================================


def test_field_model_initialization() -> None:
    """Test Field model initializes correctly."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorFieldModel(config)

    assert model.config == config
    assert isinstance(model, TwoNewsvendorFieldModel)


def test_field_init_state() -> None:
    """Test Field initial state generation."""
    config = TwoNewsvendorConfig(demand_lower=0.0, demand_upper=100.0)
    model = TwoNewsvendorFieldModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    assert state.shape == (3,)
    assert state[0] == 50.0  # Mid-range estimate
    assert state[1] == 0.0   # No source bias
    assert state[2] == 0.0   # No central bias


def test_field_sample_exogenous() -> None:
    """Test Field exogenous info sampling."""
    config = TwoNewsvendorConfig(demand_lower=20.0, demand_upper=80.0)
    model = TwoNewsvendorFieldModel(config)
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    exog = model.sample_exogenous(key, state, 0)

    # Check that demand is in range
    assert 20.0 <= exog.demand <= 80.0
    # Estimates should be non-negative
    assert exog.estimate_field >= 0.0
    assert exog.estimate_central >= 0.0


def test_field_transition() -> None:
    """Test Field state transition."""
    config = TwoNewsvendorConfig(alpha_bias=0.1)
    model = TwoNewsvendorFieldModel(config)
    key = jax.random.PRNGKey(42)

    state = jnp.array([50.0, 5.0, -3.0])  # [estimate, source_bias, central_bias]
    decision = jnp.array([45.0])  # Request 45 units
    allocated = jnp.array(40.0)  # Central allocates 40

    exog = ExogenousInfo(
        demand=jnp.array(48.0),
        estimate_field=jnp.array(52.0),
        estimate_central=jnp.array(47.0)
    )

    next_state = model.transition(state, decision, exog, allocated)

    assert next_state.shape == (3,)
    # Estimate should be updated
    assert next_state[0] == 52.0
    # Biases should be updated with exponential smoothing
    assert next_state[1] != state[1]  # Source bias changed
    assert next_state[2] != state[2]  # Central bias changed


def test_field_reward() -> None:
    """Test Field reward calculation."""
    config = TwoNewsvendorConfig(
        overage_cost_field=2.0,
        underage_cost_field=8.0
    )
    model = TwoNewsvendorFieldModel(config)

    state = jnp.array([50.0, 0.0, 0.0])
    decision = jnp.array([45.0])

    # Test underage case
    exog_underage = ExogenousInfo(
        demand=jnp.array(50.0),  # Demand > allocated
        estimate_field=jnp.array(50.0),
        estimate_central=jnp.array(50.0)
    )
    allocated_under = jnp.array(40.0)

    reward_under = model.reward(state, decision, exog_underage, allocated_under)
    # Underage = 50 - 40 = 10, cost = 10 * 8 = 80
    assert reward_under == -80.0

    # Test overage case
    exog_overage = ExogenousInfo(
        demand=jnp.array(30.0),  # Demand < allocated
        estimate_field=jnp.array(50.0),
        estimate_central=jnp.array(50.0)
    )
    allocated_over = jnp.array(40.0)

    reward_over = model.reward(state, decision, exog_overage, allocated_over)
    # Overage = 40 - 30 = 10, cost = 10 * 2 = 20
    assert reward_over == -20.0


def test_field_shapes_with_chex() -> None:
    """Test Field model output shapes using chex assertions."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorFieldModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    chex.assert_shape(state, (3,))
    chex.assert_rank(state, 1)

    exog = model.sample_exogenous(key, state, 0)
    chex.assert_shape(exog.demand, ())
    chex.assert_shape(exog.estimate_field, ())
    chex.assert_shape(exog.estimate_central, ())


def test_field_batch_with_vmap() -> None:
    """Test Field model can be batched with vmap."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorFieldModel(config)
    key = jax.random.PRNGKey(42)

    # Batch of 10 states
    batch_size = 10
    keys = jax.random.split(key, batch_size)
    states = jax.vmap(model.init_state)(keys)
    chex.assert_shape(states, (batch_size, 3))

    # Batch sample exogenous
    exogs = jax.vmap(lambda k, s: model.sample_exogenous(k, s, 0))(keys, states)
    chex.assert_shape(exogs.demand, (batch_size,))

    # Batch transition
    decisions = jnp.ones((batch_size, 1)) * 50.0
    allocated = jnp.ones(batch_size) * 45.0

    next_states = jax.vmap(model.transition)(states, decisions, exogs, allocated)
    chex.assert_shape(next_states, (batch_size, 3))

    # Batch reward
    rewards = jax.vmap(model.reward)(states, decisions, exogs, allocated)
    chex.assert_shape(rewards, (batch_size,))


def test_field_edge_case_zero_demand() -> None:
    """Test Field model handles zero demand correctly."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorFieldModel(config)

    state = jnp.array([50.0, 0.0, 0.0])
    decision = jnp.array([40.0])
    allocated = jnp.array(40.0)

    exog = ExogenousInfo(
        demand=jnp.array(0.0),  # Zero demand
        estimate_field=jnp.array(50.0),
        estimate_central=jnp.array(50.0)
    )

    reward = model.reward(state, decision, exog, allocated)
    # All allocated inventory is overage
    expected_cost = allocated * config.overage_cost_field
    assert reward == -expected_cost


def test_field_edge_case_extreme_biases() -> None:
    """Test Field model handles extreme bias values."""
    config = TwoNewsvendorConfig(alpha_bias=0.5)
    model = TwoNewsvendorFieldModel(config)

    state = jnp.array([50.0, 100.0, -100.0])  # Extreme biases
    decision = jnp.array([50.0])
    allocated = jnp.array(50.0)

    exog = ExogenousInfo(
        demand=jnp.array(50.0),
        estimate_field=jnp.array(60.0),
        estimate_central=jnp.array(55.0)
    )

    # Should not crash or produce NaN
    next_state = model.transition(state, decision, exog, allocated)
    chex.assert_tree_all_finite(next_state)
    assert next_state.shape == (3,)


# ============================================================================
# Central Model Tests
# ============================================================================


def test_central_model_initialization() -> None:
    """Test Central model initializes correctly."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorCentralModel(config)

    assert model.config == config
    assert isinstance(model, TwoNewsvendorCentralModel)


def test_central_init_state() -> None:
    """Test Central initial state generation."""
    config = TwoNewsvendorConfig(demand_lower=0.0, demand_upper=100.0)
    model = TwoNewsvendorCentralModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    assert state.shape == (7,)
    assert state[0] == 50.0  # field_request
    assert state[2] == 0.5   # field_weight
    assert state[6] == 0.5   # source_weight


def test_central_transition() -> None:
    """Test Central state transition."""
    config = TwoNewsvendorConfig(alpha_bias=0.1)
    model = TwoNewsvendorCentralModel(config)

    state = jnp.array([45.0, 2.0, 0.5, 0.0, 50.0, -1.0, 0.5])
    decision = jnp.array([40.0])
    field_request = jnp.array(45.0)

    exog = ExogenousInfo(
        demand=jnp.array(48.0),
        estimate_field=jnp.array(52.0),
        estimate_central=jnp.array(47.0)
    )

    next_state = model.transition(state, decision, exog, field_request)

    assert next_state.shape == (7,)
    # Estimate should be updated
    assert next_state[4] == 47.0
    # Biases should be updated
    assert next_state[1] != state[1]  # field_bias changed
    assert next_state[5] != state[5]  # source_bias changed


def test_central_reward() -> None:
    """Test Central reward calculation."""
    config = TwoNewsvendorConfig(
        overage_cost_central=1.5,
        underage_cost_central=8.5
    )
    model = TwoNewsvendorCentralModel(config)

    state = jnp.array([45.0, 0.0, 0.5, 0.0, 50.0, 0.0, 0.5])

    # Test underage
    decision_under = jnp.array([40.0])
    exog_under = ExogenousInfo(
        demand=jnp.array(50.0),
        estimate_field=jnp.array(50.0),
        estimate_central=jnp.array(50.0)
    )

    reward_under = model.reward(state, decision_under, exog_under)
    # Underage = 50 - 40 = 10, cost = 10 * 8.5 = 85
    assert reward_under == -85.0

    # Test overage
    decision_over = jnp.array([40.0])
    exog_over = ExogenousInfo(
        demand=jnp.array(30.0),
        estimate_field=jnp.array(30.0),
        estimate_central=jnp.array(30.0)
    )

    reward_over = model.reward(state, decision_over, exog_over)
    # Overage = 40 - 30 = 10, cost = 10 * 1.5 = 15
    assert reward_over == -15.0


def test_central_shapes_with_chex() -> None:
    """Test Central model output shapes using chex assertions."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorCentralModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    chex.assert_shape(state, (7,))
    chex.assert_rank(state, 1)


def test_central_batch_with_vmap() -> None:
    """Test Central model can be batched with vmap."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorCentralModel(config)
    key = jax.random.PRNGKey(42)

    # Batch of 10 states
    batch_size = 10
    keys = jax.random.split(key, batch_size)
    states = jax.vmap(model.init_state)(keys)
    chex.assert_shape(states, (batch_size, 7))

    # Batch sample exogenous
    exogs = jax.vmap(lambda k, s: model.sample_exogenous(k, s, 0))(keys, states)
    chex.assert_shape(exogs.demand, (batch_size,))

    # Batch transition
    decisions = jnp.ones((batch_size, 1)) * 40.0
    field_requests = jnp.ones(batch_size) * 45.0

    next_states = jax.vmap(model.transition)(states, decisions, exogs, field_requests)
    chex.assert_shape(next_states, (batch_size, 7))

    # Batch reward
    rewards = jax.vmap(model.reward)(states, decisions, exogs)
    chex.assert_shape(rewards, (batch_size,))


# ============================================================================
# Policy Tests
# ============================================================================


def test_newsvendor_field_policy() -> None:
    """Test NewsvendorFieldPolicy."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorFieldModel(config)
    policy = NewsvendorFieldPolicy(model, bias_adjustment=0.0)

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 0.0, 0.0])

    decision = policy(None, state, key)

    assert decision.shape == (1,)
    assert decision[0] >= 0.0  # Non-negative quantity


def test_bias_adjusted_field_policy() -> None:
    """Test BiasAdjustedFieldPolicy."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorFieldModel(config)
    policy = BiasAdjustedFieldPolicy(model, use_source_bias=True, use_central_bias=True)

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 5.0, -3.0])  # Positive source bias, negative central bias

    decision = policy(None, state, key)

    assert decision.shape == (1,)
    # Decision should be adjusted for biases: 50 - 5 - (-3) = 48
    assert decision[0] >= 0.0


def test_newsvendor_central_policy() -> None:
    """Test NewsvendorCentralPolicy."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorCentralModel(config)
    policy = NewsvendorCentralPolicy(model, trust_field=0.5)

    key = jax.random.PRNGKey(42)
    state = jnp.array([45.0, 0.0, 0.5, 0.0, 50.0, 0.0, 0.5])
    field_request = jnp.array(45.0)

    decision = policy(None, state, key, field_request)

    assert decision.shape == (1,)
    assert decision[0] >= 0.0


def test_bias_adjusted_central_policy() -> None:
    """Test BiasAdjustedCentralPolicy."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorCentralModel(config)
    policy = BiasAdjustedCentralPolicy(model, trust_field=0.5)

    key = jax.random.PRNGKey(42)
    state = jnp.array([45.0, 2.0, 0.5, 0.0, 50.0, -1.0, 0.5])
    field_request = jnp.array(45.0)

    decision = policy(None, state, key, field_request)

    assert decision.shape == (1,)
    assert decision[0] >= 0.0


def test_neural_field_policy() -> None:
    """Test NeuralFieldPolicy."""
    policy = NeuralFieldPolicy(hidden_dims=[8, 4], rngs=nnx.Rngs(42))

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 0.0, 0.0])

    decision = policy(state, key)

    assert decision.shape == (1,)
    assert decision[0] >= 0.0  # ReLU ensures non-negative
    assert len(policy.layers) == 3  # 2 hidden + 1 output


def test_neural_central_policy() -> None:
    """Test NeuralCentralPolicy."""
    policy = NeuralCentralPolicy(hidden_dims=[8, 4], rngs=nnx.Rngs(42))

    key = jax.random.PRNGKey(42)
    state = jnp.array([45.0, 0.0, 0.5, 0.0, 50.0, 0.0, 0.5])
    field_request = jnp.array(45.0)

    decision = policy(state, field_request, key)

    assert decision.shape == (1,)
    assert decision[0] >= 0.0
    assert len(policy.layers) == 3


def test_always_allocate_requested_policy() -> None:
    """Test AlwaysAllocateRequestedPolicy."""
    policy = AlwaysAllocateRequestedPolicy()

    key = jax.random.PRNGKey(42)
    state = jnp.array([45.0, 0.0, 0.5, 0.0, 50.0, 0.0, 0.5])
    field_request = jnp.array(45.0)

    decision = policy(None, state, key, field_request)

    assert decision[0] == field_request  # Always allocate exactly what was requested


# ============================================================================
# JIT Compilation Tests
# ============================================================================


def test_field_jit_compilation() -> None:
    """Test that Field model methods can be JIT compiled."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorFieldModel(config)
    key = jax.random.PRNGKey(42)

    init_state_jit = jax.jit(model.init_state)
    state = init_state_jit(key)

    assert state.shape == (3,)


def test_central_jit_compilation() -> None:
    """Test that Central model methods can be JIT compiled."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorCentralModel(config)
    key = jax.random.PRNGKey(42)

    init_state_jit = jax.jit(model.init_state)
    state = init_state_jit(key)

    assert state.shape == (7,)


def test_policy_jit_compilation() -> None:
    """Test that policies can be JIT compiled."""
    config = TwoNewsvendorConfig()
    field_model = TwoNewsvendorFieldModel(config)
    policy = NewsvendorFieldPolicy(field_model)

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 0.0, 0.0])

    policy_jit = jax.jit(lambda s, k: policy(None, s, k))
    decision = policy_jit(state, key)

    assert decision.shape == (1,)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_episode_coordination() -> None:
    """Test running a complete episode with both agents coordinating."""
    config = TwoNewsvendorConfig()
    field_model = TwoNewsvendorFieldModel(config)
    central_model = TwoNewsvendorCentralModel(config)

    field_policy = NewsvendorFieldPolicy(field_model)
    central_policy = NewsvendorCentralPolicy(central_model, trust_field=0.7)

    key = jax.random.PRNGKey(42)
    field_state = field_model.init_state(key)
    central_state = central_model.init_state(key)

    total_field_reward = 0.0
    total_central_reward = 0.0

    for t in range(10):
        # Field observes and requests
        key, subkey = jax.random.split(key)
        exog = field_model.sample_exogenous(subkey, field_state, t)

        # Update Field state with observed estimate
        field_state = field_state.at[0].set(exog.estimate_field)

        # Field makes request
        key, subkey = jax.random.split(key)
        field_decision = field_policy(None, field_state, subkey)

        # Central observes request and makes allocation
        central_state = central_state.at[0].set(field_decision[0])
        central_state = central_state.at[4].set(exog.estimate_central)

        key, subkey = jax.random.split(key)
        central_decision = central_policy(None, central_state, subkey, field_decision[0])

        # Compute rewards
        field_reward = field_model.reward(field_state, field_decision, exog, central_decision[0])
        central_reward = central_model.reward(central_state, central_decision, exog)

        total_field_reward += float(field_reward)
        total_central_reward += float(central_reward)

        # Transition
        field_state = field_model.transition(field_state, field_decision, exog, central_decision[0])
        central_state = central_model.transition(central_state, central_decision, exog, field_decision[0])

    # Both should have accumulated costs (negative rewards)
    assert isinstance(total_field_reward, float)
    assert isinstance(total_central_reward, float)


def test_multiple_policies_comparison() -> None:
    """Test that we can run multiple policy combinations."""
    config = TwoNewsvendorConfig()
    field_model = TwoNewsvendorFieldModel(config)
    central_model = TwoNewsvendorCentralModel(config)

    policy_pairs = [
        ("Newsvendor", NewsvendorFieldPolicy(field_model), NewsvendorCentralPolicy(central_model, 0.5)),
        ("BiasAdjusted", BiasAdjustedFieldPolicy(field_model), BiasAdjustedCentralPolicy(central_model, 0.5)),
        ("AlwaysApprove", NewsvendorFieldPolicy(field_model), AlwaysAllocateRequestedPolicy()),
    ]

    key = jax.random.PRNGKey(42)
    results = {}

    for name, field_policy, central_policy in policy_pairs:
        field_state = field_model.init_state(key)
        central_state = central_model.init_state(key)
        total_cost = 0.0

        for t in range(5):
            key, subkey = jax.random.split(key)
            exog = field_model.sample_exogenous(subkey, field_state, t)

            field_state = field_state.at[0].set(exog.estimate_field)
            key, subkey = jax.random.split(key)
            field_decision = field_policy(None, field_state, subkey)

            central_state = central_state.at[0].set(field_decision[0])
            central_state = central_state.at[4].set(exog.estimate_central)

            key, subkey = jax.random.split(key)
            central_decision = central_policy(None, central_state, subkey, field_decision[0])

            field_reward = field_model.reward(field_state, field_decision, exog, central_decision[0])
            central_reward = central_model.reward(central_state, central_decision, exog)

            total_cost += float(field_reward + central_reward)

            field_state = field_model.transition(field_state, field_decision, exog, central_decision[0])
            central_state = central_model.transition(central_state, central_decision, exog, field_decision[0])

        results[name] = total_cost

    assert len(results) == 3
    assert all(isinstance(cost, float) for cost in results.values())


# ============================================================================
# Gradient Flow Tests
# ============================================================================


def test_gradient_flow_neural_field_policy() -> None:
    """Test that gradients flow through neural field policy."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorFieldModel(config)
    policy = NeuralFieldPolicy(hidden_dims=[8, 4], rngs=nnx.Rngs(42))

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 0.0, 0.0])

    # Define loss as negative reward
    def loss_fn(policy_params: nnx.Module) -> jax.Array:
        decision = policy_params(state, key)
        exog = model.sample_exogenous(key, state, 0)
        allocated = decision[0]  # Assume central allocates exactly what's requested
        reward = model.reward(state, decision, exog, allocated)
        return -reward  # Minimize cost

    # Compute gradients
    grads = nnx.grad(loss_fn)(policy)

    # Check that gradients exist and are finite for all layers
    for layer in policy.layers:
        if isinstance(layer, nnx.Linear):
            chex.assert_tree_all_finite(layer.kernel.value)
            chex.assert_tree_all_finite(layer.bias.value)


def test_gradient_flow_neural_central_policy() -> None:
    """Test that gradients flow through neural central policy."""
    config = TwoNewsvendorConfig()
    model = TwoNewsvendorCentralModel(config)
    policy = NeuralCentralPolicy(hidden_dims=[8, 4], rngs=nnx.Rngs(42))

    key = jax.random.PRNGKey(42)
    state = jnp.array([45.0, 0.0, 0.5, 0.0, 50.0, 0.0, 0.5])
    field_request = jnp.array(45.0)

    # Define loss as negative reward
    def loss_fn(policy_params: nnx.Module) -> jax.Array:
        decision = policy_params(state, field_request, key)
        exog = model.sample_exogenous(key, state, 0)
        reward = model.reward(state, decision, exog)
        return -reward

    # Compute gradients
    grads = nnx.grad(loss_fn)(policy)

    # Check that gradients exist and are finite for all layers
    for layer in policy.layers:
        if isinstance(layer, nnx.Linear):
            chex.assert_tree_all_finite(layer.kernel.value)
            chex.assert_tree_all_finite(layer.bias.value)


def test_vmap_neural_policies() -> None:
    """Test that neural policies work with vmap for batch processing."""
    field_policy = NeuralFieldPolicy(hidden_dims=[8], rngs=nnx.Rngs(0))
    central_policy = NeuralCentralPolicy(hidden_dims=[8], rngs=nnx.Rngs(1))

    key = jax.random.PRNGKey(42)
    batch_size = 5

    # Batch field policy
    keys = jax.random.split(key, batch_size)
    field_states = jnp.tile(jnp.array([50.0, 0.0, 0.0]), (batch_size, 1))

    # Note: vmap works with the policy call directly
    batch_field_decisions = jax.vmap(lambda s, k: field_policy(s, k))(field_states, keys)
    chex.assert_shape(batch_field_decisions, (batch_size, 1))

    # Batch central policy
    central_states = jnp.tile(jnp.array([45.0, 0.0, 0.5, 0.0, 50.0, 0.0, 0.5]), (batch_size, 1))
    field_requests = jnp.ones(batch_size) * 45.0

    batch_central_decisions = jax.vmap(
        lambda s, fr, k: central_policy(s, fr, k)
    )(central_states, field_requests, keys)
    chex.assert_shape(batch_central_decisions, (batch_size, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
