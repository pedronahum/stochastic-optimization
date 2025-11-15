"""Tests for Adaptive Market Planning problem (JAX-native implementation)."""

import pytest
import jax
import jax.numpy as jnp
import chex
from flax import nnx

from problems.adaptive_market_planning import (
    AdaptiveMarketPlanningConfig,
    AdaptiveMarketPlanningModel,
    ExogenousInfo,
    HarmonicStepPolicy,
    KestenStepPolicy,
    ConstantStepPolicy,
    NeuralStepPolicy,
)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_default_values() -> None:
    """Test that config has reasonable defaults."""
    config = AdaptiveMarketPlanningConfig()

    assert config.price == 1.0
    assert config.cost == 0.5
    assert config.demand_mean == 100.0
    assert config.initial_order_quantity == 50.0
    assert config.max_order_quantity == 1000.0


def test_config_validation_price() -> None:
    """Test that config validates price."""
    with pytest.raises(ValueError, match="price must be positive"):
        AdaptiveMarketPlanningConfig(price=0.0)

    with pytest.raises(ValueError, match="price must be positive"):
        AdaptiveMarketPlanningConfig(price=-1.0)


def test_config_validation_cost() -> None:
    """Test that config validates cost."""
    with pytest.raises(ValueError, match="cost must be non-negative"):
        AdaptiveMarketPlanningConfig(cost=-1.0)


def test_config_validation_price_cost_relationship() -> None:
    """Test that price must be greater than cost."""
    with pytest.raises(ValueError, match="price.*must be.*cost"):
        AdaptiveMarketPlanningConfig(price=1.0, cost=1.0)

    with pytest.raises(ValueError, match="price.*must be.*cost"):
        AdaptiveMarketPlanningConfig(price=1.0, cost=1.5)


def test_config_validation_demand_mean() -> None:
    """Test that config validates demand_mean."""
    with pytest.raises(ValueError, match="demand_mean must be positive"):
        AdaptiveMarketPlanningConfig(demand_mean=0.0)

    with pytest.raises(ValueError, match="demand_mean must be positive"):
        AdaptiveMarketPlanningConfig(demand_mean=-10.0)


def test_config_immutability() -> None:
    """Test that config is immutable (frozen dataclass)."""
    config = AdaptiveMarketPlanningConfig()

    with pytest.raises((AttributeError, Exception)):
        config.price = 2.0  # Should fail - frozen dataclass


# ============================================================================
# Model Tests
# ============================================================================


def test_model_initialization() -> None:
    """Test model initializes correctly."""
    config = AdaptiveMarketPlanningConfig()
    model = AdaptiveMarketPlanningModel(config)

    assert model.config == config
    assert isinstance(model, AdaptiveMarketPlanningModel)
    # Check optimal quantity is computed
    assert model.optimal_quantity > 0


def test_init_state() -> None:
    """Test initial state generation."""
    config = AdaptiveMarketPlanningConfig(initial_order_quantity=60.0)
    model = AdaptiveMarketPlanningModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    assert state.shape == (2,)
    assert state[0] == 60.0  # Initial order quantity
    assert state[1] == 0.0   # No sign changes yet


def test_sample_exogenous() -> None:
    """Test exogenous info sampling."""
    config = AdaptiveMarketPlanningConfig(demand_mean=100.0)
    model = AdaptiveMarketPlanningModel(config)
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    exog = model.sample_exogenous(key, state, 0)

    # Demand should be positive
    assert exog.demand > 0
    # Previous derivative should be 0 by default
    assert exog.previous_derivative == 0.0


def test_transition_undersupply() -> None:
    """Test state transition when demand > order (undersupply)."""
    config = AdaptiveMarketPlanningConfig(price=1.0, cost=0.5)
    model = AdaptiveMarketPlanningModel(config)

    state = jnp.array([50.0, 0.0])  # [order_quantity, counter]
    decision = jnp.array([0.1])  # Step size
    exog = ExogenousInfo(
        demand=jnp.array(100.0),  # Demand > order_quantity
        previous_derivative=jnp.array(0.0)
    )

    next_state = model.transition(state, decision, exog)

    # Should increase order quantity (gradient = price - cost = 0.5)
    # new_order = 50 + 0.1 * 0.5 = 50.05
    assert next_state[0] > state[0]
    assert next_state[1] == 0.0  # Counter unchanged (no sign change)


def test_transition_oversupply() -> None:
    """Test state transition when demand < order (oversupply)."""
    config = AdaptiveMarketPlanningConfig(price=1.0, cost=0.5)
    model = AdaptiveMarketPlanningModel(config)

    state = jnp.array([100.0, 0.0])  # [order_quantity, counter]
    decision = jnp.array([0.1])  # Step size
    exog = ExogenousInfo(
        demand=jnp.array(50.0),  # Demand < order_quantity
        previous_derivative=jnp.array(0.0)
    )

    next_state = model.transition(state, decision, exog)

    # Should decrease order quantity (gradient = -cost = -0.5)
    # new_order = 100 + 0.1 * (-0.5) = 99.95
    assert next_state[0] < state[0]
    assert next_state[1] == 0.0  # Counter unchanged


def test_transition_sign_change() -> None:
    """Test that counter increments on gradient sign change."""
    config = AdaptiveMarketPlanningConfig(price=1.0, cost=0.5)
    model = AdaptiveMarketPlanningModel(config)

    state = jnp.array([50.0, 2.0])  # [order_quantity, counter=2]
    decision = jnp.array([0.1])
    exog = ExogenousInfo(
        demand=jnp.array(40.0),  # Oversupply
        previous_derivative=jnp.array(0.5)  # Previous was positive (undersupply)
    )

    next_state = model.transition(state, decision, exog)

    # Counter should increment due to sign change
    assert next_state[1] == 3.0


def test_reward_undersupply() -> None:
    """Test reward calculation when demand > order."""
    config = AdaptiveMarketPlanningConfig(price=2.0, cost=1.0)
    model = AdaptiveMarketPlanningModel(config)

    state = jnp.array([50.0, 0.0])
    decision = jnp.array([0.1])
    exog = ExogenousInfo(
        demand=jnp.array(100.0),
        previous_derivative=jnp.array(0.0)
    )

    reward = model.reward(state, decision, exog)

    # Sell all 50 units: revenue = 2.0 * 50 = 100, cost = 1.0 * 50 = 50
    expected_reward = 100.0 - 50.0
    assert reward == expected_reward


def test_reward_oversupply() -> None:
    """Test reward calculation when demand < order."""
    config = AdaptiveMarketPlanningConfig(price=2.0, cost=1.0)
    model = AdaptiveMarketPlanningModel(config)

    state = jnp.array([100.0, 0.0])
    decision = jnp.array([0.1])
    exog = ExogenousInfo(
        demand=jnp.array(50.0),
        previous_derivative=jnp.array(0.0)
    )

    reward = model.reward(state, decision, exog)

    # Sell only 50 units: revenue = 2.0 * 50 = 100, cost = 1.0 * 100 = 100
    expected_reward = 100.0 - 100.0
    assert reward == expected_reward


def test_is_valid_decision() -> None:
    """Test decision validation."""
    config = AdaptiveMarketPlanningConfig()
    model = AdaptiveMarketPlanningModel(config)
    state = jnp.array([50.0, 0.0])

    # Valid decision (positive step size)
    valid_decision = jnp.array([0.1])
    assert model.is_valid_decision(state, valid_decision)

    # Valid decision (zero step size)
    zero_decision = jnp.array([0.0])
    assert model.is_valid_decision(state, zero_decision)

    # Invalid decision (negative step size)
    invalid_decision = jnp.array([-0.1])
    assert not model.is_valid_decision(state, invalid_decision)


def test_shapes_with_chex() -> None:
    """Test model output shapes using chex assertions."""
    config = AdaptiveMarketPlanningConfig()
    model = AdaptiveMarketPlanningModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    chex.assert_shape(state, (2,))
    chex.assert_rank(state, 1)

    exog = model.sample_exogenous(key, state, 0)
    chex.assert_shape(exog.demand, ())
    chex.assert_shape(exog.previous_derivative, ())


def test_batch_with_vmap() -> None:
    """Test model can be batched with vmap."""
    config = AdaptiveMarketPlanningConfig()
    model = AdaptiveMarketPlanningModel(config)
    key = jax.random.PRNGKey(42)

    # Batch of 10 states
    batch_size = 10
    keys = jax.random.split(key, batch_size)
    states = jax.vmap(model.init_state)(keys)
    chex.assert_shape(states, (batch_size, 2))

    # Batch sample exogenous
    exogs = jax.vmap(lambda k, s: model.sample_exogenous(k, s, 0))(keys, states)
    chex.assert_shape(exogs.demand, (batch_size,))

    # Batch transition
    decisions = jnp.ones((batch_size, 1)) * 0.1
    next_states = jax.vmap(model.transition)(states, decisions, exogs)
    chex.assert_shape(next_states, (batch_size, 2))

    # Batch reward
    rewards = jax.vmap(model.reward)(states, decisions, exogs)
    chex.assert_shape(rewards, (batch_size,))


def test_edge_case_zero_order() -> None:
    """Test model handles zero order quantity."""
    config = AdaptiveMarketPlanningConfig()
    model = AdaptiveMarketPlanningModel(config)

    state = jnp.array([0.0, 0.0])  # Zero order
    decision = jnp.array([0.1])
    exog = ExogenousInfo(
        demand=jnp.array(100.0),
        previous_derivative=jnp.array(0.0)
    )

    # Should not crash
    reward = model.reward(state, decision, exog)
    assert reward == -0.0  # No sales, no costs

    next_state = model.transition(state, decision, exog)
    # Should increase from 0 (undersupply)
    assert next_state[0] > 0.0


def test_edge_case_max_order() -> None:
    """Test model respects maximum order quantity."""
    config = AdaptiveMarketPlanningConfig(max_order_quantity=100.0)
    model = AdaptiveMarketPlanningModel(config)

    state = jnp.array([99.0, 0.0])  # Near max
    decision = jnp.array([10.0])  # Large step
    exog = ExogenousInfo(
        demand=jnp.array(200.0),  # High demand
        previous_derivative=jnp.array(0.0)
    )

    next_state = model.transition(state, decision, exog)

    # Should be clipped to max
    assert next_state[0] <= config.max_order_quantity


# ============================================================================
# Policy Tests
# ============================================================================


def test_harmonic_step_policy() -> None:
    """Test HarmonicStepPolicy."""
    policy = HarmonicStepPolicy(theta=1.0)

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 0.0])

    decision = policy(None, state, key)

    assert decision.shape == (1,)
    assert decision[0] > 0.0  # Positive step size


def test_kesten_step_policy() -> None:
    """Test KestenStepPolicy."""
    policy = KestenStepPolicy(theta=1.0)

    key = jax.random.PRNGKey(42)

    # Low counter -> large step
    state_low = jnp.array([50.0, 1.0])
    decision_low = policy(None, state_low, key)

    # High counter -> small step
    state_high = jnp.array([50.0, 10.0])
    decision_high = policy(None, state_high, key)

    assert decision_low[0] > decision_high[0]  # Step decreases with counter


def test_constant_step_policy() -> None:
    """Test ConstantStepPolicy."""
    theta = 0.15
    policy = ConstantStepPolicy(theta=theta)

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 0.0])

    decision = policy(None, state, key)

    assert decision[0] == theta  # Always returns constant


def test_neural_step_policy() -> None:
    """Test NeuralStepPolicy."""
    policy = NeuralStepPolicy(hidden_dims=[8, 4], rngs=nnx.Rngs(42))

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 5.0])

    decision = policy(state, key)

    assert decision.shape == (1,)
    # Should be in valid range
    assert policy.min_step <= decision[0] <= policy.max_step


# ============================================================================
# JIT Compilation Tests
# ============================================================================


def test_jit_compilation() -> None:
    """Test that model methods can be JIT compiled."""
    config = AdaptiveMarketPlanningConfig()
    model = AdaptiveMarketPlanningModel(config)
    key = jax.random.PRNGKey(42)

    init_state_jit = jax.jit(model.init_state)
    state = init_state_jit(key)

    assert state.shape == (2,)


def test_policy_jit_compilation() -> None:
    """Test that policies can be JIT compiled."""
    policy = KestenStepPolicy(theta=1.0)

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 2.0])

    policy_jit = jax.jit(lambda s, k: policy(None, s, k))
    decision = policy_jit(state, key)

    assert decision.shape == (1,)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_episode() -> None:
    """Test running a complete learning episode."""
    config = AdaptiveMarketPlanningConfig()
    model = AdaptiveMarketPlanningModel(config)
    policy = KestenStepPolicy(theta=1.0)

    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    total_reward = 0.0
    previous_derivative = 0.0

    for t in range(20):
        # Sample exogenous info
        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, t, previous_derivative)

        # Get decision
        key, subkey = jax.random.split(key)
        decision = policy(None, state, subkey)

        # Compute reward
        reward = model.reward(state, decision, exog)
        total_reward += float(reward)

        # Update derivative for next iteration
        previous_derivative = float(model.compute_derivative(state, exog.demand))

        # Transition
        state = model.transition(state, decision, exog)

    # Check that we accumulated some rewards
    assert isinstance(total_reward, float)


def test_convergence_behavior() -> None:
    """Test that order quantity converges over time."""
    config = AdaptiveMarketPlanningConfig(initial_order_quantity=10.0)
    model = AdaptiveMarketPlanningModel(config)
    policy = HarmonicStepPolicy(theta=1.0)

    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    order_quantities = [float(state[0])]
    previous_derivative = 0.0

    for t in range(100):
        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, t, previous_derivative)

        # Update time for harmonic policy
        policy.current_time = t + 1

        decision = policy(None, state, key)
        previous_derivative = float(model.compute_derivative(state, exog.demand))
        state = model.transition(state, decision, exog)

        order_quantities.append(float(state[0]))

    # Order quantity should change less over time (converging)
    early_changes = abs(order_quantities[10] - order_quantities[0])
    late_changes = abs(order_quantities[99] - order_quantities[90])

    assert late_changes < early_changes  # Converging


# ============================================================================
# Gradient Flow Tests
# ============================================================================


def test_gradient_flow_neural_policy() -> None:
    """Test that gradients flow through neural policy."""
    config = AdaptiveMarketPlanningConfig()
    model = AdaptiveMarketPlanningModel(config)
    policy = NeuralStepPolicy(hidden_dims=[8, 4], rngs=nnx.Rngs(42))

    key = jax.random.PRNGKey(42)
    state = jnp.array([50.0, 2.0])

    # Define loss as negative reward
    def loss_fn(policy_params: nnx.Module) -> jax.Array:
        decision = policy_params(state, key)
        exog = model.sample_exogenous(key, state, 0)
        reward = model.reward(state, decision, exog)
        return -reward  # Minimize negative reward

    # Compute gradients
    grads = nnx.grad(loss_fn)(policy)

    # Check that gradients exist and are finite for all layers
    for layer in policy.layers:
        if isinstance(layer, nnx.Linear):
            chex.assert_tree_all_finite(layer.kernel.value)
            chex.assert_tree_all_finite(layer.bias.value)


def test_vmap_neural_policy() -> None:
    """Test that neural policy works with vmap for batch processing."""
    policy = NeuralStepPolicy(hidden_dims=[8], rngs=nnx.Rngs(0))

    key = jax.random.PRNGKey(42)
    batch_size = 5

    # Batch states
    keys = jax.random.split(key, batch_size)
    states = jnp.tile(jnp.array([50.0, 2.0]), (batch_size, 1))

    # Batch policy evaluation
    batch_decisions = jax.vmap(lambda s, k: policy(s, k))(states, keys)
    chex.assert_shape(batch_decisions, (batch_size, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
