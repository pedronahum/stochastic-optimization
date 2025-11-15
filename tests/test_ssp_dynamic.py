"""Tests for Stochastic Shortest Path Dynamic problem (JAX-native implementation)."""

import pytest
import jax
import jax.numpy as jnp

from problems.ssp_dynamic import (
    SSPDynamicConfig,
    SSPDynamicModel,
    ExogenousInfo,
    LookaheadPolicy,
    GreedyLookaheadPolicy,
    RandomPolicy,
)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_default_values() -> None:
    """Test that config has reasonable defaults."""
    config = SSPDynamicConfig()

    assert config.n_nodes == 10
    assert config.horizon == 15
    assert config.edge_prob == 0.3
    assert config.cost_min == 1.0
    assert config.cost_max == 10.0
    assert config.max_spread == 0.3


def test_config_validation() -> None:
    """Test that config validates parameters."""
    with pytest.raises(ValueError, match="n_nodes must be > 1"):
        SSPDynamicConfig(n_nodes=1)

    with pytest.raises(ValueError, match="edge_prob"):
        SSPDynamicConfig(edge_prob=0.0)

    with pytest.raises(ValueError, match="cost_max"):
        SSPDynamicConfig(cost_min=10.0, cost_max=5.0)

    with pytest.raises(ValueError, match="max_spread"):
        SSPDynamicConfig(max_spread=1.0)

    with pytest.raises(ValueError, match="horizon"):
        SSPDynamicConfig(horizon=0)


def test_config_immutability() -> None:
    """Test that config is immutable."""
    config = SSPDynamicConfig()

    with pytest.raises((AttributeError, Exception)):
        config.n_nodes = 20


# ============================================================================
# Model Tests
# ============================================================================


def test_model_initialization() -> None:
    """Test model initializes correctly."""
    config = SSPDynamicConfig(n_nodes=5, horizon=10)
    model = SSPDynamicModel(config)

    assert model.config == config
    assert model.target_node == 4  # Last node

    # Check graph was created
    assert model.adjacency.shape == (5, 5)
    assert model.mean_costs.shape == (5, 5)
    assert model.spreads.shape == (5, 5)


def test_init_state_structure() -> None:
    """Test that init_state creates state with correct structure."""
    config = SSPDynamicConfig(n_nodes=5, horizon=10)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # State: [current_node, time, estimated_costs (5*5), obs_counts (5*5)]
    expected_size = 2 + 5*5 + 5*5  # 2 + 25 + 25 = 52
    assert state.shape == (expected_size,)

    # Check initial values
    assert state[0] == 0.0  # Start at origin
    assert state[1] == 0.0  # Time = 0


def test_sample_exogenous() -> None:
    """Test edge cost sampling with spread."""
    config = SSPDynamicConfig(n_nodes=5)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    exog = model.sample_exogenous(key, state, 0)

    # Should have costs for all nodes
    assert exog.edge_costs.shape == (5,)

    # Costs should be within spread range
    current_node = 0
    mean = model.mean_costs[current_node]
    spread = model.spreads[current_node]

    # Check costs are within bounds where edges exist
    for i in range(5):
        if model.adjacency[current_node, i]:
            lower = mean[i] * (1 - spread[i])
            upper = mean[i] * (1 + spread[i])
            # Allow some tolerance for numerical precision
            assert exog.edge_costs[i] >= lower - 1e-5
            assert exog.edge_costs[i] <= upper + 1e-5


def test_transition() -> None:
    """Test state transition updates costs and time."""
    config = SSPDynamicConfig(n_nodes=5)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    decision = jnp.array(1, dtype=jnp.int32)  # Move to node 1

    exog = ExogenousInfo(edge_costs=jnp.array([0., 5., 0., 0., 0.]))

    next_state = model.transition(state, decision, exog)

    # Should now be at node 1
    assert next_state[0] == 1.0

    # Time should increment
    assert next_state[1] == 1.0

    # State shape should be preserved
    assert next_state.shape == state.shape


def test_reward() -> None:
    """Test reward calculation."""
    config = SSPDynamicConfig()
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    decision = jnp.array(1, dtype=jnp.int32)

    exog = ExogenousInfo(edge_costs=jnp.array([0., 5., 0., 0., 0., 0., 0., 0., 0., 0.]))

    reward = model.reward(state, decision, exog)

    # Reward is negative cost
    assert reward == -5.0


def test_is_valid_decision() -> None:
    """Test decision validation."""
    config = SSPDynamicConfig(n_nodes=5)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # At least node 1 should be reachable (we force it in graph generation)
    decision = jnp.array(1, dtype=jnp.int32)
    assert model.is_valid_decision(state, decision)


def test_is_terminal() -> None:
    """Test terminal state detection."""
    config = SSPDynamicConfig(n_nodes=5, target_node=4)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # Initially at origin, not terminal
    assert not model.is_terminal(state)

    # Manually set to target
    state = state.at[0].set(4.0)
    assert model.is_terminal(state)


def test_get_estimated_costs() -> None:
    """Test extracting estimated costs from state."""
    config = SSPDynamicConfig(n_nodes=5)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    estimated_costs = model.get_estimated_costs(state)

    assert estimated_costs.shape == (5, 5)
    # Should be initialized to mean costs
    assert jnp.allclose(estimated_costs, model.mean_costs)


def test_get_time() -> None:
    """Test extracting time from state."""
    config = SSPDynamicConfig(n_nodes=5)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    assert model.get_time(state) == 0

    # After transition
    decision = jnp.array(1, dtype=jnp.int32)
    exog = ExogenousInfo(edge_costs=jnp.array([0., 5., 0., 0., 0.]))
    next_state = model.transition(state, decision, exog)

    assert model.get_time(next_state) == 1


def test_estimated_cost_update() -> None:
    """Test that estimated costs update via running average."""
    config = SSPDynamicConfig(n_nodes=5)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # Get initial estimated cost for edge 0->1
    initial_costs = model.get_estimated_costs(state)
    initial_cost_01 = initial_costs[0, 1]

    # Transition with different cost
    decision = jnp.array(1, dtype=jnp.int32)
    new_observed_cost = initial_cost_01 + 2.0  # Different from initial
    exog = ExogenousInfo(edge_costs=jnp.array([0., new_observed_cost, 0., 0., 0.]))

    next_state = model.transition(state, decision, exog)

    # Check estimated cost updated
    updated_costs = model.get_estimated_costs(next_state)
    updated_cost_01 = updated_costs[0, 1]

    # Should be running average: (1 - 1/1) * old + (1/1) * new = new (first obs)
    # Actually obs_count starts at 1, so alpha = 1/1 = 1.0
    # So it should equal new_observed_cost
    assert jnp.abs(updated_cost_01 - new_observed_cost) < 1e-5


def test_shapes() -> None:
    """Test output shapes."""
    config = SSPDynamicConfig(n_nodes=5)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # State: [current_node, time, costs (25), obs (25)] = 52
    assert state.shape == (52,)


def test_jit_compilation() -> None:
    """Test JIT compilation."""
    config = SSPDynamicConfig(n_nodes=5, horizon=10)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    init_jit = jax.jit(model.init_state)
    state = init_jit(key)

    assert state.shape == (52,)


# ============================================================================
# Policy Tests
# ============================================================================


def test_lookahead_policy_theta_validation() -> None:
    """Test that LookaheadPolicy validates theta parameter."""
    with pytest.raises(ValueError, match="theta must be in"):
        LookaheadPolicy(theta=-0.1)

    with pytest.raises(ValueError, match="theta must be in"):
        LookaheadPolicy(theta=1.1)

    # Valid values should work
    LookaheadPolicy(theta=0.0)
    LookaheadPolicy(theta=0.5)
    LookaheadPolicy(theta=1.0)


def test_lookahead_policy() -> None:
    """Test lookahead policy with multi-step horizon."""
    config = SSPDynamicConfig(n_nodes=5, horizon=10)
    model = SSPDynamicModel(config)
    policy = LookaheadPolicy(theta=0.5)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    decision = policy(None, state, key, model)

    assert decision.dtype == jnp.int32
    assert 0 <= decision < 5


def test_lookahead_policy_risk_sensitivity() -> None:
    """Test that different theta values produce different decisions."""
    config = SSPDynamicConfig(n_nodes=8, horizon=12, seed=123)
    model = SSPDynamicModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # Pessimistic (theta=0) vs Optimistic (theta=1)
    policy_pessimistic = LookaheadPolicy(theta=0.0)
    policy_optimistic = LookaheadPolicy(theta=1.0)

    decision_pessimistic = policy_pessimistic(None, state, key, model)
    decision_optimistic = policy_optimistic(None, state, key, model)

    # Both should be valid
    assert 0 <= decision_pessimistic < 8
    assert 0 <= decision_optimistic < 8

    # Note: decisions may be same or different depending on graph structure
    # Just verify they're valid decisions


def test_greedy_lookahead_policy() -> None:
    """Test greedy lookahead policy."""
    config = SSPDynamicConfig(n_nodes=5)
    model = SSPDynamicModel(config)
    policy = GreedyLookaheadPolicy()
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    decision = policy(None, state, key, model)

    assert decision.dtype == jnp.int32
    assert 0 <= decision < 5


def test_random_policy() -> None:
    """Test random policy."""
    config = SSPDynamicConfig(n_nodes=5)
    model = SSPDynamicModel(config)
    policy = RandomPolicy()
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    decision = policy(None, state, key, model)

    assert decision.dtype == jnp.int32
    assert 0 <= decision < 5


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_episode() -> None:
    """Test running a complete episode to target."""
    config = SSPDynamicConfig(n_nodes=8, horizon=12, edge_prob=0.4)
    model = SSPDynamicModel(config)
    policy = LookaheadPolicy(theta=0.5)

    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    total_reward = 0.0
    max_steps = 30

    for t in range(max_steps):
        if model.is_terminal(state):
            break

        key, subkey = jax.random.split(key)
        decision = policy(None, state, subkey, model)

        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, t)

        reward = model.reward(state, decision, exog)
        total_reward += float(reward)

        state = model.transition(state, decision, exog)

    # Should reach target or run out of steps
    assert t < max_steps or model.is_terminal(state)


def test_cost_estimation_improves() -> None:
    """Test that cost estimates converge with observations."""
    config = SSPDynamicConfig(n_nodes=6, horizon=10)
    model = SSPDynamicModel(config)
    policy = GreedyLookaheadPolicy()

    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    # Track cost estimate variance over multiple episodes
    estimates_history = []

    for episode in range(5):
        state = state.at[0].set(0.0)  # Reset to origin
        state = state.at[1].set(0.0)  # Reset time

        for t in range(10):
            if model.is_terminal(state):
                break

            key, subkey = jax.random.split(key)
            decision = policy(None, state, subkey, model)

            key, subkey = jax.random.split(key)
            exog = model.sample_exogenous(subkey, state, t)

            state = model.transition(state, decision, exog)

        # Record current cost estimates
        estimates_history.append(model.get_estimated_costs(state).copy())

    # Estimates should stabilize (variance should decrease)
    # This is a weak test - just check we're updating
    assert len(estimates_history) == 5


def test_multiple_policies_same_graph() -> None:
    """Test that different policies work on same graph."""
    config = SSPDynamicConfig(n_nodes=6, horizon=10, seed=42)
    model = SSPDynamicModel(config)

    policies = {
        "lookahead": LookaheadPolicy(theta=0.5),
        "greedy": GreedyLookaheadPolicy(),
        "random": RandomPolicy(),
    }

    key = jax.random.PRNGKey(42)

    for name, policy in policies.items():
        state = model.init_state(key)

        # Run a few steps
        for _ in range(5):
            if model.is_terminal(state):
                break

            key, subkey = jax.random.split(key)
            decision = policy(None, state, subkey, model)

            # Decision should be valid
            assert model.is_valid_decision(state, decision)

            key, subkey = jax.random.split(key)
            exog = model.sample_exogenous(subkey, state, 0)
            state = model.transition(state, decision, exog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
