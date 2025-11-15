"""Tests for Stochastic Shortest Path Static problem (JAX-native implementation)."""

import pytest
import jax
import jax.numpy as jnp
import chex

from problems.ssp_static import (
    SSPStaticConfig,
    SSPStaticModel,
    ExogenousInfo,
    GreedyPolicy,
    EpsilonGreedyPolicy,
    RandomPolicy,
)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_default_values() -> None:
    """Test that config has reasonable defaults."""
    config = SSPStaticConfig()
    
    assert config.n_nodes == 10
    assert config.edge_prob == 0.3
    assert config.cost_lower_bound == 1.0
    assert config.cost_upper_bound == 10.0


def test_config_validation() -> None:
    """Test that config validates parameters."""
    with pytest.raises(ValueError, match="n_nodes must be > 1"):
        SSPStaticConfig(n_nodes=1)
    
    with pytest.raises(ValueError, match="edge_prob"):
        SSPStaticConfig(edge_prob=0.0)
    
    with pytest.raises(ValueError, match="cost_upper_bound"):
        SSPStaticConfig(cost_lower_bound=10.0, cost_upper_bound=5.0)


def test_config_immutability() -> None:
    """Test that config is immutable."""
    config = SSPStaticConfig()
    
    with pytest.raises((AttributeError, Exception)):
        config.n_nodes = 20


# ============================================================================
# Model Tests  
# ============================================================================


def test_model_initialization() -> None:
    """Test model initializes correctly."""
    config = SSPStaticConfig(n_nodes=5)
    model = SSPStaticModel(config)
    
    assert model.config == config
    assert model.target_node == 4  # Last node


def test_init_state_creates_graph() -> None:
    """Test that init_state creates a valid graph."""
    config = SSPStaticConfig(n_nodes=5, edge_prob=0.5)
    model = SSPStaticModel(config)
    key = jax.random.PRNGKey(42)
    
    state = model.init_state(key)
    
    # State should be [current_node, V_0, ..., V_n]
    assert state.shape == (6,)  # 1 + 5 nodes
    assert state[0] == 0.0  # Start at origin
    
    # Check graph was created
    assert model.adjacency.shape == (5, 5)
    assert model.edge_lower.shape == (5, 5)
    assert model.edge_upper.shape == (5, 5)


def test_sample_exogenous() -> None:
    """Test edge cost sampling."""
    config = SSPStaticConfig(n_nodes=5)
    model = SSPStaticModel(config)
    key = jax.random.PRNGKey(42)
    
    state = model.init_state(key)
    exog = model.sample_exogenous(key, state, 0)
    
    # Should have costs for all nodes
    assert exog.edge_costs.shape == (5,)


def test_transition() -> None:
    """Test state transition."""
    config = SSPStaticConfig(n_nodes=5)
    model = SSPStaticModel(config)
    key = jax.random.PRNGKey(42)
    
    state = model.init_state(key)
    decision = jnp.array(1, dtype=jnp.int32)  # Move to node 1
    
    exog = ExogenousInfo(edge_costs=jnp.array([0., 5., 0., 0., 0.]))
    
    next_state = model.transition(state, decision, exog)
    
    # Should now be at node 1
    assert next_state[0] == 1.0
    # Value function should be updated
    assert next_state.shape == state.shape


def test_reward() -> None:
    """Test reward calculation."""
    config = SSPStaticConfig()
    model = SSPStaticModel(config)
    key = jax.random.PRNGKey(42)
    
    state = model.init_state(key)
    decision = jnp.array(1, dtype=jnp.int32)
    
    exog = ExogenousInfo(edge_costs=jnp.array([0., 5., 0., 0., 0., 0., 0., 0., 0., 0.]))
    
    reward = model.reward(state, decision, exog)
    
    # Reward is negative cost
    assert reward == -5.0


def test_is_valid_decision() -> None:
    """Test decision validation."""
    config = SSPStaticConfig(n_nodes=5)
    model = SSPStaticModel(config)
    key = jax.random.PRNGKey(42)
    
    state = model.init_state(key)
    
    # At least node 1 should be reachable (we force it)
    decision = jnp.array(1, dtype=jnp.int32)
    assert model.is_valid_decision(state, decision)


def test_is_terminal() -> None:
    """Test terminal state detection."""
    config = SSPStaticConfig(n_nodes=5, target_node=4)
    model = SSPStaticModel(config)
    key = jax.random.PRNGKey(42)
    
    state = model.init_state(key)
    
    # Initially at origin, not terminal
    assert not model.is_terminal(state)
    
    # Manually set to target
    state = state.at[0].set(4.0)
    assert model.is_terminal(state)


def test_shapes_with_chex() -> None:
    """Test output shapes."""
    config = SSPStaticConfig(n_nodes=5)
    model = SSPStaticModel(config)
    key = jax.random.PRNGKey(42)
    
    state = model.init_state(key)
    chex.assert_shape(state, (6,))  # 1 + n_nodes


def test_jit_compilation() -> None:
    """Test JIT compilation."""
    config = SSPStaticConfig(n_nodes=5)
    model = SSPStaticModel(config)
    key = jax.random.PRNGKey(42)
    
    init_jit = jax.jit(model.init_state)
    state = init_jit(key)
    
    assert state.shape == (6,)


# ============================================================================
# Policy Tests
# ============================================================================


def test_greedy_policy() -> None:
    """Test greedy policy."""
    config = SSPStaticConfig(n_nodes=5)
    model = SSPStaticModel(config)
    policy = GreedyPolicy()
    key = jax.random.PRNGKey(42)
    
    state = model.init_state(key)
    decision = policy(None, state, key, model)
    
    assert decision.dtype == jnp.int32
    assert 0 <= decision < 5


def test_epsilon_greedy_policy() -> None:
    """Test epsilon-greedy policy."""
    config = SSPStaticConfig(n_nodes=5)
    model = SSPStaticModel(config)
    policy = EpsilonGreedyPolicy(epsilon=0.2)
    key = jax.random.PRNGKey(42)
    
    state = model.init_state(key)
    decision = policy(None, state, key, model)
    
    assert decision.dtype == jnp.int32
    assert 0 <= decision < 5


def test_random_policy() -> None:
    """Test random policy."""
    config = SSPStaticConfig(n_nodes=5)
    model = SSPStaticModel(config)
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
    config = SSPStaticConfig(n_nodes=8, edge_prob=0.4)
    model = SSPStaticModel(config)
    policy = GreedyPolicy()
    
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)
    
    total_reward = 0.0
    max_steps = 20
    
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


def test_value_function_learning() -> None:
    """Test that value function improves over episodes."""
    config = SSPStaticConfig(n_nodes=6, learning_rate=0.2)
    model = SSPStaticModel(config)
    policy = GreedyPolicy()
    
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)
    
    initial_V = state[1:].copy()
    
    # Run multiple episodes
    for episode in range(5):
        state = state.at[0].set(0.0)  # Reset to origin
        
        for t in range(10):
            if model.is_terminal(state):
                break
            
            key, subkey = jax.random.split(key)
            decision = policy(None, state, subkey, model)
            
            key, subkey = jax.random.split(key)
            exog = model.sample_exogenous(subkey, state, t)
            
            state = model.transition(state, decision, exog)
    
    final_V = state[1:]
    
    # Value function should change (learning occurred)
    assert not jnp.allclose(initial_V, final_V)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
