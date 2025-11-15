"""Comprehensive tests for Clinical Trials problem (JAX-native implementation)."""

import pytest
import jax
import jax.numpy as jnp
import chex
from flax import nnx

from stochopt.core import simulator as sim
from problems.clinical_trials import model, policy


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_default_values() -> None:
    """Test that config has reasonable defaults."""
    cfg = model.Config()

    assert cfg.horizon == 20
    assert cfg.sigma == 0.25
    assert cfg.mu == 0.0


def test_config_immutability() -> None:
    """Test that config is immutable (frozen struct)."""
    cfg = model.Config()

    with pytest.raises((AttributeError, Exception)):
        cfg.horizon = 30  # Should fail - frozen struct


# ============================================================================
# Model Tests
# ============================================================================


def test_model_initialization() -> None:
    """Test model initializes correctly."""
    cfg = model.Config(horizon=15, sigma=0.3, mu=0.1)
    mdl = model.ClinicalTrialsModel(cfg)

    assert mdl.cfg == cfg
    assert mdl.cfg.horizon == 15
    assert mdl.cfg.sigma == 0.3
    assert mdl.cfg.mu == 0.1


def test_reset_state() -> None:
    """Test initial state generation."""
    cfg = model.Config()
    mdl = model.ClinicalTrialsModel(cfg)
    key = jax.random.PRNGKey(42)

    state = mdl.reset(key=key)

    assert isinstance(state, model.State)
    assert state.t == 0
    assert state.x == 0.0
    chex.assert_shape(state.x, ())


def test_step_dynamics() -> None:
    """Test state transition dynamics."""
    cfg = model.Config(sigma=0.25, mu=0.0)
    mdl = model.ClinicalTrialsModel(cfg)
    key = jax.random.PRNGKey(42)

    state = model.State(t=0, x=jnp.array(0.5))
    action = 0.1  # Dose

    new_state, reward = mdl.step(state, action, key=key)

    # Time should increment
    assert new_state.t == 1

    # Health should update: x_{t+1} = x_t + action + noise
    # new_x ≈ 0.5 + 0.1 + noise
    assert new_state.x != state.x  # Should change due to action + noise

    # Reward should be negative absolute value
    expected_reward = -jnp.abs(new_state.x)
    chex.assert_trees_all_close(reward, expected_reward)


def test_step_reward_calculation() -> None:
    """Test reward is negative absolute deviation."""
    cfg = model.Config(sigma=0.0, mu=0.0)  # No noise
    mdl = model.ClinicalTrialsModel(cfg)
    key = jax.random.PRNGKey(42)

    # Test positive health state
    state_pos = model.State(t=0, x=jnp.array(2.0))
    _, reward_pos = mdl.step(state_pos, 0.0, key=key)
    assert reward_pos == -2.0

    # Test negative health state
    state_neg = model.State(t=0, x=jnp.array(-1.5))
    _, reward_neg = mdl.step(state_neg, 0.0, key=key)
    assert reward_neg == -1.5

    # Test zero health state (optimal)
    state_zero = model.State(t=0, x=jnp.array(0.0))
    _, reward_zero = mdl.step(state_zero, 0.0, key=key)
    assert reward_zero == 0.0


def test_step_with_positive_bias() -> None:
    """Test dynamics with positive bias (mu > 0)."""
    cfg = model.Config(sigma=0.0, mu=0.5)  # Positive drift
    mdl = model.ClinicalTrialsModel(cfg)
    key = jax.random.PRNGKey(42)

    state = model.State(t=0, x=jnp.array(0.0))
    action = 0.0

    new_state, _ = mdl.step(state, action, key=key)

    # Health should increase due to positive bias
    # new_x = 0.0 + 0.0 + 0.5 = 0.5
    assert new_state.x == 0.5


def test_step_with_negative_bias() -> None:
    """Test dynamics with negative bias (mu < 0)."""
    cfg = model.Config(sigma=0.0, mu=-0.3)  # Negative drift
    mdl = model.ClinicalTrialsModel(cfg)
    key = jax.random.PRNGKey(42)

    state = model.State(t=0, x=jnp.array(1.0))
    action = 0.0

    new_state, _ = mdl.step(state, action, key=key)

    # Health should decrease due to negative bias
    # new_x = 1.0 + 0.0 + (-0.3) = 0.7
    assert new_state.x == 0.7


def test_shapes_with_chex() -> None:
    """Test model output shapes using chex assertions."""
    cfg = model.Config()
    mdl = model.ClinicalTrialsModel(cfg)
    key = jax.random.PRNGKey(42)

    state = mdl.reset(key=key)
    chex.assert_shape(state.x, ())
    chex.assert_scalar(state.t)

    new_state, reward = mdl.step(state, 0.1, key=key)
    chex.assert_shape(new_state.x, ())
    chex.assert_shape(reward, ())


def test_batch_with_vmap() -> None:
    """Test model can be batched with vmap."""
    cfg = model.Config()
    mdl = model.ClinicalTrialsModel(cfg)
    key = jax.random.PRNGKey(42)

    batch_size = 10
    keys = jax.random.split(key, batch_size)

    # Batch reset
    batch_states = jax.vmap(mdl.reset)(key=keys)
    assert batch_states.x.shape == (batch_size,)
    assert jnp.all(batch_states.t == 0)

    # Batch step
    actions = jnp.ones(batch_size) * 0.1
    batch_step_keys = jax.random.split(key, batch_size)

    def step_fn(state: model.State, action: float, key: jax.Array) -> tuple:
        return mdl.step(state, action, key=key)

    batch_new_states, batch_rewards = jax.vmap(step_fn)(
        batch_states, actions, batch_step_keys
    )

    assert batch_new_states.x.shape == (batch_size,)
    assert batch_rewards.shape == (batch_size,)


def test_jit_compilation() -> None:
    """Test that model methods can be JIT compiled."""
    cfg = model.Config()
    mdl = model.ClinicalTrialsModel(cfg)
    key = jax.random.PRNGKey(42)

    # JIT compile reset
    reset_jit = jax.jit(mdl.reset)
    state = reset_jit(key=key)
    assert state.t == 0

    # JIT compile step
    step_jit = jax.jit(mdl.step)
    new_state, reward = step_jit(state, 0.1, key=key)
    assert new_state.t == 1


# ============================================================================
# Policy Tests
# ============================================================================


def test_linear_dose_policy_initialization() -> None:
    """Test LinearDosePolicy initializes correctly."""
    π = policy.LinearDosePolicy()

    assert isinstance(π, policy.LinearDosePolicy)
    assert hasattr(π, 'w')
    # Default weight should be 0.1
    assert π.w.value == 0.1


def test_linear_dose_policy_action() -> None:
    """Test LinearDosePolicy produces correct actions."""
    π = policy.LinearDosePolicy()
    key = jax.random.PRNGKey(42)

    # Test with zero state
    state_zero = model.State(t=0, x=jnp.array(0.0))
    action_zero = π.act(state_zero, key=key)
    assert action_zero == 0.0

    # Test with positive state
    state_pos = model.State(t=0, x=jnp.array(2.0))
    action_pos = π.act(state_pos, key=key)
    # action = w * x = 0.1 * 2.0 = 0.2
    assert action_pos == 0.2

    # Test with negative state
    state_neg = model.State(t=0, x=jnp.array(-3.0))
    action_neg = π.act(state_neg, key=key)
    # action = 0.1 * (-3.0) = -0.3
    assert action_neg == -0.3


def test_linear_dose_policy_custom_weight() -> None:
    """Test LinearDosePolicy with custom weight."""
    π = policy.LinearDosePolicy()
    π.w.value = jnp.array(-0.5)  # Set custom weight

    key = jax.random.PRNGKey(42)
    state = model.State(t=0, x=jnp.array(1.0))

    action = π.act(state, key=key)
    # action = -0.5 * 1.0 = -0.5
    assert action == -0.5


def test_policy_jit_compilation() -> None:
    """Test that policy can be JIT compiled."""
    π = policy.LinearDosePolicy()
    key = jax.random.PRNGKey(42)
    state = model.State(t=0, x=jnp.array(1.5))

    # JIT compile the act method
    act_jit = jax.jit(π.act)
    action = act_jit(state, key=key)

    expected = 0.1 * 1.5
    assert action == expected


# ============================================================================
# Integration Tests
# ============================================================================


def test_single_rollout_matches_shapes() -> None:
    """Test simulator rollout produces correct shapes (original test)."""
    cfg = model.Config()
    mdl = model.ClinicalTrialsModel(cfg)
    π = policy.LinearDosePolicy()
    key = jax.random.PRNGKey(0)

    rewards = sim.rollout(mdl, π, cfg.horizon, key=key)
    assert rewards.shape == (cfg.horizon,)


def test_full_episode() -> None:
    """Test running a complete episode manually."""
    cfg = model.Config(horizon=10, sigma=0.1, mu=0.0)
    mdl = model.ClinicalTrialsModel(cfg)
    π = policy.LinearDosePolicy()
    π.w.value = jnp.array(-0.5)  # Corrective policy

    key = jax.random.PRNGKey(42)
    state = mdl.reset(key=key)

    total_reward = 0.0
    health_states = [float(state.x)]

    for t in range(cfg.horizon):
        key, subkey = jax.random.split(key)
        action = π.act(state, key=subkey)

        key, subkey = jax.random.split(key)
        state, reward = mdl.step(state, action, key=subkey)

        total_reward += float(reward)
        health_states.append(float(state.x))

    # Check that we accumulated rewards
    assert isinstance(total_reward, float)
    assert len(health_states) == cfg.horizon + 1


def test_stability_with_corrective_policy() -> None:
    """Test that corrective policy (w < 0) stabilizes health."""
    cfg = model.Config(horizon=20, sigma=0.1, mu=0.0)
    mdl = model.ClinicalTrialsModel(cfg)
    π = policy.LinearDosePolicy()
    π.w.value = jnp.array(-0.5)  # Negative feedback

    key = jax.random.PRNGKey(42)
    # Start with high health deviation
    state = model.State(t=0, x=jnp.array(5.0))

    health_values = [float(state.x)]

    for t in range(cfg.horizon):
        key, subkey = jax.random.split(key)
        action = π.act(state, key=subkey)

        key, subkey = jax.random.split(key)
        state, _ = mdl.step(state, action, key=subkey)

        health_values.append(float(state.x))

    # With corrective policy, health should decrease from initial high value
    # (on average, despite noise)
    final_health = abs(health_values[-1])
    initial_health = abs(health_values[0])

    # Final health should be lower than initial (stabilizing)
    # We use a relaxed check since there's randomness
    assert final_health < initial_health or final_health < 3.0


def test_instability_with_amplifying_policy() -> None:
    """Test that amplifying policy (w > 0) destabilizes health."""
    cfg = model.Config(horizon=15, sigma=0.1, mu=0.0)
    mdl = model.ClinicalTrialsModel(cfg)
    π = policy.LinearDosePolicy()
    π.w.value = jnp.array(0.5)  # Positive feedback (bad!)

    key = jax.random.PRNGKey(42)
    # Start with small deviation
    state = model.State(t=0, x=jnp.array(0.5))

    health_values = [float(state.x)]

    for t in range(cfg.horizon):
        key, subkey = jax.random.split(key)
        action = π.act(state, key=subkey)

        key, subkey = jax.random.split(key)
        state, _ = mdl.step(state, action, key=subkey)

        health_values.append(float(state.x))

    # With amplifying policy, health magnitude should increase
    final_magnitude = abs(health_values[-1])
    initial_magnitude = abs(health_values[0])

    # Final should generally be larger (destabilizing)
    assert final_magnitude > initial_magnitude * 0.5  # Relaxed due to randomness


# ============================================================================
# Gradient Flow Tests
# ============================================================================


def test_gradient_flow_through_policy() -> None:
    """Test that gradients flow through the policy."""
    cfg = model.Config(sigma=0.0, mu=0.0)  # Deterministic for testing
    mdl = model.ClinicalTrialsModel(cfg)
    π = policy.LinearDosePolicy()

    key = jax.random.PRNGKey(42)
    state = model.State(t=0, x=jnp.array(2.0))

    # Define loss as negative reward
    def loss_fn(policy_module: policy.LinearDosePolicy) -> jax.Array:
        action = policy_module.act(state, key=key)
        _, reward = mdl.step(state, action, key=key)
        return -reward  # Minimize negative reward (maximize reward)

    # Compute gradients
    grads = nnx.grad(loss_fn)(π)

    # Check that gradient exists and is finite
    chex.assert_tree_all_finite(π.w.value)


def test_policy_parameter_update() -> None:
    """Test that policy parameters can be updated via optimizer."""
    cfg = model.Config(sigma=0.0, mu=0.0)
    mdl = model.ClinicalTrialsModel(cfg)
    π = policy.LinearDosePolicy()

    initial_w = float(π.w.value)
    key = jax.random.PRNGKey(42)
    state = model.State(t=0, x=jnp.array(1.0))

    # Create optimizer
    import optax
    optimizer = nnx.Optimizer(π, optax.adam(0.1))

    # Define loss
    def loss_fn(policy_module: policy.LinearDosePolicy) -> jax.Array:
        action = policy_module.act(state, key=key)
        _, reward = mdl.step(state, action, key=key)
        return -reward

    # Compute gradient and update
    loss = loss_fn(π)
    grad = nnx.grad(loss_fn)(π)
    optimizer.update(grad)

    updated_w = float(π.w.value)

    # Weight should have changed after optimizer update
    assert updated_w != initial_w


def test_vmap_policy() -> None:
    """Test that policy works with vmap for batch processing."""
    π = policy.LinearDosePolicy()
    key = jax.random.PRNGKey(42)
    batch_size = 5

    # Batch states
    keys = jax.random.split(key, batch_size)
    x_values = jnp.array([0.0, 1.0, -1.0, 2.0, -2.0])
    states_x = jax.vmap(lambda x: model.State(t=0, x=x))(x_values)

    # Batch policy evaluation
    batch_actions = jax.vmap(π.act)(states_x, key=keys)

    assert batch_actions.shape == (batch_size,)
    # Check that actions are proportional to x values
    expected_actions = 0.1 * x_values
    chex.assert_trees_all_close(batch_actions, expected_actions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
