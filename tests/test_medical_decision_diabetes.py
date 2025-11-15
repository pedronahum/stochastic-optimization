"""Tests for Medical Decision Diabetes problem (JAX-native implementation)."""

import pytest
import jax
import jax.numpy as jnp
import chex

from problems.medical_decision_diabetes import (
    MedicalDecisionDiabetesConfig,
    MedicalDecisionDiabetesModel,
    ExogenousInfo,
    UCBPolicy,
    IntervalEstimationPolicy,
    PureExploitationPolicy,
    PureExplorationPolicy,
    ThompsonSamplingPolicy,
)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_default_values() -> None:
    """Test that config has reasonable defaults."""
    config = MedicalDecisionDiabetesConfig()

    assert config.n_drugs == 5
    assert config.initial_mu == 0.5
    assert config.initial_sigma == 0.2
    assert config.measurement_sigma == 0.05
    assert config.use_fixed_truth is True


def test_config_validation_n_drugs() -> None:
    """Test that config validates n_drugs."""
    with pytest.raises(ValueError, match="n_drugs must be positive"):
        MedicalDecisionDiabetesConfig(n_drugs=0)

    with pytest.raises(ValueError, match="n_drugs must be positive"):
        MedicalDecisionDiabetesConfig(n_drugs=-1)


def test_config_validation_sigmas() -> None:
    """Test that config validates sigma parameters."""
    with pytest.raises(ValueError, match="initial_sigma must be positive"):
        MedicalDecisionDiabetesConfig(initial_sigma=0.0)

    with pytest.raises(ValueError, match="measurement_sigma must be positive"):
        MedicalDecisionDiabetesConfig(measurement_sigma=-0.1)


def test_config_immutability() -> None:
    """Test that config is immutable (frozen dataclass)."""
    config = MedicalDecisionDiabetesConfig()

    with pytest.raises((AttributeError, Exception)):
        config.n_drugs = 10  # Should fail - frozen dataclass


def test_config_get_true_mu_array() -> None:
    """Test getting true mu values as array."""
    config = MedicalDecisionDiabetesConfig()
    true_mu = config.get_true_mu_array()

    assert true_mu.shape == (5,)
    # Check that Peptide Analog (index 4) is best
    assert true_mu[4] == config.true_mu_PA
    assert true_mu[4] > true_mu[0]  # PA > Metformin


# ============================================================================
# Model Tests
# ============================================================================


def test_model_initialization() -> None:
    """Test model initializes correctly."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)

    assert model.config == config
    assert isinstance(model, MedicalDecisionDiabetesModel)
    assert model.true_mu.shape == (5,)


def test_init_state() -> None:
    """Test initial state generation."""
    config = MedicalDecisionDiabetesConfig(
        initial_mu=0.6,
        initial_sigma=0.3
    )
    model = MedicalDecisionDiabetesModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)

    # State should be [5 × 3]
    assert state.shape == (5, 3)

    # All drugs should start with same prior
    for i in range(5):
        assert state[i, 0] == 0.6  # mu_empirical
        assert state[i, 2] == 0.0  # N_trials


def test_sample_exogenous() -> None:
    """Test exogenous info sampling."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    # Try drug 0 (Metformin)
    exog = model.sample_exogenous(key, state, 0, decision=0)

    # Reduction should be close to true_mu with some noise
    assert exog.reduction.shape == ()
    assert exog.true_mu == model.true_mu[0]
    # Should be within reasonable range
    assert abs(exog.reduction - exog.true_mu) < 1.0


def test_transition_bayesian_update() -> None:
    """Test Bayesian belief update."""
    config = MedicalDecisionDiabetesConfig(measurement_sigma=0.05)
    model = MedicalDecisionDiabetesModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    initial_mu = state[0, 0]
    initial_beta = state[0, 1]

    # Try drug 0
    decision = jnp.array(0, dtype=jnp.int32)
    exog = ExogenousInfo(
        reduction=jnp.array(0.7),  # Observed high reduction
        true_mu=jnp.array(0.6),
        measurement_precision=jnp.array(1.0 / 0.05**2)
    )

    next_state = model.transition(state, decision, exog)

    # Beta should increase (more information)
    assert next_state[0, 1] > initial_beta

    # Mu should move toward observation
    assert next_state[0, 0] != initial_mu

    # Trial count should increment
    assert next_state[0, 2] == 1.0

    # Other drugs should be unchanged
    for i in range(1, 5):
        chex.assert_trees_all_close(next_state[i], state[i])


def test_reward_returns_true_mu() -> None:
    """Test reward is true mean effectiveness."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    decision = jnp.array(4, dtype=jnp.int32)  # Peptide Analog (best)

    exog = ExogenousInfo(
        reduction=jnp.array(0.65),
        true_mu=jnp.array(0.7),
        measurement_precision=jnp.array(400.0)
    )

    reward = model.reward(state, decision, exog)

    # Reward should be true_mu
    assert reward == 0.7


def test_is_valid_decision() -> None:
    """Test decision validation."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    # Valid decisions (0-4)
    for i in range(5):
        assert model.is_valid_decision(state, jnp.array(i, dtype=jnp.int32))

    # Invalid decisions
    assert not model.is_valid_decision(state, jnp.array(-1, dtype=jnp.int32))
    assert not model.is_valid_decision(state, jnp.array(5, dtype=jnp.int32))
    assert not model.is_valid_decision(state, jnp.array(10, dtype=jnp.int32))


def test_get_drug_names() -> None:
    """Test getting drug names."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)

    names = model.get_drug_names()

    assert len(names) == 5
    assert "Metformin" in names
    assert "Peptide Analog" in names


def test_get_posterior_std() -> None:
    """Test getting posterior standard deviation."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    std = model.get_posterior_std(state, 0)

    # Should be 1/sqrt(beta)
    expected_std = 1.0 / jnp.sqrt(state[0, 1])
    assert std == expected_std


def test_shapes_with_chex() -> None:
    """Test model output shapes using chex assertions."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)
    key = jax.random.PRNGKey(42)

    state = model.init_state(key)
    chex.assert_shape(state, (5, 3))
    chex.assert_rank(state, 2)

    exog = model.sample_exogenous(key, state, 0, decision=0)
    chex.assert_shape(exog.reduction, ())
    chex.assert_shape(exog.true_mu, ())


def test_batch_with_vmap() -> None:
    """Test model can be batched with vmap."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)
    key = jax.random.PRNGKey(42)

    batch_size = 10
    keys = jax.random.split(key, batch_size)

    # Batch init
    batch_states = jax.vmap(model.init_state)(keys)
    chex.assert_shape(batch_states, (batch_size, 5, 3))

    # Batch transition
    decisions = jnp.zeros(batch_size, dtype=jnp.int32)
    exogs_list = [model.sample_exogenous(keys[i], batch_states[i], 0, decision=0)
                  for i in range(batch_size)]

    # Create batch exog (note: need to stack properly)
    exogs = ExogenousInfo(
        reduction=jnp.array([e.reduction for e in exogs_list]),
        true_mu=jnp.array([e.true_mu for e in exogs_list]),
        measurement_precision=jnp.array([e.measurement_precision for e in exogs_list])
    )

    next_states = jax.vmap(model.transition)(batch_states, decisions, exogs)
    chex.assert_shape(next_states, (batch_size, 5, 3))


def test_jit_compilation() -> None:
    """Test that model methods can be JIT compiled."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)
    key = jax.random.PRNGKey(42)

    init_jit = jax.jit(model.init_state)
    state = init_jit(key)

    assert state.shape == (5, 3)


# ============================================================================
# Policy Tests
# ============================================================================


def test_ucb_policy() -> None:
    """Test UCB policy."""
    policy = UCBPolicy(theta=2.0)
    key = jax.random.PRNGKey(42)

    # Create state where drug 2 has highest mu
    state = jnp.array([
        [0.5, 100.0, 10.0],  # Drug 0
        [0.55, 100.0, 10.0],  # Drug 1
        [0.6, 100.0, 10.0],  # Drug 2 (best mean)
        [0.45, 100.0, 10.0],  # Drug 3
        [0.5, 100.0, 10.0],  # Drug 4
    ])

    decision = policy(None, state, key, time=20)

    # Should select drug with high UCB
    assert decision.dtype == jnp.int32
    assert 0 <= decision < 5


def test_interval_estimation_policy() -> None:
    """Test Interval Estimation policy."""
    policy = IntervalEstimationPolicy(theta=1.0)
    key = jax.random.PRNGKey(42)

    state = jnp.array([
        [0.5, 400.0, 5.0],  # High precision
        [0.55, 100.0, 2.0],  # Lower precision (more uncertainty)
        [0.45, 200.0, 3.0],
        [0.5, 150.0, 4.0],
        [0.5, 300.0, 6.0],
    ])

    decision = policy(None, state, key)

    assert decision.dtype == jnp.int32
    assert 0 <= decision < 5


def test_pure_exploitation_policy() -> None:
    """Test Pure Exploitation policy."""
    policy = PureExploitationPolicy()
    key = jax.random.PRNGKey(42)

    state = jnp.array([
        [0.5, 100.0, 10.0],
        [0.55, 100.0, 10.0],
        [0.7, 100.0, 10.0],  # Best mean (drug 2)
        [0.45, 100.0, 10.0],
        [0.6, 100.0, 10.0],
    ])

    decision = policy(None, state, key)

    # Should always pick drug with highest mean (drug 2)
    assert decision == 2


def test_pure_exploration_policy() -> None:
    """Test Pure Exploration policy."""
    policy = PureExplorationPolicy(n_drugs=5)
    key = jax.random.PRNGKey(42)

    state = jnp.zeros((5, 3))  # State doesn't matter

    # Run multiple times to check randomness
    decisions = [policy(None, state, jax.random.PRNGKey(i)) for i in range(20)]

    # Should have some variety
    unique_decisions = len(set([int(d) for d in decisions]))
    assert unique_decisions > 1  # Not always the same


def test_thompson_sampling_policy() -> None:
    """Test Thompson Sampling policy."""
    policy = ThompsonSamplingPolicy()
    key = jax.random.PRNGKey(42)

    state = jnp.array([
        [0.5, 100.0, 10.0],
        [0.55, 100.0, 10.0],
        [0.6, 100.0, 10.0],
        [0.45, 100.0, 10.0],
        [0.5, 100.0, 10.0],
    ])

    decision = policy(None, state, key)

    assert decision.dtype == jnp.int32
    assert 0 <= decision < 5


def test_policy_jit_compilation() -> None:
    """Test that policies can be JIT compiled."""
    policy = UCBPolicy(theta=2.0)
    key = jax.random.PRNGKey(42)
    state = jnp.zeros((5, 3))

    policy_jit = jax.jit(lambda s, k, t: policy(None, s, k, t))
    decision = policy_jit(state, key, 10)

    assert decision.dtype == jnp.int32


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_episode() -> None:
    """Test running a complete learning episode."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)
    policy = UCBPolicy(theta=2.0)

    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    total_reward = 0.0
    n_timesteps = 50

    for t in range(n_timesteps):
        # Select drug
        key, subkey = jax.random.split(key)
        decision = policy(None, state, subkey, time=t+1)

        # Sample outcome
        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, t, decision=int(decision))

        # Compute reward
        reward = model.reward(state, decision, exog)
        total_reward += float(reward)

        # Update state
        state = model.transition(state, decision, exog)

    # Should have accumulated positive rewards
    assert total_reward > 0
    # Should have tried some drugs
    assert jnp.sum(state[:, 2]) == n_timesteps


def test_learning_convergence() -> None:
    """Test that agent learns to select best drug."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)
    policy = UCBPolicy(theta=1.5)

    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    decisions_last_20 = []

    for t in range(100):
        key, subkey = jax.random.split(key)
        decision = policy(None, state, subkey, time=t+1)

        # Track last 20 decisions
        if t >= 80:
            decisions_last_20.append(int(decision))

        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, t, decision=int(decision))

        state = model.transition(state, decision, exog)

    # After learning, should pick best drug (Peptide Analog = 4) reasonably often
    # The policy balances exploration and exploitation, so not always best
    best_drug_count = sum(1 for d in decisions_last_20 if d == 4)

    # Should pick best drug at least 35% of the time (relaxed due to exploration)
    assert best_drug_count >= 7


def test_multiple_policies_comparison() -> None:
    """Test running multiple policies."""
    config = MedicalDecisionDiabetesConfig()
    model = MedicalDecisionDiabetesModel(config)

    policies = {
        "UCB": UCBPolicy(theta=2.0),
        "IE": IntervalEstimationPolicy(theta=1.0),
        "Greedy": PureExploitationPolicy(),
        "Thompson": ThompsonSamplingPolicy(),
    }

    key = jax.random.PRNGKey(42)
    results = {}

    for name, pol in policies.items():
        state = model.init_state(key)
        total_reward = 0.0

        for t in range(30):
            key, subkey = jax.random.split(key)

            if name == "UCB":
                decision = pol(None, state, subkey, time=t+1)
            else:
                decision = pol(None, state, subkey)

            key, subkey = jax.random.split(key)
            exog = model.sample_exogenous(subkey, state, t, decision=int(decision))

            reward = model.reward(state, decision, exog)
            total_reward += float(reward)

            state = model.transition(state, decision, exog)

        results[name] = total_reward

    # All should achieve positive rewards
    assert all(r > 0 for r in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
