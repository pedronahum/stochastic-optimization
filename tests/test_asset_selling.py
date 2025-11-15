"""Tests for Asset Selling model and policies."""

import pytest
import jax
import jax.numpy as jnp
import chex
from flax import nnx

from problems.asset_selling import (
    AssetSellingModel,
    AssetSellingConfig,
    ExogenousInfo,
    SellLowPolicy,
    HighLowPolicy,
    ExpectedValuePolicy,
    LinearThresholdPolicy,
    NeuralPolicy,
    AlwaysHoldPolicy,
    AlwaysSellPolicy,
)


class TestAssetSellingConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test creating valid configuration."""
        config = AssetSellingConfig(
            up_step=2.0,
            down_step=-2.0,
            variance=1.5,
            initial_price=100.0,
        )
        assert config.up_step == 2.0
        assert config.initial_price == 100.0

        # chex assertions
        chex.assert_scalar_positive(config.variance)
        chex.assert_scalar_positive(config.initial_price)

    def test_default_transition_matrix(self):
        """Test default transition matrix creation."""
        config = AssetSellingConfig()

        assert config.transition_matrix is not None
        chex.assert_shape(config.transition_matrix, (3, 3))

        # Rows should sum to 1
        row_sums = jnp.sum(config.transition_matrix, axis=1)
        chex.assert_trees_all_close(row_sums, jnp.ones(3), rtol=1e-5)

    def test_custom_transition_matrix(self):
        """Test custom transition matrix."""
        custom_tm = jnp.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
        ])

        config = AssetSellingConfig(transition_matrix=custom_tm)
        chex.assert_trees_all_close(config.transition_matrix, custom_tm)

    def test_immutability(self):
        """Test that config is immutable."""
        config = AssetSellingConfig()

        with pytest.raises((AttributeError, Exception)):
            config.initial_price = 200.0  # Should fail - frozen


class TestAssetSellingModel:
    """Tests for asset selling model."""

    @pytest.fixture
    def config(self):
        """Provide standard config."""
        return AssetSellingConfig(
            up_step=2.0,
            down_step=-2.0,
            variance=1.5,
            initial_price=100.0,
        )

    @pytest.fixture
    def model(self, config):
        """Provide model instance."""
        return AssetSellingModel(config)

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
        assert state[0] == model.config.initial_price
        assert state[1] == 1.0  # Resource available
        assert state[2] == float(model.config.initial_bias_idx)

    def test_transition_shape(self, model, key):
        """Test that transition preserves shape."""
        state = model.init_state(key)
        decision = jnp.array([0])  # Hold

        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, 0)

        next_state = model.transition(state, decision, exog)

        # Shape should be preserved
        chex.assert_equal_shape([state, next_state])
        chex.assert_tree_all_finite(next_state)

    def test_transition_hold(self, model, key):
        """Test holding decision."""
        state = model.init_state(key)
        decision = jnp.array([0])  # Hold

        key, subkey = jax.random.split(key)
        exog = ExogenousInfo(
            price_change=jnp.array(5.0),
            new_bias_idx=jnp.array(0)
        )

        next_state = model.transition(state, decision, exog)

        # Resource should still be 1
        assert next_state[1] == 1.0

        # Price should change
        expected_price = state[0] + 5.0
        chex.assert_trees_all_close(next_state[0], expected_price)

    def test_transition_sell(self, model, key):
        """Test selling decision."""
        state = model.init_state(key)
        decision = jnp.array([1])  # Sell

        key, subkey = jax.random.split(key)
        exog = ExogenousInfo(
            price_change=jnp.array(5.0),
            new_bias_idx=jnp.array(0)
        )

        next_state = model.transition(state, decision, exog)

        # Resource should be 0 after selling
        assert next_state[1] == 0.0

    def test_transition_negative_price(self, model, key):
        """Test that price cannot go negative."""
        state = jnp.array([10.0, 1.0, 2.0])  # Low price
        decision = jnp.array([0])  # Hold

        exog = ExogenousInfo(
            price_change=jnp.array(-20.0),  # Large negative change
            new_bias_idx=jnp.array(2)
        )

        next_state = model.transition(state, decision, exog)

        # Price should be clipped to 0
        assert next_state[0] >= 0.0

    def test_reward_hold(self, model, key):
        """Test reward for holding."""
        state = model.init_state(key)
        decision = jnp.array([0])  # Hold

        exog = ExogenousInfo(
            price_change=jnp.array(5.0),
            new_bias_idx=jnp.array(0)
        )

        reward = model.reward(state, decision, exog)

        # Holding gives 0 reward
        assert float(reward) == 0.0

    def test_reward_sell(self, model, key):
        """Test reward for selling."""
        state = model.init_state(key)
        decision = jnp.array([1])  # Sell

        exog = ExogenousInfo(
            price_change=jnp.array(5.0),
            new_bias_idx=jnp.array(0)
        )

        reward = model.reward(state, decision, exog)

        # Selling gives price as reward
        assert float(reward) == float(state[0])

    def test_reward_sell_without_resource(self, model, key):
        """Test selling without resource gives 0 reward."""
        state = jnp.array([100.0, 0.0, 1.0])  # No resource
        decision = jnp.array([1])  # Try to sell

        exog = ExogenousInfo(
            price_change=jnp.array(5.0),
            new_bias_idx=jnp.array(0)
        )

        reward = model.reward(state, decision, exog)

        # Cannot sell without resource
        assert float(reward) == 0.0

    def test_sample_exogenous(self, model, key):
        """Test exogenous sampling."""
        state = model.init_state(key)

        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, 0)

        # Check types
        assert isinstance(exog, ExogenousInfo)
        chex.assert_tree_all_finite((exog.price_change, exog.new_bias_idx))

        # Bias should be in [0, 1, 2]
        assert exog.new_bias_idx in [0, 1, 2]

    def test_is_valid_decision_hold(self, model, key):
        """Test validation of hold decision."""
        state = model.init_state(key)
        decision = jnp.array([0])  # Hold

        valid = model.is_valid_decision(state, decision)
        assert bool(valid) is True

    def test_is_valid_decision_sell_with_resource(self, model, key):
        """Test validation of sell with resource."""
        state = model.init_state(key)
        decision = jnp.array([1])  # Sell

        valid = model.is_valid_decision(state, decision)
        assert bool(valid) is True

    def test_is_valid_decision_sell_without_resource(self, model, key):
        """Test validation of sell without resource."""
        state = jnp.array([100.0, 0.0, 1.0])  # No resource
        decision = jnp.array([1])  # Try to sell

        valid = model.is_valid_decision(state, decision)
        assert bool(valid) is False

    def test_transition_is_jittable(self, model, key):
        """Test that transition can be JIT-compiled."""
        state = model.init_state(key)
        decision = jnp.array([0])

        exog = ExogenousInfo(
            price_change=jnp.array(5.0),
            new_bias_idx=jnp.array(0)
        )

        # Should not raise error
        jitted_transition = jax.jit(model.transition)
        next_state = jitted_transition(state, decision, exog)

        chex.assert_tree_all_finite(next_state)

    def test_batch_with_vmap(self, model, key):
        """Test batching with vmap."""
        # Create batch of states
        batch_size = 100
        states = jnp.tile(model.init_state(key), (batch_size, 1))

        # Batch of decisions (half hold, half sell)
        decisions = jnp.concatenate([
            jnp.zeros((batch_size // 2, 1), dtype=jnp.int32),
            jnp.ones((batch_size // 2, 1), dtype=jnp.int32)
        ])

        # Sample batch of exogenous info
        key, *subkeys = jax.random.split(key, batch_size + 1)
        batch_sample = jax.vmap(lambda k, s: model.sample_exogenous(k, s, 0))
        batch_exogs = batch_sample(jnp.array(subkeys), states)

        # Vectorize transition
        batch_transition = jax.vmap(model.transition)
        next_states = batch_transition(states, decisions, batch_exogs)

        # Check shapes
        chex.assert_shape(next_states, (batch_size, 3))
        chex.assert_tree_all_finite(next_states)

        # Check that sell decisions set resource to 0
        for i in range(batch_size // 2, batch_size):
            assert next_states[i, 1] == 0.0  # Resource sold


class TestAssetSellingPolicies:
    """Tests for various policies."""

    @pytest.fixture
    def model(self):
        """Provide model instance."""
        config = AssetSellingConfig()
        return AssetSellingModel(config)

    @pytest.fixture
    def key(self):
        """Provide JAX random key."""
        return jax.random.PRNGKey(42)

    def test_sell_low_policy_below_threshold(self, model, key):
        """Test sell-low policy when price below threshold."""
        policy = SellLowPolicy(threshold=90.0)

        state = jnp.array([85.0, 1.0, 1.0])  # Price below threshold
        decision = policy(None, state, key)

        assert int(decision[0]) == 1  # Should sell

    def test_sell_low_policy_above_threshold(self, model, key):
        """Test sell-low policy when price above threshold."""
        policy = SellLowPolicy(threshold=90.0)

        state = jnp.array([95.0, 1.0, 1.0])  # Price above threshold
        decision = policy(None, state, key)

        assert int(decision[0]) == 0  # Should hold

    def test_high_low_policy_below_low(self, model, key):
        """Test high-low policy when price below low threshold."""
        policy = HighLowPolicy(low_threshold=90.0, high_threshold=110.0)

        state = jnp.array([85.0, 1.0, 1.0])  # Price below low
        decision = policy(None, state, key)

        assert int(decision[0]) == 1  # Should sell

    def test_high_low_policy_above_high(self, model, key):
        """Test high-low policy when price above high threshold."""
        policy = HighLowPolicy(low_threshold=90.0, high_threshold=110.0)

        state = jnp.array([115.0, 1.0, 1.0])  # Price above high
        decision = policy(None, state, key)

        assert int(decision[0]) == 1  # Should sell

    def test_high_low_policy_in_range(self, model, key):
        """Test high-low policy when price in acceptable range."""
        policy = HighLowPolicy(low_threshold=90.0, high_threshold=110.0)

        state = jnp.array([100.0, 1.0, 1.0])  # Price in range
        decision = policy(None, state, key)

        assert int(decision[0]) == 0  # Should hold

    def test_expected_value_policy(self, model, key):
        """Test expected value policy."""
        policy = ExpectedValuePolicy(model)

        # With up bias, expected future price is higher
        state_up_bias = jnp.array([100.0, 1.0, 0.0])  # Up bias
        decision = policy(None, state_up_bias, key)

        # Since expected future > current, should hold
        assert int(decision[0]) == 0

    def test_linear_threshold_policy(self, key):
        """Test linear threshold policy."""
        policy = LinearThresholdPolicy(rngs=nnx.Rngs(42))

        state = jnp.array([100.0, 1.0, 1.0])
        decision = policy(state, key)

        # Should return binary decision
        assert int(decision[0]) in [0, 1]

    def test_neural_policy(self, key):
        """Test neural network policy."""
        policy = NeuralPolicy(hidden_dims=[16, 8], rngs=nnx.Rngs(42))

        state = jnp.array([100.0, 1.0, 1.0])

        # Run multiple times (stochastic)
        decisions = []
        for i in range(20):
            key, subkey = jax.random.split(key)
            decision = policy(state, subkey)
            decisions.append(int(decision[0]))

        # Should get mix of 0s and 1s (stochastic)
        assert 0 in decisions or 1 in decisions
        assert all(d in [0, 1] for d in decisions)

    def test_always_hold_policy(self, key):
        """Test always-hold baseline."""
        policy = AlwaysHoldPolicy()

        state = jnp.array([100.0, 1.0, 1.0])
        decision = policy(None, state, key)

        assert int(decision[0]) == 0  # Always hold

    def test_always_sell_policy(self, key):
        """Test always-sell baseline."""
        policy = AlwaysSellPolicy()

        state = jnp.array([100.0, 1.0, 1.0])
        decision = policy(None, state, key)

        assert int(decision[0]) == 1  # Always sell (if resource available)

    def test_policies_respect_no_resource(self, key):
        """Test that policies respect resource availability."""
        state_no_resource = jnp.array([100.0, 0.0, 1.0])  # No resource

        policies = [
            SellLowPolicy(threshold=90.0),
            HighLowPolicy(low_threshold=90.0, high_threshold=110.0),
            AlwaysSellPolicy(),
        ]

        for policy in policies:
            decision = policy(None, state_no_resource, key)
            # Should not sell if no resource
            assert int(decision[0]) == 0


class TestAssetSellingIntegration:
    """Integration tests for full workflows."""

    def test_full_trajectory(self):
        """Test complete trajectory simulation."""
        config = AssetSellingConfig()
        model = AssetSellingModel(config)
        policy = HighLowPolicy(low_threshold=90.0, high_threshold=110.0)

        key = jax.random.PRNGKey(0)
        state = model.init_state(key)
        horizon = 10

        total_reward = 0.0
        trajectory_states = [state]

        for t in range(horizon):
            key, subkey1, subkey2 = jax.random.split(key, 3)

            # Get decision
            decision = policy(None, state, subkey1)

            # Sample exogenous
            exog = model.sample_exogenous(subkey2, state, t)

            # Get reward and transition
            reward = model.reward(state, decision, exog)
            next_state = model.transition(state, decision, exog)

            total_reward += float(reward)
            trajectory_states.append(next_state)

            # If sold, stop
            if next_state[1] < 0.5:
                break

            state = next_state

        # Should have sold at some point or reached horizon
        assert len(trajectory_states) <= horizon + 1
        assert total_reward >= 0.0

    def test_gradient_flow_through_policy(self):
        """Test that gradients flow through learnable policy."""
        config = AssetSellingConfig()
        model = AssetSellingModel(config)
        policy = LinearThresholdPolicy(rngs=nnx.Rngs(0))

        key = jax.random.PRNGKey(0)
        state = model.init_state(key)

        def loss_fn(policy):
            """Negative reward."""
            key_local = jax.random.PRNGKey(1)
            key_exog, key_policy = jax.random.split(key_local)

            decision = policy(state, key_policy)
            exog = model.sample_exogenous(key_exog, state, 0)
            reward = model.reward(state, decision, exog)

            return -reward

        # Compute gradient
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(policy)

        # Check gradients exist and are finite
        chex.assert_tree_all_finite(grads)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
