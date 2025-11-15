"""JAX-native policies for Asset Selling.

This module implements various decision policies for the asset selling problem:
- Threshold policies (sell_low, high_low)
- Parametric neural network policies
- Myopic/greedy policies
"""

from typing import Optional, List
from functools import partial
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
import jax
import jax.numpy as jnp
import chex
from flax import nnx


# Type aliases
State = Float[Array, "3"]  # [price, resource, bias_idx]
Decision = Int[Array, "1"]  # [sell]
Key = PRNGKeyArray


@chex.dataclass(frozen=True)
class ThresholdPolicyConfig:
    """Configuration for threshold-based policies.

    Attributes:
        sell_low: Lower price threshold for sell_low policy.
        sell_high: Upper price threshold for high_low policy.
    """
    sell_low: float = 90.0
    sell_high: float = 110.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        chex.assert_scalar_positive(self.sell_low)
        chex.assert_scalar_positive(self.sell_high)
        if self.sell_high < self.sell_low:
            raise ValueError(
                f"sell_high ({self.sell_high}) must be >= sell_low ({self.sell_low})"
            )


class SellLowPolicy:
    """Sell-low policy: Sell if price drops below threshold.

    This policy sells the asset when the price falls below a specified
    threshold, implementing a stop-loss strategy.

    Example:
        >>> policy = SellLowPolicy(threshold=90.0)
        >>> state = jnp.array([85.0, 1.0, 1.0])  # Price below threshold
        >>> decision = policy(None, state, key)
        >>> print(decision)  # [1] - sell
    """

    def __init__(self, threshold: float = 90.0) -> None:
        """Initialize policy.

        Args:
            threshold: Price below which to sell.
        """
        self.threshold = threshold

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Get decision based on sell-low rule.

        Args:
            params: Unused (no learnable parameters).
            state: Current state [price, resource, bias_idx].
            key: Random key (unused for deterministic policy).

        Returns:
            Decision: [1] if price < threshold and resource available, else [0].
        """
        price, resource, _ = state[0], state[1], state[2]

        # Sell if price below threshold and resource available
        sell = jnp.where(
            (price < self.threshold) & (resource > 0.5),
            1,
            0
        )

        return jnp.array([sell])


class HighLowPolicy:
    """High-low policy: Sell if price is too low or too high.

    This policy combines stop-loss (sell if too low) with profit-taking
    (sell if too high), implementing a bounded price strategy.

    Example:
        >>> policy = HighLowPolicy(low_threshold=90.0, high_threshold=110.0)
        >>> state = jnp.array([115.0, 1.0, 1.0])  # Price above high threshold
        >>> decision = policy(None, state, key)
        >>> print(decision)  # [1] - sell
    """

    def __init__(self, low_threshold: float = 90.0, high_threshold: float = 110.0) -> None:
        """Initialize policy.

        Args:
            low_threshold: Lower price threshold (stop-loss).
            high_threshold: Upper price threshold (take-profit).
        """
        if high_threshold < low_threshold:
            raise ValueError(
                f"high_threshold ({high_threshold}) must be >= low_threshold ({low_threshold})"
            )
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Get decision based on high-low rule.

        Args:
            params: Unused (no learnable parameters).
            state: Current state [price, resource, bias_idx].
            key: Random key (unused for deterministic policy).

        Returns:
            Decision: [1] if price outside bounds and resource available, else [0].
        """
        price, resource, _ = state[0], state[1], state[2]

        # Sell if price outside [low, high] bounds and resource available
        sell = jnp.where(
            ((price < self.low_threshold) | (price > self.high_threshold)) & (resource > 0.5),
            1,
            0
        )

        return jnp.array([sell])


class ExpectedValuePolicy:
    """Expected value policy: Sell if current price exceeds expected future value.

    This myopic policy compares the current price to the expected price
    in the next period, accounting for the bias state.

    Example:
        >>> from stochopt.problems.asset_selling.model import AssetSellingModel, AssetSellingConfig
        >>> config = AssetSellingConfig()
        >>> model = AssetSellingModel(config)
        >>> policy = ExpectedValuePolicy(model)
    """

    def __init__(self, model: "AssetSellingModel") -> None:  # type: ignore[name-defined]
        """Initialize policy.

        Args:
            model: AssetSellingModel instance for computing expected values.
        """
        self.model = model

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Get decision based on expected value comparison.

        Args:
            params: Unused (no learnable parameters).
            state: Current state [price, resource, bias_idx].
            key: Random key (unused for deterministic policy).

        Returns:
            Decision: [1] if current price > expected future price, else [0].
        """
        price, resource, bias_idx = state[0], state[1], state[2]

        # Compute expected price change based on current bias
        # E[price_change | bias] = bias_step (variance averages to 0)
        bias_idx_int = jnp.array(bias_idx, dtype=jnp.int32)
        expected_step = self.model.bias_steps[bias_idx_int]

        # Expected future price (ignoring bias transitions for simplicity)
        expected_future_price = price + expected_step

        # Sell if current price exceeds expected future price
        sell = jnp.where(
            (price > expected_future_price) & (resource > 0.5),
            1,
            0
        )

        return jnp.array([sell])


class LinearThresholdPolicy(nnx.Module):
    """Learnable linear threshold policy.

    The selling threshold is a linear function of state features:
        threshold = w0 + w1 * bias_indicator

    This allows the policy to learn different thresholds for different
    bias states (Up/Neutral/Down).

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> policy = LinearThresholdPolicy(rngs=nnx.Rngs(0))
        >>> state = jnp.array([100.0, 1.0, 1.0])
        >>> decision = policy(state, key)
    """

    def __init__(self, rngs: nnx.Rngs) -> None:
        """Initialize policy with random parameters.

        Args:
            rngs: Flax NNX random number generator state.
        """
        # Initialize learnable parameters
        # w[0] = base threshold, w[1] = Up bias adjustment, w[2] = Down bias adjustment
        self.weights = nnx.Param(
            jax.random.uniform(rngs(), shape=(3,), minval=80.0, maxval=120.0)
        )

    def __call__(self, state: State, key: Key) -> Decision:
        """Get decision based on learned threshold.

        Args:
            state: Current state [price, resource, bias_idx].
            key: Random key (unused for deterministic policy).

        Returns:
            Decision: [1] if should sell, else [0].
        """
        price, resource, bias_idx = state[0], state[1], state[2]

        # Compute threshold based on bias
        # threshold = base + adjustment[bias]
        bias_idx_int = jnp.array(bias_idx, dtype=jnp.int32)

        # Create one-hot encoding of bias
        bias_onehot = jax.nn.one_hot(bias_idx_int, 3)

        # Compute threshold: w[0] is base, w[1:] are adjustments
        base_threshold = self.weights.value[0]
        bias_adjustment = jnp.dot(bias_onehot, self.weights.value)

        threshold = base_threshold + bias_adjustment

        # Sell if price below threshold
        sell = jnp.where(
            (price < threshold) & (resource > 0.5),
            1,
            0
        )

        return jnp.array([sell])


class NeuralPolicy(nnx.Module):
    """Neural network policy for asset selling.

    This policy uses a small feedforward neural network to map state
    to selling probability, then samples a binary decision.

    The network architecture:
        state -> [hidden1] -> [hidden2] -> logit -> sigmoid -> prob

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> policy = NeuralPolicy(hidden_dims=[16, 16], rngs=nnx.Rngs(0))
        >>> state = jnp.array([100.0, 1.0, 1.0])
        >>> key_decision = jax.random.PRNGKey(1)
        >>> decision = policy(state, key_decision)
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        rngs: Optional[nnx.Rngs] = None
    ) -> None:
        """Initialize neural network policy.

        Args:
            hidden_dims: List of hidden layer dimensions.
            rngs: Flax NNX random number generator state.
        """
        if hidden_dims is None:
            hidden_dims = [32, 16]
        if rngs is None:
            rngs = nnx.Rngs(0)

        # State dimension is 3 (price, resource, bias_idx)
        state_dim = 3

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim

        # Output layer: single logit for selling probability
        layers.append(nnx.Linear(prev_dim, 1, rngs=rngs))

        self.layers = layers

    def __call__(self, state: State, key: Key) -> Decision:
        """Get decision from neural network.

        Args:
            state: Current state [price, resource, bias_idx].
            key: Random key for stochastic decision.

        Returns:
            Decision: [0] or [1] sampled from learned distribution.
        """
        # Forward pass through network
        x = state

        # Hidden layers with ReLU activation
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)

        # Output layer (logit)
        logit = self.layers[-1](x)

        # Convert to probability
        prob_sell = nnx.sigmoid(logit[0])

        # Sample binary decision
        decision = jax.random.bernoulli(key, prob_sell)

        return jnp.array([decision], dtype=jnp.int32)


class AlwaysHoldPolicy:
    """Baseline policy that never sells (always holds).

    Useful as a baseline for comparison. This policy only sells
    at the terminal time when forced.
    """

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Always return hold decision.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).

        Returns:
            Decision: Always [0] (hold).
        """
        return jnp.array([0])


class AlwaysSellPolicy:
    """Baseline policy that sells immediately if possible.

    Useful as a baseline for comparison.
    """

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Sell if resource available.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).

        Returns:
            Decision: [1] if resource available, else [0].
        """
        resource = state[1]
        sell = jnp.where(resource > 0.5, 1, 0)
        return jnp.array([sell])


# Example usage
if __name__ == "__main__":
    print("JAX-Native Asset Selling Policies")
    print("=" * 70)

    # Create test state
    state_low = jnp.array([85.0, 1.0, 1.0])  # Low price, resource available
    state_high = jnp.array([115.0, 1.0, 0.0])  # High price, resource available
    state_mid = jnp.array([100.0, 1.0, 1.0])  # Mid price, resource available
    state_sold = jnp.array([100.0, 0.0, 1.0])  # Resource already sold

    key = jax.random.PRNGKey(42)

    # Test SellLowPolicy
    print("\n1. SellLowPolicy (threshold=90)")
    print("-" * 70)
    policy_sl = SellLowPolicy(threshold=90.0)

    for name, state in [("Low price", state_low), ("Mid price", state_mid), ("High price", state_high)]:
        decision = policy_sl(None, state, key)
        print(f"  {name} (${state[0]:.0f}): Decision = {int(decision[0])} ({'Sell' if decision[0] else 'Hold'})")

    # Test HighLowPolicy
    print("\n2. HighLowPolicy (low=90, high=110)")
    print("-" * 70)
    policy_hl = HighLowPolicy(low_threshold=90.0, high_threshold=110.0)

    for name, state in [("Low price", state_low), ("Mid price", state_mid), ("High price", state_high)]:
        decision = policy_hl(None, state, key)
        print(f"  {name} (${state[0]:.0f}): Decision = {int(decision[0])} ({'Sell' if decision[0] else 'Hold'})")

    # Test LinearThresholdPolicy
    print("\n3. LinearThresholdPolicy (learnable)")
    print("-" * 70)
    policy_lt = LinearThresholdPolicy(rngs=nnx.Rngs(42))

    print(f"  Learned weights: {policy_lt.weights.value}")

    for name, state in [("Low price", state_low), ("Mid price", state_mid), ("High price", state_high)]:
        decision = policy_lt(state, key)
        print(f"  {name} (${state[0]:.0f}): Decision = {int(decision[0])} ({'Sell' if decision[0] else 'Hold'})")

    # Test NeuralPolicy
    print("\n4. NeuralPolicy (stochastic)")
    print("-" * 70)
    policy_nn = NeuralPolicy(hidden_dims=[16, 8], rngs=nnx.Rngs(42))

    # Run multiple times to see stochasticity
    print(f"  Running neural policy 10 times on mid-price state:")
    decisions = []
    for i in range(10):
        key, subkey = jax.random.split(key)
        decision = policy_nn(state_mid, subkey)
        decisions.append(int(decision[0]))

    print(f"  Decisions: {decisions}")
    print(f"  Sell probability: {sum(decisions)/len(decisions):.2f}")

    # Test with sold resource
    print("\n5. Testing with sold resource")
    print("-" * 70)

    for name, pol in [
        ("SellLow", policy_sl),
        ("HighLow", policy_hl),
    ]:
        decision = pol(None, state_sold, key)  # type: ignore[operator]
        print(f"  {name}: Decision = {int(decision[0])} (should be 0/Hold)")

    # Test baselines
    print("\n6. Baseline Policies")
    print("-" * 70)

    policy_hold = AlwaysHoldPolicy()
    policy_sell = AlwaysSellPolicy()

    decision_hold = policy_hold(None, state_mid, key)
    decision_sell = policy_sell(None, state_mid, key)

    print(f"  AlwaysHold: {int(decision_hold[0])} (Hold)")
    print(f"  AlwaysSell: {int(decision_sell[0])} (Sell)")

    print("\n" + "=" * 70)
    print("✓ All policy tests completed!")
    print("=" * 70)
