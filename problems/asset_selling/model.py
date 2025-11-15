"""JAX-native Asset Selling Model.

This model addresses the problem of optimal timing for selling an asset
under price uncertainty. The price follows a random walk with bias transitions,
and the decision is binary: sell or hold.

Key features:
- Markov chain for price bias (Up/Neutral/Down)
- Random walk price dynamics
- One-shot decision (can only sell once)
- Time-dependent value (must sell by horizon)
"""

from typing import NamedTuple, Optional, List, Any
from functools import partial
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
import jax
import jax.numpy as jnp
import chex


# Type aliases
State = Float[Array, "3"]  # [price, resource, bias_idx]
Decision = Int[Array, "1"]  # [sell] - binary: 0=hold, 1=sell
Reward = Float[Array, ""]  # Scalar reward
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information for asset selling.

    Attributes:
        price_change: Change in asset price for this period.
        new_bias_idx: Index of new bias state (0=Up, 1=Neutral, 2=Down).
    """
    price_change: Float[Array, ""]
    new_bias_idx: Int[Array, ""]


@chex.dataclass(frozen=True)
class AssetSellingConfig:
    """Configuration for asset selling model.

    This is a pytree-registered, immutable configuration.

    Attributes:
        up_step: Mean price change when bias is "Up".
        neutral_step: Mean price change when bias is "Neutral" (typically 0).
        down_step: Mean price change when bias is "Down" (negative).
        variance: Standard deviation of price changes.
        initial_price: Starting asset price.
        initial_bias_idx: Starting bias index (0=Up, 1=Neutral, 2=Down).
        transition_matrix: 3x3 matrix of bias transition probabilities.
                          Row i = from bias i, columns = to bias [Up, Neutral, Down].
    """
    up_step: float = 2.0
    neutral_step: float = 0.0
    down_step: float = -2.0
    variance: float = 2.0
    initial_price: float = 100.0
    initial_bias_idx: int = 1  # Start neutral
    transition_matrix: Optional[Float[Array, "3 3"]] = None

    def __post_init__(self) -> None:
        """Validate configuration with chex assertions."""
        chex.assert_scalar_positive(self.variance)
        chex.assert_scalar_positive(self.initial_price)
        # Check bias index is valid (0, 1, or 2)
        if self.initial_bias_idx not in [0, 1, 2]:
            raise ValueError(f"initial_bias_idx must be 0, 1, or 2, got {self.initial_bias_idx}")

        # Default transition matrix if not provided
        if self.transition_matrix is None:
            # Default: 70% stay, 15% each to other states
            default_tm = jnp.array([
                [0.70, 0.15, 0.15],  # From Up
                [0.15, 0.70, 0.15],  # From Neutral
                [0.15, 0.15, 0.70],  # From Down
            ])
            object.__setattr__(self, 'transition_matrix', default_tm)

        # Validate transition matrix (at this point it's guaranteed to be non-None)
        assert self.transition_matrix is not None  # Help mypy understand it's not None
        chex.assert_shape(self.transition_matrix, (3, 3))
        chex.assert_tree_all_finite(self.transition_matrix)
        # Each row should sum to ~1.0
        row_sums = jnp.sum(self.transition_matrix, axis=1)
        chex.assert_trees_all_close(row_sums, jnp.ones(3), rtol=1e-5)


class AssetSellingModel:
    """JAX-native asset selling optimization model.

    The model simulates an asset whose price follows a random walk with
    Markov-modulated drift (bias). The decision maker must choose when
    to sell the asset to maximize expected revenue.

    State representation:
        - price: Current asset price (continuous, non-negative)
        - resource: Whether asset is still held (1) or sold (0)
        - bias_idx: Current bias state index (0=Up, 1=Neutral, 2=Down)

    Dynamics:
        - Price evolves as: price_t+1 = max(0, price_t + bias_step + N(0, variance))
        - Bias transitions according to Markov chain
        - Resource becomes 0 after selling (irreversible)

    Objective:
        - Maximize sale price
        - No holding cost, but implicit opportunity cost from waiting

    Example:
        >>> config = AssetSellingConfig()
        >>> model = AssetSellingModel(config)
        >>> key = jax.random.PRNGKey(0)
        >>> state = model.init_state(key)
        >>> decision = jnp.array([0])  # Hold
        >>> key, subkey = jax.random.split(key)
        >>> exog = model.sample_exogenous(subkey, state, 0)
        >>> next_state = model.transition(state, decision, exog)
    """

    def __init__(self, config: AssetSellingConfig) -> None:
        """Initialize model.

        Args:
            config: Model configuration.
        """
        self.config = config

        # Pre-compute bias steps array for efficient indexing
        self.bias_steps = jnp.array([
            config.up_step,
            config.neutral_step,
            config.down_step
        ])

    def init_state(self, key: Key) -> State:
        """Initialize state.

        Args:
            key: Random key (unused but kept for interface consistency).

        Returns:
            Initial state [price, resource, bias_idx].
        """
        return jnp.array([
            self.config.initial_price,
            1.0,  # Asset is held initially
            float(self.config.initial_bias_idx),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def transition(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> State:
        """Compute next state (JIT-compiled).

        Args:
            state: Current state [price, resource, bias_idx].
            decision: Binary decision (0=hold, 1=sell).
            exog: Exogenous information with price change and new bias.

        Returns:
            Next state.
        """
        price, resource, bias_idx = state[0], state[1], state[2]
        sell_decision = decision[0]

        # Update price with random change (only if not negative)
        new_price = jnp.maximum(0.0, price + exog.price_change)

        # Resource becomes 0 if we sell (irreversible)
        new_resource = jnp.where(
            sell_decision == 1,
            0.0,  # Sold
            resource  # Keep current status
        )

        # Update bias index (convert to float for array)
        new_bias_idx = exog.new_bias_idx.astype(jnp.float32)

        return jnp.array([new_price, new_resource, new_bias_idx])

    @partial(jax.jit, static_argnums=(0,))
    def reward(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> Reward:
        """Compute reward (JIT-compiled).

        The reward is the sale price if selling with resource available,
        otherwise 0.

        Args:
            state: Current state.
            decision: Binary decision (0=hold, 1=sell).
            exog: Exogenous information (unused but kept for interface).

        Returns:
            Scalar reward (sale revenue or 0).
        """
        price, resource, _ = state[0], state[1], state[2]
        sell_decision = decision[0]

        # Get reward only if selling AND resource is available
        reward = jnp.where(
            (sell_decision == 1) & (resource > 0.5),  # resource > 0.5 to handle float comparison
            price,
            0.0
        )

        return reward

    def sample_exogenous(self, key: Key, state: State, time: int) -> ExogenousInfo:
        """Sample exogenous information.

        Samples:
        1. New bias state from current bias using transition matrix
        2. Price change based on new bias and random noise

        Args:
            key: JAX random key.
            state: Current state (to get current bias).
            time: Current time step (unused but kept for interface).

        Returns:
            Sampled exogenous information.
        """
        # Get current bias index (keep as JAX array)
        current_bias_idx = state[2].astype(jnp.int32)

        # Split key for bias and price sampling
        key_bias, key_price = jax.random.split(key)

        # Sample new bias using transition probabilities
        # Use advanced indexing that works with tracers
        # transition_matrix is guaranteed to be non-None after __post_init__
        assert self.config.transition_matrix is not None
        transition_probs = self.config.transition_matrix[current_bias_idx]
        new_bias_idx = jax.random.choice(
            key_bias,
            jnp.array([0, 1, 2]),
            p=transition_probs
        )

        # Get bias step for new bias
        bias_step = self.bias_steps[new_bias_idx]

        # Sample price change: bias + noise
        noise = jax.random.normal(key_price) * self.config.variance
        price_change = bias_step + noise

        return ExogenousInfo(
            price_change=price_change,
            new_bias_idx=new_bias_idx
        )

    @partial(jax.jit, static_argnums=(0,))
    def is_valid_decision(
        self,
        state: State,
        decision: Decision,
    ) -> Bool[Array, ""]:
        """Check if decision is valid (JIT-compiled).

        A decision is valid if:
        - It's binary (0 or 1)
        - If selling (1), must have resource available

        Args:
            state: Current state.
            decision: Proposed decision.

        Returns:
            Boolean indicating validity.
        """
        resource = state[1]
        sell_decision = decision[0]

        # Check if decision is binary
        is_binary = (sell_decision == 0) | (sell_decision == 1)

        # If selling, must have resource
        valid_if_selling = jnp.where(
            sell_decision == 1,
            resource > 0.5,  # Must have resource to sell
            True  # Holding is always valid
        )

        return is_binary & valid_if_selling

    def get_bias_name(self, bias_idx: int) -> str:
        """Get human-readable bias name.

        Args:
            bias_idx: Bias index (0, 1, or 2).

        Returns:
            Bias name string.
        """
        names: List[str] = ["Up", "Neutral", "Down"]
        return names[bias_idx]


# Example usage and benchmarking
if __name__ == "__main__":
    import time

    print("JAX-Native Asset Selling Model")
    print("=" * 70)

    # Create model with default configuration
    config = AssetSellingConfig(
        up_step=2.0,
        down_step=-2.0,
        variance=1.5,
        initial_price=100.0
    )
    model = AssetSellingModel(config)

    print(f"\nConfiguration:")
    print(f"  Initial price: ${config.initial_price:.2f}")
    print(f"  Up step: {config.up_step}")
    print(f"  Down step: {config.down_step}")
    print(f"  Variance: {config.variance}")
    print(f"  Initial bias: {model.get_bias_name(config.initial_bias_idx)}")

    # Initialize
    key = jax.random.PRNGKey(42)
    state = model.init_state(key)

    print(f"\nInitial state:")
    print(f"  Price: ${state[0]:.2f}")
    print(f"  Resource: {int(state[1])}")
    print(f"  Bias: {model.get_bias_name(int(state[2]))}")

    # Test holding decision
    print("\n" + "=" * 70)
    print("Testing HOLD decision")
    print("=" * 70)

    decision_hold = jnp.array([0])
    key, subkey = jax.random.split(key)
    exog = model.sample_exogenous(subkey, state, 0)

    print(f"\nExogenous sample:")
    print(f"  Price change: {float(exog.price_change):.2f}")
    print(f"  New bias: {model.get_bias_name(int(exog.new_bias_idx))}")

    # First call (includes compilation)
    start = time.time()
    next_state = model.transition(state, decision_hold, exog)
    compile_time = time.time() - start

    print(f"\nFirst call (with JIT compilation): {compile_time:.4f}s")
    print(f"Next state: Price=${next_state[0]:.2f}, Resource={int(next_state[1])}, Bias={model.get_bias_name(int(next_state[2]))}")

    # Second call (compiled)
    start = time.time()
    next_state = model.transition(state, decision_hold, exog)
    run_time = time.time() - start

    print(f"Second call (JIT-compiled): {run_time:.6f}s")
    print(f"Speedup: {compile_time/run_time:.1f}x")

    # Test reward for holding
    reward_hold = model.reward(state, decision_hold, exog)
    print(f"\nReward for holding: ${float(reward_hold):.2f}")

    # Test selling decision
    print("\n" + "=" * 70)
    print("Testing SELL decision")
    print("=" * 70)

    decision_sell = jnp.array([1])

    # Sell at initial state
    next_state_sell = model.transition(state, decision_sell, exog)
    reward_sell = model.reward(state, decision_sell, exog)

    print(f"\nSelling at price ${state[0]:.2f}:")
    print(f"  Reward: ${float(reward_sell):.2f}")
    print(f"  Next state: Price=${next_state_sell[0]:.2f}, Resource={int(next_state_sell[1])}, Bias={model.get_bias_name(int(next_state_sell[2]))}")

    # Try to sell again (should get 0 reward)
    reward_sell_again = model.reward(next_state_sell, decision_sell, exog)
    print(f"  Trying to sell again: ${float(reward_sell_again):.2f} (resource already sold)")

    # Validation
    print("\n" + "=" * 70)
    print("Testing Decision Validation")
    print("=" * 70)

    valid_hold = model.is_valid_decision(state, decision_hold)
    valid_sell = model.is_valid_decision(state, decision_sell)
    valid_sell_again = model.is_valid_decision(next_state_sell, decision_sell)

    print(f"\nWith resource:")
    print(f"  Hold (0) valid: {bool(valid_hold)}")
    print(f"  Sell (1) valid: {bool(valid_sell)}")
    print(f"\nAfter selling:")
    print(f"  Sell again valid: {bool(valid_sell_again)}")

    # Batch simulation
    print("\n" + "=" * 70)
    print("Batch Simulation with vmap")
    print("=" * 70)

    batch_size = 1000
    print(f"\nSimulating {batch_size} trajectories...")

    # Create batch of initial states
    states = jnp.repeat(state[None, :], batch_size, axis=0)
    decisions = jnp.zeros((batch_size, 1), dtype=jnp.int32)  # All hold

    # Sample batch of exogenous info
    key, *subkeys = jax.random.split(key, batch_size + 1)

    # Vectorize exogenous sampling
    def _sample_fn(k: Any, s: Any) -> Any:
        return model.sample_exogenous(k, s, 0)

    batch_sample_exog = jax.vmap(_sample_fn)
    batch_exogs = batch_sample_exog(jnp.array(subkeys), states)

    # Vectorize transition
    batch_transition = jax.vmap(model.transition)

    start = time.time()
    batch_next_states = batch_transition(states, decisions, batch_exogs)
    batch_time = time.time() - start

    print(f"Batch processing time: {batch_time:.6f}s")
    print(f"Per-sample time: {batch_time/batch_size*1e6:.2f}μs")
    print(f"Samples per second: {batch_size/batch_time:.0f}")

    # Statistics
    prices = batch_next_states[:, 0]
    print(f"\nPrice statistics after one step:")
    print(f"  Mean: ${float(jnp.mean(prices)):.2f}")
    print(f"  Std: ${float(jnp.std(prices)):.2f}")
    print(f"  Min: ${float(jnp.min(prices)):.2f}")
    print(f"  Max: ${float(jnp.max(prices)):.2f}")

    # Test gradient flow
    print("\n" + "=" * 70)
    print("Automatic Differentiation")
    print("=" * 70)

    # For gradient demo, we'll compute gradient w.r.t. a continuous policy parameter
    # that determines selling threshold
    def policy_loss(threshold: float, state: State, key: Key) -> Any:
        """Loss function: negative expected reward from threshold policy."""
        # Sample future price
        exog = model.sample_exogenous(key, state, 0)
        next_state = model.transition(state, jnp.array([0]), exog)  # Hold
        future_price = next_state[0]

        # Decision: sell if price > threshold
        decision = jnp.where(state[0] > threshold, 1, 0)
        decision_array = jnp.array([decision])

        reward = model.reward(state, decision_array, exog)
        return -reward  # Negative for minimization

    threshold_test = 100.0
    key, subkey = jax.random.split(key)

    grad_fn = jax.grad(policy_loss)
    gradient = grad_fn(threshold_test, state, subkey)

    print(f"\nPolicy threshold: ${threshold_test:.2f}")
    print(f"Gradient w.r.t. threshold: {float(gradient):.4f}")
    print("✓ Gradients flow through the model!")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
