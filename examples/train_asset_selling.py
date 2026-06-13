"""Training script for Asset Selling with neural network policy.

This script demonstrates how to train a neural network policy for the
asset selling problem using policy gradient methods with JAX, Flax NNX, and Optax.

The training loop:
1. Samples batch of trajectories under current policy
2. Computes returns (discounted cumulative rewards)
3. Computes policy gradient using REINFORCE
4. Updates policy parameters with Adam optimizer
"""

from typing import Dict

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from problems.asset_selling import (
    AssetSellingConfig,
    AssetSellingModel,
    HighLowPolicy,
    NeuralPolicy,
)


def as_state_key_policy(policy):
    """Adapt a baseline ``(params, state, key)`` policy to the ``(state, key)``
    calling convention used by the learnable nnx policies (and by
    :func:`simulate_episode`)."""
    return lambda state, key: policy(None, state, key)


def simulate_episode(
    model: AssetSellingModel,
    policy: nnx.Module,
    key: jax.random.PRNGKey,
    horizon: int = 10,
    discount: float = 0.99,
) -> tuple[float, int]:
    """Simulate a single episode.

    Args:
        model: Asset selling model.
        policy: Policy to follow.
        key: Random key.
        horizon: Maximum number of steps.
        discount: Discount factor.

    Returns:
        Tuple of (total_discounted_reward, num_steps).
    """
    init_state = model.init_state(key)

    def step(carry, t):
        state, k, discount_factor, done = carry
        k, key_policy, key_exog = jax.random.split(k, 3)

        # Get decision from policy
        decision = policy(state, key_policy)

        # Sample exogenous information
        exog = model.sample_exogenous(key_exog, state, t)

        # Discounted reward (reward is 0 after the asset has been sold)
        reward = model.reward(state, decision, exog) * discount_factor

        # Transition
        next_state = model.transition(state, decision, exog)

        # Detect the selling step (resource goes from held -> sold) and count
        # steps until (and including) the sale.
        sold_now = (state[1] > 0.5) & (next_state[1] < 0.5)
        counted = jnp.where(done, 0, 1)

        return (
            (next_state, k, discount_factor * discount, done | sold_now),
            (reward, counted),
        )

    (final_state, _, final_discount, done), (rewards, counts) = jax.lax.scan(
        step, (init_state, key, 1.0, jnp.array(False)), jnp.arange(horizon)
    )

    # Didn't sell within horizon - force sell at final price.
    final_reward = jnp.where(done, 0.0, final_state[0] * final_discount)
    total_reward = jnp.sum(rewards) + final_reward
    num_steps = jnp.sum(counts)  # steps until the sale, or `horizon` if never sold

    return total_reward, num_steps


def batch_simulate_episodes(
    model: AssetSellingModel,
    policy: nnx.Module,
    keys: jax.random.PRNGKey,
    horizon: int = 10,
    discount: float = 0.99,
) -> Dict[str, float]:
    """Simulate batch of episodes using vmap.

    Args:
        model: Asset selling model.
        policy: Policy to follow.
        keys: Batch of random keys.
        horizon: Maximum number of steps.
        discount: Discount factor.

    Returns:
        Dictionary with statistics.
    """
    # Vectorize simulation
    batch_simulate = jax.vmap(
        lambda k: simulate_episode(model, policy, k, horizon, discount)
    )

    rewards, steps = batch_simulate(keys)

    return {
        "mean_reward": float(jnp.mean(rewards)),
        "std_reward": float(jnp.std(rewards)),
        "min_reward": float(jnp.min(rewards)),
        "max_reward": float(jnp.max(rewards)),
        "mean_steps": float(jnp.mean(steps)),
    }


def evaluate_policy(
    model: AssetSellingModel,
    policy: nnx.Module,
    key: jax.random.PRNGKey,
    n_episodes: int = 100,
    horizon: int = 10,
) -> Dict[str, float]:
    """Evaluate policy over multiple episodes.

    Args:
        model: Asset selling model.
        policy: Policy to evaluate.
        key: Random key.
        n_episodes: Number of episodes to run.
        horizon: Episode horizon.

    Returns:
        Dictionary of statistics.
    """
    keys = jax.random.split(key, n_episodes)
    return batch_simulate_episodes(model, policy, keys, horizon)


def compute_reinforce_loss(
    policy: nnx.Module,
    model: AssetSellingModel,
    keys: jax.random.PRNGKey,
    horizon: int,
    discount: float,
) -> float:
    """Compute REINFORCE policy gradient loss.

    Loss is negative expected return (to minimize).

    Args:
        policy: Policy being optimized.
        model: Environment model.
        keys: Batch of random keys.
        horizon: Episode horizon.
        discount: Discount factor.

    Returns:
        Negative mean return (loss to minimize).
    """
    # Vectorise over the batch of episode keys, keeping everything as JAX
    # arrays (batch_simulate_episodes returns Python floats and so cannot be
    # used inside a traced/jitted loss).
    rewards, _ = jax.vmap(
        lambda k: simulate_episode(model, policy, k, horizon, discount)
    )(keys)
    return -jnp.mean(rewards)  # Negative for minimization


def train_neural_policy(
    model: AssetSellingModel,
    policy: nnx.Module,
    optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,
    n_iterations: int = 1000,
    batch_size: int = 32,
    horizon: int = 10,
    discount: float = 0.99,
    eval_every: int = 50,
) -> tuple[nnx.Module, list[float]]:
    """Train neural network policy using REINFORCE.

    Args:
        model: Asset selling model.
        policy: Neural network policy (will be modified in-place).
        optimizer: Flax NNX optimizer.
        key: Random key.
        n_iterations: Number of training iterations.
        batch_size: Number of episodes per gradient update.
        horizon: Episode horizon.
        discount: Discount factor.
        eval_every: Evaluate every N iterations.

    Returns:
        Tuple of (trained_policy, loss_history).
    """
    loss_history = []
    eval_rewards = []

    # JIT-compile loss and gradient computation
    @nnx.jit
    def train_step(policy, optimizer, keys):
        """Single training step."""
        loss, grads = nnx.value_and_grad(compute_reinforce_loss)(
            policy, model, keys, horizon, discount
        )
        optimizer.update(policy, grads)
        return loss

    print("Training Neural Network Policy")
    print("=" * 70)
    print(f"Iterations: {n_iterations}")
    print(f"Batch size: {batch_size}")
    print(f"Horizon: {horizon}")
    print(f"Discount: {discount}")
    print()

    for iteration in range(n_iterations):
        # Sample batch of episodes
        key, subkey = jax.random.split(key)
        episode_keys = jax.random.split(subkey, batch_size)

        # Training step
        loss = train_step(policy, optimizer, episode_keys)
        loss_history.append(float(loss))

        # Evaluation
        if iteration % eval_every == 0:
            key, eval_key = jax.random.split(key)
            eval_stats = evaluate_policy(model, policy, eval_key, n_episodes=100)

            print(f"Iter {iteration:4d}: "
                  f"Loss = {loss:7.2f}, "
                  f"Eval Reward = {eval_stats['mean_reward']:7.2f} ± "
                  f"{eval_stats['std_reward']:5.2f}, "
                  f"Steps = {eval_stats['mean_steps']:.1f}")

            eval_rewards.append(eval_stats['mean_reward'])

    print()
    print("Training completed!")
    print("=" * 70)

    return policy, loss_history


def compare_policies(model: AssetSellingModel, key: jax.random.PRNGKey):
    """Compare different policies on the asset selling problem.

    Args:
        model: Asset selling model.
        key: Random key.
    """
    print("\nComparing Policies")
    print("=" * 70)

    policies = {
        "High-Low (90, 110)": HighLowPolicy(90.0, 110.0),
        "High-Low (95, 105)": HighLowPolicy(95.0, 105.0),
        "High-Low (85, 115)": HighLowPolicy(85.0, 115.0),
    }

    results = {}
    for name, policy in policies.items():
        key, subkey = jax.random.split(key)
        stats = evaluate_policy(model, as_state_key_policy(policy), subkey, n_episodes=1000)
        results[name] = stats

        print(f"\n{name}:")
        print(f"  Mean reward: ${stats['mean_reward']:.2f} ± ${stats['std_reward']:.2f}")
        print(f"  Range: [${stats['min_reward']:.2f}, ${stats['max_reward']:.2f}]")
        print(f"  Mean steps: {stats['mean_steps']:.1f}")

    return results


def plot_training_results(
    loss_history: list[float],
    eval_every: int,
    n_iterations: int,
):
    """Plot training curves.

    Args:
        loss_history: List of losses per iteration.
        eval_every: Evaluation frequency.
        n_iterations: Total iterations.
    """
    try:
        import matplotlib.pyplot as plt  # optional dependency (install with `.[viz]`)
    except ImportError:
        print("\n[skip] matplotlib not installed; skipping training plot. "
              "Install with `pip install '.[viz]'` to enable.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(loss_history, alpha=0.3, color='blue', label='Raw')

    # Moving average
    window = 50
    if len(loss_history) > window:
        moving_avg = jnp.convolve(
            jnp.array(loss_history),
            jnp.ones(window) / window,
            mode='valid'
        )
        ax1.plot(range(window-1, len(loss_history)), moving_avg,
                color='red', linewidth=2, label=f'MA({window})')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (Negative Reward)')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot reward (negative loss)
    rewards = [-loss for loss in loss_history]
    ax2.plot(rewards, alpha=0.3, color='green', label='Raw')

    if len(rewards) > window:
        moving_avg = jnp.convolve(
            jnp.array(rewards),
            jnp.ones(window) / window,
            mode='valid'
        )
        ax2.plot(range(window-1, len(rewards)), moving_avg,
                color='darkgreen', linewidth=2, label=f'MA({window})')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Average Return')
    ax2.set_title('Training Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('asset_selling_training.png', dpi=150, bbox_inches='tight')
    print("\nSaved training plot to: asset_selling_training.png")
    plt.show()


def main():
    """Main training script."""
    # Configuration
    config = AssetSellingConfig(
        up_step=2.0,
        down_step=-2.0,
        variance=2.0,
        initial_price=100.0,
    )
    model = AssetSellingModel(config)

    print("Asset Selling - Policy Learning")
    print("=" * 70)
    print("\nModel Configuration:")
    print(f"  Initial price: ${config.initial_price:.2f}")
    print(f"  Up step: {config.up_step}")
    print(f"  Down step: {config.down_step}")
    print(f"  Variance: {config.variance}")

    # Random key
    key = jax.random.PRNGKey(42)

    # Compare baseline policies
    key, subkey = jax.random.split(key)
    baseline_results = compare_policies(model, subkey)

    # Find best baseline
    best_baseline = max(baseline_results.items(), key=lambda x: x[1]['mean_reward'])
    print(f"\n{'=' * 70}")
    print(f"Best baseline: {best_baseline[0]} with ${best_baseline[1]['mean_reward']:.2f}")
    print(f"{'=' * 70}")

    # Train neural network policy
    print("\n\nTraining Neural Network Policy")
    print("=" * 70)

    key, subkey = jax.random.split(key)
    neural_policy = NeuralPolicy(hidden_dims=[32, 16], rngs=nnx.Rngs(int(subkey[0])))

    # Create optimizer
    learning_rate = 1e-3
    optimizer = nnx.Optimizer(neural_policy, optax.adam(learning_rate), wrt=nnx.Param)

    # Train
    key, subkey = jax.random.split(key)
    trained_policy, loss_history = train_neural_policy(
        model=model,
        policy=neural_policy,
        optimizer=optimizer,
        key=subkey,
        n_iterations=500,
        batch_size=64,
        horizon=10,
        discount=0.99,
        eval_every=25,
    )

    # Final evaluation
    print("\nFinal Evaluation")
    print("=" * 70)

    key, subkey = jax.random.split(key)
    final_stats = evaluate_policy(model, trained_policy, subkey, n_episodes=1000)

    print("\nNeural Policy:")
    print(f"  Mean reward: ${final_stats['mean_reward']:.2f} ± ${final_stats['std_reward']:.2f}")
    print(f"  Range: [${final_stats['min_reward']:.2f}, ${final_stats['max_reward']:.2f}]")
    print(f"  Mean steps: {final_stats['mean_steps']:.1f}")

    print(f"\nComparison to best baseline ({best_baseline[0]}):")
    improvement = final_stats['mean_reward'] - best_baseline[1]['mean_reward']
    improvement_pct = (improvement / best_baseline[1]['mean_reward']) * 100
    print(f"  Improvement: ${improvement:.2f} ({improvement_pct:+.1f}%)")

    # Plot results
    plot_training_results(loss_history, eval_every=25, n_iterations=500)

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
