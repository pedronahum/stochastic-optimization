"""Training example for Energy Storage policies using REINFORCE.

This script demonstrates how to train parametric policies for the battery
energy storage problem using policy gradient methods.

Example:
    $ python -m stochopt.examples.train_energy_storage
"""

from typing import Any, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from problems.energy_storage import (
    EnergyStorageModel,
    EnergyStorageConfig,
    LinearPolicy,
    NeuralPolicy,
    ThresholdPolicy,
    ThresholdPolicyConfig,
    TimeOfDayPolicy,
    AlwaysHoldPolicy,
)


def simulate_episode(
    model: EnergyStorageModel,
    policy: Any,
    key: jax.Array,
    max_steps: int = 24,
) -> Tuple[float, int]:
    """Simulate one episode and return total reward.

    Args:
        model: Energy storage model.
        policy: Policy to evaluate (must have __call__ method).
        key: Random key.
        max_steps: Maximum number of steps.

    Returns:
        Tuple of (total_reward, num_steps).
    """
    state = model.init_state(key)
    total_reward = 0.0
    num_steps = 0

    for t in range(max_steps):
        # Get decision from policy
        key, subkey = jax.random.split(key)

        # Handle different policy interfaces
        if isinstance(policy, (LinearPolicy, NeuralPolicy)):
            # Learnable policies need to sample price first
            key_exog = jax.random.PRNGKey(t)
            exog_temp = model.sample_exogenous(key_exog, state, t)
            decision = policy(state, exog_temp.price, subkey)
        else:
            # Other policies take (params, state, key)
            decision = policy(None, state, subkey)

        # Sample exogenous info
        key, subkey = jax.random.split(key)
        exog = model.sample_exogenous(subkey, state, t)

        # Get reward
        reward = model.reward(state, decision, exog)
        total_reward += float(reward)

        # Transition
        state = model.transition(state, decision, exog)
        num_steps += 1

    return total_reward, num_steps


def evaluate_policy(
    model: EnergyStorageModel,
    policy: Any,
    num_episodes: int = 100,
    key: jax.Array = jax.random.PRNGKey(0),
) -> Tuple[float, float]:
    """Evaluate a policy over multiple episodes.

    Args:
        model: Energy storage model.
        policy: Policy to evaluate.
        num_episodes: Number of episodes to run.
        key: Random key.

    Returns:
        Tuple of (mean_reward, std_reward).
    """
    rewards = []

    for i in range(num_episodes):
        key, subkey = jax.random.split(key)
        reward, _ = simulate_episode(model, policy, subkey)
        rewards.append(reward)

    return float(jnp.mean(jnp.array(rewards))), float(jnp.std(jnp.array(rewards)))


def reinforce_update(
    model: EnergyStorageModel,
    policy: Any,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: jax.Array,
    num_episodes: int = 10,
) -> Tuple[Any, optax.OptState, float]:
    """Perform one REINFORCE update.

    Args:
        model: Energy storage model.
        policy: Learnable policy (LinearPolicy or NeuralPolicy).
        optimizer: Optax optimizer.
        opt_state: Optimizer state.
        key: Random key.
        num_episodes: Number of episodes per update.

    Returns:
        Tuple of (updated_policy, updated_opt_state, mean_reward).
    """

    def loss_fn(policy_model: Any) -> jax.Array:
        """Compute negative expected reward (we minimize loss)."""

        def episode_reward(ep_key: jax.Array) -> jax.Array:
            state = model.init_state(ep_key)
            total_reward = 0.0

            for t in range(24):  # Fixed 24-hour horizon
                ep_key, subkey = jax.random.split(ep_key)

                # Sample exogenous info
                ep_key, subkey_exog = jax.random.split(ep_key)
                exog = model.sample_exogenous(subkey_exog, state, t)

                # Get decision
                decision = policy_model(state, exog.price, subkey)

                # Get reward
                reward = model.reward(state, decision, exog)
                total_reward = total_reward + reward

                # Transition
                state = model.transition(state, decision, exog)

            return total_reward

        # Generate episode keys
        episode_keys = jax.random.split(key, num_episodes)

        # Compute rewards for batch of episodes
        rewards = jax.vmap(episode_reward)(episode_keys)

        # Return negative mean reward (minimize)
        return -jnp.mean(rewards)

    # Compute loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(policy)

    # Apply updates
    updates = optimizer.update(grads, opt_state)
    nnx.update(policy, updates)

    return policy, opt_state, -float(loss)  # Return positive reward


def train_linear_policy() -> None:
    """Train a linear policy using REINFORCE."""
    print("\n" + "=" * 70)
    print("Training Linear Policy with REINFORCE")
    print("=" * 70)

    # Create model
    config = EnergyStorageConfig(
        capacity=1000.0,
        max_charge_rate=100.0,
        efficiency=0.95,
    )
    model = EnergyStorageModel(config)

    # Create policy
    policy = LinearPolicy(rngs=nnx.Rngs(42))

    # Create optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(nnx.state(policy))

    # Training loop
    num_iterations = 50
    eval_interval = 10

    print(f"\nInitial weights: {policy.weights.value}")

    key = jax.random.PRNGKey(42)

    for iteration in range(num_iterations):
        key, subkey = jax.random.split(key)

        # Update policy
        policy, opt_state, mean_reward = reinforce_update(
            model, policy, optimizer, opt_state, subkey, num_episodes=10
        )

        # Periodic evaluation
        if iteration % eval_interval == 0:
            key, eval_key = jax.random.split(key)
            eval_mean, eval_std = evaluate_policy(model, policy, num_episodes=50, key=eval_key)
            print(f"Iter {iteration:3d}: Train Reward = {mean_reward:8.2f} | "
                  f"Eval Reward = {eval_mean:8.2f} ± {eval_std:6.2f}")

    print(f"\nFinal weights: {policy.weights.value}")
    print("=" * 70)


def train_neural_policy() -> None:
    """Train a neural network policy using REINFORCE."""
    print("\n" + "=" * 70)
    print("Training Neural Policy with REINFORCE")
    print("=" * 70)

    # Create model
    config = EnergyStorageConfig(
        capacity=1000.0,
        max_charge_rate=100.0,
        efficiency=0.95,
    )
    model = EnergyStorageModel(config)

    # Create policy
    policy = NeuralPolicy(hidden_dims=[32, 16], rngs=nnx.Rngs(42))

    # Create optimizer
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(nnx.state(policy))

    # Training loop
    num_iterations = 100
    eval_interval = 20

    print(f"\nNetwork architecture: {[32, 16]} -> 1")
    print(f"Number of parameters: {sum(p.size for p in jax.tree.leaves(nnx.state(policy)))}")

    key = jax.random.PRNGKey(42)

    for iteration in range(num_iterations):
        key, subkey = jax.random.split(key)

        # Update policy
        policy, opt_state, mean_reward = reinforce_update(
            model, policy, optimizer, opt_state, subkey, num_episodes=10
        )

        # Periodic evaluation
        if iteration % eval_interval == 0:
            key, eval_key = jax.random.split(key)
            eval_mean, eval_std = evaluate_policy(model, policy, num_episodes=50, key=eval_key)
            print(f"Iter {iteration:3d}: Train Reward = {mean_reward:8.2f} | "
                  f"Eval Reward = {eval_mean:8.2f} ± {eval_std:6.2f}")

    print("=" * 70)


def compare_policies() -> None:
    """Compare different policies."""
    print("\n" + "=" * 70)
    print("Policy Comparison")
    print("=" * 70)

    # Create model
    config = EnergyStorageConfig(
        capacity=1000.0,
        max_charge_rate=100.0,
        efficiency=0.95,
    )
    model = EnergyStorageModel(config)

    # Create policies
    policies = {
        "Threshold (40/60)": ThresholdPolicy(
            model,
            ThresholdPolicyConfig(
                buy_threshold=40.0,
                sell_threshold=60.0,
                charge_rate=0.8,
                discharge_rate=0.8,
            )
        ),
        "Threshold (45/55)": ThresholdPolicy(
            model,
            ThresholdPolicyConfig(
                buy_threshold=45.0,
                sell_threshold=55.0,
                charge_rate=0.8,
                discharge_rate=0.8,
            )
        ),
        "Time of Day": TimeOfDayPolicy(
            model,
            peak_start=9,
            peak_end=20,
            charge_rate=0.8,
            discharge_rate=0.8,
        ),
        "Always Hold": AlwaysHoldPolicy(),
        "Linear (Random)": LinearPolicy(rngs=nnx.Rngs(42)),
        "Neural (Random)": NeuralPolicy(hidden_dims=[16, 8], rngs=nnx.Rngs(42)),
    }

    # Evaluate each policy
    key = jax.random.PRNGKey(42)
    results = []

    for name, policy in policies.items():
        key, subkey = jax.random.split(key)
        mean_reward, std_reward = evaluate_policy(model, policy, num_episodes=100, key=subkey)
        results.append((name, mean_reward, std_reward))

    # Sort by mean reward
    results.sort(key=lambda x: x[1], reverse=True)

    # Print results
    print("\nPolicy Performance (100 episodes each):")
    print("-" * 70)
    print(f"{'Policy':<20} {'Mean Reward':>15} {'Std Reward':>15}")
    print("-" * 70)

    for name, mean_reward, std_reward in results:
        print(f"{name:<20} {mean_reward:>15.2f} {std_reward:>15.2f}")

    print("-" * 70)
    print(f"Best policy: {results[0][0]} with mean reward {results[0][1]:.2f}")
    print("=" * 70)


def main() -> None:
    """Run all training examples."""
    print("\n" + "=" * 70)
    print("Energy Storage Policy Training Examples")
    print("=" * 70)

    # 1. Compare baseline policies
    compare_policies()

    # 2. Train linear policy
    train_linear_policy()

    # 3. Train neural policy
    train_neural_policy()

    # 4. Final comparison with trained policies
    print("\n" + "=" * 70)
    print("Final Comparison (with trained policies)")
    print("=" * 70)

    config = EnergyStorageConfig()
    model = EnergyStorageModel(config)

    # Train linear policy briefly
    linear_policy = LinearPolicy(rngs=nnx.Rngs(42))
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(nnx.state(linear_policy))

    key = jax.random.PRNGKey(42)
    for _ in range(20):
        key, subkey = jax.random.split(key)
        linear_policy, opt_state, _ = reinforce_update(
            model, linear_policy, optimizer, opt_state, subkey, num_episodes=10
        )

    # Evaluate
    policies_final = {
        "Threshold (Best)": ThresholdPolicy(
            model,
            ThresholdPolicyConfig(buy_threshold=40.0, sell_threshold=60.0)
        ),
        "Time of Day": TimeOfDayPolicy(model),
        "Linear (Trained)": linear_policy,
        "Always Hold": AlwaysHoldPolicy(),
    }

    results_final = []
    for name, policy in policies_final.items():
        key, subkey = jax.random.split(key)
        mean_reward, std_reward = evaluate_policy(model, policy, num_episodes=100, key=subkey)
        results_final.append((name, mean_reward, std_reward))

    results_final.sort(key=lambda x: x[1], reverse=True)

    print("\nFinal Results:")
    print("-" * 70)
    print(f"{'Policy':<20} {'Mean Reward':>15} {'Std Reward':>15}")
    print("-" * 70)

    for name, mean_reward, std_reward in results_final:
        print(f"{name:<20} {mean_reward:>15.2f} {std_reward:>15.2f}")

    print("-" * 70)
    print("\n" + "=" * 70)
    print("✓ All training examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
