"""JAX-native policies for Medical Decision Diabetes.

This module implements various bandit algorithms for diabetes treatment selection:
- UCB (Upper Confidence Bound): Optimistic exploration
- Interval Estimation: Similar to UCB but based on posterior uncertainty
- Pure Exploitation: Greedy selection (best known drug)
- Pure Exploration: Random selection
- Thompson Sampling: Bayesian optimal exploration
"""

from typing import Optional
from functools import partial
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
import jax
import jax.numpy as jnp


# Type aliases
State = Float[Array, "5 3"]
Decision = Int[Array, ""]
Key = PRNGKeyArray


class UCBPolicy:
    """Upper Confidence Bound (UCB) policy.

    Selects drug with highest upper confidence bound:
    UCB_i = mu_i + theta * sqrt(log(t) / N_i)

    Example:
        >>> policy = UCBPolicy(theta=2.0)
        >>> # Higher theta = more exploration
    """

    def __init__(self, theta: float = 2.0) -> None:
        """Initialize policy.

        Args:
            theta: Exploration parameter (higher = more exploration).
        """
        self.theta = theta

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        time: int = 1,
    ) -> Decision:
        """Select drug using UCB.

        Args:
            params: Unused.
            state: Current state [5 × 3].
            key: Random key (unused).
            time: Current time step (for log(t) term).

        Returns:
            Drug index with highest UCB.
        """
        mu = state[:, 0]  # Posterior means
        n_trials = state[:, 2]  # Trial counts

        # UCB = mu + theta * sqrt(log(t) / (N + 1))
        # Add 1 to avoid division by zero
        exploration_bonus = self.theta * jnp.sqrt(
            jnp.log(time + 1) / (n_trials + 1)
        )

        ucb_values = mu + exploration_bonus

        # Return index of maximum UCB
        return jnp.argmax(ucb_values).astype(jnp.int32)


class IntervalEstimationPolicy:
    """Interval Estimation (IE) policy.

    Selects drug with highest upper confidence bound based on posterior:
    IE_i = mu_i + theta / sqrt(beta_i)
    where beta_i is posterior precision.

    Example:
        >>> policy = IntervalEstimationPolicy(theta=1.0)
    """

    def __init__(self, theta: float = 1.0) -> None:
        """Initialize policy.

        Args:
            theta: Exploration parameter.
        """
        self.theta = theta

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Select drug using Interval Estimation.

        Args:
            params: Unused.
            state: Current state [5 × 3].
            key: Random key (unused).

        Returns:
            Drug index with highest IE value.
        """
        mu = state[:, 0]  # Posterior means
        beta = state[:, 1]  # Posterior precisions

        # IE = mu + theta / sqrt(beta)
        ie_values = mu + self.theta / jnp.sqrt(beta)

        return jnp.argmax(ie_values).astype(jnp.int32)


class PureExploitationPolicy:
    """Pure Exploitation (Greedy) policy.

    Always selects drug with highest posterior mean (no exploration).

    Example:
        >>> policy = PureExploitationPolicy()
    """

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Select drug with highest posterior mean.

        Args:
            params: Unused.
            state: Current state [5 × 3].
            key: Random key (unused).

        Returns:
            Drug index with highest mean.
        """
        mu = state[:, 0]  # Posterior means

        return jnp.argmax(mu).astype(jnp.int32)


class PureExplorationPolicy:
    """Pure Exploration (Random) policy.

    Selects drug uniformly at random (ignores information).

    Example:
        >>> policy = PureExplorationPolicy()
    """

    def __init__(self, n_drugs: int = 5) -> None:
        """Initialize policy.

        Args:
            n_drugs: Number of drug options.
        """
        self.n_drugs = n_drugs

    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Select drug uniformly at random.

        Args:
            params: Unused.
            state: Current state (unused).
            key: Random key.

        Returns:
            Random drug index.
        """
        return jax.random.randint(key, (), 0, self.n_drugs).astype(jnp.int32)


class ThompsonSamplingPolicy:
    """Thompson Sampling policy.

    Samples from posterior distribution for each drug and selects maximum.
    Bayesian optimal exploration-exploitation tradeoff.

    Example:
        >>> policy = ThompsonSamplingPolicy()
    """

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Select drug via Thompson Sampling.

        Args:
            params: Unused.
            state: Current state [5 × 3].
            key: Random key for sampling.

        Returns:
            Drug index with highest sampled value.
        """
        mu = state[:, 0]  # Posterior means
        beta = state[:, 1]  # Posterior precisions

        # Sample from posterior: N(mu_i, 1/beta_i)
        sigma = 1.0 / jnp.sqrt(beta)

        # Split key for each drug
        keys = jax.random.split(key, 5)

        # Sample effectiveness for each drug
        sampled_values = mu + sigma * jax.random.normal(keys[0], (5,))

        return jnp.argmax(sampled_values).astype(jnp.int32)


class EpsilonGreedyPolicy:
    """Epsilon-Greedy policy.

    With probability epsilon, explores randomly.
    With probability 1-epsilon, exploits best known drug.

    Example:
        >>> policy = EpsilonGreedyPolicy(epsilon=0.1)
    """

    def __init__(self, epsilon: float = 0.1, n_drugs: int = 5) -> None:
        """Initialize policy.

        Args:
            epsilon: Exploration probability (0-1).
            n_drugs: Number of drug options.
        """
        self.epsilon = epsilon
        self.n_drugs = n_drugs

    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Select drug using epsilon-greedy.

        Args:
            params: Unused.
            state: Current state [5 × 3].
            key: Random key.

        Returns:
            Drug index (random with prob epsilon, greedy otherwise).
        """
        key1, key2 = jax.random.split(key)

        # With probability epsilon, explore
        explore = jax.random.uniform(key1) < self.epsilon

        # Exploration: random drug
        random_drug = jax.random.randint(key2, (), 0, self.n_drugs)

        # Exploitation: best drug
        mu = state[:, 0]
        best_drug = jnp.argmax(mu)

        # Return exploration or exploitation
        return jnp.where(explore, random_drug, best_drug).astype(jnp.int32)
