"""JAX-native policies for Stochastic Shortest Path (Static).

This module implements policies for graph navigation:
- Greedy: Select neighbor with lowest estimated cost-to-go
- Epsilon-Greedy: Mix greedy and random exploration
- Random: Uniform random neighbor selection
"""

from typing import Optional
from functools import partial
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
import jax
import jax.numpy as jnp


# Type aliases
State = Float[Array, "..."]
Decision = Int[Array, ""]
Key = PRNGKeyArray


class GreedyPolicy:
    """Greedy policy for SSP.

    Selects neighbor that minimizes: edge_cost + V(neighbor)
    This is the optimal policy given current value function estimate.

    Example:
        >>> policy = GreedyPolicy()
    """

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "SSPStaticModel",  # type: ignore[name-defined]
    ) -> Decision:
        """Select neighbor with minimum cost + value.

        Args:
            params: Unused.
            state: Current state [current_node, V_0, ..., V_n].
            key: Random key (unused for greedy).
            model: Model instance for graph access.

        Returns:
            Next node index (greedy choice).
        """
        current_node_idx = state[0].astype(jnp.int32)
        V = state[1:]

        # For each neighbor, compute: edge_cost_mean + V(neighbor)
        # Use mean edge cost: (lower + upper) / 2
        edge_costs_mean = (
            model.edge_lower[current_node_idx] +
            model.edge_upper[current_node_idx]
        ) / 2.0

        # Cost-to-go for each neighbor
        cost_to_go = edge_costs_mean + V

        # Mask out non-neighbors with large value
        valid_edges = model.adjacency[current_node_idx]
        cost_to_go = jnp.where(valid_edges, cost_to_go, jnp.inf)

        # Select minimum
        return jnp.argmin(cost_to_go).astype(jnp.int32)


class EpsilonGreedyPolicy:
    """Epsilon-Greedy policy for SSP.

    With probability epsilon, selects random neighbor.
    With probability 1-epsilon, selects greedy neighbor.

    Example:
        >>> policy = EpsilonGreedyPolicy(epsilon=0.1)
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        """Initialize policy.

        Args:
            epsilon: Exploration probability (0-1).
        """
        self.epsilon = epsilon
        self.greedy_policy = GreedyPolicy()

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "SSPStaticModel",  # type: ignore[name-defined]
    ) -> Decision:
        """Select neighbor using epsilon-greedy.

        Args:
            params: Unused.
            state: Current state.
            key: Random key.
            model: Model instance.

        Returns:
            Next node index.
        """
        key1, key2 = jax.random.split(key)

        # Decide: explore or exploit
        explore = jax.random.uniform(key1) < self.epsilon

        # Get greedy choice
        greedy_decision = self.greedy_policy(params, state, key2, model)

        # Get random choice from valid neighbors
        current_node_idx = state[0].astype(jnp.int32)
        valid_edges = model.adjacency[current_node_idx]

        # Use categorical sampling weighted by valid edges
        # This avoids boolean indexing which is non-concrete
        neighbor_probs = valid_edges / jnp.sum(valid_edges)
        random_decision = jax.random.choice(
            key2,
            jnp.arange(model.config.n_nodes),
            p=neighbor_probs
        )

        # Return exploration or exploitation
        result: jax.Array = jnp.where(explore, random_decision, greedy_decision).astype(jnp.int32)
        return result


class RandomPolicy:
    """Random policy for SSP.

    Selects uniformly at random from valid neighbors.
    Useful as baseline for comparison.

    Example:
        >>> policy = RandomPolicy()
    """

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "SSPStaticModel",  # type: ignore[name-defined]
    ) -> Decision:
        """Select random valid neighbor.

        Args:
            params: Unused.
            state: Current state.
            key: Random key.
            model: Model instance.

        Returns:
            Random next node index.
        """
        current_node_idx = state[0].astype(jnp.int32)

        # Get valid neighbors using categorical sampling
        # This avoids boolean indexing which is non-concrete
        valid_edges = model.adjacency[current_node_idx]
        neighbor_probs = valid_edges / jnp.sum(valid_edges)

        # Random choice weighted by valid edges
        decision = jax.random.choice(
            key,
            jnp.arange(model.config.n_nodes),
            p=neighbor_probs
        )
        return decision.astype(jnp.int32)


class BellmanGreedyPolicy:
    """Bellman-Greedy policy with sampled edge costs.

    Like greedy but uses sampled edge costs instead of mean.
    Useful when edge cost samples are available.

    Example:
        >>> policy = BellmanGreedyPolicy()
    """

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "SSPStaticModel",  # type: ignore[name-defined]
        edge_costs: Optional[Float[Array, "..."]] = None,
    ) -> Decision:
        """Select neighbor using sampled edge costs + V.

        Args:
            params: Unused.
            state: Current state.
            key: Random key.
            model: Model instance.
            edge_costs: Sampled edge costs (if None, use mean).

        Returns:
            Next node index.
        """
        current_node_idx = state[0].astype(jnp.int32)
        V = state[1:]

        if edge_costs is None:
            # Use mean costs
            edge_costs_use = (
                model.edge_lower[current_node_idx] +
                model.edge_upper[current_node_idx]
            ) / 2.0
        else:
            edge_costs_use = edge_costs

        # Cost-to-go
        cost_to_go = edge_costs_use + V

        # Mask non-neighbors
        valid_edges = model.adjacency[current_node_idx]
        cost_to_go = jnp.where(valid_edges, cost_to_go, jnp.inf)

        return jnp.argmin(cost_to_go).astype(jnp.int32)
