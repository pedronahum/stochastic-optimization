"""JAX-native Stochastic Shortest Path (Static) Model.

This module implements a shortest path problem on a directed graph with stochastic
edge costs. The agent navigates from an origin node to a target node, learning
optimal paths through value iteration.

Graph Representation:
- Adjacency matrix: adj[i,j] = 1 if edge i→j exists
- Edge cost bounds: lower[i,j], upper[i,j] for uniform sampling
- Value function: V[i] = expected cost from node i to target

State: [current_node, V_estimate_0, V_estimate_1, ..., V_estimate_n]
Decision: next_node (which neighbor to visit)
Exogenous: Sampled edge costs from uniform distributions
"""

from typing import NamedTuple, Tuple
from functools import partial
from jaxtyping import Array, Float, Int, PRNGKeyArray, Bool
import jax
import jax.numpy as jnp
import chex


# Type aliases
State = Float[Array, "..."]  # [current_node, V_0, V_1, ..., V_n]
Decision = Int[Array, ""]  # Next node index
Reward = Float[Array, ""]  # Negative cost
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information for SSP Static.

    Attributes:
        edge_costs: Sampled costs for all edges from current node.
    """
    edge_costs: Float[Array, "..."]  # Costs to neighbors


@chex.dataclass(frozen=True)
class SSPStaticConfig:
    """Configuration for SSP Static model.

    Attributes:
        n_nodes: Number of nodes in graph.
        edge_prob: Probability of edge existence (for random graph).
        cost_lower_bound: Lower bound for edge cost sampling.
        cost_upper_bound: Upper bound for edge cost sampling.
        origin_node: Starting node index.
        target_node: Goal node index.
        learning_rate: Step size for value function updates.
    """
    n_nodes: int = 10
    edge_prob: float = 0.3
    cost_lower_bound: float = 1.0
    cost_upper_bound: float = 10.0
    origin_node: int = 0
    target_node: int = -1  # -1 means last node
    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_nodes <= 1:
            raise ValueError(f"n_nodes must be > 1, got {self.n_nodes}")
        if not (0.0 < self.edge_prob <= 1.0):
            raise ValueError(
                f"edge_prob must be in (0, 1], got {self.edge_prob}"
            )
        if self.cost_lower_bound < 0:
            raise ValueError(
                f"cost_lower_bound must be non-negative, got {self.cost_lower_bound}"
            )
        if self.cost_upper_bound <= self.cost_lower_bound:
            raise ValueError(
                f"cost_upper_bound ({self.cost_upper_bound}) must be > "
                f"cost_lower_bound ({self.cost_lower_bound})"
            )
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be in (0, 1], got {self.learning_rate}"
            )


class SSPStaticModel:
    """JAX-native model for Stochastic Shortest Path (Static) problem.

    Navigates a directed graph with stochastic edge costs:
    - Adjacency matrix defines graph structure
    - Edge costs sampled from uniform distributions
    - Value function learned via Bellman updates
    - Goal: Reach target node with minimum cost

    Example:
        >>> config = SSPStaticConfig(n_nodes=5, edge_prob=0.4)
        >>> model = SSPStaticModel(config)
        >>> key = jax.random.PRNGKey(0)
        >>> state = model.init_state(key)
    """

    def __init__(self, config: SSPStaticConfig) -> None:
        """Initialize model and generate random graph.

        Args:
            config: Model configuration.
        """
        self.config = config

        # Resolve target node (-1 means last node)
        self.target_node = (
            config.target_node if config.target_node >= 0
            else config.n_nodes - 1
        )

        # Will be set during init_state
        self.adjacency: Bool[Array, "n n"] = jnp.zeros((config.n_nodes, config.n_nodes), dtype=bool)
        self.edge_lower: Float[Array, "n n"] = jnp.zeros((config.n_nodes, config.n_nodes))
        self.edge_upper: Float[Array, "n n"] = jnp.zeros((config.n_nodes, config.n_nodes))

    def init_state(self, key: Key) -> State:
        """Initialize state and generate random graph.

        Args:
            key: Random key for graph generation.

        Returns:
            Initial state [current_node, V_0, ..., V_n].
        """
        # Generate random graph
        key1, key2, key3 = jax.random.split(key, 3)

        # Create adjacency matrix (random edges)
        edge_probs = jax.random.uniform(
            key1, (self.config.n_nodes, self.config.n_nodes)
        )
        adjacency = (edge_probs < self.config.edge_prob) & \
                   (jnp.arange(self.config.n_nodes)[:, None] !=
                    jnp.arange(self.config.n_nodes)[None, :])  # No self-loops

        # Ensure path from origin to target exists (add edges if needed)
        # Simple approach: ensure each node i has edge to at least i+1
        for i in range(self.config.n_nodes - 1):
            adjacency = adjacency.at[i, i + 1].set(True)

        self.adjacency = adjacency

        # Sample edge cost bounds
        edge_lower = jax.random.uniform(
            key2,
            (self.config.n_nodes, self.config.n_nodes),
            minval=self.config.cost_lower_bound,
            maxval=self.config.cost_upper_bound
        ) * adjacency  # Zero out non-edges

        edge_upper = edge_lower + jax.random.uniform(
            key3,
            (self.config.n_nodes, self.config.n_nodes),
            minval=0.0,
            maxval=self.config.cost_upper_bound - self.config.cost_lower_bound
        ) * adjacency

        self.edge_lower = edge_lower
        self.edge_upper = edge_upper

        # Initialize value function (all zeros except target)
        V = jnp.zeros(self.config.n_nodes)
        V = V.at[self.target_node].set(0.0)

        # State: [current_node, V_0, V_1, ..., V_n]
        current_node = float(self.config.origin_node)
        state = jnp.concatenate([jnp.array([current_node]), V])

        return state

    @partial(jax.jit, static_argnums=(0,))
    def transition(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> State:
        """Move to next node and update value function.

        Args:
            state: Current state [current_node, V_0, ..., V_n].
            decision: Next node to visit.
            exog: Sampled edge costs.

        Returns:
            New state with updated current node and value function.
        """
        current_node_idx = state[0].astype(jnp.int32)
        V = state[1:]

        # Update value function using Bellman equation
        # V(current_node) ← (1-α)V(current_node) + α*(cost + V(next_node))
        edge_cost = exog.edge_costs[decision]
        next_V_value = V[decision]

        # Bellman update
        new_V_current = (
            (1 - self.config.learning_rate) * V[current_node_idx] +
            self.config.learning_rate * (edge_cost + next_V_value)
        )

        # Update value function
        V_new = V.at[current_node_idx].set(new_V_current)

        # New state: move to next node
        new_state = jnp.concatenate([jnp.array([decision], dtype=jnp.float32), V_new])

        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def reward(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> Reward:
        """Compute reward as negative edge cost.

        Args:
            state: Current state.
            decision: Next node.
            exog: Sampled edge costs.

        Returns:
            Negative cost (reward).
        """
        edge_cost = exog.edge_costs[decision]
        return -edge_cost

    def sample_exogenous(
        self,
        key: Key,
        state: State,
        time: int,
    ) -> ExogenousInfo:
        """Sample edge costs for current node's outgoing edges.

        Args:
            key: Random key.
            state: Current state.
            time: Current time step (unused).

        Returns:
            Sampled edge costs.
        """
        current_node_idx = state[0].astype(jnp.int32)

        # Sample costs for all edges from current node
        edge_costs = jax.random.uniform(
            key,
            (self.config.n_nodes,),
            minval=self.edge_lower[current_node_idx],
            maxval=self.edge_upper[current_node_idx]
        )

        # Zero out costs for non-edges
        edge_costs = edge_costs * self.adjacency[current_node_idx]

        return ExogenousInfo(edge_costs=edge_costs)

    @partial(jax.jit, static_argnums=(0,))
    def is_valid_decision(self, state: State, decision: Decision) -> jax.Array:
        """Check if decision is valid (edge exists).

        Args:
            state: Current state.
            decision: Next node to check.

        Returns:
            Boolean: True if edge exists.
        """
        current_node_idx = state[0].astype(jnp.int32)
        return self.adjacency[current_node_idx, decision]

    def get_neighbors(self, state: State) -> jax.Array:
        """Get valid neighbors of current node.

        Args:
            state: Current state.

        Returns:
            Indices of neighboring nodes.
        """
        current_node_idx = state[0].astype(jnp.int32)
        neighbors = jnp.where(
            self.adjacency[current_node_idx],
            jnp.arange(self.config.n_nodes),
            -1  # Mark non-neighbors
        )
        # Filter out -1s
        return neighbors[neighbors >= 0]

    def is_terminal(self, state: State) -> bool:
        """Check if current node is the target.

        Args:
            state: Current state.

        Returns:
            True if at target node.
        """
        current_node_idx = int(state[0])
        return current_node_idx == self.target_node

    def get_value_function(self, state: State) -> jax.Array:
        """Extract value function from state.

        Args:
            state: Current state.

        Returns:
            Value function array V.
        """
        return state[1:]

    def get_current_node(self, state: State) -> int:
        """Get current node index.

        Args:
            state: Current state.

        Returns:
            Current node index.
        """
        return int(state[0])
