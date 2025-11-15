"""SSP Dynamic model - JAX-native implementation with lookahead."""

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray as Key

# Type aliases
State = Float[Array, "..."]  # [current_node, time, estimated_costs..., obs_count...]
Decision = Int[Array, ""]  # Next node index
Reward = Float[Array, ""]  # Negative cost


@dataclass(frozen=True)
class ExogenousInfo:
    """Exogenous information for SSP Dynamic.

    Attributes:
        edge_costs: Sampled costs for edges from current node.
    """

    edge_costs: Float[Array, "n_nodes"]


# Register ExogenousInfo as a JAX pytree
jax.tree_util.register_pytree_node(
    ExogenousInfo,
    lambda obj: ((obj.edge_costs,), None),  # flatten
    lambda aux, children: ExogenousInfo(*children),  # unflatten
)


@dataclass(frozen=True)
class SSPDynamicConfig:
    """Configuration for SSP Dynamic problem.

    Attributes:
        n_nodes: Number of nodes in the graph.
        horizon: Lookahead horizon for dynamic programming.
        edge_prob: Probability of edge existing between nodes.
        cost_min: Minimum edge cost.
        cost_max: Maximum edge cost.
        max_spread: Maximum relative spread for edge costs (as fraction).
        target_node: Target/destination node (default: last node).
        seed: Random seed for graph generation.
    """

    n_nodes: int = 10
    horizon: int = 15
    edge_prob: float = 0.3
    cost_min: float = 1.0
    cost_max: float = 10.0
    max_spread: float = 0.3  # Maximum 30% spread
    target_node: Optional[int] = None
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_nodes <= 1:
            raise ValueError("n_nodes must be > 1")
        if not 0.0 < self.edge_prob <= 1.0:
            raise ValueError("edge_prob must be in (0, 1]")
        if self.cost_max <= self.cost_min:
            raise ValueError("cost_max must be > cost_min")
        if not 0.0 <= self.max_spread < 1.0:
            raise ValueError("max_spread must be in [0, 1)")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


class SSPDynamicModel:
    """SSP Dynamic model with multi-step lookahead.

    This model implements a stochastic shortest path problem with:
    - Time-indexed value function computation
    - Running average cost estimation
    - Risk-sensitive percentile-based policies

    State representation: [current_node, time, est_cost_0, ..., est_cost_n, obs_0, ..., obs_n]
    - current_node: Current position
    - time: Current time step
    - est_cost_i: Estimated mean cost for edges from node i
    - obs_i: Number of observations for node i (for running average)

    Example:
        >>> config = SSPDynamicConfig(n_nodes=5, horizon=10)
        >>> model = SSPDynamicModel(config)
        >>> key = jax.random.PRNGKey(42)
        >>> state = model.init_state(key)
    """

    def __init__(self, config: SSPDynamicConfig) -> None:
        """Initialize SSP Dynamic model.

        Args:
            config: Problem configuration.
        """
        self.config = config
        self.target_node = (
            config.target_node if config.target_node is not None else config.n_nodes - 1
        )

        # Generate graph structure
        key = jax.random.PRNGKey(config.seed)
        self._generate_graph(key)

    def _generate_graph(self, key: Key) -> None:
        """Generate random directed graph with guaranteed path to target.

        Args:
            key: Random key for generation.
        """
        key1, key2, key3, key4 = jax.random.split(key, 4)

        # Generate adjacency matrix
        edge_prob_matrix = jax.random.uniform(
            key1, (self.config.n_nodes, self.config.n_nodes)
        )
        adjacency = (edge_prob_matrix < self.config.edge_prob) & (
            jnp.arange(self.config.n_nodes)[:, None]
            != jnp.arange(self.config.n_nodes)[None, :]
        )

        # Force at least one outgoing edge per node (except target)
        for i in range(self.config.n_nodes - 1):
            if not jnp.any(adjacency[i]):
                # Add edge to next node
                adjacency = adjacency.at[i, min(i + 1, self.target_node)].set(True)

        # Generate mean costs for edges
        mean_costs = jax.random.uniform(
            key2,
            (self.config.n_nodes, self.config.n_nodes),
            minval=self.config.cost_min,
            maxval=self.config.cost_max,
        )

        # Generate spreads for edges
        spreads = jax.random.uniform(
            key3, (self.config.n_nodes, self.config.n_nodes), maxval=self.config.max_spread
        )

        # Zero out non-edges
        mean_costs = jnp.where(adjacency, mean_costs, 0.0)
        spreads = jnp.where(adjacency, spreads, 0.0)

        # Target node self-loop
        adjacency = adjacency.at[self.target_node, self.target_node].set(True)
        mean_costs = mean_costs.at[self.target_node, self.target_node].set(0.0)
        spreads = spreads.at[self.target_node, self.target_node].set(0.0)

        self.adjacency: Bool[Array, "n n"] = adjacency
        self.mean_costs: Float[Array, "n n"] = mean_costs
        self.spreads: Float[Array, "n n"] = spreads

    def init_state(self, key: Key) -> State:
        """Initialize state at origin node.

        State structure:
        [current_node, time, est_cost_0_0, ..., est_cost_0_n, ..., est_cost_n_n, obs_0, ..., obs_n]

        Args:
            key: Random key (unused but kept for API consistency).

        Returns:
            Initial state at node 0, time 0, with estimated costs initialized to mean costs.
        """
        n = self.config.n_nodes

        # Initialize: current_node=0, time=0
        current_node = jnp.array(0.0)
        time = jnp.array(0.0)

        # Initialize estimated costs to mean costs (flattened)
        estimated_costs = self.mean_costs.flatten()

        # Initialize observation counts to 1 for all node pairs
        obs_counts = jnp.ones(n * n)

        # Concatenate all components
        state = jnp.concatenate([
            jnp.array([current_node, time]),
            estimated_costs,
            obs_counts
        ])

        return state

    @partial(jax.jit, static_argnums=(0,))
    def transition(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> State:
        """Move to next node and update estimated costs.

        Args:
            state: Current state.
            decision: Next node to visit.
            exog: Sampled edge costs.

        Returns:
            New state with updated position, time, and cost estimates.
        """
        current_node_idx = state[0].astype(jnp.int32)
        time = state[1]
        n = self.config.n_nodes

        # Extract estimated costs and observation counts
        estimated_costs = state[2:2 + n * n].reshape(n, n)
        obs_counts = state[2 + n * n:].reshape(n, n)

        # Update estimated cost for the edge we just traversed
        edge_cost = exog.edge_costs[decision]
        obs_count = obs_counts[current_node_idx, decision]

        # Running average: alpha = 1 / obs_count
        alpha = 1.0 / obs_count
        new_estimated_cost = (1 - alpha) * estimated_costs[current_node_idx, decision] + alpha * edge_cost

        # Update arrays
        estimated_costs = estimated_costs.at[current_node_idx, decision].set(new_estimated_cost)
        obs_counts = obs_counts.at[current_node_idx, decision].set(obs_count + 1)

        # New state: move to next node, increment time
        new_state = jnp.concatenate([
            jnp.array([decision], dtype=jnp.float32),
            jnp.array([time + 1]),
            estimated_costs.flatten(),
            obs_counts.flatten()
        ])

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

        Costs are sampled uniformly from:
        [(1 - spread) * mean, (1 + spread) * mean]

        Args:
            key: Random key.
            state: Current state.
            time: Current time step (unused).

        Returns:
            Sampled edge costs.
        """
        current_node_idx = state[0].astype(jnp.int32)

        # Get mean and spread for current node's edges
        mean = self.mean_costs[current_node_idx]
        spread = self.spreads[current_node_idx]

        # Sample from uniform distribution with spread
        deviation = jax.random.uniform(
            key,
            (self.config.n_nodes,),
            minval=-spread,
            maxval=spread
        ) * mean

        edge_costs = mean + deviation

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

    def is_terminal(self, state: State) -> bool:
        """Check if current node is the target.

        Args:
            state: Current state.

        Returns:
            True if at target node.
        """
        current_node = int(state[0])
        return current_node == self.target_node

    def get_estimated_costs(self, state: State) -> Float[Array, "n n"]:
        """Extract estimated costs matrix from state.

        Args:
            state: Current state.

        Returns:
            Estimated costs matrix [n_nodes, n_nodes].
        """
        n = self.config.n_nodes
        return state[2:2 + n * n].reshape(n, n)

    def get_time(self, state: State) -> int:
        """Extract current time from state.

        Args:
            state: Current state.

        Returns:
            Current time step.
        """
        return int(state[1])
