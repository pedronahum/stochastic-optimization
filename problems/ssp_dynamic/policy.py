"""Policies for SSP Dynamic problem."""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray as Key, PyTree

# Type aliases
State = Float[Array, "..."]
Decision = Int[Array, ""]


class LookaheadPolicy:
    """Risk-sensitive lookahead policy with percentile-based costs.

    Uses backward induction to compute optimal decisions over a finite horizon.
    Incorporates risk-sensitivity via percentile parameter theta:
    - theta=0.0: pessimistic (uses (1-spread)*mean costs)
    - theta=0.5: neutral (uses mean costs)
    - theta=1.0: optimistic (uses (1+spread)*mean costs)

    Example:
        >>> policy = LookaheadPolicy(theta=0.5)
        >>> # theta=0.5 uses mean costs (risk-neutral)
    """

    def __init__(self, theta: float = 0.5) -> None:
        """Initialize lookahead policy.

        Args:
            theta: Risk parameter in [0, 1]. Higher = more optimistic.
        """
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")
        self.theta = theta

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "SSPDynamicModel",  # type: ignore[name-defined]
    ) -> Decision:
        """Compute optimal decision using backward induction.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).
            model: Model instance.

        Returns:
            Optimal next node based on lookahead.
        """
        current_node_idx = state[0].astype(jnp.int32)
        n = model.config.n_nodes
        horizon = model.config.horizon

        # Get estimated costs and compute percentile-adjusted costs
        estimated_costs = model.get_estimated_costs(state)

        # Compute percentile value: (1 - spread) + 2*spread*theta
        # This maps theta ∈ [0,1] to percentile ∈ [(1-spread), (1+spread)]
        percentile_factor = (1 - model.spreads) + 2 * model.spreads * self.theta
        adjusted_costs = estimated_costs * percentile_factor

        # Initialize value function V[t][node] = inf, except target
        V = jnp.full((horizon + 1, n), jnp.inf)
        V = V.at[:, model.target_node].set(0.0)

        # Backward induction from horizon-1 to 0
        def backward_step(t: int, V: Float[Array, "H n"]) -> Float[Array, "H n"]:
            """Single backward induction step."""
            # For each node, compute min cost-to-go
            def compute_node_value(k: Int[Array, ""]) -> Float[Array, ""]:
                """Compute V[t][k] = min_l (cost[k,l] + V[t+1][l])."""
                # Cost to each neighbor + future value
                cost_to_go = adjusted_costs[k] + V[t + 1]

                # Mask non-neighbors with inf
                valid_edges = model.adjacency[k]
                cost_to_go = jnp.where(valid_edges, cost_to_go, jnp.inf)

                # Return minimum
                return jnp.min(cost_to_go)

            # Vectorize over all nodes
            new_V_t = jax.vmap(compute_node_value)(jnp.arange(n))
            return V.at[t].set(new_V_t)

        # Run backward induction
        for t in range(horizon - 1, -1, -1):
            V = backward_step(t, V)

        # At time 0, select best neighbor for current node
        cost_to_go = adjusted_costs[current_node_idx] + V[0]
        valid_edges = model.adjacency[current_node_idx]
        cost_to_go = jnp.where(valid_edges, cost_to_go, jnp.inf)

        decision = jnp.argmin(cost_to_go).astype(jnp.int32)
        return decision


class GreedyLookaheadPolicy:
    """Greedy policy using single-step lookahead with estimated costs.

    Similar to LookaheadPolicy but only looks ahead one step.
    Uses risk-neutral estimated costs (no percentile adjustment).

    Example:
        >>> policy = GreedyLookaheadPolicy()
    """

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "SSPDynamicModel",  # type: ignore[name-defined]
    ) -> Decision:
        """Select neighbor with minimum estimated cost.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).
            model: Model instance.

        Returns:
            Next node with minimum estimated cost.
        """
        current_node_idx = state[0].astype(jnp.int32)

        # Get estimated costs for current node
        estimated_costs = model.get_estimated_costs(state)
        edge_costs = estimated_costs[current_node_idx]

        # Mask non-neighbors
        valid_edges = model.adjacency[current_node_idx]
        edge_costs = jnp.where(valid_edges, edge_costs, jnp.inf)

        # Select minimum
        decision = jnp.argmin(edge_costs).astype(jnp.int32)
        return decision


class RandomPolicy:
    """Random policy - selects uniformly from valid neighbors.

    Useful as a baseline for comparison.

    Example:
        >>> policy = RandomPolicy()
    """

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "SSPDynamicModel",  # type: ignore[name-defined]
    ) -> Decision:
        """Select random valid neighbor.

        Args:
            params: Unused.
            state: Current state.
            key: Random key.
            model: Model instance.

        Returns:
            Random next node.
        """
        current_node_idx = state[0].astype(jnp.int32)

        # Get valid neighbors using categorical sampling
        valid_edges = model.adjacency[current_node_idx]
        neighbor_probs = valid_edges / jnp.sum(valid_edges)

        # Random choice weighted by valid edges
        decision = jax.random.choice(
            key,
            jnp.arange(model.config.n_nodes),
            p=neighbor_probs
        )
        return decision.astype(jnp.int32)
