"""Stochastic Shortest Path (Static) problem for stochastic optimization.

This module implements a shortest path problem on a static directed graph with
stochastic edge costs:
- Navigate from origin to target node
- Edge costs are random (uniform distribution)
- Learn value function through Bellman updates
- Minimize expected path cost

Key components:
- SSPStaticModel: JAX-native graph navigation with value iteration
- Various policies: Greedy (w.r.t. value function), Random exploration

Example:
    >>> from problems.ssp_static import (
    ...     SSPStaticConfig,
    ...     SSPStaticModel,
    ...     GreedyPolicy
    ... )
    >>> import jax
    >>>
    >>> # Create model with random graph
    >>> config = SSPStaticConfig(n_nodes=10, edge_prob=0.3)
    >>> model = SSPStaticModel(config)
    >>>
    >>> # Create policy
    >>> policy = GreedyPolicy()
    >>>
    >>> # Simulate
    >>> key = jax.random.PRNGKey(0)
    >>> state = model.init_state(key)
    >>> decision = policy(None, state, key, model)
"""

from .model import (
    SSPStaticModel,
    SSPStaticConfig,
    ExogenousInfo,
    State,
    Decision,
    Reward,
)

from .policy import (
    GreedyPolicy,
    EpsilonGreedyPolicy,
    RandomPolicy,
)

__all__ = [
    # Model
    "SSPStaticModel",
    "SSPStaticConfig",
    "ExogenousInfo",
    "State",
    "Decision",
    "Reward",
    # Policies
    "GreedyPolicy",
    "EpsilonGreedyPolicy",
    "RandomPolicy",
]
