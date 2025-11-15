"""Stochastic Shortest Path Dynamic problem (JAX-native implementation).

This module implements a dynamic stochastic shortest path problem with:
- Time-indexed multi-step lookahead policies
- Risk-sensitive decision making using percentiles
- Running average cost estimation
- Backward induction for value function computation

Example:
    >>> from problems.ssp_dynamic import (
    ...     SSPDynamicConfig,
    ...     SSPDynamicModel,
    ...     LookaheadPolicy,
    ... )
    >>> config = SSPDynamicConfig(n_nodes=10, horizon=15)
    >>> model = SSPDynamicModel(config)
    >>> policy = LookaheadPolicy(theta=0.5)
"""

from problems.ssp_dynamic.model import (
    SSPDynamicConfig,
    SSPDynamicModel,
    ExogenousInfo,
)
from problems.ssp_dynamic.policy import (
    LookaheadPolicy,
    GreedyLookaheadPolicy,
    RandomPolicy,
)

__all__ = [
    "SSPDynamicConfig",
    "SSPDynamicModel",
    "ExogenousInfo",
    "LookaheadPolicy",
    "GreedyLookaheadPolicy",
    "RandomPolicy",
]
