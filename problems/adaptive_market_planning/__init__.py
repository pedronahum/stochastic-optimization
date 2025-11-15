"""Adaptive Market Planning problem for stochastic optimization.

This module implements a gradient-based learning problem where an agent
learns optimal order quantities through repeated market interactions:
- Agent places order quantity and observes demand
- Receives gradient signal (profit if undersupply, loss if oversupply)
- Updates order quantity using step size rules
- Learns to converge to optimal order quantity over time

Key components:
- AdaptiveMarketPlanningModel: JAX-native model for gradient learning
- Various policies: harmonic step size, Kesten's rule, constant step

Example:
    >>> from problems.adaptive_market_planning import (
    ...     AdaptiveMarketPlanningConfig,
    ...     AdaptiveMarketPlanningModel,
    ...     HarmonicStepPolicy
    ... )
    >>> import jax
    >>>
    >>> # Create model
    >>> config = AdaptiveMarketPlanningConfig(
    ...     price=1.0,
    ...     cost=0.5,
    ...     demand_mean=100.0
    ... )
    >>> model = AdaptiveMarketPlanningModel(config)
    >>>
    >>> # Create policy
    >>> policy = HarmonicStepPolicy(theta=1.0)
    >>>
    >>> # Simulate
    >>> key = jax.random.PRNGKey(0)
    >>> state = model.init_state(key)
    >>> decision = policy(None, state, key)
"""

from .model import (
    AdaptiveMarketPlanningModel,
    AdaptiveMarketPlanningConfig,
    ExogenousInfo,
    State,
    Decision,
    Reward,
)

from .policy import (
    HarmonicStepPolicy,
    KestenStepPolicy,
    ConstantStepPolicy,
    NeuralStepPolicy,
)

__all__ = [
    # Model
    "AdaptiveMarketPlanningModel",
    "AdaptiveMarketPlanningConfig",
    "ExogenousInfo",
    "State",
    "Decision",
    "Reward",
    # Policies
    "HarmonicStepPolicy",
    "KestenStepPolicy",
    "ConstantStepPolicy",
    "NeuralStepPolicy",
]
