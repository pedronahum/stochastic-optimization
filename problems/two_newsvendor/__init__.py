"""Two Newsvendor problem for stochastic optimization.

This module implements a two-agent newsvendor coordination problem where:
- Field agent observes demand estimate and requests quantity
- Central agent allocates quantity based on own estimate and Field's request
- Both agents learn biases over time

Key components:
- TwoNewsvendorFieldModel: JAX-native model for Field agent
- TwoNewsvendorCentralModel: JAX-native model for Central agent
- Various policies: newsvendor formula, bias-adjusted, neural networks

Example:
    >>> from problems.two_newsvendor import (
    ...     TwoNewsvendorConfig,
    ...     TwoNewsvendorFieldModel,
    ...     TwoNewsvendorCentralModel,
    ...     NewsvendorFieldPolicy,
    ...     NewsvendorCentralPolicy
    ... )
    >>> import jax
    >>>
    >>> # Create models
    >>> config = TwoNewsvendorConfig()
    >>> field_model = TwoNewsvendorFieldModel(config)
    >>> central_model = TwoNewsvendorCentralModel(config)
    >>>
    >>> # Create policies
    >>> field_policy = NewsvendorFieldPolicy(field_model)
    >>> central_policy = NewsvendorCentralPolicy(central_model, trust_field=0.5)
    >>>
    >>> # Simulate
    >>> key = jax.random.PRNGKey(0)
    >>> field_state = field_model.init_state(key)
    >>> field_decision = field_policy(None, field_state, key)
"""

from .model import (
    TwoNewsvendorFieldModel,
    TwoNewsvendorCentralModel,
    TwoNewsvendorConfig,
    ExogenousInfo,
    StateField,
    StateCentral,
    DecisionField,
    DecisionCentral,
    Reward,
)

from .policy import (
    NewsvendorFieldPolicy,
    BiasAdjustedFieldPolicy,
    NewsvendorCentralPolicy,
    BiasAdjustedCentralPolicy,
    NeuralFieldPolicy,
    NeuralCentralPolicy,
    AlwaysAllocateRequestedPolicy,
)

__all__ = [
    # Models
    "TwoNewsvendorFieldModel",
    "TwoNewsvendorCentralModel",
    "TwoNewsvendorConfig",
    "ExogenousInfo",
    "StateField",
    "StateCentral",
    "DecisionField",
    "DecisionCentral",
    "Reward",
    # Policies
    "NewsvendorFieldPolicy",
    "BiasAdjustedFieldPolicy",
    "NewsvendorCentralPolicy",
    "BiasAdjustedCentralPolicy",
    "NeuralFieldPolicy",
    "NeuralCentralPolicy",
    "AlwaysAllocateRequestedPolicy",
]
