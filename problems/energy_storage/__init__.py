"""Energy Storage problem for stochastic optimization.

This module implements the battery energy storage problem where an agent must
decide when to charge/discharge a battery under price uncertainty and degradation.

Key components:
- EnergyStorageModel: JAX-native model with JIT-compiled dynamics
- Various policies: threshold-based, time-of-day, neural networks
- Configuration with chex dataclasses for type safety

Example:
    >>> from problems.energy_storage import (
    ...     EnergyStorageModel,
    ...     EnergyStorageConfig,
    ...     ThresholdPolicy,
    ...     ThresholdPolicyConfig
    ... )
    >>> import jax
    >>>
    >>> # Create model
    >>> config = EnergyStorageConfig(capacity=1000.0)
    >>> model = EnergyStorageModel(config)
    >>>
    >>> # Create policy
    >>> policy_config = ThresholdPolicyConfig(buy_threshold=40.0, sell_threshold=60.0)
    >>> policy = ThresholdPolicy(model, policy_config)
    >>>
    >>> # Simulate
    >>> key = jax.random.PRNGKey(0)
    >>> state = model.init_state(key)
    >>> decision = policy(None, state, key)
"""

from .model import (
    EnergyStorageModel,
    EnergyStorageConfig,
    ExogenousInfo,
    State,
    Decision,
    Reward,
)

from .policy import (
    ThresholdPolicy,
    ThresholdPolicyConfig,
    TimeOfDayPolicy,
    MyopicPolicy,
    LinearPolicy,
    NeuralPolicy,
    AlwaysHoldPolicy,
)

__all__ = [
    # Model
    "EnergyStorageModel",
    "EnergyStorageConfig",
    "ExogenousInfo",
    "State",
    "Decision",
    "Reward",
    # Policies
    "ThresholdPolicy",
    "ThresholdPolicyConfig",
    "TimeOfDayPolicy",
    "MyopicPolicy",
    "LinearPolicy",
    "NeuralPolicy",
    "AlwaysHoldPolicy",
]
