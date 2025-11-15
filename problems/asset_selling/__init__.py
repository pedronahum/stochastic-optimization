"""Asset Selling problem for stochastic optimization.

This module implements the asset selling problem where an agent must decide
when to sell an asset under price uncertainty. The price follows a random walk
with Markov-modulated drift (bias states: Up, Neutral, Down).

Key components:
- AssetSellingModel: JAX-native model with JIT-compiled dynamics
- Various policies: threshold-based, expected value, neural networks
- Configuration with chex dataclasses for type safety

Example:
    >>> from problems.asset_selling import (
    ...     AssetSellingModel,
    ...     AssetSellingConfig,
    ...     HighLowPolicy
    ... )
    >>> import jax
    >>>
    >>> # Create model
    >>> config = AssetSellingConfig(initial_price=100.0)
    >>> model = AssetSellingModel(config)
    >>>
    >>> # Create policy
    >>> policy = HighLowPolicy(low_threshold=90.0, high_threshold=110.0)
    >>>
    >>> # Simulate
    >>> key = jax.random.PRNGKey(0)
    >>> state = model.init_state(key)
    >>> decision = policy(None, state, key)
"""

from .model import (
    AssetSellingModel,
    AssetSellingConfig,
    ExogenousInfo,
    State,
    Decision,
    Reward,
)

from .policy import (
    SellLowPolicy,
    HighLowPolicy,
    ExpectedValuePolicy,
    LinearThresholdPolicy,
    NeuralPolicy,
    AlwaysHoldPolicy,
    AlwaysSellPolicy,
    ThresholdPolicyConfig,
)

__all__ = [
    # Model
    "AssetSellingModel",
    "AssetSellingConfig",
    "ExogenousInfo",
    "State",
    "Decision",
    "Reward",
    # Policies
    "SellLowPolicy",
    "HighLowPolicy",
    "ExpectedValuePolicy",
    "LinearThresholdPolicy",
    "NeuralPolicy",
    "AlwaysHoldPolicy",
    "AlwaysSellPolicy",
    "ThresholdPolicyConfig",
]
