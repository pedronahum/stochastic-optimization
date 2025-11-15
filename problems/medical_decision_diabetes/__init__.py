"""Medical Decision Diabetes problem for stochastic optimization.

This module implements a Bayesian multi-armed bandit problem for selecting
optimal diabetes treatments:
- 5 drug options (Metformin, Sensitizer, Secretagogue, AGI, Peptide Analog)
- Bayesian learning of drug effectiveness through patient trials
- Exploration-exploitation tradeoff in treatment selection
- Goal: Maximize A1C reduction over time

Key components:
- MedicalDecisionDiabetesModel: JAX-native Bayesian bandit model
- Various policies: UCB, Interval Estimation, Pure Exploitation/Exploration

Example:
    >>> from problems.medical_decision_diabetes import (
    ...     MedicalDecisionDiabetesConfig,
    ...     MedicalDecisionDiabetesModel,
    ...     UCBPolicy
    ... )
    >>> import jax
    >>>
    >>> # Create model
    >>> config = MedicalDecisionDiabetesConfig()
    >>> model = MedicalDecisionDiabetesModel(config)
    >>>
    >>> # Create policy
    >>> policy = UCBPolicy(theta=2.0)
    >>>
    >>> # Simulate
    >>> key = jax.random.PRNGKey(0)
    >>> state = model.init_state(key)
    >>> decision = policy(None, state, key)
"""

from .model import (
    MedicalDecisionDiabetesModel,
    MedicalDecisionDiabetesConfig,
    ExogenousInfo,
    State,
    Decision,
    Reward,
    DrugIndex,
)

from .policy import (
    UCBPolicy,
    IntervalEstimationPolicy,
    PureExploitationPolicy,
    PureExplorationPolicy,
    ThompsonSamplingPolicy,
)

__all__ = [
    # Model
    "MedicalDecisionDiabetesModel",
    "MedicalDecisionDiabetesConfig",
    "ExogenousInfo",
    "State",
    "Decision",
    "Reward",
    "DrugIndex",
    # Policies
    "UCBPolicy",
    "IntervalEstimationPolicy",
    "PureExploitationPolicy",
    "PureExplorationPolicy",
    "ThompsonSamplingPolicy",
]
