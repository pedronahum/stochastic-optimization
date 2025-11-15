"""Blood Management problem (JAX-native implementation).

This module implements a blood bank inventory optimization problem with:
- Multiple blood types (O-, O+, A-, A+, B-, B+, AB-, AB+)
- Age-dependent inventory (blood expires after MAX_AGE days)
- Demand from urgent and elective surgeries
- Blood type substitution rules (e.g., O- is universal donor)
- Stochastic donations and demands with surge events
- Optimal allocation via projected gradient optimization

Example:
    >>> from problems.blood_management import (
    ...     BloodManagementConfig,
    ...     BloodManagementModel,
    ...     GreedyPolicy,
    ... )
    >>> config = BloodManagementConfig(max_age=5)
    >>> model = BloodManagementModel(config)
"""

from problems.blood_management.model import (
    BloodManagementConfig,
    BloodManagementModel,
    ExogenousInfo,
)
from problems.blood_management.policy import (
    GreedyPolicy,
    FIFOPolicy,
    RandomPolicy,
)

__all__ = [
    "BloodManagementConfig",
    "BloodManagementModel",
    "ExogenousInfo",
    "GreedyPolicy",
    "FIFOPolicy",
    "RandomPolicy",
]
