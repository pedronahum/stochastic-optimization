"""Clinical Trials problem (faithful port of the original Powell problem).

A drug program enrolls patients over successive trials, tracking a
Beta(success, failure) belief about the success rate and deciding how many to
enroll and whether to continue or stop (declaring success/failure).
"""

from problems.clinical_trials.model import (
    ClinicalTrialsModel,
    Config,
    Decision,
    ExogenousInfo,
    Reward,
    State,
)
from problems.clinical_trials.policy import FixedEnrollPolicy, StoppingPolicy

__all__ = [
    "Config",
    "ClinicalTrialsModel",
    "ExogenousInfo",
    "State",
    "Decision",
    "Reward",
    "StoppingPolicy",
    "FixedEnrollPolicy",
]
