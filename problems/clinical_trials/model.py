"""JAX-native Clinical Trials model — faithful port of the original Powell problem.

Ported from ``legacy/old_problems/ClinicalTrials`` (Raluca Cobzaru / Donghun Lee,
2018). A drug program enrolls patients over successive trials; each period we
decide how many to enroll and whether to continue the program (or stop and
declare success/failure). Beliefs about the success rate are tracked as
Beta(success, failure) counts and the enrollment response rate ``l_response`` is
smoothed toward the observed arrival rate.

  state      = [potential_pop, success, failure, l_response]
  decision   = [enroll, prog_continue, drug_success]
  exogenous  = (new_patients ~ Poisson(true_l_response*(pop+enroll)),
                succ_count    ~ Binomial(new_patients, true_succ_rate))
  transition : potential_pop' = prog_continue*(pop + enroll)
               l_response'     = (1-alpha)*l_response + alpha*new_patients/(pop+enroll)
               success'        = success + succ_count
               failure'        = failure + (new_patients - succ_count)
  reward     : (1-prog_continue)*drug_success*success_rev
               - prog_continue*(program_cost + patient_cost*enroll)
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, PRNGKeyArray

# Type aliases
State = Float[Array, "4"]  # [potential_pop, success, failure, l_response]
Decision = Float[Array, "3"]  # [enroll, prog_continue, drug_success]
Reward = Float[Array, ""]
Key = PRNGKeyArray


class ExogenousInfo(NamedTuple):
    """Exogenous information for a trial period.

    Attributes:
        new_patients: Newly enrolled patients this period.
        succ_count: Successes among the new patients.
    """
    new_patients: Float[Array, ""]
    succ_count: Float[Array, ""]


@struct.dataclass
class Config:
    """Configuration for the clinical trials model (original Trials Parameters).

    Attributes:
        alpha: Smoothing weight for the enrollment-response update.
        patient_cost: Cost per enrolled patient.
        program_cost: Per-period cost of continuing the program.
        success_rev: Revenue if the drug is declared a success.
        true_l_response: True enrollment-response rate (data-generating).
        true_succ_rate: True per-patient success probability (data-generating).
        theta_stop_low / theta_stop_high: Stopping thresholds on the success belief.
        init_potential_pop, init_success, init_failure, init_l_response: initial state.
    """
    alpha: float = 0.1
    patient_cost: float = 500.0
    program_cost: float = 10000.0
    success_rev: float = 5_000_000.0
    true_l_response: float = 0.05
    true_succ_rate: float = 0.85
    theta_stop_low: float = 0.78
    theta_stop_high: float = 0.80
    init_potential_pop: float = 0.0
    init_success: float = 195.0
    init_failure: float = 50.0
    init_l_response: float = 0.06


class ClinicalTrialsModel:
    """Clinical trials enrollment model (faithful to the original).

    Example:
        >>> import jax
        >>> model = ClinicalTrialsModel(Config())
        >>> state = model.init_state(jax.random.PRNGKey(0))
        >>> decision = jnp.array([20.0, 1.0, 0.0])  # enroll 20, continue
        >>> exog = model.sample_exogenous(jax.random.PRNGKey(1), state, decision)
        >>> next_state = model.transition(state, decision, exog)
    """

    def __init__(self, cfg: Config) -> None:
        """Initialize the model with a configuration."""
        self.cfg = cfg

    def init_state(self, key: Key) -> State:
        """Initial state ``[potential_pop, success, failure, l_response]``."""
        c = self.cfg
        return jnp.array([c.init_potential_pop, c.init_success, c.init_failure, c.init_l_response])

    @partial(jax.jit, static_argnums=(0,))
    def transition(self, state: State, decision: Decision, exog: ExogenousInfo) -> State:
        """Update beliefs, response rate and population (original ``transition_fn``)."""
        potential_pop, success, failure, l_response = state[0], state[1], state[2], state[3]
        enroll, prog_continue = decision[0], decision[1]

        enrolled_base = potential_pop + enroll
        new_potential_pop = prog_continue * enrolled_base
        new_l_response = (1.0 - self.cfg.alpha) * l_response + \
            self.cfg.alpha * exog.new_patients / enrolled_base
        new_success = success + exog.succ_count
        new_failure = failure + (exog.new_patients - exog.succ_count)
        return jnp.array([new_potential_pop, new_success, new_failure, new_l_response])

    @partial(jax.jit, static_argnums=(0,))
    def reward(self, state: State, decision: Decision, exog: ExogenousInfo) -> Reward:
        """Per-period contribution (original ``objective_fn``)."""
        enroll, prog_continue, drug_success = decision[0], decision[1], decision[2]
        return (
            (1.0 - prog_continue) * drug_success * self.cfg.success_rev
            - prog_continue * (self.cfg.program_cost + self.cfg.patient_cost * enroll)
        )

    def sample_exogenous(self, key: Key, state: State, decision: Decision) -> ExogenousInfo:
        """Sample new patients (Poisson) and successes (Binomial) for this period."""
        potential_pop, enroll = state[0], decision[0]
        k1, k2 = jax.random.split(key)
        lam = self.cfg.true_l_response * (potential_pop + enroll)
        new_patients = jax.random.poisson(k1, lam).astype(jnp.float32)
        succ_count = jax.random.binomial(k2, new_patients, self.cfg.true_succ_rate)
        return ExogenousInfo(new_patients=new_patients, succ_count=succ_count)
