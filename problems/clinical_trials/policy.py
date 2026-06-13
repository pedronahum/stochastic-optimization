"""Policies for the Clinical Trials problem (faithful to the original).

The original lookahead/ADP policies stop the program when the success belief
crosses a threshold. ``StoppingPolicy`` captures that rule: estimate the success
probability ``p = success / (success + failure)`` and

  * stop & declare success when ``p >= theta_stop_high``,
  * stop & declare failure when ``p <= theta_stop_low``,
  * otherwise continue and enroll a fixed batch.
"""

from functools import partial
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

if TYPE_CHECKING:
    from problems.clinical_trials.model import ClinicalTrialsModel

State = Float[Array, "4"]
Decision = Float[Array, "3"]
Key = PRNGKeyArray


class StoppingPolicy:
    """Threshold stopping policy on the Beta(success, failure) success belief.

    Example:
        >>> policy = StoppingPolicy(model, enroll=20.0)
        >>> decision = policy(None, state, key)
    """

    def __init__(self, model: "ClinicalTrialsModel", enroll: float = 20.0) -> None:
        """Initialize policy.

        Args:
            model: Clinical trials model (provides the stop thresholds).
            enroll: Batch size to enroll while continuing.
        """
        self.model = model
        self.enroll = enroll

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, params: Optional[PyTree], state: State, key: Key) -> Decision:
        """Continue/enroll, or stop and declare success/failure by belief."""
        success, failure = state[1], state[2]
        p_hat = success / (success + failure)
        cfg = self.model.cfg
        declare_success = p_hat >= cfg.theta_stop_high
        stop = (p_hat >= cfg.theta_stop_high) | (p_hat <= cfg.theta_stop_low)
        enroll = jnp.where(stop, 0.0, self.enroll)
        prog_continue = jnp.where(stop, 0.0, 1.0)
        drug_success = jnp.where(declare_success, 1.0, 0.0)
        return jnp.array([enroll, prog_continue, drug_success])


class FixedEnrollPolicy:
    """Baseline: always continue and enroll a fixed batch."""

    def __init__(self, enroll: float = 20.0) -> None:
        """Initialize with a fixed enroll size."""
        self.enroll = enroll

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, params: Optional[PyTree], state: State, key: Key) -> Decision:
        """Always enroll ``enroll`` patients and continue the program."""
        return jnp.array([self.enroll, 1.0, 0.0])
