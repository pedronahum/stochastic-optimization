from typing import Any
import jax.numpy as jnp
from flax import nnx  # ← note the new import

from core.simulator import PRNGKey

from .model import State  # local import


class LinearDosePolicy(nnx.Module):
    """Parametric policy  a_t = w * x_t  ."""

    def __init__(self) -> None:
        self.w = nnx.Param(jnp.array(0.1))  # trainable scalar parameter

    def act(self, state: State, *, key: PRNGKey) -> Any:  # Model.Policy protocol
        del key  # deterministic
        return self.w * state.x
