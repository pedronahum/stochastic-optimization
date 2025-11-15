from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from core.simulator import PRNGKey


@struct.dataclass
class State:
    t: int
    x: jnp.ndarray  # patient health metric (shape = ())


@struct.dataclass
class Config:
    horizon: int = 20
    sigma: float = 0.25
    mu: float = 0.0


class ClinicalTrialsModel:
    """Simple stochastic patient evolution x_{t+1}=x_t+μ+σ·ε."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ---- Model API --------------------------------------------------------
    def reset(self, *, key: PRNGKey) -> State:
        return State(t=0, x=jnp.zeros(()))

    def step(self, state: State, action: float, *, key: PRNGKey) -> tuple[State, jax.Array]:
        """`action` ∈ ℝ is the dose; reward = −|x_t| penalises deviation."""
        ε = jr.normal(key) * self.cfg.sigma + self.cfg.mu
        new_x = state.x + action + ε
        reward = -jnp.abs(new_x)
        new_state = State(t=state.t + 1, x=new_x)
        return new_state, reward
