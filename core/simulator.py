# core/simulator.py
# mypy: disable-error-code="no-any-return"
from __future__ import annotations

from typing import Any, Protocol, Tuple

import jax

PRNGKey = jax.Array  # alias for readability
Array = jax.Array


class Model(Protocol):
    def reset(self, *, key: PRNGKey) -> Any: ...
    def step(self, state: Any, action: Any, *, key: PRNGKey) -> Tuple[Any, float]: ...


class Policy(Protocol):
    def act(self, state: Any, *, key: PRNGKey) -> Any: ...


def rollout(model: Model, policy: Policy, horizon: int, *, key: PRNGKey) -> Array:
    """Vectorised, JIT‑friendly simulation."""

    def _body(carry: tuple[Any, PRNGKey], _: None) -> tuple[tuple[Any, PRNGKey], Any]:
        state, k = carry
        k, sub = jax.random.split(k)
        action = policy.act(state, key=sub)
        k, sub = jax.random.split(k)
        state, reward = model.step(state, action, key=sub)
        return (state, k), reward

    # Reset
    state = model.reset(key=key)
    (_, _), rewards = jax.lax.scan(_body, (state, key), xs=None, length=horizon)
    return rewards
