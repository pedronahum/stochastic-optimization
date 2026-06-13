"""Smoke test: run a clinical-trials episode end-to-end with the stopping policy."""

import jax
import jax.numpy as jnp

from problems.clinical_trials import ClinicalTrialsModel, Config, StoppingPolicy


def test_episode_runs() -> None:
    model = ClinicalTrialsModel(Config())
    policy = StoppingPolicy(model, enroll=20.0)
    key = jax.random.PRNGKey(0)
    state = model.init_state(key)

    total = 0.0
    for _ in range(10):
        key, k_dec, k_exog = jax.random.split(key, 3)
        decision = policy(None, state, k_dec)
        exog = model.sample_exogenous(k_exog, state, decision)
        total += float(model.reward(state, decision, exog))
        state = model.transition(state, decision, exog)

    assert state.shape == (4,)
    assert jnp.isfinite(total)
