import jax

from stochopt.core import simulator as sim
from problems.clinical_trials import model, policy


def test_single_rollout_matches_shapes():
    cfg = model.Config()
    mdl = model.ClinicalTrialsModel(cfg)
    π = policy.LinearDosePolicy()
    key = jax.random.PRNGKey(0)

    rewards = sim.rollout(mdl, π, cfg.horizon, key=key)
    assert rewards.shape == (cfg.horizon,)
