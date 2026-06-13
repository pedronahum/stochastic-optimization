"""Clinical Trials example — run the enrollment program under the stopping policy.

Simulates the drug-enrollment trial: each period the stopping policy decides
whether to continue enrolling patients or to stop and declare success/failure
based on the Beta(success, failure) belief. We average the total program
contribution over several sample paths.
"""

import jax
import jax.numpy as jnp

from problems.clinical_trials import ClinicalTrialsModel, Config, StoppingPolicy

MAX_TRIALS = 30
N_PATHS = 200


def run_path(model: ClinicalTrialsModel, policy: StoppingPolicy, key: jax.Array) -> float:
    """Run one sample path and return the total contribution."""
    state = model.init_state(key)
    total = 0.0
    for _ in range(MAX_TRIALS):
        key, k_dec, k_exog = jax.random.split(key, 3)
        decision = policy(None, state, k_dec)
        exog = model.sample_exogenous(k_exog, state, decision)
        total += float(model.reward(state, decision, exog))
        state = model.transition(state, decision, exog)
        if float(decision[1]) == 0.0:  # program stopped
            break
    return total


def main() -> None:
    """Evaluate the stopping policy over many sample paths."""
    cfg = Config()
    model = ClinicalTrialsModel(cfg)
    policy = StoppingPolicy(model, enroll=20.0)

    print("Clinical Trials — enrollment program under the stopping policy")
    print("=" * 70)
    print(f"Initial belief: success={cfg.init_success:.0f}, failure={cfg.init_failure:.0f} "
          f"(p={cfg.init_success / (cfg.init_success + cfg.init_failure):.3f})")
    print(f"Stop thresholds: low={cfg.theta_stop_low}, high={cfg.theta_stop_high}")

    keys = jax.random.split(jax.random.PRNGKey(0), N_PATHS)
    contributions = jnp.array([run_path(model, policy, k) for k in keys])

    print(f"\nMean contribution over {N_PATHS} paths: {float(jnp.mean(contributions)):,.0f}")
    print(f"Std: {float(jnp.std(contributions)):,.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
