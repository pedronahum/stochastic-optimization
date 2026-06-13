"""Tests for the Clinical Trials problem (faithful port of the original)."""

import chex
import jax
import jax.numpy as jnp

from problems.clinical_trials import (
    ClinicalTrialsModel,
    Config,
    ExogenousInfo,
    FixedEnrollPolicy,
    StoppingPolicy,
)

# ============================================================================
# Configuration / model setup
# ============================================================================


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.alpha == 0.1
    assert cfg.success_rev == 5_000_000.0
    assert cfg.init_success == 195.0
    assert cfg.init_failure == 50.0


def test_init_state() -> None:
    model = ClinicalTrialsModel(Config())
    state = model.init_state(jax.random.PRNGKey(0))
    assert state.shape == (4,)
    # [potential_pop, success, failure, l_response]
    assert state[1] == 195.0
    assert state[2] == 50.0
    assert jnp.isclose(state[3], 0.06)


# ============================================================================
# Transition  (matches original transition_fn)
# ============================================================================


def test_transition_belief_counts() -> None:
    model = ClinicalTrialsModel(Config(alpha=0.1))
    state = jnp.array([0.0, 195.0, 50.0, 0.06])  # pop, succ, fail, lambda
    decision = jnp.array([20.0, 1.0, 0.0])  # enroll 20, continue
    exog = ExogenousInfo(new_patients=jnp.array(8.0), succ_count=jnp.array(6.0))
    nxt = model.transition(state, decision, exog)
    # potential_pop' = prog_continue * (pop + enroll) = 1*(0+20) = 20
    assert jnp.isclose(nxt[0], 20.0)
    # success' = 195 + 6 ; failure' = 50 + (8-6)
    assert jnp.isclose(nxt[1], 201.0)
    assert jnp.isclose(nxt[2], 52.0)
    # l_response' = 0.9*0.06 + 0.1*8/20
    assert jnp.isclose(nxt[3], 0.9 * 0.06 + 0.1 * 8.0 / 20.0)


def test_transition_stop_zeroes_population() -> None:
    model = ClinicalTrialsModel(Config())
    state = jnp.array([20.0, 195.0, 50.0, 0.06])
    decision = jnp.array([0.0, 0.0, 1.0])  # stop
    exog = ExogenousInfo(new_patients=jnp.array(0.0), succ_count=jnp.array(0.0))
    nxt = model.transition(state, decision, exog)
    assert nxt[0] == 0.0  # prog_continue=0 zeroes potential_pop


# ============================================================================
# Reward  (matches original objective_fn)
# ============================================================================


def test_reward_continue_costs() -> None:
    cfg = Config(program_cost=10000.0, patient_cost=500.0)
    model = ClinicalTrialsModel(cfg)
    state = jnp.array([0.0, 195.0, 50.0, 0.06])
    decision = jnp.array([20.0, 1.0, 0.0])  # continue, enroll 20
    exog = ExogenousInfo(new_patients=jnp.array(0.0), succ_count=jnp.array(0.0))
    # -(program_cost + patient_cost*enroll) = -(10000 + 500*20)
    assert jnp.isclose(model.reward(state, decision, exog), -(10000.0 + 500.0 * 20.0))


def test_reward_declare_success() -> None:
    cfg = Config(success_rev=5_000_000.0)
    model = ClinicalTrialsModel(cfg)
    state = jnp.array([0.0, 195.0, 50.0, 0.06])
    decision = jnp.array([0.0, 0.0, 1.0])  # stop & declare success
    exog = ExogenousInfo(new_patients=jnp.array(0.0), succ_count=jnp.array(0.0))
    assert jnp.isclose(model.reward(state, decision, exog), 5_000_000.0)


# ============================================================================
# Exogenous sampling
# ============================================================================


def test_sample_exogenous_shapes_and_bounds() -> None:
    model = ClinicalTrialsModel(Config())
    state = jnp.array([100.0, 195.0, 50.0, 0.06])
    decision = jnp.array([20.0, 1.0, 0.0])
    exog = model.sample_exogenous(jax.random.PRNGKey(0), state, decision)
    chex.assert_shape(exog.new_patients, ())
    chex.assert_shape(exog.succ_count, ())
    assert exog.new_patients >= 0
    assert 0 <= exog.succ_count <= exog.new_patients


# ============================================================================
# Policies
# ============================================================================


def test_stopping_policy_declares_success() -> None:
    model = ClinicalTrialsModel(Config(theta_stop_high=0.8, theta_stop_low=0.78))
    policy = StoppingPolicy(model, enroll=20.0)
    # p_hat = 90/100 = 0.9 >= theta_high -> stop & declare success
    state = jnp.array([0.0, 90.0, 10.0, 0.06])
    decision = policy(None, state, jax.random.PRNGKey(0))
    assert decision[1] == 0.0  # prog_continue
    assert decision[2] == 1.0  # drug_success


def test_stopping_policy_continues_when_uncertain() -> None:
    model = ClinicalTrialsModel(Config(theta_stop_high=0.8, theta_stop_low=0.78))
    policy = StoppingPolicy(model, enroll=20.0)
    # p_hat = 79/100 = 0.79 is between thresholds -> continue
    state = jnp.array([0.0, 79.0, 21.0, 0.06])
    decision = policy(None, state, jax.random.PRNGKey(0))
    assert decision[0] == 20.0  # enroll
    assert decision[1] == 1.0  # prog_continue


def test_fixed_enroll_policy() -> None:
    policy = FixedEnrollPolicy(enroll=30.0)
    decision = policy(None, jnp.array([0.0, 195.0, 50.0, 0.06]), jax.random.PRNGKey(0))
    assert jnp.allclose(decision, jnp.array([30.0, 1.0, 0.0]))


# ============================================================================
# JAX transforms
# ============================================================================


def test_jit_transition_reward() -> None:
    model = ClinicalTrialsModel(Config())
    state = jnp.array([0.0, 195.0, 50.0, 0.06])
    decision = jnp.array([20.0, 1.0, 0.0])
    exog = ExogenousInfo(new_patients=jnp.array(5.0), succ_count=jnp.array(4.0))
    assert jax.jit(model.transition)(state, decision, exog).shape == (4,)
    chex.assert_shape(jax.jit(model.reward)(state, decision, exog), ())


def test_vmap_reward() -> None:
    model = ClinicalTrialsModel(Config())
    states = jnp.tile(jnp.array([0.0, 195.0, 50.0, 0.06]), (4, 1))
    decisions = jnp.tile(jnp.array([20.0, 1.0, 0.0]), (4, 1))
    exogs = ExogenousInfo(new_patients=jnp.arange(4.0), succ_count=jnp.zeros(4))
    rewards = jax.vmap(model.reward)(states, decisions, exogs)
    assert rewards.shape == (4,)
