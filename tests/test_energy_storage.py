"""Tests for the Energy Storage problem (faithful port of the original)."""

import jax
import jax.numpy as jnp
import pytest

from problems.energy_storage import (
    AlwaysHoldPolicy,
    BuyLowSellHighPolicy,
    EnergyStorageConfig,
    EnergyStorageModel,
    ExogenousInfo,
    grid_search,
    simulate,
)

# ============================================================================
# Configuration
# ============================================================================


def test_config_default_values() -> None:
    cfg = EnergyStorageConfig()
    assert cfg.eta == 0.9
    assert cfg.capacity == 1.0
    assert cfg.initial_energy == 1.0


def test_config_validation_eta() -> None:
    with pytest.raises(ValueError, match="eta must be in"):
        EnergyStorageConfig(eta=1.5)
    with pytest.raises(ValueError, match="eta must be in"):
        EnergyStorageConfig(eta=0.0)


def test_config_validation_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        EnergyStorageConfig(capacity=-1.0)


def test_config_validation_initial_energy() -> None:
    with pytest.raises(ValueError, match="initial_energy"):
        EnergyStorageConfig(capacity=1.0, initial_energy=2.0)


# ============================================================================
# Model
# ============================================================================


def test_init_state() -> None:
    model = EnergyStorageModel(EnergyStorageConfig(), prices=jnp.array([20.0, 30.0]))
    state = model.init_state(jax.random.PRNGKey(0))
    assert state.shape == (2,)
    assert state[0] == 1.0  # initial energy
    assert state[1] == 20.0  # first price


def test_transition_energy_update() -> None:
    # energy' = energy + eta*buy - sell
    cfg = EnergyStorageConfig(eta=0.9, capacity=10.0, initial_energy=2.0)
    model = EnergyStorageModel(cfg)
    state = jnp.array([2.0, 50.0])
    decision = jnp.array([1.0, 0.0])  # buy 1
    exog = ExogenousInfo(price=jnp.array(40.0))
    nxt = model.transition(state, decision, exog)
    assert jnp.isclose(nxt[0], 2.0 + 0.9 * 1.0)  # 2.9
    assert nxt[1] == 40.0  # next price


def test_reward_sell_and_buy() -> None:
    # reward = price * (eta*sell - buy)
    cfg = EnergyStorageConfig(eta=0.9)
    model = EnergyStorageModel(cfg)
    state = jnp.array([5.0, 50.0])
    exog = ExogenousInfo(price=jnp.array(0.0))
    # selling 1 unit at price 50 with eta 0.9 -> 50 * 0.9 = 45
    assert jnp.isclose(model.reward(state, jnp.array([0.0, 1.0]), exog), 45.0)
    # buying 1 unit at price 50 -> 50 * (-1) = -50
    assert jnp.isclose(model.reward(state, jnp.array([1.0, 0.0]), exog), -50.0)


def test_apply_constraints() -> None:
    cfg = EnergyStorageConfig(eta=0.9, capacity=1.0)
    model = EnergyStorageModel(cfg)
    # positive buy charges to capacity: (Rmax - energy)/eta
    d = model.apply_constraints(jnp.array([0.25, 0.0]), jnp.array([1.0, 0.0]))
    assert jnp.isclose(d[0], (1.0 - 0.25) / 0.9)
    # sell beyond stored energy is clipped to energy
    d = model.apply_constraints(jnp.array([0.3, 50.0]), jnp.array([0.0, 1.0]))
    assert jnp.isclose(d[1], 0.3)


def test_sample_exogenous_uses_price_series() -> None:
    model = EnergyStorageModel(EnergyStorageConfig(), prices=jnp.array([10.0, 20.0, 30.0]))
    state = model.init_state(jax.random.PRNGKey(0))
    exog = model.sample_exogenous(jax.random.PRNGKey(0), state, 0)
    assert exog.price == 20.0  # prices[time + 1]


# ============================================================================
# Policies
# ============================================================================


def test_buy_low_sell_high_decisions() -> None:
    cfg = EnergyStorageConfig(eta=0.9, capacity=1.0, initial_energy=0.5)
    model = EnergyStorageModel(cfg)
    policy = BuyLowSellHighPolicy(model, theta_buy=20.0, theta_sell=60.0)
    key = jax.random.PRNGKey(0)
    # low price -> buy (positive)
    assert policy(None, jnp.array([0.5, 10.0]), key)[0] > 0
    # high price -> sell (positive)
    assert policy(None, jnp.array([0.5, 80.0]), key)[1] > 0
    # mid price -> hold
    mid = policy(None, jnp.array([0.5, 40.0]), key)
    assert jnp.allclose(mid, jnp.array([0.0, 0.0]))


def test_always_hold_policy() -> None:
    policy = AlwaysHoldPolicy()
    assert jnp.allclose(policy(None, jnp.array([0.5, 50.0]), jax.random.PRNGKey(0)),
                        jnp.array([0.0, 0.0]))


def test_simulate_and_grid_search() -> None:
    # prices oscillate: buy low (10), sell high (90) is profitable
    prices = jnp.array([10.0, 90.0, 10.0, 90.0, 10.0, 90.0])
    cfg = EnergyStorageConfig(eta=0.9, capacity=1.0, initial_energy=0.0)
    model = EnergyStorageModel(cfg, prices=prices)
    good = simulate(model, BuyLowSellHighPolicy(model, 20.0, 60.0), len(prices))
    hold = simulate(model, BuyLowSellHighPolicy(model, -1.0, 1e9), len(prices))  # never trades
    assert good > hold
    (tb, ts), best = grid_search(
        model, len(prices), jnp.arange(10.0, 50.0, 10.0), jnp.arange(50.0, 100.0, 10.0))
    assert best >= good - 1e-6  # grid search finds at least as good


# ============================================================================
# JAX transforms
# ============================================================================


def test_jit_compilation() -> None:
    model = EnergyStorageModel(EnergyStorageConfig())
    state = jnp.array([0.5, 50.0])
    exog = ExogenousInfo(price=jnp.array(40.0))
    nxt = jax.jit(model.transition)(state, jnp.array([1.0, 0.0]), exog)
    assert nxt.shape == (2,)


def test_vmap_batching() -> None:
    model = EnergyStorageModel(EnergyStorageConfig())
    states = jnp.stack([jnp.array([0.5, p]) for p in [10.0, 50.0, 90.0]])
    decisions = jnp.tile(jnp.array([0.0, 0.5]), (3, 1))
    exogs = ExogenousInfo(price=jnp.array([1.0, 2.0, 3.0]))
    rewards = jax.vmap(model.reward)(states, decisions, exogs)
    assert rewards.shape == (3,)
