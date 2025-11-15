# Quick Start Guide

## 🚀 Getting Started with the Stochastic Optimization Library

This guide will help you get started with the modernized JAX-native library in minutes.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/pedronahum/stochastic-optimization.git
cd stochastic-optimization

# Install dependencies
pip install jax jaxlib jaxtyping chex numpy pytest

# Install package in development mode
pip install -e .
```

---

## Your First Problem

Let's run a simple blood management simulation:

```python
import jax
import jax.numpy as jnp
from stochopt.problems.blood_management import (
    BloodManagementConfig,
    BloodManagementModel,
    GreedyPolicy
)

# 1. Configure the problem
config = BloodManagementConfig(
    max_age=5,              # Blood expires after 5 days
    surge_prob=0.1,         # 10% chance of demand surge
    seed=42
)

# 2. Create model and policy
model = BloodManagementModel(config)
policy = GreedyPolicy()

# 3. Initialize
key = jax.random.PRNGKey(42)
state = model.init_state(key)

# 4. Run simulation
total_reward = 0.0
for t in range(30):
    # Get decision from policy
    key, subkey = jax.random.split(key)
    decision = policy(None, state, subkey, model)

    # Sample random events
    key, subkey = jax.random.split(key)
    exog = model.sample_exogenous(subkey, state, t)

    # Compute reward and transition
    reward = model.reward(state, decision, exog)
    total_reward += float(reward)

    state = model.transition(state, decision, exog)

    # Print progress
    inventory = model.get_inventory(state)
    print(f"Day {t}: Reward={reward:.1f}, Total Inventory={jnp.sum(inventory):.0f}")

print(f"\nFinal Total Reward: {total_reward:.2f}")
```

---

## Running Tests

### Test Everything
```bash
pytest tests/ -v
```

### Test Specific Problem
```bash
pytest tests/test_blood_management.py -v
```

### Quick Test (Just Count)
```bash
pytest tests/ -q
```

---

## Type Checking

```bash
# Check all problems
mypy problems/ --strict

# Check specific problem
mypy problems/blood_management/ --strict
```

---

## Available Problems

### 1. **Blood Management** - Inventory optimization
```python
from stochopt.problems.blood_management import BloodManagementConfig, BloodManagementModel
```

### 2. **Clinical Trials** - Adaptive dose optimization
```python
from stochopt.problems.clinical_trials import ClinicalTrialsConfig, ClinicalTrialsModel
```

### 3. **SSP Dynamic** - Path planning with lookahead
```python
from stochopt.problems.ssp_dynamic import SSPDynamicConfig, SSPDynamicModel
```

### 4. **SSP Static** - Shortest path with risk
```python
from stochopt.problems.ssp_static import SSPStaticConfig, SSPStaticModel
```

### 5. **Adaptive Market Planning** - Dynamic pricing
```python
from stochopt.problems.adaptive_market_planning import AdaptiveMarketConfig, AdaptiveMarketModel
```

### 6. **Medical Decision Diabetes** - Glucose management
```python
from stochopt.problems.medical_decision_diabetes import DiabetesConfig, DiabetesModel
```

### 7. **Two Newsvendor** - Multi-agent coordination
```python
from stochopt.problems.two_newsvendor import TwoNewsvendorConfig, TwoNewsvendorModel
```

### 8. **Asset Selling** - Optimal liquidation
```python
from stochopt.problems.asset_selling import AssetSellingConfig, AssetSellingModel
```

### 9. **Energy Storage** - Battery management
```python
from stochopt.problems.energy_storage import EnergyStorageConfig, EnergyStorageModel
```

---

## JAX Transformations

### JIT Compilation (Speed up)
```python
@jax.jit
def run_step(state, key, model, policy):
    key1, key2 = jax.random.split(key)
    decision = policy(None, state, key1, model)
    exog = model.sample_exogenous(key2, state, 0)
    reward = model.reward(state, decision, exog)
    next_state = model.transition(state, decision, exog)
    return next_state, reward

# Much faster!
next_state, reward = run_step(state, key, model, policy)
```

### Vectorization (Parallel simulations)
```python
# Run 100 simulations in parallel
batch_size = 100
keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
states = jax.vmap(model.init_state)(keys)

# Batched step function
def batched_step(states, keys):
    return jax.vmap(run_step, in_axes=(0, 0, None, None))(
        states, keys, model, policy
    )

# Run in parallel
next_states, rewards = batched_step(states, keys)
print(f"Batch rewards: {rewards}")  # 100 rewards at once!
```

### Automatic Differentiation (Gradient-based learning)
```python
from stochopt.problems.clinical_trials import LinearDosePolicy

# Policy with parameters
policy = LinearDosePolicy(weight=1.0)

def loss_fn(params, state, key, model):
    decision = policy(params, state, key, model)
    # ... compute loss
    return loss_value

# Compute gradients
grad_fn = jax.grad(loss_fn)
grads = grad_fn(params, state, key, model)

# Update parameters
params = jax.tree_map(lambda p, g: p - 0.01 * g, params, grads)
```

---

## Common Patterns

### Running Multiple Episodes
```python
def run_episode(key, model, policy, horizon=30):
    state = model.init_state(key)
    total_reward = 0.0

    for t in range(horizon):
        key, k1, k2 = jax.random.split(key, 3)
        decision = policy(None, state, k1, model)
        exog = model.sample_exogenous(k2, state, t)
        reward = model.reward(state, decision, exog)
        total_reward += reward
        state = model.transition(state, decision, exog)

    return total_reward

# Run 10 episodes
keys = jax.random.split(jax.random.PRNGKey(0), 10)
rewards = [run_episode(k, model, policy) for k in keys]
print(f"Average reward: {jnp.mean(jnp.array(rewards)):.2f}")
```

### Comparing Policies
```python
from stochopt.problems.blood_management import GreedyPolicy, FIFOPolicy, RandomPolicy

policies = {
    "Greedy": GreedyPolicy(),
    "FIFO": FIFOPolicy(),
    "Random": RandomPolicy(),
}

for name, policy in policies.items():
    key = jax.random.PRNGKey(42)
    reward = run_episode(key, model, policy)
    print(f"{name:10s}: {reward:.2f}")
```

---

## Troubleshooting

### "Module not found" error
```bash
# Make sure you installed in development mode
pip install -e .
```

### "JAX not found" error
```bash
# Install JAX
pip install jax jaxlib
```

### "TracerBoolConversionError"
This means you're using Python `if` with JAX arrays. Use `jnp.where()` instead:
```python
# ❌ Bad (causes error in JIT)
if x > 0:
    y = x
else:
    y = 0

# ✅ Good (works with JIT)
y = jnp.where(x > 0, x, 0)
```

### Tests failing
```bash
# Run with verbose output to see errors
pytest tests/test_blood_management.py -v --tb=short

# Run specific test
pytest tests/test_blood_management.py::test_config_default_values -v
```

---

## Next Steps

1. **Explore examples**: Check `examples/` for more detailed usage
2. **Read the docs**: See `README.md` for comprehensive documentation
3. **Run tests**: Explore `tests/` to understand problem behavior
4. **Modify policies**: Create your own policy classes
5. **Experiment**: Try different configurations and parameters

---

## Quick Reference

### Model API
- `init_state(key)` - Initialize state
- `transition(state, decision, exog)` - State dynamics
- `reward(state, decision, exog)` - Compute reward
- `sample_exogenous(key, state, time)` - Sample random events

### Policy API
- `__call__(params, state, key, model)` - Compute decision

### Common Imports
```python
import jax
import jax.numpy as jnp
from stochopt.problems.<problem> import <Problem>Config, <Problem>Model
```

---

## Support

- **Issues**: Check GitHub issues or create new one
- **Documentation**: See `README.md` and `REPOSITORY_MODERNIZATION.md`
- **Legacy Code**: See `legacy/` for old implementations (reference only)

---

**Happy Optimizing! 🚀**
