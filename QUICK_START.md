# Quick Start

Get going with the JAX-native stochastic-optimization library in a few minutes.
See [README.md](README.md) for the full overview and per-problem details.

---

## Install

Requires **Python ≥ 3.11**.

```bash
git clone https://github.com/pedronahum/stochastic-optimization.git
cd stochastic-optimization
pip install -e .            # CPU; for GPU also: pip install "jax[cuda13]"
```

Core deps (`jax`, `jaxlib`, `flax`, `optax`, `chex`, `jaxtyping`, `ott-jax`) are
installed by `pip install -e .`. Extras: `.[viz]` (matplotlib + networkx for the
notebooks), `.[dev]` (ruff, mypy, pytest).

> On a unified-memory GPU, prefix commands with `JAX_PLATFORMS=cpu` for the small
> models/tests to avoid JAX over-preallocating shared memory.

---

## Your first problem

Every problem exposes the same `model` API: `init_state`, `transition`,
`reward`, `sample_exogenous`.

```python
import jax
from problems.asset_selling import AssetSellingConfig, AssetSellingModel, SellLowPolicy

model = AssetSellingModel(AssetSellingConfig(initial_price=100.0, up_step=2.0, down_step=-2.0))
policy = SellLowPolicy(threshold=95.0)   # sell once price drops below 95

key = jax.random.PRNGKey(0)
state = model.init_state(key)

total = 0.0
for t in range(20):
    key, k1, k2 = jax.random.split(key, 3)
    decision = policy(None, state, k1)          # most policies: (params, state, key)
    exog = model.sample_exogenous(k2, state, t)
    total += float(model.reward(state, decision, exog))
    state = model.transition(state, decision, exog)

print(f"Total reward: {total:.2f}")
```

Policy call conventions: most policies are `policy(params, state, key)`. A few
take extra context — e.g. blood management's `OTAllocationPolicy(model)` is
`policy(None, state, key, demand)`, and the SSP policies take the `model`.

---

## The nine problems

```python
from problems.adaptive_market_planning import AdaptiveMarketPlanningConfig, AdaptiveMarketPlanningModel
from problems.asset_selling            import AssetSellingConfig, AssetSellingModel
from problems.blood_management         import BloodManagementConfig, BloodManagementModel
from problems.clinical_trials          import Config as ClinicalConfig, ClinicalTrialsModel
from problems.energy_storage           import EnergyStorageConfig, EnergyStorageModel
from problems.medical_decision_diabetes import MedicalDecisionDiabetesConfig, MedicalDecisionDiabetesModel
from problems.ssp_dynamic              import SSPDynamicConfig, SSPDynamicModel
from problems.ssp_static               import SSPStaticConfig, SSPStaticModel
from problems.two_newsvendor           import TwoNewsvendorConfig, TwoNewsvendorFieldModel
```

See the README for what each one models and which policies it ships.

---

## JAX transforms

```python
import jax

# JIT a single step (the model methods are already jitted internally)
@jax.jit
def step(state, key):
    k1, k2 = jax.random.split(key)
    decision = policy(None, state, k1)
    exog = model.sample_exogenous(k2, state, 0)
    return model.transition(state, decision, exog), model.reward(state, decision, exog)

# Vectorise N independent initial states with vmap
keys = jax.random.split(jax.random.PRNGKey(0), 100)
states = jax.vmap(model.init_state)(keys)
```

Gradient-based training is shown end-to-end in `examples/` (e.g.
`train_asset_selling.py` trains a neural policy with `optax` via REINFORCE).

---

## Run the tests / checks

```bash
JAX_PLATFORMS=cpu pytest tests/ -q                 # 212 tests
ruff check .
mypy --python-executable .venv/bin/python core problems   # strict-clean
```

mypy must see the deps installed (jax/jaxtyping ship `py.typed`); point it at the
interpreter of the env where the package is installed, as above.

---

## Compare against the originals

The new code is benchmarked for *parity* against Powell's reference
implementations in `legacy/`:

```bash
JAX_PLATFORMS=cpu python benchmarks/parity.py                       # all 9 problems
JAX_PLATFORMS=cpu MPLBACKEND=Agg python benchmarks/run_notebooks.py # run notebooks headless
```

See [benchmarks/PARITY.md](benchmarks/PARITY.md) for the results.

---

## Troubleshooting

- **`ModuleNotFoundError`** — run `pip install -e .` (or run from the repo root).
- **`TracerBoolConversionError`** — you used a Python `if` on a traced JAX array
  inside a jitted function; use `jnp.where(cond, a, b)` instead.
- **GPU out-of-memory on small models** — run with `JAX_PLATFORMS=cpu`.

---

## Next steps

- `README.md` — full library overview and per-problem reference
- `examples/` — runnable training scripts
- `notebooks/` — Colab-runnable tutorials (see `notebooks/README.md`)
- `benchmarks/PARITY.md` — how the ports are validated against the originals
