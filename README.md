# Stochastic Optimization Library

A modern **JAX-native** library of sequential decision-making problems under
uncertainty — faithful reimplementations of the classic problems from Warren
Powell's [stochastic-optimization](https://github.com/wbpowell328/stochastic-optimization)
course (Princeton, Castle Lab), modernized for GPU/TPU, autodiff, and type safety.

---

## ✨ Highlights

- ✅ **JAX-native** — JIT-compiled, vectorizable (`vmap`), GPU/TPU-ready
- ✅ **9 problems**, each with a unified `model` / `policy` API
- ✅ **Faithful to the originals** — the new code is benchmarked for *parity*
  against Powell's reference implementations in `legacy/` (see
  [`benchmarks/PARITY.md`](benchmarks/PARITY.md)); 8/9 match exactly or
  analytically, blood management matches the reference LP to ~1e-2
- ✅ **Typed** — `flax.struct` configs; mypy strict-clean on `core/` + `problems/`
- ✅ **Tested** — 212 tests; ruff-clean

---

## 📦 Installation

Requires **Python ≥ 3.11**. Core dependencies: `jax`, `jaxlib`, `flax`,
`optax`, `chex`, `jaxtyping`, and `ott-jax` (entropic OT, used by blood
management).

```bash
git clone https://github.com/pedronahum/stochastic-optimization.git
cd stochastic-optimization
pip install -e .            # CPU
# GPU: install the matching jaxlib, e.g. pip install "jax[cuda13]"
```

Optional extras: `.[viz]` (matplotlib + networkx for the notebooks), `.[dev]`
(ruff, mypy, pytest, …).

> **Note:** on unified-memory GPUs, run with `JAX_PLATFORMS=cpu` for the small
> models/tests to avoid JAX over-preallocating the shared memory pool.

---

## 🚀 Quickstart

```python
import jax
from problems.asset_selling import AssetSellingConfig, AssetSellingModel, SellLowPolicy

model = AssetSellingModel(AssetSellingConfig(initial_price=100.0, up_step=2.0, down_step=-2.0))
policy = SellLowPolicy(threshold=95.0)

key = jax.random.PRNGKey(0)
state = model.init_state(key)
decision = policy(None, state, key)          # sell / hold
exog = model.sample_exogenous(key, state, 0)
reward = model.reward(state, decision, exog)
state = model.transition(state, decision, exog)
```

Every problem follows the same shape:

```python
class Model:
    def init_state(self, key) -> State: ...
    def transition(self, state, decision, exog) -> State: ...
    def reward(self, state, decision, exog) -> Reward: ...
    def sample_exogenous(self, key, state, time) -> ExogenousInfo: ...
```

Most policies are called as `policy(params, state, key)`. A few problems whose
decision depends on the realised exogenous data or the model take extra
arguments — e.g. blood management's `OTAllocationPolicy(model)` is called
`policy(None, state, key, demand)`.

---

## 📓 Notebooks

One-click runnable in Google Colab (each notebook clones this repo and installs
deps):

> ⚠️ **Colab: restart the runtime once after the setup cell.** The first cell
> installs the package, which upgrades Colab's preinstalled `jax`/`jaxlib`. Colab
> keeps the *old* jax loaded in the running kernel, so the next cell can fail with
> a jax/jaxlib error (e.g. `jit() missing 1 required positional argument: 'fun'`).
> Just do **Runtime → Restart session** and run the cells again — the setup cell
> is safe to re-run, and after the restart everything imports cleanly.

| Problem | Colab |
|---|---|
| Blood Management | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/blood_management.ipynb) |
| Clinical Trials | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/clinical_trials.ipynb) |
| SSP Dynamic | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/ssp_dynamic.ipynb) |
| SSP Static | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/ssp_static.ipynb) |
| Adaptive Market Planning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/adaptive_market_planning.ipynb) |
| Medical Decision (Diabetes) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/medical_decision_diabetes.ipynb) |
| Two Newsvendor | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/two_newsvendor.ipynb) |
| Asset Selling | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/asset_selling.ipynb) |
| Energy Storage | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/energy_storage.ipynb) |

Training scripts live in [`examples/`](examples/) (`train_asset_selling.py`,
`train_clinical.py`, `train_energy_storage.py`).

---

## 🎯 Problem Domains

Nine problems under `problems/`. Each has `model.py`, `policy.py`, `__init__.py`.

### Adaptive Market Planning — `problems/adaptive_market_planning/`
Newsvendor order-quantity learning by stochastic gradient ascent; converges to
the analytic optimum `q* = μ·ln(p/c)` for exponential demand.
Policies: `HarmonicStepPolicy`, `KestenStepPolicy`, `ConstantStepPolicy`,
`AdaptiveStepPolicy`, `NeuralStepPolicy`.
```python
from problems.adaptive_market_planning import AdaptiveMarketPlanningConfig, AdaptiveMarketPlanningModel
model = AdaptiveMarketPlanningModel(AdaptiveMarketPlanningConfig(price=1.5, cost=0.8, demand_mean=100.0))
```

### Asset Selling — `problems/asset_selling/`
Optimal stopping: when to sell an asset whose price follows a Markov-modulated
random walk. Policies: `SellLowPolicy`, `HighLowPolicy`, `ExpectedValuePolicy`,
`LinearThresholdPolicy`, `NeuralPolicy`, `AlwaysHold/AlwaysSell`.
```python
from problems.asset_selling import AssetSellingConfig, AssetSellingModel
model = AssetSellingModel(AssetSellingConfig(initial_price=100.0, up_step=2.0, down_step=-2.0))
```

### Blood Management — `problems/blood_management/`
Blood-bank inventory: 8 types with ABO/Rh substitution, FIFO aging, urgent/
elective demand, surges. The per-period allocation is the original min-cost-flow
LP, solved JAX-natively by **entropic OT** (`OTAllocationPolicy`, via `ott-jax`).
The full **ADP** (Powell SPAR/CAVE — learned concave value-of-holding) is in
`adp.py` (`train_spar`, `ADPPolicy`). Baselines: `GreedyPolicy`, `FIFOPolicy`,
`RandomPolicy`.
```python
from problems.blood_management import BloodManagementConfig, BloodManagementModel, OTAllocationPolicy
model = BloodManagementModel(BloodManagementConfig(max_age=5, surge_prob=0.1))
```

### Clinical Trials — `problems/clinical_trials/`
Drug-program enrollment MDP: decide how many patients to enroll and whether to
continue or stop (declaring success/failure), tracking a Beta(success, failure)
belief. State `[potential_pop, success, failure, l_response]`.
Policies: `StoppingPolicy`, `FixedEnrollPolicy`.
```python
from problems.clinical_trials import Config, ClinicalTrialsModel, StoppingPolicy
model = ClinicalTrialsModel(Config())   # success_rev, program/patient cost, stop thresholds, …
```

### Energy Storage — `problems/energy_storage/`
Battery arbitrage against an exogenous price series (historically PJM RT LMP):
buy/sell to maximise `price·(η·sell − buy)`. Policy: `BuyLowSellHighPolicy`
(grid-searchable via `grid_search`), `AlwaysHoldPolicy`.
```python
import jax.numpy as jnp
from problems.energy_storage import EnergyStorageConfig, EnergyStorageModel
model = EnergyStorageModel(EnergyStorageConfig(eta=0.9, capacity=1.0), prices=jnp.array([20., 50., 15.]))
```

### Medical Decision — Diabetes — `problems/medical_decision_diabetes/`
Bayesian **multi-armed bandit** for choosing among drugs, with conjugate
Normal belief updates. Policies: `UCBPolicy`, `IntervalEstimationPolicy`,
`ThompsonSamplingPolicy`, `EpsilonGreedyPolicy`, `PureExploration/Exploitation`.
```python
from problems.medical_decision_diabetes import MedicalDecisionDiabetesConfig, MedicalDecisionDiabetesModel
model = MedicalDecisionDiabetesModel(MedicalDecisionDiabetesConfig(n_drugs=5))
```

### Stochastic Shortest Path — Dynamic — `problems/ssp_dynamic/`
Risk-sensitive lookahead routing on a graph with stochastic edge costs; the
lookahead reproduces Dijkstra shortest paths. Policies: `LookaheadPolicy`,
`GreedyLookaheadPolicy`, `RandomPolicy`.
```python
from problems.ssp_dynamic import SSPDynamicConfig, SSPDynamicModel
model = SSPDynamicModel(SSPDynamicConfig(n_nodes=10, horizon=15))
```

### Stochastic Shortest Path — Static — `problems/ssp_static/`
Shortest path with online value-function learning. Policies: `GreedyPolicy`,
`EpsilonGreedyPolicy`, `BellmanGreedyPolicy`, `RandomPolicy`.
```python
from problems.ssp_static import SSPStaticConfig, SSPStaticModel
model = SSPStaticModel(SSPStaticConfig(n_nodes=8, edge_prob=0.3))
```

### Two Newsvendor — `problems/two_newsvendor/`
Two-agent (field/central) newsvendor coordination with biased demand estimates.
Policies: `NewsvendorFieldPolicy`, `BiasAdjustedFieldPolicy`,
`NewsvendorCentralPolicy`, `BiasAdjustedCentralPolicy`, `Neural*`,
`AlwaysAllocateRequestedPolicy`.
```python
from problems.two_newsvendor import TwoNewsvendorConfig, TwoNewsvendorFieldModel
model = TwoNewsvendorFieldModel(TwoNewsvendorConfig(demand_lower=0.0, demand_upper=100.0))
```

---

## ✅ Parity with the originals — `benchmarks/`

The original Powell implementations are archived in `legacy/old_problems/`. The
benchmark suite checks the new JAX code against them rather than trusting it
blindly:

```bash
JAX_PLATFORMS=cpu python benchmarks/parity.py        # deterministic / analytical parity, all 9
JAX_PLATFORMS=cpu MPLBACKEND=Agg python benchmarks/run_notebooks.py   # run all notebooks headless
```

`benchmarks/PARITY.md` documents the methodology and results: 8/9 problems match
the originals' `transition`/`reward`/closed-form policies exactly (or hit the
known analytical optimum); blood management's allocation matches the reference
LP to ~1e-2 (entropic OT), and its ADP value functions reproduce the SPAR
behaviour (concave, and beating the myopic allocation).

---

## 🧪 Testing & quality

```bash
JAX_PLATFORMS=cpu pytest tests/ -q            # 212 tests
mypy --python-executable .venv/bin/python core problems   # strict-clean
ruff check .
```

mypy must see the deps installed (jax/jaxtyping ship `py.typed`), otherwise the
shape-alias types collapse to `Any`; point it at the env's interpreter as above.

---

## 🗂️ Repository layout

```
core/        # shared simulator / protocols
problems/    # the 9 JAX-native problems (model.py, policy.py)
examples/    # training / demo scripts
notebooks/   # Colab-runnable notebooks
benchmarks/  # parity vs. the originals + notebook runner + PARITY.md
legacy/      # archived original Powell implementations (reference only)
```

---

## 👥 Credits

The problems originate from Warren Powell's *Sequential Decision Analytics*
course code (Princeton, Castle Lab) — original authors include Donghun Lee,
Grace Lee, Joy Hii, Robert Raveanu, Raluca Cobzaru, Juliana Nascimento, and
others. This repository is a JAX-native reimplementation that has been
benchmarked for parity against those originals.

## 📄 License

See [LICENSE](LICENSE).
