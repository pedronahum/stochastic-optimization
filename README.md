# Stochastic Optimization Library

A modern JAX-native library for sequential decision-making problems under uncertainty.

**Princeton University - Castle Lab**

---

## 🚀 Modern JAX-Native Implementation (2025)

This library has been completely modernized with:
- ✅ **JAX-native implementations** - Full GPU/TPU acceleration support
- ✅ **JIT compilation** - Optimized performance with XLA
- ✅ **Automatic differentiation** - End-to-end gradient computation
- ✅ **Type safety** - 100% mypy strict compliance
- ✅ **Comprehensive testing** - 230+ tests with 100% pass rate
- ✅ **Functional programming** - Immutable state, pure functions

---

## 📦 Installation

### Requirements
- Python 3.10+
- JAX 0.4+
- jaxtyping
- chex
- numpy
- pytest (for development)

### Install from source

```bash
git clone https://github.com/pedronahum/stochastic-optimization.git
cd stochastic-optimization
pip install -e .
```

### Dependencies

```bash
pip install jax jaxlib jaxtyping chex numpy pytest
```

---

## 📓 Interactive Notebooks Gallery

Explore problems interactively with our Jupyter notebooks - all runnable in Google Colab with one click:

| Problem | Notebook | Description |
|---------|----------|-------------|
| **Blood Management** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/blood_management.ipynb) | Blood bank inventory with 8 types, aging, substitution rules |
| **Clinical Trials** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/clinical_trials.ipynb) | Adaptive dose optimization for patient outcomes |
| **SSP Dynamic** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/ssp_dynamic.ipynb) | Shortest path with lookahead and cost estimation |
| **SSP Static** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/ssp_static.ipynb) | Classical shortest path with percentile risk |
| **Market Planning** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/adaptive_market_planning.ipynb) | Dynamic pricing and demand forecasting |
| **Diabetes Management** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/medical_decision_diabetes.ipynb) | Glucose-insulin dynamics for diabetes treatment |
| **Two Newsvendor** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/two_newsvendor.ipynb) | Multi-agent inventory coordination |
| **Asset Selling** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/asset_selling.ipynb) | Optimal liquidation with price dynamics |
| **Energy Storage** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/energy_storage.ipynb) | Battery management with price arbitrage |

See [notebooks/README.md](notebooks/README.md) for detailed descriptions and learning paths.

---

## 🎯 Problem Domains

The library includes 9 fully-implemented stochastic optimization problems:

### 1. Clinical Trials
**Path**: `problems/clinical_trials/`

Adaptive dose optimization for clinical trial design with patient outcomes and safety constraints.

**Features**: Dose-response modeling, adaptive trial design, patient safety constraints
**Tests**: 22/22 passing
**Policies**: LinearDosePolicy

```python
from problems.clinical_trials import Config, ClinicalTrialsModel

config = Config(horizon=50, mu=0.1, sigma=0.5)
model = ClinicalTrialsModel(config)
```

---

### 2. Stochastic Shortest Path - Dynamic
**Path**: `problems/ssp_dynamic/`

Multi-step lookahead path planning with time-varying costs and running average estimation.

**Features**: Dynamic programming, risk-sensitive policies, cost estimation
**Tests**: 39/39 passing
**Policies**: LookaheadPolicy, GreedyLookaheadPolicy, RandomPolicy

```python
from problems.ssp_dynamic import SSPDynamicConfig, SSPDynamicModel

config = SSPDynamicConfig(n_nodes=10, horizon=15)
model = SSPDynamicModel(config)
```

---

### 3. Stochastic Shortest Path - Static
**Path**: `problems/ssp_static/`

Classical shortest path with static graph structure and percentile-based risk measures.

**Features**: Bellman-Ford algorithm, percentile optimization, risk sensitivity
**Tests**: 34/34 passing
**Policies**: ShortestPathPolicy, RandomPolicy

```python
from problems.ssp_static import SSPStaticConfig, SSPStaticModel

config = SSPStaticConfig(n_nodes=8, edge_prob=0.3)
model = SSPStaticModel(config)
```

---

### 4. Adaptive Market Planning
**Path**: `problems/adaptive_market_planning/`

Dynamic pricing and demand forecasting with market adaptation.

**Features**: Price optimization, demand modeling, market dynamics
**Tests**: 29/29 passing
**Policies**: NeuralPolicy, heuristic policies

```python
from problems.adaptive_market_planning import AdaptiveMarketPlanningConfig, AdaptiveMarketPlanningModel

config = AdaptiveMarketPlanningConfig(price=1.5, cost=0.8, demand_mean=100.0)
model = AdaptiveMarketPlanningModel(config)
```

---

### 5. Medical Decision - Diabetes
**Path**: `problems/medical_decision_diabetes/`

Glucose-insulin dynamics for diabetes management with meal planning and health monitoring.

**Features**: Physiological modeling, treatment policies, health state tracking
**Tests**: 25/25 passing
**Policies**: Multiple treatment strategies

```python
from problems.medical_decision_diabetes import MedicalDecisionDiabetesConfig, MedicalDecisionDiabetesModel

config = MedicalDecisionDiabetesConfig(n_drugs=5, initial_mu=0.5)
model = MedicalDecisionDiabetesModel(config)
```

---

### 6. Two Newsvendor
**Path**: `problems/two_newsvendor/`

Multi-agent inventory coordination with demand uncertainty and allocation strategies.

**Features**: Coordination mechanisms, inventory allocation, demand forecasting
**Tests**: 37/37 passing
**Policies**: NewsvendorFieldPolicy, NeuralPolicies, coordination strategies

```python
from problems.two_newsvendor import TwoNewsvendorConfig, TwoNewsvendorFieldModel

config = TwoNewsvendorConfig(demand_lower=0.0, demand_upper=100.0)
model = TwoNewsvendorFieldModel(config)
```

---

### 7. Asset Selling
**Path**: `problems/asset_selling/`

Optimal asset liquidation with price dynamics and market volatility.

**Features**: Price processes, optimal stopping, market timing
**Tests**: 23/23 passing
**Policies**: Threshold policies, time-based strategies

```python
from problems.asset_selling import AssetSellingConfig, AssetSellingModel

config = AssetSellingConfig(initial_price=100.0, up_step=2.0, down_step=-2.0)
model = AssetSellingModel(config)
```

---

### 8. Energy Storage
**Path**: `problems/energy_storage/`

Battery management with price arbitrage and capacity constraints.

**Features**: Battery dynamics, price-based charging, capacity management
**Tests**: 20/20 passing
**Policies**: Price-based policies, threshold strategies

```python
from problems.energy_storage import EnergyStorageConfig, EnergyStorageModel

config = EnergyStorageConfig(capacity=100.0, initial_energy=50.0)
model = EnergyStorageModel(config)
```

---

### 9. Blood Management
**Path**: `problems/blood_management/`

Blood bank inventory optimization with age-dependent inventory, blood type substitution, and stochastic demand.

**Features**: 8 blood types with substitution rules, FIFO aging, urgent/elective demands, surge events
**Tests**: 21/21 passing
**Policies**: GreedyPolicy, FIFOPolicy, RandomPolicy

```python
from problems.blood_management import BloodManagementConfig, BloodManagementModel

config = BloodManagementConfig(max_age=5, surge_prob=0.1)
model = BloodManagementModel(config)
```

---

## 🏗️ Architecture

All problems follow a consistent API:

### Model Interface
```python
class Model:
    def init_state(self, key: PRNGKey) -> State:
        """Initialize state."""

    def transition(self, state: State, decision: Decision, exog: ExogenousInfo) -> State:
        """State transition dynamics."""

    def reward(self, state: State, decision: Decision, exog: ExogenousInfo) -> Reward:
        """Reward/cost function."""

    def sample_exogenous(self, key: PRNGKey, state: State, time: int) -> ExogenousInfo:
        """Sample random exogenous information."""
```

### Policy Interface
```python
class Policy:
    def __call__(self, params: PyTree, state: State, key: PRNGKey, model: Model) -> Decision:
        """Compute decision given current state."""
```

---

## 📊 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific problem tests:
```bash
pytest tests/test_blood_management.py -v
```

Check type safety:
```bash
mypy problems/ --strict
```

**Current Status**:
- ✅ 230/230 tests passing
- ✅ 100% mypy strict compliance
- ✅ 28 source files

---

## 🔬 Example Usage

### Running a Full Episode

```python
import jax
import jax.numpy as jnp
from stochopt.problems.blood_management import (
    BloodManagementConfig,
    BloodManagementModel,
    GreedyPolicy
)

# Initialize
config = BloodManagementConfig(max_age=5)
model = BloodManagementModel(config)
policy = GreedyPolicy()

key = jax.random.PRNGKey(42)
state = model.init_state(key)

# Run episode
total_reward = 0.0
for t in range(30):
    # Get decision from policy
    key, subkey = jax.random.split(key)
    decision = policy(None, state, subkey, model)

    # Sample exogenous information
    key, subkey = jax.random.split(key)
    exog = model.sample_exogenous(subkey, state, t)

    # Get reward and transition
    reward = model.reward(state, decision, exog)
    total_reward += float(reward)

    state = model.transition(state, decision, exog)

print(f"Total reward: {total_reward:.2f}")
```

### Using JAX Transformations

```python
# JIT compilation
@jax.jit
def rollout_step(state, key, model, policy):
    key1, key2 = jax.random.split(key)
    decision = policy(None, state, key1, model)
    exog = model.sample_exogenous(key2, state, 0)
    reward = model.reward(state, decision, exog)
    next_state = model.transition(state, decision, exog)
    return next_state, reward

# Vectorization with vmap
batch_rollout = jax.vmap(rollout_step, in_axes=(0, 0, None, None))
```

---

## 📈 Modernization Achievements (2024)

### From NumPy to JAX
- **Before**: NumPy-based implementations with limited performance
- **After**: JAX-native with GPU/TPU support and automatic differentiation

### Type Safety
- **Before**: Minimal type hints, runtime errors
- **After**: 100% mypy strict compliance, compile-time error detection

### Testing
- **Before**: Limited test coverage
- **After**: 230+ comprehensive tests, 100% pass rate

### Performance
- **Before**: Python loops, CPU-only
- **After**: JIT-compiled, vectorized, hardware-accelerated

### Code Quality
- **Before**: Inconsistent APIs across problems
- **After**: Unified API, consistent patterns, extensive documentation

---

## 📚 Documentation

Each problem directory contains:
- `model.py` - Core dynamics implementation
- `policy.py` - Policy implementations
- `__init__.py` - Public API exports

Test files located in `tests/`:
- `test_<problem>.py` - Comprehensive test suites

---

## 🗂️ Legacy Code

Original implementations are archived in the `legacy/` directory for historical reference. See [legacy/README.md](legacy/README.md) for details.

**All new development should use the JAX-native implementations in `stochopt/`.**

---

## 👥 Contributors

### Original Implementation (Pre-2024)
- Donghun Lee: d.lee@princeton.edu (dhl)
- Grace Lee: gylee@princeton.edu (or gayeonglee95@gmail.com) (gl)
- Joy Hii: jhii@princeton.edu (jh)
- Robert Raveanu: rraveanu@princeton.edu (rr)
- Raluca Cobzaru: rcobzaru@princeton.edu
- Juilana Nascimento: jnascime@princeton.edu (jn)
- And others (agraur, ckn)

### JAX Migration (2024)
- Complete modernization to JAX-native implementation
- 100% test coverage and type safety
- Performance optimization and API unification

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🔗 Citation

If you use this library in your research, please cite:

```bibtex
@software{stochastic_optimization_jax,
  title = {Stochastic Optimization Library (JAX)},
  author = {Castle Lab, Princeton University},
  year = {2024},
  url = {https://github.com/pedronahum/stochastic-optimization}
}
```

---

## 🎓 Princeton University - Castle Lab

Sequential Decision Problem Modeling Library
**Castle Lab**
Princeton University

For more information, visit [Castle Lab](https://castlelab.princeton.edu/)
