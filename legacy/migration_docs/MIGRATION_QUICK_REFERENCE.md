# Migration Priority Quick Reference

## TL;DR

**Migrate AdaptiveMarketPlanning next.** It's the smallest (543 LOC), simplest (2 state dims, 1 decision var), and easiest to JAX-ify (pure math, no external dependencies).

Timeline: 1-2 days. Then use it as a template for the other 4 problems.

---

## The Five Problems at a Glance

```
Problem                    LOC    State Dims  Decision Vars  Complexity  Time to Migrate
─────────────────────────────────────────────────────────────────────────────────────
1. AdaptiveMarketPlanning  543    2          1              ⭐           1-2 days   ← START HERE
2. MedicalDecisionDiabetes 721    15         5 (categorical) ⭐⭐          3-4 days
3. SSP_Static              683    1+dict     1 (categorical) ⭐⭐          4-5 days
4. SSP_Dynamic             598    1+dict     1 (categorical) ⭐⭐⭐        5-6 days
5. BloodManagement         2167   160-200+   160+ discrete   ⭐⭐⭐⭐⭐    10-14 days
```

---

## Why Start with AdaptiveMarketPlanning?

### Pros (Why It's Best)
- ✅ **Smallest**: 543 lines vs 2,167 for BloodManagement
- ✅ **Simplest State**: Just 2 values (quantity, counter)
- ✅ **No Dependencies**: Pure math, no LP solvers or networkx
- ✅ **Quick Win**: Achievable in 1-2 days
- ✅ **Great Template**: Shows all key JAX patterns
- ✅ **Low Risk**: Fewer things to go wrong
- ✅ **Clear Speedup**: Small enough to benchmark
- ✅ **Momentum**: Success will motivate next migrations

### Cons (Minor)
- None significant. It's genuinely the best choice.

---

## Why NOT the Others (Yet)

| Problem | Why Wait |
|---------|----------|
| **MedicalDecisionDiabetes** | Wait for AdaptiveMarketPlanning patterns first. Good follow-up though (Tier 2). |
| **SSP_Static** | Wait for MedicalDecisionDiabetes. Bellman iteration patterns then clear. |
| **SSP_Dynamic** | More complex than Static. Do Static first. |
| **BloodManagement** | CVXOPT LP solver + 160+ dims + advanced algorithms. Do last after learning from simpler ones. |

---

## Key Numbers

### Size Comparison
- AdaptiveMarketPlanning: 543 LOC
- MedicalDecisionDiabetes: 721 LOC (+33%)
- SSP_Static: 683 LOC (+26%)
- SSP_Dynamic: 598 LOC (+10%)
- BloodManagement: 2,167 LOC (+299%)

### State Space Complexity
- AdaptiveMarketPlanning: 2 dimensions
- MedicalDecisionDiabetes: 15 dimensions (7.5x more)
- SSP_Static: 1-10 dimensions
- SSP_Dynamic: 1 dimension (+ context)
- BloodManagement: 160-200+ dimensions (80-100x more!)

### External Dependencies
- AdaptiveMarketPlanning: None
- MedicalDecisionDiabetes: None
- SSP_Static: networkx
- SSP_Dynamic: networkx
- BloodManagement: CVXOPT, networkx, Excel

---

## What You'll Learn from AdaptiveMarketPlanning

1. **JAX Basics**
   - jax.jit for compilation
   - jax.vmap for vectorization
   - jax.random with PRNG keys

2. **Type Safety**
   - jaxtyping Float[Array, "..."]
   - chex.dataclass for configs
   - chex assertions for shape/value checks

3. **Policy Patterns**
   - Multiple policy variants (harmonic, Kesten, constant)
   - Step size selection
   - Parameter-driven behavior

4. **Testing Patterns**
   - Property-based testing with hypothesis
   - Shape assertions with chex
   - Batched operation testing

5. **Documentation**
   - Docstrings with jaxtyping
   - Examples in docstrings
   - Type hints as documentation

---

## Migration Plan for AdaptiveMarketPlanning

### Phase 1: Setup (2 hours)
1. Create `stochopt/problems/adaptive_market_planning/`
2. Create `__init__.py`, `config.py`, `model.py`, `policy.py`
3. Copy test file structure from `energy_storage_model_jax.py` example

### Phase 2: Model (4 hours)
1. Implement `AdaptiveMarketPlanningConfig` as chex.dataclass
2. Implement model with:
   - `init_state(key)` -> Float[Array, "2"]
   - `transition(state, decision, exog)` -> Float[Array, "2"]
   - `reward(state, decision, exog)` -> Float[Array, ""]
   - `sample_exogenous(key, time)` -> exog dict
3. JIT-compile hot paths with @jax.jit

### Phase 3: Policies (2 hours)
1. Implement three policy variants:
   - Harmonic: `harmonic_rule()`
   - Kesten: `kesten_rule()`
   - Constant: `constant_rule()`
2. Each returns decision from policy_params and state

### Phase 4: Testing (3 hours)
1. Write unit tests for model initialization
2. Write tests for state transitions
3. Write tests for reward calculation
4. Property-based tests with hypothesis
5. Integration test: full trajectory simulation
6. Benchmark: compare JAX vs original numpy

### Phase 5: Documentation (2 hours)
1. Write docstrings with examples
2. Create example notebook
3. Add to library documentation
4. Create migration guide for next problems

**Total: ~13 hours spread over 1-2 days**

---

## Files to Create

```
stochopt/problems/adaptive_market_planning/
├── __init__.py                 # Exports model, policy, config
├── config.py                   # AdaptiveMarketPlanningConfig (chex.dataclass)
├── model.py                    # AdaptiveMarketPlanningModel
├── policy.py                   # Three policy variants
└── __pycache__/

tests/
└── unit/
    └── models/
        └── test_adaptive_market_planning.py

examples/
└── adaptive_market_planning_example.ipynb
```

---

## Key Code Patterns from the Example

### State Definition
```python
from jaxtyping import Float, Array

State = Float[Array, "2"]  # [order_quantity, counter]
Decision = Float[Array, "1"]  # [step_size]
Reward = Float[Array, ""]  # scalar

@chex.dataclass(frozen=True)
class AdaptiveMarketPlanningConfig:
    price: float = 1.0
    cost: float = 0.7
    initial_quantity: float = 50.0
```

### Model Structure
```python
class AdaptiveMarketPlanningModel:
    def __init__(self, config: AdaptiveMarketPlanningConfig):
        self.config = config
    
    def init_state(self, key: Key) -> State:
        return jnp.array([self.config.initial_quantity, 0.0])
    
    @partial(jax.jit, static_argnums=(0,))
    def transition(self, state: State, decision: Decision, exog: dict) -> State:
        # Pure function, JIT-compilable
        ...
        return new_state
```

### Policy Pattern
```python
class AdaptiveMarketPlanningPolicy:
    def __init__(self, config: AdaptiveMarketPlanningConfig, theta_step: float):
        self.config = config
        self.theta_step = theta_step
    
    def harmonic_rule(self, t: int) -> Decision:
        step = self.theta_step / (self.theta_step + t)
        return jnp.array([step])
    
    def kesten_rule(self, counter: int) -> Decision:
        step = self.theta_step / (self.theta_step + counter)
        return jnp.array([step])
```

### Testing Pattern
```python
import pytest
import jax
import chex

class TestAdaptiveMarketPlanningModel:
    def test_state_shape(self):
        model = AdaptiveMarketPlanningModel(AdaptiveMarketPlanningConfig())
        state = model.init_state(jax.random.PRNGKey(0))
        
        chex.assert_shape(state, (2,))
        chex.assert_rank(state, 1)
    
    def test_jit_compilation(self):
        model = AdaptiveMarketPlanningModel(AdaptiveMarketPlanningConfig())
        state = model.init_state(jax.random.PRNGKey(0))
        decision = jnp.array([0.01])
        
        jitted_fn = jax.jit(model.transition)
        next_state = jitted_fn(state, decision, {'demand': 100.0})
        
        chex.assert_equal_shape([state, next_state])
```

---

## Success Criteria

After migration, you should have:

1. ✅ JAX-native AdaptiveMarketPlanning model
2. ✅ All three policy variants implemented
3. ✅ >80% test coverage
4. ✅ Successful JIT compilation
5. ✅ Demonstrated speedup vs numpy (5-10x on CPU, 100x+ on GPU)
6. ✅ Complete docstrings with examples
7. ✅ Migration guide for next problems
8. ✅ Integration with stochopt package structure

---

## Estimated Timeline

| Task | Duration | Day |
|------|----------|-----|
| Setup directories & structure | 30 min | Day 1 AM |
| Implement model | 2 hours | Day 1 AM-PM |
| Implement three policies | 1 hour | Day 1 PM |
| Write tests | 2 hours | Day 1 PM-Evening |
| Fix issues, iterate | 1 hour | Day 1 Evening |
| Documentation & examples | 1 hour | Day 2 AM |
| Benchmark & optimize | 1 hour | Day 2 AM |
| Final review & commit | 30 min | Day 2 AM |

**Total: ~9 hours spread over 1-2 days**

---

## Next Steps After AdaptiveMarketPlanning

1. **Tier 2 (Days 3-8)**: MedicalDecisionDiabetes
   - Use AdaptiveMarketPlanning patterns
   - Learn policy vectorization with vmap
   - Practice Bayesian updates in JAX

2. **Tier 3 (Days 8-13)**: StochasticShortestPath_Static
   - Learn graph structures as pytrees
   - Practice Bellman iteration
   - Value function approximation

3. **Tier 4 (Days 13-19)**: StochasticShortestPath_Dynamic
   - More complex graph operations
   - Lookahead DP patterns
   - Horizon-based planning

4. **Tier 5 (Days 19+)**: BloodManagement
   - Largest project
   - LP solver integration research
   - Multi-dimensional state handling
   - Advanced VFA algorithms

---

## Resources

### Templates to Use
- `/Users/pedro/programming/python/stochastic-optimization/Modernization/base_protocols_jax.py`
- `/Users/pedro/programming/python/stochastic-optimization/Modernization/energy_storage_model_jax.py`

### Documentation
- `/Users/pedro/programming/python/stochastic-optimization/Modernization/JAX_MODERNIZATION_PLAN.md`
- `/Users/pedro/programming/python/stochastic-optimization/Modernization/QUICK_START.md`

### Full Analysis
- `/Users/pedro/programming/python/stochastic-optimization/LEGACY_PROBLEM_ANALYSIS.md`

---

## Questions?

Refer to the full analysis in `LEGACY_PROBLEM_ANALYSIS.md` for detailed information about each problem.

The Modernization directory has comprehensive guides for JAX patterns, type safety, testing, and deployment.

Good luck with the migration!
