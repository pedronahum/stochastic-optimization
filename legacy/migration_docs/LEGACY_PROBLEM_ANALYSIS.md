# Legacy Problem Directory Analysis & Migration Recommendation

## Executive Summary

This analysis evaluates the 5 remaining legacy problem directories for migration to JAX-native implementation. Based on code complexity, state space size, decision variables, and dynamics, **AdaptiveMarketPlanning is the clear priority for migration next**.

---

## Detailed Problem Analysis

### 1. BloodManagement

#### Overview
Complex blood inventory management system with multi-dimensional state and network-based optimization.

#### File Structure
- **Files**: 4 Python files + parameters/output
- **Total LOC**: 2,167 lines
- **Driver**: BloodManagementDriverScript.py (1,500+ lines)
- **Model**: BloodManagementModel.py (224 lines)
- **Policy**: BloodManagementPolicy.py (409 lines)
- **Network**: BloodManagementNetwork.py (121 lines)

#### State Space Complexity
**VERY HIGH - Highest complexity of all 5 problems**

```
State Variables:
- Blood Inventory: NUM_BLD_NODES dimensions (blood type × age combinations)
  - 4 blood types (O-, O+, B-, B+)
  - Multiple age categories (MAX_AGE = 42 days typical)
  - Result: ~160+ dimensional state space
- Demand tracking
- Current age tracking per blood type

Total State Dimension: 160-200+
```

#### Decision Variables
**HIGH - Complex discrete and continuous decisions**
```
Decisions:
- Hold vector: NUM_BLD_NODES dimensions (how much blood to hold at each node)
- Contribution value (objective term)
- Discrete constraints on blood type compatibility
- Substitution rules matrix for compatibility

Total Decision Dimension: 160+ continuous + discrete constraints
```

#### Problem Dynamics
**HIGHLY COMPLEX - Multi-stage, inventory-based system**
```
Key Features:
1. Blood aging mechanics - explicit age progression
2. Donation randomness - stochastic blood unit arrivals
3. Demand randomness - stochastic surgery demand by type
4. Network flow - complex constraint satisfaction
5. Value function approximation (VFA) with linear program solver
6. Parallel arc capacity constraints
7. Multi-dimensional substitution rules
```

#### Special Features & Challenges
1. **Linear Programming Integration**: Uses CVXOPT solver (cvxopt.solvers.lp)
   - Complex matrix construction for A, G, h constraints
   - Dual variable extraction for value iteration
   
2. **Advanced Algorithm**: Stochastic Shortest Paths with gradient-based updates
   - Projection algorithms for concavity maintenance
   - Adaptive step size selection (AdaGrad)
   - Value function approximation with multiple parallel arcs
   
3. **Network-based Representation**: Specialized Graph class
   - Blood nodes, demand nodes, hold nodes
   - Supersink node for aggregation
   - Parallel edges with capacity constraints
   
4. **Parameter Complexity**: Parameters.xlsx with extensive configuration
   - Blood types, ages, surgery types, substitution rules
   - Multiple penalty/bonus terms
   - Surge probability configurations
   
5. **Data Dependencies**: Multiple parameters from external files
   - Depends on numpy random state management
   - Excel parameter loading

#### Migration Difficulty: **VERY HIGH** ⭐⭐⭐⭐⭐
- Requires reimplementing LP solving (could use JAX solver libraries)
- Complex network data structures need pytree registration
- Intricate projection algorithms
- Extensive state dimension requires careful JAX array handling

---

### 2. AdaptiveMarketPlanning (RECOMMENDED FIRST)

#### Overview
Simple price-learning problem where order quantity adapts based on observed demand and price-cost difference.

#### File Structure
- **Files**: 3-4 Python files + parameters
- **Total LOC**: 543 lines (smallest among all 5)
- **Model Variants**: 2 (base + parametric)
- **Driver**: AdaptiveMarketPlanningDriverScript.py (84 lines)
- **Model**: AdaptiveMarketPlanningModel.py (130 lines)
- **Policy**: AdaptiveMarketPlanningPolicy.py (58 lines)
- **Parametric Model**: ParametricModel.py + ParametricModelDriverScript.py

#### State Space Complexity
**VERY LOW - Simplest of all 5 problems**

```
State Variables:
- Order quantity: single continuous scalar [0, ∞)
- Counter: single integer (counts derivative sign changes)

Total State Dimension: 2 (1 continuous + 1 integer)
Example: state = (order_quantity=100.0, counter=3)
```

#### Decision Variables
**MINIMAL - Single parameter**
```
Decisions:
- Step size: single continuous scalar [0, ∞)

Total Decision Dimension: 1
Example: decision = (step_size=0.01)
```

#### Problem Dynamics
**SIMPLE - Gradient-based stochastic optimization**

```
Dynamics:
1. Observe demand from exponential distribution
2. Compute gradient: (price - cost) if order < demand, else -cost
3. Update order_quantity via gradient descent: 
   new_q = max(0, old_q + step_size × gradient)
4. Count derivative sign changes (convergence metric)
5. Profit = min(quantity, demand) × price - quantity × cost
```

#### Special Features & Challenges
1. **Three Policy Variants**:
   - Harmonic rule: step_size / (step_size + t)
   - Kesten's rule: step_size / (step_size + counter)
   - Constant rule: fixed step_size
   
2. **Lightweight Integration**:
   - No external dependencies beyond numpy
   - No LP solvers needed
   - No complex graph structures
   - Simple math operations
   
3. **Learning Tracking**:
   - history of order quantities
   - derivative sign change detection
   
4. **Parametric Variant**:
   - Additional variant for parametric model testing
   - Similar simplicity

#### Migration Difficulty: **VERY LOW** ⭐
- Minimal state dimension (2 values)
- Single decision variable
- Pure mathematical operations (no LP, no networks)
- No external solver dependencies
- Straightforward to JAX array operations

---

### 3. MedicalDecisionDiabetes

#### Overview
Multi-armed bandit problem for diabetes treatment selection with Bayesian learning of medication efficacy.

#### File Structure
- **Files**: 2 Python files + parameters
- **Total LOC**: 721 lines
- **Model**: MedicalDecisionDiabetesModel.py (274 lines)
- **Policy**: MedicalDecisionDiabetesPolicy.py (86 lines)
- **Driver**: MedicalDecisionDiabetesDriverScript.py (361 lines)

#### State Space Complexity
**MODERATE - Multidimensional belief state**

```
State Variables:
Per drug (5 drugs: Metformin, Sensitizer, Secretagogue, AGI, Peptide):
- mu: posterior mean A1C reduction [mu_0, ∞)
- beta: posterior precision (inverse variance) [0, ∞)
- N: count of times drug administered [0, T]

Total State: 5 drugs × 3 parameters = 15 dimensions
Example: 5 × [mu_estimate, precision, count_used]
```

#### Decision Variables
**LOW - Categorical choice**
```
Decisions:
- Drug choice: discrete selection from {M, Sens, Secr, AGI, PA}
- Observation weight (implicit in update)

Total Decision Dimension: 1 categorical (5 options)
```

#### Problem Dynamics
**MODERATE - Bayesian belief update with observation model**

```
Dynamics:
1. Sample treatment outcome: W_x ~ N(μ_x, σ_w²)
   - Uses current posterior estimate of μ_x
   - Fixed measurement noise σ_w
   
2. Bayesian update: combine prior belief with new observation
   - Update precision: β_new = β_old + β_W
   - Update mean: μ_new = (β_old × μ_old + β_W × W) / β_new
   - Increment count: N_new = N_old + 1
   
3. Objective: maximize expected A1C reduction μ_x
```

#### Special Features & Challenges
1. **Bayesian Learning**:
   - Conjugate prior-likelihood (Normal with known variance)
   - Posterior update equations
   - Depends on prior parameters from Excel
   
2. **Multiple Policy Types**:
   - Upper Confidence Bound (UCB)
   - Interval Estimation (IE)
   - Pure Exploitation
   - Pure Exploration
   
3. **Parameter Configurations**:
   - Truth type selection (fixed_uniform, prior_uniform, known, normal)
   - Requires Excel parameter loading (MDDMparameters.xlsx)
   - Per-drug priors and truth parameters
   
4. **Stochastic Observations**:
   - Normal random observation model
   - Measurement noise parameter σ_w

#### Migration Difficulty: **MODERATE** ⭐⭐
- Moderate state dimension (15 elements)
- Simple mathematical operations (normal distribution, linear updates)
- No external LP solvers
- No complex data structures
- Policy selection can use vmap elegantly
- JAX random number generation straightforward

---

### 4. StochasticShortestPath_Dynamic

#### Overview
Shortest path problem with uncertain edge costs using lookahead policy with horizon-based optimization.

#### File Structure
- **Files**: 3 Python files + parameters
- **Total LOC**: 598 lines
- **Model**: Model.py (174 lines)
- **Policy**: Policy.py (74 lines)
- **Driver**: Driver.py (83 lines)
- **GraphGenerator**: GraphGenerator.py (235 lines)

#### State Space Complexity
**LOW-MODERATE - Single node identifier**

```
State Variables:
- Current node: integer index [0, vertexCount)
- Cost estimates for outgoing edges: depends on node degree

Total State: 1 integer (node ID) + context-dependent edge costs
Example: state = node_5
```

#### Decision Variables
**LOW - Neighbor selection**
```
Decisions:
- Next node: discrete choice from neighbors(current_node)
- Depends on current node: variable branching factor

Total Decision Dimension: 1 categorical (2-10 neighbors typical)
```

#### Problem Dynamics
**MODERATE - Graph traversal with stochastic costs**

```
Dynamics:
1. Sample edge cost: cost ~ Uniform[mean×(1-spread), mean×(1+spread)]
2. Transition: move to selected neighbor
3. Accumulate cost
4. Stop when reaching target node

Objective: minimize total cost to reach target
Constraints: deadline (cost > deadline incurs penalty)
```

#### Special Features & Challenges
1. **Lookahead Policy**:
   - Backward-time dynamic programming
   - Solves Bellman equations over horizon
   - Percentile-based decision using estimated cost distributions
   
2. **Graph Management**:
   - Dynamic graph generation via GraphGenerator
   - networkx integration for path finding
   - Edge cost distributions parameterized by mean + spread
   
3. **Optimization Parameters**:
   - theta (risk parameter) ∈ [0,1]
   - horizon (lookahead depth)
   - deadline constraint for penalty
   
4. **Multiple Graph Creation Methods**:
   - createNetworkSteps(): layered graph
   - createNetworkChance(): random Erdos-Renyi style
   
5. **Complex Graph Structure**:
   - Variable node degrees
   - Multiple paths between nodes
   - Path enumeration for analysis

#### Migration Difficulty: **MODERATE-HIGH** ⭐⭐⭐
- Simple state (single integer)
- Graph structures less natural in JAX
- networkx integration needed (can use JAX for core dynamics)
- Backward DP solution could be JIT-compiled
- Graph generation can use numpy/jax random

---

### 5. StochasticShortestPath_Static

#### Overview
Extended shortest path problem using point estimates and Bellman equation solutions with static graph structure.

#### File Structure
- **Files**: 5 Python files + parameters
- **Total LOC**: 683 lines
- **Main Model**: StaticModelAdaptive.py (253 lines)
- **Solution Model**: StaticModelAdaptiveSolution.py (269 lines)
- **Policy**: PolicyAdaptive.py (29 lines)
- **Driver**: DriverScriptAdaptive.py (132 lines)

#### State Space Complexity
**LOW - Single node with edge costs**

```
State Variables:
- Current node: integer index [0, vertexCount)
- CurrentNodeLinksCost: edge costs to neighbors {neighbor: cost}

Total State: 1 integer + dictionary of costs
Example: state = (current_node=5, edge_costs={3: 2.5, 7: 1.8, ...})
```

#### Decision Variables
**LOW - Neighbor selection**
```
Decisions:
- Next node: discrete choice from neighbors(current_node)

Total Decision Dimension: 1 categorical (variable degree)
```

#### Problem Dynamics
**MODERATE - Graph traversal with value iteration**

```
Dynamics:
1. Initialize V values (expected cost to target from each node)
2. Bellman iteration: V[v] = min_u(cost[v][u] + V[u])
3. Transition: move to neighbor with minimum expected cost
4. Update VFA: combine V estimate with Bellman value
5. Accumulate cost

Key: Bellman equation-based decisions, not lookahead
```

#### Special Features & Challenges
1. **Value Function Approximation (VFA)**:
   - Stores V estimates for all nodes
   - Updates via: V[v] ← (1-α)V[v] + α × vhat
   - Step size selection (constant or harmonic)
   
2. **Graph Generation Classes**:
   - StochasticGraph class with Bellman solver
   - truebellman() method for deterministic distances
   - bellman() method for expected costs
   - randomgraphChoice() construction
   
3. **Complex Graph Objects**:
   - Node list, edges, upper/lower bounds per edge
   - Pytree-like structure (custom dict-based)
   
4. **Comparison with Dynamic**:
   - No horizon-based lookahead (uses global V)
   - Simpler policy (argmin of Bellman)
   - Accumulates actual costs, not estimated
   
5. **Multiple Solution Methods**:
   - Point estimates (averages of bounds)
   - Value iteration solution
   - Adaptive update rules

#### Migration Difficulty: **MODERATE** ⭐⭐
- Simple state (single integer + dict)
- Core computation is Bellman iteration (easily JIT-compiled)
- Graph structures can be pytree-registered
- Value iteration naturally parallelizable
- Random graph generation straightforward

---

## Comparison Matrix

| Aspect | BloodManagement | **AdaptiveMarketPlanning** | MedicalDecision | SSP_Dynamic | SSP_Static |
|--------|-----------------|--------|------------|---------|----------|
| **Total LOC** | 2,167 | **543** | 721 | 598 | 683 |
| **State Dimension** | 160-200+ | **2** | 15 | 1 | 1-10 |
| **Decision Dimension** | 160+ | **1** | 1 | 1 | 1 |
| **External Solvers** | CVXOPT LP | **None** | None | networkx | networkx |
| **Data Dependencies** | Excel, complex | **None** | Excel | networkx | networkx |
| **Algorithm Complexity** | VFA + AdaGrad | **Gradient Descent** | Bayesian Update | Lookahead DP | Bellman |
| **JAX Suitability** | Difficult | **Excellent** | Good | Moderate | Good |
| **Migration Effort** | Very High | **Very Low** | Moderate | High | Moderate |
| **Testing Effort** | Very High | **Very Low** | Moderate | Moderate | Moderate |
| **Educational Value** | High | **High** | Moderate | Moderate | Moderate |
| **Reusability** | Low | **High** | Moderate | Moderate | Moderate |

---

## Migration Priority Recommendation

### TIER 1 (Next): AdaptiveMarketPlanning ✅

**Rationale:**
1. **Minimal Complexity**: Only 2 state dimensions, 1 decision variable
2. **No External Dependencies**: Pure mathematical operations, no LP/graph libraries
3. **Quick Wins**: 543 LOC → can migrate in 1-2 days
4. **Learning Foundation**: Perfect template for other models
   - Shows policy variants (harmonic, Kesten, constant)
   - Demonstrates step size selection patterns
   - Simple-enough to understand JAX idioms fully
5. **Low Risk**: Simple dynamics reduces testing burden
6. **High Value**: Good pedagogical example for the library
7. **Demonstrable Performance**: Small model can show speedup clearly

**Timeline**: 1-2 days (including full testing and documentation)

### TIER 2: MedicalDecisionDiabetes

**Rationale:**
1. Moderate complexity (15 state dims, 5 decision options)
2. No external solver dependencies
3. Natural for policy vectorization (5 policy types)
4. Builds on AdaptiveMarketPlanning learnings
5. Standard Bayesian update is JAX-friendly
6. Good next step for practicing JAX patterns

**Timeline**: 3-4 days

### TIER 3: StochasticShortestPath_Static

**Rationale:**
1. Moderate complexity
2. Bellman iteration naturally JIT-compilable
3. Good for learning value iteration in JAX
4. Can reuse utilities from SSP_Dynamic
5. networkx dependency manageable

**Timeline**: 4-5 days

### TIER 4: StochasticShortestPath_Dynamic

**Rationale:**
1. Lookahead policy more complex than static
2. Graph operations less natural in JAX
3. Learn patterns from Static first

**Timeline**: 5-6 days

### TIER 5 (Last): BloodManagement

**Rationale:**
1. **Highest Complexity**: 160-200+ dimensional state
2. **External Dependencies**: CVXOPT LP solver integration
3. **Advanced Algorithms**: VFA with AdaGrad, projection methods
4. **Do Last**: Leverage patterns learned from 4 simpler problems
5. **Consider Refactoring**: May want to redesign LP solver integration for JAX

**Timeline**: 10-14 days (major project)

---

## Key Migration Pattern Insights

### From AdaptiveMarketPlanning Analysis:

**What makes it JAX-ready:**
```python
# Simple state and decision
State = Float[Array, "2"]  # [order_quantity, counter]
Decision = Float[Array, "1"]  # [step_size]

# Pure mathematical operations
@jax.jit
def transition(state, decision, exog):
    order_q, counter = state[0], state[1]
    step = decision[0]
    demand = exog['demand']
    
    # Gradient-based update
    derivative = price - cost if order_q < demand else -cost
    new_q = jnp.maximum(0.0, order_q + step * derivative)
    new_counter = counter + (1 if sign_change else 0)
    
    return jnp.array([new_q, new_counter])

# Trivial to parallelize
batch_transition = jax.vmap(lambda s, d: transition(s, d, exog))
```

**No JAX-unfriendly patterns:**
- ✅ No in-place operations
- ✅ No custom data structures
- ✅ No external solvers
- ✅ No complex branching
- ✅ No imperative style
- ✅ All pure functions

---

## Modernization Directory Status

The `Modernization/` directory contains comprehensive planning documents:

1. **FINAL_SUMMARY.md**: Executive overview (14K)
   - Technology stack decisions
   - JAX-native recommendation
   - Project structure
   
2. **JAX_MODERNIZATION_PLAN.md**: Complete 13-week roadmap (30K)
   - Type system (jaxtyping + chex)
   - Testing strategy
   - Neural network integration
   - Configuration examples
   
3. **NUMPY_VS_JAX_COMPARISON.md**: Decision framework (15K)
   - Performance benchmarks (10-1000x speedups!)
   - Type system comparison
   - Migration patterns
   
4. **QUICK_START.md**: Implementation guide (10K)
   - Environment setup
   - Daily workflow
   - Common issues
   
5. **Code Examples**:
   - `base_protocols_jax.py`: Type-safe protocols (16K)
   - `energy_storage_model_jax.py`: Complete example (16K)

**Key Recommendation from Modernization docs:**
- JAX-native is the clear choice (not NumPy-native)
- 10-100x speedup on CPU, 100-1000x on GPU
- Superior type safety with jaxtyping + chex
- Automatic differentiation for policy optimization
- GPU/TPU support out of the box

---

## Action Plan

### Immediate (Next 1-2 days):
1. **Migrate AdaptiveMarketPlanning to JAX**
   - Use `base_protocols_jax.py` as template
   - Implement `AdaptiveMarketPlanningModel` with chex.dataclass
   - Create JAX-native policy with three variants
   - Write comprehensive tests using chex assertions
   - Benchmark vs numpy implementation

2. **Document Migration Pattern**
   - Create migration guide for next problems
   - Record decisions and workarounds
   - Update JAX_MODERNIZATION_PLAN.md with learnings

### Following Week (Days 3-8):
3. **Migrate MedicalDecisionDiabetes**
   - Leverage AdaptiveMarketPlanning patterns
   - Focus on policy vectorization
   - Multiple policy variants good vmap practice

4. **Migrate first StochasticShortestPath variant**
   - Choose Static or Dynamic based on learnings
   - Graph management patterns
   - Bellman iteration optimization

### Later:
5. **Return to BloodManagement last**
   - Major undertaking after simpler examples
   - May need custom solver integration research
   - Highest complexity, highest impact

---

## Conclusion

**AdaptiveMarketPlanning is unequivocally the best choice for next migration because:**

1. ✅ **Minimal Risk**: Simplest codebase (543 LOC, 2 state dims)
2. ✅ **Quick Win**: Can complete in 1-2 days
3. ✅ **Pattern Foundation**: Perfect template for subsequent migrations
4. ✅ **Educational Value**: Clean example of gradient-based optimization in JAX
5. ✅ **No Dependencies**: Pure math, no external solvers
6. ✅ **Obvious Speedup**: Small enough to demonstrate JAX benefits clearly
7. ✅ **Testing**: Easiest to achieve high coverage
8. ✅ **Momentum**: Success will guide larger migrations

After this, the learning curve accelerates for subsequent models.

---

**Report Generated**: 2025-11-14
**Analysis Scope**: 5 legacy problems, 4,312 total LOC
**Recommendation**: Start with AdaptiveMarketPlanning immediately
