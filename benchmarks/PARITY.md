# Examples, notebooks & parity vs. the original Powell code

This documents (1) running every example and notebook against the current code,
and (2) benchmarking the new JAX implementations for **parity** against the
original implementations in `legacy/old_problems/` (a copy of
[wbpowell328/stochastic-optimization](https://github.com/wbpowell328/stochastic-optimization)).

## Environment

A dedicated project venv (`.venv`, Python 3.12) was created with the same
JAX/CUDA stack as the working reference env: `jax[cuda13]==0.10.0`, `flax 0.12.7`,
`optax 0.2.8`, `chex 0.1.91`, `jaxtyping 0.3.9`, plus `openpyxl` (to read the
originals' `.xlsx` params), the Jupyter stack, and `networkx`. `jax.devices()`
reports `[CudaDevice(id=0)]` on the GB10. Everything below runs with
`JAX_PLATFORMS=cpu` (the migration is platform-independent and this avoids the
GB10 unified-memory OOM) and `MPLBACKEND=Agg`.

## 1. Examples — 3/3 pass

`python examples/<name>.py` (CPU):

| Example | Result |
|---|---|
| `train_clinical.py` | ✓ trained dose-weight `w = -0.879` |
| `train_energy_storage.py` | ✓ trains linear/neural, evaluates 4 policies |
| `train_asset_selling.py` | ✓ REINFORCE 500 iters, eval ≈ $90.5 |

## 2. Notebooks — 9/9 pass

Runner: `benchmarks/run_notebooks.py` (executes each headless on CPU). The
notebooks are **Google Colab bootstraps** — their first cell `!pip install`s and
`git clone`s the *published* GitHub repo and `chdir`s into it; the runner
neutralises that cell so they execute against the **local** working copy.

Fixes required to get to 9/9 (committed):

- `two_newsvendor.ipynb` — a code cell was missing the `outputs`/`execution_count`
  keys (invalid `nbformat`).
- `adaptive_market_planning.ipynb` & `medical_decision_diabetes.ipynb` — demo
  cells referenced undefined `key`/`state` and used wrong decision shapes / wrong
  domain framing (a 3-vector "pricing" decision for a scalar step-size problem; a
  glucose loop for a 5-arm drug bandit). Rewritten to use the real model/policy API.
- `ssp_dynamic.ipynb` — `boxplot(labels=…)` → `tick_labels=` (matplotlib ≥ 3.9).
- **`networkx`** was missing — it is imported by the two SSP notebooks but is
  **not declared** in `pyproject.toml` (see Findings).

## 3. Parity vs. the original — `benchmarks/parity.py`

We do **not** compare Monte-Carlo objective *means* — those depend on each
codebase's RNG stream and are not a clean signal. Instead we check what a
faithful migration must preserve:

- **deterministic-core equivalence** — feed identical inputs to the original and
  new `reward`/`transition`/closed-form policy and require agreement to ~1e-4, and
- **analytical anchors** — closed-form optima (newsvendor fractile, etc.).

### Executable parity results (all pass)

| Problem | Check | Result |
|---|---|---|
| `adaptive_market_planning` | original `transition_fn`/`objective_fn` vs new `transition`/`reward` on 6 random matched inputs | **exact match** (≤1e-4) |
| `adaptive_market_planning` | newsvendor optimum `q* = μ·ln(p/c)` anchor | 26.24 ✓ |
| `asset_selling` | original `sell_low_policy`/`high_low_policy` vs new `SellLowPolicy`/`HighLowPolicy` over a 24-point price grid | **identical decisions** |
| `two_newsvendor` | new `reward` vs closed-form newsvendor cost; `argmax_q E[reward]` vs critical fractile `q*` | **q\* = 90 (CR 0.9)** ✓ |
| `medical_decision_diabetes` | original `transition_fn` belief update vs new `transition` on 5 random matched inputs (posterior mean + precision) | **exact match** (≤1e-4) |
| `medical_decision_diabetes` | best arm = `argmax true_mu` | drug 4 ✓ |
| `ssp_dynamic` | risk-neutral `LookaheadPolicy` next node vs networkx Dijkstra on the model's mean-cost graph | **7/7 nodes identical** |
| `ssp_static` | Bellman value iteration on the model's graph vs networkx Dijkstra; `reward == −edge_cost` | **match** |
| `blood_management` | reward(fulfil) > reward(unmet) sanity (reformulation — see below) | ✓ |

Run: `JAX_PLATFORMS=cpu python benchmarks/parity.py`

### Classification of all 9 problems

| Problem | Relation to original | Parity status |
|---|---|---|
| `adaptive_market_planning` | **faithful port** | ✅ verified exact (deterministic transition/reward + newsvendor analytical) |
| `asset_selling` | **faithful port** | ✅ verified exact (policy decisions identical; reward identical by construction) |
| `two_newsvendor` | **faithful** (same newsvendor economics) | ✅ verified vs analytical critical fractile |
| `medical_decision_diabetes` | **faithful port** | ✅ verified exact (Bayesian belief update identical to original `transition_fn`) + best-arm anchor |
| `ssp_dynamic` | **faithful** (same SSP + lookahead) | ✅ verified — lookahead reproduces Dijkstra shortest-path decisions |
| `ssp_static` | **faithful** (same SSP; online value-iteration learner) | ✅ verified — graph optimum matches Dijkstra/Bellman, reward charges edge cost. (The online learner itself is not run to convergence.) |
| `energy_storage` | **faithful port** | ✅ verified exact — `(buy, sell)` decision, `transition energy' = energy + η·buy − sell`, `reward = price·(η·sell − buy)`, prices from the historical series. (Re-written from the earlier reformulation; the `$1000/cycle` degradation term was removed.) |
| `clinical_trials` | **faithful port** | ✅ verified exact — drug-enrollment MDP: state `[potential_pop, success, failure, l_response]`, decision `[enroll, prog_continue, drug_success]`, original `transition_fn`/`objective_fn` reproduced. (Re-written from the earlier dose-control toy.) |
| `blood_management` | **faithful port (approximate solve)** | ✅ verified close — the original per-period allocation **LP** (min-cost network flow, glpk) is reproduced with **entropic OT (Sinkhorn, `ott-jax`)**, JAX-native and differentiable. The OT objective matches the exact LP (scipy/HiGHS) to **~0.01** (gap from entropic regularisation). Original `contribution()` weights + ABO/RhD substitution rules used. |

**Verdict:** all **9 of 9** problems are faithful ports passing executable parity — 8 exact/analytical, and `blood_management` matching the exact LP to ~0.01 via entropic OT.

`blood_management` also reproduces the original's **ADP layer** (Powell's SPAR/CAVE) in `problems/blood_management/adp.py`: separable concave piecewise-linear value functions for the value-to-go of holding blood, trained by feeding the per-period LP dual (`vhat`) into a stepsize update with concavity projection. Parity here is **behavioural**, not numeric (learned value functions depend on RNG/stepsize/projection): the trained value functions come out concave and the ADP policy **beats the myopic OT allocation** (~4% more total contribution on surge-prone instances). The training LP uses exact duals (scipy/HiGHS, faithful to the original glpk).

### Why entropic OT for blood (and not jaxopt)

The blood allocation is a degenerate transportation LP. `jaxopt.BoxOSQP` (first-order ADMM) only reached ~1% gaps and sometimes returned slightly infeasible points (didn't converge). Entropic OT / Sinkhorn (`ott-jax`) is the natural, robust, differentiable JAX-native solver for transportation problems and matches the exact LP to ~1e-2 with small regularisation (`epsilon`). `jaxopt` is pulled in transitively by `ott-jax`.

## Findings (beyond run/parity)

1. **`clinical_trials` and `energy_storage` were reformulations — now re-ported**
   faithfully to the originals (drug-enrollment MDP, and buy/sell against a price
   series respectively); both pass exact parity.
2. **`networkx`** is declared in the `viz` extra (used by the SSP notebooks).
3. **`NewsvendorFieldPolicy` computes `critical_ratio` but never uses it** — it
   orders the demand *estimate* (≈ mean), not the critical-fractile optimum. The
   reward economics are correct (optimum at `q*=90`); the policy is suboptimal.
4. **Notebooks are Colab-only** — they clone the published GitHub repo, so as
   shipped they never exercise local changes (acceptable per project owner).
5. **`blood_management`** is now a faithful port end-to-end: the per-period
   allocation LP via entropic OT (`ott-jax` Sinkhorn, ~1e-2 of exact), **and**
   the original's learned value-to-go via the SPAR/CAVE ADP
   (`problems/blood_management/adp.py`) — concave PWL value functions trained
   from LP duals; the ADP policy beats the myopic allocation.

## Reproduce

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg .venv/bin/python benchmarks/run_notebooks.py
JAX_PLATFORMS=cpu              .venv/bin/python benchmarks/parity.py
```
