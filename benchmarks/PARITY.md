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

Run: `JAX_PLATFORMS=cpu python benchmarks/parity.py`

### Classification of all 9 problems

| Problem | Relation to original | Parity status |
|---|---|---|
| `adaptive_market_planning` | **faithful port** | ✅ verified exact (deterministic + analytical) |
| `asset_selling` | **faithful port** | ✅ verified exact (policy decisions; reward identical by construction) |
| `two_newsvendor` | **faithful** (same newsvendor economics) | ✅ verified vs analytical fractile |
| `energy_storage` | same domain, **reformulated** | ⚠️ revenue term maps to original `price·(η·sell − buy)`, **but** the new reward adds a `$1000/cycle` degradation cost the original lacks, and uses a single signed charge decision instead of `(buy, sell)`. Not numerically equal. |
| `medical_decision_diabetes` | same domain (Bayesian drug bandit) | ◻️ assessed by inspection — comparable; executable belief-update parity is feasible (follow-up) |
| `blood_management` | same domain (allocation) | ◻️ assessed by inspection — executable contribution parity feasible (follow-up) |
| `ssp_static` | same domain (shortest path) | ◻️ assessed by inspection — analytical check vs Dijkstra feasible (follow-up) |
| `ssp_dynamic` | same domain (SSP + lookahead) | ◻️ assessed by inspection — analytical check vs Bellman feasible (follow-up) |
| `clinical_trials` | **reformulation — different problem** | ❌ parity N/A. The original models drug-program *enrollment* (potential population, success/failure counts, program revenue). The new `clinical_trials` is a scalar dose-control toy: `x_{t+1}=x_t+a+noise`, reward `−|x|`. They are not the same problem. |

## Findings (beyond run/parity)

1. **`clinical_trials` is not a port** — it is an unrelated, much simpler model.
   If a faithful clinical-trials port is intended, the new model needs rewriting;
   otherwise it should be renamed to reflect that it's a dose-control toy.
2. **`networkx` is an undeclared dependency** (used by the SSP notebooks). Add it
   to `pyproject.toml` (a `notebooks`/`viz` extra).
3. **`NewsvendorFieldPolicy` computes `critical_ratio` but never uses it** — it
   orders the demand *estimate* (≈ mean), not the critical-fractile optimum. The
   reward economics are correct (optimum at `q*=90`); the policy is suboptimal.
4. **Notebooks are Colab-only** — they clone the published GitHub repo, so as
   shipped they never exercise local changes. Consider a local-run path.
5. **`energy_storage` adds a degradation cost** absent from the original — a
   deliberate modelling change worth documenting in the model docstring.

## Reproduce

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg .venv/bin/python benchmarks/run_notebooks.py
JAX_PLATFORMS=cpu              .venv/bin/python benchmarks/parity.py
```
