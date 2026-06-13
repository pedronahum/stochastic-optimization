# Notebooks

Interactive tutorials for all 9 problems. Each notebook is self-contained and
runs in Google Colab: the first cell installs dependencies and clones this repo,
so no local setup is needed. Click a badge to open.

| Problem | What it shows | Colab |
|---|---|---|
| **Blood Management** | OT allocation across 8 blood types (ABO/Rh substitution), inventory aging, surges | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/blood_management.ipynb) |
| **Clinical Trials** | Drug-enrollment MDP: enroll vs. stop on a Beta(success, failure) belief | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/clinical_trials.ipynb) |
| **SSP Dynamic** | Risk-sensitive lookahead routing on a stochastic graph (NetworkX viz) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/ssp_dynamic.ipynb) |
| **SSP Static** | Shortest path with online value-function learning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/ssp_static.ipynb) |
| **Adaptive Market Planning** | Newsvendor order-quantity learning → analytic optimum `q*=μ·ln(p/c)` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/adaptive_market_planning.ipynb) |
| **Medical Decision (Diabetes)** | Bayesian multi-armed bandit over drugs (UCB) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/medical_decision_diabetes.ipynb) |
| **Two Newsvendor** | Field/central newsvendor coordination, critical-fractile economics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/two_newsvendor.ipynb) |
| **Asset Selling** | Optimal stopping under a Markov-modulated price walk | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/asset_selling.ipynb) |
| **Energy Storage** | Battery arbitrage (buy low / sell high) against a price series | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/energy_storage.ipynb) |

## Each notebook

1. **Setup** — `pip install` deps and `git clone` the repo (Colab).
2. **Configure** a model + policy.
3. **Simulate** an episode.
4. **Visualize** the trajectory with matplotlib.

## Running locally

The notebooks' first cell is written for Colab (it clones the *published* repo).
To run them against your **local** checkout instead, use the headless runner,
which neutralizes that bootstrap cell and executes against this working copy:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python benchmarks/run_notebooks.py
```

Or open them directly:

```bash
pip install -e ".[viz]"          # adds matplotlib + networkx
jupyter notebook notebooks/
```

## Requirements

- Python ≥ 3.11
- `jax` / `jaxlib`, plus `matplotlib` and `networkx` (the `viz` extra) for plots
- Jupyter or Google Colab (Colab auto-installs everything)

## More

- Library overview: [../README.md](../README.md)
- Quick start: [../QUICK_START.md](../QUICK_START.md)
- Parity vs. the originals: [../benchmarks/PARITY.md](../benchmarks/PARITY.md)
