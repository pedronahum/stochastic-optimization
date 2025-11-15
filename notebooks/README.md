# Jupyter Notebooks

Interactive tutorials for all 9 stochastic optimization problems. Each notebook is standalone and works seamlessly in Google Colab.

## 🚀 Quick Start

Click any badge below to open in Google Colab:

### 1. Blood Management
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/blood_management.ipynb)

Blood bank inventory optimization with 8 blood types, age-dependent inventory, and stochastic demand.

**Topics**: Inventory management, substitution rules, FIFO dynamics, surge events

---

### 2. Clinical Trials
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/clinical_trials.ipynb)

Adaptive dose optimization for clinical trials with patient outcomes.

**Topics**: Dose-response modeling, adaptive design, safety constraints

---

### 3. SSP Dynamic
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/ssp_dynamic.ipynb)

Stochastic shortest path with dynamic programming and lookahead.

**Topics**: Graph algorithms, dynamic programming, risk sensitivity, cost estimation

---

### 4. SSP Static
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/ssp_static.ipynb)

Classical shortest path with percentile-based risk measures.

**Topics**: Bellman-Ford, percentile optimization, graph search

---

### 5. Adaptive Market Planning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/adaptive_market_planning.ipynb)

Dynamic pricing and demand forecasting with market adaptation.

**Topics**: Price optimization, demand modeling, market dynamics

---

### 6. Medical Decision - Diabetes
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/medical_decision_diabetes.ipynb)

Glucose-insulin dynamics for diabetes management.

**Topics**: Physiological modeling, treatment policies, health monitoring

---

### 7. Two Newsvendor
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/two_newsvendor.ipynb)

Multi-agent inventory coordination with demand uncertainty.

**Topics**: Coordination mechanisms, inventory allocation, multi-agent systems

---

### 8. Asset Selling
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/asset_selling.ipynb)

Optimal asset liquidation with price dynamics.

**Topics**: Optimal stopping, price processes, market timing

---

### 9. Energy Storage
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/stochastic-optimization/blob/main/notebooks/energy_storage.ipynb)

Battery management with price arbitrage.

**Topics**: Battery dynamics, price-based charging, capacity management

---

## 📚 What's in Each Notebook?

Each notebook includes:

1. **Problem Overview** - Clear description and motivation
2. **Mathematical Formulation** - Complete equations and notation
   - State space definition
   - Dynamics and transitions
   - Reward/cost function
   - Objective formulation
3. **Setup Instructions** - Automatic installation for Colab
4. **Interactive Code** - Run simulations step-by-step
5. **Visualizations** - Charts and plots of results
6. **Policy Comparison** - Compare different strategies
7. **Key Insights** - Interpretation and takeaways
8. **Extensions** - Ideas for experimentation

## 🎯 Learning Path

### Beginner
Start with simpler problems:
1. **Asset Selling** - Single agent, basic dynamics
2. **Energy Storage** - Battery constraints, price arbitrage
3. **SSP Static** - Graph algorithms fundamentals

### Intermediate
Move to multi-dimensional problems:
4. **SSP Dynamic** - Dynamic programming, lookahead
5. **Clinical Trials** - Adaptive policies, safety
6. **Two Newsvendor** - Multi-agent coordination

### Advanced
Tackle complex inventory problems:
7. **Medical Decision Diabetes** - Physiological modeling
8. **Adaptive Market Planning** - Market dynamics
9. **Blood Management** - Multi-type inventory, substitution

## 💻 Running Locally

If not using Colab:

```bash
# Clone repository
git clone https://github.com/pedronahum/stochastic-optimization.git
cd stochastic-optimization

# Install dependencies
pip install jax jaxlib jaxtyping chex numpy matplotlib jupyter

# Launch Jupyter
jupyter notebook notebooks/
```

## 🔧 Requirements

- Python 3.10+
- JAX 0.4+
- matplotlib (for visualizations)
- Jupyter or Google Colab

All dependencies are automatically installed in Colab.

## 📖 Documentation

For more details:
- **Main README**: [../README.md](../README.md)
- **Quick Start Guide**: [../QUICK_START.md](../QUICK_START.md)
- **Code Documentation**: See `problems/<problem>/` for source code

## 🤝 Contributing

Want to improve a notebook?
1. Fork the repository
2. Make your changes
3. Test in Colab
4. Submit a pull request

## 📄 License

See [LICENSE](../LICENSE) for details.

---

**Princeton University - Castle Lab**
https://github.com/pedronahum/stochastic-optimization
