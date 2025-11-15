# Quick Start: Implementing the Modernization

This guide provides step-by-step instructions to begin modernizing the stochastic-optimization repository.

## Prerequisites

- Python 3.10 or higher
- Git
- pip or conda for package management

## Step 1: Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/wbpowell328/stochastic-optimization.git
cd stochastic-optimization

# Create a new branch for modernization
git checkout -b modernization-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install --upgrade pip
pip install mypy pytest pytest-cov hypothesis black isort ruff pre-commit
pip install numpy scipy pandas matplotlib jax flax optax
```

## Step 2: Create New Project Structure

```bash
# Create the new source structure
mkdir -p src/stochastic_optimization/{base,models,policies,utils,drivers}
mkdir -p tests/{unit,integration}
mkdir -p docs examples scripts

# Create __init__.py files
touch src/stochastic_optimization/__init__.py
touch src/stochastic_optimization/base/__init__.py
touch src/stochastic_optimization/models/__init__.py
touch src/stochastic_optimization/policies/__init__.py
touch tests/__init__.py
```

## Step 3: Add Modern Configuration Files

### pyproject.toml

Copy the pyproject.toml from the MODERNIZATION_PLAN.md and place it in the repository root.

```bash
# Create pyproject.toml with the content from the plan
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "stochastic-optimization"
version = "2.0.0"
description = "Sequential Decision Problem Modeling Library"
# ... (rest of the configuration from MODERNIZATION_PLAN.md)
EOF
```

### mypy.ini

```bash
cat > mypy.ini << 'EOF'
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
# ... (rest from MODERNIZATION_PLAN.md)
EOF
```

### .pre-commit-config.yaml

```bash
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies: [numpy, pandas, types-all]
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
EOF

# Install pre-commit hooks
pre-commit install
```

## Step 4: Create Base Protocol Files

```bash
# Copy the base_protocols.py to the new location
cp /path/to/base_protocols.py src/stochastic_optimization/base/protocols.py

# Create types.py for type aliases
cat > src/stochastic_optimization/types.py << 'EOF'
"""Common type definitions for stochastic optimization."""

from typing import Protocol, Dict, Any, TypeVar
import numpy as np
import numpy.typing as npt

# Type aliases
State = npt.NDArray[np.float64]
Decision = npt.NDArray[np.float64]
Reward = float
Time = int

__all__ = ['State', 'Decision', 'Reward', 'Time']
EOF
```

## Step 5: Migrate Existing Models

### Example: Energy Storage Model

```bash
# Create energy storage directory
mkdir -p src/stochastic_optimization/models/energy_storage

# Copy and adapt the energy storage model
cp /path/to/energy_storage_model.py \
   src/stochastic_optimization/models/energy_storage/model.py

# Add __init__.py
cat > src/stochastic_optimization/models/energy_storage/__init__.py << 'EOF'
"""Energy storage optimization model."""

from .model import (
    EnergyStorageModel,
    EnergyStorageConfig,
    EnergyStorageExogenousInfo,
)

__all__ = [
    'EnergyStorageModel',
    'EnergyStorageConfig',
    'EnergyStorageExogenousInfo',
]
EOF
```

### Repeat for Other Models

For each existing model (asset_selling, blood_management, etc.):

1. Create directory: `mkdir -p src/stochastic_optimization/models/{model_name}`
2. Add type hints to existing code
3. Create config dataclass
4. Implement Model protocol
5. Add comprehensive docstrings
6. Create `__init__.py`

## Step 6: Add Type Hints to Existing Code

### Strategy: One File at a Time

```python
# Example transformation of existing code

# BEFORE (original code)
def transition(state, decision, exog):
    new_state = state.copy()
    new_state[0] += decision[0]
    return new_state

# AFTER (modernized with types)
def transition(
    self,
    state: State,
    decision: Decision,
    exogenous_info: ExogenousInfo,
) -> State:
    """Compute next state after applying decision.
    
    Args:
        state: Current state vector.
        decision: Decision/action vector.
        exogenous_info: Exogenous information.
    
    Returns:
        Next state vector.
    """
    new_state = state.copy()
    new_state[0] += decision[0]
    return new_state
```

### Use mypy to Find Missing Types

```bash
# Run mypy on each file to find missing types
mypy src/stochastic_optimization/models/energy_storage/model.py

# Fix errors one by one
# Repeat until mypy passes with no errors
```

## Step 7: Create Tests

### For Each Model

```bash
# Create test file
mkdir -p tests/unit/models/energy_storage
cp /path/to/test_energy_storage.py \
   tests/unit/models/energy_storage/test_model.py

# Create conftest.py for shared fixtures
cat > tests/conftest.py << 'EOF'
"""Shared test fixtures."""

import pytest
import numpy as np

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
EOF
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Open coverage report
# open htmlcov/index.html  # On Mac
# xdg-open htmlcov/index.html  # On Linux
```

## Step 8: Set Up CI/CD

### GitHub Actions

```bash
mkdir -p .github/workflows

# Create test workflow
cat > .github/workflows/tests.yml << 'EOF'
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
EOF

# Similar files for type checking and linting
# (See MODERNIZATION_PLAN.md for complete workflows)
```

## Step 9: Add Documentation

### Sphinx Setup

```bash
# Create docs directory
mkdir -p docs
cd docs

# Initialize Sphinx
sphinx-quickstart

# Configure conf.py (see MODERNIZATION_PLAN.md)

# Generate API docs
sphinx-apidoc -o api ../src/stochastic_optimization

# Build docs
make html

# View docs
# open _build/html/index.html
```

## Step 10: Iterative Migration

### Recommended Order

1. ✅ Set up infrastructure (Steps 1-3)
2. ✅ Create base protocols (Step 4)
3. 🔄 Migrate one model completely (Step 5-7)
   - Energy storage is a good starting point
4. 🔄 Add tests for that model
5. 🔄 Ensure mypy passes
6. 🔄 Add documentation
7. 🔄 Repeat for other models
8. 🔄 Add neural network policies
9. 🔄 Performance optimization
10. ✅ Release v2.0.0

## Daily Workflow

### Morning Routine

```bash
# Update local branch
git pull origin modernization-v2

# Activate environment
source venv/bin/activate

# Run quick checks
mypy src/
pytest tests/ -x  # Stop on first failure
```

### Before Each Commit

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/

# Run tests
pytest tests/ -v

# If all pass, commit
git add .
git commit -m "Add type hints to energy storage model"
git push origin modernization-v2
```

## Common Issues and Solutions

### Issue: mypy errors on numpy arrays

**Solution**: Use `numpy.typing`

```python
# Instead of
def foo(arr):
    return arr

# Use
import numpy.typing as npt
import numpy as np

def foo(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return arr
```

### Issue: Tests fail on different platforms

**Solution**: Use `pytest.approx()` for floating point comparisons

```python
# Instead of
assert result == 1.23456789

# Use
assert result == pytest.approx(1.23456789, rel=1e-5)
```

### Issue: Import errors after restructuring

**Solution**: Install package in editable mode

```bash
pip install -e .
```

## Measuring Progress

### Metrics to Track

```bash
# Type coverage (aim for 100%)
mypy src/ --strict | grep "Success"

# Test coverage (aim for >80%)
pytest --cov=src --cov-report=term-missing | grep "TOTAL"

# Documentation coverage
sphinx-build -b coverage docs docs/_build/coverage
cat docs/_build/coverage/python.txt

# Number of files modernized
find src/ -name "*.py" -exec grep -l "npt.NDArray" {} \; | wc -l
```

### Weekly Goals

- Week 1: Infrastructure setup, 1 model modernized
- Week 2: 3 models modernized
- Week 3-4: All models modernized with type hints
- Week 5-6: Tests for all models, >80% coverage
- Week 7-8: Documentation complete
- Week 9-10: Neural network policies
- Week 11: Performance optimization
- Week 12: Release preparation

## Getting Help

### Resources

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [mypy Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Quick Start](https://hypothesis.readthedocs.io/en/latest/quickstart.html)
- [Flax NNX Tutorial](https://flax.readthedocs.io/en/latest/nnx/index.html)

### Questions?

- Check the MODERNIZATION_PLAN.md for detailed guidance
- Review the example implementations
- Run the example tests to see patterns
- Refer to the base_protocols.py for interfaces

## Success Criteria

✅ **You'll know you're successful when:**

1. All files have complete type hints
2. `mypy --strict` passes on entire codebase  
3. Test coverage > 80%
4. All tests pass on Python 3.10, 3.11, 3.12
5. Documentation builds without errors
6. Examples run successfully
7. Code is faster or same speed as v1
8. Users report fewer issues

Good luck with the modernization! 🚀
