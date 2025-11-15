"""JAX-native base protocols for stochastic optimization.

This module provides type-safe interfaces using JAX, jaxtyping, and chex.
All operations are designed to be JIT-compilable and work with JAX transformations
like vmap, grad, and scan.
"""

from typing import Protocol, Callable, Any, NamedTuple
from functools import partial
from jaxtyping import Array, Float, Int, Bool, PyTree, PRNGKeyArray
import jax
import jax.numpy as jnp
import chex

# Type aliases with shape annotations
State = Float[Array, "state_dim"]
StateBatch = Float[Array, "batch state_dim"]
Decision = Float[Array, "action_dim"]
DecisionBatch = Float[Array, "batch action_dim"]
Reward = Float[Array, ""]  # Scalar
RewardBatch = Float[Array, "batch"]
Key = PRNGKeyArray
Time = Int[Array, ""] | int

# Function types
TransitionFn = Callable[[State, Decision, Any], State]
RewardFn = Callable[[State, Decision, Any], Reward]
PolicyFn = Callable[[PyTree, State, Key], Decision]


class ExogenousInfo(NamedTuple):
    """Base class for exogenous information.
    
    Using NamedTuple for automatic pytree registration and immutability.
    Subclass this for specific problem domains.
    
    Example:
        >>> class EnergyExog(ExogenousInfo):
        ...     price: Float[Array, ""]
        ...     demand: Float[Array, ""]
        ...     
        >>> exog = EnergyExog(price=jnp.array(50.0), demand=jnp.array(100.0))
    """
    pass


@chex.dataclass(frozen=True)
class ModelConfig:
    """Base configuration for models.
    
    Using chex.dataclass for:
    - Automatic pytree registration
    - Immutability enforcement
    - Runtime validation with __post_init__
    
    Subclass this for specific models and add validation in __post_init__.
    
    Example:
        >>> @chex.dataclass(frozen=True)
        ... class MyConfig(ModelConfig):
        ...     capacity: float
        ...     rate: float
        ...     
        ...     def __post_init__(self):
        ...         chex.assert_scalar_positive(self.capacity)
    """
    pass


class Model(Protocol):
    """Protocol for JAX-native sequential decision models.
    
    All methods must be pure functions (no side effects) and should be
    JIT-compilable. Models work with immutable state - each transition
    returns a new state rather than modifying the input.
    
    Attributes:
        config: Immutable configuration for the model.
    
    Example:
        >>> class MyModel:
        ...     def __init__(self, config: MyConfig):
        ...         self.config = config
        ...     
        ...     def init_state(self, key: Key) -> State:
        ...         return jnp.zeros(self.config.state_dim)
        ...     
        ...     @partial(jax.jit, static_argnums=(0,))
        ...     def transition(self, state, decision, exog):
        ...         return state + decision
    """
    
    config: ModelConfig
    
    def init_state(self, key: Key) -> State:
        """Initialize state (pure function).
        
        Args:
            key: JAX random key for stochastic initialization.
        
        Returns:
            Initial state vector.
        
        Note:
            Can be JIT-compiled but usually not necessary since it's
            called once per simulation.
        """
        ...
    
    def transition(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> State:
        """Compute next state (pure function, should be JIT-compiled).
        
        This implements: S_{t+1} = S^M(S_t, x_t, W_{t+1})
        
        Args:
            state: Current state vector.
            decision: Decision/action vector.
            exog: Exogenous information.
        
        Returns:
            Next state vector.
        
        Note:
            Decorate with @partial(jax.jit, static_argnums=(0,))
            for performance. The static_argnums=(0,) keeps 'self'
            as a static argument.
        """
        ...
    
    def reward(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> Reward:
        """Compute reward (pure function, should be JIT-compiled).
        
        This implements: C(S_t, x_t, W_{t+1})
        
        Args:
            state: Current state.
            decision: Decision taken.
            exog: Exogenous information.
        
        Returns:
            Scalar reward (higher is better).
        
        Note:
            Decorate with @partial(jax.jit, static_argnums=(0,))
            for performance.
        """
        ...
    
    def sample_exogenous(self, key: Key, time: Time) -> ExogenousInfo:
        """Sample exogenous information (pure function).
        
        Args:
            key: JAX random key.
            time: Current time step (may affect distribution).
        
        Returns:
            Sampled exogenous information.
        
        Note:
            This uses JAX random key splitting for reproducibility.
            Do not use numpy.random here.
        """
        ...
    
    def is_valid_decision(
        self,
        state: State,
        decision: Decision,
    ) -> Bool[Array, ""]:
        """Check if decision is valid (pure function, JIT-compilable).
        
        Args:
            state: Current state.
            decision: Proposed decision.
        
        Returns:
            Boolean scalar indicating validity.
        
        Note:
            Use JAX operations (jnp.where, jnp.logical_and, etc.)
            instead of Python conditionals for JIT compatibility.
        """
        ...


class Policy(Protocol):
    """Protocol for JAX-native policies.
    
    Policies are represented as pure functions that map states to decisions.
    Parameters are stored in pytrees for compatibility with JAX optimizers.
    
    Example:
        >>> class LinearPolicy:
        ...     def __init__(self, key):
        ...         self.params = {'weights': jax.random.normal(key, (n, m))}
        ...     
        ...     @partial(jax.jit, static_argnums=(0,))
        ...     def __call__(self, params, state, key):
        ...         return params['weights'] @ state
    """
    
    def __call__(
        self,
        params: PyTree,
        state: State,
        key: Key,
    ) -> Decision:
        """Get decision for current state.
        
        Args:
            params: Policy parameters (pytree).
            state: Current state vector.
            key: Random key for stochastic policies.
        
        Returns:
            Decision/action vector.
        
        Note:
            This should be a pure function of (params, state, key).
            No internal state should be modified.
        """
        ...


# Utility functions for simulation

@partial(jax.jit, static_argnums=(0, 2))
def simulate_step(
    model: Model,
    policy_fn: PolicyFn,
    state: State,
    policy_params: PyTree,
    key: Key,
    time: int,
) -> tuple[State, Reward, ExogenousInfo, Decision]:
    """Single simulation step (JIT-compiled).
    
    This function orchestrates one time step:
    1. Sample exogenous information
    2. Get decision from policy
    3. Compute reward
    4. Transition to next state
    
    Args:
        model: Problem model.
        policy_fn: Policy function.
        state: Current state.
        policy_params: Policy parameters.
        key: Random key.
        time: Current time step.
    
    Returns:
        Tuple of (next_state, reward, exogenous_info, decision).
    
    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> state = model.init_state(key)
        >>> next_state, reward, exog, decision = simulate_step(
        ...     model, policy_fn, state, policy_params, key, 0
        ... )
    """
    # Split key for different random operations
    key_exog, key_policy = jax.random.split(key)
    
    # Sample exogenous info
    exog = model.sample_exogenous(key_exog, time)
    
    # Get decision from policy
    decision = policy_fn(policy_params, state, key_policy)
    
    # Assert valid decision (disabled in JIT mode, useful for debugging)
    chex.assert_tree_all_finite(decision)
    
    # Compute reward
    reward = model.reward(state, decision, exog)
    
    # Transition to next state
    next_state = model.transition(state, decision, exog)
    
    # Assert finite values
    chex.assert_tree_all_finite((next_state, reward))
    
    return next_state, reward, exog, decision


def simulate_trajectory(
    model: Model,
    policy_fn: PolicyFn,
    policy_params: PyTree,
    key: Key,
    horizon: int,
    discount: float = 1.0,
) -> dict[str, Array]:
    """Simulate full trajectory using jax.lax.scan.
    
    This is much more efficient than a Python loop because:
    1. Everything is JIT-compiled
    2. Uses scan for efficient sequential operations
    3. Memory usage is constant (scan doesn't unroll)
    
    Args:
        model: Problem model.
        policy_fn: Policy function.
        policy_params: Policy parameters.
        key: Initial random key.
        horizon: Number of time steps.
        discount: Discount factor for rewards.
    
    Returns:
        Dictionary with:
            - states: Shape (horizon+1, state_dim) - includes initial state
            - decisions: Shape (horizon, action_dim)
            - rewards: Shape (horizon,)
            - exogenous: Pytree with shape (horizon,)
            - total_reward: Scalar - discounted cumulative reward
    
    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> trajectory = simulate_trajectory(
        ...     model, policy_fn, policy_params, key, horizon=100
        ... )
        >>> print(f"Total reward: {trajectory['total_reward']}")
    """
    
    def step_fn(carry: tuple, t: int) -> tuple[tuple, dict]:
        """Single step for scan."""
        state, key, cumulative_reward, discount_power = carry
        
        # Split key
        key, subkey = jax.random.split(key)
        
        # Simulate one step
        next_state, reward, exog, decision = simulate_step(
            model, policy_fn, state, policy_params, subkey, t
        )
        
        # Update cumulative reward
        discounted_reward = reward * discount_power
        new_cumulative = cumulative_reward + discounted_reward
        new_discount_power = discount_power * discount
        
        # Store outputs
        outputs = {
            'state': state,
            'decision': decision,
            'reward': reward,
            'exog': exog,
        }
        
        new_carry = (next_state, key, new_cumulative, new_discount_power)
        return new_carry, outputs
    
    # Initialize
    init_key, key = jax.random.split(key)
    init_state = model.init_state(init_key)
    init_carry = (init_state, key, jnp.array(0.0), jnp.array(1.0))
    
    # Scan over time steps
    final_carry, trajectory = jax.lax.scan(
        step_fn,
        init_carry,
        jnp.arange(horizon)
    )
    
    final_state, _, total_reward, _ = final_carry
    
    # Add final state to trajectory
    trajectory['states'] = jnp.concatenate([
        trajectory['state'],
        final_state[None, :]
    ])
    trajectory['total_reward'] = total_reward
    
    # Remove intermediate 'state' (we have 'states' now)
    del trajectory['state']
    
    return trajectory


@partial(jax.jit, static_argnums=(0, 1))
def batch_simulate_trajectories(
    model: Model,
    policy_fn: PolicyFn,
    policy_params: PyTree,
    keys: PRNGKeyArray,
    horizon: int,
    discount: float = 1.0,
) -> dict[str, Array]:
    """Simulate multiple trajectories in parallel using vmap.
    
    This is extremely efficient - all trajectories run in parallel on GPU.
    
    Args:
        model: Problem model.
        policy_fn: Policy function.
        policy_params: Policy parameters.
        keys: Batch of random keys with shape (n_trajectories,).
        horizon: Number of time steps.
        discount: Discount factor.
    
    Returns:
        Dictionary with batched trajectories:
            - states: Shape (n_trajectories, horizon+1, state_dim)
            - decisions: Shape (n_trajectories, horizon, action_dim)
            - rewards: Shape (n_trajectories, horizon)
            - total_reward: Shape (n_trajectories,)
            - mean_reward: Scalar - average across trajectories
            - std_reward: Scalar - standard deviation
    
    Example:
        >>> n_trajectories = 100
        >>> key = jax.random.PRNGKey(0)
        >>> keys = jax.random.split(key, n_trajectories)
        >>> results = batch_simulate_trajectories(
        ...     model, policy_fn, policy_params, keys, horizon=100
        ... )
        >>> print(f"Mean reward: {results['mean_reward']:.2f}")
    """
    # Vectorize simulate_trajectory over keys
    batch_sim = jax.vmap(
        lambda k: simulate_trajectory(
            model, policy_fn, policy_params, k, horizon, discount
        )
    )
    
    # Run all trajectories in parallel
    trajectories = batch_sim(keys)
    
    # Add summary statistics
    trajectories['mean_reward'] = jnp.mean(trajectories['total_reward'])
    trajectories['std_reward'] = jnp.std(trajectories['total_reward'])
    
    return trajectories


def optimize_policy(
    model: Model,
    policy_fn: PolicyFn,
    init_params: PyTree,
    optimizer: Any,  # optax optimizer
    key: Key,
    n_iterations: int,
    horizon: int,
    batch_size: int = 32,
) -> tuple[PyTree, list[float]]:
    """Optimize policy using policy gradient.
    
    This is a simple policy gradient implementation that:
    1. Simulates trajectories
    2. Computes gradient of expected return
    3. Updates policy parameters with optimizer
    
    Args:
        model: Problem model.
        policy_fn: Policy function.
        init_params: Initial policy parameters.
        optimizer: Optax optimizer.
        key: Random key.
        n_iterations: Number of optimization iterations.
        horizon: Trajectory length.
        batch_size: Number of trajectories per iteration.
    
    Returns:
        Tuple of (optimized_params, loss_history).
    
    Example:
        >>> import optax
        >>> optimizer = optax.adam(learning_rate=1e-3)
        >>> params, losses = optimize_policy(
        ...     model, policy_fn, init_params, optimizer,
        ...     key, n_iterations=1000, horizon=100
        ... )
    """
    opt_state = optimizer.init(init_params)
    params = init_params
    loss_history = []
    
    def loss_fn(params: PyTree, key: Key) -> Float[Array, ""]:
        """Negative expected return (to minimize)."""
        keys = jax.random.split(key, batch_size)
        results = batch_simulate_trajectories(
            model, policy_fn, params, keys, horizon
        )
        return -results['mean_reward']  # Negative for minimization
    
    # JIT-compile the update step
    @jax.jit
    def update_step(
        params: PyTree,
        opt_state: Any,
        key: Key,
    ) -> tuple[PyTree, Any, float]:
        """Single optimization step."""
        loss, grads = jax.value_and_grad(loss_fn)(params, key)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    # Optimization loop
    for i in range(n_iterations):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = update_step(params, opt_state, subkey)
        loss_history.append(float(loss))
        
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")
    
    return params, loss_history


# Example usage
if __name__ == "__main__":
    print("JAX-Native Base Protocols")
    print("=" * 50)
    print("\nKey features:")
    print("- Type-safe with jaxtyping")
    print("- Runtime checks with chex")
    print("- JIT-compilable functions")
    print("- GPU/TPU compatible")
    print("- Functional programming paradigm")
    print("\nSee energy_storage_model_jax.py for concrete implementation.")
