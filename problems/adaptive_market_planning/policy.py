"""JAX-native policies for Adaptive Market Planning.

This module implements step size selection policies for gradient-based learning:
- Harmonic rule: step_size = theta / (theta + t - 1)
- Kesten's rule: step_size = theta / (theta + counter - 1) - adapts to sign changes
- Constant rule: step_size = theta
- Neural policy: learns optimal step size from state
"""

from typing import Optional, List, Any
from functools import partial
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
import jax
import jax.numpy as jnp
from flax import nnx


# Type aliases
State = Float[Array, "2"]
Decision = Float[Array, "1"]
Key = PRNGKeyArray


class HarmonicStepPolicy:
    """Harmonic step size rule: alpha_t = theta / (theta + t - 1).

    Classic diminishing step size that decreases with time.
    Guarantees convergence under standard conditions.

    Example:
        >>> policy = HarmonicStepPolicy(theta=1.0)
        >>> # At t=1: step_size = 1.0 / (1.0 + 0) = 1.0
        >>> # At t=2: step_size = 1.0 / (1.0 + 1) = 0.5
        >>> # At t=3: step_size = 1.0 / (1.0 + 2) = 0.33
    """

    def __init__(self, theta: float = 1.0, start_time: int = 1) -> None:
        """Initialize policy.

        Args:
            theta: Step size parameter (larger = slower decay).
            start_time: Starting time index (usually 1).
        """
        self.theta = theta
        self.start_time = start_time
        self.current_time = start_time

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Compute step size using harmonic rule.

        Args:
            params: Unused (for consistency with other policies).
            state: Current state (unused, time tracked internally).
            key: Random key (unused).

        Returns:
            Step size decision.
        """
        # Note: In practice, time should be passed as an argument
        # For now, we use a simple time-independent formulation
        # that can be adapted during simulation
        step_size = self.theta / (self.theta + self.current_time - 1)
        return jnp.array([step_size])

    def reset(self) -> None:
        """Reset time counter."""
        self.current_time = self.start_time

    def increment_time(self) -> None:
        """Increment time counter."""
        self.current_time += 1


class KestenStepPolicy:
    """Kesten's rule: alpha_t = theta / (theta + counter - 1).

    Adaptive step size that decreases based on the number of sign changes
    in the gradient. More robust than harmonic rule as it adapts to
    oscillations around the optimum.

    Example:
        >>> policy = KestenStepPolicy(theta=1.0)
        >>> # If counter=1: step_size = 1.0 / (1.0 + 0) = 1.0
        >>> # If counter=5: step_size = 1.0 / (1.0 + 4) = 0.2
    """

    def __init__(self, theta: float = 1.0) -> None:
        """Initialize policy.

        Args:
            theta: Step size parameter (larger = slower decay).
        """
        self.theta = theta

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Compute step size using Kesten's rule.

        Args:
            params: Unused.
            state: Current state [order_quantity, counter].
            key: Random key (unused).

        Returns:
            Step size decision.
        """
        counter = state[1]

        # Kesten's rule: step size decreases with sign changes
        # Add 1 to counter to avoid division by zero at start
        step_size = self.theta / (self.theta + jnp.maximum(counter, 1.0) - 1.0)

        return jnp.array([step_size])


class ConstantStepPolicy:
    """Constant step size rule: alpha_t = theta for all t.

    Simple policy that uses fixed step size. May not converge but can be
    useful for tracking time-varying optima.

    Example:
        >>> policy = ConstantStepPolicy(theta=0.1)
        >>> # Always returns step_size = 0.1
    """

    def __init__(self, theta: float = 0.1) -> None:
        """Initialize policy.

        Args:
            theta: Fixed step size.
        """
        self.theta = theta

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
    ) -> Decision:
        """Return constant step size.

        Args:
            params: Unused.
            state: Current state (unused).
            key: Random key (unused).

        Returns:
            Step size decision.
        """
        return jnp.array([self.theta])


class AdaptiveStepPolicy:
    """Adaptive step size based on recent gradient magnitudes.

    Increases step size when gradients are consistent, decreases when
    they oscillate. Uses exponential moving average of gradient magnitudes.

    Example:
        >>> policy = AdaptiveStepPolicy(base_theta=0.1, smoothing=0.9)
    """

    def __init__(
        self,
        base_theta: float = 0.1,
        smoothing: float = 0.9,
        min_step: float = 0.001,
        max_step: float = 1.0
    ) -> None:
        """Initialize policy.

        Args:
            base_theta: Base step size.
            smoothing: Smoothing factor for gradient EMA (0-1).
            min_step: Minimum allowed step size.
            max_step: Maximum allowed step size.
        """
        self.base_theta = base_theta
        self.smoothing = smoothing
        self.min_step = min_step
        self.max_step = max_step
        self.gradient_ema = 0.0

    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        gradient: float = 0.0
    ) -> Decision:
        """Compute adaptive step size.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).
            gradient: Current gradient value (must be provided).

        Returns:
            Step size decision.
        """
        # Update EMA of gradient magnitude
        self.gradient_ema = (
            self.smoothing * self.gradient_ema +
            (1 - self.smoothing) * abs(gradient)
        )

        # Adapt step size based on gradient magnitude
        # Larger gradients -> smaller steps (near boundary)
        # Smaller gradients -> larger steps (converging)
        adapted_step = self.base_theta / (1.0 + self.gradient_ema)

        # Clip to valid range
        step_size = jnp.clip(adapted_step, self.min_step, self.max_step)

        return jnp.array([step_size])


class NeuralStepPolicy(nnx.Module):
    """Neural network policy for learning optimal step sizes.

    Maps [order_quantity, counter] -> step_size using a neural network.
    Can learn complex adaptive strategies beyond hand-crafted rules.

    Example:
        >>> policy = NeuralStepPolicy(hidden_dims=[16, 8], rngs=nnx.Rngs(0))
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        rngs: Optional[nnx.Rngs] = None,
        min_step: float = 0.001,
        max_step: float = 1.0
    ) -> None:
        """Initialize neural network policy.

        Args:
            hidden_dims: List of hidden layer dimensions.
            rngs: Flax NNX random number generator state.
            min_step: Minimum allowed step size.
            max_step: Maximum allowed step size.
        """
        if hidden_dims is None:
            hidden_dims = [16, 8]
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.min_step = min_step
        self.max_step = max_step

        # Build network: 2 inputs -> hidden layers -> 1 output
        layers: List[Any] = []
        prev_dim = 2

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim

        layers.append(nnx.Linear(prev_dim, 1, rngs=rngs))

        self.layers = layers

    def __call__(self, state: State, key: Key) -> Decision:
        """Get step size from neural network.

        Args:
            state: Current state [order_quantity, counter].
            key: Random key (unused).

        Returns:
            Step size decision.
        """
        x = state

        # Forward pass through hidden layers
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)

        # Output layer with sigmoid to map to [0, 1]
        raw_output = self.layers[-1](x)[0]
        normalized = nnx.sigmoid(raw_output)

        # Scale to [min_step, max_step]
        step_size = self.min_step + (self.max_step - self.min_step) * normalized

        return jnp.array([step_size])
