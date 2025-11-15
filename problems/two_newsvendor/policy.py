"""JAX-native policies for Two Newsvendor.

This module implements decision policies for both Field and Central agents:
- Newsvendor formula policies (optimal under certain assumptions)
- Bias-adjusted policies
- Neural network policies
"""

from typing import Optional, List, Any
from functools import partial
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
import jax
import jax.numpy as jnp
from flax import nnx


# Type aliases
StateField = Float[Array, "3"]
StateCentral = Float[Array, "7"]
DecisionField = Float[Array, "1"]
DecisionCentral = Float[Array, "1"]
Key = PRNGKeyArray


class NewsvendorFieldPolicy:
    """Newsvendor formula policy for Field agent.

    Uses critical ratio: CR = underage_cost / (overage_cost + underage_cost)
    Requests quantity = estimate + bias_adjustment

    Example:
        >>> from stochopt.problems.two_newsvendor.model import TwoNewsvendorConfig, TwoNewsvendorFieldModel
        >>> config = TwoNewsvendorConfig()
        >>> model = TwoNewsvendorFieldModel(config)
        >>> policy = NewsvendorFieldPolicy(model, bias_adjustment=0.0)
    """

    def __init__(
        self,
        model: "TwoNewsvendorFieldModel",  # type: ignore[name-defined]
        bias_adjustment: float = 0.0
    ) -> None:
        """Initialize policy.

        Args:
            model: Field model instance.
            bias_adjustment: Fixed bias to add to estimate.
        """
        self.model = model
        self.bias_adjustment = bias_adjustment

        # Compute critical ratio
        self.critical_ratio = (
            model.config.underage_cost_field /
            (model.config.overage_cost_field + model.config.underage_cost_field)
        )

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: StateField,
        key: Key,
    ) -> DecisionField:
        """Get decision based on newsvendor formula.

        Args:
            params: Unused.
            state: Current state [estimate, source_bias, central_bias].
            key: Random key (unused).

        Returns:
            Quantity to request.
        """
        estimate = state[0]

        # Simple newsvendor: request estimate + bias adjustment
        # For uniform demand, optimal is at critical ratio quantile
        # Here we use estimate as proxy
        quantity = jnp.maximum(0.0, estimate + self.bias_adjustment)

        return jnp.array([quantity])


class BiasAdjustedFieldPolicy:
    """Bias-adjusted policy for Field agent.

    Adjusts request based on learned biases from both source and central.

    Example:
        >>> policy = BiasAdjustedFieldPolicy(model, use_source_bias=True, use_central_bias=True)
    """

    def __init__(
        self,
        model: "TwoNewsvendorFieldModel",  # type: ignore[name-defined]
        use_source_bias: bool = True,
        use_central_bias: bool = True,
    ) -> None:
        """Initialize policy.

        Args:
            model: Field model instance.
            use_source_bias: Whether to adjust for source bias.
            use_central_bias: Whether to adjust for central bias.
        """
        self.model = model
        self.use_source_bias = use_source_bias
        self.use_central_bias = use_central_bias

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: StateField,
        key: Key,
    ) -> DecisionField:
        """Get decision with bias adjustment.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).

        Returns:
            Quantity to request.
        """
        estimate, source_bias, central_bias = state[0], state[1], state[2]

        # Start with estimate
        quantity = estimate

        # Adjust for source bias (estimate tends to be off)
        if self.use_source_bias:
            quantity = quantity - source_bias

        # Adjust for central bias (central tends to allocate more/less than requested)
        if self.use_central_bias:
            quantity = quantity - central_bias

        quantity = jnp.maximum(0.0, quantity)

        return jnp.array([quantity])


class NewsvendorCentralPolicy:
    """Newsvendor formula policy for Central agent.

    Allocates based on own estimate and Field's request.

    Example:
        >>> policy = NewsvendorCentralPolicy(model, trust_field=0.5)
    """

    def __init__(
        self,
        model: "TwoNewsvendorCentralModel",  # type: ignore[name-defined]
        trust_field: float = 0.5,
    ) -> None:
        """Initialize policy.

        Args:
            model: Central model instance.
            trust_field: Weight for Field's request vs own estimate (0-1).
        """
        self.model = model
        self.trust_field = jnp.clip(trust_field, 0.0, 1.0)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: StateCentral,
        key: Key,
        field_request: Float[Array, ""],
    ) -> DecisionCentral:
        """Get decision based on weighted estimate.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).
            field_request: Quantity requested by Field.

        Returns:
            Quantity to allocate.
        """
        estimate = state[4]  # Central's estimate

        # Weighted combination of Field's request and own estimate
        quantity = (
            self.trust_field * field_request +
            (1.0 - self.trust_field) * estimate
        )

        quantity = jnp.maximum(0.0, quantity)

        return jnp.array([quantity])


class BiasAdjustedCentralPolicy:
    """Bias-adjusted policy for Central agent.

    Adjusts allocation based on learned biases.

    Example:
        >>> policy = BiasAdjustedCentralPolicy(model, trust_field=0.5)
    """

    def __init__(
        self,
        model: "TwoNewsvendorCentralModel",  # type: ignore[name-defined]
        trust_field: float = 0.5,
    ) -> None:
        """Initialize policy.

        Args:
            model: Central model instance.
            trust_field: Weight for Field's request vs own estimate.
        """
        self.model = model
        self.trust_field = jnp.clip(trust_field, 0.0, 1.0)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: StateCentral,
        key: Key,
        field_request: Float[Array, ""],
    ) -> DecisionCentral:
        """Get decision with bias adjustment.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).
            field_request: Quantity requested by Field.

        Returns:
            Quantity to allocate.
        """
        # Unpack state
        _, field_bias, _, _, estimate, source_bias, _ = (
            state[0], state[1], state[2], state[3], state[4], state[5], state[6]
        )

        # Adjust Field's request for known bias
        adjusted_field_request = field_request - field_bias

        # Adjust own estimate for known bias
        adjusted_estimate = estimate - source_bias

        # Weighted combination
        quantity = (
            self.trust_field * adjusted_field_request +
            (1.0 - self.trust_field) * adjusted_estimate
        )

        quantity = jnp.maximum(0.0, quantity)

        return jnp.array([quantity])


class NeuralFieldPolicy(nnx.Module):
    """Neural network policy for Field agent.

    Maps [estimate, source_bias, central_bias] -> quantity_requested

    Example:
        >>> policy = NeuralFieldPolicy(hidden_dims=[16, 8], rngs=nnx.Rngs(0))
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        rngs: Optional[nnx.Rngs] = None
    ) -> None:
        """Initialize neural network policy.

        Args:
            hidden_dims: List of hidden layer dimensions.
            rngs: Flax NNX random number generator state.
        """
        if hidden_dims is None:
            hidden_dims = [16, 8]
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Build network: 3 inputs -> hidden layers -> 1 output
        layers: List[Any] = []
        prev_dim = 3

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim

        layers.append(nnx.Linear(prev_dim, 1, rngs=rngs))

        self.layers = layers

    def __call__(self, state: StateField, key: Key) -> DecisionField:
        """Get decision from neural network.

        Args:
            state: Current state.
            key: Random key (unused).

        Returns:
            Quantity to request.
        """
        x = state

        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)

        # Output layer with ReLU to ensure non-negative
        quantity = nnx.relu(self.layers[-1](x)[0])

        return jnp.array([quantity])


class NeuralCentralPolicy(nnx.Module):
    """Neural network policy for Central agent.

    Maps [state + field_request] -> quantity_allocated

    Example:
        >>> policy = NeuralCentralPolicy(hidden_dims=[16, 8], rngs=nnx.Rngs(0))
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        rngs: Optional[nnx.Rngs] = None
    ) -> None:
        """Initialize neural network policy.

        Args:
            hidden_dims: List of hidden layer dimensions.
            rngs: Flax NNX random number generator state.
        """
        if hidden_dims is None:
            hidden_dims = [16, 8]
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Build network: 8 inputs (7 state + 1 field_request) -> hidden -> 1 output
        layers: List[Any] = []
        prev_dim = 8

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim

        layers.append(nnx.Linear(prev_dim, 1, rngs=rngs))

        self.layers = layers

    def __call__(
        self,
        state: StateCentral,
        field_request: Float[Array, ""],
        key: Key
    ) -> DecisionCentral:
        """Get decision from neural network.

        Args:
            state: Current state.
            field_request: Quantity requested by Field.
            key: Random key (unused).

        Returns:
            Quantity to allocate.
        """
        # Concatenate state and field_request
        x = jnp.concatenate([state, jnp.array([field_request])])

        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)

        # Output layer with ReLU
        quantity = nnx.relu(self.layers[-1](x)[0])

        return jnp.array([quantity])


class AlwaysAllocateRequestedPolicy:
    """Baseline Central policy: always allocate exactly what Field requests.

    Example:
        >>> policy = AlwaysAllocateRequestedPolicy()
    """

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        params: Optional[PyTree],
        state: StateCentral,
        key: Key,
        field_request: Float[Array, ""],
    ) -> DecisionCentral:
        """Always allocate exactly what was requested.

        Args:
            params: Unused.
            state: Current state (unused).
            key: Random key (unused).
            field_request: Quantity requested by Field.

        Returns:
            Quantity to allocate (equals field_request).
        """
        return jnp.array([field_request])
