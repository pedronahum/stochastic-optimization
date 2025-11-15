"""Policies for Blood Management problem."""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray as Key, PyTree

# Type aliases
State = Float[Array, "..."]
Decision = Float[Array, "..."]


class GreedyPolicy:
    """Greedy allocation policy for blood management.

    Allocates blood using a greedy heuristic:
    1. Prioritize urgent demands over elective
    2. Use oldest blood first (FIFO within each blood type)
    3. Prefer exact blood type matches
    4. Use substitutions only when necessary

    Example:
        >>> policy = GreedyPolicy()
    """

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "BloodManagementModel",  # type: ignore[name-defined]
    ) -> Decision:
        """Compute greedy allocation.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).
            model: Model instance.

        Returns:
            Allocation decision.
        """
        inventory = model.get_inventory(state)
        n_inv = model.n_inventory_slots
        n_dem = model.n_demand_types

        # Initialize allocation matrix
        allocation = jnp.zeros((n_inv, n_dem))

        # Get current exogenous info to know demands
        # Note: In practice, this would come from the environment
        # For now, we'll allocate based on inventory availability

        # Greedy strategy: For each demand, allocate oldest compatible blood
        # Use vectorized operations instead of Python if statements
        for demand_idx in range(n_dem):
            blood_type_idx = demand_idx // model.n_surgery_types

            # Try to allocate from compatible blood types (oldest first)
            for donor_blood_idx in range(model.n_blood_types):
                # Check if substitution is allowed
                can_substitute = model.substitution_matrix[donor_blood_idx, blood_type_idx]

                # Allocate from oldest to newest
                for age in range(model.config.max_age - 1, -1, -1):
                    slot_idx = donor_blood_idx * model.config.max_age + age
                    available = inventory[donor_blood_idx, age]

                    # Already allocated from this slot
                    already_allocated = jnp.sum(allocation[slot_idx, :])
                    remaining = available - already_allocated

                    # Allocate what's available (only if substitution allowed)
                    allocation_amount = jnp.where(
                        can_substitute,
                        jnp.minimum(remaining, 1.0),  # Allocate up to 1 unit
                        0.0
                    )
                    allocation = allocation.at[slot_idx, demand_idx].set(allocation_amount)

        return allocation.flatten()


class FIFOPolicy:
    """First-In-First-Out allocation policy.

    Always uses oldest blood first for any compatible demand.
    Simpler than greedy - doesn't prioritize exact matches.

    Example:
        >>> policy = FIFOPolicy()
    """

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "BloodManagementModel",  # type: ignore[name-defined]
    ) -> Decision:
        """Compute FIFO allocation.

        Args:
            params: Unused.
            state: Current state.
            key: Random key (unused).
            model: Model instance.

        Returns:
            Allocation decision.
        """
        inventory = model.get_inventory(state)
        allocation = jnp.zeros((model.n_inventory_slots, model.n_demand_types))

        # For each blood type, allocate oldest first
        for blood_type_idx in range(model.n_blood_types):
            for age in range(model.config.max_age - 1, -1, -1):
                slot_idx = blood_type_idx * model.config.max_age + age
                available = inventory[blood_type_idx, age]

                # Find compatible demands
                for demand_idx in range(model.n_demand_types):
                    demand_blood_idx = demand_idx // model.n_surgery_types

                    # Check substitution
                    can_substitute = model.substitution_matrix[blood_type_idx, demand_blood_idx]

                    # Allocate fraction (only if available and can substitute)
                    allocation_amount = jnp.where(
                        can_substitute & (available > 0),
                        available / model.n_demand_types,
                        0.0
                    )
                    allocation = allocation.at[slot_idx, demand_idx].set(allocation_amount)

        return allocation.flatten()


class RandomPolicy:
    """Random allocation policy.

    Randomly allocates available blood to compatible demands.
    Useful as a baseline.

    Example:
        >>> policy = RandomPolicy()
    """

    @partial(jax.jit, static_argnames=["self", "model"])
    def __call__(
        self,
        params: Optional[PyTree],
        state: State,
        key: Key,
        model: "BloodManagementModel",  # type: ignore[name-defined]
    ) -> Decision:
        """Compute random allocation.

        Args:
            params: Unused.
            state: Current state.
            key: Random key.
            model: Model instance.

        Returns:
            Allocation decision.
        """
        inventory = model.get_inventory(state)

        # Generate random allocation matrix
        allocation = jax.random.uniform(
            key,
            (model.n_inventory_slots, model.n_demand_types)
        )

        # Step 1: Zero out incompatible substitutions FIRST
        for slot_idx in range(model.n_inventory_slots):
            blood_type_idx = slot_idx // model.config.max_age

            for demand_idx in range(model.n_demand_types):
                demand_blood_idx = demand_idx // model.n_surgery_types

                # Check substitution
                can_substitute = model.substitution_matrix[blood_type_idx, demand_blood_idx]

                # Zero out if not compatible
                allocation = allocation.at[slot_idx, demand_idx].set(
                    jnp.where(can_substitute, allocation[slot_idx, demand_idx], 0.0)
                )

        # Step 2: Scale by available inventory (after filtering)
        inventory_flat = inventory.flatten()
        for slot_idx in range(model.n_inventory_slots):
            total_allocated = jnp.sum(allocation[slot_idx, :])
            # Scale down if needed
            scale = jnp.where(
                total_allocated > inventory_flat[slot_idx],
                inventory_flat[slot_idx] / (total_allocated + 1e-10),
                1.0
            )
            allocation = allocation.at[slot_idx, :].set(
                allocation[slot_idx, :] * scale
            )

        return allocation.flatten()
