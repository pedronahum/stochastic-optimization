"""Blood Management model - JAX-native implementation."""

from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray as Key

# Type aliases
State = Float[Array, "..."]  # [inventory (blood_types × max_age), time]
Decision = Float[Array, "inventory demands"]  # Allocation matrix
Reward = Float[Array, ""]  # Contribution from allocations

# Blood types in order
BLOOD_TYPES = ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]

# Blood substitution matrix: can blood_type_i substitute for blood_type_j?
# Universal donor: O- can substitute for all
# Type-specific rules based on Rh factor and antigens
SUBSTITUTION_MATRIX = {
    ("O-", "O-"): True,   ("O-", "O+"): True,   ("O-", "A-"): True,   ("O-", "A+"): True,
    ("O-", "B-"): True,   ("O-", "B+"): True,   ("O-", "AB-"): True,  ("O-", "AB+"): True,
    ("O+", "O+"): True,   ("O+", "A+"): True,   ("O+", "B+"): True,   ("O+", "AB+"): True,
    ("A-", "A-"): True,   ("A-", "A+"): True,   ("A-", "AB-"): True,  ("A-", "AB+"): True,
    ("A+", "A+"): True,   ("A+", "AB+"): True,
    ("B-", "B-"): True,   ("B-", "B+"): True,   ("B-", "AB-"): True,  ("B-", "AB+"): True,
    ("B+", "B+"): True,   ("B+", "AB+"): True,
    ("AB-", "AB-"): True, ("AB-", "AB+"): True,
    ("AB+", "AB+"): True,
}


@dataclass(frozen=True)
class ExogenousInfo:
    """Exogenous information for Blood Management.

    Attributes:
        demand: Demand for each (blood_type, surgery_type) combination.
        donation: New blood donations by blood type.
    """

    demand: Float[Array, "n_demands"]
    donation: Float[Array, "n_blood_types"]


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    ExogenousInfo,
    lambda obj: ((obj.demand, obj.donation), None),
    lambda aux, children: ExogenousInfo(*children),
)


@dataclass(frozen=True)
class BloodManagementConfig:
    """Configuration for Blood Management problem.

    Attributes:
        max_age: Maximum age of blood in days before expiry.
        max_demand_urgent: Maximum units demanded for urgent surgeries per type.
        max_demand_elective: Maximum units demanded for elective surgeries per type.
        max_donation: Maximum units donated per blood type.
        surge_prob: Probability of demand surge.
        surge_factor: Multiplier for demand during surge.
        urgent_bonus: Reward for fulfilling urgent demand.
        elective_bonus: Reward for fulfilling elective demand.
        no_substitution_bonus: Bonus for exact blood type match.
        discard_penalty: Penalty for discarding expired blood.
        shortage_penalty: Penalty for unfulfilled demand.
        seed: Random seed.
    """

    max_age: int = 5
    max_demand_urgent: float = 10.0
    max_demand_elective: float = 5.0
    max_donation: float = 15.0
    surge_prob: float = 0.1
    surge_factor: float = 3.0
    urgent_bonus: float = 100.0
    elective_bonus: float = 50.0
    no_substitution_bonus: float = 10.0
    discard_penalty: float = -50.0
    shortage_penalty: float = -200.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_age < 1:
            raise ValueError("max_age must be >= 1")
        if self.surge_prob < 0 or self.surge_prob > 1:
            raise ValueError("surge_prob must be in [0, 1]")


class BloodManagementModel:
    """Blood Management model with inventory dynamics.

    This model implements blood bank inventory management with:
    - 8 blood types (O-, O+, A-, A+, B-, B+, AB-, AB+)
    - Age-dependent inventory (expires after max_age days)
    - Two surgery types: Urgent and Elective
    - Blood substitution rules
    - Stochastic donations and demands

    State representation: [inventory (8 × max_age), time]
    - inventory[i, j]: units of blood type i with age j days

    Example:
        >>> config = BloodManagementConfig(max_age=5)
        >>> model = BloodManagementModel(config)
        >>> key = jax.random.PRNGKey(42)
        >>> state = model.init_state(key)
    """

    def __init__(self, config: BloodManagementConfig) -> None:
        """Initialize Blood Management model.

        Args:
            config: Problem configuration.
        """
        self.config = config
        self.n_blood_types = len(BLOOD_TYPES)
        self.n_surgery_types = 2  # Urgent, Elective
        self.n_demand_types = self.n_blood_types * self.n_surgery_types
        self.n_inventory_slots = self.n_blood_types * config.max_age

        # Create substitution matrix as JAX array (8x8)
        sub_matrix = jnp.zeros((self.n_blood_types, self.n_blood_types), dtype=bool)
        for i, blood_i in enumerate(BLOOD_TYPES):
            for j, blood_j in enumerate(BLOOD_TYPES):
                sub_matrix = sub_matrix.at[i, j].set(
                    SUBSTITUTION_MATRIX.get((blood_i, blood_j), False)
                )
        self.substitution_matrix = sub_matrix

    def init_state(self, key: Key) -> State:
        """Initialize state with empty inventory.

        Args:
            key: Random key.

        Returns:
            Initial state: [inventory (flattened), time]
        """
        # Initialize with small random inventory
        key1, key2 = jax.random.split(key)

        # Small initial inventory (0-5 units per slot)
        inventory = jax.random.uniform(key1, (self.n_inventory_slots,), maxval=5.0)
        inventory = jnp.floor(inventory)

        time = jnp.array(0.0)

        state = jnp.concatenate([inventory, jnp.array([time])])
        return state

    @partial(jax.jit, static_argnums=(0,))
    def transition(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> State:
        """Age blood, add donations, and apply allocations.

        Args:
            state: Current state.
            decision: Allocation decision (flattened matrix).
            exog: Donations and demands.

        Returns:
            New state after aging, donations, and allocations.
        """
        # Extract inventory and time
        inventory = state[:-1].reshape(self.n_blood_types, self.config.max_age)
        time = state[-1]

        # Step 1: Apply allocations (subtract allocated blood)
        # Decision is flattened: [blood_type_age × demand_type]
        allocation = decision.reshape(self.n_inventory_slots, self.n_demand_types)

        # Sum allocations across all demands for each inventory slot
        total_allocated = jnp.sum(allocation, axis=1).reshape(self.n_blood_types, self.config.max_age)
        inventory_after_allocation = jnp.maximum(0, inventory - total_allocated)

        # Step 2: Age the blood (shift ages by 1, oldest expires)
        inventory_aged = jnp.zeros_like(inventory_after_allocation)

        # Age increases: slot j becomes slot j+1
        for blood_type in range(self.n_blood_types):
            # Shift ages: age 0 → age 1, age 1 → age 2, ..., age (max-1) → discarded
            for age in range(self.config.max_age - 1):
                inventory_aged = inventory_aged.at[blood_type, age + 1].set(
                    inventory_after_allocation[blood_type, age]
                )
            # Age 0 will be filled with donations

        # Step 3: Add new donations (age 0)
        for blood_type in range(self.n_blood_types):
            inventory_aged = inventory_aged.at[blood_type, 0].set(
                exog.donation[blood_type]
            )

        # Increment time
        new_time = time + 1

        # Flatten and concatenate
        new_state = jnp.concatenate([inventory_aged.flatten(), jnp.array([new_time])])
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def reward(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> Reward:
        """Compute reward from allocations.

        Reward components:
        - Bonus for fulfilling urgent/elective demand
        - Bonus for exact blood type match
        - Penalty for blood that will expire
        - Penalty for unmet demand

        Args:
            state: Current state.
            decision: Allocation decision.
            exog: Demands and donations.

        Returns:
            Total reward.
        """
        inventory = state[:-1].reshape(self.n_blood_types, self.config.max_age)
        allocation = decision.reshape(self.n_inventory_slots, self.n_demand_types)

        reward_value: jax.Array = jnp.array(0.0)

        # Reward for fulfilling demands
        for demand_idx in range(self.n_demand_types):
            blood_type_idx = demand_idx // self.n_surgery_types
            surgery_type = demand_idx % self.n_surgery_types  # 0=Urgent, 1=Elective

            # Total allocated to this demand
            allocated_to_demand = jnp.sum(allocation[:, demand_idx])

            # Reward based on surgery type (use jnp.where instead of if)
            bonus = jnp.where(
                surgery_type == 0,
                self.config.urgent_bonus,
                self.config.elective_bonus
            )
            reward_value = reward_value + jnp.minimum(allocated_to_demand, exog.demand[demand_idx]) * bonus

            # Bonus for exact blood type match
            for age in range(self.config.max_age):
                slot_idx = blood_type_idx * self.config.max_age + age
                exact_match_allocation = allocation[slot_idx, demand_idx]
                reward_value = reward_value + exact_match_allocation * self.config.no_substitution_bonus

        # Penalty for unmet demand
        total_allocated_per_demand = jnp.sum(allocation, axis=0)
        unmet_demand = jnp.maximum(0, exog.demand - total_allocated_per_demand)

        # Urgent unmet demand gets higher penalty
        for demand_idx in range(self.n_demand_types):
            surgery_type = demand_idx % self.n_surgery_types
            penalty_mult = jnp.where(surgery_type == 0, 2.0, 1.0)
            reward_value = reward_value + unmet_demand[demand_idx] * self.config.shortage_penalty * penalty_mult

        # Penalty for blood that will be discarded (oldest age after allocation)
        allocation_sum = jnp.sum(allocation, axis=1).reshape(self.n_blood_types, self.config.max_age)
        remaining = jnp.maximum(0, inventory - allocation_sum)
        oldest_blood = remaining[:, -1]  # Last age column
        reward_value = reward_value + jnp.sum(oldest_blood) * self.config.discard_penalty

        return reward_value

    def sample_exogenous(
        self,
        key: Key,
        state: State,
        time: int,
    ) -> ExogenousInfo:
        """Sample demands and donations.

        Args:
            key: Random key.
            state: Current state.
            time: Current time step.

        Returns:
            Exogenous information (demand, donation).
        """
        key1, key2, key3 = jax.random.split(key, 3)

        # Check for surge
        is_surge = jax.random.uniform(key1) < self.config.surge_prob
        surge_multiplier = jnp.where(is_surge, self.config.surge_factor, 1.0)

        # Sample demands: Poisson distribution for each (blood_type, surgery_type)
        urgent_rate = self.config.max_demand_urgent * surge_multiplier
        elective_rate = self.config.max_demand_elective * surge_multiplier

        # Generate all demands at once
        demands_urgent = jax.random.poisson(key2, urgent_rate, shape=(self.n_blood_types,))
        demands_elective = jax.random.poisson(key2, elective_rate, shape=(self.n_blood_types,))

        # Interleave: [urgent0, elective0, urgent1, elective1, ...]
        demand = jnp.zeros(self.n_demand_types)
        for i in range(self.n_blood_types):
            demand = demand.at[i * 2].set(demands_urgent[i])
            demand = demand.at[i * 2 + 1].set(demands_elective[i])

        # Sample donations: Poisson distribution for each blood type
        donation = jax.random.poisson(key3, self.config.max_donation, shape=(self.n_blood_types,))
        donation = donation.astype(jnp.float32)
        demand = demand.astype(jnp.float32)

        return ExogenousInfo(demand=demand, donation=donation)

    def is_valid_decision(self, state: State, decision: Decision) -> jax.Array:
        """Check if allocation decision is feasible.

        Args:
            state: Current state.
            decision: Allocation decision.

        Returns:
            Boolean: True if feasible.
        """
        inventory = state[:-1]
        allocation = decision.reshape(self.n_inventory_slots, self.n_demand_types)

        # Check 1: Allocations are non-negative
        non_negative = jnp.all(allocation >= -1e-6)  # Small tolerance for numerical errors

        # Check 2: Total allocated per slot doesn't exceed inventory
        total_per_slot = jnp.sum(allocation, axis=1)
        within_inventory = jnp.all(total_per_slot <= inventory + 1e-6)  # Small tolerance

        return non_negative & within_inventory

    def get_inventory(self, state: State) -> Float[Array, "n_blood_types max_age"]:
        """Extract inventory matrix from state.

        Args:
            state: Current state.

        Returns:
            Inventory matrix [n_blood_types, max_age].
        """
        return state[:-1].reshape(self.n_blood_types, self.config.max_age)

    def get_time(self, state: State) -> int:
        """Extract time from state.

        Args:
            state: Current state.

        Returns:
            Current time step.
        """
        return int(state[-1])
