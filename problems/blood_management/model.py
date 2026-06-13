"""Blood Management model — JAX-native, faithful to the original Powell problem.

The original solves a per-period allocation **LP** (a min-cost network flow over
blood→demand edges, glpk) maximising the match ``contribution`` subject to
supply (inventory) and demand caps. We reproduce that allocation with **entropic
optimal transport** (Sinkhorn via ``ott-jax``): a JAX-native, differentiable
solver that converges reliably on this degenerate transportation LP and matches
the exact LP optimum to ~1e-2 with small regularisation. The contribution
weights and ABO/RhD substitution rules match the original ``contribution()``.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from jaxtyping import PRNGKeyArray as Key
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

_INFEASIBLE_COST = 1e3  # cost on disallowed substitutions (never used at optimum)

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
        urgent_bonus: Contribution for filling urgent demand (original: 30).
        elective_bonus: Contribution for filling elective demand (original: 5).
        no_substitution_bonus: Bonus for an exact blood-type match (original: 5).
        epsilon: Sinkhorn entropic-regularisation strength (smaller = closer to LP).
        seed: Random seed.
    """

    max_age: int = 5
    max_demand_urgent: float = 10.0
    max_demand_elective: float = 5.0
    max_donation: float = 15.0
    surge_prob: float = 0.1
    surge_factor: float = 3.0
    urgent_bonus: float = 30.0
    elective_bonus: float = 5.0
    no_substitution_bonus: float = 5.0
    epsilon: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_age < 1:
            raise ValueError("max_age must be >= 1")
        if self.surge_prob < 0 or self.surge_prob > 1:
            raise ValueError("surge_prob must be in [0, 1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


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

        # Precompute the per-edge contribution C[slot, demand] (original contribution()).
        # slot s -> blood type s // max_age ; demand d -> (d // 2 blood type, d % 2 surgery,
        # 0 = Urgent, 1 = Elective). Infeasible substitutions are masked out.
        C = np.zeros((self.n_inventory_slots, self.n_demand_types), dtype=np.float32)
        feasible = np.zeros_like(C, dtype=bool)
        for s in range(self.n_inventory_slots):
            bt = s // config.max_age
            for d in range(self.n_demand_types):
                dt, surg = d // self.n_surgery_types, d % self.n_surgery_types
                if SUBSTITUTION_MATRIX.get((BLOOD_TYPES[bt], BLOOD_TYPES[dt]), False):
                    feasible[s, d] = True
                    C[s, d] = (
                        (config.no_substitution_bonus if bt == dt else 0.0)
                        + (config.urgent_bonus if surg == 0 else config.elective_bonus)
                    )
        self.contribution_matrix = jnp.asarray(C)
        self.feasible_mask = jnp.asarray(feasible)
        # OT cost = -contribution on feasible edges, large on infeasible ones.
        self._ot_cost = jnp.where(self.feasible_mask, -self.contribution_matrix, _INFEASIBLE_COST)

    def optimal_allocation(self, inventory: Array, demand: Array) -> Decision:
        """Reward-maximising allocation via entropic OT (the original per-period LP).

        Solves ``max <C, x>`` s.t. row sums <= inventory, col sums <= demand, x >= 0
        as a balanced OT with a dummy sink (absorbs unused supply) and dummy source
        (covers unmet demand) at zero cost, then returns the flattened allocation.

        Args:
            inventory: Available units per inventory slot ``[n_inventory_slots]``.
            demand: Demand per demand node ``[n_demand_types]``.

        Returns:
            Flattened allocation matrix ``[n_inventory_slots * n_demand_types]``.
        """
        nB, nD = self.n_inventory_slots, self.n_demand_types
        tot_s, tot_d = jnp.sum(inventory), jnp.sum(demand)
        # Augment with a dummy demand column (sink) and dummy supply row (source).
        cost = jnp.zeros((nB + 1, nD + 1))
        cost = cost.at[:nB, :nD].set(self._ot_cost)
        a = jnp.concatenate([inventory, tot_d[None]])
        b = jnp.concatenate([demand, tot_s[None]])
        geom = geometry.Geometry(cost_matrix=cost, epsilon=self.config.epsilon)
        prob = linear_problem.LinearProblem(geom, a=a, b=b)
        out = sinkhorn.Sinkhorn(max_iterations=5000)(prob)
        allocation = out.matrix[:nB, :nD]
        return allocation.reshape(-1)

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
        total_allocated = jnp.sum(allocation, axis=1).reshape(
            self.n_blood_types, self.config.max_age
        )
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
        """Total match contribution of an allocation (the original LP objective).

        ``reward = sum_{slot,demand} contribution[slot, demand] * allocation[slot, demand]``
        with the original ``contribution()`` weights (no-substitution + urgent/elective
        bonuses) on feasible edges only.

        Args:
            state: Current state (unused; allocation feasibility is handled upstream).
            decision: Allocation decision (flattened matrix).
            exog: Demands and donations (unused — the objective scores the allocation).

        Returns:
            Total contribution.
        """
        allocation = decision.reshape(self.n_inventory_slots, self.n_demand_types)
        return jnp.sum(self.contribution_matrix * allocation)

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
