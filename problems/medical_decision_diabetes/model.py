"""JAX-native Medical Decision Diabetes Model.

This module implements a Bayesian multi-armed bandit problem for diabetes treatment
selection. The agent learns the effectiveness of 5 different drugs through patient
trials and Bayesian belief updates:

Drugs (arms):
- 0: Metformin (M)
- 1: Sensitizer (Sens)
- 2: Secretagogue (Secr)
- 3: Alpha-glucosidase inhibitor (AGI)
- 4: Peptide Analog (PA)

State: For each drug: [mu_empirical, beta (precision), N_trials]
  - mu_empirical: Posterior mean of A1C reduction
  - beta: Posterior precision (1/variance²)
  - N_trials: Number of times drug was tried

Decision: Which drug to try (integer 0-4)
Exogenous: Observed A1C reduction (true_mu + measurement_noise)
"""

from typing import NamedTuple
from functools import partial
from jaxtyping import Array, Float, Int, PRNGKeyArray
import jax
import jax.numpy as jnp
import chex


# Type aliases
State = Float[Array, "5 3"]  # 5 drugs × [mu, beta, N]
Decision = Int[Array, ""]  # Drug index 0-4
Reward = Float[Array, ""]  # A1C reduction
Key = PRNGKeyArray
DrugIndex = int  # For type clarity


class ExogenousInfo(NamedTuple):
    """Exogenous information for Medical Decision Diabetes.

    Attributes:
        reduction: Observed A1C reduction (with measurement noise).
        true_mu: True mean effectiveness of the drug (for evaluation).
        measurement_precision: Precision of measurement (1/sigma_w²).
    """
    reduction: Float[Array, ""]
    true_mu: Float[Array, ""]
    measurement_precision: Float[Array, ""]


@chex.dataclass(frozen=True)
class MedicalDecisionDiabetesConfig:
    """Configuration for Medical Decision Diabetes model.

    Attributes:
        n_drugs: Number of drug options.
        initial_mu: Initial belief mean for all drugs.
        initial_sigma: Initial belief std dev for all drugs.
        measurement_sigma: Std dev of measurement noise.
        true_mu: True mean effectiveness for each drug.
        true_sigma: True std dev of effectiveness (if sampled).
        use_fixed_truth: If True, use true_mu directly; else sample from normal.
    """
    n_drugs: int = 5
    initial_mu: float = 0.5
    initial_sigma: float = 0.2
    measurement_sigma: float = 0.05
    # Default true effectiveness values (can be customized)
    true_mu_M: float = 0.6  # Metformin
    true_mu_Sens: float = 0.55  # Sensitizer
    true_mu_Secr: float = 0.5  # Secretagogue
    true_mu_AGI: float = 0.45  # AGI
    true_mu_PA: float = 0.7  # Peptide Analog (best)
    true_sigma: float = 0.1
    use_fixed_truth: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_drugs <= 0:
            raise ValueError(f"n_drugs must be positive, got {self.n_drugs}")
        if self.initial_sigma <= 0:
            raise ValueError(
                f"initial_sigma must be positive, got {self.initial_sigma}"
            )
        if self.measurement_sigma <= 0:
            raise ValueError(
                f"measurement_sigma must be positive, got {self.measurement_sigma}"
            )

    def get_true_mu_array(self) -> jax.Array:
        """Get array of true mu values for all drugs."""
        return jnp.array([
            self.true_mu_M,
            self.true_mu_Sens,
            self.true_mu_Secr,
            self.true_mu_AGI,
            self.true_mu_PA,
        ])


class MedicalDecisionDiabetesModel:
    """JAX-native model for Medical Decision Diabetes problem.

    Bayesian multi-armed bandit for diabetes treatment selection:
    - Maintain belief (mu, beta, N) for each drug
    - Select drug, observe A1C reduction
    - Update belief using Bayesian conjugate prior (Normal-Gamma)
    - Goal: Maximize cumulative A1C reduction

    Example:
        >>> config = MedicalDecisionDiabetesConfig()
        >>> model = MedicalDecisionDiabetesModel(config)
        >>> key = jax.random.PRNGKey(0)
        >>> state = model.init_state(key)
        >>> decision = 0  # Try Metformin
    """

    def __init__(self, config: MedicalDecisionDiabetesConfig) -> None:
        """Initialize model.

        Args:
            config: Model configuration.
        """
        self.config = config

        # Sample or set true drug effectiveness
        if config.use_fixed_truth:
            self.true_mu = config.get_true_mu_array()
        else:
            # Will be sampled in init_state
            self.true_mu = jnp.zeros(config.n_drugs)

    def init_state(self, key: Key) -> State:
        """Initialize state with prior beliefs.

        Args:
            key: Random key.

        Returns:
            Initial state [5 × 3]: each row is [mu_empirical, beta, N_trials].
        """
        # Sample true mu if not using fixed values
        if not self.config.use_fixed_truth:
            self.true_mu = jax.random.normal(key, (self.config.n_drugs,)) * \
                          self.config.true_sigma + self.config.initial_mu

        # Initialize belief: same prior for all drugs
        initial_beta = 1.0 / (self.config.initial_sigma ** 2)

        # State: [mu_empirical, beta, N_trials] for each drug
        state = jnp.array([
            [self.config.initial_mu, initial_beta, 0.0],  # Metformin
            [self.config.initial_mu, initial_beta, 0.0],  # Sensitizer
            [self.config.initial_mu, initial_beta, 0.0],  # Secretagogue
            [self.config.initial_mu, initial_beta, 0.0],  # AGI
            [self.config.initial_mu, initial_beta, 0.0],  # Peptide Analog
        ])

        return state

    @partial(jax.jit, static_argnums=(0,))
    def transition(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> State:
        """Update belief via Bayesian update for selected drug.

        Uses conjugate Normal-Gamma prior update:
        - Prior: N(mu | mu_0, 1/beta_0), Gamma(beta | ...)
        - Likelihood: N(W | mu, 1/beta_W)
        - Posterior: N(mu | mu_1, 1/beta_1) where
          - beta_1 = beta_0 + beta_W
          - mu_1 = (beta_0 * mu_0 + beta_W * W) / beta_1

        Args:
            state: Current state [5 × 3].
            decision: Drug index to try (0-4).
            exog: Observed reduction and measurement precision.

        Returns:
            Updated state with new belief for selected drug.
        """
        # Extract current belief for selected drug
        drug_state = state[decision]
        mu_prior, beta_prior, n_trials = drug_state[0], drug_state[1], drug_state[2]

        # Bayesian update
        beta_posterior = beta_prior + exog.measurement_precision
        mu_posterior = (
            beta_prior * mu_prior + exog.measurement_precision * exog.reduction
        ) / beta_posterior

        # Update trial count
        n_trials_new = n_trials + 1

        # Create new state row for this drug
        new_drug_state = jnp.array([mu_posterior, beta_posterior, n_trials_new])

        # Update state (only the selected drug changes)
        new_state = state.at[decision].set(new_drug_state)

        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def reward(
        self,
        state: State,
        decision: Decision,
        exog: ExogenousInfo,
    ) -> Reward:
        """Compute reward as true mean effectiveness of selected drug.

        Args:
            state: Current state (unused).
            decision: Drug index.
            exog: Exogenous information.

        Returns:
            True mean effectiveness (A1C reduction).
        """
        # Reward is the true mean effectiveness
        return exog.true_mu

    def sample_exogenous(
        self,
        key: Key,
        state: State,
        time: int,
        decision: int = 0,
    ) -> ExogenousInfo:
        """Sample exogenous information for a drug trial.

        Args:
            key: Random key.
            state: Current state (unused).
            time: Current time step (unused).
            decision: Which drug is being tried.

        Returns:
            Observed A1C reduction with measurement noise.
        """
        # True mean for this drug
        true_mu = self.true_mu[decision]

        # Observed reduction = true_mu + measurement_noise
        noise = jax.random.normal(key) * self.config.measurement_sigma
        reduction = true_mu + noise

        # Measurement precision
        measurement_precision = 1.0 / (self.config.measurement_sigma ** 2)

        return ExogenousInfo(
            reduction=reduction,
            true_mu=true_mu,
            measurement_precision=jnp.array(measurement_precision),
        )

    @partial(jax.jit, static_argnums=(0,))
    def is_valid_decision(self, state: State, decision: Decision) -> jax.Array:
        """Check if decision is valid.

        Args:
            state: Current state.
            decision: Drug index to validate.

        Returns:
            Boolean array: True if decision is valid (0 <= decision < n_drugs).
        """
        return (decision >= 0) & (decision < self.config.n_drugs)

    def get_drug_names(self) -> list[str]:
        """Get list of drug names.

        Returns:
            List of drug names.
        """
        return ["Metformin", "Sensitizer", "Secretagogue", "AGI", "Peptide Analog"]

    def get_posterior_std(self, state: State, drug_idx: int) -> jax.Array:
        """Get posterior standard deviation for a drug.

        Args:
            state: Current state.
            drug_idx: Drug index.

        Returns:
            Posterior standard deviation.
        """
        beta = state[drug_idx, 1]
        return 1.0 / jnp.sqrt(beta)

    def get_drug_statistics(self, state: State) -> dict[str, jax.Array]:
        """Get statistics for all drugs.

        Args:
            state: Current state.

        Returns:
            Dictionary with mu, sigma, and N for each drug.
        """
        return {
            "mu": state[:, 0],  # Posterior means
            "sigma": 1.0 / jnp.sqrt(state[:, 1]),  # Posterior std devs
            "N": state[:, 2],  # Trial counts
        }
