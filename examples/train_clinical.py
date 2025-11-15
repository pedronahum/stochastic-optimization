import jax
import jax.numpy as jnp
import optax
from flax import nnx

from stochopt.core.simulator import rollout
from problems.clinical_trials.model import ClinicalTrialsModel, Config
from problems.clinical_trials.policy import LinearDosePolicy

cfg = Config()
model = ClinicalTrialsModel(cfg)
policy = LinearDosePolicy()
optimizer = nnx.Optimizer(policy, optax.adam(1e-2))

RNG = jax.random.PRNGKey(0)


def loss_fn(pi: nnx.Module) -> jnp.ndarray:
    rewards = rollout(model, pi, cfg.horizon, key=RNG)
    return -jnp.sum(rewards)


for step in range(1_000):
    # get (loss, grads) w.r.t. all nnx.Param leaves
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)

    if step % 100 == 0:
        w = optimizer.model.w
        print(f"Step {step:4d} → loss {loss:.3f},  w = {float(w):.4f}")

print("Trained dose‐weight w =", float(optimizer.model.w))
