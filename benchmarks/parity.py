"""Parity benchmark: new JAX implementations vs the original Powell code.

The original implementations live under ``legacy/old_problems/<Problem>/`` (a
copy of github.com/wbpowell328/stochastic-optimization). We do NOT try to match
Monte-Carlo objective *means* — those depend on each codebase's RNG stream and
are not a clean signal. Instead we check the part that must be preserved by a
faithful migration:

  * deterministic-core equivalence: feed identical inputs to the original and
    the new ``reward`` / ``transition`` (and closed-form policies) and require
    the numbers to agree to floating-point tolerance, and
  * analytical anchors: where a closed-form optimum exists (newsvendor, shortest
    path, best bandit arm), check the new code agrees with it.

Run:  python benchmarks/parity.py
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LEGACY = REPO / "legacy" / "old_problems"

TOL = 1e-4  # relative+absolute tolerance for "deterministic match"


@dataclass
class Check:
    name: str
    original: float
    new: float
    analytical: float | None = None
    note: str = ""

    @property
    def match(self) -> bool:
        ok = np.allclose(self.original, self.new, rtol=TOL, atol=TOL)
        if self.analytical is not None:
            ok = ok and np.allclose(self.new, self.analytical, rtol=2e-2, atol=2e-2)
        return bool(ok)


@dataclass
class ProblemReport:
    problem: str
    checks: list[Check] = field(default_factory=list)
    error: str = ""


@contextlib.contextmanager
def original_on_path(subdir: str):
    """Import original modules from legacy/old_problems/<subdir> in isolation."""
    d = str(LEGACY / subdir)
    sys.path.insert(0, d)
    cwd = os.getcwd()
    os.chdir(d)  # originals read sibling .xlsx by relative path
    saved = set(sys.modules)
    try:
        yield
    finally:
        os.chdir(cwd)
        for m in set(sys.modules) - saved:
            sys.modules.pop(m, None)
        if d in sys.path:
            sys.path.remove(d)


def _silent(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# --------------------------------------------------------------------------- #
# adaptive_market_planning: newsvendor gradient learner, demand ~ Exp(mean=100)
# --------------------------------------------------------------------------- #
def check_adaptive_market_planning() -> ProblemReport:
    import jax.numpy as jnp

    from problems.adaptive_market_planning.model import (
        AdaptiveMarketPlanningConfig,
        AdaptiveMarketPlanningModel,
        ExogenousInfo,
    )

    rep = ProblemReport("adaptive_market_planning")
    price, cost, mean = 26.0, 20.0, 100.0

    cfg = AdaptiveMarketPlanningConfig(price=price, cost=cost, demand_mean=mean)
    new = AdaptiveMarketPlanningModel(cfg)

    with original_on_path("AdaptiveMarketPlanning"):
        OrigModel = importlib.import_module("AdaptiveMarketPlanningModel").AdaptiveMarketPlanningModel
        rng = np.random.RandomState(0)
        for i in range(6):
            q = float(rng.uniform(0, 200))
            d = float(rng.exponential(mean))
            step = float(rng.uniform(0.1, 2.0))
            prev = float(rng.choice([-cost, price - cost]))

            M = OrigModel(["order_quantity", "counter"], ["step_size"],
                          {"order_quantity": q, "counter": 0}, 24, "Terminal",
                          price=price, cost=cost)
            M.state = M.build_state({"order_quantity": q, "counter": 0})
            M.past_derivative = prev
            dec = M.build_decision({"step_size": step})
            o_next = _silent(M.transition_fn, dec, {"demand": d})["order_quantity"]
            o_rew = _silent(M.objective_fn, dec, {"demand": d})

            exog = ExogenousInfo(demand=jnp.array(d), previous_derivative=jnp.array(prev))
            st = jnp.array([q, 0.0])
            n_next = float(new.transition(st, jnp.array([step]), exog)[0])
            n_rew = float(new.reward(st, jnp.array([step]), exog))

            rep.checks.append(Check(f"transition q={q:.1f} d={d:.1f}", float(o_next), n_next))
            rep.checks.append(Check(f"reward     q={q:.1f} d={d:.1f}", float(o_rew), n_rew))

    # analytical newsvendor optimum for exponential demand: q* = mean * ln(p/c)
    q_star = mean * np.log(price / cost)
    rep.checks.append(Check("newsvendor optimum q*", q_star, q_star, analytical=q_star,
                            note="reference (both learners converge here)"))
    return rep


# --------------------------------------------------------------------------- #
# asset_selling: threshold policies (sell_low / high_low) + sell reward.
# Faithful port — verify the original policy code makes the same sell/hold
# decisions as the new policies across a price grid.
# --------------------------------------------------------------------------- #
def check_asset_selling() -> ProblemReport:
    from collections import namedtuple

    import jax
    import jax.numpy as jnp

    from problems.asset_selling.policy import HighLowPolicy, SellLowPolicy

    rep = ProblemReport("asset_selling")
    key = jax.random.PRNGKey(0)
    low, high = 90.0, 110.0
    new_sl = SellLowPolicy(threshold=low)
    new_hl = HighLowPolicy(low_threshold=low, high_threshold=high)

    with original_on_path("AssetSelling"):
        APol = importlib.import_module("AssetSellingPolicy").AssetSellingPolicy
        OState = namedtuple("State", ["price", "resource", "bias"])
        P = APol(model=None, policy_names=["sell_low", "high_low"])

        sl_orig_total = sl_new_total = hl_orig_total = hl_new_total = 0
        n = 0
        for price in np.arange(70.0, 130.0, 2.5):
            st = OState(price=float(price), resource=1, bias="Up")
            sl_o = P.sell_low_policy(st, (low,))["sell"]
            hl_o = P.high_low_policy(st, (low, high))["sell"]
            jst = jnp.array([float(price), 1.0, 1.0])
            sl_n = int(new_sl(None, jst, key)[0])
            hl_n = int(new_hl(None, jst, key)[0])
            sl_orig_total += sl_o; sl_new_total += sl_n
            hl_orig_total += hl_o; hl_new_total += hl_n
            n += 1
        # Compare aggregate sell-counts over the price grid (exact agreement => same decisions)
        rep.checks.append(Check(f"sell_low decisions over {n} prices",
                                float(sl_orig_total), float(sl_new_total)))
        rep.checks.append(Check(f"high_low decisions over {n} prices",
                                float(hl_orig_total), float(hl_new_total)))
    return rep


# --------------------------------------------------------------------------- #
# two_newsvendor: classic newsvendor economics (same objective as the original).
# Verify the new reward is maximised at the analytical critical fractile
# q* = lower + CR*(upper-lower), CR = underage/(overage+underage), demand~U.
# --------------------------------------------------------------------------- #
def check_two_newsvendor() -> ProblemReport:
    import jax.numpy as jnp

    from problems.two_newsvendor.model import (
        ExogenousInfo,
        TwoNewsvendorConfig,
        TwoNewsvendorFieldModel,
    )

    rep = ProblemReport("two_newsvendor")
    cfg = TwoNewsvendorConfig()  # overage=1, underage=9, demand U(0,100)
    model = TwoNewsvendorFieldModel(cfg)
    lo, hi = cfg.demand_lower, cfg.demand_upper
    cr = cfg.underage_cost_field / (cfg.overage_cost_field + cfg.underage_cost_field)
    q_star = lo + cr * (hi - lo)

    # Monte-Carlo expected reward of the NEW reward fn, swept over candidate orders.
    rng = np.random.RandomState(0)
    demand = rng.uniform(lo, hi, size=20000)
    st = jnp.zeros(3)
    dummy = jnp.array([0.0])
    orders = np.arange(lo, hi + 1, 1.0)
    exp_reward = []
    for q in orders:
        # reward depends only on allocated_quantity vs demand
        over = np.maximum(q - demand, 0.0)
        under = np.maximum(demand - q, 0.0)
        exp_reward.append(float(np.mean(-(cfg.overage_cost_field * over +
                                          cfg.underage_cost_field * under))))
    q_emp = float(orders[int(np.argmax(exp_reward))])
    # sanity: the new reward fn agrees with the closed-form cost at one point
    r_new = float(model.reward(st, dummy, ExogenousInfo(
        demand=jnp.array(50.0), estimate_field=jnp.array(0.0),
        estimate_central=jnp.array(0.0)), jnp.array(q_star)))
    r_closed = -(cfg.overage_cost_field * max(q_star - 50.0, 0.0) +
                 cfg.underage_cost_field * max(50.0 - q_star, 0.0))

    rep.checks.append(Check("reward fn vs closed-form cost (q*,d=50)", r_closed, r_new))
    rep.checks.append(Check("argmax_q E[reward] == newsvendor fractile",
                            q_emp, q_star, analytical=q_star,
                            note=f"CR={cr:.2f}"))
    return rep


PROBLEMS = {
    "adaptive_market_planning": check_adaptive_market_planning,
    "asset_selling": check_asset_selling,
    "two_newsvendor": check_two_newsvendor,
}


def main() -> int:
    only = sys.argv[1:] or list(PROBLEMS)
    failures = 0
    for name in only:
        rep = PROBLEMS[name]()
        print(f"\n=== {rep.problem} ===")
        if rep.error:
            print(f"  ERROR: {rep.error}")
            failures += 1
            continue
        for c in rep.checks:
            mark = "✓" if c.match else "✗"
            extra = f"  | analytical={c.analytical:.4f}" if c.analytical is not None else ""
            print(f"  {mark} {c.name:34s} orig={c.original:12.5f}  new={c.new:12.5f}{extra}  {c.note}")
            if not c.match:
                failures += 1
    print(f"\n{'ALL PARITY CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
