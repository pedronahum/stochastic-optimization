"""Energy Storage example — grid-search the buy-low/sell-high thresholds.

This mirrors the original Powell driver: the policy buys when the price is at or
below ``theta_buy`` and sells at or above ``theta_sell``; we grid-search the two
thresholds over an exogenous price series and report the best contribution.

By default it loads the original historical PJM RT LMP prices from
``legacy/old_problems/EnergyStorage_I/Parameters.xlsx`` (falling back to a
synthetic oscillating series if that file is unavailable).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from problems.energy_storage import (
    BuyLowSellHighPolicy,
    EnergyStorageConfig,
    EnergyStorageModel,
    grid_search,
    simulate,
)

T = 192  # original horizon


def load_prices() -> jnp.ndarray:
    """Load the historical price series, or synthesise one if unavailable."""
    xlsx = (Path(__file__).resolve().parent.parent
            / "legacy/old_problems/EnergyStorage_I/Parameters.xlsx")
    try:
        import pandas as pd

        raw = pd.ExcelFile(xlsx).parse("Raw Data")
        prices = raw["PJM RT LMP"].to_numpy()[:T]
        print(f"Loaded {len(prices)} historical PJM RT LMP prices")
        return jnp.asarray(prices, dtype=jnp.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"(historical prices unavailable: {exc}; using a synthetic series)")
        t = jnp.arange(T)
        return 40.0 + 30.0 * jnp.sin(2 * jnp.pi * t / 24.0)


def main() -> None:
    """Grid-search thresholds and report the best contribution."""
    prices = load_prices()
    horizon = int(prices.shape[0])
    config = EnergyStorageConfig(eta=0.9, capacity=1.0, initial_energy=1.0)
    model = EnergyStorageModel(config, prices=prices)

    print("\nEnergy Storage — buy-low / sell-high grid search")
    print("=" * 70)

    lo, hi = float(jnp.min(prices)), float(jnp.max(prices))
    buy_grid = jnp.linspace(lo, (lo + hi) / 2, 12)
    sell_grid = jnp.linspace((lo + hi) / 2, hi, 12)

    (theta_buy, theta_sell), best = grid_search(model, horizon, buy_grid, sell_grid)
    hold = simulate(model, BuyLowSellHighPolicy(model, lo - 1.0, hi + 1.0), horizon)

    print(f"Price range: [{lo:.2f}, {hi:.2f}] over {horizon} periods")
    print(f"Best thresholds: buy<= {theta_buy:.2f}, sell>= {theta_sell:.2f}")
    print(f"Best contribution: {best:.2f}")
    print(f"Never-trade baseline: {hold:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
