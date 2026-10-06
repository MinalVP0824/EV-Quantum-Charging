"""Synthetic EV charging-site load generator.

Produces hourly aggregate charging demand (kW) for one feeder that serves a
group of public chargers. The pattern includes a morning commute peak, a
stronger evening peak, weekend behaviour, a heat effect and a slow upward
trend that mimics growing EV adoption.

The generator is only a stand-in so the demo runs offline. Any real dataset
with an hourly timestamp and a demand column (for example an aggregated slice
of ACN-Data) can be passed to the forecaster instead.
"""

import numpy as np
import pandas as pd


def generate_site_load(days: int = 90, seed: int = 42, start: str = "2026-06-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    hour = idx.hour.values
    dow = idx.dayofweek.values
    weekend = (dow >= 5).astype(float)
    n = len(idx)

    temp = (
        26
        + 5 * np.sin(2 * np.pi * (hour - 9) / 24)
        + 2 * np.sin(2 * np.pi * np.arange(n) / (24 * 30))
        + rng.normal(0, 1.0, n)
    )

    morning = 18 * np.exp(-0.5 * ((hour - 9) / 1.5) ** 2)
    evening = 34 * np.exp(-0.5 * ((hour - 18.3) / 1.8) ** 2)
    midday_weekend = 6 * weekend * np.exp(-0.5 * ((hour - 14) / 3) ** 2)

    base = 8 + morning * (1 - 0.6 * weekend) + evening * (1 - 0.2 * weekend) + midday_weekend
    heat = 0.8 * np.clip(temp - 28, 0, None)
    trend = np.linspace(0, 4, n)
    noise = rng.normal(0, 2.5, n)

    demand = np.clip(base + heat + trend + noise, 0, None)
    return pd.DataFrame({"timestamp": idx, "demand_kw": demand, "temp_c": temp})
