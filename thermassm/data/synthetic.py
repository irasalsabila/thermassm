"""Synthetic temperature generator for offline development and smoke tests."""
from __future__ import annotations

import numpy as np

from .insolation import daily_insolation, solar_declination


def generate_synthetic(
    lat_deg: float = 40.0,
    start_year: int = 1979,
    end_year: int = 2022,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a daily temperature series driven by an analytic EBM."""
    rng = np.random.default_rng(seed)
    dates = np.arange(
        np.datetime64(f"{start_year}-01-01"),
        np.datetime64(f"{end_year + 1}-01-01"),
        np.timedelta64(1, "D"),
    )
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    ins = daily_insolation(lat_deg, doy.astype(float))

    annual_cycle = 8.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)
    seasonal_amp = 1.0 - 0.5 * abs(lat_deg) / 90.0
    seasonal = annual_cycle * seasonal_amp

    trend = 0.02 * np.arange(len(dates)) / 365.0
    ar1 = np.zeros(len(dates))
    ar1[0] = rng.normal(0, 1)
    for i in range(1, len(dates)):
        ar1[i] = 0.8 * ar1[i - 1] + rng.normal(0, 0.6)

    mean_temp = 288.0 - 0.6 * abs(lat_deg) / 90.0
    t2m = mean_temp + seasonal + trend + ar1 + 0.5 * ins / 340.0

    return dates, t2m.astype(np.float32)
