"""Solar insolation computation from day-of-year and latitude."""
from __future__ import annotations

import numpy as np


def solar_declination(day_of_year: np.ndarray) -> np.ndarray:
    return 23.44 * np.sin(np.deg2rad((360.0 / 365.0) * (day_of_year - 81.0)))


def hour_angle(lat_deg: np.ndarray, decl_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    decl = np.deg2rad(decl_deg)
    cos_h = -np.tan(lat) * np.tan(decl)
    cos_h = np.clip(cos_h, -1.0, 1.0)
    return np.arccos(cos_h)


def daily_insolation(
    lat_deg: float,
    day_of_year: np.ndarray,
    solar_constant: float = 1361.0,
) -> np.ndarray:
    """Daily mean top-of-atmosphere insolation (W/m^2)."""
    lat = np.deg2rad(lat_deg)
    decl = np.deg2rad(solar_declination(day_of_year))
    h = hour_angle(lat_deg, solar_declination(day_of_year))
    cos_term = np.cos(lat) * np.cos(decl) * np.sin(h)
    sin_term = h * np.sin(lat) * np.sin(decl)
    return (solar_constant / np.pi) * np.clip(sin_term + cos_term, 0.0, None)


def instantaneous_insolation(
    lat_deg: float,
    day_of_year: np.ndarray,
    hour: float = 12.0,
    solar_constant: float = 1361.0,
) -> np.ndarray:
    """Instantaneous TOA insolation at a given local solar hour."""
    lat = np.deg2rad(lat_deg)
    decl = np.deg2rad(solar_declination(day_of_year))
    omega = np.deg2rad(15.0 * (hour - 12.0))
    mu = np.sin(decl) * np.sin(lat) + np.cos(decl) * np.cos(lat) * np.cos(omega)
    return solar_constant * np.clip(mu, 0.0, None)


def insolation_for_timeseries(
    lat_deg: float,
    dates: np.ndarray,
    solar_constant: float = 1361.0,
) -> np.ndarray:
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    return daily_insolation(lat_deg, doy, solar_constant)
