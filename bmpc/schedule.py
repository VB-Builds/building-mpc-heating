"""Occupancy, comfort bands, internal gains and time-of-use tariff (all assumptions, labelled in README)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .weather import TZ

ZONES = ("living", "bedroom", "office")
NZ = len(ZONES)

# Comfort lower bound [degC] when occupied / unoccupied; upper bound [degC].
T_LOW_OCC = np.array([20.5, 18.0, 21.0])
T_LOW_UNOCC = np.array([17.0, 16.0, 16.0])
T_HIGH_OCC = np.array([25.0, 24.0, 25.0])
T_HIGH_UNOCC = np.array([27.0, 27.0, 27.0])
# Internal gains [kW] while occupied (people + appliances), small base load when not.
GAINS_OCC = np.array([0.35, 0.15, 0.60])
GAINS_UNOCC = np.array([0.05, 0.02, 0.05])

# Illustrative two-rate tariff (assumption, not a real utility tariff).
PRICE_HT = 0.30  # CHF/kWh, Mon-Fri 06:00-22:00
PRICE_NT = 0.18  # CHF/kWh, otherwise


def local_clock(index_utc: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Local hour-of-day and weekday (0=Mon) for each hour start."""
    loc = index_utc.tz_convert(TZ)
    return np.asarray(loc.hour), np.asarray(loc.day_of_week)


def occupancy(hour: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """Boolean occupancy, shape (NZ, N)."""
    weekday = dow < 5
    living = np.where(weekday, ((hour >= 6) & (hour < 8)) | ((hour >= 17) & (hour < 23)), (hour >= 8) & (hour < 23))
    bedroom = (hour >= 21) | (hour < 7)
    office = weekday & (hour >= 8) & (hour < 18)
    return np.vstack([living, bedroom, office])


def comfort_bands(index_utc: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Per hour interval: occupancy, lower/upper comfort bounds and internal gains, each (NZ, N)."""
    hour, dow = local_clock(index_utc)
    occ = occupancy(hour, dow)
    low = np.where(occ, T_LOW_OCC[:, None], T_LOW_UNOCC[:, None])
    high = np.where(occ, T_HIGH_OCC[:, None], T_HIGH_UNOCC[:, None])
    gains = np.where(occ, GAINS_OCC[:, None], GAINS_UNOCC[:, None])
    return {"occ": occ, "low": low, "high": high, "gains": gains}


def tariff(index_utc: pd.DatetimeIndex) -> np.ndarray:
    """CHF/kWh for each hour start."""
    hour, dow = local_clock(index_utc)
    ht = (dow < 5) & (hour >= 6) & (hour < 22)
    return np.where(ht, PRICE_HT, PRICE_NT).astype(float)
