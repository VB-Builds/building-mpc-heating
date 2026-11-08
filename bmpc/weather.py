"""Weather loading and hour-interval conventions.

Data: Open-Meteo Historical Weather API (https://open-meteo.com/en/docs/historical-weather-api),
Zurich (47.3769 N, 8.5417 E), hourly 2 m temperature and shortwave (global horizontal) radiation,
2024-09-28 .. 2025-04-30 (UTC). Open-Meteo reanalysis data is CC BY 4.0.
Temperature is the value at the timestamp; radiation is the mean of the preceding hour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "zurich_2024-10_2025-04.json"
TZ = "Europe/Zurich"


@dataclass
class Weather:
    """Hourly weather. Interval i covers [t_i, t_i + 1 h)."""

    index: pd.DatetimeIndex  # UTC, hourly, length N+1 (labels)
    temp: np.ndarray  # degC at each label, length N+1
    solar: np.ndarray  # W/m2, mean of the hour preceding each label, length N+1
    source: str = ""

    def __post_init__(self) -> None:
        n = len(self.index)
        if not (len(self.temp) == len(self.solar) == n):
            raise ValueError("weather arrays must have equal length")
        if n < 2:
            raise ValueError("need at least two hourly samples")
        if np.isnan(self.temp).any() or np.isnan(self.solar).any():
            raise ValueError("weather contains NaN")
        if not ((self.index[1:] - self.index[:-1]) == pd.Timedelta(hours=1)).all():
            raise ValueError("weather index must be strictly hourly and contiguous")

    @property
    def n_intervals(self) -> int:
        return len(self.index) - 1

    def interval_temp(self) -> np.ndarray:
        """Mean outdoor temperature per hour interval (trapezoid of the two labels)."""
        return 0.5 * (self.temp[:-1] + self.temp[1:])

    def interval_solar_kw_m2(self) -> np.ndarray:
        """Mean global horizontal irradiance per hour interval in kW/m2."""
        return self.solar[1:] / 1000.0

    def slice_local(self, start: str, end: str, pad_before_h: int = 0, pad_after_h: int = 0) -> tuple[Weather, int]:
        """Cut [start, end) given as local dates; returns (weather, offset of `start` inside it)."""
        t0 = pd.Timestamp(start, tz=TZ).tz_convert("UTC")
        t1 = pd.Timestamp(end, tz=TZ).tz_convert("UTC")
        i0 = int(self.index.searchsorted(t0)) - pad_before_h
        i1 = int(self.index.searchsorted(t1)) + pad_after_h
        if i0 < 0 or i1 >= len(self.index):
            raise ValueError("requested window exceeds available weather data")
        w = Weather(self.index[i0 : i1 + 1], self.temp[i0 : i1 + 1], self.solar[i0 : i1 + 1], self.source)
        return w, pad_before_h


def load_weather(path: str | Path | None = None) -> Weather:
    p = Path(path) if path else DATA_FILE
    d = json.loads(p.read_text(encoding="utf-8"))
    h = d["hourly"]
    idx = pd.DatetimeIndex(pd.to_datetime(h["time"]), tz="UTC")
    temp = np.array([np.nan if v is None else v for v in h["temperature_2m"]], dtype=float)
    sol = np.array([np.nan if v is None else v for v in h["shortwave_radiation"]], dtype=float)
    src = "Open-Meteo Historical Weather API (CC BY 4.0), Zurich 47.3769N 8.5417E"
    return Weather(idx, temp, sol, src)


def synthetic_weather(n_hours: int = 24 * 14, seed: int = 0, start: str = "2024-12-01") -> Weather:
    """Deterministic fake winter weather for tests (not used for reported results)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n_hours + 1, freq="h")
    hod = np.asarray(idx.hour)
    temp = 2.0 - 4.0 * np.cos((hod - 4) / 24 * 2 * np.pi) + np.cumsum(rng.normal(0, 0.15, n_hours + 1))
    sol = np.clip(250 * np.sin((hod - 6) / 12 * np.pi), 0, None) * rng.uniform(0.2, 1.0, n_hours + 1)
    return Weather(idx, temp, sol, "synthetic")
