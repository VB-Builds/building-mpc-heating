import numpy as np
import pandas as pd
import pytest

from bmpc.schedule import PRICE_HT, PRICE_NT, comfort_bands, occupancy, tariff
from bmpc.weather import Weather, load_weather


def test_real_weather_loads_clean():
    w = load_weather()
    assert w.n_intervals == 5159
    assert not np.isnan(w.temp).any()
    assert -20 < w.temp.min() < w.temp.max() < 40
    assert w.solar.min() >= 0
    assert w.index.tz is not None


def test_weather_rejects_nan():
    idx = pd.date_range("2024-12-01", periods=3, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="NaN"):
        Weather(idx, np.array([1.0, np.nan, 2.0]), np.zeros(3))


def test_weather_rejects_gap():
    idx = pd.DatetimeIndex(["2024-12-01 00:00", "2024-12-01 01:00", "2024-12-01 03:00"], tz="UTC")
    with pytest.raises(ValueError, match="hourly"):
        Weather(idx, np.zeros(3), np.zeros(3))


def test_weather_rejects_length_mismatch():
    idx = pd.date_range("2024-12-01", periods=3, freq="h", tz="UTC")
    with pytest.raises(ValueError):
        Weather(idx, np.zeros(2), np.zeros(3))


def test_interval_values():
    idx = pd.date_range("2024-12-01", periods=3, freq="h", tz="UTC")
    w = Weather(idx, np.array([0.0, 2.0, 6.0]), np.array([0.0, 100.0, 300.0]))
    assert w.interval_temp().tolist() == [1.0, 4.0]
    assert w.interval_solar_kw_m2().tolist() == [0.1, 0.3]  # backward-averaged radiation


def test_slice_local_and_bounds():
    w = load_weather()
    s, off = w.slice_local("2024-12-01", "2024-12-03", pad_before_h=5, pad_after_h=7)
    assert off == 5
    assert s.n_intervals == 48 + 5 + 7
    assert s.index[off] == pd.Timestamp("2024-11-30 23:00", tz="UTC")  # CET midnight
    with pytest.raises(ValueError, match="exceeds"):
        w.slice_local("2025-04-29", "2025-05-05")


def test_office_occupied_weekday_daytime_only():
    hour = np.arange(24)
    wd = occupancy(hour, np.full(24, 2))[2]
    we = occupancy(hour, np.full(24, 6))[2]
    assert wd.tolist() == [8 <= h < 18 for h in range(24)]
    assert not we.any()


def test_bedroom_occupied_at_night():
    hour = np.arange(24)
    bed = occupancy(hour, np.full(24, 1))[1]
    assert bed[23] and bed[3] and not bed[12]


def test_tariff_uses_local_time():
    # 2024-12-04 is a Wednesday; CET = UTC+1, so 05:00 UTC = 06:00 local = first high-tariff hour
    idx = pd.DatetimeIndex(["2024-12-04 04:00", "2024-12-04 05:00", "2024-12-04 20:00", "2024-12-04 21:00"], tz="UTC")
    assert tariff(idx).tolist() == [PRICE_NT, PRICE_HT, PRICE_HT, PRICE_NT]


def test_tariff_weekend_is_low():
    idx = pd.date_range("2024-12-07 08:00", periods=10, freq="h", tz="UTC")  # Saturday
    assert (tariff(idx) == PRICE_NT).all()


def test_bands_consistent():
    idx = pd.date_range("2024-12-02", periods=24 * 7, freq="h", tz="UTC")
    b = comfort_bands(idx)
    assert b["low"].shape == b["high"].shape == b["gains"].shape == (3, 24 * 7)
    assert (b["low"] < b["high"]).all()
    assert b["gains"][b["occ"]].min() > b["gains"][~b["occ"]].max()
