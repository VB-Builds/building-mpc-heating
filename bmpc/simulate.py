"""Closed-loop simulation engine: plant at 15 min, controllers at 15 min (baseline) or 1 h (MPC)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .building import Plant
from .schedule import NZ, comfort_bands, tariff
from .weather import Weather

SUB = 4  # plant sub-steps per hour
SENSOR_SIGMA = 0.1  # K, room temperature sensor noise


@dataclass
class Env:
    """Everything the plant and controllers see, per hour interval (length n)."""

    wx: Weather
    tout: np.ndarray  # degC, interval mean
    tout_pt: np.ndarray  # degC at labels (n+1)
    solar: np.ndarray  # kW/m2, interval mean
    occ: np.ndarray  # (NZ, n)
    low: np.ndarray
    high: np.ndarray
    gains: np.ndarray  # scheduled (known to controllers)
    gains_true: np.ndarray  # what the building really gets
    solar_factor: np.ndarray  # true / nominal irradiance-to-gain factor
    price: np.ndarray  # CHF/kWh
    index: object

    @property
    def n(self) -> int:
        return len(self.tout)


def make_env(wx: Weather, seed: int = 1) -> Env:
    """Prepare inputs. True gains and solar transmission deviate from what controllers assume."""
    rng = np.random.default_rng(seed)
    idx = wx.index[:-1]
    b = comfort_bands(idx)
    day_id, _ = pd.factorize(idx.tz_convert("Europe/Zurich").normalize())
    n_days = day_id.max() + 1
    g_fac = np.exp(rng.normal(0.0, 0.25, (NZ, n_days)))[:, day_id]
    s_fac = np.exp(rng.normal(0.0, 0.15, n_days))[day_id]
    return Env(
        wx=wx,
        tout=wx.interval_temp(),
        tout_pt=wx.temp,
        solar=wx.interval_solar_kw_m2(),
        occ=b["occ"],
        low=b["low"],
        high=b["high"],
        gains=b["gains"],
        gains_true=b["gains"] * g_fac,
        solar_factor=s_fac,
        price=tariff(idx),
        index=idx,
    )


def forecast(env: Env, k: int, horizon: int, scale: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Weather forecast issued at hour k for intervals k..k+H-1: truth plus lead-time-dependent errors.

    Temperature error std grows from 0.3 K to 1.5 K over 24 h (scale=1), AR(1)-correlated in lead time.
    Irradiance error is multiplicative lognormal with sigma 0.35 (scale=1), also correlated.
    """
    t_true = env.tout[k : k + horizon]
    s_true = env.solar[k : k + horizon]
    if len(t_true) < horizon:
        raise ValueError("forecast window exceeds environment")
    if scale <= 0:
        return t_true.copy(), s_true.copy()
    lead = np.arange(horizon)
    sd = scale * (0.3 + 1.2 * np.minimum(lead / 24.0, 1.0))
    e = _ar1(rng, horizon, 0.85) * sd
    f = _ar1(rng, horizon, 0.85) * 0.35 * scale
    return t_true + e, np.clip(s_true * np.exp(f), 0.0, None)


def _ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    z = rng.normal(size=n)
    out = np.empty(n)
    out[0] = z[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + np.sqrt(1 - rho**2) * z[i]
    return out


@dataclass
class Result:
    k0: int
    k1: int
    ti: np.ndarray  # (steps, NZ) true air temperature at step start
    q: np.ndarray  # (steps, NZ) delivered heat kW
    p_el: np.ndarray  # (steps,) kW electric
    tsup: np.ndarray
    price: np.ndarray  # per step
    low: np.ndarray  # (steps, NZ)
    high: np.ndarray
    solve_ms: list[float] = field(default_factory=list)
    label: str = ""


def run(env: Env, controller, plant: Plant, k0: int, k1: int, seed: int = 7, t_init: float = 20.0) -> Result:
    """Simulate hours k0..k1-1. `controller.act(k, s, y)` returns (q_cmd, valve), one of them None."""
    if k1 > env.n or k0 < 0 or k0 >= k1:
        raise ValueError("invalid simulation window")
    rng = np.random.default_rng(seed)
    plant.reset(t_init)
    steps = (k1 - k0) * SUB
    ti = np.empty((steps, NZ))
    qd = np.empty((steps, NZ))
    pe = np.empty(steps)
    ts = np.empty(steps)
    pr = np.empty(steps)
    lo = np.empty((steps, NZ))
    hi = np.empty((steps, NZ))
    j = 0
    for k in range(k0, k1):
        for s in range(SUB):
            y = plant.t_air + rng.normal(0.0, SENSOR_SIGMA, NZ)
            q_cmd, valve = controller.act(k, s, y)
            t_out = env.tout_pt[k] + (env.tout_pt[k + 1] - env.tout_pt[k]) * (s + 0.5) / SUB
            ti[j] = plant.t_air
            q, tsup, p_el = plant.delivered(q_cmd, valve, t_out)
            qd[j], ts[j], pe[j] = q, tsup, p_el
            pr[j] = env.price[k]
            lo[j], hi[j] = env.low[:, k], env.high[:, k]
            plant.step(q, t_out, env.solar[k], env.gains_true[:, k], env.solar_factor[k])
            j += 1
        if hasattr(controller, "end_hour"):
            controller.end_hour(k, qd[j - SUB : j].mean(axis=0), env)
    return Result(k0, k1, ti, qd, pe, ts, pr, lo, hi, list(getattr(controller, "solve_ms", [])))


def metrics(res: Result, skip_hours: int = 0) -> dict[str, float]:
    """Season metrics over steps after `skip_hours` of warm-up."""
    a = skip_hours * SUB
    dt = 1.0 / SUB
    p = res.p_el[a:]
    viol = np.maximum(res.low[a:] - res.ti[a:], 0.0) + np.maximum(res.ti[a:] - res.high[a:], 0.0)
    e = float(p.sum() * dt)
    th = float(res.q[a:].sum() * dt)
    out = {
        "electricity_kwh": e,
        "heat_kwh": th,
        "cost_chf": float((p * res.price[a:]).sum() * dt),
        "comfort_kh": float(viol.sum() * dt),
        "max_deviation_k": float(viol.max()),
        "peak_kw": float(p.max()),
        "scop": th / e if e > 0 else float("nan"),
    }
    sm = np.array(res.solve_ms) if res.solve_ms else None
    if sm is not None and len(sm):
        out["mpc_ms_mean"] = float(sm.mean())
        out["mpc_ms_p95"] = float(np.percentile(sm, 95))
        out["mpc_ms_max"] = float(sm.max())
    return out
