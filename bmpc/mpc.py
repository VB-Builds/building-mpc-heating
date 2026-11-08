"""Model predictive control: a convex (LP) heating planner, a Kalman state estimator and a controller."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are

from . import heatpump as hp
from .building import N_STATE, GreyBox
from .schedule import NZ
from .simulate import SENSOR_SIGMA, Env, forecast

TI_IDX = np.arange(NZ) * 2
MARGIN = 0.3  # K, default planning margin above the comfort floor (tuned on the tuning fortnight)
W_COMFORT = 5.0  # CHF per Kh of band violation (soft constraint, near-hard)


@dataclass
class Plan:
    q: np.ndarray  # (NZ, H) thermal kW per zone
    ti: np.ndarray  # (NZ, H+1) predicted air temperature
    status: str
    solve_ms: float
    cost: float


class Planner:
    """24 h (default) receding-horizon heating plan.

    minimise  sum_k price_k * Qtot_k / COP_k  +  W * band violation
    s.t.      x_{k+1} = Ad x_k + Bq Q_k + Bd d_k   (identified grey-box model, 1 h steps)
              0 <= Q_zone <= Q_zone_max, sum Q <= Q_hp_max
              low_k <= Ti_{k+1} + slack <= high_k   (bounds apply to the state at the END of step k)
    COP_k comes from the forecast outdoor temperature and the heating-curve supply temperature
    (conservative: demand-driven low supply temperatures are not credited in the plan).
    """

    def __init__(self, model: GreyBox, horizon: int = 24, solver: str = "CLARABEL"):
        if horizon < 2:
            raise ValueError("horizon must be >= 2")
        self.model = model
        self.h = horizon
        self.solver = solver
        ad, bd = model.discrete(1.0)
        self.ad, self.bd = ad, bd
        h = horizon
        self.x0 = cp.Parameter(N_STATE)
        self.d = cp.Parameter((2 + NZ, h))
        self.low = cp.Parameter((NZ, h))
        self.high = cp.Parameter((NZ, h))
        self.cprice = cp.Parameter(h, nonneg=True)
        x = cp.Variable((N_STATE, h + 1))
        q = cp.Variable((NZ, h), nonneg=True)
        slo = cp.Variable((NZ, h), nonneg=True)
        shi = cp.Variable((NZ, h), nonneg=True)
        sel = np.zeros((NZ, N_STATE))
        sel[np.arange(NZ), TI_IDX] = 1.0
        cons = [
            x[:, 0] == self.x0,
            x[:, 1:] == ad @ x[:, :-1] + bd[:, :NZ] @ q + bd[:, NZ:] @ self.d,
            q <= hp.Q_ZONE_MAX,
            cp.sum(q, axis=0) <= hp.Q_HP_MAX,
            sel @ x[:, 1:] + slo >= self.low,
            sel @ x[:, 1:] - shi <= self.high,
        ]
        cost = self.cprice @ cp.sum(q, axis=0) + W_COMFORT * (cp.sum(slo) + cp.sum(shi))
        self.prob = cp.Problem(cp.Minimize(cost), cons)
        self._x, self._q = x, q

    def plan(
        self,
        x0: np.ndarray,
        tout: np.ndarray,
        solar: np.ndarray,
        gains: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        price: np.ndarray,
    ) -> Plan:
        h = self.h
        for name, arr, shape in (
            ("x0", x0, (N_STATE,)),
            ("tout", tout, (h,)),
            ("solar", solar, (h,)),
            ("gains", gains, (NZ, h)),
            ("low", low, (NZ, h)),
            ("high", high, (NZ, h)),
            ("price", price, (h,)),
        ):
            if np.shape(arr) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {np.shape(arr)}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(np.asarray(price) < 0):
            raise ValueError("price must be non-negative")
        if np.any(np.asarray(low) > np.asarray(high)):
            raise ValueError("comfort low exceeds high")
        t0 = time.perf_counter()
        self.x0.value = np.asarray(x0, dtype=float)
        self.d.value = np.vstack([tout, solar, gains])
        self.low.value = np.asarray(low, dtype=float)
        self.high.value = np.asarray(high, dtype=float)
        tsup = hp.heating_curve(tout)
        self.cprice.value = np.asarray(price, dtype=float) / hp.cop(tout, tsup)
        try:
            self.prob.solve(solver=self.solver)
        except cp.error.SolverError:
            self.prob.solve(solver="HIGHS")
        ms = (time.perf_counter() - t0) * 1000.0
        if self._q.value is None:
            raise RuntimeError(f"MPC solve failed: {self.prob.status}")
        return Plan(
            q=np.clip(self._q.value, 0.0, hp.Q_ZONE_MAX),
            ti=self._x.value[TI_IDX],
            status=self.prob.status,
            solve_ms=ms,
            cost=float(self.prob.value),
        )


class Estimator:
    """Steady-state Kalman filter on the hourly model; measures only room air temperature."""

    def __init__(self, model: GreyBox, proc_sd: float = 0.05, meas_sd: float = SENSOR_SIGMA):
        self.ad, self.bd = model.discrete(1.0)
        self.c = np.zeros((NZ, N_STATE))
        self.c[np.arange(NZ), TI_IDX] = 1.0
        w = np.eye(N_STATE) * proc_sd**2
        v = np.eye(NZ) * meas_sd**2
        p = solve_discrete_are(self.ad.T, self.c.T, w, v)
        self.gain = p @ self.c.T @ np.linalg.inv(self.c @ p @ self.c.T + v)
        self.x = None

    def init(self, y: np.ndarray) -> None:
        x = np.zeros(N_STATE)
        x[TI_IDX] = y
        x[TI_IDX + 1] = y  # envelope assumed at room temperature until data says otherwise
        self.x = x

    def update(self, y: np.ndarray) -> np.ndarray:
        if self.x is None:
            self.init(y)
        self.x = self.x + self.gain @ (y - self.c @ self.x)
        return self.x.copy()

    def predict(self, q: np.ndarray, tout: float, solar: float, gains: np.ndarray) -> None:
        u = np.concatenate([q, [tout, solar], gains])
        self.x = self.ad @ self.x + self.bd @ u


class MPCController:
    """Plans once per hour, holds the zone heat set-points for the hour."""

    def __init__(self, env: Env, model: GreyBox, horizon: int = 24, forecast_scale: float = 1.0, seed: int = 11,
                 margin: float = MARGIN, flat_price: bool = False):
        self.env = env
        self.planner = Planner(model, horizon)
        self.est = Estimator(model)
        self.h = horizon
        self.scale = forecast_scale
        self.margin = margin  # K added to the comfort floor in the plan (robustness to model/forecast error)
        self.rng = np.random.default_rng(seed)
        self.q_hold = np.zeros(NZ)
        self.solve_ms: list[float] = []
        self.plans: dict[int, Plan] = {}
        self.keep_plans = False
        self.flat_price = flat_price  # ablation: plan as if the tariff were flat

    def act(self, k: int, s: int, y: np.ndarray):
        if s == 0:
            x0 = self.est.update(y)
            e = self.env
            t_fc, s_fc = forecast(e, k, self.h, self.scale, self.rng)
            sl = slice(k, k + self.h)
            nx = slice(k + 1, k + self.h + 1)
            # the state at the end of step j is also the start of step j+1: honour both bands
            low = np.maximum(e.low[:, sl], e.low[:, nx]) + self.margin
            high = np.minimum(e.high[:, sl], e.high[:, nx])
            price = np.full(self.h, e.price.mean()) if self.flat_price else e.price[sl]
            plan = self.planner.plan(x0, t_fc, s_fc, e.gains[:, sl], low, high, price)
            self.solve_ms.append(plan.solve_ms)
            self.q_hold = plan.q[:, 0]
            if self.keep_plans:
                self.plans[k] = plan
        return self.q_hold.copy(), None

    def end_hour(self, k: int, q_delivered: np.ndarray, env: Env) -> None:
        # heat meters and outdoor sensor are known after the fact
        self.est.predict(q_delivered, env.tout[k], env.solar[k], env.gains[:, k])


