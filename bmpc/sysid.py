"""Grey-box identification of the 2R2C multi-zone model from logged (simulated) sensor data.

Excitation: random-hold heat inputs per zone on the plant, room temperatures sampled hourly with
sensor noise. The fit minimises free-run (multi-step) prediction error, not one-step error, so the
parameters are good for the 24 h predictions an MPC needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .building import N_IN, N_STATE, GreyBox, Plant, default_guess
from .schedule import NZ
from .simulate import SENSOR_SIGMA, SUB, Env, run


class PRBSController:
    """Random-hold heat input per zone, levels in kW."""

    LEVELS = np.array([0.0, 0.8, 1.8, 3.0])

    def __init__(self, n_hours: int, seed: int = 3):
        rng = np.random.default_rng(seed)
        self.q = np.zeros((n_hours, NZ))
        for z in range(NZ):
            k = 0
            while k < n_hours:
                hold = int(rng.integers(1, 9))
                self.q[k : k + hold, z] = rng.choice(self.LEVELS)
                k += hold
        self.k0 = 0
        self.solve_ms: list[float] = []

    def act(self, k: int, s: int, y: np.ndarray):
        return self.q[k - self.k0].copy(), None


@dataclass
class Experiment:
    y: np.ndarray  # (N+1, NZ) noisy hourly room temperature
    y_true: np.ndarray
    u: np.ndarray  # (N, N_IN) model inputs
    t_hours: int


def make_experiment(env: Env, k0: int, hours: int, seed: int = 3) -> Experiment:
    """Run the excitation on the plant starting at env interval k0."""
    ctl = PRBSController(hours, seed)
    ctl.k0 = k0
    plant = Plant()
    res = run(env, ctl, plant, k0, k0 + hours, seed=seed + 100, t_init=20.0)
    rng = np.random.default_rng(seed + 200)
    ti = res.ti[::SUB]  # hour starts
    last = np.vstack([ti, plant.t_air[None, :]])
    y_true = last
    y = y_true + rng.normal(0.0, SENSOR_SIGMA, y_true.shape)
    q_h = res.q.reshape(hours, SUB, NZ).mean(axis=1)
    sl = slice(k0, k0 + hours)
    u = np.column_stack([q_h, env.tout[sl], env.solar[sl], env.gains[:, sl].T])
    assert u.shape[1] == N_IN
    return Experiment(y=y, y_true=y_true, u=u, t_hours=hours)


def _simulate(theta: np.ndarray, y0: np.ndarray, u: np.ndarray) -> np.ndarray:
    n_par = 7 * NZ
    model = GreyBox.from_vector(np.exp(theta[:n_par]))
    x0 = np.zeros(N_STATE)
    x0[0::2] = y0
    x0[1::2] = y0 + theta[n_par:]
    return model.simulate(x0, u)[:, 0::2]


def fit(exp: Experiment, n_train: int, ridge: float = 0.02, max_nfev: int = 200) -> tuple[GreyBox, dict]:
    """Fit on the first n_train hours, validate (free run continuing) on the rest."""
    if not 24 <= n_train <= exp.t_hours:
        raise ValueError("n_train must be between 24 and the experiment length")
    g0 = default_guess().to_vector()
    th0 = np.concatenate([np.log(g0), np.zeros(NZ)])
    lo = np.concatenate([np.log(g0) - 3.0, np.full(NZ, -10.0)])
    hi = np.concatenate([np.log(g0) + 3.0, np.full(NZ, 10.0)])
    ytr = exp.y[: n_train + 1]

    def resid(theta):
        pred = _simulate(theta, exp.y[0], exp.u[:n_train])
        r = (pred - ytr).ravel() / SENSOR_SIGMA
        reg = ridge * (theta[: 7 * NZ] - np.log(g0))
        return np.concatenate([r, reg])

    sol = least_squares(resid, th0, bounds=(lo, hi), max_nfev=max_nfev, x_scale=1.0)
    model = GreyBox.from_vector(np.exp(sol.x[: 7 * NZ]))
    full = _simulate(sol.x, exp.y[0], exp.u)
    err_tr = full[: n_train + 1] - exp.y_true[: n_train + 1]
    stats = {
        "train_hours": int(n_train),
        "train_rmse_k": float(np.sqrt(np.mean(err_tr**2))),
        "train_rmse_vs_noisy_k": float(np.sqrt(np.mean((full[: n_train + 1] - ytr) ** 2))),
        "sensor_noise_k": SENSOR_SIGMA,
        "iterations": int(sol.nfev),
    }
    if exp.t_hours > n_train:
        err_va = full[n_train:] - exp.y_true[n_train:]
        stats["val_hours"] = int(exp.t_hours - n_train)
        stats["val_rmse_k"] = float(np.sqrt(np.mean(err_va**2)))
        stats["val_rmse_per_zone_k"] = [float(v) for v in np.sqrt(np.mean(err_va**2, axis=0))]
    return model, stats


def horizon_error(model: GreyBox, exp: Experiment, start: int, hours: int = 24) -> float:
    """RMSE (K, vs noise-free truth) of a `hours`-ahead prediction started at `start` using a state guess
    taken from the measurements only (Te = Ti). Shows how good the model is as an MPC predictor."""
    x0 = np.zeros(N_STATE)
    x0[0::2] = exp.y[start]
    x0[1::2] = exp.y[start]
    pred = model.simulate(x0, exp.u[start : start + hours])[:, 0::2]
    return float(np.sqrt(np.mean((pred - exp.y_true[start : start + hours + 1]) ** 2)))
