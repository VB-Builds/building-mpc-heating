"""Thermal models.

* `GreyBox`: 2R2C per zone (air node Ti, envelope/slab node Te), conductances between zones.
  This is the model the identification fits and the MPC uses.
* `Plant`: a richer 3-node-per-zone network (air, inner mass, outer mass) with different
  parameters, per-day gain noise and 15-minute sub-stepping. It plays the real building, so the
  grey-box model is structurally wrong in a realistic way.

Units: kW, K, kWh/K, hours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from . import heatpump as hp
from .schedule import NZ

FA = 0.6  # share of emitter heat delivered to the air node (rest to the mass node)
PAIRS = ((0, 1), (0, 2), (1, 2))
N_STATE = 2 * NZ
# model input order: Q(3), Tout, Isol[kW/m2], gains(3)
N_IN = NZ + 2 + NZ


def discretize(ac: np.ndarray, bc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation."""
    n, m = bc.shape
    big = np.zeros((n + m, n + m))
    big[:n, :n] = ac
    big[:n, n:] = bc
    e = expm(big * dt)
    return e[:n, :n], e[:n, n:]


def _couple(a: np.ndarray, p: int, q: int, g: float, cp: float, cq: float) -> None:
    a[p, p] -= g / cp
    a[p, q] += g / cp
    a[q, q] -= g / cq
    a[q, p] += g / cq


@dataclass
class GreyBox:
    Ci: np.ndarray
    Ce: np.ndarray
    Gie: np.ndarray
    Geo: np.ndarray
    Gio: np.ndarray
    As: np.ndarray
    Gz: np.ndarray  # coupling for PAIRS

    NAMES = ("Ci", "Ce", "Gie", "Geo", "Gio", "As", "Gz")

    def to_vector(self) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(self, n), dtype=float) for n in self.NAMES])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> GreyBox:
        v = np.asarray(v, dtype=float)
        if v.size != 7 * NZ:
            raise ValueError(f"expected {7 * NZ} parameters, got {v.size}")
        return cls(*[v[i * NZ : (i + 1) * NZ].copy() for i in range(7)])

    def to_dict(self) -> dict[str, list[float]]:
        return {n: [float(x) for x in getattr(self, n)] for n in self.NAMES}

    @classmethod
    def from_dict(cls, d: dict) -> GreyBox:
        try:
            gb = cls(*[np.asarray(d[n], dtype=float) for n in cls.NAMES])
        except KeyError as e:
            raise ValueError(f"model is missing parameter {e}") from e
        if any(np.asarray(getattr(gb, n)).shape != (NZ,) for n in cls.NAMES):
            raise ValueError(f"each parameter needs {NZ} values")
        if (gb.to_vector() <= 0).any():
            raise ValueError("all model parameters must be positive")
        return gb

    def continuous(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.zeros((N_STATE, N_STATE))
        b = np.zeros((N_STATE, N_IN))
        for z in range(NZ):
            i, e = 2 * z, 2 * z + 1
            a[i, i] -= self.Gio[z] / self.Ci[z]
            a[e, e] -= self.Geo[z] / self.Ce[z]
            _couple(a, i, e, self.Gie[z], self.Ci[z], self.Ce[z])
            b[i, z] = FA / self.Ci[z]
            b[e, z] = (1 - FA) / self.Ce[z]
            b[i, NZ] = self.Gio[z] / self.Ci[z]
            b[e, NZ] = self.Geo[z] / self.Ce[z]
            b[i, NZ + 1] = self.As[z] / self.Ci[z]
            b[i, NZ + 2 + z] = 1.0 / self.Ci[z]
        for (p, q), g in zip(PAIRS, self.Gz, strict=True):
            _couple(a, 2 * p, 2 * q, g, self.Ci[p], self.Ci[q])
        return a, b

    def discrete(self, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        return discretize(*self.continuous(), dt)

    def simulate(self, x0: np.ndarray, u: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Free run. u shape (N, N_IN); returns states (N+1, N_STATE)."""
        ad, bd = self.discrete(dt)
        x = np.empty((len(u) + 1, N_STATE))
        x[0] = x0
        for k in range(len(u)):
            x[k + 1] = ad @ x[k] + bd @ u[k]
        return x

    def ua_total(self) -> float:
        """Steady-state conductance to outdoors, summed over zones (kW/K), ignoring inter-zone flow."""
        return float(np.sum(self.Gio + 1.0 / (1.0 / self.Gie + 1.0 / self.Geo)))


def default_guess() -> GreyBox:
    """Generic starting point for identification (not the truth)."""
    return GreyBox(
        Ci=np.full(NZ, 0.8),
        Ce=np.full(NZ, 5.0),
        Gie=np.full(NZ, 0.8),
        Geo=np.full(NZ, 0.08),
        Gio=np.full(NZ, 0.05),
        As=np.full(NZ, 1.5),
        Gz=np.full(NZ, 0.05),
    )


# ---------------------------------------------------------------- plant

P_NODES = 3  # per zone: air, inner mass, outer mass


class Plant:
    """Ground-truth building. `loss_scale` multiplies all envelope conductances (mismatch tests)."""

    DT = 0.25  # h

    def __init__(self, loss_scale: float = 1.0, supply_mode: str = "demand"):
        if supply_mode not in ("demand", "curve"):
            raise ValueError("supply_mode must be 'demand' or 'curve'")
        self.supply_mode = supply_mode
        self.loss_scale = loss_scale
        zs = np.array([1.15, 0.90, 1.00])  # zone envelope size factors
        ci = np.array([1.2, 0.8, 0.9])
        self.aperture = np.array([3.2, 1.1, 2.2])  # m2 effective solar aperture
        n = NZ * P_NODES
        a = np.zeros((n, n))
        b = np.zeros((n, NZ + 1 + NZ + NZ))  # Q(3), Tout, Gsol(3), Gint(3)
        for z in range(NZ):
            i, w, m = 3 * z, 3 * z + 1, 3 * z + 2
            c = (ci[z], 2.5 * zs[z], 6.0 * zs[z])
            gi, gm = 1.2 * zs[z], 0.5 * zs[z]
            go = 0.12 * zs[z] * loss_scale
            gv = 0.04 * zs[z] * loss_scale
            _couple(a, i, w, gi, c[0], c[1])
            _couple(a, w, m, gm, c[1], c[2])
            a[m, m] -= go / c[2]
            b[m, NZ] += go / c[2]
            a[i, i] -= gv / c[0]
            b[i, NZ] += gv / c[0]
            b[i, z] = FA / c[0]
            b[w, z] = (1 - FA) / c[1]
            b[i, NZ + 1 + z] = 0.7 / c[0]
            b[w, NZ + 1 + z] = 0.3 / c[1]
            b[i, NZ + 1 + NZ + z] = 1.0 / c[0]
        gz = np.array([0.06, 0.06, 0.03])
        for (p, q), g in zip(PAIRS, gz, strict=True):
            _couple(a, 3 * p, 3 * q, g, ci[p], ci[q])
        self.ad, self.bd = discretize(a, b, self.DT)
        self.air = np.array([0, 3, 6])
        self.x = np.zeros(n)

    def reset(self, t_init: float = 20.0) -> None:
        self.x = np.full(NZ * P_NODES, float(t_init))

    @property
    def t_air(self) -> np.ndarray:
        return self.x[self.air].copy()

    def step(self, q: np.ndarray, t_out: float, solar_kw_m2: float, gains: np.ndarray, solar_factor: float) -> None:
        u = np.concatenate([q, [t_out], self.aperture * solar_kw_m2 * solar_factor, gains])
        self.x = self.ad @ self.x + self.bd @ u

    def delivered(
        self, q_cmd: np.ndarray | None, valve: np.ndarray | None, t_out: float
    ) -> tuple[np.ndarray, float, float]:
        """Resolve a controller action into (zone heat kW, supply temp degC, electric kW)."""
        ti = self.t_air
        if valve is not None:
            tsup = float(hp.heating_curve(t_out))
            q = np.clip(valve, 0.0, 1.0) * hp.K_EMITTER * np.maximum(tsup - ti, 0.0)
        else:
            q = np.clip(np.asarray(q_cmd, dtype=float), 0.0, hp.Q_ZONE_MAX)
            if self.supply_mode == "curve":
                tsup = float(hp.heating_curve(t_out))
            else:
                need = np.where(q > 1e-9, ti + q / hp.K_EMITTER, hp.T_SUP_MIN)
                tsup = float(np.clip(need.max(), hp.T_SUP_MIN, hp.T_SUP_MAX))
            q = np.minimum(q, hp.K_EMITTER * np.maximum(tsup - ti, 0.0))
        q = np.minimum(q, hp.Q_ZONE_MAX)
        tot = float(q.sum())
        if tot > hp.Q_HP_MAX:
            q = q * (hp.Q_HP_MAX / tot)
            tot = hp.Q_HP_MAX
        p_el = tot / float(hp.cop(t_out, tsup)) if tot > 0 else 0.0
        return q, tsup, p_el
