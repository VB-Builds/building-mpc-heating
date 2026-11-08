"""Heat pump and emitter model: Carnot-fraction COP and supply-temperature rules."""

from __future__ import annotations

import numpy as np

ETA = 0.40  # fraction of Carnot COP
COP_MIN, COP_MAX = 1.5, 5.5
K_EMITTER = 0.30  # kW/K per zone (floor heating / large radiators)
Q_ZONE_MAX = 5.0  # kW thermal per zone
Q_HP_MAX = 12.0  # kW thermal, whole heat pump
T_SUP_MIN = 25.0
T_SUP_MAX = 50.0


def cop(t_out, t_sup):
    """COP = eta * T_sup / (T_sup - T_out) in kelvin, clipped. Works on arrays."""
    t_out = np.asarray(t_out, dtype=float)
    t_sup = np.asarray(t_sup, dtype=float)
    lift = np.maximum(t_sup - t_out, 5.0)
    return np.clip(ETA * (t_sup + 273.15) / lift, COP_MIN, COP_MAX)


def heating_curve(t_out):
    """Weather-compensated supply temperature used by the baseline (and as MPC COP assumption)."""
    return np.clip(25.0 + 0.7 * (20.0 - np.asarray(t_out, dtype=float)), T_SUP_MIN, T_SUP_MAX)
