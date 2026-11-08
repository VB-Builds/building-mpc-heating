"""Baseline: weather-compensated heating curve plus per-room proportional thermostat valves."""

from __future__ import annotations

import numpy as np

from .schedule import T_LOW_OCC, T_LOW_UNOCC
from .simulate import Env

# Defaults picked by `bmpc tune` (cheapest setting with <= 1 Kh comfort violation on a tuning fortnight, 2024-11-04..18)
OFFSET_OCC = 1.0  # K above the comfort floor while occupied
OFFSET_UNOCC = 0.5
BAND_K = 1.5  # proportional band: valve fully open this far below setpoint
PREHEAT_H = 5  # setpoint returns to the occupied value this many hours early


class BaselineController:
    """`setback=True`: night/absence setback with a fixed pre-heat lead. `False`: constant occupied setpoint."""

    def __init__(self, env: Env, setback: bool = True, preheat_h: int = PREHEAT_H, offset_occ: float = OFFSET_OCC):
        self.env = env
        self.offset_occ = offset_occ
        self.setback = setback
        self.preheat_h = preheat_h
        self.solve_ms: list[float] = []

    def setpoint(self, k: int) -> np.ndarray:
        occ = self.env.occ
        if not self.setback:
            return T_LOW_OCC + self.offset_occ
        ahead = occ[:, k : k + self.preheat_h + 1].any(axis=1)
        return np.where(ahead, T_LOW_OCC + self.offset_occ, T_LOW_UNOCC + OFFSET_UNOCC)

    def act(self, k: int, s: int, y: np.ndarray):
        valve = np.clip((self.setpoint(k) - y) / BAND_K, 0.0, 1.0)
        return None, valve
