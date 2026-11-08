"""Stateless planning interface: JSON in, JSON out. Used by `bmpc plan` (CLI) and `bmpc serve` (REST)."""

from __future__ import annotations

import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import heatpump as hp
from .building import GreyBox
from .mpc import Planner
from .schedule import NZ, ZONES

DEFAULT_MODEL = Path(__file__).parent / "data" / "model_identified.json"
MAX_BODY = 1_000_000


def load_model(path: str | Path | None = None) -> GreyBox:
    p = Path(path) if path else DEFAULT_MODEL
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {p} (run `bmpc sysid` to create one)")
    d = json.loads(p.read_text(encoding="utf-8"))
    return GreyBox.from_dict(d["model"] if "model" in d else d)


@lru_cache(maxsize=8)
def _planner(model_key: str, horizon: int) -> Planner:
    return Planner(GreyBox.from_dict(json.loads(model_key)), horizon)


def _vec(payload: dict, key: str, n: int, default=None) -> np.ndarray:
    if key not in payload:
        if default is None:
            raise ValueError(f"missing field: {key}")
        return np.full(n, float(default))
    v = np.asarray(payload[key], dtype=float)
    if v.ndim == 0:
        v = np.full(n, float(v))
    if v.shape != (n,):
        raise ValueError(f"{key} must have {n} values, got shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{key} contains non-finite values")
    return v


def _mat(payload: dict, key: str, h: int, default: float | None = None) -> np.ndarray:
    """(NZ, H) from a scalar, a per-zone list of NZ, or an NZ x H nested list."""
    if key not in payload:
        if default is None:
            raise ValueError(f"missing field: {key}")
        return np.full((NZ, h), default)
    v = np.asarray(payload[key], dtype=float)
    if v.ndim == 0:
        v = np.full((NZ, h), float(v))
    elif v.shape == (NZ,):
        v = np.repeat(v[:, None], h, axis=1)
    if v.shape != (NZ, h):
        raise ValueError(f"{key} must be a scalar, {NZ} values or {NZ}x{h}, got shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{key} contains non-finite values")
    return v


def plan_request(payload: dict, model: GreyBox) -> dict:
    """Compute the next-hour heat set-points for the zones.

    Bounds in `comfort_low_c` / `comfort_high_c` apply to the room temperature at the END of each hour step.
    """
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    if "outdoor_temp_c" not in payload:
        raise ValueError("missing field: outdoor_temp_c")
    h = len(payload["outdoor_temp_c"])
    if not 2 <= h <= 72:
        raise ValueError("horizon (length of outdoor_temp_c) must be between 2 and 72 hours")
    tout = _vec(payload, "outdoor_temp_c", h)
    solar = _vec(payload, "solar_w_m2", h, default=0.0) / 1000.0
    if np.any(solar < 0):
        raise ValueError("solar_w_m2 must be >= 0")
    room = _vec(payload, "room_temp_c", NZ)
    env_t = _vec(payload, "envelope_temp_c", NZ) if "envelope_temp_c" in payload else room
    x0 = np.zeros(2 * NZ)
    x0[0::2], x0[1::2] = room, env_t
    low = _mat(payload, "comfort_low_c", h)
    high = _mat(payload, "comfort_high_c", h, default=27.0)
    gains = _mat(payload, "internal_gains_kw", h, default=0.1)
    price = _vec(payload, "price_chf_kwh", h, default=0.25)
    planner = _planner(json.dumps(model.to_dict(), sort_keys=True), h)
    plan = planner.plan(x0, tout, solar, gains, low, high, price)
    q0 = plan.q[:, 0]
    need = np.where(q0 > 1e-9, room + q0 / hp.K_EMITTER, hp.T_SUP_MIN)
    tsup = float(np.clip(need.max(), hp.T_SUP_MIN, hp.T_SUP_MAX))
    return {
        "status": plan.status,
        "horizon_h": h,
        "heat_kw": {z: round(float(q), 4) for z, q in zip(ZONES, q0, strict=True)},
        "supply_temp_c": round(tsup, 2),
        "predicted_room_temp_c": {z: [round(float(t), 3) for t in plan.ti[i]] for i, z in enumerate(ZONES)},
        "planned_heat_kw": {z: [round(float(t), 4) for t in plan.q[i]] for i, z in enumerate(ZONES)},
        "cost_objective": round(plan.cost, 5),
        "solve_ms": round(plan.solve_ms, 2),
    }


def make_server(model: GreyBox, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/model":
                self._send(200, {"zones": list(ZONES), "model": model.to_dict()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/plan":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n <= 0 or n > MAX_BODY:
                    raise ValueError("body missing or too large")
                payload = json.loads(self.rfile.read(n))
                self._send(200, plan_request(payload, model))
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # solver failure etc.
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):  # quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)
