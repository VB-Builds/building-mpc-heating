"""Season experiments: identification, baseline vs MPC, ablations and sensitivity checks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baseline import BaselineController
from .building import GreyBox, Plant
from .mpc import MPCController
from .simulate import Env, Result, make_env, metrics, run
from .sysid import fit, horizon_error, make_experiment
from .weather import Weather, load_weather

WARMUP_H = 72
PAD_AFTER_H = 48
SEASON = ("2024-10-15", "2025-04-15")
SMOKE = ("2024-12-09", "2024-12-16")
TUNE = ("2024-11-04", "2024-11-18")
SYSID_WINDOW = ("2024-09-28", "2024-10-26")
SYSID_HOURS = 21 * 24
SYSID_TRAIN_H = 16 * 24


@dataclass
class Prepared:
    model: GreyBox
    fit_stats: dict
    wx: Weather


def identify(wx: Weather) -> tuple[GreyBox, dict]:
    w, _ = wx.slice_local(*SYSID_WINDOW)
    env = make_env(w, seed=5)
    exp = make_experiment(env, 0, SYSID_HOURS, seed=3)
    model, stats = fit(exp, SYSID_TRAIN_H)
    # 24 h-ahead prediction error from measured room temperatures only (Te unknown), on held-out days
    errs = [horizon_error(model, exp, s) for s in range(SYSID_TRAIN_H, SYSID_HOURS - 24, 12)]
    stats["horizon24_rmse_k_mean"] = float(np.mean(errs))
    stats["horizon24_rmse_k_max"] = float(np.max(errs))
    return model, stats


def season_env(wx: Weather, window: tuple[str, str], seed: int) -> tuple[Env, int, int]:
    w, off = wx.slice_local(window[0], window[1], pad_before_h=WARMUP_H, pad_after_h=PAD_AFTER_H)
    env = make_env(w, seed=seed)
    return env, 0, env.n - PAD_AFTER_H


def run_case(env: Env, k1: int, kind: str, model: GreyBox, loss_scale: float = 1.0, **kw) -> Result:
    plant = Plant(loss_scale=loss_scale, supply_mode=kw.pop("supply_mode", "demand"))
    if kind == "baseline":
        ctl = BaselineController(env, **kw)
    elif kind == "mpc":
        ctl = MPCController(env, model, **kw)
    else:
        raise ValueError(kind)
    return run(env, ctl, plant, 0, k1, seed=7)


def _pct(new: float, ref: float) -> float:
    return float(100.0 * (ref - new) / ref) if ref else float("nan")


def compare(m: dict, ref: dict) -> dict:
    return {
        "cost_saving_pct": _pct(m["cost_chf"], ref["cost_chf"]),
        "electricity_saving_pct": _pct(m["electricity_kwh"], ref["electricity_kwh"]),
        "heat_delta_pct": -_pct(m["heat_kwh"], ref["heat_kwh"]),
    }


def run_all(smoke: bool = False, log=print) -> dict:
    """Run every scenario. Returns the results dict; time series for the example week under '_series'."""
    wx = load_weather()
    t0 = time.time()
    model, fit_stats = identify(wx)
    log(f"identified model: validation RMSE {fit_stats.get('val_rmse_k', float('nan')):.2f} K")
    window = SMOKE if smoke else SEASON
    seeds = [1] if smoke else [1, 2, 3]
    out: dict = {
        "window": list(window),
        "smoke": smoke,
        "identification": fit_stats,
        "model": model.to_dict(),
        "scenarios": {},
        "seeds": {},
    }
    series: dict[str, Result] = {}

    def do(env, k1, name, kind, loss=1.0, keep=False, **kw):
        t = time.time()
        res = run_case(env, k1, kind, model, loss_scale=loss, **kw)
        mt = metrics(res, WARMUP_H)
        log(f"  {name:<28} cost {mt['cost_chf']:8.2f} CHF  elec {mt['electricity_kwh']:7.1f} kWh  "
            f"comfort {mt['comfort_kh']:6.1f} Kh  ({time.time() - t:.0f}s)")
        if keep:
            series[name] = res
        return mt

    for sd in seeds:
        env, _, k1 = season_env(wx, window, seed=sd)
        keep = sd == 1
        log(f"seed {sd}: {window[0]} .. {window[1]} ({(k1 - WARMUP_H) // 24} days)")
        base = do(env, k1, "baseline", "baseline", keep=keep)
        mpc = do(env, k1, "mpc", "mpc", keep=keep)
        out["seeds"][str(sd)] = {"baseline": base, "mpc": mpc, **compare(mpc, base)}
        if sd != 1:
            continue
        sc = out["scenarios"]
        sc["baseline"] = base
        sc["mpc"] = {**mpc, **compare(mpc, base)}
        const = do(env, k1, "baseline_constant_setpoint", "baseline", setback=False, offset_occ=0.7, keep=keep)
        sc["baseline_constant_setpoint"] = const
        sc["mpc_vs_constant_setpoint"] = compare(mpc, const)
        sc["mpc_same_supply_rule"] = {
            **(m := do(env, k1, "mpc_same_supply_rule", "mpc", supply_mode="curve")),
            **compare(m, base),
        }
        sc["mpc_flat_price_plan"] = {
            **(m := do(env, k1, "mpc_flat_price_plan", "mpc", flat_price=True)),
            **compare(m, base),
        }
        for label, scale in (("perfect_forecast", 0.0), ("forecast_error_x3", 3.0)):
            sc[f"mpc_{label}"] = {
                **(m := do(env, k1, f"mpc_{label}", "mpc", forecast_scale=scale)),
                **compare(m, base),
            }
        for label, loss in (("envelope_loss_x0.8", 0.8), ("envelope_loss_x1.25", 1.25)):
            b2 = do(env, k1, f"baseline_{label}", "baseline", loss=loss)
            m2 = do(env, k1, f"mpc_{label}", "mpc", loss=loss)
            sc[f"mismatch_{label}"] = {"baseline": b2, "mpc": {**m2, **compare(m2, b2)}}

    out["runtime_s"] = time.time() - t0
    out["_series"] = series
    out["_env"] = season_env(wx, window, seed=1)[0]
    return out


def tune(log=print) -> dict:
    """Pick baseline (offset, preheat) and MPC margin on the tuning fortnight: cheapest with <= 1 Kh."""
    wx = load_weather()
    model, _ = identify(wx)
    env, _, k1 = season_env(wx, TUNE, seed=1)
    rows = []
    for off in (0.7, 1.0, 1.2, 1.5):
        for ph in (2, 3, 4, 5, 6):
            mt = metrics(run_case(env, k1, "baseline", model, offset_occ=off, preheat_h=ph), WARMUP_H)
            rows.append({"kind": "baseline", "offset_occ": off, "preheat_h": ph, **mt})
            log(f"baseline off={off} ph={ph}: {mt['cost_chf']:.2f} CHF {mt['comfort_kh']:.1f} Kh")
    for mg in (0.0, 0.15, 0.3, 0.45, 0.6, 0.8):
        mt = metrics(run_case(env, k1, "mpc", model, margin=mg), WARMUP_H)
        rows.append({"kind": "mpc", "margin": mg, **mt})
        log(f"mpc margin={mg}: {mt['cost_chf']:.2f} CHF {mt['comfort_kh']:.1f} Kh")
    best = {}
    for kind in ("baseline", "mpc"):
        ok = [r for r in rows if r["kind"] == kind and r["comfort_kh"] <= 1.0]
        best[kind] = min(ok, key=lambda r: r["cost_chf"]) if ok else None
    return {"window": list(TUNE), "rows": rows, "best": best}


def save_series(res: dict, path: Path) -> None:
    """Keep the 15-minute traces of the nominal baseline and MPC runs so figures can be rebuilt offline."""
    arrays = {}
    for name in ("baseline", "mpc"):
        r = res["_series"][name]
        for f in ("ti", "q", "p_el", "tsup", "price", "low", "high"):
            arrays[f"{name}_{f}"] = np.round(getattr(r, f), 4)
    arrays["k1"] = np.array(res["_series"]["baseline"].k1)
    np.savez_compressed(path, **arrays)


def load_for_figures(out_dir: Path) -> dict:
    """Rebuild the objects the figure code needs from results.json + series.npz."""
    meta = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    z = np.load(out_dir / "series.npz")
    k1 = int(z["k1"])
    ser = {}
    for name in ("baseline", "mpc"):
        ser[name] = Result(0, k1, *[z[f"{name}_{f}"] for f in ("ti", "q", "p_el", "tsup", "price", "low", "high")])
    window = tuple(meta["window"])
    env, _, _ = season_env(load_weather(), window, seed=1)
    return {**meta, "_series": ser, "_env": env}


def save_results(res: dict, path: Path) -> None:
    clean = {k: v for k, v in res.items() if not k.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
