"""Result figures: README figure (1600x900) and LinkedIn post figure (960x700, dark)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .simulate import SUB  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="findfont")

PAL = {
    "bg": "#16213A", "grid": "#2A3750", "text": "#F8FAFC", "muted": "#A9B6CB", "accent": "#38BDF8",
    "green": "#34D399", "amber": "#FBBF24", "red": "#F87171", "grey": "#64748B", "violet": "#A78BFA",
}
SRC = "Weather: Open-Meteo (Zurich, CC BY 4.0). Building, tariff, occupancy: simulated."


def coldest_window(env, warm_h: int, k1: int, days: int, weekdays=(0,)) -> tuple[int, int]:
    """Hour range [a, b) of the `days`-day span with the lowest mean outdoor temperature, starting at local
    midnight on one of `weekdays` (0 = Monday)."""
    local = env.index.tz_convert("Europe/Zurich")
    starts = [k for k in range(warm_h, k1 - 24 * days) if local[k].hour == 0 and local[k].dayofweek in weekdays]
    if not starts:
        return warm_h, min(k1, warm_h + 24 * days)
    means = [env.tout[k : k + 24 * days].mean() for k in starts]
    a = starts[int(np.argmin(means))]
    return a, a + 24 * days


def _steps(a: int, b: int, warm_h: int) -> slice:
    return slice((a - 0) * SUB, (b - 0) * SUB)


def _x_hours(a: int, b: int) -> np.ndarray:
    return np.arange((b - a) * SUB) / SUB


def _style(dark: bool):
    if dark:
        plt.rcParams.update({
            "figure.facecolor": PAL["bg"], "axes.facecolor": PAL["bg"], "savefig.facecolor": PAL["bg"],
            "axes.edgecolor": PAL["grid"], "axes.labelcolor": PAL["muted"], "xtick.color": PAL["muted"],
            "ytick.color": PAL["muted"], "text.color": PAL["text"], "grid.color": PAL["grid"],
        })
    else:
        plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def _ht_spans(ax, env, a: int, b: int, color: str, alpha: float):
    from .schedule import PRICE_HT

    pr = env.price[a:b] >= PRICE_HT - 1e-9
    h = 0
    while h < len(pr):
        if pr[h]:
            e = h
            while e < len(pr) and pr[e]:
                e += 1
            ax.axvspan(h, e, color=color, alpha=alpha, lw=0)
            h = e
        else:
            h += 1


def make_main_figure(res: dict, path: Path) -> None:
    _style(dark=False)
    env, ser = res["_env"], res["_series"]
    base, mpc = ser["baseline"], ser["mpc"]
    warm = 72
    a, b = coldest_window(env, warm, base.k1, 7)
    sl = _steps(a, b, warm)
    x = _x_hours(a, b)
    sc, bm, mm = res["scenarios"], res["scenarios"]["baseline"], res["scenarios"]["mpc"]
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=100)
    fig.suptitle("Predictive heating (MPC) vs heating curve + thermostat: simulated 3-zone building, "
                 f"{res['window'][0]} to {res['window'][1]}", x=0.01, ha="left", fontsize=14, fontweight="bold")

    ax = axs[0, 0]
    z = 2
    _ht_spans(ax, env, a, b, "#fde68a", 0.35)
    ax.step(x, env.low[z, a:b].repeat(SUB), color="#94a3b8", lw=1.2, label="comfort floor")
    ax.plot(x, base.ti[sl, z], color="#dc2626", lw=1.3, label="baseline")
    ax.plot(x, mpc.ti[sl, z], color="#0284c7", lw=1.5, label="MPC")
    ax.set_title("Office air temperature, coldest week (shaded = high-tariff hours)", fontsize=11, loc="left")
    ax.set_ylabel("degC")
    ax.legend(loc="upper right", fontsize=9, ncol=3)

    ax = axs[0, 1]
    _ht_spans(ax, env, a, b, "#fde68a", 0.35)
    ax.plot(x, base.p_el[sl], color="#dc2626", lw=1.0, label="baseline")
    ax.plot(x, mpc.p_el[sl], color="#0284c7", lw=1.2, label="MPC")
    ax.set_title("Heat pump electric power, same week", fontsize=11, loc="left")
    ax.set_ylabel("kW")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    for ax in axs[0]:
        ax.set_xlim(0, (b - a))
        ax.set_xticks(np.arange(0, b - a + 1, 24))
        ax.set_xticklabels([f"d{d + 1}" for d in range(0, 8)])
        ax.grid(alpha=0.3)

    ax = axs[1, 0]
    labels, vals, cols = [], [], []
    spec = [
        ("MPC (nominal)", mm["cost_saving_pct"], "#0284c7"),
        ("perfect forecast", sc["mpc_perfect_forecast"]["cost_saving_pct"], "#0284c7"),
        ("forecast error x3", sc["mpc_forecast_error_x3"]["cost_saving_pct"], "#0284c7"),
        ("envelope loss -20%", sc["mismatch_envelope_loss_x0.8"]["mpc"]["cost_saving_pct"], "#d97706"),
        ("envelope loss +25%", sc["mismatch_envelope_loss_x1.25"]["mpc"]["cost_saving_pct"], "#d97706"),
        ("same supply-temp rule", sc["mpc_same_supply_rule"]["cost_saving_pct"], "#7c3aed"),
        ("plans on flat price", sc["mpc_flat_price_plan"]["cost_saving_pct"], "#7c3aed"),
    ]
    for lab, v, c in spec:
        labels.append(lab)
        vals.append(v)
        cols.append(c)
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, vals, color=cols)
    for yv, v in zip(yy, vals, strict=True):
        ax.text(v + (0.15 if v >= 0 else -0.15), yv, f"{v:.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#475569", lw=0.8)
    ax.set_xlim(min(0, min(vals) - 1), max(vals) * 1.15)
    ax.set_title("Season cost saving vs baseline, sensitivity checks (%)", fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.3)

    ax = axs[1, 1]
    dt = 1.0 / SUB
    w0 = warm * SUB
    cb = np.cumsum(base.p_el[w0:] * base.price[w0:]) * dt
    cm = np.cumsum(mpc.p_el[w0:] * mpc.price[w0:]) * dt
    xs = np.arange(len(cb)) / SUB / 24
    ax.plot(xs, cb, color="#dc2626", label=f"baseline {bm['cost_chf']:.0f} CHF, {bm['comfort_kh']:.1f} Kh outside band")
    ax.plot(xs, cm, color="#0284c7", label=f"MPC {mm['cost_chf']:.0f} CHF, {mm['comfort_kh']:.1f} Kh outside band")
    ax.set_title("Cumulative electricity cost over the evaluation window", fontsize=11, loc="left")
    ax.set_xlabel("day")
    ax.set_ylabel("CHF")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.005, SRC, fontsize=8, color="#64748B")
    fig.tight_layout(rect=(0, 0.015, 1, 0.96))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def make_post_figure(res: dict, path: Path) -> None:
    _style(dark=True)
    env, ser = res["_env"], res["_series"]
    base, mpc = ser["baseline"], ser["mpc"]
    warm = 72
    a, b = coldest_window(env, warm, base.k1, 3, weekdays=(0, 1, 2))
    sl = _steps(a, b, warm)
    x = _x_hours(a, b)
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=PAL["bg"])
    fig.text(0.04, 0.955, "MPC pre-heats in cheap hours, same comfort", fontsize=14, fontweight="bold",
             color=PAL["text"], va="center")
    fig.text(0.04, 0.915, "Simulated 3-zone building | Zurich weather: Open-Meteo, coldest 3 days",
             fontsize=11.5, color=PAL["muted"], va="center")
    ax1 = fig.add_axes((0.11, 0.55, 0.86, 0.29))
    ax2 = fig.add_axes((0.11, 0.10, 0.86, 0.34), sharex=ax1)
    for ax in (ax1, ax2):
        ax.set_facecolor(PAL["bg"])
        _ht_spans(ax, env, a, b, PAL["amber"], 0.22)
        ax.grid(color=PAL["grid"], lw=0.8)
        for sp in ax.spines.values():
            sp.set_color(PAL["grid"])
        ax.tick_params(labelsize=11.5, colors=PAL["muted"])
    z = 2
    ax1.step(x, env.low[z, a:b].repeat(SUB), color=PAL["grey"], lw=2, label="comfort floor")
    ax1.plot(x, base.ti[sl, z], color=PAL["red"], lw=2.4, label="baseline")
    ax1.plot(x, mpc.ti[sl, z], color=PAL["accent"], lw=2.6, label="MPC")
    ax1.set_ylabel("Office, degC", fontsize=11.5, color=PAL["muted"])
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=11.5, ncol=3, frameon=False,
               labelcolor=PAL["text"])
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.plot(x, base.p_el[sl], color=PAL["red"], lw=2.0, label="baseline")
    ax2.plot(x, mpc.p_el[sl], color=PAL["accent"], lw=2.2, label="MPC")
    ax2.set_ylabel("Electric power, kW", fontsize=11.5, color=PAL["muted"])
    ax2.set_xlim(0, b - a)
    ax2.set_xticks(np.arange(0, b - a + 1, 12))
    ax2.set_xticklabels([f"{(h % 24):02d}:00" for h in range(0, b - a + 1, 12)])
    ax2.set_xlabel("local time (amber = high-tariff hours)", fontsize=11.5, color=PAL["muted"])
    fig.savefig(path, dpi=100, facecolor=PAL["bg"])
    plt.close(fig)
