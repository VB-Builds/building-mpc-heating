"""Command line: demo, tune, sysid, plan, serve."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import experiments as ex
from .api import DEFAULT_MODEL, load_model, make_server, plan_request


def _fmt_row(name: str, m: dict) -> str:
    return (f"{name:<30}{m['electricity_kwh']:>10.0f}{m['cost_chf']:>10.1f}{m['comfort_kh']:>9.1f}"
            f"{m['peak_kw']:>8.2f}{m['scop']:>7.2f}")


def cmd_demo(args) -> int:
    from .figures import make_main_figure, make_post_figure

    res = ex.run_all(smoke=args.smoke)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ex.save_results(res, out / "results.json")
    ex.save_series(res, out / "series.npz")
    (out / "model_identified.json").write_text(json.dumps({"model": res["model"]}, indent=2), encoding="utf-8")
    make_main_figure(res, out / "figure.png")
    make_post_figure(res, out / "post-figure.png")
    if args.docs_dir:
        d = Path(args.docs_dir)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "figure.png", d / "figure.png")
        shutil.copy(out / "post-figure.png", d / "post-figure.png")
    sc = res["scenarios"]
    print(f"\nwindow {res['window'][0]} .. {res['window'][1]}   (kWh = electricity)")
    print(f"{'case':<30}{'kWh':>10}{'CHF':>10}{'Kh out':>9}{'peak kW':>8}{'SCOP':>7}")
    print(_fmt_row("baseline (curve+thermostat)", sc["baseline"]))
    print(_fmt_row("baseline, constant setpoint", sc["baseline_constant_setpoint"]))
    print(_fmt_row("MPC", sc["mpc"]))
    m = sc["mpc"]
    print(f"MPC vs baseline: cost {m['cost_saving_pct']:+.1f}%  electricity {m['electricity_saving_pct']:+.1f}%")
    print(f"MPC solve time: mean {m['mpc_ms_mean']:.1f} ms, p95 {m['mpc_ms_p95']:.1f} ms")
    print(f"outputs in {out}")
    return 0


def cmd_figures(args) -> int:
    from .figures import make_main_figure, make_post_figure

    out = Path(args.out)
    res = ex.load_for_figures(out)
    make_main_figure(res, out / "figure.png")
    make_post_figure(res, out / "post-figure.png")
    if args.docs_dir:
        d = Path(args.docs_dir)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "figure.png", d / "figure.png")
        shutil.copy(out / "post-figure.png", d / "post-figure.png")
    return 0


def cmd_tune(args) -> int:
    res = ex.tune()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("best:", json.dumps(res["best"], indent=2))
    return 0


def cmd_sysid(args) -> int:
    from .weather import load_weather

    model, stats = ex.identify(load_weather())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    doc = {"model": model.to_dict(), "identification": stats}
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


def cmd_plan(args) -> int:
    try:
        payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
        print(json.dumps(plan_request(payload, load_model(args.model)), indent=2))
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_serve(args) -> int:
    srv = make_server(load_model(args.model), args.host, args.port)
    print(f"listening on http://{args.host}:{srv.server_address[1]}  (POST /plan, GET /health, GET /model)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bmpc", description="Predictive heating controller: simulate, identify, plan.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="identify, simulate baseline vs MPC, write results and figures")
    d.add_argument("--smoke", action="store_true", help="one winter week instead of the full heating season")
    d.add_argument("--out", default="results")
    d.add_argument("--docs-dir", default=None, help="also copy figures here")
    d.set_defaults(fn=cmd_demo)
    f = sub.add_parser("figures", help="rebuild figures from saved results")
    f.add_argument("--out", default="results")
    f.add_argument("--docs-dir", default=None)
    f.set_defaults(fn=cmd_figures)
    t = sub.add_parser("tune", help="grid-search baseline settings and MPC margin on the tuning fortnight")
    t.add_argument("--out", default="results/tuning.json")
    t.set_defaults(fn=cmd_tune)
    s = sub.add_parser("sysid", help="identify the grey-box model and save it")
    s.add_argument("--out", default=str(DEFAULT_MODEL))
    s.set_defaults(fn=cmd_sysid)
    p = sub.add_parser("plan", help="compute next-hour set-points from a JSON request file")
    p.add_argument("request")
    p.add_argument("--model", default=None)
    p.set_defaults(fn=cmd_plan)
    v = sub.add_parser("serve", help="REST server (POST /plan)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8080)
    v.add_argument("--model", default=None)
    v.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
