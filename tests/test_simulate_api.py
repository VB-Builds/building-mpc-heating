import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from bmpc import cli
from bmpc.api import load_model, make_server, plan_request
from bmpc.baseline import BaselineController
from bmpc.building import Plant
from bmpc.mpc import MPCController
from bmpc.simulate import SUB, Result, forecast, metrics, run

# ------------------------------------------------------------------ simulation


def test_forecast_perfect_and_noisy(env):
    rng = np.random.default_rng(0)
    t0, s0 = forecast(env, 10, 24, 0.0, rng)
    assert np.array_equal(t0, env.tout[10:34])
    t1, s1 = forecast(env, 10, 24, 1.0, np.random.default_rng(0))
    assert not np.array_equal(t1, t0) and (s1 >= 0).all()
    t2, _ = forecast(env, 10, 24, 1.0, np.random.default_rng(0))
    assert np.array_equal(t1, t2)  # deterministic given the generator


def test_forecast_error_grows_with_lead():
    from bmpc.simulate import make_env
    from bmpc.weather import synthetic_weather

    e = make_env(synthetic_weather(24 * 30, seed=2))
    rng = np.random.default_rng(1)
    errs = np.array([forecast(e, 0, 24, 1.0, rng)[0] - e.tout[:24] for _ in range(400)])
    sd = errs.std(axis=0)
    assert sd[-4:].mean() > 2.0 * sd[:3].mean()


def test_forecast_past_end_raises(env):
    with pytest.raises(ValueError):
        forecast(env, env.n - 5, 24, 1.0, np.random.default_rng(0))


def test_metrics_known_values():
    steps = 8
    res = Result(
        k0=0, k1=2, ti=np.full((steps, 3), 19.0), q=np.full((steps, 3), 1.0), p_el=np.full(steps, 2.0),
        tsup=np.full(steps, 35.0), price=np.full(steps, 0.5), low=np.full((steps, 3), 20.0),
        high=np.full((steps, 3), 25.0),
    )
    m = metrics(res)
    assert m["electricity_kwh"] == pytest.approx(2.0 * 8 * 0.25)
    assert m["cost_chf"] == pytest.approx(2.0 * 8 * 0.25 * 0.5)
    assert m["comfort_kh"] == pytest.approx(1.0 * 3 * 8 * 0.25)  # 1 K short in 3 zones for 2 h
    assert m["peak_kw"] == 2.0
    assert m["scop"] == pytest.approx(1.5)
    skipped = metrics(res, skip_hours=1)
    assert skipped["electricity_kwh"] == pytest.approx(2.0)


def test_run_rejects_bad_window(env):
    ctl = BaselineController(env)
    with pytest.raises(ValueError):
        run(env, ctl, Plant(), 5, 5)
    with pytest.raises(ValueError):
        run(env, ctl, Plant(), 0, env.n + 1)


def test_baseline_setback_lowers_setpoint_when_unoccupied(env):
    sp_on = BaselineController(env, setback=True)
    sp_flat = BaselineController(env, setback=False)
    k_night = next(k for k in range(env.n) if not env.occ[2, k : k + 6].any())  # office empty for 6 h
    assert sp_on.setpoint(k_night)[2] < sp_flat.setpoint(k_night)[2]
    assert (sp_flat.setpoint(0) == sp_flat.setpoint(7)).all()


def test_baseline_valve_saturates(env):
    ctl = BaselineController(env)
    _, v = ctl.act(0, 0, np.array([5.0, 5.0, 5.0]))
    assert (v == 1.0).all()
    _, v = ctl.act(0, 0, np.array([40.0, 40.0, 40.0]))
    assert (v == 0.0).all()


def test_closed_loop_baseline_keeps_rooms_sane(env):
    res = run(env, BaselineController(env), Plant(), 0, 72)
    assert 10 < res.ti.min() and res.ti.max() < 30
    assert res.p_el.sum() > 0
    assert len(res.p_el) == 72 * SUB


def test_closed_loop_mpc_runs_and_holds_comfort(env, model):
    ctl = MPCController(env, model, horizon=24)
    res = run(env, ctl, Plant(), 0, 96)
    m = metrics(res, 48)
    assert len(ctl.solve_ms) == 96
    assert m["comfort_kh"] < 8.0
    assert m["mpc_ms_mean"] > 0


# ------------------------------------------------------------------ api


def _request(h=24):
    return {
        "room_temp_c": [19.0, 18.0, 20.0],
        "outdoor_temp_c": [0.0] * h,
        "solar_w_m2": [0.0] * h,
        "comfort_low_c": [20.5, 18.0, 21.0],
        "price_chf_kwh": [0.3] * h,
    }


def test_plan_request_returns_setpoints(model):
    out = plan_request(_request(), model)
    assert out["status"] == "optimal"
    assert set(out["heat_kw"]) == {"living", "bedroom", "office"}
    assert out["heat_kw"]["office"] > 0
    assert len(out["predicted_room_temp_c"]["living"]) == 25
    assert 25 <= out["supply_temp_c"] <= 50


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda r: r.pop("room_temp_c"), "room_temp_c"),
        (lambda r: r.pop("outdoor_temp_c"), "outdoor_temp_c"),
        (lambda r: r.update(room_temp_c=[1.0, 2.0]), "room_temp_c"),
        (lambda r: r.update(outdoor_temp_c=[0.0]), "horizon"),
        (lambda r: r.update(solar_w_m2=[-5.0] * 24), "solar"),
        (lambda r: r.update(comfort_low_c=[[20.0] * 5] * 3), "comfort_low_c"),
        (lambda r: r.update(price_chf_kwh=[float("nan")] * 24), "price"),
    ],
)
def test_plan_request_rejects_bad_input(model, mutate, msg):
    r = _request()
    mutate(r)
    with pytest.raises(ValueError, match=msg):
        plan_request(r, model)


def test_plan_request_requires_object(model):
    with pytest.raises(ValueError):
        plan_request([1, 2, 3], model)


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.json")


def test_rest_server_roundtrip(model):
    srv = make_server(model, "127.0.0.1", 0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        assert json.load(urllib.request.urlopen(base + "/health"))["status"] == "ok"
        assert "Ci" in json.load(urllib.request.urlopen(base + "/model"))["model"]
        req = urllib.request.Request(base + "/plan", data=json.dumps(_request()).encode(), method="POST")
        out = json.load(urllib.request.urlopen(req))
        assert out["heat_kw"]["office"] > 0
        bad = urllib.request.Request(base + "/plan", data=b'{"outdoor_temp_c": [1]}', method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
        notjson = urllib.request.Request(base + "/plan", data=b"not json", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(notjson)
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/nothing")
        assert e.value.code == 404
    finally:
        srv.shutdown()
        srv.server_close()


# ------------------------------------------------------------------ cli


def test_cli_plan_prints_json(tmp_path, model, capsys):
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps({"model": model.to_dict()}))
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(_request()))
    assert cli.main(["plan", str(rp), "--model", str(mp)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "optimal"


def test_cli_plan_bad_request_exit_code(tmp_path, model, capsys):
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(model.to_dict()))
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"outdoor_temp_c": [1.0] * 24}))
    assert cli.main(["plan", str(rp), "--model", str(mp)]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])
