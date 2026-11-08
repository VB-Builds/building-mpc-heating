import numpy as np
import pytest

from bmpc import heatpump as hp
from bmpc.building import FA, GreyBox, Plant, default_guess, discretize


def test_cop_carnot_value():
    assert hp.cop(7.0, 35.0) == pytest.approx(0.4 * 308.15 / 28.0)


def test_cop_clipped_and_monotonic():
    assert hp.cop(30.0, 31.0) <= hp.COP_MAX
    assert hp.cop(-30.0, 60.0) >= hp.COP_MIN
    t = np.linspace(-10, 15, 20)
    assert (np.diff(hp.cop(t, 35.0)) >= 0).all()  # warmer outside -> better COP


def test_heating_curve_bounds_and_slope():
    t = np.linspace(-20, 30, 30)
    c = hp.heating_curve(t)
    assert c.min() >= hp.T_SUP_MIN and c.max() <= hp.T_SUP_MAX
    assert (np.diff(c) <= 0).all()
    assert hp.heating_curve(0.0) == pytest.approx(39.0)


def test_discretize_scalar_matches_analytic():
    a, b, dt = -0.5, 2.0, 1.5
    ad, bd = discretize(np.array([[a]]), np.array([[b]]), dt)
    assert ad[0, 0] == pytest.approx(np.exp(a * dt))
    assert bd[0, 0] == pytest.approx((b / a) * (np.exp(a * dt) - 1))


def test_vector_roundtrip(model):
    again = GreyBox.from_vector(model.to_vector())
    assert np.allclose(again.to_vector(), model.to_vector())
    with pytest.raises(ValueError):
        GreyBox.from_vector(np.ones(5))


def test_from_dict_validation(model):
    d = model.to_dict()
    assert np.allclose(GreyBox.from_dict(d).Ci, model.Ci)
    bad = dict(d)
    del bad["Ce"]
    with pytest.raises(ValueError, match="missing"):
        GreyBox.from_dict(bad)
    with pytest.raises(ValueError, match="positive"):
        GreyBox.from_dict(dict(d, Ci=[-1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="values"):
        GreyBox.from_dict(dict(d, Ci=[1.0, 1.0]))


def test_unheated_building_relaxes_to_outdoor(model):
    u = np.zeros((24 * 60, 8))
    u[:, 3] = -5.0
    x = model.simulate(np.full(6, 20.0), u)
    assert np.allclose(x[-1], -5.0, atol=0.05)


def test_steady_state_heat_balance():
    # three identical zones: inter-zone flows cancel, so Ti - Tout = Q / UA per zone
    g = GreyBox(*[np.full(3, v) for v in (1.5, 8.0, 0.4, 0.14, 0.02, 1.0, 0.05)])
    u = np.zeros((24 * 100, 8))
    u[:, :3] = 2.0
    x = g.simulate(np.zeros(6), u)
    ua_zone = g.ua_total() / 3
    # part of the emitter heat (1 - FA) enters the envelope node directly and bypasses the air node
    direct = 1.0 - g.Geo[0] * (1.0 - FA) / (g.Gie[0] + g.Geo[0])
    assert x[-1, 0] == pytest.approx(2.0 * direct / ua_zone, rel=1e-3)


def test_solar_and_gains_warm_the_room(model):
    u = np.zeros((48, 8))
    base = model.simulate(np.full(6, 10.0), u)[-1, 0]
    u[:, 4] = 0.5
    sun = model.simulate(np.full(6, 10.0), u)[-1, 0]
    u[:, 4] = 0.0
    u[:, 5] = 1.0
    gain = model.simulate(np.full(6, 10.0), u)[-1, 0]
    assert sun > base and gain > base


def test_default_guess_is_valid():
    assert (default_guess().to_vector() > 0).all()


def test_plant_is_stable_and_relaxes():
    p = Plant()
    assert np.abs(np.linalg.eigvals(p.ad)).max() < 1.0
    p.reset(20.0)
    for _ in range(24 * 4 * 60):
        p.step(np.zeros(3), 0.0, 0.0, np.zeros(3), 1.0)
    assert np.allclose(p.t_air, 0.0, atol=0.3)


def test_plant_loss_scale_cools_faster():
    cold = []
    for ls in (0.8, 1.25):
        p = Plant(loss_scale=ls)
        p.reset(20.0)
        for _ in range(24 * 4 * 3):
            p.step(np.zeros(3), 0.0, 0.0, np.zeros(3), 1.0)
        cold.append(p.t_air.mean())
    assert cold[1] < cold[0]


def test_plant_delivered_limits_and_power():
    p = Plant()
    p.reset(20.0)
    q, tsup, p_el = p.delivered(np.array([5.0, 5.0, 5.0]), None, 0.0)
    assert q.sum() == pytest.approx(hp.Q_HP_MAX)
    assert p_el == pytest.approx(q.sum() / hp.cop(0.0, tsup))
    q0, _, p0 = p.delivered(None, np.zeros(3), 0.0)
    assert q0.sum() == 0 and p0 == 0
    qv, tv, _ = p.delivered(None, np.ones(3), 0.0)
    assert tv == pytest.approx(float(hp.heating_curve(0.0)))
    assert (qv <= hp.Q_ZONE_MAX + 1e-9).all()


def test_plant_rejects_bad_mode():
    with pytest.raises(ValueError):
        Plant(supply_mode="magic")
