import numpy as np
import pytest

from bmpc import heatpump as hp
from bmpc.mpc import Estimator, Planner
from bmpc.sysid import Experiment, fit

H = 24


def _arrays(h=H, tout=0.0, low=20.0, high=27.0, price=0.25):
    return {
        "tout": np.full(h, tout),
        "solar": np.zeros(h),
        "gains": np.full((3, h), 0.1),
        "low": np.full((3, h), low),
        "high": np.full((3, h), high),
        "price": np.full(h, price),
    }


def _x0(t=21.0, te=None):
    x = np.zeros(6)
    x[0::2] = t
    x[1::2] = t if te is None else te
    return x


# ------------------------------------------------------------------ identification

def test_fit_recovers_noise_free_data(model):
    rng = np.random.default_rng(0)
    n = 12 * 24
    u = np.zeros((n, 8))
    for z in range(3):
        k = 0
        while k < n:
            hold = int(rng.integers(1, 8))
            u[k : k + hold, z] = rng.choice([0.0, 0.8, 1.8, 3.0])
            k += hold
    u[:, 3] = 5.0 + 4.0 * np.sin(np.arange(n) / 24 * 2 * np.pi)
    u[:, 4] = np.clip(np.sin((np.arange(n) % 24 - 6) / 12 * np.pi), 0, None) * 0.2
    u[:, 5:] = 0.1
    x = model.simulate(np.full(6, 20.0), u)
    y = x[:, 0::2]
    exp = Experiment(y=y, y_true=y, u=u, t_hours=n)
    fitted, stats = fit(exp, 9 * 24)
    assert stats["train_rmse_k"] < 0.15
    assert stats["val_rmse_k"] < 0.3
    assert len(stats["val_rmse_per_zone_k"]) == 3
    assert (fitted.to_vector() > 0).all()


def test_fit_rejects_bad_train_length(model):
    exp = Experiment(y=np.zeros((50, 3)), y_true=np.zeros((50, 3)), u=np.zeros((49, 8)), t_hours=49)
    with pytest.raises(ValueError):
        fit(exp, 10)
    with pytest.raises(ValueError):
        fit(exp, 500)


# ------------------------------------------------------------------ planner

def test_planner_idle_when_warm_and_mild(model):
    p = Planner(model, H)
    plan = p.plan(_x0(23.0), **_arrays(tout=18.0, low=18.0))
    assert plan.status == "optimal"
    assert plan.q[:, :3].max() < 1e-3


def test_planner_heats_when_cold(model):
    p = Planner(model, H)
    plan = p.plan(_x0(16.0), **_arrays(tout=-5.0))
    assert plan.q[:, 0].sum() > 1.0


def test_planner_respects_limits_and_band(model):
    p = Planner(model, H)
    plan = p.plan(_x0(20.5), **_arrays(tout=-8.0, low=20.0))
    assert plan.q.min() >= -1e-9
    assert plan.q.max() <= hp.Q_ZONE_MAX + 1e-6
    assert plan.q.sum(axis=0).max() <= hp.Q_HP_MAX + 1e-6
    assert plan.ti[:, 1:].min() >= 20.0 - 1e-3  # feasible case: no slack needed


def test_planner_prefers_cheap_hours(model):
    a = _arrays(tout=0.0, low=17.0)
    a["low"][:, 12:] = 21.0  # comfort need starts at hour 12
    cheap_then_dear = np.where(np.arange(H) < 12, 0.10, 0.50)
    plan = Planner(model, H).plan(_x0(17.5), **{**a, "price": cheap_then_dear})
    flat = Planner(model, H).plan(_x0(17.5), **{**a, "price": np.full(H, 0.3)})
    early = plan.q[:, :12].sum()
    assert early > flat.q[:, :12].sum() + 1.0  # pre-heats while cheap
    assert plan.ti[:, 12].min() >= 21.0 - 1e-3


def test_planner_soft_comfort_when_infeasible(model):
    # 5 kW per zone cannot hold 26 degC at -20 degC: slack absorbs it, no solver failure
    plan = Planner(model, H).plan(_x0(15.0), **_arrays(tout=-20.0, low=26.0, high=28.0))
    assert plan.status == "optimal"
    assert plan.q[:, 0].sum() >= hp.Q_HP_MAX - 1e-3


@pytest.mark.parametrize(
    "field,value",
    [
        ("tout", np.zeros(5)),
        ("solar", np.full(H, np.nan)),
        ("gains", np.zeros((2, H))),
        ("price", np.full(H, -0.1)),
    ],
)
def test_planner_validates_inputs(model, field, value):
    arrays = _arrays()
    arrays[field] = value
    with pytest.raises(ValueError):
        Planner(model, H).plan(_x0(), **arrays)


def test_planner_rejects_inverted_band(model):
    with pytest.raises(ValueError, match="exceeds"):
        Planner(model, H).plan(_x0(), **_arrays(low=25.0, high=20.0))


def test_planner_rejects_tiny_horizon(model):
    with pytest.raises(ValueError):
        Planner(model, 1)


def test_planner_model_prediction_matches_simulation(model):
    p = Planner(model, H)
    a = _arrays(tout=2.0, low=20.0)
    plan = p.plan(_x0(21.0), **a)
    u = np.column_stack([plan.q.T, a["tout"], a["solar"], a["gains"].T])
    x = model.simulate(_x0(21.0), u)
    assert np.allclose(x[:, 0::2].T, plan.ti, atol=1e-3)


# ------------------------------------------------------------------ estimator

def test_estimator_converges_from_wrong_envelope_guess(model):
    rng = np.random.default_rng(2)
    est = Estimator(model)
    x_true = _x0(21.0, te=14.0)
    est.init(x_true[0::2])
    for _ in range(72):
        y = x_true[0::2] + rng.normal(0, 0.1, 3)
        est.update(y)
        q = np.array([1.0, 1.0, 1.0])
        u = np.concatenate([q, [0.0, 0.0], [0.1, 0.1, 0.1]])
        x_true = est.ad @ x_true + est.bd @ u
        est.predict(q, 0.0, 0.0, np.full(3, 0.1))
    assert np.abs(est.x - x_true).max() < 1.0


def test_estimator_gain_shape_and_stability(model):
    est = Estimator(model)
    assert est.gain.shape == (6, 3)
    assert np.abs(np.linalg.eigvals(est.ad - est.gain @ est.c @ est.ad)).max() < 1.0
