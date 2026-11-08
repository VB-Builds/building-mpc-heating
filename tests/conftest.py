import numpy as np
import pytest

from bmpc.building import GreyBox
from bmpc.simulate import make_env
from bmpc.weather import synthetic_weather


@pytest.fixture(scope="session")
def model() -> GreyBox:
    return GreyBox(
        Ci=np.array([1.5, 1.2, 1.3]),
        Ce=np.array([8.0, 7.0, 7.5]),
        Gie=np.array([0.40, 0.33, 0.37]),
        Geo=np.array([0.14, 0.12, 0.13]),
        Gio=np.array([0.02, 0.02, 0.03]),
        As=np.array([2.0, 1.0, 1.4]),
        Gz=np.array([0.05, 0.05, 0.03]),
    )


@pytest.fixture(scope="session")
def wx():
    return synthetic_weather(24 * 12, seed=1)


@pytest.fixture(scope="session")
def env(wx):
    return make_env(wx, seed=1)
