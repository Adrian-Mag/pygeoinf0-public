"""Law tests on 2D/3D domains (tensor backend). Small sizes: run with a memory cap on the laptop."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg

DOMAINS = {
    "rectangle": (pg.Interval("[0, 1]") * pg.Interval("[0, 2]"), lambda x, y: x * y, 8 / 9),
    "disk": (pg.Disk(1.0), lambda x, y: x ** 2 + y ** 2, math.pi / 3),
    "box": (pg.Box("[0, 1]", "[0, 1]", "[0, 1]"), lambda x, y, z: x + y + z, 3 * (1 / 3) + 6 * (1 / 4)),
    "cylinder": (pg.Cylinder(1.0, z=(0, 2)), lambda x, y, z: z * np.hypot(x, y), 4 * math.pi / 3),
    "ball": (pg.Ball(1.0), lambda x, y, z: x ** 2 + y ** 2 + z ** 2, 4 * math.pi / 7),
}


@pytest.mark.parametrize("name", list(DOMAINS))
def test_norms_against_closed_forms(name):
    D, f, exact = DOMAINS[name]
    M = pg.L2(D)
    g = M(f)
    assert g @ g == pytest.approx(exact, rel=1e-13)
    assert M == pg.L2(pg.LebesgueMeasure(D))


def test_cartesian_coordinates_on_curved_domains():
    M = pg.L2(pg.Ball(1.0))
    assert M.x @ M.x == pytest.approx(4 * math.pi / 15, rel=1e-13)      # analytic in (r, theta, phi)
    C = pg.L2(pg.Disk(2.0, centre=(1, 1)))
    assert C.x.integral() == pytest.approx(4 * math.pi, rel=1e-13)        # centroid times area


def test_values_and_outside():
    M = pg.L2(pg.Ball(1.0))
    f = M(lambda x, y, z: np.sin(3 * x) * np.cos(2 * y) + z ** 2)
    P = np.random.default_rng(0).uniform(-0.5, 0.5, (2000, 3))
    exact = np.sin(3 * P[:, 0]) * np.cos(2 * P[:, 1]) + P[:, 2] ** 2
    assert np.abs(f(P) - exact).max() < 1e-11
    assert np.isnan(f(2.0, 0.0, 0.0))


def test_localised_function_integral():
    M = pg.L2(pg.Disk(2.0, centre=(1, 1)))
    g = M(lambda x, y: np.exp(-((x - 1.5) ** 2 + (y - 0.5) ** 2) / 0.1 ** 2))
    assert g.integral() == pytest.approx(math.pi * 0.01, rel=1e-12)


@pytest.mark.parametrize("D", [pg.Disk(1.0), pg.Box("[0, 1]", "[0, 1]", "[0, 1]")])
def test_adjoint_and_minimum_norm(D):
    M = pg.L2(D)
    rng = np.random.default_rng(1)
    lo = np.array([a for a, _ in D.bounding_box])
    hi = np.array([b for _, b in D.bounding_box])
    c = lo + (hi - lo) * rng.uniform(0.25, 0.75, (12, D.dim))
    if D.dim == 2:
        K = M.family(lambda x, y, cx, cy: np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.3 ** 2), cx=c[:, 0], cy=c[:, 1])
    else:
        K = M.family(lambda x, y, z, cx, cy, cz: np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 0.4 ** 2),
                     cx=c[:, 0], cy=c[:, 1], cz=c[:, 2])
    G = pg.Functionals(K)
    m = pg.probability.random_features(M).sample(seed=2)
    y = G.codomain.random(seed=3)
    assert G(m) @ y == pytest.approx(m @ G.adjoint(y), rel=1e-11)
    d = G(m)
    mdag = G.adjoint((G @ G.adjoint).inverse(pg.Eigh(1e-13))(d))
    assert (G(mdag) - d).norm() <= 1e-9 * d.norm()
    assert mdag.norm() <= m.norm() * (1 + 1e-9)
