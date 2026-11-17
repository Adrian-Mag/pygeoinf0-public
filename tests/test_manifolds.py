"""Circle, sphere and flat torus: measures, closed forms, product kernels, grid evaluation, least squares."""
import math

import numpy as np
import pytest
from scipy import special

import pygeoinf0 as pg


def test_measures_and_closed_forms():
    C = pg.L2(pg.Circle(2.0))
    assert C(1.0).integral() == pytest.approx(4 * math.pi, rel=1e-14)                 # length 2 pi R
    f = C(lambda x, y: np.exp(x / 2))                                                  # e^{cos phi}
    assert f.integral() == pytest.approx(2 * 2 * math.pi * special.iv(0, 1), rel=1e-13)
    S = pg.L2(pg.Sphere(2.0))
    assert S(1.0).integral() == pytest.approx(16 * math.pi, rel=1e-14)                # area 4 pi R^2
    z = S.coordinate(2)
    assert z @ z == pytest.approx(4 * math.pi * 2 ** 4 / 3, rel=1e-13)                 # int z^2 dA = 4 pi R^4 / 3
    T = pg.L2(pg.Torus(f"[0, {2 * math.pi!r}]", "[0, 1]"))
    h = T(lambda x, y: np.sin(x) ** 2 * np.cos(2 * np.pi * y) ** 2)
    assert h.integral() == pytest.approx(math.pi / 2, rel=1e-13)
    assert h(0.3 + 2 * math.pi, 1.7) == pytest.approx(h(0.3, 0.7), abs=1e-14)          # periodic evaluation


def test_points_off_the_manifold_are_nan():
    S = pg.L2(pg.Sphere(1.0))
    f = S(lambda x, y, z: x + z)
    assert np.isnan(f(0.5, 0.0, 0.0))
    assert f(0.6, 0.0, 0.8) == pytest.approx(1.4, abs=1e-13)
    assert pg.Sphere(1.0).contains(np.array([0.0, 0.0, 1.0])) and not pg.Sphere(1.0).contains(np.array([0.0, 0.0, 0.5]))


def test_sphere_caps_gram_closed_form():
    S = pg.L2(pg.Sphere(1.0))
    rng = np.random.default_rng(0)
    C = rng.standard_normal((12, 3))
    C /= np.linalg.norm(C, axis=1)[:, None]
    kap = 8.0
    K = S.family(lambda x, y, z, a, b, c: np.exp(kap * (x * a + y * b + z * c - 1)), a=C[:, 0], b=C[:, 1], c=C[:, 2])
    s = kap * np.linalg.norm(C[:, None] + C[None], axis=2)
    exact = 4 * math.pi * np.sinh(s) / s * math.exp(-2 * kap)
    assert np.allclose(pg.Functionals(K).gram(), exact, rtol=1e-12, atol=1e-14)


def test_circle_von_mises_gram_and_grid_values():
    C = pg.L2(pg.Circle(1.0))
    c = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    kap = 20.0
    K = C.product_family([lambda phi, c: np.exp(kap * (np.cos(phi - c) - 1))], c=c)
    D = np.abs(np.cos(0.5 * np.subtract.outer(c, c)))
    exact = 2 * np.pi * special.ive(0, 2 * kap * D) * np.exp(2 * kap * D - 2 * kap)
    assert np.allclose(pg.Functionals(K).gram(), exact, rtol=1e-12, atol=1e-15)
    m = pg.Functionals(K).adjoint(pg.Euclidean(16)(np.arange(16.0)))
    phi = np.linspace(0, 2 * np.pi, 50)
    assert np.allclose(m.on_grid(phi), m(np.cos(phi), np.sin(phi)), atol=1e-12)


def test_on_grid_matches_points_for_product_expansions():
    T = pg.L2(pg.Torus(f"[0, {2 * math.pi!r}]", f"[0, {2 * math.pi!r}]"))
    c = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    cx, cy = [a.ravel() for a in np.meshgrid(c, c, indexing="ij")]
    K = T.product_family([lambda x, cx: np.exp(4 * (np.cos(x - cx) - 1)), lambda y, cy: np.exp(4 * (np.cos(y - cy) - 1))],
                         cx=cx, cy=cy)
    m = pg.Functionals(K).adjoint(pg.Euclidean(25)(np.random.default_rng(1).standard_normal(25)))
    x, y = np.linspace(0, 6, 7), np.linspace(0, 6, 9)
    X, Y = np.meshgrid(x, y, indexing="ij")
    assert np.allclose(m.on_grid(x, y), m(X, Y), atol=1e-13)


def test_damped_least_squares_on_the_sphere_against_the_closed_form():
    import _referee as problems
    P = problems.build("sphere", 60)
    S = pg.L2(pg.Sphere(1.0))
    Cc, kap = P.params["centres"], P.params["kappa"]
    K = S.family(lambda x, y, z, a, b, c: np.exp(kap * (x * a + y * b + z * c - 1)), a=Cc[:, 0], b=Cc[:, 1], c=Cc[:, 2])
    D = pg.Euclidean(P.n, covariance=P.sigma ** 2)
    G = pg.Functionals(K, into=D)
    m = G.adjoint((G @ G.adjoint + P.alpha * pg.Identity(D)).inv(D(P.d)))
    assert P.score(m.on_grid(*P.axes))["err"] < 1e-11
