"""Spherical-harmonic realisations on the sphere: orthonormality, exactness, adaptivity, agreement with the tensor
basis, product kernels, grid evaluation and least squares."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.realisations.series import spherical as SHm


def test_basis_is_orthonormal_and_transforms_are_exact():
    L = 12
    C = np.random.default_rng(0).standard_normal(((L + 1) ** 2, 3))
    V = SHm.synthesise_on(C, L, L)
    assert np.allclose(SHm.analyse(V, L), C, atol=1e-13)                      # analysis inverts synthesis at degree L
    theta, w, phi, _ = SHm.grid(2 * L)
    V2 = SHm.synthesise_on(np.eye((L + 1) ** 2), L, 2 * L)                  # every Y_lm on a finer grid
    G = np.einsum("tpi,tpj,t->ij", V2, V2, w) * (2 * np.pi / phi.size)
    assert np.allclose(G, np.eye((L + 1) ** 2), atol=1e-13)                   # int Y_i Y_j dOmega = delta_ij
    t, p = np.array([0.3, 1.2, 2.9]), np.array([0.1, 3.0, 5.5])
    assert np.allclose(SHm.evaluate_points(C, L, t, p), SHm.synthesise_grid(C, L, t, p)[[0, 1, 2], [0, 1, 2]], atol=1e-13)


def test_low_degree_closed_forms():
    S = pg.L2(pg.Sphere(2.0))
    assert S.backend.summary(S(1.0).realisation) == "spherical harmonics: degree 0"
    z = S.coordinate(2)
    assert z.realisation.L == 1
    assert z @ z == pytest.approx(4 * math.pi * 2 ** 4 / 3, rel=1e-14)
    f = S(lambda x, y, z: x * y * z + z ** 2)
    assert f.realisation.L == 3
    assert f.integral() == pytest.approx(4 * math.pi * 2 ** 4 / 3, rel=1e-14)
    assert f(0.0, 0.0, 2.0) == pytest.approx(4.0, abs=1e-13)
    assert f(1.2, 1.6 * 0.6, 1.6 * 0.8) == pytest.approx(1.2 * 0.96 * 1.28 + 1.28 ** 2, abs=1e-13)


def test_agrees_with_the_tensor_basis_and_uses_fewer_coefficients():
    fun = lambda x, y, z: np.exp(-((x - 0.3) ** 2 + y ** 2 + (z - 0.2) ** 2) / 0.1) + np.sin(3 * x) * z  # noqa: E731
    g = lambda x, y, z: 1.0 + x * y  # noqa: E731
    out = {}
    for basis in ("harmonics", "tensor"):
        S = pg.L2(pg.Sphere(1.0), realisation=basis)
        f = S(fun)
        out[basis] = (f.norm(), f @ S(g), f.integral(), (f * f).integral(), f.realisation.size,
                      f(0.6, 0.0, 0.8))
    a, b = out["harmonics"], out["tensor"]
    assert np.allclose(a[:4], b[:4], rtol=1e-12)
    assert a[5] == pytest.approx(fun(0.6, 0.0, 0.8), abs=1e-12)
    assert a[4] < b[4] / 2


def test_product_family_and_on_grid():
    import warnings
    S = pg.L2(pg.Sphere(1.0))
    tc = np.array([1.0, 1.5, 2.0])            # theta factors negligible (1e-19) at the poles: smooth products
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        K = S.product_family([lambda theta, tc: np.exp(-((theta - tc) / 0.15) ** 2),
                              lambda phi, pc: np.exp(4 * (np.cos(phi - pc) - 1))], tc=tc, pc=np.array([0.0, 1.0, 4.0]))
    th, ph = np.linspace(0.1, 3.0, 7), np.linspace(0, 6, 5)
    T, P = np.meshgrid(th, ph, indexing="ij")
    want = np.exp(-((T - 1.5) / 0.15) ** 2) * np.exp(4 * (np.cos(P - 1.0) - 1))
    k1 = K[1]
    assert np.allclose(k1.on_grid(th, ph), want, atol=1e-12)
    assert np.allclose(k1(*pg.Sphere(1.0).from_angles(T, P)), want, atol=1e-12)


def test_damped_least_squares_against_the_referee():
    import _referee as problems
    P = problems.build("sphere", 200)
    S = pg.L2(pg.Sphere(1.0))
    Cc, kap = P.params["centres"], P.params["kappa"]
    K = S.family(lambda x, y, z, a, b, c: np.exp(-0.5 * kap * ((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2)),
                 a=Cc[:, 0], b=Cc[:, 1], c=Cc[:, 2])
    assert np.allclose(pg.Functionals(K).gram(), P.gram, rtol=1e-12, atol=1e-14 * np.abs(P.gram).max())
    D = pg.Euclidean(P.n, covariance=P.sigma ** 2)
    G = pg.Functionals(K, into=D)
    m = G.adjoint((G @ G.adjoint + P.alpha * pg.Identity(D)).inv(D(P.d)))
    assert P.score(m.on_grid(*P.axes))["err"] < 1e-12
