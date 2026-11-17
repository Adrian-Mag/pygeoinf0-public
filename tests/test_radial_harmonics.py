"""Plan 009 P2, G1: the radial x harmonic realisation of L2 on balls, shells and layered balls, band projections, and
the harmonic conventions (EXP-030 DESIGN G1). References: the tensor realisation (Legendre in r, theta; Fourier in phi),
scipy quadrature, explicit formulas for the harmonics and scipy's sph_harm_y."""
import math

import numpy as np
import pytest
from scipy.integrate import quad, tplquad
from scipy.special import sph_harm_y

import pygeoinf0 as pg
from pygeoinf0.spaces import harmonic as H

F = lambda x, y, z: np.exp(0.5 * x) * np.cos(y + 0.3 * z)        # noqa: E731
G = lambda x, y, z: x * y + z ** 2 + 1.0                          # noqa: E731


def test_default_realisation_on_balls_is_unchanged():
    assert pg.L2(pg.Ball(1.0)).realisation == "tensor"
    assert pg.L2(pg.Shell(0.5, 1.0)).realisation == "tensor"
    assert pg.L2(pg.Ball(1.0), realisation="radial_harmonics").realisation == "radial_harmonics"
    with pytest.raises(ValueError):
        pg.L2(pg.Disk(1.0), realisation="radial_harmonics")


@pytest.mark.parametrize("dom", [pg.Ball(1.3), pg.Shell(0.5, 1.2), pg.Ball(1.0, centre=(0.2, -0.1, 0.3))])
def test_inner_products_against_the_tensor_realisation(dom):
    M, T = pg.L2(dom, realisation="radial_harmonics"), pg.L2(dom)
    a, b, c, d = M(F), M(G), T(F), T(G)
    assert M.inner(a, b) == pytest.approx(T.inner(c, d), rel=1e-13, abs=1e-13)
    assert a.norm() == pytest.approx(c.norm(), rel=1e-13)
    assert a.integral() == pytest.approx(c.integral(), rel=1e-13)
    assert a.error < 1e-12
    p = np.array(dom.to_cartesian(np.array([0.6, 0.9]), np.array([0.4, 2.0]), np.array([1.0, 5.0]))).T
    assert np.allclose(a(p), F(*p.T), rtol=0, atol=1e-13)


def test_inner_product_against_scipy_quadrature_on_a_shell():
    S = pg.Shell(0.5, 1.0)
    M = pg.L2(S, realisation="radial_harmonics")
    f = lambda x, y, z: np.sin(x + 2 * y) * z + x * x               # noqa: E731

    def integrand(phi, theta, r):
        x, y, z = S.to_cartesian(r, theta, phi)
        return f(x, y, z) * G(x, y, z) * r * r * np.sin(theta)
    ref = tplquad(integrand, 0.5, 1.0, 0.0, math.pi, 0.0, 2 * math.pi, epsabs=1e-13, epsrel=1e-13)[0]
    assert M.inner(M(f), M(G)) == pytest.approx(ref, rel=1e-11)


def test_layered_ball_and_jumps_at_interfaces():
    P = pg.LayeredBall([0.4, 1.0], names=("core", "mantle"))
    V = pg.PiecewiseSpace.lebesgue(P, realisation="radial_harmonics")
    W = pg.PiecewiseSpace.lebesgue(P)
    x = V({"core": F, "mantle": G})
    y = W({"core": F, "mantle": G})
    assert x.norm() == pytest.approx(y.norm(), rel=1e-13)
    assert all(V.summand(n).realisation == "radial_harmonics" for n in V.names)


def test_radial_multiplication_is_exact():
    M = pg.L2(pg.Ball(1.0), realisation="radial_harmonics")
    a = M(lambda x, y, z: 1.0 + x * x + y * y + z * z)              # radial: 1 + r^2
    f = M(F)
    g = f * a
    p = np.array([[0.2, 0.3, -0.4], [0.6, 0.1, 0.2]])
    assert np.allclose(g(p), F(*p.T) * (1 + (p ** 2).sum(1)), rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------------------------- band projections
@pytest.mark.parametrize("space", [pg.L2(pg.Ball(1.0), realisation="radial_harmonics"), pg.L2(pg.Sphere(2.0)),
                                   pg.L2(pg.Shell(0.3, 1.0), realisation="radial_harmonics")])
def test_band_projection_is_an_orthogonal_projection(space):
    P = pg.BandProjection(space, [0, 2, 3, 5])
    f, g = space(lambda x, y, z: np.exp(x - y) * (1 + z)), space(lambda x, y, z: np.cos(x * y + z))
    Pf = P(f)
    assert (P(Pf) - Pf).norm() <= 1e-15 * f.norm()                   # idempotent (exactly: coefficients zeroed)
    assert space.inner(Pf, g) == pytest.approx(space.inner(f, P(g)), rel=1e-14, abs=1e-15)   # self-adjoint
    assert Pf.norm() <= f.norm()
    assert ((f - Pf) @ Pf) == pytest.approx(0.0, abs=1e-14 * f.norm() ** 2)
    assert P.adjoint is P and P.claims.status("rotation invariant") == (pg.Truth.TRUE, "theorem")


def test_band_projection_commutes_with_rotations():
    M = pg.L2(pg.Ball(1.0), realisation="radial_harmonics")
    Q = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    f = lambda x, y, z: np.exp(x - 0.5 * y) * (1 + z) + x * y * z     # noqa: E731
    rot = lambda x, y, z: f(*(sum(Q[k, i] * c for k, c in enumerate((x, y, z))) for i in range(3)))            # noqa: E731  f o Q^-1
    P = pg.BandProjection(M, [1, 2, 4])
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(20, 3))
    pts *= (rng.uniform(0.1, 0.9, 20) / np.linalg.norm(pts, axis=1))[:, None]
    a = P(M(rot))(pts)
    b = P(M(f))(pts @ Q)                                            # (P f)(Q^T x)
    assert np.allclose(a, b, rtol=0, atol=1e-13)


def test_band_projection_on_a_direct_sum():
    M = pg.DirectSum({"v": pg.L2(pg.Ball(1.0), realisation="radial_harmonics"), "h": pg.L2(pg.Sphere(1.0))})
    P = pg.BandProjection(M, range(2, 9, 2))
    x = M({"v": F, "h": G})
    assert (P(P(x)) - P(x)).norm() <= 1e-15 * x.norm()
    assert P(x)["h"].norm() == pytest.approx(pg.BandProjection(M.summand("h"), range(2, 9, 2))(M.summand("h")(G)).norm())
    with pytest.raises(pg.errors.Unsupported):
        pg.BandProjection(pg.L2(pg.Ball(1.0)), [1])                  # the tensor realisation has no harmonic structure


# ---------------------------------------------------------------------------------------------- conventions
def test_package_harmonics_against_explicit_formulas():
    th = np.array([0.3, 1.1, 2.5])
    ph = np.array([0.2, 2.0, 4.0])
    V = H.values(2, None, (th, ph))
    c = math.sqrt(3 / (4 * math.pi))
    assert np.allclose(V[:, 0], 1 / math.sqrt(4 * math.pi))
    assert np.allclose(V[:, 1], c * np.cos(th))
    assert np.allclose(V[:, 2], c * np.sin(th) * np.cos(ph))         # no Condon-Shortley sign
    assert np.allclose(V[:, 3], c * np.sin(th) * np.sin(ph))
    assert np.allclose(V[:, 4], math.sqrt(5 / (16 * math.pi)) * (3 * np.cos(th) ** 2 - 1))
    assert np.allclose(V[:, 7], math.sqrt(15 / (16 * math.pi)) * np.sin(th) ** 2 * np.cos(2 * ph))


def test_thesis_convention_is_scipy_with_condon_shortley_and_a_sign_per_order():
    rng = np.random.default_rng(0)
    th, ph = rng.uniform(0, np.pi, 30), rng.uniform(0, 2 * np.pi, 30)
    L = 8
    lab = H.labels(L)
    ref = np.empty((30, len(lab)))
    for k, (l, j) in enumerate(lab):
        y = sph_harm_y(l, (j + 1) // 2, th, ph)
        ref[:, k] = y.real if j == 0 else math.sqrt(2) * (y.real if j % 2 else y.imag)
    assert np.allclose(H.THESIS.values(L, None, (th, ph)), ref, rtol=0, atol=1e-13)
    sign = (-1.0) ** ((lab[:, 1] + 1) // 2)
    assert np.allclose(H.values(L, None, (th, ph)) * sign, ref, rtol=0, atol=1e-13)
    C = rng.normal(size=((L + 1) ** 2, 2))
    assert np.allclose(H.PACKAGE.convert(H.THESIS.convert(C), to=H.THESIS), C)
    cat = H.HarmonicConvention("condon-shortley", scale=H.SQRT2_NONZONAL)
    assert np.allclose(cat.values(L, None, (th, ph)) * np.where(lab[:, 1] == 0, 1.0, math.sqrt(2)), ref, atol=1e-13)


def test_funk_hecke_coefficients_against_scipy_quad():
    psi0 = 0.4
    lam, n2 = H.funk_hecke("raised-cosine", psi0, 20)
    w = lambda p: np.cos(0.5 * math.pi * p / psi0) ** 2                # noqa: E731
    mass = 2 * math.pi * quad(lambda p: w(p) * math.sin(p), 0, psi0, epsabs=1e-15, epsrel=1e-15)[0]
    for l in (0, 3, 10, 20):
        from scipy.special import eval_legendre
        ref = 2 * math.pi * quad(lambda p: w(p) * eval_legendre(l, math.cos(p)) * math.sin(p), 0, psi0, epsabs=1e-16,
                                 epsrel=1e-14, limit=200)[0] / mass
        assert lam[l] == pytest.approx(ref, rel=1e-12, abs=1e-14)
    ref2 = 2 * math.pi * quad(lambda p: (w(p) / mass) ** 2 * math.sin(p), 0, psi0, epsabs=1e-16, epsrel=1e-14)[0]
    assert n2 == pytest.approx(ref2, rel=1e-12)
