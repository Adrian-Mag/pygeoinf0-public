"""Sobolev spaces H^s_l: definitions checked against independent references, Riesz representers of points and rays,
the adjoint identity, damped least squares, refusals."""
import math

import numpy as np
import pytest
from scipy import integrate

import pygeoinf0 as pg
from pygeoinf0.core.errors import NormUnavailable, PointEvaluationUndefined, UnboundedFunctional


def test_one_dimensional_closed_forms():
    l = 0.1
    H = pg.Sobolev("[0, 1]", 1, length_scale=l)                  # kernel exp(-|x - y|/l) / (2 l)
    x = np.array([0.1, 0.35, 0.9])
    K = H.inner(H.riesz([pg.DiracMeasure(v) for v in x]), H.riesz([pg.DiracMeasure(v) for v in x]))
    assert np.allclose(K, np.exp(-np.abs(np.subtract.outer(x, x)) / l) / (2 * l), rtol=1e-14)
    # sub-interval measures: int_a^b int_c^d exp(-|x - y|/l)/(2l), closed form for disjoint [a, b] < [c, d]
    I1, I2 = pg.Interval(0.1, 0.3), pg.Interval(0.5, 0.8)
    E = H.inner(H.riesz(I1), H.riesz(I2))
    ref = l / 2 * (math.exp(-(0.5 - 0.3) / l) - math.exp(-(0.5 - 0.1) / l)) * (1 - math.exp(-(0.8 - 0.5) / l))
    assert E == pytest.approx(ref, rel=1e-13)
    E11 = H.norm(H.riesz(I1)) ** 2                                # int int over [0, 0.2]^2 = (L - l (1 - e^-L/l))
    assert E11 == pytest.approx(0.2 - l * (1 - math.exp(-0.2 / l)), rel=1e-13)
    assert pg.Sobolev("[0, 1]", 0, length_scale=l) == pg.L2("[0, 1]")


def test_ray_gram_against_mpmath_and_reproducing_property():
    H = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.1)           # exponential kernel (nu = 1/2)
    r1, r2 = pg.Segment((0, -1), (0, 1)), pg.Segment((-1, 0), (1, 0))
    k = H.riesz([r1, r2])
    E = H.inner(k, k)
    assert E[0, 1] == pytest.approx(0.999752826247969513616, rel=2e-15)    # mpmath, 30 digits
    f = H(lambda x, y: np.cos(2 * x) * (1 + y) + x * y)
    G = pg.Functionals(k)
    ref = integrate.quad(lambda t: np.cos(0.0) * (1 + t), -1, 1, epsabs=1e-15)[0]   # along r1: x = 0, y = t
    assert G(f).values[0] == pytest.approx(ref, rel=1e-12)
    ref2 = integrate.quad(lambda t: np.cos(2 * t), -1, 1, epsabs=1e-15)[0]
    assert G(f).values[1] == pytest.approx(ref2, rel=1e-12)


def test_adjoint_identity_and_damped_least_squares():
    rng = np.random.default_rng(0)
    D = pg.Disk(1.0)
    H = pg.Sobolev(D, 1.5, length_scale=0.2)
    ang = rng.uniform(0, 2 * np.pi, (40, 2))
    rays = [pg.Segment((math.cos(a), math.sin(a)), (math.cos(b), math.sin(b))) for a, b in ang]
    sigma = 1e-3
    Dd = pg.Euclidean(len(rays), covariance=sigma ** 2)
    G = pg.Functionals(H.riesz(rays), into=Dd)
    m_true = H(lambda x, y: np.exp(-((x - 0.3) ** 2 + y ** 2) / 0.1))
    d = G(m_true) + Dd.standard_gaussian().sample(seed=1)
    y = Dd(rng.standard_normal(len(rays)))
    m1 = G.adjoint(y)
    assert G(m1) @ y == pytest.approx(m1 @ G.adjoint(y), rel=1e-12)
    m = G.adjoint((G @ G.adjoint + 1.0 * pg.Identity(Dd)).inv(d))
    assert np.isfinite(m.norm())
    assert np.isfinite(m(0.1, 0.2)) and np.isnan(m(1.5, 0.0))
    chi2 = (G(m) - d).norm() ** 2
    assert chi2 < 10 * len(rays)
    with pytest.raises(NormUnavailable):
        m_true.norm()
    L = pg.L2(D)                                                  # representers have kinks along the rays:
    mL = pg.Inclusion(H, L, tol=1e-6)(m)                          # their L2 realisation converges algebraically
    assert mL.space == L
    X = np.random.default_rng(3).uniform(-0.6, 0.6, (400, 2))
    err = np.sqrt(np.mean((mL(X[:, 0], X[:, 1]) - m(X[:, 0], X[:, 1])) ** 2) * math.pi)
    assert err < 10 * mL.error and mL.error < 1e-4 * m.norm()     # the estimate is honest (no 1e-16 claims)


def test_refusals():
    D = pg.Disk(1.0)
    with pytest.raises(UnboundedFunctional, match="s > \\(d - k\\)/2 = 0.5"):
        pg.Sobolev(D, 0.5, length_scale=0.1).riesz(pg.Segment((0, 0), (0.5, 0)))
    with pytest.raises(PointEvaluationUndefined):
        pg.Sobolev(D, 1.0, length_scale=0.1).evaluation((0.1, 0.1))
    H = pg.Sobolev(D, 1.5, length_scale=0.1)
    with pytest.raises(ValueError, match="clip"):
        H.riesz(pg.Segment((0, 0), (2, 0)))
    assert H.evaluation((0.1, 0.1))(H(lambda x, y: x + y)) == pytest.approx(0.2, rel=1e-12)
    assert len(H.claims) >= 5 and H == pg.Sobolev(D, 1.5, length_scale=0.1) != pg.Sobolev(D, 1.5, length_scale=0.2)


def test_periodic_spaces_match_the_spectral_definition():
    # torus [0, 2 pi)^2, s = 2.5: K(x, y) = (2 pi)^-2 sum_n (1 + l^2 |n|^2)^-s e^{i n (x - y)}
    l, s = 0.3, 2.5
    H = pg.Sobolev(pg.Torus(f"[0, {2 * math.pi!r}]", f"[0, {2 * math.pi!r}]"), s, length_scale=l)
    P = np.array([[0.1, 0.2], [3.0, 6.0], [6.2, 0.1]])
    k = H.riesz([pg.DiracMeasure(p) for p in P])
    K = H.inner(k, k)
    n = np.arange(-1500, 1501)                                    # spectral tail beyond: ~ 6e-9 relative
    N1, N2 = np.meshgrid(n, n, indexing="ij")
    w = (1 + l ** 2 * (N1 ** 2 + N2 ** 2)) ** -s
    ref = np.array([[np.sum(w * np.cos(N1 * (a[0] - b[0]) + N2 * (a[1] - b[1]))) for b in P] for a in P]) / (2 * math.pi) ** 2
    assert np.allclose(K, ref, rtol=2e-8, atol=1e-12)
    # circle of radius 2, s = 1: K = (2 pi R)^-1 sum_k (1 + l^2 k^2 / R^2)^-1 e^{i k (phi - psi)}
    R, l = 2.0, 0.5
    C = pg.Sobolev(pg.Circle(R), 1, length_scale=l)
    phis = np.array([0.3, 2.0, 5.9])
    pts = [pg.DiracMeasure((R * math.cos(p), R * math.sin(p))) for p in phis]
    kk = C.riesz(pts)
    K = C.inner(kk, kk)
    a = l / R                                                     # sum_k cos(k t)/(1 + a^2 k^2), t in [0, 2 pi]:
    t = np.mod(np.subtract.outer(phis, phis), 2 * math.pi)        # (pi/a) cosh((pi - t)/a) / sinh(pi/a)
    ref = (math.pi / a) * np.cosh((math.pi - t) / a) / math.sinh(math.pi / a) / (2 * math.pi * R)
    assert np.allclose(K, ref, rtol=1e-12)


@pytest.mark.parametrize("s, ref", [(0.75, 0.99995208818646192613),     # nu = -1/4 (singular kernel)
                                    (1.3, 0.99982907648600971008),      # nu = 0.3 (scipy kv path)
                                    (2.0, 0.99945744577005413395)])     # nu = 1 (integer order)
def test_general_orders_against_mpmath(s, ref):
    H = pg.Sobolev(pg.Disk(1.0), s, length_scale=0.1)
    k = H.riesz([pg.Segment((0, -1), (0, 1)), pg.Segment((-1, 0), (1, 0))])
    assert H.inner(k, k)[0, 1] == pytest.approx(ref, rel=1e-13)          # mpmath, 25 digits (polar form about the crossing)
