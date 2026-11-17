"""Properties of the polynomial and Fourier families that the contract cannot see (plan 021 P1, work package D):
special values, orthonormality (Gram matrices against an independent quadrature), eigenvalue equations, the r = 0
limits, explicit index layouts, and refusals."""
import itertools
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.functions import geometry as geo
from pygeoinf0.spaces import harmonic as harmonics

I01 = pg.Interval(-1.0, 3.0)                      # not [-1, 1] and not [0, 1]: the affine map is exercised
RECT = pg.Box((0.0, 2.0), (-1.0, 1.0))
BOX3 = pg.Box((0.0, 1.0), (0.0, 2.0), (-1.0, 0.5))
TORUS = pg.Torus((0.0, 2.0), (-1.0, 3.0))


def gram(fs, D, n=40):
    X, w = geo.quadrature(D, n)
    V = np.array([f(*X.T) for f in fs])
    return (V * w) @ V.T


def build(name, D, **params):
    return F.entries()[name].builder(D, **params)


# --------------------------------------------------------------------------------------------- monomial
def test_monomial_values_origin_and_layout():
    f = build("monomial", RECT, exponents=(2, 1))
    assert f(1.5, 0.5) == pytest.approx(1.5 ** 2 * 0.5)                  # exponents follow the coordinate order (x, y)
    g = build("monomial", RECT, exponents=(2, 1), origin=(1.0, 0.0))
    assert g(1.5, 0.5) == pytest.approx(0.25 * 0.5)
    assert build("monomial", I01, exponents=0)(0.3) == 1.0               # 0^0 = 1 and a bare integer on an interval
    assert build("monomial", I01, exponents=(0,))(0.0) == 1.0
    assert build("monomial", I01, exponents=3).params["exponents"] == (3,)
    # elementary integrals
    assert build("monomial", I01, exponents=2).integral()[0] == pytest.approx((27 + 1) / 3)
    assert build("monomial", RECT, exponents=(1, 2)).closed["l2_norm_sq"][0] == pytest.approx(
        (2.0 ** 3 / 3) * (2.0 / 5))


# --------------------------------------------------------------------------------------------- legendre
def test_legendre_special_values_and_mapping():
    for n in range(7):
        f = build("legendre", I01, index=n)
        assert f(3.0) == pytest.approx(1.0) and f(-1.0) == pytest.approx((-1.0) ** n)      # P_n(1) = 1, P_n(-1) = (-1)^n
        assert f.closed["max"][0] == 1.0
    f = build("legendre", I01, index=2)                                   # P_2(t) = (3 t^2 - 1)/2, t = (x - 1)/2
    assert f(2.0) == pytest.approx(-0.125) and f(1.0) == pytest.approx(-0.5)
    g = build("legendre", RECT, index=(1, 2))                             # first index = x: P_1(x - 1) P_2(y)
    assert g(1.5, 0.5) == pytest.approx(0.5 * (3 * 0.25 - 1) / 2)


@pytest.mark.parametrize("D, indices", [
    (I01, [(i,) for i in range(5)]),
    (RECT, list(itertools.product(range(3), repeat=2))),
    (BOX3, list(itertools.product(range(2), repeat=3)))])
def test_legendre_orthonormal_and_orthogonal(D, indices):
    fs = [build("legendre", D, index=i, norm="orthonormal") for i in indices]
    assert np.allclose(gram(fs, D), np.eye(len(fs)), atol=1e-12)
    for f in fs:
        assert f.closed["l2_norm_sq"][0] == pytest.approx(1.0)
    standard = [build("legendre", D, index=i) for i in indices]
    G = gram(standard, D)
    for a, f in enumerate(standard):
        assert G[a, a] == pytest.approx(f.closed["l2_norm_sq"][0], rel=1e-12)
    assert np.allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-12)


def test_legendre_integrals_and_gradient_layout():
    assert build("legendre", I01, index=0).integral()[0] == pytest.approx(4.0)
    assert build("legendre", I01, index=3).integral()[0] == 0.0
    assert build("legendre", RECT, index=(0, 0)).integral()[0] == pytest.approx(4.0)
    assert build("legendre", RECT, index=(0, 1)).integral()[0] == 0.0
    f = build("legendre", RECT, index=(1, 2))
    g = f.gradient(1.5, 0.5)
    assert g.shape == (2,) and g[0] == pytest.approx(1.0 * (3 * 0.25 - 1) / 2)       # d/dx P_1(x - 1) = 1
    assert g[1] == pytest.approx(0.5 * 3 * 0.5)                                       # P_1... = (x - 1) * 3 y


def test_legendre_three_term_recurrence():
    x = np.linspace(-1.0, 3.0, 11)
    t = (x - 1.0) / 2.0
    P = [build("legendre", I01, index=n)(x) for n in range(6)]
    for n in range(1, 5):
        assert np.allclose((n + 1) * P[n + 1], (2 * n + 1) * t * P[n] - n * P[n - 1], atol=1e-12)


# --------------------------------------------------------------------------------------------- chebyshev
def test_chebyshev_cosine_identity_and_constants():
    unit = pg.Interval(-1.0, 1.0)
    u = np.linspace(0.0, math.pi, 13)
    for n in range(8):
        f = build("chebyshev", unit, index=n)
        assert np.allclose(f(np.cos(u)), np.cos(n * u), atol=1e-12)       # T_n(cos u) = cos(n u)
        assert f(1.0) == pytest.approx(1.0)
    # the weighted orthogonality constants (Gauss-Chebyshev, weight (1 - t^2)^(-1/2)): pi for n = 0, pi/2 otherwise
    N = 64
    nodes = np.cos((2 * np.arange(N) + 1) * math.pi / (2 * N))
    fs = [build("chebyshev", unit, index=n) for n in range(6)]
    G = np.array([[math.pi / N * np.sum(f(nodes) * g(nodes)) for g in fs] for f in fs])
    assert np.allclose(G, np.diag([math.pi] + [math.pi / 2] * 5), atol=1e-12)


def test_chebyshev_lebesgue_constants_on_mapped_domains():
    for n in range(5):
        f = build("chebyshev", I01, index=n)
        X, w = geo.quadrature(I01, 200)
        assert f.closed["l2_norm_sq"][0] == pytest.approx(w @ f(*X.T) ** 2, rel=1e-12)
    assert build("chebyshev", I01, index=2).integral()[0] == pytest.approx(-2.0 / 3 * 2.0)    # (b - a)/2 = 2
    assert build("chebyshev", I01, index=3).integral()[0] == 0.0
    f = build("chebyshev", RECT, index=(2, 1))
    assert f(1.5, 1.0) == pytest.approx(math.cos(2 * math.acos(0.5)) * 1.0)    # T_2(0.5) T_1(1)


# --------------------------------------------------------------------------------------------- zernike
ZERNIKE_INDICES = [(n, m) for n in range(5) for m in range(-n, n + 1, 2)]


@pytest.mark.parametrize("R, centre", [(1.0, (0.0, 0.0)), (2.5, (0.3, -0.4))])
def test_zernike_orthonormal(R, centre):
    D = pg.Disk(R, centre)
    fs = [build("zernike", D, n=n, m=m, norm="orthonormal") for n, m in ZERNIKE_INDICES]
    assert np.allclose(gram(fs, D, 60), np.eye(len(fs)), atol=1e-11)
    for n, m in ZERNIKE_INDICES:
        f = build("zernike", D, n=n, m=m)
        assert f.closed["l2_norm_sq"][0] == pytest.approx(gram([f], D, 60)[0, 0], rel=1e-11)
        assert f.closed["integral"][0] == pytest.approx(math.pi * R * R if n == 0 else 0.0, abs=1e-11)


def test_zernike_closed_polynomials_and_centre_limits():
    D = pg.Disk(2.0, (1.0, -1.0))
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-1, 1, (2, 20)) + np.array([[1.0], [-1.0]])
    u, v = (x - 1.0) / 2.0, (y + 1.0) / 2.0                               # Cartesian coordinates of the unit disk
    r2 = u * u + v * v
    known = {(0, 0): np.ones_like(u), (1, 1): u, (1, -1): v, (2, 0): 2 * r2 - 1, (2, 2): u * u - v * v, (2, -2): 2 * u * v,
             (3, 1): (3 * r2 - 2) * u, (3, -1): (3 * r2 - 2) * v, (3, 3): u ** 3 - 3 * u * v * v,
             (3, -3): 3 * u * u * v - v ** 3, (4, 0): 6 * r2 ** 2 - 6 * r2 + 1}
    for (n, m), expected in known.items():
        assert np.allclose(build("zernike", D, n=n, m=m)(x, y), expected, atol=1e-13), (n, m)
    # r = 0: exact values, no angle is taken (R_n^m(0) = (-1)^(n/2) for m = 0, else 0)
    for n, m in ZERNIKE_INDICES:
        want = (-1.0) ** (n // 2) if m == 0 else 0.0
        assert build("zernike", D, n=n, m=m)(1.0, -1.0) == pytest.approx(want, abs=1e-14)
    # R_n^m(1) = 1 on the boundary: the cosine mode at phi = 0, the sine mode at phi = pi/(2|m|)
    for n, m in ZERNIKE_INDICES:
        phi = 0.0 if m >= 0 else math.pi / (2 * abs(m))
        assert build("zernike", D, n=n, m=m)(1.0 + 2.0 * math.cos(phi), -1.0 + 2.0 * math.sin(phi)) == pytest.approx(1.0)


def test_zernike_radial_orthogonality():
    # R_n^m and R_k^m (same m) are orthogonal for the weight rho on [0, 1], with norm 1/(2 (n + 1))
    D = pg.Disk(1.0)
    rho = np.polynomial.legendre.leggauss(40)
    r, w = 0.5 * (rho[0] + 1), 0.5 * rho[1]
    for m in (0, 1, 2):
        ns = [n for n in range(m, 9) if (n - m) % 2 == 0]
        R = np.array([build("zernike", D, n=n, m=m)(r, 0.0) for n in ns])
        G = (R * (w * r)) @ R.T
        assert np.allclose(G, np.diag(1.0 / (2 * (np.array(ns) + 1))), atol=1e-13)


def test_zernike_refusals():
    D = pg.Disk(1.0)
    for bad in ({"n": 2, "m": 1}, {"n": 2, "m": 4}, {"n": -1, "m": 0}, {"n": 2.0, "m": 0}, {"n": 100, "m": 0},
                {"n": 2, "m": 0, "norm": "unit"}):
        with pytest.raises(pg.Refused):
            build("zernike", D, **bad)
    with pytest.raises(pg.Refused, match="not defined on a ball"):
        build("zernike", pg.Ball(1.0), n=0, m=0)


# --------------------------------------------------------------------------------------------- spherical harmonics
def test_spherical_harmonics_match_the_package_and_are_orthonormal():
    S = pg.Sphere(1.0)
    L = 3
    labels = [(l, j) for l in range(L + 1) for j in range(2 * l + 1)]
    fs = [build("spherical_harmonic", S, degree=l, j=j) for l, j in labels]
    assert np.allclose(gram(fs, S, 24), np.eye(len(fs)), atol=1e-12)
    rng = np.random.default_rng(5)
    P = rng.normal(size=(9, 3))
    P /= np.linalg.norm(P, axis=1)[:, None]
    V = harmonics.values(L, P)
    for (l, j), f in zip(labels, fs):
        assert np.allclose(f(*P.T), V[:, harmonics.index(l, j)], atol=1e-14)       # the package's packed layout


def test_spherical_harmonic_special_values_and_conventions():
    S = pg.Sphere(2.0, (1.0, 2.0, 3.0))
    c = np.array(S.centre)
    north = c + np.array([0.0, 0.0, 2.0])
    f00 = build("spherical_harmonic", S, degree=0, j=0)
    assert f00(*north) == pytest.approx(1 / math.sqrt(4 * math.pi))
    assert f00.closed["integral"][0] == pytest.approx(4 * math.pi * 4.0 / math.sqrt(4 * math.pi))      # area 16 pi
    assert build("spherical_harmonic", S, degree=1, j=0)(*north) == pytest.approx(math.sqrt(3 / (4 * math.pi)))
    # the radius scales the measure only: the norm is R^2, the values are those of the unit sphere
    g = build("spherical_harmonic", S, degree=2, j=3)
    X, w = geo.quadrature(S, 24)
    assert g.closed["l2_norm_sq"][0] == pytest.approx(4.0) and w @ g(*X.T) ** 2 == pytest.approx(4.0, rel=1e-12)
    # packed j: j = 2m - 1 is cos(m phi), j = 2m is sin(m phi): at phi = 0 the sine member vanishes
    pt = c + 2.0 * np.array([math.sin(1.0), 0.0, math.cos(1.0)])
    assert build("spherical_harmonic", S, degree=2, j=4)(*pt) == pytest.approx(0.0, abs=1e-14)
    assert build("spherical_harmonic", S, degree=2, j=3)(*pt) != 0.0
    # the Condon-Shortley convention differs from the package's by (-1)^m
    for j in range(5):
        m = (j + 1) // 2
        a = build("spherical_harmonic", S, degree=2, j=j)(*pt)
        b = build("spherical_harmonic", S, degree=2, j=j, convention="condon-shortley")(*pt)
        assert b == pytest.approx((-1.0) ** m * a, abs=1e-14)
    assert build("spherical_harmonic", S, degree=2, j=3, convention="condon-shortley").closed["l2_norm_sq"][0] == \
        pytest.approx(4.0)
    sc = harmonics.HarmonicConvention("package", scale=harmonics.SQRT2_NONZONAL)
    assert build("spherical_harmonic", pg.Sphere(1.0), degree=2, j=3, convention=sc).closed["l2_norm_sq"][0] == \
        pytest.approx(0.5)


@pytest.mark.parametrize("l, j", [(0, 0), (1, 1), (2, 3), (3, 6), (4, 5)])
def test_spherical_harmonic_is_a_laplace_beltrami_eigenfunction(l, j):
    # Y(x/|x|) is homogeneous of degree 0, so its R^3 Laplacian at |x| = 1 is the Laplace-Beltrami operator: -l(l+1) Y
    f = build("spherical_harmonic", pg.Sphere(1.0), degree=l, j=j)
    p = np.array([0.3, -0.5, 0.6])
    p = p / np.linalg.norm(p)
    h, lap = 3e-4, 0.0
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        lap += (f(*(p + e)) - 2 * f(*p) + f(*(p - e))) / h ** 2
    assert lap == pytest.approx(-l * (l + 1) * f(*p), abs=1e-5)


def test_spherical_harmonic_addition_theorem_and_poles():
    S = pg.Sphere(1.0)
    rng = np.random.default_rng(1)
    P = rng.normal(size=(7, 3))
    P /= np.linalg.norm(P, axis=1)[:, None]
    for l in (1, 2, 5):
        total = sum(build("spherical_harmonic", S, degree=l, j=j)(*P.T) ** 2 for j in range(2 * l + 1))
        assert np.allclose(total, (2 * l + 1) / (4 * math.pi), atol=1e-12)
    for pole in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)):                    # the poles are regular points
        vals = [build("spherical_harmonic", S, degree=3, j=j)(*pole) for j in range(7)]
        assert np.all(np.isfinite(vals)) and np.allclose(vals[1:], 0.0, atol=1e-14)


def test_spherical_harmonic_refusals():
    S = pg.Sphere(1.0)
    for bad in ({"degree": 2, "j": 5}, {"degree": -1, "j": 0}, {"degree": 2, "j": -1}, {"degree": 1.5, "j": 0},
                {"degree": 2, "j": 1, "convention": "thesis2"}, {"degree": 99, "j": 0}):
        with pytest.raises(pg.Refused):
            build("spherical_harmonic", S, **bad)


# --------------------------------------------------------------------------------------------- fourier
def test_fourier_orthonormal_on_interval_box_torus():
    for D, d in ((I01, 1), (RECT, 2), (TORUS, 2), (BOX3, 3)):
        ks = range(-1, 2) if d < 3 else (-1, 0, 1)
        idx = list(itertools.product(ks, repeat=d))[: (27 if d == 3 else 9)]
        fs = [build("fourier", D, index=i, norm="orthonormal") for i in idx]
        assert np.allclose(gram(fs, D, 24 if d < 3 else 16), np.eye(len(fs)), atol=1e-11), D
        for i in idx:
            f = build("fourier", D, index=i)
            assert f.closed["integral"][0] == pytest.approx(math.prod(
                (s.b - s.a) for s in ([D] if d == 1 else D.sides)) if all(k == 0 for k in i) else 0.0, abs=1e-11)


def test_fourier_layout_and_boundary():
    D = pg.Interval(1.0, 3.0)                                              # L = 2, a = 1
    assert build("fourier", D, index=0)(1.7) == 1.0
    assert build("fourier", D, index=1)(1.5) == pytest.approx(math.cos(2 * math.pi * 0.25))       # k > 0: cos(2 pi k t)
    assert build("fourier", D, index=-1)(1.5) == pytest.approx(math.sin(2 * math.pi * 0.25))      # k < 0: sin(2 pi |k| t)
    assert build("fourier", D, index=-2)(1.25) == pytest.approx(math.sin(4 * math.pi * 0.125))
    f = build("fourier", RECT, index=(-1, 2))
    assert f(0.5, 0.5) == pytest.approx(math.sin(2 * math.pi * 0.25) * math.cos(4 * math.pi * 0.75))
    # a sine mode in every axis vanishes on the whole boundary (claimed order 1); others make no claim
    s = build("fourier", RECT, index=(-1, -3))
    assert s.regularity.vanishes_at_boundary == 1
    for x, y in ((0.0, 0.3), (2.0, -0.7), (0.9, -1.0), (1.3, 1.0)):
        assert s(x, y) == pytest.approx(0.0, abs=1e-12)
    assert build("fourier", RECT, index=(-1, 1)).regularity.vanishes_at_boundary is None
    assert build("fourier", RECT, index=(-1, 1)).element(pg.L2(RECT)) is not None


def test_fourier_periodicity_on_torus_and_circle():
    f = build("fourier", TORUS, index=(2, -1))
    assert f(0.3, 0.2) == pytest.approx(f(0.3 + 2.0, 0.2)) and f(0.3, 0.2) == pytest.approx(f(0.3, 0.2 + 4.0))
    C = pg.Circle(2.0, (1.0, -1.0))
    phi = np.linspace(-3.0, 3.0, 9)
    pts = (1.0 + 2.0 * np.cos(phi), -1.0 + 2.0 * np.sin(phi))
    assert np.allclose(build("fourier", C, index=3)(*pts), np.cos(3 * phi), atol=1e-13)
    assert np.allclose(build("fourier", C, index=-2)(*pts), np.sin(2 * phi), atol=1e-13)
    assert np.allclose(build("fourier", C, index=0)(*pts), 1.0)
    for k in (0, 1, -1, 4):
        f = build("fourier", C, index=k)
        assert f.closed["l2_norm_sq"][0] == pytest.approx(4 * math.pi if k == 0 else 2 * math.pi)   # arc length 4 pi
    fs = [build("fourier", C, index=k, norm="orthonormal") for k in range(-3, 4)]
    assert np.allclose(gram(fs, C, 200), np.eye(7), atol=1e-11)


def test_fourier_refusals():
    for D, bad in ((RECT, (1,)), (RECT, (1, 2, 3)), (RECT, (1.5, 0)), (RECT, (True, 0)), (RECT, (99, 0)),
                   (pg.Circle(1.0), (1, 2)), (TORUS, "ab")):
        with pytest.raises(pg.Refused):
            build("fourier", D, index=bad)
    with pytest.raises(pg.Refused, match="norm"):
        build("fourier", RECT, index=(0, 0), norm="l2")
    with pytest.raises(pg.Refused, match="not defined on a sphere"):
        build("fourier", pg.Sphere(1.0), index=0)


def test_polynomial_families_refuse_bad_indices():
    for name, key in (("legendre", "index"), ("chebyshev", "index"), ("monomial", "exponents")):
        for bad in (-1, (0, 0), 1.5, 61, None, "1"):
            with pytest.raises(pg.Refused):
                build(name, I01, **{key: bad})
        with pytest.raises(pg.Refused, match="not defined on a disk"):
            build(name, pg.Disk(1.0), **{key: 0})
    with pytest.raises(pg.Refused, match="norm"):
        build("legendre", I01, index=1, norm="unit")
    with pytest.raises(pg.Refused, match="origin"):
        build("monomial", RECT, exponents=(1, 1), origin=(0.0,))


def test_catalogue_lists_the_families_with_their_kinds():
    cat = {n: kinds for n, kinds, _ in F.catalogue()}
    assert cat["legendre"] == ("interval", "box") and cat["zernike"] == ("disk",)
    assert cat["spherical_harmonic"] == ("sphere",)
    assert set(cat["fourier"]) == {"interval", "box", "circle", "torus"}
    assert F.on(pg.Disk(1.0)).zernike(n=2, m=0).family == "zernike"
