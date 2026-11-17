"""The sphere's certified L1 bracket with local remainders (plan 061): the upper end converges with rtol for
concentrated functions (a cap anywhere on the sphere, against the exact zonal value of the same cap at the pole), for
sign-changing functions against an independent quadrature, and its local curvature bounds hold where sampled.
Synthetic functions only."""
import math

import numpy as np
import pytest
from numpy.polynomial import legendre as npleg

import pygeoinf0 as pg
from pygeoinf0.realisations.bounds import sphere_l1 as S
from pygeoinf0.spaces.harmonic import values

R_CORE = 3479.95776


def cap_space():
    return pg.DirectSum({"cmb": pg.L2(pg.Sphere(1.0))}, weights={"cmb": 1 / (4 * math.pi * R_CORE ** 2)})


def cap_coefficients(X, centre, deg, radius_km=1000.0):
    T = pg.CapAverages(X, radius_km / R_CORE, np.array([centre]), summand="cmb", degree=deg)
    q = T.adjoint(T.codomain(np.array([1.0])))
    return q, np.asarray(q.realisation.parts[0][0].C, float)[:, 0]


def zonal_series(C, L):
    """The Legendre series of a function zonal about the north pole from its packed coefficients: Y_l0 =
    sqrt((2l + 1) / (4 pi)) P_l(cos theta)."""
    l = np.arange(L + 1)
    return C[l * l] * np.sqrt((2 * l + 1) / (4 * math.pi))


def top_hat(L, psi0):
    """lambda_l = 2 pi int_(cos psi0)^1 P_l: the zonal top-hat cap truncated at degree L (Funk-Hecke)."""
    a = math.cos(psi0)
    lam = np.zeros(L + 1)
    lam[0] = 2 * math.pi * (1 - a)
    for l in range(1, L + 1):
        e = np.zeros(l + 2)
        e[l - 1], e[l + 1] = 1.0, -1.0
        lam[l] = 2 * math.pi * npleg.legval(a, e) / (2 * l + 1)
    return lam


def rotated(lam, xi, L, degrees=None):
    """Packed coefficients of sum_l lambda_l (2l+1)/(4 pi) P_l(xi . eta) (the addition theorem)."""
    Y = values(L, xi[None])[0]
    C = np.zeros((L + 1) ** 2)
    for s in (range(L + 1) if degrees is None else degrees):
        C[s * s:(s + 1) ** 2] = lam[s] * Y[s * s:(s + 1) ** 2]
    return C


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


@pytest.mark.parametrize("deg", [24, 48])
def test_an_equatorial_cap_box_support_converges_to_the_exact_zonal_value_at_the_pole(deg):
    """The box support is rotation invariant for a constant bound, so the equatorial cap's support equals the polar
    cap's, which is exact (zonal_l1). Before plan 061 the upper end was 6.98 at degree 24 against 5.54, for every rtol."""
    X = cap_space()
    box = pg.PointwiseBox(X, envelope={"cmb": 5.0})
    _, Cp = cap_coefficients(X, [90.0, 0.0], deg)
    exact = 5.0 * S.zonal_l1(zonal_series(Cp, deg)) / (4 * math.pi * R_CORE ** 2)        # the summand weight
    q, _ = cap_coefficients(X, [0.0, 0.0], deg)
    widths = []
    for rtol in (1e-2, 1e-4):
        b = box.support(q, rtol=rtol)
        lo, hi = b.lower, b.upper
        assert lo <= exact * (1 + 1e-10) and exact <= hi * (1 + 1e-10)
        assert hi - lo <= rtol * lo
        widths.append(hi - lo)
    assert widths[1] < widths[0]
    assert exact <= pg.PointwiseBox(X, envelope={"cmb": 5.0}, rtol=1e-3).support(q) <= exact * (1 + 1e-3)


def test_a_generic_cap_bracket_shrinks_with_rtol_and_always_holds_the_exact_value():
    L = 32
    lam = top_hat(L, 0.3)
    c = (2 * np.arange(L + 1) + 1) / (4 * math.pi) * lam
    exact = S.zonal_l1(c)
    C = rotated(lam, unit(23.0, -61.0), L)
    prev = math.inf
    for rtol in (1e-2, 1e-3, 1e-4, 1e-5):
        lo, hi, info = S.l1_bracket(C[:, None], L, rtol=rtol)
        lo, hi = lo[0], hi[0]
        assert lo <= exact * (1 + 1e-12) and exact <= hi * (1 + 1e-12)
        assert hi - lo <= rtol * lo and info[0]["converged"] and info[0]["stop"] in ("rtol", "decided")
        assert hi - lo < prev
        prev = hi - lo


def quadrature(C, L, nz):
    """int_S |g| by Gauss-Legendre in z and the trapezoid rule in longitude (independent of sphere_l1)."""
    z, w = npleg.leggauss(nz)
    lon = np.arange(2 * nz) * math.pi / nz
    total = 0.0
    for i in range(0, nz, 32):
        Z, LO = np.meshgrid(z[i:i + 32], lon, indexing="ij")
        Xp = np.stack([np.sqrt(1 - Z ** 2) * np.cos(LO), np.sqrt(1 - Z ** 2) * np.sin(LO), Z], -1).reshape(-1, 3)
        V = np.abs(values(L, Xp) @ C).reshape(len(Z), -1)
        total += float((V * w[i:i + 32, None]).sum()) * math.pi / nz
    return total


def test_a_sign_changing_function_against_a_fine_quadrature():
    """A top-hat cap minus SOLA-like even-degree terms (degrees 2..14) of a wider cap elsewhere: many sign changes."""
    L = 20
    C = rotated(top_hat(L, 0.35), unit(-12.0, 140.0), L) \
        - 0.8 * rotated(top_hat(L, 0.9), unit(40.0, 100.0), L, degrees=range(2, 15, 2))
    lo, hi, info = S.l1_bracket(C[:, None], L, rtol=1e-5)
    q1, q2 = quadrature(C, L, 600), quadrature(C, L, 1200)
    tol = 2 * abs(q2 - q1) + 1e-12
    assert lo[0] - tol <= q2 <= hi[0] + tol and hi[0] - lo[0] <= 1e-5 * lo[0]
    assert abs(q2 - q1) < 1e-4 * q2                                    # the reference itself is that good


def test_local_curvature_bounds_the_sampled_second_derivatives():
    L = 12
    C = rotated(top_hat(L, 0.4), unit(10.0, 30.0), L)
    F = S.torus_coefficients(C[:, None], L)[0]
    M = 8 * (L + 1)
    Sc = S.local_curvature(F, L, M)
    kf, km = np.arange(-(L + 1), L + 2), np.arange(-L, L + 1)
    rng = np.random.default_rng(1)
    delta = 2 * math.pi / M
    for _ in range(200):
        i, j = rng.integers(M // 2), rng.integers(M)
        ph = -math.pi / 2 + (i + rng.random(16)) * delta
        la = (j + rng.random(16)) * delta
        ep, el = np.exp(1j * np.outer(ph, kf)), np.exp(1j * np.outer(la, km))
        d = [np.einsum("pk,km,pm->p", ep * (1j * kf) ** a, F, el * (1j * km) ** b).real for a, b in
             ((2, 0), (1, 1), (0, 2))]
        assert np.all(np.abs(d[0]) + 2 * np.abs(d[1]) + np.abs(d[2]) <= Sc[i, j])
    assert Sc.max() / Sc.min() > 50                                    # local: much smaller far from the cap


def test_the_de_la_vallee_poussin_formula_and_its_cell_majorant():
    n, M = 7, 24
    rng = np.random.default_rng(2)
    a = rng.standard_normal(2 * n + 1) + 1j * rng.standard_normal(2 * n + 1)
    k = np.arange(-n, n + 1)

    def q(x):
        return (np.exp(1j * np.outer(x, k)) @ a).real
    A, B = M - n, n
    kk = np.arange(-A, A + 1)
    vhat = (A * np.maximum(1 - np.abs(kk) / A, 0) - B * np.maximum(1 - np.abs(kk) / B, 0)) / (A - B)

    def V(t):
        return (np.exp(1j * np.outer(t, kk)) @ vhat).real
    xj = 0.3 + 2 * math.pi * np.arange(M) / M
    x = rng.uniform(0, 2 * math.pi, 50)
    interp = np.array([np.sum(q(xj) * V(xx - xj)) / M for xx in x])
    assert np.allclose(interp, q(x), atol=1e-12)
    K = S.vp_cell_kernel(n, M)
    delta = 2 * math.pi / M
    for d in range(M):
        t = (d + np.linspace(0, 1, 401)) * delta
        assert np.all(np.abs(V(t)) <= K[d] * (1 + 1e-12))
    with pytest.raises(ValueError):
        S.vp_cell_kernel(n, 2 * n)


def test_the_torus_values_equal_the_harmonics_on_the_whole_torus():
    L = 9
    C = np.random.default_rng(4).standard_normal(((L + 1) ** 2, 2))
    N = 2 * L + 4
    ang = 2 * math.pi * np.arange(N) / N
    PH, LA = np.meshgrid(ang, ang, indexing="ij")
    Xp = np.stack([np.cos(PH) * np.cos(LA), np.cos(PH) * np.sin(LA), np.sin(PH)], -1).reshape(-1, 3)
    ref = ((values(L, Xp) @ C) * np.cos(PH).reshape(-1, 1)).reshape(N, N, -1)
    assert np.allclose(S.torus_values(C, L, N), ref, atol=1e-13)


def test_the_box_rtol_is_validated():
    with pytest.raises(ValueError):
        pg.PointwiseBox(cap_space(), envelope={"cmb": 1.0}, rtol=0.0)
