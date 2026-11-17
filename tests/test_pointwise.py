"""Pointwise boxes on function spaces and certified box BGP (plan 009 P5, G8): exact zonal L1 values (the thesis's
Pacific caps), certified brackets against independent quadratures, exact interval supports, the box BGP on L2(0, 1)
with piecewise-constant kernels against an independent LP (convex/lp.py) and an SLSQP primal, witnesses re-verified
by independent code, operator ellipsoids against norm balls, and an amplitude bound on the ladder's interval problem."""
import math

import numpy as np
import pytest
from numpy.polynomial import legendre as npleg
from scipy.optimize import minimize

import pygeoinf0 as pg
from pygeoinf0.convex.feasible import FeasibleSet
from pygeoinf0.optimisers.lp import lp_max

from pygeoinf0.realisations.bounds import sphere_l1 as S
from pygeoinf0.spaces.harmonic import funk_hecke, values

L14 = 14
R_CMB = 3479.95776


def cap_series(radius_km, L=L14):
    """Legendre series of the band-projected (even degrees 2..14) raised-cosine cap weight w_B (EXP-030 DESIGN 1.4)."""
    lam = funk_hecke("raised-cosine", radius_km / R_CMB, L)[0]
    c = np.zeros(L + 1)
    for s in range(2, L + 1, 2):
        c[s] = (2 * s + 1) / (4 * math.pi) * lam[s]
    return c, lam


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def rotated(lam, xi, L=L14):
    """Packed coefficients of w_B(xi . eta) (Funk-Hecke / addition theorem)."""
    Y = values(L, xi[None])[0]
    C = np.zeros((L + 1) ** 2)
    for s in range(2, L + 1, 2):
        C[s * s:(s + 1) ** 2] = lam[s] * Y[s * s:(s + 1) ** 2]
    return C


@pytest.mark.parametrize("radius, exact, notes_bracket", [(1000, 33.1869, (33.186922, 33.188401)),
                                                          (2000, 26.0193, (26.019277, 26.019348)),
                                                          (5000, 13.4524, (13.452371, 13.452385))])
def test_exact_zonal_box_half_widths_and_certified_brackets(radius, exact, notes_bracket):
    c, lam = cap_series(radius)
    hw = 15.0 * S.zonal_l1(c)                                   # A_CMB = 15 km, a = 1 (EXP-030 R7)
    assert hw == pytest.approx(exact, abs=5e-5)
    assert notes_bracket[0] <= hw <= notes_bracket[1]           # phase R's brackets (NOTES 8) contain the exact value
    lo, hi, info = S.l1_bracket(rotated(lam, unit(-4.5, -157.5))[:, None], L14, rtol=1e-7)
    assert 15 * lo[0] <= hw <= 15 * hi[0] and hi[0] - lo[0] <= 2e-7 * lo[0]


def test_zonal_sign_moments_against_quadrature():
    c, _ = cap_series(2000)
    m = S.zonal_sign_moments(c, 14)
    x = np.linspace(-1, 1, 2_000_001)
    for s in (0, 3, 8):
        e = np.zeros(s + 1)
        e[s] = 1
        assert m[s] == pytest.approx(np.trapezoid(np.sign(npleg.legval(x, c)) * npleg.legval(x, e), x), abs=2e-6)


def test_brackets_contain_a_fine_quadrature_of_random_band_limited_functions():
    L = 6
    C = np.random.default_rng(3).standard_normal(((L + 1) ** 2, 2))
    lo, hi, _ = S.l1_bracket(C, L, rtol=1e-6)

    def quad(nz):
        z, w = npleg.leggauss(nz)
        lon = np.arange(2 * nz) * math.pi / nz
        Z, LO = np.meshgrid(z, lon, indexing="ij")
        X = np.stack([np.sqrt(1 - Z ** 2) * np.cos(LO), np.sqrt(1 - Z ** 2) * np.sin(LO), Z], -1).reshape(-1, 3)
        V = np.abs(values(L, X) @ C).reshape(nz, 2 * nz, -1)
        return (V * w[:, None, None]).sum((0, 1)) * math.pi / nz
    r1, r2 = quad(400), quad(800)
    tol = np.abs(r2 - r1) + 1e-12
    assert np.all(lo - tol <= r2) and np.all(r2 <= hi + tol) and np.all(hi - lo <= 1e-6 * lo + 1e-12)


def test_rect_abs_linear_against_quadrature():
    x = np.linspace(-0.5, 0.5, 2001)
    y = np.linspace(-0.4, 0.4, 2001)
    X, Y = np.meshgrid(x, y)
    for c, a, b in ((0.3, 1.2, -0.7), (0.0, 0.0, 1.0), (0.2, 0.0, 0.0), (-0.1, 0.5, 0.0)):
        ref = np.trapezoid(np.trapezoid(np.abs(c + a * X + b * Y), y, axis=0), x)
        assert S.rect_abs_linear(np.array([c]), np.array([a]), np.array([b]), 0.5, 0.4)[0] == pytest.approx(ref, abs=1e-6)


def test_radial_box_support_on_a_ball_is_exact_for_separable_fields():
    """f(r, xi) = rho(r) w_B(xi . eta): int A |f| dV = A int |rho| r^2 dr * int_S |w_B| (zonal, exact)."""
    from pygeoinf0.spaces.elements import Function
    from pygeoinf0.realisations.series.radial_harmonic import RadialHarmonicSeries
    M = pg.L2(pg.Ball(1.0), realisation="radial_harmonics")
    c, lam = cap_series(2000, L=6)
    C = rotated(lam, unit(30.0, 40.0), L=6)
    rho = [np.array([0.2, 1.0, 0.3]), np.array([1.0, -0.5])]          # Legendre coefficients on [0, .4], [.4, 1]
    breaks = [0.0, 0.4, 1.0]
    coefs = [np.einsum("d,j->dj", r, C)[:, :, None] for r in rho]
    f = Function(M, pl=RadialHarmonicSeries(6, breaks, coefs, 0.0))
    exact_r = 0.0
    for (a, b), r in zip(zip(breaks[:-1], breaks[1:]), rho):
        t = np.linspace(-1, 1, 400001)
        rr = 0.5 * (a + b) + 0.5 * (b - a) * t
        exact_r += 0.5 * (b - a) * np.trapezoid(np.abs(npleg.legval(t, r)) * rr ** 2, t)
    exact = 2.0 * exact_r * S.zonal_l1(c)
    b = pg.PointwiseBox(M, envelope=2.0).support(f, rtol=1e-3)
    lo, hi = b.lower, b.upper
    assert lo <= exact * (1 + 1e-9) and exact <= hi * (1 + 1e-9) and hi - lo <= 2e-3 * hi


def test_piecewise_affine_radial_bracket_contains_the_exact_value():
    """f(r, xi) = rho(r) w_B(xi . eta) with rho affine on two pieces (a sign change inside the second), weight 1:
    int |f| = int |rho| dr * int_S |w_B| exactly (zonal)."""
    c, lam = cap_series(1000, L=6)
    C = rotated(lam, unit(-10.0, 100.0), L=6)
    ends = [(0.2, 1.0), (1.0, -0.5)]
    widths = np.array([0.4, 0.6])
    exact_r = 0.4 * 0.6 + 0.6 * (1.0 ** 2 + 0.5 ** 2) / (2 * 1.5)
    lo, hi, per = S.affine_radial_l1_bracket(np.array([a * C for a, _ in ends]), np.array([b * C for _, b in ends]),
                                             widths, np.zeros(2), 6, rtol=1e-7)
    exact = exact_r * S.zonal_l1(c)
    assert lo <= exact <= hi and per.shape == (2, 2)
    assert per[0, 1] - per[0, 0] <= 1e-6 * per[0, 1]           # a sign-definite piece: midpoint = trapezoid


def test_interval_support_is_exact_with_function_bounds():
    M = pg.L2(pg.Interval(0, 1))
    q = M(lambda x: np.sin(7 * x) - 0.3 + x ** 2)
    u = M(lambda x: 1 + x)
    B = pg.PointwiseBox(M, lower=-0.5, upper=u)
    x = np.linspace(0, 1, 2_000_001)
    qv = np.sin(7 * x) - 0.3 + x ** 2
    ref = np.trapezoid(np.where(qv > 0, (1 + x) * qv, -0.5 * qv), x)
    b = B.support(q)
    lo, hi = b.lower, b.upper
    assert lo == hi and lo == pytest.approx(ref, abs=1e-9)
    assert B.member(M(lambda x: 0.9 + 0.0 * x)) is pg.Truth.TRUE
    assert B.member(M(lambda x: 1.5 - x)) is pg.Truth.FALSE
    assert pg.PointwiseBox(M, lower=0.0).support(M(lambda x: -1 - x)) == pytest.approx(0.0)
    assert pg.PointwiseBox(M, lower=0.0).support(q) == math.inf


def piecewise_problem(noise="box", seed=0, ncell=10, nd=6):
    """L2(0, 1), kernels constant on 10 cells: the box BGP is the LP (or SOCP) in the cell values."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, 1, ncell + 1)
    K = rng.uniform(-1, 1, (nd, ncell))
    M = pg.L2(pg.Interval(0, 1))
    D = pg.Euclidean(nd)

    def pc(vals):
        return M(lambda x, v=vals: v[np.clip(np.searchsorted(edges, x, side="right") - 1, 0, ncell - 1)],
                 breaks=edges)
    G = pg.Functionals(M.family([pc(k) for k in K]), into=D)
    truth = rng.uniform(-0.8, 0.8, ncell)
    h = 1.0 / ncell
    d = K @ truth * h + rng.normal(0, 0.01, nd)
    w = np.full(nd, 0.03)
    N = pg.CoordinateBox(D, -w, w) if noise == "box" else pg.NormBall(D, 0.05)
    c = pc(rng.uniform(-1, 1, ncell))
    return M, D, G, K, h, d, N, w, c, edges


def test_box_bgp_on_piecewise_constant_kernels_equals_the_lp():
    M, D, G, K, h, d, N, w, c, edges = piecewise_problem("box")
    box = pg.PointwiseBox(M, envelope=1.0)
    Kset = FeasibleSet(box, G, N, D(d))
    b = Kset.bound(c)
    t = np.array([c.integral(over=pg.Interval(a, e)) for a, e in zip(edges[:-1], edges[1:])])
    H = K.T * h                                                  # (cells, data)
    n = len(t)
    A = np.r_[np.eye(n), -np.eye(n), H.T, -H.T]
    rhs = np.r_[np.ones(n), np.ones(n), d + w, -(d - w)]
    ref, _ = lp_max(t, A, rhs)
    assert b.status == "EXACT" and b.lower <= ref + 1e-9 and ref <= b.upper + 1e-9 and b.upper - b.lower < 1e-8
    # the witness re-verified independently: cell values in the box, data residual inside the noise box
    v = b.witness                                                # values on the 40 witness cells (4 per kernel cell)
    H4, t4 = np.repeat(H, 4, axis=0) / 4, np.repeat(t, 4) / 4      # c and the kernels are constant on each cell
    assert np.all(np.abs(v) <= 1 + 1e-12) and np.all(np.abs(d - H4.T @ v) <= w * (1 + 1e-9))
    assert t4 @ v == pytest.approx(b.lower, abs=1e-12)


def test_box_bgp_with_ball_noise_against_slsqp_and_the_dual():
    M, D, G, K, h, d, N, w, c, edges = piecewise_problem("ball", seed=4)
    box = pg.PointwiseBox(M, envelope=1.0)
    K_ = pg.update(box, pg.Additive(G, N), D(d))
    b = K_.bound(c)
    t = np.array([c.integral(over=pg.Interval(a, e)) for a, e in zip(edges[:-1], edges[1:])])
    H = K.T * h
    cons = [dict(type="ineq", fun=lambda v: 0.05 ** 2 - np.sum((d - H.T @ v) ** 2))]
    res = minimize(lambda v: -t @ v, np.zeros(len(t)), jac=lambda v: -t, bounds=[(-1, 1)] * len(t),
                   constraints=cons, method="SLSQP", options=dict(ftol=1e-14, maxiter=500))
    assert b.lower - 1e-7 <= -res.fun <= b.upper + 1e-7 and b.status in ("EXACT", "BRACKET")
    assert b.upper - b.lower <= 1e-6 * max(1.0, abs(b.upper))
    v = b.witness
    H4 = np.repeat(H, 4, axis=0) / 4
    assert np.all(np.abs(v) <= 1 + 1e-12) and np.linalg.norm(d - H4.T @ v) <= 0.05 * (1 + 1e-9)


def test_an_inconsistent_box_gives_a_certified_empty_set():
    M, D, G, K, h, d, N, w, c, edges = piecewise_problem("ball", seed=2)
    box = pg.PointwiseBox(M, envelope=1e-3)                      # far too small to fit the data
    Kset = FeasibleSet(box, G, N, D(d * 50))
    assert Kset.is_empty() is pg.Truth.TRUE and Kset.emptiness.status == "EMPTY"


def test_operator_ellipsoid_against_a_scaled_norm_ball():
    M = pg.L2(pg.Interval(0, 1))
    A = 4.0 * pg.Identity(M)
    E = pg.Ellipsoid(M, 1.0, shape=A, inverse=0.25 * pg.Identity(M))
    q = M(lambda x: np.cos(3 * x))
    assert E.support(q) == pytest.approx(pg.NormBall(M, 0.5).support(q), rel=1e-13)
    assert E.member(M(lambda x: 0.4 + 0 * x)) is pg.Truth.TRUE and E.member(M(lambda x: 0.6 + 0 * x)) is pg.Truth.FALSE
    D = pg.Euclidean(5, covariance=0.01 ** 2)
    G = pg.Functionals(M.riesz([pg.Interval(a, a + 0.15) for a in np.linspace(0, 0.85, 5)]), into=D)
    d = G(M(lambda x: 0.3 * np.sin(4 * x)))
    T = pg.Functionals(M.riesz([pg.Interval(0.4, 0.6)]))
    for S_, ball in ((E, pg.NormBall(M, 0.5)),):
        lo_e, hi_e = pg.push(pg.update(S_, pg.Additive(G, pg.NormBall(D, 3.0)), d), T).bounds(0)
        lo_b, hi_b = pg.push(pg.update(ball, pg.Additive(G, pg.NormBall(D, 3.0)), d), T).bounds(0)
        assert hi_e.value == pytest.approx(hi_b.value, rel=1e-10) and lo_e.value == pytest.approx(lo_b.value, rel=1e-10)
        assert hi_e.status == "EXACT"
    assert pg.Ellipsoid(pg.Euclidean(2), 1.0, covariance=[1.0, 4.0]).support(np.array([0.0, 1.0])) == pytest.approx(2.0)


def test_an_amplitude_bound_on_the_ladders_interval_problem():
    """Outside use: the ladder's L1 data (averages over random subintervals) with the bound |m| <= 1.5 on L2(0, 1),
    against the norm ball of the same box: the box interval is certified, contains the truth, and lies inside the
    ball's interval (B is a subset of the ball of radius 1.5)."""
    import ladder
    P = ladder.build("L1", 30)
    M = pg.L2(pg.Interval(0, 1))
    D = pg.Euclidean(30)
    G = pg.Functionals(M.riesz(P.measures), into=D)
    r = math.sqrt(30 + 2 * math.sqrt(60)) * P.sigma
    noise = pg.NormBall(D, r)
    d = D(P.d)
    T = pg.Functionals(M.riesz([pg.Interval(0.4, 0.6)]) * 5.0)
    xs = np.linspace(0.4, 0.6, 20001)
    true = np.trapezoid(P.truth(xs), xs) * 5.0
    assert np.max(np.abs(P.truth(np.linspace(0, 1, 10001)))) < 1.5
    Kb = pg.update(pg.PointwiseBox(M, envelope=1.5), pg.Additive(G, noise), d)
    lo, hi = pg.push(Kb, T).bounds(0)
    lo_n, hi_n = pg.push(pg.update(pg.NormBall(M, 1.5), pg.Additive(G, noise), d), T).bounds(0)
    for b in (lo, hi):
        assert b.status in ("EXACT", "BRACKET") and b.upper - b.lower <= 1e-3 * (hi.upper - lo.lower)
    assert lo.lower <= true <= hi.upper
    assert lo_n.value - 1e-9 <= lo.lower and hi.upper <= hi_n.value + 1e-9
