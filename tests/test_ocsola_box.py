"""oCSOLA with a pointwise-box prior (plan 062): on L2 of an interval equal to slab BGP's exact one-slab route; on a
sphere inside the slab-BGP bracket of the cell route (plan 061) and never wider than tau = 0 (the prior alone) or
tau = 1 (cSOLA); on a direct sum of an interval and a sphere valid and no wider than either; vectorised equal to the
per-property loop; one certified support per end on the sphere; gamma equal to SOLA's X; refusals. Every test also
checks validity against a model in the box that fits the data exactly (so it lies in every slab). Synthetic only."""
import math

import numpy as np
import pytest
from scipy.stats import norm

import pygeoinf0 as pg
from pygeoinf0.core.errors import Refused, Unsupported
from pygeoinf0.inference.methods import decomposition_of

I = pg.L2(pg.Interval(0, 1))


def gaussians(M, centres, width=0.02):
    return [M(lambda x, c=c: np.exp(-(x - c) ** 2 / width), resolution=0.02) for c in centres]


def slab_of(X, noise, D, alpha=0.05):
    w = norm.ppf(1 - alpha / 2) * np.sqrt(np.diag(np.atleast_2d(X(noise).covariance.matrix())))
    return pg.inference.slab_family(D, X.matrix().T, sigma=w, mass=1.0)


def phi(box, t, a, s, w, tau, sign=1.0):
    """The certified dual at tau (the upper end of sigma_B's bracket)."""
    return tau * s + abs(tau) * w + box.support(t * sign - a * tau)


def interval_problem():
    D = pg.Euclidean(6)
    G = pg.Functionals(gaussians(I, np.linspace(0.1, 0.9, 6)), into=D)
    T = pg.Functionals([I(lambda x: ((x > 0.4) & (x < 0.6)) * 5.0, breaks=[0, 0.4, 0.6, 1]),
                        I(lambda x: np.sin(3 * x), resolution=0.05)], into=pg.Euclidean(2))
    noise = pg.GaussianMeasure(D, 1e-4 * pg.Identity(D))
    mt = I(lambda x: 0.5 * np.sin(6 * x), resolution=0.05)
    return G, T, noise, mt, D


@pytest.mark.parametrize("varying", [False, True])
def test_interval_equals_the_exact_one_slab_route(varying):
    G, T, noise, mt, D = interval_problem()
    L = pg.Additive(G, noise)
    d = G(mt)
    env = I(lambda x: 1.0 + 0.5 * x, breaks=[0, 1]) if varying else 1.0
    box = pg.PointwiseBox(I, envelope=env)
    X = decomposition_of(L).sola_operator(T, 1e-2)
    r = pg.inference.ocsola(box, L, T, d, X=X)
    slabs = slab_of(X, noise, D)
    for j in range(2):
        K = pg.FeasibleSet(box, G, slabs[j], d)
        t = T.adjoint(pg.Euclidean(2).family(np.eye(2)))[j]
        up, lo = K.bound(t), K.bound(t * -1.0)
        assert "EXACT" in (up.status, lo.status)                 # (a witness may fail membership by rounding: OUTER)
        assert r[1][j] == pytest.approx(up.upper, rel=1e-12, abs=1e-12)
        assert r[0][j] == pytest.approx(-lo.upper, rel=1e-12, abs=1e-12)
        assert r[0][j] <= T(mt).values[j] <= r[1][j]
    assert np.all(r.lower_bracket[:, 0] == r[0]) and np.all(r.upper_bracket[:, 1] == r[1])


R_CORE = 3479.95776
XS = pg.DirectSum({"cmb": pg.L2(pg.Sphere(1.0))}, weights={"cmb": 1 / (4 * math.pi * R_CORE ** 2)})


def sphere_problem(nd=12, P=3, deg=14, seed=0):
    rng = np.random.default_rng(seed)
    lat = lambda n: np.degrees(np.arcsin(rng.uniform(-1, 1, n)))     # noqa: E731
    G = pg.CapAverages(XS, 1500.0 / R_CORE, np.c_[lat(nd), rng.uniform(0, 360, nd)], summand="cmb", degree=deg)
    T = pg.CapAverages(XS, 1000.0 / R_CORE, np.c_[lat(P), rng.uniform(0, 360, P)], summand="cmb", degree=deg)
    D = G.codomain
    noise = pg.GaussianMeasure(D, 0.01 * pg.Identity(D))
    S2 = XS.summands[0]
    mt = XS({"cmb": S2(lambda x, y, z: 3.0 * x * y + 2.0 * z, resolution=1.0)})      # |m| <= 5 on the sphere
    return G, T, noise, mt, D


def test_sphere_inside_the_slab_bgp_bracket_and_never_wider_than_tau_0_or_1():
    G, T, noise, mt, D = sphere_problem()
    L = pg.Additive(G, noise)
    d = G(mt)
    box = pg.PointwiseBox(XS, envelope={"cmb": 5.0}, rtol=1e-4)
    X = decomposition_of(L).sola_operator(T, 0.05)
    r = pg.inference.ocsola(box, L, T, d, X=X)
    assert np.all(r.supports == 1)                                   # one certified support per end
    P = T.codomain
    tf, af = T.adjoint(P.family(np.eye(3))), G.adjoint(X.adjoint(P.family(np.eye(3))))
    s = X(d).values
    w = norm.ppf(0.975) * np.sqrt(np.diag(X(noise).covariance.matrix()))
    slabs = slab_of(X, noise, D)
    truth = T(mt).values
    for j in range(3):
        assert r[0][j] <= truth[j] <= r[1][j]
        for sign, end in ((1.0, r[1][j]), (-1.0, -r[0][j])):
            assert end <= phi(box, tf[j], af[j], s[j], w[j], 0.0, sign) + 1e-12       # the prior alone
            assert end <= phi(box, tf[j], af[j], s[j], w[j], 1.0, sign) + 1e-12       # cSOLA
        K = pg.FeasibleSet(box, G, slabs[j], d)
        up, lo = K.bound(tf[j]), K.bound(tf[j] * -1.0)
        # inside BGP's bracket, up to the certified supports' width (rtol 1e-4 of sigma_B in both) and tau's
        # quadrature error (measured: at most 7e-5 relative here)
        assert up.lower <= r[1][j] <= up.upper + 2e-4 * abs(up.upper)
        assert lo.lower <= -r[0][j] <= lo.upper + 2e-4 * abs(lo.upper)
    refined = pg.inference.ocsola(box, L, T, d, X=X, refine=4)
    assert np.all(refined.supports <= 5)
    assert np.all(refined[1] <= r[1] + 1e-15) and np.all(refined[0] >= r[0] - 1e-15)


def test_vectorised_equals_the_per_property_loop():
    G, T, noise, mt, D = sphere_problem(P=4, deg=10, seed=3)
    L = pg.Additive(G, noise)
    d = G(mt)
    box = pg.PointwiseBox(XS, envelope={"cmb": 5.0}, rtol=1e-4)
    X = decomposition_of(L).sola_operator(T, 0.05)
    r = pg.inference.ocsola(box, L, T, d, X=X)
    Xm = X.matrix()
    for j in range(4):
        Tj = pg.CapAverages(XS, 1000.0 / R_CORE, np.atleast_2d(T.centres[j]), summand="cmb", degree=10)
        Xj = pg.Matrix(Xm[j:j + 1], D, pg.Euclidean(1))
        one = pg.inference.ocsola(box, L, Tj, d, X=Xj)
        assert np.ndim(one[0]) == 0
        assert one[0] == pytest.approx(r[0][j], rel=1e-9) and one[1] == pytest.approx(r[1][j], rel=1e-9)   # (rounding
        # of q moves the certified bracket's cell decisions: 3e-11 seen)
        assert one.tau_upper[0] == pytest.approx(r.tau_upper[j], rel=1e-10, abs=1e-14)
        assert one.tau_lower[0] == pytest.approx(r.tau_lower[j], rel=1e-10, abs=1e-14)


def test_direct_sum_of_an_interval_and_a_sphere():
    S2 = pg.L2(pg.Sphere(1.0))
    M = pg.DirectSum({"line": I, "cap": S2}, weights={"line": 1.0, "cap": 1 / (4 * math.pi)})

    def el(f, g):
        return M({"line": I(f, resolution=0.05), "cap": S2(g, resolution=1.0)})
    D = pg.Euclidean(5)
    G = pg.Functionals(M.family([el(lambda x, c=c: np.exp(-(x - c) ** 2 / 0.05), lambda x, y, z, c=c: 1 + c * z)
                                 for c in np.linspace(0.1, 0.9, 5)]), into=D)
    T = pg.Functionals(M.family([el(lambda x: np.cos(2 * x), lambda x, y, z: x + y),
                                 el(lambda x: x, lambda x, y, z: 0.5 * z * z)]), into=pg.Euclidean(2))
    noise = pg.GaussianMeasure(D, 1e-4 * pg.Identity(D))
    L = pg.Additive(G, noise)
    mt = el(lambda x: 0.4 * np.sin(5 * x), lambda x, y, z: 0.5 * x * z)
    d = G(mt)
    box = pg.PointwiseBox(M, envelope={"line": 1.0, "cap": 1.0}, rtol=1e-5)
    X = decomposition_of(L).sola_operator(T, 1e-3)
    r = pg.inference.ocsola(box, L, T, d, X=X)
    P = T.codomain
    tf, af = T.adjoint(P.family(np.eye(2))), G.adjoint(X.adjoint(P.family(np.eye(2))))
    s = X(d).values
    w = norm.ppf(0.975) * np.sqrt(np.diag(X(noise).covariance.matrix()))
    truth = T(mt).values
    for j in range(2):
        assert r[0][j] <= truth[j] <= r[1][j]
        for sign, end in ((1.0, r[1][j]), (-1.0, -r[0][j])):
            assert end <= phi(box, tf[j], af[j], s[j], w[j], 0.0, sign) + 1e-12
            assert end <= phi(box, tf[j], af[j], s[j], w[j], 1.0, sign) + 1e-12
    assert np.all(r.supports == 1)                                   # a sphere summand: no default refinement
    assert r[1][0] - r[0][0] < 0.9 * (2 * phi(box, tf[0], af[0], s[0], w[0], 0.0))   # the data help


def test_gamma_is_solas_x_and_refusals():
    G, T, noise, mt, D = interval_problem()
    L = pg.Additive(G, noise)
    d = G(mt)
    box = pg.PointwiseBox(I, envelope=1.0)
    X = decomposition_of(L).sola_operator(T, 1e-2)
    a, b = pg.inference.ocsola(box, L, T, d, X=X), pg.inference.ocsola(box, L, T, d, gamma=1e-2)
    assert np.allclose(a[0], b[0], rtol=1e-10) and np.allclose(a[1], b[1], rtol=1e-10)
    assert a.claims.find("valid ends").status == "theorem"
    with pytest.raises(Refused):
        pg.inference.ocsola(box, L, T, d)
    with pytest.raises(Refused):
        pg.inference.ocsola(box, L, T, d, gamma=[1e-2, 1e-1])
    with pytest.raises(Unsupported):
        pg.inference.ocsola(pg.PointwiseBox(I, lower=0.0), L, T, d, X=X)


def test_the_search_minimiser_and_window_against_brute_force():
    """The weighted-median minimiser is the minimum of the piecewise-linear approximation, and the refinement window
    holds every breakpoint below the level (a random rule; the rule's internals only)."""
    from types import SimpleNamespace
    from pygeoinf0.inference.ocsola_box import approx_value, minimise, window
    rng = np.random.default_rng(5)
    n = 300
    rule = SimpleNamespace(om=rng.uniform(0.1, 1, n), lo=-rng.uniform(0.5, 1, n), hi=rng.uniform(0.5, 1, n))
    tv, av = rng.standard_normal((n, 3)), rng.standard_normal((n, 3))
    s, w = rng.standard_normal(3), rng.uniform(0, 1, 3)
    tau, val, unb = minimise(tv, av, s, w, rule)
    grid = np.linspace(-30, 30, 120001)
    for j in range(3):
        brute = approx_value(np.repeat(tv[:, j:j + 1], grid.size, 1), np.repeat(av[:, j:j + 1], grid.size, 1), grid,
                             np.full(grid.size, s[j]), np.full(grid.size, w[j]), rule)
        assert not unb[j] and val[j] <= brute.min() + 1e-12
        lev = val[j] + 0.5
        a, b = window(tv[:, j], av[:, j], s[j], w[j], rule, tau[j], lev)
        below = grid[brute <= lev]
        assert a <= below.min() and below.max() <= b
