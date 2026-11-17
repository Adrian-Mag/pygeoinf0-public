"""Separable functionals on spaces with a density (plan 066): G h = int k h dlambda against the domain's own measure
(R^2 dOmega on a sphere, r^2 dr dOmega on a ball or shell, or against=); the space's measure mu = w lambda enters only
through the representer k dlambda/dmu. (a) On L2(w dOmega) with weight 1 they equal those on L2(dOmega) with
direct-sum weight w: data, Grams, structured Grams, representers' norms, SOLA's X, oCSOLA box ends. (b) A radial density on a
shell: G h against an independent quadrature of int k h_lj r^2 dr; against= kept; the adjoint identity. (c) G h does not
depend on the density at all for a fixed function h. Synthetic only."""
import math
import warnings

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

import pygeoinf0 as pg
from pygeoinf0.inference.methods import decomposition_of
from pygeoinf0.spaces import harmonic as HA

W_CMB = 1 / (4 * math.pi * 3479.95776 ** 2)
S2 = pg.Sphere(1.0)
LAB = [(l, j) for l in range(4) for j in range(2 * l + 1)]


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def field(H):
    return H(lambda x, y, z: 1.0 + 2.0 * x * y - z + 0.5 * z ** 3, resolution=0.5)


def sphere_numbers(X):
    rng = np.random.default_rng(1)
    kern = list(rng.standard_normal(len(LAB)))
    G = pg.Functionals.separable(X, {"cmb": kern}, LAB, into=pg.Euclidean(len(LAB)))
    H = X.summands[0]
    h = X({"cmb": field(H)})
    T = pg.CapAverages(X, 0.3, [[10.0, 20.0], [-30.0, 140.0]], summand="cmb", degree=6)
    noise = pg.GaussianMeasure(G.codomain, 1e-6 * pg.Identity(G.codomain))
    L = pg.Additive(G, noise)
    Xs = decomposition_of(L).sola_operator(T, 0.1)
    d = G(h)
    box = pg.PointwiseBox(X, envelope={"cmb": 4.0}, rtol=1e-6)
    r = pg.inference.ocsola(box, L, T, d, X=Xs)
    y = G.codomain(rng.standard_normal(len(LAB)))
    return dict(data=d.values, gram=G.gram(), structured=G.structured_gram().matrix(), X=Xs.matrix(),
                ends=np.r_[r[0], r[1]], rep_norm=X.norm(G.adjoint(y)) ** 2,
                adjoint=(G.codomain.inner(d, y), X.inner(h, G.adjoint(y))))


def test_a_sphere_density_equals_the_direct_sum_weight_for_separable_functionals():
    a = sphere_numbers(pg.DirectSum({"cmb": pg.L2(S2)}, weights={"cmb": W_CMB}))
    b = sphere_numbers(pg.DirectSum({"cmb": pg.L2(pg.LebesgueMeasure(S2).with_density(W_CMB))}, weights={"cmb": 1.0}))
    for k in a:
        assert rel(b[k], a[k]) <= 1e-11, k
    assert a["adjoint"][0] == pytest.approx(a["adjoint"][1], rel=1e-12)
    # the PLI finding: the data are the kernels against R^2 dOmega, 1e-3-sized, not w times smaller
    c = pg.DirectSum({"cmb": pg.L2(S2)}, weights={"cmb": 1.0})
    G1 = pg.Functionals.separable(c, {"cmb": [1.0]}, [(0, 0)], into=pg.Euclidean(1))
    Gd = pg.Functionals.separable(pg.DirectSum({"cmb": pg.L2(pg.LebesgueMeasure(S2).with_density(W_CMB))}),
                                  {"cmb": [1.0]}, [(0, 0)], into=pg.Euclidean(1))
    h1, hd = c({"cmb": field(c.summands[0])}), Gd.domain({"cmb": field(Gd.domain.summands[0])})
    assert Gd(hd).values[0] == pytest.approx(G1(h1).values[0], rel=1e-13)


def test_rotated_families_and_cap_averages_on_a_weighted_sphere():
    """A rotated family of a functional given by its representer t0 (an element) is <t0 rotated, h> in the space's
    measure on L2(c dOmega) too; cap averages do not depend on c."""
    cen = [[10.0, 30.0], [-50.0, 200.0]]
    caps = []
    for X in (pg.L2(S2), pg.L2(pg.LebesgueMeasure(S2).with_density(0.02))):
        h = field(X)
        t0 = X(lambda x, y, z: np.exp(-(1.0 - z) / 0.2), resolution=0.3)
        R = pg.RotatedFamily(pg.Functional.from_representer(t0), cen)
        for i, (lat, lon) in enumerate(cen):
            la, lo = math.radians(lat), math.radians(lon)
            n = (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))
            t = X(lambda x, y, z, n=n: np.exp(-(1.0 - n[0] * x - n[1] * y - n[2] * z) / 0.2), resolution=0.3)
            assert R(h).values[i] == pytest.approx(X.inner(t, h), rel=1e-9)
        caps.append(pg.CapAverages(X, 0.4, cen, degree=8)(h).values)
    assert rel(caps[1], caps[0]) <= 1e-12


SHELL = pg.Shell(0.5, 1.0)


def rho(r):
    return 1.0 + 2.0 * r ** 2 + np.sin(3.0 * r)


def spaces():
    return [pg.L2(SHELL, realisation="radial_harmonics"),
            pg.L2(pg.LebesgueMeasure(SHELL).with_density(0.3), realisation="radial_harmonics"),
            pg.L2(pg.LebesgueMeasure(SHELL).with_density(rho, radial=True), realisation="radial_harmonics")]


def coefficients_by_quadrature(f, r, lab, n=64):
    """h_lj(r) = int h(r xi) Y_lj(xi) dxi on a Gauss x equispaced grid (independent of the package's fits)."""
    t, wt = leggauss(n)
    th = np.arccos(t)
    ph = np.linspace(0, 2 * math.pi, 2 * n + 1)[:-1]
    T, P = np.meshgrid(th, ph, indexing="ij")
    Wa = (wt[:, None] * np.full(ph.size, 2 * math.pi / ph.size)[None, :]).ravel()
    Y = HA.values(3, None, theta_phi=(T.ravel(), P.ravel()))[:, [HA.index(l, j) for l, j in lab]]
    st, ct, cp, sp = np.sin(T).ravel(), np.cos(T).ravel(), np.cos(P).ravel(), np.sin(P).ravel()
    out = np.empty((len(r), len(lab)))
    for i, ri in enumerate(r):
        out[i] = (Wa * f(ri * st * cp, ri * st * sp, ri * ct)) @ Y
    return out


def test_radial_density_functionals_against_independent_quadrature():
    """G h = int k_i h_(l_i j_i) r^2 dr whatever the density; against= nu integrates against nu; G G* and the adjoint
    identity in the space's measure."""
    f = (lambda x, y, z: 1.0 + x * y + z ** 2 - 0.3 * z)
    lab = [(0, 0), (1, 0), (2, 1), (2, 0)]
    kern = [lambda r: r, lambda r: np.cos(r), lambda r: r * r, lambda r: np.exp(-r)]
    tr, wr = leggauss(48)
    r, wr = 0.75 + 0.25 * tr, 0.25 * wr
    H = coefficients_by_quadrature(f, r, lab)
    ref = np.array([float(np.sum(wr * r * r * kern[i](r) * H[:, i])) for i in range(len(lab))])
    nu = pg.LebesgueMeasure(pg.Interval(0.5, 1.0)).with_density(lambda x: 1.0 + x)
    ref_nu = np.array([float(np.sum(wr * (1 + r) * kern[i](r) * H[:, i])) for i in range(len(lab))])
    for M in spaces():
        G = pg.Functionals.separable(M, kern, lab, into=pg.Euclidean(len(lab)))
        h = M(f, resolution=0.2)
        assert np.max(np.abs(G(h).values - ref)) <= 1e-10 * np.max(np.abs(ref)), M
        Gn = pg.Functionals.separable(M, kern, lab, against=nu, into=pg.Euclidean(len(lab)))
        assert np.max(np.abs(Gn(h).values - ref_nu)) <= 1e-9 * np.max(np.abs(ref_nu)), M
        assert rel(G.structured_gram().matrix(), G.gram()) <= 1e-12
        y = G.codomain(np.array([0.3, -1.0, 0.7, 2.0]))
        assert G.codomain.inner(G(h), y) == pytest.approx(M.inner(h, G.adjoint(y)), rel=1e-12)
    # G G* = int k_i k_i / (c rho) r^2 dr on the diagonal blocks (the representer k dlambda/dmu)
    G = pg.Functionals.separable(spaces()[2], kern, lab, into=pg.Euclidean(len(lab)))
    assert G.gram()[0, 0] == pytest.approx(float(np.sum(wr * r * r * r * r / rho(r))), rel=1e-10)


def test_g_h_does_not_depend_on_the_density():
    """(c) the same function h, the same kernels: the same data on L2(dx), L2(0.3 dx), L2(rho dx) and in a direct sum
    with any weight."""
    f = (lambda x, y, z: np.exp(z) * (1 + x))
    vals = []
    for M in spaces():
        G = pg.Functionals.separable(M, [lambda r: r, lambda r: 1.0 / r], [(0, 0), (1, 1)], into=pg.Euclidean(2))
        vals.append(G(M(f, resolution=0.2)).values)
        X = pg.DirectSum({"m": M}, weights={"m": 7.0})
        GX = pg.Functionals.separable(X, {"m": [lambda r: r, lambda r: 1.0 / r]}, [(0, 0), (1, 1)], into=pg.Euclidean(2))
        vals.append(GX(X({"m": M(f, resolution=0.2)})).values)
    for v in vals[1:]:
        assert np.max(np.abs(v - vals[0])) <= 1e-10 * np.max(np.abs(vals[0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sph = [pg.L2(S2), pg.L2(pg.LebesgueMeasure(S2).with_density(3.0))]
        d = [pg.Functionals.separable(M, [0.5, -2.0], [(1, 0), (2, 2)], into=pg.Euclidean(2))(field(M)).values
             for M in sph]
    assert rel(d[1], d[0]) <= 1e-13
