"""oCSOLA and the reduction to the span of the representers do not depend on the representers' scale (plan 037).

The defect: optimisers/dual/reduction.reduce dropped eigen-directions of the joint Gram matrix below 1e-13 of its largest
eigenvalue, so a representer much shorter than the others (oCSOLA's r~ = G* X* 1 for a large gamma, or a datum in
small units) lost its component orthogonal to the others; oCSOLA then returned a falsely tight interval that need not
contain the truth. The closed-form route also formed <t, r~> by a cancelling difference. The references here are the
exact two-dimensional problem (QR of the actual vectors t and r~, then the ball-slab closed form) and BGP on the
slab acceptance set, which do not pass through the reduction of the 2 x 2 Gram matrix. Everything goes through public routes (ocsola,
sola, push/update) so the tests check any fix (europa's plan 037 or the Mac's plan 036)."""
import numpy as np
import pytest
from scipy.stats import norm

import pygeoinf0 as pg
from pygeoinf0.optimisers.dual import ball_slab
from pygeoinf0.convex.subspaces import Zero


def problem(n=1000, nd=30, kscale=1e-6, sig=1e-4):
    x = np.linspace(0, 1, n)
    A = kscale * np.exp(-(x[None] - np.linspace(0.05, 0.95, nd)[:, None]) ** 2 / (2 * 0.05 ** 2))
    blk = (x > 0.4) & (x < 0.6)
    t = blk / blk.sum()                                          # a block average
    M, D = pg.Euclidean(n), pg.Euclidean(nd)
    G, T = pg.Matrix(A, M, D), pg.Matrix(t[None], M, pg.Euclidean(1))
    mt = 300 + 50 * np.sin(7 * x)
    d = D(A @ mt + sig * np.random.default_rng(1).normal(size=nd))
    L = pg.Additive(G, pg.GaussianMeasure(D, sig ** 2 * pg.Identity(D)))
    r = 1.5 * np.linalg.norm(mt)
    return A, t, mt, d, L, T, pg.NormBall(M, r), r, sig


def exact(A, t, xv, d, sig, r, alpha=0.05):
    """The oCSOLA interval from the vectors themselves: max/min <t, m> over ||m|| <= r, |X d - <G* X* 1, m>| <= w."""
    rt = A.T @ xv
    _, R = np.linalg.qr(np.c_[t, rt])
    s, w = float(xv @ d.values), norm.ppf(1 - alpha / 2) * sig * np.linalg.norm(xv)
    return -ball_slab(-R[:, 0], R[:, 1], s, w, r)[0], ball_slab(R[:, 0], R[:, 1], s, w, r)[0]


@pytest.mark.parametrize("gamma", [1e-12, 1e-6, 1.0, 1e6, 1e10])
def test_ocsola_matches_the_exact_two_dimensional_interval_and_bgp(gamma):
    A, t, mt, d, L, T, S, r, sig = problem()
    X = pg.inference.sola(L, T, d, gamma=gamma).X
    xv = X.matrix()[0]
    rr_over_tt = (A.T @ xv) @ (A.T @ xv) / (t @ t)
    ref = exact(A, t, xv, d, sig, r)
    P = pg.push(pg.update(S, L.acceptance(0.05, shape=X), d), T)
    lo, hi = P.bounds(0)
    assert (lo.value, hi.value) == pytest.approx(ref, rel=1e-9)          # the values
    out = P.interval(0)                                                     # the certified outer ends contain ref
    tol = 1e-9 * max(abs(ref[0]), abs(ref[1]))                              # EXACT ends: to TOL
    assert out[0] <= ref[0] + tol and ref[1] <= out[1] + tol
    assert pg.inference.ocsola(S, L, T, d, X=X) == pytest.approx(ref, rel=1e-9)
    assert pg.inference.ocsola(S, L, T, d, gamma=gamma) == pytest.approx(ref, rel=1e-9)
    assert ref[0] < t @ mt < ref[1]                              # the truth is in the interval
    if gamma >= 1e6:
        assert rr_over_tt < 1e-13                                # the regime the old threshold mishandled


@pytest.mark.parametrize("c", [1e-12, 1e-6, 1e6, 1e12, 1e15])
def test_ocsola_does_not_depend_on_the_scale_of_X(c):
    """X and c X give the same slab; the reference is the exact interval, not ocsola itself."""
    A, t, mt, d, L, T, S, r, sig = problem()
    X = pg.inference.sola(L, T, d, gamma=1.0).X
    ref = exact(A, t, X.matrix()[0], d, sig, r)
    assert pg.inference.ocsola(S, L, T, d, X=c * X) == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("c", [1e-9, 1e-15])
def test_bgp_does_not_depend_on_the_units_of_one_datum(c):
    """Datum 0 in units c times smaller (its kernel, value and noise sd): the same acceptance set, the same interval."""
    A, t, mt, d, L, T, S, r, sig = problem(n=400)
    M, D = pg.Euclidean(A.shape[1]), pg.Euclidean(A.shape[0])

    def interval(scale):
        G = pg.Matrix(scale[:, None] * A, M, D)
        Ls = pg.Additive(G, pg.GaussianMeasure(D, pg.Matrix(np.diag((sig * scale) ** 2), D, D)))
        return pg.push(pg.update(S, Ls.acceptance(0.05), D(scale * d.values)), T).interval(0)
    one = np.ones(A.shape[0])
    ref = interval(one)
    small = one.copy()
    small[0] = c
    assert interval(small) == pytest.approx(ref, rel=1e-8)


@pytest.mark.parametrize("c", [1e-6, 1e-9, 1e-12, 1e-15])
def test_bgp_with_exact_data_does_not_depend_on_the_units_of_one_datum(c):
    """Exact data: G m = d with row 0 (and its value) in units c times smaller is the same constraint."""
    n, nd = 400, 6
    x = np.linspace(0, 1, n)
    A = np.exp(-(x[None] - np.linspace(0.1, 0.9, nd)[:, None]) ** 2 / (2 * 0.1 ** 2))
    t = ((x > 0.4) & (x < 0.6)) / np.sum((x > 0.4) & (x < 0.6))
    mt = 300 + 50 * np.sin(7 * x)
    M, D = pg.Euclidean(n), pg.Euclidean(nd)
    S, T = pg.NormBall(M, 1.5 * np.linalg.norm(mt)), pg.Matrix(t[None], M, pg.Euclidean(1))

    def interval(scale):
        G = pg.Matrix(scale[:, None] * A, M, D)
        return pg.push(pg.update(S, pg.Additive(G, Zero(D)), D(G.matrix() @ mt)), T).interval(0)
    ref = interval(np.ones(nd))
    assert ref[0] < t @ mt < ref[1]
    assert interval(np.r_[c, np.ones(nd - 1)]) == pytest.approx(ref, rel=1e-8)
