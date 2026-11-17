"""Plan 054 fix round: data directions at the rounding floor, against exact answers, each with a planted-defect control
(the former behaviour, shown to fail the same check).

    optimisers/dual/reduction.ball_exact, none_ball  a data direction at the rounding floor, not exactly zero, is refused with the
    (and SpectralBall with no prior)   choices named; the data declared dependent (K.dependent) drop it, as before
    convex/feasible.FeasibleSet        the data Gram's correlation eigenvalues at the floor, less the structural
                                       n - dim M, decide it (the reduction drops them before the solver sees them)
    convex/subspaces.Span              its dimension at the floor eps k max of the Gram (was 1e-12)
References are exact for the given double inputs: mpmath at 60-80 digits, or closed forms from known factors."""
import math

import mpmath as mp
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.optimisers.dual import reduction as dual
from pygeoinf0.convex.feasible import FeasibleSet
from pygeoinf0.core.errors import Refused
from pygeoinf0.convex import subspaces
from pygeoinf0.convex.subspaces import Zero


def _mpm(A):
    return mp.matrix([[mp.mpf(float(x)) for x in row] for row in np.atleast_2d(A)])


def _near_dependent(delta, seed=0):
    """Five data representers in R^12 with g5 = g4 + delta h (delta = 0: a bitwise duplicate); the target t = g5."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((5, 12))
    G[4] = G[3] + delta * rng.standard_normal(12) if delta else G[3]
    m = rng.standard_normal(12)
    d = G @ (0.5 * m / np.linalg.norm(m))
    return G, d, G[4].copy(), rng


def _mp_none(G, d, t, rho):
    """sup t.m over ||d - G m|| <= rho with no prior, G of full row rank in exact arithmetic: mu.d + rho |mu|."""
    with mp.workdps(60):
        Gm, tm = _mpm(G), _mpm(t).T
        mu = mp.lu_solve(Gm * Gm.T, Gm * tm)
        return float(mp.fsum(mu[i] * mp.mpf(d[i]) for i in range(len(d))) + mp.mpf(rho) * mp.norm(mu))


def _mp_exact(G, d, t, r):
    """sup t.m over ||m|| <= r, G m = d, G of full row rank in exact arithmetic."""
    with mp.workdps(60):
        Gm, tm, dm = _mpm(G), _mpm(t).T, _mpm(d).T
        GGt = Gm * Gm.T
        m0 = Gm.T * mp.lu_solve(GGt, dm)
        pt = tm - Gm.T * mp.lu_solve(GGt, Gm * tm)
        return float(mp.fsum(tm[i] * m0[i] for i in range(len(t))) + mp.sqrt(r ** 2 - mp.norm(m0) ** 2) * mp.norm(pt))


@pytest.mark.parametrize("route", [False, None])                 # the reduction per direction, the decomposition
def test_near_dependent_data_are_refused(route):
    """g5 = g4 + 1e-9 h: the data Gram's correlation eigenvalue (about 1e-18) is below the floor. Exact answers:
    no prior, a noise ball: 0.0431 (none_ball); a prior ball, exact data: 0.0432 (ball_exact). Refused, naming the
    choices. Control: declaring the data dependent gives the former result, 13 % wrong and a false EMPTY."""
    G, d0, t, rng = _near_dependent(1e-9)
    X, D = pg.Euclidean(12), pg.Euclidean(5)
    Gop, c, rho = pg.Matrix(G, X, D), X(t), 0.1
    dn = d0 + 0.3 * rho * rng.standard_normal(5) / math.sqrt(5)
    none, exact = _mp_none(G, dn, t, rho), _mp_exact(G, d0, t, 1.0)

    def sets(dependent):
        return (FeasibleSet(None, Gop, pg.NormBall(D, rho), D(dn), decomposition=route, dependent=dependent),
                FeasibleSet(pg.NormBall(X, 1.0), Gop, Zero(D), D(d0), decomposition=False, dependent=dependent))
    for K in sets(False):
        with pytest.raises(Refused, match="rounding floor") as err:
            K.bound(c)
        assert any("K.dependent" in n for n in err.value.needs) and any("regularise" in n for n in err.value.needs)
    Kn, Ke = sets(True)                                              # the planted defect: the former drop
    assert abs(Kn.bound(c).value - none) > 0.1 * abs(none)
    assert Ke.bound(c).status == "EMPTY" and math.isfinite(exact)


@pytest.mark.parametrize("route", [False, None])
def test_declared_duplicates_are_exact(route):
    """g5 = g4 bitwise: the data are exactly dependent; declared so (K.dependent = True), and the target t = g5
    declared in their span (estimable=True: K.dependent declares the data only, plan 054's estimable targets), the
    bounds are those of the four distinct data (exact answers from the reduced problem in mpmath). Undeclared: refused
    (a duplicate cannot be told from a near duplicate by the Gram matrix); the data alone declared: not EXACT (the
    target's own floor direction is admitted). Control: the reduced reference with the duplicate's datum moved by 1e-6
    is missed by the same check."""
    G, d0, t, rng = _near_dependent(0.0)
    X, D = pg.Euclidean(12), pg.Euclidean(5)
    Gop, c, rho = pg.Matrix(G, X, D), X(t), 0.1
    dn = d0 + 0.3 * rho * rng.standard_normal(5) / math.sqrt(5)
    with mp.workdps(60):                                             # the duplicate: rows g1..g3, sqrt 2 g4
        Gm = _mpm(G[:4])
        for j in range(12):
            Gm[3, j] *= mp.sqrt(2)
        rr = mp.sqrt(mp.mpf(rho) ** 2 - (mp.mpf(dn[3]) - mp.mpf(dn[4])) ** 2 / 2)
        dm = [mp.mpf(dn[0]), mp.mpf(dn[1]), mp.mpf(dn[2]), (mp.mpf(dn[3]) + mp.mpf(dn[4])) / mp.sqrt(2)]
        mu = mp.lu_solve(Gm * Gm.T, Gm * _mpm(t).T)
        none = float(mp.fsum(mu[i] * dm[i] for i in range(4)) + rr * mp.norm(mu))
        wrong = float(mp.fsum(mu[i] * dm[i] for i in range(4)) + mp.mpf(1e-6) * mu[3] + rr * mp.norm(mu))
    exact = _mp_exact(G[:4], d0[:4], t, 1.0)
    Kn = FeasibleSet(None, Gop, pg.NormBall(D, rho), D(dn), decomposition=route)
    Ke = FeasibleSet(pg.NormBall(X, 1.0), Gop, Zero(D), D(d0), decomposition=False)
    for K in (Kn, Ke):
        with pytest.raises(Refused, match="rounding floor"):
            K.bound(c)
        K.dependent = True
        assert K.bound(c).status == "BRACKET"                       # the data declared, the target not
    bn, be = Kn.bound(c, estimable=True), Ke.bound(c, estimable=True)
    assert abs(bn.value - none) <= 1e-12 * abs(none) and abs(be.value - exact) <= 1e-12 * abs(exact)
    assert "declared estimable (estimable=True)" in bn.note and bn.status == be.status == "EXACT"
    assert abs(bn.value - wrong) > 1e-8 * abs(none)                  # the check resolves a 1e-6 change of a datum


def test_a_singular_value_at_the_floor_is_refused():
    """B = U diag(s) V^T with s from 1 to 1e-15 (below the floor eps max(n, k) = 2.7e-15, not zero), d = B y: the
    exact value of ball_exact (mpmath, 80 digits) is not what dropping gives. Refused; declared dependent, the former
    value, measured 37 % wrong in plan 054 P2 (seed 1)."""
    n, k = 8, 12
    rng = np.random.default_rng(1)
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    V = np.linalg.qr(rng.standard_normal((k, k)))[0][:, :n]
    B = U @ np.diag(np.geomspace(1.0, 1e-15, n)) @ V.T
    y = rng.standard_normal(k)
    d = B @ (0.5 * y / np.linalg.norm(y))
    a = rng.standard_normal(k)
    with mp.workdps(80):
        Bm, dm, am = _mpm(B), _mpm(d).T, _mpm(a).T
        BBt = Bm * Bm.T
        y0 = Bm.T * mp.lu_solve(BBt, dm)
        pa = am - Bm.T * mp.lu_solve(BBt, Bm * am)
        ref = float(mp.fsum(am[i] * y0[i] for i in range(k)) + mp.sqrt(1 - mp.norm(y0) ** 2) * mp.norm(pa))
    with pytest.raises(Refused, match="1 of 8 data directions are at the rounding floor"):
        dual.ball_exact(a, B, d, 1.0)
    with pytest.raises(Refused):
        dual.none_ball(B.T @ rng.standard_normal(n), B, d, 0.1)
    info = {}
    val = dual.ball_exact(a, B, d, 1.0, info, dependent=True)[0]     # the planted defect: the former drop
    assert info["declared"] and info["data_dropped"] == 1 and abs(val - ref) > 0.3 * abs(ref)


def test_more_data_than_unknowns_is_not_refused(monkeypatch):
    """Seven exact data of a model in R^4: rank G G* <= 4, so three eigenvalues are zero by structure (n - dim M), not
    an ambiguity. The feasible set is the single model m = G^+ d, so sup <t, m> = t . m (closed form). Control: without
    the structural count the same set is refused."""
    rng = np.random.default_rng(3)
    G, mt = rng.standard_normal((7, 4)), rng.standard_normal(4)
    X, D = pg.Euclidean(4), pg.Euclidean(7)
    K = FeasibleSet(pg.NormBall(X, 2.0 * np.linalg.norm(mt)), pg.Matrix(G, X, D), Zero(D), D(G @ mt), decomposition=False)
    t = rng.standard_normal(4)
    b = K.bound(X(t))
    assert K.dependency()[1:3] == (3, 3) and abs(b.value - t @ mt) <= 1e-12 * np.linalg.norm(t) * np.linalg.norm(mt)
    monkeypatch.setattr(dual, "structural_zeros", lambda n, G: 0)  # the planted defect: no structural count
    K2 = FeasibleSet(K.prior, K.G, K.noise, K.d, decomposition=False)
    with pytest.raises(Refused, match="3 of 7"):
        K2.bound(X(t))


def test_span_keeps_directions_that_hold_digits(monkeypatch):
    """span{v1, 1e-7 v2} and span{v1, v1 + 1e-6 v2} are span{v1, v2} (exact: the projection of x is its first two
    coordinates, to the Gram matrix's rounding cond eps); their Gram eigenvalues, 1e-14 and 5e-13 of the largest, lie
    above the floor eps k = 4.4e-16.
    span{v1, v1} has dimension 1 and says it dropped one. Control: the former cut 1e-12 gives dimension 1 and loses
    half of x's projection."""
    X = pg.Euclidean(5)
    e = np.eye(5)
    x = X(np.array([0.3, -1.2, 0.7, 0.1, 2.0]))
    exact = np.array([0.3, -1.2, 0.0, 0.0, 0.0])

    def spans():
        return pg.Span([X(e[0]), X(1e-7 * e[1])]), pg.Span([X(e[0]), X(e[0] + 1e-6 * e[1])])
    for S, tol in zip(spans(), (1e-12, 1e-3)):                      # measured 0 and 1.1e-4: the second Gram's
        assert S.dim == 2 and S.dropped == 0                         # cond 2e12 costs digits (cond eps |x|^2 = 3e-3)
        assert np.abs(S.project(x).values - exact).max() < tol
    dup = pg.Span([X(e[0]), X(e[0])])
    assert dup.dim == 1 and "1 correlation eigenvalue(s)" in dup.claims.find("dimension").statement
    monkeypatch.setattr(subspaces, "EPS", 1e-12 / 2)                 # the planted defect: the former cut 1e-12
    first, second = spans()
    assert first.dim == 2                                            # scale-free since the general rule: 1e-7 v2 is kept
    assert second.dim == 1 and np.abs(second.project(x).values - exact).max() > 0.5
