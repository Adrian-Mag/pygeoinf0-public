"""Plan 054, estimable targets (the author's decision of 2026-11-06). With no prior a target in the data's span only to
rounding cannot be told from one outside it, whose bound is +inf, so estimability is decided by how the target was made
(core/estimable.py), the same in both routes:

    by construction   G.adjoint(y), a datum's functional (G*(e_k), G[k].representer), their combinations, a map L @ G or
                      R.select(rows) of G: EXACT where the digits hold, the note says "by construction"
    declared          K.bound(c, estimable=True), K.image(T, estimable=True), P.estimable = True: the caller's
                      responsibility (a false declaration gives a wrong value, and the note says so; one the numbers
                      contradict beyond rounding is refused)
    neither           the infinite bracket, the note naming both; K.dependent declares the data, not the target

References are exact: y.d + rho |y| for G* y over the noise ball rho (the target exactly in the span), mpmath at 60
digits for the given doubles, +inf off the span. Each test has a planted-defect control."""
import math
import warnings

import mpmath as mp
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.convex import feasible
from pygeoinf0.convex.feasible import FeasibleSet
from pygeoinf0.core import estimable
from pygeoinf0.core.errors import Refused
from pygeoinf0.relations import select

TOL, RHO = 1e-9, 0.1
ROUTES = [None, False]                                   # the decomposition (spectral) and the reduction per direction


def _problem(smin=0.5, seed=2):
    """m_unb.py's problem: G = U diag(s) V^T in R^(8 x 20), s geometric 1 .. smin; y unit in R^8, w unit off the span."""
    n, m = 8, 20
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    V = np.linalg.qr(rng.standard_normal((m, m)))[0]
    G = (U * np.geomspace(1, smin, n)) @ V[:, :n].T
    y = rng.standard_normal(n)
    y /= np.linalg.norm(y)
    w = V[:, n:] @ rng.standard_normal(m - n)
    d = G @ rng.standard_normal(m) + 0.3 * RHO * rng.standard_normal(n) / math.sqrt(n)
    X, D = pg.Euclidean(m), pg.Euclidean(n)
    return X, D, pg.Matrix(G, X, D), G, d, y, w / np.linalg.norm(w)


def _exact(coef, d):
    """sup <G* coef, m> over ||d - G m|| <= RHO, no prior: coef.d + RHO |coef| (mpmath, 60 digits)."""
    with mp.workdps(60):
        c = [mp.mpf(float(x)) for x in coef]
        return float(mp.fsum(ci * mp.mpf(float(di)) for ci, di in zip(c, d)) + RHO * mp.sqrt(mp.fsum(ci * ci for ci in c)))


def _K(Gop, D, d, route, dependent=False):
    return FeasibleSet(None, Gop, pg.NormBall(D, RHO), D(d), decomposition=route, dependent=dependent)


@pytest.mark.parametrize("route", ROUTES)
def test_targets_built_from_the_data_functionals_are_estimable(route, monkeypatch):
    """G* y, a datum's functional G* e_0 and 2 G* e_0 - 0.5 G* e_3 + G* y: EXACT at the exact value, the note says
    "by construction". The same numbers given as X(t): the infinite bracket. Control: the record ignored gives the
    infinite bracket for the built targets too."""
    X, D, Gop, G, d, y, w = _problem()
    e = np.eye(8)
    built = [(Gop.adjoint(D(y)), y), (Gop.adjoint(D(e[0])), e[0]),
             (2.0 * Gop.adjoint(D(e[0])) - 0.5 * Gop.adjoint(D(e[3])) + Gop.adjoint(D(y)), 2 * e[0] - 0.5 * e[3] + y)]
    for c, coef in built:
        ref = _exact(coef, d)
        b = _K(Gop, D, d, route).bound(c)
        assert b.status == "EXACT" and abs(b.value - ref) <= TOL * max(1.0, abs(ref)) and "by construction" in b.note
        nb = _K(Gop, D, d, route).bound(X(np.array(c.values)))
        assert nb.status == "BRACKET" and math.isinf(nb.upper) and "neither declared" in nb.note
    monkeypatch.setattr(estimable, "target", lambda c, G, declared=False: None)
    b0 = _K(Gop, D, d, route).bound(built[0][0])
    assert b0.status == "BRACKET" and math.isinf(b0.upper)


@pytest.mark.parametrize("route", ROUTES)
def test_property_maps_made_from_the_data_are_estimable(route, monkeypatch):
    """T = L @ G and R.select([0, 3]) of G: every direction estimable by construction (EXACT at the exact value);
    T = Matrix(L G) from numbers: the infinite bracket, unless declared (P.estimable = True). Control: the composition's
    record removed (estimable.within a no-op) gives the infinite bracket."""
    X, D, Gop, G, d, y, w = _problem()
    L = np.random.default_rng(7).standard_normal((2, 8))
    q = np.array([1.0, 0.0])
    for T, coef in ((pg.Matrix(L, D, pg.Euclidean(2)) @ Gop, L[0]), (select(Gop, [0, 3]), np.eye(8)[0])):
        ref = _exact(coef, d)
        b = _K(Gop, D, d, route).image(T).bound(q)
        assert b.status == "EXACT" and abs(b.value - ref) <= TOL * max(1.0, abs(ref)) and "by construction" in b.note
    P = _K(Gop, D, d, route).image(pg.Matrix(L @ G, X, pg.Euclidean(2)))
    assert math.isinf(P.bound(q).upper)
    P.estimable = True
    b = P.bound(q)
    assert b.status == "EXACT" and abs(b.value - _exact(L[0], d)) <= TOL * max(1.0, abs(b.value))
    monkeypatch.setattr(estimable, "within", lambda out, left=None, right=None: out)
    T0 = pg.Matrix(L, D, pg.Euclidean(2)) @ Gop
    assert math.isinf(_K(Gop, D, d, route).image(T0).bound(q).upper)


@pytest.mark.parametrize("route", ROUTES)
def test_combining_with_an_unmarked_element_loses_the_mark(route, monkeypatch):
    """G* y + X(1e-12 |t| w) is not in the span (truth +inf): the infinite bracket. Control: a record that keeps what
    either term has gives a finite EXACT, wrong."""
    X, D, Gop, G, d, y, w = _problem()
    c = Gop.adjoint(D(y))
    off = X(1e-12 * np.linalg.norm(c.values) * w)
    b = _K(Gop, D, d, route).bound(c + off)
    assert b.status == "BRACKET" and math.isinf(b.upper)
    monkeypatch.setattr(estimable, "meet", lambda out, x, y: estimable._set(out, estimable.of(x) | estimable.of(y)))
    b0 = _K(Gop, D, d, route).bound(c + off)
    assert b0.status == "EXACT" and math.isfinite(b0.value)          # the planted defect: the truth is +inf


@pytest.mark.parametrize("route", ROUTES)
def test_a_declaration_is_the_callers_responsibility(route):
    """Declared, X(t) with t = G^T y: EXACT at the exact value of the given doubles, the note says "declared". A false
    declaration (t + 1e-10 |t| w, truth +inf) gives a finite value, wrong, with the same note: the control. One the
    numbers contradict beyond rounding (t + 1e-2 |t| w) is refused with the choices named."""
    X, D, Gop, G, d, y, w = _problem()
    t = G.T @ y
    with mp.workdps(60):
        Gm = mp.matrix(G.tolist())
        mu = mp.lu_solve(Gm * Gm.T, Gm * mp.matrix(t.tolist()))
    ref = _exact([float(x) for x in mu], d)
    b = _K(Gop, D, d, route).bound(X(t), estimable=True)
    assert b.status == "EXACT" and abs(b.value - ref) <= TOL * max(1.0, abs(ref)) and "caller's responsibility" in b.note
    false = _K(Gop, D, d, route).bound(X(t + 1e-10 * np.linalg.norm(t) * w), estimable=True)
    assert math.isfinite(false.upper) and "declared estimable" in false.note     # wrong: the truth is +inf
    with pytest.raises(Refused, match="contradict") as err:
        _K(Gop, D, d, route).bound(X(t + 1e-2 * np.linalg.norm(t) * w), estimable=True)
    assert any("prior" in n for n in err.value.needs)
    assert _K(Gop, D, d, route).bound(X(t + 1e-2 * np.linalg.norm(t) * w)).status == "UNBOUNDED"


@pytest.mark.parametrize("route", ROUTES)
def test_k_dependent_declares_the_data_not_the_target(route, monkeypatch):
    """t = G^T y + 1e-12 |t| w (truth +inf) with K.dependent = True: the infinite bracket. Control: K.dependent read as
    a declaration for targets too (the former meaning) gives a finite EXACT, wrong."""
    X, D, Gop, G, d, y, w = _problem()
    t = G.T @ y
    c = X(t + 1e-12 * np.linalg.norm(t) * w)
    b = _K(Gop, D, d, route, dependent=True).bound(c)
    assert b.status == "BRACKET" and math.isinf(b.upper)
    monkeypatch.setattr(estimable, "target", lambda c, G, declared=False: estimable.DECLARED)
    b0 = _K(Gop, D, d, route, dependent=True).bound(c)
    assert b0.status == "EXACT" and math.isfinite(b0.value)


@pytest.mark.parametrize("route", ROUTES)
def test_function_space_kernels(route):
    """Gaussian kernels in L2[0, 1]: G[0].representer and G* y are estimable by construction (EXACT at the closed form
    e_0.d + rho, y.d + rho |y|); the same kernel made again by the caller from its formula: the infinite bracket (the
    control: the record, not the values, decides)."""
    M = pg.L2(pg.Interval(0, 1), tol=1e-8)
    cs = np.linspace(0.1, 0.9, 8)
    G = pg.Functionals([M(lambda x, c=c: np.exp(-(x - c) ** 2 / 0.02)) for c in cs])
    D = G.codomain
    rng = np.random.default_rng(8)
    d = D(0.1 * rng.standard_normal(8))
    y = rng.standard_normal(8)
    K = FeasibleSet(None, G, pg.NormBall(D, RHO), d, decomposition=route)
    for c, coef in ((G[0].representer, np.eye(8)[0]), (G.adjoint(D(y)), y)):
        ref = float(coef @ d.values + RHO * np.linalg.norm(coef))
        b = K.bound(c)
        assert b.status == "EXACT" and abs(b.value - ref) <= TOL * max(1.0, abs(ref)) and "by construction" in b.note
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = K.bound(M(lambda x: np.exp(-(x - cs[0]) ** 2 / 0.02)))
    assert again.status == "BRACKET" and math.isinf(again.upper)


def test_an_empty_test_without_a_prior_agrees_on_both_routes(monkeypatch):
    """K.is_empty() with no prior: FALSE on both routes (direction 0 = G* 0 is in the span by construction); it was
    UNKNOWN through the decomposition (an infinite bracket). Control: direction 0 without the record."""
    X, D, Gop, G, d, y, w = _problem()
    assert all(_K(Gop, D, d, route).is_empty() is pg.Truth.FALSE for route in ROUTES)
    monkeypatch.setattr(feasible._how, "BUILT", None)               # feasible's name for core/estimable.py
    assert _K(Gop, D, d, None).is_empty() is pg.Truth.UNKNOWN


def test_many_properties_through_the_decomposition_say_which_applied():
    """P.intervals() (vectorised, the spectral route) with no prior: T = L @ G gives EXACT ends at the exact values and
    a claim "by construction"; T from numbers gives BRACKET ends (infinite brackets) and the claim "neither" (the
    control); declared,
    EXACT again with the claim "declared"."""
    X, D, Gop, G, d, y, w = _problem()
    L = np.random.default_rng(9).standard_normal((3, 8))
    lo, hi, sl, su, cl = _K(Gop, D, d, None).image(pg.Matrix(L, D, pg.Euclidean(3)) @ Gop).intervals()
    refs = [(-_exact(-L[k], d), _exact(L[k], d)) for k in range(3)]
    assert all(s == "EXACT" for s in list(sl) + list(su))
    assert np.allclose(np.c_[lo, hi], refs, rtol=TOL, atol=TOL) and "by construction" in str(cl)
    P = _K(Gop, D, d, None).image(pg.Matrix(L @ G, X, pg.Euclidean(3)))
    lo0, hi0, sl0, su0, cl0 = P.intervals()                       # the values assume it estimable; not EXACT
    assert all(s == "BRACKET" for s in list(sl0) + list(su0)) and "neither declared" in str(cl0)
    P.estimable = True
    lo1, hi1, sl1, su1, cl1 = P.intervals()
    assert all(s == "EXACT" for s in list(sl1) + list(su1)) and "declared estimable" in str(cl1)
