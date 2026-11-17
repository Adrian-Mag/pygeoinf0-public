"""Plan 054, pre-existing bugs and consistency (the author's rulings of 2026-11-06: "These rules must be generally true";
a wrong result that can be detected is fixed, or refused with the choices named). Each test is checked against an exact
answer and has a planted-defect control (the former behaviour, put back).

    optimisers/dual/spectral.py, exactness.py   no prior in the eigenbasis: estimability decided by the stated error of the part
                                       outside the data's span (it was a_perp > 1e-12 |a| with a_perp = sqrt(eps) |t|
                                       from the subtraction), the infinite bracket of the reduce route's lift where the
                                       stated error admits both answers, and first order for a target declared
                                       estimable (estimable=True; K.dependent until plan 054's estimable targets)
    solvers/dataspace/cancellation.py  SOLA's mismatch and the posterior variance formed from the residual element
    probability/posterior_covariance.py   the dense posterior covariance in Joseph's form
    operators/functionals.py           the kernels' a-posteriori Gram error: ||G~ - G'|| + G''s own bound
    solvers.direct (compression)       a compressed Gram whose worst direction has no digit is refused
References: mpmath at 60 digits for the given doubles; the erf closed form for Gaussian kernels; exact energies."""
import math
import sys
import warnings

import mpmath as mp
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import solvers
from pygeoinf0.optimisers.dual import exactness
from pygeoinf0.convex.feasible import FeasibleSet
from pygeoinf0.core.errors import PrecisionWarning, RealisationWarning, Refused
from pygeoinf0.solvers.dataspace import cancellation
from pygeoinf0.operators import functionals

TOL = 1e-9


# ---------------------------------------------------------------------------------------------- items 1 and 2
def _unb(smin, seed, perp=None):
    """m_unb.py's problem: G in R^(8 x 20), noise ball 0.1, no prior; t = G^T y (+ 10^-perp |t| w, w off the span)."""
    n, m = 8, 20
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    V = np.linalg.qr(rng.standard_normal((m, m)))[0]
    G = (U * np.geomspace(1, smin, n)) @ V[:, :n].T
    y = rng.standard_normal(n)
    t = G.T @ (y / np.linalg.norm(y))
    w = V[:, n:] @ rng.standard_normal(m - n)
    d = G @ rng.standard_normal(m) + 0.03 * rng.standard_normal(n) / math.sqrt(n)
    if perp is not None:
        t = t + 10.0 ** -perp * np.linalg.norm(t) * w / np.linalg.norm(w)
    return G, d, t


def _estimable_value(G, d, t, rho=0.1):
    mp.mp.dps = 60
    Gm, tm, dm = mp.matrix(G.tolist()), mp.matrix(t.tolist()), mp.matrix(d.tolist())
    mu = mp.lu_solve(Gm * Gm.T, Gm * tm)
    return float(mp.fsum(mu[i] * dm[i] for i in range(G.shape[0])) + rho * mp.norm(mu))


def _bound(G, d, t, route, declared=False):
    """The no-prior bound of X(t); declared: the target declared estimable (estimable=True)."""
    X, D = pg.Euclidean(G.shape[1]), pg.Euclidean(G.shape[0])
    K = FeasibleSet(None, pg.Matrix(G, X, D), pg.NormBall(D, 0.1), D(d), decomposition=route)
    return K.bound(X(t), estimable=declared)


def _former(G, d, t, planted=None):
    """The former decision in the same coordinates: UNBOUNDED where a_perp > 1e-12 |a|, else the finite bound; and whether
    a_perp^2 is within the stated error e2. planted: a fraction f, a_perp^2 = f e2 in place of the computed difference
    cc - |a_pos|^2, whose rounding (+-1e-16 here) has the sign of the machine's BLAS: a value the stated error admits."""
    X, D = pg.Euclidean(G.shape[1]), pg.Euclidean(G.shape[0])
    K = FeasibleSet(None, pg.Matrix(G, X, D), pg.NormBall(D, 0.1), D(d))
    sp = K._spectral()
    beta, cc = sp.dec._coef(K.G(X(t)).values), float(t @ t)
    a_pos = beta[sp.pos] / sp.s
    e2 = exactness.perp_error(sp, a_pos / sp.s, cc, a_pos)
    a_perp = math.sqrt(max(cc - float(a_pos @ a_pos) if planted is None else planted * e2, 0.0))
    return ("UNBOUNDED" if a_perp > 1e-12 * math.sqrt(a_perp ** 2 + float(a_pos @ a_pos)) else "finite"), a_perp ** 2 <= e2


def test_no_prior_estimability_is_decided_by_the_stated_error():
    """An estimable target (seed 2 at cond 4 was UNBOUNDED before) is not UNBOUNDED: the stated error admits both answers,
    so the bracket is infinite, and declared (estimable=True) it is EXACT at the exact value; a target 1e-2 off the span is
    UNBOUNDED, one 1e-10 off (EXACT before, the truth +inf) gets the infinite bracket. Control: the former test."""
    G, d, t = _unb(0.5, 2)
    ref = _estimable_value(G, d, t)
    b = _bound(G, d, t, None)
    assert b.status == "BRACKET" and b.lower == -math.inf and b.upper == math.inf and "estimable=True" in b.note
    bd = _bound(G, d, t, None, declared=True)
    assert bd.status == "EXACT" and abs(bd.value - ref) <= TOL * max(1.0, abs(ref))
    assert _bound(*_unb(0.5, 2, perp=2), None).status == "UNBOUNDED"
    far = _bound(*_unb(1.0, 1, perp=10), None)
    assert far.status == "BRACKET" and far.upper == math.inf
    # control: the former test decided by the rounding of a_perp: two values the stated error admits, two answers
    assert _former(G, d, t, planted=0.25) == ("UNBOUNDED", True) and _former(G, d, t, planted=0.0) == ("finite", True)
    assert _former(*_unb(1.0, 1, perp=10)) == ("finite", True)


@pytest.mark.parametrize("smin, seed", [(1.0, 0), (0.5, 2), (1e-2, 1), (1e-5, 0)])
def test_both_routes_follow_one_rule_for_an_estimable_target(smin, seed):
    """The reduce route (its lifted re-solve) and the spectral route give the same status: an infinite bracket unless
    declared, and declared the same value (EXACT where it holds its digits, else a bracket holding the exact value).
    Control: the spectral route without the rule disagrees with the reduce route."""
    G, d, t = _unb(smin, seed)
    ref = _estimable_value(G, d, t)
    for dep in (False, True):
        bs, br = _bound(G, d, t, None, dep), _bound(G, d, t, False, dep)
        assert bs.status == br.status
        for b in (bs, br):
            slack = TOL * max(1.0, abs(ref))
            assert b.lower - slack <= ref <= b.upper + slack
            if b.status == "EXACT":
                assert abs(b.value - ref) <= TOL * max(1.0, abs(ref))
        if not dep:
            assert math.isinf(bs.upper) and math.isinf(br.upper)
    assert _former(G, d, t)[1]                     # control: the former test decided what the stated error leaves open


def test_a_declared_no_prior_bound_is_judged_by_first_order(monkeypatch):
    """Declared, at cond(GG*) 1e10 (seed 0): EXACT before with error 13 TOL; now a bracket holding the exact value.
    Control: first order and the re-solves off give the wrong EXACT back."""
    G, d, t = _unb(1e-5, 0)
    ref = _estimable_value(G, d, t)
    b = _bound(G, d, t, None, declared=True)
    assert b.status == "BRACKET" and b.lower <= ref <= b.upper
    monkeypatch.setattr(exactness, "first_order_none", lambda *a: 0.0)
    b0 = _bound(G, d, t, None, declared=True)
    assert b0.status == "EXACT" and abs(b0.value - ref) > TOL * max(1.0, abs(ref))


# ---------------------------------------------------------------------------------------------- item 3
def _dec2(smin=1e-1, seed=0):
    """P4's m4_dec2 problem: G in R^(30 x 60), sigma 1e-2, T = [G^T r (estimable), random]."""
    n, m = 30, 60
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    V = np.linalg.qr(rng.standard_normal((m, n)))[0]
    G = (U * np.geomspace(1.0, smin, n)) @ V.T
    T = np.vstack([G.T @ rng.standard_normal(n), rng.standard_normal(m)])
    d = G @ rng.standard_normal(m) + 1e-2 * rng.standard_normal(n)
    return G, T, d


def _mpm(A):
    return mp.matrix([[mp.mpf(float(x)) for x in r] for r in np.atleast_2d(A)])


def test_sola_mismatch_is_formed_from_the_residual(monkeypatch):
    """gamma = 1e-6 at cond(W) 1e2: the Gram route's mismatch of the estimable target was 1e3 relative off; formed from the
    residual element it is within 1e-4 of the exact one, and within its stated error. Control: without the residual
    route, the Gram route's value (no correct digit) is refused by default and warned of with refuse=False."""
    mp.mp.dps = 60
    G, T, d = _dec2()
    n, m, g = 30, 60, 1e-6
    Gm, Tm = _mpm(G), _mpm(T)
    X = Tm * Gm.T * mp.inverse(Gm * Gm.T + g * mp.mpf(1e-2) ** 2 * mp.eye(n))
    R = Tm - X * Gm
    exact = [float(mp.fsum(R[i, j] ** 2 for j in range(m))) for i in range(2)]
    M, D, P = pg.Euclidean(m), pg.Euclidean(n, covariance=1e-4), pg.Euclidean(2)
    dec = pg.decompose(pg.Matrix(G, M, D), D.standard_gaussian())
    Top = pg.Matrix(T, M, P)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        r = dec.sola(Top, g, D(d))
    err = np.abs(r.mismatch - exact)
    assert np.all(err <= 1e-4 * np.abs(exact)) and np.all(err <= r.mismatch_error)
    monkeypatch.setattr(cancellation, "residuals", lambda *a: None)
    dec._residual_cache = {}                    # the route kept for the next call (plan 054 fix2) goes with it
    with pytest.raises(Refused):
        dec.sola(Top, g, D(d))
    with pytest.warns(PrecisionWarning):
        r0 = dec.sola(Top, g, D(d), refuse=False)
    # no correct digit: off by the value or more (europa, OpenBLAS: more; the Mac, Accelerate: 0.0 for 7e-18), within the
    # stated error
    assert abs(r0.mismatch[0] - exact[0]) >= abs(exact[0]) and abs(r0.mismatch[0] - exact[0]) <= r0.mismatch_error[0]


def _exact_post_std(G, T, a, sigma=1e-2):
    mp.mp.dps = 60
    Gm, Tm, a = _mpm(G), _mpm(T), mp.mpf(a)
    A = a ** 2 * Gm * Gm.T + mp.mpf(sigma) ** 2 * mp.eye(G.shape[0])
    cov = a ** 2 * Tm * Tm.T - a ** 4 * Tm * Gm.T * mp.inverse(A) * Gm * Tm.T
    return np.array([float(mp.sqrt(cov[i, i])) for i in range(2)])


def test_posterior_std_is_formed_without_the_subtraction(monkeypatch):
    """a = 1e6 at cond(W) 1e2: the closed form's std of the estimable property was 8e-2 to 1.0 relative off (P4); now
    within 1e-9 and its stated error. Control: without the residual route the Gram route is refused (no digit)."""
    G, T, d = _dec2()
    exact = _exact_post_std(G, T, 1e6)
    M, D, P = pg.Euclidean(60), pg.Euclidean(30, covariance=1e-4), pg.Euclidean(2)
    dec = pg.decompose(pg.Matrix(G, M, D), D.standard_gaussian())
    Top = pg.Matrix(T, M, P)
    post = dec.posterior(Top, D(d), a=1e6, nu=1.0)
    err = np.abs(post.std - exact)
    assert np.all(err <= 1e-9 * exact) and np.all(err <= post.std_error)
    monkeypatch.setattr(cancellation, "residuals", lambda *a: None)
    with pytest.raises(Refused):
        dec.posterior(Top, D(d), a=1e6, nu=1.0)


def test_dense_posterior_covariance_in_josephs_form(monkeypatch):
    """pg.update + push at a = 1e6, cond 1e2: std within 1e-12 of the exact one (it was 8e-2 off, no warning). Control:
    the former operator C - K S^-1 G C."""
    G, T, d = _dec2()
    exact = _exact_post_std(G, T, 1e6)
    M, D, P = pg.Euclidean(60), pg.Euclidean(30, covariance=1e-4), pg.Euclidean(2)

    def std():
        prior = pg.GaussianMeasure(M, 1e12 * pg.Identity(M))
        post = pg.push(pg.update(prior, pg.Additive(pg.Matrix(G, M, D), D.standard_gaussian()), D(d)), pg.Matrix(T, M, P))
        return np.sqrt(np.diag(post.covariance.matrix()))
    assert np.all(np.abs(std() - exact) <= 1e-12 * exact)
    from pygeoinf0.probability import posterior_covariance as pc
    monkeypatch.setattr(pc, "PosteriorCovariance", lambda C, G, Ce, Sinv, **kw: C - C @ G.adjoint @ Sinv @ G @ C)
    assert np.max(np.abs(std() - exact) / exact) > 1e-3


# ---------------------------------------------------------------------------------------------- item 4
def test_kernel_estimate_adds_the_refinements_own_bound(monkeypatch):
    """12 Gaussian kernels at tol 1e-5 (the suite's case): the a-posteriori Gram error is ||G~ - G'|| + G''s own first-
    order bound, a bound of the true error (erf closed form at 40 digits). Control: the difference alone (round 1)."""
    s2, cs = 0.02, np.linspace(0.1, 0.9, 12)
    mp.mp.dps = 40

    def g(a, b):
        a, b = mp.mpf(a), mp.mpf(b)
        mid, q, r = (a + b) / 2, (a - b) ** 2 / (2 * s2), mp.sqrt(2 / mp.mpf(s2))
        return mp.e ** (-q) * mp.sqrt(mp.pi * s2 / 2) * (mp.erf(r * (1 - mid)) - mp.erf(-r * mid)) / 2
    exact = np.array([[float(g(a, b)) for b in cs] for a in cs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)
        M = pg.L2(pg.Interval(0, 1), tol=1e-5)
        Gk = pg.Functionals([M(lambda x, c=c: np.exp(-(x - c) ** 2 / s2)) for c in cs])
        A = Gk @ Gk.adjoint
        e, _, how = solvers.stated_error(A, sharp=True)
        true = np.linalg.norm(A.matrix() - exact, 2)
        ref = functionals._refined(Gk, A.matrix())
    assert "plus its own first-order bound" in how and e == pytest.approx(ref[0] + ref[2]) and ref[2] > 0
    assert true <= e
    monkeypatch.setattr(functionals, "_refined", lambda G, Gr, f=functionals._refined: (lambda r: (r[0], r[1], 0.0))(
        f(G, Gr)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)
        assert solvers.stated_error(Gk @ Gk.adjoint, sharp=True)[0] == pytest.approx(ref[0])


# ---------------------------------------------------------------------------------------------- item 5
def test_compression_refuses_a_direction_without_a_digit(monkeypatch):
    """280 ray kernels in Sobolev(Ball, 2, 0.5) and 20 near-duplicates (ends rotated by 1e-3), compressed at 1e-2: the
    worst direction's error against exact energies is about 4 (no digit), and the default A.inv refuses. Control: the
    compression's statement removed (rounding only), A.inv gives the inverse: a false acceptance."""
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import ladder
    rng = np.random.default_rng(0)
    base = []
    for _ in range(280):
        p = ladder._unit(rng.normal(size=3))
        v = rng.normal(size=3)
        base.append((p, ladder._unit(v - (v @ p) * p), rng.uniform(np.radians(30), np.radians(150))))
    ms = [ladder._curved_ray(p, u, D, 8)[0] for p, u, D in base]
    for j in range(20):
        p, u, D = base[j]
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        Rm = np.eye(3) + math.sin(1e-3) * Kx + (1 - math.cos(1e-3)) * Kx @ Kx
        ms.append(ladder._curved_ray(Rm @ p, Rm @ u, D, 8)[0])
    P = pg.Euclidean(len(ms))
    Ge = pg.Functionals(pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5).riesz(ms), into=P)
    Ae = (Ge @ Ge.adjoint).matrix()
    with pg.settings.using(dense_gram="compressed", compression=1e-2):
        Gc = pg.Functionals(pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5).riesz(ms), into=P)
        Ac = Gc @ Gc.adjoint
        Mc = Ac.matrix()
        assert np.linalg.norm(np.linalg.solve(Mc, Ae) - np.eye(len(ms)), 2) >= 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PrecisionWarning)
            with pytest.raises(Refused):
                Ac.inv.factorisation()
            monkeypatch.setattr(Ac, "_stated_error", lambda: None)
            monkeypatch.setattr(Ac, "_sharper_error", lambda: None)
            Ac._inv = None
            Ac.inv.factorisation()                                   # the planted defect: given
