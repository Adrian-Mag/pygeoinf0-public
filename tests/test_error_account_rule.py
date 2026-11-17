"""Plan 054, the author's general rule: a result is EXACT (or an inverse given) only if the sensitivity of the result to
its input errors times the stated size of those errors is within the tolerance; otherwise it is downgraded or refused,
with the choices named. Each test is checked against an exact answer and has a planted-defect control.

    optimisers/dual/exactness.py     spectral BGP: EXACT needs E <= TOL max(1, |value|), E from the multiplier (first order) and
                            re-solves under perturbations of the stated size; the bracket widened by E otherwise
    solvers.direct          A.inv: the sharpest stated error of A (an a-posteriori estimate for realised kernels:
                            operators/refinement._refined) over |lambda|_min; refused from 1 (no correct digit)
    convex/subspaces        Span: its dimension decided on normalised generators (scale-free)
References: P2's Gaussian-kernel problems (mpmath at 60 and 90 digits, mp_bgp.sup of plan 054 P2, agreeing to 1e-54;
values below), and the erf closed form of Gaussian kernels in L2[0, 1] at 40 digits."""
import math
import warnings

import mpmath as mp
import numpy as np
import pytest
from scipy.stats import chi2  # noqa: F401  (pg.update's acceptance)

import pygeoinf0 as pg
from pygeoinf0.core.precision import TOL
from pygeoinf0.optimisers.dual import exactness
from pygeoinf0.core.errors import PrecisionWarning, RealisationWarning, Refused
from pygeoinf0.operators import functionals
from pygeoinf0.convex import subspaces

# exact [lower, upper] of the target's mean over the feasible set, for the given doubles (P2's m_rcond.prob, n = 200)
EXACT = {(0.15, 20, "block"): (0.33138758348552199706, 2.0115589951998589706),
         (0.1, 10, "estimable"): (7.7550870868192278706, 7.7571059694146760624),
         (0.1, 10, "block"): (0.34700931738459572795, 1.9995045941384441624)}


def _problem(width, nd, tk, n=200, sig=1e-3, seed=0, rounding=None):
    """rounding: a seed for a componentwise relative change of the kernel matrix and the datum by up to eps / 2, the
    last bit another linear-algebra library may set otherwise (tests/test_error_account_blas.py); t is built first."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    A = np.exp(-(x[None] - np.linspace(0.05, 0.95, nd)[:, None]) ** 2 / (2 * width ** 2))
    t = ((x > 0.4) & (x < 0.6)) / np.sum((x > 0.4) & (x < 0.6)) if tk == "block" else A.T @ rng.standard_normal(nd) / nd
    dv = A @ (1 + 0.3 * np.sin(5 * x)) + sig * rng.standard_normal(nd)
    if rounding is not None:
        q = np.random.default_rng(1000 + rounding)
        A = A * (1 + 1.1e-16 * q.uniform(-1, 1, A.shape))
        dv = dv * (1 + 1.1e-16 * q.uniform(-1, 1, nd))
    M, D = pg.Euclidean(n), pg.Euclidean(nd)
    G, T = pg.Matrix(A, M, D), pg.Matrix(t[None], M, pg.Euclidean(1))
    d = D(dv)
    L = pg.Additive(G, pg.GaussianMeasure(D, sig ** 2 * pg.Identity(D)))
    S = pg.NormBall(M, 1.5 * np.linalg.norm(1 + 0.3 * np.sin(5 * x)))
    return pg.push(pg.update(S, L.acceptance(0.05), d), T)


@pytest.mark.parametrize("case", [(0.15, 20, "block"), (0.1, 10, "estimable")])
def test_spectral_exact_needs_its_digits(case):
    """P2's Gaussian kernels: a bound whose value is more than TOL from the exact one is not EXACT and says so, an EXACT
    one is within TOL, and every bracket (widened) holds the exact value. Which bounds miss by more than TOL is the
    machine's rounding (europa, OpenBLAS: 25 to 80 x TOL in both cases; the Mac, Accelerate: in the first only), so the
    instance and the control are in test_former_account_misses_on_p2."""
    b = _problem(*case).bounds(0)
    for x, ref in zip(b, EXACT[case]):
        if abs(x.value - ref) > TOL * max(1.0, abs(x.value)):
            assert x.status != "EXACT" and "not EXACT" in x.note
        # what the rule guarantees whatever the rounding: the stated error is 25 to 800 x TOL, so no end is EXACT
        assert x.error > 10 * TOL * max(1.0, abs(x.value)) and x.status != "EXACT"
        assert x.lower - 1e-3 * (x.upper - x.lower) <= ref <= x.upper + 1e-3 * (x.upper - x.lower)


def _wrong(x, ref, k):
    """EXACT with an error above k TOL, or a bracket that misses the exact end by more than TOL."""
    t = TOL * max(1.0, abs(ref))
    return (x.status == "EXACT" and abs(x.value - ref) > k * t) or not (x.lower - t <= ref <= x.upper + t)


def test_former_account_misses_on_p2(monkeypatch):
    """Across P2's two problems some bound is more than TOL from the exact one (the rule is exercised on this machine) and
    none is wrong. Control: no stated input error (C = 0, the former account) gives a wrong bound: EXACT with an error
    above 5 TOL (europa) or a bracket that misses the exact end (the Mac: the first problem's upper end, 3e-8 below it)."""
    cases = [(0.15, 20, "block"), (0.1, 10, "estimable")]
    pairs = [(x, ref) for c in cases for x, ref in zip(_problem(*c).bounds(0), EXACT[c])]
    assert any(abs(x.value - ref) > TOL * max(1.0, abs(x.value)) for x, ref in pairs)
    assert not any(_wrong(x, ref, 1) for x, ref in pairs)
    monkeypatch.setattr(exactness, "C", 0.0)
    assert any(_wrong(x, ref, 5) for c in cases for x, ref in zip(_problem(*c).bounds(0), EXACT[c]))


def test_well_conditioned_bounds_stay_exact():
    """cond(GG*) 2.6e3: EXACT, within TOL of the exact value, and the rule leaves it (E about 5e-4 TOL)."""
    lo, hi = EXACT[(0.1, 10, "block")]
    b = _problem(0.1, 10, "block").bounds(0)
    for x, ref in zip(b, (lo, hi)):
        assert x.status == "EXACT" and abs(x.value - ref) <= TOL * max(1.0, abs(ref))


def test_stated_errors_are_what_separates():
    """The same first-order estimate on a decomposition that states blocks (exact zeros between them) and on the same
    numbers stated as one dense block: smaller where the stated error is smaller. Not a rule about structure: E1 only
    reads the stated error per block."""
    rng = np.random.default_rng(3)
    lam = np.exp(rng.uniform(-20, 5, 40))

    class Dec:
        def __init__(self, blocks):
            self.lam = lam
            self._lam_b = [lam[a:b] for a, b in zip(blocks[:-1], blocks[1:])]
            self.offsets = np.asarray(blocks)

        def _coef(self, e):
            return np.asarray(e, float)

        def positive(self, rcond):
            return self.lam > 0

        def above_floor(self):
            return self.lam > 0

        def zero_count(self, rcond):
            return 0, 0, len(self.lam)

        def describe(self):
            return "test"
    from pygeoinf0.optimisers.dual import SpectralBall
    delta = rng.normal(size=40)
    x = rng.normal(size=(40, 1))
    E = []
    for blocks in (list(range(41)), [0, 40]):
        sp = SpectralBall(Dec(blocks), delta, 1.0, 3.0)
        E.append(exactness.first_order(sp, x, np.array([1.0]), np.array([1.0]))[0])
    assert E[0] < E[1] / 20


def _gauss_gram(k, tol, s2=0.02):
    cs = np.linspace(0.1, 0.9, k)
    M = pg.L2(pg.Interval(0, 1), tol=tol)
    G = pg.Functionals([M(lambda x, c=c: np.exp(-(x - c) ** 2 / s2)) for c in cs])
    return G @ G.adjoint, cs


def _exact_gram(cs, s2=0.02):
    mp.mp.dps = 40

    def g(a, b):
        a, b = mp.mpf(a), mp.mpf(b)
        m, q, r = (a + b) / 2, (a - b) ** 2 / (2 * s2), mp.sqrt(2 / mp.mpf(s2))
        return mp.e ** (-q) * mp.sqrt(mp.pi * s2 / 2) * (mp.erf(r * (1 - m)) - mp.erf(-r * m)) / 2
    return mp.matrix([[g(a, b) for b in cs] for a in cs])


def _worst(A, Aex):
    """max over directions of the relative error of A~^-1 y against A^-1 y: ||A~^-1 A - I||_2, in 40 digits."""
    Z = mp.inverse(mp.matrix(A.matrix().tolist())) * Aex - mp.eye(Aex.rows)
    return float(max(mp.svd_r(Z, compute_uv=False)))


def test_inverse_refuses_only_where_no_digit_is_left(monkeypatch):
    """The default A.inv of realised Gaussian kernels: 12 kernels at tol 1e-5 (worst-direction error 5.5e-5, measured
    for y 4.2e-6) is given, stating an a-posteriori ratio 2.5e-4; 16 kernels at tol 1e-4 (worst direction 4.5: no
    digit) is refused. Control: the first-order bound (no a-posteriori estimate) refuses the first, ratio 11."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)
        warnings.simplefilter("ignore", PrecisionWarning)
        A, cs = _gauss_gram(12, 1e-5)
        Ainv = A.inv
        y = np.random.default_rng(1).standard_normal(len(cs))
        x = Ainv(pg.Euclidean(len(cs))(y)).values
        st = Ainv.claims.find("inherited error").statement
        assert "a posteriori" in st
        ratio = float(st.split("at most ")[1].split(",")[0])
        assert _worst(A, _exact_gram(cs)) <= ratio < 1
        xr = np.array(mp.lu_solve(_exact_gram(cs), mp.matrix(y.tolist())).tolist(), float).ravel()
        assert np.linalg.norm(x - xr) / np.linalg.norm(xr) < 1e-5
        B, cb = _gauss_gram(16, 1e-4)
        assert _worst(B, _exact_gram(cb)) > 1
        with pytest.raises(Refused, match="no correct digit"):
            B.inv.factorisation()
        monkeypatch.setattr(functionals, "_refined", lambda G, Gr: None)        # the planted defect: the former bound
        A2, _ = _gauss_gram(12, 1e-5)
        with pytest.raises(Refused, match="no correct digit"):
            A2.inv.factorisation()


def test_span_dimension_is_scale_free(monkeypatch):
    """span{v1, 1e-9 v2} has dimension 2 (its projection exact), span{v1, v1 + 1e-9 v2} dimension 1 (a combination at
    the floor), and with scale= a generator at the rounding of its scale is dropped. Control: the former cut, eps k
    times the largest Gram eigenvalue, gives dimension 1 for span{v1, 1e-9 v2}."""
    X = pg.Euclidean(4)
    e = np.eye(4)
    x = X(np.array([0.3, -1.2, 0.7, 2.0]))
    S = pg.Span([X(e[0]), X(1e-9 * e[1])])
    assert S.dim == 2 and S.dropped == 0
    assert np.abs(S.project(x).values - [0.3, -1.2, 0.0, 0.0]).max() < 1e-15
    D = pg.Span([X(e[0]), X(e[0] + 1e-9 * e[1])])
    assert D.dim == 1 and "1 correlation eigenvalue(s)" in D.claims.find("dimension").statement
    Z = pg.Span([X(e[0]), X(1e-17 * e[2])], scale=1.0)
    assert Z.dim == 1 and Z.zero_generators == 1 and "1 zero generator(s)" in Z.claims.find("dimension").statement
    lam = np.linalg.eigvalsh(S.gram)                                      # the planted defect: the former cut
    assert int(np.sum(lam > subspaces.EPS * 2 * lam.max())) == 1
    assert math.isclose(float(S.project(X(e[1])).values[1]), 1.0, rel_tol=1e-15)
