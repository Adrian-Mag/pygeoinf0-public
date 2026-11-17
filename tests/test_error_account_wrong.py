"""Plan 054, the inventory rows that were wrong (not merely unstated), fixed or refused with the choices named (the author,
2026-11-06: "Can we know when it's terribly wrong and then throw an error? ... Why can it not be fixed?"; "These rules
must be generally true"). Each test is checked against an exact answer and has a planted-defect control (the former
behaviour, put back).

    probability/posterior_covariance.py   S2/P2: the dense posterior mean refined with the operators' residual, its
                                          error stated, warned of below 9 digits, refused at none
    solvers.solve_damping                 S13: a discrepancy root outside [eps n max lam, 1e20 max lam] refused (it was
                                          the bracket's end, claimed "solved exactly")
    functions/families/smooth.py          Q31: the Gaussian's integral on a circle by i0e (it was inf, nan from x ~ 713)
    quadrature.graded (was measures/weights.nodes_on) Q7: the innermost graded cell by Gauss-Jacobi (it was dropped)
    inference/methods.pli                 I17: the posterior's stated variance errors delivered
    realisations/rkhs/hmatrix.py               G3: an H-matrix built looser than asked is rebuilt; claims state its own
    spaces/complex.py                       P7: a complex metric checked as the real one (rtol 1e-12)
    core/adaptive.py                      Q6: an unresolved weight-adapted rule warns
References: mpmath at 60 digits for the given doubles; closed forms; exact energies."""
import math
import sys
import warnings

import mpmath as mp
import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.core import adaptive
from pygeoinf0 import quadrature
from pygeoinf0.core.errors import PrecisionWarning, RealisationWarning, Refused
from pygeoinf0.probability import posterior_covariance as pc
from pygeoinf0.solvers import solve_damping


# ------------------------------------------------------------------------------------------- S2/P2: dense mean
def _dense(smin, seed, a, n=30, m=60, sigma=1e-2):
    """m_mean.py's problem (G = U diag(s) V^T, 30 x 60), prior N(0, a^2 I), noise N(0, sigma^2 I): (posterior, exact
    mean)."""
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    V = np.linalg.qr(rng.standard_normal((m, n)))[0]
    G = (U * np.geomspace(1.0, smin, n)) @ V.T
    d = G @ rng.standard_normal(m) + sigma * rng.standard_normal(n)
    mp.mp.dps = 60
    Gm = mp.matrix(G.tolist())
    A = a ** 2 * Gm * Gm.T + mp.eye(n) * mp.mpf(sigma) ** 2
    exact = np.array([float(x) for x in a ** 2 * Gm.T * mp.lu_solve(A, mp.matrix(d.tolist()))])
    M, D = pg.Euclidean(m), pg.Euclidean(n, covariance=sigma ** 2)
    prior = pg.GaussianMeasure(M, a ** 2 * pg.Identity(M))
    return (lambda: pg.update(prior, pg.Additive(pg.Matrix(G, M, D), D.standard_gaussian()), D(d))), exact


def test_dense_posterior_mean_is_refined_and_its_error_stated(monkeypatch):
    post, exact = _dense(1e-5, 0, 1e6)                       # cond(S) 1e10, just below P1's warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        p = post()
    x = p.mean.values
    err = np.linalg.norm(x - exact)
    assert err <= 1e-10 * np.linalg.norm(exact) and err <= p.mean_error
    monkeypatch.setattr(pc, "REFINE", 0)                     # control: the former single solve
    q = post()
    err0 = np.linalg.norm(q.mean.values - exact)
    assert err0 > 1e-9 * np.linalg.norm(exact) and err0 > q.mean_error


def test_dense_posterior_mean_warns_below_nine_digits():
    post, exact = _dense(1e-7, 0, 1e6)                       # cond(S) 1e14: the refined mean holds ~8 digits
    with pytest.warns(PrecisionWarning, match="posterior mean"):
        p = post()
    assert np.linalg.norm(p.mean.values - exact) <= p.mean_error


def test_the_mean_rule():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pc.judge_mean("m", 1.0, 1e-12)                       # nine digits: nothing to say
    with pytest.warns(PrecisionWarning):
        pc.judge_mean("m", 1.0, 1e-6)
    with pytest.raises(Refused, match="no correct digit|fewer than 9"):
        pc.judge_mean("m", 1.0, 2.0)


# ------------------------------------------------------------------------------------------- S13: discrepancy alpha
def test_an_unreachable_discrepancy_target_is_refused():
    M, D = pg.Euclidean(3), pg.Euclidean(2)
    G = pg.Matrix(np.array([[1.0, 0, 0], [1.0, 0, 0]]), M, D)
    d = D(np.array([1.0, -1.0]))                             # in the null space of G G*: chi^2 = 2 for every alpha
    with pytest.raises(Refused, match="rounding floor"):
        pg.solvers.damping_for_misfit(G @ G.adjoint, d, target=1.0)
    lam, b2 = np.array([2.0, 0.0]), np.array([0.0, 2.0])
    former = 10 ** (math.log10(2.0) - 20)                    # control: what was returned, "solved exactly"
    assert float(np.sum((former / (lam + former)) ** 2 * b2)) == pytest.approx(2.0)


def test_a_discrepancy_root_inside_the_range_is_exact():
    lam, b2 = np.array([1.0, 1e-3, 1e-9]), np.array([1.0, 1.0, 1.0])
    alpha, chi2 = solve_damping(lam, b2, 0.5)
    assert chi2(alpha) == pytest.approx(0.5, rel=1e-12)
    with pytest.raises(Refused, match="rounding floor"):    # a root below eps n max lam has no correct digit
        solve_damping(np.array([1.0, 1e-30]), np.array([1e-6, 1.0]), 0.5)


# ------------------------------------------------------------------------------------------- Q31: circle Gaussian
@pytest.mark.parametrize("x", [1.0, 720.0, 800.0, 1e4, 1e8])
def test_gaussian_integral_on_a_circle(x):
    R = 1.5
    mp.mp.dps = 40
    exact = float(2 * mp.pi * R * mp.exp(-x) * mp.besseli(0, x))
    v, status, _ = F.on(pg.Circle(R)).gaussian(width=R / math.sqrt(x)).integral()
    assert status == "derived" and v == pytest.approx(exact, rel=1e-14)
    with np.errstate(over="ignore", invalid="ignore"):          # control: the former formula
        former = 2 * math.pi * R * math.exp(-x) * float(np.i0(x))
    assert x < 700 or not math.isfinite(former)


# ------------------------------------------------------------------------------------------- Q7: singular cell
@pytest.mark.parametrize("a", [0.9, 0.99])
def test_the_innermost_cell_at_a_singular_point_is_integrated(a, monkeypatch):
    mu = pg.LebesgueMeasure(pg.Interval(0.0, 0.5)).with_density(lambda x: np.abs(x) ** -a, singular_at=[0.0],
                                                                 exponent=a)
    X = pg.Continuous("[0, 1]")
    exact = 0.5 ** (1 - a) / (1 - a)
    mp.mp.dps = 30
    exact_e = float(mp.nsum(lambda k: mp.mpf(0.5) ** (k + 1 - a) / ((k + 1 - a) * mp.factorial(k)), [0, mp.inf]))
    m, e = mu.mass()
    assert m == pytest.approx(exact, rel=1e-13) and abs(m - exact) <= max(e, 1e-14 * exact)
    assert pg.Integrals(X, [mu])(X(np.exp)).values[0] == pytest.approx(exact_e, rel=1e-12)
    monkeypatch.setattr(quadrature, "_singular_cell", lambda *args: (np.zeros(0), np.zeros(0)))   # control: dropped
    m0, e0 = mu.mass()
    assert abs(m0 - exact) / exact > 0.5 * (quadrature.GRADING ** quadrature.LEVELS) ** (1 - a) and abs(m0 - exact) > e0


# ------------------------------------------------------------------------------------------- I17: PLI errors
def test_pli_through_a_problem_delivers_the_stated_variance_errors():
    n, m, sigma, a = 30, 60, 1e-2, 1e6
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    V = np.linalg.qr(rng.standard_normal((m, n)))[0]
    G = (U * np.geomspace(1.0, 1e-3, n)) @ V.T
    T = np.vstack([G.T @ rng.standard_normal(n), rng.standard_normal(m)])
    d = G @ rng.standard_normal(m) + sigma * rng.standard_normal(n)
    M, D, P = pg.Euclidean(m), pg.Euclidean(n, covariance=sigma ** 2), pg.Euclidean(2)
    prior = pg.GaussianMeasure(M, a ** 2 * pg.Identity(M))
    rel = pg.Additive(pg.Matrix(G, M, D), D.standard_gaussian())
    out = pg.inference.Problem(M=M, D=D, P=P, prior=prior, mdr=rel, mpr=pg.Matrix(T, M, P), datum=D(d)).run("pli")
    mp.mp.dps = 50
    Gm, Tm = mp.matrix(G.tolist()), mp.matrix(T.tolist())
    K = a ** 2 * Tm * Gm.T * mp.inverse(a ** 2 * Gm * Gm.T + mp.eye(n) * mp.mpf(sigma) ** 2)
    cov = a ** 2 * Tm * Tm.T - K * a ** 2 * Gm * Tm.T
    exact = np.array([float(cov[i, i]) for i in range(2)])
    assert out.variance_error.shape == (2,) and np.all(np.abs(out.variances - exact) <= out.variance_error)
    dec = pg.decompose(pg.Matrix(G, M, D), D.standard_gaussian())      # control: the same errors as the closed form's
    res = dec.posterior(pg.Matrix(T, M, P), D(d), a=a)
    assert np.allclose(out.variance_error, res.variance_error, rtol=1e-12)


# ------------------------------------------------------------------------------------------- G3: H-matrix cache
def test_an_hmatrix_built_looser_than_asked_is_rebuilt():
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import ladder
    P = ladder.build("L3", 300, segments=8)
    Ae = (lambda G: (G @ G.adjoint).matrix())(pg.Functionals(pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5)
                                                               .riesz(P.measures), into=P.D))
    with pg.settings.using(dense_gram="compressed", compression=1e-2):
        K = pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5).riesz(P.measures)

    def gram(**kw):
        with pg.settings.using(**kw):
            G = pg.Functionals(K, into=P.D)
            A = G @ G.adjoint
            return np.linalg.norm(A.matrix() - Ae) / np.linalg.norm(Ae), str(A.claims)
    loose, _ = gram(dense_gram="compressed", compression=1e-2)
    assert loose > 1e-7
    tight, claims = gram(dense_gram="compressed", compression=1e-8)
    assert tight < 1e-12 and "settings.compression = 1e-08 when the H-matrix was built" in claims
    exact, _ = gram()
    assert exact < 1e-14
    atoms = K.realisation.kx.atoms                              # control: a stale cache put back
    with pg.settings.using(dense_gram="compressed", compression=1e-2):
        atoms.hmatrix = atoms.self_energy = None
        gram(dense_gram="compressed", compression=1e-2)
    atoms.hmatrix.compression = 0.0
    stale, _ = gram(dense_gram="compressed", compression=1e-8)
    assert stale > 1e-7


# ------------------------------------------------------------------------------------------- P7: complex metric
def test_a_complex_metric_is_checked_as_a_real_one():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    W = Z @ Z.conj().T + 4 * np.eye(4)
    E = rng.standard_normal((4, 4))
    E = E - E.T
    pg.Euclidean(4, metric=W, field="C")                            # control: Hermitian, accepted
    for delta in (1e-11, 1e-8):
        with pytest.raises(ValueError, match="Hermitian"):
            pg.Euclidean(4, metric=W + delta * np.abs(W).max() * E, field="C")
        with pytest.raises(ValueError, match="symmetric"):
            pg.Euclidean(4, metric=(W + delta * np.abs(W).max() * E).real)


# ------------------------------------------------------------------------------------------- Q6: resolved flag
def test_an_unresolved_adapted_rule_warns():
    c, a = 0.3, 0.9

    def w(t):                                                   # an undeclared singular point
        return np.abs(t - c) ** -a
    exact = (c ** (1 - a) + (1 - c) ** (1 - a)) / (1 - a)
    with pytest.warns(RealisationWarning, match="not resolved"):
        x, W, ok = adaptive.adapted_panels([0.0, 1.0], 32, w)
    assert not ok and abs(W @ w(x) - exact) > 1e-3 * exact      # what the warning is about
    with warnings.catch_warnings():                             # control: a weight the rule resolves
        warnings.simplefilter("error", RealisationWarning)
        x, W, ok = adaptive.adapted_panels([0.0, 1.0], 32, lambda t: np.exp(-t))
    assert ok and W @ np.exp(-x) == pytest.approx(1 - math.exp(-1), rel=1e-13)
