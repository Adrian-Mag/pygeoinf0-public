"""Plan 054, the general rule, round 2 (the author's rulings of 2026-11-06): the rule (EXACT, or an inverse given, only if
the sensitivity to the stated input errors times their stated size is within the tolerance) in the reduce route and at
oCSOLA's endpoints, the widened bracket's safety factor, and a-posteriori input errors for truncated covariances and
compressed energies. Each test is checked against an exact answer and has a planted-defect control.

    optimisers/dual/exactness_reduce.py   reduce route: E from the reduction's backward error (first order, re-solves, the lifted
                                 problem where directions were dropped); oCSOLA's ends moved outwards and warned
    optimisers/dual/exactness.WIDEN       the widened bracket: 2 x E
    probability/covariance.py    Covariance.sharper_error: the covariance with more of it known, ||C~ - C'|| + ||R'||
    operators/functionals.py     compressed energies compressed again at compression / 100
References: mpmath at 60 digits for the given doubles (the fix round's near-dependent problems; the ball-slab closed
form for oCSOLA), and operators formed directly in double where their rounding is far below the errors measured."""
import math
import sys
import warnings

import mpmath as mp
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import solvers
from pygeoinf0.core.precision import TOL
from pygeoinf0.optimisers.dual import exactness_reduce
from pygeoinf0.convex.feasible import FeasibleSet
from pygeoinf0.core.errors import PrecisionWarning, RealisationWarning, Refused
from pygeoinf0.operators import functionals
from pygeoinf0.convex.subspaces import Zero

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_error_account_cuts import near_parallel  # noqa: E402

# exact values for the given doubles (mpmath, 60 digits; /disks/data/tmp/plan54-rule2-scratch/m_reduce.py)
REDUCE = {("estimable", 1e-8, 1, "ball_ball"): 0.3470921046943879,
          ("estimable", 1e-6, 0, "none_ball"): 0.13312163025266943,
          ("generic", 1e-3, 0, "ball_ball"): 2.3073897533650523,
          ("generic", 1e-7, 3, "ball_exact"): 2.744146984586715}


def _reduce_problem(which, delta, seed, solver):
    """The fix round's data g1..g5 in R^12 with g5 = g4 + delta h, through the reduce route (decomposition=False)."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((5, 12))
    h = rng.standard_normal(12)
    G[4] = G[3] + delta * h
    mt = rng.standard_normal(12)
    mt *= 0.5 / np.linalg.norm(mt)
    d0 = G @ mt
    t = G[4].copy() if which == "estimable" else rng.standard_normal(12)
    X, D = pg.Euclidean(12), pg.Euclidean(5)
    dn = d0 + 0.03 * rng.standard_normal(5) / math.sqrt(5)
    Gop = pg.Matrix(G, X, D)
    K = {"ball_exact": lambda: FeasibleSet(pg.NormBall(X, 1.0), Gop, Zero(D), D(d0), decomposition=False),
         "none_ball": lambda: FeasibleSet(None, Gop, pg.NormBall(D, 0.1), D(dn), decomposition=False),
         "ball_ball": lambda: FeasibleSet(pg.NormBall(X, 1.0), Gop, pg.NormBall(D, 0.1), D(dn), decomposition=False)}
    return K[solver]().bound(X(t))


@pytest.mark.parametrize("case", [("estimable", 1e-8, 1, "ball_ball"), ("estimable", 1e-6, 0, "none_ball"),
                                  ("generic", 1e-7, 3, "ball_exact")])
def test_reduce_route_exact_needs_its_digits(case, monkeypatch):
    """Near-dependent data: the reduce route gave EXACT 13 and 7700 x TOL from the exact value, and a bracket that did
    not hold it (0.58 off). With the rule: not EXACT, and the widened bracket holds the exact value. Control: the rule off
    (judge -> 0) gives the former results."""
    ref = REDUCE[case]
    b = _reduce_problem(*case)
    assert b.status != "EXACT" and "not EXACT" in b.note
    assert b.lower <= ref <= b.upper
    monkeypatch.setattr(exactness_reduce, "judge", lambda *a, **k: 0.0)
    b0 = _reduce_problem(*case)
    assert abs(b0.value - ref) > 10 * TOL * max(1.0, abs(ref))
    assert b0.status == "EXACT" or not (b0.lower <= ref <= b0.upper)


def test_reduce_route_well_conditioned_stays_exact():
    """delta = 1e-3, a generic target: EXACT and within TOL of the exact value with the rule (error 2e-15)."""
    ref = REDUCE[("generic", 1e-3, 0, "ball_ball")]
    b = _reduce_problem("generic", 1e-3, 0, "ball_ball")
    assert b.status == "EXACT" and abs(b.value - ref) <= TOL * max(1.0, abs(ref))


def _ball_slab_mp(a, b, s, w, r):
    """sup <a, m> over ||m|| <= r, |s - <b, m>| <= w for the given double vectors, in mpmath (closed form)."""
    with mp.workdps(60):
        a, b = [mp.mpf(float(x)) for x in a], [mp.mpf(float(x)) for x in b]
        s, w, r = mp.mpf(float(s)), mp.mpf(float(w)), mp.mpf(float(r))
        na2, nb2, c = mp.fsum(x * x for x in a), mp.fsum(x * x for x in b), mp.fsum(x * y for x, y in zip(a, b))
        t0 = r * c / mp.sqrt(na2)
        if s - w <= t0 <= s + w:
            return float(r * mp.sqrt(na2))
        h = s + w if t0 > s + w else s - w
        return float(c * h / nb2 + mp.sqrt(r * r - h * h / nb2) * mp.sqrt(na2 - c * c / nb2))


def _ocsola(theta):
    from scipy.stats import norm
    p = near_parallel(theta, 0)
    xv, A, t = p["X"].matrix()[0], p["L"].effective[0].matrix(), p["T"].matrix()[0]
    s, w, r = float(xv @ p["d"].values), norm.ppf(0.975) * 1e-3 * np.linalg.norm(xv), p["S"].radius
    ref = (-_ball_slab_mp(-t, A.T @ xv, s, w, r), _ball_slab_mp(t, A.T @ xv, s, w, r))
    return (lambda: pg.inference.ocsola(p["S"], p["L"], p["T"], p["d"], X=p["X"])), ref


def test_ocsola_ends_need_their_digits(monkeypatch):
    """A target at 1e-7 to r~: both ends up to 24 x TOL from the exact interval (by library: 2e-2 TOL on another
    OpenBLAS kernel). With the rule: a PrecisionWarning and the ends moved outwards, holding the exact interval.
    Control: the rule off gives the computed ends silently, more than 10 TOL inside the rule's."""
    f, ref = _ocsola(1e-7)
    with pytest.warns(PrecisionWarning, match="oCSOLA: the upper end moves"):
        lo, hi = f()
    assert lo <= ref[0] and hi >= ref[1]
    monkeypatch.setattr(exactness_reduce, "judge", lambda *a, **k: 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        lo0, hi0 = f()
    assert min(lo0 - lo, hi - hi0) > 10 * TOL * max(1.0, abs(ref[1]))


def test_ocsola_well_separated_is_left_alone():
    """theta = 0.1: no warning, the ends within 2e-5 TOL of the exact ones."""
    f, ref = _ocsola(0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        lo, hi = f()
    assert abs(lo - ref[0]) <= TOL * max(1.0, abs(ref[0])) and abs(hi - ref[1]) <= TOL * max(1.0, abs(ref[1]))


def test_widened_bracket_has_the_safety_factor(monkeypatch):
    """P2's Gaussian kernels (0.1, 10, estimable), the lower end: widened by E alone it missed the exact value by 3.4e-11
    on europa (OpenBLAS; 4.4e-3 TOL; round 1); widened by 2 E it holds it. Whether E alone misses is the machine's rounding
    (on the Mac, Accelerate, the value is 2.5e-10 from the exact one with E = 9e-7, and the end has no upper side), so the
    control checks the factor itself: WIDEN = 1 halves the widening about the same value."""
    from pygeoinf0.optimisers.dual import exactness
    from test_error_account_rule import EXACT, _problem
    lo = EXACT[(0.1, 10, "estimable")][0]
    b = _problem(0.1, 10, "estimable").bounds(0)[0]
    assert b.status != "EXACT" and b.lower <= lo <= b.upper
    # under 10 roundings of the last bit (what another library may give): never EXACT, and the bracket holds the end
    bs = [_problem(0.1, 10, "estimable", rounding=k).bounds(0)[0] for k in range(10)]
    assert all(x.status != "EXACT" and x.lower <= lo <= x.upper for x in bs)
    monkeypatch.setattr(exactness, "WIDEN", 1.0)
    b1 = _problem(0.1, 10, "estimable").bounds(0)[0]
    assert b1.value == b.value and math.isclose(b.value - b.lower, 2 * (b1.value - b1.lower), rel_tol=1e-3)


# ------------------------------------------------------------------------------------------------ truncation
def _spectral(k, sigma2, refine=True, seed=1):
    """S~ = G C_k G* + sigma^2 I on R^60 with C the spectral covariance lam_j = 1/j^2 on R^400 (P3's case), given a
    refinement with 4 k eigenpairs (400: exact), and the exact S."""
    rng = np.random.default_rng(seed)
    N, n = 400, 60
    Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
    lam = 1.0 / (1.0 + np.arange(N)) ** 2
    E, D = pg.Euclidean(N), pg.Euclidean(n)
    Gm = rng.standard_normal((n, N)) / np.sqrt(N)

    def cov(k):
        return pg.Covariance.spectral(lam[:k], pg.VectorFamily(E, Q[:, :k].T), tail=float(lam[k:].sum()),
                                      tail_max=float(lam[k]) if k < N else None,
                                      refine=(lambda: cov(min(4 * k, N))) if (refine and k < N) else None)
    S = cov(k).congruence(pg.Matrix(Gm, domain=E, codomain=D)) + sigma2 * pg.Identity(D)
    return S, Gm @ (Q * lam) @ Q.T @ Gm.T + sigma2 * np.eye(n), D, rng.standard_normal(n)


def _worst(S, Sex):
    return float(np.linalg.norm(np.linalg.solve(S.matrix(), Sex) - np.eye(len(Sex)), 2))


def test_truncated_covariance_a_posteriori_gives_the_inverse():
    """k = 100, sigma^2 = 1e-4: first order states 1.78 (refused), a posteriori 0.444 = the true ||S~ - S|| / lam_min;
    the worst direction's error is 0.27, so the default A.inv gives it and its solve error is within the ratio.
    Control: the same covariance without a refinement is refused."""
    S, Sex, D, y = _spectral(100, 1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        x = S.inv(D(y)).values
    xr = np.linalg.solve(Sex, y)
    e, status, how = solvers.stated_error(S, sharp=True)
    small = float(np.abs(np.linalg.eigvalsh(S.matrix())).min())
    true = float(np.linalg.norm(S.matrix() - Sex, 2))
    assert "a posteriori" in how and true <= e * (1 + 1e-9) and e < 1.01 * true and e / small < 1
    assert np.linalg.norm(x - xr) / np.linalg.norm(xr) <= e / small and _worst(S, Sex) < 1
    S0, _, _, _ = _spectral(100, 1e-4, refine=False)
    with pytest.raises(Refused, match="no correct digit"):
        S0.inv.factorisation()


def test_truncated_covariance_without_a_digit_stays_refused():
    """k = 20, sigma^2 = 1e-4: a posteriori 7.7 (true 5.0), the worst direction's error 4.1: refused (no false
    acceptance). Planted defect: the a-posteriori bound divided by 10 would accept it."""
    S, Sex, D, y = _spectral(20, 1e-4)
    with pytest.raises(Refused, match="no correct digit"):
        S.inv.factorisation()
    e = solvers.stated_error(S, sharp=True)[0]
    small = float(np.abs(np.linalg.eigvalsh(S.matrix())).min())
    assert _worst(S, Sex) >= 1 and e / small >= 1 and (e / 10) / small < 1


def test_interval_bessel_potential_a_posteriori():
    """(1 + L/16)^-1 on [0, 1] with 32 Galerkin modes through 12 Gaussian kernels, sigma^2 = 1e-6: first order 6.1
    (refused), a posteriori 0.40 (refinement: 128 modes); the worst direction's error 1e-4 against 320 modes. Control:
    without the refinement it is refused."""
    from pygeoinf0.operators.unbounded import SturmLiouville
    op = SturmLiouville("[0, 1]", 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)
        C = op.bessel_potential(4.0, 2.0, modes=32)
        X = C.domain
        G = pg.Functionals([X(lambda x, c=c: np.exp(-(x - c) ** 2 / 0.005)) for c in np.linspace(0.1, 0.9, 12)])
        S = C.congruence(G) + 1e-6 * pg.Identity(G.codomain)
        Sex = op.bessel_potential(4.0, 2.0, modes=320, space=X).congruence(G).matrix() + 1e-6 * np.eye(12)
        S.inv.factorisation()                                            # given
        assert _worst(S, Sex) < 1e-2
        C._refine = None                                                 # the control: no refinement
        S0 = C.congruence(G) + 1e-6 * pg.Identity(G.codomain)
        with pytest.raises(Refused, match="no correct digit"):
            S0.inv.factorisation()


# ------------------------------------------------------------------------------------------------ compression
def test_compressed_energies_a_posteriori(monkeypatch):
    """settings.compression = 1e-2 (ladder L3, 300 segments): the first estimate states 1.52 (refused), the a-posteriori
    one (energies compressed again at 1e-4) 0.38, the true ||A~ - A|| / lam_min 0.36 against exact energies; the worst
    direction's error 0.053: given. Control: without the re-compression it is refused."""
    import ladder
    P = ladder.build("L3", 300, segments=8)
    H = pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5)
    Ae = (pg.Functionals(H.riesz(P.measures), into=P.D) @ pg.Functionals(H.riesz(P.measures), into=P.D).adjoint).matrix()

    def gram():
        Hc = pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5)
        Gc = pg.Functionals(Hc.riesz(P.measures), into=P.D)
        return Gc @ Gc.adjoint
    with pg.settings.using(dense_gram="compressed", compression=1e-2), warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        A = gram()
        A.inv.factorisation()                                            # given
        e = solvers.stated_error(A, sharp=True)[0]
        M = A.matrix()
        assert np.linalg.norm(M - Ae, 2) <= e and _worst(A, Ae) < 1
        monkeypatch.setattr(functionals, "_recompressed", lambda *a: None)
        with pytest.raises(Refused, match="no correct digit"):
            gram().inv.factorisation()
