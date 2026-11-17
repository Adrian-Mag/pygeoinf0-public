"""Plan 054, verification levels (the author's ruling of 2026-11-06: "there should be a choice of how accurate or safe one
wants to be, but the option should be there"): pg.settings.verification sets how hard an inverse checks the error it
inherits from its operator's inputs where their first-order statement leaves no correct digit. Each test is checked
against an exact answer and has a planted-defect control.

    settings.verification = 0   quick: nothing formed again; the inverse is given with a warning that it is not verified
    settings.verification = 1   one refinement (the default, as before)
    settings.verification = d   d nested refinements, each about 100 x tighter, the first-order statement at the deepest
    operators/refinement.py     kernels realised again at tol / 100^j; energies compressed again at compression / 100^d
    probability/covariance.py   Covariance.sharper_error: the refinement taken d times
References: the erf closed form of Gaussian kernels at 40 digits; exact covariances; exact energies."""
import sys
import warnings

import mpmath as mp
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import solvers
from pygeoinf0.core.errors import PrecisionWarning, RealisationWarning, Refused
from pygeoinf0.operators import refinement

S2 = 0.02


def _gauss(k, tol):
    """k Gaussian kernels exp(-(x - c)^2 / 0.02) in L2[0, 1] realised at tol: the Gram operator, the exact Gram (40
    digits, rounded) and the worst direction's error ||A~^-1 A - I||_2 at 40 digits (>= 1: no correct digit)."""
    mp.mp.dps = 40
    cs = np.linspace(0.1, 0.9, k)

    def g(a, b):
        a, b = mp.mpf(a), mp.mpf(b)
        mid, q, r = (a + b) / 2, (a - b) ** 2 / (2 * S2), mp.sqrt(2 / mp.mpf(S2))
        return mp.e ** (-q) * mp.sqrt(mp.pi * S2 / 2) * (mp.erf(r * (1 - mid)) - mp.erf(-r * mid)) / 2
    Aex = mp.matrix([[g(a, b) for b in cs] for a in cs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)
        M = pg.L2(pg.Interval(0, 1), tol=tol)
        G = pg.Functionals([M(lambda x, c=c: np.exp(-(x - c) ** 2 / S2)) for c in cs])
        A = G @ G.adjoint
        Am = A.matrix()
    worst = float(max(mp.svd_r(mp.inverse(mp.matrix(Am.tolist())) * Aex - mp.eye(k), compute_uv=False)))
    return A, np.array(Aex.tolist(), float), worst


def _sharp(A):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)
        return solvers.stated_error(A, sharp=True)


def test_the_setting():
    """An integer >= 0, default 1, scoped by settings.using, in the repr. Control: a name or a negative is refused."""
    assert pg.settings.verification == 1 and "verification=1" in repr(pg.settings)
    with pg.settings.using(verification=3):
        assert pg.settings.verification == 3
    assert pg.settings.verification == 1
    for bad in (-1, "quick", 1.5, True):
        with pytest.raises(ValueError):
            pg.settings.verification = bad
    assert pg.settings.verification == 1


def test_quick_gives_the_numbers_unverified_and_says_so():
    """16 kernels at tol 1e-3: the worst direction's error is 4.6 (no correct digit). Quick (0): nothing is realised
    again, the inverse is given with a PrecisionWarning, and its claim says it was NOT VERIFIED and may hold no digit,
    which is the case. Control: the default (1) refuses it, the refusal quick skips."""
    A, _, worst = _gauss(16, 1e-3)
    assert worst >= 1
    with pg.settings.using(verification=0):
        Ainv = A.inverse()
        with pytest.warns(PrecisionWarning, match="NOT VERIFIED"):
            Ainv.factorisation()
        claim = Ainv.claims.find("inherited error").statement
    assert "NOT VERIFIED" in claim and "may hold no correct digit" in claim and "_sharper_cache" not in A.__dict__
    with pytest.raises(Refused, match="no correct digit"):
        A.inverse().factorisation()


def test_quick_leaves_a_verified_statement_alone():
    """12 kernels at tol 1e-7: the first-order ratio is 0.033, so quick and the default give the same inverse and claim,
    without the not-verified words; the true ratio (exact Gram) is below it. Control: the first-order statement divided
    by 1e6 would be below the true error."""
    A, Aex, _ = _gauss(12, 1e-7)
    with pg.settings.using(verification=0):
        claim0 = A.inverse().claims.find("inherited error").statement
    claim1 = A.inverse().claims.find("inherited error").statement
    assert claim0 == claim1 and "NOT VERIFIED" not in claim0
    e = solvers.stated_error(A)[0]
    true = np.linalg.norm(A.matrix() - Aex, 2)
    assert true <= e and not true <= e / 1e6


def test_nested_levels_lift_a_false_refusal(monkeypatch):
    """12 kernels at tol 1e-3: the worst direction's error is 8.8e-2 (a correct digit). One refinement refuses it (its
    own first-order bound dominates); three nested refinements give it, the stated error a bound of the true one (exact
    Gram) within 1 %, and the claim names the level and the chain's differences, which fall. Control: the depth
    ignored (always one refinement) refuses it again."""
    A, Aex, worst = _gauss(12, 1e-3)
    true = np.linalg.norm(A.matrix() - Aex, 2)
    assert worst < 1
    with pytest.raises(Refused, match="no correct digit"):
        A.inverse().factorisation()
    with pg.settings.using(verification=3):
        Ainv = A.inverse()
        Ainv.factorisation()                                             # given
        e, _, how = _sharp(A)
        claim = Ainv.claims.find("inherited error").statement
    assert "verification level 3;" in claim and true <= e < 1.01 * true
    diffs = [float(x.split(" at tol")[0]) for x in how.split("||G^(j-1) - G^(j)|| = ")[1].split(";")[0].split(", ")]
    assert len(diffs) == 3 and diffs[0] > 100 * diffs[1] > 100 * diffs[2]
    monkeypatch.setattr(refinement, "depth", lambda: 1)
    A2, _, _ = _gauss(12, 1e-3)
    with pg.settings.using(verification=3), pytest.raises(Refused, match="no correct digit"):
        A2.inverse().factorisation()


def test_each_level_is_a_bound_that_tightens(monkeypatch):
    """12 kernels at tol 1e-4: the a-posteriori statements at levels 1, 2, 3 bound the true ||G~ - G|| (exact Gram) and
    fall towards it (level 3 within 2 %). Planted defect: a 'refinement' that returns the same Gram, claiming 1e6 x
    smaller kernel errors, states less than the true error: the bound check catches it."""
    A, Aex, _ = _gauss(12, 1e-4)
    true = np.linalg.norm(A.matrix() - Aex, 2)
    es = []
    for d in (1, 2, 3):
        with pg.settings.using(verification=d):
            es.append(_sharp(A)[0])
    assert true <= es[2] <= es[1] <= es[0] and es[2] < 1.02 * true
    same = lambda G, M, tol: (np.asarray(G.kernels.space.inner(G.kernels, G.kernels), float),  # noqa: E731
                              np.asarray(G._kerr, float) / 1e6, np.sqrt(np.diag(A.matrix())))
    monkeypatch.setattr(refinement, "_realise", same)
    B, _, _ = _gauss(12, 1e-4)
    with pg.settings.using(verification=1):
        assert _sharp(B)[0] < true


def test_a_level_too_deep_stops_and_says_why():
    """12 kernels at tol 1e-6, level 4 asked: the realisation at tol 1e-14 does not state errors 10 x smaller, so the
    chain stops at level 3 and the claim says so; the statement still bounds the true error. Control: level 3 asked
    reaches it without the note."""
    A, Aex, _ = _gauss(12, 1e-6)
    true = np.linalg.norm(A.matrix() - Aex, 2)
    with pg.settings.using(verification=4):
        e, _, how = _sharp(A)
    assert "verification level 3 of the 4 asked (the realisation at tol 1e-14 states errors not 10 x smaller)" in how
    assert true <= e
    with pg.settings.using(verification=3):
        assert "verification level 3;" in _sharp(A)[2]


# ------------------------------------------------------------------------------------------- truncated covariances
def _spectral(k, sigma2, seed=1, lie_at=None):
    """S~ = G C_k G* + sigma^2 I on R^60, C the spectral covariance lam_j = 1/j^2 on R^400 (round 2's case), refined to
    4 k eigenpairs (400: exact, nothing omitted); the exact S. lie_at: a planted refinement with that many eigenpairs
    that states nothing omitted."""
    rng = np.random.default_rng(seed)
    N, n = 400, 60
    Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
    lam = 1.0 / (1.0 + np.arange(N)) ** 2
    E, D = pg.Euclidean(N), pg.Euclidean(n)
    Gm = rng.standard_normal((n, N)) / np.sqrt(N)

    def cov(k, lie=False):
        if lie:
            return pg.Covariance.spectral(lam[:k], pg.VectorFamily(E, Q[:, :k].T))
        nxt = (lambda: cov(lie_at, True)) if lie_at and 4 * k >= lie_at else (lambda: cov(min(4 * k, N)))
        if k == N:
            return pg.Covariance.spectral(lam, pg.VectorFamily(E, Q.T))
        return pg.Covariance.spectral(lam[:k], pg.VectorFamily(E, Q[:, :k].T), tail=float(lam[k:].sum()),
                                      tail_max=float(lam[k]), refine=nxt)
    S = cov(k).congruence(pg.Matrix(Gm, domain=E, codomain=D)) + sigma2 * pg.Identity(D)
    return S, Gm @ (Q * lam) @ Q.T @ Gm.T + sigma2 * np.eye(n)


def test_truncated_covariance_levels():
    """k = 20, sigma^2 = 1e-3: level 1 (80 eigenpairs) states 1.5 x the true ||S~ - S||; level 2 reaches 320 and level 3
    the exact 400 (1280 capped), where the statement is the true error to rounding, and says that nothing more is
    omitted at the level asked. Control: a planted refinement at 80 that states nothing omitted is believed at level 2
    and states less than the true error: the check catches it."""
    S, Sex = _spectral(20, 1e-3)
    true = float(np.linalg.norm(S.matrix() - Sex, 2))
    es = {}
    for d in (1, 2, 3, 4):
        with pg.settings.using(verification=d):
            es[d] = solvers.stated_error(S, sharp=True)
    assert true <= es[1][0] and es[3][0] <= es[2][0] <= es[1][0]
    assert abs(es[3][0] - true) <= 1e-9 * true and "verification level 3," in es[3][2]
    assert "verification level 3 of the 4 asked (nothing omitted)" in es[4][2]
    P, Pex = _spectral(20, 1e-3, lie_at=80)
    with pg.settings.using(verification=2):
        assert solvers.stated_error(P, sharp=True)[0] < float(np.linalg.norm(P.matrix() - Pex, 2))


def test_interval_bessel_potential_false_refusal_lifted(monkeypatch):
    """(1 + L/16)^-1 on [0, 1], 16 Galerkin modes, 12 Gaussian kernels (width 0.05), sigma^2 = 1e-8: one refinement
    (64 modes) states 1.6 x lam_min and refuses; two (256 modes) state 0.12 and give it. Against 320 modes (its own
    omitted part 1.6e-5) the worst direction's error is 4e-3 (a digit, even with that reference error, 0.31 lam_min).
    Control: the depth ignored (always one) refuses."""
    from pygeoinf0.operators.unbounded import SturmLiouville
    op = SturmLiouville("[0, 1]", 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)
        C = op.bessel_potential(4.0, 2.0, modes=16)
        X = C.domain
        G = pg.Functionals([X(lambda x, c=c: np.exp(-(x - c) ** 2 / (2 * 0.05 ** 2))) for c in np.linspace(0.1, 0.9, 12)])
        S = C.congruence(G) + 1e-8 * pg.Identity(G.codomain)
        Cr = op.bessel_potential(4.0, 2.0, modes=320, space=X)
        Sex = Cr.congruence(G).matrix() + 1e-8 * np.eye(12)
        small = float(np.abs(np.linalg.eigvalsh(S.matrix())).min())
        assert np.linalg.norm(np.linalg.solve(S.matrix(), Sex) - np.eye(12), 2) + Cr.rest["norm"] / small < 1
        with pytest.raises(Refused, match="no correct digit"):
            S.inverse().factorisation()
        with pg.settings.using(verification=2):
            Ainv = S.inverse()
            Ainv.factorisation()                                          # given
            assert "verification level 2," in Ainv.claims.find("inherited error").statement
            monkeypatch.setattr(refinement, "depth", lambda: 1)
            with pytest.raises(Refused, match="no correct digit"):
                (C.congruence(G) + 1e-8 * pg.Identity(G.codomain)).inverse().factorisation()


# ------------------------------------------------------------------------------------------------ compression
def test_compressed_energies_levels(monkeypatch):
    """Ladder L3, 300 segments, compression 1e-2: the first estimate states 1.52 x lam_min. Quick gives it unverified;
    level 1 (compression 1e-4) states 0.38, level 2 (1e-6) 0.36, the true ||A~ - A|| / lam_min against exact energies to
    1 %, an estimate (the H-matrix's), not a bound. Control: the depth ignored states level 1's 5 % above the truth."""
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import ladder
    P = ladder.build("L3", 300, segments=8)
    H = pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5)
    Ae = (pg.Functionals(H.riesz(P.measures), into=P.D) @ pg.Functionals(H.riesz(P.measures), into=P.D).adjoint).matrix()
    with pg.settings.using(dense_gram="compressed", compression=1e-2), warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        Gc = pg.Functionals(pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5).riesz(P.measures), into=P.D)
        A = Gc @ Gc.adjoint
        true = np.linalg.norm(A.matrix() - Ae, 2)
        with pg.settings.using(verification=0):
            assert "NOT VERIFIED" in A.inverse().claims.find("inherited error").statement
        with pg.settings.using(verification=2):
            e2, _, how = solvers.stated_error(A, sharp=True)
        assert "an estimate" in how and "verification level 2" in how and "compressed again at 1e-06" in how
        assert abs(e2 - true) <= 0.01 * true
        monkeypatch.setattr(refinement, "depth", lambda: 1)
        with pg.settings.using(verification=2):
            e1 = solvers.stated_error(A, sharp=True)[0]
        assert e1 > 1.03 * true


def test_the_refusal_names_a_deeper_level(monkeypatch):
    """The author (2026-11-06): the inherited-error refusal names pg.settings.verification with deeper levels among its
    choices, next to the others, and its statement says so; following it gives the inverse (12 kernels at tol 1e-3,
    worst direction 8.8e-2: a false refusal at level 1, given at 3). Control: the other choices (the former ones) do
    not name it, so the new entry is what names it."""
    from pygeoinf0 import solvers
    A, _, worst = _gauss(12, 1e-3)
    with pytest.raises(Refused) as err:
        A.inverse().factorisation()
    named = [n for n in err.value.needs if "pg.settings.verification = 2 or 3" in n]
    assert len(named) == 1 and len(err.value.needs) == 4 and any("shift" in n for n in err.value.needs)
    assert "a deeper level, 2 or 3" in err.value.reason and worst < 1
    with pg.settings.using(verification=3):
        A.inverse().factorisation()                                     # the named choice gives it
    former = solvers.inherited_refusal("A", "x").needs[1:]    # control: the former choices
    assert not any("verification" in n for n in former)
