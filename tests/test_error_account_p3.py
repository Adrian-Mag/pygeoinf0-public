"""Plan 054 P3: stated approximations and the remainder, measured against exact answers, each with a planted-defect
control (the former statement, or a wrong one, shown to fail the same check).

    inherited error      A.inv states what it inherits from the stated error of A (a covariance applied as its known
                         part, compressed energies, realised kernels): every solve's relative error <= e / |lambda|_min
    sampling             a dense coordinate covariance is factored with nothing added (was S + 1e-14 tr(S)/n I)
    narrow features      a callable's realisation can lose a feature narrower than its sampling without a warning;
                         the element's claims say so, breaks= at the support resolve it
    randomised route     W_k <= W holds only to rounding in double, so the variance bounds are two-sided

References are exact: mpmath (variances for the stored W, the L2 error of a fit, a Gram in closed form), or the operator
meant formed directly in double where its rounding is far below the error measured."""
import sys
import warnings

import mpmath as mp
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import solvers
from pygeoinf0.core.errors import PrecisionWarning, RealisationWarning, Refused
from pygeoinf0.solvers.dataspace import partial

EPS = np.finfo(float).eps


def _ratio(Ainv):
    """The stated bound on the relative error inherited by a solve, from the claim 'inherited error'."""
    c = Ainv.claims.find("inherited error")
    assert c is not None, Ainv.claims
    return float(c.statement.split("at most ")[1].split(" ")[0].rstrip(","))


def _rel(x, xr):
    return float(np.linalg.norm(x - xr) / np.linalg.norm(xr))


# ------------------------------------------------------------------------------------------------ inherited error
def _truncated(k, sigma2, seed=1):
    """S~ = G C_k G* + sigma^2 I with C_k the known part of C = sum_j lam_j q_j q_j^T on R^400, and the exact S."""
    rng = np.random.default_rng(seed)
    N, n = 400, 60
    Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
    lam = 1.0 / (1.0 + np.arange(N)) ** 2
    E, D = pg.Euclidean(N), pg.Euclidean(n)
    Gm = rng.standard_normal((n, N)) / np.sqrt(N)
    C = pg.Covariance.spectral(lam[:k], pg.VectorFamily(E, Q[:, :k].T), tail=float(lam[k:].sum()),
                               tail_max=float(lam[k]))
    S = C.congruence(pg.Matrix(Gm, domain=E, codomain=D)) + sigma2 * pg.Identity(D)
    Sex = Gm @ (Q * lam) @ Q.T @ Gm.T + sigma2 * np.eye(n)
    return S, Sex, D, rng.standard_normal(n)


def test_inverse_states_what_a_truncated_covariance_costs():
    """The covariance is applied as its known part (omitted R >= 0, ||R|| <= m): S~^-1 y against the exact S^-1 y."""
    S, Sex, D, y = _truncated(50, 1e-4)
    Sinv = S.inverse(pg.Direct(refuse=False))
    x = Sinv(D(y)).values
    xr = np.linalg.solve(Sex, y)
    lam = np.linalg.eigvalsh(Sex)
    meas, r = _rel(x, xr), _ratio(Sinv)
    assert meas <= r                                                   # measured 0.39 <= 7.35
    assert meas > 1e6 * (lam.max() / lam.min()) * EPS                  # the rounding-only account (P1) is far off
    assert "omitted part" in Sinv.claims.find("inherited error").statement
    # planted defect: an understated error (e / 100) gives a bound the measured error breaks
    e, status, how = solvers.stated_error(S)
    small = float(np.abs(np.linalg.eigvalsh(S.matrix())).min())
    assert meas > (e / 100) / small


def test_no_correct_digit_inherited_is_warned():
    """k = 20, sigma^2 = 1e-6: the stated error exceeds the smallest eigenvalue; measured relative error 60."""
    S, Sex, D, y = _truncated(20, 1e-6)
    with pytest.warns(PrecisionWarning, match="no correct digit in the worst direction"):
        Sinv = S.inverse(pg.Direct(refuse=False))
        x = Sinv(D(y)).values
    assert _ratio(Sinv) >= 1 and _rel(x, np.linalg.solve(Sex, y)) > 1


def test_inverse_states_what_compressed_energies_cost():
    """settings.dense_gram = "compressed": the Gram from the H-matrix, against the exact-energy Gram."""
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import ladder
    P = ladder.build("L3", 300, segments=8)
    H = pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5)        # the ladder's kernels decay too fast to compress
    G = pg.Functionals(H.riesz(P.measures), into=P.D)
    A = G @ G.adjoint
    Ae = A.matrix()
    assert A.inv.claims.find("inherited error") is None             # exact energies: rounding only, nothing stated
    y = np.random.default_rng(2).standard_normal(P.n)
    with pg.settings.using(dense_gram="compressed", compression=1e-4):
        Hc = pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.5)
        K = Hc.riesz(P.measures)
        Gc = pg.Functionals(K, into=P.D)
        Ac = Gc @ Gc.adjoint
        Hm = K.realisation.kx.atoms.hmatrix
        assert Hm is not None and Hm.rel_error > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PrecisionWarning)
            Ainv = Ac.inverse(pg.Direct(refuse=False))
            x = Ainv(P.D(y)).values
        r = _ratio(Ainv)
        e = solvers.stated_error(Ac)[0]
    xr = np.linalg.solve(Ae, y)
    lam = np.linalg.eigvalsh(Ae)
    meas = _rel(x, xr)
    assert np.linalg.norm(Ac.matrix() - Ae, 2) <= e                 # the H-matrix estimate covers the Gram's error
    assert meas <= r + 10 * (lam.max() / lam.min()) * EPS
    assert meas > 10 * (lam.max() / lam.min()) * EPS                # beyond the rounding-only account
    small = float(np.abs(np.linalg.eigvalsh(Ac.matrix())).min())
    assert meas > (np.linalg.norm(Ac.matrix() - Ae, 2) / 1e3) / small  # planted defect: e / 1e3 is broken


def test_inverse_states_what_realised_kernels_cost():
    """Gaussian kernels in L2[0, 1] realised at tol 1e-5: their Gram against the closed form (erf) in mpmath."""
    mp.mp.dps = 40
    s2, cs = 0.02, np.linspace(0.1, 0.9, 12)

    def g(a, b):
        a, b = mp.mpf(a), mp.mpf(b)
        m, q, r = (a + b) / 2, (a - b) ** 2 / (2 * s2), mp.sqrt(2 / mp.mpf(s2))
        return mp.e ** (-q) * mp.sqrt(mp.pi * s2 / 2) * (mp.erf(r * (1 - m)) - mp.erf(-r * m)) / 2
    Aex = mp.matrix([[g(a, b) for b in cs] for a in cs])
    y = np.random.default_rng(1).standard_normal(len(cs))
    xr = np.array(mp.lu_solve(Aex, mp.matrix(y.tolist())).tolist(), float).ravel()
    cond = float(np.linalg.cond(np.array(Aex.tolist(), float)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)
        warnings.simplefilter("ignore", PrecisionWarning)
        M = pg.L2(pg.Interval(0, 1), tol=1e-5)
        G = pg.Functionals([M(lambda x, c=c: np.exp(-(x - c) ** 2 / s2)) for c in cs])
        Ainv = (G @ G.adjoint).inverse(pg.Direct(refuse=False))
        x = Ainv(pg.Euclidean(len(cs))(y)).values
    meas = _rel(x, xr)
    assert meas <= _ratio(Ainv)                                       # measured 5e-6, stated 11 (first-order bound)
    assert meas > 100 * cond * EPS                                    # the rounding-only account is broken
    assert "kernels' realisations" in Ainv.claims.find("inherited error").statement


# ------------------------------------------------------------------------------------------------------- sampling
class _Eye:
    """A generator whose standard normals are the identity: a dense route's draws are then its factor L itself."""

    def standard_normal(self, shape):
        return np.eye(*shape)


@pytest.mark.parametrize("spectrum", ["cond 1e20", "rank 30"])
def test_dense_sampling_has_the_covariance_itself(spectrum):
    """L L^T against S along every eigenvector: the former jitter 1e-14 tr(S)/n is 10-50 x the rounding."""
    rng = np.random.default_rng(0)
    n = 60
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    lam = np.geomspace(1, 1e-20, n) if spectrum == "cond 1e20" else np.r_[np.geomspace(1, 1e-3, 30), np.zeros(30)]
    S = (Q * lam) @ Q.T
    S = 0.5 * (S + S.T)
    E = pg.Euclidean(n)
    mu = pg.GaussianMeasure(E, covariance=pg.Matrix(S, domain=E, codomain=E))
    L = mu._centred(n, _Eye()).values.T
    m = np.trace(S) / n
    d = np.einsum("ik,ij,jk->k", Q, L @ L.T, Q)                         # the draws' variance along each q_k
    assert np.abs(d - lam)[n // 2:].max() <= 5e-15 * m                  # the smaller half: measured 3.5e-15, 6.8e-16
    jit = np.linalg.cholesky(S + 1e-14 * m * np.eye(n))                # planted defect: the former factor
    dj = np.einsum("ik,ij,jk->k", Q, jit @ jit.T, Q)
    assert np.abs(dj - lam)[n // 2:].max() > 5e-15 * m                  # measured 1.0e-14 in both


def test_circular_sampling_has_the_covariance_itself():
    rng = np.random.default_rng(0)
    n = 40
    U = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    lam = np.r_[np.geomspace(1, 1e-3, 20), np.zeros(20)]
    S = (U * lam) @ U.conj().T
    S = 0.5 * (S + S.conj().T)
    mu = pg.probability.circular.circular(pg.Euclidean(n, field="C"), covariance=S)
    Z = mu._centred(n, _Eye()).values.T                                 # L (1 + 1j) / sqrt 2
    d = np.real(np.einsum("ik,ij,jk->k", U.conj(), Z @ Z.conj().T, U))
    m = lam.sum() / n
    assert np.abs(d[lam == 0]).max() <= 5e-15 * m                       # measured 1.0e-15; the former jitter added 1e-14
    assert np.abs(d - lam).max() <= 5e-14 * m


def test_a_covariance_that_is_not_positive_semidefinite_has_no_samples():
    from pygeoinf0.probability.gaussian import coordinate_factor
    S = np.diag([1.0, 0.5, -1e-6])
    with pytest.raises(Refused, match="not positive semidefinite"):
        coordinate_factor(S)
    L = coordinate_factor(np.diag([1.0, 0.5, -1e-17]))                 # within the rounding floor: 0
    assert np.allclose(L @ L.T, np.diag([1.0, 0.5, 0.0]), atol=1e-16)


# ------------------------------------------------------------------------------------------------ narrow features
def _true_error(M, F, f_mp, pts):
    B = M.backend
    R = F.realisation
    fr = lambda x: float(B.evaluate(R, np.array([float(x)]))[0, 0])   # noqa: E731
    return float(mp.sqrt(mp.quad(lambda x: (f_mp(x) - fr(x)) ** 2, sorted(set(pts) | set(R.breaks.tolist())))))


def test_a_narrow_bump_is_lost_silently_and_the_claims_say_it_can_be():
    """A bump of width 1e-4 (a fifth of the scout spacing 1/2048) on [0, 1], centred on a scout point: no warning, error
    estimate 1e-15, the true L2 error is the whole bump (mpmath). breaks= at its support resolve it; at its centre they do
    not. (Centred at 0.3141 it is lost in part: 6.6e-4 of 1.1e-2.)"""
    mp.mp.dps = 30
    c, w = 1000 / 2048, 1e-4
    M = pg.L2(pg.Interval(0, 1))
    f = lambda x: 1 + np.exp(-((x - c) / w) ** 2)                     # noqa: E731
    f_mp = lambda x: 1 + mp.e ** (-((x - c) / w) ** 2)                 # noqa: E731
    pts = [0.0, c - 8 * w, c, c + 8 * w, 1.0]
    bump = float(mp.sqrt(mp.quad(lambda x: mp.e ** (-2 * ((x - c) / w) ** 2), pts)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RealisationWarning)               # no warning is raised
        F = M(f)
        F.realisation
    assert F.error < 1e-12
    assert _true_error(M, F, f_mp, pts) > 0.5 * bump                   # measured: all of it, 1.1e-2
    claim = F.claims.find("sampled")
    assert claim is not None and "0.000488" in claim.statement
    Fc = M.element(f, breaks=[c])                                       # at the centre: still lost
    assert _true_error(M, Fc, f_mp, pts) > 0.5 * bump
    Fs = M.element(f, breaks=[c - 4 * w, c + 4 * w])                    # at the support: resolved
    assert _true_error(M, Fs, f_mp, pts) < 1e-6 * bump


# ------------------------------------------------------------------------------------------------ randomised route
class _Dec:
    def __init__(self, n):
        self.n, self._claims = n, []

    def _finish(self, rows, wts, parts, route, k):
        self.lam, self.U = parts[0][0], parts[0][1]


def test_nystrom_variance_bound_is_two_sided(monkeypatch):
    """W = Q diag(1e12 .. 1, 0 ...) Q^T (n = 100, rank 50): the closed-form variance from W_k against the exact one for
    the stored W (mpmath, 30 digits). The former one-sided claim (var_k <= var) fails by 2e-7 of the prior variance;
    |var_k - var| <= E ||R w||^2 / (1 - E) holds."""
    mp.mp.dps = 30
    rng = np.random.default_rng(2)
    n = 100
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    lam = np.r_[np.geomspace(1e12, 1, 50), np.zeros(n - 50)]
    Wm = (Q * lam) @ Q.T
    Wm = 0.5 * (Wm + Wm.T)
    monkeypatch.setattr(partial, "operator", lambda dec: ((lambda V: Wm @ V), None))
    dec = _Dec(n)
    partial.nystrom(dec, tail=1e-2, seed=0)
    assert "of either sign" in dec._claims[-1].statement and "W_k <= W and" not in dec._claims[-1].statement
    Z = rng.standard_normal((n, 4))
    Wf = ((Q * np.sqrt(lam)) @ Q.T) @ Z
    tt = (Z * Z).sum(0)
    A = mp.matrix(Wm.tolist()) + mp.eye(n)
    ex = np.array([float(mp.mpf(tt[p]) - (mp.matrix(Wf[:, p].tolist()).T * mp.lu_solve(A, mp.matrix(Wf[:, p].tolist())))[0])
                   for p in range(Z.shape[1])])
    C = dec.U.T @ Wf
    perp = Wf - dec.U @ C
    vk = tt - np.einsum("kp,k->p", C * C, 1.0 / (dec.lam + 1.0)) - (perp * perp).sum(0)
    Rw2 = np.einsum("kp,k->p", C * C, 1.0 / (dec.lam + 1.0) ** 2) + (perp * perp).sum(0)
    E = dec.error
    vb = E * Rw2 * partial.two_sided(E)
    assert np.all(np.abs(vk - ex) <= vb)                               # two-sided: holds (ratio 4e-3)
    assert np.max(vk - ex) > 1e-12 * tt.max()                          # planted defect: the one-sided claim fails
    assert np.all(partial.std_error(vk, vb) >= np.abs(np.sqrt(ex) - np.sqrt(vk)))
