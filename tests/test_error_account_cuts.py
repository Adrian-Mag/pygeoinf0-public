"""Plan 054 P2: the numerical cuts outside the solvers, measured against exact answers, each with a planted-defect
control (the former or a wrong cut, shown to fail the same check).

    optimisers/dual/reduction.reduce       correlation eigenvalues <= eps q max dropped (was 1e-13: intervals too narrow by 1e-2 of
                             their width when the target is within ~4e-7 rad of the representers' span)
    optimisers/dual/reduction.ball_slab    a . y written as a . y0 + rem ||pa|| (was a @ y: no correct digit when the target is
                             parallel to the slab normal; values far outside the certificate's bound)
    reduction.data_svd       singular values <= eps max(n, k) s_max count as zero (was 1e-12: errors up to 50 %)
    reduction.whiten         nuisance columns with |R_jj| <= eps n max dropped (was 1e-12: a nuisance direction lost)
    solvers/dataspace/partial._orth  bookkeeping: the stated bound and the rank do not depend on the cut
    RCOND (spectral, closed forms) and Eigh("rounding"): the count dropped is stated
References are exact for the given double inputs: the 2-D problem from a QR of the actual vectors, closed forms from
known factors, the same set written so that no cut applies."""
import math

import numpy as np
import pytest
from scipy.stats import norm

import pygeoinf0 as pg
from pygeoinf0.optimisers.dual import reduction as dual
from pygeoinf0.solvers.dataspace import partial

EPS = np.finfo(float).eps


def _reduce_with(cut):
    """dual.reduce with a fixed relative cut (the planted defect: the former 1e-13)."""
    def reduce(J, n_dirs, info=None):
        J = 0.5 * (J + J.T)
        d = np.sqrt(np.clip(np.diag(J), 0.0, None))
        d = np.where(d > 0.0, d, 1.0)
        lam, U = np.linalg.eigh(J / np.outer(d, d))
        keep = lam > cut * max(lam.max(initial=0.0), 1e-300)
        Y = np.sqrt(lam[keep])[:, None] * U[:, keep].T * d
        return Y[:, :n_dirs], Y[:, n_dirs:].T
    return reduce


def _ball_slab_2d(t, rt, s, w, r):
    """The exact interval of <t, m> over ||m|| <= r, |s - <rt, m>| <= w from a QR of the two vectors."""
    _, R = np.linalg.qr(np.c_[t, rt])
    return -dual.ball_slab(-R[:, 0], R[:, 1], s, w, r)[0], dual.ball_slab(R[:, 0], R[:, 1], s, w, r)[0]


def near_parallel(theta, seed, n=200, nd=10, sig=1e-3):
    """A target at angle theta to r~ = G* X* 1 (the slab's normal in the model space)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    A = np.exp(-(x[None] - np.linspace(0.05, 0.95, nd)[:, None]) ** 2 / (2 * 0.1 ** 2))
    xv = rng.standard_normal(nd)
    rt = A.T @ xv
    u = rng.standard_normal(n)
    u -= rt * (rt @ u) / (rt @ rt)
    t = np.cos(theta) * rt / np.linalg.norm(rt) + np.sin(theta) * u / np.linalg.norm(u)
    M, D = pg.Euclidean(n), pg.Euclidean(nd)
    G, T, X = pg.Matrix(A, M, D), pg.Matrix(t[None], M, pg.Euclidean(1)), pg.Matrix(xv[None], D, pg.Euclidean(1))
    mt = rng.standard_normal(n)
    d = D(A @ mt + sig * rng.standard_normal(nd))
    L = pg.Additive(G, pg.GaussianMeasure(D, sig ** 2 * pg.Identity(D)))
    r = 1.5 * np.linalg.norm(mt)
    s, w = float(xv @ d.values), norm.ppf(0.975) * sig * np.linalg.norm(xv)
    return dict(S=pg.NormBall(M, r), L=L, T=T, X=X, d=d, ref=_ball_slab_2d(t, rt, s, w, r))


def _errors(p):
    ref = np.array(p["ref"])
    width = ref[1] - ref[0]
    oc = np.array(pg.inference.ocsola(p["S"], p["L"], p["T"], p["d"], X=p["X"]))
    P = pg.push(pg.update(p["S"], p["L"].acceptance(0.05, shape=p["X"]), p["d"]), p["T"])
    lo, hi = P.bounds(0)                                  # the values (interval(0) reports the certified outer ends)
    bgp = np.array([lo.value, hi.value])
    return np.abs(oc - ref).max() / width, np.abs(bgp - ref).max() / width


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reduce_keeps_the_directions_that_hold_digits(seed, monkeypatch):
    """theta = 3e-7: the correlation eigenvalue 1 - cos(theta) = 4.5e-14 lies above the floor 2 eps and below the
    former cut. oCSOLA and BGP agree with the exact interval to 1e-3 of its width; the former cut does not."""
    p = near_parallel(3e-7, seed)
    assert max(_errors(p)) < 1e-3
    ref, width = p["ref"], p["ref"][1] - p["ref"][0]
    outer = pg.push(pg.update(p["S"], p["L"].acceptance(0.05, shape=p["X"]), p["d"]), p["T"]).interval(0)
    assert outer[0] <= ref[0] + 1e-9 * width and ref[1] <= outer[1] + 1e-9 * width     # the certified ends contain it
    monkeypatch.setattr(dual, "reduce", _reduce_with(1e-13))           # the planted defect: the former cut
    assert min(_errors(p)) > 3e-3


def test_reduce_states_what_it_drops():
    J = np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) + np.diag([0.0, 1.0, 1.0])
    info = {}
    dual.reduce(J, 1, info)
    assert (info["dropped"], info["size"]) == (0, 3)
    dual.reduce(np.outer([1.0, 2.0], [1.0, 2.0]), 1, info)               # truly parallel
    assert (info["dropped"], info["size"]) == (1, 2)
    assert int(info["reduction"][4].sum()) == 1                          # the kept eigenvalues, for the general rule


def _old_ball_slab(a, beta, s, w, r):
    """optimisers/dual/reduction.ball_slab before plan 054 P2 (the planted defect)."""
    nb, na = np.linalg.norm(beta), np.linalg.norm(a)
    yf = r * a / na
    t = beta @ yf
    if s - w <= t <= s + w:
        return r * na, yf, "EXACT"
    h = s + w if t > s + w else s - w
    y0 = h * beta / nb ** 2
    pa = a - (a @ beta) * beta / nb ** 2
    rem = math.sqrt(max(r * r - h * h / nb ** 2, 0.0))
    y = y0 + rem * pa / np.linalg.norm(pa)
    return float(a @ y), y, "EXACT"


def test_bgp_slab_bound_when_the_target_is_parallel_to_the_slab_normal(monkeypatch):
    """An estimable target resolved by SOLA's X: t and r~ = G* X* 1 parallel to rounding. The value lies in its own
    certificate's bracket and equals the exact interval; the former formula returned values ~1000 widths away."""
    n, nd, sig = 300, 10, 1e-3
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, n)
    A = np.exp(-(x[None] - np.linspace(0.05, 0.95, nd)[:, None]) ** 2 / (2 * 0.05 ** 2))
    t = A.T @ rng.standard_normal(nd) / nd
    M, D = pg.Euclidean(n), pg.Euclidean(nd)
    G, T = pg.Matrix(A, M, D), pg.Matrix(t[None], M, pg.Euclidean(1))
    mt = 1 + 0.3 * np.sin(5 * x)
    d = D(A @ mt + sig * rng.standard_normal(nd))
    L = pg.Additive(G, pg.GaussianMeasure(D, sig ** 2 * pg.Identity(D)))
    S = pg.NormBall(M, 1.5 * np.linalg.norm(mt))
    X = pg.inference.sola(L, T, d, gamma=1e-6).X
    xv = X.matrix()[0]
    ref = np.array(_ball_slab_2d(t, A.T @ xv, float(xv @ d.values), norm.ppf(0.975) * sig * np.linalg.norm(xv),
                                 S.radius))
    width = ref[1] - ref[0]

    def bounds():
        return pg.push(pg.update(S, L.acceptance(0.05, shape=X), d), T).bounds(0)
    for b in bounds():
        assert b.lower - 1e-9 * abs(b.value) <= b.value <= b.upper + 1e-9 * abs(b.value)
    assert np.abs(np.array([b.value for b in bounds()]) - ref).max() < 1e-6 * width
    monkeypatch.setattr(dual, "ball_slab", _old_ball_slab)             # the planted defect
    assert np.abs(np.array([b.value for b in bounds()]) - ref).max() > 1.0 * width


def _svd_with(cut):
    def _svd(B):
        U, s, Vt = np.linalg.svd(B, full_matrices=True)
        s2 = np.zeros(B.shape[1])
        s2[:len(s)] = s ** 2
        return U, s, Vt.T, s2, int(np.sum(s > cut * s.max()))
    return _svd


@pytest.mark.parametrize("seed", [0, 1])
def test_svd_rank_keeps_singular_values_that_hold_digits(seed, monkeypatch):
    """B = diag(s) V^T (orthogonal rows of graded lengths, s down to 1e-13 > eps max(n, k)): ball_exact and none_ball
    against their closed forms from the known factors (errors <= 6e-4 measured over six seeds); the former cut 1e-12
    drops two rows of the datum (errors 5e-3 to 1.5e-1)."""
    n, k = 8, 12
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.standard_normal((k, k)))[0][:, :n]
    s = np.geomspace(1.0, 1e-13, n)
    B = s[:, None] * V.T
    y = rng.standard_normal(k)
    d = B @ (0.5 * y / np.linalg.norm(y))
    a = rng.standard_normal(k)
    y0 = V @ (d / s)
    exact = a @ y0 + math.sqrt(1.0 - y0 @ y0) * np.linalg.norm(a - V @ (V.T @ a))
    mu0 = rng.standard_normal(n)
    a2, rho = B.T @ mu0, 0.1
    none = mu0 @ d + rho * np.linalg.norm(mu0)

    def errors(dependent=False):
        info = {}
        val = dual.ball_exact(a, B, d, 1.0, info, dependent)[0]
        val2 = dual.none_ball(a2, B, d, rho, None, dependent)[0]
        return abs(val - exact) / abs(exact), abs(val2 - none) / abs(none), info["data_dropped"]
    e1, e2, dropped = errors()
    assert e1 < 1e-3 and e2 < 1e-3 and dropped == 0
    monkeypatch.setattr(dual, "data_svd", _svd_with(1e-12))                # the planted defect: the former cut
    with pytest.raises(pg.errors.Refused, match="rounding floor"):     # it drops a nonzero singular value: refused
        errors()                                                        # (plan 054 fix round), and if declared
    e1, e2, _ = errors(dependent=True)                                  # dependent the value is the wrong one
    assert max(e1, e2) > 5e-3


def _whiten_with(cut):
    def _whiten(form, e, B):
        Phi, F = form["Phi"], form["F"]
        n = Phi.shape[0]
        Q, R = np.linalg.qr(Phi @ F)
        Q = Q[:, np.abs(np.diag(R)) > cut * np.abs(np.diag(R)).max()]
        Z = np.linalg.qr(np.c_[Q, np.eye(n)])[0][:, Q.shape[1]:n]
        M = Z.T @ Phi
        return M @ (e - form["centre"]), M @ B, M
    return _whiten


def test_whiten_keeps_a_nuisance_direction_that_holds_digits(monkeypatch):
    """span{f1} + span{f1 + delta g} = span{f1, g} for delta = 1e-13 (above the floor eps n); an exact duplicate
    (delta = 0) is span{f1}: dropping it is bookkeeping. The former cut 1e-12 lost g."""
    rng = np.random.default_rng(6)
    Xe, D = pg.Euclidean(30), pg.Euclidean(9)
    A, T = rng.normal(size=(9, 30)), pg.Matrix(rng.normal(size=(1, 30)), Xe, pg.Euclidean(1))
    d = rng.normal(size=9)
    Em = np.eye(3)[[0, 0, 0, 1, 1, 1, 1, 2, 2]]
    f1, g = Em[:, 0], Em[:, 1]
    rho = 0.5 * np.linalg.norm(d)

    def interval(N):
        return np.array(pg.push(pg.update(pg.NormBall(Xe, 3.0), pg.Additive(pg.Matrix(A, Xe, D), N), D(d)),
                                T).interval(0))

    def near(delta):
        return pg.Span([D(f1)]) + (pg.Span([D(f1 + delta * g)]) + pg.NormBall(D, rho))
    ref = interval(pg.Span([D(f1), D(g)]) + pg.NormBall(D, rho))
    one = interval(pg.Span([D(f1)]) + pg.NormBall(D, rho))
    width = ref[1] - ref[0]
    assert np.abs(interval(near(1e-13)) - ref).max() < 1e-3 * width
    assert np.abs(interval(near(0.0)) - one).max() < 1e-9 * width
    assert np.abs(one - ref).max() > 1e-2 * width                      # g matters
    monkeypatch.setattr(dual, "whiten", _whiten_with(1e-12))          # the planted defect: the former cut
    assert np.abs(interval(near(1e-13)) - ref).max() > 1e-2 * width


def _orth_with(cut):
    def _orth(V, Q):
        for _ in range(2):
            if Q.shape[1]:
                V = V - Q @ (Q.T @ V)
        Qv, R = np.linalg.qr(V)
        return Qv[:, np.abs(np.diag(R)) > cut * max(float(np.abs(np.diag(R)).max(initial=0.0)), 1e-300)]
    return _orth


class _Dec:
    def __init__(self, n):
        self.n, self._claims = n, []

    def _finish(self, rows, wts, parts, route, k):
        self.lam, self.U = parts[0][0], parts[0][1]


def _nystrom(Wm, tail, monkeypatch, orth=None):
    monkeypatch.setattr(partial, "operator", lambda dec: ((lambda V: Wm @ V), None))
    monkeypatch.setattr(partial, "_orth", orth or _ORTH)
    dec = _Dec(Wm.shape[0])
    partial.nystrom(dec, tail=tail, seed=0)
    actual = np.linalg.norm(Wm - dec.U @ np.diag(dec.lam) @ dec.U.T, 2)
    return len(dec.lam), dec.error, actual


_ORTH = partial._orth


def test_orth_is_bookkeeping(monkeypatch):
    """The randomised route on a dense W of known spectrum (two clusters, 1e14 and 0.1): the stated bound holds and
    the rank and bound are the same for the cut 1e-12, the floor eps n and 1e-6. Control: keeping every column (cut
    0) changes them, so the comparison is not vacuous."""
    rng = np.random.default_rng(0)
    n = 600
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    Wm = (Q * np.r_[1e14 * np.ones(30), 1e-1 * np.ones(n - 30)]) @ Q.T
    Wm = 0.5 * (Wm + Wm.T)
    runs = [_nystrom(Wm, 1e-2, monkeypatch, orth) for orth in (None, _orth_with(EPS * n), _orth_with(1e-6))]
    for k, est, actual in runs:
        assert actual <= est
    assert len({(k, est) for k, est, _ in runs}) == 1
    k0, est0, actual0 = _nystrom(Wm, 1e-2, monkeypatch, _orth_with(0.0))   # the control
    assert actual0 <= est0 and (k0, est0) != runs[0][:2]


def _spectral_problem(dup):
    rng = np.random.default_rng(1)
    n, nd = 50, 6
    A = rng.standard_normal((nd, n))
    if dup:
        A[-1] = A[0]                                                    # one zero eigenvalue by construction
    M, D = pg.Euclidean(n), pg.Euclidean(nd)
    G = pg.Matrix(A, M, D)
    L = pg.Additive(G, pg.GaussianMeasure(D, 1e-2 * pg.Identity(D)))
    d = D(A @ rng.standard_normal(n))
    return M, D, G, L, d, pg.Matrix(rng.standard_normal((1, n)), M, pg.Euclidean(1))


@pytest.mark.parametrize("dup, zero", [(False, 0), (True, 1)])
def test_rcond_count_is_stated(dup, zero):
    """The spectral route's note and the feasibility claims count the eigenvalues treated as zero (1/Lam). Control:
    a duplicated datum gives exactly one."""
    M, D, G, L, d, T = _spectral_problem(dup)
    b = pg.push(pg.update(pg.NormBall(M, 10.0), L.acceptance(0.05), d), T).bounds(0)[1]
    assert f"{zero} of 6 eigenvalues <= 1e-15" in b.note
    dec = pg.decompose(G, pg.GaussianMeasure(D, 1e-2 * pg.Identity(D)))
    claims = " ".join(c.statement for c in dec.feasibility(d, a=1.0).claims)
    assert f"{zero} of 6 eigenvalues <= 1e-15" in claims


def test_eigh_rounding_states_its_threshold_and_count():
    """MinimumNorm with a 2-dimensional nuisance at cond(G G*) = 1e16: exactly dim mul = 2 dropped (the zero eigenvalues
    by construction), the result as accurate as cond eps allows. Control: rcond 1e-12 would drop a third direction,
    which the stated count shows. B8 and F13 state their threshold."""
    from pygeoinf0.operators.inverses import MinimumNorm
    from pygeoinf0.solvers import Eigh
    rng = np.random.default_rng(0)
    nd, nm = 10, 30
    U = np.linalg.qr(rng.standard_normal((nd, nd)))[0]
    V = np.linalg.qr(rng.standard_normal((nm, nm)))[0][:, :nd]
    A = U @ np.diag(np.geomspace(1.0, 1e-8, nd)) @ V.T
    X, D = pg.Euclidean(nm), pg.Euclidean(nd)
    F = rng.standard_normal((nd, 2))
    mn = MinimumNorm(pg.Matrix(A, X, D), pg.Span([D(F[:, 0]), D(F[:, 1])]))
    text = " ".join(c.statement for c in mn.claims)
    assert "eps n max|lam|" in text and ": 2 of 10; dim mul = 2" in text
    assert mn.Ainv.A.inverse(Eigh(1e-12)).factorisation().dropped == 3      # the control: a wrong cut is visible
    Qc = np.linalg.qr(np.c_[F, rng.standard_normal((nd, nd - 2))])[0][:, 2:]   # the exact minimum-norm element
    y = rng.standard_normal(nd)
    x = A.T @ Qc @ np.linalg.solve(Qc.T @ A @ A.T @ Qc, Qc.T @ y)
    assert np.linalg.norm(mn(D(y)).values - x) < 1e-2 * np.linalg.norm(x)
    H = pg.Euclidean(5)
    G = pg.Matrix(rng.normal(size=(3, 5)), H, pg.Euclidean(3))
    post = pg.update(pg.GaussianMeasure(H, pg.Identity(H)), G, pg.Euclidean(3)(rng.normal(size=3)))
    assert any("(G C G*)^+: the pseudo-inverse, eigenvalues with |lam| <= eps n max|lam|" in c.statement
               for c in post.claims)
    Jp = pg.push(pg.Ordering.norm(H), pg.Matrix(rng.normal(size=(2, 5)), H, pg.Euclidean(2)))
    assert any("(T T*)^+: the pseudo-inverse" in c.statement and ": 0 of 2" in c.statement for c in Jp.claims)
