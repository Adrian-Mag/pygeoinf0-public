"""Feasible sets, property bounds and posteriors through the cells (plan 008 P3), against LAND-001's recorded numbers
(EXP-001, EXP-019, EXP-020), an independent numerical dual, two routes for each identity (cSOLA, oCSOLA, origin times,
stacking) and the inner-product invariance of BGP."""
import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.optimize import minimize

import pygeoinf0 as pg


def gauss_overlap(a, s1, b, s2):
    v = s1 ** 2 + s2 ** 2
    return np.sqrt(2 * np.pi * s1 ** 2 * s2 ** 2 / v) * np.exp(-(a - b) ** 2 / (2 * v))


def exp001():
    """EXP-001's problem (M = L2(R), Gaussian kernels, analytic pairings) on the exact span of its representers."""
    Nd, sg, xs = 8, 0.08, np.linspace(0.1, 0.9, 8)
    tc, st, mc, sm = 0.5, 0.03, 0.45, 0.05
    centres, widths = np.r_[tc, xs], np.r_[st, np.full(Nd, sg)]
    J = gauss_overlap(centres[:, None], widths[:, None], centres[None, :], widths[None, :])
    L = np.linalg.cholesky(J)                                  # rows: coordinates of t, g_1..g_8 in an orthonormal basis
    X = pg.Euclidean(Nd + 1)
    T, G = pg.Matrix(L[:1], X, pg.Euclidean(1)), pg.Matrix(L[1:], X, pg.Euclidean(Nd))
    amp = 0.8 / np.sqrt(gauss_overlap(mc, sm, mc, sm))
    d_clean = amp * gauss_overlap(xs, sg, mc, sm)
    rng = np.random.default_rng(0)
    eta = rng.normal(size=Nd)
    eta *= 0.5 * 0.02 / np.linalg.norm(eta)
    return X, G, T, G.codomain(d_clean + eta)


def test_bgp_reproduces_exp001():
    X, G, T, d = exp001()
    K = pg.update(pg.NormBall(X, 1.0), pg.Additive(G, pg.NormBall(G.codomain, 0.02)), d)
    lo, hi = pg.push(K, T).bounds(0)
    assert lo.value == pytest.approx(0.01516775, abs=1e-8) and hi.value == pytest.approx(0.21075796, abs=1e-8)
    for b in (lo, hi):
        assert b.status == "EXACT" and b.lower - 1e-12 <= b.value <= b.upper + 1e-12 and b.upper - b.lower <= 1e-9
    res = pg.inference.bgp(pg.NormBall(X, 1.0), pg.Additive(G, pg.NormBall(G.codomain, 0.02)), T, d)
    assert res.interval(0) == pytest.approx((lo.value, hi.value), abs=1e-12)


def exp019(H):
    x0 = lambda x, c, s: np.exp(-(x - c) ** 2 / (2 * s ** 2))   # noqa: E731
    xs = np.linspace(0.1, 0.9, 8)
    G = pg.Functionals([H(lambda x, a=a: x0(x, a, 0.08)) for a in xs])
    T = pg.Functionals([H(lambda x: x0(x, 0.5, 0.03))])
    mt = H(lambda x: x0(x, 0.45, 0.05))
    mt = mt * (0.8 / mt.norm())
    rng = np.random.default_rng(0)
    eta = rng.normal(size=8)
    eta *= 0.5 * 0.02 / np.linalg.norm(eta)
    return G, T, G(mt) + G.codomain(eta)


def test_bgp_on_l2_lies_in_exp019s_rigorous_bracket():
    H = pg.L2("[0, 1]")
    G, T, d = exp019(H)
    lo, hi = pg.push(pg.update(pg.NormBall(H, 1.0), pg.Additive(G, pg.NormBall(G.codomain, 0.02)), d), T).bounds(0)
    assert 0.2107355 <= hi.value <= 0.2107385                    # EXP-019 at N = 256: INNER 0.210736, OUTER 0.210738
    assert hi.value == pytest.approx(0.2107362357, abs=5e-6)    # its reference truncates at 20000 sine modes
    assert lo.value == pytest.approx(0.0152217207, abs=5e-6)
    A, b, tt = G.gram(), np.asarray(H.inner(G.kernels, T.kernels), float).ravel(), float(T.gram()[0, 0])

    def phi(lam, s=1.0):                                         # an independent dual, EXP-001's way
        return lam @ d.values + np.sqrt(max(tt - 2 * s * lam @ b + lam @ A @ lam, 0.0)) + 0.02 * np.linalg.norm(lam)
    r = minimize(phi, np.zeros(8), method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-14, maxiter=40000,
                                                                       maxfev=80000))
    r = minimize(phi, r.x, method="BFGS", options=dict(gtol=1e-12))
    assert hi.value == pytest.approx(r.fun, rel=1e-7)


@pytest.mark.parametrize("q, V", [(2.0, 1.522628e-04), (1.2, 8.029713e-04)])
def test_pli_reproduces_exp020(q, V):
    H = pg.L2("[0, 1]")
    n = np.arange(1, 257)
    sines = H.family(lambda x, k: np.sqrt(2) * np.sin(k * np.pi * x), k=n)
    prior = pg.GaussianMeasure(H, pg.FiniteRank(sines, (1 + (n / 3.0) ** 2) ** (-q)))
    x0 = lambda x, c, s: np.exp(-(x - c) ** 2 / (2 * s ** 2))   # noqa: E731
    D = pg.Euclidean(8, covariance=1e-4)
    G = pg.Functionals([H(lambda x, a=a: x0(x, a, 0.08)) for a in np.linspace(0.1, 0.9, 8)], into=D)
    T = pg.Functionals([H(lambda x: x0(x, 0.5, 0.03))])
    post = pg.inference.pli(prior, pg.Additive(G, D.standard_gaussian()), T, D(np.zeros(8)))
    assert float(post.covariance.matrix()[0, 0]) == pytest.approx(V, rel=2e-6)


def test_csola_contains_bgp_and_ocsola_is_bgp_with_the_slab():
    X, G, T, d = exp001()
    D = G.codomain
    S, N = pg.NormBall(X, 1.0), pg.NormBall(D, 0.02)
    Xs = pg.inference.sola(pg.Additive(G, D.standard_gaussian()), T, d, gamma=1e-3).X
    E = pg.inference.csola(S, pg.Additive(G, N), T, d, X=Xs)
    lo, hi = pg.push(pg.update(S, pg.Additive(G, N), d), T).interval(0)
    elo, ehi = E.interval(0)
    assert elo <= lo + 1e-12 and hi <= ehi + 1e-12
    w = Xs.adjoint(pg.Euclidean(1)([1.0]))                       # X* 1: the cSOLA certificate
    resid = T.adjoint(pg.Euclidean(1)([1.0])) - G.adjoint(w)
    assert ehi == pytest.approx(float(Xs(d).values[0]) + resid.norm() + 0.02 * w.norm(), rel=1e-12)
    L = pg.Additive(G, pg.GaussianMeasure(D, 1e-4 * pg.Identity(D)))
    oc = pg.inference.ocsola(S, L, T, d, X=Xs, alpha=0.05)
    slab = pg.push(pg.update(S, L.acceptance(0.05, shape=Xs), d), T).interval(0)
    assert oc == pytest.approx(slab, rel=1e-8)


def test_bgp_does_not_depend_on_the_data_inner_product():
    X, G, T, d = exp001()
    A = np.random.default_rng(4).normal(size=(8, 8))
    C = A @ A.T / 8 + np.eye(8) * 0.5
    Dw = pg.Euclidean(8, covariance=C)
    Gw = pg.Matrix(G.A, X, Dw)
    r = 0.05
    one = pg.push(pg.update(pg.NormBall(X, 1.0), pg.Additive(G, pg.Ellipsoid(G.codomain, r, covariance=C)), d), T)
    two = pg.push(pg.update(pg.NormBall(X, 1.0), pg.Additive(Gw, pg.NormBall(Dw, r)), Dw(d.values)), T)
    assert one.interval(0) == pytest.approx(two.interval(0), rel=1e-9)


def test_an_unattainable_datum_gives_an_empty_set_with_a_certificate():
    X, G, T, d = exp001()
    K = pg.update(pg.NormBall(X, 0.05), pg.Additive(G, pg.NormBall(G.codomain, 1e-3)), d)
    assert K.is_empty() is pg.Truth.TRUE
    lam = K.emptiness.certificate                                 # check phi(lam, 0) < 0 by a second formula
    B = G.A
    assert lam @ d.values + 0.05 * np.linalg.norm(B.T @ lam) + 1e-3 * np.linalg.norm(lam) < 0
    assert pg.push(K, T).bound(np.array([1.0])).status == "EMPTY"


def origin_time_problem(seed=6):
    rng = np.random.default_rng(seed)
    nm, N = 30, 9
    Xe, D = pg.Euclidean(nm), pg.Euclidean(N, weights=rng.uniform(0.5, 2, N))
    Ge = pg.Matrix(rng.normal(size=(N, nm)), Xe, D, name="G")
    Em = np.eye(3)[[0, 0, 0, 1, 1, 1, 1, 2, 2]]
    E = pg.Span([D(Em[:, j]) for j in range(3)], name="origin times")
    T = pg.Matrix(rng.normal(size=(1, nm)), Xe, pg.Euclidean(1), name="T")
    return Xe, D, Ge, Em, E, T, D(rng.normal(size=N))


def test_origin_times_exact_data_against_contrasts_and_a_direct_sum():
    Xe, D, Ge, Em, E, T, d = origin_time_problem()
    K = pg.update(pg.NoPrior(Xe), pg.Additive(Ge, E), d)
    Q = np.r_[np.eye(9)[1:3] - np.eye(9)[[0, 0]], np.eye(9)[4:7] - np.eye(9)[[3, 3, 3]], np.eye(9)[[8]] - np.eye(9)[[7]]]
    m_q = np.linalg.pinv(Q @ Ge.A) @ (Q @ d.values)              # within-event differencing
    m_c = np.linalg.pinv(null_space(Em.T).T @ Ge.A) @ (null_space(Em.T).T @ d.values)
    assert np.allclose(K.summary().values, m_q, atol=1e-11) and np.allclose(m_q, m_c, atol=1e-11)
    errs = []
    for w in (1e-1, 1e-3, 1e-5):                                 # origin times as a weighted model summand (plan 007)
        Xs = pg.DirectSum({"m": Xe, "t0": pg.Euclidean(3)}, weights=[1.0, w])
        Gs = pg.Functionals(Xs.riesz([{"m": Ge.A[i], "t0": Em[i]} for i in range(9)]), into=D)
        x = Gs.adjoint((Gs @ Gs.adjoint).inv(d))
        m = Xs.part(x, "m")
        assert pg.Additive(Ge, E)(m).member(d) is pg.Truth.TRUE   # its model part is in the relation's fibre
        errs.append(np.abs(m.values - m_q).max() / np.abs(m_q).max())
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-3 and errs[0] > 1e-3


def test_origin_times_gaussian_quotient_against_contrasts_and_a_vague_prior():
    Xe, D, Ge, Em, E, T, d = origin_time_problem()
    prior = pg.GaussianMeasure(Xe, pg.Identity(Xe))
    post = pg.update(prior, pg.Additive(Ge, D.standard_gaussian(), nuisance=E), d)
    Wi = np.linalg.inv(D.metric_matrix())                       # noise coordinates N(0, W^-1)
    Q = null_space(Em.T).T
    S = Q @ Ge.A @ Ge.A.T @ Q.T + Q @ Wi @ Q.T
    mean_q = Ge.A.T @ Q.T @ np.linalg.solve(S, Q @ d.values)
    assert np.allclose(post.mean.values, mean_q, atol=1e-11)
    for s, tol in ((1e2, 1e-3), (1e4, 1e-7)):
        Sv = Ge.A @ Ge.A.T + Wi + s ** 2 * Em @ Em.T             # c ~ N(0, s^2 I) integrated out
        assert np.allclose(Ge.A.T @ np.linalg.solve(Sv, d.values), mean_q, atol=tol * np.abs(mean_q).max())


def test_origin_times_bgp_certificates_ignore_the_origin_times():
    Xe, D, Ge, Em, E, T, d = origin_time_problem()
    r = 0.5 * D.norm(d)
    P1 = pg.push(pg.update(pg.NormBall(Xe, 3.0), pg.Additive(Ge, E + pg.NormBall(D, r)), d), T)
    lo, hi = P1.bounds(0)
    lam = hi.certificate
    assert np.abs(Em.T @ D.apply_metric(lam)).max() <= 1e-10 * np.abs(lam).max()   # weights sum to 0 per event
    q = pg.LinearRelation(Ge, mul=E).quotient_map()
    P2 = pg.push(pg.update(pg.NormBall(Xe, 3.0), pg.Additive(q @ Ge, pg.NormBall(q.codomain, r)), q(d)), T)
    assert (lo.value, hi.value) == pytest.approx(P2.interval(0), rel=1e-9)


def test_stacking_rules():
    X, G, T, d = exp001()
    D = G.codomain
    G1, G2 = pg.Matrix(G.A[:4], X, pg.Euclidean(4)), pg.Matrix(G.A[4:], X, pg.Euclidean(4))
    d1, d2 = G1.codomain(d.values[:4]), G2.codomain(d.values[4:])
    R1, R2 = pg.Additive(G1, pg.NormBall(G1.codomain, 0.02)), pg.Additive(G2, pg.NormBall(G2.codomain, 0.02))
    S = pg.NormBall(X, 1.0)
    u1 = pg.push(pg.update(S, R1, d1), T).interval(0)
    u2 = pg.push(pg.update(S, R2, d2), T).interval(0)
    anyR = pg.stack(R1, R2, rule="any")
    dd = pg.relations.join(anyR, d1, d2)
    ua = pg.push(pg.update(S, anyR, dd), T)
    assert ua.support(np.array([1.0])) == pytest.approx(max(u1[1], u2[1]), rel=1e-12)
    allR = pg.stack(R1, R2, rule="all")
    Ka = pg.update(S, allR, dd)
    b = pg.push(Ka, T).bound(np.array([1.0]))
    assert b.upper <= min(u1[1], u2[1]) + 1e-9                   # one model explains both blocks
    assert b.lower <= b.upper + 1e-12


def _slab_interval(tt, tr, rr, s, w, r):
    """The exact [inf, sup] of <t, m> on ||m|| <= r, |<r~, m> - s| <= w, in the plane of t and r~ (closed form: concave
    in the component a of m along r~, so the optimum is the unconstrained one clipped to the slab)."""
    nt, nr = np.sqrt(tt), np.sqrt(rr)
    c = tr / (nt * nr)
    sn = np.sqrt(max(1.0 - c * c, 0.0))
    lo, hi = max(-r, (s - w) / nr), min(r, (s + w) / nr)
    a_up, a_lo = np.clip(r * c, lo, hi), np.clip(-r * c, lo, hi)
    return (-nt * (-c * a_lo + sn * np.sqrt(r * r - a_lo ** 2)), nt * (c * a_up + sn * np.sqrt(r * r - a_up ** 2)))


def test_reduce_keeps_a_direction_at_any_scale():
    from pygeoinf0.optimisers.dual import reduce
    u, v = np.array([1.0, 0.0]), np.array([0.6, 0.8]) * 1e-20
    J = np.array([[u @ u, u @ v], [u @ v, v @ v]])
    Yt, B = reduce(J, 1)
    Y = np.c_[Yt, B.T]
    assert Y.shape == (2, 2)
    assert (Y.T @ Y) / np.sqrt(np.outer(np.diag(J), np.diag(J))) == pytest.approx(
        J / np.sqrt(np.outer(np.diag(J), np.diag(J))), abs=1e-12)
    assert reduce(np.outer([1.0, 2.0], [1.0, 2.0]), 1)[0].shape[0] == 1          # truly parallel: one direction


def test_ocsola_does_not_depend_on_the_scale_of_x():
    X, G, T, d = exp001()
    D = G.codomain
    S, L = pg.NormBall(X, 1.0), pg.Additive(G, pg.GaussianMeasure(D, 1e-4 * pg.Identity(D)))
    Xs = pg.inference.sola(pg.Additive(G, D.standard_gaussian()), T, d, gamma=1e-3).X
    oc = pg.inference.ocsola(S, L, T, d, X=Xs, alpha=0.05)
    for scale in (1e-16, 1e-8, 1e8):
        assert pg.inference.ocsola(S, L, T, d, X=scale * Xs, alpha=0.05) == pytest.approx(oc, rel=1e-6)


@pytest.mark.parametrize("gamma", [1e-6, 1e-3, 1.0, 1e3])
def test_ocsola_closed_with_small_kernels_is_the_exact_interval(gamma):
    """The same problem in other units: kernels 1e-8 of EXP-001's, models 1e8 of them. The whitened G G* is tiny against
    gamma, so X is tiny and <t, G* X* 1> is far below ||t||^2; the interval is still the slab's, not a falsely narrow one."""
    from scipy.stats import norm
    X, G, T, d = exp001()
    A, C, r = G.matrix() * 1e-8, 1e-4, 1e8
    D = pg.Euclidean(A.shape[0])
    G8, y = pg.Matrix(A, X, D), D(d.values)
    L = pg.Additive(G8, pg.GaussianMeasure(D, C * pg.Identity(D)))
    t = T.matrix()[0]
    x = np.linalg.solve(A @ A.T + gamma * C * np.eye(A.shape[0]), A @ t)
    rt = A.T @ x
    exact = _slab_interval(t @ t, t @ rt, rt @ rt, x @ y.values, norm.ppf(0.975) * np.sqrt(C) * np.linalg.norm(x), r)
    pb = pg.Problem(M=X, D=D, mdr=L, P=T.codomain, mpr=T, prior=pg.NormBall(X, r), datum=y)
    assert pb.run("ocsola", gamma=gamma, alpha=0.05) == pytest.approx(exact, rel=1e-6)


def test_blocks_are_not_detected_because_the_kernels_are_small_against_the_noise():
    X, G, T, d = exp001()
    A = G.matrix() * 1e-8
    D = pg.Euclidean(A.shape[0])
    L = pg.Additive(pg.Matrix(A, X, D), pg.GaussianMeasure(D, 1.0 * pg.Identity(D)))
    dec = pg.Problem(M=X, D=D, mdr=L, P=T.codomain, mpr=T, prior=pg.NormBall(X, 1.0), datum=D(d.values)).decomposition()
    assert dec.route == "dense"
    assert np.sort(dec.lam) == pytest.approx(np.linalg.eigvalsh(A @ A.T), rel=1e-10)


def test_bgp_scales_with_the_property():
    X, G, T, d = exp001()
    S, N = pg.NormBall(X, 1.0), pg.NormBall(G.codomain, 0.02)
    K = pg.update(S, pg.Additive(G, N), d)
    lo, hi = pg.push(K, T).interval(0)
    big = pg.Matrix(T.matrix() * 1e9, X, T.codomain)
    assert pg.push(pg.update(S, pg.Additive(G, N), d), big).interval(0) == pytest.approx((1e9 * lo, 1e9 * hi), rel=1e-6)
