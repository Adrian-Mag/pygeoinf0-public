"""Data-space decompositions (plan 009 G3; EXP-030 DESIGN.md G3, NOTES.md section 4): every closed form against a
NumPy-only dense reference on random problems, BGP against an SLSQP primal and the per-direction route, the evidence
against a direct log-determinant, a* against bisection with direct solves, block routes against the dense one, the
Lanczos route, the cache on pg.Problem, and the ladder's discrepancy rule unchanged bit for bit."""
import math

import numpy as np
import pytest
from scipy.optimize import minimize

import pygeoinf0 as pg
from pygeoinf0.convex.feasible import FeasibleSet
from pygeoinf0.solvers.dataspace.decomposition import DataDecomposition


def problem(seed=0, n=7, m=12, p=2, dense_noise=False, weights=True):
    """M = R^m, G = A, T = B, data space with a diagonal noise metric (or a dense noise measure), datum d."""
    rng = np.random.default_rng(seed)
    M = pg.Euclidean(m)
    A, B = rng.normal(size=(n, m)), rng.normal(size=(p, m))
    s2 = rng.uniform(0.5, 2.0, n)
    if dense_noise:
        D = pg.Euclidean(n)
        Q = rng.normal(size=(n, n))
        Sig = Q @ Q.T / n + np.diag(s2)
        noise = pg.GaussianMeasure(D, pg.Matrix(Sig, D, D, name="C"))
    else:
        D = pg.Euclidean(n, covariance=s2) if weights else pg.Euclidean(n)
        Sig = np.diag(s2) if weights else np.eye(n)
        noise = D.standard_gaussian()
    G, T = pg.Matrix(A, M, D, name="G"), pg.Matrix(B, M, pg.Euclidean(p), name="T")
    d = D(rng.normal(size=n))
    return M, D, G, T, d, A, B, Sig, noise


def close(x, y, rel=1e-12, scale=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.max(np.abs(y)) if scale is None else scale
    return np.max(np.abs(x - y)) <= rel * max(s, 1e-300)


@pytest.mark.parametrize("dense_noise", [False, True])
def test_sola_closed_forms_against_dense_solves(dense_noise):
    M, D, G, T, d, A, B, Sig, noise = problem(1, dense_noise=dense_noise)
    dec = pg.decompose(G, noise)
    gam = np.array([1e-3, 0.1, 1.0, 30.0])
    res = dec.sola(T, gam, d)
    for i, g in enumerate(gam):
        X = B @ A.T @ np.linalg.inv(A @ A.T + g * Sig)
        R = B - X @ A
        ref = dict(estimate=X @ d.values, variance=np.diag(X @ Sig @ X.T), mismatch=np.diag(R @ R.T),
                   delivered=np.diag(X @ A @ A.T @ X.T))
        tt = np.diag(B @ B.T).max()
        for k, v in ref.items():
            assert close(getattr(res, k)[i], v, scale=max(np.abs(v).max(), tt if k != "estimate" else 0)), k
        Xm = dec.sola_operator(T, g).matrix()
        assert close(Xm, X)
    h = 1e-6                                                       # derivatives in gamma
    for k, dk in (("variance", "d_variance"), ("mismatch", "d_mismatch")):
        fd = (getattr(dec.sola(T, 1.0 + h, d), k) - getattr(dec.sola(T, 1.0 - h, d), k)) / (2 * h)
        assert np.allclose(getattr(res, dk)[2], fd, rtol=1e-6)
    assert np.allclose(res.d_mismatch / res.d_variance, -gam[:, None])     # d TRM / d variance = -gamma
    one = pg.inference.sola(pg.Additive(G, noise), T, d, gamma=0.1)        # the method: scalar gamma
    assert close(one.estimate.values, res.estimate[1]) and "decomposition" in repr(one.claims)
    curve = pg.inference.sola(pg.Additive(G, noise), T, d, gamma=gam)      # an array: the curve
    assert close(curve.std, res.std)


def test_sola_components_per_summand():
    rng = np.random.default_rng(3)
    X = pg.DirectSum({"a": pg.Euclidean(5), "b": pg.Euclidean(4)}, weights=[1.0, 2.5])
    A, B = rng.normal(size=(6, 9)), rng.normal(size=(1, 9))
    G = pg.Functionals(X.riesz([{"a": r[:5], "b": r[5:]} for r in A]))
    T = pg.Functionals(X.riesz([{"a": B[0, :5], "b": B[0, 5:]}]))
    d = G.codomain(rng.normal(size=6))
    dec = pg.decompose(G, G.codomain.standard_gaussian())
    res = dec.sola(T, [0.01, 1.0], d, fields=True)
    w = np.r_[np.ones(5), np.full(4, 2.5)]                      # representers: r_k / w_k in each summand
    Ag, Bt = A / w, B / w
    K = Ag @ np.diag(w) @ Ag.T
    for i, g in enumerate([0.01, 1.0]):
        x = np.linalg.solve(K + g * np.eye(6), Ag @ np.diag(w) @ Bt.T)       # weights of the delivered representer
        for key, sl in (("a", slice(0, 5)), ("b", slice(5, 9))):
            Wf = np.diag(w[sl])
            dl = x.T @ Ag[:, sl] @ Wf @ Ag[:, sl].T @ x
            r = Bt[:, sl] - x.T @ Ag[:, sl]
            assert close(res.components[key]["delivered"][i], np.diag(dl), scale=(Bt @ np.diag(w) @ Bt.T).item())
            assert close(res.components[key]["mismatch"][i], np.diag(r @ Wf @ r.T), scale=(Bt @ np.diag(w) @ Bt.T).item())
        tot = sum(res.components[k]["mismatch"][i] for k in "ab")
        assert np.allclose(tot, res.mismatch[i], rtol=1e-11)


def test_pli_posterior_for_any_scaling_and_the_method():
    M, D, G, T, d, A, B, Sig, noise = problem(2)
    rng = np.random.default_rng(5)
    Q = rng.normal(size=(12, 12))
    C1 = Q @ Q.T / 12 + 0.1 * np.eye(12)
    m0 = M(rng.normal(size=12))
    prior = pg.GaussianMeasure(M, pg.Matrix(C1, M, M, name="C1"), mean=m0)
    dec = pg.decompose(G, noise, middle=prior)
    a, nu = np.array([[0.5], [2.0]]), np.array([1.0, 3.0])
    res = dec.posterior(T, d, a=a, nu=nu, prior_mean=m0)
    assert res.mean.shape == (2, 2, 2) and res.covariance.shape == (2, 2, 2, 2)
    for i, av in enumerate(a[:, 0]):
        for j, nv in enumerate(nu):
            S = av ** 2 * A @ C1 @ A.T + nv ** 2 * Sig
            Kc = av ** 2 * B @ C1 @ A.T
            mean = B @ m0.values + Kc @ np.linalg.solve(S, d.values - A @ m0.values)
            cov = av ** 2 * B @ C1 @ B.T - Kc @ np.linalg.solve(S, Kc.T)
            assert close(res.mean[i, j], mean)
            assert close(res.covariance[i, j], cov, scale=np.abs(av ** 2 * B @ C1 @ B.T).max())
    post = pg.inference.pli(prior, pg.Additive(G, noise), T, d)               # the method: the same measure
    cell = pg.push(pg.update(prior, pg.Additive(G, noise), d), T)
    assert close(post.mean.values, cell.mean.values) and close(post.covariance.matrix(), cell.covariance.matrix())
    assert "decomposition" in repr(post.claims)


def bgp_problem(seed=4, n=6, m=10, r=2.0, rho=1.0):
    rng = np.random.default_rng(seed)
    M = pg.Euclidean(m)
    w = rng.uniform(0.5, 2.0, n)
    D = pg.Euclidean(n, weights=w)
    A = rng.normal(size=(n, m))
    G, T = pg.Matrix(A, M, D, name="G"), pg.Matrix(rng.normal(size=(2, m)), M, pg.Euclidean(2), name="T")
    m0 = M(0.1 * rng.normal(size=m))
    d = D(A @ rng.normal(size=m) * 0.3 + 0.2 * rng.normal(size=n))
    return M, D, G, T, d, m0, w, pg.NormBall(M, r, centre=m0), pg.NormBall(D, rho)


def test_bgp_through_the_decomposition_against_slsqp_and_the_per_direction_route():
    M, D, G, T, d, m0, w, S, N = bgp_problem()
    A = G.A
    new = pg.push(FeasibleSet(S, G, N, d), T)
    old = pg.push(FeasibleSet(S, G, N, d, decomposition=False), T)
    assert new.K._spectral() is not None and old.K._spectral() is None
    for k in range(2):
        for b_new, b_old in zip(new.bounds(k), old.bounds(k)):
            assert b_new.status == "EXACT" and b_new.value == pytest.approx(b_old.value, rel=1e-9, abs=1e-12)
            assert b_new.lower - 1e-12 <= b_new.value <= b_new.upper + 1e-12
    for q in (np.array([1.0, 0.0]), np.array([0.3, -1.2])):
        c = T.A.T @ q
        cons = [dict(type="ineq", fun=lambda x: 2.0 ** 2 - (x - m0.values) @ (x - m0.values)),
                dict(type="ineq", fun=lambda x: 1.0 - (d.values - A @ x) @ (w * (d.values - A @ x)))]
        best = -math.inf
        for x0 in (m0.values, np.zeros(10)):
            res = minimize(lambda x: -(c @ x), x0, jac=lambda x: -c, constraints=cons, method="SLSQP",
                           options=dict(ftol=1e-15, maxiter=2000))
            if all(cc["fun"](res.x) >= -1e-9 for cc in cons):
                best = max(best, -res.fun)
        b = new.bound(q)
        assert b.value == pytest.approx(best, rel=1e-7)
        mu = D.apply_metric(b.certificate)                     # the plain multiplier; its dual value by a second formula
        phi = c @ m0.values + mu @ (d.values - A @ m0.values) + 2.0 * np.linalg.norm(c - A.T @ mu) \
            + 1.0 * math.sqrt(mu @ (mu / w))
        assert phi == pytest.approx(b.upper, rel=1e-10) and phi >= best - 1e-9
    # the problem's decompositions are shared by every bound
    dec = new.K._spectral().dec
    assert len(dec._targets) == 1 and dec.route == "dense"


def test_bgp_empty_and_no_prior_through_the_decomposition():
    M, D, G, T, d, m0, w, S, N = bgp_problem(rho=0.5)
    K = FeasibleSet(pg.NormBall(M, 0.01), G, pg.NormBall(D, 1e-3), d)
    assert K.is_empty() is pg.Truth.TRUE
    mu = D.apply_metric(K.emptiness.certificate)
    A = G.A
    assert mu @ d.values + 0.01 * np.linalg.norm(A.T @ mu) + 1e-3 * math.sqrt(mu @ (mu / w)) < 0
    free = pg.push(FeasibleSet(pg.Whole(M), G, pg.NormBall(D, 0.5), d), T)
    assert free.bound(np.array([1.0, 0.0])).status == "UNBOUNDED"           # m > n: T is not estimable
    Gm = pg.Matrix(A, M, D)
    Te = pg.Matrix(A[:1], M, pg.Euclidean(1))                                # an estimable property: a datum
    lo, hi = pg.push(FeasibleSet(pg.Whole(M), Gm, pg.NormBall(D, 0.5), d), Te).bounds(0)
    lo2, hi2 = pg.push(FeasibleSet(pg.Whole(M), Gm, pg.NormBall(D, 0.5), d, decomposition=False), Te).bounds(0)
    assert (lo.value, hi.value) == pytest.approx((lo2.value, hi2.value), rel=1e-9)
    assert hi.value == pytest.approx(d.values[0] + 0.5 / math.sqrt(w[0]), rel=1e-9)


def test_ellipsoid_bound_and_feasibility_against_direct_solves():
    M, D, G, T, d, A, B, Sig, noise = problem(6, n=9, m=5)            # more data than unknowns: a misfit floor
    dec = pg.decompose(G, noise)
    Si = np.linalg.inv(Sig)
    F = np.linalg.cholesky(Si).T                                        # F^T F = Sigma^-1

    def rho_min(a):
        """min ||F (d - A m)|| over ||m|| <= a by bisection on the multiplier, with direct solves."""
        def m_of(mu):
            return np.linalg.solve(A.T @ Si @ A + mu * np.eye(5), A.T @ Si @ d.values)
        if np.linalg.norm(m_of(0.0)) <= a:
            m = m_of(0.0)
        else:
            lo, hi = 0.0, 1.0
            while np.linalg.norm(m_of(hi)) > a:
                hi *= 2
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if np.linalg.norm(m_of(mid)) > a else (lo, mid)
            m = m_of(hi)
        return np.linalg.norm(F @ (d.values - A @ m))
    floor = rho_min(1e6)
    top = np.linalg.norm(F @ d.values)
    fe = dec.feasibility(d, a=[0.05, 0.2, 1e6], r=[0.5 * (floor + top), 0.9 * floor, 2 * top])
    for a, v in zip([0.05, 0.2, 1e6], fe.rho_min):
        assert v == pytest.approx(rho_min(a), rel=1e-10)
    assert fe.floor == pytest.approx(floor, rel=1e-10)
    r = 0.5 * (floor + top)
    lo, hi = 0.0, 1e3                                                  # a* by bisection with direct solves
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rho_min(mid) > r else (lo, mid)
    assert fe.a_star[0] == pytest.approx(hi, rel=1e-9)
    assert fe.a_star[1] == math.inf and fe.a_star[2] == 0.0
    # the ellipsoid bound with a = a* + margin is finite; the empty side has a Farkas certificate
    q = np.array([1.0, 0.0])
    ok = dec.ellipsoid_bound(T, a=1.5 * hi, r=r, datum=d, q=q)
    no = dec.ellipsoid_bound(T, a=0.5 * hi, r=r, datum=d, q=q)
    assert ok.status == "EXACT" and no.status == "EMPTY"


def test_evidence_against_a_direct_log_determinant_and_its_gradient():
    from scipy.stats import multivariate_normal
    M, D, G, T, d, A, B, Sig, noise = problem(7, dense_noise=True)
    rng = np.random.default_rng(8)
    Q = rng.normal(size=(12, 12))
    C1 = Q @ Q.T / 12 + 0.1 * np.eye(12)
    dec = pg.decompose(G, noise, middle=pg.Matrix(C1, M, M))
    for a, nu in ((0.7, 1.3), (2.0, 0.4)):
        ref = multivariate_normal(np.zeros(7), a * a * A @ C1 @ A.T + nu * nu * Sig).logpdf(d.values)
        out = dec.evidence(d, a=a, nu=nu, gradient=True)
        assert out.log_evidence == pytest.approx(ref, rel=1e-12)
        h = 1e-6
        fa = (dec.evidence(d, a + h, nu).log_evidence - dec.evidence(d, a - h, nu).log_evidence) / (2 * h)
        fn = (dec.evidence(d, a, nu + h).log_evidence - dec.evidence(d, a, nu - h).log_evidence) / (2 * h)
        assert out.d_a == pytest.approx(fa, rel=1e-6) and out.d_nu == pytest.approx(fn, rel=1e-6)
    grid = dec.evidence(d, a=np.array([0.7, 2.0]), nu=np.array([1.3, 0.4]))
    assert grid.log_evidence.shape == (2,)


def test_discrepancy_and_predictive_against_dense_formulas():
    M, D, G, T, d, A, B, Sig, noise = problem(9)
    dec = pg.decompose(G, noise)
    alpha = dec.damping_for_misfit(d, D.dim)
    assert float(alpha) == pytest.approx(float(pg.solvers.damping_for_misfit(G @ G.adjoint, d, D.dim)), rel=1e-10)
    r = alpha * Sig @ np.linalg.solve(A @ A.T + alpha * Sig, d.values)    # the damped residual, chi^2 = n
    assert r @ np.linalg.solve(Sig, r) == pytest.approx(7, rel=1e-8)
    a, nu = 0.8, 1.1
    Cov = a * a * A @ A.T + nu * nu * Sig
    H = a * a * A @ A.T @ np.linalg.inv(Cov)
    pr = dec.predictive(d, a=a, nu=nu)
    assert pr.chi2 == pytest.approx(d.values @ np.linalg.solve(Cov, d.values), rel=1e-12)
    assert pr.trace_hat == pytest.approx(np.trace(H), rel=1e-12)
    res = d.values - H @ d.values                                         # the posterior mean's predicted residual
    assert pr.residual_chi2 == pytest.approx(res @ np.linalg.solve(nu * nu * Sig, res), rel=1e-11)
    assert np.sum(pr.residual_weights) == pytest.approx(7 - pr.trace_hat, rel=1e-12)


def test_the_ladder_discrepancy_is_unchanged_bit_for_bit():
    """solvers.damping_for_misfit (the root of the discrepancy principle, solvers.discrepancy before plan 070) runs
    through a decomposition: the same operations, so the same bits (plan 009); and so does the meta rule."""
    from pygeoinf0.solvers import solve_damping
    from pygeoinf0.solvers.direct import orthonormal_form, _square_matrix
    import ladder
    for name in ("L1", "L2"):
        P = ladder.build(name, 40)
        G = pg.Functionals(P.H.riesz(P.measures), into=P.D)
        A = G @ G.adjoint
        S = orthonormal_form(A, _square_matrix(A))                      # the rule as it was written before plan 009
        lam, U = np.linalg.eigh(0.5 * (S + S.T))
        old, _ = solve_damping(lam, (U.T @ P.D.to_orthonormal(P.d.values)) ** 2, P.D.dim)
        assert float(pg.solvers.damping_for_misfit(A, P.d, P.D.dim)) == old
        assert float(pg.meta.discrepancy(pg.Problem(mdr=G, datum=P.d)).gamma) == old


def test_the_ladder_rule_and_posterior_from_one_decomposition():
    import ladder
    P = ladder.build("L1", 40)
    G, alpha, post = ladder.solve(P)
    dec = pg.decompose(G)                                                 # noise: the data space's own
    assert float(dec.damping_for_misfit(P.d, P.D.dim)) == pytest.approx(float(alpha), rel=1e-10)
    T = pg.Functionals(P.H.riesz(P.measures[:3]))
    res = dec.posterior(T, P.d, a=1.0 / math.sqrt(alpha), nu=1.0)
    assert np.allclose(res.std, post.std(P.measures[:3]), rtol=1e-9)


def blocked(seed=10, sizes=(3, 4, 2, 4), m_per=5):
    """A problem whose data blocks see disjoint parts of the model: K = A A^T is block diagonal (rows shuffled)."""
    rng = np.random.default_rng(seed)
    m = m_per * len(sizes)
    rows = []
    for b, k in enumerate(sizes):
        R = np.zeros((k, m))
        R[:, b * m_per:(b + 1) * m_per] = rng.normal(size=(k, m_per))
        rows.append(R)
    A = np.vstack(rows)
    perm = rng.permutation(len(A))
    A = A[perm]
    M = pg.Euclidean(m)
    s2 = rng.uniform(0.5, 2.0, len(A))
    D = pg.Euclidean(len(A), covariance=s2)
    G, T = pg.Matrix(A, M, D), pg.Matrix(rng.normal(size=(2, m)), M, pg.Euclidean(2))
    return M, D, G, T, D(rng.normal(size=len(A))), A, s2


def test_blocks_found_from_values_agree_with_the_dense_route():
    M, D, G, T, d, A, s2 = blocked()
    auto, one = pg.decompose(G), pg.decompose(G, blocks=None)
    assert auto.route == "blocks" and len(auto._rows) == 4 and one.route == "dense"
    assert any(c.status == "checked" and "found from the values" in c.statement for c in auto.claims)
    g = np.array([0.01, 1.0])
    for k in ("estimate", "variance", "mismatch", "delivered"):
        assert close(getattr(auto.sola(T, g, d), k), getattr(one.sola(T, g, d), k), rel=1e-11)
    assert close(auto.posterior(T, d, a=0.7).covariance, one.posterior(T, d, a=0.7).covariance, rel=1e-11)
    assert auto.evidence(d, 0.7, 1.2).log_evidence == pytest.approx(one.evidence(d, 0.7, 1.2).log_evidence, rel=1e-12)
    S, N = pg.NormBall(M, 1.5), pg.NormBall(D, 3.0)
    lo, hi = pg.push(FeasibleSet(S, G, N, d), T).bounds(1)
    lo2, hi2 = pg.push(FeasibleSet(S, G, N, d, decomposition=False), T).bounds(1)
    assert (lo.value, hi.value) == pytest.approx((lo2.value, hi2.value), rel=1e-9)


def test_structured_gram_with_shared_blocks_and_noise_that_depends_on_the_order():
    """Rotation-invariant Grams repeat across orders t (shared blocks), the noise does not: Gamma's blocks differ,
    their Gram parts are equal, and every closed form matches the dense matrices (EXP-030 DESIGN G2)."""
    rng = np.random.default_rng(11)
    Ks = []
    for m in (3, 2):                                     # two "degrees", one Gram block each
        Q = rng.normal(size=(m, m + 2))
        Ks.append(Q @ Q.T)
    shared = [0, 0, 0, 1, 1]                             # degree 0 for three orders, degree 1 for two
    n = 3 * 3 + 2 * 2
    order = rng.permutation(n)                           # the blocks interleave in the data's own order
    D = pg.Euclidean(n)
    K = pg.BlockDiagonal(D, Ks, shared=shared, order=order, name="K")
    Kd = K.matrix()
    for b, part in enumerate(K.partition):               # the stored blocks are the dense matrix's blocks
        assert np.array_equal(Kd[np.ix_(part, part)], Ks[shared[b]])
    s2 = rng.uniform(0.5, 2.0, n)                        # t-dependent errors
    Gamma = K + pg.Diagonal(s2, D)
    assert isinstance(Gamma, pg.BlockDiagonal) and len(Gamma.matrices) == 2
    blocks = [Gamma.block(b) for b in range(5)]
    assert not np.allclose(blocks[0], blocks[1])         # distinct Gamma blocks ...
    assert np.array_equal(blocks[0] - np.diag(np.diag(blocks[0])), blocks[1] - np.diag(np.diag(blocks[1])))
    assert np.allclose(Gamma.matrix(), Kd + np.diag(s2), rtol=0, atol=1e-14)     # ... with equal Gram parts
    v = rng.normal(size=(n, 3))
    assert np.allclose(Gamma(D.family(v.T)).values.T, (Kd + np.diag(s2)) @ v, rtol=1e-13)
    assert np.allclose(Gamma.adjoint.matrix(), Gamma.matrix().T)
    cross = rng.normal(size=(n, 1))
    tt = np.array([[(cross.T @ np.linalg.pinv(Kd) @ cross).item() + 1.0]])
    d = rng.normal(size=n)
    t_dep = DataDecomposition(gram=K, metric=s2, space=D)
    t_ind = DataDecomposition(gram=K, metric=np.ones(n), space=D)
    assert t_dep.route == "blocks" and len(t_dep._rows) == 5
    assert "5 distinct" in repr(t_dep.claims) and "2 distinct" in repr(t_ind.claims)
    g = np.array([0.1, 2.0])
    res = t_dep.sola(gamma=g, datum=d, pairings=(cross, tt))
    for i, gv in enumerate(g):
        x = np.linalg.solve(Kd + gv * np.diag(s2), cross)
        assert close(res.estimate[i], x.T @ d) and close(res.variance[i], x.T @ np.diag(s2) @ x)
        assert close(res.mismatch[i], tt[0] - 2 * cross.T @ x + x.T @ Kd @ x, scale=tt[0, 0])
    bd = t_dep.ellipsoid_bound(a=3.0, r=math.sqrt(n), datum=d, q=np.array([1.0]), pairings=(cross, tt))
    dd = DataDecomposition(gram=Kd, metric=s2, space=D, blocks=None)
    assert dd.ellipsoid_bound(a=3.0, r=math.sqrt(n), datum=d, q=np.array([1.0]), pairings=(cross, tt)).value \
        == pytest.approx(bd.value, rel=1e-10)


def test_block_diagonal_detect_and_round_trip():
    rng = np.random.default_rng(12)
    M, D, G, T, d, A, s2 = blocked(12)
    Kd = A @ A.T
    B = pg.BlockDiagonal.detect(Kd)
    assert B.n_blocks == 4 and np.array_equal(B.matrix(), Kd)
    assert B.claims.find("block diagonal").status == "checked"
    v = rng.normal(size=len(Kd))
    assert np.allclose(B(pg.Euclidean(len(Kd))(v)).values, Kd @ v, rtol=1e-13)


def test_lanczos_route_above_dense_max():
    rng = np.random.default_rng(13)
    n, m = 60, 30                                       # rank 30: the rest of the spectrum is exactly 0
    M, D = pg.Euclidean(m), pg.Euclidean(n, weights=rng.uniform(0.5, 2, n))
    G, T = pg.Matrix(rng.normal(size=(n, m)), M, D), pg.Matrix(rng.normal(size=(2, m)), M, pg.Euclidean(2))
    d = D(rng.normal(size=n))
    full = pg.decompose(G)
    with pg.settings.using(dense_max=40):
        part = pg.decompose(G, tail=1e-8, route="lanczos")
    assert part.partial and part.route == "lanczos" and full.route == "dense"
    assert any("Lanczos" in c.statement and c.status == "checked" for c in part.claims)
    g = np.array([0.01, 1.0])
    for k in ("estimate", "variance", "mismatch"):
        assert close(getattr(part.sola(T, g, d), k), getattr(full.sola(T, g, d), k), rel=1e-8)
    assert part.evidence(d, 0.5, 1.0).log_evidence == pytest.approx(full.evidence(d, 0.5, 1.0).log_evidence, rel=1e-10)
    assert close(part.sola_operator(T, 0.3).matrix(), full.sola_operator(T, 0.3).matrix(), rel=1e-8)
    with pytest.raises(pg.errors.Unsupported):
        part.feasibility(d, a=1.0)


def test_problem_cache_shares_one_decomposition_across_methods():
    M, D, G, T, d, m0, w, S, N = bgp_problem(14)
    L = pg.Additive(G, D.standard_gaussian())
    pb = pg.Problem(M=M, D=D, mdr=L, P=T.codomain, mpr=T, datum=d)
    curve = pb.run("sola", gamma=np.logspace(-2, 2, 9))
    assert curve.std.shape == (9, 2) and len(pb._cache) == 1
    pb2 = pb.with_(mdr=L.acceptance(0.05), prior=S)
    out = pb2.run("bgp")
    assert len(pb._cache) == 1 and pb2._cache is pb._cache            # the norm ball of D: the same whitening
    assert "decomposition" in out.bounds[0][1].note
    dec = pb2.decomposition()
    assert dec is next(iter(pb._cache.values()))
    T1 = pg.Matrix(T.A[:1], M, pg.Euclidean(1), name="T1")              # oCSOLA: one property, SOLA's X for gamma
    pb3 = pb.with_(mpr=T1, P=T1.codomain, prior=S)
    closed = pb3.run("ocsola", gamma=0.5)
    X = pg.inference.sola(L, T1, d, gamma=0.5).X
    assert closed == pytest.approx(pg.inference.ocsola(S, L, T1, d, X=X), rel=1e-8)
    many = pb3.run("ocsola", gamma=[0.5, 2.0])
    assert many[0][0] == pytest.approx(closed[0], rel=1e-12) and many[1].shape == (2,)


def test_blocks_run_in_worker_processes_above_the_threshold(monkeypatch):
    import pygeoinf0.solvers.dataspace.decomposition as dmod
    M, D, G, T, d, A, s2 = blocked(15)
    serial = pg.decompose(G)
    monkeypatch.setattr(dmod, "PARALLEL_FLOPS", 0.0)
    with pg.settings.using(workers=2):
        par = pg.decompose(G)
    assert any("worker processes" in c.statement for c in par.claims)
    assert np.array_equal(par.lam, serial.lam)
    g = np.array([0.1, 1.0])
    assert close(par.sola(T, g, d).estimate, serial.sola(T, g, d).estimate, rel=1e-14)


def test_feasibility_resolves_blocks_whose_scales_differ_by_many_orders():
    """Plan 009 P4a: two blocks of a Gram matrix with scales 1e10 apart (as the CMB problem's whitened blocks, condition
    up to 1e15). A cut relative to the global largest eigenvalue would treat the small block as unseen by the data;
    the cut is relative to each block (BGP's SpectralBall uses the same mask). Reference: the dense lambda-bisection
    of the thesis code for rho_min, NumPy only."""
    rng = np.random.default_rng(7)
    n1, n2 = 6, 5
    B = np.zeros((n1 + n2, n1 + n2 + 4))                       # K = B B^T: G = B on a model space R^(n + 4)
    B[:n1, :n1 + 2] = 1e5 * rng.normal(size=(n1, n1 + 2))
    B[n1:, n1 + 2:] = 0.01 * rng.normal(size=(n2, n2 + 2))
    K = B @ B.T
    n = n1 + n2
    D = pg.Euclidean(n)
    d = rng.normal(size=n) * np.r_[1e5 * np.ones(n1), np.ones(n2)]
    dec = DataDecomposition(gram=K, metric=np.ones(n), space=D)
    assert dec.route == "blocks"

    def fit(lam):
        z = np.linalg.solve(lam * np.eye(n) + K, d)
        return float(z @ K @ z), float(np.linalg.norm(d - K @ z))
    for a in (0.5, 3.0):
        lo, hi = 1e-14, 1e14
        while hi / lo > 1 + 1e-14:
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if fit(mid)[0] > a * a else (lo, mid)
        ref = fit(math.sqrt(lo * hi))[1]
        got = float(np.atleast_1d(dec.feasibility(d, a=np.array([a])).rho_min)[0])
        assert got == pytest.approx(ref, rel=1e-9)
    from pygeoinf0.optimisers.dual.spectral import RCOND
    assert dec.positive(RCOND).all()                                         # every eigenvalue of both blocks counts
    assert not (dec.lam > 1e-13 * dec.lam.max()).all()                       # a cut relative to the global max would not
