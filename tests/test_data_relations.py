"""Data-space relations (plan 009 G9; EXP-030 DESIGN.md G9, NOTES.md section 9): restriction to some of the data is
exact deletion (against a dense reference with the rows deleted, and the limit of inflated noise), known offsets are
shifted data, labelled data spaces round-trip through selection and restriction, and noise families are fitted by
the Gaussian evidence as a meta record."""
import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import pygeoinf0 as pg
from pygeoinf0.meta import NoiseFamily, fit_noise


def problem(seed=0, n=8, m=10):
    rng = np.random.default_rng(seed)
    M = pg.Euclidean(m)
    s2 = rng.uniform(0.5, 2.0, n)
    D = pg.Euclidean(n, covariance=s2, labels={"mode": [f"m{i // 2}" for i in range(n)], "s": [2, 2, 4, 4] * (n // 4),
                                               "t": [0, 1] * (n // 2)})
    A, B = rng.normal(size=(n, m)), rng.normal(size=(2, m))
    Q = rng.normal(size=(m, m))
    C1 = Q @ Q.T / m + 0.2 * np.eye(m)
    prior = pg.GaussianMeasure(M, pg.Matrix(C1, M, M, name="C1"))
    G, T = pg.Matrix(A, M, D, name="G"), pg.Matrix(B, M, pg.Euclidean(2), name="T")
    return M, D, G, T, D(rng.normal(size=n)), A, B, C1, s2, prior


def dense_posterior(A, B, C1, Sig, d):
    S = A @ C1 @ A.T + Sig
    K = B @ C1 @ A.T
    return K @ np.linalg.solve(S, d), B @ C1 @ B.T - K @ np.linalg.solve(S, K.T)


def test_restriction_is_exact_deletion():
    M, D, G, T, d, A, B, C1, s2, prior = problem()
    rows = np.array([0, 2, 3, 6])
    L = pg.Additive(G, D.standard_gaussian())
    LJ = L.select(rows)
    DJ = LJ.codomain
    assert DJ.dim == 4 and np.allclose(DJ.weights, 1 / s2[rows])
    assert LJ.claims.find("selection").status == "derived"
    dJ = DJ(d.values[rows])
    mean, cov = dense_posterior(A[rows], B, C1, np.diag(s2[rows]), d.values[rows])
    for post in (pg.inference.pli(prior, LJ, T, dJ), pg.push(pg.update(prior, LJ, dJ), T)):
        assert np.allclose(post.mean.values, mean, rtol=1e-12, atol=1e-13)
        assert np.allclose(post.covariance.matrix(), cov, rtol=1e-12, atol=1e-13)
    by_label = L.select(lambda lab: (lab.t == 0 and lab.mode != "m2") or lab.mode == "m1")   # the same rows
    assert np.array_equal(by_label.rows, rows) and by_label.codomain.labels == tuple(D.labels[i] for i in rows)


def test_inflated_noise_converges_to_deletion_as_f_to_the_minus_two():
    M, D, G, T, d, A, B, C1, s2, prior = problem(1)
    keep = np.array([1, 2, 5, 7])
    out = np.setdiff1d(np.arange(8), keep)
    mean, cov = dense_posterior(A[keep], B, C1, np.diag(s2[keep]), d.values[keep])
    errs = []
    for f in (1e2, 1e3, 1e6):
        s2f = s2.copy()
        s2f[out] *= f * f
        Df = pg.Euclidean(8, covariance=s2f)
        post = pg.inference.pli(prior, pg.Additive(pg.Matrix(A, M, Df), Df.standard_gaussian()), T, Df(d.values))
        errs.append(max(np.abs(post.mean.values - mean).max(), np.abs(post.covariance.matrix() - cov).max()))
    assert 0.3e-2 < errs[1] / errs[0] < 3e-2                                 # O(f^-2)
    assert errs[2] < 1e-9 * max(np.abs(mean).max(), np.abs(cov).max())       # the thesis's sigma x 1e6 (pli:149)
    LJ = pg.Additive(G, D.standard_gaussian()).select(keep)
    post = pg.inference.pli(prior, LJ, T, LJ.codomain(d.values[keep]))
    assert np.allclose(post.mean.values, mean, rtol=1e-12, atol=1e-13)


def test_restricted_sets_project_their_noise():
    M, D, G, T, d, A, B, C1, s2, prior = problem(2)
    rows = [1, 4, 5]
    R = pg.Additive(G, pg.NormBall(D, 2.0)).select(rows)
    DJ = R.codomain
    direct = pg.Additive(pg.Matrix(A[rows], M, pg.Euclidean(3, weights=1 / s2[rows])),
                         pg.NormBall(pg.Euclidean(3, weights=1 / s2[rows]), 2.0))
    S = pg.NormBall(M, 3.0)
    one = pg.push(pg.update(S, R, DJ(d.values[rows])), T).interval(1)
    two = pg.push(pg.update(S, direct, direct.codomain(d.values[rows])), T).interval(1)
    assert one == pytest.approx(two, rel=1e-10)
    Cd = np.cov(np.random.default_rng(3).normal(size=(8, 40)))                # a dense ellipsoid: its marginal shape
    E = pg.Ellipsoid(pg.Euclidean(8), 1.5, covariance=Cd)
    EJ = pg.Additive(pg.Matrix(A, M, pg.Euclidean(8)), E).select(rows).noise
    assert np.allclose(EJ.shape, Cd[np.ix_(rows, rows)]) and EJ.radius == 1.5


def test_a_known_offset_is_a_shifted_datum():
    M, D, G, T, d, A, B, C1, s2, prior = problem(4)
    b = D(np.random.default_rng(5).normal(size=8), name="crust")
    Lb, L = pg.Additive(G, D.standard_gaussian(), offset=b), pg.Additive(G, D.standard_gaussian())
    assert Lb.claims.find("offset").status == "asserted" and Lb.offset is b
    one, two = pg.inference.pli(prior, Lb, T, d), pg.inference.pli(prior, L, T, d - b)
    assert np.allclose(one.mean.values, two.mean.values, rtol=1e-12, atol=1e-14)
    cell = pg.push(pg.update(prior, Lb, d), T)
    assert np.allclose(cell.mean.values, two.mean.values, rtol=1e-12, atol=1e-14)
    s1, s2_ = pg.inference.sola(Lb, T, d, gamma=0.3), pg.inference.sola(L, T, d - b, gamma=0.3)
    assert np.allclose(s1.estimate.values, s2_.estimate.values, rtol=1e-12)
    S = pg.NormBall(M, 30.0)
    for noise in (pg.NormBall(D, 2.5), None):                                # a noise set, and exact data
        Rb, R = pg.Additive(G, noise, offset=b), pg.Additive(G, noise)
        u = pg.push(pg.update(S, Rb, d), T).interval(0)
        v = pg.push(pg.update(S, R, d - b), T).interval(0)
        assert u == pytest.approx(v, rel=1e-9)
    rows = [0, 3, 4, 7]                                                      # restriction keeps the offset
    post = pg.inference.pli(prior, Lb.select(rows), T, Lb.select(rows).codomain(d.values[rows]))
    mean, cov = dense_posterior(A[rows], B, C1, np.diag(s2[rows]), d.values[rows] - b.values[rows])
    assert np.allclose(post.mean.values, mean, rtol=1e-12, atol=1e-13)


def test_labels_round_trip():
    lab = {"mode": ["0S2", "0S2", "1S2", "1S2", "0S3", "0S3"], "s": [2, 2, 2, 2, 3, 3], "t": [0, 1, 0, 1, 0, 1]}
    D = pg.Euclidean(6, labels=lab)
    assert D.labels[3] == ("1S2", 2, 1) and D.labels[3].s == 2 and D.fields == ("mode", "s", "t")
    assert list(D.select(s=2)) == [0, 1, 2, 3] and list(D.select(lambda l: l.t == 1, s=[2, 3])) == [1, 3, 5]
    assert list(D.select(mode=lambda m: m.startswith("0"))) == [0, 1, 4, 5]
    g = D.groups("s", "t")
    assert list(g) == [(2, 0), (2, 1), (3, 0), (3, 1)] and list(g[(2, 1)]) == [1, 3]
    E = D.with_covariance(np.full(6, 2.0))
    assert E.labels == D.labels and E.standard is D
    assert pg.Euclidean(6) == D                                              # unlabelled spaces ignore labels
    assert D != pg.Euclidean(6, labels=list("abcdef"))
    G = pg.Matrix(np.eye(6), pg.Euclidean(6), D)
    GJ = G.select(dict(s=2, t=1))
    assert GJ.codomain.labels == (D.labels[1], D.labels[3]) and list(GJ.codomain.select(mode="1S2")) == [1]
    assert np.array_equal(GJ.matrix(), np.eye(6)[[1, 3]])
    with pytest.raises(ValueError):
        pg.Euclidean(3, labels=["a", "a", "b"])


def fit_problem(seed=6, blocks=6, rows=40, per=30, theta=(1.5, 0.7)):
    """Data blocks that see disjoint parts of the model, in two groups (label s) with their own noise multiplier."""
    rng = np.random.default_rng(seed)
    n, m = blocks * rows, blocks * per
    A = np.zeros((n, m))
    for b in range(blocks):
        A[b * rows:(b + 1) * rows, b * per:(b + 1) * per] = rng.normal(size=(rows, per)) / math.sqrt(per)
    s = np.repeat([2 if b % 2 == 0 else 3 for b in range(blocks)], rows)
    D = pg.Euclidean(n, labels={"s": s, "i": np.arange(n)})
    M = pg.Euclidean(m)
    G = pg.Matrix(A, M, D, name="G")
    th = np.where(s == 2, theta[0], theta[1])
    d = A @ rng.normal(size=m) + th * rng.normal(size=n)
    return M, D, G, A, D(d), th


def test_fit_noise_by_the_closed_form_evidence():
    M, D, G, A, d, th = fit_problem()
    prior = pg.GaussianMeasure(M, pg.Identity(M))
    F = NoiseFamily(D, sigma=1.0, groups="s")
    fit = fit_noise(pg.Problem(mdr=G, prior=prior, datum=d), family=F)
    assert fit.route == "closed" and set(fit.theta) == {"theta[(2,)]", "theta[(3,)]"}
    assert fit.theta["theta[(2,)]"] == pytest.approx(1.5, rel=0.25)
    assert fit.theta["theta[(3,)]"] == pytest.approx(0.7, rel=0.25)
    assert fit.claims.find("data-dependent rule").status == "asserted"
    Cov = fit.a ** 2 * A @ A.T + np.diag(F.covariance(fit.theta_vector))
    ref = multivariate_normal(np.zeros(D.dim), Cov).logpdf(d.values)
    assert fit.log_evidence == pytest.approx(ref, rel=1e-10)
    h = 1e-5                                                                 # a stationary point of the direct evidence
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        up = multivariate_normal(np.zeros(D.dim), fit.a ** 2 * A @ A.T + np.diag(F.covariance(fit.theta_vector * (1 + e))))
        dn = multivariate_normal(np.zeros(D.dim), fit.a ** 2 * A @ A.T + np.diag(F.covariance(fit.theta_vector * (1 - e))))
        assert abs(up.logpdf(d.values) - dn.logpdf(d.values)) / (2 * h) < 1e-4 * D.dim
    mu = fit.measure()
    assert mu.space == D and np.allclose(np.diag(mu.covariance.matrix()), F.covariance(fit.theta_vector))
    alt = NoiseFamily(D, sigma=1.0, groups=[np.arange(0, D.dim, 2), np.arange(1, D.dim, 2)])   # splits the blocks
    fd = fit_noise(pg.Problem(mdr=G, prior=prior, datum=d), family=alt, a=fit.a)
    assert fd.route == "dense" and fd.a == fit.a
    Cov = fit.a ** 2 * A @ A.T + np.diag(alt.covariance(fd.theta_vector))
    assert fd.log_evidence == pytest.approx(multivariate_normal(np.zeros(D.dim), Cov).logpdf(d.values), rel=1e-10)
