"""Law tests for Gaussian measures: covariances, push-forwards, conditioning (exact and by sampling)."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.probability import legendre_decay, random_features


def test_standard_gaussian_of_a_data_space_is_its_noise():
    n = 6
    i = np.arange(n)
    Cd = 0.01 * np.exp(-np.abs(i[:, None] - i[None, :]) / 2.0)
    D = pg.Euclidean(n, covariance=Cd)
    X = D.standard_gaussian().sample(size=60000, seed=0).values
    assert np.abs(np.cov(X.T) - Cd).max() <= 0.03 * np.abs(Cd).max()          # coordinates N(0, C_d)
    u, v = D(np.ones(n)), D(np.arange(n) * 1.0)
    emp = np.mean((X @ D.apply_metric(u.values)) * (X @ D.apply_metric(v.values)))
    assert emp == pytest.approx(u @ v, rel=0.03)                               # E <x,u><x,v> = <u, v>: covariance I


def test_no_standard_gaussian_on_L2():
    with pytest.raises(ValueError):
        pg.GaussianMeasure(pg.L2("[0, 1]"))


def test_legendre_decay_on_an_interval_is_its_stated_covariance():
    M = pg.L2("[0, 1]")
    mu = legendre_decay(M, 0.7)
    u, v = M(np.sin), M(lambda x: x ** 2)
    S = mu.sample(size=20000, seed=1)
    a, b = M.inner(S, u), M.inner(S, v)
    assert np.mean(a * b) == pytest.approx(M.inner(mu.covariance(u), v), rel=0.05)
    assert mu.variance(u) == pytest.approx(np.var(a), rel=0.05)
    T, K = mu.covariance.trace(), len(mu.covariance.realised.lam)
    assert T.lower <= sum(0.7 ** (2 * k) for k in range(K)) * (1 + 1e-12) and T.contains(1 / (1 - 0.49))
    assert "trace class" in repr(mu.claims)


def test_pushforward_and_posterior_exact():
    M = pg.L2("[0, 1]")
    prior = legendre_decay(M, 0.7)
    n = 12
    ks = [np.exp(-((M.x - c) / 0.08) ** 2) for c in np.linspace(0.05, 0.95, n)]
    sigma = 0.003
    D = pg.Euclidean(n, covariance=sigma ** 2)
    G = pg.Functionals(ks, into=D)
    m_true = prior.sample(seed=2)
    d = G(m_true) + D.standard_gaussian().sample(seed=3)
    # push-forward: covariance of G x is B L B^T in coordinates (B = G psi)
    C = prior.covariance.realised                                              # the FiniteRank applied
    B = G(C.generators).values.T                                               # (n, N)
    Ld = np.diag(C.lam)
    assert np.allclose(G(prior).covariance.matrix(), B @ Ld @ B.T @ np.eye(n) / sigma ** 2, rtol=1e-10, atol=1e-14)
    # posterior in the generators' coefficients: x = sum_j c_j psi_j, c ~ N(0, L), d = B c + e, e ~ N(0, sigma^2 I)
    post = prior.condition(G, d)
    Lp = np.linalg.inv(np.linalg.inv(Ld) + B.T @ B / sigma ** 2)
    cm = Lp @ B.T @ d.values / sigma ** 2
    mean_ref = pg.FiniteRank(C.generators, np.ones(C.rank)).generators
    ref = M.sum([c * f for c, f in zip(cm, mean_ref)])
    assert post.mean.distance(ref) <= 1e-8 * ref.norm()
    u = M(np.cos)
    a = M.inner(C.generators, u)
    assert post.variance(u) == pytest.approx(a @ Lp @ a, rel=1e-8)
    # Matheron samples have the posterior moments
    S = post.sample(size=4000, seed=4)
    vals = M.inner(S, u)
    assert np.mean(vals) == pytest.approx(post.mean @ u, abs=4 * np.sqrt(a @ Lp @ a / 4000))
    assert np.var(vals) == pytest.approx(a @ Lp @ a, rel=0.1)


def test_measure_algebra():
    D = pg.Euclidean(3)
    mu, nu = D.standard_gaussian(), pg.GaussianMeasure(D, pg.Matrix(np.diag([1.0, 4.0, 9.0]), D, D))
    s = mu + nu
    X = s.sample(size=50000, seed=5).values
    assert np.allclose(np.cov(X.T), np.diag([2.0, 5.0, 10.0]), atol=0.15)
    shifted = mu + D([1.0, 2.0, 3.0])
    assert np.allclose(shifted.sample(size=50000, seed=6).values.mean(axis=0), [1, 2, 3], atol=0.03)
    A = pg.Matrix([[1.0, 1.0, 0.0]], domain=D)
    pa = A(nu)
    assert pa.variance(pg.Euclidean(1)([1.0])) == pytest.approx(5.0)


def test_random_features_on_a_ball():
    M = pg.L2(pg.Ball(1.0))
    mu = random_features(M, features=32)
    f = mu.sample(seed=0)
    assert np.isfinite(f.norm()) and f.norm() > 0
    assert mu.covariance.realised.rank == 32
