"""Law tests for weighted Euclidean spaces: the metric is an operator on the standard R^n."""
import numpy as np
import pytest
import scipy.sparse as sps

import pygeoinf0 as pg


def _spd(n, seed):
    A = np.random.default_rng(seed).standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _spaces():
    """(name, space, dense W) for every kind of metric, on R^12 (a 3 x 4 lattice for the Kronecker case)."""
    n = 12
    E = pg.Euclidean(n)
    w = np.linspace(1, 3, n)
    W = _spd(n, 0)
    T = sps.diags([np.full(n - 1, -1.0), np.full(n, 4.0), np.full(n - 1, -1.0)], [-1, 0, 1]).tocsr()
    A1, A2 = _spd(3, 1), _spd(4, 2)
    K = pg.KroneckerMatrix.from_factors([A1, A2], space=E)
    C = _spd(n, 3)
    return [
        ("weights", pg.Euclidean(n, weights=w), np.diag(w)),
        ("diagonal operator", E.with_metric(pg.Diagonal(w, E)), np.diag(w)),
        ("dense", pg.Euclidean(n, metric=W), W),
        ("sparse", pg.Euclidean(n, metric=T), T.toarray()),
        ("kronecker", E.with_metric(K), np.kron(A1, A2)),
        ("covariance dense", pg.Euclidean(n, covariance=C), np.linalg.inv(C)),
        ("covariance scalar", pg.Euclidean(n, covariance=0.04), np.eye(n) / 0.04),
        ("covariance vector", pg.Euclidean(n, covariance=w), np.diag(1 / w)),
        ("metric = C.inv", E.with_metric(pg.Matrix(C, E, E).inv), np.linalg.inv(C)),
    ]


CASES = _spaces()


@pytest.mark.parametrize("name, D, W", CASES, ids=[c[0] for c in CASES])
def test_inner_product_and_metric_operations(name, D, W):
    rng = np.random.default_rng(5)
    x, y = rng.standard_normal(D.dim), rng.standard_normal(D.dim)
    assert D(x) @ D(y) == pytest.approx(x @ W @ y, rel=1e-11)
    V = rng.standard_normal((D.dim, 3))
    assert np.allclose(D.apply_metric(V), W @ V, rtol=1e-11, atol=1e-11 * np.abs(W @ V).max())
    assert np.allclose(D.solve_metric(D.apply_metric(V)), V, rtol=1e-9, atol=1e-10)
    assert D.claims[1].status in ("checked", "derived", "asserted")


@pytest.mark.parametrize("name, D, W", [c for c in CASES if c[0] != "sparse"], ids=[c[0] for c in CASES if c[0] != "sparse"])
def test_orthonormal_coordinates(name, D, W):
    V = np.random.default_rng(6).standard_normal((D.dim, 4))
    Z = D.to_orthonormal(V)
    assert np.allclose(Z.T @ Z, V.T @ W @ V, rtol=1e-10, atol=1e-10)           # orthonormal: dot products = inner products
    assert np.allclose(D.from_orthonormal(Z), V, rtol=1e-9, atol=1e-10)
    S = D.random(size=40000, seed=7).values                                     # the space's standard Gaussian N(0, W^-1)
    C = np.linalg.inv(W)
    assert np.abs(np.cov(S.T) - C).max() <= 0.05 * np.abs(C).max()


@pytest.mark.parametrize("name, D, W", CASES, ids=[c[0] for c in CASES])
def test_adjoints_into_the_weighted_space(name, D, W):
    rng = np.random.default_rng(8)
    M = pg.L2("[0, 1]")
    ks = [np.exp(-((M.x - c) / 0.1) ** 2) for c in np.linspace(0.05, 0.95, D.dim)]
    G = pg.Functionals(ks, into=D)
    m, y = pg.probability.legendre_decay(M).sample(seed=9), D(rng.standard_normal(D.dim))
    assert G(m) @ y == pytest.approx(m @ G.adjoint(y), rel=1e-10)
    A = pg.Matrix(rng.standard_normal((D.dim, 5)), codomain=D)               # R^5 -> D
    x = pg.Euclidean(5)(rng.standard_normal(5))
    assert A(x) @ y == pytest.approx(x @ A.adjoint(y), rel=1e-10)
    B = pg.Matrix(rng.standard_normal((5, D.dim)), domain=D)                  # D -> R^5
    z = pg.Euclidean(5)(rng.standard_normal(5))
    assert B(y) @ z == pytest.approx(y @ B.adjoint(z), rel=1e-10)


def test_damped_least_squares_with_correlated_noise():
    M = pg.L2("[0, 1]")
    n = 30
    ks = [np.exp(-((M.x - c) / 0.05) ** 2) for c in np.linspace(0.05, 0.95, n)]
    i = np.arange(n)
    C = 1e-4 * np.exp(-np.abs(i[:, None] - i[None, :]) / 3.0)                 # exponentially correlated noise
    D = pg.Euclidean(n, covariance=C)
    G = pg.Functionals(ks, into=D)
    d = D(G(M(np.sin)).values)
    m = G.adjoint((G @ G.adjoint + 1.0 * pg.Identity(D)).inv(d))
    # the same solution from the explicit formula m = sum_i a_i k_i, a = C^-1 (K C^-1 + I)^-1 d ... in matrix form
    K = pg.Functionals(ks).gram()
    a = np.linalg.solve(K + C, d.values)
    assert np.allclose(G.gram() @ a, G(m).values, rtol=1e-8, atol=1e-12)


def test_identity_adjoint_is_the_metric_and_errors():
    E = pg.Euclidean(4)
    W = _spd(4, 11)
    D = E.with_metric(pg.Matrix(W, E, E))
    J = pg.Matrix(np.eye(4), domain=E, codomain=D)                            # the identity map E -> D
    assert np.allclose(J.adjoint.matrix(), W)                                  # J* = W
    assert pg.Euclidean(4, weights=np.ones(4)) == E
    assert pg.Euclidean(4, weights=[1, 2, 3, 4]) == E.with_metric(pg.Diagonal([1, 2, 3, 4], E))
    assert pg.Euclidean(4, covariance=[1, 0.5, 0.25, 2]) == pg.Euclidean(4, weights=[1, 2, 4, 0.5])
    with pytest.raises(ValueError):
        pg.Euclidean(4, metric=-np.eye(4))
    with pytest.raises(ValueError):
        D.with_metric(pg.Matrix(W, E, E))                                      # only from the standard space
    with pytest.raises(ValueError):
        pg.Euclidean(4, metric=np.triu(W))                                     # not symmetric


def test_zero_noise_is_refused_with_an_explanation():
    with pytest.raises(ValueError, match="sigma = 0"):
        pg.Euclidean(3, weights=np.full(3, np.inf))
    with pytest.raises(ValueError, match="sigma = 0"):
        pg.Euclidean(3, covariance=0.0)
    with pytest.raises(ValueError):
        pg.Euclidean(2, metric=np.array([[1.0, 0.0], [0.0, np.nan]]))
