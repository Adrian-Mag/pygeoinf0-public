"""Plan 063 P3: solver backends (dense LAPACK, sparse Cholesky), sparse forms of structured operators, matrices of
compositions and sums without the identity product, and the memory check of recommend()."""
import numpy as np
import pytest
import scipy.sparse as sp

import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported


def _ray_like(N=300, M=200, seed=1):
    rng = np.random.default_rng(seed)
    G = sp.random(N, M, density=0.05, random_state=rng, format="csr")
    V = rng.uniform(0.5, 2.0, M)
    return G, V


def _identity_route(A):
    """The default matrix: A applied to the identity."""
    return np.array(A(A.domain.family(np.eye(A.domain.dim))).values.T)


def test_composition_and_sum_matrices_equal_the_identity_route():
    G_, V = _ray_like()
    MV, DN = pg.Euclidean(G_.shape[1], weights=V), pg.Euclidean(G_.shape[0])
    G = pg.SparseMatrix(G_, MV, DN)
    C = pg.Diagonal(np.linspace(1, 2, DN.dim), DN)
    S = G @ G.adjoint + 0.3 * C
    ref = _identity_route(S)
    assert np.allclose((G @ G.adjoint).matrix(), _identity_route(G @ G.adjoint), rtol=0, atol=1e-13 * np.abs(ref).max())
    assert np.allclose(S.matrix(), ref, rtol=0, atol=1e-13 * np.abs(ref).max())
    # a dense factor on the left, a sparse one on the right
    B = pg.Matrix(np.random.default_rng(2).standard_normal((5, DN.dim)), DN, pg.Euclidean(5))
    assert np.allclose((B @ G).matrix(), _identity_route(B @ G), rtol=0, atol=1e-12)


def test_sparse_forms():
    G_, V = _ray_like()
    DN = pg.Euclidean(G_.shape[0])
    G = pg.SparseMatrix(G_, pg.Euclidean(G_.shape[1]), DN)
    d = np.linspace(1, 2, DN.dim)
    assert (pg.Diagonal(d, DN).sparse() != sp.diags(d)).nnz == 0
    assert pg.Identity(DN).sparse().nnz == DN.dim
    A = pg.SparseMatrix(G_ @ G_.T, DN, DN) + 2.0 * pg.Diagonal(d, DN)
    assert np.allclose(A.sparse().toarray(), (G_ @ G_.T).toarray() + 2.0 * np.diag(d))
    assert (G @ G.adjoint).sparse() is None                       # products are not multiplied out
    assert pg.Matrix(np.eye(3)).sparse() is None


def _laplacian(n, shift, weights=None):
    L = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    X = pg.Euclidean(n) if weights is None else pg.Euclidean(n, weights=weights)
    if weights is not None:
        L = sp.diags(1 / weights) @ L                             # self-adjoint in the weighted inner product
    return pg.SparseMatrix(L, X, X) + shift * pg.Identity(X), X


@pytest.mark.parametrize("weighted", [False, True])
def test_sparse_cholesky_agrees_with_dense(weighted):
    n = 400
    w = np.linspace(0.5, 1.5, n) if weighted else None
    A, X = _laplacian(n, 0.1, w)
    y = X(np.sin(np.arange(n)))
    xd = pg.Inverse(A, pg.Cholesky())(y)
    xs = pg.Inverse(A, pg.Cholesky(backend="sparse"))(y)
    assert np.allclose(xs.values, xd.values, rtol=1e-11, atol=1e-11 * np.abs(xd.values).max())
    assert np.allclose(A(xs).values, y.values, atol=1e-10)


def test_sparse_cholesky_refusals():
    A, X = _laplacian(50, -1.0)                                   # indefinite
    with pytest.raises(np.linalg.LinAlgError):
        pg.Inverse(A, pg.Cholesky(backend="sparse"))(X(np.ones(50)))
    B = pg.Matrix(np.eye(4) * 2)
    with pytest.raises(Unsupported):
        pg.Inverse(B, pg.Cholesky(backend="sparse"))(B.domain(np.ones(4)))
    with pytest.raises(ValueError, match="no backend"):
        pg.Cholesky(backend="gpu")


def test_backends_listed():
    b = pg.solvers.backends()
    assert set(b) == {"dense", "sparse"}
    assert all({"computes", "when", "error", "cost", "available"} <= set(v) for v in b.values())


def test_recommend_checks_memory():
    A, X = _laplacian(300, 0.1)
    with pg.settings.using(memory_gb=1e-6):
        r = pg.solvers.recommend(A)
    assert "sparse" in r.solver.name and any("memory_gb" in f for f in r.facts)
    D = pg.Matrix(np.diag(np.linspace(1, 2, 300)))
    D._sa = (True, "checked", "diagonal")
    with pg.settings.using(memory_gb=1e-6):
        r = pg.solvers.recommend(D)
    assert r.solver.name.startswith("CG")
    assert pg.solvers.recommend(A).solver.name == "Cholesky"   # within the budget: dense, as before
