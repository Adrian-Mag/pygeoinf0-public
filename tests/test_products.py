"""Law tests for product (separated) kernel families, expansions and sparse Gram matrices."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.realisations.series import separated as SEP


def _box_family(n=6, wf=0.7):
    M = pg.L2(pg.Box((0, 1), (0, 1), (0, 1)))
    c = np.linspace(0.1, 0.9, n)
    w = wf * (c[1] - c[0])
    C = [a.ravel() for a in np.meshgrid(c, c, c, indexing="ij")]
    g = lambda t, s: np.exp(-((t - s) / w) ** 2)
    K = M.product_family([lambda x, a: g(x, a), lambda y, b: g(y, b), lambda z, e: g(z, e)], a=C[0], b=C[1], e=C[2])
    return M, K, C, g


def test_product_kernels_equal_dense_kernels():
    M, K, C, g = _box_family()
    m = M(lambda x, y, z: np.sin(3 * x) * np.cos(2 * y) + z ** 2)
    G = pg.Functionals(K)
    d = G(m).values
    for i in (0, 77, 215):
        ki = M(lambda x, y, z: g(x, C[0][i]) * g(y, C[1][i]) * g(z, C[2][i]))
        assert d[i] == pytest.approx(ki @ m, abs=1e-15, rel=1e-12)
        assert K[i].norm() == pytest.approx(ki.norm(), rel=1e-12)
    assert len(K.realisation.axes[0].U.err) == 6          # 6 distinct factors per axis, 216 members


def test_sparse_gram_equals_dense(monkeypatch):
    M, K, _, _ = _box_family(n=8)
    F = K.realisation
    Gd = SEP.gram_sf_sf(F, F)
    F._gram = None
    monkeypatch.setattr(SEP, "DENSE_GRAM_MAX", 10)
    Gs = SEP.gram_sf_sf(F, F)
    assert Gs.nnz < F.N ** 2
    assert np.abs(Gd - Gs.toarray()).max() <= 1e-14 * np.diag(Gd).max()


def test_expansion_norms_and_adjoint():
    M, K, _, _ = _box_family()
    D = pg.Euclidean(len(K), weights=np.linspace(1, 2, len(K)))
    G = pg.Functionals(K, into=D)
    y = D.random(seed=3)
    m = pg.probability.random_features(M).sample(seed=4)
    assert G(m) @ y == pytest.approx(m @ G.adjoint(y), rel=1e-12)
    e = G.adjoint(y)
    alpha = D.apply_metric(y.values)
    assert e.norm() ** 2 == pytest.approx(alpha @ (G.gram() @ alpha), rel=1e-12)
    assert (e - e).norm() == pytest.approx(0.0, abs=1e-12 * e.norm())


@pytest.mark.parametrize("dom, facs, prm", [
    (pg.Disk(1.0), [lambda r, a: np.exp(-((r - a) / 0.1) ** 2), lambda phi, b: np.exp((np.cos(phi - b) - 1) / 0.2)],
     dict(a=np.repeat([0.4, 0.7], 6), b=np.tile(np.linspace(0, 2 * np.pi, 6, endpoint=False), 2))),
    (pg.Cylinder(1.0, z=(0, 1)), [lambda r, a: (r / a) ** 2 * np.exp(1 - (r / a) ** 2), None,
                                  lambda z, b: np.exp(-((z - b) / 0.1) ** 2)],
     dict(a=np.repeat([0.3, 0.6], 5), b=np.tile(np.linspace(0.1, 0.9, 5), 2))),
    (pg.Ball(1.0), [lambda r, a: np.exp(-((r - a) / 0.1) ** 2), lambda theta, b: np.exp(-((theta - b) / 0.3) ** 2), None],
     dict(a=np.repeat([0.4, 0.8], 4), b=np.tile(np.linspace(0.6, 2.5, 4), 2))),
])
def test_minimum_norm_with_product_kernels(dom, facs, prm):
    M = pg.L2(dom)
    K = M.product_family(facs, **prm)
    G = pg.Functionals(K)
    m = M(lambda *X: 1 + X[0] + X[1] ** 2)
    d = G(m)
    md = G.adjoint((G @ G.adjoint).inv(d))
    assert (G(md) - d).norm() <= 1e-12 * d.norm()
    assert md.norm() <= m.norm()
    assert abs(md @ (m - md)) <= 1e-12 * m.norm() ** 2


def test_expansion_values_match_dense_sum():
    M, K, C, g = _box_family(n=4)
    c = np.random.default_rng(0).standard_normal(len(K))
    G = pg.Functionals(K)
    e = G.adjoint(pg.Euclidean(len(K))(c))
    P = np.random.default_rng(1).uniform(0, 1, (50, 3))
    direct = sum(c[i] * g(P[:, 0], C[0][i]) * g(P[:, 1], C[1][i]) * g(P[:, 2], C[2][i]) for i in range(len(K)))
    assert np.abs(e(P) - direct).max() <= 1e-11 * np.abs(direct).max()


def test_kronecker_gram_matches_dense(monkeypatch):
    M, K, C, g = _box_family(n=5)
    N = len(K)
    sigma = 0.01
    D = pg.Euclidean(N, weights=np.full(N, 1 / sigma ** 2))
    G = pg.Functionals(K, into=D)
    m = M(lambda x, y, z: np.sin(3 * x) * np.cos(2 * y) + z ** 2)
    d = D(G(m).values + sigma * np.random.default_rng(0).standard_normal(N))
    dense_A = G @ G.adjoint
    ref = G.adjoint((dense_A + 30.0 * pg.Identity(D)).inv(d))
    ref_mn = G.adjoint(dense_A.inverse(pg.Eigh(1e-12))(d))
    monkeypatch.setattr(SEP, "DENSE_GRAM_MAX", 10)
    K2 = M.product_family([lambda x, a: g(x, a), lambda y, b: g(y, b), lambda z, e: g(z, e)], a=C[0], b=C[1], e=C[2])
    G2 = pg.Functionals(K2, into=D)
    A2 = G2 @ G2.adjoint
    assert isinstance(A2, pg.KroneckerMatrix)
    y = D.random(seed=1)
    assert np.abs(A2(y).values - dense_A(y).values).max() <= 1e-13 * np.abs(dense_A(y).values).max()
    B2 = A2 + 30.0 * pg.Identity(D)
    assert isinstance(B2.recommend_solver().solver, pg.KroneckerEigh)
    sol = G2.adjoint(B2.inv(d))
    mn = G2.adjoint(A2.inv(d))
    assert (G2(mn) - d).norm() <= 1e-6 * d.norm() and mn.norm() <= ref_mn.norm() * (1 + 1e-6)
    P = np.random.default_rng(2).uniform(0, 1, (40, 3))      # values, not sol.distance(ref): the two live on different
    assert np.abs(sol(P) - ref(P)).max() <= 1e-9 * np.abs(ref(P)).max()   # families (sqrt(eps) floor; README limits)
