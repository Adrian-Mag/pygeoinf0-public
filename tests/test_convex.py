"""Convex sets by support functions (plan 008 P2): closed forms against direct maximisation, enumeration of vertices,
linear programming and sampled points; the set algebra; three-valued membership; products on a direct sum."""
import itertools

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.convex import CartesianProduct, Image, image

rng = np.random.default_rng(11)


def test_norm_ball_support_in_a_function_space_is_attained_on_the_span():
    M = pg.Sobolev("[0, 1]", 1, length_scale=0.2)
    k = [M.riesz(pg.DiracMeasure(0.4)), M.riesz(pg.Interval(0.6, 0.9)), M.riesz(pg.Interval(0.1, 0.3))]
    c = k[2] * 0.5
    B = pg.NormBall(M, 2.0, centre=c)
    v = k[0] + k[1] * 2.0
    s = B.support(v)
    m_star = c + v * (2.0 / v.norm())                              # the maximiser c + r v / ||v||
    assert B.member(m_star) is pg.Truth.TRUE
    assert M.inner(v, m_star) == pytest.approx(s, rel=1e-12)
    Q = pg.Span(k).basis()                                          # direct maximisation in an orthonormal basis
    coords = np.asarray(M.inner(Q, v), float).ravel()
    assert s == pytest.approx(M.inner(v, c) + 2.0 * np.linalg.norm(coords), rel=1e-12)
    for _ in range(20):                                             # points of the ball never exceed the support
        w = rng.normal(size=len(Q))
        w *= 2.0 * rng.random() / np.linalg.norm(w)
        m = c + sum((Q[i] * float(w[i]) for i in range(len(Q))), M.zero)
        assert M.inner(v, m) <= s + 1e-12


def test_ball_of_a_covariance_space_is_the_noise_ellipsoid():
    A = rng.normal(size=(3, 3))
    C = A @ A.T + np.eye(3)
    D = pg.Euclidean(3, covariance=C)
    B, E = pg.NormBall(D, 1.5), pg.Ellipsoid(D.standard, 1.5, covariance=C)
    for k in range(3):
        assert B.interval(k) == pytest.approx(E.interval(k), rel=1e-12)
        assert E.interval(k)[1] == pytest.approx(1.5 * np.sqrt(C[k, k]), rel=1e-12)
    a = rng.normal(size=3)
    L = np.linalg.cholesky(C)
    z = rng.normal(size=(3, 20000))
    pts = 1.5 * L @ (z / np.linalg.norm(z, axis=0))                 # boundary points of the ellipsoid
    assert (a @ pts).max() <= E.support(a) + 1e-12 and (a @ pts).max() >= E.support(a) * 0.99
    assert E.member(D.standard(pts[:, 0])) is pg.Truth.TRUE


def test_box_and_polytope_against_vertices_and_linear_programming():
    D = pg.Euclidean(3)
    lo, hi = np.array([-1.0, 0.0, 2.0]), np.array([1.0, 0.5, 3.0])
    box = pg.CoordinateBox(D, lo, hi)
    poly = pg.Polytope(D, np.r_[np.eye(3), -np.eye(3)], np.r_[hi, -lo])
    vertices = np.array(list(itertools.product(*zip(lo, hi))))
    for _ in range(10):
        a = rng.normal(size=3)
        assert box.support(a) == pytest.approx((vertices @ a).max(), rel=1e-12)
        assert poly.support(a) == pytest.approx((vertices @ a).max(), rel=1e-9)
    empty = pg.Polytope(D, [[1.0, 0, 0], [-1.0, 0, 0]], [0.0, -1.0])            # x <= 0 and x >= 1
    assert empty.is_empty() is pg.Truth.TRUE and empty.support([1.0, 0, 0]) == -np.inf
    half = pg.Polytope(D, [[1.0, 0, 0]], [0.0])
    assert half.support([0.0, 1.0, 0.0]) == np.inf
    assert box.member(D([0.0, 0.25, 2.5])) is pg.Truth.TRUE and box.member(D([0.0, 0.25, 4.0])) is pg.Truth.FALSE


def test_set_algebra_identities():
    D = pg.Euclidean(3, weights=[1.0, 2.0, 0.5])
    c1, c2 = D(rng.normal(size=3)), D(rng.normal(size=3))
    B1, B2 = pg.NormBall(D, 0.7, centre=c1), pg.NormBall(D, 1.3, centre=c2)
    both = pg.NormBall(D, 2.0, centre=c1 + c2)
    for _ in range(5):
        v = D(rng.normal(size=3))
        assert (B1 + B2).support(v) == pytest.approx(both.support(v), rel=1e-12)
        assert (2.0 * B1).support(v) == pytest.approx(pg.NormBall(D, 1.4, centre=c1 * 2.0).support(v), rel=1e-12)
        assert (pg.NormBall(D, 0.7) + c1).support(v) == pytest.approx(B1.support(v), rel=1e-12)
        assert (B1 | B2).support(v) == max(B1.support(v), B2.support(v))
    assert (B1 | B2).member(c2) is pg.Truth.TRUE
    T = pg.Matrix(rng.normal(size=(2, 3)), D, pg.Euclidean(2))
    TB = image(T, pg.NormBall(D, 1.0))
    assert isinstance(TB, Image)
    q = np.array([0.3, -1.1])
    Lt = np.linalg.cholesky(D.metric_matrix()).T                    # ||x||_D = ||Lt x||: boundary x = Lt^-1 u
    u = rng.normal(size=(3, 20000))
    X = np.linalg.solve(Lt, u / np.linalg.norm(u, axis=0))
    assert (q @ T.A @ X).max() <= TB.support(q) + 1e-12 and (q @ T.A @ X).max() >= 0.99 * TB.support(q)
    K = image(T, pg.AffineSubset(c1, pg.Span([D([1.0, 0, 0])])))
    assert isinstance(K, pg.AffineSubset) and K.direction.dim == 1


def test_three_valued_membership():
    D = pg.Euclidean(2)
    B = pg.NormBall(D, 1.0)
    e = D([0.6, 0.8])
    assert B.member(e) is pg.Truth.TRUE
    assert B.member(e * (1 + 1e-9)) is pg.Truth.UNKNOWN
    assert B.member(e * (1 + 1e-6)) is pg.Truth.FALSE
    with pytest.raises(TypeError):
        bool(pg.Truth.UNKNOWN)
    assert (pg.Truth.FALSE & pg.Truth.UNKNOWN) is pg.Truth.FALSE and (pg.Truth.TRUE | pg.Truth.UNKNOWN) is pg.Truth.TRUE


def test_slab_support_is_finite_only_along_its_normal():
    D = pg.Euclidean(3, weights=[1.0, 4.0, 1.0])
    a = D([1.0, 1.0, 0.0])
    S = pg.Slab(D, a, 0.5)
    assert S.support(a * 3.0) == pytest.approx(1.5, rel=1e-14)
    assert S.support(D([0.0, 0.0, 1.0])) == np.inf
    assert S.member(D([0.1, 0.0, 7.0])) is pg.Truth.TRUE


def test_cartesian_product_on_a_direct_sum():
    Ea, Eb = pg.Euclidean(2), pg.Euclidean(3)
    X = pg.DirectSum({"a": Ea, "b": Eb}, weights=[2.0, 0.5])
    S = CartesianProduct([pg.NormBall(Ea, 1.0), pg.CoordinateBox(Eb, -1.0, 1.0)], X)
    v = X({"a": [3.0, 4.0], "b": [0.5, -2.0, 0.0]})
    best = X({"a": [0.6, 0.8], "b": [1.0, -1.0, 0.0]})               # the maximiser, part by part
    assert S.support(v) == pytest.approx(X.inner(v, best), rel=1e-12)
    assert S.support(v) == pytest.approx(2.0 * 5.0 + 0.5 * 2.5, rel=1e-12)
    assert S.member(best) is pg.Truth.TRUE


def test_the_simplex_against_vertex_enumeration():
    from scipy.optimize import nnls
    from pygeoinf0.optimisers.lp import lp_max
    r = np.random.default_rng(5)
    checked = 0
    for _ in range(20):
        A = r.normal(size=(9, 3))
        b = r.uniform(0.5, 2.0, 9)                              # 0 is strictly feasible
        c = r.normal(size=3)
        val, x = lp_max(c, A, b)
        if nnls(A.T, c)[1] > 1e-9:                              # c not a nonnegative combination of rows: unbounded
            assert val == np.inf
            continue
        verts = []
        for rows in itertools.combinations(range(9), 3):
            M = A[list(rows)]
            if abs(np.linalg.det(M)) > 1e-12:
                v = np.linalg.solve(M, b[list(rows)])
                if np.all(A @ v <= b + 1e-9):
                    verts.append(v)
        assert val == pytest.approx(max(c @ v for v in verts), rel=1e-9) and np.all(A @ x <= b + 1e-9)
        checked += 1
    assert checked >= 5
    assert lp_max([1.0, 0.0], [[0.0, 1.0]], [1.0])[0] == np.inf          # unbounded
    assert lp_max([1.0], [[1.0], [-1.0]], [0.0, -1.0])[0] == -np.inf      # x <= 0 and x >= 1
