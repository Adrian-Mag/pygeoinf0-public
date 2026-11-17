"""The plain G G* of separable functionals takes the structured route on any diagonal metric (plan 045), and is what a dense
route in D computes: compared with dense formulas, on invertible and singular Grams, with and without a shift."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.operators.blocks import BlockDiagonal
from pygeoinf0.operators.plain_gram import WeightedGram

BALL = pg.Ball(1.0)
X = pg.L2(BALL, realisation="radial_harmonics")
LAB = [(0, 0), (1, 0), (1, 0), (1, 1), (2, 0), (2, 0), (2, 0), (2, 3)]
N = len(LAB)
SIGMA = 0.1 * (1.0 + np.arange(N))


def kernels(repeat=False):
    k = [(lambda r, p=i: r ** (0.5 * p + 1.0)) for i in range(N)]    # independent, so only repeat= makes K singular
    if repeat:
        k[2] = k[1]                                               # the same functional twice: K is singular
    return k


def separable(metric=True, repeat=False):
    D = pg.Euclidean(N, covariance=SIGMA ** 2) if metric else pg.Euclidean(N)
    return pg.Functionals.separable(X, kernels(repeat), LAB, into=D), D


def dense_k(G, D):
    """The plain Gram from G and its adjoint, column by column: <g_i, g_j>, undoing D's metric."""
    A = np.column_stack([G(G.adjoint(D(e))).values for e in np.eye(N)])
    return A / (1.0 / np.asarray(D.weights, float))[None, :] if D.weights is not None else A


def pinv_in_d(K, w, y, s=0.0, rcond=1e-12):
    r = np.sqrt(w)
    e, Q = np.linalg.eigh(r[:, None] * K * r[None, :] + s * np.eye(N))
    keep = e > rcond * e.max()
    return (Q[:, keep] / e[keep]) @ (Q[:, keep].T @ (r * y)) / r


def test_the_plain_expression_is_structured_on_either_metric():
    G, D = separable()
    assert isinstance(G @ G.adjoint, WeightedGram)
    G0, D0 = separable(metric=False)
    assert isinstance(G0 @ G0.adjoint, BlockDiagonal)
    assert (G @ G.adjoint).n_blocks == len(set(LAB))
    assert G.structured_gram().domain.is_standard                  # always the plain Gram


def test_it_is_g_g_star_and_self_adjoint_in_d():
    G, D = separable()
    A = G @ G.adjoint
    want = np.column_stack([G(G.adjoint(D(e))).values for e in np.eye(N)])
    assert A.matrix() == pytest.approx(want, rel=1e-11, abs=1e-12)
    rng = np.random.default_rng(1)
    u, v = D(rng.normal(size=N)), D(rng.normal(size=N))
    assert D.inner(A(u), v) == pytest.approx(D.inner(u, A(v)), rel=1e-11)
    assert A.adjoint is A and A.is_selfadjoint()[0]


def test_the_inverse_and_minimum_norm_on_an_invertible_gram():
    G, D = separable()
    G0, D0 = separable(metric=False)
    y = np.random.default_rng(2).normal(size=N)
    A, A0 = G @ G.adjoint, G0 @ G0.adjoint
    assert A(A.inv(D(y))).values == pytest.approx(y, rel=1e-8, abs=1e-8)
    m, m0 = G.adjoint(A.inv(D(y))), G0.adjoint(A0.inv(D0(y)))        # G* (G G*)^-1 does not depend on D's metric
    assert X.norm(m - m0) <= 1e-7 * X.norm(m0)


def test_on_a_singular_gram_it_is_the_pseudo_inverse_in_d():
    """A datum outside the range is fitted in D's norm: W^-1 K^+ would be another inverse, and fail here."""
    G, D = separable(repeat=True)
    K = G.structured_gram().matrix()
    w = np.asarray(D.weights, float)
    y = np.random.default_rng(3).normal(size=N)
    with pytest.raises(pg.errors.Refused):
        (G @ G.adjoint).inv(D(y))
    got = (G @ G.adjoint).pseudo_inverse(1e-12)(D(y)).values
    assert got == pytest.approx(pinv_in_d(K, w, y), rel=1e-7, abs=1e-7)


def test_a_family_of_data():
    G, D = separable()
    A = G @ G.adjoint
    V = np.random.default_rng(4).normal(size=(3, N))
    from pygeoinf0.spaces.elements import VectorFamily
    out = A.inv(VectorFamily(D, V))
    for i in range(3):
        assert out.values[i] == pytest.approx(A.inv(D(V[i])).values, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("metric", [True, False])
@pytest.mark.parametrize("s", [1.0, 1e-4])
def test_a_small_shift_keeps_every_direction(metric, s):
    """Even on an ill-conditioned Gram: the shift determines all of them, and a pseudo-inverse cut would drop what it keeps."""
    G, D = separable(metric, repeat=True)
    y = np.random.default_rng(5).normal(size=N)
    A = (G @ G.adjoint) + pg.Diagonal(np.full(N, s), D)
    K = G.structured_gram().matrix()
    w = np.asarray(D.weights, float) if metric else np.ones(N)
    want = pinv_in_d(K, w, y, s=s, rcond=1e-16)
    assert A.inv(D(y)).values == pytest.approx(want, rel=1e-7, abs=1e-7)
    assert A(A.inv(D(y))).values == pytest.approx(y, rel=1e-7, abs=1e-7)


@pytest.mark.parametrize("metric", [True, False])
def test_a_shift_below_the_pseudo_inverse_cut_is_not_dropped(metric):
    """A repeated kernel leaves one eigenvalue of K + s at s, below rcond times the largest: that direction is what s determines."""
    G, D = separable(metric, repeat=True)
    K = G.structured_gram().matrix()
    w = np.asarray(D.weights, float) if metric else np.ones(N)
    r = np.sqrt(w)
    s = 1e-13 * np.linalg.eigvalsh(r[:, None] * K * r[None, :]).max()
    y = np.random.default_rng(8).normal(size=N)
    got = ((G @ G.adjoint) + pg.Diagonal(np.full(N, s), D)).inv(D(y)).values
    want = pinv_in_d(K, w, y, s=s, rcond=1e-16)
    assert np.linalg.norm(r * (got - want)) <= 0.05 * np.linalg.norm(r * want)
    assert np.linalg.norm(r * want) > 1e3 * np.linalg.norm(r * pinv_in_d(K, w, y))     # the direction dominates


def test_an_identity_shift_on_a_metric():
    G, D = separable()
    y = D(np.random.default_rng(6).normal(size=N))
    A = (G @ G.adjoint) + pg.Identity(D)
    assert isinstance(A, WeightedGram)
    assert A(A.inv(y)).values == pytest.approx(y.values, rel=1e-8, abs=1e-8)


def test_decompose_reaches_the_blocks_from_the_plain_expression():
    G, D = separable()
    G0, D0 = separable(metric=False)
    w = np.asarray(D.weights, float)
    dec = pg.decompose(G, D.standard_gaussian())
    assert dec.route in ("blocks", "dense")
    K = G.structured_gram().matrix()
    r = np.sqrt(w)
    assert np.sort(dec.lam) == pytest.approx(np.linalg.eigvalsh(r[:, None] * K * r[None, :]), rel=1e-9, abs=1e-9)
    noise = pg.GaussianMeasure(D0, pg.Diagonal(SIGMA ** 2, D0))
    assert np.sort(pg.decompose(G0, noise).lam) == pytest.approx(np.sort(dec.lam), rel=1e-9, abs=1e-9)


def test_a_prior_by_degree_on_the_weighted_space_is_the_plain_route():
    G, D = separable()
    G0, D0 = separable(metric=False)
    C = pg.Laplacian(BALL, bc={"boundary": "dirichlet"}, centre="regular").bessel_potential(10.0, 2.0, degrees=[0, 1, 2], modes=16)
    y = np.random.default_rng(7).normal(size=N)
    T = pg.Functionals.separable(X, kernels()[:1], [(1, 0)], into=pg.Euclidean(1))
    a = pg.decompose(G, D.standard_gaussian(), middle=C).posterior(T, D(y), 1.0, 1.0)
    b = pg.decompose(G0, pg.GaussianMeasure(D0, pg.Diagonal(SIGMA ** 2, D0)), middle=C).posterior(T, D0(y), 1.0, 1.0)
    assert np.ravel(a.mean) == pytest.approx(np.ravel(b.mean), rel=1e-9)
    assert np.ravel(a.covariance) == pytest.approx(np.ravel(b.covariance), rel=1e-9)


def test_a_full_metric_stays_dense():
    C = np.eye(N) + 0.1 * np.ones((N, N))
    D = pg.Euclidean(N, covariance=C)
    G = pg.Functionals.separable(X, kernels(), LAB, into=D)
    assert not isinstance(G @ G.adjoint, (WeightedGram, BlockDiagonal))
    assert G.structured_gram() is None
