"""Plan 065 P4, A8 (architecture review 2026-11-13): an ellipsoid given by its form, pg.Ellipsoid(D, r, shape=A), is
the same set as pg.Ellipsoid(D, r, covariance=S) with S = (W A)^-1 in standard coordinates (W the metric of D), and it
has the same interface: shape, factor, centre_values, shape_operator, inverse. Its consumers (BGP's whitening, the
csola decomposition, the set engine's dual, restriction to data rows) read that interface and give the same answers
for both. Before the repair each consumer crashed on the operator ellipsoid (no .shape, no .factor, a centre of None)."""
import numpy as np
import pytest

import pygeoinf0 as pg


def problem(seed=0, n=5, m=7):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, n)
    D = pg.Euclidean(n, weights=w)                              # a weighted data space: the metric matters
    M = pg.Euclidean(m)
    Q = rng.normal(size=(n, n))
    P = Q @ Q.T / n + 0.5 * np.eye(n)                           # <A e, e>_D = e^T P e: A = W^-1 P is D-self-adjoint
    A = pg.Matrix(np.diag(1 / w) @ P, D, D, name="A")
    c = rng.normal(size=n)
    G, T = pg.Matrix(rng.normal(size=(n, m)), M, D, name="G"), pg.Matrix(rng.normal(size=(2, m)), M, pg.Euclidean(2))
    d = D(G.A @ rng.normal(size=m) * 0.1 + c)
    return M, D, G, T, d, A, P, c


def pair(r=0.7, centred=True):
    M, D, G, T, d, A, P, c = problem()
    centre = D(c) if centred else None
    E_op = pg.Ellipsoid(D, r, shape=A, centre=centre)
    E_mat = pg.Ellipsoid(D, r, covariance=np.linalg.inv(P), centre=c if centred else None)
    return M, D, G, T, d, E_op, E_mat


def test_the_two_parameterisations_are_one_set():
    M, D, G, T, d, E_op, E_mat = pair()
    rng = np.random.default_rng(1)
    for _ in range(5):
        v = D(rng.normal(size=D.dim))
        assert E_op.support(v) == pytest.approx(E_mat.support(v), rel=1e-12)


@pytest.mark.parametrize("centred", [True, False])
def test_one_interface(centred):
    M, D, G, T, d, E_op, E_mat = pair(centred=centred)
    for E in (E_op, E_mat):
        assert isinstance(E, pg.Ellipsoid) and E.radius == 0.7
    assert np.allclose(E_op.shape, E_mat.shape, rtol=1e-12, atol=1e-14)
    assert np.allclose(E_op.factor @ E_op.factor.T, E_mat.shape, rtol=1e-12, atol=1e-14)
    assert np.allclose(E_op.centre_values, E_mat.centre_values, rtol=0, atol=0)
    v = D(np.arange(1.0, D.dim + 1))
    for E in (E_op, E_mat):                                     # the operators of the form, on both
        assert np.allclose(E.inverse(E.shape_operator(v)).values, v.values, rtol=1e-12)
    assert np.allclose(E_mat.shape_operator(v).values, E_op.shape_operator(v).values, rtol=1e-12)


def test_an_operator_ellipsoid_off_rn_refuses_coordinates():
    X = pg.L2(pg.Interval(0, 1))
    E = pg.Ellipsoid(X, 1.0, shape=4.0 * pg.Identity(X))
    for attr in ("shape", "factor", "centre_values"):
        with pytest.raises(pg.Refused, match="shape_operator"):
            getattr(E, attr)


def test_a_non_self_adjoint_form_is_refused_when_coordinates_are_read():
    D = pg.Euclidean(2)
    E = pg.Ellipsoid(D, 1.0, shape=pg.Matrix(np.array([[2.0, 1.0], [0.0, 2.0]]), D, D))
    with pytest.raises(pg.Refused, match="self-adjoint"):
        E.shape


def test_the_set_engine_reads_it():
    M, D, G, T, d, E_op, E_mat = pair()
    S = pg.NormBall(M, 3.0)
    one = pg.push(pg.update(S, pg.Additive(G, E_op), d), T).interval(0)
    two = pg.push(pg.update(S, pg.Additive(G, E_mat), d), T).interval(0)
    assert one == pytest.approx(two, rel=1e-9)


def test_restriction_reads_it():
    M, D, G, T, d, E_op, E_mat = pair()
    rows = [0, 2, 3]
    one = pg.Additive(G, E_op).select(rows).noise
    two = pg.Additive(G, E_mat).select(rows).noise
    assert np.allclose(one.shape, two.shape, rtol=1e-12) and np.allclose(one.centre_values, two.centre_values)
    assert one.radius == two.radius


def test_whitening_and_csola_read_it():
    M, D, G, T, d, E_op, E_mat = pair()
    S = pg.NormBall(M, 3.0)
    one = pg.inference.csola(S, pg.Additive(G, E_op), T, d, gamma=0.3)
    two = pg.inference.csola(S, pg.Additive(G, E_mat), T, d, gamma=0.3)
    assert one.interval(0) == pytest.approx(two.interval(0), rel=1e-9)
    from pygeoinf0.convex.forms import noise_metric                  # the set's shape, as the engine takes it
    a, b = noise_metric(E_op, D), noise_metric(E_mat, D)
    assert a.kind == b.kind and np.allclose(a.data, b.data, rtol=1e-12) and np.allclose(a.offset, b.offset)
