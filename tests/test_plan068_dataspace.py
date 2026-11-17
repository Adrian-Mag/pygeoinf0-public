"""Plan 068 P5 (decisions Q1 and Q9, review B12): the data space as a named concept (DataSpace: a plain R^n with
labelled coordinates; D.whitened(C) the whitened coordinates as a named construction), a direct sum of Euclideans as a
Euclidean with block labels (euclidean_sum), and the column operator x -> (G_k x)_k (operators/blocks.ColumnOperator),
which replaces relations/algebra.StackedOperator. The laws are in test_laws.py."""
import importlib

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.claims import THEOREM
from pygeoinf0.core.errors import Refused, SpaceMismatch
from pygeoinf0.core.logic import Truth
from pygeoinf0.operators.blocks import ColumnOperator
from pygeoinf0.operators.whitening import Whitening
from pygeoinf0.spaces import DataSpace, euclidean_sum


def spd(n, seed):
    A = np.random.default_rng(seed).normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


# ------------------------------------------------------------------------------------------------ the data space
def test_a_data_space_is_a_plain_labelled_euclidean():
    D = pg.DataSpace(3, labels={"station": ["a", "b", "c"]})
    assert D is not None and isinstance(D, pg.Euclidean) and isinstance(D, DataSpace)
    assert D.is_standard and D.dim == 3 and D.field == "R"
    assert D.fields == ("station",) and list(D.select(station="b")) == [1]
    assert D == pg.Euclidean(3)                                          # the same space as R^3 (labels name coordinates)
    assert D != pg.DataSpace(3, labels=["x", "y", "z"])                  # different labels: different spaces
    assert "data space" in repr(D)
    d = D([1.0, 2.0, 3.0])
    assert D.norm(d) == pytest.approx(np.sqrt(14.0), rel=1e-15)


@pytest.mark.parametrize("kw", [dict(weights=[1.0, 2.0]), dict(metric=np.eye(2)), dict(covariance=0.5)])
def test_a_data_space_refuses_a_metric_and_names_the_choices(kw):
    with pytest.raises(Refused) as e:
        pg.DataSpace(2, **kw)
    text = str(e.value) + " ".join(e.value.needs)
    assert "whitened" in text and "Additive" in text and "Euclidean" in text


def test_a_data_space_with_another_inner_product_is_a_euclidean_not_a_data_space():
    D = pg.DataSpace(2, labels=["p", "q"])
    E = D.with_covariance(np.array([1.0, 4.0]))
    assert isinstance(E, pg.Euclidean) and not isinstance(E, DataSpace) and E.labels == ("p", "q")


# ------------------------------------------------------------------------------------------------ whitening
@pytest.mark.parametrize("C", [2.0, np.array([1.0, 4.0, 0.25]), spd(3, 1)], ids=["number", "diagonal", "dense"])
def test_whitened_coordinates_are_an_isometry_onto_r_n(C):
    D = pg.DataSpace(3)
    W = D.whitened(C)
    assert isinstance(W, Whitening) and W.covariance is not None
    Cm = np.diag(np.broadcast_to(np.asarray(C, float), (3,))) if np.ndim(C) < 2 else C
    assert isinstance(W.codomain, DataSpace) and W.codomain.is_standard
    assert W.domain == pg.Euclidean(3, covariance=C)                    # D's vectors with the C^-1 inner product
    d = D([1.0, -2.0, 0.5])
    z = W(d)                                                             # an element of D is accepted (the same vectors)
    assert z.space == W.codomain
    assert W.codomain.norm(z) ** 2 == pytest.approx(d.values @ np.linalg.solve(Cm, d.values), rel=1e-12)
    M = W.matrix()
    assert np.allclose(M @ Cm @ M.T, np.eye(3), atol=1e-12)              # whitened noise: N(0, I)
    back = W.inv(z)
    assert back.space == W.domain and np.allclose(back.values, d.values, rtol=1e-13, atol=1e-14)
    assert W.adjoint is W.inv                                            # an isometry onto: adjoint = inverse
    assert W.claims.status("isometry") == (Truth.TRUE, THEOREM)
    assert W.norm_bound() == 1.0


def test_whitening_keeps_labels_where_the_coordinates_stay_apart():
    D = pg.DataSpace(2, labels=["p", "q"])
    assert D.whitened(np.array([1.0, 9.0])).codomain.labels == ("p", "q")   # diagonal: z_i = d_i / sigma_i
    assert D.whitened(spd(2, 4)).codomain.labels is None                    # dense: z_k mixes d_1, ..., d_k
    assert D.whitened(spd(2, 4)).domain.labels == ("p", "q")


def test_whitening_refuses_elements_of_other_spaces():
    W = pg.DataSpace(2).whitened(1.0)
    with pytest.raises(SpaceMismatch):
        W(pg.Euclidean(3)([1.0, 2.0, 3.0]))


def test_whitened_on_a_euclidean_space_starts_from_its_vectors():
    E = pg.Euclidean(2, weights=[1.0, 3.0])                          # D's own metric is not the noise (Q1)
    W = E.whitened(np.array([4.0, 1.0]))
    assert W.domain == pg.Euclidean(2, covariance=np.array([4.0, 1.0]))
    assert np.allclose(W(E([2.0, 1.0])).values, [1.0, 1.0])


# ------------------------------------------------------------------------------------------------ direct sums of Euclideans
def test_a_direct_sum_of_plain_data_spaces_is_a_data_space_with_block_labels():
    E = euclidean_sum({"P": pg.DataSpace(2, labels=["s1", "s2"]), "S": pg.Euclidean(3)})
    assert isinstance(E, DataSpace) and E.dim == 5 and E.is_standard
    assert E.fields == ("block", "coordinate")
    assert E.labels[0] == ("P", "s1") and E.labels[4] == ("S", 2)
    assert list(E.select(block="S")) == [2, 3, 4]
    assert list(E.groups("block")) == [("P",), ("S",)]


def test_a_weighted_direct_sum_of_euclideans_has_the_sums_inner_product():
    Ea, Eb = pg.Euclidean(2, weights=[1.0, 2.0]), pg.Euclidean(2, metric=spd(2, 3))
    E = euclidean_sum([Ea, Eb], weights=[3.0, 0.5])
    X = pg.DirectSum([Ea, Eb], weights=[3.0, 0.5])
    assert isinstance(E, pg.Euclidean) and not isinstance(E, DataSpace)
    a, b = np.array([1.0, -1.0, 2.0, 0.5]), np.array([0.3, 2.0, -1.0, 1.0])
    x, y = X({0: a[:2], 1: a[2:]}), X({0: b[:2], 1: b[2:]})
    assert E.inner(E(a), E(b)) == pytest.approx(X.inner(x, y), rel=1e-13)
    assert list(E.select(block=1)) == [2, 3]
    assert isinstance(euclidean_sum([Ea, Ea], weights=[1.0, 2.0]), pg.Euclidean)
    assert euclidean_sum([Ea, Ea], weights=[1.0, 2.0]).weights is not None   # diagonal metrics stay diagonal


def test_euclidean_sum_refuses_other_summands():
    with pytest.raises(TypeError, match="DirectSum"):
        euclidean_sum([pg.Euclidean(2), pg.L2(pg.Interval(0, 1))])
    with pytest.raises(TypeError, match="DirectSum"):
        euclidean_sum([pg.Euclidean(2), pg.Euclidean(2, field="C")])


# ------------------------------------------------------------------------------------------------ the column operator
def test_the_column_operator_into_euclideans_is_the_stacked_matrix():
    X = pg.Euclidean(3, metric=spd(3, 5))
    A, B = np.random.default_rng(1).normal(size=(2, 3)), np.random.default_rng(2).normal(size=(1, 3))
    Da, Db = pg.Euclidean(2, covariance=np.array([0.5, 2.0])), pg.DataSpace(1)
    G = ColumnOperator({"a": pg.Matrix(A, X, Da, name="A"), "b": pg.Matrix(B, X, Db, name="B")})
    assert isinstance(G.codomain, pg.Euclidean) and list(G.codomain.select(block="b")) == [2]
    assert G.ranges == [(0, 2), (2, 3)] and G.names == ("a", "b")
    assert np.allclose(G.matrix(), np.vstack([A, B]))
    Wd = np.diag([2.0, 0.5, 1.0])
    Wx = X.metric_matrix()
    assert np.allclose(G.adjoint.matrix(), np.linalg.solve(Wx, np.vstack([A, B]).T @ Wd), rtol=1e-12)
    x = X([1.0, 2.0, -1.0])
    assert G.block(G(x), "a").space == Da and np.allclose(G.block(G(x), 1).values, B @ x.values)
    assert pg.relations.status(G, "linear")[0] is Truth.TRUE and pg.relations.status(G, "bounded")[0] is Truth.TRUE


def test_the_column_operator_into_a_direct_sum_of_hilbert_spaces():
    I = pg.Interval(0.0, 1.0)
    H = pg.L2(I)
    X = pg.Euclidean(2)
    f, one = H.element(lambda t: t, resolution=0.5), H.element(lambda t: 1.0 + 0.0 * t, resolution=0.5)
    J = pg.LinearOperator(X, H, apply=lambda x: x.values[0] * f + x.values[1] * one,           # x -> x_1 t + x_2
                          adjoint=lambda u: X([H.inner(u, f), H.inner(u, one)]), name="J")
    G = ColumnOperator([J, pg.Matrix(np.array([[1.0, 2.0]]), X, pg.Euclidean(1))])
    assert isinstance(G.codomain, pg.DirectSum) and G.ranges is None
    x, y = X([1.0, -0.5]), G.codomain({0: H.element(lambda t: t ** 2, resolution=0.5), 1: [3.0]})
    assert G.codomain.inner(G(x), y) == pytest.approx(X.inner(x, G.adjoint(y)), rel=1e-10)


def test_column_operator_refuses_blocks_on_different_domains():
    with pytest.raises(SpaceMismatch):
        ColumnOperator([pg.Identity(pg.Euclidean(2)), pg.Identity(pg.Euclidean(3))])


# ------------------------------------------------------------------------------------------------ stack and the old path
def test_stack_builds_a_column_operator_with_block_labels():
    X = pg.Euclidean(2)
    G1, G2 = pg.Matrix(np.eye(2), X, pg.Euclidean(2)), pg.Matrix(np.ones((1, 2)), X, pg.Euclidean(1))
    G = pg.stack(G1, G2, rule="all")
    assert isinstance(G, ColumnOperator) and isinstance(G.codomain, DataSpace)
    assert G.codomain.fields == ("block", "coordinate")
    R = pg.stack(pg.Additive(G1, pg.NormBall(G1.codomain, 1.0)), pg.Additive(G2, pg.NormBall(G2.codomain, 1.0)),
                 rule="all")
    assert R.noise.blocks == [(0, 2), (2, 3)]


def test_stacked_operator_is_gone():
    from pygeoinf0.relations import algebra
    assert not hasattr(algebra, "StackedOperator") and not hasattr(algebra, "stacked_space")
    assert not hasattr(pg.relations, "StackedOperator")
    with pytest.raises(ImportError):
        exec("from pygeoinf0.relations import StackedOperator")
    assert importlib.import_module("pygeoinf0.operators.blocks").ColumnOperator is pg.operators.ColumnOperator
    assert pg.DataSpace is DataSpace and pg.spaces.euclidean_sum is euclidean_sum
