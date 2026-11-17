"""Relations (plan 008 P1): kinds, composition laws by enumeration on finite sets, embeddings (unit laws), converses,
refusals, claims of the additive constructors, and LinearOperator as a point relation without behaviour change."""
import itertools

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.claims import ASSERTED, DERIVED, UNKNOWN
from pygeoinf0.convex import NormBall
from pygeoinf0.sets import PredicateSubset
from pygeoinf0.core.kinds import MEASURE, POINT, SET
from pygeoinf0.relations import MarkovKernel, Map, SetValued, as_kind, compose, status

X, Y, Z, W = (pg.FiniteSet(range(n)) for n in (4, 3, 5, 2))


def random_relation(A, B, seed, p=0.45):
    rng = np.random.default_rng(seed)
    table = {(float(a), float(b)): bool(rng.random() < p) for a in A.points for b in B.points}
    return SetValued(A, B, member=lambda a, b: table[(float(a), float(b))], name=f"R{seed}")


def random_kernel(A, B, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((len(A), len(B)))
    P /= P.sum(axis=1, keepdims=True)
    law = lambda a: pg.DiscreteMeasure({float(b): float(P[int(a), j]) for j, b in enumerate(B.points)}, on=B)  # noqa
    return MarkovKernel(A, B, law, name=f"K{seed}"), P


def test_kinds():
    D = pg.Euclidean(3)
    G = pg.Matrix(np.ones((3, 2)), codomain=D)
    assert isinstance(G, pg.Relation) and G.kind is POINT
    assert pg.Additive(G, pg.Span([D([1.0, 0, 0])])).kind is SET
    assert pg.Additive(G, D.standard_gaussian()).kind is MEASURE
    assert isinstance(G.dom, pg.Whole) and G.mul.dim == 0


def test_set_composition_is_associative_and_the_converse_reverses_it():
    R, S, T = random_relation(X, Y, 1), random_relation(Y, Z, 2), random_relation(Z, W, 3)
    SR = compose(S, R)
    for x, z in itertools.product(X.points, Z.points):        # the definition, by enumeration
        assert bool(SR.member(x, z)) == any(bool(R.member(x, y)) and bool(S.member(y, z)) for y in Y.points)
    left, right = compose(T, SR), compose(compose(T, S), R)
    assert all(bool(left.member(x, w)) == bool(right.member(x, w)) for x in X.points for w in W.points)
    Rc = R.converse()
    assert Rc.converse() is R
    assert all(bool(Rc.member(y, x)) == bool(R.member(x, y)) for x in X.points for y in Y.points)
    lhs, rhs = SR.converse(), compose(R.converse(), S.converse())       # (S o R)^-1 = R^-1 o S^-1
    assert all(bool(lhs.member(z, x)) == bool(rhs.member(z, x)) for x in X.points for z in Z.points)


def test_kernel_composition_is_chapman_kolmogorov():
    (K, P), (L, Q), (M, S) = random_kernel(X, Y, 4), random_kernel(Y, Z, 5), random_kernel(Z, W, 6)
    a, b = compose(M, compose(L, K)), compose(compose(M, L), K)
    ref = P @ Q @ S
    for i, x in enumerate(X.points):
        assert np.allclose([a(x).at(w) for w in W.points], ref[i], rtol=0, atol=1e-15)
        assert np.allclose([b(x).at(w) for w in W.points], ref[i], rtol=0, atol=1e-15)


def test_embeddings_of_points_are_units():
    f = Map(X, Y, lambda x: float(int(x) % 3), name="f")
    S = random_relation(Y, Z, 7)
    Sf = compose(S, as_kind(f, SET))                        # singletons: S o {f} = S(f(.))
    assert all(bool(Sf.member(x, z)) == bool(S.member(f(x), z)) for x in X.points for z in Z.points)
    L, _ = random_kernel(Y, Z, 8)
    Lf = compose(L, as_kind(f, MEASURE))                    # Dirac: L o delta_f = L(f(.))
    assert all(dict(Lf(x).items()) == pytest.approx(dict(L(f(x)).items()), abs=1e-15) for x in X.points)
    assert as_kind(f, POINT) is f


def test_off_diagonal_compositions_and_conversions_are_refused():
    R, (K, _) = random_relation(X, Y, 9), random_kernel(Y, Z, 10)
    with pytest.raises(pg.Refused, match="family of measures"):
        compose(K, R)
    D = pg.Euclidean(2)
    G = pg.Matrix(np.eye(2), D, D)
    L = pg.Additive(G, D.standard_gaussian())
    B = pg.Additive(G, NormBall(D, 1.0))
    with pytest.raises(pg.Refused, match="random set"):
        compose(B, L)
    with pytest.raises(pg.Refused) as e:
        L.converse()
    assert any("prior" in n for n in e.value.needs)
    with pytest.raises(pg.Refused, match="no canonical"):
        as_kind(B, MEASURE)
    with pytest.raises(pg.Refused, match="no canonical"):
        as_kind(L, SET)
    with pytest.raises(pg.Refused, match="stated rule"):
        pg.stack(G, G)


def test_a_map_and_its_fibres():
    D = pg.Euclidean(2)
    f = Map(D, pg.Euclidean(1), lambda x: pg.Euclidean(1)([x.values @ x.values]), name="|x|^2")
    fc = f.converse()
    assert fc.member(pg.Euclidean(1)([2.0]), D([1.0, 1.0])) is pg.Truth.TRUE
    assert fc.member(pg.Euclidean(1)([2.0]), D([1.0, 0.0])) is pg.Truth.FALSE
    half = Map(D, D, lambda x: x, dom=PredicateSubset(D, lambda x: x.values[0] >= 0))
    with pytest.raises(pg.Refused, match="not in dom"):
        half(D([-1.0, 0.0]))


def test_claims_of_the_additive_constructors():
    D = pg.Euclidean(3)
    G = pg.Matrix(np.random.default_rng(0).normal(size=(3, 2)), codomain=D)
    B = pg.Additive(G, NormBall(D, 1.0))
    assert B.claims.status("convex graph") == (pg.Truth.TRUE, DERIVED)
    assert B.claims.status("closed graph") == (pg.Truth.TRUE, DERIVED)
    open_set = pg.Additive(G, PredicateSubset(D, lambda e: e.norm() < 1))     # an open ball: no claim
    assert open_set.claims.status("closed graph") == (pg.Truth.UNKNOWN, UNKNOWN)
    L = pg.Additive(G, D.standard_gaussian())
    assert L.claims.status("measurable") == (pg.Truth.TRUE, DERIVED)
    assert L.claims.status("independent noise") == (pg.Truth.TRUE, ASSERTED)
    M = pg.L2("[0, 1]")
    H = pg.LinearOperator(M, D, apply=lambda m: D(np.zeros(3)), adjoint=lambda y: M.zero)    # bounded: not on record
    assert status(H, "bounded") == (pg.Truth.UNKNOWN, UNKNOWN)
    assert pg.Additive(H, NormBall(D, 1.0)).claims.status("closed graph") == (pg.Truth.UNKNOWN, UNKNOWN)
    pg.assert_claim(H, "bounded", "the zero map")
    assert status(H, "bounded") == (pg.Truth.TRUE, ASSERTED)


def test_linear_operator_is_unchanged_and_composes_with_relations():
    D = pg.Euclidean(3)
    rng = np.random.default_rng(1)
    A, B = pg.Matrix(rng.normal(size=(3, 3)), D, D), pg.Matrix(rng.normal(size=(3, 3)), D, D)
    x = D(rng.normal(size=3))
    assert np.array_equal((A @ B)(x).values, (A.A @ B.A) @ x.values)
    assert np.array_equal(A(x).values, A.A @ x.values)
    R = pg.Additive(B, pg.Span([D([1.0, 1.0, 0.0])]))
    AR = A @ R
    assert isinstance(AR, pg.LinearRelation) and AR.mul.dim == 1
    assert AR(x).member(A(B(x)) + D(A.A @ np.array([2.0, 2.0, 0.0]))) is pg.Truth.TRUE
    from pygeoinf0.operators.sparse import KroneckerMatrix, SparseMatrix       # their module (plan 065 E4: no forwarder)
    assert SparseMatrix is pg.SparseMatrix and KroneckerMatrix is pg.KroneckerMatrix


def test_gaussian_kernels_compose_by_chapman_kolmogorov():
    D = pg.Euclidean(2)
    rng = np.random.default_rng(2)
    G, H = pg.Matrix(rng.normal(size=(2, 2)), D, D), pg.Matrix(rng.normal(size=(2, 2)), D, D)
    L1, L2 = pg.Additive(G, D.standard_gaussian()), pg.Additive(H, D.standard_gaussian())
    L = compose(L2, L1)
    x = D([0.3, -0.2])
    law = L(x)
    assert np.allclose(law.mean.values, H.A @ G.A @ x.values)
    assert np.allclose(law.covariance.matrix(), H.A @ H.A.T + np.eye(2))
