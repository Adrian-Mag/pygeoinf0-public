"""Plan 067 P4: states of knowledge (review B3, A9, D8; decision Q12; the author's amendment 3).

The laws:
- the kinds are closed and live in core/kinds.py: set, measure and ordering are the kinds of state, point, set and
  measure the kinds of relation; NONE, FAMILY, RANDOM_SET and register_kind are gone, and core/relation.py holds none;
- every state declares its kind on its class (the State protocol: space, kind, claims); an instance attribute or other
  attributes never make a state; elements, families, base measures and relations are not states;
- NoPrior(X) is Whole(X): one state, so update and push give one answer by one route (cells B1 and B5 were two);
- a point is the singleton set or the Dirac measure (states.as_set, as_measure); the conversions between kinds live in
  states/ and take their stated choice as an argument;
- D8: methods and cells choose by kind and claims, not by class names; as_kind(POINT) of a single-valued total
  LinearRelation is its operator part.
"""
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core import kinds
from pygeoinf0.core.kinds import MEASURE, ORDERING, POINT, SET

PKG = Path(pg.__file__).parent
rng = np.random.default_rng(67)


def linear_problem(n=6, k=3, seed=1):
    r = np.random.default_rng(seed)
    X, D = pg.Euclidean(n), pg.Euclidean(k)
    G = pg.Matrix(r.normal(size=(k, n)), X, D)
    return X, D, G, D(r.normal(size=k))


def some_states():
    X, D, G, d = linear_problem()
    M = pg.Sobolev("[0, 1]", 1, length_scale=0.2)
    mu = X.standard_gaussian()
    return [pg.NormBall(X, 1.0), pg.Ellipsoid(X, 1.0, np.ones(6)), pg.CoordinateBox(X, -1.0, 1.0),
            pg.Span([X(np.eye(6)[0])]), pg.Whole(X), pg.EmptySet(X), pg.convex.sets.singleton(X.zero),
            pg.sets.PredicateSubset(X, lambda x: True), pg.NormBall(X, 1.0) & pg.NormBall(X, 2.0),
            pg.update(pg.NormBall(X, 1.0), pg.Additive(G, pg.NormBall(D, 0.5)), d),
            mu.harden(pg.Identity(X), 0.9), pg.probability.cylindrical(M, alpha=2.0), mu, pg.Dirac(X.zero),
            pg.DiscreteMeasure({0.0: 0.5, 1.0: 0.5}), pg.Ordering.norm(X)]


# ------------------------------------------------------------------------------------------------ the kinds
def test_the_kinds_are_closed_in_core_kinds():
    assert kinds.STATE_KINDS == (SET, MEASURE, ORDERING)
    assert kinds.RELATION_KINDS == (POINT, SET, MEASURE)
    for gone in ("NONE", "FAMILY", "RANDOM_SET", "register_kind", "KINDS"):
        assert not hasattr(kinds, gone), gone
    from pygeoinf0.core import relation
    for name in ("Kind", "POINT", "SET", "MEASURE", "ORDERING", "NONE", "FAMILY", "RANDOM_SET", "register_kind",
                 "KINDS"):
        assert not hasattr(relation, name), name                  # kinds live in core/kinds.py, not with Relation


def test_every_state_declares_its_kind_on_its_class():
    for s in some_states():
        k = type(s).kind
        assert k in kinds.STATE_KINDS, s
        assert any("kind" in vars(c) for c in type(s).__mro__), s
        assert pg.state_kind(s) is k and isinstance(s, pg.State), s
        assert hasattr(s, "space") and hasattr(s, "claims"), s
        assert not hasattr(s, "is_state") and not hasattr(s, "is_set"), s     # the declared kind replaces the flags
    expected = [SET] * 11 + [MEASURE] * 4 + [ORDERING]
    assert [pg.state_kind(s) for s in some_states()] == expected


def test_what_is_not_a_state():
    X, D, G, d = linear_problem()

    class Lookalike:                                              # an instance attribute is not a declaration
        def __init__(self):
            self.space, self.kind, self.claims = X, SET, ()

    class Sniffed:                                                # attributes are not a declaration
        space, covariance = X, None

        def sample(self):
            return X.zero
    for obj in (Lookalike(), Sniffed(), X.family(np.eye(6)), pg.LebesgueMeasure("[0, 1]"), G,
                pg.Additive(G, pg.NormBall(D, 1.0)), pg.Additive(G, D.standard_gaussian())):
        assert not isinstance(obj, pg.State), obj
        with pytest.raises(TypeError):
            pg.state_kind(obj)


def test_a_point_is_a_singleton_or_a_dirac_not_a_state_kind():
    X, D, G, d = linear_problem()
    x = X(rng.normal(size=6))
    assert kinds.declared_kind(x) is POINT and not isinstance(x, pg.State)
    with pytest.raises(TypeError, match="singleton"):
        pg.state_kind(x)
    S = pg.states.as_set(x)
    assert pg.state_kind(S) is SET and S.member(x) is pg.Truth.TRUE and S.member(X.zero) is pg.Truth.FALSE
    m = pg.states.as_measure(x)
    assert isinstance(m, pg.Dirac) and np.array_equal(m.mean.values, x.values) and pg.state_kind(m) is MEASURE
    # the table still accepts an element as the asserted point: its answer is the image of the singleton
    assert np.array_equal(pg.push(x, G).values, G(x).values)
    assert pg.push(S, G).member(G(x)) is pg.Truth.TRUE


def test_the_cells_are_keyed_by_state_kinds_and_the_point_argument():
    from pygeoinf0.cells import BACKWARD, FORWARD
    for c in FORWARD + BACKWARD:
        assert c.state in kinds.STATE_KINDS + (POINT,), c
        assert c.relation in kinds.RELATION_KINDS, c
    names = {c.name for c in FORWARD + BACKWARD}
    assert not names & {"B1", "B2", "B3", "F1", "F2", "F3"}         # the rows of "none": the whole space's rows now
    assert "| none |" not in pg.cells.table()


# ------------------------------------------------------------------------------------------------ NoPrior is Whole
def test_no_prior_is_the_whole_space():
    M = pg.L2("[0, 1]")
    W = pg.NoPrior(M)
    assert isinstance(W, pg.Whole) and W == pg.Whole(M) and pg.state_kind(W) is SET
    f = M(np.sin, resolution=0.01)
    assert W.member(f) is pg.Truth.TRUE
    assert W.support(f) == np.inf and W.support(M.zero) == 0.0


def _same(K1, K2, X):
    assert type(K1) is type(K2) and K1.cell == K2.cell
    if hasattr(K1, "summary"):
        assert X.norm(K1.summary() - K2.summary()) == 0.0


def test_one_state_one_route_backward():
    X, D, G, d = linear_problem()
    for R in (G, pg.Additive(G, pg.NormBall(D, 0.5)), pg.relations.LinearRelation(G),
              pg.relations.LinearRelation(G, mul=pg.Span([D(np.eye(3)[0])]))):
        _same(pg.update(pg.NoPrior(X), R, d), pg.update(pg.Whole(X), R, d), X)
    K = pg.update(pg.Whole(X), G, d)                             # the fibre G^-1(d), exact: m-dagger + ker G
    assert isinstance(K, pg.AffineSubset) and K.cell == "B5"
    m = K.summary()
    assert np.allclose(m.values, np.linalg.lstsq(G.A, d.values, rcond=None)[0], atol=1e-12)
    assert K.member(m) is pg.Truth.TRUE and K.member(m + X(G.A[0])) is pg.Truth.FALSE
    f = pg.Map(X, D, lambda x: G(x), name="f")                     # a map not claimed linear: membership
    for S in (pg.Whole(X), pg.NoPrior(X)):
        P = pg.update(S, f, d)
        assert P.member(m) is pg.Truth.TRUE and P.member(X.zero) is pg.Truth.FALSE


def test_one_state_one_route_forward():
    X, D, G, d = linear_problem()
    R = pg.Additive(G, pg.NormBall(D, 0.5))
    for Rel in (G, R):
        a, b = pg.push(pg.NoPrior(X), Rel), pg.push(pg.Whole(X), Rel)
        assert type(a) is type(b) and a.cell == b.cell
    assert pg.push(pg.Whole(X), G).member(G(X(rng.normal(size=6)))) is pg.Truth.TRUE


def test_no_prior_refusals_are_the_set_refusals():
    X, D, G, d = linear_problem()
    L = pg.Additive(G, D.standard_gaussian())
    with pytest.raises(pg.Refused, match="family"):
        pg.push(pg.NoPrior(X), L)
    with pytest.raises(pg.Refused, match="acceptance"):
        pg.update(pg.NoPrior(X), L, d)


# ------------------------------------------------------------------------------------------------ conversions
def test_conversions_take_their_stated_choice():
    X, D, G, d = linear_problem()
    mu = X.standard_gaussian()
    B = pg.NormBall(X, 1.0)
    assert pg.states.as_set(B) is B
    with pytest.raises(pg.Refused, match="harden"):
        pg.states.as_set(mu)                                      # a measure gives a set only with a stated choice
    S = pg.states.as_set(mu, form=pg.Identity(X), mass=0.9)
    T = pg.states.harden(mu, pg.Identity(X), 0.9)
    assert pg.state_kind(S) is SET and S.threshold == T.threshold == mu.harden(pg.Identity(X), 0.9).threshold
    with pytest.raises(pg.Refused, match="not developed"):
        pg.states.as_set(pg.Ordering.norm(X))
    with pytest.raises(pg.Refused, match="soft"):
        pg.states.as_measure(B)                                   # a set gives no probabilities
    assert pg.states.as_measure(mu) is mu


# ------------------------------------------------------------------------------------------------ D8
def test_as_kind_point_of_a_single_valued_total_linear_relation_is_its_operator():
    X, D, G, d = linear_problem()
    R = pg.relations.LinearRelation(G)
    assert R.as_kind(POINT) is G
    from pygeoinf0.convex.subspaces import Span
    with pytest.raises(pg.Refused, match="single-valued"):
        pg.relations.LinearRelation(G, mul=Span([D(np.eye(3)[0])])).as_kind(POINT)
    with pytest.raises(pg.Refused, match="total"):
        pg.relations.LinearRelation(G, dom=Span([X(np.eye(6)[0])])).as_kind(POINT)
    with pytest.raises(pg.Refused):
        pg.Additive(G, D.standard_gaussian()).as_kind(POINT)


_NAME_TEST = re.compile(r"__name__\s*(==|in\b)|_is\(\s*['\"]")


def test_no_class_name_tests_in_the_dispatch():
    for rel in ("inference/methods.py", "cells/backward.py", "cells/forward.py", "cells/registry.py",
                "relations/algebra.py"):
        text = (PKG / rel).read_text()
        assert not _NAME_TEST.search(text), (rel, _NAME_TEST.search(text).group(0))


def test_a_lookalike_class_name_does_not_choose_a_cell():
    X, D, G, d = linear_problem()

    class AdditiveSet(pg.relations.SetValued):                    # the name alone must not pick the BGP cell
        pass
    R = AdditiveSet(X, D, member=lambda x, y: pg.Truth.of(D.norm(G(x) - y) <= 1.0), name="R")
    R.G, R.noise = G, pg.NormBall(D, 1.0)
    K = pg.update(pg.NormBall(X, 1.0), R, d)
    assert not isinstance(K, pg.convex.feasible.FeasibleSet)


def test_methods_choose_by_kind_and_structure_not_by_name():
    X, D, G, d = linear_problem()
    T = pg.Matrix(np.ones((1, 6)), X, pg.Euclidean(1))
    from pygeoinf0.relations.kernels import AdditiveMarkovKernel
    from pygeoinf0.relations.additive import AdditiveSet

    class MyKernel(AdditiveMarkovKernel):
        pass

    class MySet(AdditiveSet):
        pass

    class MyBall(pg.NormBall):
        pass
    L = MyKernel(G, D.standard_gaussian())
    prob = pg.Problem(M=X, D=D, mdr=L, mpr=T, datum=d, prior=X.standard_gaussian())
    assert ("pli", True) in [(name, ok) for name, ok, _, _ in prob.methods()]
    assert isinstance(pg.inference.pli(X.standard_gaussian(), L, T, d), pg.GaussianMeasure)
    S = MySet(G, pg.NormBall(D, 0.5))
    assert pg.inference.METHODS["csola"].missing({"prior": pg.NormBall(X, 1.0), "mdr": S, "mpr": T, "datum": d}) == []
    assert pg.inference.METHODS["ocsola"].missing({"prior": MyBall(X, 1.0), "mdr": L, "mpr": T, "datum": d}) == []

    class Sniffed:                                                # attributes are not a measure state
        space, covariance = X, X.standard_gaussian().covariance
    assert pg.inference.METHODS["pli"].missing({"prior": Sniffed(), "mdr": L, "mpr": T, "datum": d})
    assert pg.inference.METHODS["bgp"].missing({"prior": pg.NoPrior(X), "mdr": S, "mpr": T, "datum": d}) == []


def test_relations_push_states_by_their_declared_kind():
    X, D, G, d = linear_problem()
    f = pg.Map(X, D, lambda x: G(x), name="f", linear="a matrix")

    class Sniffed:                                                # not a state: applied as an argument, and refused
        space, covariance = X, None
    with pytest.raises(Exception) as e:
        f(Sniffed())
    assert "not a state" not in str(e.value)                      # applied as an argument, not sent to the cells
    B = pg.NormBall(X, 1.0)
    with pytest.raises(pg.errors.Unsupported):                    # a set state is pushed (F7 for a rule map)
        f(B)
