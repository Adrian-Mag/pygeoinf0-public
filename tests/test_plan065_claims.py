"""Plan 065 P1 (architecture review 2026-11-13, A3, A4, A5, A9): claims that were not true. Each test failed on main
a45553c (the false statement) and passes with the repair (the true statement or a refusal naming the choices)."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN, status_of
from pygeoinf0.spaces.elements import Vector
from pygeoinf0.core.errors import HypothesisUnmet, Refused, Unsupported
from pygeoinf0.core.kinds import POINT
from pygeoinf0.operators.linear import LinearOperator
from pygeoinf0.relations.algebra import status


class _Square(LinearOperator):
    """x -> x^2 entrywise: a subclass that states nothing about itself (it is not linear)."""

    def __init__(self, X):
        super().__init__(X, X, name="sq")

    def _apply(self, x):
        return Vector(self.codomain, x.values ** 2)


# ------------------------------------------------------------------------------------------------ A3
def test_a3_a_bare_subclass_is_not_linear_by_its_class():
    E = pg.Euclidean(2)
    sq = _Square(E)
    assert status(sq, "linear") == (pg.Truth.UNKNOWN, UNKNOWN)            # main: (True, theorem) for x -> x^2
    assert status(sq, "bounded") == (pg.Truth.UNKNOWN, UNKNOWN)           # main: (True, theorem) by the Euclidean rule alone


def test_a3_package_classes_state_their_linearity_with_a_reason():
    E = pg.Euclidean(2)
    A = pg.Matrix(np.array([[1.0, 2.0], [0.0, 1.0]]))
    for op in (A, pg.Identity(E), pg.Diagonal([1.0, 2.0]), A @ A, A + A, A.inv):
        assert status(op, "linear") == (pg.Truth.TRUE, THEOREM) or status(op, "linear")[1] == DERIVED, op
        assert status(op, "bounded")[0] is pg.Truth.TRUE, op
    assert status(pg.LinearOperator(E, E, lambda x: x), "linear") == (pg.Truth.TRUE, ASSERTED)
    assert status(pg.LinearOperator(E, E, lambda x: x), "bounded") == (pg.Truth.TRUE, ASSERTED)   # finite dimension, given linear


def test_a3_a_subclass_of_a_package_class_that_changes_the_action_states_nothing():
    class Twisted(pg.Matrix):
        def _apply(self, x):
            return Vector(self.codomain, np.sin(x.values))
    assert status(Twisted(np.eye(2)), "linear") == (pg.Truth.UNKNOWN, UNKNOWN)


def test_a3_the_forward_cells_choose_their_row_by_claims():
    E = pg.Euclidean(2)
    S = pg.NormBall(E, 1.0)
    with pytest.raises(HypothesisUnmet):                     # main: the image of S under x -> x^2 by the linear rule
        pg.push(S, _Square(E))
    f = pg.Map(E, E, lambda x: x, linear="the identity, by inspection")
    with pytest.raises(Unsupported) as e:                    # main: "the image of a set under a nonlinear map"
        pg.push(S, f)
    assert "nonlinear" not in str(e.value) and "LinearOperator" in str(e.value)


# ------------------------------------------------------------------------------------------------ A4
def test_a4_the_domain_of_an_unbounded_operator_is_not_claimed_closed():
    I = pg.Interval("[0, 1]")
    R = pg.SobolevForm(I, 1, length_scale=0.2).operator()                       # a relation (plan 068 P3)
    held, st = R.dom.claims.status("closed")
    assert held is not pg.Truth.TRUE                                  # main: "closed linear subspace" at theorem
    assert R.dom.claims.status("dense") == (pg.Truth.TRUE, THEOREM)
    assert R.claims.status("closed graph") == (pg.Truth.TRUE, THEOREM)  # Kato VI.2.1, not the bounded-G rule
    assert all("G is bounded" not in c.reason for c in R.claims)


def test_a4_a_linear_relation_adds_no_claim_whose_premises_it_has_not_checked():
    E = pg.Euclidean(2)
    R = pg.LinearRelation(domain=E, codomain=E)               # no operator part
    assert all("G is" not in c.reason for c in R.claims)      # main: "G is linear", "G is bounded" with no G
    Rs = pg.LinearRelation(_Square(E))                        # an operator part whose linearity is not on record
    assert Rs.claims.status("linear")[0] is not pg.Truth.TRUE
    assert Rs.claims.status("closed graph")[0] is not pg.Truth.TRUE
    Rm = pg.LinearRelation(pg.Matrix(np.eye(2)))
    assert Rm.claims.status("linear") == (pg.Truth.TRUE, THEOREM) and Rm.claims.status("closed graph")[0] is pg.Truth.TRUE


# ------------------------------------------------------------------------------------------------ A5
def test_a5_matrix_products_keep_the_derived_positivity():
    G = pg.Matrix(np.arange(6.0).reshape(2, 3) + np.eye(2, 3))
    GG = G @ G.adjoint
    assert isinstance(GG, pg.Matrix)                          # the structural shortcut is kept
    assert GG.positivity()[:2] == ("psd", DERIVED)            # main: (None, unknown) while Composition says derived
    assert GG.is_selfadjoint()[:2] == (True, DERIVED)
    D = pg.Diagonal([1.0, 2.0])
    assert (D @ D.adjoint).positivity()[0] in ("psd", "pd")


def test_a5_self_adjointness_and_positivity_live_in_the_claims():
    I = pg.Identity(pg.Euclidean(2))
    assert status(I, "self-adjoint") == (pg.Truth.TRUE, THEOREM)       # main: (None, unknown): kept beside the claims
    assert I.claims.status("positive definite") == (pg.Truth.TRUE, THEOREM)
    A = pg.Matrix(np.eye(2))
    A._sa = (True, ASSERTED, "stated")
    assert A._sa == (True, ASSERTED, "stated") and A.claims.find("self-adjoint").reason == "stated"
    A._sa = None
    assert A.claims.find("self-adjoint") is None or A.claims.find("self-adjoint").status == CHECKED


def test_a5_a_missing_or_failed_record_is_not_promoted():
    assert status_of(None) == UNKNOWN                         # main: asserted, a hypothesis nobody stated
    assert status_of((False, CHECKED, "W A is not symmetric")) == UNKNOWN    # main: checked, for a property that fails


def test_a5_a_covariance_checked_not_self_adjoint_is_refused():
    E = pg.Euclidean(2)
    C = pg.Matrix(np.array([[2.0, 1.0], [0.0, 1.0]]))         # not symmetric
    with pytest.raises(Refused):                             # main: a measure; its posterior covariance "self-adjoint,
        pg.GaussianMeasure(E, covariance=C)                   # checked"


def test_a5_the_posterior_covariance_cites_the_prior_measures_assertion():
    E, D = pg.Euclidean(2), pg.Euclidean(1)
    C = pg.LinearOperator(E, E, lambda x: x, adjoint=lambda x: x, name="C")    # no record of self-adjointness
    C.is_selfadjoint = lambda: (None, UNKNOWN, "not known")
    prior = pg.GaussianMeasure(E, covariance=C)
    assert prior.claims[-2].status == ASSERTED                # a covariance is assumed by building the measure
    post = prior.condition(pg.Matrix(np.array([[1.0, 1.0]]), domain=E, codomain=D), D(np.zeros(1)), D.standard_gaussian())
    sa = post.covariance._sa
    assert sa[1] == ASSERTED and "premise: the covariance is self-adjoint" in sa[2]   # named, not derived from nothing


def test_a5_definiteness_of_g_c_g_plus_noise_needs_a_definite_noise_on_record():
    E, D = pg.Euclidean(2), pg.Euclidean(2)
    Ce = pg.Matrix(np.diag([1.0, 0.0]), name="Ce")
    Ce._pos = ("psd", ASSERTED, "stated")                       # semidefinite only
    prior, noise = pg.GaussianMeasure(E), pg.GaussianMeasure(D, covariance=Ce)
    G = pg.Matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    import pygeoinf0.operators.linear as L
    real_sum = L.LinearOperator.__add__
    made = []

    def add(a, b):
        out = real_sum(a, b)
        made.append(out)
        return out
    L.LinearOperator.__add__ = add
    try:
        with pytest.raises(Refused):
            prior._condition(G, D(np.zeros(2)), noise)            # G C G* + C_e = diag(2, 0): singular, refused
    finally:
        L.LinearOperator.__add__ = real_sum
    S = made[0]
    assert S._pos is None or S._pos[0] == "psd"                 # main: "pd", "C_e positive definite", never stated


# ------------------------------------------------------------------------------------------------ A9
def test_a9_a_family_is_not_a_point_state():
    E = pg.Euclidean(2)
    F = E.family(np.eye(2))
    with pytest.raises(TypeError):                            # main: "point"
        pg.state_kind(F)
    H = pg.DirectSum({"a": E, "b": E})
    Fh = H.family([H({"a": np.ones(2), "b": np.zeros(2)}), H({"a": np.zeros(2), "b": np.ones(2)})])
    with pytest.raises(TypeError):
        pg.state_kind(Fh)
    from pygeoinf0.core.kinds import declared_kind
    from pygeoinf0.states import is_element
    for x in (E(np.ones(2)), H({"a": np.ones(2), "b": np.zeros(2)})):   # elements declare the kind point (plan 067
        assert declared_kind(x) is POINT and is_element(x)               # P4: a point is not a kind of state)
        assert not is_element(F) and not is_element(Fh)


def test_a9_the_kind_is_declared_not_sniffed():
    class Lookalike:
        space, _memo = None, {}
    with pytest.raises(TypeError):                            # main: any object with .space and ._memo is a point
        pg.state_kind(Lookalike())
