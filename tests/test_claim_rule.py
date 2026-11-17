"""Plan 023: the three defects EXP-036 found, as plain regression tests (no Hypothesis needed; test_laws.py has the
property versions with their planted-defect controls)."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.numbers import Enclosure
from pygeoinf0.core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN, by_rule, weakest
from pygeoinf0.spaces.metrics import Metric


def test_by_rule_is_the_weakest_of_derived_and_the_premises():
    assert by_rule() == DERIVED
    assert by_rule(THEOREM, THEOREM) == DERIVED
    assert by_rule(THEOREM, CHECKED) == CHECKED
    assert by_rule(DERIVED, ASSERTED) == ASSERTED
    assert by_rule(CHECKED, UNKNOWN) == UNKNOWN
    assert weakest() == UNKNOWN


def test_a_sum_of_checked_self_adjoint_matrices_is_checked_not_derived():
    E = pg.Euclidean(2)
    A = pg.Matrix(np.eye(2), domain=E, codomain=E)
    assert A.is_selfadjoint()[1] == CHECKED
    assert (A + A).is_selfadjoint()[1] == CHECKED                    # EXP-036: it was "derived"
    assert (3.0 * A).is_selfadjoint()[1] == CHECKED


def test_theorem_premises_still_give_derived():
    E = pg.Euclidean(3)
    Id = pg.Identity(E)
    assert (Id + 2.0 * Id).is_selfadjoint()[1] == DERIVED
    assert (Id + 2.0 * Id).positivity() == ("pd", DERIVED, "a positive combination of positive operators, one of them "
                                                         "definite")


def test_asserted_positivity_stays_asserted_through_inverse_blocks_and_metric():
    E = pg.Euclidean(2)
    A = pg.Matrix(np.diag([1.0, 2.0]), domain=E, codomain=E)
    A._sa, A._pos = (True, ASSERTED, "stated"), ("pd", ASSERTED, "stated")
    assert A.inv.positivity()[1] == ASSERTED
    X = pg.DirectSum({"a": E, "b": E})
    D = pg.BlockDiagonal(X, [A, pg.Identity(E)])
    assert D.is_selfadjoint()[1] == ASSERTED and D.positivity()[1] == ASSERTED
    assert Metric(2.0 * A, E, "metric").ops.claims[-1].status == ASSERTED    # it took the stronger status before


def test_conditioning_does_not_promote_an_asserted_prior():
    E, D = pg.Euclidean(2), pg.Euclidean(1)
    C = pg.Matrix(np.diag([1.0, 2.0]), domain=E, codomain=E)
    C._sa, C._pos = (True, ASSERTED, "stated"), ("pd", ASSERTED, "stated")
    prior = pg.GaussianMeasure(E, covariance=C)
    post = prior.condition(pg.Matrix(np.array([[1.0, 1.0]]), domain=E, codomain=D), D(np.zeros(1)), D.standard_gaussian())
    assert post.covariance.is_selfadjoint()[1] == ASSERTED
    assert post.covariance.positivity()[1] == ASSERTED


def test_a_nearly_symmetric_dense_metric_gives_an_exactly_symmetric_inner_product():
    W = np.eye(2) + 4e-15 * np.array([[0.0, 1.0], [-1.0, 0.0]])        # accepted: symmetric within the tolerance
    E = pg.Euclidean(2, metric=W)
    e0, e1 = E(np.array([1.0, 0.0])), E(np.array([0.0, 1.0]))
    assert (e1 @ e0) == (e0 @ e1)                                       # EXP-036: they differed by 8e-15


@pytest.mark.parametrize("value, lower, upper", [(math.inf, 0.0, math.inf), (1.0, -math.inf, math.inf),
                                                 (-math.inf, -math.inf, 0.0)])
def test_scaling_a_bracket_by_zero_gives_zero(value, lower, upper):
    s = Enclosure(value, lower, upper).scaled(0)
    assert (float(s), s.lower, s.upper) == (0.0, 0.0, 0.0)              # EXP-036: the value was NaN


def test_scaling_a_bracket_by_a_negative_number_swaps_its_ends():
    s = Enclosure(1.0, 0.5, 2.0, CHECKED, "r").scaled(-2)
    assert (float(s), s.lower, s.upper, s.status) == (-2.0, -4.0, -1.0, CHECKED)
