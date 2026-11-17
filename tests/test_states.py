"""States of knowledge (plan 008 P2; plan 067 P4): the kinds of existing objects, the prior 'none' (the whole space),
orderings with a feasible set."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.kinds import MEASURE, ORDERING, SET


def test_kinds_of_existing_objects():
    D = pg.Euclidean(3)
    M = pg.Sobolev("[0, 1]", 1, length_scale=0.2)
    for x in (D([1.0, 2.0, 3.0]), M.riesz(0.3)):                  # a point is not a kind of state (plan 067 P4)
        with pytest.raises(TypeError, match="singleton"):
            pg.state_kind(x)
    assert pg.state_kind(D.standard_gaussian()) is MEASURE and pg.state_kind(pg.probability.cylindrical(M, alpha=2.0)) is MEASURE
    assert pg.state_kind(pg.NormBall(M, 1.0)) is SET and pg.state_kind(pg.Span([D([1.0, 0, 0])])) is SET
    assert pg.state_kind(pg.NoPrior(M)) is SET and pg.state_kind(pg.Ordering.norm(M)) is ORDERING
    with pytest.raises(pg.Refused, match="not stated"):
        pg.state_kind(None)
    with pytest.raises(TypeError):
        pg.state_kind(pg.Matrix(np.eye(3)))                        # a relation is not a state


def test_no_prior_is_the_whole_space_for_the_set_cells():
    M = pg.L2("[0, 1]")
    W = pg.NoPrior(M)
    f = M(np.sin, resolution=0.01)
    assert isinstance(W, pg.Whole) and W.member(f) is pg.Truth.TRUE
    assert W.support(f) == np.inf and W.support(M.zero) == 0.0


def test_orderings_carry_their_feasible_set():
    D = pg.Euclidean(2, weights=[1.0, 4.0])
    J = pg.Ordering.norm(D, centre=D([1.0, 0.0]))
    x = D([3.0, 1.0])
    assert J.cost(x) == pytest.approx(0.5 * (4.0 + 4.0))
    assert J.compare(D([1.0, 0.1]), x) is pg.Truth.TRUE
    K = pg.Ordering(D, J._cost, feasible=pg.NormBall(D, 1.0), name="J on B")
    assert K.cost(x) == np.inf and K.cost(D([0.5, 0.0])) == pytest.approx(0.125)
    with pytest.raises(pg.errors.Unsupported):
        K.summary()                                               # a summary is computed by a cell, not invented
    assert J.is_norm and not K.is_norm
