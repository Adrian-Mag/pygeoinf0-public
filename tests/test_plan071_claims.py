"""Plan 071 P2 (Mac review items 1, 3, 7): claims whose premises were not checked.

Item 1: a functional given by a callable is linear by the caller's word (ASSERTED), not by a theorem; from_operator
states "bounded" only with its premise (the map's own claims and the package's rules). Item 3: SturmLiouville states
self-adjoint, bounded below, closed and densely defined only with p, w > 0 bounded away from 0 and q bounded below
checked (exactly for numbers and tables, on samples for callables), and refuses a p or w that is not positive. Item 7:
Scale(S, 0) gives an Enclosure with the claim status of S's support (a Bound's outcome is not a claim status), and
minkowski(Scale(K, 0), B) has a support.
"""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN
from pygeoinf0.core.errors import Refused
from pygeoinf0.core.logic import Truth
from pygeoinf0.core.numbers import Enclosure

R1, R3 = pg.Euclidean(1), pg.Euclidean(3)


# ------------------------------------------------------------------ item 1: functionals
def _square():
    return pg.Functional(R3, lambda x: float(np.sum(x.values ** 2)), name="|x|^2")


def test_a_callable_functional_is_linear_by_the_callers_word_only():
    l = _square()
    x = R3(np.array([1.0, 1.0, 1.0]))
    assert l(R3(2 * x.values)) == 12.0 and 2 * l(x) == 6.0            # not linear: no theorem may say it is
    held, st = l.claims.status("linear")
    assert held is Truth.TRUE and st == ASSERTED, l.claims
    assert "caller" in l.claims.find("linear").reason


def test_package_functionals_keep_their_theorem():
    for l in (R3.functional(np.array([1.0, 0.5, -1.0])), pg.L2(pg.Interval(0, 1)).functional(pg.LebesgueMeasure(
            pg.Interval(0.2, 0.6))), R3.dual().zero):
        held, st = l.claims.status("linear")
        assert held is Truth.TRUE and st == THEOREM, (l, l.claims)


def test_a_combination_is_as_linear_as_its_terms():
    r = R3.functional(np.array([1.0, 0.0, 0.0]))
    assert (r + r).claims.status("linear")[1] in (THEOREM, DERIVED)
    c = r + _square()
    held, st = c.claims.status("linear")
    assert held is Truth.TRUE and st == ASSERTED, c.claims
    assert c.claims.status("bounded")[1] not in (THEOREM, DERIVED), c.claims       # |x|^2 states no bound


def test_from_operator_states_bounded_only_with_its_premise():
    f = pg.LinearOperator(R3, R1, lambda x: R1(np.array([np.sum(x.values ** 2)])))   # linear by the caller's word
    l = pg.Functional.from_operator(f)
    assert l.claims.status("linear")[1] == ASSERTED, l.claims
    assert l.claims.status("bounded")[1] == ASSERTED, l.claims          # finite-dimensional, if linear
    Lp = pg.Lp(pg.Interval(0, 1), 3)
    g = pg.Functional.from_operator(pg.LinearOperator(Lp, R1, lambda u: R1(np.array([u.integral()]))))
    assert g.claims.status("bounded") == (Truth.UNKNOWN, UNKNOWN), g.claims       # nothing decides it on L^3
    assert g.claims.status("linear")[1] == ASSERTED, g.claims
    m = pg.Functional.from_operator(pg.Matrix(np.array([[1.0, -1.0, 2.0]]), domain=R3, codomain=R1))
    assert m.claims.status("linear") == (Truth.TRUE, THEOREM), m.claims
    assert m.claims.status("bounded") == (Truth.TRUE, THEOREM), m.claims


def test_from_operator_of_a_map_with_nothing_on_record_is_not_linear_by_theorem():
    class Map:                                                             # callable, into R^1, no claims at all
        domain, codomain, name = R3, R1, "F"

        def __call__(self, x):
            return R1(np.array([float(np.sum(x.values ** 2))]))
    l = pg.Functional.from_operator(Map())
    assert l.claims.status("linear")[1] not in (THEOREM, DERIVED), l.claims
    assert l.claims.status("bounded")[1] not in (THEOREM, DERIVED), l.claims


# ------------------------------------------------------------------ item 3: Sturm-Liouville
def _sl(*a, **k):
    from pygeoinf0.operators.unbounded import SturmLiouville
    return SturmLiouville(*a, **k)


@pytest.mark.parametrize("p, w", [(-1.0, 1.0), (0.0, 1.0), (lambda x: x - 0.5, 1.0), (1.0, -2.0),
                                  (pg.Tabulated("[0, 1]", [0.0, 1.0], [1.0, -1.0]), 1.0)])
def test_sturm_liouville_refuses_a_coefficient_that_is_not_positive(p, w):
    with pytest.raises(Refused, match="p > 0|w > 0"):
        _sl("[0, 1]", p, 0.0, w)


@pytest.mark.parametrize("p, q, w", [(1.0, 0.0, 1.0), (2.0, -5.0, 3.0),
                                     (pg.Tabulated("[0, 1]", [0.0, 1.0], [1.0, 3.0]), 0.5, 2.0)])
def test_exact_coefficients_keep_the_theorem(p, q, w):
    L = _sl("[0, 1]", p, q, w)
    for key in ("self-adjoint", "unbounded", "closed graph"):
        assert L.claims.status(key) == (Truth.TRUE, THEOREM), (key, L.claims)
    assert L.dom.claims.status("dense") == (Truth.TRUE, THEOREM)


def test_callable_coefficients_give_checked_claims():
    L = _sl("[0, 1]", lambda x: 1.0 + x, lambda x: np.cos(x), 1.0)
    for key in ("self-adjoint", "unbounded", "closed graph", "densely defined"):
        held, st = L.claims.status(key)
        assert held is Truth.TRUE and st == CHECKED, (key, L.claims)
    assert "sampled" in L.claims.find("self-adjoint").reason
    assert L.dom.claims.status("dense")[1] == CHECKED
    assert all(c.status == CHECKED for c in L.claims if c.statement.startswith(("self-adjoint, bounded below",
                                                                                 "the n-th eigenfunction")))


def test_the_radial_laplacian_keeps_the_theorem():
    from pygeoinf0.operators.unbounded import SturmLiouville
    for L in (SturmLiouville.radial_laplacian(0.0, 1.0, 2), SturmLiouville.radial_laplacian(0.5, 1.0, 0)):
        assert L.claims.status("self-adjoint") == (Truth.TRUE, THEOREM), L.claims


# ------------------------------------------------------------------ item 7: Scale(S, 0)
def _outer_set():
    from pygeoinf0.convex.sets import ConvexSet
    from pygeoinf0.optimisers.dual.bound import Bound
    D2 = pg.Euclidean(2)

    class K(ConvexSet):                       # a set not known non-empty: its supports are OUTER Bounds
        def __init__(self):
            super().__init__(D2, "K", [])

        def support(self, v):
            return Bound(0.7, -math.inf, 0.7, "OUTER", claim=CHECKED)
    return D2, K()


def test_scale_by_zero_passes_a_claim_status():
    from pygeoinf0.convex.algebra import Scale
    D2, K = _outer_set()
    e = Scale(K, 0).support(D2([1.0, 0.0]))
    assert isinstance(e, Enclosure) and e.status == CHECKED, repr(e)
    assert e.lower == -math.inf and e.upper == 0.0


def test_minkowski_with_a_set_scaled_by_zero():
    from pygeoinf0.convex.algebra import Scale, minkowski
    D2, K = _outer_set()
    e = minkowski(Scale(K, 0), pg.NormBall(D2, 1.0)).support(D2([1.0, 0.0]))
    assert isinstance(e, Enclosure) and e.status == CHECKED, repr(e)
    assert e.lower == -math.inf and abs(e.upper - 1.0) < 1e-12


def test_the_claim_status_of_a_bound_is_not_its_outcome():
    from pygeoinf0.optimisers.dual.bound import Bound
    b = Bound(0.7, -math.inf, 0.7, "OUTER", claim=CHECKED)
    assert b.status == "OUTER" and b.claim_status == CHECKED and b.enclosure.status == CHECKED
    assert Enclosure(1.0, 0.5, 1.5, DERIVED).claim_status == DERIVED
