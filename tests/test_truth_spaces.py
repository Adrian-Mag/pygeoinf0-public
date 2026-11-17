"""Plan 065 P2 (architecture review A1, A2): spaces that accepted non-elements.

A1  H^s took a discontinuous function and reported its L2 norm. Now the regularity rule of functions/membership is
    consulted by every element() (a function that carries a Regularity), and the realisation of H^s refuses a fit that
    jumps at a break for s >= 1/2 (a jump of the derivative for s >= 3/2, of the m-th derivative for s >= m + 1/2). A
    plain callable without a jump stays under an asserted claim: a fit cannot prove membership.
A2  L^p (p != 2) and C have no Riesz map and C has no inner product: no such methods (plan 068 P1; NotDefined before),
    distinct from Unsupported (not built).
Each test failed on main a45553c (the plan's rule: a failing test first).
"""
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.core.claims import ASSERTED, THEOREM
from pygeoinf0.core.errors import Refused, Unsupported

I = pg.Interval("[0, 1]")


def _errors():
    from pygeoinf0.core import errors
    return errors


def H(s=1):
    return pg.Sobolev(I, s, length_scale=0.1)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


# ------------------------------------------------------------------------------------------ A1
def test_a1_bank_function_with_a_jump_is_refused_by_the_space_itself():
    """f.element(H) refused it and H(f) accepted it with norm 0.7071 (its L2 norm): one rule for both entrances."""
    f = F.on(I).indicator(centre=0.5, radius=0.25)
    with pytest.raises(Refused, match="H\\^s for s < 0.5"):
        f.element(H())
    with pytest.raises(Refused, match="H\\^s for s < 0.5"):
        H()(f)
    _quiet(lambda: pg.L2(I)(f))                  # in L2 it is an element


def test_a1_bank_function_in_the_space_carries_its_membership_theorem():
    f = F.on(I).bump(centre=0.5, radius=0.2)
    for u in (_quiet(lambda: H()(f)), _quiet(lambda: f.element(H()))):
        c = u.claims[0]
        assert c.status == THEOREM and "is in H^1" in c.statement


def test_a1_callable_with_a_jump_at_a_given_break_is_refused():
    """H(1_{x > 1/2}, breaks=[0.5]) gave the norm 0.7746 (the form of each piece, the jump ignored)."""
    X = H()
    u = X(lambda x: (np.asarray(x) > 0.5).astype(float), breaks=[0.5], resolution=0.01)
    with pytest.raises(_errors().NotAnElement, match="jumps") as e:
        X.norm(u)
    assert e.value.needs                         # the choices are named
    with pytest.raises(_errors().NotAnElement):
        u.claims                                 # no claim "is in H^1" for it


def test_a1_callable_with_a_jump_found_by_the_fit_is_refused():
    X = H()
    u = X(lambda x: np.where(np.asarray(x) > 0.3, 1.0, 0.0), resolution=0.01)
    with pytest.raises(_errors().NotAnElement):
        _quiet(lambda: X.norm(u))


def test_a1_a_kink_is_in_h1_but_not_in_h2():
    kink = lambda x: np.abs(np.asarray(x) - 0.5)     # noqa: E731
    X1 = H(1)
    u = X1(kink, breaks=[0.5], resolution=0.01)
    assert np.isfinite(X1.norm(u))                  # a kink is in H^1 (s < 3/2)
    X2 = H(2)
    v = X2(kink, breaks=[0.5], resolution=0.01)
    with pytest.raises(_errors().NotAnElement, match="derivative"):
        X2.norm(v)


def test_a1_a_jump_of_the_second_derivative_is_refused_in_h3_only():
    f = lambda x: np.abs(np.asarray(x) - 0.4) * (np.asarray(x) - 0.4)     # noqa: E731  C^1, f'' jumps by 4
    _quiet(lambda: H(2)(f, breaks=[0.4], resolution=0.01).realisation)
    with pytest.raises(_errors().NotAnElement, match="order 2"):
        _quiet(lambda: H(3)(f, breaks=[0.4], resolution=0.01).realisation)


def test_a1_a_refitted_representer_is_accepted():
    """A combination of representers evaluated and fitted again (as operators/pieces_sobolev does) is in H^s."""
    for s in (1.5, 2, 3):
        X = H(s)
        k = X.riesz([0.3, 0.71])
        u = X(lambda x: k[0](np.asarray(x)) - 2 * k[1](np.asarray(x)), resolution=0.01)
        assert u.realisation is not None


def test_a1_smooth_callable_with_breaks_is_accepted_in_h2():
    X2 = H(2)
    u = X2(np.sin, breaks=[0.3, 0.7], resolution=0.01)
    assert np.isfinite(X2.norm(u))


def test_a1_a_plain_callable_stays_asserted():
    """The claim was CHECKED "its realisation converged with finite norm": a fit cannot prove membership in H^s."""
    u = H()(np.sin, resolution=0.01)
    c = u.claims[0]
    assert c.status == ASSERTED and "is in H^1" in c.statement


def test_a1_low_order_spaces_keep_jumps():
    """s < 1/2: piecewise smooth functions with jumps are elements; nothing to refuse."""
    X = pg.Sobolev(I, 0.4, length_scale=0.1)
    u = X(lambda x: (np.asarray(x) > 0.5).astype(float), breaks=[0.5], resolution=0.01)
    assert u.realisation is not None


# ------------------------------------------------------------------------------------------ A2
def test_a2_notdefined_is_a_refusal_not_unsupported():
    E = _errors()
    assert issubclass(E.NotDefined, Refused) and not issubclass(E.NotDefined, Unsupported)


def test_a2_lp_has_no_riesz_map():
    """L^3.riesz(I) returned an "element of L^3" (and |x|^-1/2 dx a finite norm 437 for a non-element). Plan 065 made it
    refuse (NotDefined); plan 068 P1 took the method out of the class: L^p for p != 2 has no Riesz map to call."""
    X = pg.Lp(I, 3)
    assert not hasattr(X, "riesz") and not hasattr(pg.Lp(I, 1), "riesz")
    assert "L^1.5" in str(X.dual().claims)            # its dual is named instead
    assert pg.L2(I).riesz(I) is not None             # p = 2: the Hilbert space keeps its Riesz map


def test_a2_c_has_no_inner_product_and_no_riesz_map():
    X = pg.Continuous("[0, 1]")
    assert not hasattr(X, "inner") and not hasattr(X, "riesz")
