"""Plan 071 P1, truth and tolerance (Mac review of restructure 9e3a38f7: Truth vs bool, items 2 and 9).

- A Truth compared with a bool (==, !=, either side, numpy bools too) raises TruthComparedWithBool (core/errors.py),
  whose message names the right tests (`t is Truth.TRUE`, Claim.holds). Truth == Truth is kept, and so is hashing.
  Before, Truth.TRUE == True was silently False, so (holds, status) == (True, "checked") was always False.
- Item 2: the support of a subspace is 0 only where v's component in the subspace computes to exactly 0; inside the
  rounding band it is the Enclosure [0, +inf] (undecided), never 0 EXACT; +inf above it. No norm underflow:
  Whole(R^3).support((1e-300, 0, 0)) is +inf.
- Item 9: f.equals(g) and NormBall.member answer UNKNOWN inside the tolerance (TRUE only at distance or excess 0), as
  Interval and Disk never say TRUE outside.
"""
import math
import pickle

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core import errors
from pygeoinf0.core.logic import Truth
from pygeoinf0.core.numbers import EXACT


# ------------------------------------------------------------------ Truth vs bool
@pytest.mark.parametrize("b", [True, False, np.True_, np.False_])
@pytest.mark.parametrize("t", list(Truth))
def test_truth_compared_with_a_bool_raises(t, b):
    for compare in (lambda: t == b, lambda: b == t, lambda: t != b, lambda: b != t):
        with pytest.raises(errors.TruthComparedWithBool) as info:
            compare()
        assert "is Truth.TRUE" in str(info.value) and "holds" in str(info.value)


def test_truth_compared_with_a_bool_inside_a_tuple_raises():
    status = (Truth.TRUE, "checked")
    with pytest.raises(errors.TruthComparedWithBool):
        status == (True, "checked")                     # noqa: B015 - was silently False
    claims = pg.Claims([pg.Claim("bounded", "checked", "why", key="bounded")])
    with pytest.raises(errors.TruthComparedWithBool):
        claims.status("bounded") == (True, "checked")   # noqa: B015
    assert claims.status("bounded") == (Truth.TRUE, "checked")


def test_truth_comparison_error_is_a_type_error_in_core_errors():
    assert issubclass(errors.TruthComparedWithBool, TypeError)
    assert "TruthComparedWithBool" in errors.__doc__


def test_truth_with_truth_and_hashing_are_kept():
    assert Truth.TRUE == Truth.TRUE and Truth.TRUE != Truth.FALSE and not (Truth.UNKNOWN != Truth.UNKNOWN)
    assert {Truth.TRUE: 1, Truth.FALSE: 0}[Truth.TRUE] == 1
    assert len({Truth.TRUE, Truth.TRUE, Truth.FALSE, Truth.UNKNOWN}) == 3
    assert hash(Truth.TRUE) == hash(Truth.TRUE)
    assert Truth.TRUE in (Truth.FALSE, Truth.TRUE) and Truth.UNKNOWN not in (Truth.FALSE, Truth.TRUE)
    assert pickle.loads(pickle.dumps(Truth.UNKNOWN)) is Truth.UNKNOWN
    assert (Truth.TRUE == "true") is False and (Truth.TRUE == 1.0) is False       # other types: unequal, not raised
    assert Truth.of(True) is Truth.TRUE and bool(Truth.TRUE) is True


# ------------------------------------------------------------------ item 2: supports of subspaces
def test_span_support_off_the_span_is_not_zero_exact():
    R3 = pg.Euclidean(3)
    S = pg.Span([R3(np.array([1.0, 0.0, 0.0]))])
    s = S.support(R3(np.array([1e-10, 1.0, 0.0])))      # true value +inf: <v, t e1> = 1e-10 t
    assert not (s.how == EXACT and float(s) == 0.0)
    assert s.upper == math.inf and s.lower <= 0.0 and s.how != EXACT
    s = S.support(R3(np.array([1e-13, 1.0, 0.0])))
    assert s.upper == math.inf and s.how != EXACT
    s = S.support(R3(np.array([1e-3, 1.0, 0.0])))
    assert float(s) == math.inf and s.lower == math.inf
    s = S.support(R3(np.array([0.0, 1.0, 0.0])))       # exactly in the annihilator
    assert float(s) == 0.0 and s.upper == 0.0


def test_whole_support_has_no_norm_underflow():
    R3 = pg.Euclidean(3)
    W = pg.Whole(R3)
    for x in (1e-300, 1e-320):
        s = W.support(R3(np.array([x, 0.0, 0.0])))
        assert float(s) == math.inf and s.lower == math.inf
    assert float(W.support(R3.zero)) == 0.0
    assert W.perp().member(R3(np.array([1e-300, 0.0, 0.0]))) is Truth.FALSE


def test_complement_support_off_its_annihilator():
    R3 = pg.Euclidean(3)
    C = pg.Span([R3(np.array([1.0, 0.0, 0.0]))]).perp()
    s = C.support(R3(np.array([1.0, 1e-10, 0.0])))      # component 1e-10 in C: +inf
    assert s.upper == math.inf and s.how != EXACT
    assert float(C.support(R3(np.array([2.0, 0.0, 0.0])))) == 0.0


def test_span_member_has_no_norm_underflow():
    R3 = pg.Euclidean(3)
    S = pg.Span([R3(np.array([1.0, 0.0, 0.0]))])
    assert S.member(R3(np.array([0.0, 1e-300, 0.0]))) is Truth.FALSE
    assert S.member(R3(np.array([1e-300, 0.0, 0.0]))) is Truth.TRUE


# ------------------------------------------------------------------ item 9: tolerance is not truth
def test_equals_is_unknown_inside_the_tolerance():
    M = pg.L2(pg.Interval(0, 1))
    assert M.constant(1).equals(M.constant(1 + 1e-13)) is Truth.UNKNOWN
    assert M.constant(1).equals(M.constant(1.0)) is Truth.TRUE
    assert M.constant(1).equals(M.constant(1.5)) is Truth.FALSE


def test_normball_member_is_unknown_inside_the_tolerance():
    R2 = pg.Euclidean(2)
    B = pg.NormBall(R2, 1.0)
    assert B.member(R2(np.array([1 + 1e-11, 0.0]))) is Truth.UNKNOWN
    assert B.member(R2(np.array([1 + 1e-13, 0.0]))) is Truth.UNKNOWN
    assert B.member(R2(np.array([1.0, 0.0]))) is Truth.TRUE
    assert B.member(R2(np.array([0.5, 0.5]))) is Truth.TRUE
    assert B.member(R2(np.array([1.1, 0.0]))) is Truth.FALSE
    assert pg.Interval(0, 1).member(1 + 1e-11) is not Truth.TRUE
