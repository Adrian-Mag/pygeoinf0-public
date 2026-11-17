"""Plan 067 P1, logic and enclosures (architecture review 2026-11-13, B8 and the Enclosure part of A7).

- Kleene's logic (Truth, decide) lives in core/logic.py; core/sets.py no longer holds it (a clean break, no alias).
- Three-valued answers are Truth: Claims.status, relations.algebra.status, the boundedness Decision, atoms_inside,
  inside_closure and BallPiece.inside (UNKNOWN where they returned None, or True for "not decided here").
- Enclosure (core/numbers.py): a number with an interval that contains the exact one, with two axes: how (EXACT,
  BRACKET, OUTER) and the claim status; numbers.Bracket is gone (Enclosure, same arguments). Its arithmetic (sum,
  positive scaling, max, negation) keeps the direction of every end, rounding included; optimisers.dual.Bound is an
  Enclosure plus certificate and witness, with its public behaviour kept.
- f.equals(g) is a Truth beside the tolerance == (unchanged).
"""
import math
import pickle
from fractions import Fraction

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core import claims as claims_mod
from pygeoinf0 import sets as sets_mod                  # plan 067 P2: core/sets.py moved to sets/
from pygeoinf0.core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN, Claim, Claims


def logic():
    from pygeoinf0.core import logic as mod
    return mod


# ------------------------------------------------------------------ core/logic.py
def test_truth_and_decide_live_in_core_logic():
    L = logic()
    assert pg.Truth is L.Truth
    assert L.decide(0.0, 1.0) is L.Truth.TRUE and L.decide(1.0, 1.0) is L.Truth.FALSE
    assert not hasattr(sets_mod, "Truth") and not hasattr(sets_mod, "decide")       # a clean break: no alias


def test_truth_semantics_are_kept():
    T = logic().Truth
    with pytest.raises(TypeError):
        bool(T.UNKNOWN)
    assert bool(T.TRUE) is True and bool(T.FALSE) is False
    assert (T.FALSE & T.UNKNOWN) is T.FALSE and (T.TRUE | T.UNKNOWN) is T.TRUE and ~T.UNKNOWN is T.UNKNOWN


# ------------------------------------------------------------------ three-valued answers are Truth
def test_claims_status_is_truth_valued():
    T = logic().Truth
    cl = Claims([Claim("closed", DERIVED, "r", key="closed"), Claim("not bounded", THEOREM, "r", key="bounded",
                                                                     holds=False)])
    held, st = cl.status("closed")
    assert held is T.TRUE and st == DERIVED
    held, st = cl.status("bounded")
    assert held is T.FALSE and st == THEOREM
    held, st = cl.status("convex")
    assert held is T.UNKNOWN and st == UNKNOWN                                       # was (None, "unknown")


def test_relation_status_is_truth_valued():
    T = logic().Truth
    from pygeoinf0.relations.algebra import status
    A = pg.Matrix(np.eye(2), pg.Euclidean(2), pg.Euclidean(2))
    assert status(A, "linear")[0] is T.TRUE and status(A, "bounded")[0] is T.TRUE
    assert status(object(), "linear") == (T.UNKNOWN, UNKNOWN)
    assert status(object(), "closed") == (T.UNKNOWN, UNKNOWN)


def test_the_boundedness_decision_is_truth_valued():
    T = logic().Truth
    from pygeoinf0.spaces.dual import bounded, require_bounded
    D = pg.Interval(0, 1)
    point = pg.measures.as_measure(0.5)
    assert bounded(point, pg.Sobolev(D, 1.0, length_scale=0.2)).bounded is T.TRUE
    assert bounded(point, pg.Sobolev(D, 0.25, length_scale=0.2)).bounded is T.FALSE

    class NoRule:                                     # a space no rule of the decision covers
        domain = D
    d = bounded(point, NoRule())
    assert d.bounded is T.UNKNOWN and "not decided" in d.reason                      # was Decision(True, ...)
    require_bounded(point, NoRule())                  # not refused: the decision is not negative (kept)


def test_containment_of_atoms_and_supports_is_truth_valued():
    T = logic().Truth
    from pygeoinf0.domains import inside_closure
    from pygeoinf0.realisations.rkhs.atoms import atoms_inside
    from pygeoinf0.realisations.rkhs.atoms import atoms_from
    A = atoms_from([pg.measures.as_measure(0.5)], pg.Interval(0, 1), lambda X: np.asarray(X, float))
    assert atoms_inside(A, pg.Interval(0, 0.6)) is T.TRUE and atoms_inside(A, pg.Interval(0.6, 1)) is T.FALSE
    assert atoms_inside(A, pg.Simplex((0, 0), (1, 0), (0, 1))) is T.UNKNOWN          # was None
    B = pg.Ball(1.0)
    assert inside_closure(pg.Sphere(0.5), B) is T.TRUE and inside_closure(pg.Sphere(0.5, (0.8, 0, 0)), B) is T.FALSE
    assert inside_closure(pg.Curve(lambda t: (t, t * t, 0 * t), 0, 1), B) is T.UNKNOWN  # was None


def test_ball_piece_inside_is_truth_valued():
    T = logic().Truth
    from pygeoinf0.domains.pieces import BallPiece, meet
    shell = pg.Shell(0.5, 1.0)
    J = meet(pg.Ball(0.15, (0.0, 0.0, 0.55)), shell)
    assert isinstance(J, BallPiece) and J.inside(shell) is T.TRUE                  # was True


# ------------------------------------------------------------------ Enclosure
def numbers():
    from pygeoinf0.core import numbers as mod
    return mod


def test_bracket_is_now_enclosure():
    N = numbers()
    assert not hasattr(N, "Bracket")                                                    # a clean break: no alias
    e = N.Enclosure(1.0, 0.5, 2.0, CHECKED, "r")
    assert float(e) == 1.0 and (e.lower, e.upper, e.status, e.reason) == (0.5, 2.0, CHECKED, "r")
    assert e.how == N.BRACKET and N.Enclosure(2.0, 2.0, 2.0).how == N.EXACT
    assert N.Enclosure(3.0, -math.inf, 3.0, how=N.OUTER).how == N.OUTER
    assert pickle.loads(pickle.dumps(e)).status == CHECKED


def _exact(x):
    return Fraction(x)


def test_sum_keeps_the_direction_of_its_ends_through_rounding():
    N = numbers()
    a = N.Enclosure(0.1, 0.1, 0.1, DERIVED)
    b = N.Enclosure(0.2, 0.2, 0.2, DERIVED)
    s = a + b                                         # 0.1 + 0.2 rounds up to 0.30000000000000004 in double
    assert isinstance(s, N.Enclosure)
    assert _exact(s.lower) <= _exact(0.1) + _exact(0.2) <= _exact(s.upper)
    assert s.lower < 0.1 + 0.2                        # rounded down: the nearest sum is above the exact one
    assert s.status == DERIVED and s.how == N.EXACT
    t = sum([a, b, 1.0])
    assert isinstance(t, N.Enclosure) and _exact(t.lower) <= _exact(0.1) + _exact(0.2) + 1 <= _exact(t.upper)


def test_positive_scaling_max_and_negation_keep_directions():
    N = numbers()
    e = N.Enclosure(1.0, 0.5, 2.0, CHECKED)
    s = e.scaled(0.1)
    assert _exact(s.lower) <= _exact(0.1) * _exact(0.5) and _exact(0.1) * _exact(2.0) <= _exact(s.upper)
    m = N.maximum(e, N.Enclosure(1.5, 1.2, 1.7, THEOREM))
    assert (float(m), m.lower, m.upper, m.status) == (1.5, 1.2, 2.0, CHECKED)
    n = e.negated()
    assert (float(n), n.lower, n.upper, n.status) == (-1.0, -2.0, -0.5, CHECKED)
    o = N.Enclosure(3.0, -math.inf, 3.0, how=N.OUTER)
    assert o.negated().how == N.OUTER and (o + o).how == N.OUTER and (o + N.Enclosure(1, 0, 2)).how == N.BRACKET


def test_an_undefined_sum_is_refused():
    N = numbers()
    with pytest.raises(pg.Refused):
        N.Enclosure(-math.inf, -math.inf, -math.inf) + N.Enclosure(math.inf, -math.inf, math.inf, how=N.OUTER)


def test_exact_values_lie_in_their_enclosures():
    """Known cases: a chi-square cdf in closed form (mpmath at 50 digits), a finite-rank trace, and the support of a
    norm ball, alone and summed (sigma_{B1 + B2} = sigma_B1 + sigma_B2)."""
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50
    Q = pg.WeightedChiSquare([2.0], [3])
    F = Q.cdf(4.0)
    exact = mpmath.gammainc(1.5, 0, mpmath.mpf(4.0) / 4, regularized=True)        # P(2 chi^2_3 <= 4)
    assert mpmath.mpf(F.lower) <= exact <= mpmath.mpf(F.upper)
    from pygeoinf0.probability.covariance import Covariance
    X = pg.Euclidean(3)
    tr = Covariance(pg.FiniteRank(X.family([X(np.array([1.0, 0, 0])), X(np.array([0, 2.0, 0]))]), [0.1, 0.3])).trace()
    exact = Fraction(0.1) + 4 * Fraction(0.3)                                        # sum lam_j ||psi_j||^2
    assert Fraction(tr.lower) <= exact <= Fraction(tr.upper) and isinstance(tr, numbers().Enclosure)


def test_a_bound_is_an_enclosure_with_its_public_behaviour_kept():
    N = numbers()
    from pygeoinf0.optimisers.dual import Bound, LazyBound
    b = Bound(1.0, 0.5, 2.0, "BRACKET", certificate=np.array([1.0]), witness="w", note="n")
    assert isinstance(b, N.Enclosure)
    assert (b.value, b.lower, b.upper, b.status, b.note) == (1.0, 0.5, 2.0, "BRACKET", "n")
    assert b.how == N.BRACKET and b.enclosure.status == UNKNOWN                     # no claim stated: unknown
    assert float(b) == 1.0 and b.witness == "w" and b.certificate[0] == 1.0
    n = b.negated()
    assert isinstance(n, Bound) and (n.value, n.lower, n.upper, n.status) == (-1.0, -2.0, -0.5, "BRACKET")
    for outcome, how in (("EXACT", N.EXACT), ("OUTER", N.OUTER), ("EMPTY", N.EXACT), ("UNBOUNDED", N.OUTER)):
        v = {"EMPTY": -math.inf, "UNBOUNDED": math.inf}.get(outcome, 1.0)
        lo = {"EMPTY": -math.inf, "UNBOUNDED": -math.inf, "OUTER": -math.inf}.get(outcome, 1.0)
        hi = {"EMPTY": -math.inf, "UNBOUNDED": math.inf}.get(outcome, 1.0)
        assert Bound(v, lo, hi, outcome).how == how, outcome
    lz = LazyBound(1.0, 1.0, 1.0, "EXACT", lambda: np.array([2.0]))
    assert lz.certificate[0] == 2.0 and lz.negated().certificate[0] == 2.0
    s = b + Bound(1.0, 1.0, 1.0, "EXACT", claim=DERIVED)
    assert type(s) is N.Enclosure and (s.lower, s.upper, s.status) == (1.5, 3.0, UNKNOWN)
    c = pickle.loads(pickle.dumps(b))
    assert (c.value, c.lower, c.upper, c.status, c.note, c.witness) == (1.0, 0.5, 2.0, "BRACKET", "n", "w")


def test_a_bound_from_bgp_keeps_its_outcome():
    """sup m_2 over {||m|| <= 1, |m_1 - 0.5| <= 0.1} is sqrt(1 - 0.4^2) (m_1 = 0.4)."""
    X = pg.Euclidean(2)
    G, T = pg.Matrix(np.array([[1.0, 0.0]]), X, pg.Euclidean(1)), pg.Matrix(np.array([[0.0, 1.0]]), X, pg.Euclidean(1))
    K = pg.update(pg.NormBall(X, 1.0), pg.Additive(G, pg.NormBall(G.codomain, 0.1)), G.codomain(np.array([0.5])))
    lo, hi = pg.push(K, T).bounds(0)
    for b in (lo, hi):
        assert isinstance(b, numbers().Enclosure) and b.status == "EXACT" and b.how == numbers().EXACT
    assert hi.lower - 1e-9 <= math.sqrt(1 - 0.4 ** 2) <= hi.upper + 1e-9
    assert -hi.value == pytest.approx(lo.value, abs=1e-9)


# ------------------------------------------------------------------ f.equals(g) -> Truth
def test_function_equals_is_truth_and_eq_is_unchanged():
    T = logic().Truth
    X = pg.L2(pg.Interval(0, 1))
    f = X(lambda t: np.sin(3 * t))
    g = X(lambda t: np.sin(3 * t) + 0.0)
    h = X(lambda t: np.cos(3 * t))
    assert f.equals(g) is T.TRUE and (f == g) is True
    assert f.equals(h) is T.FALSE and (f == h) is False
