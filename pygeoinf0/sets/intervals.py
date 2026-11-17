"""Intervals of the real line (plan 067 P2: from core/sets.py; Regions of R^1, sets/regions.py).

Membership is decided by the endpoints (closed or open), with the point convention of every Region: numbers, or
arrays of shape (..., 1). Endpoints matter for membership (and for spaces such as C(I)); they do not matter for
Lebesgue measure, so L2 on [0, 1] and on (0, 1) is one and the same space. I & J is an Interval, a FiniteSet (one
point: closed intervals that touch) or EmptySet(R), exactly.
"""
import math
import re

import numpy as np

from .base import EmptySet
from .finite import FiniteSet, _fmt
from .regions import Chart, R, Region

_PAT = re.compile(r"^\s*([\[\(])\s*([^,]+?)\s*,\s*([^\]\)]+?)\s*([\]\)])\s*$")
_CONST = {"inf": math.inf, "+inf": math.inf, "-inf": -math.inf, "∞": math.inf, "-∞": -math.inf,
          "pi": math.pi, "π": math.pi, "-pi": -math.pi, "-π": -math.pi, "e": math.e}


def _number(s):
    s = s.strip().replace(" ", "")
    if s in _CONST:
        return _CONST[s]
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"([+-]?[0-9.]*)\*?(pi|π)(?:/([0-9.]+))?", s)
    if m:
        c = m.group(1)
        c = float(c + "1") if c in ("", "+", "-") else float(c)
        return c * math.pi / (float(m.group(3)) if m.group(3) else 1.0)
    raise ValueError(f"cannot read the endpoint {s!r}; write a number, inf, pi, 2pi or pi/2")


class Interval(Region):
    """An interval of the real line, written as in mathematics.

    Interval("[0, 1)")             a string, with [ ] for closed and ( ) for open ends
    Interval(0, 1)                 [0, 1]
    Interval(0, 1, closed="left")  [0, 1); closed is "both", "left", "right" or "neither"

    An infinite endpoint is always open. The interval needs a < b (a single point is a FiniteSet).
    """

    __slots__ = ("a", "b", "left_closed", "right_closed")

    def __init__(self, a, b=None, closed="both"):
        if isinstance(a, str):
            if b is not None:
                raise TypeError("give either a string such as '[0, 1)' or two endpoints")
            m = _PAT.match(a)
            if not m:
                raise ValueError(f"cannot read {a!r} as an interval; write for example '[0, 1)'")
            lb, lo, hi, rb = m.groups()
            a, b = _number(lo), _number(hi)
            lc, rc = lb == "[", rb == "]"
            if (lc and math.isinf(a)) or (rc and math.isinf(b)):
                raise ValueError("an infinite endpoint cannot belong to the interval; write '(' or ')' there")
        else:
            if b is None:
                raise TypeError("give both endpoints, or a string such as '[0, 1]'")
            if closed not in ("both", "left", "right", "neither"):
                raise ValueError("closed must be 'both', 'left', 'right' or 'neither'")
            a, b = float(a), float(b)
            lc = closed in ("both", "left") and not math.isinf(a)
            rc = closed in ("both", "right") and not math.isinf(b)
        if math.isnan(a) or math.isnan(b) or not a < b:
            raise ValueError(f"an interval needs a < b (got a={a}, b={b}); a single point is a FiniteSet")
        object.__setattr__(self, "a", float(a))
        object.__setattr__(self, "b", float(b))
        object.__setattr__(self, "left_closed", lc)
        object.__setattr__(self, "right_closed", rc)

    def __setattr__(self, *_):
        raise AttributeError("intervals are immutable")

    dim = cartesian_dim = 1
    ambient_dim = None
    boundary_parts = ("left", "right")      # the named parts of the boundary (spaces/constructions.py)

    @property
    def bounding_box(self):
        return [(self.a, self.b)]

    def to_cartesian(self, x):
        return (np.asarray(x, float),)

    def charts(self):
        """One chart, the identity on [a, b] with Lebesgue measure (pygeoinf0.quadrature.rule: Gauss-Legendre)."""
        return (Chart(((self.a, self.b),), (False,), None, self.to_cartesian, None, None),)

    @classmethod
    def closed_interval(cls, a, b):
        return cls(a, b, "both")

    @classmethod
    def open_interval(cls, a, b):
        return cls(a, b, "neither")

    @property
    def length(self):
        return self.b - self.a

    @property
    def volume(self):
        """Its Hausdorff measure H^1 (the Region protocol): the length."""
        return self.b - self.a

    @property
    def ambient(self):
        return R

    @property
    def is_bounded(self):
        return math.isfinite(self.a) and math.isfinite(self.b)

    @property
    def is_closed(self):
        return (self.left_closed or math.isinf(self.a)) and (self.right_closed or math.isinf(self.b))

    @property
    def is_open(self):
        return not self.left_closed and not self.right_closed

    @property
    def closure(self):
        return Interval(self.a, self.b, "both")

    @property
    def interior(self):
        return Interval(self.a, self.b, "neither")

    @property
    def boundary(self):
        return FiniteSet([p for p in (self.a, self.b) if math.isfinite(p)])

    @property
    def midpoint(self):
        return 0.5 * (self.a + self.b)

    def _inside(self, x):
        lo = x >= self.a if self.left_closed else x > self.a
        hi = x <= self.b if self.right_closed else x < self.b
        return lo & hi

    def same_up_to_null_set(self, other):
        """True when the two intervals differ at most in their endpoints (a Lebesgue-null set)."""
        return isinstance(other, Interval) and self.a == other.a and self.b == other.b

    def __mul__(self, other):
        """The Cartesian product I * J (a rectangle), or I * J * K (a box)."""
        from ..domains import Box
        if isinstance(other, (Interval, str)):
            return Box(self, other)
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, Interval):
            return meet_intervals(self, other)
        return Region.__and__(self, other)

    def __eq__(self, other):
        return (isinstance(other, Interval) and self.a == other.a and self.b == other.b
                and self.left_closed == other.left_closed and self.right_closed == other.right_closed)

    def __hash__(self):
        return hash((self.a, self.b, self.left_closed, self.right_closed))

    def __repr__(self):
        return f"{'[' if self.left_closed else '('}{_fmt(self.a)}, {_fmt(self.b)}{']' if self.right_closed else ')'}"


def meet_intervals(I, J):
    """I & J exactly: an Interval, a FiniteSet (closed ends that touch) or EmptySet(R)."""
    a, b = max(I.a, J.a), min(I.b, J.b)
    lc = (I.left_closed if I.a == a else True) and (J.left_closed if J.a == a else True)
    rc = (I.right_closed if I.b == b else True) and (J.right_closed if J.b == b else True)
    if a > b or (a == b and not (lc and rc)):
        return EmptySet(R, why=f"{I!r} and {J!r} are disjoint")
    if a == b:
        return FiniteSet([a])
    return Interval(a, b, {(True, True): "both", (True, False): "left",
                           (False, True): "right", (False, False): "neither"}[(lc, rc)])
