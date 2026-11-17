"""The number type of Euclidean elements (plan 020 P3a): float for real spaces, complex for complex ones. Complex
values are never truncated to their real parts: a real space refuses them (numpy would drop imaginary parts with
only a warning).

Enclosures (plan 067 P1, unifying numbers.Bracket and the ends of optimisers.dual.Bound; architecture review 2026-11-13,
B8 and A7): a computed value with an interval that contains the exact number (plan 009 P3: traces of covariances,
weighted chi-square probabilities and quantiles, eigenvalues of Sturm-Liouville operators; support values of BGP).

    e = Enclosure(value, lower, upper, status, reason, how=None)    float(e) == value, lower <= exact <= upper
    e.width, e.error, e.contains(x), e.claim(what)
    a + b, sum([...]), e.scaled(c), maximum(a, b, ...), e.negated()

An Enclosure is a float (it can be used wherever its value can); the interval is what is claimed about the exact
number. It has two axes:
  status  how the interval is known, a claim status (core/claims.py): derived (a stated bound proved from how the
          number was computed, rounding included where it matters), checked (a numerical estimate of an error, not a
          proof), asserted (given by the user) or unknown;
  how     what the ends are: EXACT (they agree to the precision rule's tolerance: the value is the number), BRACKET
          (the number lies between ends that differ; the value is a computed approximation between them), OUTER
          (only one end is informative, the other infinite; the value is that end). By default how is EXACT when the
          ends agree to TOL (core/precision.py) and BRACKET otherwise.
An unbounded side is +-inf.

Arithmetic keeps the direction of every end, rounding included: the lower end of a result is a lower bound of the
exact result whenever each operand's lower end is a lower bound of its exact number, and the same for upper ends.
Sums and products are formed exactly (as rationals) and rounded outward: the lower end down, the upper end up.
    a + b          [l_a + l_b, u_a + u_b]; a plain number is an exact operand
    e.scaled(c)    c >= 0: [c l, c u] (0 times an infinite end is 0, the measure-theoretic convention); c < 0: the
                   negation scaled by |c|
    maximum(...)   [max l, max u]: the max of numbers in the intervals lies there (no rounding)
    e.negated()    -value with [-u, -l]: the lower end of -x is minus the upper end of x (no rounding)
Other float operations (a - b, a * b, ...) give plain floats. A result's status is the rule's (claims.by_rule: the
weakest of derived and the operands'); its how is EXACT when every operand is EXACT, OUTER when every other operand
is OUTER with its value at the same end, else BRACKET. A sum of -inf and +inf ends is not defined: refused.
"""


import math
from fractions import Fraction

import numpy as np

from .claims import Claim, by_rule


def element_values(space, values):
    """The values of an element of a Euclidean space, in the space's number type (float, or complex for a complex
    space: plan 020 P3). Complex values with a nonzero imaginary part are refused by a real space, never truncated."""
    dtype = getattr(space, "dtype", float)
    v = np.asarray(values)
    if np.iscomplexobj(v) and dtype is float:
        if np.any(v.imag != 0):
            raise TypeError(f"complex values in the real space {space}: their imaginary parts would be lost "
                            "(use pg.Euclidean(n, field=\"C\"))")
        v = v.real
    return np.array(v, dtype=dtype)


def field_scalar(space, c):
    """A scalar multiplier: real, or complex for a complex space (refused, with the reason, for a real one)."""
    if np.iscomplexobj(c) and getattr(space, "dtype", float) is float:
        if np.imag(c) != 0:
            raise TypeError(f"a complex scalar times an element of the real space {space}")
        c = np.real(c)
    return complex(c) if getattr(space, "dtype", float) is complex else float(c)


# ---------------------------------------------------------------- enclosures (plan 067 P1; Bracket before)
EXACT, BRACKET, OUTER = "EXACT", "BRACKET", "OUTER"
HOWS = (EXACT, BRACKET, OUTER)


_BIG = Fraction(np.finfo(float).max)


def _down(r):
    """The largest float at most the rational r (-inf below the floats, the largest float above them)."""
    if abs(r) > _BIG:
        return -math.inf if r < 0 else float(np.finfo(float).max)
    f = float(r)
    return math.nextafter(f, -math.inf) if Fraction(f) > r else f


def _up(r):
    """The smallest float at least the rational r (+inf above the floats, minus the largest float below them)."""
    if abs(r) > _BIG:
        return math.inf if r > 0 else -float(np.finfo(float).max)
    f = float(r)
    return math.nextafter(f, math.inf) if Fraction(f) < r else f


def _how(value, lower, upper):
    from .precision import TOL
    if lower == upper or (math.isfinite(lower) and math.isfinite(upper)
                          and upper - lower <= TOL * max(1.0, abs(value) if math.isfinite(value) else 1.0)):
        return EXACT
    return BRACKET


def _end(e):
    """Which end the value sits at: "upper", "lower", "both" (EXACT) or None."""
    if e.how == EXACT:
        return "both"
    v = float(e)
    return "upper" if v == e.upper else "lower" if v == e.lower else None


def _combined_how(terms):
    hows = [t.how for t in terms]
    if all(h == EXACT for h in hows):
        return EXACT
    others = [t for t in terms if t.how != EXACT]
    if all(t.how == OUTER for t in others) and len({_end(t) for t in others}) == 1 and _end(others[0]) is not None:
        return OUTER
    return BRACKET


def _as_enclosure(x):
    if isinstance(x, Enclosure):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return None
    raise TypeError(f"{type(x).__name__} is not a number with an enclosure")


def _sum(a, b, inf):
    """a + b of two ends (floats), rounded outward towards inf; -inf + inf refused."""
    if math.isinf(a) or math.isinf(b):
        if math.isinf(a) and math.isinf(b) and a != b:
            from .errors import Refused
            raise Refused("the sum of an end at -inf and an end at +inf is not defined",
                          needs=["decide the sum of the sets first (an empty set's -inf absorbs everything)"])
        return a if math.isinf(a) else b
    r = Fraction(a) + Fraction(b)
    return _up(r) if inf > 0 else _down(r)


def _times(c, a, inf):
    """c * a (c > 0 a float, a an end), rounded outward towards inf."""
    if math.isinf(a):
        return a
    r = Fraction(c) * Fraction(a)
    return _up(r) if inf > 0 else _down(r)


class Enclosure(float):
    """A value with an interval containing the exact number, its claim status and how its ends were found (module
    docstring). Enclosure(value, lower=-inf, upper=inf, status="derived", reason="", how=None): the interval is widened
    to contain the value (a value outside its own interval would make it a false statement about the value)."""

    __slots__ = ("lower", "upper", "how", "_status", "reason")

    def __new__(cls, value, lower=-math.inf, upper=math.inf, status="derived", reason="", how=None):
        out = float.__new__(cls, value)
        out.lower, out.upper, out._status, out.reason = float(lower), float(upper), status, reason
        if not out.lower <= float(value) <= out.upper and not math.isnan(value):
            out.lower, out.upper = min(out.lower, float(value)), max(out.upper, float(value))
        if how is not None and how not in HOWS:
            raise ValueError(f"how is one of {HOWS}, not {how!r}")
        out.how = _how(float(out), out.lower, out.upper) if how is None else how
        return out

    @property
    def status(self):
        """The claim status of the interval (core/claims.py). A subclass may read status otherwise (optimisers.dual.Bound:
        its outcome); claim_status is the claim status for every Enclosure."""
        return self._status

    @property
    def claim_status(self):
        """The claim status of the interval for every Enclosure, a Bound included (whose status is its outcome; plan
        071 P2, review item 7): what a computation that combines enclosures passes on."""
        return self._status

    @property
    def value(self):
        return float(self)

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def error(self):
        """The largest distance from the value to an end of the interval."""
        return max(float(self) - self.lower, self.upper - float(self))

    def contains(self, x):
        return self.lower <= float(x) <= self.upper

    # -------------------------------------------------------- arithmetic that keeps the direction of every end
    def __add__(self, other):
        o = _as_enclosure(other)
        if o is None:                                   # a plain number: exact
            c = float(other)
            return Enclosure(float(self) + c, _sum(self.lower, c, -1), _sum(self.upper, c, 1), by_rule(self.claim_status),
                             self.reason, how=self.how)
        return Enclosure(float(self) + float(o), _sum(self.lower, o.lower, -1), _sum(self.upper, o.upper, 1),
                         by_rule(self.claim_status, o.claim_status), "the sum of enclosures (rounded outward)",
                         how=_combined_how([self, o]))

    __radd__ = __add__

    def scaled(self, c, reason=None):
        """c times the enclosure (c a number). c = 0 gives the enclosure 0, also for infinite ends: 0 * inf = 0, the
        measure-theoretic convention (0 * inf is NaN in floating point, which broke lower <= value <= upper). c < 0
        is the negation scaled by |c|: the lower end of c x is c times the upper end of x."""
        if not c:
            return Enclosure(0.0, 0.0, 0.0, by_rule(self.claim_status), reason or self.reason, how=EXACT)
        if c < 0:
            return self.negated().scaled(-c, reason)
        c = float(c)
        return Enclosure(c * float(self), _times(c, self.lower, -1), _times(c, self.upper, 1), by_rule(self.claim_status),
                         reason or self.reason, how=self.how)

    def negated(self):
        """-value with [-upper, -lower] (exact: no rounding); how is kept, the value's end flips."""
        return Enclosure(-float(self), -self.upper, -self.lower, self.claim_status, self.reason, how=self.how)

    def claim(self, what, key=None):
        return Claim(f"{what} in [{self.lower:.12g}, {self.upper:.12g}] (value {float(self):.12g})", self.claim_status,
                     self.reason, key=key)

    def __reduce__(self):
        return (Enclosure, (float(self), self.lower, self.upper, self._status, self.reason, self.how))

    def __repr__(self):
        return f"{float(self):.12g} [{self._status}: {self.lower:.12g}, {self.upper:.12g}]"


def maximum(*terms):
    """The max of enclosures and numbers: [max l, max u] encloses the max of numbers in the intervals (exact)."""
    given = [_as_enclosure(t) for t in terms]
    es = [Enclosure(float(t), float(t), float(t), how=EXACT) if e is None else e for t, e in zip(terms, given)]
    return Enclosure(max(float(e) for e in es), max(e.lower for e in es), max(e.upper for e in es),
                     by_rule(*[e.claim_status for e in given if e is not None]), "the max of enclosures",
                     how=_combined_how(es))
