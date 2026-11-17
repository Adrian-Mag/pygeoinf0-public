"""The result of a dual solve: a Bound, sup over a feasible set in one direction with its certified bracket, the dual
certificate and the primal witness (moved from convex/dual.py by plan 070 P4; the statuses are those of
optimisers/dual/reduction.py's module docstring)."""
import math

from ...core.claims import CHECKED, UNKNOWN
from ...core.numbers import BRACKET, EXACT, OUTER, Enclosure

CERTIFIED = CHECKED      # the claim status the producers of convex/ state for a Bound's interval (Bound's docstring)
_HOW = {"EXACT": EXACT, "BRACKET": BRACKET, "OUTER": OUTER, "EMPTY": EXACT, "UNBOUNDED": OUTER}


def _how_of(outcome, value, lower, upper):
    """The Enclosure's how of a Bound's outcome: EXACT, BRACKET, OUTER as they are; EMPTY (-inf at both ends) EXACT;
    UNBOUNDED (+inf, the only informative end) OUTER; PRIMAL by its shape (OUTER when the value is its one finite
    end, else BRACKET)."""
    if outcome in _HOW:
        return _HOW[outcome]
    if (value == lower and not math.isfinite(upper)) or (value == upper and not math.isfinite(lower)):
        return OUTER
    return BRACKET


class Bound(Enclosure):
    """sup over a feasible set in one direction, with its bracket and certificate (module docstring): an Enclosure
    (core/numbers.py: value, lower, upper, how) plus the dual certificate and the primal witness (plan 067 P1).

    status is the Bound's outcome, as before: EXACT, BRACKET, OUTER, PRIMAL, UNBOUNDED or EMPTY; how is the Enclosure
    axis (EXACT, BRACKET, OUTER) it maps to. The ends are certified in their directions (plan 067 P3, review A7): the
    lower end is the value at a witness checked feasible (to TOL, convex/feasible.py solve; the cell witnesses of
    optimisers/dual/nonsmooth.py and slabs.py checked by the noise set's violation and the prior's membership), or -inf; the
    upper end is the dual function phi at a certificate (weak duality: an upper bound for every certificate, evaluated
    with the upper ends of the sets' supports), or +inf; each to the rounding of its evaluation (the ends of an EXACT
    bound may cross by it). The claim status of the interval, b.claim_status (= b.enclosure.status; plan 071 P2), is
    stated by the producer (claim=): the producers of convex/ state CERTIFIED (checked: a numerical witness and a numerical evaluation of phi, not a
    proof); a Bound built without claim= says unknown. The ends are kept as computed (no widening to the value). error: E, the change of the bound that its stated input errors can make
    (optimisers/dual/exactness.py), where the rule judged it, else None; a BRACKET it made is [lower, upper] widened by WIDEN
    x E. Arithmetic with Bounds (a + b, scaled, maximum) gives plain Enclosures."""

    __slots__ = ("_outcome", "certificate", "witness", "error", "cells")

    def __new__(cls, value, lower=-math.inf, upper=math.inf, status="EXACT", certificate=None, witness=None,
                note="", error=None, claim=UNKNOWN):
        out = float.__new__(cls, value)
        out.lower, out.upper, out._outcome, out._status, out.reason = float(lower), float(upper), status, claim, note
        out.how = _how_of(status, float(value), out.lower, out.upper)
        out.certificate, out.witness, out.error = certificate, witness, error
        out.cells = None                                    # a box route's CellProblem (its witness's cells)
        return out

    @property
    def status(self):
        """The outcome: EXACT, BRACKET, OUTER, PRIMAL, UNBOUNDED or EMPTY (module docstring)."""
        return self._outcome

    @property
    def note(self):
        return self.reason

    @note.setter
    def note(self, text):
        self.reason = text

    @property
    def enclosure(self):
        """The Enclosure of this bound: value, [lower, upper], how and the claim status of the interval."""
        return Enclosure(self.value, self.lower, self.upper, self._status, self.reason, how=self.how)

    def negated(self):
        """This bound of the direction -q read as a bound of the infimum along q: -value with [-upper, -lower]."""
        return Bound(-self.value, -self.upper, -self.lower, self.status, self.certificate, self.witness, self.note,
                     self.error, claim=self._status)

    def __reduce__(self):
        return (_rebuild, (type(self), self.value, self.lower, self.upper, self._outcome, self._status),
                (None, {"certificate": self.certificate, "witness": self.witness, "note": self.note,
                        "error": self.error, "cells": self.cells}))

    def __repr__(self):
        return f"{self.value:.10g} [{self.status}: {self.lower:.10g} <= σ <= {self.upper:.10g}]" + \
            (f" {self.note}" if self.note else "")


def _rebuild(cls, value, lower, upper, outcome, claim):
    out = Bound.__new__(cls, value, lower, upper, outcome, claim=claim)
    if cls is LazyBound:
        out._make, out._cert = None, None
    return out


class LazyBound(Bound):
    """A Bound whose certificate (a vector of the data space) is formed when first read: a map of many directions
    reads values, not certificates."""

    __slots__ = ("_cert", "_make")

    def __new__(cls, value, lower, upper, status, make, witness=None, note="", error=None, claim=UNKNOWN):
        out = Bound.__new__(cls, value, lower, upper, status, None, witness, note, error, claim)
        out._make, out._cert = make, None
        return out

    @property
    def certificate(self):
        if self._cert is None and self._make is not None:
            self._cert, self._make = self._make(), None
        return self._cert

    @certificate.setter
    def certificate(self, value):
        self._cert = value

    def negated(self):
        out = LazyBound(-self.value, -self.upper, -self.lower, self.status, None, self.witness, self.note, self.error,
                        self._status)
        out._cert, out._make = self._cert, self._make
        return out


