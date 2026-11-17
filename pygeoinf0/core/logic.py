"""Kleene's three-valued logic: Truth and decide (plan 067 P1, from core/sets.py; architecture review 2026-11-13, L0).

Every three-valued answer of the package is a Truth: membership, single-valuedness, Claims.status, the boundedness
decision, containment of supports. UNKNOWN never reads as FALSE: bool(Truth.UNKNOWN) raises, so a caller that needs a
decided answer says what it does with UNKNOWN (`answer is Truth.TRUE`, `answer is not Truth.FALSE`). A Truth compared
with a bool by == or != raises TruthComparedWithBool (plan 071 P1): Truth.TRUE == True was silently False.
"""
import enum

import numpy as np

from .errors import TruthComparedWithBool
from .precision import DECIDE_TOL, UNDECIDED


class Truth(enum.Enum):
    """A three-valued answer (membership, single-valuedness): TRUE, FALSE or UNKNOWN. UNKNOWN never reads as FALSE:
    bool(Truth.UNKNOWN) raises. & and | follow Kleene's logic (FALSE & UNKNOWN is FALSE, TRUE | UNKNOWN is TRUE)."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @classmethod
    def of(cls, x):
        return x if isinstance(x, Truth) else (cls.TRUE if x else cls.FALSE)

    def __bool__(self):
        if self is Truth.UNKNOWN:
            raise TypeError("Truth.UNKNOWN is undecided: it cannot be read as True or False")
        return self is Truth.TRUE

    def __eq__(self, other):
        if isinstance(other, (bool, np.bool_)):
            raise _compared(self, other, "==")
        return self is other

    def __ne__(self, other):
        if isinstance(other, (bool, np.bool_)):
            raise _compared(self, other, "!=")
        return self is not other

    __hash__ = enum.Enum.__hash__

    def __and__(self, other):
        other = Truth.of(other)
        if Truth.FALSE in (self, other):
            return Truth.FALSE
        return Truth.TRUE if self is other is Truth.TRUE else Truth.UNKNOWN

    def __or__(self, other):
        other = Truth.of(other)
        if Truth.TRUE in (self, other):
            return Truth.TRUE
        return Truth.FALSE if self is other is Truth.FALSE else Truth.UNKNOWN

    def __invert__(self):
        return {Truth.TRUE: Truth.FALSE, Truth.FALSE: Truth.TRUE}.get(self, Truth.UNKNOWN)


def _compared(t, b, op):
    return TruthComparedWithBool(
        f"{t} {op} {bool(b)}: a Truth is not a bool, and the comparison would be silently "
        f"{'False' if op == '==' else 'True'}. Test `t is Truth.TRUE` (or `t is not Truth.FALSE`, UNKNOWN counted as "
        "possible), or read a Claim's .holds (a bool); Claims.status(key)[0] is a Truth")


def decide(residual, scale, tol=DECIDE_TOL, undecided=UNDECIDED):
    """TRUE when residual <= tol * scale, FALSE when residual > undecided * scale, UNKNOWN between (rounding level); the
    defaults are the precision rule's (core/precision.py)."""
    if residual <= tol * scale:
        return Truth.TRUE
    return Truth.FALSE if residual > undecided * scale else Truth.UNKNOWN
