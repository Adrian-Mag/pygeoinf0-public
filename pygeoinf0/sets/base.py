"""The root of the hierarchy (plan 067 P2; architecture review 2026-11-13, L1 and B1; decision Q3): a Set and its
membership, and the set algebra every set shares.

    S.member(x)       a Truth: TRUE, FALSE or UNKNOWN (core/logic.py), the one membership protocol
    x in S            bool(S.member(x)); raises MembershipUndecided (a TypeError) on UNKNOWN, never reads it as FALSE
    S & T, S | T      the intersection and the union: lazy sets known by membership (Kleene's & and |), unless a
                      subclass constructs them (Region: intervals, boxes, round sets)
    EmptySet(A)       the empty subset of the ambient set A: the identity of | and absorbing for &, with the same ambient
    S.ambient         the set S is a subset of (S itself unless a subclass says otherwise: R^d for a Region, the space
                      for a vector space); & and | of sets in different ambients are refused (SpaceMismatch)
    S.claims          what is known about the set, with its status

The sets of a space as states of knowledge (convex/: Subset, ConvexSet) and measures on sets (measures/) build
on this root in later phases (P3, P5).
"""
from ..core.claims import Claims
from ..core.errors import MembershipUndecided, SpaceMismatch
from ..core.logic import Truth


class Set:
    """A set. member(x) -> Truth decides membership where it can (UNKNOWN where it cannot); `x in S` is bool of it."""

    def member(self, x):
        return Truth.UNKNOWN

    def __contains__(self, x):
        t = Truth.of(self.member(x))
        if t is Truth.UNKNOWN:
            raise MembershipUndecided(f"whether {_short(x)} is in {self!r} is not decided here: S.member(x) gives the "
                                      "three-valued answer (Truth.UNKNOWN)")
        return t is Truth.TRUE

    @property
    def ambient(self):
        return self

    @property
    def claims(self):
        return Claims()

    def is_empty(self):
        return Truth.UNKNOWN

    def __and__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        A = common_ambient(self, other)
        if isinstance(other, EmptySet):
            return other
        if other is self:
            return self
        return Intersection([self, other], A)

    def __or__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        A = common_ambient(self, other)
        if isinstance(other, EmptySet) or other is self:
            return self
        return Union([self, other], A)


def _short(x):
    r = repr(x)
    return r if len(r) <= 60 else r[:57] + "..."


def common_ambient(S, T):
    """The ambient set of S and T, refused (SpaceMismatch) when they differ."""
    A, B = S.ambient, T.ambient
    if A is not B and A != B:
        raise SpaceMismatch(f"{S!r} is a subset of {A!r} and {T!r} of {B!r}: their intersection and union are not "
                            "formed across ambient sets")
    return A


class EmptySet(Set):
    """The empty subset of the ambient set A (a finding, not an error: an empty feasible set says the data and the
    prior disagree). member is FALSE for everything; S | EmptySet(A) is S and S & EmptySet(A) is EmptySet(A)."""

    def __init__(self, ambient, why="empty"):
        self._ambient, self.why = ambient, why

    @property
    def ambient(self):
        return self._ambient

    def member(self, x):
        return Truth.FALSE

    def contains(self, points):
        """Every point is outside: the answer of the ambient Region's point convention, all False."""
        import numpy as np
        return np.zeros(np.shape(self._ambient.contains(points)), bool) if hasattr(self._ambient, "contains") \
            else False

    def is_empty(self):
        return Truth.TRUE

    @property
    def volume(self):
        return 0.0                              # the measure of the empty set, in every dimension

    def __and__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        common_ambient(self, other)
        return self

    __rand__ = __and__

    def __or__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        common_ambient(self, other)
        return other

    __ror__ = __or__

    def __eq__(self, other):
        return isinstance(other, EmptySet) and (other.ambient is self.ambient or other.ambient == self.ambient)

    def __hash__(self):
        return hash(("EmptySet", repr(self._ambient)))

    def __repr__(self):
        return f"∅ ⊆ {self._ambient!r}"


class Intersection(Set):
    """S_1 ∩ ... ∩ S_k known by membership: member is Kleene's & of the members (FALSE as soon as one says FALSE)."""

    def __init__(self, parts, ambient):
        self.parts = tuple(p for S in parts for p in (S.parts if type(S) is type(self) else (S,)))
        self._ambient = ambient

    @property
    def ambient(self):
        return self._ambient

    def member(self, x):
        t = Truth.TRUE
        for S in self.parts:
            t = t & S.member(x)
            if t is Truth.FALSE:
                break
        return t

    def __repr__(self):
        return " ∩ ".join(f"({S!r})" for S in self.parts)


class Union(Set):
    """S_1 ∪ ... ∪ S_k known by membership: member is Kleene's | of the members (TRUE as soon as one says TRUE)."""

    def __init__(self, parts, ambient):
        self.parts = tuple(p for S in parts for p in (S.parts if type(S) is type(self) else (S,)))
        self._ambient = ambient

    @property
    def ambient(self):
        return self._ambient

    def member(self, x):
        t = Truth.FALSE
        for S in self.parts:
            t = t | S.member(x)
            if t is Truth.TRUE:
                break
        return t

    def __repr__(self):
        return " ∪ ".join(f"({S!r})" for S in self.parts)
