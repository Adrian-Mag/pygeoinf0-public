"""Subsets of a space (plan 067 P3; architecture review 2026-11-13, B1 and the author's amendment 1): the sets of
knowledge about an element of a space, under the one Set root.

    Subset(space, name, claims)     a subset of the space: member(x) -> Truth, `x in S`, &, |; S.ambient is the space
    PredicateSubset(space, test)    {x : test(x)}, known by membership only (test returns a bool or a Truth)
    SubsetIntersection, SubsetUnion S & T and S | T of subsets of one space, known by membership (Kleene's & and |)

A subset is a state of kind set (a prior set before the datum, a feasible set after it). What is computable about the
convex ones (support functions, the convex set algebra, feasible sets) lives in convex/: ConvexSet(Subset) is that
subfamily, not a kind of state. The empty subset of a space is convex.EmptySet(space), the root's EmptySet with a
support function; S & EmptySet is the EmptySet and S | EmptySet is S, as for every Set. & and | of subsets of different
spaces are refused (SpaceMismatch).
"""
from ..core.claims import Claims
from ..core.errors import Unsupported
from ..core.logic import Truth
from ..core.kinds import SET
from .base import EmptySet, Intersection, Set, Union, common_ambient


class Subset(Set):
    """A subset of a space, used as a state (kind set). Subclasses give member; ConvexSet (convex/) adds support."""

    kind = SET                            # a state of kind set, declared by the class (plan 067 P4)
    convex = False

    def __init__(self, space, name="S", claims=()):
        self.space, self.name = space, name
        self._claims = list(claims)

    @property
    def ambient(self):
        return self.space

    @property
    def claims(self):
        return Claims(self._claims)

    def member(self, x):
        return Truth.UNKNOWN

    def support(self, v):
        raise Unsupported(f"{self.name} has no support function here (a set known by membership only; the convex sets "
                          "with a support function are pygeoinf0.convex.ConvexSet)")

    def is_empty(self):
        return Truth.UNKNOWN

    def _meet(self, other):
        return SubsetIntersection([self, other])

    def _join(self, other):
        return SubsetUnion([self, other])

    def __and__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        common_ambient(self, other)
        if isinstance(other, EmptySet):
            return other
        if other is self:
            return self
        return self._meet(other)

    def __or__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        common_ambient(self, other)
        if isinstance(other, EmptySet) or other is self:
            return self
        return self._join(other)

    def __repr__(self):
        return f"{self.name} ⊆ {self.space!r}"


class PredicateSubset(Subset):
    """{x : test(x)}: membership only (test returns a bool or a Truth)."""

    def __init__(self, space, test, name="S"):
        super().__init__(space, name)
        self.test = test

    def member(self, x):
        return Truth.of(self.test(x))


class SubsetIntersection(Subset, Intersection):
    """S_1 ∩ ... ∩ S_k of subsets of one space, known by membership (the root's Intersection with the space)."""

    def __init__(self, sets, name=None):
        sets = list(sets)
        Intersection.__init__(self, sets, sets[0].space)
        Subset.__init__(self, sets[0].space, name or " ∩ ".join(S.name for S in self.parts))

    member = Intersection.member

    def __repr__(self):
        return Intersection.__repr__(self)


class SubsetUnion(Subset, Union):
    """S_1 ∪ ... ∪ S_k of subsets of one space, known by membership (the root's Union with the space)."""

    def __init__(self, sets, name=None):
        sets = list(sets)
        Union.__init__(self, sets, sets[0].space)
        Subset.__init__(self, sets[0].space, name or " ∪ ".join(S.name for S in self.parts))

    member = Union.member

    def is_empty(self):
        out = Truth.TRUE
        for S in self.parts:
            out = out & S.is_empty()
        return out

    def __repr__(self):
        return Union.__repr__(self)
