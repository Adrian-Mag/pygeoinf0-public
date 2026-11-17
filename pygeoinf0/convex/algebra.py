"""Lazy algebra of sets by support functions (LAND findings 40, section 5; DESIGN.md 2.2):

    image(T, S)        T(S) for a linear T: sigma_TS(q) = sigma_S(T* q); affine sets map to affine sets exactly
    minkowski(A, B)    A + B: sigma_{A+B} = sigma_A + sigma_B; a point gives a translate
    Translate(S, x)    x + S: sigma + <v, x>
    Scale(S, c)        c S (c >= 0): c sigma
    Union(sets)        sigma = max of the sigmas (exact, DESIGN 2.5 (f)); member: any
    Intersection(sets) member: all; its support needs a dual (pygeoinf0.convex.feasible builds the BGP case)
    CartesianProduct   a product of sets on the blocks of a stacked R^n or the summands of a direct sum

Supports are Enclosures (convex/sets.py ConvexSet; plan 067 P3, review A7), and the algebra composes their ends in
their directions (core/numbers.py: sums and positive scalings rounded outward, max exact): the lower end of a sum is
the sum of the lower ends, and so on. A part whose support is a plain number is refused (convex/sets.support_of).
Union and Intersection are the root's (sets.base, via sets/subsets.py) with a support function.
"""
import functools
import math

from ..core.claims import DERIVED, THEOREM, Claim
from ..core.errors import SpaceMismatch, Unsupported
from ..core.kinds import SET, stated_kind
from ..core.logic import Truth
from ..core.numbers import Enclosure, maximum
from ..sets.subsets import SubsetIntersection, SubsetUnion
from .sets import AffineSubset, ConvexSet, EmptySet, as_element, exact, support_of
from ..spaces.identity import require


def _claims_of(sets, why):
    """closed / convex / bounded of a combination: derived when every part has the property on record."""
    return [Claim(key, DERIVED, why, key=key) for key in ("convex", "bounded")
            if all(S.claims.status(key)[0] is Truth.TRUE for S in sets)]


class Image(ConvexSet):
    """T(S) for a linear operator T and a convex set S of its domain: sigma(q) = sigma_S(T* q)."""

    def __init__(self, T, S):
        self.T, self.base = T, S
        cl = _claims_of([S], f"the image of a convex set under the linear map {T.name}")
        if S.claims.status("weakly compact")[0] is Truth.TRUE:
            cl.append(Claim("closed", DERIVED, "the image of a weakly compact set under a bounded linear map", key="closed"))
        super().__init__(T.codomain, f"{T.name}({S.name})", cl)

    def support(self, q):
        return support_of(self.base, self.T.adjoint(as_element(self.space, q)))

    def is_empty(self):
        return self.base.is_empty()


def image(T, S):
    """T(S), computed exactly where the structure allows (affine sets, subspaces, sums), lazily otherwise."""
    from .subspaces import ClosedSubspace, image as subspace_image
    require(T.domain, S.space, f"{T.name} acts on {T.domain}; {S.name} is a subset of {S.space}", T.name)
    if isinstance(S, ClosedSubspace):
        return subspace_image(T, S)
    if isinstance(S, EmptySet):
        return EmptySet(T.codomain)
    if isinstance(S, AffineSubset):
        return AffineSubset(T(S.point), subspace_image(T, S.direction), name=f"{T.name}({S.name})")
    if isinstance(S, Translate):
        return Translate(image(T, S.base), T(S.shift))
    if isinstance(S, Minkowski):
        return minkowski(image(T, S.parts[0]), image(T, S.parts[1]))
    if hasattr(S, "image"):
        return S.image(T)                                   # feasible sets keep their dual
    return Image(T, S)


class Minkowski(ConvexSet):
    """A + B = {a + b}: sigma = sigma_A + sigma_B (app-add)."""

    def __init__(self, A, B):
        require(A.space, B.space, "a Minkowski sum needs two subsets of one space", "a Minkowski sum")
        self.parts = (A, B)
        super().__init__(A.space, f"{A.name} + {B.name}", _claims_of([A, B], "a Minkowski sum of convex sets"))

    def support(self, v):
        a = support_of(self.parts[0], v)
        if a.upper == -math.inf:                             # the first part is empty: so is the sum
            return a
        b = support_of(self.parts[1], v)
        return b if b.upper == -math.inf else a + b

    def is_empty(self):
        return self.parts[0].is_empty() | self.parts[1].is_empty()


def minkowski(A, B):
    """A + B for sets, subspaces or points (an element gives a translate)."""
    from .subspaces import Zero
    if stated_kind(A) is not SET:
        A, B = B, A
    if stated_kind(B) is not SET:
        return Translate(A, B)
    if isinstance(B, Zero):
        return A
    if isinstance(A, Zero):
        return B
    return Minkowski(A, B)


class Translate(ConvexSet):
    """x + S."""

    def __init__(self, S, shift):
        if getattr(shift, "space", None) != S.space:
            raise SpaceMismatch(f"cannot translate a subset of {S.space} by an element of another space")
        self.base, self.shift = S, shift
        super().__init__(S.space, f"{getattr(shift, 'name', 'x')} + {S.name}", list(S.claims))
        self.convex = getattr(S, "convex", False)

    def member(self, x):
        return self.base.member(x - self.shift)

    def support(self, v):
        v = as_element(self.space, v)
        s = support_of(self.base, v)
        return s if s.lower == s.upper and math.isinf(s.upper) else s + float(self.space.inner(v, self.shift))

    def is_empty(self):
        return self.base.is_empty()


class Scale(ConvexSet):
    """c S for c >= 0."""

    def __init__(self, S, c):
        if not c >= 0:
            raise ValueError("a convex set is scaled by c >= 0 (use the image under -I otherwise)")
        self.base, self.c = S, float(c)
        super().__init__(S.space, f"{c:g}·{S.name}", list(S.claims))

    def member(self, x):
        if self.c == 0:
            return Truth.of(self.space.norm(x) == 0) & ~self.base.is_empty()
        return self.base.member(x * (1.0 / self.c))

    def support(self, v):
        s = support_of(self.base, v)
        if self.c == 0:                                      # {0} if S is non-empty, else empty
            if s.lower > -math.inf:
                return exact(0.0, "0 S = {0} for a non-empty S")
            if s.upper == -math.inf:
                return s
            return Enclosure(0.0, -math.inf, 0.0, s.claim_status, "0 S: {0} or empty (S not known non-empty)")
        return s.scaled(self.c)


class Union(SubsetUnion):
    """A union of subsets of one space: support max (the support of its closed convex hull); membership: any. Not a
    ConvexSet (a union of convex sets is not convex in general), but it has interval and outer_polytope by its
    support."""

    interval, outer_polytope, polytope, plot = (ConvexSet.interval, ConvexSet.outer_polytope, ConvexSet.polytope,
                                                ConvexSet.plot)

    def __init__(self, sets, name=None):
        sets = list(sets)
        for S in sets:
            require(S.space, sets[0].space, "a union needs subsets of one space", "a union")
        super().__init__(sets, name)
        self._claims = [Claim("not convex in general", THEOREM, "a union of convex sets", key="convex", holds=False)]

    def support(self, v):
        return maximum(*[support_of(S, v) for S in self.parts])


class Intersection(SubsetIntersection):
    """A lazy intersection: membership of every part. Its support function needs a dual problem: for a prior set and a
    relation's converse, pg.update builds the feasible set with one (pygeoinf0.convex.feasible)."""

    def __init__(self, sets):
        sets = list(sets)
        for S in sets:
            require(S.space, sets[0].space, "an intersection needs subsets of one space", "an intersection")
        super().__init__(sets)
        self._claims = _claims_of(self.parts, "an intersection of convex sets")
        self.convex = all(getattr(S, "convex", False) for S in self.parts)

    def support(self, v):
        raise Unsupported("the support function of a general intersection needs a dual problem; the feasible set of "
                          "pg.update(prior, relation, d) carries one")


class CartesianProduct(ConvexSet):
    """S_1 x ... x S_K on the blocks of a stacked R^n (stack(..., rule='all')) or the summands of a direct sum:
    sigma(v) = sum_k sigma_k(v_k), with the summand weights w_k on a direct sum."""

    def __init__(self, sets, space, blocks=None):
        self.parts, self.blocks = tuple(sets), blocks
        super().__init__(space, " × ".join(S.name for S in self.parts),
                         _claims_of(self.parts, "a product of convex sets"))

    def _pieces(self, v):
        if self.blocks is not None:
            vals = as_element(self.space, v).values
            return [(1.0, S.space(vals[a:b])) for S, (a, b) in zip(self.parts, self.blocks)]
        X = self.space
        return [(w, X.part(v, n)) for w, n in zip(X.weights, X.names)]

    def support(self, v):
        terms = [support_of(S, p).scaled(w) for (w, p), S in zip(self._pieces(v), self.parts)]
        if any(t.upper == -math.inf for t in terms):         # an empty factor: the product is empty
            return next(t for t in terms if t.upper == -math.inf)
        return functools.reduce(lambda a, b: a + b, terms)

    def member(self, x):
        out = Truth.TRUE
        for (_, p), S in zip(self._pieces(x), self.parts):
            out = out & S.member(p)
        return out

