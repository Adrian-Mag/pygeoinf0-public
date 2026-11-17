"""Regions: subsets of R^d (plan 067 P2; architecture review 2026-11-13, L1, B1 and B7; decision Q3).

    Region            a subset of R^d with
        dim               its (Hausdorff) dimension k: 0 for finite sets, 1 for intervals and curves, 2 for surfaces
        cartesian_dim     d, the number of Cartesian coordinates of its points (ambient_dim when it is larger than dim)
        volume            its k-dimensional Hausdorff measure H^k: length, area, volume, or the number of points
        contains(points)  membership of arrays of points, the one point convention below
        member(x)         a Truth for one point x; `x in S` is bool of it (sets/base.py)
        S & T, S | T      constructed exactly where sets/intersections.py can (intervals, boxes, round sets, finite
                          sets), else the intersection or union known by membership; never None: the empty
                          intersection is EmptySet(R^d)
    RealSpace(d)      R^d itself, the ambient of every region in it; RealLine() = R is R^1
    charts()          the charts that cover it up to a null set (Chart, Simplices), or None when it states none: the
                      geometry through which the numerical column integrates over it (pygeoinf0.quadrature.rule; plan
                      069 P2: regions carry no quadrature)

The point convention (decision Q3), one for every Region: points of R^d are arrays of shape (..., d) and the answer
has shape (...); one point of shape (d,) gives a bool. This holds for d = 1 too: n points of R^1 are an array of shape
(n, 1) (t[:, None] for n numbers t), and shape (1,) is one point. A number is also one point of R^1. A flat array of
n >= 2 numbers is refused (ValueError naming t[:, None]), never read silently as n points: the open decision of plan
067 P2 records the alternative (numbers as points of R^1, as M.riesz reads them).
"""
import collections
import math

import numpy as np

from ..core.errors import SpaceMismatch, Unsupported
from ..core.logic import Truth
from .base import EmptySet, Intersection, Set, Union, common_ambient


def as_points(points, d):
    """(X, shape): the points as an (n, d) array and the shape of the answer (module docstring)."""
    X = np.asarray(points, dtype=float)
    if d == 1 and X.ndim == 0:
        return X.reshape(1, 1), ()                                # a number: one point of R^1
    if X.ndim == 0 or X.shape[-1] != d:
        raise ValueError(f"points of R^{d} are arrays of shape (..., {d}) (got shape {X.shape}); one point is an array "
                         f"of shape ({d},), m points (m, {d})" + (": n numbers t are the points t[:, None]" if d == 1
                                                                 else ""))
    return X.reshape(-1, d), X.shape[:-1]


def as_point(x, d):
    """x as one point of R^d, shape (d,), or None when x is not numeric (so it is no point of R^d); ValueError when x is
    numeric but of another shape (an array of several points: use contains)."""
    try:
        X = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if X.shape == (d,) or (d == 1 and X.ndim == 0):
        return X.reshape(d)
    raise ValueError(f"member(x) takes one point of R^{d}, shape ({d},){' or a number' if d == 1 else ''} (got shape "
                     f"{X.shape}); for arrays of points use contains(points)")


Chart = collections.namedtuple("Chart", "reference periodic breaks to_cartesian jacobians density")
Chart.__doc__ = """A box chart of a region: to_cartesian(*xi) maps coordinate arrays of one shape on the reference box
((lo_k, hi_k) per axis) to a tuple of cartesian_dim arrays of that shape, and the region's measure is
prod_k J_k(xi_k) density(xi) dxi there. periodic: per axis, whether it is an angle (the map has period hi - lo);
breaks: per axis the coordinates where integrands of interest may lose smoothness, or None; jacobians: per axis a
function J_k of that coordinate, or None for 1 (or None for all); density: a function of all the coordinate arrays (the
part of the measure that is no product), or None."""

Simplices = collections.namedtuple("Simplices", "origins edges scale lift")
Simplices.__doc__ = """Simplicial cells of a region: cell j is {o_j + u E_j : u in the reference k-simplex} (origins (m, d),
edges (m, k, d)) with the flat measure scale_j du (scale_j = k! times its volume); lift(X, j) maps flat points X of
cells j onto the region (a radial projection onto a sphere) and returns (points, the factor of the measure), or lift is
None (the flat cells are the region)."""


class Region(Set):
    """A subset of R^d (module docstring). Subclasses give dim, cartesian_dim (or ambient_dim), volume and the test of
    membership: _inside(*coordinates) (coordinate arrays of one shape) or _inside_points(X) (X of shape (n, d)).
    decides_membership = False for a region whose membership is not decided here (member UNKNOWN, contains refused)."""

    dim = None
    ambient_dim = None
    decides_membership = True

    @property
    def cartesian_dim(self):
        return self.ambient_dim or self.dim

    @property
    def ambient(self):
        return RealSpace(int(self.cartesian_dim))

    @property
    def volume(self):
        raise Unsupported(f"the Hausdorff measure of {self!r} is not computed here")

    def charts(self):
        """The charts covering the region up to a null set (a tuple of Chart or Simplices), or None: a region known by
        membership only states none, and no rule integrates over it (pygeoinf0.quadrature.rule refuses)."""
        return None

    def _inside(self, *x):
        raise Unsupported(f"membership in {self!r} is not decided here")

    def _inside_points(self, X):
        return np.broadcast_to(np.asarray(self._inside(*X.T), bool), (len(X),))

    def contains(self, points):
        """Whether each point lies in the set: points of shape (..., d), the answer of shape (...) (a bool for one
        point; module docstring for R^1)."""
        if not self.decides_membership:
            raise Unsupported(f"membership in {self!r} is not decided here (member(x) answers Truth.UNKNOWN)")
        X, shape = as_points(points, int(self.cartesian_dim))
        r = np.asarray(self._inside_points(X), bool).reshape(shape)
        return bool(r) if r.ndim == 0 else r

    def member(self, x):
        p = as_point(x, int(self.cartesian_dim))
        if p is None:
            return Truth.FALSE                               # not a point of R^d
        if not self.decides_membership:
            return Truth.UNKNOWN
        return Truth.of(bool(np.all(self.contains(p[None])[0])))

    def __and__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        if not isinstance(other, Region):
            A = common_ambient(self, other)
            return other if isinstance(other, EmptySet) else Intersection([self, other], A)
        from .intersections import intersect
        return intersect(self, other)

    def __or__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        A = common_ambient(self, other)
        if isinstance(other, EmptySet) or other is self or other == self:
            return self
        if not isinstance(other, Region):
            return Union([self, other], A)
        return RegionUnion([self, other])


class RealSpace(Region):
    """R^d as a set (finite points), the ambient of the regions in it."""

    def __init__(self, d):
        if int(d) != d or d < 1:
            raise ValueError("R^d needs an integer d >= 1")
        self.d = int(d)

    @property
    def dim(self):
        return self.d

    @property
    def cartesian_dim(self):
        return self.d

    @property
    def ambient(self):
        return self

    @property
    def volume(self):
        return math.inf

    def _inside_points(self, X):
        return np.all(np.isfinite(X), axis=1)

    def __eq__(self, other):
        return isinstance(other, RealSpace) and other.d == self.d

    def __hash__(self):
        return hash(("R^d", self.d))

    def __repr__(self):
        return "R" if self.d == 1 else f"R^{self.d}"


class RealLine(RealSpace):
    """The real line R = R^1 (finite floats)."""

    def __init__(self):
        super().__init__(1)


R = RealLine()


class RegionIntersection(Intersection, Region):
    """S_1 ∩ ... ∩ S_k of regions in one R^d, known by membership (the intersections not constructed in
    sets/intersections.py): member is Kleene's &; contains needs every part to decide; dim and volume are those given
    (a stated fact), else not computed."""

    def __init__(self, parts, dim=None, volume=None, why=""):
        Intersection.__init__(self, parts, None)
        d = {int(S.cartesian_dim) for S in self.parts}
        if len(d) != 1:
            raise SpaceMismatch(f"regions in R^{min(d)} and R^{max(d)} have no intersection here")
        self._d, self.dim, self._volume, self.why = d.pop(), dim, volume, why

    @property
    def cartesian_dim(self):
        return self._d

    @property
    def ambient(self):
        return RealSpace(self._d)

    @property
    def decides_membership(self):
        return all(S.decides_membership for S in self.parts)

    member = Intersection.member

    def _inside_points(self, X):
        r = np.ones(len(X), bool)
        for S in self.parts:
            r &= np.atleast_1d(S.contains(X))
        return r

    @property
    def volume(self):
        if self._volume is None:
            raise Unsupported(f"the Hausdorff measure of {self!r} is not computed here (an intersection known by "
                              "membership)")
        return self._volume


class RegionUnion(Union, Region):
    """S_1 ∪ ... ∪ S_k of regions in one R^d, known by membership: member is Kleene's |; dim is the largest of the
    parts' (when known); volume is the one given (a Partition knows its union's), else not computed."""

    def __init__(self, parts, dim=None, volume=None, why=""):
        Union.__init__(self, parts, None)
        d = {int(S.cartesian_dim) for S in self.parts}
        if len(d) != 1:
            raise SpaceMismatch(f"regions in R^{min(d)} and R^{max(d)} have no union here")
        dims = [S.dim for S in self.parts]
        self._d, self._volume, self.why = d.pop(), volume, why
        self.dim = dim if dim is not None else (max(dims) if None not in dims else None)

    @property
    def cartesian_dim(self):
        return self._d

    @property
    def ambient(self):
        return RealSpace(self._d)

    @property
    def decides_membership(self):
        return all(S.decides_membership for S in self.parts)

    member = Union.member

    def _inside_points(self, X):
        r = np.zeros(len(X), bool)
        for S in self.parts:
            r |= np.atleast_1d(S.contains(X))
        return r

    @property
    def volume(self):
        if self._volume is None:
            raise Unsupported(f"the Hausdorff measure of {self!r} is not computed here (a union known by membership)")
        return self._volume
