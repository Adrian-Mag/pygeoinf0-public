"""Intersections of regions (plan 067 P2: from core/intersections.py, with S & T added).

    intersect(S, T)     -> S & T exactly (the set algebra of Regions, sets/regions.py): never None
    intersection(S, T, D) -> a set, None when S cap T is null for the measure of the sets, or pg.Refused

intersect(S, T) for regions of one R^d (SpaceMismatch otherwise): equal sets and R^d give the other; intervals give an
Interval, a FiniteSet (closed ends that touch) or EmptySet(R) (sets/intervals.py); a finite set gives the points of it
that lie in the other set; boxes of one dimension give a Box, EmptySet(R^d) when apart, else (touching: a face of lower
dimension) the intersection known by membership; round sets (balls, shells, their pieces, bands of one sphere) give
domains/pieces.meet's set; when meet finds the intersection null, EmptySet(R^d) for balls, shells and their pieces
apart by more than meet's tolerance, else (touching, or bands) the intersection known by membership; anything else is the intersection known by
membership (RegionIntersection: exact membership, dimension and measure not computed).

intersection(S, T, D) is S cap T for the exact L2 pairings of symbolic representers (spaces/representers.py; EXP-031,
EXP-032 finding 1), up to null sets:
S and T are sets of the dimension k of the domain D (L2 pairings exist only for measures of full dimension), possibly
inside a higher-dimensional Cartesian space (a curve in R^2 or R^3, a patch in R^3). Decided, in this order:

1. intervals: S & T; equal sets, or one of them the whole domain: the other;
2. sub-paths of a curve D (a segment, polyline, great-circle arc or Curve, not closed): their ranges in D's arc-length
   coordinate, [s0, s1] cap [t0, t1], by D's own geometry (D.from_cartesian of the end points and of quadrature nodes,
   which also checks that the sets lie on D); a partial overlap is D.subpiece(lo, hi), of the same kind as D; on a
   closed curve with subpiece (a circle: arcs from Circle.arc) the ranges are periodic, unwrapped along each set's
   parametrisation; arcs that overlap twice are refused;
2b. round sets (balls, shells and their pieces, domains/pieces.py; plan 035): meet(S, T), exactly: a ball, a shell, a
   BallPiece (a ball cut by a ball or shell about another centre) or None; bounded by more spheres: refused;
2c. bands and polar caps of one sphere (domains/bands.py; plan 035 P5): their colatitude intervals, exactly;
3. flat parallelograms in a flat parallelogram D (patches in a plane, e.g. z = 0): their images in D's natural
   coordinates; rectangles there intersect in a rectangle, D.subpiece(lo, hi);
4. bounding boxes as closed intervals (touching allowed): boxes apart (a gap in some coordinate) mean disjoint sets;
   an overlap of the boxes with fewer than k coordinates of positive width lies in an affine set of dimension < k, so
   S cap T is null; two coordinate boxes intersect in a box; a set whose box lies in a coordinate box T is inside T;
5. convex polytopes of full dimension in R^k (boxes, parallelograms and parallelepipeds in any orientation, single
   simplices: tubes about rays, EXP-032 finding 2): their intersection, exactly, as a Simplicial set (half-spaces,
   a linear program for a common interior point, simplices from its centre to the facets of its hull);
6. otherwise refused (not representable here).

Rule 4 alone was the old test, with open intervals: the boxes of sets lying in a coordinate plane (curves and patches
in z = 0, segments parallel to an axis) have zero width in that coordinate, so overlapping sets were declared disjoint
and their pairing was 0 (EXP-032 finding 1).
"""
import numpy as np

from ..core.errors import Refused, Unsupported
from ..core.logic import Truth
from .base import EmptySet, common_ambient
from .finite import FiniteSet
from .intervals import Interval, meet_intervals
from .regions import RealSpace, RegionIntersection

TOL = 1e-12          # relative to the size of the domain: coordinates closer than this are equal


def intersect(S, T):
    """S & T for regions of one R^d, exactly where constructed, else known by membership; never None (module
    docstring)."""
    from ..domains import Box
    from ..domains.pieces import disjoint, meet
    A = common_ambient(S, T)
    if S is T or S == T or isinstance(T, RealSpace):
        return S
    if isinstance(S, RealSpace):
        return T
    if isinstance(S, Interval) and isinstance(T, Interval):
        return meet_intervals(S, T)
    if isinstance(S, FiniteSet) or isinstance(T, FiniteSet):
        F, O = (S, T) if isinstance(S, FiniteSet) else (T, S)
        if O.decides_membership:
            return F & O
        return RegionIntersection([S, T])
    if isinstance(S, Box) and isinstance(T, Box) and S.dim == T.dim:
        sides = [meet_intervals(a, b) for a, b in zip(S.sides, T.sides)]
        if any(isinstance(J, EmptySet) for J in sides):
            return EmptySet(A, why=f"{S!r} and {T!r} are apart")
        if all(isinstance(J, Interval) for J in sides):
            return Box(*sides)
        return RegionIntersection([S, T], why="boxes that touch: a face of lower dimension")
    try:
        J = meet(S, T)
    except Unsupported:
        return RegionIntersection([S, T])
    if J is not None:
        return J
    if disjoint(S, T) is Truth.TRUE:                      # null: apart (empty) or touching (not empty)
        return EmptySet(A, why=f"{S!r} and {T!r} are apart")
    return RegionIntersection([S, T], why="sets that touch (their intersection is null for their measure)")


def _ends_and_nodes(S):
    """Points of S (m, d): its two end points (from its parametrisation) and quadrature nodes; None if S is no curve."""
    try:
        par = S.parametrisation()
    except (AttributeError, TypeError):
        return None, None
    ends = np.asarray(par.gamma(np.array([par.t0, par.t1], float)), float).reshape(2, -1)
    from ..quadrature import rule
    X, _ = rule(S, 8)
    return ends, np.asarray(X, float)


def _range_on_curve(S, D):
    """[s0, s1]: the range of S in the arc-length coordinate of the curve D, or None when S does not lie on D."""
    if S == D or S.same_up_to_null_set(D):
        return tuple(D.reference[0])
    ends, X = _ends_and_nodes(S)
    if ends is None:
        return None
    s = np.asarray(D.from_cartesian(*np.vstack([ends, X]).T)[0], float)
    if not np.all(np.isfinite(s)):
        return None
    # S is connected and D is a simple open curve, so the image of S is an interval of D's coordinate; its end points
    # are the images of S's end points
    return float(min(s[0], s[1])), float(max(s[0], s[1]))


def _range_on_closed(S, D):
    """[s0, s1] (s0 in the period [lo, lo + P), s1 - s0 < P) for an arc S of the closed curve D in D's coordinate:
    D.from_cartesian along S's parametrisation, unwrapped; None when S is no arc of D (not on it, or not traversed
    monotonically: S is connected, so a monotone image is the arc [s0, s1])."""
    try:
        par = S.parametrisation()
    except (AttributeError, TypeError):
        return None
    lo, hi = D.reference[0]
    P = hi - lo
    t = np.linspace(par.t0, par.t1, 257)
    s = np.asarray(D.from_cartesian(*np.asarray(par.gamma(t), float).reshape(len(t), -1).T)[0], float)
    if not np.all(np.isfinite(s)):
        return None
    s = np.unwrap(s - lo, period=P)
    d, length = np.diff(s), abs(float(s[-1] - s[0]))
    if not (np.all(d > 0) or np.all(d < 0)) or length >= P * (1 - TOL):
        return None
    s0 = float(min(s[0], s[-1]))
    shift = np.floor(s0 / P) * P
    return lo + s0 - shift, lo + s0 - shift + length


def _from_periodic(S, T, D, a, b):
    """S cap T for arcs a, b of a closed curve (rule 2): one arc, None, or refused when they overlap twice."""
    P = D.reference[0][1] - D.reference[0][0]
    parts = [(max(a[0], b[0] + j * P), min(a[1], b[1] + j * P)) for j in (-1, 0, 1)]
    parts = [(x, y) for x, y in parts if y - x > TOL * P]
    if not parts:
        return None
    if len(parts) > 1:
        raise Refused(f"{S!r} and {T!r} overlap in two arcs of {D!r}; their L2 pairing is the sum of two pieces, "
                      "not constructed here", needs=("arcs whose union is not the whole curve",))
    x, y = parts[0]
    if abs(x - a[0]) <= TOL * P and abs(y - a[1]) <= TOL * P:
        return S
    if abs(x - b[0] - round((x - b[0]) / P) * P) <= TOL * P and abs(y - x - (b[1] - b[0])) <= TOL * P:
        return T
    return D.subpiece(np.array([x]), np.array([y]))


def _box_in_coordinates(S, D):
    """(lo, hi) of S in the natural coordinates of the flat piece D when the image of S there is a coordinate box;
    None otherwise (S not a flat piece of D's dimension lying in D, or a parallelogram not aligned with D's axes)."""
    if getattr(S, "dim", None) != D.dim or not hasattr(S, "edges") or S.cartesian_dim != D.cartesian_dim:
        return None
    o = np.asarray(S.origin, float)
    pts = np.array([o] + [o + np.asarray(e, float) for e in S.edges])
    C = np.stack([np.asarray(c, float) for c in D.from_cartesian(*pts.T)], axis=1)        # (1 + dim, dim)
    if not np.all(np.isfinite(C)):
        return None
    E = C[1:] - C[0]
    scale = max(float(np.max(np.abs(E))), 1e-300)
    nz = np.abs(E) > 1e-10 * scale
    if not (np.all(nz.sum(axis=1) == 1) and np.all(nz.sum(axis=0) == 1)):
        return None
    corners = C[0] + np.array([sum(c[k] * E[k] for k in range(D.dim)) for c in np.ndindex(*(2,) * D.dim)])
    return corners.min(axis=0), corners.max(axis=0)


def _from_ranges(S, T, D, a, b):
    """S cap T from their coordinate boxes a = (lo, hi) and b in D (closed: touching gives a null intersection)."""
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    size = max(float(np.max(np.asarray(D.reference, float)[:, 1] - np.asarray(D.reference, float)[:, 0])), 1e-300)
    if np.any(hi - lo <= TOL * size):
        return None
    if np.all(np.abs(lo - a[0]) <= TOL * size) and np.all(np.abs(hi - a[1]) <= TOL * size):
        return S
    if np.all(np.abs(lo - b[0]) <= TOL * size) and np.all(np.abs(hi - b[1]) <= TOL * size):
        return T
    subpiece = getattr(D, "subpiece", None)
    if subpiece is None:
        raise Refused(f"the overlap of {S!r} and {T!r} is a part of {D!r} that is not constructed here",
                      needs=("nested or disjoint sets",))
    return subpiece(lo, hi)


def _refuse(S, T):
    return Refused(f"the intersection of {S!r} and {T!r} is not representable here, so the inner product of their "
                   "L2 representers is not computed (it would need a quadrature of an indicator)",
                   needs=("equal or nested-in-the-domain sets, intervals, boxes, sub-paths of a path, coordinate "
                          "rectangles of a flat patch, or disjoint sets",))


def intersection(S, T, D):
    """S cap T (a set), None when it is null, or Refused when it is not representable here (module docstring)."""
    from ..domains import Box
    if isinstance(S, Interval) and isinstance(T, Interval):
        J = meet_intervals(S, T)
        return J if isinstance(J, Interval) else None        # a point or nothing: null for Lebesgue measure
    if S == T or S.same_up_to_null_set(T) or T == D or T.same_up_to_null_set(D):
        return S
    if S == D or S.same_up_to_null_set(D):
        return T
    k = int(D.dim)
    if k == 1 and not any(getattr(D, "periodic", (True,))) and hasattr(D, "from_cartesian"):     # rule 2
        a, b = _range_on_curve(S, D), _range_on_curve(T, D)
        if a is not None and b is not None:
            return _from_ranges(S, T, D, (np.array([a[0]]), np.array([a[1]])), (np.array([b[0]]), np.array([b[1]])))
    if k == 1 and any(getattr(D, "periodic", ())) and hasattr(D, "subpiece"):                # rule 2, a circle
        a, b = _range_on_closed(S, D), _range_on_closed(T, D)
        if a is not None and b is not None:
            return _from_periodic(S, T, D, a, b)
    if hasattr(D, "edges") and k >= 2:                                                          # rule 3
        a, b = _box_in_coordinates(S, D), _box_in_coordinates(T, D)
        if a is not None and b is not None:
            return _from_ranges(S, T, D, a, b)
    from ..domains.pieces import annuli, meet                                                  # rule 2b
    if annuli(S) is not None and annuli(T) is not None:
        try:
            return meet(S, T)
        except NotImplementedError:
            raise _refuse(S, T) from None
    from ..domains.bands import colatitudes, meet_bands                                          # rule 2c
    if colatitudes(S) is not None and colatitudes(T) is not None:
        return meet_bands(S, T)
    A, B = np.array(S.bounding_box, float), np.array(T.bounding_box, float)                     # rule 4
    lo, hi = np.maximum(A[:, 0], B[:, 0]), np.minimum(A[:, 1], B[:, 1])
    scale = max(float(np.max(np.maximum(A[:, 1] - A[:, 0], B[:, 1] - B[:, 0]))), 1e-300)
    if np.any(lo > hi + TOL * scale):
        return None                                        # closed boxes apart: the sets are disjoint
    if np.sum(hi - lo > TOL * scale) < k:
        return None                                        # the overlap lies in an affine set of dimension < k
    if isinstance(S, Box) and isinstance(T, Box):
        return Box(*[(x, y) for x, y in zip(lo, hi)])
    for inner_, outer in ((S, T), (T, S)):                 # a set inside a box (convex: its bounding box suffices)
        if isinstance(outer, Box):
            box = np.array(inner_.bounding_box, float)
            if all(s.a <= x0 and x1 <= s.b for s, (x0, x1) in zip(outer.sides, box)):
                return inner_
    a, b = _polytope(S, k), _polytope(T, k)                                                       # rule 5
    if a is not None and b is not None:
        return _convex_intersection(a, b, scale)
    raise _refuse(S, T)


def _polytope(S, k):
    """The vertices (m, k) of S when it is a convex polytope of full dimension k in R^k (a box, a parallelogram or
    parallelepiped, one simplex), else None."""
    from ..domains import Box
    from ..domains.simplicial import Simplicial
    if getattr(S, "dim", None) != k or getattr(S, "cartesian_dim", None) != k:
        return None
    if isinstance(S, Box):
        return np.array([[s.b if c else s.a for s, c in zip(S.sides, cs)] for cs in np.ndindex(*(2,) * k)], float)
    if isinstance(S, Simplicial):
        return S.vertices[S.cells[0]] if S.chart is None and len(S.cells) == 1 else None
    if hasattr(S, "edges") and hasattr(S, "origin"):
        E = np.asarray(S.edges, float)
        return np.asarray(S.origin, float) + np.array([np.asarray(c, float) @ E for c in np.ndindex(*(2,) * k)])
    return None


def _convex_intersection(A, B, scale):
    """The intersection of the convex hulls of the vertex sets A and B (exact up to rounding), a Simplicial set, or None
    when it is null (EXP-032 finding 2: overlapping parallelogram tubes). Every vertex of the intersection (k <= 3) is
    a vertex of one polytope inside the other, or where an edge of one crosses a facet plane of the other: these points
    (segments between all pairs of vertices, a superset of the edges, whose extra points lie inside and do not change
    the hull), kept when inside both, span it. Its cells: simplices from its centre to the facets of its hull."""
    from scipy.spatial import ConvexHull, QhullError

    from ..domains.simplicial import Simplicial
    k = A.shape[1]
    HA, HB = ConvexHull(A).equations, ConvexHull(B).equations                # a . x + b <= 0 inside, |a| = 1
    eps = 1e-12 * scale

    def inside(X, H):
        return np.all(X @ H[:, :k].T + H[:, k] <= eps, axis=1)

    pts = [A[inside(A, HB)], B[inside(B, HA)]]
    for V, H, other in ((A, HB, HA), (B, HA, HB)):
        i, j = np.triu_indices(len(V), 1)
        P, Q = V[i], V[j]
        fp, fq = P @ H[:, :k].T + H[:, k], Q @ H[:, :k].T + H[:, k]           # (pairs, facets)
        cross = (fp * fq < 0)
        t = np.where(cross, fp / np.where(cross, fp - fq, 1.0), 0.0)
        X = (P[:, None, :] + t[..., None] * (Q - P)[:, None, :])[cross]
        pts.append(X[inside(X, H) & inside(X, other)])
    X = np.vstack(pts)
    if len(X) <= k:
        return None
    try:
        hull = ConvexHull(X)
    except QhullError:
        return None                                         # the points lie in a hyperplane: a null intersection
    if hull.volume <= TOL * scale ** k:
        return None                                         # touching or apart
    c = hull.points[hull.vertices].mean(axis=0)
    V = np.vstack([c, hull.points])
    cells = np.c_[np.zeros(len(hull.simplices), int), hull.simplices + 1]
    vol = np.abs(np.linalg.det(V[cells[:, 1:]] - V[cells[:, :1]]))          # slivers of facets split by Qhull: dropped
    return Simplicial(V, cells[vol > 1e-11 * float(np.max(np.ptp(V, axis=0))) ** k])
