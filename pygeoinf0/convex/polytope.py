"""Vertices and hulls of polytopes {x : A x <= b} in R^1, R^2, R^3 (plan 058), in standard coordinates.

vertices(A, b): the exact intersection of all the half-spaces (scipy's HalfspaceIntersection about the Chebyshev
centre, an LP of optimisers/lp.py), each vertex then re-solved from its active constraints by least squares, so it is
the intersection point to rounding. Every half-space counts: a redundant one changes nothing, and the polygon is
never the one through consecutive supporting lines, which is larger when a direction is redundant. Refused, with the
reason: an empty polytope, an unbounded one (a coordinate LP unbounded), one without interior (the largest inscribed
ball has radius 0 to rounding: lower-dimensional) and dimensions above 3.

hull(points): the half-space form (A, b) of the convex hull of points (ConvexHull's facets; in R^1 the interval).

inner_polytope(P, Q): for a feasible image P = T(K) with a pointwise box prior, the hull of T(m_j) over the
witnesses m_j of the bounds in the directions Q (cell fields, or the one-slab route's elements; each in K: its data
constraint checked in floating point), so the hull lies in P by convexity. The other routes' witnesses are coordinates of their reduction, not elements of the
model space: not built there (refused).
"""
import math

import numpy as np

from ..core.errors import NotDefined, Unsupported
from ..optimisers.lp import lp_max


def chebyshev(A, b):
    """(centre, radius) of the largest ball inside {A x <= b} (max r with A x + r |a_i| <= b): radius +inf with no
    finite largest ball; negative when empty (every point violates some half-space by at least -r)."""
    A, b = np.atleast_2d(np.asarray(A, float)), np.asarray(b, float).ravel()
    n = A.shape[1]
    val, x = lp_max(np.r_[np.zeros(n), 1.0], np.c_[A, np.linalg.norm(A, axis=1)], b)
    if not math.isfinite(val):
        return None, val
    return np.asarray(x, float)[:n], float(val)


def bounded(A, b):
    """Truth of boundedness as a bool: every coordinate LP max of +-x_k finite (a non-empty polytope)."""
    A, b = np.atleast_2d(np.asarray(A, float)), np.asarray(b, float).ravel()
    n = A.shape[1]
    return all(math.isfinite(lp_max(s * np.eye(n)[k], A, b)[0]) for k in range(n) for s in (1.0, -1.0))


def _scale(A, b):
    return max(1.0, float(np.abs(b).max(initial=0.0)), float(np.abs(A).max(initial=0.0)))


def vertices(A, b):
    """The vertices (k, n) of the bounded polytope {A x <= b} with interior (module docstring); in R^2 in
    counter-clockwise order, in R^1 the two ends."""
    A, b = np.atleast_2d(np.asarray(A, float)), np.asarray(b, float).ravel()
    n = A.shape[1]
    if n > 3:
        raise Unsupported(f"vertices of polytopes in R^{n}: only R^1, R^2 and R^3 (project onto 2 or 3 coordinates)")
    c, r = chebyshev(A, b)
    tol = 1e-12 * _scale(A, b)
    if r < -tol:                                                # every point violates some half-space by r or more
        raise Unsupported("the polytope is empty: it has no vertices")
    if r == math.inf or not bounded(A, b):
        raise NotDefined("the polytope is unbounded (a coordinate is unbounded on it): it has no finite vertex list")
    if r <= tol:
        raise Unsupported(f"the polytope has no interior (largest inscribed ball radius {r:.3g}): it is "
                          "lower-dimensional; its vertices are not computed here")
    if n == 1:
        return np.array([[-lp_max(np.array([-1.0]), A, b)[0]], [lp_max(np.array([1.0]), A, b)[0]]])
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    pts = HalfspaceIntersection(np.c_[A, -b], c).intersections
    pts = pts[ConvexHull(pts).vertices]
    out = []
    for p in pts:                                               # the intersection of its active constraints, to rounding
        res = b - A @ p
        act = np.nonzero(res <= 1e-8 * _scale(A, b))[0]
        if len(act) >= n:
            p = np.linalg.lstsq(A[act], b[act], rcond=None)[0]
        out.append(p)
    return np.array(out)


def hull(points):
    """(A, b) with conv(points) = {A x <= b}: ConvexHull's facets (R^2, R^3) or the interval (R^1). Points spanning
    fewer dimensions than the space are refused (no interior)."""
    P = np.atleast_2d(np.asarray(points, float))
    n = P.shape[1]
    if n == 1:
        return np.array([[1.0], [-1.0]]), np.array([P.max(), -P.min()])
    if n > 3:
        raise Unsupported(f"hulls in R^{n}: only R^1, R^2 and R^3")
    from scipy.spatial import ConvexHull, QhullError
    try:
        H = ConvexHull(P)
    except QhullError as err:
        raise Unsupported(f"the {len(P)} points span fewer than {n} dimensions: their hull has no interior") from err
    return H.equations[:, :n], -H.equations[:, n]


def inner_polytope(P, directions):
    """The inner Polytope of a feasible image (module docstring)."""
    from .feasible import FeasibleImage
    from .sets import Polytope
    if not (isinstance(P, FeasibleImage) and P.K._box()):
        raise Unsupported(f"no inner polytope of {P.name}: it needs feasible points as elements, which only a feasible "
                          "image with a pointwise box prior gives (its cell witnesses); other routes' witnesses are in "
                          "the coordinates of their reduction")
    pts = []
    for q in np.atleast_2d(np.asarray(directions, float)):
        bd = P.bound(P.space(P.space.solve_metric(q)))
        cp = getattr(bd, "cells", None)
        if bd.witness is None or not math.isfinite(bd.lower):
            continue
        if hasattr(bd.witness, "space"):                     # an element (the one-slab route): T m itself
            pts.append(np.asarray(P.T(bd.witness).values, float))
            continue
        if cp is None or cp.images is None:
            raise Unsupported("the cells were given by the caller (K.cells): their witnesses are not mapped to models")
        pts.append(cp.images(P.T).T @ np.asarray(bd.witness, float))
    if not pts:
        raise Unsupported(f"no witness in any of the directions: no inner polytope of {P.name}")
    A, b = hull(np.array(pts))
    return Polytope(P.space, A, b, f"inner({P.name})", "inner", P)
