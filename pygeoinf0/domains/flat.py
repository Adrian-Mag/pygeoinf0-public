"""Paths and flat pieces: segments, parallelograms, polylines, great-circle arcs.

They are Domains with fewer natural coordinates than Cartesian ones (dim < ambient_dim), like Circle and Sphere, so
they carry function spaces (L2(path)), measures (LebesgueMeasure(path) is arc length or area) and quadratures, and
they are the supports of the measures that define ray integrals:

    Segment(p, q)          the straight segment from p to q in R^2 or R^3; natural coordinate s in [0, |q - p|]
                           (arc length)
    Patch(o, e1, e2)       the flat parallelogram o + a e1 + b e2 (a, b in [0, 1]) in R^3; natural coordinates
                           the lengths along e1 and e2; area measure (constant weight sin of the angle)
    Polyline(points)       the path through the points; arc length, with the vertices as reference breaks
    Arc(sphere, p, q)      the shorter great-circle arc on a sphere from p to q; arc length

Segment and Patch are one class, Flat: an affine image of a box. Points given in Cartesian coordinates are accepted
within a relative distance 1e-8 of the set (others give NaN), like on circles and spheres.
"""
import math

import numpy as np

from .base import Domain, Parametrisation, _fmt, scaled

ON = 1e-8


def _vec(p):
    return tuple(float(c) for c in np.asarray(p, float).ravel())


class Flat(Domain):
    """An affine image o + sum_k a_k e_k (a_k in [0, 1]) of a box, of dimension len(edges) in R^len(o)."""

    def __init__(self, origin, edges):
        o = np.asarray(origin, float).ravel()
        E = np.atleast_2d(np.asarray(edges, float))
        if E.shape[1] != o.size or not 1 <= len(E) <= o.size:
            raise ValueError("a flat piece needs edges in the ambient space, at most as many as its dimension")
        lens = np.linalg.norm(E, axis=1)
        if np.any(lens <= 0):
            raise ValueError("edges must have positive length")
        U = E / lens[:, None]
        if len(E) == 2:
            c = float(U[0] @ U[1])
            if abs(c) > 1 - 1e-12:
                raise ValueError("the edges of a patch must not be parallel")
            sin = math.sqrt(1 - c * c)
        elif len(E) == 3:                                   # a parallelepiped: |det| of the unit edges
            sin = abs(float(np.linalg.det(U)))
            if sin < 1e-12:
                raise ValueError("the edges of a parallelepiped must be linearly independent")
        else:
            sin = 1.0
        self._J = sin
        self.origin, self.edges = _vec(o), tuple(_vec(e) for e in E)
        self._o, self._U, self._lens = o, U, lens
        self.dim, self.ambient_dim = len(E), o.size
        self.reference = tuple((0.0, float(L)) for L in lens)
        self.jacobian = (scaled(sin, "1"),) + ("1",) * (len(E) - 1)
        self.periodic = (False,) * len(E)
        self.internal_names = ("s", "t", "u")[: len(E)]
        # coordinates of a point: solve for (s, t) by least squares in the span of the unit edges
        self._pinv = np.linalg.pinv(U.T)

    def to_cartesian(self, *s):
        out = [self._o[i] + sum(np.asarray(s[k], float) * self._U[k, i] for k in range(self.dim))
               for i in range(self.ambient_dim)]
        return tuple(out)

    def from_cartesian(self, *x):
        X = np.stack(np.broadcast_arrays(*[np.asarray(a, float) for a in x]), axis=-1) - self._o
        S = X @ self._pinv.T
        resid = np.linalg.norm(X - S @ self._U, axis=-1)
        scale = float(self._lens.max())
        ok = resid <= ON * scale
        for k in range(self.dim):
            ok &= (S[..., k] >= -ON * scale) & (S[..., k] <= self._lens[k] * (1 + ON))
        return tuple(np.where(ok, np.clip(S[..., k], 0, self._lens[k]), np.nan) for k in range(self.dim))

    def _inside(self, *x):
        return np.isfinite(self.from_cartesian(*x)[0])

    @property
    def bounding_box(self):
        corners = np.array([self._o + sum(c[k] * self._lens[k] * self._U[k] for k in range(self.dim))
                            for c in np.ndindex(*(2,) * self.dim)])
        return [(float(lo), float(hi)) for lo, hi in zip(corners.min(axis=0), corners.max(axis=0))]

    @property
    def volume(self):
        v = float(np.prod(self._lens))
        return v * (math.sqrt(1 - float(self._U[0] @ self._U[1]) ** 2) if self.dim == 2 else self._J)

    def parametrisation(self):
        """A segment as a curve (the Curve interface): gamma(t) = p + t u, t in [0, length], speed 1."""
        if self.dim != 1:
            raise TypeError(f"{self!r} is not a curve")
        o, u = self._o, self._U[0]
        return Parametrisation(lambda t: o + np.multiply.outer(np.asarray(t, float), u), lambda t: np.broadcast_to(
            u, np.shape(t) + u.shape), 0.0, float(self._lens[0]), (), False)

    @property
    def endpoints(self):
        """For a segment: its two end points (p, q)."""
        return np.array(self._o), np.array(self._o + self._lens[0] * self._U[0])

    def subpiece(self, lo, hi):
        """The part whose natural coordinates lie in the box [lo, hi] (lo < hi inside the reference box): a Flat piece
        with the same edge directions (exact)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return Flat(np.array(self.to_cartesian(*lo)), [(hi[k] - lo[k]) * self._U[k] for k in range(self.dim)])

    def key(self):
        return ("Flat", self.origin, self.edges)

    def __repr__(self):
        p = lambda v: "(" + ", ".join(_fmt(round(c, 12)) for c in v) + ")"  # noqa: E731
        if self.dim == 1:
            q = tuple(a + b for a, b in zip(self.origin, self.edges[0]))
            return f"Segment({p(self.origin)}, {p(q)})"
        if self.dim == 3:
            return f"Flat({p(self.origin)}, [{', '.join(p(e) for e in self.edges)}])"
        return f"Patch({p(self.origin)}, {p(self.edges[0])}, {p(self.edges[1])})"


def Segment(p, q):
    """The straight segment from p to q (points of R^2 or R^3); natural coordinate: arc length from p."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.shape == q.shape and np.array_equal(p, q):
        raise ValueError(f"Segment(p, q) needs p != q (got p = q = {tuple(p)}: a ray of length zero carries no data)")
    return Flat(p, [q - p])


def Patch(origin, e1, e2):
    """The flat parallelogram origin + a e1 + b e2, a, b in [0, 1], in R^3 (area measure), or in R^2 (a parallelogram
    of full dimension: Lebesgue measure; EXP-031)."""
    return Flat(origin, [e1, e2])


class Polyline(Domain):
    """The path through points p_0, ..., p_m (m >= 1) of R^2 or R^3; natural coordinate: arc length; the vertices are
    reference breaks (functions on it, and restrictions of smooth fields to it, may have kinks there)."""

    dim = 1
    periodic = (False,)
    internal_names = ("s",)
    jacobian = ("1",)

    def __init__(self, points):
        P = np.asarray(points, float)
        if P.ndim != 2 or len(P) < 2 or P.shape[1] not in (2, 3):
            raise ValueError("a polyline needs at least two points of R^2 or R^3")
        seg = np.diff(P, axis=0)
        L = np.linalg.norm(seg, axis=1)
        if np.any(L <= 0):
            raise ValueError("consecutive points of a polyline must differ")
        self.points = tuple(_vec(p) for p in P)
        self._P, self._U, self._L = P, seg / L[:, None], L
        self._cum = np.concatenate([[0.0], np.cumsum(L)])
        self.ambient_dim = P.shape[1]
        self.reference = ((0.0, float(self._cum[-1])),)
        self.reference_breaks = (tuple(float(c) for c in self._cum[1:-1]),)

    @property
    def segments(self):
        """The segments (p_i, p_i+1) as arrays (m, d), (m, d)."""
        return self._P[:-1].copy(), self._P[1:].copy()

    def pieces(self):
        """The segments as Segment domains (the pieces of the set)."""
        return tuple(Segment(p, q) for p, q in zip(self._P[:-1], self._P[1:]))

    def parametrisation(self):
        """The polyline as a curve: gamma(s) = to_cartesian(s), s in [0, length] (arc length, speed 1), breaks at the
        vertices."""
        def gamma(t):
            return np.stack(self.to_cartesian(np.asarray(t, float)), axis=-1)

        def derivative(t):
            t = np.asarray(t, float)
            i = np.clip(np.searchsorted(self._cum, t, side="right") - 1, 0, len(self._L) - 1)
            return self._U[i]
        return Parametrisation(gamma, derivative, 0.0, float(self._cum[-1]), tuple(self._cum[1:-1]), False)

    def to_cartesian(self, s):
        s = np.asarray(s, float)
        i = np.clip(np.searchsorted(self._cum, s, side="right") - 1, 0, len(self._L) - 1)
        loc = s - self._cum[i]
        return tuple(self._P[i, k] + loc * self._U[i, k] for k in range(self.ambient_dim))

    def from_cartesian(self, *x):
        X = np.stack(np.broadcast_arrays(*[np.asarray(a, float) for a in x]), axis=-1)
        rel = X[..., None, :] - self._P[:-1]                                # (..., m, d)
        t = np.clip(np.einsum("...md,md->...m", rel, self._U), 0, self._L)
        dist = np.linalg.norm(rel - t[..., None] * self._U, axis=-1)
        i = np.argmin(dist, axis=-1)
        best = np.take_along_axis(dist, i[..., None], axis=-1)[..., 0]
        s = self._cum[i] + np.take_along_axis(t, i[..., None], axis=-1)[..., 0]
        return (np.where(best <= ON * self._cum[-1], s, np.nan),)

    def _inside(self, *x):
        return np.isfinite(self.from_cartesian(*x)[0])

    @property
    def bounding_box(self):
        return [(float(lo), float(hi)) for lo, hi in zip(self._P.min(axis=0), self._P.max(axis=0))]

    @property
    def volume(self):
        return float(self._cum[-1])

    def subpiece(self, lo, hi):
        """The sub-path between arc lengths lo[0] < hi[0] (exact: the vertices between them are kept)."""
        a, b = float(np.ravel(lo)[0]), float(np.ravel(hi)[0])
        inner = [self._P[i] for i in range(1, len(self._P) - 1) if a < self._cum[i] < b]
        ends = [np.array(self.to_cartesian(np.array(v))).ravel() for v in (a, b)]
        return Polyline([ends[0]] + inner + [ends[1]])

    def key(self):
        return ("Polyline", self.points)

    def __repr__(self):
        return f"Polyline({len(self.points)} points, length {_fmt(round(self.volume, 12))})"


class Arc(Domain):
    """The shorter great-circle arc from p to q on a sphere (p, q on the sphere, not antipodal); arc length."""

    dim, ambient_dim = 1, 3
    periodic = (False,)
    internal_names = ("s",)
    jacobian = ("1",)

    def __init__(self, sphere, p, q):
        c, R = np.asarray(sphere.centre, float), float(sphere.radius)
        a, b = np.asarray(p, float) - c, np.asarray(q, float) - c
        if abs(np.linalg.norm(a) - R) > 1e-8 * R or abs(np.linalg.norm(b) - R) > 1e-8 * R:
            raise ValueError("the end points of an arc must lie on the sphere")
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        ang = math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b))
        if ang <= 0 or ang >= math.pi - 1e-12:
            raise ValueError("the end points must differ and not be antipodal (the great circle is then not unique)")
        e = b - (a @ b) * a
        e /= np.linalg.norm(e)
        self.sphere, self.p, self.q = sphere, _vec(p), _vec(q)
        self._c, self._R, self._a, self._e, self.angle = c, R, a, e, ang
        self.reference = ((0.0, R * ang),)

    def to_cartesian(self, s):
        th = np.asarray(s, float) / self._R
        return tuple(self._c[k] + self._R * (np.cos(th) * self._a[k] + np.sin(th) * self._e[k]) for k in range(3))

    def from_cartesian(self, x, y, z):
        X = np.stack(np.broadcast_arrays(*[np.asarray(v, float) for v in (x, y, z)]), axis=-1) - self._c
        n = np.cross(self._a, self._e)
        on = (np.abs(np.linalg.norm(X, axis=-1) - self._R) <= ON * self._R) & (np.abs(X @ n) <= ON * self._R)
        th = np.arctan2(X @ self._e, X @ self._a)
        ok = on & (th >= -ON) & (th <= self.angle + ON)
        return (np.where(ok, np.clip(th, 0, self.angle) * self._R, np.nan),)

    def _inside(self, x, y, z):
        return np.isfinite(self.from_cartesian(x, y, z)[0])

    @property
    def bounding_box(self):
        P = np.stack(self.to_cartesian(np.linspace(0, self.reference[0][1], 257)), axis=1)
        return [(float(lo), float(hi)) for lo, hi in zip(P.min(axis=0), P.max(axis=0))]

    @property
    def volume(self):
        return self._R * self.angle

    def subpiece(self, lo, hi):
        """The sub-arc between arc lengths lo[0] < hi[0] (exact: an arc of the same great circle)."""
        a, b = float(np.ravel(lo)[0]), float(np.ravel(hi)[0])
        return Arc(self.sphere, *[np.array(self.to_cartesian(np.array(v))).ravel() for v in (a, b)])

    def parametrisation(self):
        """The arc as a curve: gamma(s) = to_cartesian(s), s in [0, R angle] (arc length, speed 1)."""
        a, e, R = self._a, self._e, self._R

        def gamma(t):
            return np.stack(self.to_cartesian(np.asarray(t, float)), axis=-1)

        def derivative(t):
            th = np.asarray(t, float)[..., None] / R
            return -np.sin(th) * a + np.cos(th) * e
        return Parametrisation(gamma, derivative, 0.0, R * self.angle, (), False)

    def key(self):
        return ("Arc", self.sphere.key(), self.p, self.q)

    def __repr__(self):
        return f"Arc on {self.sphere!r}, {math.degrees(self.angle):.4g} degrees"


def clip(segment, domain):
    """The part of a segment inside a Box, Disk, Cylinder or Ball of the same dimension, as a Segment, or None when
    they meet in at most a point (or in a chord shorter than 1e-12 of the segment). Exact: these domains are convex, so a line meets each in an
    interval, cut out by the bounding slabs and, for the round ones, by the quadratic |x - c|^2 <= R^2 (in x, y for a
    cylinder)."""
    from .solids import Ball
    from .solids import Box
    from .solids import Cylinder
    from .solids import Disk
    if not (isinstance(segment, Flat) and segment.dim == 1):
        raise TypeError("clip takes a Segment")
    if not isinstance(domain, (Box, Disk, Cylinder, Ball)) or domain.cartesian_dim != segment.cartesian_dim:
        raise TypeError(f"clip needs a Box, Disk, Cylinder or Ball of dimension {segment.cartesian_dim}, not {domain!r}")
    o, u, L = segment._o, segment._U[0], float(segment._lens[0])
    lo, hi = 0.0, L
    for k, (a, b) in enumerate(domain.bounding_box):
        if u[k] == 0:
            if not a <= o[k] <= b:
                return None
            continue
        t0, t1 = sorted(((a - o[k]) / u[k], (b - o[k]) / u[k]))
        lo, hi = max(lo, t0), min(hi, t1)
    if not isinstance(domain, Box):
        idx = slice(0, 2) if isinstance(domain, Cylinder) else slice(None)
        w, v = o[idx] - np.asarray(domain.centre, float)[idx], u[idx]
        a, b, c = float(v @ v), float(w @ v), float(w @ w) - domain.radius ** 2
        if a == 0:
            if c > 0:
                return None
        else:
            disc = b * b - a * c
            if disc <= 0:
                return None
            r = math.sqrt(disc)
            lo, hi = max(lo, (-b - r) / a), min(hi, (-b + r) / a)
    if hi - lo <= 1e-12 * L:
        return None
    return Segment(o + lo * u, o + hi * u)
