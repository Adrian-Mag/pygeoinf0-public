"""Polyhedral sets: unions of simplices (EXP-031, DESIGN 4.2; question 4), and exact closure checks (DESIGN 4.3).

    Simplicial(vertices, cells, chart=None)   the union of the k-simplices `cells` (rows of k + 1 indices into
                                              `vertices`, points of R^d, k <= d): a continuum set, the generalisation of
                                              Polyline (its k = 1 case), not a discretisation. Its measure is the
                                              k-dimensional Hausdorff measure (length, area, volume).
    Simplex(*vertices)                        the one-cell case.

With chart=sphere (a Sphere), the cells are the radial projections of the flat triangles onto the sphere: curved pieces
of the sphere, exact (area element R^2 |(x - c) . n| / |x - c|^3 on the flat triangle); then the set is the union of
those spherical triangles, which is the sphere when they cover it. Without a chart the set is the polyhedron, exactly.
The package never triangulates a curved set itself (question 5).

A Simplicial set is a Region (sets/regions.py), not a Domain: function spaces on it are out of scope (question 4). Its charts
are its cells (sets/regions.Simplices); pygeoinf0.quadrature.rule(S, n) is the collapsed (Duffy) Gauss-Jacobi rule
per cell, exact for polynomials of total degree 2n - 1 on flat cells.

inside_closure(S, D) decides S in closure(D) exactly where convexity or roundness allows it, else returns Truth.UNKNOWN.
"""
import math

import numpy as np

from ..core.logic import Truth
from ..quadrature import rule
from ..sets import Region
from ..sets.regions import Simplices
from .flat import ON


class Simplicial(Region):
    """A union of simplices (module docstring)."""

    def __init__(self, vertices, cells, chart=None, name=None):
        from .manifolds import Sphere
        V = np.atleast_2d(np.asarray(vertices, float))
        C = np.atleast_2d(np.asarray(cells, int))
        if V.shape[1] not in (2, 3) or C.ndim != 2 or not 2 <= C.shape[1] <= V.shape[1] + 1:
            raise ValueError("a simplicial set needs vertices in R^2 or R^3 and cells of k + 1 indices, 1 <= k <= d")
        if C.min() < 0 or C.max() >= len(V):
            raise ValueError("cell indices must refer to the vertices")
        self.vertices, self.cells, self.chart, self.name = V, C, chart, name
        self.dim, self.ambient_dim = C.shape[1] - 1, V.shape[1]
        self._o = V[C[:, 0]]
        self._E = V[C[:, 1:]] - self._o[:, None, :]                     # (m, k, d)
        G = np.einsum("mid,mjd->mij", self._E, self._E)
        self._gram = np.sqrt(np.maximum(np.linalg.det(G), 0.0))         # k! times the flat volume of each cell
        scale = float(np.max(np.ptp(V, axis=0))) or 1.0
        if np.any(self._gram <= 1e-12 * scale ** self.dim):
            raise ValueError("degenerate cells (zero volume)")
        if chart is not None:
            if not (isinstance(chart, Sphere) and self.dim == 2 and self.ambient_dim == 3):
                raise NotImplementedError("charts are implemented for triangles projected radially onto a Sphere")
            c = np.array(chart.centre)
            n = np.cross(self._E[:, 0], self._E[:, 1])
            self._n = n / np.linalg.norm(n, axis=1)[:, None]
            if np.any(np.abs(np.sum((self._o - c) * self._n, axis=1)) <= 1e-12 * chart.radius):
                raise ValueError("a cell's plane passes through the centre of the sphere: its radial projection is "
                                 "not a surface piece")
        self._volume = None

    @property
    def cartesian_dim(self):
        return self.ambient_dim

    # ---------------------------------------------------------------- geometry
    def _map(self, j, U):
        """Points of cells j (q,) at reference coordinates U (q, k), flat."""
        return self._o[j] + np.einsum("qk,qkd->qd", U, self._E[j])

    def _project(self, X, j):
        """(points on the sphere, area factor) for flat points X of cells j (radial projection)."""
        c, R = np.array(self.chart.centre), self.chart.radius
        Y = X - c
        r = np.linalg.norm(Y, axis=1)
        return c + R * Y / r[:, None], R * R * np.abs(np.sum(Y * self._n[j], axis=1)) / r ** 3

    def charts(self):
        """The cells (sets/regions.Simplices): o_j + u E_j with scale k! |cell_j|, lifted radially onto the sphere with
        the chart's area factor (pygeoinf0.quadrature.rule: the collapsed Gauss-Jacobi rule on every cell)."""
        return (Simplices(self._o, self._E, self._gram, None if self.chart is None else self._project),)

    @property
    def volume(self):
        if self._volume is None:
            if self.chart is None:
                self._volume = float(np.sum(self._gram)) / math.factorial(self.dim)
            else:
                self._volume = float(np.sum(rule(self, 24)[1]))
        return self._volume

    @property
    def bounding_box(self):
        P = self.vertices if self.chart is None else np.vstack([self.chart_points(self.vertices),
                                                                rule(self, 6)[0]])
        return [(float(lo), float(hi)) for lo, hi in zip(P.min(axis=0), P.max(axis=0))]

    def chart_points(self, X):
        if self.chart is None:
            return X
        c, R = np.array(self.chart.centre), self.chart.radius
        return c + R * (X - c) / np.linalg.norm(X - c, axis=1)[:, None]

    def pieces(self):
        """The cells, each a one-cell Simplicial set (with the chart)."""
        return tuple(Simplicial(self.vertices[c], [list(range(self.dim + 1))], self.chart) for c in self.cells)

    def _locate(self, X):
        """(cell index, badness) per point: badness is the distance to the cell (flat) or to the radially projected
        cell (chart), relative to the diameter of the set."""
        X = np.atleast_2d(np.asarray(X, float))
        if self.chart is not None:                     # back to the cell planes along the rays from the centre
            c = np.array(self.chart.centre)
            on = np.abs(np.linalg.norm(X - c, axis=1) - self.chart.radius) / self.chart.radius
        best, idx = np.full(len(X), np.inf), np.zeros(len(X), int)
        scale = float(np.max(np.ptp(self.vertices, axis=0))) or 1.0
        for j in range(len(self.cells)):
            o, E = self._o[j], self._E[j]
            if self.chart is None:
                Y = X - o
            else:
                Y0 = X - c
                t = np.sum((o - c) * self._n[j]) / np.sum(Y0 * self._n[j], axis=1)
                Y = np.where((t > 0)[:, None], c + t[:, None] * Y0 - o, np.inf)
            lam, *_ = np.linalg.lstsq(E.T, np.where(np.isfinite(Y), Y, 0).T, rcond=None)
            lam = lam.T
            resid = np.linalg.norm(Y - lam @ E, axis=1)
            outside = np.maximum(np.maximum(-lam.min(axis=1), lam.sum(axis=1) - 1), 0.0)
            bad = np.where(np.isfinite(resid), (resid + outside * scale) / scale, np.inf)
            if self.chart is not None:
                bad = np.maximum(bad, on)
            better = bad < best
            best[better], idx[better] = bad[better], j
        return idx, best

    def locate(self, X):
        """The index of the cell containing each point (the nearest one)."""
        return self._locate(X)[0]

    def _inside_points(self, X):
        return self._locate(X)[1] <= ON

    def key(self):
        ck = None if self.chart is None else self.chart.key()
        return ("Simplicial", self.vertices.tobytes(), self.cells.tobytes(), ck)

    def same_up_to_null_set(self, other):
        return getattr(other, "key", lambda: None)() == self.key()

    def __eq__(self, other):
        return isinstance(other, Simplicial) and other.key() == self.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        if self.name:
            return self.name
        kind = {1: "segments", 2: "triangles", 3: "tetrahedra"}[self.dim]
        on = "" if self.chart is None else f" on {self.chart!r}"
        return f"Simplicial({len(self.cells)} {kind} in R^{self.ambient_dim}{on})"


def Simplex(*vertices):
    """The simplex with the given vertices (2 to 4 points of R^2 or R^3): a one-cell Simplicial set."""
    V = np.asarray(vertices[0] if len(vertices) == 1 else vertices, float)
    return Simplicial(V, [list(range(len(V)))])


# ------------------------------------------------------------------------------------------ closure


def _extreme_points(S):
    """Points whose convex hull contains S (exact), or None."""
    from .solids import Box
    from .flat import Flat, Polyline
    if isinstance(S, Simplicial) and S.chart is None:
        return S.vertices
    if isinstance(S, Polyline):
        return S._P
    if isinstance(S, Flat):
        return np.array([S._o + sum(c[k] * S._lens[k] * S._U[k] for k in range(S.dim)) for c in np.ndindex(*(2,) * S.dim)])
    if isinstance(S, Box):
        return np.array(np.meshgrid(*[(s.a, s.b) for s in S.sides], indexing="ij")).reshape(S.dim, -1).T
    return None


def _round(S):
    """(centre, radius) of a set contained in the ball B(centre, radius) and touching its sphere (exact extent)."""
    from .solids import Ball
    from .solids import Disk
    from .manifolds import Circle, Sphere
    if isinstance(S, (Ball, Disk, Sphere, Circle)):
        return np.array(S.centre, float), float(S.radius)
    return None


def inside_closure(S, D, tol=1e-10):
    """Whether S lies in the closure of the domain D, a Truth (plan 067 P1; was True, False or None): TRUE or FALSE
    where it is decided exactly (convex D with a polytope S; round S in a ball or disk; S with an exact bounding box in
    a box), UNKNOWN when it is not decided here (callers then sample)."""
    from .solids import Ball
    from .solids import Box
    from .solids import Cylinder
    from .solids import Disk
    P, rnd = _extreme_points(S), _round(S)
    if P is None and rnd is None:
        return Truth.UNKNOWN
    if isinstance(D, Box):
        sc = max(1.0, max(s.length for s in D.sides))
        return Truth.of(all(s.a - tol * sc <= lo and hi <= s.b + tol * sc
                            for s, (lo, hi) in zip(D.sides, S.bounding_box)))
    if isinstance(D, (Ball, Disk, Cylinder)):
        c = np.array(D.centre if not isinstance(D, Cylinder) else (*D.centre, 0.5 * (D.z.a + D.z.b)), float)
        if P is not None and P.shape[1] == len(c):           # convex D: the extreme points decide
            Y = c + (P - c) * (1 - tol)
            return Truth.of(bool(np.all(D.contains(Y))))
        if rnd is not None and isinstance(D, (Ball, Disk)) and len(rnd[0]) == len(c):
            return Truth.of(bool(np.linalg.norm(rnd[0] - c) + rnd[1] <= D.radius * (1 + tol)))
    return Truth.UNKNOWN
