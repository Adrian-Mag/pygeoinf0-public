"""Shells and layered balls.

    Shell(inner, outer)               {inner <= |x - c| <= outer}: natural coordinates (r, theta, phi), theta the
                                      colatitude; Lebesgue measure r^2 sin(theta) dr dtheta dphi (the ball's, no centre)
    LayeredBall(radii, names=...)     the ball of radius radii[-1] cut by the spheres of radii[:-1]: layer 0 is a Ball,
                                      the others Shells; interface j is the Sphere between layers j and j + 1

A layered ball is a partition of a Ball, not a new domain: its layers carry the spaces (one per layer, summands of a
direct sum: pg.PiecewiseSpace, spaces/piecewise.py), and it answers the geometric questions that need all of them:

    P.layer_of(points, side=None)     which layer holds each point (-1 outside the ball). A point on an interface
                                      belongs to two closed layers: say side="below" or side="above", or it is refused
    P.split(mu)                       the restrictions of a measure to the closed layers that it charges: {name: mu_k},
                                      sum_k mu_k = mu (interfaces are null sets for points off them, for paths and for
                                      volumes; volume and density measures go to the exact sets S cap layer)
    P.crossings(polyline)             the points where a polyline meets an interface, with the side and the incidence
                                      (|cos| of the angle to the normal) of the pieces before and after each one

A polyline never lies in an interface: a chord between two points of a sphere passes strictly inside it. So once a
polyline is cut where it crosses the interfaces, every piece lies in one closed layer, determined by its interior, and
paths need no side. Points (evaluation, Dirac measures) on an interface do.
"""
import collections
import math

import numpy as np

from ..sets import Region
from .solids import Ball
from .base import TWO_PI, Domain, _fmt
from .manifolds import Sphere

ON = 1e-9           # distance from a sphere, relative to the outer radius, within which a point counts as on it

Crossing = collections.namedtuple("Crossing", "interface index point vertex sides cosines")
Crossing.__doc__ = """A point where a polyline meets an interface: the interface's name and position, the point, its
vertex in the cut polyline (LayeredBall.cut), the sides ("below"/"above"; None at an end of the path) of the pieces
before and after it, and their incidence |n . t| (n the outward normal, t the piece's direction)."""


class Shell(Domain):
    """The closed spherical shell inner <= |x - centre| <= outer (0 < inner < outer); natural coordinates
    (r, theta, phi) with theta the colatitude, Lebesgue measure r^2 sin(theta) dr dtheta dphi. Points within ON times the
    outer radius of either sphere count as on it."""

    dim = 3
    periodic = (False, False, True)
    internal_names = ("r", "theta", "phi")
    boundary_parts = ("inner", "outer")     # the named parts of the boundary (spaces/constructions.py)

    def __init__(self, inner, outer, centre=(0.0, 0.0, 0.0)):
        self.inner, self.outer = float(inner), float(outer)
        self.centre = tuple(float(c) for c in centre)
        if not 0 < self.inner < self.outer or len(self.centre) != 3:
            raise ValueError("a shell needs radii 0 < inner < outer and a centre (cx, cy, cz)")
        self.reference = ((self.inner, self.outer), (0.0, math.pi), (0.0, TWO_PI))
        self.jacobian = ("r2", "sin", "1")

    def to_cartesian(self, r, theta, phi):
        s = np.sin(theta)
        cx, cy, cz = self.centre
        return cx + r * s * np.cos(phi), cy + r * s * np.sin(phi), cz + r * np.cos(theta)

    def from_cartesian(self, x, y, z):
        dx, dy, dz = x - self.centre[0], y - self.centre[1], z - self.centre[2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        tol = ON * self.outer
        ok = (r >= self.inner - tol) & (r <= self.outer + tol)
        phi = np.mod(np.arctan2(dy, dx), TWO_PI)
        theta = np.arccos(np.clip(dz / np.where(r > 0, r, 1.0), -1, 1))
        return (np.where(ok, np.clip(r, self.inner, self.outer), np.nan), np.where(ok, theta, np.nan),
                np.where(ok, phi, np.nan))

    def _inside(self, x, y, z):
        r = np.sqrt((x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2 + (z - self.centre[2]) ** 2)
        tol = ON * self.outer
        return (r >= self.inner - tol) & (r <= self.outer + tol)

    @property
    def inner_sphere(self):
        return Sphere(self.inner, self.centre)

    @property
    def outer_sphere(self):
        return Sphere(self.outer, self.centre)

    @property
    def bounding_box(self):
        return [(c - self.outer, c + self.outer) for c in self.centre]

    @property
    def volume(self):
        return 4.0 / 3.0 * math.pi * (self.outer ** 3 - self.inner ** 3)

    def key(self):
        return ("Shell", self.inner, self.outer, self.centre)

    def __repr__(self):
        c = "" if self.centre == (0.0, 0.0, 0.0) else f", centre=({', '.join(_fmt(v) for v in self.centre)})"
        return f"Shell({_fmt(self.inner)}, {_fmt(self.outer)}{c})"


class LayeredBall(Region):
    """The ball of radius radii[-1] cut by concentric spheres at radii[:-1] (increasing). names: one per layer, from
    the centre out (default "layer 0", ...); interfaces: one name per interface (default "lower/upper").
    As a set (a Region, plan 067 P2) it is the ball: its membership, dimension and volume are the ball's."""

    dim = 3

    def __init__(self, radii, names=None, interfaces=None, centre=(0.0, 0.0, 0.0)):
        r = np.asarray(radii, float).ravel()
        if len(r) < 1 or r[0] <= 0 or np.any(np.diff(r) <= 0):
            raise ValueError("the radii must be positive and increasing (the last one is the outer radius)")
        self.radii, self.centre = tuple(float(x) for x in r), tuple(float(c) for c in centre)
        names = [f"layer {k}" for k in range(len(r))] if names is None else list(names)
        if len(names) != len(r) or len(set(names)) != len(names):
            raise ValueError(f"{len(r)} layers need {len(r)} distinct names")
        inames = [f"{a}/{b}" for a, b in zip(names[:-1], names[1:])] if interfaces is None else list(interfaces)
        if len(inames) != len(r) - 1 or len(set(inames) | set(names)) != len(inames) + len(names):
            raise ValueError(f"{len(r) - 1} interfaces need {len(r) - 1} distinct names, different from the layers'")
        self.names, self.interface_names = tuple(names), tuple(inames)
        self.ball = Ball(r[-1], self.centre)
        self.layers = (Ball(r[0], self.centre),) + tuple(Shell(a, b, self.centre) for a, b in zip(r[:-1], r[1:]))
        self.interfaces = tuple(Sphere(x, self.centre) for x in r[:-1])
        self.surface = Sphere(r[-1], self.centre)
        self.tol = ON * r[-1]
        from .partition import Partition
        self._partition = Partition(self.ball, dict(zip(self.names, self.layers)))      # plan 035 P3

    # ---------------------------------------------------------------- structure
    def index(self, key):
        """The position of a layer, by name or position."""
        if key in self.names:
            return self.names.index(key)
        if isinstance(key, (int, np.integer)) and -len(self.names) <= key < len(self.names):
            return int(key) % len(self.names)
        raise KeyError(f"{key!r} is not a layer of {self} (layers: {', '.join(self.names)})")

    def layer(self, key):
        return self.layers[self.index(key)]

    @property
    def partition(self):
        """The radial pg.Partition of the ball into the layers (by name; plan 035 P3): checked to tile it, with the
        interfaces as its seams. split and the rest of this class keep their own rules (side= as layer_of)."""
        return self._partition

    def __getitem__(self, key):
        return self.layer(key)

    def interface_index(self, key):
        """The position of an interface, by name, position, or the pair of layer names (lower, upper)."""
        if key in self.interface_names:
            return self.interface_names.index(key)
        if isinstance(key, tuple) and len(key) == 2:
            a, b = self.index(key[0]), self.index(key[1])
            if b == a + 1:
                return a
        if isinstance(key, (int, np.integer)) and -len(self.interfaces) <= key < len(self.interfaces):
            return int(key) % len(self.interfaces)
        raise KeyError(f"{key!r} is not an interface of {self} (interfaces: {', '.join(self.interface_names)})")

    def interface(self, key):
        """The sphere between two adjacent layers (by interface name, position, or (lower, upper) layer names)."""
        return self.interfaces[self.interface_index(key)]

    # ---------------------------------------------------------------- points
    def _radius(self, X):
        return np.linalg.norm(np.asarray(X, float) - np.array(self.centre), axis=-1)

    def layer_of(self, points, side=None):
        """The layer of each point ((m, 3) or (3,)): -1 outside the ball. A point within self.tol of an interface is
        on it and belongs to the layers on both sides: side="below" or "above" picks one, None refuses."""
        if side not in (None, "below", "above"):
            raise ValueError("side must be 'below', 'above' or None")
        r = self._radius(points)
        R = np.asarray(self.radii)
        k = np.searchsorted(R, r, side="left")
        near = np.abs(r[..., None] - R[:-1]) <= self.tol if len(R) > 1 else np.zeros(r.shape + (0,), bool)
        on = near.any(axis=-1)
        if np.any(on):
            if side is None:
                j = int(np.argmax(near[on][0]))
                raise ValueError(f"a point is on the interface {self.interface_names[j]!r} (radius {R[j]:g}), where a "
                                 "field may have two values: say side='below' or side='above'")
            j = np.argmax(near, axis=-1)
            k = np.where(on, j + (side == "above"), k)
        k = np.where(r > R[-1] + self.tol, -1, np.minimum(k, len(R) - 1))
        return int(k) if np.ndim(k) == 0 else k

    # ---------------------------------------------------------------- paths
    def cut(self, points, weights=None):
        """The polyline through points with a vertex added wherever a piece crosses an interface (each new piece keeps
        its parent's weight): (points, weights, layer of each piece). Pieces shorter than ON times the outer radius
        (rounding at an existing vertex) are not created."""
        X, w, rm = cut_at_spheres(points, weights, self.centre, self.radii[:-1], self.tol)
        R = np.asarray(self.radii)
        lay = np.minimum(np.searchsorted(R, rm, side="left"), len(R) - 1)
        touch = np.abs(rm[:, None] - R[:-1]) <= self.tol if len(R) > 1 else np.zeros((len(rm), 0), bool)
        if touch.any():                  # a piece touching an interface at its middle lies above it (radius is convex)
            lay = np.where(touch.any(axis=1), np.argmax(touch, axis=1) + 1, lay)
        return X, w, lay

    def split(self, mu, side=None):
        """The restrictions of mu to the closed layers it charges, {layer name: measure}; they add up to mu. mu: a Dirac
        measure or a point (side as in layer_of), arc length on a Segment or Polyline (or the set itself), a piecewise
        density on a Polyline, a volume, density or bank measure (f.measure()): restricted to each layer on the exact set
        S cap layer (mu.restricted, domains/pieces.py), a sphere measure (side= on an interface), or a signed
        combination of these. A layer charged by several separate pieces of a path gets their sum (a SignedMeasure with
        unit coefficients)."""
        from ..measures import DiracMeasure, LebesgueMeasure, PieceDensity, SignedMeasure, terms_of
        from .flat import Flat, Polyline
        parts = {}
        for c, m in terms_of(mu if not isinstance(mu, Region) else LebesgueMeasure(mu)):
            if not isinstance(m, (DiracMeasure, LebesgueMeasure, PieceDensity)) and isinstance(m, Region):
                m = LebesgueMeasure(m)
            if isinstance(m, DiracMeasure) or (not hasattr(m, "set") and np.ndim(m) == 1):
                m = m if isinstance(m, DiracMeasure) else DiracMeasure(m)
                k = self.layer_of(np.asarray(m.x0, float), side)
                if k < 0:
                    raise ValueError(f"{m} is outside {self.ball}")
                parts.setdefault(self.names[k], []).append((c, m))
                continue
            S = m.set
            if isinstance(S, Flat) and S.dim == 1 and isinstance(m, LebesgueMeasure):
                pts, w, weighted = np.array(S.endpoints), None, False
            elif isinstance(S, Polyline) and isinstance(m, (LebesgueMeasure, PieceDensity)):
                pts, w, weighted = S._P, getattr(m, "weights", None), isinstance(m, PieceDensity)
            else:                                    # volumes, densities, spheres: mu restricted to each layer
                for name, layer in zip(self.names, self.layers):
                    piece = m.restricted(layer, side)
                    if piece is not None:
                        parts.setdefault(name, []).append((c, piece))
                continue
            if np.any(self._radius(pts) > self.radii[-1] + self.tol):
                raise ValueError(f"{m} leaves {self.ball}: clip it first")
            X, wx, lay = self.cut(pts, w)
            starts = np.r_[0, np.flatnonzero(np.diff(lay)) + 1]
            for a, b in zip(starts, np.r_[starts[1:], len(lay)]):
                run = Polyline(X[a:b + 1])
                piece = LebesgueMeasure(run).with_density(wx[a:b]) if weighted else LebesgueMeasure(run)
                parts.setdefault(self.names[lay[a]], []).append((c, piece))
        return {name: (t[0][1] if len(t) == 1 and t[0][0] == 1.0 else SignedMeasure(t))
                for name, t in sorted(parts.items(), key=lambda kv: self.index(kv[0]))}

    def crossings(self, points):
        """The crossings of the polyline through points (or a Polyline) with the interfaces, in path order (see
        Crossing). Vertices are those of cut(points)."""
        P = np.asarray(getattr(points, "_P", points), float)
        X, _, lay = self.cut(P)
        r = self._radius(X)
        out = []
        for i in range(len(X)):
            near = np.flatnonzero(np.abs(r[i] - np.asarray(self.radii[:-1])) <= self.tol)
            if not len(near):
                continue
            j = int(near[0])
            n = (X[i] - np.array(self.centre)) / r[i]
            sides, cos = [], []
            for s in (i - 1, i):                                 # the pieces before and after the vertex
                if 0 <= s < len(lay):
                    t = X[s + 1] - X[s]
                    sides.append("below" if lay[s] <= j else "above")
                    cos.append(abs(float(n @ t)) / float(np.linalg.norm(t)))
                else:
                    sides.append(None)
                    cos.append(None)
            out.append(Crossing(self.interface_names[j], j, X[i].copy(), i, tuple(sides), tuple(cos)))
        return out

    # ---------------------------------------------------------------- the set
    def _inside(self, x, y, z):
        return self.ball._inside(x, y, z)

    @property
    def volume(self):
        return self.ball.volume

    @property
    def bounding_box(self):
        return self.ball.bounding_box

    # ---------------------------------------------------------------- identity
    def __eq__(self, other):
        return isinstance(other, LayeredBall) and (self.radii, self.names, self.interface_names, self.centre) == \
            (other.radii, other.names, other.interface_names, other.centre)

    def __hash__(self):
        return hash(("LayeredBall", self.radii, self.names, self.centre))

    def __repr__(self):
        parts = ", ".join(f"{n} {_fmt(a)}-{_fmt(b)}" for n, a, b in zip(self.names, (0.0,) + self.radii[:-1], self.radii))
        return f"LayeredBall({parts})"


def cut_at_spheres(points, weights, centre, radii, tol):
    """The polyline through points with a vertex added wherever a piece crosses one of the spheres |x - centre| = r,
    r in radii (each new piece keeps its parent's weight): (points, weights, radius of each piece's midpoint). Pieces
    shorter than tol (rounding at an existing vertex) are not created."""
    P = np.asarray(points, float)
    w = np.ones(len(P) - 1) if weights is None else np.asarray(weights, float)
    c = np.array(centre, float)
    out, wout = [P[0]], []
    for j in range(len(P) - 1):
        p, d = P[j] - c, P[j + 1] - P[j]
        a, b, q = d @ d, p @ d, p @ p
        ts = []
        for rj in radii:
            disc = b * b - a * (q - rj * rj)
            if disc > 0:
                s = math.sqrt(disc)
                ts += [t for t in ((-b - s) / a, (-b + s) / a) if 0 < t < 1]
        L = math.sqrt(a)
        for t in sorted(ts):
            if t * L > tol and (1 - t) * L > tol and np.linalg.norm(P[j] + t * d - out[-1]) > tol:
                out.append(P[j] + t * d)
                wout.append(w[j])
        out.append(P[j + 1])
        wout.append(w[j])
    X = np.array(out)
    return X, np.array(wout), np.linalg.norm(0.5 * (X[1:] + X[:-1]) - c, axis=-1)
