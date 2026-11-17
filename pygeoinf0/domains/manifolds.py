"""Closed manifolds: the circle, the sphere and the flat torus.

    Circle(R)   a curve in R^2; natural coordinate phi; measure R dphi (arc length)
    Sphere(R)   a surface in R^3; natural coordinates (theta, phi), theta the colatitude; measure R^2 sin(theta) dtheta dphi
                (surface area)
    Torus(I, J) the flat torus I x J with opposite sides identified; natural coordinates (x, y); measure dx dy

L2 of a sphere is realised in spherical harmonics (realisations/series/spherical.py; L2(sphere, realisation="tensor") selects
the theta x phi tensor basis instead). Functions on the circle and the sphere take the Cartesian coordinates of the ambient space, f(x, y) and f(x, y, z), and
are evaluated only at points on the curve or surface (others give NaN). Functions on the torus take (x, y) and are
evaluated periodically. Every natural coordinate that is an angle (and both torus coordinates) is periodic, so the
tensor backend uses a Fourier basis along it.
"""
import math

import numpy as np

from .base import TWO_PI, Domain, Parametrisation, _fmt, _interval, scaled

ON = 1e-8       # relative distance from the curve or surface within which a Cartesian point counts as on it


class Circle(Domain):
    """The circle of the given radius and centre in R^2, with arc-length measure R dphi."""

    dim, ambient_dim = 1, 2
    periodic = (True,)
    internal_names = ("phi",)
    boundary_parts = ()                     # a closed manifold

    def __init__(self, radius=1.0, centre=(0.0, 0.0)):
        self.radius, self.centre = float(radius), tuple(float(c) for c in centre)
        if self.radius <= 0 or len(self.centre) != 2:
            raise ValueError("a circle needs a positive radius and a centre (cx, cy)")
        self.reference = ((0.0, TWO_PI),)
        self.jacobian = (scaled(self.radius, "1"),)

    def to_cartesian(self, phi):
        return self.centre[0] + self.radius * np.cos(phi), self.centre[1] + self.radius * np.sin(phi)

    def parametrisation(self):
        """The circle as a closed curve: gamma(phi) = centre + R (cos phi, sin phi), phi in [0, 2 pi], speed R."""
        c, R = np.array(self.centre), self.radius

        def gamma(t):
            t = np.asarray(t, float)[..., None]
            return c + R * np.concatenate([np.cos(t), np.sin(t)], axis=-1)

        def derivative(t):
            t = np.asarray(t, float)[..., None]
            return R * np.concatenate([-np.sin(t), np.cos(t)], axis=-1)
        return Parametrisation(gamma, derivative, 0.0, TWO_PI, (), True)

    def arc(self, phi0, phi1, name=None):
        """The arc phi0 <= phi <= phi1 of the circle (radians; it may cross phi = 0), a Curve: a set for measures on
        sub-arcs (sub-arc functionals in Sobolev spaces on the circle, EXP-032 finding 6)."""
        from .curves import Curve
        c, R = np.array(self.centre), self.radius
        return Curve(lambda t: c + R * np.stack([np.cos(t), np.sin(t)], axis=-1), phi0, phi1,
                     derivative=lambda t: R * np.stack([-np.sin(t), np.cos(t)], axis=-1),
                     name=name or f"arc [{_fmt(phi0)}, {_fmt(phi1)}] of {self!r}")

    def subpiece(self, lo, hi):
        """The arc lo[0] <= phi <= hi[0] (the natural coordinate; hi may exceed 2 pi: the arc crosses phi = 0)."""
        return self.arc(float(np.ravel(lo)[0]), float(np.ravel(hi)[0]))

    def from_cartesian(self, x, y):
        dx, dy = x - self.centre[0], y - self.centre[1]
        ok = np.abs(np.hypot(dx, dy) - self.radius) <= ON * self.radius
        return (np.where(ok, np.mod(np.arctan2(dy, dx), TWO_PI), np.nan),)

    def _inside(self, x, y):
        return np.abs(np.hypot(x - self.centre[0], y - self.centre[1]) - self.radius) <= ON * self.radius

    @property
    def bounding_box(self):
        return [(c - self.radius, c + self.radius) for c in self.centre]

    @property
    def volume(self):
        return TWO_PI * self.radius

    def key(self):
        return ("Circle", self.radius, self.centre)

    def __repr__(self):
        return f"Circle(radius={_fmt(self.radius)}, centre=({', '.join(_fmt(c) for c in self.centre)}))"


class Sphere(Domain):
    """The sphere of the given radius and centre in R^3, with surface measure R^2 sin(theta) dtheta dphi."""

    dim, ambient_dim = 2, 3
    periodic = (False, True)
    internal_names = ("theta", "phi")
    boundary_parts = ()                     # a closed manifold

    def __init__(self, radius=1.0, centre=(0.0, 0.0, 0.0)):
        self.radius, self.centre = float(radius), tuple(float(c) for c in centre)
        if self.radius <= 0 or len(self.centre) != 3:
            raise ValueError("a sphere needs a positive radius and a centre (cx, cy, cz)")
        self.reference = ((0.0, math.pi), (0.0, TWO_PI))
        self.jacobian = (scaled(self.radius ** 2, "sin"), "1")

    def to_cartesian(self, theta, phi):
        s, R = np.sin(theta), self.radius
        cx, cy, cz = self.centre
        return cx + R * s * np.cos(phi), cy + R * s * np.sin(phi), cz + R * np.cos(theta)

    def from_cartesian(self, x, y, z):
        dx, dy, dz = x - self.centre[0], y - self.centre[1], z - self.centre[2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        ok = np.abs(r - self.radius) <= ON * self.radius
        theta = np.arccos(np.clip(dz / np.where(r > 0, r, 1.0), -1, 1))
        phi = np.mod(np.arctan2(dy, dx), TWO_PI)
        return np.where(ok, theta, np.nan), np.where(ok, phi, np.nan)

    def band(self, theta0, theta1):
        """The latitude band theta0 <= theta <= theta1 (colatitude, radians; domains/bands.py)."""
        from .bands import SphereBand
        return SphereBand(self, theta0, theta1)

    def cap(self, theta, pole="north"):
        """The polar cap theta <= theta (pole="north") or theta >= theta (pole="south") about the z axis."""
        if pole not in ("north", "south"):
            raise ValueError("pole must be 'north' or 'south'")
        return self.band(0.0, theta) if pole == "north" else self.band(theta, math.pi)

    def from_angles(self, theta, phi):
        """Cartesian points of the sphere from colatitude and longitude (radians): a convenience for evaluation."""
        return self.to_cartesian(np.asarray(theta, float), np.asarray(phi, float))

    def _inside(self, x, y, z):
        r = np.sqrt((x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2 + (z - self.centre[2]) ** 2)
        return np.abs(r - self.radius) <= ON * self.radius

    @property
    def bounding_box(self):
        return [(c - self.radius, c + self.radius) for c in self.centre]

    @property
    def volume(self):
        return 4.0 * math.pi * self.radius ** 2

    def key(self):
        return ("Sphere", self.radius, self.centre)

    def __repr__(self):
        return f"Sphere(radius={_fmt(self.radius)}, centre=({', '.join(_fmt(c) for c in self.centre)}))"


class Torus(Domain):
    """The flat torus: a box with opposite sides identified, Torus("[0, 2 pi]", "[0, 2 pi]"). Measure dx dy (dz)."""

    boundary_parts = ()                     # a closed manifold

    def __init__(self, *sides):
        if len(sides) not in (2, 3):
            raise ValueError("a flat torus here is 2- or 3-dimensional: give 2 or 3 intervals (periods)")
        self.sides = tuple(_interval(s) for s in sides)
        if not all(s.is_bounded and s.length > 0 for s in self.sides):
            raise ValueError("the sides of a torus must be bounded intervals of positive length")
        self.dim = len(self.sides)
        self.reference = tuple((s.a, s.b) for s in self.sides)
        self.jacobian = ("1",) * self.dim
        self.periodic = (True,) * self.dim
        self.internal_names = ("x", "y", "z")[: self.dim]

    def to_cartesian(self, *xi):
        return tuple(xi)

    def from_cartesian(self, *x):
        return tuple(s.a + np.mod(np.asarray(a, float) - s.a, s.length) for s, a in zip(self.sides, x))

    def _inside(self, *x):
        return np.ones(np.broadcast(*x).shape, bool) if np.ndim(x[0]) else True

    @property
    def bounding_box(self):
        return [(s.a, s.b) for s in self.sides]

    @property
    def volume(self):
        return float(np.prod([s.length for s in self.sides]))

    def key(self):
        return ("Torus",) + tuple((s.a, s.b) for s in self.sides)

    def __repr__(self):
        return "Torus(" + ", ".join(f"[{_fmt(s.a)}, {_fmt(s.b)})" for s in self.sides) + ")"
