"""Box: a rectangle or a 3-D box with Cartesian (natural) coordinates; Lebesgue measure dx dy (dz).

From ball.py (merged here by plan 063):
Ball: natural coordinates (r, theta, phi), theta the colatitude; Lebesgue measure r^2 sin(theta) dr dtheta dphi.

From cylinder.py (merged here by plan 063):
Cylinder (vertical axis): natural coordinates (r, phi, z); Lebesgue measure r dr dphi dz.

From disk.py (merged here by plan 063):
Disk: natural coordinates (r, phi); Lebesgue measure r dr dphi.
"""


import math

import numpy as np

from ..sets import Interval, Set
from .base import TWO_PI, Domain, _fmt, _interval


class Box(Domain):
    """A box: Box("[0, 1]", "[0, 2]") (a rectangle), Box((0, 1), (0, 1), (0, 0.5)) (3D), or I * J (* K).
    Natural coordinates are the Cartesian ones; Lebesgue measure is dx dy (dz)."""

    def __init__(self, *sides):
        if len(sides) == 1 and isinstance(sides[0], (list, tuple)) and len(sides[0]) in (2, 3) \
                and not isinstance(sides[0][0], (int, float)):
            sides = tuple(sides[0])
        if len(sides) not in (2, 3):
            raise ValueError("a box here is 2- or 3-dimensional: give 2 or 3 intervals")
        self.sides = tuple(_interval(s) for s in sides)
        for s in self.sides:
            if not s.is_bounded:
                raise ValueError("box sides must be bounded")
        self.dim = len(self.sides)
        self.reference = tuple((s.a, s.b) for s in self.sides)
        self.jacobian = ("1",) * self.dim
        self.periodic = (False,) * self.dim
        self.internal_names = ("x", "y", "z")[: self.dim]
        self.boundary_parts = tuple(f"{a}{e}" for a in self.internal_names for e in "-+")    # faces x = a, x = b, ...

    def to_cartesian(self, *xi):
        return tuple(xi)

    def from_cartesian(self, *x):
        inside = self._inside_closed(*x)
        return tuple(np.where(inside, a, np.nan) for a in x)

    def _inside(self, *x):
        r = True
        for s, a in zip(self.sides, x):
            r = r & s.contains(np.asarray(a, float)[..., None])
        return r

    def _inside_closed(self, *x):
        r = True
        for s, a in zip(self.sides, x):
            r = r & (a >= s.a) & (a <= s.b)
        return r

    @property
    def bounding_box(self):
        return [(s.a, s.b) for s in self.sides]

    @property
    def volume(self):
        return float(np.prod([s.length for s in self.sides]))

    def __mul__(self, other):
        if isinstance(other, (Interval, str)) and self.dim == 2:
            return Box(*self.sides, _interval(other))
        return NotImplemented

    def key(self):
        return ("Box",) + tuple((s.a, s.b) for s in self.sides)

    def __repr__(self):
        return " × ".join(repr(s) for s in self.sides)


# ---------------------------------------------------------------- from ball.py (plan 063)
class Ball(Domain):
    """The closed ball of the given radius and centre. Natural coordinates (r, theta, phi) with theta the
    colatitude; Lebesgue measure r^2 sin(theta) dr dtheta dphi. (Smooth Cartesian functions are analytic in
    (r, theta, phi); in mu = cos(theta) they would have square-root singularities at the poles.)"""

    dim = 3
    periodic = (False, False, True)
    internal_names = ("r", "theta", "phi")

    def __init__(self, radius=1.0, centre=(0.0, 0.0, 0.0)):
        if isinstance(radius, Set):
            raise TypeError(f"pg.Ball(radius, centre) is the 3-D ball domain, not a ball of {radius!r}: the closed ball "
                            "{x : ||x - c|| <= r} of a space is pg.NormBall(space, r, centre=c) (plan 067 P3)")
        self.radius, self.centre = float(radius), tuple(float(c) for c in centre)
        if self.radius <= 0 or len(self.centre) != 3:
            raise ValueError("a ball needs a positive radius and a centre (cx, cy, cz)")
        self.reference = ((0.0, self.radius), (0.0, math.pi), (0.0, TWO_PI))
        self.jacobian = ("r2", "sin", "1")

    def to_cartesian(self, r, theta, phi):
        s = np.sin(theta)
        cx, cy, cz = self.centre
        return cx + r * s * np.cos(phi), cy + r * s * np.sin(phi), cz + r * np.cos(theta)

    def from_cartesian(self, x, y, z):
        dx, dy, dz = x - self.centre[0], y - self.centre[1], z - self.centre[2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        ok = r <= self.radius * (1 + 1e-14)
        mu = np.where(r > 0, dz / np.where(r > 0, r, 1.0), 1.0)
        phi = np.mod(np.arctan2(dy, dx), TWO_PI)
        return (np.where(ok, np.minimum(r, self.radius), np.nan),
                np.where(ok, np.arccos(np.clip(mu, -1, 1)), np.nan), np.where(ok, phi, np.nan))

    def _inside(self, x, y, z):
        return (x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2 + (z - self.centre[2]) ** 2 <= self.radius ** 2

    @property
    def bounding_box(self):
        return [(c - self.radius, c + self.radius) for c in self.centre]

    @property
    def volume(self):
        return 4.0 / 3.0 * math.pi * self.radius ** 3

    def key(self):
        return ("Ball", self.radius, self.centre)

    def __repr__(self):
        return f"Ball(radius={_fmt(self.radius)}, centre=({', '.join(_fmt(c) for c in self.centre)}))"


# ---------------------------------------------------------------- from cylinder.py (plan 063)
class Cylinder(Domain):
    """A solid circular cylinder with axis parallel to z: Cylinder(radius, z=(z0, z1), centre=(cx, cy)).
    Natural coordinates (r, phi, z); measure r dr dphi dz."""

    dim = 3
    periodic = (False, True, False)
    internal_names = ("r", "phi", "z")

    def __init__(self, radius=1.0, z=(0.0, 1.0), centre=(0.0, 0.0)):
        self.radius, self.centre = float(radius), tuple(float(c) for c in centre)
        self.z = _interval(z)
        if self.radius <= 0 or len(self.centre) != 2:
            raise ValueError("a cylinder needs a positive radius, a z-interval and an axis position (cx, cy)")
        self.reference = ((0.0, self.radius), (0.0, TWO_PI), (self.z.a, self.z.b))
        self.jacobian = ("r", "1", "1")

    def to_cartesian(self, r, phi, z):
        return self.centre[0] + r * np.cos(phi), self.centre[1] + r * np.sin(phi), z + 0.0 * r

    def from_cartesian(self, x, y, z):
        dx, dy = x - self.centre[0], y - self.centre[1]
        r = np.hypot(dx, dy)
        phi = np.mod(np.arctan2(dy, dx), TWO_PI)
        ok = (r <= self.radius * (1 + 1e-14)) & (z >= self.z.a) & (z <= self.z.b)
        return (np.where(ok, np.minimum(r, self.radius), np.nan), np.where(ok, phi, np.nan),
                np.where(ok, z, np.nan))

    def _inside(self, x, y, z):
        return (np.hypot(x - self.centre[0], y - self.centre[1]) <= self.radius) & self.z.contains(np.asarray(z, float)[..., None])

    @property
    def bounding_box(self):
        return [(c - self.radius, c + self.radius) for c in self.centre] + [(self.z.a, self.z.b)]

    @property
    def volume(self):
        return math.pi * self.radius ** 2 * self.z.length

    def key(self):
        return ("Cylinder", self.radius, self.centre, self.z.a, self.z.b)

    def __repr__(self):
        return (f"Cylinder(radius={_fmt(self.radius)}, z=[{_fmt(self.z.a)}, {_fmt(self.z.b)}], "
                f"centre=({', '.join(_fmt(c) for c in self.centre)}))")


# ---------------------------------------------------------------- from disk.py (plan 063)
class Disk(Domain):
    """The closed disk of the given radius and centre. Natural coordinates (r, phi); measure r dr dphi."""

    dim = 2
    periodic = (False, True)
    internal_names = ("r", "phi")

    def __init__(self, radius=1.0, centre=(0.0, 0.0)):
        self.radius, self.centre = float(radius), tuple(float(c) for c in centre)
        if self.radius <= 0 or len(self.centre) != 2:
            raise ValueError("a disk needs a positive radius and a centre (cx, cy)")
        self.reference = ((0.0, self.radius), (0.0, TWO_PI))
        self.jacobian = ("r", "1")

    def to_cartesian(self, r, phi):
        return self.centre[0] + r * np.cos(phi), self.centre[1] + r * np.sin(phi)

    def from_cartesian(self, x, y):
        dx, dy = x - self.centre[0], y - self.centre[1]
        r = np.hypot(dx, dy)
        phi = np.mod(np.arctan2(dy, dx), TWO_PI)
        ok = r <= self.radius * (1 + 1e-14)
        return np.where(ok, np.minimum(r, self.radius), np.nan), np.where(ok, phi, np.nan)

    def _inside(self, x, y):
        return np.hypot(x - self.centre[0], y - self.centre[1]) <= self.radius

    @property
    def bounding_box(self):
        return [(c - self.radius, c + self.radius) for c in self.centre]

    @property
    def volume(self):
        return math.pi * self.radius ** 2

    def key(self):
        return ("Disk", self.radius, self.centre)

    def __repr__(self):
        return f"Disk(radius={_fmt(self.radius)}, centre=({', '.join(_fmt(c) for c in self.centre)}))"
