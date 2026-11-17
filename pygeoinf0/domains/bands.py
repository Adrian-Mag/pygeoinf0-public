"""Latitude bands and polar caps of a sphere (plan 035 P5, design D5's sphere row): the pieces of a sphere whose tiling
is decided exactly.

    S = pg.Sphere(1.0)
    S.band(theta0, theta1)        {theta0 <= theta <= theta1} (colatitude about the sphere's own z axis, radians)
    S.cap(theta1)                 the north polar cap theta <= theta1; S.cap(theta0, pole="south") theta >= theta0
    pg.Partition(S, {"north": S.cap(0.6), "belt": S.band(0.6, 2.0), "south": S.cap(2.0, pole="south")})

A band is a Domain with natural coordinates (theta, phi) on the reference box [theta0, theta1] x [0, 2 pi) and the
measure R^2 sin(theta) dtheta dphi (surface area), so L2 of a band is realised in the theta x phi tensor basis
(Legendre in theta, Fourier in phi; spherical harmonics are the whole sphere's). Membership is decided from the numbers
(the colatitude of the point), so a partition of a sphere into bands about its axis is decided exactly; caps about
other axes and other shapes (spherical polygons, Voronoi cells) are not constructed and partitions of them are refused.
The seams of such a partition are the parallels theta = const; side="below" is the piece of smaller colatitude (nearer
the north pole theta = 0), "above" the other, as for intervals.
"""
import math

import numpy as np

from .base import TWO_PI, Domain, _fmt, scaled
from .manifolds import Sphere

TOL = 1e-12          # in radians: colatitudes closer than this are equal


def colatitudes(S):
    """(sphere, theta0, theta1) of a Sphere (0, pi) or a SphereBand, else None."""
    if isinstance(S, SphereBand):
        return S.sphere, S.theta0, S.theta1
    if isinstance(S, Sphere):
        return S, 0.0, math.pi
    return None


def same_sphere(A, B):
    return abs(A.radius - B.radius) <= TOL * max(A.radius, B.radius) and \
        float(np.linalg.norm(np.subtract(A.centre, B.centre))) <= TOL * max(A.radius, B.radius)


class SphereBand(Domain):
    """{x on the sphere: theta0 <= theta(x) <= theta1}, theta the colatitude about the sphere's z axis (module
    docstring); theta0 = 0 is a north cap, theta1 = pi a south cap."""

    dim, ambient_dim = 2, 3
    periodic = (False, True)
    internal_names = ("theta", "phi")
    boundary_parts = ()

    def __init__(self, sphere, theta0, theta1):
        t0, t1 = float(theta0), float(theta1)
        if not (0.0 <= t0 < t1 <= math.pi):
            raise ValueError(f"a band needs 0 <= theta0 < theta1 <= pi (radians), not {t0:g}, {t1:g}")
        if t0 <= TOL and t1 >= math.pi - TOL:
            raise ValueError("the band 0 <= theta <= pi is the whole sphere: use the Sphere itself")
        self.sphere, self.theta0, self.theta1 = sphere, (0.0 if t0 <= TOL else t0), (math.pi if t1 >= math.pi - TOL
                                                                                      else t1)
        self.radius, self.centre = sphere.radius, sphere.centre
        self.reference = ((self.theta0, self.theta1), (0.0, TWO_PI))
        self.jacobian = (scaled(self.radius ** 2, "sin"), "1")

    def to_cartesian(self, theta, phi):
        return self.sphere.to_cartesian(theta, phi)

    def from_cartesian(self, x, y, z):
        theta, phi = self.sphere.from_cartesian(x, y, z)
        ok = (theta >= self.theta0 - TOL) & (theta <= self.theta1 + TOL)
        return np.where(ok, theta, np.nan), np.where(ok, phi, np.nan)

    def colatitude(self, *x):
        """The colatitude of points on the sphere (NaN off it)."""
        return self.sphere.from_cartesian(*[np.asarray(a, float) for a in x])[0]

    def _inside(self, x, y, z):
        theta = self.colatitude(x, y, z)
        with np.errstate(invalid="ignore"):
            return np.isfinite(theta) & (theta >= self.theta0 - TOL) & (theta <= self.theta1 + TOL)

    _inside_closed = _inside

    @property
    def is_cap(self):
        return self.theta0 == 0.0 or self.theta1 == math.pi

    @property
    def bounding_box(self):
        R, (cx, cy, cz) = self.radius, self.centre
        s = 1.0 if self.theta0 <= math.pi / 2 <= self.theta1 else max(math.sin(self.theta0), math.sin(self.theta1))
        return [(cx - R * s, cx + R * s), (cy - R * s, cy + R * s),
                (cz + R * math.cos(self.theta1), cz + R * math.cos(self.theta0))]

    @property
    def volume(self):
        """Its area 2 pi R^2 (cos theta0 - cos theta1)."""
        return TWO_PI * self.radius ** 2 * (math.cos(self.theta0) - math.cos(self.theta1))

    def key(self):
        return ("SphereBand", self.radius, self.centre, self.theta0, self.theta1)

    def __repr__(self):
        what = "north cap" if self.theta0 == 0 else "south cap" if self.theta1 == math.pi else "band"
        return f"{what} {_fmt(self.theta0)} <= theta <= {_fmt(self.theta1)} of {self.sphere!r}"


def meet_bands(S, D):
    """S cap D for spheres and bands of one sphere: a band, the sphere, or None (null); Unsupported for other
    spheres (their intersection is not constructed here)."""
    from ..core.errors import Unsupported
    (A, a0, a1), (B, b0, b1) = colatitudes(S), colatitudes(D)
    if not same_sphere(A, B):
        raise Unsupported(f"{S!r} cap {D!r}: bands and caps of different spheres are not intersected here")
    lo, hi = max(a0, b0), min(a1, b1)
    if hi - lo <= TOL:
        return None
    if lo == a0 and hi == a1:
        return S
    if lo == b0 and hi == b1:
        return D
    return SphereBand(A, lo, hi)


def point_side(D, x, tol=1e-10):
    """Where the point x lies relative to the closed band D: "in", "out", "lower" (on its boundary theta = theta0 > 0)
    or "upper" (on theta = theta1 < pi)."""
    theta = float(D.colatitude(*np.atleast_1d(np.asarray(x, float))))
    if not np.isfinite(theta):
        return "out"
    if D.theta0 > 0 and abs(theta - D.theta0) <= tol:
        return "lower"
    if D.theta1 < math.pi and abs(theta - D.theta1) <= tol:
        return "upper"
    return "in" if D.theta0 - tol <= theta <= D.theta1 + tol else "out"          # a pole is inside its cap


def cut_arc(m, D, side=None):
    """The arc-length measure m of a great-circle Arc restricted to the band D (or the sphere): the sub-arcs inside D,
    cut where the arc crosses D's parallels (closed form: along the arc z = R rho cos(t - psi)), a list of measures;
    an arc lying on a parallel of D's boundary (the equator) needs side=, like a point there."""
    from ..core.errors import Unsupported
    from ..measures import LebesgueMeasure
    from ..measures.restrict import point_in
    A = m.set
    if not same_sphere(A.sphere, colatitudes(D)[0]):
        raise Unsupported(f"restricting {m} to {D!r}: the arc lies on another sphere")
    if colatitudes(D)[1:] == (0.0, math.pi):
        return [m]
    az, ez = float(A._a[2]), float(A._e[2])
    rho, psi = math.hypot(az, ez), math.atan2(ez, az)
    if rho <= TOL:                                         # the arc lies on the equator
        return [m] if point_in(D, np.array(A.p), side) else []
    cuts = {0.0, A.angle}
    for th in (D.theta0, D.theta1):
        c = math.cos(th) / rho
        if 0 < th < math.pi and abs(c) < 1 - 1e-15:
            for t in (psi + math.acos(c), psi - math.acos(c)):
                t = math.fmod(t, 2 * math.pi) % (2 * math.pi)
                if TOL < t < A.angle - TOL:
                    cuts.add(t)
    t = sorted(cuts)
    runs, R = [], A._R
    for a, b in zip(t[:-1], t[1:]):
        mid = np.array(A.to_cartesian(np.array(0.5 * (a + b) * R))).ravel()
        if point_side(D, mid) == "in":
            if runs and abs(runs[-1][1] - a) <= TOL:
                runs[-1][1] = b
            else:
                runs.append([a, b])
    if len(runs) == 1 and runs[0] == [0.0, A.angle]:
        return [m]
    return [LebesgueMeasure(A.subpiece(a * R, b * R)) for a, b in runs]
