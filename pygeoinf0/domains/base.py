"""Domain: the common interface (natural coordinates on a reference box, product Lebesgue measure), the per-axis
Jacobians of Lebesgue measure (JACOBIANS) and small helpers shared by the concrete domains. A domain keeps its chart,
Jacobian and measure; the rules that integrate over it are pygeoinf0.quadrature's (rule(D, n), through D.charts();
plan 069 P2), and so are the Gauss counts for the Jacobians (quadrature.jacobian_nodes)."""
import collections
import math

import numpy as np

from ..core.errors import SupportOutsideDomain
from ..sets import Interval, Region
from ..sets.regions import Chart

TWO_PI = 2.0 * math.pi

Parametrisation = collections.namedtuple("Parametrisation", "gamma derivative t0 t1 breaks closed")
Parametrisation.__doc__ = """A curve as a map: gamma(t) (points, shape t.shape + (d,)) for t in [t0, t1], its derivative
(same shape), the parameters where gamma' may jump, and whether gamma(t0) = gamma(t1) (the Curve interface)."""

class _Jacobians(dict):
    """Jacobians of Lebesgue (or surface) measure along one natural axis: name -> function. A name "c*w" is the
    Jacobian w times the constant c (e.g. "4*sin" on a sphere of radius 2), so that a measure with a constant factor
    stays a product of one-dimensional Jacobians."""

    def __missing__(self, name):
        c, sep, base = str(name).partition("*")
        if not sep or base not in self:
            raise KeyError(name)
        c = float(c)
        f = self[base]
        self[name] = lambda t, f=f, c=c: c * f(t)
        return self[name]


JACOBIANS = _Jacobians({"1": lambda t: np.ones_like(t), "r": lambda t: t, "r2": lambda t: t * t, "sin": np.sin})


def scaled(c, w):
    """The weight name for c times the weight w ("1", "r", "r2", "sin")."""
    return w if c == 1 else f"{float(c)!r}*{w}"


def _interval(I):
    if isinstance(I, Interval):
        return I
    if isinstance(I, str):
        return Interval(I)
    a, b = I
    return Interval(float(a), float(b))


def _fmt(x):
    return str(int(x)) if float(x) == int(x) else f"{x:g}"


class Domain(Region):
    """A bounded domain in R^d with natural coordinates.

    reference      internal coordinate ranges [(lo, hi), ...] (the reference box of the backend)
    jacobian       per internal axis, the name of the weight ("1", "r", "r2", "sin"); Lebesgue measure =
                   prod_k J_k(xi_k) dxi
    periodic       per internal axis, whether it is an angle (period 2 pi)
    internal_names names of the internal coordinates
    ambient_dim    the number of Cartesian coordinates (larger than dim on a curve or surface: a circle in R^2,
                   a sphere in R^3)
    reference_breaks per internal axis, points where functions of interest may lose smoothness (the vertices of a
                   polyline); realisations start their pieces there

    A Domain is a parametrised Region (sets/regions.py): the image of its reference box under to_cartesian, with the
    measure prod J_k dxi_k (its Lebesgue or Hausdorff measure: volume, area, arc length). Membership is the Region's:
    contains(points) on arrays of shape (..., cartesian_dim), member(x) -> Truth, `x in D` (plan 067 P2: the
    coordinate signature contains(*x) is gone); subclasses decide it in _inside(*coordinates). Curves and surfaces are Domains with
    dim < ambient_dim, so they can carry function spaces and measures like any other domain.
    """

    dim = None
    ambient_dim = None          # the number of Cartesian coordinates functions take (default: dim)
    reference = ()
    jacobian = ()
    periodic = ()
    internal_names = ()
    reference_breaks = None

    def to_cartesian(self, *xi):
        raise NotImplementedError

    def from_cartesian(self, *x):
        raise NotImplementedError

    @property
    def bounding_box(self):
        raise NotImplementedError

    @property
    def volume(self):
        raise NotImplementedError

    def jacobian_value(self, k, t):
        return JACOBIANS[self.jacobian[k]](np.asarray(t, float))

    def charts(self):
        """One chart (sets/regions.Chart): the reference box, its periodic axes and breaks, to_cartesian and the
        per-axis Jacobians (pygeoinf0.quadrature.rule integrates over the domain through it)."""
        jac = tuple((lambda t, k=k: self.jacobian_value(k, t)) for k in range(len(self.reference)))
        return (Chart(tuple(self.reference), tuple(self.periodic), self.reference_breaks, self.to_cartesian, jac, None),)

    def pieces(self):
        """The pieces of the set (DESIGN 2.1): a domain is one chart piece (its reference box and to_cartesian) unless it
        says otherwise (a polyline: its segments)."""
        return (self,)

    def same_up_to_null_set(self, other):
        return self.key() == getattr(other, "key", lambda: None)()

    def __eq__(self, other):
        return isinstance(other, Domain) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


def require_in_closure(domain, X, what, segment=False):
    """Refuse (SupportOutsideDomain) unless the points X (m, d) lie in the closure of the domain, with a relative
    tolerance: points are pulled towards the centre of the bounding box by 1e-10 of their distance, which is exact for
    convex domains. what names the support in the message; segment: whether it is a segment (pg.clip can cut those).
    (Was realisations/atoms._check_closure; plan 069 P2: a geometric check, used by spaces/ and realisations/.)"""
    box = np.array(domain.bounding_box, float)
    c = box.mean(axis=1)
    Y = c + (X - c) * (1 - 1e-10)
    ok = np.atleast_1d(domain.contains(Y))
    fix = "clip it (pg.clip(segment, domain))" if segment else "move or shrink it (pg.clip takes segments only)"
    if not np.any(ok):
        raise SupportOutsideDomain(f"no sampled point of the support of {what} lies in {domain}: if it crosses the "
                                   f"domain, {fix}; otherwise it does not meet the domain")
    if not np.all(ok):
        raise SupportOutsideDomain(f"the support of {what} leaves {domain}: {fix} (the functional would depend on "
                                   "values outside the domain, which elements of a space on the domain do not have)")


def round_inside(domain, c, R):
    """Whether the sphere or circle with centre c and radius R lies in the closure of the domain: exact for balls,
    disks, boxes and cylinders (convex: compare the extents along their own axes), always on tori (periodic); None for
    other domains (then sampled). (Was realisations/atoms._round_inside, plan 069 P2.)"""
    from .manifolds import Torus
    from .solids import Ball, Box, Cylinder, Disk
    tol = 1e-10 * max(R, 1.0)
    if isinstance(domain, Torus):
        return True
    if isinstance(domain, (Ball, Disk)):
        return bool(np.linalg.norm(c - np.array(domain.centre)) + R <= domain.radius + tol)
    if isinstance(domain, Box):
        return all(s.a - tol <= x - R and x + R <= s.b + tol for s, x in zip(domain.sides, c))
    if isinstance(domain, Cylinder):
        return bool(np.linalg.norm(c[:2] - np.array(domain.centre)) + R <= domain.radius + tol
                    and domain.z.a - tol <= c[2] - R and c[2] + R <= domain.z.b + tol)
    return None
