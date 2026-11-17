"""Shared pieces of the bands and plateaus (plan 021 P1, work package H; bands.py and plateau.py).

step(u), dstep(u)            the C-infinity step S(u) = e(u) / (e(u) + e(1 - u)), e(u) = exp(-1/u) for u > 0 else 0:
                             0 for u <= 0, 1 for u >= 1, S(u) + S(1 - u) = 1, flat to all orders at 0 and 1
gl(f, breaks)                composite Gauss-Legendre (4 panels x 64 nodes between consecutive breaks)
point(D, centre)             a validated centre (default: the domain's own); on the manifold for spheres and circles
sides(D)                     the coordinate intervals of an interval, box or torus
SharpFunction                a function with jumps and an exact mass: but the contract's quadrature cannot
                             verify it (a discontinuous integrand), so the claim is kept apart from `closed`
"""

import itertools
import math
from types import MappingProxyType

import numpy as np

from ...core.claims import DERIVED, Claim, Claims
from ...core.errors import Refused
from ...quadrature import gauss
from .. import geometry as geo
from ..base import BankFunction

STEP_REFERENCE = (
    "Hormander 1990, The Analysis of Linear Partial Differential Operators I, 1.2: e(u) = exp(-1/u) is "
    "C-infinity, and S = e(u)/(e(u) + e(1-u)) is a quotient of C-infinity functions with a positive "
    "denominator"
)
_GL = gauss(64)


def step(u):
    """The C-infinity step S: 0 for u <= 0, 1 for u >= 1, strictly increasing between."""
    u = np.asarray(u, float)
    flat = np.atleast_1d(u).ravel()
    out = np.where(flat >= 1.0, 1.0, 0.0)
    m = (flat > 0) & (flat < 1)
    t = flat[m]
    out[m] = 1.0 / (1.0 + np.exp(np.clip(1.0 / t - 1.0 / (1.0 - t), -700.0, 700.0)))
    return out.reshape(u.shape)


def dstep(u):
    """S'(u) = S (1 - S) (1/u^2 + 1/(1-u)^2) on (0, 1), 0 elsewhere."""
    u = np.asarray(u, float)
    s = step(u)
    t = np.clip(u, 1e-3, 1 - 1e-3)
    return np.where((u > 0) & (u < 1), s * (1 - s) * (1 / t**2 + 1 / (1 - t) ** 2), 0.0)


def gl(f, breaks, panels=4):
    """The integral of f over [breaks[0], breaks[-1]]: Gauss-Legendre, `panels` panels of 64 nodes between breaks."""
    t, w = _GL
    total = 0.0
    for a, b in itertools.pairwise(breaks):
        for lo, hi in itertools.pairwise(np.linspace(a, b, panels + 1)):
            total += 0.5 * (hi - lo) * float(np.sum(w * f(0.5 * (hi - lo) * (t + 1) + lo)))
    return total


def sides(D):
    return [(D.a, D.b)] if geo.kind(D) == "interval" else [(s.a, s.b) for s in D.sides]


def point(D, centre, who):
    """The centre as an array: the domain's own when None; refused when it has the wrong length, is not finite, is not
    on the sphere or circle, or lies outside the coordinate ranges of a torus."""
    try:
        c = geo.centre(D) if centre is None else np.asarray(centre, float).reshape(-1)
    except (TypeError, ValueError):
        raise Refused(f"{who}: the centre must be numbers (got {centre!r})") from None
    k = geo.kind(D)
    if c.size != geo.ambient(D) or not np.all(np.isfinite(c)):
        raise Refused(f"{who}: a centre on a {k} is {geo.ambient(D)} finite Cartesian coordinates (got {c})")
    if k in ("sphere", "circle") and abs(np.linalg.norm(c - np.asarray(D.centre, float)) - D.radius) > 1e-9 * D.radius:
        raise Refused(f"{who}: the centre {tuple(c)} is not on the {k} (distance from its centre must be {D.radius:g})")
    if k == "torus" and any(not a <= ci <= b for ci, (a, b) in zip(c, sides(D))):
        raise Refused(f"{who}: the centre {tuple(c)} lies outside the torus' coordinate ranges {sides(D)}")
    return c


def positive(value, name, who):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise Refused(f"{who}: {name} must be a number (got {value!r})") from None
    if not (math.isfinite(v) and v > 0):
        raise Refused(f"{who}: {name} must be positive and finite (got {v})")
    return v


def nonnegative(value, name, who):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise Refused(f"{who}: {name} must be a number (got {value!r})") from None
    if not (math.isfinite(v) and v >= 0):
        raise Refused(f"{who}: {name} must be finite and >= 0 (got {v})")
    return v


class SharpFunction(BankFunction):
    """A bank function with jumps and an exact mass. `exact` holds closed forms that follow from the construction
    (derived) but that the contract's quadrature cannot check on a discontinuous integrand (name -> (value, status,
    reason)); `integral()` returns the exact integral and the claims list it."""

    def __init__(self, *args, exact=None, **kw):
        super().__init__(*args, **kw)
        self.exact = MappingProxyType(dict(exact or {}))

    def integral(self):
        return self.exact["integral"] if "integral" in self.exact else super().integral()

    @property
    def claims(self):
        extra = [Claim(f"{key} = {v:.15g}", status, why, key=key) for key, (v, status, why) in self.exact.items()]
        return Claims(tuple(super().claims) + tuple(extra))

    def _with_unit_integral(self, status, why):
        """The claim stays in `exact` (not duplicated into `closed`)."""
        self.exact = MappingProxyType({**self.exact, "integral": (1.0, status, why)})

    def scaled(self, c, name=None):
        out = super().scaled(c, name)
        exact = {key: (float(c) * v, st, f"{why}; scaled by {float(c):g}") for key, (v, st, why) in self.exact.items()}
        return SharpFunction(
            out.family,
            out.domain,
            out.params,
            out.fn,
            out.support,
            out.regularity,
            out.closed,
            out.gradient,
            out.name,
            out.reasons,
            out.breaks,
            exact=exact,
        )
