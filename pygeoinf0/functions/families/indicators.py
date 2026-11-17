"""Indicators of geodesic balls (plan 021 WP-C): sharp and smooth-edged.

    indicator(D, *, centre=None, radius=None)               1 on the closed geodesic ball, 0 elsewhere
    smooth_indicator(D, *, centre=None, radius=None, inner=None)
                                                            1 within `inner`, C-infinity transition to 0 at `radius`:
                                                            psi((radius - d)/(radius - inner)) with
                                                            psi(t) = f(t)/(f(t) + f(1 - t)), f(t) = exp(-1/t) (t > 0)

From indicators_common.py (merged here by plan 063):
Shared pieces of the indicator families (plan 021 WP-C; EXP-037 DESIGN.md 5, 6).

An indicator takes the values 0 and 1 and jumps across the boundary of its set, so its integral cannot be checked
against a quadrature rule to the contract's tolerance (the error of any rule is O(h) at a jump, and the contract's
reference rules put no node at the jump). The exact integrals of this package are therefore kept in `exact`, not in
`closed`: BankFunction.integral() and the claims report them (status derived, with the derivation), and
test_functions_indicators.py checks them by identities (partitions of the domain add up to its volume) and by fine
quadrature. Everything else about the base class is unchanged.

    IndicatorFunction   a BankFunction with `exact` (name -> (value, status, reason))
    jump_function(...)  builds one from a boolean mask (the support set itself) and the exact volume
    ball_volume(D, rho) the measure of a closed geodesic ball of the domain

From indicators_boxes.py (merged here by plan 063):
Indicators of boxes, radial bands, latitude bands and slabs (plan 021 WP-C).

    box_indicator(D, *, lower=None, upper=None)     lower <= x <= upper componentwise (interval, box)
    band_indicator(D, *, lower=None, upper=None)    lower <= |x - centre| <= upper (disk, ball, shell: radii about the
                                                    domain's own centre); lower <= latitude <= upper (sphere, radians)
    slab_indicator(D, *, normal=None, lower=None, upper=None)
                                                    lower <= x . normal <= upper (box, disk, ball, cylinder)
"""


import itertools
import math
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from ...core.claims import DERIVED, Claim, Claims
from ...core.errors import Refused
from ...quadrature import gauss
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register

# ---------------------------------------------------------------- from indicators_common.py (plan 063)
TOL = 1e-12                       # relative slack for "touches the boundary" and "lies inside the domain"
ALL_KINDS = ("interval", "box", "disk", "ball", "shell", "sphere", "circle", "cylinder", "torus")
GRISVARD = ("Grisvard 1985, Elliptic Problems in Nonsmooth Domains, 1.4 (piecewise smooth functions with jumps lie in "
            "H^s exactly for s < 1/2)")
HORMANDER = ("Hormander 1990, The Analysis of Linear Partial Differential Operators I, 1.2 (exp(-1/t) is "
             "C-infinity on R)")


class IndicatorFunction(BankFunction):
    """A BankFunction whose exact closed forms (`exact`) are not checkable by the contract's quadrature (module doc)."""

    def __init__(self, family, domain, params, fn, support, regularity, *, closed=None, exact=None, reasons=None,
                 name=None):
        super().__init__(family, domain, params, fn, support, regularity, closed=closed, name=name, reasons=reasons)
        self.exact = MappingProxyType(dict(exact or {}))

    def integral(self):
        if "integral" in self.exact:
            return self.exact["integral"]
        value, status, why = super().integral()
        if "integral" in self.closed:
            return value, status, why
        return value, status, (why + "; the function jumps, so this rule is only O(h) accurate: the error is not "
                               "bounded here and can be several percent")

    def _with_unit_integral(self, status, why):
        """The claim stays in `exact` (not duplicated into `closed`)."""
        self.exact = MappingProxyType({**self.exact, "integral": (1.0, status, why)})

    @property
    def claims(self):
        extra = [Claim(f"{k} = {v:.15g}", status, why, key=k) for k, (v, status, why) in self.exact.items()]
        return Claims(tuple(super().claims) + tuple(extra))

    def scaled(self, c, name=None):
        out = super().scaled(c, name)
        c = float(c)
        exact = {k: ((c * v if k == "integral" else c * c * v), st, f"{why}; scaled by {c:g}")
                 for k, (v, st, why) in self.exact.items()}
        return IndicatorFunction(out.family, out.domain, out.params, out.fn, out.support, out.regularity,
                                 closed=out.closed, exact=exact, reasons=out.reasons, name=out.name)


def jump_function(family, D, params, mask, *, kind, description, centre=None, radius=None, volume=None, why="",
                  reasons=None, box=None):
    """The 0/1 function of the support set `mask(*X) -> bool array` (the closed set itself: the function equals 1 where
    `mask` is true). kind: "interior", "touches" or "full" (the set is the whole domain: the function is the constant
    1, C-infinity). volume: the exact measure of the set (derived; `why` is the derivation)."""
    def fn(*X):
        return np.asarray(mask(*X), bool).astype(float)

    full = kind == "full"
    support = Support("full", description) if full else Support(kind, description, centre, radius, contains=mask,
                                                                 box=box)
    if full:
        regularity = Regularity(math.inf)
        reasons = {"regularity": "the constant 1 on the whole domain"}
    else:
        regularity = Regularity(-1, piecewise=True, vanishes_at_boundary="all orders" if kind == "interior" else None)
    closed = {"max": (1.0, DERIVED, "the function takes the values 0 and 1, and 1 on the set")}
    exact = {}
    if volume is not None:
        exact = {"integral": (float(volume), DERIVED, why),
                 "l2_norm_sq": (float(volume), DERIVED, "f takes only the values 0 and 1, so int f^2 = int f; " + why)}
    return IndicatorFunction(family, D, params, fn, support, regularity, closed=closed, exact=exact, reasons=reasons)


def centre_arg(D, centre):
    """The centre as a float array. On a closed manifold (geo.room is inf there, so the domain check of the caller
    cannot see it) the centre must lie on the manifold: |c - o| = R within 1e-9 R on a sphere or circle (then
    projected onto it), inside the closed fundamental cell on a torus."""
    try:
        c = geo.centre(D) if centre is None else np.asarray(centre, float).reshape(-1)
    except (TypeError, ValueError):
        raise Refused(f"a centre is a point (numbers), not {centre!r}") from None
    if c.size != geo.ambient(D) or not np.all(np.isfinite(c)):
        raise Refused(f"a centre on {D!r} has {geo.ambient(D)} finite Cartesian coordinates, not {c.size}")
    k = geo.kind(D)
    if k in ("sphere", "circle"):
        o = np.asarray(D.centre, float)
        dist = float(np.linalg.norm(c - o))
        if abs(dist - D.radius) > 1e-9 * D.radius:
            raise Refused(f"the centre {tup(c)} lies outside the domain {D!r}: it is at distance {dist:g} from the "
                          f"middle, not on the {k} of radius {D.radius:g}")
        c = o + D.radius * (c - o) / dist
    elif k == "torus":
        for x, s in zip(c, D.sides):
            if not (s.a - TOL * (s.b - s.a) <= x <= s.b + TOL * (s.b - s.a)):
                raise Refused(f"the centre {tup(c)} lies outside the domain {D!r}: it is outside the fundamental "
                              f"cell [{s.a:g}, {s.b:g}] of the torus")
        c = np.clip(c, [s.a for s in D.sides], [s.b for s in D.sides])
    return c


def ball_volume(D, rho):
    """(measure, derivation) of the closed geodesic ball of radius rho (inside the domain, below the injectivity
    radius): the Euclidean ball pi^(d/2)/Gamma(d/2+1) rho^d; a spherical cap 2 pi R^2 (1 - cos(rho/R)); an arc 2 rho."""
    k = geo.kind(D)
    if k == "sphere":
        return (2 * math.pi * D.radius ** 2 * (1 - math.cos(rho / D.radius)),
                "area of a spherical cap: 2 pi R^2 (1 - cos(rho/R))")
    if k == "circle":
        return 2.0 * rho, "length of an arc of half-length rho: 2 rho"
    d = geo.dimension(D)
    return (math.pi ** (d / 2) / math.gamma(d / 2 + 1) * rho ** d,
            f"volume of a Euclidean ball in R^{d}: pi^(d/2)/Gamma(d/2+1) rho^d (the ball lies in the domain)")


def tup(v, digits=6):
    """A plain tuple of rounded floats (for descriptions and parameter records)."""
    return tuple(round(float(x), digits) for x in np.asarray(v, float).reshape(-1))


def positive(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise Refused(f"{name} must be a number (got {value!r})") from None
    if not (math.isfinite(value) and value > 0):
        raise Refused(f"{name} must be positive and finite (got {value})")
    return value


FLAT = ("interval", "box", "disk", "ball", "shell", "cylinder")
RADIAL = ("measure of the geodesic ball of radius inner (psi = 1) plus the radial reduction of the integral over the "
          "transition, composite Gauss-Legendre (16 x 64 nodes) in t after r = radius - width t")
LEE = "Lee 2013, Introduction to Smooth Manifolds, 2nd ed., chapter 2 (smooth bump functions from exp(-1/t))"


def _reach(D, c):
    """(room, injectivity radius, largest allowed radius) of geodesic balls about c."""
    room, inj = geo.room(D, c), geo.injectivity(D)
    if room < 0:
        raise Refused(f"the centre {tup(c)} lies outside the domain {D!r}")
    return room, inj


def _farthest(D, c):
    """The distance from c to the farthest point of the domain, where the ball could cover it (interval, disk, ball);
    inf elsewhere (no admissible ball contains the whole of the other kinds)."""
    k = geo.kind(D)
    if k == "interval":
        return float(max(abs(c[0] - D.a), abs(D.b - c[0])))
    if k in ("disk", "ball"):
        return float(np.linalg.norm(c - np.asarray(D.centre, float)) + D.radius)
    return math.inf


# --------------------------------------------------------------------------------------------- indicator
def _indicator_examples(D):
    c = geo.centre(D)
    room, inj = geo.room(D, c), geo.injectivity(D)
    reach = min(room, 0.9 * inj)                               # closed manifolds: the ball stays below the injectivity
    return [{}, {"radius": 0.3 * reach}, {"centre": tuple(c), "radius": reach}]


@register("indicator", ALL_KINDS, "indicator of a closed geodesic ball (jump: order -1)",
          references=(GRISVARD,), examples=_indicator_examples)
def indicator(D, *, centre=None, radius=None):
    """1 on the closed geodesic ball {x : d(x, centre) <= radius}, 0 elsewhere.

    Defaults: the domain's centre; half of min(room to the boundary, injectivity radius). The ball must lie in the
    domain: radius <= the distance from the centre to the boundary (a radius above it by at most the relative slack
    1e-12 is reduced to it; equal: the support touches the boundary), and below the injectivity radius on closed
    manifolds (a cap of a sphere, an arc of a circle, a ball of a torus below half the shortest period); on a closed
    manifold the centre must lie on it (a sphere or circle: within 1e-9 R, then projected; a torus: in the closed
    fundamental cell). Exact integral: the measure of the ball (Euclidean pi^(d/2)/Gamma(d/2+1) r^d, a spherical cap
    2 pi R^2 (1 - cos(r/R)), an arc 2r). Regularity: order -1 (jump). Only a ball that contains every point of the
    domain (compared exactly, without tolerance: radius >= the distance from the centre to the farthest point) is
    the constant 1, C-infinity; a ball that misses any of the domain, however little, is a jump."""
    c = centre_arg(D, centre)
    room, inj = _reach(D, c)
    radius = 0.5 * min(room, inj) if radius is None else positive("radius", radius)
    if radius > room:
        if radius > room * (1 + TOL):
            raise Refused(f"indicator: radius {radius:g} leaves the domain (room to the boundary {room:g}); the ball "
                          "must lie inside")
        radius = room                                          # within the slack: the ball is the largest one inside
    if radius >= inj:
        raise Refused(f"indicator: radius {radius:g} reaches the injectivity radius {inj:g}: a geodesic ball is "
                      "defined below it")
    vol, why = ball_volume(D, radius)
    kind = "full" if radius >= _farthest(D, c) else ("touches" if radius >= room else "interior")

    def mask(*X):
        return geo.distance(D, c, *X) <= radius

    return jump_function("indicator", D, {"centre": tup(c, 15), "radius": radius}, mask, kind=kind,
                         description=f"the closed geodesic ball of radius {radius:g} about {tup(c)}",
                         centre=tup(c, 15), radius=radius, volume=vol, why=why)


# --------------------------------------------------------------------------------------------- smooth indicator
def _psi(t):
    """psi(t) = f(t)/(f(t) + f(1 - t)), f(t) = exp(-1/t): 0 for t <= 0, 1 for t >= 1, C-infinity and increasing."""
    t = np.asarray(t, float)
    mid = (t > 0) & (t < 1)
    tt = np.where(mid, t, 0.5)
    with np.errstate(over="ignore"):
        val = 1.0 / (1.0 + np.exp(np.minimum(1.0 / tt - 1.0 / (1.0 - tt), 700.0)))     # f(1-t)/f(t) = exp(1/t - 1/(1-t))
    return np.where(t <= 0, 0.0, np.where(t >= 1, 1.0, val))


_GL = gauss(64)


def _shell_density(D, r):
    """d(measure of the geodesic ball of radius r)/dr: the area of the geodesic sphere of radius r."""
    k = geo.kind(D)
    if k == "sphere":
        return 2 * math.pi * D.radius * np.sin(r / D.radius)
    if k == "circle":
        return 2.0 + 0.0 * r
    d = geo.ambient(D)
    return geo.SPHERE_AREA[d] * r ** (d - 1)


def _transition_mass(D, inner, radius, power, panels=16):
    """int over D of psi((radius - d)/width)^power, with d the distance from the centre: vol(ball(inner)) + width
    int_0^1 psi(t)^power density(radius - width t) dt. The transition is integrated in t on [0, 1], where psi does not
    depend on the width, so a transition much narrower than the radius loses nothing."""
    width = radius - inner
    edges = np.linspace(0.0, 1.0, panels + 1)
    t, w = _GL
    total = 0.0
    for a, b in itertools.pairwise(edges):
        tt = 0.5 * (b - a) * (t + 1) + a
        total += np.sum(0.5 * (b - a) * w * _psi(tt) ** power * _shell_density(D, radius - width * tt))
    return float(ball_volume(D, inner)[0] + width * total)


def _smooth_examples(D):
    c = geo.centre(D)
    room = min(geo.room(D, c), geo.injectivity(D))
    return [{}, {"radius": 0.3 * room, "inner": 0.1 * room}, {"centre": tuple(c), "radius": 0.6 * room, "inner": 0.0}]


@register("smooth_indicator", ALL_KINDS, "1 within `inner`, C-infinity transition to 0 at `radius` (smooth step of the "
          "distance)", references=(HORMANDER, LEE), examples=_smooth_examples)
def smooth_indicator(D, *, centre=None, radius=None, inner=None):
    """psi((radius - d)/(radius - inner)), d the geodesic distance from `centre`, psi(t) = f(t)/(f(t) + f(1 - t)),
    f(t) = exp(-1/t) for t > 0 and 0 otherwise: equal to 1 for d <= inner, 0 for d >= radius, strictly decreasing
    between, C-infinity (f is, and f(t) + f(1 - t) > 0). psi(t) + psi(1 - t) = 1, so the value is 1/2 at the middle
    radius (inner + radius)/2. Defaults: the domain's centre, radius half of min(room to the boundary, injectivity
    radius) (on a sphere, circle or torus the room is infinite, so half the injectivity radius), inner = radius/2;
    0 <= inner < radius < min(room, injectivity radius) (the support lies inside the domain); on a closed manifold
    the centre must lie on it (see `indicator`). Exact integral and squared norm (derived): the measure of the ball of
    radius `inner` plus the radial integral over the transition, taken after the substitution r = radius - width t,
    which keeps a narrow transition accurate; maximum 1 at the centre."""
    c = centre_arg(D, centre)
    room, inj = _reach(D, c)
    radius = 0.5 * min(room, inj) if radius is None else positive("radius", radius)
    try:
        inner = 0.5 * radius if inner is None else float(inner)
    except (TypeError, ValueError):
        raise Refused(f"smooth_indicator: inner must be a number (got {inner!r})") from None
    if not (math.isfinite(inner) and 0 <= inner < radius):
        raise Refused(f"smooth_indicator: need 0 <= inner < radius (inner {inner}, radius {radius:g})")
    if radius >= room or radius >= inj:
        raise Refused(f"smooth_indicator: radius {radius:g} leaves the domain (room to the boundary {room:g}, "
                      f"injectivity radius {inj:g}); the support lies inside")
    width = radius - inner

    def profile(r):
        return _psi((radius - np.asarray(r, float)) / width)

    def fn(*X):
        return profile(geo.distance(D, c, *X))

    grad = None
    if geo.kind(D) in FLAT:
        def grad(*X):
            P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
            r = np.linalg.norm(P - c, axis=-1)
            t = (radius - r) / width
            m = (t > 0) & (t < 1) & (r > 0)
            tm = np.where(m, t, 0.5)
            p = _psi(tm)
            dp = np.where(m, p * (1 - p) * (1 / tm ** 2 + 1 / (1 - tm) ** 2) * (-1.0 / width) / np.where(r > 0, r, 1.0),
                          0.0)
            return dp[..., None] * (P - c)

    closed = {"integral": (_transition_mass(D, inner, radius, 1), DERIVED, RADIAL),
              "l2_norm_sq": (_transition_mass(D, inner, radius, 2), DERIVED, RADIAL),
              "max": (1.0, DERIVED, "psi(1) = 1: the value at the centre")}
    return BankFunction("smooth_indicator", D, {"centre": tup(c, 15), "radius": radius, "inner": inner}, fn,
                        Support("interior", f"the closed geodesic ball of radius {radius:g} about "
                                f"{tup(c)}", tup(c, 15), radius, breaks=(inner,)),
                        Regularity(math.inf, vanishes_at_boundary="all orders"), closed, grad,
                        reasons={"regularity": HORMANDER + "; " + LEE,
                                 "support": f"psi vanishes for d >= radius; room to the boundary {room:g} > radius"})


# ---------------------------------------------------------------- from indicators_boxes.py (plan 063)
IRWIN = "Irwin 1927 / Hall 1927 (the distribution of a sum of independent uniform variables, here with weights)"


def _vector(name, value, d, default):
    v = np.asarray(default if value is None else value, float).reshape(-1)
    if v.size != d or not np.all(np.isfinite(v)):
        raise Refused(f"{name} needs {d} finite coordinates (got {v.size})")
    return v


def _kind_of_support(covers, touches):
    return "full" if covers else ("touches" if touches else "interior")


# --------------------------------------------------------------------------------------------- box_indicator
def _box_examples(D):
    sides = [(D.a, D.b)] if geo.kind(D) == "interval" else [(s.a, s.b) for s in D.sides]
    out = [{}]
    out.append({"lower": tuple(a + 0.1 * (b - a) for a, b in sides), "upper": tuple(a + 0.4 * (b - a) for a, b in sides)})
    out.append({"lower": tuple(a for a, _ in sides), "upper": tuple(a + 0.5 * (b - a) for a, b in sides)})
    out.append({"lower": tuple(a for a, _ in sides), "upper": tuple(b for _, b in sides)})
    return out


@register("box_indicator", ("interval", "box"), "indicator of a sub-interval or sub-box, lower <= x <= upper",
          references=(GRISVARD,), examples=_box_examples)
def box_indicator(D, *, lower=None, upper=None):
    """1 on the closed sub-box prod_i [lower_i, upper_i], 0 elsewhere (a sub-interval on an interval; scalars allowed
    there). Defaults: the central half of every side. Requires a_i <= lower_i < upper_i <= b_i (the sub-box lies in the
    domain and has positive measure); touching a side of the domain makes the support touch the boundary, and the whole
    domain gives the constant 1. A bound outside the domain by at most the relative slack 1e-12 of the side is moved
    onto the domain; every other comparison is exact, so a box that misses the domain, however little, is a jump and
    only a box equal to the domain is the constant 1. Exact integral: the product of the side lengths
    prod_i (upper_i - lower_i)."""
    sides = [(D.a, D.b)] if geo.kind(D) == "interval" else [(s.a, s.b) for s in D.sides]
    d = len(sides)
    a, b = np.array([s[0] for s in sides], float), np.array([s[1] for s in sides], float)
    lo = _vector("lower", lower, d, a + 0.25 * (b - a))
    hi = _vector("upper", upper, d, a + 0.75 * (b - a))
    slack = TOL * (b - a)
    if np.any(lo < a - slack) or np.any(hi > b + slack):
        raise Refused(f"box_indicator: the sub-box [{lo}, {hi}] leaves the domain [{a}, {b}]")
    if not np.all(hi > lo):
        raise Refused(f"box_indicator: need lower < upper in every coordinate (lower {lo}, upper {hi})")
    lo, hi = np.maximum(lo, a), np.minimum(hi, b)                # within the slack: onto the domain
    covers = bool(np.all(lo <= a) and np.all(hi >= b))
    touches = bool(np.any(lo <= a) or np.any(hi >= b))
    c, radius = tup(0.5 * (lo + hi), 15), 0.5 * float(np.linalg.norm(hi - lo))
    room = geo.room(D, np.array(c))
    inside = radius < room or touches                           # the bounding ball is stated only where it lies inside

    def mask(*X):
        out = True                         # pairwise: the coordinates may be broadcast (open meshgrids of the fits)
        for x, a_, b_ in zip(X, lo, hi):
            out = out & (np.asarray(x, float) >= a_) & (np.asarray(x, float) <= b_)
        return np.asarray(out)

    return jump_function("box_indicator", D, {"lower": tup(lo, 15), "upper": tup(hi, 15)}, mask,
                         kind=_kind_of_support(covers, touches),
                         description=f"the closed box {tup(lo)} .. {tup(hi)}",
                         centre=c if inside else None, radius=radius if inside else None,
                         volume=float(np.prod(hi - lo)), why="product of the side lengths prod_i (upper_i - lower_i)",
                         box=tuple(zip(lo.tolist(), hi.tolist())))


# --------------------------------------------------------------------------------------------- band_indicator
def _band_examples(D):
    k = geo.kind(D)
    if k == "sphere":
        h = math.pi / 2
        return [{}, {"lower": 0.0, "upper": h}, {"lower": -h, "upper": -0.3}, {"lower": -h, "upper": h}]
    lo, hi = (D.inner, D.outer) if k == "shell" else (0.0, D.radius)
    t = hi - lo
    return [{}, {"lower": lo, "upper": lo + 0.4 * t}, {"lower": lo + 0.3 * t, "upper": hi}, {"lower": lo, "upper": hi}]


@register("band_indicator", ("disk", "ball", "shell", "sphere"),
          "indicator of a radial band lower <= r <= upper (disk, ball, shell) or a latitude band (sphere)",
          references=(GRISVARD,), examples=_band_examples)
def band_indicator(D, *, lower=None, upper=None):
    """Disk, ball, shell: 1 where lower <= |x - centre| <= upper, the centre being the domain's own centre (radii; for
    a shell inner <= lower < upper <= outer, for a disk or ball 0 <= lower < upper <= radius). Sphere: 1 where the
    latitude lies in [lower, upper] (radians, -pi/2 <= lower < upper <= pi/2, latitude measured from the equator of
    the z axis through the sphere's centre: z - z0 = R sin(latitude)). Defaults: the middle half of the radii; the
    band [-pi/6, pi/6] about the equator. A band reaching the domain's outer (or a shell's inner) boundary touches it;
    a band filling the domain is the constant 1. A bound outside the domain by at most the relative slack 1e-12 is
    moved onto it; every other comparison is exact (a band missing the domain, however little, is a jump). Exact integrals: pi (b^2 - a^2) (disk), 4 pi/3 (b^3 - a^3) (ball,
    shell), 2 pi R^2 (sin(upper) - sin(lower)) (sphere; Archimedes' hat-box theorem)."""
    k = geo.kind(D)
    if k == "sphere":
        return _latitude_band(D, lower, upper)
    lo0, hi0 = (D.inner, D.outer) if k == "shell" else (0.0, D.radius)
    t = hi0 - lo0
    lo = lo0 + 0.25 * t if lower is None else float(lower)
    hi = lo0 + 0.75 * t if upper is None else float(upper)
    slack = TOL * hi0
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < lo0 - slack or hi > hi0 + slack:
        raise Refused(f"band_indicator: the radii [{lo:g}, {hi:g}] must lie in [{lo0:g}, {hi0:g}], the radii of the {k}")
    lo, hi = max(lo, lo0), min(hi, hi0)                          # within the slack: onto the domain; then exact tests
    if not hi > lo:
        raise Refused(f"band_indicator: need lower < upper (lower {lo:g}, upper {hi:g})")
    centre = np.asarray(D.centre, float)
    covers = lo <= lo0 and hi >= hi0
    touches = hi >= hi0 or (k == "shell" and lo <= lo0)
    d = geo.dimension(D)
    vol = math.pi * (hi ** 2 - lo ** 2) if d == 2 else 4.0 * math.pi / 3.0 * (hi ** 3 - lo ** 3)
    why = ("area of an annulus pi (b^2 - a^2)" if d == 2 else "volume of a spherical shell 4 pi/3 (b^3 - a^3)")

    def mask(*X):
        r = geo.distance(D, centre, *X)
        return (r >= lo) & (r <= hi)

    return jump_function("band_indicator", D, {"lower": lo, "upper": hi}, mask, kind=_kind_of_support(covers, touches),
                         description=f"the closed band {lo:g} <= |x - centre| <= {hi:g}",
                         centre=None if k == "shell" else tup(centre, 15), radius=None if k == "shell" else hi,
                         volume=vol, why=why)


def _latitude_band(D, lower, upper):
    h = math.pi / 2
    lo = -math.pi / 6 if lower is None else float(lower)
    hi = math.pi / 6 if upper is None else float(upper)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < -h - TOL or hi > h + TOL:
        raise Refused(f"band_indicator: latitudes [{lo:g}, {hi:g}] must lie in [-pi/2, pi/2] (radians)")
    lo, hi = max(lo, -h), min(hi, h)                             # within the slack: onto the sphere; then exact tests
    if not hi > lo:
        raise Refused(f"band_indicator: need lower < upper (lower {lo:g}, upper {hi:g})")
    z0, R = float(D.centre[2]), D.radius
    zlo, zhi = R * math.sin(lo), R * math.sin(hi)
    covers = lo <= -h and hi >= h

    def mask(*X):
        z = np.asarray(X[2], float) - z0
        return (z >= zlo) & (z <= zhi)

    return jump_function("band_indicator", D, {"lower": lo, "upper": hi}, mask, kind="full" if covers else "interior",
                         description=f"the closed latitude band {lo:g} <= latitude <= {hi:g}",
                         volume=2 * math.pi * R ** 2 * (math.sin(hi) - math.sin(lo)),
                         why="area of a zone 2 pi R^2 (sin(upper) - sin(lower)) (Archimedes: the area of a sphere "
                             "between two parallel planes is 2 pi R times the distance between them)")


# --------------------------------------------------------------------------------------------- slab_indicator
def _slab_examples(D):
    n, k = geo.ambient(D), geo.kind(D)
    e = np.eye(n)
    normals = [tuple(e[0]), tuple(e[-1]), (0.6, 0.8) if n == 2 else (1 / 3, 2 / 3, 2 / 3)]
    if k == "cylinder":
        normals = [(0.6, 0.8, 0.0), (0.0, 0.0, 1.0), (0.0, 0.6, 0.8)]       # the last is oblique: no closed form
    out = []
    for v in normals:
        lo, hi = _extent(D, np.asarray(v, float))
        out.append({"normal": v, "lower": lo + 0.2 * (hi - lo), "upper": lo + 0.55 * (hi - lo)})
    out.append({"normal": normals[0]})
    return out


def _extent(D, n):
    """(min, max) of x . n over the domain."""
    k = geo.kind(D)
    if k == "box":
        lo = sum(min(ni * s.a, ni * s.b) for ni, s in zip(n, D.sides))
        hi = sum(max(ni * s.a, ni * s.b) for ni, s in zip(n, D.sides))
        return lo, hi
    c = np.asarray(geo.centre(D), float)
    if k == "cylinder":
        return (float(n @ c) - (D.radius * np.hypot(n[0], n[1]) + abs(n[2]) * 0.5 * (D.z.b - D.z.a)),
                float(n @ c) + (D.radius * np.hypot(n[0], n[1]) + abs(n[2]) * 0.5 * (D.z.b - D.z.a)))
    return float(n @ c) - D.radius, float(n @ c) + D.radius


def _disk_area(R, u1, u2):
    """Area of the part of the disk of radius R between the chords at signed distances u1 < u2 from the centre."""
    u1, u2 = max(u1, -R), min(u2, R)
    if u2 <= u1:
        return 0.0

    def F(u):
        return u * math.sqrt(max(R * R - u * u, 0.0)) + R * R * math.asin(max(-1.0, min(1.0, u / R)))
    return F(u2) - F(u1)


def _box_slab_volume(D, n, a, b):
    """Volume of {a <= x . n <= b} inside the box. With s = x . n = smin + sum c_i y_i, c_i = |n_i|, y_i uniform on
    [0, w_i]: the volume below level t is prod_{n_i = 0} w_i (1/(m! prod c_i)) sum_S (-1)^|S| (t - smin - sum_{i in S}
    c_i w_i)_+^m (m = number of c_i != 0). The alternating sum cancels badly in floating point when a coefficient
    c_i w_i is small against the others (an elongated box, a nearly axial normal), so it is evaluated in exact
    rational arithmetic on the given floats (fractions.Fraction): the result is the volume of the polytope described
    by those floats, rounded once to a float, whatever n, a, b and the sides are."""
    sides = [(Fraction(s.a), Fraction(s.b)) for s in D.sides]
    nf = [Fraction(float(x)) for x in n]
    w = [hi - lo for lo, hi in sides]
    idx = [i for i, x in enumerate(nf) if x != 0]
    c, wn = [abs(nf[i]) for i in idx], [w[i] for i in idx]
    smin = sum(min(x * lo, x * hi) for x, (lo, hi) in zip(nf, sides))
    m = len(idx)
    flat = Fraction(1)
    for i, x in enumerate(nf):
        if x == 0:
            flat *= w[i]
    denom = math.factorial(m) * math.prod(c)
    a, b = Fraction(float(a)), Fraction(float(b))

    def below(t):
        total = Fraction(0)
        for r in range(m + 1):
            for S in itertools.combinations(range(m), r):
                total += (-1) ** r * max(t - smin - sum((c[i] * wn[i] for i in S), Fraction(0)), Fraction(0)) ** m
        return flat * total / denom
    return float(below(b) - below(a))


def _slab_volume(D, n, a, b):
    """(volume, derivation) of the slab inside the domain, or (None, "") when there is no closed form."""
    k = geo.kind(D)
    if k == "box":
        return (_box_slab_volume(D, n, a, b),
                "the volume of a box cut by two parallel planes: differences of the weighted Irwin-Hall formula, "
                "evaluated in exact rational arithmetic; " + IRWIN)
    c = np.asarray(geo.centre(D), float)
    off = float(n @ c)
    if k == "disk":
        return _disk_area(D.radius, a - off, b - off), "area between two chords of a disk: [u sqrt(R^2-u^2) + R^2 asin(u/R)]"
    if k == "ball":
        u1, u2 = max(a - off, -D.radius), min(b - off, D.radius)
        return (math.pi * (D.radius ** 2 * (u2 - u1) - (u2 ** 3 - u1 ** 3) / 3.0),
                "volume between two planes of a ball: pi [R^2 u - u^3/3] (Archimedes)")
    H = D.z.b - D.z.a
    if n[2] == 0.0:
        return (H * _disk_area(D.radius, a - float(n[:2] @ np.asarray(D.centre, float)),
                               b - float(n[:2] @ np.asarray(D.centre, float))),
                "axial height times the area between two chords of the cross-section disk (normal orthogonal to the axis)")
    if n[0] == 0.0 and n[1] == 0.0:
        z1, z2 = sorted((a / n[2], b / n[2]))
        return (math.pi * D.radius ** 2 * max(0.0, min(D.z.b, z2) - max(D.z.a, z1)),
                "disk area times the axial length of the slab (normal along the axis)")
    return None, ""


@register("slab_indicator", ("box", "disk", "ball", "cylinder"),
          "indicator of the slab lower <= x . normal <= upper (unit normal)", references=(GRISVARD, IRWIN),
          examples=_slab_examples)
def slab_indicator(D, *, normal=None, lower=None, upper=None):
    """1 where lower <= x . normal <= upper (x the Cartesian coordinates), 0 elsewhere; `normal` a unit vector (norm
    within 1e-9 of 1, then normalised). Defaults: the last coordinate axis, and the middle half of the range of x . normal
    over the domain. The slab must meet the interior of the domain (lower < max, upper > min); its support touches
    the boundary, and a slab containing the whole domain (compared exactly) is the constant 1. Exact integrals
    (derived): box, any normal (Irwin-Hall in exact rational arithmetic on the given floats, so no normal or side
    lengths lose digits); disk and ball, any normal (chord and cross-section areas); cylinder, normal along or
    orthogonal to the axis. Otherwise (an oblique normal in a cylinder) the integral falls back to the quadrature,
    reported with status unknown because its error is not bounded."""
    n = np.asarray(np.eye(geo.ambient(D))[-1] if normal is None else normal, float).reshape(-1)
    if n.size != geo.ambient(D) or not np.all(np.isfinite(n)):
        raise Refused(f"slab_indicator: the normal needs {geo.ambient(D)} finite coordinates (got {n.size})")
    norm = float(np.linalg.norm(n))
    if abs(norm - 1.0) > 1e-9:
        raise Refused(f"slab_indicator: the normal must be a unit vector (norm {norm:.12g})")
    n = n / norm
    smin, smax = _extent(D, n)
    a = smin + 0.25 * (smax - smin) if lower is None else float(lower)
    b = smin + 0.75 * (smax - smin) if upper is None else float(upper)
    if not (math.isfinite(a) and math.isfinite(b)) or not b > a:
        raise Refused(f"slab_indicator: need lower < upper (lower {a:g}, upper {b:g})")
    if not (a < smax and b > smin):
        raise Refused(f"slab_indicator: the slab [{a:g}, {b:g}] does not meet the domain, where x . normal ranges over "
                      f"[{smin:g}, {smax:g}]")
    covers = a <= smin and b >= smax                            # exact: a slab that misses any of the domain is a jump
    vol, why = _slab_volume(D, n, a, b)

    def mask(*X):
        s = sum(ni * np.asarray(x, float) for ni, x in zip(n, X))
        return (s >= a) & (s <= b)

    return jump_function("slab_indicator", D, {"normal": tup(n, 15), "lower": a, "upper": b}, mask,
                         kind="full" if covers else "touches",
                         description=f"the closed slab {a:g} <= x . normal <= {b:g}, normal {tup(n)}",
                         volume=vol, why=why)
