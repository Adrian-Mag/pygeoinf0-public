"""Tapers and radial profiles (plan 021 P1, work package A; EXP-037 DESIGN.md 5, 6): compactly supported functions of
the distance to a centre, equal to 1 (or peaking at 1) at the centre and vanishing from `radius` on. With t = distance
/ radius (geodesic distance on spheres and circles, periodic minimum image on tori) and p the profile:

    cosine_taper(centre, radius, inner)  p = 1 for r <= inner; (1 + cos(pi s))/2, s = (r - inner)/(radius - inner),
                                         for inner < r < radius; 0 from radius on.            C^1, piecewise
    raised_cosine(centre, radius)        p = (1 + cos(pi t))/2 for t < 1, else 0 (cosine_taper with inner = 0)
                                                                                              C^1, piecewise
    smoothstep(centre, radius, order)    p = 1 - S_order(t) for t < 1, else 0, S_1 = 3t^2 - 2t^3,
                                         S_2 = 10t^3 - 15t^4 + 6t^5, S_3 = 35t^4 - 84t^5 + 70t^6 - 20t^7    C^order, piecewise
    bspline_profile(centre, radius, degree)  p(t) = B_degree(t (degree + 1)/2) / B_degree(0), B_degree the cardinal
                                         B-spline (knots at the integers, support [-(degree+1)/2, (degree+1)/2])
                                                                                              C^(degree-1), piecewise

All four have interior support (the closed geodesic ball of `radius`, which must lie inside the domain at a positive
distance from the boundary) and peak 1 at the centre. Closed forms: the integral and the squared L2 norm by
geometry.radial_mass (derived), the maximum 1.
"""
import itertools
import math
import numbers

import numpy as np

from ...core.claims import DERIVED
from ...core.errors import Refused
from ...quadrature import gauss
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register

TAPER_KINDS = ("interval", "box", "disk", "ball", "shell", "sphere", "circle", "cylinder", "torus")
FLAT = ("interval", "box", "disk", "ball", "shell", "cylinder")
RADIAL = "radial reduction of the integral over a geodesic ball, composite Gauss-Legendre (8 x 64 nodes) in r"
HARRIS = ("Harris 1978, On the use of windows for harmonic analysis with the discrete Fourier transform, Proc. IEEE "
          "66, 51-83 (cosine-tapered (Tukey) and raised-cosine (Hann) windows)")
COMPOSITION = ("a radial function p(distance) is as smooth as p is, because the distance is smooth away from the centre "
               "(inside the injectivity radius) and p is an even function of r near r = 0 or has its only singular "
               "term |r|^m (m odd) of class C^(m-1)")
BETA = ("Abramowitz & Stegun 1964, Handbook of Mathematical Functions, 26.5 (S_n is the regularised incomplete beta "
        "function I_t(n+1, n+1): S_n'(t) = (2n+1)!/(n!)^2 (t(1-t))^n has a zero of order n at 0 and 1, so S_n is C^n "
        "across t = 0 and t = 1 and S_n^(n+1) jumps)")
DEBOOR = ("de Boor 2001, A Practical Guide to Splines, IX; Schoenberg 1973, Cardinal Spline Interpolation, 1 (a spline "
          "of degree m with simple knots is C^(m-1), and its (m)-th derivative jumps at each knot)")


# --------------------------------------------------------------------------------------------- validation
def _real(family, name, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
        raise Refused(f"{family}: {name} must be a finite real number (got {v!r})")
    return float(v)


def _integer(family, name, v, allowed):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral) or int(v) not in allowed:
        raise Refused(f"{family}: {name} must be one of {tuple(allowed)} (got {v!r})")
    return int(v)


def _centre(family, D, centre):
    """The centre as an array; refused if it has the wrong size, is not finite, is off a sphere/circle or outside a
    torus' fundamental box (the minimum-image distance needs it inside)."""
    if centre is None:
        return geo.centre(D)
    try:
        c = np.asarray(centre, float).reshape(-1)
    except (TypeError, ValueError):
        raise Refused(f"{family}: the centre must be a sequence of {geo.ambient(D)} numbers (got {centre!r})") from None
    if c.size != geo.ambient(D):
        raise Refused(f"{family}: a centre on {D!r} has {geo.ambient(D)} Cartesian coordinates, not {c.size}")
    if not np.all(np.isfinite(c)):
        raise Refused(f"{family}: the centre must be finite (got {centre!r})")
    k = geo.kind(D)
    if k in ("sphere", "circle") and abs(np.linalg.norm(c - np.asarray(D.centre, float)) - D.radius) > 1e-9 * D.radius:
        raise Refused(f"{family}: the centre {tuple(c)} does not lie on the {k}")
    if k == "torus" and not all(s.a <= ci <= s.b for ci, s in zip(c, D.sides)):
        raise Refused(f"{family}: the centre {tuple(c)} lies outside the torus' fundamental box")
    return c


def _radius(family, D, c, radius):
    """The support radius: default half the room to the boundary (or to the injectivity radius); the closed ball must
    lie inside the domain at a positive distance from the boundary."""
    room, inj = geo.room(D, c), geo.injectivity(D)
    if radius is None:
        radius = 0.5 * min(room, inj)
    radius = _real(family, "radius", radius)
    if not radius > 0:
        raise Refused(f"{family}: radius must be positive (got {radius:g})")
    if radius >= room or radius >= inj:
        raise Refused(f"{family}: radius {radius:g} leaves the domain (room to the boundary {room:g}, largest geodesic "
                      f"ball {inj:g}); the support lies inside")
    return radius, room


# --------------------------------------------------------------------------------------------- the builder
def _shell_mass(D, g, lo, hi, panels=8, nodes=64):
    """The integral over D of g(distance(c, x)) restricted to lo <= distance <= hi (0 <= lo < hi, inside the injectivity
    radius): composite Gauss-Legendre (panels x nodes) in r directly on [lo, hi], with the radial weight of
    geometry.radial_mass (area of the sphere of radius r: SPHERE_AREA[d] r^(d-1) in R^d, 2 pi R sin(r/R) on the sphere of
    radius R, 2 on the circle). Integrating on [lo, hi] itself, not as radial_mass(hi) - radial_mass(lo), keeps the
    accuracy when g oscillates on [0, lo] (a taper whose flat top is nearly as wide as its support)."""
    k = geo.kind(D)
    x, w = gauss(nodes)
    edges = np.linspace(lo, hi, panels + 1)
    total = 0.0
    for a, b in itertools.pairwise(edges):
        r = 0.5 * (b - a) * (x + 1) + a
        wr = 0.5 * (b - a) * w
        if k == "sphere":
            R = D.radius
            total += 2 * math.pi * R ** 2 * np.sum(wr * g(r) * np.sin(r / R) / R)
        elif k == "circle":
            total += 2.0 * np.sum(wr * g(r))
        else:
            d = geo.ambient(D)
            total += geo.SPHERE_AREA[d] * np.sum(wr * g(r) * r ** (d - 1))
    return float(total)


def _masses(D, radius, breaks, pieces):
    """(integral, squared L2 norm) over D of the profile that equals the smooth function pieces[i](t) for t in
    [breaks[i], breaks[i+1]] (t = distance/radius). Each piece is integrated by Gauss-Legendre on its own stretch
    [breaks[i] radius, breaks[i+1] radius] (the first, starting at the centre, by geometry.radial_mass), so that no kink
    falls inside a panel and no cancellation between two integrals from 0 occurs."""
    out = []
    for power in (1, 2):
        total = 0.0
        for lo, hi, q in zip(breaks[:-1], breaks[1:], pieces):
            def g(r, q=q, power=power):
                return q(np.asarray(r, float) / radius) ** power
            if lo > 0:
                total += _shell_mass(D, g, lo * radius, hi * radius)
            else:
                total += geo.radial_mass(D, g, hi * radius)
        out.append(total)
    return out


def _build(D, family, params, c, radius, room, p, dp, order, regularity_reason, peak_why, pieces=None):
    """The BankFunction of a radial profile p(t), t = distance/radius in [0, 1) (dp = dp/dt or None). pieces =
    (breaks, [smooth functions of t]) when p has kinks inside (0, 1); default the one piece p."""
    def profile(r):
        t = np.asarray(r, float) / radius
        out = np.zeros_like(t)
        m = t < 1
        out[m] = p(t[m])
        return out

    def fn(*X):
        return profile(geo.distance(D, c, *X))

    grad = None
    if dp is not None and geo.kind(D) in FLAT:
        def grad(*X):
            P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
            r = np.linalg.norm(P - c, axis=-1)
            t = r / radius
            g = np.zeros_like(t)
            m = (t < 1) & (r > 0)
            g[m] = dp(t[m]) / (radius * r[m])                         # the gradient is dp/dr * (x - c)/r, 0 at r = 0
            return g[..., None] * (P - c)

    integral, l2 = _masses(D, radius, *(pieces or ([0.0, 1.0], [p])))
    kinks = tuple(float(b) * radius for b in (pieces[0] if pieces else ()) if 0.0 < b < 1.0)
    closed = {"integral": (integral, DERIVED, RADIAL + ", one piece per smooth stretch of the profile"),
              "l2_norm_sq": (l2, DERIVED, RADIAL + ", one piece per smooth stretch of the profile"),
              "max": (1.0, DERIVED, peak_why)}
    return BankFunction(
        family, D, {"centre": tuple(c), "radius": radius, **params}, fn,
        Support("interior", f"the closed geodesic ball of radius {radius:g} about {tuple(np.round(c, 6))}",
                tuple(c), radius, breaks=kinks),
        Regularity(order, piecewise=True, vanishes_at_boundary="all orders"), closed, grad,
        reasons={"regularity": regularity_reason,
                 "support": f"t = distance/radius < 1 inside, 0 outside; room to the boundary {room:g} > radius"})


def _examples(full, order_of, kinked=lambda e: False):
    """The registered examples: every parameter set of full(D). (Until plan 021 P2 the sets whose interior kinks the
    contract's bounding-box reference integral could not resolve were left out; the support now declares its kink
    radii, Support.breaks, and the contract integrates in polar coordinates split there.)"""
    return full


def _specs(D, *specs):
    """The default parameters plus one example per spec (where, fraction of the room, extra parameters): where is
    "default" (the domain's centre) or "shifted" (a second centre)."""
    c = geo.centre(D)
    room = min(geo.room(D, c), geo.injectivity(D))
    shifted = _shifted(D, c)
    out = [{}]
    for where, frac, extra in specs:
        centre = {} if where == "default" or shifted is None else {"centre": tuple(shifted)}
        out.append({**centre, "radius": frac * room, **extra})
    return out


def _shifted(D, c):
    """A second centre of the domain, at least 0.8 of the default centre's room from the boundary (None: no second
    centre; the default one is used)."""
    k = geo.kind(D)
    if k == "sphere":
        return np.asarray(D.centre, float) + D.radius * np.array([math.sin(0.7), 0.0, math.cos(0.7)])
    if k == "circle":
        return np.asarray(D.centre, float) + D.radius * np.array([math.cos(0.7), math.sin(0.7)])
    if k == "torus":
        return c + 0.1 * np.array([s.b - s.a for s in D.sides])
    if k == "shell":
        return c + np.array([0.0, 0.1 * (D.outer - D.inner), 0.0])
    return None


# --------------------------------------------------------------------------------------------- cosine taper
def _cosine_full(D):
    out = _specs(D, ("default", 0.3, {}), ("shifted", 0.45, {}), ("default", 0.6, {}))
    for e, frac in zip(out[1:], (0.1, 0.5, 0.0)):                      # inner as a fraction of the radius
        e["inner"] = frac * e["radius"]
    return out


@register("cosine_taper", TAPER_KINDS,
          "cosine (Tukey) taper: 1 up to `inner`, then (1 + cos(pi s))/2 down to 0 at `radius`; C^1, piecewise",
          references=(HARRIS,), examples=_examples(_cosine_full, lambda e: 1, lambda e: e.get("inner", 1.0) > 0))
def cosine_taper(D, *, centre=None, radius=None, inner=None):
    """p(r) = 1 for r <= inner; (1 + cos(pi (r - inner)/(radius - inner)))/2 for inner < r < radius; 0 for r >= radius.

    r is the geodesic distance from `centre` (default: the domain's centre). `radius` (default: half the room to the
    boundary) must leave the closed ball inside the domain; 0 <= `inner` < `radius` (default: radius/2); inner = 0 is
    raised_cosine. Support: the closed ball of `radius`. Regularity: C^1, piecewise. With W = radius - inner (a length)
    and s = (r - inner)/W, dp/dr = -(pi/(2W)) sin(pi s) vanishes at r = inner and r = radius while d2p/dr2 =
    -(pi/W)^2 cos(pi s)/2 jumps there; p vanishes to all orders at the domain's boundary. The integral and the squared
    L2 norm are integrated on [inner, radius] directly, so they are accurate for any inner < radius."""
    c = _centre("cosine_taper", D, centre)
    radius, room = _radius("cosine_taper", D, c, radius)
    inner = 0.5 * radius if inner is None else _real("cosine_taper", "inner", inner)
    if not 0.0 <= inner < radius:
        raise Refused(f"cosine_taper: inner must satisfy 0 <= inner < radius (inner {inner:g}, radius {radius:g})")
    a = inner / radius                                                 # the flat top's share of the radius
    w = 1.0 - a                                                        # the taper's width in t = r/radius

    def p(t):
        out = np.ones_like(t)
        m = t > a
        out[m] = 0.5 * (1.0 + np.cos(math.pi * (t[m] - a) / w))
        return out

    def dp(t):
        out = np.zeros_like(t)
        m = t > a
        out[m] = -0.5 * math.pi / w * np.sin(math.pi * (t[m] - a) / w)
        return out

    def taper(t):
        return 0.5 * (1.0 + np.cos(math.pi * (t - a) / w))

    pieces = ([0.0, a, 1.0], [np.ones_like, taper]) if a > 0 else None
    return _build(D, "cosine_taper", {"inner": inner}, c, radius, room, p, dp, 1,
                  f"{HARRIS}: p = 1 on [0, inner], p(radius) = 0; with W = radius - inner and s = (r - inner)/W, "
                  f"dp/dr = -(pi/(2W)) sin(pi s) is 0 at r = inner and at r = radius while d2p/dr2 = "
                  f"-(pi/W)^2 cos(pi s)/2 jumps from 0 to -(pi/W)^2/2 at inner and from (pi/W)^2/2 to 0 at radius, so "
                  f"p is C^1, not C^2; {COMPOSITION}",
                  "p = 1 on the flat top [0, inner] and p decreases from there to 0", pieces)


# --------------------------------------------------------------------------------------------- raised cosine
def _raised_full(D):
    return _specs(D, ("default", 0.3, {}), ("shifted", 0.45, {}))


@register("raised_cosine", TAPER_KINDS,
          "raised cosine (Hann) (1 + cos(pi t))/2 of a geodesic ball, t = distance/radius; C^1, piecewise",
          references=(HARRIS,), examples=_examples(_raised_full, lambda e: 1))
def raised_cosine(D, *, centre=None, radius=None):
    """p(r) = (1 + cos(pi r/radius))/2 for r < radius, 0 for r >= radius, r the geodesic distance from `centre`
    (default: the domain's centre; `radius` default: half the room to the boundary). Support: the closed ball of
    `radius`. Regularity: C^1, piecewise (p' = -(pi/(2 radius)) sin(pi t) is 0 at r = radius, p'' jumps there); p is an
    even entire function of r, so smooth at the centre. Vanishes to all orders at the domain's boundary."""
    c = _centre("raised_cosine", D, centre)
    radius, room = _radius("raised_cosine", D, c, radius)

    def p(t):
        return 0.5 * (1.0 + np.cos(math.pi * t))

    def dp(t):
        return -0.5 * math.pi * np.sin(math.pi * t)

    return _build(D, "raised_cosine", {}, c, radius, room, p, dp, 1,
                  f"{HARRIS}: p(radius) = 0, p' = -(pi/(2 radius)) sin(pi t) = 0 at t = 1 while p'' = "
                  f"-(pi/radius)^2 cos(pi t)/2 jumps from (pi/radius)^2/2 to 0, so p is C^1, not C^2; {COMPOSITION}",
                  "p(0) = 1 and p decreases on [0, radius]")


# --------------------------------------------------------------------------------------------- smoothstep
# S_n = (2n+1)!/(n!)^2 int_0^t (u(1-u))^n du, ascending coefficients (tested against scipy.special.betainc(n+1, n+1, t))
SMOOTHSTEP = {1: (0.0, 0.0, 3.0, -2.0),
              2: (0.0, 0.0, 0.0, 10.0, -15.0, 6.0),
              3: (0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0)}


def _smoothstep_full(D):
    return _specs(D, ("default", 0.3, {"order": 2}), ("shifted", 0.45, {"order": 3}), ("default", 0.6, {"order": 1}))


@register("smoothstep", TAPER_KINDS,
          "polynomial smoothstep 1 - S_n(t) (n = 1: 3t^2 - 2t^3, 2, 3) of a geodesic ball; C^n, piecewise",
          references=(BETA,), examples=_examples(_smoothstep_full, lambda e: e.get("order", 1)))
def smoothstep(D, *, centre=None, radius=None, order=1):
    """p(r) = 1 - S_order(t) for t = r/radius < 1, 0 for t >= 1, with the smoothstep polynomials S_1 = 3t^2 - 2t^3,
    S_2 = 10t^3 - 15t^4 + 6t^5, S_3 = 35t^4 - 84t^5 + 70t^6 - 20t^7 (order in {1, 2, 3}, default 1). r is the geodesic
    distance from `centre` (default: the domain's centre; `radius` default: half the room to the boundary). Support:
    the closed ball of `radius`. Regularity: C^order, piecewise (S_n' = c (t(1-t))^n has a zero of order n at t = 1;
    S_n^(n+1) jumps there). Vanishes to all orders at the domain's boundary."""
    c = _centre("smoothstep", D, centre)
    order = _integer("smoothstep", "order", order, tuple(SMOOTHSTEP))
    radius, room = _radius("smoothstep", D, c, radius)
    S = np.polynomial.Polynomial(SMOOTHSTEP[order])
    dS = S.deriv()

    return _build(D, "smoothstep", {"order": order}, c, radius, room, lambda t: 1.0 - S(t), lambda t: -dS(t), order,
                  f"{BETA}; here n = {order}; {COMPOSITION}",
                  "p(0) = 1 - S_n(0) = 1 and p decreases on [0, 1]")


# --------------------------------------------------------------------------------------------- B-spline profile
def cardinal_bspline(x, degree, deriv=0):
    """The cardinal B-spline of `degree` (knots at the integers shifted to centre the support: support
    [-(degree+1)/2, (degree+1)/2], integral 1, sum_k B(x - k) = 1) or its derivative of order deriv < degree, by
    B(x) = 1/degree! sum_{k=0}^{degree+1} (-1)^k C(degree+1, k) (x + (degree+1)/2 - k)_+^degree."""
    n = degree + 1
    u = np.asarray(x, float) + 0.5 * n
    m = degree - deriv
    out = np.zeros_like(u)
    for k in range(n + 1):
        out += (-1) ** k * math.comb(n, k) * np.maximum(u - k, 0.0) ** m
    return out / math.factorial(m)


def _bspline_full(D):
    return _specs(D, ("default", 0.3, {"degree": 1}), ("shifted", 0.45, {"degree": 4}), ("default", 0.6, {"degree": 5}))


def _bspline_piece(m, degree, h, b0):
    """The polynomial of t that the profile equals for u = h (1 + t) in [m, m + 1]: the truncated powers k <= m."""
    n = degree + 1

    def q(t):
        u = h * (1.0 + np.asarray(t, float))
        return sum((-1) ** k * math.comb(n, k) * (u - k) ** degree for k in range(m + 1)) / (math.factorial(degree) * b0)
    return q


@register("bspline_profile", TAPER_KINDS,
          "cardinal B-spline of degree 1..5 as a radial profile, support the geodesic ball of `radius`, peak 1; "
          "C^(degree-1), piecewise", references=(DEBOOR,),
          examples=_examples(_bspline_full, lambda e: e.get("degree", 3) - 1))
def bspline_profile(D, *, centre=None, radius=None, degree=3):
    """p(r) = B(t (degree + 1)/2) / B(0), t = r/radius < 1, 0 for t >= 1, B the cardinal B-spline of `degree` in
    1..5 (default 3, the cubic) with support [-(degree+1)/2, (degree+1)/2]; the division by B(0) makes the peak 1
    (B(0) = 1, 3/4, 2/3, 115/192, 11/20 for degree 1..5). r is the geodesic distance from `centre` (default: the
    domain's centre; `radius` default: half the room to the boundary). Support: the closed ball of `radius`.
    Regularity: C^(degree-1), piecewise (B has simple knots, so its (degree)-th derivative jumps at each; for odd
    degree the knot at 0 gives a term |r|^degree at the centre, still C^(degree-1)). Degree 1 is the cone 1 - t, which
    is continuous with a kink at the centre and at the boundary of the support; it has no gradient claim. Vanishes to
    all orders at the domain's boundary."""
    c = _centre("bspline_profile", D, centre)
    degree = _integer("bspline_profile", "degree", degree, (1, 2, 3, 4, 5))
    radius, room = _radius("bspline_profile", D, c, radius)
    h, b0 = 0.5 * (degree + 1), float(cardinal_bspline(0.0, degree))

    def p(t):
        return cardinal_bspline(h * t, degree) / b0

    dp = None if degree == 1 else (lambda t: h * cardinal_bspline(h * t, degree, 1) / b0)
    n = degree + 1
    m0 = math.floor(h)
    breaks_u = [h] + list(range(m0 + 1, n + 1))                           # the knots in u = x + h, from the centre out
    pieces = ([(u - h) / h for u in breaks_u],
              [_bspline_piece(m, degree, h, b0) for m in range(m0, n)])
    return _build(D, "bspline_profile", {"degree": degree}, c, radius, room, p, dp, degree - 1,
                  f"{DEBOOR}; here degree {degree}, so C^{degree - 1}; {COMPOSITION}",
                  "the cardinal B-spline is symmetric and nonincreasing on [0, inf) (Schoenberg 1973), so the maximum "
                  "B(0)/B(0) = 1 is at the centre", pieces)


# the full example lists (the registered ones leave out what the contract's reference integral cannot verify, _examples)
FULL_EXAMPLES = {"cosine_taper": _cosine_full, "raised_cosine": _raised_full, "smoothstep": _smoothstep_full,
                 "bspline_profile": _bspline_full}
