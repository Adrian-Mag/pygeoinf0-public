"""Reference families (plan 021 P0): the C-infinity bump of a geodesic ball, and the Gaussian (EXP-037 DESIGN.md 5).

    bump(D, centre=None, radius=None, k=1.0)   exp(-k / (1 - t^2)) for t = distance / radius < 1, else 0
    gaussian(D, centre=None, width=None)       exp(-|x - c|^2 / (2 width^2)), |.| Euclidean in the ambient space
                                               (chordal on spheres and circles: smooth everywhere, unlike the geodesic
                                               distance, whose square has a kink at the antipode)
"""
import itertools
import math

import numpy as np

from ...core.claims import CHECKED, DERIVED
from ...core.errors import Refused
from ...quadrature import gauss
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register
from .band_common import gl

BUMP_KINDS = ("interval", "box", "disk", "ball", "shell", "sphere", "circle", "cylinder", "torus")
FLAT = ("interval", "box", "disk", "ball", "shell", "cylinder")
HORMANDER = "Hormander 1990, The Analysis of Linear Partial Differential Operators I, 1.2 (exp(-1/(1-t^2)) is C-infinity)"
CLIP_SLACK = 1e-12
CUT = "the two arms of the cut interval, composite Gauss-Legendre (64 nodes per panel) graded towards the ends"
RADIAL = "radial reduction of the integral over a geodesic ball, composite Gauss-Legendre (8 x 64 nodes) in r"


def _centre(D, centre, inside=True):
    if centre is None:
        return geo.centre(D)
    try:
        np.asarray(centre, float)
    except (TypeError, ValueError):
        raise Refused(f"a centre is a point (numbers), not {centre!r}") from None
    return geo.check_centre(D, centre, inside)


def _number(value, name, who):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise Refused(f"{who}: {name} must be a number (got {value!r})") from None


def _bump_examples(D):
    c = geo.centre(D)
    room = min(geo.room(D, c), geo.injectivity(D))
    return [{}, {"radius": 0.3 * room, "k": 2.0}, {"centre": tuple(c), "radius": 0.6 * room, "k": 0.5}]


@register("bump", BUMP_KINDS, "C-infinity bump exp(-k/(1-t^2)) of a geodesic ball, t = distance/radius",
          references=(HORMANDER,), examples=_bump_examples)
def bump(D, centre=None, radius=None, k=1.0, clip=False):
    """The bump of the geodesic ball of `radius` about `centre` (default: the domain's centre, half the room to the
    boundary). Its support lies in the domain: radius <= the room to the boundary (equal, within the relative slack
    1e-12: the support touches the boundary, and the bump still vanishes there to all orders). clip=True on a flat
    domain: the bump restricted to the domain when its ball leaves it (e.g. centred on an end of an interval): C-infinity
    on the closed domain but not zero at the boundary; closed forms then only on an interval (exact cut integrals)."""
    c = _centre(D, centre)
    room, inj = geo.room(D, c), geo.injectivity(D)
    if radius is None:
        radius = 0.5 * min(room, inj)
    radius, k = _number(radius, "radius", "bump"), _number(k, "k", "bump")
    if not (radius > 0 and k > 0):
        raise Refused(f"bump: radius and k must be positive (radius {radius}, k {k})")
    if radius >= inj:
        raise Refused(f"bump: radius {radius:g} reaches the largest geodesic ball {inj:g}; a bump is defined below it")
    cut = radius > room * (1 + CLIP_SLACK)
    if cut and not (clip and geo.kind(D) in FLAT):
        raise Refused(f"bump: radius {radius:g} leaves the domain (room to the boundary {room:g}). Choose: a radius <= "
                      f"{room:g} (the support inside, touching at equality), or clip=True for the bump restricted to "
                      "the domain (flat domains; it does not vanish at the boundary)")
    if not cut and radius > room:
        radius = room                                          # within the slack: the ball touches the boundary

    def profile(r):
        t = np.asarray(r, float) / radius
        out = np.zeros_like(t)
        m = t < 1
        out[m] = np.exp(-k / (1 - t[m] ** 2))
        return out

    def fn(*X):
        return profile(geo.distance(D, c, *X))

    grad = None
    if geo.kind(D) in FLAT:
        def grad(*X):
            P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
            r = np.linalg.norm(P - c, axis=-1)
            t = r / radius
            dp = np.zeros_like(t)
            m = (t < 1) & (r > 0)
            dp[m] = -2 * k * t[m] / (radius * (1 - t[m] ** 2) ** 2) * np.exp(-k / (1 - t[m] ** 2)) / r[m]
            return dp[..., None] * (P - c)

    ball = f"the closed geodesic ball of radius {radius:g} about {tuple(np.round(c, 6))}"
    params = {"centre": tuple(c), "radius": radius, "k": k}
    if not cut:
        closed = {"integral": (geo.radial_mass(D, profile, radius), DERIVED, RADIAL),
                  "l2_norm_sq": (geo.radial_mass(D, lambda r: profile(r) ** 2, radius), DERIVED, RADIAL),
                  "max": (math.exp(-k), DERIVED, "the profile's maximum, at the centre")}
        return BankFunction("bump", D, params, fn,
                            Support("touches" if radius >= room else "interior", ball, tuple(c), radius, breaks=()),
                            Regularity(math.inf, vanishes_at_boundary="all orders"), closed, grad,
                            reasons={"regularity": HORMANDER,
                                     "support": f"t = distance/radius < 1 inside; room to the boundary {room:g} >= radius"})
    closed = {"max": (math.exp(-k), DERIVED, "the profile's maximum, at the centre (a point of the closed domain)")}
    if geo.kind(D) == "interval":
        arms = [min(radius, c[0] - D.a), min(radius, D.b - c[0])]
        cut_mass = lambda p: sum(_arm(p, a) for a in arms if a > 0)
        closed |= {"integral": (cut_mass(profile), DERIVED, CUT), "l2_norm_sq": (cut_mass(lambda r: profile(r) ** 2),
                                                                              DERIVED, CUT)}
    from .indicators import _farthest
    covers = radius >= _farthest(D, c)
    return BankFunction("bump", D, {**params, "clip": True}, fn,
                        Support("full" if covers else "touches", f"{ball} cut by the domain", tuple(c), radius,
                                breaks=()),
                        Regularity(math.inf), closed, grad,
                        reasons={"regularity": HORMANDER + "; restricted to the closed domain (not zero on its boundary)",
                                 "support": f"clip=True: the ball (radius {radius:g}) leaves the domain (room {room:g})"})


def _arm(p, a, panels=8):
    """int_0^a p(r) dr, composite Gauss-Legendre (64 nodes per panel) graded towards a (geometry._graded)."""
    t, w = gauss(64)
    total = 0.0
    for lo, hi in itertools.pairwise(geo._graded(0.0, a, panels)):
        total += float(np.sum(0.5 * (hi - lo) * w * p(0.5 * (hi - lo) * (t + 1) + lo)))
    return total


# --------------------------------------------------------------------------------------------- Gaussian
def _gauss_closed(D, c, w):
    """Closed forms of the integral and the squared L2 norm where they exist (interval, box, sphere, circle); on a curved
    boundary the integral is _gaussian_mass."""
    k = geo.kind(D)
    from math import erf, sqrt

    def flat(sig):
        sides = [(D.a, D.b)] if k == "interval" else [(s.a, s.b) for s in D.sides]
        v = 1.0
        for (a, b), ci in zip(sides, c):
            v *= sig * sqrt(math.pi / 2) * (erf((b - ci) / (sig * sqrt(2))) - erf((a - ci) / (sig * sqrt(2))))
        return v

    def sphere(sig):
        return 2 * math.pi * sig ** 2 * (1 - math.exp(-2 * D.radius ** 2 / sig ** 2))

    def circle(sig):
        from scipy.special import i0e                  # exp(-x) I0(x) without forming I0(x) (it overflows from x ~ 713)
        return 2 * math.pi * D.radius * float(i0e(D.radius ** 2 / sig ** 2))

    form = {"interval": flat, "box": flat, "sphere": sphere, "circle": circle}.get(k)
    if form is None:
        return {}
    why = {"interval": "products of error functions", "box": "products of error functions",
           "sphere": "int_S exp(-|x-c|^2/(2 s^2)) dS = 2 pi s^2 (1 - exp(-2 R^2/s^2)) (|x-c|^2 = 2 R^2 (1 - cos theta))",
           "circle": "R int_0^2pi exp(-R^2 (1 - cos phi)/s^2) dphi = 2 pi R exp(-R^2/s^2) I_0(R^2/s^2)"}[k]
    return {"integral": (form(w), DERIVED, why), "l2_norm_sq": (form(w / math.sqrt(2)), DERIVED, why + "; width/sqrt 2")}


_erf = np.vectorize(math.erf, otypes=[float])


def _radial(d, t, sig):
    """int_0^t exp(-r^2/(2 sig^2)) r^(d-1) dr for d = 2, 3 (closed forms)."""
    t = np.asarray(t, float)
    z = t / (sig * math.sqrt(2.0))
    if d == 2:
        return -sig * sig * np.expm1(-z * z)
    return sig**3 * math.sqrt(math.pi / 2.0) * _erf(z) - sig * sig * t * np.exp(-z * z)


def _converged(f, breaks, what, full):
    """The integral of f over the breaks: composite Gauss-Legendre, panels doubled from 4 to 128 until two successive
    results differ by at most 1e-12 of the result plus 1e-15 of `full`, the Gaussian's mass on the whole space (the
    floor is rounding in the closed forms, which is relative to `full`)."""
    prev = gl(f, breaks, 4)
    for panels in (8, 16, 32, 64, 128):
        cur = gl(f, breaks, panels)
        if abs(cur - prev) <= 1e-12 * abs(cur) + 1e-15 * full:
            return cur
        prev = cur
    raise Refused(f"gaussian: the angular quadrature of the Gaussian's mass on the {what} did not converge")


def _ball_gauss(d, a, s, sig, what):
    """int over the ball (disk if d = 2) of radius a about o of exp(-|x - c|^2/(2 sig^2)), |c - o| = s. In polar
    coordinates about c with the axis through o: only the angle g to that axis matters; the weight of the angle is
    2 (d = 2) or 2 pi sin g (d = 3), the ray from c meets the ball in [t-, t+] (t- = 0 when c is inside)."""

    def weight(g):
        return 2.0 * np.pi * np.sin(g) if d == 3 else 2.0

    if s <= a:  # g from the outward direction: t+ = -s cos g + sqrt(a^2 - s^2 sin^2 g)

        def f(g):
            t = -s * np.cos(g) + np.sqrt(np.maximum(a * a - (s * np.sin(g)) ** 2, 0.0))
            return weight(g) * _radial(d, np.maximum(t, 0.0), sig)

        return _converged(f, [0.0, 0.5 * math.pi, math.pi], what, (2.0 * math.pi * sig * sig) ** (d / 2.0))
    gm = math.asin(a / s)  # c outside: the rays through the ball are at angle g < gm from the direction to o

    def f(v):  # g = gm (1 - v^2) removes the square-root edge at gm
        g = gm * (1.0 - v * v)
        q = np.sqrt(np.maximum(a * a - (s * np.sin(g)) ** 2, 0.0))
        return weight(g) * (_radial(d, s * np.cos(g) + q, sig) - _radial(d, s * np.cos(g) - q, sig)) * 2.0 * gm * v

    return _converged(f, [0.0, 1.0], what, (2.0 * math.pi * sig * sig) ** (d / 2.0))


def _gaussian_mass(D, c, sig):
    """int_D of exp(-|x - c|^2/(2 sig^2)) for a domain with a curved boundary (disk, ball, shell, cylinder): the
    module docstring's polar-coordinate method; the Gaussian of a shell is the one of the outer ball minus the one of
    the inner ball, of a cylinder the product of the disk's and the exact error-function factor in z."""
    k = geo.kind(D)
    o = np.asarray(D.centre, float)
    if k == "disk":
        mass, d = _ball_gauss(2, D.radius, float(np.linalg.norm(c - o)), sig, k), 2
    elif k == "ball":
        mass, d = _ball_gauss(3, D.radius, float(np.linalg.norm(c - o)), sig, k), 3
    elif k == "shell":
        s, d = float(np.linalg.norm(c - o)), 3
        mass = _ball_gauss(3, D.outer, s, sig, k) - _ball_gauss(3, D.inner, s, sig, k)
    else:  # cylinder
        s, d = float(np.linalg.norm(c[:2] - o)), 3
        zf = (
            sig
            * math.sqrt(math.pi / 2.0)
            * (math.erf((D.z.b - c[2]) / (sig * math.sqrt(2.0))) - math.erf((D.z.a - c[2]) / (sig * math.sqrt(2.0))))
        )
        mass = _ball_gauss(2, D.radius, s, sig, k) * zf
    full = (2.0 * math.pi * sig * sig) ** (d / 2.0)
    if not mass > 1e-6 * full:
        raise Refused(
            f"gaussian: the Gaussian of width {sig:g} about {tuple(np.round(c, 6))} has only {mass / full:.2g} of "
            f"its full-space mass in the {k}; its integral cannot be given reliably (move the centre inside)"
        )
    return float(mass)


def _gauss_examples(D):
    c = geo.centre(D)
    size = geo.volume(D) ** (1.0 / geo.dimension(D))
    return [{}, {"width": 0.05 * size}, {"centre": tuple(c), "width": 0.5 * size}]


@register("gaussian", ("interval", "box", "disk", "ball", "shell", "sphere", "circle", "cylinder"),
          "Gaussian exp(-|x-c|^2/(2 width^2)), Euclidean (chordal) distance", examples=_gauss_examples)
def gaussian(D, centre=None, width=None):
    """The Gaussian about `centre` (default: the domain's centre) with standard deviation `width` (default: a tenth of
    the domain's size). Smooth up to the boundary; it does not vanish there. The centre may lie outside a flat domain
    (the function is defined everywhere); on spheres and circles it must lie on the manifold."""
    c = _centre(D, centre, inside=False)
    w = 0.1 * geo.volume(D) ** (1.0 / geo.dimension(D)) if width is None else _number(width, "width", "gaussian")
    if not w > 0:
        raise Refused(f"gaussian: width must be positive (got {w})")

    def fn(*X):
        P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
        return np.exp(-np.sum((P - c) ** 2, axis=-1) / (2 * w * w))

    grad = None
    if geo.kind(D) in FLAT:
        def grad(*X):
            P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
            return -(P - c) / (w * w) * fn(*X)[..., None]

    closed, reasons = _gauss_closed(D, c, w), {"regularity": "the restriction of an entire function of the Cartesian coordinates"}
    if "integral" not in closed:
        try:
            mass = _gaussian_mass(D, c, w)
            closed["integral"] = (mass, CHECKED, "polar coordinates about the centre: the radial integral in closed form, the "
                                  "angular one by composite Gauss-Legendre with doubled panels agreeing to 1e-12 (relative, plus "
                                  f"1e-15 of the full-space mass); the Gaussian truncated to a {geo.kind(D)} has no closed form")
        except Refused as e:
            reasons["integral_refused"] = str(e)
    return BankFunction("gaussian", D, {"centre": tuple(c), "width": w}, fn,
                        Support("full", "the whole domain (positive everywhere)"), Regularity(math.inf), closed, grad,
                        reasons=reasons)
