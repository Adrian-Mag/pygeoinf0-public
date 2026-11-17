"""Functions along curves (plan 021 P1, work package I; EXP-037 DESIGN.md 5, 6): arc-length bumps and Gaussians.

    curve_bump(D, *, centre=None, radius=None, k=1.0)   exp(-k / (1 - t^2)), t = |s - centre| / radius < 1, else 0
    curve_gaussian(D, *, centre=None, width=None)       exp(-(s - centre)^2 / (2 width^2))

D is a Segment, a Polyline, an Arc or a Curve (kind "curve"). s is the arc length, the natural coordinate of these
domains (L2 and Sobolev spaces on them are the interval realisations in s); `centre` is an arc length in [0, length].
The functions are written in s, so they are C-infinity in s also across the vertices of a Polyline, where the
restriction of a smooth ambient field would have a kink. A bank function takes Cartesian coordinates: the arc length of a
point is D.from_cartesian (nearest point of the curve), and a point that is not on the curve (relative distance above
1e-8 of the length) gives NaN. Curves are injective (Curve's own requirement), so s is well defined on them.

On a closed Curve the distance is the periodic minimum image |s - centre| mod length (a bump needs radius < length / 2),
and the Gaussian is the periodic sum over ALL images, f(s) = sum_{n in Z} exp(-(s - centre + n length)^2 / (2 width^2)),
which converges with every derivative locally uniformly: a C-infinity, exactly periodic function. It is evaluated by the
sum over |n| <= N = max(1, ceil(9 width / length)) with s reduced to [0, length]: the dropped terms are below 1e-17
(absolute, the value at the centre is at least 1), so the evaluated values are those of f to that accuracy and, being
a truncation, show a seam jump of the same size (below 1e-17). The closed forms are those of the full sum (exact), not
of the truncation: integral = width sqrt(2 pi) and l2_norm_sq = width sqrt(pi) sum_m exp(-(m length)^2 / (4 width^2)).
The shared one-dimensional helpers (wrapped_*) are used by the torus Gaussian of curves_solid.py.

From curves_solid.py (merged here by plan 063):
Functions on the cylinder and the torus (plan 021 P1, work package I; EXP-037 DESIGN.md 5, 6).

    cylinder_product(D, *, centre=None, radius=None, half_height=None, k_radial=1.0, k_axial=1.0)
        exp(-k_radial / (1 - (r/radius)^2)) * exp(-k_axial / (1 - ((z - zc)/half_height)^2)), r the distance from the
        axis through `centre`, zero outside the product set {r < radius} x {|z - zc| < half_height}
    torus_gaussian(D, *, centre=None, width=None)
        the periodic Gaussian: prod_i sum_{n in Z} exp(-(x_i - c_i + n L_i)^2 / (2 width^2)), L_i the periods (the full
        sum, C-infinity and periodic; evaluated by a truncation accurate to 1e-17 per axis)

A torus bump is the bump of functions.families.smooth (kind "torus" there: the bump in the periodic minimum-image
distance), so only the Gaussian, which smooth.gaussian does not give on a torus, is added here. The one-dimensional
wrapped-sum helpers are those of curves.py.
"""


import itertools
import math

import numpy as np

from ...core.claims import DERIVED
from ...core.errors import Refused
from ...quadrature import gauss
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register

HORMANDER = "Hormander 1990, The Analysis of Linear Partial Differential Operators I, 1.2 (exp(-1/(1-t^2)) is C-infinity)"
ARC = ("radial reduction in arc length: 2 radius int_0^1 exp(-p k/(1-t^2)) dt (p = 1, 2), Gauss-Legendre on panels graded "
       "geometrically toward t = 1 (see bump_moment)")
MAX_K_FORMS = 300.0          # the closed forms of the bump families are stated for k up to this (exp(-2k) must not underflow)
MIN_K_FORMS = 1e-12          # ... and verified from this k on (relative 1e-13); below it the graded panels still cover [0, 1)
MAX_WIDTHS = 10.0            # a wrapped Gaussian wider than this many periods is refused (cost of the periodic sum)
_GL = gauss(64)


# ------------------------------------------------------------------------------------------- shared helpers
def number(family, name, value, positive=False):
    """float(value), or Refused with the reason when it is not a finite number (positive when asked)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise Refused(f"{family}: {name} must be a number (got {value!r})") from None
    if not math.isfinite(v) or (positive and not v > 0):
        raise Refused(f"{family}: {name} must be {'positive and ' if positive else ''}finite (got {v})")
    return v


def bump_moment(k, moment=0, power=1, bulk=8, levels=60):
    """int_0^1 t^moment exp(-power k / (1 - t^2)) dt, moment 0 or 1, for k > 0.

    The integrand is smooth but, for small k, has a steep edge at t = 1 (it varies on the scale k there). With
    u = 1 - t, 1 - t^2 = u (2 - u) (no cancellation), the part t in [0, 1/2] is cut into `bulk` equal panels and the edge
    u in (0, 1/2] into the geometric panels [2^-(j+1), 2^-j], j = 1..`levels`; every panel carries 64 Gauss-Legendre
    nodes. On a panel [a, 2a] the integrand is analytic at distance a from its singularity at u = 0, so the rule converges
    geometrically on each; the part u < 2^-levels is dropped (below 1e-18 in absolute terms). Verified against
    scipy.integrate.quad (epsrel 2.3e-14, one call per panel) to 1.6e-14 relative for 1e-12 <= power k <= 600."""
    x, w = _GL

    def panel(lo, hi):
        u = 0.5 * (hi - lo) * (x + 1) + lo
        return 0.5 * (hi - lo) * np.sum(w * (1.0 - u) ** moment * np.exp(-power * k / (u * (2.0 - u))))
    edge = sum(panel(2.0 ** -(j + 1), 2.0 ** -j) for j in range(1, levels + 1))
    edges = np.linspace(0.5, 1.0, bulk + 1)
    return float(edge + sum(panel(lo, hi) for lo, hi in itertools.pairwise(edges)))


def bump_profile(r, radius, k):
    """exp(-k / (1 - t^2)) for t = r / radius < 1, else 0 (r an array of distances)."""
    t = np.asarray(r, float) / radius
    out = np.zeros_like(t)
    m = t < 1
    out[m] = np.exp(-k / (1 - t[m] ** 2))
    return out


def wrapped_images(width, period):
    """N: the periodic sum over |n| <= N images. For points and centres in one period the first dropped image is at
    distance >= N * period >= 9 width, so the dropped terms are below 2 sum_{j>=N} exp(-(j period)^2 / (2 width^2)) <
    1e-17 (the sum's value at the centre is at least 1)."""
    return max(1, math.ceil(9.0 * width / period))


def wrapped_sum(x, c, w, period, N, slope=False):
    """sum_{|n|<=N} exp(-(x - c + n period)^2 / (2 w^2)), or its x-derivative (slope=True)."""
    d = np.asarray(x, float)[..., None] - c + period * np.arange(-N, N + 1)
    g = np.exp(-d * d / (2 * w * w))
    return np.sum(-d / (w * w) * g, axis=-1) if slope else np.sum(g, axis=-1)


def wrapped_integral(w):
    """int over one period of the full periodic sum sum_{n in Z} exp(-(x - c + n period)^2 / (2 w^2)): the images tile
    the line, so it is the full-line integral w sqrt(2 pi), whatever the period and the centre c."""
    return w * math.sqrt(2.0 * math.pi)


def wrapped_l2(w, period):
    """int over one period of (full periodic sum)^2 = sum_m int_R g(x) g(x + m period) dx
    = w sqrt(pi) sum_{m in Z} exp(-(m period)^2 / (4 w^2)) (unfolding one factor over the line; g the Gaussian about 0).
    The sum is cut at |m| <= M = max(1, ceil(14 w / period)): the first dropped term is below exp(-49) = 5e-22 of the
    m = 0 term and the terms decay at least geometrically, so the cut sum is accurate to 1e-20 relative."""
    M = max(1, math.ceil(14.0 * w / period))
    return w * math.sqrt(math.pi) * sum(math.exp(-((m * period) ** 2) / (4.0 * w * w)) for m in range(-M, M + 1))


# ------------------------------------------------------------------------------------------- curve geometry
def _curve_data(family, D, centre):
    """(length, closed, centre s0) of the curve; Refused when the centre is not an arc length of the curve."""
    L, closed = float(D.volume), bool(D.periodic[0])
    s0 = 0.5 * L if centre is None else number(family, "centre", centre)
    if not 0.0 <= s0 <= L:
        raise Refused(f"{family}: the centre is an arc length in [0, {L:g}] (got {s0:g})")
    return L, closed, s0


def _arclength(D, X):
    return np.asarray(D.from_cartesian(*X)[0], float)


def _s_distance(s, s0, L, closed):
    d = np.abs(s - s0)
    return np.minimum(d, L - d) if closed else d


def _cartesian(D, s):
    return tuple(float(np.ravel(a)[0]) for a in D.to_cartesian(np.array([float(s)])))


# ------------------------------------------------------------------------------------------- curve_bump
def _bump_examples(D):
    L = float(D.volume)
    return [{}, {"centre": 0.3 * L, "radius": 0.2 * L, "k": 2.0}, {"centre": 0.5 * L, "radius": 0.45 * L, "k": 0.5}]


@register("curve_bump", ("curve",),
          "C-infinity bump exp(-k/(1-t^2)) in arc length along a Segment, Polyline, Arc or Curve, t = |s - centre|/radius",
          references=(HORMANDER,), examples=_bump_examples)
def curve_bump(D, *, centre=None, radius=None, k=1.0):
    """f(x) = exp(-k / (1 - t^2)) for t = |s(x) - centre| / radius < 1, else 0, s the arc length of x on the curve.

    `centre`: arc length of the middle (default: half the length); `radius`: arc-length radius (default: half the
    room to the nearer end, a quarter of the length on a closed curve); k > 0 (default 1). The support lies inside the
    curve at positive arc length from its ends (radius < centre, radius < length - centre; on a closed curve
    radius < length / 2). Support: |s - centre| <= radius. Regularity: C-infinity in s, all derivatives vanish at the
    ends. Closed forms (derived): integral = 2 int_0^radius exp(-k/(1-(r/radius)^2)) dr, l2_norm_sq the same with
    exp(-2k/(...)), both by quadrature graded toward the steep edge (bump_moment: relative error below 1e-13 for
    1e-12 <= k <= 300; they are not stated for k > 300, where exp(-2k) underflows); max = exp(-k) at the centre."""
    L, closed, s0 = _curve_data("curve_bump", D, centre)
    room = 0.5 * L if closed else min(s0, L - s0)
    radius = 0.5 * room if radius is None else number("curve_bump", "radius", radius, positive=True)
    k = number("curve_bump", "k", k, positive=True)
    if not radius > 0:
        raise Refused(f"curve_bump: no room for a bump at the arc length {s0:g} of a curve of length {L:g}")
    if radius >= room:
        raise Refused(f"curve_bump: radius {radius:g} leaves the curve (room from arc length {s0:g} to "
                      f"{'the largest period ball' if closed else 'the nearer end'}: {room:g}); a bump's support lies "
                      "inside")

    def fn(*X):
        s = _arclength(D, X)
        return np.where(np.isnan(s), np.nan, bump_profile(_s_distance(s, s0, L, closed), radius, k))

    def contains(*X):
        return _s_distance(_arclength(D, X), s0, L, closed) <= radius

    c = _cartesian(D, s0)
    closed_forms = {"max": (math.exp(-k), DERIVED, "the profile's maximum, at the centre")}
    if k <= MAX_K_FORMS:
        closed_forms["integral"] = (2 * radius * bump_moment(k), DERIVED, ARC)
        closed_forms["l2_norm_sq"] = (2 * radius * bump_moment(k, power=2), DERIVED, ARC + ", squared profile")
    return BankFunction(
        "curve_bump", D, {"centre": s0, "radius": radius, "k": k}, fn,
        Support("interior", f"the arc |s - {s0:g}| <= {radius:g} (arc length s)", c, radius, contains),
        Regularity(math.inf, vanishes_at_boundary="all orders"), closed_forms, None,
        reasons={"regularity": HORMANDER,
                 "support": f"t = |s - centre|/radius < 1 inside; room {room:g} > radius, so the ends are not reached"})


# ------------------------------------------------------------------------------------------- curve_gaussian
def _gaussian_examples(D):
    L = float(D.volume)
    return [{}, {"centre": 0.3 * L, "width": 0.05 * L}, {"centre": 0.5 * L, "width": 0.5 * L}]


@register("curve_gaussian", ("curve",),
          "Gaussian exp(-(s-centre)^2/(2 width^2)) in arc length along a Segment, Polyline, Arc or Curve",
          examples=_gaussian_examples)
def curve_gaussian(D, *, centre=None, width=None):
    """f(x) = exp(-(s(x) - centre)^2 / (2 width^2)), s the arc length of x on the curve.

    `centre`: arc length in [0, length] (default: half the length); `width` > 0 (default: a tenth of the length).
    Support: the whole curve (positive everywhere). Regularity: C-infinity in s up to the ends; no claim of vanishing
    there. Closed forms (derived), open curve: integral = w sqrt(pi/2) (erf((L - c)/(w sqrt 2)) + erf(c/(w sqrt 2))),
    l2_norm_sq the same with width / sqrt 2. On a closed Curve f is the full periodic sum over the images (module
    docstring; width at most 10 lengths), evaluated by a truncation accurate to 1e-17, with the exact integral
    w sqrt(2 pi) and squared norm w sqrt(pi) sum_m exp(-(m L)^2 / (4 w^2)) of the full sum."""
    L, closed, s0 = _curve_data("curve_gaussian", D, centre)
    w = 0.1 * L if width is None else number("curve_gaussian", "width", width, positive=True)
    if closed and w > MAX_WIDTHS * L:
        raise Refused(f"curve_gaussian: width {w:g} exceeds {MAX_WIDTHS:g} lengths of the closed curve ({L:g}); the "
                      "periodic sum is not formed")
    if closed:
        N = wrapped_images(w, L)

        def profile(s):
            return wrapped_sum(s, s0, w, L, N)
        closed_forms = {
            "integral": (wrapped_integral(w), DERIVED,
                         "the images tile the line: one period of the full periodic sum integrates to w sqrt(2 pi)"),
            "l2_norm_sq": (wrapped_l2(w, L), DERIVED,
                           ("unfolding one factor of the square: w sqrt(pi) sum_m exp(-(m L)^2/(4 w^2)), cut at "
                            "relative 1e-20"))}
        why = ("the periodic sum over all images centre + n length, n in Z: each term is entire and the series converges "
               "with every derivative locally uniformly (Weierstrass M-test), so it is C-infinity and exactly periodic; "
               "the evaluation truncates it at |n| <= " + str(N) + " (absolute error below 1e-17)")
    else:
        def profile(s):
            return np.exp(-((s - s0) ** 2) / (2 * w * w))

        def erfdiff(sig):
            z = sig * math.sqrt(2.0)
            return sig * math.sqrt(math.pi / 2) * (math.erf((L - s0) / z) + math.erf(s0 / z))
        closed_forms = {"integral": (erfdiff(w), DERIVED, "w sqrt(pi/2) (erf((L - c)/(w sqrt 2)) + erf(c/(w sqrt 2)))"),
                        "l2_norm_sq": (erfdiff(w / math.sqrt(2.0)), DERIVED,
                                       "the square is a Gaussian of width w/sqrt 2: the same erf formula")}
        why = "an entire function of the arc length s"

    def fn(*X):
        s = _arclength(D, X)
        return np.where(np.isnan(s), np.nan, profile(s))

    return BankFunction("curve_gaussian", D, {"centre": s0, "width": w}, fn,
                        Support("full", "the whole curve (positive everywhere)"), Regularity(math.inf), closed_forms,
                        None, reasons={"regularity": why})


# ---------------------------------------------------------------- from curves_solid.py (plan 063)
def _point(family, D, centre, name="centre"):
    c = geo.centre(D) if centre is None else np.asarray(centre, float).reshape(-1)
    if c.size != geo.ambient(D) or not np.all(np.isfinite(c)):
        raise Refused(f"{family}: {name} needs {geo.ambient(D)} finite Cartesian coordinates (got {centre!r})")
    return c


# ------------------------------------------------------------------------------------------- cylinder_product
def _cylinder_examples(D):
    R, (z0, z1) = D.radius, (D.z.a, D.z.b)
    cx, cy = D.centre
    h = z1 - z0
    # wide supports and larger k: the contract's reference quadrature (polar, 48 points) resolves these to 1e-5
    return [{},
            {"radius": 0.8 * R, "half_height": 0.4 * h, "k_radial": 3.0, "k_axial": 3.0},
            {"centre": (cx + 0.2 * R, cy - 0.1 * R, z0 + 0.45 * h), "radius": 0.7 * R, "half_height": 0.4 * h,
             "k_radial": 2.0, "k_axial": 2.0}]


@register("cylinder_product", ("cylinder",),
          "product of a radial bump about an axis and an axial bump, both exp(-k/(1-t^2)), on a cylinder",
          references=(HORMANDER,), examples=_cylinder_examples)
def cylinder_product(D, *, centre=None, radius=None, half_height=None, k_radial=1.0, k_axial=1.0):
    """f(x, y, z) = exp(-kr / (1 - (r/radius)^2)) exp(-kz / (1 - ((z - zc)/half_height)^2)) inside the product set
    {r < radius} x {|z - zc| < half_height}, else 0; r is the distance of (x, y) from the axis through `centre`.

    `centre` = (cx, cy, zc) (default: the cylinder's centre), `radius`, `half_height` (defaults: 0.9 of the room to the
    lateral wall, 0.9 of the room to the nearer end face), `k_radial`, `k_axial` > 0 (default 1). The product set lies
    inside the cylinder at positive distance from its wall and end faces: radius < R - |(cx, cy) - axis|,
    half_height < min(zc - z0, z1 - zc). Support: that set. Regularity: C-infinity (the radial factor is a smooth
    function of r^2, the axial one of z), every derivative vanishes at the boundary. Closed forms (derived), the
    integral splits into the factors: integral = (2 pi int_0^radius p(r) r dr) (2 int_0^half_height a(u) du), p and a
    the two profiles; l2_norm_sq the same with p^2, a^2 (quadrature graded toward the steep edge, bump_moment:
    relative error below 1e-13 for 1e-12 <= k <= 300; not stated when a k exceeds 300, where exp(-2k) underflows);
    max = exp(-kr - kz) at the centre."""
    c = _point("cylinder_product", D, centre)
    ax = np.asarray(D.centre, float)
    lateral = float(D.radius - np.linalg.norm(c[:2] - ax))
    axial = float(min(c[2] - D.z.a, D.z.b - c[2]))
    if not (lateral > 0 and axial > 0):
        raise Refused(f"cylinder_product: the centre {tuple(c)} is not inside the cylinder (room to the wall {lateral:g}, "
                      f"to the nearer end face {axial:g})")
    rho = 0.9 * lateral if radius is None else number("cylinder_product", "radius", radius, positive=True)
    h = 0.9 * axial if half_height is None else number("cylinder_product", "half_height", half_height, positive=True)
    kr = number("cylinder_product", "k_radial", k_radial, positive=True)
    kz = number("cylinder_product", "k_axial", k_axial, positive=True)
    if rho >= lateral:
        raise Refused(f"cylinder_product: radius {rho:g} leaves the cylinder (room to the lateral wall {lateral:g}); "
                      "the support lies inside")
    if h >= axial:
        raise Refused(f"cylinder_product: half_height {h:g} leaves the cylinder (room to the nearer end face "
                      f"{axial:g}); the support lies inside")
    cx, cy, zc = (float(v) for v in c)

    def fn(*X):
        x, y, z = (np.asarray(v, float) for v in X)
        return bump_profile(np.hypot(x - cx, y - cy), rho, kr) * bump_profile(np.abs(z - zc), h, kz)

    def gradient(*X):
        x, y, z = np.broadcast_arrays(*[np.asarray(v, float) for v in X])
        dx, dy, dz = x - cx, y - cy, z - zc
        tr2, tz2 = (dx * dx + dy * dy) / rho ** 2, (dz / h) ** 2
        inr, inz = tr2 < 1, tz2 < 1
        f = bump_profile(np.hypot(dx, dy), rho, kr) * bump_profile(np.abs(dz), h, kz)
        cr = np.zeros_like(f)
        cz = np.zeros_like(f)
        cr[inr] = -2 * kr / (rho ** 2 * (1 - tr2[inr]) ** 2)
        cz[inz] = -2 * kz / (h ** 2 * (1 - tz2[inz]) ** 2)
        return np.stack([f * cr * dx, f * cr * dy, f * cz * dz], axis=-1)

    def contains(*X):
        x, y, z = (np.asarray(v, float) for v in X)
        return (np.hypot(x - cx, y - cy) <= rho) & (np.abs(z - zc) <= h)

    def mass(power):
        # radial: 2 pi int_0^rho p^power r dr = 2 pi rho^2 int_0^1 t exp(-power kr/(1-t^2)) dt; axial: 2 h int_0^1 ...
        return (2 * math.pi * rho ** 2 * bump_moment(kr, moment=1, power=power)) * (2 * h * bump_moment(kz, power=power))

    why = ("product of the radial integral 2 pi rho^2 int_0^1 t exp(-k_r/(1-t^2)) dt and the axial integral "
           "2 h int_0^1 exp(-k_z/(1-t^2)) dt, each by Gauss-Legendre on panels graded toward t = 1 (bump_moment)")
    closed = {"max": (math.exp(-kr - kz), DERIVED, "the product of the two profiles' maxima, at the centre")}
    if max(kr, kz) <= MAX_K_FORMS:
        closed["integral"] = (mass(1), DERIVED, why)
        closed["l2_norm_sq"] = (mass(2), DERIVED, why + ", profiles squared")
    return BankFunction(
        "cylinder_product", D, {"centre": (cx, cy, zc), "radius": rho, "half_height": h, "k_radial": kr, "k_axial": kz},
        fn, Support("interior", f"the set r <= {rho:g} about the axis through ({cx:g}, {cy:g}), |z - {zc:g}| <= {h:g}",
                    contains=contains),
        Regularity(math.inf, vanishes_at_boundary="all orders"), closed, gradient,
        reasons={"regularity": HORMANDER + "; a product of C-infinity functions of (x, y) and of z",
                 "support": f"room to the wall {lateral:g} > radius, to the nearer end face {axial:g} > half_height"})


# ------------------------------------------------------------------------------------------- torus_gaussian
def _torus_examples(D):
    a = np.array([s.a for s in D.sides])
    L = np.array([s.b - s.a for s in D.sides])
    m = float(L.min())
    return [{}, {"width": 0.05 * m},
            {"centre": tuple(a + 0.02 * L), "width": 0.12 * m},       # near the seam: the images matter
            {"centre": tuple(a + 0.7 * L), "width": 0.4 * m}]


@register("torus_gaussian", ("torus",),
          "periodic Gaussian on a flat torus: the sum over all images of exp(-|x-c|^2/(2 width^2)), evaluated truncated",
          examples=_torus_examples)
def torus_gaussian(D, *, centre=None, width=None):
    """f(x) = prod_i sum_{n in Z} exp(-(x_i - c_i + n L_i)^2 / (2 width^2)), L_i the periods: the full periodic sum.

    `centre` (default: the torus's centre; reduced modulo the periods), `width` > 0 (default: a tenth of the shortest
    period; at most 10 periods). f is the full sum over all images, which converges with every derivative locally
    uniformly (each term is entire, M-test), so f is exactly C-infinity and periodic. It is evaluated with the
    coordinates reduced to one period and the images |n| <= N_i = max(1, ceil(9 width / L_i)): the first dropped image
    is at distance >= 9 width, so the neglected terms are below 1e-17 per axis (absolute; the value at the centre is at
    least 1). The evaluated values are those of f to that accuracy; as a truncation they show a seam jump of that size
    (below 1e-17), which is not a property of f. Support: the whole torus (no boundary). Closed forms (derived) are
    exact for the full sum f, not for the truncation: integral = prod_i w sqrt(2 pi) = (2 pi)^(d/2) w^d (each period
    cell is tiled by the images), l2_norm_sq = prod_i w sqrt(pi) sum_m exp(-(m L_i)^2 / (4 w^2)) (unfolding one
    factor of the square; the sum cut at relative 1e-20)."""
    c = _point("torus_gaussian", D, centre)
    w = 0.1 * min(s.b - s.a for s in D.sides) if width is None else number("torus_gaussian", "width", width, positive=True)
    sides = [(float(s.a), float(s.b)) for s in D.sides]
    Ls = [b - a for a, b in sides]
    if w > MAX_WIDTHS * min(Ls):
        raise Refused(f"torus_gaussian: width {w:g} exceeds {MAX_WIDTHS:g} periods (shortest period {min(Ls):g}); the "
                      "periodic sum is not formed")
    c = np.array([a + math.fmod(ci - a, L) + (L if math.fmod(ci - a, L) < 0 else 0.0)
                  for ci, (a, _), L in zip(c, sides, Ls)])
    Ns = [wrapped_images(w, L) for L in Ls]

    def factors(X, slope_axis=None):
        out = []
        for i, (x, (a, _), L, N) in enumerate(zip(X, sides, Ls, Ns)):
            xr = a + np.mod(np.asarray(x, float) - a, L)
            out.append(wrapped_sum(xr, c[i], w, L, N, slope=(i == slope_axis)))
        return out

    def fn(*X):
        return np.prod(np.broadcast_arrays(*factors(X)), axis=0)

    def gradient(*X):
        cols = []
        for i in range(len(X)):
            cols.append(np.prod(np.broadcast_arrays(*factors(X, slope_axis=i)), axis=0))
        return np.stack(np.broadcast_arrays(*cols), axis=-1)

    why = "product over the axes of w sqrt(2 pi): the images tile each period cell (exact for the full periodic sum)"
    closed = {"integral": (math.prod(wrapped_integral(w) for _ in Ls), DERIVED, why),
              "l2_norm_sq": (math.prod(wrapped_l2(w, L) for L in Ls), DERIVED,
                             ("product over the axes of w sqrt(pi) sum_m exp(-(m L)^2/(4 w^2)) (unfolding one factor of "
                              "the square; full periodic sum; the m-sum cut at relative 1e-20)"))}
    return BankFunction("torus_gaussian", D, {"centre": tuple(float(v) for v in c), "width": w}, fn,
                        Support("full", "the whole torus (positive everywhere, no boundary)"), Regularity(math.inf),
                        closed, gradient,
                        reasons={"regularity": "the full sum over all images: each term is entire and the series converges "
                                               "with every derivative locally uniformly (Weierstrass M-test), so f is "
                                               "C-infinity and periodic; the evaluation truncates it (absolute error "
                                               "below 1e-17 per axis)"})
