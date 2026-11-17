"""A band with C-infinity edges, in the radius or in the depth (plan 034; a sharp band is band_indicator, slab_indicator or
box_indicator).

    smooth_band(D, *, lower=None, upper=None, edge=None, coordinate=None)

f(x) = w(s(x)) with s the coordinate: the radius |x - o| about the centre of a disk, ball or shell, or the depth below the top,
which is top - last coordinate on an interval, box or cylinder and outer radius - |x - o| on a disk, ball or shell. The profile
is 1 on the plateau inside [lower, upper], with edges of width `edge` inside that range: w(s) = S((s - lower)/edge) S((upper - s)/edge)
for the C-infinity step S of band_common, 0 < edge <= (upper - lower)/2. On a box or cylinder the band is constant across the
lateral directions, so its support meets the lateral boundary and it claims no vanishing there. Its integral is int w(s) m(s) ds
with m the measure of the level set {coordinate = s}, by Gauss-Legendre with breaks at the edges (derived).
"""
import math
from types import SimpleNamespace

import numpy as np

from ...core.claims import DERIVED
from ...core.errors import Refused
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register
from .band_common import STEP_REFERENCE, dstep, gl, step
from .smooth import FLAT

RADIAL = ("disk", "ball", "shell")
KINDS = ("interval", "box", "disk", "ball", "shell", "cylinder")


def _stack(X):
    return np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)


def _axis(D, depth):
    """The window coordinate s(x) of D and its measure: a namespace with s, grad (of s), density m(s), the range
    [lo, hi] of s, whether the ends are boundary (lo_bd, hi_bd), whether the level sets reach the lateral boundary."""
    k, d = geo.kind(D), geo.ambient(D)
    if k in ("disk", "ball", "shell"):
        o = np.asarray(D.centre, float)
        r_lo, r_hi = (D.inner, D.outer) if k == "shell" else (0.0, D.radius)
        area = geo.SPHERE_AREA[d]
        sign = -1.0 if depth else 1.0

        def radius(*X):
            return np.linalg.norm(_stack(X) - o, axis=-1)

        def grad(*X):
            p = _stack(X) - o
            r = np.linalg.norm(p, axis=-1)
            return sign * p / np.where(r > 0, r, 1.0)[..., None]

        if depth:
            return SimpleNamespace(
                s=lambda *X: r_hi - radius(*X),
                grad=grad,
                lo=0.0,
                hi=r_hi - r_lo,
                density=lambda s: area * (r_hi - s) ** (d - 1),
                lo_bd=True,
                hi_bd=k == "shell",
                lateral=False,
                flat=True,
                name="depth below the outer sphere",
            )
        return SimpleNamespace(
            s=radius,
            grad=grad,
            lo=r_lo,
            hi=r_hi,
            density=lambda s: area * s ** (d - 1),
            lo_bd=k == "shell",
            hi_bd=True,
            lateral=False,
            flat=True,
            name="radius",
        )
    if k == "interval":
        a, b, cross = D.a, D.b, 1.0
    elif k == "box":
        a, b, cross = D.sides[-1].a, D.sides[-1].b, float(np.prod([s.b - s.a for s in D.sides[:-1]]))
    else:
        a, b, cross = D.z.a, D.z.b, math.pi * D.radius**2

    def grad(*X):
        g = np.zeros(np.broadcast(*X).shape + (d,))
        g[..., -1] = -1.0
        return g

    return SimpleNamespace(
        s=lambda *X: b - np.asarray(X[-1], float),
        grad=grad,
        lo=0.0,
        hi=b - a,
        density=lambda s: np.full_like(np.asarray(s, float), cross),
        lo_bd=True,
        hi_bd=True,
        lateral=k != "interval",
        flat=True,
        name="depth below the top (last coordinate)",
    )


def _examples(D):
    out = []
    for coordinate in (("radius", "depth") if geo.kind(D) in RADIAL else ("depth",)):
        ax = _axis(D, coordinate == "depth")
        t = ax.hi - ax.lo
        out += [{"coordinate": coordinate, "lower": ax.lo + 0.1 * t, "upper": ax.lo + 0.9 * t, "edge": 0.4 * t},
                {"coordinate": coordinate, "lower": ax.lo, "upper": ax.hi, "edge": 0.5 * t}]
    return out


def _number(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise Refused(f"smooth_band: {name} must be a number (got {value!r})") from None


@register("smooth_band", KINDS, "1 on a range of the radius or of the depth, with C-infinity edges inside the range",
          references=(STEP_REFERENCE,), examples=_examples)
def smooth_band(D, *, lower=None, upper=None, edge=None, coordinate=None):
    """The band of the module docstring. `coordinate` is "radius" (disk, ball, shell; the default there) or "depth" (the default
    elsewhere). Defaults: the middle half of the coordinate's range, and an edge of a quarter of the band's width."""
    kind = geo.kind(D)
    coordinate = ("radius" if kind in RADIAL else "depth") if coordinate is None else coordinate
    if coordinate not in ("radius", "depth"):
        raise Refused(f"smooth_band: coordinate is 'radius' or 'depth' (got {coordinate!r})")
    if coordinate == "radius" and kind not in RADIAL:
        raise Refused(f"smooth_band: a {kind} has no radius about its centre here; use coordinate='depth'")
    ax = _axis(D, coordinate == "depth")
    span = ax.hi - ax.lo
    s1 = ax.lo + 0.25 * span if lower is None else _number(lower, "lower")
    s2 = ax.lo + 0.75 * span if upper is None else _number(upper, "upper")
    if not (math.isfinite(s1) and math.isfinite(s2) and ax.lo <= s1 < s2 <= ax.hi):
        raise Refused(f"smooth_band: need {ax.lo:g} <= lower < upper <= {ax.hi:g} (the {ax.name} ranges over "
                      f"[{ax.lo:g}, {ax.hi:g}] on this domain); got lower = {s1:g}, upper = {s2:g}")
    edge = 0.25 * (s2 - s1) if edge is None else _number(edge, "edge")
    if not (math.isfinite(edge) and edge > 0):
        raise Refused(f"smooth_band: edge must be positive (got {edge:g}); a sharp band is band_indicator, slab_indicator or "
                      "box_indicator")
    if edge > 0.5 * (s2 - s1) * (1 + 1e-12):
        raise Refused(f"smooth_band: edge {edge:g} exceeds half the width {0.5 * (s2 - s1):g} of the band: no plateau remains")
    edge = min(edge, 0.5 * (s2 - s1))

    def w(s):
        s = np.asarray(s, float)
        return step((s - s1) / edge) * step((s2 - s) / edge)

    def dw(s):
        return (dstep((s - s1) / edge) * step((s2 - s) / edge) - step((s - s1) / edge) * dstep((s2 - s) / edge)) / edge

    breaks = [s1, s1 + edge, s2 - edge, s2]
    mass = gl(lambda s: w(s) * ax.density(s), breaks)
    l2 = gl(lambda s: w(s) ** 2 * ax.density(s), breaks)
    tol = 1e-12 * max(1.0, ax.hi)
    touches = ax.lateral or (ax.lo_bd and s1 <= ax.lo + tol) or (ax.hi_bd and s2 >= ax.hi - tol)

    def fn(*X):
        return w(ax.s(*X))

    def contains(*X):
        s = ax.s(*X)
        return (s >= s1) & (s <= s2)

    grad = None
    if kind in FLAT:
        def grad(*X):
            return dw(ax.s(*X))[..., None] * ax.grad(*X)

    desc = f"lower = {s1:g} <= {ax.name} <= upper = {s2:g}" + (" (the whole cross-section)" if ax.lateral else "")
    support = Support("touches" if touches else "interior", desc, contains=contains)
    why = f"int w(s) m(s) ds = {mass:.15g}, Gauss-Legendre (4 x 64 nodes between the edges)"
    closed = {"integral": (mass, DERIVED, why), "l2_norm_sq": (l2, DERIVED, why.replace("w(s)", "w(s)^2")),
              "max": (1.0, DERIVED, "the plateau's value")}
    regularity = Regularity(math.inf, vanishes_at_boundary=None if ax.lateral else "all orders")
    return BankFunction("smooth_band", D, {"coordinate": coordinate, "lower": s1, "upper": s2, "edge": edge}, fn, support, regularity,
                        closed, grad, reasons={"regularity": STEP_REFERENCE})
