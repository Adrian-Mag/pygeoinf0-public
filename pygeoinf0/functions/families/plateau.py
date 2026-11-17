"""The plateau: a product of one-dimensional plateaus, an indicator of a box with optional C-infinity edges.

    plateau(D, *, centre=None, half_width=None, edge=0.0)

Value 1 on the box of half-widths `half_width` (a number or one per axis) about `centre`, with C-infinity edges of width
`edge` outside it (edge = 0: the indicator of the box). Its integral is prod(2 h_i + edge), because int_0^1 S = 1/2 for the
step S of band_common. Defined on interval, box, disk, ball, shell, cylinder, torus: the support lies inside the domain at
positive distance from the boundary by its circumscribed ball, so a box of the shell needs no convex domain (the ball's room
is min(r - inner, outer - r)).
"""
import math

import numpy as np

from ...core.claims import DERIVED
from ...core.errors import Refused
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register
from .band_common import STEP_REFERENCE, SharpFunction, dstep, nonnegative, point, sides, step

KINDS = ("interval", "box", "disk", "ball", "shell", "cylinder", "torus")


def _plateau(D, c, scale, edge, who):
    d = geo.ambient(D)
    span = min(geo.room(D, c), geo.injectivity(D))
    if scale is None:
        h = np.full(d, 0.5 * span / math.sqrt(d) - edge)
    else:
        try:
            arr = np.asarray(scale, float)
            h = np.broadcast_to(arr, (d,)).copy() if arr.ndim <= 1 and arr.size in (1, d) else None
        except (TypeError, ValueError):
            h = None
        if h is None:
            raise Refused(f"{who}: the half-width is a number or {d} numbers (got {scale!r})")
    if not (np.all(np.isfinite(h)) and np.all(h > 0)):
        raise Refused(
            f"{who}: the half-widths must be positive (got {tuple(h)}; with the edge {edge:g} the default "
            f"is {0.5 * span / math.sqrt(d) - edge:g})"
        )
    ext = h + edge
    rho = float(np.linalg.norm(ext))
    if rho >= span:
        raise Refused(
            f"{who}: the box (half-widths {tuple(ext)} with its edge) has circumscribed radius {rho:g}, which "
            f"must be below {span:g}, the room to the boundary / largest geodesic ball"
        )
    if geo.kind(D) == "torus" and any(not a < ci - e and ci + e < b for ci, e, (a, b) in zip(c, ext, sides(D))):
        raise Refused(f"{who}: the box leaves the torus' coordinate ranges {sides(D)}")
    sharp = edge == 0

    def factors(X):
        out = []
        for x, ci, hi in zip(X, c, h):
            x = np.asarray(x, float)
            if sharp:
                out.append((np.abs(x - ci) <= hi).astype(float))
            else:
                out.append(step((x - (ci - hi - edge)) / edge) * step(((ci + hi + edge) - x) / edge))
        return out

    def raw(*X):
        out = 1.0
        for f in factors(X):
            out = out * f
        return out

    grad = None
    if not sharp:

        def grad(*X):
            F = factors(X)
            cols = []
            for i, (x, ci, hi) in enumerate(zip(X, c, h)):
                x = np.asarray(x, float)
                a, b = ci - hi - edge, ci + hi + edge
                dp = (
                    dstep((x - a) / edge) * step((b - x) / edge) - step((x - a) / edge) * dstep((b - x) / edge)
                ) / edge
                for j, f in enumerate(F):
                    if j != i:
                        dp = dp * f
                cols.append(dp)
            return np.stack(np.broadcast_arrays(*cols), axis=-1)

    mass = float(np.prod(2 * h + edge))
    why = (
        f"prod(2 h_i + edge) = {mass:.15g}: each plateau of half-width h_i has two edges of width {edge:g}, each of mass "
        "edge/2 because S(u) + S(1-u) = 1"
    )

    def contains(*X):
        return np.all(
            np.stack(
                np.broadcast_arrays(*[np.abs(np.asarray(x, float) - ci) <= e for x, ci, e in zip(X, c, ext)]), axis=-1
            ),
            axis=-1,
        )

    support = Support(
        "interior",
        f"the closed box of half-widths {tuple(np.round(ext, 6))} about {tuple(np.round(c, 6))}",
        tuple(c),
        rho,
        contains,
    )
    breaks = tuple(sorted({ci - e, ci - hi, ci + hi, ci + e}) for ci, hi, e in zip(c, h, ext))
    reg = Regularity(-1 if sharp else math.inf, vanishes_at_boundary="all orders")
    hs = tuple(float(x) for x in h)
    params = {"centre": tuple(c), "half_width": hs, "edge": edge}
    closed = {"max": (1.0, DERIVED, "the plateau's value")}
    if sharp:
        closed["l2_norm_sq"] = (mass, DERIVED, "the indicator's square is itself")
        return SharpFunction("plateau", D, params, raw, support, reg, closed, grad, breaks=breaks,
                             exact={"integral": (mass, DERIVED, why)})
    closed["integral"] = (mass, DERIVED, why)
    return BankFunction("plateau", D, params, raw, support, reg, closed, grad, reasons={"regularity": STEP_REFERENCE},
                        breaks=breaks)


def _examples(D):
    span = min(geo.room(D, geo.centre(D)), geo.injectivity(D))
    return [{"half_width": 0.2 * span}, {"half_width": 0.2 * span, "edge": 0.1 * span}, {}]


@register("plateau", KINDS, "1 on a box of given half-widths, with optional C-infinity edges outside it",
          references=(STEP_REFERENCE,), examples=_examples)
def plateau(D, *, centre=None, half_width=None, edge=0.0):
    """The plateau of the module docstring. `centre` defaults to the domain's centre; `half_width` to what fits, half the
    room to the boundary across the diagonal, less the edge."""
    who = "plateau"
    return _plateau(D, point(D, centre, who), half_width, nonnegative(edge, "edge", who), who)
