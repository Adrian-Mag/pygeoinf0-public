"""Explicit quadrature rules as realisation options (plan 009 G10, substitution S1 of EXP-030 DESIGN 1.9).

The package integrates piecewise-polynomial objects exactly (Gauss-Legendre per piece). A fixed rule is a different
realisation of the same integrals, chosen on purpose (to reproduce a computation that used it); every result computed
with it states the rule and its measured error against the exact value:

    q = pg.realisations.Trapezoid(1024)      the n-node uniform trapezoid rule on each interval (nodes at both ends)
    q = pg.realisations.Trapezoid(1024, over=(a, b))   one rule on the parent interval [a, b], restricted to whatever
                                             sub-interval it is asked for (the parent's nodes in [a', b') with the
                                             parent's weights; the last piece closed): a computation that used one grid
                                             across pieces of a decomposed space (layers, zones) is reproduced piece by
                                             piece, and the pieces of a partition add up to the parent rule exactly
    q.integrate(f, a, b)                     int_a^b f by the rule (f vectorised)
    q.error_claim(value, exact, what)        the claim that records the rule and |value - exact|

From moments.py (merged here by plan 063):
int g dm for a realised g with kinks or jumps that a fixed Gauss rule on the set misses (plan 044): the realisation's own
coordinates, each axis split at its breaks, inside the natural box that holds the term's support, at rising orders until two
agree. A radial x harmonic g pairs coefficients instead (realisations/series/symbolic_harmonics.realised_moment). None when no such box is
known or the last orders do not agree (a jump of the set itself on no coordinate surface): the caller keeps its rule and its warning.
"""


import numpy as np

from ..core.claims import CHECKED, Claim
from ..quadrature import gauss, trapezoid


class Trapezoid:
    """The n-node uniform trapezoid rule on [a, b] (n >= 2; pygeoinf0.quadrature.trapezoid): weights h/2, h, ..., h,
    h/2 with h = (b - a)/(n - 1). Exact for linear integrands only; its error for smooth ones is O(h^2), and larger
    where the integrand has kinks between nodes. over=(a, b): the rule on that parent interval, restricted to sub-intervals (module docstring)."""

    def __init__(self, n, over=None):
        n = int(n)
        if n < 2:
            raise ValueError("the trapezoid rule needs at least 2 nodes")
        self.n = n
        self.over = None if over is None else (float(over[0]), float(over[1]))
        if self.over is not None and not self.over[1] > self.over[0]:
            raise ValueError("over=(a, b) needs a < b")
        self.name = f"{n}-node trapezoid rule" + ("" if over is None else
                                                  f" on [{self.over[0]:g}, {self.over[1]:g}], restricted")

    def _full(self, a, b):
        return trapezoid(self.n, a, b)

    def nodes(self, a, b):
        """(x, w): the nodes and weights on [a, b] (with over=: the parent's nodes in [a, b), or [a, b] when b is the
        parent's end; [a, b] must lie in the parent interval)."""
        if self.over is None:
            return self._full(a, b)
        A, B = self.over
        tol = 1e-12 * max(abs(A), abs(B), 1.0)
        if a < A - tol or b > B + tol or b < a:
            raise ValueError(f"[{a:g}, {b:g}] is not inside the rule's interval [{A:g}, {B:g}]")
        x, w = self._full(A, B)
        keep = (x >= a) & ((x <= b) if b >= B - tol else (x < b))
        return x[keep], w[keep]

    def integrate(self, f, a, b):
        x, w = self.nodes(a, b)
        return np.tensordot(w, np.asarray(f(x), float), axes=(0, 0))

    def error_claim(self, value, exact, what="the integral"):
        err = np.max(np.abs(np.asarray(value, float) - np.asarray(exact, float)))
        scale = max(float(np.max(np.abs(exact))), 1e-300)
        return Claim(f"{what} by the {self.name} (a realisation option), not exactly: largest difference from the exact "
                     f"value {err:.3g} ({err / scale:.3g} relative)", CHECKED, "measured against Gauss-Legendre per piece")

    def __repr__(self):
        return f"Trapezoid({self.n})"


# ---------------------------------------------------------------- from moments.py (plan 063)
AGREE = 1e-13           # relative difference of two orders that ends the refinement
ACCEPT = 1e-8           # the finest order stands when the last two differ by this; beyond it, the caller's rule and warning
ORDERS = (48, 72, 108, 162, 243)          # points per axis over the box, at least, shared among the cells of the axis
MAX_POINTS = 3e7
SLICE_POINTS = 2e5      # points evaluated at a time, whatever the grid: the memory of a moment stays near this times the basis


def natural_box(m, D):
    """((lo, hi) per natural axis of the domain D) holding the term's support, or None."""
    from .series.symbolic_harmonics import _balls, support_box
    S = m.set
    if S is D or S == D:
        return tuple(D.reference)
    kind = type(D).__name__
    if kind in ("Ball", "Shell"):
        a, b = (float(x) for x in D.reference[0])
        if type(S).__name__ == "Shell" and tuple(S.centre) == tuple(D.centre):
            return (max(a, S.inner), min(b, S.outer)), tuple(D.reference[1]), tuple(D.reference[2])
        return support_box(m, D.centre, a, b)
    if kind == "Box":
        sides = getattr(S, "sides", None)
        if sides is not None:
            return tuple((float(s.a), float(s.b)) for s in sides)
        balls = _balls(m)
        if balls and len(balls) == 1:
            c, rho = balls[0]
            return tuple((max(lo, x - rho), min(hi, x + rho)) for x, (lo, hi) in zip(c, D.reference))
    return None


def _axis_breaks(B, R, dim):
    br = B.breaks(R)
    if br is None and hasattr(B, "cells"):
        br = B.cells.nodes
    if br is None:
        return (None,) * dim
    return tuple(None if b is None else np.asarray(b, float).ravel() for b in br)


def _degree(R, k):
    bases = getattr(R, "bases", None)
    return 0 if bases is None else int(bases[k].n)


def _axis(lo, hi, breaks, order, degree):
    """(nodes, weights) of Gauss rules on the cells of [lo, hi] cut at breaks."""
    inner = [] if breaks is None else [x for x in breaks if lo < x < hi]
    edges = np.r_[lo, inner, hi]
    n = max(int(np.ceil(order / (len(edges) - 1))), degree // 2 + 2, 4)
    t, w = gauss(n)
    h = np.diff(edges)
    return (edges[:-1, None] + 0.5 * h[:, None] * (t + 1.0)).ravel(), (0.5 * h[:, None] * w).ravel()


def _at_order(space, m, R, box, breaks, order):
    from ..spaces.representers import members_at, weight_values
    B, D = space.backend, space.domain
    rules = [_axis(lo, hi, br, order, _degree(R, k)) for k, ((lo, hi), br) in enumerate(zip(box, breaks))]
    if np.prod([len(x) for x, _ in rules], dtype=float) > MAX_POINTS:
        return None
    weights = [w * D.jacobian_value(k, x) for k, (x, w) in enumerate(rules)]
    rest = np.einsum(",".join("abcd"[:len(rules) - 1]) + "->" + "abcd"[:len(rules) - 1], *weights[1:]) if len(rules) > 2 else weights[1]
    total = 0.0
    x0, w0 = rules[0][0], weights[0]
    chunk = max(1, int(SLICE_POINTS / max(rest.size, 1)))
    for s in range(0, len(x0), chunk):
        axes = [x0[s:s + chunk]] + [x for x, _ in rules[1:]]
        G = np.meshgrid(*axes, indexing="ij")
        P = np.stack([np.asarray(c, float).ravel() for c in D.to_cartesian(*G)], axis=1)
        keep = members_at(m.set, P)
        if not keep.any():
            continue
        wt = np.zeros(len(P))
        wt[keep] = weight_values(m, P[keep])
        v = np.asarray(B.grid_values(R, axes), float).reshape(len(P), -1)
        scale = (w0[s:s + chunk, None] * rest.ravel()[None]).ravel()
        total = total + (scale * wt) @ v
    return np.atleast_1d(total) if np.ndim(total) else None


def moment(space, m, R):
    """int g dm (N,) by the module's rule, or None."""
    D = space.domain
    box = natural_box(m, D)
    if box is None or any(hi <= lo for lo, hi in box):
        return None
    breaks = _axis_breaks(space.backend, R, len(box))
    last, gap = None, np.inf
    for order in ORDERS:
        now = _at_order(space, m, R, box, breaks, order)
        if now is None:                                   # the grid's cap: what the orders so far support
            break
        if last is not None:
            gap = float(np.max(np.abs(now - last)) / max(float(np.abs(now).max()), 1e-300))
            if gap <= AGREE:
                return now
        last = now
    return last if gap <= ACCEPT else None
