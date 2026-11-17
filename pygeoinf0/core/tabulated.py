"""Tabulated inputs as declared functions, integrated exactly (plan 009 G10; EXP-030 DESIGN G10).

    k = pg.Tabulated(domain, nodes, values, rule="linear", segments=None, extrapolation="flat", against=None)

A table is not a function until a rule says how to read it; the rule is a modelling choice (ASSERTED), the reading is
then exact. Rules:

    "constant"         nodes are cell edges (n + 1), values one per cell: k = v_i on [x_i, x_(i+1))
    "linear"           nodes are points (n), values one per point: linear between consecutive points
    "midpoint-linear"  nodes are cell edges (n + 1), values one per cell, read at the cell midpoints and joined linearly
                       (cells of zero width are skipped; their value must be 0): the rule of the thesis's kernels

segments: points (inside the domain) across which "linear" and "midpoint-linear" never interpolate: each segment
[s_k, s_(k+1)) is read from its own points only (the last segment is closed), so a discontinuity of the tabulated
quantity at a known radius stays a jump. extrapolation: beyond the first and last point of each segment (and of the
domain), "flat" (the nearest value: the thesis's rule), "zero" or "linear". against: the measure the table is a density
against (default Lebesgue measure on the domain: k(r) dr); pg.Functionals converts it to the space's measure (G13).

The result is exactly piecewise polynomial of degree <= 1 (k.pl, a one-column legendre.PiecewiseLegendre; k.breaks its breakpoints), so
integrals of k against polynomials and polynomial weights are exact by Gauss-Legendre per piece (k.integral,
k.inner), the breakpoints are passed to every realisation that fits k or functions made from it, and k.order_at(x0)
gives its exact local order (0 where k(x0) != 0, 1 where it vanishes linearly, inf where it vanishes on a piece), which
the boundedness checks of pg.Functionals use. k.integral(quadrature=pg.realisations.Trapezoid(n)) uses the n-node
trapezoid rule instead (a realisation option, S1) and states its error against the exact value.
"""
import math

import numpy as np

from .claims import ASSERTED, DERIVED, Claim, Claims
from ..sets import Interval

RULES = ("constant", "linear", "midpoint-linear")
EXTRAPOLATIONS = ("flat", "zero", "linear")


class Tabulated:
    """A tabulated function on an interval, read by a declared rule (module docstring)."""

    def __init__(self, domain, nodes, values, rule="linear", segments=None, extrapolation="flat", against=None,
                 name="k"):
        I = Interval(domain) if isinstance(domain, str) else domain if isinstance(domain, Interval) else \
            Interval(float(domain[0]), float(domain[1]))
        if rule not in RULES:
            raise ValueError(f"rule is one of {RULES}")
        if extrapolation not in EXTRAPOLATIONS:
            raise ValueError(f"extrapolation is one of {EXTRAPOLATIONS}")
        x, v = np.asarray(nodes, float).ravel(), np.asarray(values, float).ravel()
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
            raise ValueError("nodes and values must be finite")
        if np.any(np.diff(x) < 0):
            raise ValueError("nodes must be non-decreasing")
        self.domain, self.rule, self.extrapolation, self.against, self.name = I, rule, extrapolation, against, name
        self.nodes, self.values = x, v
        a, b = I.a, I.b
        seg = np.unique(np.asarray([] if segments is None else segments, float))
        self.segments = seg[(seg > a) & (seg < b)]
        if rule == "constant":
            if len(v) != len(x) - 1:
                raise ValueError("the constant rule takes cell edges (n + 1) and one value per cell (n)")
            pieces = self._constant(x, v, a, b)
        else:
            if rule == "midpoint-linear":
                if len(v) != len(x) - 1:
                    raise ValueError("the midpoint-linear rule takes cell edges (n + 1) and one value per cell (n)")
                w = np.diff(x)
                if np.any((w == 0) & (v != 0)):
                    raise ValueError("a cell of zero width carries a nonzero value")
                x, v = 0.5 * (x[:-1] + x[1:])[w > 0], v[w > 0]
            elif len(v) != len(x):
                raise ValueError("the linear rule takes one value per node")
            if np.any(np.diff(x) == 0):
                raise ValueError("repeated points: use segments= for a jump")
            pieces = self._linear(x, v, a, b)
        pieces = [p for p in pieces if p[1] > p[0]]
        self._x0 = np.array([p[0] for p in pieces])
        self._x1 = np.array([p[1] for p in pieces])
        self._v0 = np.array([p[2] for p in pieces])
        self._v1 = np.array([p[3] for p in pieces])
        self.breaks = np.r_[self._x0, self._x1[-1]]
        self._pl = None

    # ---------------------------------------------------------------------------------------------- construction
    def _ends(self, xs, vs, lo, hi):
        """The pieces before the first and after the last point of the segment [lo, hi], by the extrapolation rule."""
        out = []
        for inner, x_in, edge, near in ((0, xs[0], lo, slice(0, 2)), (-1, xs[-1], hi, slice(-2, None))):
            if (inner == 0 and x_in <= lo) or (inner == -1 and x_in >= hi):
                continue
            v_in = float(vs[inner])
            if self.extrapolation == "zero":
                e, v_in = 0.0, 0.0
            elif self.extrapolation == "flat" or len(xs) < 2:
                e = v_in
            else:
                x2, v2 = xs[near], vs[near]
                e = float(v2[0] + (v2[1] - v2[0]) * (edge - x2[0]) / (x2[1] - x2[0]))
            out.append((edge, x_in, e, v_in) if inner == 0 else (x_in, edge, v_in, e))
        return out

    def _linear(self, x, v, a, b):
        edges = np.r_[a, self.segments, b]
        out = []
        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k + 1]
            last = k == len(edges) - 2
            sel = (x >= lo) & ((x <= hi) if last else (x < hi))
            if not sel.any():
                raise ValueError(f"the segment [{lo:g}, {hi:g}) contains no tabulated point")
            xs, vs = x[sel], v[sel]
            out += self._ends(xs, vs, lo, hi)
            out += [(xs[i], xs[i + 1], vs[i], vs[i + 1]) for i in range(len(xs) - 1)]
        return sorted(out, key=lambda p: p[0])

    def _constant(self, x, v, a, b):
        out = []
        if x[0] > a:
            c = 0.0 if self.extrapolation == "zero" else v[0]
            out.append((a, x[0], c, c))
        out += [(max(x[i], a), min(x[i + 1], b), v[i], v[i]) for i in range(len(v)) if x[i + 1] > a and x[i] < b]
        if x[-1] < b:
            c = 0.0 if self.extrapolation == "zero" else v[-1]
            out.append((x[-1], b, c, c))
        return out

    # ---------------------------------------------------------------------------------------------- values
    def __call__(self, t):
        t = np.asarray(t, float)
        k = np.clip(np.searchsorted(self.breaks, t, side="right") - 1, 0, len(self._x0) - 1)
        s = (t - self._x0[k]) / (self._x1[k] - self._x0[k])
        out = self._v0[k] + (self._v1[k] - self._v0[k]) * s
        out = np.where((t < self.domain.a) | (t > self.domain.b), np.nan, out)
        return float(out) if out.ndim == 0 else out

    @property
    def pl(self):
        """The function as a one-column legendre.PiecewiseLegendre (degree <= 1 per piece): exact. Built once (a table is immutable)."""
        if self._pl is None:
            from ..realisations.series.legendre import PiecewiseLegendre
            self._pl = PiecewiseLegendre(self.breaks, [np.array([[0.5 * (p + q)], [0.5 * (q - p)]]) for p, q in zip(self._v0, self._v1)],
                          [0.0])
        return self._pl

    def order_at(self, x0):
        """The exact local order p of k at x0 from inside the domain: k ~ |x - x0|^p (0, 1 or inf)."""
        x0 = float(x0)
        right = x0 < self.domain.b
        k = int(np.clip(np.searchsorted(self.breaks, x0, side="right") - 1 if right else len(self._x0) - 1, 0,
                        len(self._x0) - 1))
        v0, v1 = (self._v0[k], self._v1[k])
        at = v0 + (v1 - v0) * (x0 - self._x0[k]) / (self._x1[k] - self._x0[k])
        scale = max(abs(v0), abs(v1), 1e-300)
        if abs(at) > 1e-14 * scale:
            return 0.0
        return 1.0 if abs(v1 - v0) > 1e-14 * scale else math.inf

    def zero_end(self, inward):
        """Where k stops vanishing identically from its left end (inward > 0) or right end (inward < 0): the first
        (last) point of its first (last) piece that is not zero; the end itself when k is nonzero there, None if k = 0."""
        nz = np.flatnonzero((self._v0 != 0) | (self._v1 != 0))
        if nz.size == 0:
            return None
        return float(self._x0[nz[0]]) if inward > 0 else float(self._x1[nz[-1]])

    # ---------------------------------------------------------------------------------------------- integrals
    def integral(self, over=None, weight=None, quadrature=None):
        """int k w dx over the domain (or the interval `over`), w a polynomial weight given as a legendre.PiecewiseLegendre (or None);
        exact by Gauss-Legendre per piece, or by `quadrature` (e.g. pg.realisations.Trapezoid(n)): then a pair
        (value, Claim stating its error against the exact integral)."""
        from ..realisations.series import legendre as LG
        J = self.domain if over is None else Interval(over) if isinstance(over, str) else over \
            if isinstance(over, Interval) else Interval(float(over[0]), float(over[1]))
        a, b = J.a, J.b
        p = self.pl if weight is None else LG.multiply(self.pl, weight)
        exact = float(LG.integral(p, a, b)[0])
        if quadrature is None:
            return exact
        val = float(quadrature.integrate(lambda t: self(t) * (1.0 if weight is None else LG.evaluate(weight, t)[..., 0]),
                                         a, b))
        return val, quadrature.error_claim(val, exact, f"int {self.name} over [{a:g}, {b:g}]")

    def inner(self, other, weight=None):
        """int k g w dx for another Tabulated (or a legendre.PiecewiseLegendre) g and a polynomial weight w (a PiecewiseLegendre): exact."""
        from ..realisations.series import legendre as LG
        g = other.pl if isinstance(other, Tabulated) else other
        if weight is None:
            return float(LG.gram(self.pl, g)[0, 0])
        return float(LG.gram_weighted(self.pl, g, weight)[0, 0])

    def element(self, space):
        """The element of a function space on this interval with this function as representative (breakpoints passed)."""
        return space.element(self, breaks=list(self.breaks[1:-1]), name=self.name)

    @property
    def claims(self):
        seg = f", split at {len(self.segments)} segment points" if len(self.segments) else ""
        return Claims([Claim(f"{self.name} is read from its table by the rule {self.rule!r}{seg}, extrapolation "
                             f"{self.extrapolation!r}", ASSERTED, "a modelling choice declared with the table"),
                       Claim("integrals against polynomials are exact", DERIVED,
                             f"piecewise linear on {len(self._x0)} pieces: Gauss-Legendre per piece")])

    def __repr__(self):
        return f"Tabulated {self.name} on {self.domain} ({len(self.nodes)} nodes, rule {self.rule!r})"
