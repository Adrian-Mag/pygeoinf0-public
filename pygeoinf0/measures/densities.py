"""Radon-Nikodym derivatives between measures, with their local orders (plan 009 G13; EXP-030 DESIGN G13, NOTES.md
section 2; EXP-031 DESIGN 3.3, 8).

A functional l(f) = int k f dnu on L2(mu) has the representer k (dnu/dmu) when nu << mu, and is bounded iff that
representer is in L2(mu) (Riesz). Near a point x0 where the densities behave like powers,

    k ~ |x - x0|^p,   dnu/dx ~ |x - x0|^q,   dmu/dx ~ |x - x0|^s,     int (k dnu/dmu)^2 dmu ~ int |x - x0|^(2p + 2q - s) dx,

so the functional is bounded near x0 iff 2p + 2q - s > -1. For radial kernels against dr on L2(r^2 dr): q = 0, s = 2,
bounded iff p > 1/2; a kernel with k(0) != 0 is not (it needs a taper that vanishes at 0, or another reading).

The interface (what `against=` accepts): any measure nu with a method nu.density_against(base) returning a Density, the
Radon-Nikodym derivative dnu/dbase (values, breakpoints, support) with its local orders at points where it vanishes or
blows up. LebesgueMeasure and DensityMeasure implement it through density_against below, on every set: on intervals with
exact local orders (polynomial densities, tables, declared singular_at=/exponent=), elsewhere as the weight w (or
w / rho), whose declared singular points spaces/dual.py reads. The Density carries .weight, the derivative as a
Weight (measures/weights.py).

This module forms densities and their orders; it decides nothing. The one decision, with the one refusal path
(pg.Refused), is spaces/dual.py: bounded(mu, space) for measures and l2_density(nu, mu, kernels, taper) for density
kernels (pg.Functionals(..., against=)), both through the same local rule.
"""
import math

import numpy as np

from ..core.errors import Unsupported
from ..sets import Interval


class Density:
    """dnu/dbase on an interval: values (fun), breakpoints, the support [c, d] of nu, and local orders {x0: q} at points
    where it vanishes or is singular (dnu/dbase ~ |x - x0|^q). exact: whether the orders are known exactly.
    absolutely_continuous False (with reason) when nu is not absolutely continuous with respect to the base.
    On sets other than intervals fun takes Cartesian points (m, d) and orders are empty. weight: the derivative as a
    Weight (measures/weights.py), when it is known as one."""

    def __init__(self, fun, breaks=(), orders=None, support=None, name="dnu/dmu", exact=True, absolutely_continuous=True,
                 reason="", weight=None):
        self.fun, self.breaks, self.orders = fun, tuple(float(b) for b in breaks), dict(orders or {})
        self.support, self.name, self.exact = support, name, exact
        self.absolutely_continuous, self.reason, self.weight = absolutely_continuous, reason, weight

    def __call__(self, x):
        x = np.asarray(x, float)
        out = np.asarray(self.fun(x), float)
        if self.support is not None:
            out = np.where((x < self.support[0]) | (x > self.support[1]), 0.0, out)
        return out

    def order(self, x0):
        return self.orders.get(float(x0), 0.0)

    def __repr__(self):
        return f"Density({self.name})"


def pl_order(pl, x0, tol=1e-10, side=1):
    """The local order at x0 of the first column of a legendre.PiecewiseLegendre, approached from the side `side` (+1: from the right,
    -1: from the left; at the ends of the PiecewiseLegendre from inside): the index of the first nonzero Taylor coefficient of the
    piece there (inf for a zero piece). Any x0 of the PiecewiseLegendre, an end, a breakpoint or a point inside a piece."""
    from numpy.polynomial import legendre as npleg, polynomial as P
    br = np.asarray(pl.breaks)
    if x0 <= br[0]:
        side = 1
    elif x0 >= br[-1]:
        side = -1
    k = np.searchsorted(br, x0, side="right" if side > 0 else "left") - 1
    k = int(np.clip(k, 0, pl.npieces - 1))
    mono = npleg.leg2poly(np.asarray(pl.coefs[k][:, 0], float))
    t0 = -1.0 + 2.0 * (x0 - br[k]) / (br[k + 1] - br[k])
    q = P.Polynomial(mono)(P.Polynomial([t0, 1.0])).coef                          # in u = t - t0
    big = np.flatnonzero(np.abs(q) > tol * max(float(np.abs(q).max(initial=0.0)), 1e-300))
    return float(big[0]) if len(big) and np.abs(q).max() > 0 else math.inf


def _lebesgue_density(m):
    """(PiecewiseLegendre of dm/dx, or None for 1 and for a declared singular weight, interval) of a measure on an interval."""
    from . import DensityMeasure, LebesgueMeasure
    if isinstance(m, LebesgueMeasure) and isinstance(m.set, Interval):
        return None, m.set
    if isinstance(m, DensityMeasure) and isinstance(m.set, Interval):
        return m.density, m.set
    raise Unsupported(f"densities with respect to {m} are not formed here (Lebesgue measures and densities on intervals; "
                      "other measures offer density_against(base))")


def declared_points(m):
    """The declared singular points (singular_at=) of the weight of a measure on an interval."""
    wt = getattr(m, "weight", None)
    return [] if wt is None or wt.singular_at is None else [float(p[0]) for p in wt.singular_at]


def measure_order(m, x0, pl=None, side=1):
    """The local order of dm/dx at x0 (dm/dx ~ |x - x0|^order): -exponent at a declared point (singular_at=,
    exponent=), the exact order of a table (order_at), else that of the polynomial density (0 for Lebesgue measure)."""
    wt = getattr(m, "weight", None)
    if wt is not None and wt.singular_at is not None:
        if np.any(np.abs(wt.singular_at[:, 0] - x0) <= 1e-12 * max(1.0, abs(x0))):
            return -wt.exponent
    if wt is not None and hasattr(wt, "order_at"):
        return float(wt.order_at(x0))
    if pl is None:
        pl = _lebesgue_density(m)[0]
    return 0.0 if pl is None else pl_order(pl, x0, side=side)


def _values(m, x):
    """dm/dx at x (any shape): the weight's own values (near a zero of the density its fitted polynomial has absolute,
    not relative, accuracy, and 1/w would not)."""
    wt = getattr(m, "weight", None)
    if wt is None:
        return np.ones_like(x)
    return np.asarray(wt.values(x.reshape(-1, 1)), float).reshape(x.shape)


def interval_density(nu, base):
    """dnu/dbase for Lebesgue measures and densities on intervals: exact local orders (measure_order) at the points of
    the closed support of nu where either density is not of order 0 (outside that support dnu/dbase vanishes)."""
    wn, In = _lebesgue_density(nu)
    wm, Im = _lebesgue_density(base)
    tol = 1e-12 * max(abs(Im.b), abs(Im.a), 1.0)
    if In.a < Im.a - tol or In.b > Im.b + tol:
        return Density(None, absolutely_continuous=False,
                       reason=f"{nu} is not absolutely continuous with respect to {base}: its support {In} leaves {Im}")
    orders, space = {}, {}
    cand = {In.a, In.b, Im.a, Im.b} | set(declared_points(nu)) | set(declared_points(base))
    for x0 in sorted(x for x in cand if In.a - tol <= x <= In.b + tol):
        sides = (1,) if x0 <= In.a + tol else (-1,) if x0 >= In.b - tol else (-1, 1)   # from inside the support
        for side in sides:
            qn, sm = measure_order(nu, x0, wn, side), measure_order(base, x0, wm, side)
            if (qn != 0.0 or sm != 0.0) and qn - sm < orders.get(float(x0), math.inf):
                orders[float(x0)], space[float(x0)] = qn - sm, sm

    def fun(x):
        x = np.asarray(x, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _values(nu, x) / _values(base, x)
    brk = ([] if wn is None else list(wn.breaks)) + ([] if wm is None else list(wm.breaks)) + declared_points(nu) + \
        declared_points(base)
    out = Density(fun, sorted(set(brk) | {In.a, In.b}), orders, (In.a, In.b),
                  name=f"d({getattr(nu, 'name', 'nu')})/d({getattr(base, 'name', 'base')})")
    out.space_orders = space                 # the order of dbase/dx at the same points, from the same side
    return out


def density_against(nu, base):
    """nu.density_against(base) for LebesgueMeasure and DensityMeasure on every set (module docstring): a Density whose
    .weight is the derivative as a Weight (w, or w / rho when base = rho dx). Refused (UnboundedFunctional) when nu lives
    on a set of lower dimension than base's (nu is then singular to base)."""
    from ..core.errors import unbounded
    from . import DensityMeasure, measure_dim
    from .weights import ConstantWeight, QuotientWeight
    k, d = measure_dim(nu), int(base.set.dim)
    if k < d:
        raise unbounded(nu, k, f"L2({base})", "the set has measure zero for the space's measure")
    w = getattr(nu, "weight", None) or ConstantWeight(nu.set, 1.0)
    wt = QuotientWeight(w, base.weight) if isinstance(base, DensityMeasure) else w
    if isinstance(nu.set, Interval) and isinstance(base.set, Interval):
        rho = interval_density(nu, base)
    else:
        rho = Density(lambda X: wt.values(np.atleast_2d(np.asarray(X, float))),
                      name=f"d({getattr(nu, 'name', 'nu')})/d({getattr(base, 'name', 'base')})")
    rho.weight = wt
    return rho


def space_order(mu, x0, rho=None, side=1):
    """The local order s of the space's measure at x0 (dmu/dx ~ |x - x0|^s): the one recorded with the Density rho
    (the same side as its order), else from mu, approached from `side`."""
    rec = getattr(rho, "space_orders", {})
    return rec[float(x0)] if float(x0) in rec else measure_order(mu, x0, side=side)


def kernel_order(k, x0, inside, scale):
    """(p, exact): the local order of a kernel at x0 from the side `inside` (+1 or -1): exactly for a Tabulated (or
    anything with order_at), else estimated from samples at x0 + inside * scale * 10^-j, j = 2..9 (a check, not a proof)."""
    if hasattr(k, "order_at"):
        return float(k.order_at(x0)), True
    if np.ndim(k) == 0 and not callable(k):
        return (0.0 if float(k) != 0 else math.inf), True
    h = scale * 10.0 ** -np.arange(2, 10)
    v = np.abs(np.asarray([float(np.asarray(k(np.array([x0 + inside * hh]))).ravel()[0]) for hh in h]))
    if not np.any(v > 0):
        return math.inf, False
    if np.all(v > 0):
        return float(np.round(np.polyfit(np.log(h[-4:]), np.log(v[-4:]), 1)[0], 2)), False
    return math.inf, False
