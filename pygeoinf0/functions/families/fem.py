"""Finite-element-like functions, work package F (plan 021 P1): the tensor machinery shared by the three families and
the P1 hat function (EXP-037 DESIGN.md 5, 6).

    hat(D, *, index, nodes=None, count=None)   the P1 hat function of node `index` of a grid on an interval, and the
                                               tensor product of one hat per axis on a box grid; continuous, piecewise
                                               affine (C^0 piecewise)
    bspline(D, *, degree, index, knots=None, count=None)         fem_bspline.py
    partition_of_unity(D, *, index, count=None, ...)             fem_unity.py

All three are products over the axes of one-dimensional factors. A `Factor` carries the value and derivative of one
factor, its support [lo, hi] inside the axis [a, b], its regularity, how it vanishes at the two ends of the axis, and the
closed forms of its integral, squared L2 norm and maximum where they were derived; `assemble` forms the product, the
support (a box, given by a `contains` predicate), the regularity (the least of the factors'; the function vanishes at the
boundary of a box to the least order of its factors at the 2 d faces), the gradient (product rule) and the closed forms
(products of the factors').

From fem_bspline.py (merged here by plan 063):
B-spline basis functions, work package F (plan 021 P1): bspline(D, *, degree, index, knots=None, count=None).

The B-spline of degree p with knots t_0 <= ... <= t_{p+1} is given by the Cox-de Boor recurrence
    N_{j,0}(x) = 1 on [t_j, t_{j+1}) (0 if t_j = t_{j+1}),
    N_{j,q}(x) = (x - t_j) / (t_{j+q} - t_j) N_{j,q-1}(x) + (t_{j+q+1} - x) / (t_{j+q+1} - t_{j+1}) N_{j+1,q-1}(x)
(terms with a zero denominator are 0), and its derivative by
    N'_{j,q} = q [ N_{j,q-1} / (t_{j+q} - t_j) - N_{j+1,q-1} / (t_{j+q+1} - t_{j+1}) ].
The last non-empty knot span that ends at the domain's upper end is closed there, so that the clamped basis sums to 1
on the closed domain. A basis function of a knot vector t_0 .. t_{n+p} is B_i = N_{0,p} for the knots t_i .. t_{i+p+1}.

From fem_unity.py (merged here by plan 063):
Smooth partitions of unity, work package F (plan 021 P1):
partition_of_unity(D, *, index, count=None, overlap=None, centres=None, radii=None, k=1.0).

On an interval, the bumps psi_j(x) = exp(-k / (1 - t_j^2)), t_j = (x - c_j) / r_j, (0 for |t_j| >= 1), j = 0 .. n-1, cover
the domain, and the j-th member is phi_j = psi_j / sum_l psi_l. On a box the member is the product of the axis members, so
that the sum over all index tuples is the product of the axis sums, 1.
"""


import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ...core.claims import CHECKED, DERIVED, THEOREM
from ...core.errors import Refused
from ...quadrature import gauss
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register

CIARLET = ("Ciarlet 1978, The Finite Element Method for Elliptic Problems (the nodal basis of the P1 Lagrange element "
           "is continuous and piecewise affine)")


@dataclass(frozen=True)
class Factor:
    """One axis of a tensor-product function (module docstring); integral, l2 and peak are None where not derived."""
    value: object                   # x (array) -> array
    deriv: object                   # x (array) -> array: the derivative (one-sided at kinks)
    lo: float                       # the support [lo, hi] inside the axis [a, b]
    hi: float
    order: float                    # math.inf or an integer >= 0: C^order on [a, b]
    piecewise: bool                 # True: only piecewise smooth (kinks); False: C-infinity on [a, b]
    vanish_left: float              # derivatives of order < vanish_left vanish at a (math.inf: all; 0: none)
    vanish_right: float             # the same at b
    integral: float = None
    l2: float = None
    peak: float = None
    kinks: tuple = ()               # points inside [lo, hi] where the factor's derivatives jump (fit breaks)


# ------------------------------------------------------------------------------------------- parameter validation
def sides(D):
    """The axes of D as a list of (a, b)."""
    return [(float(D.a), float(D.b))] if geo.kind(D) == "interval" else [(float(s.a), float(s.b)) for s in D.sides]


def to_int(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise Refused(f"{name} must be an integer (got {v!r})")
    return int(v)


def to_real(v, name):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise Refused(f"{name} must be a real number (got {v!r})") from None
    if isinstance(v, (bool, np.bool_)) or not math.isfinite(x):
        raise Refused(f"{name} must be a finite real number (got {v!r})")
    return x


def to_grid(v, name):
    try:
        g = np.array(v, dtype=float)
    except (TypeError, ValueError):
        raise Refused(f"{name} must be a sequence of real numbers (got {v!r})") from None
    if g.ndim != 1 or not np.all(np.isfinite(g)):
        raise Refused(f"{name} must be a one-dimensional sequence of finite real numbers (got {v!r})")
    return g


def per_axis(D, v, name, conv, broadcast=False):
    """The value for each axis of D: on an interval `v` itself; on a box a sequence with one entry per axis (a single
    entry is repeated when `broadcast`). Refused with the reason otherwise."""
    d = len(sides(D))
    if d == 1:
        return [conv(v, name)]
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) != d:
            raise Refused(f"{name} needs one entry per axis ({d} on {D!r}), got {len(v)}")
        return [conv(x, f"{name}[{j}]") for j, x in enumerate(v)]
    if broadcast:
        return [conv(v, name)] * d
    raise Refused(f"{name} needs one entry per axis ({d} on {D!r}); got {v!r}")


def pack(D, values):
    """The parameter record of per-axis values: the value itself on an interval, a tuple on a box."""
    vals = [tuple(float(t) for t in v) if isinstance(v, np.ndarray) else v for v in values]
    return vals[0] if len(vals) == 1 else tuple(vals)


def exactly_one(family, **given):
    """Refused unless exactly one of the keyword arguments is not None; returns its name."""
    present = [k for k, v in given.items() if v is not None]
    if len(present) != 1:
        raise Refused(f"{family}: give exactly one of {', '.join(given)} (got {', '.join(present) or 'none'})")
    return present[0]


def check_index(i, n, axis, what, family):
    if not 0 <= i < n:
        raise Refused(f"{family}: index {i} on axis {axis} is outside 0..{n - 1} ({n} {what})")


# ------------------------------------------------------------------------------------------- tensor assembly
def assemble(D, family, params, factors, reasons, closed_why):
    """The BankFunction that is the product of one factor per axis (module docstring). `reasons` as in BankFunction
    (a "regularity" entry makes the regularity claim a theorem); `closed_why` maps "integral", "l2_norm_sq", "max" to
    (status, reason) for the closed forms the factors carry."""
    def fn(*X):
        out = 1.0
        for fac, x in zip(factors, X):
            out = out * fac.value(np.asarray(x, float))
        return out

    def gradient(*X):
        X = np.broadcast_arrays(*[np.asarray(x, float) for x in X])
        vals = [fac.value(x) for fac, x in zip(factors, X)]
        cols = []
        for j, (fac, x) in enumerate(zip(factors, X)):
            col = fac.deriv(x)
            for ell, v in enumerate(vals):
                if ell != j:
                    col = col * v
            cols.append(col)
        return np.stack(cols, axis=-1)

    def contains(*X):
        ok = True
        for fac, x in zip(factors, X):
            ok = ok & (np.asarray(x) >= fac.lo) & (np.asarray(x) <= fac.hi)
        return np.asarray(ok)

    axes = sides(D)
    inside = all(fac.lo > a and fac.hi < b for fac, (a, b) in zip(factors, axes))
    box = " x ".join(f"[{fac.lo:g}, {fac.hi:g}]" for fac in factors)
    support = Support("interior" if inside else "touches", f"the box {box}" if len(factors) > 1 else f"the interval {box}",
                      contains=contains, box=tuple((fac.lo, fac.hi) for fac in factors))
    order = min(fac.order for fac in factors)
    piecewise = any(fac.piecewise for fac in factors)
    van = min(min(fac.vanish_left, fac.vanish_right) for fac in factors)
    vanishes = "all orders" if van == math.inf else (int(van) if van >= 1 else None)
    regularity = Regularity(order if order == math.inf else int(order), piecewise, vanishes)
    closed = {}
    for key, attr in (("integral", "integral"), ("l2_norm_sq", "l2"), ("max", "peak")):
        vals = [getattr(fac, attr) for fac in factors]
        if key in closed_why and all(v is not None for v in vals):
            closed[key] = (float(np.prod(vals)), *closed_why[key])
    breaks = tuple((fac.lo, fac.hi) + tuple(fac.kinks) for fac in factors)
    return BankFunction(family, D, params, fn, support, regularity, closed, gradient, reasons=reasons, breaks=breaks)


# ------------------------------------------------------------------------------------------- the hat function
def hat_factor(a, b, nodes, i):
    """The P1 hat of node i of the grid `nodes` (strictly increasing, nodes[0] = a, nodes[-1] = b)."""
    m = len(nodes) - 1
    y = np.zeros(m + 1)
    y[i] = 1.0
    slope = np.diff(y) / np.diff(nodes)

    def value(x):
        return np.interp(x, nodes, y, left=0.0, right=0.0)

    def deriv(x):
        x = np.asarray(x, float)
        cell = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, m - 1)
        return np.where((x >= nodes[0]) & (x <= nodes[-1]), slope[cell], 0.0)

    lo, hi = nodes[max(i - 1, 0)], nodes[min(i + 1, m)]
    h_left = nodes[i] - nodes[i - 1] if i > 0 else 0.0
    h_right = nodes[i + 1] - nodes[i] if i < m else 0.0
    left = math.inf if lo > a else (0 if i == 0 else 1)         # f(a) = 1 at the first node, else 0 with f' != 0
    right = math.inf if hi < b else (0 if i == m else 1)
    return Factor(value, deriv, float(lo), float(hi), math.inf if m == 1 else 0, m > 1, left, right,
                  integral=0.5 * (h_left + h_right), l2=(h_left + h_right) / 3.0, peak=1.0, kinks=(float(nodes[i]),))


def grids(D, family, nodes, count):
    """The per-axis node arrays of a grid given by `nodes` or by `count` uniform nodes per axis (exactly one)."""
    axes = sides(D)
    if exactly_one(family, nodes=nodes, count=count) == "count":
        counts = per_axis(D, count, "count", to_int, broadcast=True)
        for j, n in enumerate(counts):
            if n < 2:
                raise Refused(f"{family}: count must be at least 2 nodes per axis (axis {j} has {n})")
        return [np.linspace(a, b, n) for (a, b), n in zip(axes, counts)], pack(D, counts), "count"
    grid = per_axis(D, nodes, "nodes", to_grid)
    out = []
    for j, (g, (a, b)) in enumerate(zip(grid, axes)):
        tol = 1e-12 * (b - a)
        if len(g) < 2 or not np.all(np.diff(g) > 0):
            raise Refused(f"{family}: the nodes of axis {j} must be at least 2 and strictly increasing")
        if abs(g[0] - a) > tol or abs(g[-1] - b) > tol:
            raise Refused(f"{family}: the nodes of axis {j} must run from {a:g} to {b:g} (got {g[0]:g} to {g[-1]:g}); "
                          f"a grid spans the domain")
        g = g.copy()
        g[0], g[-1] = a, b
        out.append(g)
    return out, pack(D, out), "nodes"


def _hat_examples(D):
    axes = sides(D)
    d = len(axes)
    (a, b) = axes[0]
    L = b - a
    if d == 1:
        return [{"index": 0, "count": 5}, {"index": 2, "count": 5}, {"index": 4, "count": 5},
                {"index": 2, "nodes": (a, a + 0.15 * L, a + 0.4 * L, a + 0.7 * L, b)}]
    # Boxes: the contract's reference quadrature (a fixed Gauss-Legendre rule over the whole domain) cannot resolve the
    # kinks of a hat to its 1e-5 tolerance in 2 and 3 dimensions, so the contract examples are the affine grids (2 nodes
    # per axis, no kinks); grids with kinks are checked against exact cell-by-cell integration in test_functions_fem.py.
    two = tuple((s, t) for s, t in axes)
    return [{"index": (0,) * d, "count": 2}, {"index": (1,) + (0,) * (d - 1), "count": 2},
            {"index": (1,) * d, "nodes": two}]


@register("hat", ("interval", "box"),
          "P1 hat function of a grid node (tensor product of hats on a box grid): continuous, piecewise affine",
          references=(CIARLET,), examples=_hat_examples)
def hat(D, *, index, nodes=None, count=None):
    """The P1 (piecewise linear) hat function of node `index` of a grid, 0-based: with nodes x_0 < ... < x_m spanning
    the domain,  phi_i(x) = (x - x_{i-1}) / (x_i - x_{i-1}) on [x_{i-1}, x_i],  (x_{i+1} - x) / (x_{i+1} - x_i) on
    [x_i, x_{i+1}],  0 elsewhere  (the first and last hats have only one side). The grid is `nodes` (strictly increasing,
    from the domain's lower to its upper end) or `count` uniform nodes. On a box, `index` is a tuple with one node index
    per axis, `nodes` a tuple of grids and `count` an integer or a tuple; the function is the product of the axis hats.

    Support: the box of the neighbouring cells, [x_{i-1}, x_{i+1}] per axis; it touches the boundary for the nodes on
    the first and last cells. Regularity: C^0, piecewise affine (kinks on the grid lines); a grid of 2 nodes per axis
    gives an affine function. The function vanishes at the boundary (to order 1) only if every axis hat does.
    Closed forms (derived): integral = prod (h_left + h_right) / 2 (a triangle), squared L2 norm = prod
    (h_left + h_right) / 3, maximum 1; h_left = x_i - x_{i-1}, h_right = x_{i+1} - x_i (0 at the grid's ends).
    Sum over all nodes: 1 on the domain (partition of unity)."""
    grid, grid_record, mode = grids(D, "hat", nodes, count)
    idx = per_axis(D, index, "index", to_int)
    factors = []
    for j, (g, i, (a, b)) in enumerate(zip(grid, idx, sides(D))):
        check_index(i, len(g), j, "nodes", "hat")
        factors.append(hat_factor(a, b, g, i))
    why = {"integral": (DERIVED, "the area of the triangle(s) of height 1 over the neighbouring cells, per axis"),
           "l2_norm_sq": (DERIVED, "int_0^h (s/h)^2 ds = h/3 on each side, per axis"),
           "max": (DERIVED, "the hat is 1 at its node and below 1 elsewhere")}
    return assemble(D, "hat", {"index": pack(D, idx), mode: grid_record}, factors,
                    {"support": "the neighbouring cells of the node, per axis"}, why)


# ---------------------------------------------------------------- from fem_bspline.py (plan 063)
DE_BOOR = ("de Boor 1978, A Practical Guide to Splines (B-splines: the Cox-de Boor recurrence, the derivative formula, "
           "C^(p-m) continuity at a knot of multiplicity m, the integral (t_{i+p+1} - t_i)/(p+1))")
MAX_DEGREE = 10


def cox_de_boor(x, t, q, upper):
    """N_{0,q}(x) for the knots t[0 .. q+1]; the span ending at `upper` is closed there."""
    if q == 0:
        if t[0] >= t[1]:
            return np.zeros(np.shape(x))
        return ((x >= t[0]) & (x < t[1]) | ((x == t[1]) & (t[1] == upper))).astype(float)
    out = np.zeros(np.shape(x))
    if t[q] > t[0]:
        out = out + (x - t[0]) / (t[q] - t[0]) * cox_de_boor(x, t[:-1], q - 1, upper)
    if t[q + 1] > t[1]:
        out = out + (t[q + 1] - x) / (t[q + 1] - t[1]) * cox_de_boor(x, t[1:], q - 1, upper)
    return out


def bspline_factor(a, b, p, t):
    """The B-spline of degree p on the knots t (p + 2 values, nondecreasing, inside [a, b])."""
    def value(x):
        return cox_de_boor(np.asarray(x, float), t, p, b)

    def deriv(x):
        x = np.asarray(x, float)
        out = np.zeros(x.shape)
        if t[p] > t[0]:
            out = out + cox_de_boor(x, t[:-1], p - 1, b) / (t[p] - t[0])
        if t[p + 1] > t[1]:
            out = out - cox_de_boor(x, t[1:], p - 1, b) / (t[p + 1] - t[1])
        return p * out

    knots, mult = np.unique(t, return_counts=True)
    inner = (knots > a) & (knots < b)
    if np.any(mult[inner] > p):
        raise Refused(f"bspline: a knot of multiplicity {int(mult[inner].max())} strictly inside the domain makes the "
                      f"degree-{p} function discontinuous; multiplicities inside the domain must be at most {p}")
    order = math.inf if not inner.any() else p - int(mult[inner].max())
    left = math.inf if t[0] > a else p + 1 - int(mult[0])        # B_i ~ (x - a)^(p + 1 - m) at a knot a of multiplicity m
    right = math.inf if t[-1] < b else p + 1 - int(mult[-1])
    return Factor(value, deriv, float(t[0]), float(t[-1]), order, bool(inner.any()), left, right, kinks=tuple(knots),
                  integral=float(t[-1] - t[0]) / (p + 1))


def knot_vectors(D, degrees, knots, count):
    """Per-axis knot vectors: given (nondecreasing, inside the axis) or uniform clamped with `count` functions."""
    axes = sides(D)
    if exactly_one("bspline", knots=knots, count=count) == "count":
        counts = per_axis(D, count, "count", to_int, broadcast=True)
        out = []
        for j, (n, p, (a, b)) in enumerate(zip(counts, degrees, axes)):
            if n < p + 1:
                raise Refused(f"bspline: count must be at least degree + 1 = {p + 1} basis functions (axis {j} has {n})")
            out.append(np.r_[[a] * p, np.linspace(a, b, n - p + 1), [b] * p])
        return out, pack(D, counts), "count"
    out = []
    for j, (t, p, (a, b)) in enumerate(zip(per_axis(D, knots, "knots", to_grid), degrees, axes)):
        if len(t) < p + 2:
            raise Refused(f"bspline: a degree-{p} basis needs at least {p + 2} knots (axis {j} has {len(t)})")
        if np.any(np.diff(t) < 0):
            raise Refused(f"bspline: the knots of axis {j} must be nondecreasing")
        if t[0] < a or t[-1] > b:
            raise Refused(f"bspline: the knots of axis {j} must lie in the domain [{a:g}, {b:g}] (got {t[0]:g} to "
                          f"{t[-1]:g}), so that every basis function has its whole support in the domain")
        out.append(t)
    return out, pack(D, out), "knots"


def _bspline_examples(D):
    axes = sides(D)
    d = len(axes)
    a, b = axes[0]
    L = b - a
    if d == 1:
        return [{"degree": 2, "index": 0, "count": 6}, {"degree": 2, "index": 3, "count": 6},
                {"degree": 3, "index": 2, "count": 7},
                {"degree": 2, "index": 1, "knots": (a, a + 0.1 * L, a + 0.1 * L, a + 0.4 * L, a + 0.7 * L, b, b)}]
    return [{"degree": 3, "index": (0,) * d, "count": 4},
            {"degree": (3,) * d, "index": (1,) + (2,) * (d - 1), "count": (5,) + (4,) * (d - 1)},
            {"degree": (3,) + (4,) * (d - 1), "index": (0,) * d, "count": (5,) + (6,) * (d - 1)}]



@register("bspline", ("interval", "box"),
          "B-spline basis function of degree p (knot vector given or uniform clamped), C^(p-1); tensor product on a box",
          references=(DE_BOOR,), examples=_bspline_examples)
def bspline(D, *, degree, index, knots=None, count=None):
    """The i-th B-spline basis function B_{i,p} of degree p = `degree` >= 1 (order p + 1), 0-based `index`: the
    spline on the knots t_i <= ... <= t_{i+p+1} of the Cox-de Boor recurrence (module docstring), 0 outside
    [t_i, t_{i+p+1}]. The knot vector is `knots` (nondecreasing, all inside the domain, at least p + 2 values; n = len - p - 1
    functions) or the uniform clamped vector of `count` = n >= p + 1 functions on the domain (the end knots repeated
    p + 1 times). On a box, `degree`, `index`, `knots`, `count` are tuples with one entry per axis (`degree` and `count`
    may be one integer for all axes) and the function is the product of the axis B-splines.

    Support: [t_i, t_{i+p+1}] per axis (touching the boundary for the first and last functions of a clamped basis).
    Regularity: C^(p - m) at a knot of multiplicity m, so C^(p-1) for simple knots, piecewise polynomial (a single
    polynomial piece when no knot lies strictly inside the domain); a knot of multiplicity > p strictly inside the domain
    is refused. At a domain end that is a knot of multiplicity m of the function, B ~ (x - a)^(p + 1 - m): derivatives of
    order < p + 1 - m vanish there (none for the clamped first function, whose value is 1 at a).
    Closed form (theorem, de Boor): integral = prod (t_{i+p+1} - t_i) / (p + 1). Sum over a clamped basis: 1 on the domain
    (partition of unity; checked in the tests)."""
    deg = per_axis(D, degree, "degree", to_int, broadcast=True)
    for j, p in enumerate(deg):
        if not 1 <= p <= MAX_DEGREE:
            raise Refused(f"bspline: degree must be an integer from 1 to {MAX_DEGREE} (axis {j} has {p}; degree 0 is an "
                          f"indicator, which the bank offers elsewhere)")
    vectors, record, mode = knot_vectors(D, deg, knots, count)
    idx = per_axis(D, index, "index", to_int)
    factors = []
    for j, (p, t, i, (a, b)) in enumerate(zip(deg, vectors, idx, sides(D))):
        check_index(i, len(t) - p - 1, j, "basis functions", "bspline")
        if t[i + p + 1] <= t[i]:
            raise Refused(f"bspline: the knots t_{i}..t_{i + p + 1} of axis {j} are all equal: the function is 0")
        factors.append(bspline_factor(a, b, p, t[i:i + p + 2]))
    why = {"integral": (THEOREM, DE_BOOR)}
    return assemble(D, "bspline", {"degree": pack(D, deg), "index": pack(D, idx), mode: record}, factors,
                    {"regularity": DE_BOOR, "support": "the span of the knots t_i .. t_{i+p+1} of each axis"}, why)


# ---------------------------------------------------------------- from fem_unity.py (plan 063)
HORMANDER = "Hormander 1990, The Analysis of Linear Partial Differential Operators I, 1.2 (exp(-1/(1-t^2)) is C-infinity)"
SHRINK = 0.9            # every point of the domain lies within SHRINK * radius of some centre: sum psi >= exp(-k / (1 - SHRINK^2))
MAX_K = 50.0


def _psi(x, c, r, k):
    t = (np.asarray(x, float) - c) / r
    inside = np.abs(t) < 1
    u = np.where(inside, t, 0.0)
    return np.where(inside, np.exp(-k / (1 - u * u)), 0.0)


def _dpsi(x, c, r, k):
    t = (np.asarray(x, float) - c) / r
    inside = np.abs(t) < 1
    u = np.where(inside, t, 0.0)
    return np.where(inside, -2 * k * u / (r * (1 - u * u) ** 2) * np.exp(-k / (1 - u * u)), 0.0)


def covers(a, b, centres, radii):
    """True when every point of [a, b] lies within SHRINK * radius_j of some centre_j (a sweep over closed intervals)."""
    spans = sorted((c - SHRINK * r, c + SHRINK * r) for c, r in zip(centres, radii))
    reach = a
    while reach < b:
        new = max((hi for lo, hi in spans if lo <= reach), default=-math.inf)
        if new <= reach:
            return False
        reach = new
    return True


def unity_factor(a, b, centres, radii, k, j):
    """The member j of the partition of unity of [a, b] by the bumps (centres, radii)."""
    def parts(x):
        psi = [_psi(x, c, r, k) for c, r in zip(centres, radii)]
        return psi, sum(psi)

    def value(x):
        psi, total = parts(x)
        return np.where(total > 0, psi[j] / np.where(total > 0, total, 1.0), 0.0)

    def deriv(x):
        psi, total = parts(x)
        d = [_dpsi(x, c, r, k) for c, r in zip(centres, radii)]
        safe = np.where(total > 0, total, 1.0)
        phi = psi[j] / safe
        return np.where(total > 0, (d[j] - phi * sum(d)) / safe, 0.0)

    c, r = centres[j], radii[j]
    return Factor(value, deriv, max(c - r, a), min(c + r, b), math.inf, False,
                  math.inf if c - r >= a else 0, math.inf if c + r <= b else 0)


def covering(D, family, count, overlap, centres, radii):
    """Per-axis (centres, radii): uniform (`count` bumps at uniform nodes with radius overlap * spacing) or given."""
    axes = sides(D)
    if count is not None:
        if centres is not None or radii is not None:
            raise Refused(f"{family}: give either count and overlap, or centres and radii, not both")
        if overlap is None:
            raise Refused(f"{family}: count needs overlap (the bump radius in units of the node spacing)")
        counts = per_axis(D, count, "count", to_int, broadcast=True)
        overl = per_axis(D, overlap, "overlap", to_real, broadcast=True)
        cs, rs = [], []
        for j, (n, o, (a, b)) in enumerate(zip(counts, overl, axes)):
            if n < 2 or not o > 0:
                raise Refused(f"{family}: axis {j} needs count >= 2 and overlap > 0 (got count {n}, overlap {o:g})")
            cs.append(np.linspace(a, b, n))
            rs.append(np.full(n, o * (b - a) / (n - 1)))
        return cs, rs, {"count": pack(D, counts), "overlap": pack(D, overl)}
    if centres is None or radii is None or overlap is not None:
        raise Refused(f"{family}: give count and overlap, or centres and radii")
    cs = per_axis(D, centres, "centres", to_grid)
    rs = per_axis(D, radii, "radii", to_grid)
    for j, (c, r) in enumerate(zip(cs, rs)):
        if len(c) < 1 or len(c) != len(r) or not np.all(r > 0):
            raise Refused(f"{family}: axis {j} needs as many positive radii as centres (got {len(c)} centres, "
                          f"{len(r)} radii)")
    return cs, rs, {"centres": pack(D, cs), "radii": pack(D, rs)}


def _unity_examples(D):
    axes = sides(D)
    d = len(axes)
    a, b = axes[0]
    L = b - a
    if d == 1:
        return [{"index": 0, "count": 5, "overlap": 1.0}, {"index": 2, "count": 5, "overlap": 1.0},
                {"index": 1, "count": 4, "overlap": 0.75, "k": 2.0},
                {"index": 1, "centres": (a, a + 0.5 * L, b), "radii": (0.7 * L, 0.3 * L, 0.7 * L)}]
    return [{"index": (0,) * d, "count": 3, "overlap": 1.0}, {"index": (1,) * d, "count": 3, "overlap": 1.0},
            {"index": (1,) + (2,) * (d - 1), "count": (3,) + (4,) * (d - 1), "overlap": 1.2}]


def _product_integral(factors, knots):
    """The product over the axes of int factor dx, or None if an axis does not converge."""
    t, w = gauss(32)
    total = 1.0
    for fac, kn in zip(factors, knots):
        breaks = np.unique(np.clip(np.r_[fac.lo, kn, fac.hi], fac.lo, fac.hi))
        previous = None
        for panels in (2, 4, 8, 16, 32, 64, 128):
            value = 0.0
            for a, b in zip(breaks[:-1], breaks[1:]):
                edges = np.linspace(a, b, panels + 1)
                for lo, hi in zip(edges[:-1], edges[1:]):
                    value += 0.5 * (hi - lo) * float(np.sum(w * fac.value(0.5 * (hi - lo) * (t + 1) + lo)))
            if previous is not None and abs(value - previous) <= 1e-12 * abs(value):
                break
            previous = value
        else:
            return None
        total *= value
    return total


@register("partition_of_unity", ("interval", "box"),
          "member of a smooth (C-infinity) partition of unity: C-infinity bumps normalised by their sum",
          references=(HORMANDER,), examples=_unity_examples)
def partition_of_unity(D, *, index, count=None, overlap=None, centres=None, radii=None, k=1.0):
    """The member `index` (0-based) of the partition of unity by the bumps psi_j = exp(-k / (1 - t_j^2)) on |t_j| < 1,
    t_j = (x - c_j) / r_j:  phi_j = psi_j / sum_l psi_l. The bumps are `count` ones centred at uniform nodes spanning the
    domain, each of radius `overlap` x (node spacing), or the given `centres` and `radii` (sequences of equal length). On
    a box, `index`, `count`, `overlap`, `centres`, `radii` have one entry per axis (`count` and `overlap` may be single
    numbers) and the member is the product of the axis members.
    The bumps must cover the domain with margin: every point within 0.9 x radius of some centre (refused otherwise, with
    the axis), which bounds the denominator below by exp(-k / 0.19) > 0, so that phi_j is C-infinity. 0 < k <= 50.

    Support: [c_j - r_j, c_j + r_j] intersected with the domain, per axis; interior when that lies strictly inside the
    domain, otherwise touching the boundary. Regularity: C-infinity (quotient of C-infinity functions, the denominator
    positive); it vanishes to all orders at the boundary only if every axis support stays inside the closed domain.
    Sum over all members: 1 on the domain (checked in the tests). No closed form for the integral (the denominator has none): a checked 1-D quadrature per axis."""
    k = to_real(k, "k")
    if not 0 < k <= MAX_K:
        raise Refused(f"partition_of_unity: k must be in (0, {MAX_K:g}] (got {k:g})")
    cs, rs, record = covering(D, "partition_of_unity", count, overlap, centres, radii)
    idx = per_axis(D, index, "index", to_int)
    factors, knots = [], []
    for j, (c, r, i, (a, b)) in enumerate(zip(cs, rs, idx, sides(D))):
        if not covers(a, b, c, r):
            raise Refused(f"partition_of_unity: the bumps of axis {j} do not cover [{a:g}, {b:g}] with margin (every "
                          f"point must lie within {SHRINK} x radius of a centre); increase the overlap or the radii")
        check_index(i, len(c), j, "bumps", "partition_of_unity")
        if min(c[i] + r[i], b) <= max(c[i] - r[i], a):
            raise Refused(f"partition_of_unity: bump {i} of axis {j} does not meet the interior of the domain")
        factors.append(unity_factor(a, b, c, r, k, i))
        knots.append(np.r_[np.asarray(c) - np.asarray(r), np.asarray(c) + np.asarray(r)])
    out = assemble(D, "partition_of_unity", {"index": pack(D, idx), **record, "k": k}, factors,
                   {"regularity": HORMANDER + "; quotient of C-infinity functions with a positive denominator",
                    "support": "the closed ball where t < 1 of the member's bump, per axis, cut to the domain"}, {})
    mass = _product_integral(factors, knots)
    if mass is not None:
        out.closed = MappingProxyType({**out.closed, "integral": (mass, CHECKED, "the product of the axis integrals, each by composite "
                                                                  "Gauss-Legendre between the bumps' ends, panels doubled until two results agree to 1e-12")})
    return out
