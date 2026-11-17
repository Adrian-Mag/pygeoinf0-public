"""The quadrature rules of the package, in one place (plan 069 P5; architecture review 2026-11-13, B13 and section 2,
"The numerical column"). A rule is a function returning nodes and weights; each states its exactness on itself:
rule.exact_degree(n) is the polynomial degree it integrates exactly (per panel for the composite rules, total degree on
the simplex), and every rule here is sharp there (one degree more is not integrated exactly); rule.exactness says it in
words, with the error claim it carries for other integrands.

    gauss(n)                  Gauss-Legendre on [-1, 1], n nodes, SciPy's nodes refined (see gauss): degree 2n - 1
    gauss01(n)                the same rule on [0, 1]
    panels(edges, n)          composite Gauss-Legendre, n nodes on each panel [e_i, e_(i+1)]: degree 2n - 1 per panel
    graded_unit(n, cells)     panels on [0, 1] graded geometrically towards 0 (graded_edges), the cell [0, h] left out
    both_ends(n, cells)       panels on [0, 1] graded towards both ends (both_ends_edges)
    simplex(k, n)             the conical product (Stroud) rule on {u >= 0, sum u <= 1} in R^k, n^k nodes: degree 2n - 1
    trapezoid(n, a, b)        the n-node uniform trapezoid rule on [a, b]: degree 1, error (b - a) h^2 f''(xi) / 12

The rules of sets (plan 069 P2, review C2: a region keeps its chart, Jacobian and measure; the rule that integrates
over it lives here, reached by its geometry):

    rule(S, n)                nodes (m, d) and weights (m,) for the Hausdorff measure of the region S, built from its
                              charts S.charts() (sets/regions.py: Chart, Simplices): chart_rule on each box chart,
                              cells_rule on simplicial cells; a region without charts has none (refused)
    chart_rule(chart, n)      n Gauss points on every piece of each non-periodic axis (pieces between the chart's
                              breaks), n equispaced points on periodic axes, weights times the Jacobians and density
    cells_rule(cells, n)      simplex(k, n) mapped to every cell, times its scale (and its lift onto a curved region)
    graded(a, b, n, points, breaks, exponent)
                              panels on [a, b] split at breaks and graded geometrically towards points where integrands
                              may be singular, with the innermost cell by Gauss-Jacobi for a declared |x - x0|^-a
                              (moved from measures/weights.nodes_on)
    jacobian_nodes(name)      the extra Gauss points a tensor rule takes for a Jacobian ("1", "r", "r2", "sin", "c*w";
                              moved from domains/base.WEIGHTS)

The module is the numerical column's lowest layer: it imports NumPy and SciPy only, and every mathematical layer may
use it (tests/test_structure.py: its own rank in LAYERS, after settings and before core); it reads regions through their
charts only (duck-typed: it imports no set class). pg.realisations.Trapezoid is the realisation option built on
trapezoid (with its restriction to sub-intervals and its error claim). Some weight-adapted rules stay with the kernels
that need them (realisations/rkhs/weighted.py, core/adaptive.py).
"""
import math
from functools import lru_cache

import numpy as np
from scipy.special import roots_jacobi, roots_legendre

RATIO = 0.25          # geometric grading of graded_unit and both_ends: the ratio of consecutive cells
DEPTH = 1e-14         # graded_unit's smallest cell (relative), by default
OUTER_DEPTH = 1e-8    # both_ends's smallest cell at each end, by default
RULES = ("gauss", "gauss01", "panels", "graded_unit", "both_ends", "trapezoid", "simplex")


def _states(degree, exactness):
    """Mark a rule with its exactness: degree(n) the degree it integrates exactly, exactness the statement in words."""
    def mark(rule):
        rule.exact_degree, rule.exactness = degree, exactness
        return rule
    return mark


def _ro(a):
    a.flags.writeable = False
    return a


def _legendre_and_derivative(n, x):
    p0, p1 = np.ones_like(x), x.copy()
    for k in range(1, n):
        p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
    return p1, n * (x * p1 - p0) / (x * x - 1)


@_states(lambda n: 2 * n - 1,
         "exact for polynomials of degree 2n - 1 on [-1, 1], and sharp (P_n^2, of degree 2n, vanishes at the nodes); "
         "for f in C^(2n) the error is 2^(2n+1) (n!)^4 / ((2n+1) ((2n)!)^3) f^(2n)(xi); the weights hold the Gram of "
         "orthonormal Legendre polynomials to 5e-14 up to n = 1500 (numpy's leggauss: 1.2e-11 there)")
@lru_cache(maxsize=None)
def gauss(n):
    """Gauss-Legendre nodes and weights on [-1, 1] (read-only, cached). SciPy's nodes, refined by Newton's method on
    P_n, with the weights recomputed as 2 / ((1 - x^2) P_n'(x)^2): SciPy's own weights carry relative errors of 5e-11 at
    n = 128, which the rule then commits on every integral (measured: Gram matrices of orthonormal polynomials 1.6e-12
    from the identity, 1.7e-14 after the refinement)."""
    t, _ = roots_legendre(n)
    t = np.asarray(t, float).copy()
    if n > 1:
        for _ in range(2):
            p, dp = _legendre_and_derivative(n, t)
            t -= p / dp
        _, dp = _legendre_and_derivative(n, t)
        w = 2.0 / ((1.0 - t * t) * dp * dp)
    else:
        w = np.array([2.0])
    return _ro(t), _ro(w)


@_states(gauss.exact_degree, "gauss mapped to [0, 1]: exact for polynomials of degree 2n - 1, and sharp")
@lru_cache(maxsize=None)
def gauss01(n):
    """Gauss-Legendre nodes and weights on [0, 1] (read-only, cached): (t + 1) / 2 and w / 2 of gauss(n)."""
    t, w = gauss(n)
    return _ro(0.5 * (t + 1)), _ro(0.5 * w)


@_states(gauss.exact_degree,
         "exact for piecewise polynomials of degree 2n - 1 between consecutive edges (each panel's Gauss rule), and "
         "sharp on every panel")
def panels(edges, n):
    """Composite Gauss-Legendre: n nodes on each panel [e_i, e_(i+1)] of the increasing edges, panel by panel."""
    e = np.asarray(edges, float)
    g, w = gauss(n)
    mid, half = 0.5 * (e[1:] + e[:-1]), 0.5 * np.diff(e)
    return (mid[:, None] + half[:, None] * g).ravel(), (half[:, None] * w).ravel()


def graded_edges(cells, depth=DEPTH, ratio=RATIO):
    """The edges of graded_unit: ratio^k, ..., ratio (k = ceil(log depth / log ratio)), then `cells` uniform cells on
    [ratio, 1]."""
    k = int(math.ceil(math.log(depth) / math.log(ratio)))
    return np.unique(np.r_[ratio ** np.arange(k, 0, -1), np.linspace(ratio, 1.0, cells + 1)])


@_states(gauss.exact_degree,
         "panels on graded_edges(cells, depth, ratio): exact for piecewise polynomials of degree 2n - 1 between those "
         "edges on [h, 1], and sharp on every panel; the cell [0, h] is not covered (callers add it)")
def graded_unit(n, cells, depth=DEPTH, ratio=RATIO):
    """(nodes, weights, h) on [0, 1] graded towards 0 down to the cell [0, h] (left out: callers add it), then
    `cells` uniform cells on [ratio, 1]."""
    e = graded_edges(cells, depth, ratio)
    x, w = panels(e, n)
    return x, w, float(e[0])


def both_ends_edges(cells, depth=OUTER_DEPTH):
    """The edges of both_ends: those of graded_unit (with the cell [0, h]) halved, and their mirror image on
    [1/2, 1]."""
    e = np.r_[0.0, graded_edges(max(1, cells // 2), depth)] * 0.5
    return np.unique(np.r_[e, 1 - e[::-1]])


@_states(gauss.exact_degree,
         "panels on both_ends_edges(cells, depth): exact for piecewise polynomials of degree 2n - 1 between those "
         "edges on [0, 1], and sharp on every panel")
def both_ends(n, cells, depth=OUTER_DEPTH):
    """Nodes and weights on [0, 1] graded towards both ends."""
    x, w, h = graded_unit(n, max(1, cells // 2), depth)
    x0, w0 = panels([0.0, h], n)
    x, w = np.r_[x0, x] * 0.5, np.r_[w0, w] * 0.5
    return np.r_[x, 1 - x[::-1]], np.r_[w, w[::-1]]


@_states(gauss.exact_degree,
         "exact for polynomials of total degree 2n - 1 on the reference simplex, and sharp (in u_1, a Gauss-Jacobi "
         "rule of n nodes for the weight (1 - u_1)^(k-1)); weights sum to 1/k!; k = 0: the point, with weight 1")
def simplex(k, n):
    """Nodes (n^k, k) in the reference simplex {u >= 0, sum u <= 1} and weights (n^k,) summing to 1/k!: the conical
    product of Gauss-Jacobi rules (Stroud; the collapsed coordinates u_j = xi_j prod_(i<j) (1 - xi_i), xi_j with the
    weight (1 - xi_j)^(k-1-j)). k = 0: one point, weight 1."""
    if k == 0:
        return np.zeros((1, 0)), np.ones(1)
    axes = []
    for j in range(k):
        alpha = k - 1 - j
        x, w = roots_jacobi(n, alpha, 0)
        axes.append((0.5 * (1 + x), w / 2.0 ** (alpha + 1)))
    grids = np.meshgrid(*[a[0] for a in axes], indexing="ij")
    W = np.ones_like(grids[0])
    for j, (_, w) in enumerate(axes):
        shape = [1] * k
        shape[j] = n
        W = W * w.reshape(shape)
    xi = [g.ravel() for g in grids]
    U, rest = [], np.ones_like(xi[0])
    for j in range(k):
        U.append(xi[j] * rest)
        rest = rest * (1 - xi[j])
    return np.stack(U, axis=1), W.ravel()


@_states(lambda n: 1,
         "exact for polynomials of degree 1, and sharp (x^2 is not); for f in C^2[a, b] the error is "
         "(b - a) h^2 f''(xi) / 12 with h = (b - a) / (n - 1)")
def trapezoid(n, a, b):
    """The n-node uniform trapezoid rule on [a, b] (n >= 2): nodes linspace(a, b, n), weights h/2, h, ..., h, h/2."""
    n = int(n)
    if n < 2:
        raise ValueError("the trapezoid rule needs at least 2 nodes")
    x = np.linspace(a, b, n)
    w = np.full(n, (b - a) / (n - 1))
    w[[0, -1]] *= 0.5
    return x, w


# ------------------------------------------------------------------ rules of sets (plan 069 P2)
JACOBIAN_NODES = {"1": 0, "r": 1, "r2": 2, "sin": 12}   # extra Gauss points for a Jacobian: its polynomial degree, or
                                                         # 12 for sin (its Legendre coefficients fall below 1e-16 by
                                                         # degree about 20)


def jacobian_nodes(name):
    """The extra Gauss points a tensor rule takes so that a Jacobian named name ("1", "r", "r2", "sin", or "c*w", the
    weight w times a constant c: domains/base.scaled) costs no accuracy: exact for the polynomial ones, below 1e-16 for
    sin. KeyError for other names."""
    c, sep, base = str(name).partition("*")
    if sep:
        float(c)
        name = base
    return JACOBIAN_NODES[name]


@_states(gauss.exact_degree,
         "on each box chart, exact for integrands whose pull-back times the Jacobians and density is a polynomial of "
         "degree 2n - 1 in every non-periodic coordinate on every piece between the chart's breaks, and a "
         "trigonometric polynomial of degree < n in every periodic one; on simplicial cells, simplex(k, n)'s total "
         "degree in the flat coordinates; for other integrands a rule, not a claim (callers compare two orders)")
def rule(S, n=16):
    """Nodes (m, cartesian_dim) and weights (m,) for the Hausdorff measure of the region S (its volume, area, length),
    from its charts S.charts(): chart_rule on every box chart, cells_rule on simplicial cells, concatenated. A region
    whose charts() is None (known by membership only: R, a LayeredBall, a Partition's union) has no rule here:
    NotImplementedError naming it."""
    charts = S.charts() if callable(getattr(S, "charts", None)) else None
    if charts is None:
        raise NotImplementedError(f"{S!r} states no chart, so no rule integrates over it here (a Domain, an Interval, "
                                  "a Simplicial set or a Surface does)")
    parts = [cells_rule(c, n) if hasattr(c, "origins") else chart_rule(c, n) for c in charts]
    if len(parts) == 1:
        return parts[0]
    return np.concatenate([X for X, _ in parts]), np.concatenate([w for _, w in parts])


def chart_rule(chart, n=16):
    """The tensor rule of one box chart (sets/regions.Chart): n Gauss points per piece of every non-periodic axis
    (pieces between its breaks), n equispaced points of weight (hi - lo) / n on periodic axes; the axis weights times
    the axis Jacobians, their outer product times the density at the nodes; the nodes mapped by to_cartesian."""
    axes, wts = [], []
    brk = chart.breaks or (None,) * len(chart.reference)
    jac = chart.jacobians or (None,) * len(chart.reference)
    for k, (lo, hi) in enumerate(chart.reference):
        if chart.periodic[k]:
            t = lo + (hi - lo) * np.arange(n) / n
            w = np.full(n, (hi - lo) / n)
        else:
            e = sorted(set([lo, hi] + [b for b in (brk[k] or ()) if lo < b < hi]))
            g, gw = gauss(n)
            t = np.concatenate([0.5 * (a + b) + 0.5 * (b - a) * g for a, b in zip(e[:-1], e[1:])])
            w = np.concatenate([0.5 * (b - a) * gw for a, b in zip(e[:-1], e[1:])])
        axes.append(t)
        wts.append(w if jac[k] is None else w * jac[k](t))
    G = np.meshgrid(*axes, indexing="ij")
    W = wts[0]
    for w in wts[1:]:
        W = np.multiply.outer(W, w)
    W = W.ravel()
    if chart.density is not None:
        W = W * np.asarray(chart.density(*G), float).ravel()
    X = np.stack([np.ravel(c) for c in np.broadcast_arrays(*chart.to_cartesian(*G))], axis=1)
    return X, W


def cells_rule(cells, n=16):
    """The rule of simplicial cells (sets/regions.Simplices): simplex(k, n) mapped to every cell o_j + u E_j, its
    weights times the cell's scale, then lifted (points and measure factor) when the cells carry a lift."""
    o, E = np.asarray(cells.origins, float), np.asarray(cells.edges, float)
    U, W = simplex(E.shape[1], n)
    m = len(o)
    j = np.repeat(np.arange(m), len(W))
    X = o[j] + np.einsum("qk,qkd->qd", np.tile(U, (m, 1)), E[j])
    w = np.tile(W, m) * np.asarray(cells.scale, float)[j]
    if cells.lift is not None:
        X, f = cells.lift(X, j)
        w = w * f
    return X, w


GRADING, LEVELS = 0.15, 48   # graded's geometric grading towards singular points (ratio, number of cells)


@_states(gauss.exact_degree,
         "exact for piecewise polynomials of degree 2n - 1 on each panel between the breaks and graded cells; the "
         "innermost cell [x0, x0 + GRADING^LEVELS h] at a point x0 is Gauss-Jacobi for |x - x0|^-exponent (exact for "
         "that weight times polynomials of degree 2n - 1) when 0 < exponent < 1, else Gauss-Legendre")
def graded(a, b, n=32, points=(), breaks=(), exponent=None):
    """Nodes (m,) and weights (m,) of Lebesgue measure on [a, b]: Gauss-Legendre with n points per piece (pieces split
    at breaks, graded geometrically towards the points, where integrands may be singular). The innermost graded cell,
    [x0, x0 + GRADING^LEVELS h], is integrated too: with the declared exponent a (|w| ~ dist^-a, 0 < a < 1) by
    Gauss-Jacobi for the weight dist^-a (its weights divided by dist^-a at the nodes, so that W w is the Jacobi rule of
    w dist^a), else by Gauss-Legendre. (Moved from measures/weights.nodes_on, plan 069 P2; plan 054, wrong row Q7: the
    cell was dropped, a relative loss of (GRADING^LEVELS)^(1 - a), seen by neither order.)"""
    pts = sorted({float(p) for p in points if a <= p <= b})
    edges = sorted({a, b} | set(pts) | {float(c) for c in breaks if a < c < b})
    g, gw = gauss(n)
    cells, inner = [], []
    for lo0, hi0 in zip(edges[:-1], edges[1:]):
        left, right = lo0 in pts, hi0 in pts
        if not (left or right):
            cells.append((lo0, hi0))
            continue
        m = 0.5 * (lo0 + hi0) if left and right else None
        for lo, hi, toward in ((lo0, m if m is not None else hi0, "lo") if left else (None, None, None),
                               (m if m is not None else lo0, hi0, "hi") if right else (None, None, None)):
            if lo is None:
                continue
            q = (hi - lo) * GRADING ** np.arange(LEVELS + 1)            # cell sizes towards the singular end
            if toward == "lo":
                cells += [(lo + q[k + 1], lo + q[k]) for k in range(LEVELS)]
                inner.append((lo, q[LEVELS], 1.0))
            else:
                cells += [(hi - q[k], hi - q[k + 1]) for k in range(LEVELS)]
                inner.append((hi, q[LEVELS], -1.0))
    X = [0.5 * (c + d) + 0.5 * (d - c) * g for c, d in cells]
    W = [0.5 * (d - c) * gw for c, d in cells]
    for x0, eps, sgn in inner:
        x, w = _singular_cell(x0, eps, sgn, n, exponent, g, gw)
        X.append(x)
        W.append(w)
    return np.concatenate(X), np.concatenate(W)


def _singular_cell(x0, eps, sgn, n, exponent, g, gw):
    """Nodes and Lebesgue weights on the cell from x0 (the singular point) to x0 + sgn eps (graded)."""
    if exponent is not None and 0.0 < exponent < 1.0:
        t, lam = roots_jacobi(n, 0.0, -exponent)                    # weight (1 + t)^-a on [-1, 1]
        dist = 0.5 * eps * (1.0 + t)
        return x0 + sgn * dist, (0.5 * eps) ** (1.0 - exponent) * lam * dist ** exponent
    return x0 + sgn * 0.5 * eps * (1.0 + g), 0.5 * eps * gw
