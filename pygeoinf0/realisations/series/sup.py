"""Continuous functions on a closed interval in the sup norm: the piecewise Legendre realisation of
backends.IntervalBackend with three changes, for C([a, b]) (spaces/continuous.py).

    fit         refuses a function whose realisation jumps at a breakpoint (the adaptive fit locates jumps by bisection
                to adjacent floats and splits there; the pieces of a continuous function meet to rounding; see
                check_continuous for pieces the fit could not resolve), and one that is not finite where it is checked
    error       estimated in the sup norm: the largest difference between the function and its realisation at the
                Chebyshev points of the first kind (2 n + 9 per piece of degree n), which are not the sampling nodes;
                an estimate, not a bound (a feature between the check points is not seen)
    sup_norms   the maximum of |p|, column by column: on each piece the values on a grid uniform in theta = arccos t
                (16 n + 1 points, both ends included) are refined by bisection on p' in the bracket of every grid local
                maximum above 0.9 of the grid maximum. Bernstein's inequality |d/dtheta p(cos theta)| <= n ||p|| puts a
                grid point within (1 - pi/32) ||p|| of the maximum, so no bracket that holds the maximum is skipped

The same jump check serves the Sobolev spaces (plan 065 A1): check_sobolev refuses a realisation in H^s, s >= 1/2, that
jumps at a break, and for s >= m + 1/2 one whose m-th derivative jumps (rkhs.RKHSBackend.from_base).

Error estimates of sums and products stay sup-norm estimates (legendre.lincomb adds |alpha| err, legendre.multiply
uses sup bounds), so arithmetic keeps them meaningful.
"""
import math

import numpy as np
from numpy.polynomial import legendre as npleg

from ...core.errors import NotContinuous, NotDefined
from . import legendre as L
from ..backends import IntervalBackend

JUMP = 1e-6              # a gap between converged pieces above this (of the function's size) and 10 local errors: a jump
JUMP_UNRESOLVED = 1e-4   # the same across pieces the fit could not resolve (check_continuous)


def _refine(c, d, lo, hi):
    """A local maximum of |p| (coefficients c, derivative coefficients d) in [lo, hi]: the zero of p' there by
    bisection, or the better end when p' keeps its sign."""
    flo, fhi = npleg.legval(lo, d), npleg.legval(hi, d)
    if np.sign(flo) == np.sign(fhi) or flo == 0 or fhi == 0:
        return lo if abs(npleg.legval(lo, c)) >= abs(npleg.legval(hi, c)) else hi
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if not lo < m < hi:
            break
        fm = npleg.legval(m, d)
        if np.sign(fm) == np.sign(flo):
            lo, flo = m, fm
        else:
            hi = m
    return 0.5 * (lo + hi)


def piece_max(C):
    """max over t in [-1, 1] of |sum_k C[k, j] P_k(t)|, for each column j of C (shape (degree + 1, N))."""
    n = C.shape[0] - 1
    if n <= 1:
        return np.abs(L.polyval(np.array([-1.0, 1.0]), C)).max(axis=0)
    t = np.cos(np.linspace(0.0, math.pi, 16 * n + 1))           # from 1 down to -1
    V = np.abs(L.polyval(t, C))
    best = V.max(axis=0)
    D = npleg.legder(C)
    for j in range(C.shape[1]):
        v = V[:, j]
        loc = np.nonzero((v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] >= 0.9 * best[j]))[0] + 1
        for i in loc:
            x = _refine(C[:, j], D[:, j], t[i + 1], t[i - 1])
            best[j] = max(best[j], abs(float(npleg.legval(x, C[:, j]))))
    return best


def sup_norms(p):
    """The sup norm of each realised function (column) of the piecewise realisation p."""
    return np.max([piece_max(C) for C in p.coefs], axis=0)


def piece_errors(p, fun, N, domain):
    """(npieces, N): the largest |fun - p| on each piece at the Chebyshev points of the first kind (2 n + 9 per piece of
    degree n; not the Gauss nodes the fit sampled, and not the ends, where a function with a jump takes the value of
    one side only), at least the rounding level of the piece."""
    X, P = [], []
    for i, C in enumerate(p.coefs):
        k = 2 * C.shape[0] + 9
        t = np.cos(math.pi * (np.arange(k) + 0.5) / k)
        x0, x1 = p.breaks[i], p.breaks[i + 1]
        X.append(0.5 * (x0 + x1) + 0.5 * (x1 - x0) * t)
        P.append(L.polyval(t, C))
    x = np.concatenate(X)
    want = L._sample(fun, x, N)
    if not np.isfinite(want).all():
        bad = float(x[np.nonzero(~np.isfinite(want).all(axis=1))[0][0]])
        raise NotContinuous(f"the function is not finite at x = {bad!r}, so it is not an element of C({domain})")
    miss = np.abs(want - np.vstack(P))
    cuts = np.cumsum([0] + [len(z) for z in X])
    E = np.vstack([miss[cuts[i]:cuts[i + 1]].max(axis=0) for i in range(p.npieces)])
    floor = np.array([4 * L.EPS * max(C.shape[0], 2) * np.abs(C).sum(axis=0) for C in p.coefs])
    return np.maximum(E, floor)


def first_jump(p, E, jump=None, scale=None):
    """The first jump of the realisation p, (column, x, gap, gap / scale), or None (E: its errors per piece). Where two
    converged pieces meet, a gap above jump (default JUMP) of the function's size (scale, default p.sup_bound()) and
    above 10 times their errors is a jump: the pieces of a continuous function meet to within their errors. Pieces the
    fit could not resolve (a few ulps wide, kept as their mean: at a cusp, or at a jump it could not locate) are
    bridged, with JUMP_UNRESOLVED (a cusp |x - x0|^a has gaps of about 1e-14^a there, so continuous cusps sharper than
    a = 1/4 look like jumps). Smaller jumps are not detected: continuity is checked, not proved."""
    if p.npieces == 1:
        return None
    jump = JUMP if jump is None else jump
    a, b = p.breaks[0], p.breaks[-1]
    tiny = np.diff(p.breaks) <= 128 * L.EPS * max(b - a, abs(a), abs(b))
    right = np.vstack([L.polyval(np.array([1.0]), C) for C in p.coefs])        # value at the right end of each piece
    left = np.vstack([L.polyval(np.array([-1.0]), C) for C in p.coefs])
    scale = np.maximum(p.sup_bound() if scale is None else scale, np.finfo(float).tiny)
    i, m = 0, p.npieces
    while i < m - 1:
        if tiny[i]:
            i += 1
            continue
        j = i + 1
        while j < m and tiny[j]:
            j += 1
        if j == m:
            break
        gap = np.abs(right[i] - left[j])
        limit = np.maximum((jump if j == i + 1 else max(jump, JUMP_UNRESOLVED)) * scale, 10 * (E[i] + E[j]))
        if np.any(gap > limit):
            k = int(np.argmax(gap / limit))
            x = float(p.breaks[i + 1] if j == i + 1 else 0.5 * (p.breaks[i + 1] + p.breaks[j]))
            return k, x, float(gap[k]), float(gap[k] / scale[k])
        i = j
    return None


def check_continuous(p, E, domain):
    """Raise NotContinuous when the realisation p jumps (E: its errors per piece, from piece_errors; first_jump)."""
    hit = first_jump(p, E)
    if hit is not None:
        _, x, gap, rel = hit
        raise NotContinuous(
            f"the function jumps by {gap:.3g} at x = {x!r} ({rel:.1e} of its size), so it is not "
            f"an element of C({domain}). (A continuous function steeper there than the realisation can resolve "
            "looks the same; it cannot be realised to this tolerance.)")


# ------------------------------------------------------------------------- jumps in Sobolev spaces (plan 065 A1)
JUMP_DERIVATIVE = 1e-3   # a gap of an m-th derivative (m >= 1) above this, of the derivative's size: a jump (fits of
                         # functions smooth at a break meet there to about 1e-5 in the first derivative: measured)


def _derivative(p, m):
    """The m-th derivative of a piecewise Legendre realisation, piece by piece (exact on the polynomials)."""
    coefs = []
    for i, C in enumerate(p.coefs):
        h = p.breaks[i + 1] - p.breaks[i]
        coefs.append(npleg.legder(C, m, scl=2.0 / h, axis=0) if C.shape[0] > m else np.zeros((1, C.shape[1])))
    return L.PiecewiseLegendre(p.breaks, coefs, np.zeros(p.N))


def piece_bounds(p):
    """(npieces, N): an estimate of the sup error of each piece without samples: the realisation's own error estimate
    p.err (in L2) as a sup on a piece of degree n and width h, (n + 1) / sqrt(h) p.err (Nikolskii's inequality for
    polynomials), at least the rounding level of the piece."""
    err = np.asarray(p.err, float)
    err = np.where(np.isfinite(err), err, 0.0)
    out = []
    for i, C in enumerate(p.coefs):
        h = max(p.breaks[i + 1] - p.breaks[i], np.finfo(float).tiny)
        out.append(np.maximum(err * C.shape[0] / np.sqrt(h), 4 * L.EPS * max(C.shape[0], 2) * np.abs(C).sum(axis=0)))
    return np.vstack(out)


def sobolev_orders(s):
    """The derivative orders m whose jumps put a piecewise smooth function on an interval outside H^s: m + 1/2 <= s
    (a function with a jump of its m-th derivative is in H^t exactly for t < m + 1/2: Grisvard 1985, 1.4)."""
    return range(int(math.floor(float(s) + 0.5)))


def check_sobolev(p, s, space):
    """Raise NotAnElement when the realisation p (piecewise Legendre, in flat coordinates) jumps at a break in its m-th
    derivative for some m with m + 1/2 <= s (sobolev_orders): such a function is not in H^s. Values as for C
    (first_jump, with the errors of piece_bounds); for m >= 1 the m-th derivative with JUMP_DERIVATIVE of its size and
    ten times its estimated error E (n (n + 1) / h)^m on a piece of degree n and width h (n (n + 1) / h: d/dx P_n at
    the ends). Checked at the breaks of the realisation, not proved: a jump below these limits, or a kink inside a
    piece the fit accepted (the adaptive fit splits at jumps of values, not of derivatives), is not seen."""
    from ...core.errors import NotAnElement
    E = piece_bounds(p)
    a, b = p.breaks[0], p.breaks[-1]
    width = np.diff(p.breaks)
    tiny = width <= 128 * L.EPS * max(b - a, abs(a), abs(b))
    for m in sobolev_orders(s):
        if m == 0:
            hit = first_jump(p, E)
        else:
            q = _derivative(p, m)
            n = np.array([max(C.shape[0] - 1, 1) for C in p.coefs], float)
            Em = E * ((n * (n + 1) / np.maximum(width, np.finfo(float).tiny)) ** m)[:, None]
            big = [np.abs(C).sum(axis=0) for C, t in zip(q.coefs, tiny) if not t]
            scale = np.max(big, axis=0) if big else q.sup_bound()
            hit = first_jump(q, Em, jump=JUMP_DERIVATIVE, scale=scale)
        if hit is None:
            continue
        k, x, gap, rel = hit
        what = "jumps" if m == 0 else f"its {'derivative' if m == 1 else f'derivative of order {m}'} jumps"
        member = f" (member {k})" if p.N > 1 else ""
        raise NotAnElement(
            f"the function{member} {what} by {gap:.3g} at x = {x!r} ({rel:.1e} of its size), so it is not an element of "
            f"{space}: a piecewise smooth function whose derivative of order {m} jumps is in H^t only for t < {m + 0.5:g}",
            needs=(f"a space it is in: H^t with t < {m + 0.5:g}" + (" (or L2)" if m == 0 else ""),
                   "a function without the jump (if it is continuous there but steeper than the realisation resolves, "
                   "it cannot be realised to this tolerance: give breaks= at the steep part)"))


def with_sup_error(p, fun, N, domain):
    """Check that p is continuous, and return it with its error estimate in the sup norm."""
    E = piece_errors(p, fun, N, domain)
    check_continuous(p, E, domain)
    return L.PiecewiseLegendre(p.breaks, p.coefs, E.max(axis=0))


class SupIntervalBackend(IntervalBackend):
    """The realisation of C([a, b]): piecewise Legendre series, continuity checked, errors and norms in the sup norm."""

    sparse_families = False

    def fit(self, fun, N=1, breaks=None):
        pl, u = L.fit(fun, self.a, self.b, N=N, breaks=breaks, tol=self.tol)
        return with_sup_error(pl, fun, N, self.domain), u

    def fit_at(self, fun, N=1, breaks=None, w=None):
        pl, u = super().fit_at(fun, N, breaks, w)
        return with_sup_error(pl, fun, N, self.domain), u

    def fit_family(self, fun_idx, N, breaks=None, sparse=None, name="F", w=None):
        R = super().fit_family(fun_idx, N, breaks=breaks, sparse=False, name=name, w=w)
        idx = np.arange(N)
        return with_sup_error(R, lambda x: fun_idx(x, idx), N, self.domain)

    def error(self, R):
        return R.err

    def sup_norms(self, R):
        return sup_norms(R)

    def sqnorms(self, R):
        return sup_norms(R) ** 2

    def to_lp(self, R, p):
        """The same functions as a realisation in L^p([a, b]) of Lebesgue measure: the error estimate becomes an
        L^p one, ||f||_p <= (b - a)^(1/p) ||f||_inf."""
        return L.PiecewiseLegendre(R.breaks, R.coefs, R.err * (self.b - self.a) ** (0.0 if math.isinf(p) else 1.0 / p))

    def gram(self, A, B):
        raise NotDefined("C([a, b]) has no inner product (the sup norm fails the parallelogram law)",
                         needs=("pg.Inclusion(C, pg.L2(I)) for L2 inner products of continuous functions",))
