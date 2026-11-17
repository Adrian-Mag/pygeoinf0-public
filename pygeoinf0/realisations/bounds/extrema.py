"""Certified extrema of Legendre series on [-1, 1] (plan 057): a lower bound of the minimum by Bernstein enclosure,
refined by halving, with the least value found at points as the other end. Used for the per-cell bounds of a
pointwise box with varying bounds (optimisers/dual/nonsmooth.py cell_bounds)."""
from functools import lru_cache

import numpy as np
from numpy.polynomial import legendre as npleg

from ..series.legendre import EPS, _ro


@lru_cache(maxsize=None)
def _bernstein(n):
    """(B, |B|, F): F[j][k] the exact rational matrix taking a Legendre series c to its Bernstein coefficients (degree
    n - 1 on [-1, 1]; P_k(2s - 1) = sum_i (-1)^(k+i) C(k, i) C(k+i, i) s^i, b_j = sum_i C(j, i) / C(d, i) a_i), B = F
    rounded once."""
    from fractions import Fraction
    from math import comb
    d = n - 1
    F = []
    for j in range(n):
        w = [Fraction(comb(j, i), comb(d, i)) for i in range(j + 1)]
        F.append(tuple(sum((w[i] * ((-1) ** (k + i) * comb(k, i) * comb(k + i, i)) for i in range(min(j, k) + 1)),
                           Fraction(0)) for k in range(n)))
    B = np.array([[float(x) for x in row] for row in F])
    return _ro(B), _ro(np.abs(B)), tuple(F)


def _halves(b):
    """de Casteljau at the midpoint: the Bernstein coefficients on the two halves."""
    n = len(b)
    left, right, w = np.empty(n), np.empty(n), np.array(b, float)
    for j in range(n):
        left[j], right[n - 1 - j] = w[0], w[-1]
        w = 0.5 * (w[:-1] + w[1:])
    return left, right


def min_bracket(c, rtol=1e-13, kmax=40, steps=400):
    """(lower, upper) of min over [-1, 1] of the Legendre series c. lower is certified: on every subinterval a
    polynomial lies in the hull of its Bernstein coefficients (THEOREM), refined by halving where the minimum may lie,
    less a floating-point margin and the truncated tail sum_{k > kmax} |c_k| (|P_k| <= 1). upper is the least value
    found at points (the endpoints, the real roots of p', the subinterval ends): the minimum to rounding. They meet to
    rtol max(1, sum |c|) unless the steps run out; the gap is returned as it is."""
    import heapq

    from .sphere_l1 import _real_roots
    c = np.trim_zeros(np.asarray(c, float).ravel(), "b")
    if len(c) <= 1:
        v = float(c[0]) if len(c) else 0.0
        return v, v
    upper = float(np.min(npleg.legval(np.r_[-1.0, 1.0, _real_roots(npleg.legder(c))], c)))
    n = min(len(c), kmax + 1)
    B, A, F = _bernstein(n)
    s1 = float(np.abs(c).sum())
    tol = rtol * max(1.0, s1)
    b = B @ c[:n]
    conv = (16 * n) * EPS * (float((A @ np.abs(c[:n])).max()) + s1)          # the rounding of B c (gamma_n |B| |c|)
    if conv > 0.1 * tol:                                                       # exact: a float is a rational
        from fractions import Fraction
        cf = [Fraction(float(v)) for v in c[:n]]
        b = np.array([float(sum((f * v for f, v in zip(row, cf)), Fraction(0))) for row in F])
        conv = 2 * EPS * float(np.abs(b).max())                                  # one rounding per coefficient
    heap, depth, k = [(float(b.min()), 0, 0, b)], 0, 0
    while heap[0][0] < upper - tol and k < steps:
        _, _, lev, bb = heapq.heappop(heap)
        for h in _halves(bb):
            k += 1
            upper = min(upper, float(h[0]), float(h[-1]))
            heapq.heappush(heap, (float(h.min()), k, lev + 1, h))
        depth = max(depth, lev + 1)
    margin = conv + (4 * depth + 4) * EPS * float(np.abs(b).max()) + float(np.abs(c[n:]).sum())
    return min(heap[0][0] - margin, upper), upper


@lru_cache(maxsize=None)
def _grid(n, npts):
    t = np.linspace(-1.0, 1.0, npts)
    return _ro(npleg.legvander(t, n - 1)), 2.0 / (npts - 1)


def min_brackets(C, rtol=1e-9, npts=1025):
    """(lower, upper) of the minimum over [-1, 1] of every row of C (Legendre series, padded with zeros), vectorised.
    lower is certified: between grid points (spacing delta) a function lies above its chord less sup|p''| delta^2 / 8
    (THEOREM), sup|p''| <= sum |c''_k| (|P_k| <= 1), less a margin for the rounding of the values; upper is the least
    grid value. Rows whose gap exceeds rtol max(1, sum |c|) are refined by min_bracket (Bernstein, by halving)."""
    C = np.atleast_2d(np.asarray(C, float))
    m, n = C.shape
    if n == 1:
        return C[:, 0].copy(), C[:, 0].copy()
    V, delta = _grid(n, npts)
    vals = C @ V.T
    s1 = np.abs(C).sum(axis=1)
    m2 = np.abs(npleg.legder(C.T, 2)).sum(axis=0) * (1 + 1e-10) if n > 2 else np.zeros(m)
    upper = vals.min(axis=1)
    lower = upper - m2 * delta ** 2 / 8 - 4 * n * n * EPS * s1
    loose = np.nonzero(upper - lower > rtol * np.maximum(1.0, s1))[0]
    for i in loose:
        lo, up = min_bracket(C[i], rtol=rtol)
        lower[i], upper[i] = max(lower[i], lo), min(upper[i], up)
    return lower, upper
