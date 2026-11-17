"""Certified one-dimensional searches of the support-function routes (moved from convex/dual.py by plan 063; made public and
joined by the columns' trust-region solve of convex/spectral.py, now optimisers/dual/spectral.py, by plan 070 P4):

    bisect(f, lo, hi)                 a root of a monotone f between lo (f > 0) and hi (f <= 0), geometric when lo > 0
    trs(num, den, r)                  y = num / (den + mu) with mu >= 0 the least value giving ||y|| <= r (the
                                      trust-region multiplier), by bisect: one direction
    trs_columns(c, D0, r, mu, mask)   the same multiplier for every column (a direction each) by Newton's method on
                                      psi(mu) = 1/||y(mu)|| - 1/r from a lower bound mu (More-Sorensen); trs_floor(c, D0,
                                      r) is such a lower bound

trs and trs_columns compute the same mathematical object, the least mu >= 0 with ||c / (D0 + mu)|| <= r, by two
methods: bisection to 1e-15 relative width (one direction) and Newton from a lower bound, stopped at ||y|| <= r (1 +
1e-15), within 60 steps (many directions at once: optimisers/dual/spectral.py's arrays). They agree to the stopping
rules, not to the bit; joining them would change the bounds' last digits, so both are kept (plan 070 P4, an open
decision).
"""
import math

import numpy as np


def bisect(f, lo, hi, iters=200):
    """A root of a monotone f between lo (f > 0) and hi (f <= 0), geometric when lo > 0: the hi end (f(hi) <= 0) after
    the bracket's relative width is at most 1e-15, or iters halvings."""
    for _ in range(iters):
        mid = math.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    return hi


def trs(num, den, r):
    """y = num / (den + mu), mu >= 0 the least value with ||y|| <= r (the trust-region multiplier), by bisect."""
    def y(mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / (den + mu)
        return np.where((den + mu) == 0, np.where(num == 0, 0.0, np.inf), out)
    y0 = y(0.0)
    if np.all(np.isfinite(y0)) and np.linalg.norm(y0) <= r:
        return y0, 0.0
    hi = 2.0 * np.linalg.norm(num) / r + 1e-300
    lo = hi
    while lo > 1e-300 * hi and np.linalg.norm(y(lo)) > r:
        lo *= 1e-3
    if np.linalg.norm(y(lo)) <= r:
        lo = 0.0
    mu = bisect(lambda m: np.linalg.norm(y(m)) - r, lo, hi)
    return y(mu), mu


def trs_columns(c, D0, r, mu, mask):
    """Newton on psi for y = c / (D0 + mu) with ||y|| = r (columns: directions); mu a starting lower bound (updated)."""
    for _ in range(60):
        D = D0 + mu[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(c == 0, 0.0, c / D)
        n2 = np.einsum("kj,kj->j", y, y)
        n = np.sqrt(n2)
        act = mask & (n > r * (1 + 1e-15))
        if not act.any():
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.einsum("kj,kj->j", y * y, np.where(c == 0, 0.0, 1.0 / D))
            step = (1.0 / n - 1.0 / r) * n2 * n / q                      # -psi / psi'
        mu = np.where(act, mu - step, mu)
    D = D0 + mu[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(c == 0, 0.0, c / D)
    return y, mu


def trs_floor(c, D0, r):
    """A lower bound of the trust-region multiplier: ||y|| >= |c_k| / (D0_k + mu) >= r up to this mu."""
    return np.maximum(0.0, (np.abs(c) / r - D0).max(axis=0))
