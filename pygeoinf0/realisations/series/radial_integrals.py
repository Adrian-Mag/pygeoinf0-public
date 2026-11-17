"""Radial integrals of piecewise Legendre functions in L2(r^2 dr) on [a, b] (plan 009 P4b; out of probability/degrees.py
in plan 070 P7: the numerics of the degree-structured covariances and of the separable Grams).

    nodes(a, b, pls, rule)              Gauss rules on the union of the factors' breakpoints, an order per piece from the
                                        factors' degrees (scaled down for a polynomial factor much longer than the
                                        piece), or a declared quadrature rule (its nodes(a, b))
    coefficients(R, P, m, a, b, rule)   <R_i, m phi_n> (m a radial multiplier or None)
    mgram(P, m, a, b, rule, diag)       <m phi_n, m phi_k> (diag: the squared norms only)
"""
import math

import numpy as np

from ...quadrature import gauss
from . import legendre as LG


def _eff_degree(p, lo, hi):
    """A degree sufficient for p's pieces on [lo, hi]: its own, scaled down when [lo, hi] is a small part of a piece."""
    br = p.breaks
    k = int(np.clip(np.searchsorted(br, 0.5 * (lo + hi), side="right") - 1, 0, p.npieces - 1))
    d = p.coefs[k].shape[0] - 1
    frac = (hi - lo) / max(br[k + 1] - br[k], 1e-300)
    return d if frac >= 0.5 else min(d, int(math.ceil(d * frac)) + 12)


def nodes(a, b, pls, rule=None, extra=0):
    """(x, W): nodes and weights of the integral int_a^b f r^2 dr for products of the given legendre.PiecewiseLegendre functions (each
    on an interval containing [a, b]); rule: a declared quadrature rule (its nodes(a, b))."""
    if rule is not None:
        x, w = rule.nodes(a, b)
        return np.asarray(x, float), np.asarray(w, float) * np.asarray(x, float) ** 2
    cuts = [np.array([a, b])] + [p.breaks[(p.breaks > a) & (p.breaks < b)] for p in pls]
    nb = LG.union_breaks(*cuts)
    xs, ws = [], []
    for lo, hi in zip(nb[:-1], nb[1:]):
        n = sum(_eff_degree(p, lo, hi) for p in pls) // 2 + 3 + extra
        t, w = gauss(n)
        x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * t
        xs.append(x)
        ws.append(0.5 * (hi - lo) * w * x * x)
    return np.concatenate(xs), np.concatenate(ws)


def coefficients(R, P, m, a, b, rule=None):
    """<R_i, m phi_n> on [a, b] in L2(r^2 dr): R and P legendre.PiecewiseLegendre (columns), m a one-column PiecewiseLegendre or None."""
    pls = [R, P] + ([m] if m is not None else [])
    x, W = nodes(a, b, pls, rule)
    if m is not None:
        W = W * values(m, x)[:, 0]
    return (values(R, x) * W[:, None]).T @ values(P, x)


def mgram(P, m, a, b, rule=None, diag=False):
    """<m phi_n, m phi_k> on [a, b] in L2(r^2 dr) (diag: the squared norms only)."""
    pls = [P, P] + ([m, m] if m is not None else [])
    x, W = nodes(a, b, pls, rule)
    if m is not None:
        W = W * values(m, x)[:, 0] ** 2
    V = values(P, x)
    return np.einsum("q,qn,qn->n", W, V, V) if diag else (V * W[:, None]).T @ V


def values(p, x):
    """The columns of the PiecewiseLegendre p at the points x."""
    return LG.evaluate(p, x)
