"""Separable functionals against dense radial-harmonic functions, and such functions built directly (plan 031).

    gram_sr_rh(A, B)                                <A_i, B_j> by one table of kernels: refined once per piece, not per harmonic
    harmonic_element(space, radii, coefficients)    the element sum_lm f_lm(r) Y_lm from f_lm at radii, no sampling or fitting
"""
import math

import numpy as np
from numpy.polynomial import legendre as npleg

from ...spaces.elements import Function
from ...core.errors import Unsupported
from . import legendre as LG
from .radial_harmonic import RadialHarmonicSeries, _nodes, _refine3, rgram

GRAM_SR_RH_MAX = 5e7       # entries of the table (radial columns, harmonics, members) that _gram_sr_rh keeps at once


def gram_sr_rh(A, B):
    """<A_i, B_j> for separable A and dense B, exact. A table of int k_c (B_lm)_j r^2 dr per radial column c, harmonic lm and
    member j, on the common breaks: the kernels are refined once per piece, not once per harmonic (plan 031), and any
    member of A reads its one entry. Too many columns and harmonics for a table: one harmonic at a time."""
    keep = A.lm < B.nlm
    us, u_of = np.unique(A.lm[keep], return_inverse=True)
    cs, c_of = np.unique(A.col[keep], return_inverse=True)
    if not len(us):
        return np.zeros((A.N, B.N))
    if len(us) * len(cs) > GRAM_SR_RH_MAX:
        return _by_harmonic(A, B)
    out = np.zeros((A.N, B.N))
    step = max(1, int(GRAM_SR_RH_MAX // (len(us) * len(cs))))
    P = A.pl.take(cs)
    nb = LG.union_breaks(P.breaks, B.breaks)
    CP = LG.refine(P, nb)
    CB = _refine3(RadialHarmonicSeries(B.L, B.breaks, [C[:, us, :] for C in B.coefs], B.err), nb)
    for s in range(0, B.N, step):
        table = np.zeros((len(cs), len(us) * min(step, B.N - s)))
        for i in range(len(nb) - 1):
            C = CB[i][:, :, s:s + step]
            if not C.any():
                continue
            t, w = _nodes(nb[i], nb[i + 1], (CP[i].shape[0] + C.shape[0]) // 2 + 2)
            VB = np.tensordot(npleg.legvander(t, C.shape[0] - 1), C, axes=(1, 0))        # (q, harmonics, members)
            table += (LG.polyval(t, CP[i]) * w[:, None]).T @ VB.reshape(len(t), -1)
        table = table.reshape(len(cs), len(us), -1)
        out[np.ix_(np.flatnonzero(keep), np.arange(s, s + table.shape[2]))] = table[c_of, u_of]
    return out


def _by_harmonic(A, B):
    out = np.zeros((A.N, B.N))
    for u in np.unique(A.lm):
        if u >= B.nlm:
            continue
        ia = np.flatnonzero(A.lm == u)
        ca, inv = np.unique(A.col[ia], return_inverse=True)
        out[ia] = rgram(A.pl.take(ca), LG.PiecewiseLegendre(B.breaks, [C[:, u, :] for C in B.coefs], B.err))[inv]
    return out


def piecewise_linear(a, b, radii, coefficients):
    """The RadialHarmonicSeries realisation on [a, b] from values of the coefficients at radii (ascending): linear between them, zero outside
    them with a jump. coefficients: (radii, (L + 1)^2), order l^2 + j, the package's harmonics."""
    r, c = np.asarray(radii, float).ravel(), np.asarray(coefficients, float)
    L = math.isqrt(c.shape[1]) - 1 if c.ndim == 2 else -1
    if c.ndim != 2 or c.shape[0] != r.size or (L + 1) ** 2 != c.shape[1]:
        raise ValueError("coefficients must have shape (radii, (L + 1)^2)")
    if r.size < 2 or np.any(np.diff(r) <= 0) or r[0] < a or r[-1] > b:
        raise ValueError(f"radii must be at least two, ascending, within [{a:g}, {b:g}]")
    breaks = np.unique(np.r_[a, r, b])
    k = np.searchsorted(r, breaks[:-1], side="right") - 1                 # the interval of r each piece starts in
    coefs = [np.zeros((1, c.shape[1], 1)) if j < 0 or j >= r.size - 1 else
             np.stack([0.5 * (c[j] + c[j + 1]), 0.5 * (c[j + 1] - c[j])])[:, :, None] for j in k]
    return RadialHarmonicSeries(L, breaks, coefs, 0.0)


def harmonic_element(space, radii, coefficients, name="f"):
    """The element sum_lm f_lm(r) Y_lm of a radial-harmonic space (module docstring)."""
    build = getattr(space.backend, "piecewise_linear", None)
    if build is None:
        raise Unsupported(f"harmonic_element needs a radial-harmonic realisation, not {space.realisation!r}")
    return Function(space, pl=build(radii, coefficients), name=name)
