"""The integrals of a pointwise box per realisation (plan 069 P3: the capabilities box_support, box_member and box_rule
of realisations/registry.py). The code moved here unchanged from convex/pointwise.py (the support and membership of
PointwiseBox, module docstring there) and inference/ocsola_box.py (the fixed search quadrature of oCSOLA with a box):

    capability     realisation                      what it computes
    box_support    piecewise Legendre (interval)    int (u q+ - l q-) dmu exactly: the roots of q per piece, the bounds
                                                    constant or piecewise polynomial (Lebesgue measure only)
                   spherical harmonics              the certified L1 bracket of sphere_l1.l1_bracket, times c R^2
                   radial x harmonic                the certified radial x angular bracket (sphere_l1.radial_l1_bracket)
    box_member     piecewise Legendre (interval)    l <= q <= u a.e., exactly (the extrema per piece)
    box_rule       piecewise Legendre (interval)    Gauss nodes per piece of the common breaks (oversampled)
                   spherical harmonics              the Gauss x equispaced grid of degree max(oversample (L + 1),
                                                    least) - 1

Bounds are floats, None (no bound) or realisations of the same space; the sphere and balls take one constant per
summand (constant_bounds)."""
import math

import numpy as np
from numpy.polynomial import legendre as npleg

from ...core.errors import Unsupported
from ...core.logic import Truth

INF = math.inf


def require_lebesgue(H):
    if type(getattr(H, "measure", None)).__name__ != "LebesgueMeasure":
        raise Unsupported(f"pointwise boxes in {H}: exact supports need L2 with Lebesgue measure on an interval")


def constant_bounds(H, l, u):
    """(l, u) as floats: the one constant per summand that the sphere and balls take."""
    if not (l is None or isinstance(l, float)) or not (u is None or isinstance(u, float)):
        raise Unsupported(f"pointwise boxes in {H}: bounds that vary over the sphere are not supported here (split "
                          f"the space where the envelope changes, as for envelope zones)")
    if l is None or u is None:
        raise Unsupported(f"one-sided pointwise sets in {H}: the sign of q is not certified on this realisation")
    return l, u


# ------------------------------------------------------------------------------------ piecewise Legendre (interval)
def pieces(R, l, u):
    """Common breaks and per piece the Legendre coefficients of q, l, u (None for infinite bounds)."""
    from ..series.legendre import refine, union_breaks
    pls = [R] + [b.realisation for b in (l, u) if b is not None and not isinstance(b, float)]
    nb = union_breaks(*[p.breaks for p in pls])
    qc = [C[:, 0] for C in refine(R, nb)]

    def coef(b):
        if b is None or isinstance(b, float):
            return [None if b is None else np.array([b])] * (len(nb) - 1)
        return [C[:, 0] for C in refine(b.realisation, nb)]
    return nb, qc, coef(l), coef(u)


def signs(c):
    """Sub-intervals of [-1, 1] and the sign of the Legendre series c on each (exact roots)."""
    from .sphere_l1 import _real_roots
    pts = np.r_[-1.0, _real_roots(c), 1.0]
    mid = 0.5 * (pts[:-1] + pts[1:])
    return pts, np.sign(npleg.legval(mid, c))


def interval_exact(R, l, u):
    nb, qc, lc, uc = pieces(R, l, u)
    total = 0.0
    for i in range(len(nb) - 1):
        c = np.trim_zeros(qc[i], "b")
        if not len(c):
            continue
        pts, sg = signs(c)
        jac = 0.5 * (nb[i + 1] - nb[i])
        for k, s in enumerate(sg):
            if s == 0 or pts[k + 1] <= pts[k]:
                continue
            b = uc[i] if s > 0 else lc[i]
            if b is None:
                return INF
            F = npleg.legint(npleg.legmul(c, b))
            total += jac * float(npleg.legval(pts[k + 1], F) - npleg.legval(pts[k], F))
    return total


def interval_support(R, H, l, u, rtol=1e-8):
    require_lebesgue(H)
    s = interval_exact(R, l, u)
    return s, s


def interval_member(R, H, l, u):
    require_lebesgue(H)
    nb, qc, lc, uc = pieces(R, l, u)
    for i in range(len(nb) - 1):
        for b, sgn in ((uc[i], 1.0), (lc[i], -1.0)):
            if b is None:
                continue
            d = sgn * npleg.legsub(qc[i], b)                     # <= 0 everywhere on the piece?
            ext = np.r_[-1.0, 1.0, signs(npleg.legder(d))[0][1:-1]] if len(d) > 1 else np.array([0.0])
            if float(np.max(npleg.legval(ext, d))) > 1e-12 * max(1.0, float(np.abs(qc[i]).sum())):
                return Truth.FALSE
    return Truth.TRUE


# ------------------------------------------------------------------------------------ the sphere, balls and shells
def _combined(l, u, lo, hi, mean):
    half, mid = 0.5 * (u - l), 0.5 * (u + l)                                 # u q+ - l q- = half |q| + mid q
    return half * lo + mid * mean, half * hi + mid * mean


def sphere_support(R, H, l, u, rtol=1e-8):
    from . import sphere_l1
    l, u = constant_bounds(H, l, u)
    C, L, r2 = np.asarray(R.C, float)[:, :1], R.L, H.backend.mass          # c R^2 (plan 064: c the density)
    lo, hi, _ = sphere_l1.l1_bracket(C, L, rtol=rtol)
    mean = float(C[0, 0]) * math.sqrt(4 * math.pi)                       # int g dxi (unit sphere)
    lo, hi = r2 * lo[0], r2 * hi[0]
    mean *= r2
    return _combined(l, u, lo, hi, mean)


def times_r2(C, a, b):
    """Legendre coefficients (d + 2, nlm) on [a, b] of r^2 f for f with coefficients C (d, nlm): exact."""
    from ..series.legendre import analysis
    from ...quadrature import gauss
    d = C.shape[0] + 2
    t, _ = gauss(d)
    r = 0.5 * (a + b) + 0.5 * (b - a) * t
    return analysis(d) @ ((npleg.legvander(t, C.shape[0] - 1) @ C) * (r * r)[:, None])


def radial_support(R, H, l, u, rtol=1e-8):
    from ..series import radial_harmonic as rh
    from . import sphere_l1
    l, u = constant_bounds(H, l, u)
    B = H.backend                                   # int |q| c rho r^2 dr dxi = c int |rho q| r^2 (rho >= 0)
    mean = float(B.integral(R)[0])
    R = rh.dense(B.weighted(R))
    coefs = [times_r2(C[:, :, 0], R.breaks[i], R.breaks[i + 1]) for i, C in enumerate(R.coefs)]
    lo, hi, _ = sphere_l1.radial_l1_bracket(R.breaks, coefs, R.L, rtol=rtol)
    lo, hi = B.c * lo, B.c * hi
    return _combined(l, u, lo, hi, mean)


# ------------------------------------------------------------------------------------ the search rules of oCSOLA
class IntervalRule:
    """Gauss nodes per piece of the common breaks of the members and the bounds (oversample (degree + 1) per piece):
    w, lo, hi (n,), rho2 = weight (b - a) sup(|l|, |u|)^2, exact True; values(R, cols, N) (n, N)."""
    kind, exact = "interval", True

    def __init__(self, H, Rs, l, u, weight, oversample, least=None):
        from ..series.legendre import evaluate, union_breaks
        from ...quadrature import gauss
        require_lebesgue(H)
        wk = weight
        pls = [R for R, _ in Rs if R is not None] + [b.realisation for b in (l, u) if not isinstance(b, float)]
        nb = union_breaks(*[p.breaks for p in pls])
        deg = max(max(p.degrees) for p in pls)
        tq, wgl = gauss(oversample * (deg + 1))
        h = 0.5 * np.diff(nb)
        x = ((0.5 * (nb[:-1] + nb[1:]))[:, None] + h[:, None] * tq[None, :]).ravel()
        self.w = (h[:, None] * wgl[None, :]).ravel() * wk
        self.lo = np.full(x.size, l) if isinstance(l, float) else evaluate(l.realisation, x)[:, 0]
        self.hi = np.full(x.size, u) if isinstance(u, float) else evaluate(u.realisation, x)[:, 0]
        sup = max(abs(b) if isinstance(b, float) else float(b.realisation.sup_bound()[0]) for b in (l, u))
        self.rho2 = wk * float(nb[-1] - nb[0]) * sup * sup
        self.x = x

    def values(self, R, cols, N, chunk=None):
        from ..series.legendre import evaluate
        V = np.zeros((self.x.size, N))
        if R is not None:
            V[:, cols] = evaluate(R, self.x)
        return V


class SphereRule:
    """The Gauss-Legendre x equispaced grid of degree Lg = max(oversample (L + 1), least) - 1 on a sphere summand, for
    constant bounds: w (the area weights times c R^2 and the weight), lo, hi, rho2 = weight c 4 pi R^2 max(l^2, u^2),
    exact False; values(R, cols, N, chunk) (n, N) by synthesis."""
    kind, exact = "sphere", False

    def __init__(self, H, Rs, l, u, weight, oversample, least):
        from ..series.spherical import grid
        wk = weight
        wk *= H.backend.c                       # a constant density on the sphere (plan 064)
        l, u = constant_bounds(H, l, u)
        R0 = next(R for R, _ in Rs if R is not None)
        L = max(R.L for R, _ in Rs if R is not None)
        Lg = max(oversample * (L + 1), least) - 1
        th, wq, ph, _ = grid(Lg)
        r2 = float(R0.R) ** 2
        self.w = (wq[:, None] * np.full(ph.size, 2 * math.pi / ph.size)[None, :]).ravel() * r2 * wk
        self.lo, self.hi = np.full(self.w.size, l), np.full(self.w.size, u)
        self.rho2 = wk * 4 * math.pi * r2 * max(l * l, u * u)
        self.L, self.Lg = L, Lg

    def values(self, R, cols, N, chunk):
        from ..series.spherical import synthesise_on
        L, Lg = self.L, self.Lg
        n = (Lg + 1) * (2 * Lg + 2)
        V = np.zeros((n, N))
        if R is not None:
            C = np.asarray(R.padded(L), float)
            step = max(1, chunk // (8 * n))
            for i in range(0, C.shape[1], step):
                V[:, cols[i:i + step]] = synthesise_on(C[:, i:i + step], L, Lg).reshape(n, -1)
        return V
