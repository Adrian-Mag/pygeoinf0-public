"""Adaptive fit of functions on a ball or shell in the radial x harmonic realisation (radial_harmonic.py).

1. Angular degree: at probe radii (Gauss points of every initial radial piece) the function is analysed on the
   spherical grid of degree L (spherical.grid: exact for degree <= L) for L = 7, 15, 23, ... until the coefficient energy
   of the top eighth of the degrees is below the tolerance (relative to the norm, or the rounding floor of the
   analysis); the result is truncated to the smallest degree with the same property, and the grid is kept (analysis on
   a smaller grid would alias).
2. Radial factors: on every piece, Gauss-Legendre samples of all (L + 1)^2 x N factors at 9, 17, 33, 65, 129 points
   until the tail of the Legendre coefficients is below max(tol, 2 eps n^1.5) times the largest value of the
   function (one scale per function, not per factor: factors that are rounding noise do not force refinement), else
   the piece is split at its midpoint.
The error estimate is the discarded radial energy (times b^2, the largest r^2) plus the angular energy beyond the kept
degree, integrated over the radial samples, plus a rounding term; an estimate, not a bound. When the degree reaches the
cap with energy left in the top eighth of the degrees, that energy is added (a proxy for the tail beyond the grid) and
the fit counts as unresolved: the cap is not convergence (plan 035 P2). A least degree (NaturalBreaks.degree, from a
support) starts step 1 there, so that the grid meets a narrow support.
"""
import math

import numpy as np

from . import legendre as LG
from .legendre import EPS, TINY, analysis
from ...quadrature import gauss
from .spherical import analyse, angular_degree, grid

ANGULAR = (7, 15, 23, 31, 47, 63, 95, 127)
RADIAL = (9, 17, 33, 65, 129)
MAX_PIECES = 128


def _sample(fun, rr, L, N):
    """Coefficients (len(rr), (L + 1)^2, N) on grid(L) at radii rr, and the largest |value| per function."""
    theta, _, phi, _ = grid(L)
    V = np.asarray(fun(rr[:, None, None], theta[None, :, None], phi[None, None, :]), float)
    V = np.broadcast_to(V, (len(rr), len(theta), len(phi)) + ((N,) if V.ndim == 4 or N > 1 else ()))
    V = V.reshape(len(rr), len(theta), len(phi), N)
    if not np.all(np.isfinite(V)):
        raise ValueError("the function is not finite on the sampling grid of the ball or shell")
    vscale = np.abs(V).reshape(-1, N).max(axis=0)
    C = analyse(np.moveaxis(V, 0, 2).reshape(len(theta), len(phi), -1), L)       # ((L+1)^2, nr * N)
    return np.moveaxis(C.reshape(C.shape[0], len(rr), N), 1, 0), vscale


def _degree_energy(C, L):
    """Energy per degree, summed over radii: (L + 1, N)."""
    E = (C * C).sum(axis=0)                                                   # ((L+1)^2, N)
    return np.add.reduceat(E, np.arange(L + 1) ** 2, axis=0)


def _choose_degree(fun, probes, N, tol, lmax, lmin=0):
    last = None
    for L in [x for x in ANGULAR if lmin <= x <= lmax] or [lmax]:
        C, vscale = _sample(fun, probes, L, N)
        E = _degree_energy(C, L)
        norm = np.sqrt(E.sum(axis=0))
        thr = np.maximum(tol * norm, 2 * EPS * (L + 1) * vscale * math.sqrt(len(probes))) + TINY
        top = max(2, (L + 1) // 8)
        last = (L, E, thr)
        if np.all(np.sqrt(E[L + 1 - top:].sum(axis=0)) <= thr):
            break
    L, E, thr = last
    tail = np.sqrt(np.vstack([np.cumsum(E[::-1], axis=0)[::-1][1:], np.zeros((1, N))]))   # energy beyond l
    keep = int(np.argmax(np.all(tail <= thr[None], axis=1)))
    unresolved = int(np.sum(np.sqrt(E[L + 1 - max(2, (L + 1) // 8):].sum(axis=0)) > thr))
    return L, keep, unresolved, vscale


def fit_rh(fun, a, b, N=1, tol=1e-14, breaks=None, lmax=127, lmin=None):
    """(RadialHarmonicSeries, unresolved) for fun(r, theta, phi) -> (...) or (..., N) on a < r < b. lmin: the least angular degree
    (a support's hint, NaturalBreaks.degree; refused above lmax). At the degree cap with energy left in the top
    degrees the fit is unresolved and that energy is part of the error estimate (no claim of convergence)."""
    from .radial_harmonic import RadialHarmonicSeries
    if lmin is not None and lmin > lmax:
        from ...core.errors import RealisationError
        raise RealisationError(f"the function's support needs angular degree >= {lmin} on its sampling grid, beyond "
                               f"the cap {lmax} of the radial x harmonic realisation: use realisation='tensor' (natural "
                               "breaks at the support), or keep the measure symbolic")
    init = np.unique(np.concatenate([[a, b], np.clip(np.asarray([] if breaks is None else breaks, float), a, b)]))
    t5, _ = gauss(5)
    probes = np.concatenate([0.5 * (x0 + x1) + 0.5 * (x1 - x0) * t5 for x0, x1 in zip(init[:-1], init[1:])])
    Lg, L, unresolved, vscale = _choose_degree(fun, probes, N, tol, lmax, lmin or 0)
    nlm = (L + 1) ** 2
    top0 = max(nlm, (Lg + 1 - max(2, (Lg + 1) // 8)) ** 2) if unresolved else None    # capped: the top degrees count
    todo = [(init[i], init[i + 1]) for i in range(len(init) - 1)][::-1]
    done, e2 = [], np.zeros(N)
    while todo:
        x0, x1 = todo.pop()
        h = 0.5 * (x1 - x0)
        ok = False
        for n in RADIAL:
            t, w = gauss(n)
            rr = x0 + h * (t + 1.0)
            Cf, vs = _sample(fun, rr, Lg, N)
            np.maximum(vscale, vs, out=vscale)
            ang = (w[:, None] * h * (rr * rr)[:, None] * (Cf[:, nlm:] ** 2).sum(axis=1)).sum(axis=0)
            if top0 is not None:
                ang = ang + (w[:, None] * h * (rr * rr)[:, None] * (Cf[:, top0:] ** 2).sum(axis=1)).sum(axis=0)
            C = np.tensordot(analysis(n), Cf[:, :nlm], axes=(1, 0))                # (n, nlm, N)
            thr = max(tol, 2 * EPS * n ** 1.5) * np.maximum(vscale, TINY)
            k = max(2, n // 8)
            tail = np.abs(C[-k:]).max(axis=(0, 1))
            if np.all(tail <= thr):
                big = (np.abs(C) > thr[None, None, :]).any(axis=(1, 2))
                rows = np.nonzero(big)[0]
                cut = rows[-1] + 1 if len(rows) else 1
                hk = (x1 - x0) / (2.0 * np.arange(cut, n) + 1.0)
                e2 += b * b * (hk[:, None] * (C[cut:] ** 2).sum(axis=1)).sum(axis=0) + ang
                e2 += (EPS * vscale) ** 2 * (x1 ** 3 - x0 ** 3) / 3.0 * 4 * math.pi
                done.append((x0, x1, C[:cut]))
                ok = True
                break
        if ok:
            continue
        if len(done) + len(todo) + 2 > MAX_PIECES or x1 - x0 <= 64 * EPS * max(abs(b), 1.0):
            unresolved += 1
            done.append((x0, x1, C))
            e2 += b * b * (h * 2 / 3.0) * (C[-k:] ** 2).sum(axis=(0, 1)) * 4 + ang
            continue
        mid = 0.5 * (x0 + x1)
        todo.append((mid, x1))
        todo.append((x0, mid))
    done.sort(key=lambda z: z[0])
    return RadialHarmonicSeries(L, [z[0] for z in done] + [done[-1][1]], [z[2] for z in done], np.sqrt(e2)), unresolved


def radial_fit(fun, a, b, N=1, tol=1e-14, breaks=None):
    """A legendre.PiecewiseLegendre of N radial functions (fun(r) -> (n,) or (n, N)), with the interval module's adaptive fit."""
    return LG.fit(fun, a, b, N=N, breaks=breaks, tol=tol)


def support_error(B, fun, R, box, n=24):
    """||f - f_R|| per column for f supported in the natural box ((r0, r1), (t0, t1), (p0, p1)) of the ball or shell
    (p0 > p1: the longitudes wrap): int_box (f - f_R)^2 by Gauss quadrature there, plus the energy of f_R outside the
    box (its exact norm minus the box's part). An a posteriori check of a spectral fit whose degree test the grid can
    fool (a narrow support aliased on a coarse grid, plan 035 P2); its floor is the quadrature's, about 1e-5 ||f||."""
    (r0, r1), (t0, t1), (p0, p1) = box
    t, w = gauss(n)
    panels = [(p0, p1)] if p0 < p1 else [(p0, 2 * math.pi), (0.0, p1)]
    r, wr = 0.5 * (r0 + r1) + 0.5 * (r1 - r0) * t, 0.5 * (r1 - r0) * w
    th, wt = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * t, 0.5 * (t1 - t0) * w
    ph = np.concatenate([0.5 * (a + b) + 0.5 * (b - a) * t for a, b in panels])
    wp = np.concatenate([0.5 * (b - a) * w for a, b in panels])
    G = np.meshgrid(r, th, ph, indexing="ij")
    W = np.einsum("i,j,k->ijk", wr * r * r, wt * np.sin(th), wp).ravel()
    X = [np.ravel(c) for c in B.domain.to_cartesian(*G)]
    f = np.asarray(fun(*X), float).reshape(len(W), -1)
    g = np.asarray(B.evaluate(R, *X), float).reshape(len(W), -1)
    inside = W @ (f - g) ** 2
    from .radial_harmonic import sqnorms                 # the r^2 dr norm, as W: not the space's (a density, plan 064)
    outside = np.maximum(np.asarray(sqnorms(R), float) - W @ (g * g), 0.0)
    return np.sqrt(inside + outside)


CHECK_WARN, CHECK_REFUSE = 1e-6, 0.05      # support_error relative to the norm: unresolved above, refused above


def support_check(B, fun, R, unresolved, box):
    """(R, unresolved) after support_error: a larger error than the fit's own estimate replaces it and counts as
    unresolved (a warning); a gross one (CHECK_REFUSE of the norm) is refused (RealisationError: the grid missed the
    support's detail)."""
    from ...core.errors import RealisationError
    from .radial_harmonic import RadialHarmonicSeries
    from .radial_harmonic import sqnorms
    e = support_error(B, fun, R, box)
    nrm = np.sqrt(np.maximum(np.asarray(sqnorms(R), float), 0.0))          # the same norm as e (r^2 dr)
    if np.any(e > CHECK_REFUSE * nrm):
        j = int(np.argmax(e / np.maximum(nrm, TINY)))
        raise RealisationError(f"the radial x harmonic fit (degree {R.L}) is off by {e[j] / max(nrm[j], TINY):.1g} of "
                               "the norm on the function's support (a posteriori check): the angular grid does not "
                               "resolve it below the degree cap; use realisation='tensor' (natural breaks at the "
                               "support), or keep the measure symbolic")
    worse = e > np.maximum(R.err, CHECK_WARN * nrm)
    if not np.any(worse):
        return R, unresolved
    return RadialHarmonicSeries(R.L, R.breaks, R.coefs, np.where(worse, e, R.err)), unresolved + int(np.sum(worse))


def resolution_hint(self, breaks, w):
    """Breaks for a fit at a declared resolution w (core/fits.py): the least angular degree whose grid has spacing
    <= w / 2 at the outer radius (refused by fit_rh above its cap); the radial breaks as given."""
    from ..backends import NaturalBreaks
    L = max(angular_degree(self.b, w, ANGULAR), getattr(breaks, "degree", None) or 0)
    axes = tuple(breaks) if isinstance(breaks, tuple) and len(breaks) == 3 else (breaks, None, None)
    return NaturalBreaks(axes, degree=L, box=getattr(breaks, "box", None))
