"""The harmonic coefficients of a density kept symbolic on a ball or shell (plan 044; plan 070 P1: from
operators/harmonic_targets.py, the numerics under the property families of properties/harmonic_targets.py).

A target t cut to the pieces of a ball (a measure restricted to shells, measures/restrict.py) pairs exactly with formulas and with
other such targets, but it has no realisation: a fit would cross the pieces' boundaries. Against separable functionals
k (x) Y_lj it needs none: by orthonormality only its coefficients at those (l, j) enter,

    t_lj(r) = int_(S^2) t(r xi) Y_lj(xi) d xi        on each piece, by Gauss quadrature over the support's natural box.

coefficients(part, L)          (t_lj(r) as a PiecewiseLegendre, the quadrature's estimated relative error)
realised_moment(space, m, R)   int g dm for g radial x harmonic, on the coefficients (spaces/representers.moments)
support_box(measure, C, a, b)  the box in (r, theta, phi) about C holding a measure's support (also realisations/quadrature.py)
is_symbolic(T), members(T)     whether a functional's kernels have parts kept symbolic, and those kernels (plan 070 P3)

The expansion is exact on fields of degree <= L, up to the quadrature (stated).
"""
import math

import numpy as np

from ...quadrature import gauss
from ...spaces import harmonic as HA
from . import legendre as LG

NODES = 96              # angular Gauss nodes per axis over a support's box, at least 2 L + 32
FULL = (160, 320)       # (theta, phi) nodes over the whole sphere, when no box is known
RADIAL = 33             # Gauss nodes per radial panel
PANELS = 16             # radial panels across a support's diameter, or a part with no box


def _angular(L, box, scale=1):
    """(rows of Y_lj weighted by the quadrature, the directions as sines and cosines): Gauss in theta and phi over box."""
    (t0, t1), (p0, p1) = box if box is not None else ((0.0, math.pi), (0.0, 2 * math.pi))
    if p1 < p0:                                                   # longitudes that wrap
        p1 = p1 + 2 * math.pi
    nt, np_ = ((max(NODES, 2 * L + 32),) * 2) if box is not None else FULL
    nt, np_ = nt * scale, np_ * scale
    x, w = gauss(nt)
    theta, wt = t0 + 0.5 * (t1 - t0) * (x + 1.0), 0.5 * (t1 - t0) * w
    x, w = gauss(np_)
    phi, wp = p0 + 0.5 * (p1 - p0) * (x + 1.0), 0.5 * (p1 - p0) * w
    T, P = np.meshgrid(theta, phi, indexing="ij")
    W = (wt[:, None] * wp[None, :] * np.sin(T)).ravel()
    Y = HA.values(L, None, theta_phi=(T.ravel(), P.ravel()))
    return W[:, None] * Y, np.sin(T).ravel() * np.cos(P).ravel(), np.sin(T).ravel() * np.sin(P).ravel(), np.cos(T).ravel()


def _balls(measure):
    """The balls (c, rho) holding a measure's support, from the sets of its terms; None when a term has no such set."""
    terms = getattr(measure, "terms", None)
    if terms is not None:
        out = [_balls(m) for _, m in terms]
        return None if any(x is None for x in out) else [b for x in out for b in x]
    S = getattr(measure, "set", None)
    ball = getattr(S, "ball", None)
    if ball is not None:
        return [ball]
    if type(S).__name__ in ("Ball", "Disk"):
        return [(S.centre, S.radius)]
    return None


def support_box(measure, C, a, b):
    """((r0, r1), (theta0, theta1), (phi0, phi1)) about the centre C holding the measure's support inside a <= r <= b, or None."""
    balls = _balls(measure)
    if not balls or len(balls) > 1:
        return None
    c, rho = balls[0]
    v = np.asarray(c, float) - np.asarray(C, float)
    d = float(np.linalg.norm(v))
    radial = (max(a, d - rho), min(b, d + rho))
    if radial[1] <= radial[0]:
        return None
    if d <= rho:
        return radial, (0.0, math.pi), (0.0, 2 * math.pi)
    delta, theta = math.asin(rho / d), math.acos(v[2] / d)
    t0, t1 = theta - delta, theta + delta
    if t0 <= 0.0 or t1 >= math.pi:                                # the cap holds a pole
        return radial, (max(t0, 0.0), min(t1, math.pi)), (0.0, 2 * math.pi)
    half, phi = math.asin(min(1.0, math.sin(delta) / math.sin(theta))), math.atan2(v[1], v[0])
    return radial, (t0, t1), (phi - half, phi + half)


def balls_radius(measure):
    return float(_balls(measure)[0][1])


def _whole_scale(measure, angular):
    """The size of a coefficient of the whole target, not of this piece of it, where a bump's edge holds almost nothing: its
    weight at the support's centre over the angular box, times Y_00. 0 when unknown (the piece's own size then)."""
    balls, wt = _balls(measure), getattr(measure, "weight", None)
    terms = getattr(measure, "terms", None)
    if terms is not None and len(terms) == 1:
        c, m = terms[0]
        return abs(c) * _whole_scale(m, angular)
    if not balls or wt is None or angular is None:
        return 0.0
    (t0, t1), (p0, p1) = angular
    area = abs((p1 - p0) * (math.cos(t0) - math.cos(t1)))
    return abs(float(np.ravel(wt.values(np.asarray(balls[0][0], float)[None]))[0])) * area / math.sqrt(4 * math.pi)


def coefficients(part, L):
    """(a legendre.PiecewiseLegendre of (L + 1)^2 radial functions t_lj(r), an estimate of the angular quadrature's relative error) of the
    density t of a symbolic part on a ball or shell, on the whole radial range of the part's space."""
    D = part.space.domain
    a0, b0 = (float(x) for x in part.space.backend.domain.reference[0])
    a, b = a0, b0
    c = np.asarray(D.centre, float)
    box = support_box(part.measure, c, a, b)
    if box is not None:
        a, b = box[0]
    angular = box[1:] if box is not None else None
    A, sx, sy, sz = _angular(L, angular)
    N = A.shape[1]

    def fun(r):
        r = np.asarray(r, float)
        v = np.asarray(part(c[0] + r[:, None] * sx[None], c[1] + r[:, None] * sy[None], c[2] + r[:, None] * sz[None]), float)
        return v.reshape(len(r), -1) @ A
    # fixed panels: an adaptive fit would chase the tolerance on each coefficient, however small against the largest
    panels = PANELS if box is None else max(4, int(math.ceil(PANELS * (b - a) / max(2.0 * balls_radius(part.measure), b - a))))
    edges = np.linspace(a, b, panels + 1)
    t, _ = gauss(RADIAL)
    An = LG.analysis(RADIAL)
    coefs = [An @ fun(x0 + 0.5 * (x1 - x0) * (t + 1.0)) for x0, x1 in zip(edges[:-1], edges[1:])]
    big = max(float(np.abs(C).max()) for C in coefs)
    tail = max(float(np.abs(C[-4:]).max()) for C in coefs)
    A2, sx2, sy2, sz2 = _angular(L, angular, scale=2)
    r = np.array([a + f * (b - a) for f in (0.3, 0.5, 0.7)])
    v2 = np.asarray(part(c[0] + r[:, None] * sx2[None], c[1] + r[:, None] * sy2[None], c[2] + r[:, None] * sz2[None]), float)
    fine, coarse = v2.reshape(3, -1) @ A2, fun(r)
    err = max(tail, float(np.abs(fine - coarse).max())) / max(big, _whole_scale(part.measure, angular))
    breaks = edges.tolist()
    if a > a0:
        breaks, coefs = [a0] + breaks, [np.zeros((1, N))] + coefs
    if b < b0:
        breaks, coefs = breaks + [b0], coefs + [np.zeros((1, N))]
    return LG.PiecewiseLegendre(breaks, coefs, np.zeros(N)), err


def realised_moment(space, m, R):
    """int g dm (N,) for g the realisation R on a ball or shell, on the coefficients: exact for R and the term's t_lj, up to the
    quadrature of t_lj, where R is radial x harmonic and the support's box is known; None otherwise (moments' Gauss rule).
    Kept on the backend per term and degree: the two orders of the pairing's estimate ask twice."""
    from ...spaces.representers import DensityRepresenter
    from .radial_harmonic import RadialHarmonicSeries
    from ..registry import provides
    B = space.backend
    if not provides(B, "radial_harmonic_coefficients") or not provides(R, "radial_harmonic_coefficients"):
        return None
    if support_box(m, B.domain.centre, B.a, B.b) is None:
        return None
    cache = B.__dict__.setdefault("_target_moments", {})
    key = (id(m), R.L)
    if key not in cache:
        pl, err = coefficients(DensityRepresenter(space, m), R.L)
        cache[key] = (m, RadialHarmonicSeries(R.L, pl.breaks, [C[:, :, None] for C in pl.coefs], [err]))
        if err > 1e-8:
            import warnings
            from ...core.errors import RealisationWarning
            warnings.warn(f"the harmonic coefficients of {m} carry an estimated relative error {err:.1e}", RealisationWarning, stacklevel=4)
    return B.gram(cache[key][1], R)[0]          # <t, g> in the space's measure (c rho r^2 dr: plan 064) = int g dm


def members(T):
    """The kernels of a functional or of the rows of a pg.Functionals (None: neither)."""
    kernel = getattr(T, "representer", None)
    if kernel is not None:
        return [kernel]
    kernels = getattr(T, "kernels", None)
    if kernels is None:
        return None
    return [kernels] if not hasattr(kernels, "__len__") else [kernels[i] for i in range(len(kernels))]


def is_symbolic(T):
    """Whether T's kernels are elements of a direct sum with parts kept symbolic (the capability "symbolic_measures" of
    their realisations; plan 070 P3: from properties/harmonic_targets.py, so that the data-space engine below
    properties/ can refuse such a target and name pg.harmonic_target)."""
    from ..registry import provides
    rows = members(T)
    if not rows or getattr(rows[0].space, "names", None) is None:
        return False
    try:
        return all(any(provides(p[0] if isinstance(p, tuple) else p, "symbolic_measures") for p in m.space._pl_of(m).parts
                       if p is not None)
                   for m in rows)
    except (AttributeError, TypeError):
        return False
