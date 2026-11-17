"""The solver interface's translation of convex sets into the forms the dual solvers of optimisers/dual/ take (plan 070
P4; amendment 1: convex/ holds convex analysis, optimisers/dual/ the numerics). Only these functions read the classes of
the noise sets; the solvers see arrays.

    noise_form(N, D)    (kind, data) of a noise set: ball (whitening Phi, centre, rho, nuisance columns F), exact
                        ({0} or a span, with F), slab (normal, half-width, centre) or general (the set itself)
    span_matrix(S)      a basis of a span as matrix columns
    slab_rows(N, D)     (A, lo, hi) of slab-type noise {e : lo <= A e <= hi} (slabs, their intersections, polytopes of
                        paired half-spaces; translates shift lo and hi), or None; refused with the reason for a zero
                        normal, a negative or infinite width, a half-space without its opposite
    noise_side(N, D)    the box BGP's NoiseSide (optimisers/dual/nonsmooth.py) of a ball, an ellipsoid or a
                        coordinate box of noise (without nuisances)
    noise_metric(N, D)  the data-space engine's metric of a ball or an ellipsoid of noise (solvers/dataspace/,
                        plan 070 P3): its shape stated by the set's owner, not inferred by the engine
"""
import math

import numpy as np

from ..core.errors import Unsupported
from ..optimisers.dual import NoiseSide


def noise_form(N, D):
    """(kind, data) of a noise set for the solvers: kind in ball, exact, slab, general (optimisers/dual/reduction.py's module docstring)."""
    from .subspaces import Span, Zero
    from .algebra import Minkowski, Translate
    from .sets import Ellipsoid, NormBall, Slab
    centre, F, base = np.zeros(D.dim), np.zeros((D.dim, 0)), N
    while True:
        if isinstance(base, Translate):
            centre, base = centre + base.shift.values, base.base
        elif isinstance(base, Minkowski) and isinstance(base.parts[0], (Span, Zero)):
            F, base = np.c_[F, span_matrix(base.parts[0])], base.parts[1]
        elif isinstance(base, Minkowski) and isinstance(base.parts[1], (Span, Zero)):
            F, base = np.c_[F, span_matrix(base.parts[1])], base.parts[0]
        else:
            break
    if isinstance(base, (Span, Zero)):
        return "exact", dict(F=np.c_[F, span_matrix(base)], centre=centre, Phi=np.eye(D.dim), rho=0.0)
    if isinstance(base, NormBall):
        c = centre + (0.0 if base.centre is None else base.centre.values)
        return "ball", dict(F=F, centre=c, Phi=D.to_orthonormal(np.eye(D.dim)), rho=base.radius)
    if isinstance(base, Ellipsoid):
        return "ball", dict(F=F, centre=centre + base.centre_values, Phi=np.linalg.inv(base.factor), rho=base.radius)
    if isinstance(base, Slab) and F.shape[1] == 0:
        c = centre + (0.0 if base.centre is None else base.centre.values)
        return "slab", dict(normal=base.normal.values, w=base.half_width, centre=c)
    return "general", dict(set=N)


def span_matrix(S):
    """A basis of the span S as the columns of a matrix (none for {0})."""
    from .subspaces import Zero
    return np.zeros((S.space.dim, 0)) if isinstance(S, Zero) else S.matrix_basis()


def _rows(S, D):
    """(A (k, n), lo, hi) of a slab-type set S of D, or None if S is not one."""
    from .algebra import Intersection, Translate
    from .sets import Polytope, Slab
    if isinstance(S, Translate):
        out = _rows(S.base, D)
        if out is None:
            return None
        A, lo, hi = out
        s = A @ np.asarray(S.shift.values, float)
        return A, lo + s, hi + s
    if isinstance(S, Slab):
        row = np.asarray(D.apply_metric(S.normal.values), float)
        w = S.half_width
        if not np.any(row):
            raise Unsupported("a slab of noise with a zero normal is the whole space or empty: give a nonzero normal")
        if not (math.isfinite(w) and w >= 0):
            raise Unsupported(f"a slab of noise needs a finite half-width >= 0 (got {w}): a negative one is empty")
        c = 0.0 if S.centre is None else float(row @ np.asarray(S.centre.values, float))
        return row[None, :], np.array([c - w]), np.array([c + w])
    if isinstance(S, Polytope):
        P, b = S.A, S.b
        nrm = np.linalg.norm(P, axis=1)
        if np.any(nrm == 0):
            raise Unsupported("a polytope of noise with a zero row: drop it (0 <= b) or it is empty (b < 0)")
        U = P / nrm[:, None]
        used, rows, lo, hi = set(), [], [], []
        for i in range(len(b)):
            if i in used:
                continue
            j = next((j for j in range(i + 1, len(b)) if j not in used and np.allclose(U[j], -U[i], atol=1e-12)), None)
            if j is None:
                raise Unsupported("box BGP takes a polytope of noise only with its half-spaces in opposite pairs "
                                  f"(slabs); half-space {i} has no opposite, and one-sided constraints are not built")
            used |= {i, j}
            rows.append(P[i])
            lo.append(-b[j] * nrm[i] / nrm[j])
            hi.append(b[i])
        return np.array(rows), np.array(lo), np.array(hi)
    if isinstance(S, Intersection):
        parts = [_rows(T, D) for T in S.parts]
        if any(p is None for p in parts):
            return None
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
    return None


def slab_rows(N, D):
    """(A, lo, hi) of a slab-type noise set (module docstring), or None for other sets."""
    return _rows(N, D)


def noise_side(N, D):
    """The noise set in the cell dual's terms (optimisers/dual/nonsmooth.py NoiseSide): a coordinate box (lo, hi), or a
    ball ||Phi (eta - centre)|| <= rho (Phi diagonal kept as its diagonal)."""
    from .algebra import Translate
    from .sets import CoordinateBox
    centre, base = np.zeros(D.dim), N
    while isinstance(base, Translate):
        centre, base = centre + base.shift.values, base.base
    if isinstance(base, CoordinateBox):
        return NoiseSide.box(base.lo + centre, base.hi + centre)
    kind, form = noise_form(N, D)
    if kind != "ball" or form["F"].shape[1]:
        raise Unsupported("box BGP takes a ball, an ellipsoid, a coordinate box, slabs or a polytope of paired "
                          "half-spaces of noise (no nuisances)")
    Phi = form["Phi"]
    return NoiseSide.ball(np.diag(Phi).copy() if np.count_nonzero(Phi - np.diag(np.diag(Phi))) == 0 else Phi,
                          form["centre"], form["rho"])


def noise_metric(N, D, centre=None):
    """The data-space engine's metric (solvers.dataspace.NoiseMetric) of a noise set N of D that is a NormBall or an
    Ellipsoid, translated or not (plan 070 P3, decision Q1): a ball's shape is D's own inner product, an ellipsoid's its
    shape matrix S (the metric S^-1); the offset its centre (plus centre)."""
    from ..core.errors import SpaceMismatch
    from ..solvers.dataspace import NoiseMetric
    from .algebra import Translate
    from .sets import Ellipsoid, NormBall
    while isinstance(N, Translate):
        centre = (0.0 if centre is None else centre) + N.shift.values
        N = N.base
    if isinstance(N, NormBall):
        if N.space != D:
            raise SpaceMismatch(f"the noise ball is a subset of {N.space}, not of {D}")
        c = None if N.centre is None else N.centre.values
        off = c if centre is None else (centre if c is None else centre + c)
        return NoiseMetric("standard", None, D, off, "the shape of the noise ball (the metric of the data space)")
    if isinstance(N, Ellipsoid):
        S = N.shape
        off = N.centre_values if centre is None else centre + N.centre_values
        if not np.any(S - np.diag(np.diag(S))):
            return NoiseMetric("diag", np.diag(S).copy(), D, off, "the shape of the noise ellipsoid")
        return NoiseMetric("dense", S, D, off, "the shape of the noise ellipsoid")
    raise Unsupported(f"the metric of a noise set that is a ball or an ellipsoid of {D}, not of {type(N).__name__}")
