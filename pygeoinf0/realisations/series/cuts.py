"""Tensor realisations cut to a smaller domain (plan 043).

From joined.py (merged here by plan 063):
Realisations of pieces joined into one, with no fit (plan 043 P2): Backend.glued, the mirror of Backend.restricted.
"""


import numpy as np

from . import legendre as L
from .legendre import EPS
from .tensor import TensorSeries, AxisBasis, FourierBasis


def same_chart(A, B, samples=16):
    """Whether B's natural coordinates are A's: to_cartesian agrees at points of B's reference box, which lies inside A's."""
    if len(A.reference) != len(B.reference):
        return False
    if any(lo < a - 1e-12 * max(1.0, abs(a)) or hi > b + 1e-12 * max(1.0, abs(b))
           for (lo, hi), (a, b) in zip(B.reference, A.reference)):
        return False
    u = np.random.default_rng(0).random((samples, len(B.reference)))
    X = [lo + u[:, k] * (hi - lo) for k, (lo, hi) in enumerate(B.reference)]
    return all(np.allclose(p, q, rtol=1e-13, atol=1e-13) for p, q in zip(A.to_cartesian(*X), B.to_cartesian(*X)))


def restricted(backend, R, target):
    """A domain with the same natural coordinates on a smaller reference box: each piecewise axis cut, and a Fourier
    axis only if the target keeps its whole period. None otherwise (another chart, a part of a period, patches)."""
    if not isinstance(R, TensorSeries) or target.dim != backend.dim or not same_chart(backend.domain, target.domain):
        return None
    bases, C = list(R.bases), R.C
    for k, ((lo, hi), b) in enumerate(zip(target.domain.reference, R.bases)):
        if isinstance(b, FourierBasis):
            if abs(lo - b.lo) > 1e-12 or abs(hi - b.hi) > 1e-12:
                return None
            continue
        Ck = np.moveaxis(C, k, 0)
        rest = Ck.shape[1:]
        blocks = Ck.reshape((b.P, b.n) + rest)
        out = L.cut(list(blocks), np.asarray(b.breaks), lo, hi)
        if out is None:
            return None
        bases[k] = AxisBasis(out[0], b.n)
        C = np.moveaxis(np.concatenate(out[1], axis=0), 0, k)
    return TensorSeries(bases, C, R.err)


def glued(backend, parts):
    """Pieces that differ from the domain along one axis only, with the same bases on the others: their pieces along that axis
    in order, padded to the largest number of points, zero elsewhere. None otherwise."""
    if not all(isinstance(R, TensorSeries) for _, R in parts) or len({R.N for _, R in parts}) != 1:
        return None
    ref = backend.domain.reference
    if not all(len(B.domain.reference) == len(ref) and same_chart(backend.domain, B.domain) for B, _ in parts):
        return None
    axes = {k for B, _ in parts for k, (s, r) in enumerate(zip(B.domain.reference, ref)) if abs(s[0] - r[0]) > 1e-12 or abs(s[1] - r[1]) > 1e-12}
    if len(axes) > 1:
        return None
    k = axes.pop() if axes else 0
    first = parts[0][1].bases
    if any(isinstance(R.bases[k], FourierBasis) or any(b != f for j, (b, f) in enumerate(zip(R.bases, first)) if j != k) for _, R in parts):
        return None
    n = max(R.bases[k].n for _, R in parts)
    rest = parts[0][1].C.shape[:k] + parts[0][1].C.shape[k + 1:]

    def blocks(R):
        b = R.bases[k]
        Ck = np.moveaxis(R.C, k, 0).reshape((b.P, b.n) + rest)
        return [np.concatenate([c, np.zeros((n - b.n,) + rest)], axis=0) for c in Ck]
    from .cuts import joined as join
    out = join(ref[k][0], ref[k][1], [(B.domain.reference[k][0], B.domain.reference[k][1], R.bases[k].breaks, blocks(R)) for B, R in parts],
                   lambda: np.zeros((n,) + rest))
    if out is None:
        return None
    C = np.moveaxis(np.concatenate(out[1], axis=0), 0, k)
    bases = list(first)
    bases[k] = AxisBasis(out[0], n)
    return TensorSeries(bases, C, np.sqrt(sum(R.err ** 2 for _, R in parts)))


# ---------------------------------------------------------------- from joined.py (plan 063)
def joined(a, b, parts, zero):
    """(breaks, coefficient arrays) on [a, b] from parts [(a_k, b_k, breaks_k, coefs_k)] that do not overlap: their pieces in
    order, zero(1) on what no part covers. None when two parts overlap or a part leaves [a, b]."""
    tol = 8 * EPS * max(abs(a), abs(b), 1.0)
    br, out = [a], []
    for a_k, b_k, breaks, coefs in sorted(parts, key=lambda p: p[0]):
        if a_k < br[-1] - tol or b_k > b + tol:
            return None
        if a_k > br[-1] + tol:
            br.append(a_k)
            out.append(zero())
        br.extend(np.asarray(breaks, float)[1:-1].tolist() + [b_k])
        out.extend(coefs)
    if br[-1] < b - tol:
        br.append(b)
        out.append(zero())
    br[-1] = b
    return np.asarray(br), out


def radial_harmonic(backend, parts):
    """Concentric balls and shells: their radial pieces in order, harmonics padded to the largest L."""
    from .radial_harmonic import RadialHarmonicSeries, dense
    if any(tuple(B.domain.centre) != tuple(backend.domain.centre) for B, _ in parts):
        return None
    parts = [(B, dense(R)) for B, R in parts]
    if len({R.N for _, R in parts}) != 1:
        return None
    L, N = max(R.L for _, R in parts), parts[0][1].N
    nlm = (L + 1) ** 2

    def padded(c):
        return np.concatenate([c, np.zeros((c.shape[0], nlm - c.shape[1], N))], axis=1)
    out = joined(backend.a, backend.b, [(B.a, B.b, R.breaks, [padded(c) for c in R.coefs]) for B, R in parts],
                 lambda: np.zeros((1, nlm, N)))
    return None if out is None else RadialHarmonicSeries(L, out[0], out[1], np.sqrt(sum(R.err ** 2 for _, R in parts)))
