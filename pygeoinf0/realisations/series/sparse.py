"""Sparse realisation of many localised functions (narrow data kernels, say) on a common partition.

Each piece of the partition stores only the columns (functions) that are not negligible there, so n
localised kernels cost O(n) storage instead of O(n^2), and the Gram matrix is a loop of small blocks.
Applying such a family to a smooth family F (few pieces, moderate degree) goes through the projections
<k_i, P_k> onto F's Legendre basis, computed once per partition of F and kept.

This module also holds the dispatch used by the spaces: gram, combine, colsum, sqnorms, integral, take
accept either realisation (dense PiecewiseLegendre or sparse SparsePiecewiseLegendre).
"""
import numpy as np
from numpy.polynomial import legendre as npleg

from . import legendre as L
from .legendre import EPS, TINY, analysis, polyval
from ...quadrature import gauss


class SparsePiecewiseLegendre:
    """N functions on a common partition; piece p stores columns cols[p] with coefficients coefs[p]."""

    __slots__ = ("breaks", "cols", "coefs", "N", "err", "_proj")

    def __init__(self, breaks, cols, coefs, N, err):
        self.breaks = L._ro(np.asarray(breaks, float))
        self.cols = tuple(L._ro(np.asarray(c, np.intp)) for c in cols)
        self.coefs = tuple(L._ro(np.asarray(c, float)) for c in coefs)
        self.N = int(N)
        self.err = L._ro(np.asarray(err, float).copy())
        self._proj = {}

    @property
    def npieces(self):
        return len(self.coefs)

    @property
    def degrees(self):
        return [c.shape[0] - 1 for c in self.coefs]

    @property
    def size(self):
        return sum(c.size for c in self.coefs)

    def maxabs(self):
        return max((float(np.abs(c).max()) for c in self.coefs if c.size), default=0.0)

    def take(self, idx):
        """Dense realisation of the columns idx (runs of pieces where they all vanish are merged)."""
        idx = np.atleast_1d(np.asarray(idx, np.intp))
        out_b, out_c = [self.breaks[0]], []
        zero_run = False
        for p in range(self.npieces):
            cp = self.cols[p]
            pos = np.searchsorted(cp, idx)
            hit = (pos < len(cp)) & (cp[np.minimum(pos, max(len(cp) - 1, 0))] == idx) if len(cp) else np.zeros(len(idx), bool)
            if not hit.any():
                if zero_run:
                    out_b[-1] = self.breaks[p + 1]
                    continue
                out_c.append(np.zeros((1, len(idx))))
                out_b.append(self.breaks[p + 1])
                zero_run = True
                continue
            C = np.zeros((self.coefs[p].shape[0], len(idx)))
            C[:, hit] = self.coefs[p][:, pos[hit]]
            out_c.append(C)
            out_b.append(self.breaks[p + 1])
            zero_run = False
        return L.PiecewiseLegendre(out_b, out_c, self.err[idx])


# ------------------------------------------------------------------------------------------ construction

def _sample(fun_idx, x, idx):
    v = np.asarray(fun_idx(x, idx), float)
    return np.broadcast_to(v, (len(x), len(idx))) if v.ndim < 2 else v


def fit_sparse(fun_idx, a, b, N, tol=1e-14, scout=4097, initial=16, nseq=(17, 33, 65), max_pieces=100000):
    """Adaptive fit of N functions fun_idx(x, idx) -> (len(x), len(idx)) on a common partition, storing on
    each piece only the columns that a uniform scout sample shows to be non-negligible near it (one scout
    spacing of margin). A function that is negligible at every scout point near a piece is taken as zero
    there, so features narrower than the scout spacing (b - a) / (scout - 1) can be missed."""
    sx = np.linspace(a, b, scout)
    vscale = np.zeros(N)
    cum = np.zeros((scout + 1, N), np.int32)
    chunk = max(1, (1 << 22) // scout)
    for s in range(0, N, chunk):
        idx = np.arange(s, min(N, s + chunk))
        v = np.abs(_sample(fun_idx, sx, idx))
        v[~np.isfinite(v)] = np.inf
        vs = np.where(np.isfinite(v), v, 0).max(axis=0)
        vscale[idx] = vs
        cum[1:, idx] = np.cumsum(v > tol * np.maximum(vs, TINY), axis=0)
    minwidth = 64 * EPS * max(b - a, abs(a), abs(b))
    edges = np.linspace(a, b, initial + 1)
    todo = [(edges[i], edges[i + 1]) for i in range(initial)][::-1]
    out, err2, unresolved = [], np.zeros(N), 0
    while todo:
        x0, x1 = todo.pop()
        r0 = max(0, int(np.searchsorted(sx, x0, "left")) - 1)
        r1 = min(scout, int(np.searchsorted(sx, x1, "right")) + 1)
        idx = np.nonzero(cum[r1] - cum[r0] > 0)[0]
        if len(idx) == 0:
            out.append((x0, x1, idx, np.zeros((1, 0))))
            continue
        ok = False
        for n in nseq:
            t, _ = gauss(n)
            v = _sample(fun_idx, x0 + 0.5 * (x1 - x0) * (t + 1.0), idx)
            vscale[idx] = np.maximum(vscale[idx], np.abs(v).max(axis=0))
            C = analysis(n) @ v
            thr = max(tol, 2 * EPS * n ** 1.5) * np.maximum(vscale[idx], TINY)
            if np.all(np.abs(C[-max(4, n // 8):]).max(axis=0) <= thr):
                ok = True
                break
            plateau = L._noise_plateau(C, vscale[idx])
            if plateau is not None:                   # noisy function values (rounding): accept at the plateau
                thr = 10 * plateau[1] * np.maximum(vscale[idx], TINY)
                ok = True
                break
        if ok or x1 - x0 <= minwidth or len(out) + len(todo) > max_pieces:
            big = np.abs(C) > thr
            keepc = big.any(axis=0)
            rows = np.nonzero(big.any(axis=1))[0]
            keep = rows[-1] + 1 if len(rows) else 1
            h = (x1 - x0) / (2.0 * np.arange(C.shape[0]) + 1.0)
            err2[idx[keepc]] += (h[keep:, None] * C[keep:, keepc] ** 2).sum(axis=0)
            err2[idx[~keepc]] += (h[:, None] * C[:, ~keepc] ** 2).sum(axis=0)
            out.append((x0, x1, idx[keepc], C[:keep, keepc]))
            unresolved += not ok
        else:
            m = 0.5 * (x0 + x1)
            todo.append((m, x1))
            todo.append((x0, m))
    out.sort(key=lambda z: z[0])
    err = np.sqrt(err2 + (EPS * vscale) ** 2 * (b - a))
    return SparsePiecewiseLegendre([z[0] for z in out] + [out[-1][1]], [z[2] for z in out], [z[3] for z in out], N, err), unresolved


def from_dense_list(pls, tol=1e-14):
    """Sparse family from individually realised functions (each a PiecewiseLegendre with one column)."""
    a, b = pls[0].breaks[0], pls[0].breaks[-1]
    regions, allb = [], [np.array([a, b])]
    for p in pls:
        thr = tol * max(p.maxabs(), TINY)
        live = [i for i, C in enumerate(p.coefs) if np.abs(C).max() > thr]
        if live:
            lo, hi = p.breaks[live[0]], p.breaks[live[-1] + 1]
            allb.append(p.breaks[live[0]: live[-1] + 2])
        else:
            lo = hi = None
        regions.append((lo, hi))
    nb = L.union_breaks(*allb)
    cols = [[] for _ in range(len(nb) - 1)]
    blocks = [[] for _ in range(len(nb) - 1)]
    for j, (p, (lo, hi)) in enumerate(zip(pls, regions)):
        if lo is None:
            continue
        i0, i1 = np.searchsorted(nb, lo), np.searchsorted(nb, hi)
        sub = nb[i0: i1 + 1]
        R = L.refine(L.PiecewiseLegendre(p.breaks, p.coefs, p.err), L.union_breaks(p.breaks, sub))
        pb = L.union_breaks(p.breaks, sub)
        for k in range(i0, i1):
            q = int(np.searchsorted(pb, 0.5 * (nb[k] + nb[k + 1])) - 1)
            cols[k].append(j)
            blocks[k].append(R[q])
    coefs = []
    for k in range(len(nb) - 1):
        if not blocks[k]:
            coefs.append(np.zeros((1, 0)))
            continue
        n = max(B.shape[0] for B in blocks[k])
        coefs.append(np.hstack([L._pad(B, n) for B in blocks[k]]))
    return SparsePiecewiseLegendre(nb, [np.array(c, np.intp) for c in cols], coefs, len(pls), np.concatenate([p.err for p in pls]))


# ------------------------------------------------------------------------------------------ algebra

def _refine_sparse(s, nb):
    if len(nb) == len(s.breaks) and np.array_equal(nb, s.breaks):
        return s
    mids = 0.5 * (nb[:-1] + nb[1:])
    parent = np.clip(np.searchsorted(s.breaks, mids, side="right") - 1, 0, s.npieces - 1)
    coefs = [L.restrict(s.coefs[k], s.breaks[k], s.breaks[k + 1], nb[i], nb[i + 1]) if s.coefs[k].size else s.coefs[k]
             for i, k in enumerate(parent)]
    return SparsePiecewiseLegendre(nb, [s.cols[k] for k in parent], coefs, s.N, s.err)


def _gram_ss(A, B):
    nb = L.union_breaks(A.breaks, B.breaks)
    A, B = _refine_sparse(A, nb), _refine_sparse(B, nb)
    G = np.zeros((A.N, B.N))
    for p in range(len(nb) - 1):
        ca, cb = A.cols[p], B.cols[p]
        if len(ca) == 0 or len(cb) == 0:
            continue
        m = min(A.coefs[p].shape[0], B.coefs[p].shape[0])
        h = (nb[p + 1] - nb[p]) / (2.0 * np.arange(m) + 1.0)
        G[np.ix_(ca, cb)] += A.coefs[p][:m].T @ (h[:, None] * B.coefs[p][:m])
    return G


def _projections(A, q):
    """For each piece j of the dense realisation q: the matrix of <k_i, P_k^(j)>, shape (A.N, deg_j + 1)."""
    key = (q.breaks.tobytes(), tuple(q.degrees))
    hit = A._proj.get(key)
    if hit is not None:
        return hit
    nb = L.union_breaks(A.breaks, q.breaks)
    As = _refine_sparse(A, nb)
    Bs = [np.zeros((A.N, d + 1)) for d in q.degrees]
    owner = np.clip(np.searchsorted(q.breaks, 0.5 * (nb[:-1] + nb[1:]), side="right") - 1, 0, q.npieces - 1)
    for p in range(len(nb) - 1):
        cp = As.cols[p]
        if len(cp) == 0:
            continue
        j = owner[p]
        d = q.degrees[j]
        Cp = As.coefs[p]
        n = (Cp.shape[0] - 1 + d) // 2 + 1
        t, w = gauss(n)
        x0, x1 = nb[p], nb[p + 1]
        x = x0 + 0.5 * (x1 - x0) * (t + 1.0)
        tj = (2.0 * x - q.breaks[j] - q.breaks[j + 1]) / (q.breaks[j + 1] - q.breaks[j])
        Pv = npleg.legvander(tj, d) * (w * 0.5 * (x1 - x0))[:, None]
        Bs[j][cp] += polyval(t, Cp).T @ Pv
    if len(A._proj) > 8:
        A._proj.clear()
    A._proj[key] = Bs
    return Bs


def _gram_sd(A, q):
    Bs = _projections(A, q)
    return sum(B @ C for B, C in zip(Bs, q.coefs))


def gram(A, B):
    sa, sb = isinstance(A, SparsePiecewiseLegendre), isinstance(B, SparsePiecewiseLegendre)
    if sa and sb:
        return _gram_ss(A, B)
    if sa:
        return _gram_sd(A, B)
    if sb:
        return _gram_sd(B, A).T
    return L.gram(A, B)


def sqnorms(A):
    if not isinstance(A, SparsePiecewiseLegendre):
        return L.sqnorms(A)
    out = np.zeros(A.N)
    for p, C in enumerate(A.coefs):
        if C.size:
            h = (A.breaks[p + 1] - A.breaks[p]) / (2.0 * np.arange(C.shape[0]) + 1.0)
            np.add.at(out, A.cols[p], (h[:, None] * C * C).sum(axis=0))
    return out


def combine(A, M):
    """Dense realisation of A @ M (column combinations), M of shape (A.N, k)."""
    if not isinstance(A, SparsePiecewiseLegendre):
        return L.combine(A, M)
    M = np.asarray(M, float)
    coefs = [C @ M[A.cols[p]] if C.size else np.zeros((1, M.shape[1])) for p, C in enumerate(A.coefs)]
    return L.PiecewiseLegendre(A.breaks, coefs, np.abs(M).T @ A.err)


def colsum(A):
    return combine(A, np.ones((A.N, 1))) if isinstance(A, SparsePiecewiseLegendre) else L.colsum(A)


def integral(A, c=None, d=None):
    if not isinstance(A, SparsePiecewiseLegendre):
        return L.integral(A, c, d)
    a, b = A.breaks[0], A.breaks[-1]
    c = a if c is None else max(a, c)
    d = b if d is None else min(b, d)
    As = _refine_sparse(A, L.union_breaks(A.breaks, np.array([c, d])))
    out = np.zeros(A.N)
    for p, C in enumerate(As.coefs):
        if C.size and As.breaks[p] >= c and As.breaks[p + 1] <= d:
            np.add.at(out, As.cols[p], C[0] * (As.breaks[p + 1] - As.breaks[p]))
    return out


def take(A, idx):
    return A.take(idx)


def is_sparse(A):
    return isinstance(A, SparsePiecewiseLegendre)
