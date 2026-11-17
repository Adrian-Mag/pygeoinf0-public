"""The parts of the spectral energy engine (spectral.py) that involve no transform: the tapered kernel, Gauss nodes on
segments, the segment pairs closer than r_corr, and the exact corrections on them.

    taper, khat      T(rho) = erfc((rho - 0.9 Xi) / (Xi / 40)) / 2 and Khat T = (1 + l^2 |xi|^2)^-s T(|xi|)
    TaperedTable     K_T(r) = (2 pi)^-d int Khat T e^{i xi.x} dxi, the kernel the tapered spectral sum represents, by
                     its radial (Hankel) transform: d = 3, (2 pi^2)^-1 int Khat T rho^2 sin(rho r) / (rho r) drho;
                     d = 2, (2 pi)^-1 int Khat T rho J_0(rho r) drho; Gauss on panels of width <= min(2 / rmax,
                     Xi / 61) up to 1.05 Xi (T(1.05 Xi) < 1e-17), tabulated with its derivative at spacing 0.05 / Xi
                     for cubic Hermite interpolation (relative error ~ (0.05)^4 / 384 ~ 2e-8)
    segment_nodes    Gauss nodes on segments: Q nodes per panel of half-length OMEGA / kmax integrate e^{i xi.y} along a
                     segment to ~1e-9 for |xi| <= kmax (16 nodes and a phase of 13 per half panel: J_32(13) ~ 4e-10)
    close_pairs      segment pairs (a, b, shift) closer than r_corr: sample points at spacing <= max(r_corr, median
                     length / 16), KD-trees, then the exact distance of the segments; on tori b is translated by the
                     shifts to the neighbouring cells. Without shift: spatial blocks (grid cells of 2-4 sample radii,
                     about 2 CHUNK / (mean neighbours) samples each), one KD-tree query of pairs per block over its
                     samples and the later ones near it, so each pair of samples is seen once and nothing is counted
                     first (plan 005: 3.5x faster than blocks of segments against one tree at L3 1e4 x 27, the same
                     pairs); shifts: blocks of whole segments against the translated tree, at most CHUNK sample
                     pairs at a time. In workers, the pairs streamed back and checked against settings.memory_gb
                     (PAIR_BYTES each):
                     at fixed l and domain their number grows like n^2 (rays crossing within r_corr), so a large set
                     is refused while it is being found. Optionally only the pairs of different measures that meet
                     (distance 0: shared endpoints, crossings) or are at a small angle (sin < smax): the collinear
                     ones, whose omitted term converges slowly, and shared endpoints, without which the estimate fell
                     below the error on the fan (spectral.py; pg.settings.collinear)
    exact_energies   E_K of those pairs (kernels.segment_pair_list: exact reductions)
    taper_energies   E_{K_T} of those pairs (Gauss products of the table, panels as for the nodes)

The pair search and both energies run in blocks over forked workers (execution.map_blocks) and use numpy and scipy
only: no transform, and so no OpenMP of finufft, runs in a worker (spectral.py keeps the transforms in the parent).
"""
import math

import numpy as np
from scipy import special

from ...core.errors import Unsupported
from . import kernels as KR
from ...execution import imap_blocks, map_blocks, workers
from ...quadrature import gauss01

OMEGA, Q = 13.0, 16      # panels of half-length OMEGA / kmax with Q Gauss nodes (module docstring)
FRAC = 0.8               # the taper is 1 to rounding below FRAC Xi
H_TABLE = 0.05           # spacing of the table of K_T, times 1 / Xi
STEP = 8192              # segment pairs per block of the correction energies
SAMPLES = 10000          # sample points per block of the pair search
CHUNK = 2000000          # sample pairs found at once in a block of the pair search (~50 MB of indices and distances)
TOUCH = 1e-9             # pairs at most TOUCH r_corr apart meet (shared endpoints, crossings: corrected also with smax)
PAIR_BYTES = 120         # memory of one corrected pair: (a, b, k) and E_K kept, E_{K_T} and the sparse C per resolution,
                         # the copies of the search's merge


def taper(Xi, rho):
    """T(rho): 1 - 7.7e-9 at 0.8 Xi, 1/2 at 0.9 Xi, 7.7e-9 at Xi (an erfc of width Xi / 40)."""
    a, w = FRAC * Xi, (1 - FRAC) * Xi / 8
    return 0.5 * special.erfc((np.asarray(rho, float) - a - 4 * w) / w)


def khat(kernel, Xi, xi2):
    """Khat T at squared frequencies xi2."""
    return (1 + kernel.l ** 2 * xi2) ** (-kernel.s) * taper(Xi, np.sqrt(xi2))


class TaperedTable:
    """K_T on [0, rmax] (module docstring), evaluated by cubic Hermite interpolation."""

    def __init__(self, kernel, Xi, rmax):
        top = 1.05 * Xi
        edges = np.linspace(0.0, top, int(math.ceil(top * rmax / 2)) + 64)
        t, w = gauss01(20)
        rho = (edges[:-1, None] + np.diff(edges)[:, None] * t[None]).ravel()
        wr = (np.diff(edges)[:, None] * w[None]).ravel() * khat(kernel, Xi, rho ** 2)
        self.h = H_TABLE / Xi
        r = np.arange(0.0, rmax + 2 * self.h, self.h)
        self.f, self.df = np.empty(len(r)), np.empty(len(r))
        c = 2 * np.pi ** 2 if kernel.d == 3 else 2 * np.pi
        for i in range(0, len(r), 256):
            x = r[i:i + 256, None] * rho[None]
            if kernel.d == 3:
                z = np.where(x > 1e-8, x, 1.0)
                f = np.where(x > 1e-8, np.sin(z) / z, 1 - x * x / 6) * rho ** 2
                g = np.where(x > 1e-8, (np.cos(z) - np.sin(z) / z) / z, -x / 3) * rho ** 3
            else:
                f, g = special.j0(x) * rho, -special.j1(x) * rho ** 2
            self.f[i:i + 256], self.df[i:i + 256] = f @ wr / c, g @ wr / c
        self.df *= self.h
        self.n = len(r) - 1

    def __call__(self, r):
        u = np.asarray(r, float) / self.h
        i = np.minimum(u.astype(np.int64), self.n - 1)
        t = u - i
        t2 = t * t
        t3 = t2 * t
        return ((2 * t3 - 3 * t2 + 1) * self.f[i] + (t3 - 2 * t2 + t) * self.df[i] + (3 * t2 - 2 * t3) * self.f[i + 1]
                + (t3 - t2) * self.df[i + 1])


def far_reach(kernel, tol):
    """A distance beyond which K(r) r^(d-1) < tol times its maximum."""
    r = np.linspace(0, 200 * kernel.l, 200001)[1:]
    k = kernel(r) * r ** (kernel.d - 1)
    big = np.nonzero(k > tol * np.max(k))[0]
    return float(r[big[-1]]) if len(big) else 0.0


def segment_nodes(P, E1, weight, owner, kmax):
    """Nodes X (N, d), weights W (arc length times the atom's weight) and owning measures of the Gauss panels."""
    L = np.linalg.norm(E1, axis=1)
    npan = np.maximum(1, np.ceil(kmax * L / (2 * OMEGA))).astype(int)
    t, w = gauss01(Q)
    seg = np.repeat(np.arange(len(L)), npan)
    j = np.arange(len(seg)) - np.repeat(np.cumsum(npan) - npan, npan)
    h = 1.0 / npan[seg]
    T = (j * h)[:, None] + h[:, None] * t[None]
    X = (P[seg, None, :] + T[..., None] * E1[seg, None, :]).reshape(-1, P.shape[1])
    W = ((L[seg] * h * weight[seg])[:, None] * w[None]).ravel()
    return X, W, np.repeat(owner[seg], Q)


# ------------------------------------------------------------------------------------------ pairs closer than r_corr

def distance(P1, Q1, P2, Q2):
    """Distances of the segments P1 -> Q1 and P2 -> Q2 (rows) by clamped closest points (Ericson, Real-Time Collision
    Detection, 5.1.9)."""
    d1, d2, r = Q1 - P1, Q2 - P2, P1 - P2
    a, e, f = np.sum(d1 * d1, 1), np.sum(d2 * d2, 1), np.sum(d2 * r, 1)
    c, b = np.sum(d1 * r, 1), np.sum(d1 * d2, 1)
    den = a * e - b * b
    s = np.where(den > 1e-14 * a * e, np.clip((b * f - c * e) / np.where(den > 0, den, 1.0), 0, 1), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm(P1 + d1 * s[:, None] - P2 - d2 * t[:, None], axis=1)


def _select(a, b, P, Qe, rc, owner, smax, s):
    """Of the segment pairs (a, b), b translated by s, those closer than rc; with smax, of the pairs of different
    measures only those that meet (distance <= TOUCH rc: shared endpoints, crossings) or with sin(angle) < smax."""
    dist = distance(P[a], Qe[a], P[b] + s, Qe[b] + s)
    near = dist < rc
    a, b, dist = a[near], b[near], dist[near]
    if smax is not None:
        near = (owner[a] == owner[b]) | (dist <= TOUCH * rc) | (sine2(Qe[a] - P[a], Qe[b] - P[b]) < smax ** 2)
        a, b = a[near], b[near]
    return a, b


def _cell_blocks(X, r, kbar):
    """Blocks of samples for the search without shift: the samples sorted by a grid of cells of side c (2 r to 4 r,
    smaller where they are dense, so that a block holds about 2 CHUNK / kbar samples, kbar the mean number of
    samples within r of a sample), each cell one block, split by position when it holds more. Returns the order
    (sorted position -> sample), the cell grid (lo, c, dims, sorted cell keys, their first positions) and the
    blocks [(start, end)) of positions."""
    cap = max(256, int(2 * CHUNK / max(kbar, 1.0)))
    lo = X.min(axis=0)
    span = float(np.max(X.max(axis=0) - lo)) + r
    for mult in (4.0, 3.0, 2.0):                             # coarser cells waste less of the margin
        c = max(mult * r, span / 1000)
        G = np.floor((X - lo) / c).astype(np.int64)
        dims = G.max(axis=0) + 1
        key = np.ravel_multi_index(G.T, dims)
        cells, count = np.unique(key, return_counts=True)
        if count.mean() <= cap:
            break
    order = np.argsort(key, kind="stable")
    first = np.r_[0, np.cumsum(count)]
    blocks = [(int(s + q), int(min(s + q + cap, e))) for s, e in zip(first[:-1], first[1:])
              for q in range(0, e - s, cap)]
    return order, (lo, c, dims, cells, first), blocks


def _search_cells(X, seg, order, grid, r, P, Qe, rc, owner, smax, zero, blocks, lo_b, hi_b):
    """The pairs (a, b), a < b, closer than rc from the blocks lo_b:hi_b, without shift: the samples of a block (its
    core, sorted positions [s, e)) and the samples after it within r of the core's bounding box in one KD-tree; its
    pairs (i, j) with i in the core (core first, query_pairs gives i < j) are every pair of samples within r whose
    first sample, in sorted position, is in this block, so each pair of samples is seen in exactly one block."""
    from scipy.spatial import cKDTree
    glo, c, dims, cells, first = grid
    m, out = len(P), []
    for s, e in blocks[lo_b:hi_b]:
        core = order[s:e]
        blo, bhi = X[core].min(axis=0) - r * (1 + 1e-9), X[core].max(axis=0) + r * (1 + 1e-9)
        g0 = np.maximum(np.floor((blo - glo) / c).astype(np.int64), 0)
        g1 = np.minimum(np.floor((bhi - glo) / c).astype(np.int64), dims - 1)
        keys = np.ravel_multi_index(np.stack([g.ravel() for g in np.meshgrid(
            *[np.arange(a, b + 1) for a, b in zip(g0, g1)], indexing="ij")]), dims)
        at = np.searchsorted(cells, keys)
        ok = (at < len(cells)) & (cells[np.minimum(at, len(cells) - 1)] == keys)
        rng = [(max(int(first[i]), e), int(first[i + 1])) for i in at[ok]]
        later = np.concatenate([order[a:b] for a, b in rng if b > a] or [np.zeros(0, np.int64)])
        later = later[np.all((X[later] >= blo) & (X[later] <= bhi), axis=1)]
        pts = np.concatenate([core, later])
        pr = cKDTree(X[pts]).query_pairs(r, output_type="ndarray")
        pr = pr[pr[:, 0] < len(core)]
        sp = seg[pts]
        a, b = sp[pr[:, 0]], sp[pr[:, 1]]
        a, b = np.minimum(a, b), np.maximum(a, b)
        keep = a < b
        key = np.unique(a[keep].astype(np.int64) * m + b[keep])
        out.append(_select(key // m, key % m, P, Qe, rc, owner, smax, zero))
    return [np.concatenate(x) for x in zip(*out)] if out else [np.zeros(0, np.int64)] * 2


def _search(X, seg, tree, shifts, r, P, Qe, rc, owner, smax, ks, lo, hi):
    """The pairs (a, b, k), a <= b, closer than rc with b translated by shifts[k] (k in ks), found from the samples
    X[lo:hi] (of whole segments) in chunks of at most CHUNK sample pairs (counted first); smax as in _select."""
    from scipy.spatial import cKDTree
    m, out = len(P), []
    for k in ks:
        s = shifts[k]
        Y = X[lo:hi] - s
        cum = np.concatenate([[0], np.cumsum(tree.query_ball_point(Y, r, return_length=True))])
        keys, i0 = [], 0
        while i0 < len(Y):
            i1 = max(i0 + 1, int(np.searchsorted(cum, cum[i0] + CHUNK, side="right")) - 1)
            D = cKDTree(Y[i0:i1]).sparse_distance_matrix(tree, r, output_type="ndarray")   # |x_i - (x_j + s)| <= r
            a, b = seg[D["i"] + lo + i0], seg[D["j"]]
            keep = a <= b
            keys.append(np.unique(a[keep].astype(np.int64) * m + b[keep]))
            i0 = i1
        key = np.unique(np.concatenate(keys))
        a, b = _select(key // m, key % m, P, Qe, rc, owner, smax, s)
        out.append((a, b, np.full(len(a), k, np.int8)))
    return [np.concatenate(x) for x in zip(*out)]


def sine2(d1, d2):
    """sin^2 of the angles between the directions d1 and d2 (rows)."""
    c = np.sum(d1 * d2, 1) ** 2 / (np.sum(d1 * d1, 1) * np.sum(d2 * d2, 1))
    return np.maximum(0.0, 1.0 - c)


def close_pairs(P, Qe, rc, shifts, owner=None, smax=None, memory_gb=None):
    """Segment pairs closer than rc, each interaction once: (a, b, k) with segment b translated by shifts[k] (shifts[0]
    = 0), a <= b; the self pairs (a, a, 0) always. On a torus E_ab = sum_s E(a, b + s), so the pairs (a, b, s) with
    a < b and the self images (a, a, s), s != 0, give every close term of the lattice sums exactly once. smax: of the
    pairs of different measures (owner) only those that meet or with sin(angle) < smax. memory_gb: refused (Unsupported) as soon as
    the pairs found need more, at PAIR_BYTES each (an upper bound: a pair of segments in two blocks of the search
    without shift is counted twice).

    Without shift (every domain; the only shift on bounded ones) the samples are searched in spatial blocks with one
    KD-tree query of pairs each (_cell_blocks, _search_cells: each pair of samples once, no counting pass); the shifts
    of tori search the samples of blocks of segments against the translated tree (_search)."""
    from scipy.spatial import cKDTree
    m = len(P)
    L = np.linalg.norm(Qe - P, axis=1)
    h = max(rc, float(np.median(L)) / 16)
    ns = np.maximum(2, np.ceil(L / h).astype(int) + 1)
    seg = np.repeat(np.arange(m), ns)
    start = np.cumsum(ns) - ns
    j = np.arange(len(seg)) - np.repeat(start, ns)
    X = P[seg] + (j / (ns[seg] - 1))[:, None] * (Qe - P)[seg]
    r = rc + float(np.max(L / (ns - 1)))                 # samples of two segments closer than rc are within r
    tree = cKDTree(X)
    owner = np.zeros(m, np.int64) if owner is None else np.asarray(owner)
    probe = X[np.random.default_rng(0).choice(len(X), min(len(X), 2000), replace=False)]
    order, grid, blocks = _cell_blocks(X, r, float(np.mean(tree.query_ball_point(probe, r, return_length=True))))
    per = max(1, len(blocks) // (4 * workers())) if workers() > 1 else max(1, len(blocks) // 64)
    runs = [(_search_cells, (X, seg, order, grid, r, P, Qe, rc, owner, smax, np.zeros(P.shape[1]), blocks),
             [(i, min(i + per, len(blocks))) for i in range(0, len(blocks), per)])]
    if len(shifts) > 1:                                  # tori: the translated neighbours, by blocks of segments
        step = SAMPLES if workers() <= 1 else min(SAMPLES, max(512, -(-len(X) // (4 * workers()))))
        cuts = np.unique(start[np.minimum(np.searchsorted(start, np.arange(0, len(X), step)), m - 1)])
        runs.append((_search, (X, seg, tree, shifts, r, P, Qe, rc, owner, smax, range(1, len(shifts))),
                     list(zip(cuts.tolist(), cuts[1:].tolist() + [len(X)]))))
    parts, found = [], m
    for f, args, tasks in runs:
        results = imap_blocks(f, args, tasks)
        for _, v in results:
            parts.append(v if len(v) == 3 else (v[0], v[1], np.zeros(len(v[0]), np.int8)))
            found += len(v[0])
            if memory_gb is not None and found * PAIR_BYTES > memory_gb * 2 ** 30:
                results.close()
                raise Unsupported(f"the exact corrections of the spectral engine on the segment pairs closer than "
                                  f"r_corr = {rc:.3g} exceed settings.memory_gb = {memory_gb:g}: {found} pairs "
                                  f"({PAIR_BYTES} bytes each) after {len(parts)} blocks of the search, and their "
                                  f"number grows like n^2 at fixed l" + ("; pg.settings.collinear = 0.3 corrects only "
                                  "the pairs of different measures that meet or are nearly collinear" if smax is None
                                  else ""))
    a = np.concatenate([np.arange(m)] + [p[0] for p in parts])
    b = np.concatenate([np.arange(m)] + [p[1] for p in parts])
    k = np.concatenate([np.zeros(m, np.int64)] + [p[2].astype(np.int64) for p in parts])
    del parts
    key = np.unique((a * m + b) * len(shifts) + k)
    ab, k = np.divmod(key, len(shifts))
    a, b = np.divmod(ab, m)
    return a, b, k.astype(np.int8)


# ------------------------------------------------------------------------------------------ energies of the pairs

def _blocks(n):
    w = workers()
    step = STEP if w <= 1 else max(256, min(STEP, -(-n // (4 * w))))
    return [(lo, min(lo + step, n)) for lo in range(0, n, step)]


def _in_blocks(f, args, n):
    out = np.empty(n)
    for (lo, hi), v in map_blocks(f, args, _blocks(n)):
        out[lo:hi] = v
    return out


def _exact_block(kernel, P, Qe, a, b, k, shifts, lo, hi):
    out = np.empty(hi - lo)
    for c in range(lo, hi, 256):
        e = min(c + 256, hi)
        s = shifts[k[c:e]]
        out[c - lo:e - lo] = KR.segment_pair_list(kernel, P[a[c:e]], Qe[a[c:e]], P[b[c:e]] + s, Qe[b[c:e]] + s)
    return out


def exact_energies(kernel, P, Qe, a, b, k, shifts):
    """int int K over the pairs (exact reductions), unweighted."""
    return _in_blocks(_exact_block, (kernel, P, Qe, a, b, k, shifts), len(a))


def _taper_block(table, P, Qe, a, b, k, shifts, kmax, lo, hi):
    s = shifts[k[lo:hi]]
    return gauss_products(table, P[a[lo:hi]], Qe[a[lo:hi]], P[b[lo:hi]] + s, Qe[b[lo:hi]] + s, kmax)


def taper_energies(table, P, Qe, a, b, k, shifts, kmax):
    """int int K_T over the pairs, unweighted (K_T band-limited at kmax)."""
    return _in_blocks(_taper_block, (table, P, Qe, a, b, k, shifts, kmax), len(a))


def gauss_products(kfun, P1, Q1, P2, Q2, kmax, batch=32):
    """int int kfun(|x - y|) ds dt over the segment pairs (rows) by products of Gauss panels resolving kmax; vectorised
    over batches of pairs with equal panel counts."""
    out = np.empty(len(P1))
    t, w = gauss01(Q)
    L1, L2 = np.linalg.norm(Q1 - P1, axis=1), np.linalg.norm(Q2 - P2, axis=1)
    n1 = np.maximum(1, np.ceil(kmax * L1 / (2 * OMEGA))).astype(np.int64)
    n2 = np.maximum(1, np.ceil(kmax * L2 / (2 * OMEGA))).astype(np.int64)
    key = n1 * 100000 + n2
    for kv in np.unique(key):
        idx = np.nonzero(key == kv)[0]
        p1, p2 = int(kv // 100000), int(kv % 100000)
        t1, w1 = ((np.arange(p1)[:, None] + t[None]) / p1).ravel(), np.tile(w, p1) / p1
        t2, w2 = ((np.arange(p2)[:, None] + t[None]) / p2).ravel(), np.tile(w, p2) / p2
        for c in range(0, len(idx), batch):
            i = idx[c:c + batch]
            X = P1[i, None] + t1[None, :, None] * (Q1 - P1)[i, None]
            Y = P2[i, None] + t2[None, :, None] * (Q2 - P2)[i, None]
            D = X[:, :, None] - Y[:, None]
            R = np.sqrt(np.einsum("bijk,bijk->bij", D, D))
            out[i] = np.einsum("bij,i,j->b", kfun(R), w1, w2) * L1[i] * L2[i]
    return out
