"""Support-local tensor realisation on boxes (plan 025 P1; EXP-038 DESIGN-local.md): realisation="local".

A function with a stated support box (bank functions: functions/base.BankFunction.fit_breaks; M(f, support=...)) is
fitted on that box only (tensor.fit_tensor with the box as reference) and is zero outside it: a patch, the tensor
realisation of the box. A linear combination keeps its patches: PS holds N columns as an optional tensor part on the
whole box plus patches (c, cols, T), T a tensor realisation (tensor.TensorSeries) on its own box holding columns cols of the N,
times c. So a + b costs its patches, not a merge onto the union of their partitions across the box (the 100-bump sum of
plan 022 P6: 28.5 s in tensor.lincomb):

    values       the tensor part plus the patches whose half-open box holds the point (closed at the domain's faces)
    <A, B>       over the pairs of patches whose boxes overlap: both re-expressed exactly on the intersection (the union
                 of their breaks there, tensor._transfer_apply) and contracted with the mass blocks; the tensor part with
                 a patch on the patch's box; candidate pairs by a sweep over the boxes
    integrals    per patch (over coordinate boxes too: box_integrals.axis_row clips each piece)
    lincomb      patches with the same box and columns are added as tensors (tensor.lincomb on that box), others kept
    products     of single functions: per overlapping pair, on the intersection
    the rest     (Lp norms, products of families, separated families and expansions): the patches are extended by zero to
                 the whole box and merged (to_tensor), the route of realisation="tensor", with the same numbers

Functions without a support box are realised as with realisation="tensor". Norms are sums over pairs (<c, G c> for a
combination of patches, as in a classic Galerkin discretisation), not norms of merged coefficients: a combination that
cancels to far below its terms (a residual) has a rounding error of about eps sum |c_i c_j G_ij| (not in f.error), and
squared norms are clipped at 0. The patches are below.

From local_patches.py (merged here by plan 063):
Support-local patches (plan 025 P1): the data of realisation="local" (local.py) and their geometry.

PS holds N columns as an optional tensor part on the whole box plus patches (c, cols, R): R a tensor realisation
(tensor.TensorSeries) on its own box, times c, holding the columns cols of the N. Here: the support box carried by breaks
(Supported, M(f, support=...)), exact re-expansion of patches onto a common basis of a sub-box (transfer, onto), zero
extension (extend), candidate pairs of overlapping boxes (overlaps), and translates: a fit that equals an earlier one
shifted, to TYPE_TOL, shares its coefficients (canonical), so patches of one type at one offset share one Gram block.

From local_pairings.py (merged here by plan 063):
Exact pairings of measures with families of support-local patches (plan 025 P3): the data x basis matrices of a
discretisation by pixels, hats or other compactly supported functions (operators/galerkin.py: D.pairings).

    int phi_i dmu_j     for points (values), segments (arc length) and coordinate boxes (area or volume)

A segment P + t E, t in [0, 1], is clipped to each patch box it meets and split where it crosses the patch's breaks;
on each part the patch is one polynomial of degree at most sum_k (n_k - 1) along the line, so Gauss with
ceil((sum_k (n_k - 1) + 1) / 2) points is exact for the realisation. Patches of one translate type (the
canonical patch, one coefficient array) are evaluated together in the frame of that type's first patch, one call for all
their (segment, patch) pairs. A segment lying in a face shared by two boxes belongs to one of them (the half-open
convention of evaluation: [lo, hi), closed at the domain's upper faces). Coordinate boxes: local.integral (exact).
The tensor part of a combination (if any) is a patch on the whole box.

Candidate pairs come from the bounding boxes; the clipping removes those that miss. Returned as scipy.sparse.
"""


import math

import numpy as np
import scipy.sparse as sp
from numpy.polynomial import legendre as npleg

from ...core.errors import Unsupported
from . import tensor as T
from .legendre import EPS, analysis, union_breaks
from ...quadrature import gauss

# ---------------------------------------------------------------- from local_patches.py (plan 063)
DENSE_MAX = 6000                # Gram matrices with more rows or columns than this are returned as scipy.sparse
TYPE_TOL = 1e-13                # a fit is a translate of an earlier one when their coefficients agree to this times
#                                 their maximum: it then shares that one's (and its error grows by the difference)
MAX_TYPES = 4096                # translate representatives kept per backend


class Supported(tuple):
    """Breaks per axis (as tensor.fit takes them) that also carry the support box ((lo, hi) per axis) and, for bank
    functions, their translation (key, origin) (functions/base.BankFunction.translation)."""

    def __new__(cls, breaks, box, translation=None):
        obj = super().__new__(cls, breaks)
        obj.box = tuple((float(a), float(b)) for a, b in box)
        obj.translation = translation
        return obj

    def __reduce__(self):
        return (Supported, (tuple(self), self.box, self.translation))


class PS:
    """N columns: base (a TensorSeries on the whole reference box, or None) plus patches (c, cols, R): R a TensorSeries on its own box
    with len(cols) columns, scaled by c, holding the columns cols of the N; lo, hi the patches' boxes (P, d)."""

    __slots__ = ("N", "base", "patches", "lo", "hi", "_err", "_types")

    def __init__(self, N, base, patches):
        self.N, self.base = int(N), base
        self.patches = tuple((float(c), np.asarray(cols, np.intp), R) for c, cols, R in patches)
        d = base.d if base is not None else self.patches[0][2].d
        self.lo = np.array([[b.breaks[0] for b in R.bases] for _, _, R in self.patches]).reshape(-1, d)
        self.hi = np.array([[b.breaks[-1] for b in R.bases] for _, _, R in self.patches]).reshape(-1, d)
        self._err = None
        self._types = None

    @property
    def types(self):
        """(P,) type ids: patches that are translates of one another (the same breaks relative to their box and the same
        coefficients to TYPE_TOL) share an id; type_reps[id] is the first one."""
        if self._types is None:
            self._types = _types(self.patches, self.lo)
        return self._types

    @property
    def err(self):
        if self._err is None:
            e = np.zeros(self.N) if self.base is None else self.base.err.copy()
            for c, cols, R in self.patches:
                np.add.at(e, cols, abs(c) * R.err)
            self._err = e
        return self._err

    @property
    def size(self):
        return (0 if self.base is None else self.base.size) + sum(R.size for _, _, R in self.patches)

    @property
    def d(self):
        return self.lo.shape[1]


def with_support(f, breaks, box):
    """M(f, support=box): f times the indicator of the closed box, and breaks that hold the box's edges and the box
    (the same element on every realisation; realisation="local" samples f on the box only)."""
    box = support_box(box)
    if breaks is not None and len(box) == 1 and len(breaks) and np.ndim(breaks[0]) == 0:
        breaks = [breaks]                                           # a 1-D domain's flat breaks
    brk = tuple(sorted(set(list(b or ()) + [lo, hi])) for b, (lo, hi) in zip(breaks or [None] * len(box), box))

    def g(*X):
        X = np.broadcast_arrays(*[np.asarray(x, float) for x in X])
        inside = np.ones(X[0].shape, bool)
        for x, (lo, hi) in zip(X, box):
            inside &= (x >= lo) & (x <= hi)
        out = np.zeros(X[0].shape)
        if inside.any():
            out[inside] = np.broadcast_to(np.asarray(f(*[x[inside] for x in X]), float), (int(inside.sum()),))
        return out
    return g, Supported(brk, box)


def support_box(box):
    """support= as ((lo, hi), ...): a box of pairs, one pair (lo, hi) on a 1-D domain, an Interval or a Box."""
    if hasattr(box, "sides"):
        box = box.sides
    elif hasattr(box, "a") and hasattr(box, "b"):
        box = (box,)
    elif len(box) == 2 and all(np.ndim(v) == 0 and not hasattr(v, "a") for v in box):
        box = (tuple(box),)
    out = tuple((float(s.a), float(s.b)) if hasattr(s, "a") else (float(s[0]), float(s[1])) for s in box)
    for lo, hi in out:
        if not lo < hi:
            raise ValueError(f"support=: each side needs lo < hi, got ({lo}, {hi})")
    return out


def box_of(R):
    return tuple((b.breaks[0], b.breaks[-1]) for b in R.bases)


def transfer(C, k, old, new):
    """tensor._transfer_apply with the re-expansion matrices of all new pieces from one Vandermonde call (exact)."""
    if old == new:
        return C
    ob, nb = np.asarray(old.breaks), np.asarray(new.breaks)
    parent = np.clip(np.searchsorted(ob, 0.5 * (nb[:-1] + nb[1:]), side="right") - 1, 0, old.P - 1)
    t, _ = gauss(new.n)
    x0, x1 = ob[parent][:, None], ob[parent + 1][:, None]
    c0, c1 = nb[:-1][:, None], nb[1:][:, None]
    tp = (2.0 * (c0 + 0.5 * (c1 - c0) * (t + 1.0)) - x0 - x1) / (x1 - x0)          # (new.P, new.n) in the parent
    R = analysis(new.n)[None] @ npleg.legvander(tp, old.n - 1)                       # (new.P, new.n, old.n)
    same = (c0[:, 0] == x0[:, 0]) & (c1[:, 0] == x1[:, 0])
    if same.any():
        R[same] = np.eye(new.n, old.n)
    Ck = np.moveaxis(C, k, 0)
    out = np.matmul(R, Ck.reshape(old.P, old.n, -1)[parent])
    return np.moveaxis(out.reshape((new.S,) + Ck.shape[1:]), 0, k)


def onto(Rs, box):
    """The TRs Rs (each on a box holding box) re-expressed exactly on one basis of box: the union of their breaks
    inside it and its edges, the largest n per axis."""
    bases = []
    for k, (lo, hi) in enumerate(box):
        bs = [R.bases[k] for R in Rs]
        if any(isinstance(b, T.FourierBasis) for b in bs):
            raise Unsupported("support-local patches are piecewise Legendre (boxes, no periodic axes)")
        inner = [np.array([x for x in b.breaks if lo < x < hi]) for b in bs]      # union_breaks drops near-duplicates
        bases.append(T.AxisBasis(union_breaks(np.array([lo, hi]), *inner), max(b.n for b in bs)))
    out = []
    for R in Rs:
        C = R.C
        for k, (b, new) in enumerate(zip(R.bases, bases)):
            C = transfer(C, k, b, new)
        out.append(T.TensorSeries(bases, C, R.err))
    return out


def restrict(R, box):
    """R (a TensorSeries whose box holds box) re-expressed exactly on box: its breaks inside, the box's edges, the same n."""
    return onto([R], box)[0]


def extend(R, reference):
    """R (a TensorSeries on its box) extended by zero to the reference box."""
    C, bases = R.C, []
    for k, (b, (lo, hi)) in enumerate(zip(R.bases, reference)):
        left, right = b.breaks[0] > lo, b.breaks[-1] < hi
        if not (left or right):
            bases.append(b)
            continue
        new = T.AxisBasis(([lo] if left else []) + list(b.breaks) + ([hi] if right else []), b.n)
        pad = [(0, 0)] * C.ndim
        pad[k] = (b.n * left, b.n * right)
        C = np.pad(C, pad)
        bases.append(new)
    return T.TensorSeries(bases, C, R.err)


def _key(R):
    return (R.C.shape, tuple(b.n for b in R.bases),
            tuple(tuple(np.round((np.asarray(b.breaks) - b.breaks[0]) / (b.breaks[-1] - b.breaks[0]), 12))
                  for b in R.bases))


def _shifted(Q, R):
    """Whether R's breaks are Q's shifted (to rounding)."""
    for a, b in zip(Q.bases, R.bases):
        ra, rb = np.asarray(a.breaks) - a.breaks[0], np.asarray(b.breaks) - b.breaks[0]
        if np.max(np.abs(ra - rb)) > 8 * EPS * max(1.0, abs(a.breaks[-1]), abs(b.breaks[-1])):
            return False
    return True


def _types(patches, lo):
    """Type ids: the same coefficient array (shared by canonical, or equal) on shifted breaks."""
    ids, reps = np.empty(len(patches), np.intp), {}
    for p, (_, cols, R) in enumerate(patches):
        cand = reps.setdefault(_key(R), [])
        for t, Q in cand:
            if (Q.C is R.C or np.array_equal(Q.C, R.C)) and _shifted(Q, R):
                ids[p] = t
                break
        else:
            ids[p] = sum(len(v) for v in reps.values())
            cand.append((ids[p], R))
    return ids


def canonical(R, reps):
    """R, or the translate of an earlier fit in reps (a dict key -> [TensorSeries]) whose coefficients agree to TYPE_TOL: then
    R's bases with that fit's coefficients (shared), R's error plus the L2 norm of the difference."""
    key = _key(R)
    cand = reps.setdefault(key, [])
    scale = float(np.abs(R.C).max()) if R.C.size else 0.0
    for Q in cand:
        if _shifted(Q, R):
            D = R.C - Q.C
            if np.max(np.abs(D), initial=0.0) <= TYPE_TOL * scale:
                diff = np.sqrt(np.maximum(T._weighted_sq(R.bases, D), 0.0))
                return T.TensorSeries(R.bases, Q.C, R.err + diff)
    if sum(len(v) for v in reps.values()) < MAX_TYPES:
        cand.append(R)
    return R


def as_ps(R):
    return R if isinstance(R, PS) else PS(R.N, R, ())


def overlaps(A, B, chunk=2048):
    """(p, q) index arrays of the patches of A and B whose boxes overlap in a set of positive measure."""
    if not len(A.patches) or not len(B.patches):
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    order = np.argsort(B.lo[:, 0], kind="stable")
    blo, bhi = B.lo[order], B.hi[order]
    reach = np.maximum.accumulate(bhi[:, 0])          # boxes of B up to j end at most here: a sweep on axis 0
    ps, qs = [], []
    for s in range(0, len(A.patches), chunk):
        alo, ahi = A.lo[s:s + chunk], A.hi[s:s + chunk]
        j0 = np.searchsorted(reach, alo[:, 0].min(), side="right")
        j1 = np.searchsorted(blo[:, 0], ahi[:, 0].max(), side="left")
        if j1 <= j0:
            continue
        m = np.all((np.maximum(alo[:, None], blo[None, j0:j1]) < np.minimum(ahi[:, None], bhi[None, j0:j1])), axis=2)
        p, q = np.nonzero(m)
        ps.append(p + s)
        qs.append(order[q + j0])
    if not ps:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    return np.concatenate(ps), np.concatenate(qs)


class _Acc:
    """Accumulates blocks of an (n, m) matrix: dense when small, triplets otherwise."""

    def __init__(self, n, m):
        self.n, self.m = n, m
        self.dense = np.zeros((n, m)) if max(n, m) <= DENSE_MAX else None
        self.rows, self.cols, self.vals = [], [], []

    def add(self, rows, cols, block):
        if self.dense is not None:
            self.dense[np.ix_(rows, cols)] += block
            return
        r, c = np.meshgrid(rows, cols, indexing="ij")
        self.rows.append(r.ravel())
        self.cols.append(c.ravel())
        self.vals.append(np.asarray(block).ravel())

    def result(self):
        if self.dense is not None:
            return self.dense
        if not self.vals:
            return sp.csr_matrix((self.n, self.m))
        return sp.coo_matrix((np.concatenate(self.vals), (np.concatenate(self.rows), np.concatenate(self.cols))),
                             shape=(self.n, self.m)).tocsr()


MAX_SHAPES = 4096                # translation keys kept per backend (fit)


def _samples(d):
    """16 fixed relative sample points in [0.02, 0.98]^d: the check that a function is a translate (fit)."""
    return np.random.default_rng(20261031).uniform(0.02, 0.98, (16, d))


OFFSET_ROUND = 1e-15            # offsets of translate pairs equal to this (absolute) share one computed Gram block


def _self_key(R):
    return (id(R.C), tuple(tuple(np.round(np.diff(np.asarray(b.breaks)), 15)) for b in R.bases))


def gram(A, B, jac, diag=False, memo=None):
    """(A.N, B.N) Gram matrix of two PS (or TensorSeries); with diag, only its diagonal (A is B: the squared norms). memo: a dict
    kept by the backend for the Gram of a patch with itself (translates: one per type, keyed by the shared array)."""
    A, B = as_ps(A), as_ps(B)
    acc = np.zeros(A.N) if diag else _Acc(A.N, B.N)

    def add(rows, cols, G):
        if not diag:
            acc.add(rows, cols, G)
            return
        i, j = np.nonzero(rows[:, None] == cols[None, :])
        np.add.at(acc, rows[i], G[i, j])

    if A.base is not None and B.base is not None:
        G = T.gram(A.base, B.base, jac)
        add(np.arange(A.N), np.arange(B.N), G)
    if A.base is not None:
        for c, cols, R in B.patches:
            add(np.arange(A.N), cols, c * T.gram(*onto([A.base, R], box_of(R)), jac))
    if B.base is not None:
        for c, cols, R in A.patches:
            add(cols, np.arange(B.N), c * T.gram(*onto([R, B.base], box_of(R)), jac))
    ta, tb, cache = A.types, B.types, {}
    for p, q in zip(*overlaps(A, B)):
        a, ca, Ra = A.patches[p]
        b, cb, Rb = B.patches[q]
        if diag and not np.intersect1d(ca, cb).size:
            continue
        key = (ta[p], tb[q], tuple(np.round((B.lo[q] - A.lo[p]) / OFFSET_ROUND).astype(np.int64)))
        G = cache.get(key)
        if G is None and memo is not None and Ra.C is Rb.C and np.array_equal(A.lo[p], B.lo[q]):
            hit = memo.get(_self_key(Ra))
            if hit is None:
                hit = memo[_self_key(Ra)] = (Ra.C, T.gram(Ra, Ra, jac))      # Ra.C kept: its id stays its own
            G = cache[key] = hit[1]
        if G is None:                                       # translates at one offset: one block (like an element matrix)
            box = tuple(zip(np.maximum(A.lo[p], B.lo[q]), np.minimum(A.hi[p], B.hi[q])))
            G = cache[key] = T.gram(*onto([Ra, Rb], box), jac)
        add(ca, cb, a * b * G)
    return acc if diag else acc.result()


def sqnorms(A, jac, memo=None):
    return np.maximum(gram(A, A, jac, diag=True, memo=memo), 0.0)


def integral(A, jac, S=None, domain=None):
    """(N,): integrals over the domain, or over a coordinate box S (exact, box_integrals.box_integral)."""
    from ..rkhs.box_form import box_integral
    A = as_ps(A)
    if S is None:
        out = np.zeros(A.N) if A.base is None else np.asarray(T.integral(A.base, jac), float).copy()
        for c, cols, R in A.patches:
            np.add.at(out, cols, c * np.asarray(T.integral(R, jac), float))
        return out
    out = np.zeros(A.N)
    if A.base is not None:
        v = box_integral(A.base, S, domain, jac)
        if v is None:
            return None
        out += v
    for c, cols, R in A.patches:
        v = box_integral(R, S, domain, jac)
        if v is None:
            return None
        np.add.at(out, cols, c * v)
    return out


def _inside(xi, lo, hi, top):
    """Points in the half-open box [lo, hi) (closed where hi is the domain's upper face)."""
    m = np.ones(np.shape(xi[0]), bool)
    for x, a, b, t in zip(xi, lo, hi, top):
        m &= (x >= a) & ((x <= b) if b >= t else (x < b))
    return m


def evaluate(A, xi, reference):
    xi = [np.asarray(x, float) for x in xi]
    shape = np.shape(xi[0])
    flat = [x.ravel() for x in xi]
    out = np.zeros((flat[0].size, A.N)) if A.base is None else T.evaluate_points(A.base, flat).copy()
    top = [hi for _, hi in reference]
    for p, (c, cols, R) in enumerate(A.patches):
        m = _inside(flat, A.lo[p], A.hi[p], top)
        if m.any():
            v = T.evaluate_points(R, [x[m] for x in flat])
            idx = np.nonzero(m)[0]
            out[np.ix_(idx, cols)] += c * v
    return out.reshape(shape + (A.N,))


def grid_values(A, axes_points, reference):
    pts = [np.asarray(t, float) for t in axes_points]
    out = np.zeros(tuple(len(t) for t in pts) + (A.N,)) if A.base is None else T.values_on_grid(A.base, pts).copy()
    top = [hi for _, hi in reference]
    for p, (c, cols, R) in enumerate(A.patches):
        sel = [np.nonzero((t >= a) & ((t <= b) if b >= u else (t < b)))[0]
               for t, a, b, u in zip(pts, A.lo[p], A.hi[p], top)]
        if any(len(s) == 0 for s in sel):
            continue
        v = T.values_on_grid(R, [t[s] for t, s in zip(pts, sel)])
        out[np.ix_(*sel, cols)] += c * v
    return out


def lincomb(terms):
    """sum_j a_j R_j of TensorSeries and PS realisations with N columns: tensor parts merged (tensor.lincomb), patches kept;
    patches with the same box and columns added on that box."""
    N = terms[0][1].N
    trs, groups = [], {}
    for a, R in terms:
        if isinstance(R, T.TensorSeries):
            trs.append((a, R))
            continue
        if R.base is not None:
            trs.append((a, R.base))
        for c, cols, Q in R.patches:
            groups.setdefault((box_of(Q), cols.tobytes()), []).append((a * c, cols, Q))
    base = T.lincomb(trs) if trs else None
    patches = []
    for g in groups.values():
        if len(g) == 1:
            patches.append(g[0])
        else:
            patches.append((1.0, g[0][1], T.lincomb([(c, Q) for c, _, Q in g])))
    if not patches:
        return base
    return PS(N, base, patches)


def combine(A, M):
    """The K columns A M for an (N, K) matrix M."""
    M = np.asarray(M, float)
    K = M.shape[1]
    base = None if A.base is None else T.TensorSeries(A.base.bases, np.tensordot(A.base.C, M, axes=(A.base.d, 0)),
                                            np.abs(M).T @ A.base.err)
    patches = []
    for c, cols, R in A.patches:
        W = c * M[cols]                                            # (len(cols), K)
        nz = np.nonzero(np.any(W != 0, axis=0))[0]
        if not len(nz):
            continue
        if len(cols) == 1 and len(nz) == 1:
            patches.append((float(W[0, nz[0]]), nz, R))             # a scaled copy: no coefficients are copied
        else:
            patches.append((1.0, nz, T.TensorSeries(R.bases, np.tensordot(R.C, W[:, nz], axes=(R.d, 0)), np.abs(W[:, nz]).T @ R.err)))
    if not patches:
        return base if base is not None else None
    return PS(K, base, patches)


def take(A, idx):
    sel = np.atleast_1d(np.asarray(idx, np.intp))
    pos = np.full(A.N, -1, np.intp)
    pos[sel] = np.arange(len(sel))
    base = None if A.base is None else T.TensorSeries(A.base.bases, A.base.C[..., sel], A.base.err[sel])
    patches = []
    for c, cols, R in A.patches:
        keep = np.nonzero(pos[cols] >= 0)[0]
        if len(keep) == len(cols):
            patches.append((c, pos[cols], R))
        elif len(keep):
            patches.append((c, pos[cols[keep]], T.TensorSeries(R.bases, R.C[..., keep], R.err[keep])))
    return PS(len(sel), base, patches) if patches else base


def hstack(Rs, zero):
    """Columns side by side; zero(N) a zero TensorSeries on the whole box for parts without a tensor part."""
    Rs = [as_ps(R) for R in Rs]
    off = np.cumsum([0] + [R.N for R in Rs])
    patches = [(c, cols + o, Q) for R, o in zip(Rs, off) for c, cols, Q in R.patches]
    base = None
    if any(R.base is not None for R in Rs):
        base = T.hstack([R.base if R.base is not None else zero(R.N) for R in Rs])
    return PS(int(off[-1]), base, patches) if patches else base


def multiply(A, B):
    """A B for single functions (N = 1): per overlapping pair on the intersection."""
    A, B = as_ps(A), as_ps(B)
    terms, patches = [], []
    if A.base is not None and B.base is not None:
        terms.append((1.0, T.multiply(A.base, B.base)))
    for X, Y, swap in ((A, B, False), (B, A, True)):
        if X.base is not None:
            for c, cols, R in Y.patches:
                patches.append((c, cols, T.multiply(*onto([X.base, R], box_of(R)))))
    for p, q in zip(*overlaps(A, B)):
        a, ca, Ra = A.patches[p]
        b, cb, Rb = B.patches[q]
        box = tuple(zip(np.maximum(A.lo[p], B.lo[q]), np.minimum(A.hi[p], B.hi[q])))
        patches.append((a * b, ca, T.multiply(*onto([Ra, Rb], box))))
    if patches:
        terms.append((1.0, PS(1, None, patches)))
    if not terms:
        return None
    return lincomb(terms) if len(terms) > 1 or isinstance(terms[0][1], T.TensorSeries) else terms[0][1]


def to_tensor(A, reference):
    """The patches extended by zero to the whole box and merged: the realisation="tensor" form of A."""
    terms = [] if A.base is None else [(1.0, A.base)]
    for c, cols, R in A.patches:
        E = extend(R, reference)
        C = np.zeros(E.C.shape[:-1] + (A.N,))
        C[..., cols] = E.C
        err = np.zeros(A.N)
        err[cols] = E.err
        terms.append((c, T.TensorSeries(E.bases, C, err)))
    return T.lincomb(terms)


def breaks(A):
    parts = ([] if A.base is None else [A.base]) + [R for _, _, R in A.patches]
    return tuple(union_breaks(*[np.asarray(R.bases[k].breaks) for R in parts]) for k in range(A.d))


class LocalBackend(T.TensorBackend):
    """realisation="local": TensorBackend with support-local patches (module docstring)."""

    def __init__(self, measure, tol):              # boxes only (registry._choose)
        super().__init__(measure, tol, periodic_basis="legendre")
        self.realisation_name = "local"
        self._reps = {}                                            # translate representatives (canonical)
        self._selfgram = {}                                        # Grams of patches with themselves (local.gram)
        self._shapes = {}                                          # translation key -> an earlier fit (fit)

    def _zero(self, N):
        return self.constant(np.zeros(N))

    def _box(self, breaks):
        box = getattr(breaks, "box", None)
        if box is None:
            return None
        ref = self.domain.reference
        box = tuple((max(a, lo), min(b, hi)) for (a, b), (lo, hi) in zip(box, ref))
        if any(b <= a for a, b in box):
            return ()                                              # the support misses the domain: f = 0 here
        return box

    def sampling_box(self, breaks):
        """Where a fit with support= samples (core/sampling.py): the support's box in the domain, else the domain."""
        return self._box(breaks) or None

    def fit(self, fun, N=1, breaks=None):
        box = self._box(breaks)
        if box is None:
            return super().fit(fun, N=N, breaks=breaks)
        if box == ():
            return self._zero(N), 0
        g = self._internal(fun)
        tr = getattr(breaks, "translation", None) if N == 1 and box == getattr(breaks, "box", None) else None
        if tr is not None and tr[0] in self._shapes:
            R = self._shifted(g, *self._shapes[tr[0]], np.asarray(tr[1], float), box)
            if R is not None:
                return PS(1, None, [(1.0, np.arange(1), R)]), 0
        inner = tuple(None if b is None else [x for x in b if lo < x < hi] for b, (lo, hi) in zip(breaks, box))
        R, u = T.fit_tensor(g, box, N, tol=self.tol, breaks=inner, jac_max=self.jac_max)
        R = canonical(R, self._reps)
        if tr is not None and not u and len(self._shapes) < MAX_SHAPES:
            rel = _samples(len(box)) * np.array([b - a for a, b in box])
            vals = np.asarray(g(*[rel[:, k] + box[k][0] for k in range(len(box))]), float).ravel()
            if np.max(np.abs(vals), initial=0.0) > 0:
                self._shapes[tr[0]] = (R, np.asarray(tr[1], float), rel, vals)
        return PS(N, None, [(1.0, np.arange(N), R)]), u

    def _shifted(self, g, R0, origin0, rel, vals0, origin, box):
        """The fit R0 shifted by origin - origin0, when g equals the shifted function at the sample points (to 1e-12
        of their maximum) and the shifted box is this function's box; else None (the caller fits)."""
        shift = origin - origin0
        bases = [T.AxisBasis(np.asarray(b.breaks) + s, b.n) for b, s in zip(R0.bases, shift)]
        width = max(b - a for a, b in box)
        if any(abs(nb.breaks[0] - lo) > 1e-12 * width or abs(nb.breaks[-1] - hi) > 1e-12 * width
               for nb, (lo, hi) in zip(bases, box)):
            return None
        vals = np.asarray(g(*[rel[:, k] + origin[k] for k in range(len(box))]), float).ravel()
        if np.max(np.abs(vals - vals0)) > 1e-12 * np.max(np.abs(vals0)):
            return None
        bases = [T.AxisBasis([lo] + list(nb.breaks[1:-1]) + [hi], nb.n) for nb, (lo, hi) in zip(bases, box)]
        return T.TensorSeries(bases, R0.C, R0.err)

    def _tensor(self, R):
        return to_tensor(R, self.domain.reference) if isinstance(R, PS) else R

    def to_tensor(self, R):
        return super().to_tensor(self._tensor(R))

    def _ours(self, *Rs):
        return all(isinstance(R, (PS, T.TensorSeries)) for R in Rs) and any(isinstance(R, PS) for R in Rs)

    def lincomb(self, terms):
        if self._ours(*[R for _, R in terms]):
            return lincomb(terms)
        return super().lincomb([(a, self._tensor(R)) for a, R in terms])

    def combine(self, R, A):
        if isinstance(R, PS):
            out = combine(R, A)
            return out if out is not None else self._zero(np.shape(A)[1])
        return super().combine(R, A)

    def colsum(self, R):
        return self.combine(R, np.ones((R.N, 1))) if isinstance(R, PS) else super().colsum(R)

    def take(self, R, idx):
        return take(R, idx) if isinstance(R, PS) else super().take(R, idx)

    def hstack(self, Rs):
        if self._ours(*Rs):
            return hstack(Rs, self._zero)
        return super().hstack([self._tensor(R) for R in Rs])

    def multiply(self, A, B):
        if self._ours(A, B) and A.N == 1 and B.N == 1:
            out = multiply(A, B)
            return out if out is not None else self._zero(1)
        return super().multiply(self._tensor(A), self._tensor(B))

    def gram(self, A, B):                     # self.c: the constant density (plan 064), as in TensorBackend
        if self._ours(A, B):
            return T._times(self.c, gram(A, B, self.jac, memo=self._selfgram))
        return super().gram(self._tensor(A), self._tensor(B))

    def sqnorms(self, A):
        return T._times(self.c, sqnorms(A, self.jac, self._selfgram)) if isinstance(A, PS) else super().sqnorms(A)

    def integral(self, R, over=None):
        if isinstance(R, PS) and over is None:
            return T._times(self.c, integral(R, self.jac))
        return super().integral(self._tensor(R), over)

    def box_integral(self, R, S):
        if isinstance(R, PS):
            return integral(R, self.jac, S, self.domain)
        return super().box_integral(R, S)

    def lp_norm(self, R, p):
        return super().lp_norm(self._tensor(R), p)

    def evaluate(self, R, *X):
        if not isinstance(R, PS):
            return super().evaluate(R, *X)
        shape = np.broadcast(*X).shape
        xi = self.domain.from_cartesian(*[np.broadcast_to(np.asarray(a, float), shape) for a in X])
        return evaluate(R, xi, self.domain.reference)

    def grid_values(self, R, axes_points):
        if isinstance(R, PS):
            return grid_values(R, axes_points, self.domain.reference)
        return super().grid_values(R, axes_points)

    def error(self, R):
        return R.err if self.c == 1.0 else math.sqrt(self.c) * R.err      # R.err in L2(dx); the space's norm

    def pairings(self, R, measures):
        """(N, n) sparse: int phi_i dmu_j, exact for points, segments and coordinate boxes (the pairings below)."""
        from .local import pairings
        return pairings(as_ps(R), measures, self.domain, self.jac)

    def breaks(self, R):
        return breaks(R) if isinstance(R, PS) else super().breaks(R)

    def describe(self, R):
        if not isinstance(R, PS):
            return super().describe(R)
        return dict(kind="support-local tensor Legendre", pieces=sum(int(np.prod([b.P for b in Q.bases]))
                                                                     for _, _, Q in R.patches),
                    degree=max(b.n - 1 for _, _, Q in R.patches for b in Q.bases), size=R.size,
                    patches=len(R.patches), base="a tensor part on the whole box" if R.base is not None else "none")

    def summary(self, R):
        if not isinstance(R, PS):
            return super().summary(R)
        return (f"support-local tensor Legendre: {len(R.patches)} patch(es)"
                + (" and a tensor part" if R.base is not None else "") + f", {R.size} coefficients")


# ---------------------------------------------------------------- from local_pairings.py (plan 063)
PAIRS = 1 << 17                 # (segment, patch) pairs per vectorised block


def _parts(A):
    """(c, cols, R, offset, type) for every patch of A, the tensor part as one more (on the whole box)."""
    out = []
    if A.base is not None:
        out.append((1.0, np.arange(A.N), A.base, np.zeros(A.d), -1))
    reps = {}
    for p, ((c, cols, R), t) in enumerate(zip(A.patches, A.types)):
        first = reps.setdefault(t, p)
        out.append((c, cols, A.patches[first][2], A.lo[p] - A.lo[first], int(t)))
    return out


def segments(A, P, E, w, top):
    """(A.N, m) sparse: sum over the atoms a of segment m: w_a int_0^1 f(P_a + t E_a) |E_a| dt for the patches of A."""
    m, d = P.shape
    rows, cols, vals = [], [], []
    parts = _parts(A)
    groups = {}
    for k, part in enumerate(parts):
        groups.setdefault(part[4] if part[4] >= 0 else -1 - k, []).append(k)
    seg_lo, seg_hi = np.minimum(P, P + E), np.maximum(P, P + E)
    length = np.linalg.norm(E, axis=1)
    for members in groups.values():
        R = parts[members[0]][2]
        rlo = np.array([b.breaks[0] for b in R.bases])
        rhi = np.array([b.breaks[-1] for b in R.bases])
        off = np.array([parts[k][3] for k in members])                    # (q, d) offsets of the members
        lo, hi = rlo + off, rhi + off
        si, qi = [], []
        step = max(1, (1 << 22) // max(len(members), 1))
        for j in range(0, m, step):                                         # candidates by bounding boxes
            cand = np.all((seg_lo[j:j + step, None] <= hi[None]) & (seg_hi[j:j + step, None] >= lo[None]), axis=2)
            a_, b_ = np.nonzero(cand)
            si.append(a_ + j)
            qi.append(b_)
        si, qi = np.concatenate(si), np.concatenate(qi)
        coef = np.array([parts[k][0] for k in members])
        single = all(len(parts[k][1]) == 1 for k in members)
        col1 = np.array([parts[k][1][0] for k in members]) if single else None
        q = int(np.ceil((sum(b.n - 1 for b in R.bases) + 1) / 2))
        g, gw = gauss(max(q, 1))
        brk = [np.asarray(b.breaks) for b in R.bases]
        for s0 in range(0, len(si), PAIRS):
            s, mq = si[s0:s0 + PAIRS], qi[s0:s0 + PAIRS]
            a = P[s] - off[mq]                                              # the segment in the type's frame
            e = E[s]
            t0, t1 = np.zeros(len(s)), np.ones(len(s))
            keep = np.ones(len(s), bool)
            for k in range(d):
                par = np.abs(e[:, k]) <= 1e-15 * np.maximum(length[s], 1e-300)
                closed = rhi[k] + off[mq, k] >= top[k]
                inside = (a[:, k] >= rlo[k]) & np.where(closed, a[:, k] <= rhi[k], a[:, k] < rhi[k])
                keep &= ~par | inside
                with np.errstate(divide="ignore", invalid="ignore"):
                    u, v = (rlo[k] - a[:, k]) / e[:, k], (rhi[k] - a[:, k]) / e[:, k]
                t0 = np.where(par, t0, np.maximum(t0, np.minimum(u, v)))
                t1 = np.where(par, t1, np.minimum(t1, np.maximum(u, v)))
            keep &= t1 > t0
            if not keep.any():
                continue
            s, mq, a, e, t0, t1 = s[keep], mq[keep], a[keep], e[keep], t0[keep], t1[keep]
            ts = [t0[:, None], t1[:, None]]
            for k in range(d):
                with np.errstate(divide="ignore", invalid="ignore"):
                    c = (brk[k][None, 1:-1] - a[:, k:k + 1]) / e[:, k:k + 1]
                ts.append(np.where(np.isfinite(c), np.clip(c, t0[:, None], t1[:, None]), t0[:, None]))
            ts = np.sort(np.concatenate(ts, axis=1), axis=1)                  # (pairs, J): parts between crossings
            mid, half = 0.5 * (ts[:, 1:] + ts[:, :-1]), 0.5 * (ts[:, 1:] - ts[:, :-1])
            tt = mid[..., None] + half[..., None] * g                          # (pairs, J-1, q)
            X = [a[:, k, None, None] + tt * e[:, k, None, None] for k in range(d)]
            V = T.evaluate_points(R, [x.ravel() for x in X]).reshape(tt.shape + (R.N,))
            V = np.nan_to_num(V)                                               # points on the far face (rounding)
            integ = np.einsum("pjqn,q,pj->pn", V, gw, half)                    # int over [t0, t1] of f(a + t e) dt
            scale = coef[mq] * length[s] * w[s]
            if single:                                                         # one column per patch (a basis)
                rows.append(col1[mq])
                cols.append(s)
                vals.append(scale * integ[:, 0])
                continue
            for k_, member in enumerate(members):
                sel = mq == k_
                if sel.any():
                    ccols = parts[member][1]
                    rows.append(np.repeat(ccols[None, :], sel.sum(), axis=0).ravel())
                    cols.append(np.repeat(s[sel], len(ccols)))
                    vals.append((integ[sel] * scale[sel, None]).ravel())
    if not vals:
        return sp.csr_matrix((A.N, m))
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(A.N, m)).tocsr()


def pairings(A, measures, domain, jac):
    """(A.N, n) sparse: int phi_i dmu_j for the columns of A (a PS) and n measures with point, segment and axis-aligned
    patch or box atoms (atoms.atoms_from); other atoms are refused (the caller falls back to the space's functionals)."""
    from ...core.errors import Unsupported
    from ...domains import Box
    from ..rkhs.atoms import Kind, atoms_from
    from .local import evaluate, integral
    at = atoms_from(measures, domain, lambda X: X)
    d = A.d
    P, E1, E2 = at.P[:, :d], at.E1[:, :d], at.E2[:, :d]
    top = [hi for _, hi in domain.reference]
    owner = sp.csr_matrix((np.ones(len(at.kind)), (np.arange(len(at.kind)), at.owner)), shape=(len(at.kind), at.n))
    per_atom = sp.lil_matrix((A.N, len(at.kind)))
    seg = np.nonzero(at.kind == Kind.SEGMENT)[0]
    out = sp.csr_matrix((A.N, at.n))
    if len(seg):
        out = out + segments(A, P[seg], E1[seg], at.weight[seg], top) @ owner[seg]
    pts = np.nonzero(at.kind == Kind.POINT)[0]
    if len(pts):
        V = evaluate(A, [P[pts, k] for k in range(d)], domain.reference) * at.weight[pts, None]     # (points, N)
        out = out + sp.csr_matrix(V.T) @ owner[pts]
    for q in np.nonzero(at.kind == Kind.PATCH)[0]:
        E = np.stack([E1[q], E2[q]])
        if d != 2 or np.count_nonzero(np.abs(E) > 1e-15 * np.abs(E).max()) != 2 or np.any(np.abs(E).sum(axis=0) == 0):
            raise Unsupported("exact pairings with patches that are not coordinate rectangles")
        lo, hi = np.minimum(P[q], P[q] + E1[q] + E2[q]), np.maximum(P[q], P[q] + E1[q] + E2[q])
        v = integral(A, jac, Box(*[(float(a), float(b)) for a, b in zip(lo, hi)]), domain)
        if v is None:
            raise Unsupported("exact box integrals need Cartesian natural coordinates")
        per_atom[:, q] = (at.weight[q] * v)[:, None]
    other = np.nonzero(~np.isin(at.kind, [Kind.SEGMENT, Kind.POINT, Kind.PATCH]))[0]
    if len(other):
        raise Unsupported(f"exact pairings with atoms of kind {Kind(at.kind[other[0]]).name}")
    return (out + per_atom.tocsr() @ owner).tocsr()
