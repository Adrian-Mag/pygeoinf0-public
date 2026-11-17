"""A cell-constant realisation of L2 on a partition of a box into boxes (plan 020 P2; experiments/EXP-035-mineral-
exploration/NOTES.md: mesh-based codes such as SimPEG and pyGIMLi work with cell values).

    Cells.tensor(nodes_x, nodes_y[, nodes_z])     the cells of a tensor mesh (x fastest, then y, then z): a partition by
                                                  construction
    Cells(boxes)                                  any list of boxes; their volumes must add up to the domain's and
                                                  their interiors are checked pairwise disjoint (overlaps refused:
                                                  NotATiling); check=False: the caller (a mesher) vouches for it
    Cells(P)                                      the pieces of a domains.Partition P of a box into boxes (its leaves,
                                                  in P's order; plan 069 P4, review A11 and B9): the tiling is P's,
                                                  decided there, and C.partition is P
    pg.L2(D, realisation=cells)                   L2 of Lebesgue measure on the box D, realised on the cells

An element is realised by its averages on the cells: its orthogonal projection onto the cell-constant functions. That
is exact for cell-constant functions and an approximation for others, whose distance to the projection is estimated by
Gauss quadrature on each cell and kept as the realisation's error (so the space stays L2, not its truncation). Inner
products are sum_i vol_i f_i g_i; pg.port.Coordinates.cells(space) gives the cell values as numbers, with diagonal maps.

Complex cell values (plan 020 P3b; EXP-035 COMPLEX.md, the hybrid): pg.Complexified(M) of a cell-realised L2 stores
complex averages natively (ComplexCellsBackend: one complex array, one evaluation of a complex formula); its real inner
product is Re sum_i vol_i f_i conj(g_i). The real backend refuses complex values instead of dropping the imaginary part.

The tiling is a claim (plan 065 A11; cells.claims, the backend's claims): THEOREM for a tensor mesh, the Partition's own
claim (DERIVED, decided exactly from the pieces' descriptions) for Cells(P), CHECKED for boxes
checked on the grid of their coordinates (at most PAIR_BUDGET elementary cells) or by a sweep (pairs of cells whose
extents meet along the sweep axis, at most PAIR_BUDGET of them), ASSERTED by the caller for check=False or a check
over its budget. Without a tiling the realised inner product is not an L2
projection; the volume sum alone does not show one (overlaps can make up for gaps).

Limits: Lebesgue measure only; boxes only; integrals of an element along a path or over a set that cuts cells are
computed by quadrature on a discontinuous integrand (not exact across cell faces).

From cellwise.py (merged here by plan 063):
Integrals of piecewise-Legendre functions over the cells of a partition, and the breaks a realisation needs (plan
059). cell_integrals(p, edges) = (int over each cell of each column of p): differences of the exact antiderivative
(Legendre integration, piece by piece) at the edges, vectorised; it replaces the Gram of p against the cells'
indicators (an N x m family), with the same exactness. essential_breaks(p) drops the breaks where the two
neighbouring pieces are one polynomial (to rounding): a realisation built on a family's common partition carries the
other members' breaks.
"""


import hashlib
import itertools

import numpy as np
from numpy.polynomial import legendre as npleg

from ..core.claims import ASSERTED, CHECKED, THEOREM, Claim, Claims
from ..quadrature import gauss
from .backends import Backend
from .series.legendre import EPS, restrict

QUAD = 3                         # Gauss points per axis and cell for fitting functions that are not cell-constant
PAIR_BUDGET = 20_000_000         # candidate pairs the overlap sweep checks before it leaves the tiling ASSERTED
OVERLAP_TOL = 1e-12              # relative to the largest coordinate (as Cells.on and the backend's domain check)


def _grid_cover(lo, hi, tol):
    """The overlap test on the grid of the boxes' own coordinates (merged to tol), when it has at most PAIR_BUDGET
    elementary cells: each box covers a block of them, and two interiors overlap exactly when an elementary cell is
    covered twice (counted by a difference array). False: none overlap; True: some do; None: the grid is too large."""
    n, d = lo.shape
    axes = []
    for k in range(d):
        v = np.unique(np.r_[lo[:, k], hi[:, k]])
        axes.append(v[np.r_[True, np.diff(v) > tol]])
    if float(np.prod([len(a) - 1 for a in axes])) > PAIR_BUDGET:
        return None
    a_ = [np.searchsorted(axes[k], lo[:, k] + tol) - 1 for k in range(d)]       # first elementary index
    b_ = [np.searchsorted(axes[k], hi[:, k] + tol) - 1 for k in range(d)]       # one past the last
    diff = np.zeros([len(a) for a in axes], np.int64)
    for corner in itertools.product((0, 1), repeat=d):
        idx = tuple(b_[k] if c else a_[k] for k, c in enumerate(corner))
        np.add.at(diff, idx, (-1) ** sum(corner))
    for k in range(d):
        diff = np.cumsum(diff, axis=k)
    return bool(diff.max(initial=0) > 1)


def _overlap_check(lo, hi):
    """None when no two boxes have overlapping interiors (to OVERLAP_TOL), else (i, j) of one overlapping pair; or
    the string "budget" with the count when the candidate pairs exceed PAIR_BUDGET. A mesh-like list is decided on
    the grid of its coordinates (_grid_cover); then, or else, a sweep along the axis with the fewest candidates finds
    the pair: sorted by lower ends, the candidates of box i are the later boxes starting before i ends."""
    n, d = lo.shape
    tol = OVERLAP_TOL * max(1.0, float(np.abs(np.r_[lo.ravel(), hi.ravel()]).max(initial=0.0)))
    grid = _grid_cover(lo, hi, tol)
    if grid is False:
        return None
    best = None
    for k in range(d):
        order = np.argsort(lo[:, k], kind="stable")
        ends = np.searchsorted(lo[order, k], hi[order, k] - tol, side="left")
        counts = np.maximum(ends - np.arange(1, n + 1), 0)
        total = int(counts.sum())
        if best is None or total < best[0]:
            best = (total, order, counts)
    total, order, counts = best
    if total > PAIR_BUDGET:
        return ("overlap", None) if grid else ("budget", total)
    if total == 0:
        return None
    I = np.repeat(np.arange(n), counts)
    J = I + 1 + (np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts))
    for s in range(0, total, 1 << 20):
        a, b = order[I[s:s + (1 << 20)]], order[J[s:s + (1 << 20)]]
        ov = np.all(np.minimum(hi[a], hi[b]) - np.maximum(lo[a], lo[b]) > tol, axis=1)
        if ov.any():
            q = int(np.argmax(ov))
            return int(a[q]), int(b[q])
    return None


def partition_boxes(P):
    """(names, boxes) of the leaves of a Partition P of a box into boxes, in P's order; refused (Unsupported) for a
    partition of another kind (the cells of this realisation are boxes)."""
    from ..core.errors import Unsupported
    Q = P if P.is_flat else P.flat()
    if Q.kind != "box":
        raise Unsupported(f"a cell realisation is built on a Partition of a box into boxes; {P!r} is a partition of "
                          f"another kind ({Q.kind}): pg.Partition(pg.Box(...), {{name: pg.Box(...), ...}})")
    return tuple(Q.order), [Q.pieces[n] for n in Q.order]


class Cells:
    """A partition of a box into boxes (module docstring): the value of the realisation= argument. boxes: a list of
    boxes, or a domains.Partition of a box into boxes (then the cells are its leaves, with its tiling claim)."""

    def __init__(self, boxes, nodes=None, check=True, _tiling=None):
        from ..core.errors import NotATiling
        from ..domains.partition import Partition
        self.partition, self.names = None, None
        if isinstance(boxes, Partition):
            P = boxes
            self.names, boxes = partition_boxes(P)
            tc = P.claims.find("tiling")
            _tiling = (tc.status, f"the pieces of a Partition ({P!r}), {tc.reason}")
            self.partition = P
        lo = np.array([[s.a for s in b.sides] for b in boxes], float)
        hi = np.array([[s.b for s in b.sides] for b in boxes], float)
        if lo.ndim != 2 or np.any(hi <= lo):
            raise ValueError("cells are boxes of one dimension with positive sides")
        self.lo, self.hi, self.nodes = lo, hi, nodes
        if _tiling is None and not check:
            _tiling = (ASSERTED, "by the caller (check=False: typically a mesher vouches for it); not checked")
        if _tiling is None:
            found = _overlap_check(lo, hi)
            if found is None:
                _tiling = (CHECKED, f"no two cells overlap: interiors pairwise disjoint to {OVERLAP_TOL:g} relative "
                                    "(on the grid of their coordinates, or a sweep over the pairs whose extents meet)")
            elif found[0] == "budget":
                _tiling = (ASSERTED, f"by the caller: the overlap check is over its budget ({found[1]} candidate pairs "
                                     f"> PAIR_BUDGET = {PAIR_BUDGET}), so it was not run")
            else:
                i, j = found
                which = "two cells overlap (found on the grid of their coordinates)" if i == "overlap" else \
                    f"cells {i} and {j} overlap ({list(zip(lo[i], hi[i]))} and {list(zip(lo[j], hi[j]))})"
                raise NotATiling(f"{which}: "
                                 "the cell realisation needs a partition, or its inner product is not an L2 projection",
                                 needs=("cells that meet only on their faces",))
        self._tiling = _tiling
        self.claims = Claims([Claim("tiling: interiors pairwise disjoint", _tiling[0], _tiling[1], key="tiling")])
        self.volumes = np.prod(hi - lo, axis=1)
        self.n, self.dim = lo.shape
        digest = hashlib.sha1(np.ascontiguousarray(np.r_[lo.ravel(), hi.ravel()]).tobytes()).hexdigest()[:10]
        self.name = f"cells({self.n}, {digest})"

    @classmethod
    def tensor(cls, *nodes):
        from ..domains import Box
        nodes = [np.asarray(n, float) for n in nodes]
        edges = [list(itertools.pairwise(n)) for n in nodes]
        grids = np.meshgrid(*[np.arange(len(e)) for e in edges], indexing="ij")
        order = np.stack([g.ravel(order="F") for g in grids], axis=1)            # x fastest
        return cls([Box([edges[k][i[k]] for k in range(len(nodes))]) for i in order], nodes=nodes,
                   _tiling=(THEOREM, "the cells of a tensor mesh (strictly increasing nodes per axis): a partition by "
                                     "construction"))

    def on(self, D):
        """(the cells inside the box D, their indices) when they tile D, else None: a partition of D by whole cells, in
        this partition's order."""
        a, b = np.array([s.a for s in D.sides]), np.array([s.b for s in D.sides])
        tol = 1e-12 * max(1.0, float(np.abs(np.r_[a, b]).max()))
        idx = np.nonzero(np.all((self.lo >= a - tol) & (self.hi <= b + tol), axis=1))[0]
        if not len(idx) or abs(self.volumes[idx].sum() - np.prod(b - a)) > 1e-10 * np.prod(b - a):
            return None
        from ..domains import Box
        nodes = None if self.nodes is None else [n[(n >= lo - tol) & (n <= hi + tol)] for n, lo, hi in zip(self.nodes, a, b)]
        return Cells([Box(list(zip(self.lo[i], self.hi[i]))) for i in idx], nodes=nodes,
                     _tiling=(self._tiling[0], f"cells of {self.name}: {self._tiling[1]}")), idx

    def locate(self, X):
        """The index of the cell containing each point of X (shape (m, d)); -1 outside the cells."""
        X = np.atleast_2d(np.asarray(X, float))
        if self.nodes is not None:
            idx, inside = np.zeros(len(X), int), np.ones(len(X), bool)
            stride = 1
            for k, nd in enumerate(self.nodes):
                j = np.clip(np.searchsorted(nd, X[:, k], side="right") - 1, 0, len(nd) - 2)
                inside &= (X[:, k] >= nd[0]) & (X[:, k] <= nd[-1])
                idx += j * stride
                stride *= len(nd) - 1
            return np.where(inside, idx, -1)
        out = np.full(len(X), -1)
        for s in range(0, len(X), 256):
            P = X[s:s + 256, None, :]
            hit = np.all((P >= self.lo[None]) & (P <= self.hi[None]), axis=2)
            out[s:s + 256] = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        return out

    def __repr__(self):
        return self.name


class CellValues:
    """A realisation: V (cells x N) of cell averages and err (N,), the estimated distance to the realised functions."""

    __slots__ = ("V", "err")

    def __init__(self, V, err=None):
        V = np.asarray(V)
        self.V = V.astype(complex if np.iscomplexobj(V) else float, copy=False)
        if self.V.ndim == 1:
            self.V = self.V[:, None]
        self.err = np.zeros(self.V.shape[1]) if err is None else np.asarray(err, float)

    @property
    def N(self):
        return self.V.shape[1]


def _rule_gap():
    """The largest gap of the 1-D QUAD-point Gauss rule on [0, 1], the cell's ends included."""
    t = 0.5 * gauss(QUAD)[0] + 0.5
    return float(np.diff(np.concatenate([[0.0], t, [1.0]])).max())


def _stated(fun, C, m):
    """Tell a recording callable (core/sampling.Recorder) the spacing of the rule: its calls are not product grids."""
    from ..core.sampling import Recorder
    if isinstance(fun, Recorder):
        fun.stated = float((C.hi - C.lo).max()) * _rule_gap() / m


class CellsBackend(Backend):
    """L2 of Lebesgue measure on a box, realised by cell averages on a partition (module docstring)."""

    projection_note = "realised by its averages on the cells (this space's realisation is cell-constant)"
    CAPABILITIES = ("cell_values",)

    def __init__(self, measure, tol, cells):
        D = measure.set                                   # Lebesgue measure on a box: checked by registry.choose
        if cells.dim != D.dim:
            raise ValueError(f"the cells are {cells.dim}-dimensional, the domain {D.dim}-dimensional")
        a, b = np.array([s.a for s in D.sides]), np.array([s.b for s in D.sides])
        if np.any(cells.lo < a - 1e-12 * np.abs(a).max()) or np.any(cells.hi > b + 1e-12 * np.abs(b).max()):
            raise ValueError("cells outside the domain")
        if abs(cells.volumes.sum() - np.prod(b - a)) > 1e-10 * np.prod(b - a):
            raise ValueError("the cells' volumes do not add up to the domain's: not a partition")
        self.measure, self.tol, self.domain, self.cells, self.dim = measure, tol, D, cells, D.dim
        t = cells.claims.find("tiling")
        self.claims = Claims([Claim(f"the cells tile {D!r}", t.status,
                                    f"{t.reason}; their volumes add up to the domain's (checked, 1e-10 relative) and "
                                    "they lie inside it", key="tiling")])
        self.realisation_name = cells.name
        t, w = gauss(QUAD)
        grid = np.meshgrid(*[t] * self.dim, indexing="ij")
        self._t = np.stack([g.ravel() for g in grid], axis=1) * 0.5 + 0.5          # (q^d, d) on [0, 1]^d
        self._w = np.prod(np.meshgrid(*[w * 0.5] * self.dim, indexing="ij"), axis=0).ravel()

    def same_realisation(self, other):
        return isinstance(other, CellsBackend)

    def restricted(self, R, target):
        """The values on target's cells, when they are cells of this partition."""
        found = self.cells.on(target.domain)
        if found is None or found[0].name != target.cells.name:
            return None
        return CellValues(R.V[found[1]], R.err)

    def glued(self, parts):
        """The cells' values in place, zero on cells of no part; None unless each part's cells are cells of this partition."""
        if not all(isinstance(B, CellsBackend) for B, _ in parts):
            return None
        V, seen = None, np.zeros(self.cells.n, bool)
        for B, R in parts:
            found = self.cells.on(B.domain)
            if found is None or found[0].name != B.cells.name or seen[found[1]].any():
                return None
            if V is None:
                V = np.zeros((self.cells.n, R.N), R.V.dtype)
            V[found[1]], seen[found[1]] = R.V, True
        return CellValues(V, np.sqrt(sum(R.err ** 2 for _, R in parts)))

    def probe(self):
        c = 0.5 * (self.cells.lo + self.cells.hi)[:3]
        return tuple(c[:, k] for k in range(self.dim))

    dtype = float

    def fit(self, fun, N=1, breaks=None, m=1):
        """Cell averages by the QUAD^d Gauss rule on each cell, or on each of m^d sub-cells (fit_at)."""
        C = self.cells
        t, wq = self._t, self._w
        if m > 1:
            J = np.stack([g.ravel() for g in np.meshgrid(*[np.arange(m)] * self.dim, indexing="ij")], axis=1)
            t = ((J[:, None, :] + t[None]) / m).reshape(-1, self.dim)
            wq = np.tile(wq, len(J)) / len(J)
        _stated(fun, C, m)
        means, var = np.zeros((C.n, N), self.dtype), np.zeros((C.n, N))
        step = max(1, 4096 // (m ** self.dim))
        for s in range(0, C.n, step):
            lo, hi = C.lo[s:s + step], C.hi[s:s + step]
            X = lo[:, None, :] + (hi - lo)[:, None, :] * t[None]                      # (m, q^d, d)
            V = self._values(fun(*[X[..., k] for k in range(self.dim)]))
            V = np.broadcast_to(V, X.shape[:2] + ((N,) if V.ndim == 3 or N > 1 else ())).reshape(X.shape[:2] + (N,))
            mu = np.einsum("q,mqn->mn", wq, V)
            means[s:s + step] = mu
            var[s:s + step] = np.einsum("q,mqn->mn", wq, np.abs(V - mu[:, None, :]) ** 2)
        err = np.sqrt(np.maximum(C.volumes @ var, 0.0))                              # ||f - P f||, by quadrature
        return CellValues(means, err), 0

    def fit_at(self, fun, N=1, breaks=None, w=None):
        """fit at a declared resolution w (core/fits.py): the rule on m^d sub-cells, its nodes <= w / 2 apart."""
        g1 = _rule_gap()
        width = float((self.cells.hi - self.cells.lo).max())
        return self.fit(fun, N, breaks, m=max(1, int(np.ceil(width * g1 / (0.5 * w)))))

    def _values(self, V):
        V = np.asarray(V)
        if np.iscomplexobj(V):
            if np.any(V.imag != 0):
                from ..core.errors import Refused
                raise Refused("complex values in a real space: the imaginary part would be dropped. Use "
                              "pg.Complexified(M) for complex-valued functions")
            V = V.real
        return np.asarray(V, float)

    def lincomb(self, terms):
        return CellValues(sum(a * R.V for a, R in terms), sum(abs(a) * R.err for a, R in terms))

    def combine(self, R, A):
        A = np.asarray(A, float)
        return CellValues(R.V @ A, np.abs(A).T @ R.err)

    def take(self, R, idx):
        idx = np.atleast_1d(idx)
        return CellValues(R.V[:, idx], R.err[idx])

    def hstack(self, Rs):
        return CellValues(np.hstack([R.V for R in Rs]), np.concatenate([R.err for R in Rs]))

    def gram(self, A, B):
        return A.V.T @ (self.cells.volumes[:, None] * B.V)

    def sqnorms(self, A):
        return self.cells.volumes @ (A.V ** 2)

    def integral(self, R, over=None):
        if over is not None:
            self._unsupported("integral over a subset (cells may be cut)")
        return self.cells.volumes @ R.V

    def evaluate(self, R, *X):
        shape = np.broadcast(*X).shape
        P = np.stack([np.broadcast_to(np.asarray(x, float), shape).ravel() for x in X], axis=1)
        k = self.cells.locate(P)
        out = np.where(k[:, None] >= 0, R.V[np.maximum(k, 0)], np.nan)
        return out.reshape(shape + (R.N,))

    def error(self, R):
        return R.err

    def describe(self, R):
        return {"kind": "cell averages", "cells": self.cells.n, "members": R.N}

    def summary(self, R):
        return f"cell averages on {self.cells.n} cells"


class ComplexCellsBackend(CellsBackend):
    """Complex cell averages, stored natively (module docstring): the backend of pg.Complexified(M) for a cell-realised
    L2. Inner products are real, Re sum vol f conj(g) (the real structure); cgram gives the complex ones."""

    dtype = complex
    projection_note = "realised by its complex averages on the cells (this space's realisation is cell-constant)"

    def __init__(self, measure, tol, cells):
        super().__init__(measure, tol, cells)
        self.realisation_name = f"complex {cells.name}"

    def _values(self, V):
        return np.asarray(V, complex)

    def lincomb(self, terms):
        return CellValues(sum(complex(a) * R.V for a, R in terms), sum(abs(a) * R.err for a, R in terms))

    def cgram(self, A, B):
        """<a_i, b_j> = sum_k vol_k a_ik conj(b_jk): linear in a, conjugate-linear in b."""
        return A.V.T @ (self.cells.volumes[:, None] * B.V.conj())

    def gram(self, A, B):
        return self.cgram(A, B).real

    def sqnorms(self, A):
        return self.cells.volumes @ np.abs(A.V) ** 2

    def describe(self, R):
        return {"kind": "complex cell averages", "cells": self.cells.n, "members": R.N}

    def summary(self, R):
        return f"complex cell averages on {self.cells.n} cells"


# ---------------------------------------------------------------- from cellwise.py (plan 063)
def cumulative(p, x):
    """int_{a}^{x} of every column of p at the sorted points x (shape (len(x), N)), a = p's first break."""
    x = np.asarray(x, float)
    br = p.breaks
    jac = 0.5 * np.diff(br)
    F = [npleg.legint(C, lbnd=-1, axis=0) for C in p.coefs]           # zero at t = -1
    full = np.array([jac[k] * F[k].sum(axis=0) for k in range(p.npieces)])   # legval at t = 1: the sum
    start = np.vstack([np.zeros(p.N), np.cumsum(full, axis=0)[:-1]])
    idx = np.clip(np.searchsorted(br, x, side="right") - 1, 0, p.npieces - 1)
    out = np.empty((len(x), p.N))
    for k in np.unique(idx):
        sel = idx == k
        t = np.clip((2.0 * x[sel] - br[k] - br[k + 1]) / (br[k + 1] - br[k]), -1.0, 1.0)
        out[sel] = start[k] + jac[k] * npleg.legval(t, F[k]).T.reshape(-1, p.N)
    return out


def cell_integrals(p, edges):
    """(m, N): the integral of each column of p over each cell [edges_i, edges_i+1] (edges inside p's interval)."""
    return np.diff(cumulative(p, edges), axis=0)


def essential_breaks(p, rtol=1e-13):
    """p's breaks less those where the neighbouring pieces agree as polynomials (every column, to rtol)."""
    br = p.breaks
    keep = [br[0]]
    for k in range(1, p.npieces):
        L, R = p.coefs[k - 1], p.coefs[k]
        n = max(L.shape[0], R.shape[0])
        Lp = np.vstack([L, np.zeros((n - L.shape[0], L.shape[1]))])
        Rp = np.vstack([R, np.zeros((n - R.shape[0], R.shape[1]))])
        ext = restrict(Lp, br[k - 1], br[k], br[k], br[k + 1])          # the left polynomial continued
        scale = max(float(np.abs(Lp).sum()), float(np.abs(Rp).sum()), 1.0)
        if float(np.abs(ext - Rp).max()) > max(rtol, 64 * n * EPS) * scale:
            keep.append(br[k])
    keep.append(br[-1])
    return np.array(keep)
