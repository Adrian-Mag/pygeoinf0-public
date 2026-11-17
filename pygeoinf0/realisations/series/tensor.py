"""Tensor-product realisation on the reference box of a domain (boxes, disks, cylinders, balls).

A domain has natural coordinates xi in a reference box prod_k [lo_k, hi_k] and Lebesgue measure equal to
prod_k J_k(xi_k) dxi with one-dimensional weights J_k (1, r, r^2, sin theta). A function f is realised as
f(Phi(xi)) in a product basis: along each axis a piecewise Legendre basis (breakpoints, n points per piece).
The coefficients form an array C of shape (S_1, ..., S_d, N) for N functions (S_k = pieces_k * n_k).

    fit      adaptive per axis: raise the number of points per piece, then split pieces whose tails do not decay
    algebra  exact: common refinement per axis (exact re-expansion), then add / contract
    measure  <f, g> = C_f x_1 M_1 ... x_d M_d . C_g with per-axis block-diagonal mass matrices
             M_k = int phi_a phi_b J_k, by Gauss-Legendre quadrature: exact for J_k = 1, r, r^2, and to rounding
             for sin(theta) (12 extra points; its Legendre coefficients fall below 1e-16 by degree about 20)
    values   per-axis Legendre values gathered per piece; fast on tensor grids of natural coordinates

Every per-axis operator is block diagonal (one n x n block per piece) and is applied block by block, so
memory stays proportional to the coefficient array. Smooth functions need a few dozen points per axis.
Localised functions (narrow kernels) are better as products of one-dimensional factors (separated.py).
"""
import math
import warnings
from functools import lru_cache

import numpy as np
from numpy.polynomial import legendre as npleg

from . import legendre as L
from . import separated as SEP
from ...core.errors import Unsupported
from ..backends import Backend
from .legendre import EPS, TINY, analysis
from ...quadrature import gauss, jacobian_nodes

LADDERS = {1: (33, 65, 129), 2: (17, 33, 65), 3: (9, 17, 33)}
FOURIER_MAX = 513               # largest number of Fourier modes (2K + 1) on a periodic axis
MAX_VALUES = 1 << 23            # values per evaluation array (64 MB of float64)


def _weight(name):
    """(the Jacobian named name, the extra Gauss points its integrals take): domains.JACOBIANS and
    quadrature.jacobian_nodes (plan 069 P2: the Gauss counts left the domains)."""
    from ...domains import JACOBIANS
    return JACOBIANS[name], jacobian_nodes(name)


# ------------------------------------------------------------------------------------------ axis bases

class AxisBasis:
    """Piecewise Legendre basis on one axis: breakpoints and n points (degree n - 1) per piece."""

    __slots__ = ("breaks", "n", "P", "S", "key")

    def __init__(self, breaks, n):
        self.breaks = tuple(float(b) for b in breaks)
        self.n, self.P = int(n), len(self.breaks) - 1
        self.S = self.P * self.n
        self.key = (self.breaks, self.n)

    def __eq__(self, other):
        return isinstance(other, AxisBasis) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def nodes(self):
        return _nodes(self.key)[0]

    def weights(self):
        return _nodes(self.key)[1]

    def norms2(self):
        return _norms2(self.key)

    def mass_blocks(self, weight):
        """(P, n, n): int over each piece of P_a P_b J (J the named weight)."""
        return _mass(self.key, weight)

    def integrals(self, weight):
        return _integrals(self.key, weight)

    def vander(self, loc):
        """Basis values at local coordinates (from locate) within one piece: (m, n)."""
        return npleg.legvander(loc, self.n - 1)

    def analyse(self, C, k):
        """Coefficients from values at the nodes, along axis k of C."""
        return _apply_blocks(C, k, analysis(self.n), self.P)

    def with_n(self, n):
        return AxisBasis(self.breaks, n)

    def locate(self, t):
        """Piece index, local coordinate in [-1, 1] and a mask of points outside the axis."""
        t = np.asarray(t, float).ravel()
        br = np.asarray(self.breaks)
        span = br[-1] - br[0]
        bad = ~np.isfinite(t) | (t < br[0] - 1e-12 * span) | (t > br[-1] + 1e-12 * span)
        tt = np.clip(np.where(bad, br[0], t), br[0], br[-1])
        p = np.clip(np.searchsorted(br, tt, side="right") - 1, 0, self.P - 1)
        return p, (2.0 * tt - br[p] - br[p + 1]) / (br[p + 1] - br[p]), bad


def _ro(a):
    a.flags.writeable = False
    return a


class FourierBasis:
    """Real trigonometric basis on a periodic axis [lo, hi): 1, cos(theta), sin(theta), ..., cos(K theta), sin(K theta) with
    theta = 2 pi (t - lo) / (hi - lo); n = 2K + 1 functions, coefficients ordered by frequency. Nodes are equispaced, where
    the trapezoid rule is exact for trigonometric polynomials of degree < n, so analysis (an FFT) and the mass matrix
    (diagonal) are exact. One piece (P = 1): periodic functions are not split."""

    __slots__ = ("lo", "hi", "n", "P", "S", "key", "breaks")

    def __init__(self, lo, hi, n):
        n = int(n)
        if n % 2 == 0:
            n += 1
        self.lo, self.hi, self.n, self.P, self.S = float(lo), float(hi), n, 1, n
        self.breaks = (self.lo, self.hi)
        self.key = ("F", self.lo, self.hi, n)

    def __eq__(self, other):
        return isinstance(other, FourierBasis) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    @property
    def K(self):
        return (self.n - 1) // 2

    def _theta(self, t):
        return 2.0 * math.pi * (np.asarray(t, float) - self.lo) / (self.hi - self.lo)

    def nodes(self):
        return _ro(self.lo + (self.hi - self.lo) * np.arange(self.n) / self.n)

    def weights(self):
        return _ro(np.full(self.n, (self.hi - self.lo) / self.n))

    def norms2(self):
        L_ = self.hi - self.lo
        return _ro(np.concatenate(([L_], np.full(self.n - 1, 0.5 * L_))))

    def vander(self, t):
        th = self._theta(t)
        V = np.empty((th.size, self.n))
        V[:, 0] = 1.0
        if self.K:                                   # e^{ik theta} = (e^{i theta})^k: one exp per point, error ~ k eps
            Z = np.cumprod(np.broadcast_to(np.exp(1j * th.ravel())[:, None], (th.size, self.K)), axis=1)
            V[:, 1::2], V[:, 2::2] = Z.real, Z.imag
        return V

    def analyse(self, C, k):
        F = np.fft.rfft(np.moveaxis(C, k, 0), axis=0) / self.n          # (K + 1, ...)
        out = np.empty((self.n,) + F.shape[1:])
        out[0] = F[0].real
        out[1::2] = 2.0 * F[1:].real
        out[2::2] = -2.0 * F[1:].imag
        return np.moveaxis(out, 0, k)

    def with_n(self, n):
        return FourierBasis(self.lo, self.hi, min(int(n) + (1 - int(n) % 2), FOURIER_MAX))

    def locate(self, t):
        t = np.asarray(t, float).ravel()
        span = self.hi - self.lo
        bad = ~np.isfinite(t) | (t < self.lo - 1e-12 * span) | (t > self.hi + 1e-12 * span)
        return np.zeros(t.size, np.intp), np.where(bad, self.lo, t), bad

    def mass_blocks(self, weight):
        if weight == "1":
            return _ro(np.diag(self.norms2())[None])
        f, _ = _weight(weight)
        m = 4 * self.n + 64
        x = self.lo + (self.hi - self.lo) * np.arange(m) / m
        V = self.vander(x) * np.sqrt((self.hi - self.lo) / m * f(x))[:, None]
        return _ro((V.T @ V)[None])

    def integrals(self, weight):
        if weight == "1":
            out = np.zeros(self.n)
            out[0] = self.hi - self.lo
            return _ro(out)
        f, _ = _weight(weight)
        m = 4 * self.n + 64
        x = self.lo + (self.hi - self.lo) * np.arange(m) / m
        return _ro(self.vander(x).T @ ((self.hi - self.lo) / m * f(x)))


@lru_cache(maxsize=64)
def _nodes(key):
    breaks, n = key
    t, w = gauss(n)
    xs, ws = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        xs.append(a + 0.5 * (b - a) * (t + 1.0))
        ws.append(0.5 * (b - a) * w)
    return _ro(np.concatenate(xs)), _ro(np.concatenate(ws))


@lru_cache(maxsize=64)
def _norms2(key):
    breaks, n = key
    return _ro(np.concatenate([(b - a) / (2.0 * np.arange(n) + 1.0) for a, b in zip(breaks[:-1], breaks[1:])]))


@lru_cache(maxsize=64)
def _mass(key, weight):
    breaks, n = key
    f, extra = _weight(weight)
    t, w = gauss(n + extra + 1)
    V = npleg.legvander(t, n - 1)
    out = np.empty((len(breaks) - 1, n, n))
    for p, (a, b) in enumerate(zip(breaks[:-1], breaks[1:])):
        x = a + 0.5 * (b - a) * (t + 1.0)
        out[p] = V.T @ ((0.5 * (b - a) * w * f(x))[:, None] * V)
    return _ro(out)


@lru_cache(maxsize=64)
def _integrals(key, weight):
    breaks, n = key
    f, extra = _weight(weight)
    t, w = gauss(n + extra + 1)
    V = npleg.legvander(t, n - 1)
    out = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        x = a + 0.5 * (b - a) * (t + 1.0)
        out.append(V.T @ (0.5 * (b - a) * w * f(x)))
    return _ro(np.concatenate(out))


def _apply_blocks(C, k, blocks, P):
    """Apply a block-diagonal operator along axis k of C: blocks (n_out, n_in) shared by all pieces, or
    (P, n_out, n_in) one per piece. Axis k of C has length P * n_in; the result has P * n_out."""
    n_in = blocks.shape[-1]
    Ck = np.moveaxis(C, k, 0)
    rest = Ck.shape[1:]
    out = np.matmul(blocks, Ck.reshape(P, n_in, -1))
    return np.moveaxis(out.reshape((P * blocks.shape[-2],) + rest), 0, k)


def _transfer_apply(C, k, old, new):
    """Re-express axis k of C from basis old into basis new (new refines old's breaks, new.n >= old.n): exact."""
    if old == new:
        return C
    if isinstance(new, FourierBasis):
        if not (isinstance(old, FourierBasis) or (old.n == 1 and old.P == 1)):
            raise ValueError("a piecewise polynomial on a periodic axis cannot be re-expressed in a Fourier basis exactly")
        Ck = np.moveaxis(C, k, 0)
        out = np.zeros((new.n,) + Ck.shape[1:])
        out[:Ck.shape[0]] = Ck                       # frequencies align; a constant is the 0th coefficient
        return np.moveaxis(out, 0, k)
    if isinstance(old, FourierBasis):                # values at new's nodes, analysed: to rounding (_MIXED_PHASE)
        Ck = np.moveaxis(C, k, 0)
        vals = (old.vander(new.nodes()) @ Ck.reshape(old.n, -1)).reshape((new.S,) + Ck.shape[1:])
        return new.analyse(np.moveaxis(vals, 0, k), k)
    ob, nb = np.asarray(old.breaks), np.asarray(new.breaks)
    parent = np.clip(np.searchsorted(ob, 0.5 * (nb[:-1] + nb[1:]), side="right") - 1, 0, old.P - 1)
    t, _ = gauss(old.n)
    An = analysis(old.n)
    R = np.zeros((new.P, new.n, old.n))
    for q in range(new.P):
        p = parent[q]
        c0, c1, x0, x1 = nb[q], nb[q + 1], ob[p], ob[p + 1]
        if c0 == x0 and c1 == x1:
            R[q, :old.n] = np.eye(old.n)
        else:
            tp = (2.0 * (c0 + 0.5 * (c1 - c0) * (t + 1.0)) - x0 - x1) / (x1 - x0)
            R[q, :old.n] = An @ npleg.legvander(tp, old.n - 1)
    Ck = np.moveaxis(C, k, 0)
    rest = Ck.shape[1:]
    out = np.matmul(R, Ck.reshape(old.P, old.n, -1)[parent])
    return np.moveaxis(out.reshape((new.S,) + rest), 0, k)


def _union_basis(bases):
    fb = [b for b in bases if isinstance(b, FourierBasis)]
    pw = [b for b in bases if not isinstance(b, FourierBasis) and (b.n > 1 or b.P > 1)]
    if fb and not pw:
        return FourierBasis(fb[0].lo, fb[0].hi, max(b.n for b in bases))
    br = L.union_breaks(*[np.asarray(b.breaks) for b in bases])
    if not fb:
        return AxisBasis(br, max(b.n for b in bases))
    # piecewise (breaks given on a periodic axis, EXP-032 finding 5) and Fourier: pieces short enough for n points
    n, K = max([65] + [b.n for b in pw]), max(b.K for b in fb)
    if K:
        h = _MIXED_PHASE * n * (fb[0].hi - fb[0].lo) / (math.pi * K)
        br = np.concatenate([np.linspace(a, b, int(np.ceil((b - a) / h)) + 1)[:-1] for a, b in zip(br[:-1], br[1:])]
                            + [br[-1:]])
    return AxisBasis(br, n)


_MIXED_PHASE = 0.4      # phase half-width K pi h / period of a piece per point: n = 65 Gauss points resolve cos(a t + c)
                        # on [-1, 1] to ~1e-13 for a <= 0.49 n (measured; tests/test_exp032_fixes.py)


# ------------------------------------------------------------------------------------------ realisation

class TensorSeries:
    """N functions in a tensor product of axis bases: coefficients C of shape (S_1, ..., S_d, N)."""

    __slots__ = ("bases", "C", "err")

    def __init__(self, bases, C, err):
        self.bases = tuple(bases)
        C = np.asarray(C, float)
        self.C = _ro(C if C.flags.c_contiguous else np.ascontiguousarray(C))
        self.err = _ro(np.broadcast_to(np.asarray(err, float), (C.shape[-1],)).copy())

    @property
    def N(self):
        return self.C.shape[-1]

    @property
    def size(self):
        return self.C.size

    @property
    def d(self):
        return len(self.bases)


def to_bases(R, bases):
    C = R.C
    for k, (old, new) in enumerate(zip(R.bases, bases)):
        C = _transfer_apply(C, k, old, new)
    return C


def common_bases(*Rs):
    return tuple(_union_basis([R.bases[k] for R in Rs]) for k in range(Rs[0].d))


def _truncate(C, k, b, n_new):
    shape = C.shape
    rest = shape[:k] + shape[k + 1:]
    Ck = np.moveaxis(C, k, 0).reshape((b.P, b.n) + rest)[:, :n_new]
    return np.moveaxis(Ck.reshape((b.P * n_new,) + rest), 0, k)


def chop(bases, C, thr):
    """Lower the number of points per axis while the dropped coefficients are all below thr (absolute)."""
    d = len(bases)
    out_b = list(bases)
    for k in range(d):
        b = out_b[k]
        A = np.abs(C).max(axis=tuple(j for j in range(d + 1) if j != k))
        prof = A.reshape(b.P, b.n).max(axis=0)
        idx = np.nonzero(prof > thr)[0]
        n_new = int(idx[-1]) + 1 if len(idx) else 1
        if isinstance(b, FourierBasis) and n_new % 2 == 0:
            n_new += 1
        if n_new < b.n:
            C = _truncate(C, k, b, n_new)
            out_b[k] = b.with_n(n_new)
    return tuple(out_b), C


def _weighted_sq(bases, C):
    """sum C^2 * prod_k h_k (unweighted reference-box norm squared per column)."""
    W = C * C
    for k, b in enumerate(bases):
        shape = [1] * C.ndim
        shape[k] = b.S
        W = W * b.norms2().reshape(shape)
    return W.reshape(-1, C.shape[-1]).sum(axis=0)


def _discarded_sq(bases, C, kept):
    """sum of C^2 * prod_k h_k over the coefficients that chop() dropped (summed directly, no cancellation)."""
    W = C * C
    mask = np.ones(C.shape[:-1], bool)
    for k, (b, bk) in enumerate(zip(bases, kept)):
        shape = [1] * C.ndim
        shape[k] = b.S
        W = W * b.norms2().reshape(shape)
        keep_k = (np.arange(b.S) % b.n) < bk.n
        mshape = [1] * (C.ndim - 1)
        mshape[k] = b.S
        mask = mask & keep_k.reshape(mshape)
    return W[~mask].sum(axis=0) if (~mask).any() else np.zeros(C.shape[-1])


def _tail_sq(bases, C):
    """Weighted energy of the coefficients in the top eighth (at least 2) of the degrees on any axis."""
    W = C * C
    mask = np.zeros(C.shape[:-1], bool)
    for k, b in enumerate(bases):
        shape = [1] * C.ndim
        shape[k] = b.S
        W = W * b.norms2().reshape(shape)
        top = (np.arange(b.S) % b.n) >= b.n - max(2, b.n // 8)
        mshape = [1] * (C.ndim - 1)
        mshape[k] = b.S
        mask = mask | top.reshape(mshape)
    return W[mask].sum(axis=0) if mask.any() else np.zeros(C.shape[-1])


def _analyse(V, bases):
    C = V
    for k, b in enumerate(bases):
        C = b.analyse(C, k)
    return C


def fit_tensor(fun, reference, N, tol=1e-14, breaks=None, max_values=MAX_VALUES, max_pieces=64, jac_max=1.0,
               periodic=None):
    """Adaptive tensor fit of fun on the reference box. fun(*xi) takes broadcastable coordinate arrays and
    returns values of the broadcast shape (N = 1) or with a trailing axis of length N.
    Returns (TensorSeries, unresolved_count)."""
    d = len(reference)
    ladder = LADDERS.get(d, LADDERS[3])
    brk = []
    for k, (lo, hi) in enumerate(reference):
        extra = [] if breaks is None or breaks[k] is None else list(np.clip(np.asarray(breaks[k], float), lo, hi))
        brk.append(sorted(set([lo, hi] + extra)))
    per = (False,) * d if periodic is None else tuple(periodic)
    bases = [FourierBasis(*reference[k], ladder[0]) if per[k] else AxisBasis(brk[k], ladder[0]) for k in range(d)]
    unresolved, last = 0, None
    for _ in range(80):
        S = [b.S for b in bases]
        if int(np.prod(S)) * N > max_values:
            if last is None:
                raise L.RealisationError(f"the tensor grid {S} x {N} exceeds the memory budget; give fewer columns")
            unresolved += 1
            break
        grids = np.meshgrid(*[b.nodes() for b in bases], indexing="ij", sparse=True)
        with np.errstate(all="ignore"):
            V = np.asarray(fun(*grids), float)
        V = np.broadcast_to(V, tuple(S) + ((N,) if V.ndim == d + 1 or N > 1 else ())).reshape(tuple(S) + (N,))
        if not np.isfinite(V).all():
            V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
            unresolved += 1
        vscale = np.abs(V).reshape(-1, N).max(axis=0)
        C = _analyse(V, bases)
        del V
        nmax = max([b.n for b in bases if not isinstance(b, FourierBasis)] + [1])
        thr = max(tol, 2 * EPS * nmax ** 1.5 * d) * np.maximum(vscale, TINY)
        rel = (np.abs(C) / thr).max(axis=-1)
        last = (list(bases), C, vscale, thr)
        refine, new = False, list(bases)
        for k, b in enumerate(bases):
            prof = rel.max(axis=tuple(j for j in range(d) if j != k)).reshape(b.P, b.n)
            bad = prof[:, -max(2, b.n // 8):].max(axis=1) > 1.0
            if not bad.any():
                continue
            if isinstance(b, FourierBasis):
                if b.n < FOURIER_MAX:
                    new[k] = b.with_n(2 * b.n - 1)
                    refine = True
                else:
                    unresolved += 1
                continue
            if b.n < ladder[-1]:
                new[k] = AxisBasis(b.breaks, ladder[ladder.index(b.n) + 1])
                refine = True
            elif b.P < max_pieces:
                br = list(b.breaks)
                new[k] = AxisBasis(sorted(br + [0.5 * (br[p] + br[p + 1]) for p in np.nonzero(bad)[0]]), ladder[1])
                refine = True
            else:
                unresolved += 1
        if not refine:
            break
        bases = new
    bases, C, vscale, thr = last
    bases_c, Cc = chop(bases, C, float(np.max(thr)))
    e2 = _discarded_sq(bases, C, bases_c) * jac_max
    if unresolved:
        # the series was cut before it converged: the remainder is estimated by the energy of the last (top eighth
        # of the) coefficients on every axis, times 4 (they are not small, so this is an estimate, not a bound)
        e2 = e2 + 4.0 * _tail_sq(bases, C) * jac_max
    vol = float(np.prod([hi - lo for lo, hi in reference]))
    return TensorSeries(bases_c, Cc, np.sqrt(e2 + (EPS * vscale) ** 2 * vol * jac_max)), unresolved


def lincomb(terms):
    Rs = [R for _, R in terms]
    bases = common_bases(*Rs)
    acc, scale = 0.0, 0.0
    for a, R in terms:
        acc = acc + a * to_bases(R, bases)
        scale = max(scale, abs(a) * float(np.abs(R.C).max()) if R.C.size else 0.0)
    acc = np.asarray(acc, float)
    err = 0.0
    for a, R in terms:
        err = err + abs(a) * R.err
    b2, C2 = chop(bases, acc, 2 * EPS * max(scale, TINY))
    return TensorSeries(b2, C2, err)


def values_on_grid(R, axes_points):
    """Values on the tensor grid of internal coordinates axes_points = [t_1, ..., t_d]: shape (m_1, ..., m_d, N).
    NaN where a coordinate lies outside its axis."""
    C = R.C
    for k, (b, t) in enumerate(zip(R.bases, axes_points)):
        p, loc, bad = b.locate(t)
        Ck = np.moveaxis(C, k, 0).reshape((b.P, b.n) + C.shape[:k] + C.shape[k + 1:])
        V = b.vander(loc)
        out = np.einsum("ma,ma...->m...", V, Ck[p])
        out[bad] = np.nan
        C = np.moveaxis(out, 0, k)
    return C


def evaluate_points(R, xi):
    """Values at points given by internal coordinates xi = (xi_1, ..., xi_d) (arrays of equal shape).
    Points are grouped by the cell they lie in; within a cell the contraction is a chain of matrix products
    (the first one a single BLAS call over all points of the cell)."""
    shape = np.shape(xi[0])
    locs = [b.locate(np.asarray(a, float).ravel()) for b, a in zip(R.bases, xi)]
    m = len(locs[0][0])
    out = np.full((m, R.N), np.nan)
    bad = np.zeros(m, bool)
    for _, _, b_ in locs:
        bad |= b_
    ok = np.nonzero(~bad)[0]
    if len(ok) == 0:
        return out.reshape(shape + (R.N,))
    ns = [b.n for b in R.bases]
    cells = np.ravel_multi_index(tuple(loc[0][ok] for loc in locs), tuple(b.P for b in R.bases))
    order = np.argsort(cells, kind="stable")
    cs, starts = np.unique(cells[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    for c, s0, s1 in zip(cs, starts, ends):
        piece = np.unravel_index(c, tuple(b.P for b in R.bases))
        blk = R.C[tuple(slice(p * n, (p + 1) * n) for p, n in zip(piece, ns))]      # (n_1, ..., n_d, N)
        for a in range(s0, s1, 4096):                                               # memory m_c n_2 ... n_d: in batches
            pts = ok[order[a:min(a + 4096, s1)]]
            T = R.bases[0].vander(locs[0][1][pts]) @ blk.reshape(ns[0], -1)         # (m_c, n_2 ... n_d N)
            for k in range(1, R.d):
                T = T.reshape(len(pts), ns[k], -1)
                V = R.bases[k].vander(locs[k][1][pts])
                T = np.matmul(V[:, None, :], T)[:, 0]
            out[pts] = T.reshape(len(pts), R.N)
    return out.reshape(shape + (R.N,))


def multiply(A, B, max_n=129):
    bases = []
    for k in range(A.d):
        u = _union_basis([A.bases[k], B.bases[k]])
        cap = FOURIER_MAX if isinstance(u, FourierBasis) else max_n
        bases.append(u.with_n(max(u.n, min(A.bases[k].n + B.bases[k].n - 1, cap))))    # u.n: mixed periodic axes
    nodes = [b.nodes() for b in bases]
    va, vb = values_on_grid(A, nodes), values_on_grid(B, nodes)
    C = _analyse(va * vb, bases)
    sa = np.abs(va).reshape(-1, A.N).max(axis=0)
    sb = np.abs(vb).reshape(-1, B.N).max(axis=0)
    b2, C2 = chop(tuple(bases), C, 4 * EPS * max(float(np.abs(C).max()), TINY))
    return TensorSeries(b2, C2, sa * B.err + sb * A.err)


def apply_mass(C, bases, weights):
    T = C
    for k, b in enumerate(bases):
        T = _apply_blocks(T, k, b.mass_blocks(weights[k]), b.P)
    return T


def gram(A, B, weights):
    """(A.N, B.N) Gram matrix; computed in column chunks of A so that memory stays near MAX_VALUES."""
    bases = common_bases(A, B)
    Cb = to_bases(B, bases).reshape(-1, B.N)
    size = int(np.prod([b.S for b in bases]))
    step = max(1, MAX_VALUES // max(4 * size, 1))
    out = np.empty((A.N, B.N))
    for s in range(0, A.N, step):
        Ac = TensorSeries(A.bases, A.C[..., s:s + step], A.err[s:s + step])
        T = apply_mass(to_bases(Ac, bases), bases, weights)
        out[s:s + step] = T.reshape(-1, Ac.N).T @ Cb
    return out


def sqnorms(A, weights):
    T = apply_mass(A.C, A.bases, weights)
    return (T * A.C).reshape(-1, A.N).sum(axis=0)


def integral(R, weights):
    v = R.C
    for k, b in enumerate(R.bases):
        v = np.tensordot(b.integrals(weights[k]), v, axes=(0, 0))
    return np.asarray(v)


def hstack(Rs):
    bases = common_bases(*Rs)
    return TensorSeries(bases, np.concatenate([to_bases(R, bases) for R in Rs], axis=-1), np.concatenate([R.err for R in Rs]))


def _times(c, v):
    """c v for the constant density c of a space (plan 064); v unchanged (its type kept) when c = 1."""
    return v if c == 1.0 else c * v


# ------------------------------------------------------------------------------------------ backend

class TensorBackend(Backend):
    """Tensor-product Legendre realisation for a domain with product measure in natural coordinates: Lebesgue measure,
    or c times it for a constant density c > 0 (plan 064: every pairing times c, self.c)."""
    CAPABILITIES = ("breaks_per_axis",)

    def __init__(self, measure, tol, periodic_basis="fourier"):
        from ...measures import constant_density
        c = constant_density(measure)
        if c is None:
            raise Unsupported(f"the density of {measure} is not constant: in the {'local' if type(self).__name__ == 'LocalBackend' else 'tensor'} "
                              "realisation only constant densities are realised (plan 064); a radial density on a ball "
                              "or shell needs realisation='radial_harmonics'")
        self.c = c
        self.measure, self.tol = measure, tol
        self.domain = measure.set
        self.dim = self.domain.dim
        self.realisation_name = "legendre" if periodic_basis == "legendre" else "tensor"
        self.jac = tuple(self.domain.jacobian)
        self.periodic = tuple(self.domain.periodic) if periodic_basis == "fourier" else (False,) * self.dim
        self.jac_max = float(np.prod([np.abs(self.domain.jacobian_value(k, np.linspace(lo, hi, 257))).max()
                                      for k, (lo, hi) in enumerate(self.domain.reference)]))

    # construction --------------------------------------------------------------------------
    def probe(self):
        ref = self.domain.reference
        xi = [np.array([lo + f * (hi - lo) for f in (0.3, 0.5, 0.7)]) for lo, hi in ref]
        return tuple(np.asarray(c, float) for c in self.domain.to_cartesian(*xi))

    def _internal(self, fun):
        to_cart = self.domain.to_cartesian

        def g(*xi):
            return fun(*to_cart(*xi))
        return g

    def _breaks(self, breaks):
        """The domain's reference breaks merged with the given ones (per axis)."""
        own = self.domain.reference_breaks
        if own is None:
            return breaks
        if breaks is None:
            return own
        return tuple(sorted(set(a or ()) | set(b or ())) for a, b in zip(own, breaks))

    def _periodic(self, breaks):
        """The axes fitted in a Fourier basis: periodic ones without given breaks inside. Given breaks mark
        discontinuities (an arc's indicator) that equispaced samples can miss: piecewise there (EXP-032 finding 5)."""
        if breaks is None:
            return self.periodic
        return tuple(per and not (b is not None and any(lo + 1e-12 * (hi - lo) < float(x) < hi - 1e-12 * (hi - lo)
                                                        for x in np.ravel(b)))
                     for per, b, (lo, hi) in zip(self.periodic, breaks, self.domain.reference))

    def fit(self, fun, N=1, breaks=None):
        periodic = self._periodic(breaks)
        breaks = self._breaks(breaks)
        return fit_tensor(self._internal(fun), self.domain.reference, N, tol=self.tol, breaks=breaks,
                          jac_max=self.jac_max, periodic=periodic)

    def fit_family(self, fun_idx, N, breaks=None, sparse=None, name="F"):
        periodic = self._periodic(breaks)
        breaks = self._breaks(breaks)
        out, s, step = [], 0, 64
        while s < N:
            idx = np.arange(s, min(N, s + step))
            try:
                R, u = fit_tensor(self._internal(lambda *X, idx=idx: fun_idx(*X, idx)), self.domain.reference,
                                  len(idx), tol=self.tol, breaks=breaks, jac_max=self.jac_max,
                                  periodic=periodic)
            except L.RealisationError:
                if step == 1:
                    raise
                step = max(1, step // 4)
                continue
            if u and step > 1:                          # the memory budget limited the grid: fewer columns at a time
                step = max(1, step // 4)
                continue
            if u:
                warnings.warn(f"family {name}: {u} unresolved axis refinement(s) in columns {s}..{idx[-1]}")
            out.append(R)
            s += len(idx)
        return out[0] if len(out) == 1 else hstack(out)

    def constant(self, values):
        values = np.atleast_1d(np.asarray(values, float))
        bases = [FourierBasis(lo, hi, 1) if per else AxisBasis([lo, hi], 1)
                 for (lo, hi), per in zip(self.domain.reference, self.periodic)]
        return TensorSeries(bases, values.reshape((1,) * self.dim + (len(values),)), np.zeros(len(values)))

    def coordinate(self, k):
        R, _ = self.fit(lambda *X: X[k])
        return R

    # algebra -------------------------------------------------------------------------------
    def product_family(self, factors, params, name="K"):
        return SEP.fit_product_family(self, factors, params, name=name)

    def to_tensor(self, R):
        """A tensor realisation of any realisation (expansions and lazy combinations are fitted)."""
        if isinstance(R, TensorSeries):
            return R
        if isinstance(R, SEP.LazyCombination) and all(isinstance(Q, TensorSeries) for _, Q in R.terms):
            return lincomb(list(R.terms))

        def fun(*grids):
            G = np.broadcast_arrays(*grids)
            return SEP.evaluate(R, G, evaluate_points)
        T, u = fit_tensor(fun, self.domain.reference, R.N, tol=self.tol, jac_max=self.jac_max, periodic=self.periodic)
        if u:
            warnings.warn("converting an expansion to a tensor realisation did not fully converge; arithmetic with "
                          "it is approximate (norms and inner products of the expansion itself stay exact)")
        return T

    def lincomb(self, terms):
        if all(isinstance(R, TensorSeries) for _, R in terms):
            return lincomb(terms)
        return SEP.lincomb(terms, lincomb)

    def combine(self, R, A):
        A = np.asarray(A, float)
        if isinstance(R, SEP.ProductFamily):
            return SEP.KernelExpansion(R, A)
        if isinstance(R, SEP.KernelExpansion):
            return SEP.KernelExpansion(R.family, R.c @ A)
        if isinstance(R, SEP.LazyCombination):
            return SEP.LazyCombination([(a, self.combine(Q, A)) for a, Q in R.terms])
        return TensorSeries(R.bases, np.tensordot(R.C, A, axes=(R.d, 0)), np.abs(A).T @ R.err)

    def colsum(self, R):
        out = self.combine(R, np.ones((R.N, 1)))
        return lincomb([(1.0, out)]) if isinstance(out, TensorSeries) else out

    def take(self, R, idx):
        sel = np.atleast_1d(idx)
        if isinstance(R, SEP.ProductFamily):
            return R.take(sel)
        if isinstance(R, SEP.KernelExpansion):
            return SEP.KernelExpansion(R.family, R.c[:, sel])
        if isinstance(R, SEP.LazyCombination):
            return SEP.LazyCombination([(a, self.take(Q, idx)) for a, Q in R.terms])
        out = TensorSeries(R.bases, R.C[..., sel], R.err[sel])
        return lincomb([(1.0, out)]) if np.ndim(idx) == 0 else out

    def hstack(self, Rs):
        if all(isinstance(R, TensorSeries) for R in Rs):
            return hstack(Rs)
        if all(isinstance(R, SEP.KernelExpansion) and R.family is Rs[0].family for R in Rs):
            return SEP.KernelExpansion(Rs[0].family, np.hstack([R.c for R in Rs]))
        return hstack([self.to_tensor(R) for R in Rs])

    def multiply(self, A, B):
        return multiply(self.to_tensor(A), self.to_tensor(B))

    # measure -------------------------------------------------------------------------------
    def gram(self, A, B):                     # c: the constant density (plan 064)
        if isinstance(A, TensorSeries) and isinstance(B, TensorSeries):
            return _times(self.c, gram(A, B, self.jac))
        return _times(self.c, SEP.gram(A, B, lambda a, b: gram(a, b, self.jac)))

    def sqnorms(self, A):
        if isinstance(A, TensorSeries):
            return _times(self.c, sqnorms(A, self.jac))
        return _times(self.c, SEP.sqnorms(A, lambda a: sqnorms(a, self.jac), lambda a, b: gram(a, b, self.jac)))

    def integral(self, R, over=None):
        if over is not None:
            raise Unsupported("integrals over parts of a 2D/3D domain are not implemented yet")
        if isinstance(R, TensorSeries):
            return _times(self.c, integral(R, self.jac))
        return _times(self.c, SEP.integral(R, lambda r: integral(r, self.jac)))

    def box_integral(self, R, S):            # int_S f dx over a coordinate box, exact (box_form.py), or None
        from ..rkhs.box_form import box_integral
        return box_integral(R, S, self.domain, self.jac) if isinstance(R, TensorSeries) else None

    def lp_norm(self, R, p):
        R = self.to_tensor(R)
        bases = [b.with_n(min(2 * b.n + 8, FOURIER_MAX if isinstance(b, FourierBasis) else 129)) for b in R.bases]
        V = np.abs(values_on_grid(R, [b.nodes() for b in bases]))
        if math.isinf(p):
            return V.reshape(-1, R.N).max(axis=0)
        W = 1.0
        for k, b in enumerate(bases):
            shape = [1] * (R.d + 1)
            shape[k] = b.S
            W = W * (b.weights() * self.domain.jacobian_value(k, b.nodes())).reshape(shape)
        return (self.c * ((V ** p) * W).reshape(-1, R.N).sum(axis=0)) ** (1.0 / p)


    # values and description ----------------------------------------------------------------
    def evaluate(self, R, *X):
        shape = np.broadcast(*X).shape
        xi = self.domain.from_cartesian(*[np.broadcast_to(np.asarray(a, float), shape) for a in X])
        if isinstance(R, TensorSeries):
            return evaluate_points(R, xi)
        return SEP.evaluate(R, xi, evaluate_points)

    def grid_values(self, R, axes_points):
        """Values on a tensor grid of natural coordinates: shape (m_1, ..., m_d, N)."""
        if isinstance(R, TensorSeries):
            return values_on_grid(R, axes_points)
        return SEP.grid_values(R, axes_points, values_on_grid)

    def error(self, R):
        return R.err if self.c == 1.0 else math.sqrt(self.c) * R.err      # R.err in L2(dx); the space's norm

    def breaks(self, R):
        if not isinstance(R, TensorSeries):
            return None
        return tuple(np.asarray(b.breaks) for b in R.bases)

    def merge_breaks(self, items):
        items = [it for it in items if it is not None]
        if not items:
            return None
        return tuple(np.unique(np.concatenate([np.asarray(it[k]) for it in items])) for k in range(self.dim))

    def is_sparse(self, R):
        return False

    def describe(self, R):
        if not isinstance(R, TensorSeries):
            return SEP.describe(R)
        names = self.domain.internal_names

        def ax(nm, b):
            if isinstance(b, FourierBasis):
                return f"{nm}: Fourier, {b.K} harmonics ({b.n} coefficients)"
            return f"{nm}: {b.P} piece(s) x {b.n} pts"
        kind = "tensor Legendre" + (" / Fourier" if any(isinstance(b, FourierBasis) for b in R.bases) else "")
        return dict(kind=kind, pieces=int(np.prod([b.P for b in R.bases])),
                    degree=tuple(b.K if isinstance(b, FourierBasis) else b.n - 1 for b in R.bases), size=R.size,
                    axes=", ".join(ax(nm, b) for nm, b in zip(names, R.bases)))

    def summary(self, R):
        if not isinstance(R, TensorSeries):
            return SEP.summary(R)
        d = self.describe(R)
        names = self.domain.internal_names
        deg = ", ".join(f"{nm} {'K=' if isinstance(b, FourierBasis) else ''}{g}"
                        for nm, g, b in zip(names, d["degree"], R.bases))
        return f"{d['kind']}: {d['pieces']} cell{'s' * (d['pieces'] > 1)}, degrees ({deg})"
