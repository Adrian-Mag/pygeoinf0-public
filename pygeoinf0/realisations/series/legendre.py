"""Piecewise Legendre realisation of real functions on a bounded interval.

A realisation of N functions on [a, b] is a partition a = x_0 < ... < x_m = b and, on each piece,
an array of Legendre coefficients of shape (degree + 1, N) in the local variable t in [-1, 1].
It is built adaptively from samples at Gauss-Legendre nodes. The error estimate is the L2 norm of
the discarded coefficients (it is an estimate, not a bound: the unseen tail is assumed smaller).

Everything else is exact on the polynomials: sums, products, restriction to a finer partition,
integrals, inner products against Lebesgue measure (Legendre polynomials are orthogonal, so the
inner product is a weighted dot product of coefficients) and against a polynomial density
(Gauss-Legendre quadrature of sufficient order). This module knows nothing about spaces.
"""
from functools import lru_cache

import numpy as np
from numpy.polynomial import legendre as npleg

from ... import quadrature
from ...core.errors import RealisationError

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny
NSEQ = (17, 33, 65, 129, 257, 513, 1025)




def _ro(a):
    a.flags.writeable = False
    return a


@lru_cache(maxsize=None)
def analysis(n):
    """T with a = T @ f(t_j): a_k = (2k+1)/2 sum_j w_j P_k(t_j) f(t_j); exact for degree <= n-1."""
    t, w = quadrature.gauss(n)
    V = npleg.legvander(t, n - 1)
    return _ro(np.ascontiguousarray((V * w[:, None]).T * ((2.0 * np.arange(n) + 1.0) / 2.0)[:, None]))


def polyval(t, C):
    """Values of the Legendre series with coefficient columns C (deg+1, N) at points t: shape (len(t), N)."""
    t = np.asarray(t, float)
    if C.shape[1] == 1:
        return npleg.legval(t, C[:, 0])[:, None]
    step = max(1, (1 << 22) // max(1, C.shape[0]))
    if len(t) <= step:
        return npleg.legvander(t, C.shape[0] - 1) @ C
    return np.vstack([npleg.legvander(t[i:i + step], C.shape[0] - 1) @ C for i in range(0, len(t), step)])


class PiecewiseLegendre:
    """N real functions on common breakpoints (the realisation used by function spaces)."""

    __slots__ = ("breaks", "coefs", "err")
    CAPABILITIES = ("piecewise_legendre", "cell_integrals", "box_support", "box_member", "box_rule")

    def __init__(self, breaks, coefs, err):
        self.breaks = _ro(np.asarray(breaks, float))
        self.coefs = tuple(_ro(np.asarray(c, float)) for c in coefs)
        self.err = _ro(np.broadcast_to(np.asarray(err, float), (self.coefs[0].shape[1],)).copy())

    @property
    def N(self):
        return self.coefs[0].shape[1]

    @property
    def npieces(self):
        return len(self.coefs)

    @property
    def degrees(self):
        return [c.shape[0] - 1 for c in self.coefs]

    # capabilities (registry.CAPABILITIES; box_integrals.py) ---------------------------------------------------
    def box_support(self, H, l, u, rtol=1e-8):
        from ..bounds.box_integrals import interval_support
        return interval_support(self, H, l, u, rtol)

    def box_member(self, H, l, u):
        from ..bounds.box_integrals import interval_member
        return interval_member(self, H, l, u)

    def box_rule(self, H, Rs, l, u, weight, oversample, least=None):
        from ..bounds.box_integrals import IntervalRule
        return IntervalRule(H, Rs, l, u, weight, oversample, least)

    def cell_integrals(self, edges):
        from ..piecewise_constant import cell_integrals
        return cell_integrals(self, edges)

    @property
    def size(self):
        return sum(c.size for c in self.coefs)

    def take(self, idx):
        idx = np.atleast_1d(idx)
        return PiecewiseLegendre(self.breaks, [c[:, idx] for c in self.coefs], self.err[idx])

    def maxabs(self):
        return max(float(np.abs(c).max()) for c in self.coefs)

    def sup_bound(self):
        """Per-column bound on the sup norm of the realised polynomial (|P_k| <= 1 on [-1, 1])."""
        return np.max([np.abs(c).sum(axis=0) for c in self.coefs], axis=0)

    @staticmethod
    def constant(a, b, values):
        values = np.atleast_1d(np.asarray(values, float))
        return PiecewiseLegendre([a, b], [values[None, :]], np.zeros(len(values)))

    @staticmethod
    def identity(a, b):
        return PiecewiseLegendre([a, b], [np.array([[0.5 * (a + b)], [0.5 * (b - a)]])], [0.0])


# ---------------------------------------------------------------- partitions

def union_breaks(*bs):
    first = bs[0]
    if all(len(b) == len(first) and np.array_equal(b, first) for b in bs[1:]):
        return first
    u = np.unique(np.concatenate(bs))
    scale = max(u[-1] - u[0], abs(u[0]), abs(u[-1]))
    keep = np.concatenate(([True], np.diff(u) > 16 * EPS * scale))
    if not keep[-1]:                      # keep the right endpoint itself, drop its near-duplicate
        j = np.nonzero(keep)[0][-1]
        if j > 0:
            keep[j] = False
        keep[-1] = True
    return u[keep]


def restrict(C, x0, x1, c0, c1):
    """Coefficients on [c0, c1] of the polynomial with coefficients C on [x0, x1] (exact, same degree)."""
    if c0 == x0 and c1 == x1:
        return C
    n = C.shape[0]
    t, _ = quadrature.gauss(n)
    tp = (2.0 * (c0 + 0.5 * (c1 - c0) * (t + 1.0)) - x0 - x1) / (x1 - x0)
    return analysis(n) @ (npleg.legvander(tp, n - 1) @ C)


def cut_breaks(breaks, a, b):
    """Breaks of a piecewise function on [a, b] inside breaks: a, the interior ones, b. A break within rounding of a or b is
    not kept as a sliver; None when [a, b] is not inside."""
    br = np.asarray(breaks, float)
    tol = 8 * EPS * max(abs(br[0]), abs(br[-1]), 1.0)
    if a < br[0] - tol or b > br[-1] + tol or not a < b:
        return None
    return np.r_[a, br[(br > a + tol) & (br < b - tol)], b]


def cut(coefs, breaks, a, b):
    """(new breaks, coefficient arrays) on [a, b], axis 0 the Legendre index: whole pieces kept as they are, a straddling
    piece re-expressed exactly on its part. None when [a, b] is not inside breaks."""
    nb = cut_breaks(breaks, a, b)
    if nb is None:
        return None
    br = np.asarray(breaks, float)
    out = []
    for c0, c1 in zip(nb[:-1], nb[1:]):
        p = int(np.clip(np.searchsorted(br, 0.5 * (c0 + c1), side="right") - 1, 0, len(br) - 2))
        C = coefs[p]
        if abs(c0 - br[p]) <= 8 * EPS * max(abs(br[p]), 1.0) and abs(c1 - br[p + 1]) <= 8 * EPS * max(abs(br[p + 1]), 1.0):
            out.append(C)
        else:
            shape = C.shape
            out.append(restrict(C.reshape(shape[0], -1), br[p], br[p + 1], c0, c1).reshape(shape))
    return nb, out


def refine(p, nb):
    if len(nb) == len(p.breaks) and np.array_equal(nb, p.breaks):
        return list(p.coefs)
    mids = 0.5 * (nb[:-1] + nb[1:])
    parent = np.clip(np.searchsorted(p.breaks, mids, side="right") - 1, 0, p.npieces - 1)
    return [restrict(p.coefs[k], p.breaks[k], p.breaks[k + 1], nb[i], nb[i + 1]) for i, k in enumerate(parent)]


def _pad(C, n):
    if C.shape[0] == n:
        return C
    return np.vstack([C, np.zeros((n - C.shape[0], C.shape[1]))])


def _chop(C, thr):
    rows = np.nonzero((np.abs(C) > thr).any(axis=1))[0]
    return C[: rows[-1] + 1] if len(rows) else C[:1] * 0.0


# ---------------------------------------------------------------- linear algebra on realisations

def lincomb(terms):
    """sum_j alpha_j p_j for terms [(alpha_j, p_j)]; column counts broadcast (1 with N)."""
    nb = union_breaks(*[p.breaks for _, p in terms])
    refined = [(al, refine(p, nb)) for al, p in terms]
    scale = max(abs(al) * p.maxabs() for al, p in terms)
    coefs = []
    for i in range(len(nb) - 1):
        n = max(R[i].shape[0] for _, R in refined)
        acc = 0.0
        for al, R in refined:
            acc = acc + al * _pad(R[i], n)
        coefs.append(_chop(np.atleast_2d(acc), 2 * EPS * scale))
    err = 0.0
    for al, p in terms:
        err = err + abs(al) * p.err
    return PiecewiseLegendre(nb, coefs, err)


def pad(pl, a, b):
    """pl with exact zero pieces on [a, its first break] and [its last break, b] (ends where it is 0)."""
    z, br, cf = np.zeros((1, pl.N)), list(pl.breaks), list(pl.coefs)
    br, cf = ([a] + br, [z] + cf) if br[0] > a else (br, cf)
    return PiecewiseLegendre(br + [b], cf + [z], pl.err) if br[-1] < b else PiecewiseLegendre(br, cf, pl.err)


def combine(p, A):
    """Columns p @ A: the realisation of sum_j A[j, i] f_j for each i (A has shape (p.N, M))."""
    A = np.asarray(A, float)
    return PiecewiseLegendre(p.breaks, [c @ A for c in p.coefs], np.abs(A).T @ p.err)


def colsum(p):
    return PiecewiseLegendre(p.breaks, [c.sum(axis=1, keepdims=True) for c in p.coefs], [p.err.sum()])


def hstack(ps):
    nb = union_breaks(*[p.breaks for p in ps])
    refined = [refine(p, nb) for p in ps]
    coefs = []
    for i in range(len(nb) - 1):
        n = max(R[i].shape[0] for R in refined)
        coefs.append(np.hstack([_pad(R[i], n) for R in refined]))
    return PiecewiseLegendre(nb, coefs, np.concatenate([p.err for p in ps]))


def multiply(p, q):
    """Pointwise product (exact on the polynomials; degree adds up)."""
    nb = union_breaks(p.breaks, q.breaks)
    P, Q = refine(p, nb), refine(q, nb)
    coefs = []
    scale = p.maxabs() * q.maxabs()
    for Cp, Cq in zip(P, Q):
        n = Cp.shape[0] + Cq.shape[0] - 1
        t, _ = quadrature.gauss(n)
        C = analysis(n) @ (polyval(t, Cp) * polyval(t, Cq))
        coefs.append(_chop(C, 4 * EPS * scale))
    return PiecewiseLegendre(nb, coefs, p.sup_bound() * q.err + q.sup_bound() * p.err)


# ---------------------------------------------------------------- integrals and inner products

def gram(p, q):
    """Matrix of integrals of f_i g_j against Lebesgue measure, shape (p.N, q.N)."""
    nb = union_breaks(p.breaks, q.breaks)
    P, Q = refine(p, nb), refine(q, nb)
    G = np.zeros((p.N, q.N))
    for i in range(len(nb) - 1):
        m = min(P[i].shape[0], Q[i].shape[0])
        h = (nb[i + 1] - nb[i]) / (2.0 * np.arange(m) + 1.0)
        G += P[i][:m].T @ (h[:, None] * Q[i][:m])
    return G


def gram_weighted(p, q, w):
    """As gram(), against the measure w(x) dx, w a realised density with one column."""
    nb = union_breaks(p.breaks, q.breaks, w.breaks)
    P, Q, W = refine(p, nb), refine(q, nb), refine(w, nb)
    G = np.zeros((p.N, q.N))
    for i in range(len(nb) - 1):
        n = (P[i].shape[0] + Q[i].shape[0] + W[i].shape[0] - 3) // 2 + 1
        t, wt = quadrature.gauss(n)
        wv = polyval(t, W[i])[:, 0] * wt * 0.5 * (nb[i + 1] - nb[i])
        G += polyval(t, P[i]).T @ (wv[:, None] * polyval(t, Q[i]))
    return G


def sqnorms(p):
    out = np.zeros(p.N)
    for i, C in enumerate(p.coefs):
        h = (p.breaks[i + 1] - p.breaks[i]) / (2.0 * np.arange(C.shape[0]) + 1.0)
        out += (h[:, None] * C * C).sum(axis=0)
    return out


def integral(p, c=None, d=None):
    """Integrals of the N functions over [c, d] (default: the whole interval) against Lebesgue measure."""
    a, b = p.breaks[0], p.breaks[-1]
    c = a if c is None else max(a, c)
    d = b if d is None else min(b, d)
    if c >= d:
        return np.zeros(p.N)
    nb = union_breaks(p.breaks, np.array([c, d]))
    P = refine(p, nb)
    out = np.zeros(p.N)
    for i in range(len(nb) - 1):
        if nb[i] >= c and nb[i + 1] <= d:
            out += P[i][0] * (nb[i + 1] - nb[i])
    return out


def evaluate(p, x):
    """Values at x (shape x.shape + (N,)); NaN outside [a, b]; at an interior breakpoint the mean of
    the one-sided limits (the value of the canonical representative at a jump)."""
    x = np.asarray(x, float)
    xf = x.ravel()
    out = np.full((xf.size, p.N), np.nan)
    br = p.breaks
    m = p.npieces
    inside = (xf >= br[0]) & (xf <= br[-1])
    idx = np.clip(np.searchsorted(br, xf, side="right") - 1, 0, m - 1)
    for k in np.unique(idx[inside]):
        sel = inside & (idx == k)
        t = (2.0 * xf[sel] - br[k] - br[k + 1]) / (br[k + 1] - br[k])
        out[sel] = polyval(t, p.coefs[k])
    if m > 1:
        atb = inside & np.isin(xf, br[1:-1])
        if atb.any():
            js = np.searchsorted(br, xf[atb])
            left = np.array([p.coefs[j - 1].sum(axis=0) for j in js])
            right = np.array([(p.coefs[j] * ((-1.0) ** np.arange(p.coefs[j].shape[0]))[:, None]).sum(axis=0)
                              for j in js])
            out[atb] = 0.5 * (left + right)
    return out.reshape(x.shape + (p.N,))


# ---------------------------------------------------------------- adaptive construction

def _sample(fun, x, N):
    v = np.asarray(fun(x), dtype=float)
    if v.ndim == 0:
        v = np.full((len(x), N), float(v))
    elif v.ndim == 1:
        v = np.broadcast_to(v[:, None], (len(x), N))
    if v.shape != (len(x), N):
        raise ValueError(f"the function returned shape {v.shape} for {len(x)} points; expected {(len(x), N)}")
    return v


def _fit_piece(fun, x0, x1, N, vscale, tol, n_start=17):
    """Sample at Gauss-Legendre nodes of increasing order until the coefficient tail is below the noise
    floor of the transform (about eps n^1.5 relative) or tol; give up early when the tail stops decaying."""
    h = 0.5 * (x1 - x0)
    prev = None
    for n in NSEQ:
        if n < n_start:
            continue
        t, w = quadrature.gauss(n)
        x = x0 + h * (t + 1.0)
        v = _sample(fun, x, N)
        if not np.isfinite(v).all():
            return False, None, None, x, np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), w, False
        np.maximum(vscale, np.abs(v).max(axis=0), out=vscale)
        C = analysis(n) @ v
        thr = max(tol, 2 * EPS * n ** 1.5) * np.maximum(vscale, TINY)
        k = max(4, n // 8)
        tail = np.abs(C[-k:]).max(axis=0)
        if np.all(tail <= thr):
            # converged: chop at the threshold, which models the transform's rounding noise (about eps n^1.5);
            # the last few coefficients can sit below that noise and are not a reliable estimate of it
            rows = np.nonzero((np.abs(C) > thr).any(axis=1))[0]
            keep = rows[-1] + 1 if len(rows) else 1
            hk = (x1 - x0) / (2.0 * np.arange(keep, n) + 1.0)
            e2 = (hk[:, None] * C[keep:] ** 2).sum(axis=0) + (EPS * vscale) ** 2 * (x1 - x0)
            return True, C[:keep], e2, x, v, w, True
        plateau = _noise_plateau(C, vscale)
        if plateau is not None:                       # the function itself is noisy at this level: accept
            keep, level = plateau
            hk = (x1 - x0) / (2.0 * np.arange(keep, n) + 1.0)
            e2 = (hk[:, None] * C[keep:] ** 2).sum(axis=0) + (level * vscale) ** 2 * (x1 - x0)
            return True, C[:keep], e2, x, v, w, True
        if prev is not None and n >= 257 and np.all(tail > 0.01 * prev):
            break                                     # algebraic decay (kink, jump): splitting is cheaper
        prev = tail
    return False, None, None, x, v, w, True


def _noise_plateau(C, vscale, small=1e-8):
    """If the coefficients stop decaying at a small relative level (a plateau of rounding noise in the function's
    own values, e.g. exp((cos(t) - 1) / w^2) with small w), return (rows to keep, plateau level); else None.
    Raising the degree or splitting cannot get below such a floor."""
    n = C.shape[0]
    if n < 33:
        return None
    rel = np.abs(C) / np.maximum(vscale, TINY)
    q = n // 4
    q4 = rel[-q:].max(axis=0)
    q3 = rel[-2 * q:-q].max(axis=0)
    if np.all((q4 <= small) & (q4 >= 0.1 * q3)):
        level = float(q4.max())
        rows = np.nonzero((rel > 10 * level).any(axis=1))[0]
        return (rows[-1] + 1 if len(rows) else 1), level
    return None


def _split_point(fun, N, x0, x1, xs, vs, vscale, finite):
    """Midpoint, unless the steepest gap between samples is a jump: then locate the jump by bisection
    on function values to adjacent floats and split exactly there (both sides are then smooth)."""
    mid = 0.5 * (x0 + x1)
    if finite and len(xs) > 2:
        rel = np.abs(np.diff(vs, axis=0)) / np.maximum(vscale, TINY)
        slope = rel.max(axis=1) / np.maximum(np.diff(xs), TINY)
        j = int(np.argmax(slope))
        if slope[j] > 50.0 * max(float(np.median(slope)), TINY):
            col = int(np.argmax(rel[j]))
            lo, hi, flo, fhi = xs[j], xs[j + 1], vs[j, col], vs[j + 1, col]
            d0 = abs(fhi - flo)
            for _ in range(200):
                m = 0.5 * (lo + hi)
                if not lo < m < hi:
                    break
                fm = _sample(fun, np.array([m]), N)[0, col]
                if not np.isfinite(fm):
                    break
                if abs(fm - flo) >= abs(fhi - fm):
                    hi, fhi = m, fm
                else:
                    lo, flo = m, fm
            if abs(fhi - flo) > 0.1 * d0 and abs(fhi - flo) > 100 * EPS * vscale[col]:
                # lo and hi are adjacent floats around the jump; either is a valid breakpoint. Prefer the
                # one a person would have written (shorter decimal form), so a jump at 0.4 lands at 0.4.
                s = lo if len(repr(float(lo))) < len(repr(float(hi))) else hi
                if x0 < s < x1:
                    return s
    return min(max(mid, np.nextafter(x0, x1)), np.nextafter(x1, x0))


def fit(fun, a, b, N=1, breaks=None, tol=1e-14, max_pieces=512, scout=None, n_start=17):
    """Adaptive piecewise Legendre fit of fun on [a, b].

    fun(x) takes a 1-d array of n points and returns n values (N = 1) or an (n, N) array.
    Returns (PiecewiseLegendre, number_of_unresolved_pieces).

    Pieces that do not converge are split: at a jump located by bisection to adjacent floats when the
    steepest gap between samples is one, otherwise at the midpoint. Every accepted piece is checked
    against a uniform scout sample (default 2049 points, fewer for large N), which catches narrow
    features that the Gauss nodes stepped over; a feature narrower than the scout spacing can still be
    missed, so pass breaks (or a larger scout) when you know where the function is localised.
    A piece narrower than a few ulps is kept as its mean, with the sample spread as error estimate.
    """
    init = np.array([a, b], float) if breaks is None else np.unique(np.concatenate(([a, b], np.clip(breaks, a, b))))
    if scout is None:
        scout = 2049 if N == 1 else int(min(2049, max(129, (1 << 19) // N)))
    sx = np.unique(np.concatenate([np.linspace(init[i], init[i + 1], max(3, scout // (len(init) - 1)))
                                   for i in range(len(init) - 1)]))
    sx = sx[(sx > a) & (sx < b)]
    sv = _sample(fun, sx, N)
    finite_s = np.isfinite(sv)
    vscale = np.where(finite_s, np.abs(sv), 0.0).max(axis=0)
    minwidth = 64 * EPS * max(b - a, abs(a), abs(b))
    todo = [(init[i], init[i + 1]) for i in range(len(init) - 1)][::-1]
    done, unresolved = [], 0
    while todo:
        x0, x1 = todo.pop()
        ok, C, e2, xs, vs, w, finite = _fit_piece(fun, x0, x1, N, vscale, tol, n_start if len(init) == 2 and x0 == a and x1 == b else 17)
        s = None
        if ok:
            i0, i1 = np.searchsorted(sx, x0, side="right"), np.searchsorted(sx, x1, side="left")
            if i1 > i0:
                t = (2.0 * sx[i0:i1] - x0 - x1) / (x1 - x0)
                miss = np.where(finite_s[i0:i1], np.abs(polyval(t, C) - sv[i0:i1]), 0.0) / np.maximum(vscale, TINY)
                j = np.unravel_index(int(np.argmax(miss)), miss.shape)[0]
                if miss.max() > 1e-8:
                    ok = False
                    s = sx[i0 + j]
            if ok:
                done.append((x0, x1, C, e2))
                continue
        if x1 - x0 <= minwidth or len(done) + len(todo) + 2 > max_pieces:
            if C is None:
                mean = (w[:, None] * vs).sum(axis=0) / 2.0
                e2 = (x1 - x0) * (w[:, None] * (vs - mean) ** 2).sum(axis=0) / 2.0
                C = mean[None, :]
            if not finite:
                e2 = np.full(N, np.inf)
            done.append((x0, x1, C, e2))
            unresolved += 1
            continue
        if s is None or not x0 < s < x1:
            s = _split_point(fun, N, x0, x1, xs, vs, vscale, finite)
        todo.append((s, x1))
        todo.append((x0, s))
    done.sort(key=lambda z: z[0])
    return PiecewiseLegendre([z[0] for z in done] + [done[-1][1]], [z[2] for z in done],
              np.sqrt(np.sum([z[3] for z in done], axis=0))), unresolved


def fit_density(weight, interval, breaks=(), name="w"):
    """The piecewise Legendre series of a density weight on an interval (measures.DensityMeasure.density, made on
    first use: plan 069 P2, the measure holds its weight and the numerical column fits it on demand): split at the
    given breaks and the weight's own; RealisationError when the fit does not resolve it."""
    brk = sorted(set(breaks or ()) | {p[0] for p in weight.breaks})
    pl, unresolved = fit(lambda x: weight.values(np.asarray(x, float).reshape(-1, 1)).reshape(np.shape(x)),
                         interval.a, interval.b, breaks=brk or None)
    if unresolved or not np.all(np.isfinite(pl.err)):
        raise RealisationError(f"could not resolve the density {name} on {interval}")
    return pl

