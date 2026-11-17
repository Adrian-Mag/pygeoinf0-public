"""The grid-spectral route for weighted measures in 2-D (plan 022 P3, P4; experiments/EXP-038-weighted-energies).

A weighted measure w dA whose weight is smooth and vanishes smoothly at the boundary of its piece (a C^inf bump, a
Gaussian cut off far out, a product of such) has a Fourier transform that decays faster than any power. Then

    E_ij = int int K(x - y) w_i(x) w_j(y) dx dy = P^-2 sum_k Khat(xi_k) F_i(xi_k) conj(F_j(xi_k)),   xi_k = 2 pi k / P,

with F_i the trapezoid sum of w_i on a uniform lattice of spacing h (super-algebraically accurate for such weights:
Euler-Maclaurin), Khat = (1 + l^2 |xi|^2)^-s the spectrum of the kernel of H^s_l(R^2) and P a period larger than the
weights' extent by 2 reach (the images of K beyond reach, K < tol, do not meet the weights or the nodes they are
evaluated at). The weights decide the resolution: h is halved until every weight's unweighted indicator

    ind(h) = max_{|xi| >= pi / 2h} |F(xi)| / (h^2 sum |w|)

is below IND_MAX and falls by a factor RATE per halving (super-algebraic decay: a bump 0.05, 0.001; a weight with a jump
0.4, 0.4 and is refused). EXP-038 measured the energy error at 0.3 to 6.4 ind^2 (relative to the largest energy; radii
0.09 and 0.05, four levels each) down to a floor of 3e-12 to 1.4e-11 set by the period (the kernel's tail at reach); the
Khat-weighted share of the shell, tried first, under-estimates it by 500 and is not used.

Only the modes |xi| <= pi / 2h enter (the rest is below the indicator), and F is evaluated at them directly, by a
separable transform of the weight's own block of samples: no padded lattice transform. Since P4 the sum is split by a
spectral taper (weighted_split.py): a far part on one coarse lattice for all pairs and near parts for close pairs on small
lattices, each piece at its own level h, translated copies sharing their spectra; P3's single lattice (period E + 2 reach,
the finest weight's h for everyone) is gone. Energies of such a weight with points and segments (rays) come from its
potential k_i(y), far part by one batched type-2 NUFFT (finufft), near part on each type's small lattice; a weight the
indicator refuses, a piece that is not a disk or an axis-aligned box, any other kind of atom, or no finufft: the exact
routes, unchanged. What was used is recorded in `GridEngine.report`.

Limits (stated, not hidden): flat restriction construction (no reflection images, no periods), kernels of H^s_l(R^2) with
nu = 0 only through Khat as above, weights on disks or axis-aligned boxes; memory is bounded by settings.memory_gb.

From weighted_split.py (merged here by plan 063):
The far/near split of the grid-spectral route (plan 022 P4; experiments/EXP-038-weighted-energies).

weighted_grid.py (P3) put every weight on one lattice of period E + 2 reach with the spacing of the finest weight, so
the padding and the finest feature were paid by every weight, and narrow weights did not settle. Here the kernel is split
by a spectral taper, as spectral.py does for segments:

    Khat = Khat T + Khat (1 - T),   T(rho) = erfc((rho - ALPHA w) / w) / 2,   K = K_T + K_H

K_H = K - K_T has no long-range part (1 - T < 1e-17 below rho = ALPHA w - 5.5 w: the kernel's pole at |xi| = i / l lies in
T = 1) and decays like the taper's edge, exp(-(w r / 2)^2): EXP-038 measured |K_H(r)| < 1e-13 K(l) for r >= RANGE / w
(9.6 / w for ALPHA = 6, w = 20 ... 80; ALPHA = 5 leaks: 15 / w at w = 40), so r_H = RANGE / w with RANGE = 10.5.

    far    E_T for all pairs on one coarse lattice: modes |xi_k| <= Xi = TOP w (T < 1e-14 beyond), period E + reach (energies;
           E + 2 reach for potentials, as P3), each weight's transform at those modes from its own samples (separable).
    near   E_H only for pairs whose supports are closer than r_H, each type pair on a small lattice of period
           r_H + its span, at the finer of the two weights' resolutions (h per weight, chosen by P3's indicator).

Translation reuse: a weight is sampled on its own lattice, anchored at its piece's corner, so translated copies have equal
samples (to rounding: TYPE_TOL of the largest sample). Pieces with equal samples are one type, and E between two types is a
function of the offset of their anchors, evaluated at all the offsets of their member pairs at once:
E(D) = Re sum_k S_k e^{i xi_k . D}, S = Khat(.) F_a conj(F_b) / (P_x P_y), separable in x and y. Many different weights
take the general route instead (one matrix product over absolute spectra, as P3).

The split's r_H is chosen by a cost model over a few candidates (far modes and spectra against near type pairs). Errors:
the far sum is P3's sum at |xi| <= Xi (each weight sampled at least as finely as Xi needs), the near sums are P3's at the
pair's own h; so P3's measured bound (10 ind^2 + GRID_TOL, ind the largest indicator of the weights at their level) stands,
with the neglected K_H beyond r_H (< 1e-13 K(l) times the masses) inside GRID_TOL.
"""


import math

import numpy as np
from scipy import special

from ...core.errors import Unsupported
from .atoms import Kind

GRID_TOL = 1e-10        # relative energy error aimed at (the indicator bound below follows from it)
IND_MAX = math.sqrt(GRID_TOL / 10.0)   # EXP-038: error at most 6.4 ind^2 (radii 0.09 and 0.05, four levels each)
RATE = 0.2              # required decay of the indicator per halving of h (bumps 0.05 and below, jumps 0.4 and above)
H0 = 0.25               # the coarsest spacing tried, in kernel length scales
LEVELS = 7              # halvings tried (h down to l / 128 x ...): beyond it a weight is refused
PANEL = 0.25            # nodes along a segment: panels of this many length scales (or bump radii), 12 Gauss points each
MAX_BLOCK = 4e7         # samples of one weight at the finest level: beyond it the weight is refused (memory)


def _finufft():
    try:
        import finufft
    except ImportError:
        return None
    return finufft


def _reach(l, tol=GRID_TOL):
    """The distance beyond which K(r) / K(l) is below tol (K ~ e^(-r / l))."""
    return l * (math.log(1.0 / tol) + 3.0)


def _bbox(piece):
    from .weighted_volume import ParallelogramPiece, RoundPiece
    if isinstance(piece, RoundPiece) and piece.dim == 2:
        return piece.c - piece.R, piece.c + piece.R, piece.R
    if isinstance(piece, ParallelogramPiece):
        E = np.asarray(piece.E, float)
        if abs(E[0, 1]) < 1e-14 * abs(E[0, 0]) and abs(E[1, 0]) < 1e-14 * abs(E[1, 1]) and min(E[0, 0], E[1, 1]) > 0:
            return piece.o.copy(), piece.o + np.array([E[0, 0], E[1, 1]]), 0.5 * float(min(E[0, 0], E[1, 1]))
    return None


def _eligible_piece(piece):
    if getattr(piece, "w", None) is None or getattr(piece.w, "constant", None) is not None:
        return False
    return _bbox(piece) is not None


def _indicator(block, h):
    """max over the shell |xi| >= pi / 2h of |F| / (h^2 sum |w|), from the block zero-padded to twice its size."""
    v = block.vals
    mass = float(np.sum(np.abs(v)))
    if mass == 0.0:
        return 0.0
    n = [int(2 ** math.ceil(math.log2(2 * s))) for s in v.shape]
    F = np.abs(np.fft.fft2(v, s=n))
    kx, ky = (np.fft.fftfreq(m, d=h) * 2 * math.pi for m in n)
    shell = (kx[:, None] ** 2 + ky[None, :] ** 2) >= (math.pi / (2 * h)) ** 2
    return float(F[shell].max() / mass)


class GridEngine:
    """The route (module docstring). `energy(A, B)` returns the (A.n, B.n) matrix of energies, or None when no measure of
    A and B qualifies; `self_energies(A, idx)` the squared norms of some measures, or None."""

    def __init__(self, exact):
        self.exact = exact
        self.report = None
        self.indicator = 0.0

    # ------------------------------------------------------------------ what qualifies
    def applicable(self):
        from ...settings import settings
        from .kernels import Matern
        k = self.exact.kernel
        return (getattr(settings, "weighted_grid", "auto") != "off" and isinstance(k, Matern) and k.d == 2
                and self.exact.periods is None and self.exact.images is None)

    @staticmethod
    def _eligible(A):
        """Measures of A all of whose atoms are weighted pieces the route can sample: a boolean array (A.n,)."""
        ok = np.zeros(A.n, bool)
        seen = np.zeros(A.n, bool)
        for q in range(len(A.kind)):
            i = A.owner[q]
            if A.weight[q] == 0:
                continue
            good = A.kind[q] == Kind.WEIGHTED and A.piece[q] is not None and _eligible_piece(A.piece[q])
            ok[i] = good if not seen[i] else (ok[i] and good)
            seen[i] = True
        return ok & seen

    # ------------------------------------------------------------------ pieces
    def _pieces(self, A, idx):
        out = []
        for i in idx:
            out.append([(A.piece[q], float(A.weight[q])) for q in np.flatnonzero(A.owner == i) if A.weight[q] != 0])
        return out

    # ------------------------------------------------------------------ nodes of the other measures
    def _nodes(self, B, idx, l, rmin):
        """Nodes, weights and owners of the measures idx of B when all their atoms are points or segments, else None."""
        from ...quadrature import gauss
        g, gw = gauss(12)
        g, gw = 0.5 * (g + 1), 0.5 * gw
        X, W, own = [], [], []
        for i in idx:
            for q in np.flatnonzero(B.owner == i):
                if B.weight[q] == 0:
                    continue
                if B.kind[q] == Kind.POINT:
                    p = B.P[q][:2]
                    X.append(p[None]), W.append([B.weight[q]]), own.append([i])
                elif B.kind[q] == Kind.SEGMENT:
                    p, e = B.P[q][:2], B.E1[q][:2]
                    L = float(np.linalg.norm(e))
                    n = max(1, math.ceil(L / (PANEL * min(l, rmin))))
                    t = ((np.arange(n)[:, None] + g[None]) / n).ravel()
                    w = np.tile(gw / n, n) * L * B.weight[q]
                    X.append(p[None] + t[:, None] * e[None]), W.append(w), own.append(np.full(len(w), i))
                else:
                    return None
        if not X:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)
        return np.concatenate(X), np.concatenate(W), np.concatenate(own).astype(int)

    # ------------------------------------------------------------------ the matrix
    def energy(self, A, B):
        if not self.applicable():
            return None
        from ...core.errors import Unsupported
        from .weighted_grid import Split, Weights
        same = B is A
        ea = self._eligible(A)
        eb = ea if same else self._eligible(B)
        if not (ea.any() or eb.any()):
            return None
        kern = self.exact.kernel
        l = kern.l
        ia, ib = np.flatnonzero(ea), np.flatnonzero(eb)
        groups = self._pieces(A, ia) + ([] if same else self._pieces(B, ib))
        pieces = [p for g in groups for p, _ in g]
        if not pieces:
            return None
        W = Weights(pieces, l)                              # each piece's level by P3's indicator (EXP-038)
        own = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
        coef = np.array([c for g in groups for _, c in g])
        good = np.array([bool(np.all(W.level[own == k] >= 0)) for k in range(len(groups))])
        good_a, good_b = good[:len(ia)], (good[:len(ia)] if same else good[len(ia):])
        gi_a, gi_b = ia[good_a], ib[good_b]
        if not (len(gi_a) or len(gi_b)):
            self.report = {"route": "exact", "why": "no weight's indicator settled"}
            return None
        side = lambda ks: (np.flatnonzero(np.isin(own, ks)), ks)           # noqa: E731
        pa, ka = side(np.flatnonzero(good_a))
        pb, kb = (pa, ka) if same else side(len(ia) + np.flatnonzero(good_b))
        Ca = np.zeros((len(ka), len(pa)))
        Ca[np.searchsorted(ka, own[pa]), np.arange(len(pa))] = coef[pa]
        Cb = Ca if same else np.zeros((len(kb), len(pb)))
        if not same:
            Cb[np.searchsorted(kb, own[pb]), np.arange(len(pb))] = coef[pb]
        allp = np.union1d(pa, pb)
        sp = Split(kern, W, pa if len(pa) else pb, pb if len(pb) else pa)
        self.indicator = sp.indicator(allp)
        E = np.zeros((A.n, B.n))
        done = np.zeros(E.shape, bool)
        if len(gi_a) and len(gi_b):
            try:
                E[np.ix_(gi_a, gi_b)] = Ca @ sp.energies(same) @ Cb.T
            except Unsupported as e:          # memory (weighted_split._check_memory): the exact routes
                self.report = {"route": "exact", "why": f"memory (settings.memory_gb): {e}"}
                return None
            done[np.ix_(gi_a, gi_b)] = True
        self.report = dict(route="grid", indicator=self.indicator, weights=len(gi_a) + (0 if same else len(gi_b)),
                           levels=sorted({int(m) for m in W.level[allp]}), **sp.report)
        okA = np.zeros(A.n, bool)
        okA[gi_a] = True
        okB = okA if same else np.zeros(B.n, bool)
        okB[gi_b] = True
        unsup_B = np.zeros(B.n, bool)         # measures of the other side the potentials cannot reach: exact routes
        unsup_A = np.zeros(A.n, bool)
        rmin = float(min(W.box[i][2] for i in allp))
        for gi, C, pidx, other, ok_o, unsup, mirror in (
                (gi_a, Ca, pa, B, okB, unsup_B, False), (gi_b, Cb, pb, A, okA, unsup_A, True)):
            if mirror and same:
                break
            cand = np.flatnonzero(~ok_o)
            sup = np.array([i for i in cand if self._nodes(other, [i], l, rmin) is not None] if _finufft() else [], int)
            unsup[np.setdiff1d(cand, sup)] = True
            if not (len(gi) and len(sup)):
                continue
            X, Wn, own_n = self._nodes(other, sup, l, rmin)
            V = C @ sp.potentials(pidx, X, _finufft())
            block = np.stack([np.bincount(own_n, weights=Wn * v, minlength=other.n) for v in V])
            if not mirror:
                E[np.ix_(gi, sup)] = block[:, sup]
                done[np.ix_(gi, sup)] = True
                if same:
                    E[np.ix_(sup, gi)] = block[:, sup].T
                    done[np.ix_(sup, gi)] = True
            else:
                E[np.ix_(sup, gi)] = block[:, sup].T
                done[np.ix_(sup, gi)] = True
        if same:
            unsup_A = unsup_B                      # the transposed block is exact too
        self._exact_rest(E, done, A, B, okA, okB, unsup_A, unsup_B)
        return E

    def _exact_rest(self, E, done, A, B, okA, okB, unsup_A, unsup_B):
        """The pairs left (three rectangles, disjoint): the exact engine on subsets of the measures."""
        rects = [(~okA, ~okB), (okA, ~okB & unsup_B), (~okA & unsup_A, okB)]
        for rows, cols in rects:
            ra, rb = np.flatnonzero(rows), np.flatnonzero(cols)
            if not (len(ra) and len(rb)):
                continue
            E[np.ix_(ra, rb)] = self.exact.measure_energy(A.subset(ra), B.subset(rb))[0]
            done[np.ix_(ra, rb)] = True

    def bound(self):
        """The relative error aimed at for the last grid energies: 10 ind^2 (EXP-038, measured) plus GRID_TOL for the
        period's kernel tail; relative to sqrt(E_ii E_jj) for a pair of weights."""
        return 10.0 * (self.report or {}).get("indicator", 0.0) ** 2 + GRID_TOL

    def claim(self):
        """A Claim for the last use of the grid route (None when it was not used): its settings and their basis."""
        from ...core.claims import CHECKED, Claim
        rp = self.report
        if not rp or rp.get("route") != "grid":
            return None
        return Claim(f"energies of {rp['weights']} weighted measures by the grid-spectral route with a far/near split "
                     f"(levels h = {H0} l 2^-m, m in {rp['levels']}; r_H = {rp['r_H']:.3g}, far modes {rp['far_modes']})",
                     CHECKED,
                     f"each weight's unweighted shell indicator is below {IND_MAX:.1e} and falls by a factor of at most "
                     f"{RATE} per halving of h (EXP-038: energy error at most 6.4 ind^2, floor ~1e-11 from the period); "
                     f"the short-range kernel K_H is below 1e-13 K(l) beyond r_H (EXP-038, measured); "
                     f"error bound {self.bound():.1e} of sqrt(E_ii E_jj)",
                     key="weighted grid")

    # ------------------------------------------------------------------ norms
    def self_energies(self, A, idx):
        """(||k_mu_i||^2, error estimates) of the measures idx of A when every one qualifies, else None."""
        if not self.applicable():
            return None
        idx = np.asarray(idx, int)
        u, back = np.unique(idx, return_inverse=True)
        if not self._eligible(A)[u].all():
            return None
        S = A.subset(u)
        E = self.energy(S, S)
        if E is None or self.report.get("route") != "grid":
            return None
        d = np.abs(np.diag(E))
        return d[back], (self.bound() * d)[back]


def khat(kern, xi2):
    """The Fourier transform (1 + l^2 |xi|^2)^-s of the kernel of H^s_l (realisations/rkhs/kernels.Matern's convention)."""
    return (1.0 + kern.l ** 2 * np.asarray(xi2, float)) ** (-kern.s)


# ---------------------------------------------------------------- from weighted_split.py (plan 063)
ALPHA, RANGE, TOP = 6.0, 10.5, 11.5     # taper centre a = ALPHA w; r_H = RANGE / w; modes up to TOP w (module docstring)
TYPE_TOL = 1e-13         # samples equal to this fraction of the largest: translated copies, one type
MIN_SAMPLES = 6          # a block needs this many samples per axis to be judged (P3 passed blocks with none: EXP-038)
CANDIDATES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)   # r_H tried, in kernel length scales
OFFSETS = 4096           # offsets evaluated at once
SHARE = 4                # offsets route when there are at most n / SHARE types (else one product over all spectra)
FAR_TOL = 1e-13          # the far lattice's period: extent + reach(FAR_TOL) (K(reach) / K(l) < FAR_TOL; P3: 2 reach(1e-10))


def _check_memory(nbytes):
    from ...settings import settings
    if nbytes > 0.4e9 * float(settings.memory_gb):
        raise Unsupported(f"{nbytes / 1e9:.1f} GB of far spectra (0.4 settings.memory_gb allowed)")


def h_of(ell, m):
    return H0 * ell * 0.5 ** m


def taper(w, rho):
    return 0.5 * special.erfc((np.asarray(rho, float) - ALPHA * w) / w)


class Block:
    """A piece's weight on its own lattice anchor + h m (m >= 0), cropped to its nonzero samples: x0 is the first sample's
    position, vals the samples h^2 w, s the support relative to x0 (the samples' extent plus 2 h, a conservative margin),
    t its type at this level."""

    __slots__ = ("x0", "vals", "h", "s", "t")

    def __init__(self, x0, vals, h):
        self.x0, self.vals, self.h = x0, vals, h
        n = np.array(vals.shape)
        self.s = np.stack([np.full(2, -2 * h), (n - 1) * h + 2 * h])       # (2, 2): rows lo, hi
        self.t = None


class Weights:
    """The pieces of one energy computation: their levels (P3's indicator, per piece), blocks and types per level."""

    def __init__(self, pieces, ell):
        self.pieces, self.l = pieces, ell
        self.box = [_bbox(p) for p in pieces]
        self._blocks, self._types, self._ind = {}, {}, {}
        self.level = np.full(len(pieces), -1)
        self.ind = np.zeros(len(pieces))
        for i in range(len(pieces)):
            self._resolve(i)

    def block(self, i, m):
        key = (i, m)
        if key not in self._blocks:
            self._blocks[key] = self._sample(i, m)
        return self._blocks[key]

    def _sample(self, i, m):
        from .weighted_volume import RoundPiece
        piece, (lo, hi, _) = self.pieces[i], self.box[i]
        h = h_of(self.l, m)
        n = np.floor((hi - lo) / h + 1e-9).astype(int) + 1
        if np.prod(n.astype(float)) > MAX_BLOCK:
            return None
        gx, gy = (lo[a] + h * np.arange(n[a]) for a in (0, 1))
        X = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
        v = np.asarray(piece.w(X), float)
        if isinstance(piece, RoundPiece):
            v = np.where(np.sum((X - piece.c) ** 2, axis=1) <= piece.R ** 2 * (1 + 1e-12), v, 0.0)
        v = (h * h) * v.reshape(n[0], n[1])
        rows, cols = np.flatnonzero(np.any(v != 0, axis=1)), np.flatnonzero(np.any(v != 0, axis=0))
        if not len(rows):
            return None
        b = Block(lo + h * np.array([rows[0], cols[0]]), v[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], h)
        b.t = self._type(m, b.vals)
        return b

    def _type(self, m, v):
        """The type of samples v at level m: equal (to TYPE_TOL) to an earlier block's of the same shape."""
        top = float(np.max(np.abs(v)))
        key = (m, v.shape, float(f"{np.sum(v) / top:.8e}"))
        reps = self._types.setdefault(key, [])
        for t, rv in reps:
            if np.max(np.abs(v - rv)) <= TYPE_TOL * top:
                return t
        t = (m, len(self._types), len(reps))
        reps.append((t, v))
        return t

    def indicator(self, b):
        if b.t not in self._ind:
            self._ind[b.t] = _indicator(b, b.h)
        return self._ind[b.t]

    def _resolve(self, i):
        """The coarsest level at which the piece's indicator is below IND_MAX and falls by RATE per halving (P3's test),
        with at least MIN_SAMPLES samples per axis; -1 when no level passes (the piece is refused)."""
        for m in range(LEVELS + 1):
            b1, b2 = self.block(i, m), self.block(i, m - 1)
            if b1 is None or b2 is None or min(b1.vals.shape) < MIN_SAMPLES or min(b2.vals.shape) < 3:
                continue
            i1, i2 = self.indicator(b1), self.indicator(b2)
            if i1 <= IND_MAX and (i1 <= RATE * i2 or i1 < 1e-3 * IND_MAX):
                self.level[i], self.ind[i] = m, i1
                return

    def support(self, i, m):
        """The piece's support box (2, 2) at level m: its block's extent with the margin, inside its bounding box."""
        b = self.block(i, m)
        lo, hi, _ = self.box[i]
        return np.stack([np.maximum(b.x0 + b.s[0], lo), np.minimum(b.x0 + b.s[1], hi)])


def spectrum(b, kx, ky, origin=None):
    """F(xi) = sum_m h^2 w(x) e^{-i xi . x} over the block's samples, x relative to b.x0 (or to origin when given):
    (len(kx), len(ky)) at the frequencies kx, ky (arrays of xi along each axis), by a separable transform."""
    nx, ny = b.vals.shape
    ox, oy = (0.0, 0.0) if origin is None else b.x0 - origin
    Wx = np.exp(-1j * np.outer(kx, ox + b.h * np.arange(nx)))
    Wy = np.exp(-1j * np.outer(oy + b.h * np.arange(ny), ky))
    return (Wx @ b.vals) @ Wy


def lattice(P, cut, half=True):
    """Frequencies 2 pi k / P per axis with |xi| <= cut (k from -K to K; the second axis from 0 when half), and the
    weights of the half plane (2 for ky > 0: the sum over -xi is the conjugate)."""
    K = [max(1, math.ceil(cut * p / (2 * math.pi))) for p in P]
    kx = np.arange(-K[0], K[0] + 1) * (2 * math.pi / P[0])
    ky = np.arange(0 if half else -K[1], K[1] + 1) * (2 * math.pi / P[1])
    wy = np.where(ky > 0, 2.0, 1.0) if half else np.ones(len(ky))
    return kx, ky, wy


def at_offsets(S, kx, ky, D):
    """Re sum_{kx, ky} S e^{i (kx D_x + ky D_y)} for each row of D (n, 2), in blocks of OFFSETS."""
    out = np.empty(len(D))
    for a in range(0, len(D), OFFSETS):
        d = D[a:a + OFFSETS]
        T = S @ np.exp(1j * np.outer(ky, d[:, 1]))
        out[a:a + OFFSETS] = np.real(np.sum(np.exp(1j * np.outer(kx, d[:, 0])) * T, axis=0))
    return out


def _unique_offsets(D):
    """Distinct rows of D (to 1e-15) and the inverse map. The values returned are members' own offsets: evaluating at a
    rounded key moved E by dE/dD times the rounding (1e-12 rounding: 3.8e-12 of the largest energy, EXP-038)."""
    _, first, inv = np.unique(np.round(D, 15), axis=0, return_index=True, return_inverse=True)
    return D[first], inv.ravel()


class Split:
    """Energies and potentials of resolved pieces (Weights) by the split (module docstring). r_H is chosen per call."""

    def __init__(self, kernel, W, ia, ib):
        self.kern, self.W, self.l = kernel, W, kernel.l
        self.ia, self.ib = np.asarray(ia, int), np.asarray(ib, int)
        self.reach = _reach(self.l, FAR_TOL)
        self.report = {}
        self.rH = self._choose()
        self.w = RANGE / self.rH
        self.Xi = TOP * self.w
        self.m_top = next(m for m in range(-1, 64) if math.pi / (2 * h_of(self.l, m)) >= self.Xi)

    # ------------------------------------------------------------------ levels used
    def far_level(self, i):
        """One level finer than the piece's own: the far sum takes the trapezoid's error at low frequency in first order
        (EXP-038: 7.7e-12 in F(0) at the author's level 4, 2.7e-12 in E_T; 3e-15 one level finer), the near sums only
        at high frequency; and at least as fine as Xi needs."""
        return max(int(self.W.level[i]) + 1, self.m_top)

    def _extent(self, idx, m_of):
        sup = np.array([self.W.support(i, m_of(i)) for i in idx])
        return sup[:, 0].min(axis=0), sup[:, 1].max(axis=0)

    # ------------------------------------------------------------------ the choice of r_H
    def _near_pairs(self, rH, same):
        """Pairs (i of ia, j of ib) whose supports are closer than rH (i <= j when same)."""
        Sa = np.array([self.W.support(i, self.W.level[i]) for i in self.ia])
        Sb = Sa if same else np.array([self.W.support(j, self.W.level[j]) for j in self.ib])
        gap = np.maximum(0.0, np.maximum(Sb[None, :, 0] - Sa[:, None, 1], Sa[:, None, 0] - Sb[None, :, 1]))
        near = np.sum(gap ** 2, axis=-1) < rH * rH
        if same:
            near &= np.triu(np.ones(near.shape, bool))
        return np.argwhere(near)

    def _choose(self):
        """The candidate r_H (CANDIDATES l) of least estimated work (complex multiply-adds): the far spectra of each type
        and the far sums (offsets, or one product), the near spectra and sums of each near type pair."""
        W, ell, same = self.W, self.l, np.array_equal(self.ia, self.ib)
        allp = np.union1d(self.ia, self.ib)
        lv = {i: int(W.level[i]) for i in allp}
        t0 = {i: W.block(i, lv[i]).t for i in allp}
        nt = len(set(t0.values()))
        offs = len(_unique_offsets(np.array([[hash(t0[i]), hash(t0[j]), *(W.box[j][0] - W.box[i][0])]
                                              for i in self.ia for j in self.ib]))[0]) if nt <= len(allp) / SHARE else 0
        best = None
        for c in CANDIDATES:
            rH = c * ell
            Xi = TOP * RANGE / rH
            m_top = next(m for m in range(-1, 64) if math.pi / (2 * h_of(ell, m)) >= Xi)
            lo, hi = self._extent(allp, lambda i: max(lv[i] + 1, m_top))
            K = [math.ceil(Xi * (p + self.reach) / (2 * math.pi)) for p in hi - lo]
            M = (2 * K[0] + 1) * (K[1] + 1)
            spec = {}
            for i in allp:
                n = np.array(W.block(i, lv[i]).vals.shape) * 2.0 ** max(1, m_top - lv[i])
                spec[t0[i] if offs else i] = (2 * K[0] + 1) * n[0] * (n[1] + K[1] + 1)
            far = sum(spec.values()) + M * (offs if offs else len(self.ia) * len(self.ib))
            near = {}
            for i, j in self._near_pairs(rH, same):
                a, b = self.ia[i], self.ib[j]
                m = max(lv[a], lv[b])
                ba, bb = W.block(a, m), W.block(b, m)
                n = np.array(ba.vals.shape) + np.array(bb.vals.shape)
                Kn = (n / 2 + rH / ba.h) / 2                       # P / 4h per axis, P ~ span + 2 rH
                key = (ba.t, bb.t)
                near[key] = near.get(key, 4 * Kn[0] * Kn[1] * (1 + n.sum())) + 2 * Kn[0] * Kn[1]
            cost = far + sum(near.values())
            if best is None or cost < best[0]:
                best = (cost, rH)
        self.report["cost"] = float(best[0])
        return best[1]

    # ------------------------------------------------------------------ energies
    def energies(self, same):
        """E (len(ia), len(ib)) between the pieces ia and ib: far for all pairs, near for the close ones."""
        E = self._far(same)
        self._near(E, same)
        return E

    def _far(self, same):
        W = self.W
        allp = np.union1d(self.ia, self.ib)
        lo, hi = self._extent(allp, self.far_level)
        P = hi - lo + self.reach
        kx, ky, wy = lattice(P, self.Xi)
        Kt = khat(self.kern, kx[:, None] ** 2 + ky[None, :] ** 2) * taper(self.w, np.hypot(kx[:, None], ky[None, :]))
        Kt *= wy[None, :] / (P[0] * P[1])
        ba = [W.block(i, self.far_level(i)) for i in self.ia]
        bb = ba if same else [W.block(j, self.far_level(j)) for j in self.ib]
        types = {b.t for b in ba + bb}
        n = len(allp)
        self.report.update(far_modes=(len(kx), len(ky)), far_period=P.tolist(), far_types=len(types))
        many = len(types) > n / SHARE
        _check_memory(16.0 * Kt.size * ((len(ba) + (0 if same else len(bb))) if many else len(types)))
        if many:                                    # many different weights: one product over absolute spectra
            FA = np.stack([spectrum(b, kx, ky, origin=lo).ravel() for b in ba])
            FB = FA if same else np.stack([spectrum(b, kx, ky, origin=lo).ravel() for b in bb])
            return np.real((FA * Kt.ravel()[None]) @ FB.conj().T)
        spec = {}
        for b in ba + bb:
            if b.t not in spec:
                spec[b.t] = spectrum(b, kx, ky)
        E = np.zeros((len(ba), len(bb)))
        x0a = np.array([b.x0 for b in ba])
        x0b = x0a if same else np.array([b.x0 for b in bb])
        for a in {b.t for b in ba}:
            rows = np.array([k for k, q in enumerate(ba) if q.t == a])
            for b in {q.t for q in bb}:
                cols = np.array([k for k, q in enumerate(bb) if q.t == b])
                S = Kt * spec[a] * np.conj(spec[b])
                D = (x0b[cols][None, :, :] - x0a[rows][:, None, :]).reshape(-1, 2)
                u, inv = _unique_offsets(D)
                E[np.ix_(rows, cols)] = at_offsets(S, kx, ky, u)[inv].reshape(len(rows), len(cols))
        return E

    def _near(self, E, same):
        W, rH = self.W, self.rH
        pairs = self._near_pairs(rH, same)
        groups = {}
        for i, j in pairs:
            a, b = self.ia[i], self.ib[j]
            m = max(int(W.level[a]), int(W.level[b]))
            ba, bb = W.block(a, m), W.block(b, m)
            groups.setdefault((ba.t, bb.t), []).append((i, j, ba, bb))
        self.report.update(near_pairs=len(pairs), near_groups=len(groups), r_H=rH, Xi=self.Xi)
        for members in groups.values():
            ba, bb = members[0][2], members[0][3]
            D = np.array([q[3].x0 - q[2].x0 for q in members])
            # images of the pair at D + n P must be farther apart than rH (module docstring): per axis
            P = rH + np.maximum(D.max(axis=0) + bb.s[1] - ba.s[0], ba.s[1] - D.min(axis=0) - bb.s[0])
            P = P * (1 + 1e-9) + 1e-12
            kx, ky, wy = lattice(P, math.pi / (2 * ba.h))
            rho2 = kx[:, None] ** 2 + ky[None, :] ** 2
            Kh = khat(self.kern, rho2) * (1.0 - taper(self.w, np.sqrt(rho2))) * wy[None, :] / (P[0] * P[1])
            S = Kh * spectrum(ba, kx, ky) * np.conj(spectrum(bb, kx, ky))
            u, inv = _unique_offsets(D)
            v = at_offsets(S, kx, ky, u)[inv]
            rows = np.array([q[0] for q in members])
            cols = np.array([q[1] for q in members])
            E[rows, cols] += v
            if same:
                off = rows != cols
                E[cols[off], rows[off]] += v[off]

    # ------------------------------------------------------------------ potentials
    def potentials(self, idx, Y, fin):
        """k_i(y) = int K(y - x) w_i(x) dx of the pieces idx at points Y (m, 2): (len(idx), m), far by one batched type-2
        NUFFT on the coarse lattice (period extent + 2 reach), near on each type's small lattice."""
        W = self.W
        idx = np.asarray(idx, int)
        out = np.zeros((len(idx), len(Y)))
        if not len(Y):
            return out
        lo, hi = self._extent(idx, self.far_level)
        P = hi - lo + 2 * self.reach
        kx, ky, _ = lattice(P, self.Xi, half=False)
        rho2 = kx[:, None] ** 2 + ky[None, :] ** 2
        Kt = khat(self.kern, rho2) * taper(self.w, np.sqrt(rho2)) / (P[0] * P[1])
        inside = np.flatnonzero(np.all((Y >= lo - self.reach) & (Y <= hi + self.reach), axis=1))
        t = 2 * math.pi * (Y[inside] - lo[None]) / P[None]
        tx, ty = (np.ascontiguousarray(t[:, k]) for k in (0, 1))
        for a in range(0, len(idx), 4) if len(inside) else ():
            bs = [W.block(i, self.far_level(i)) for i in idx[a:a + 4]]
            Fs = np.stack([spectrum(b, kx, ky, origin=lo) * Kt for b in bs])      # lattice origin lo
            c = fin.nufft2d2(tx, ty, np.ascontiguousarray(Fs), eps=1e-13, isign=1)
            out[a:a + len(bs), inside] = np.real(c).reshape(len(bs), -1)
        groups = {}
        for r, i in enumerate(idx):
            b = W.block(i, int(W.level[i]))
            sup = W.support(i, int(W.level[i]))
            gap = np.maximum(0.0, np.maximum(sup[0][None] - Y, Y - sup[1][None]))
            near = np.flatnonzero(np.sum(gap ** 2, axis=1) < self.rH ** 2)
            if len(near):
                groups.setdefault(b.t, []).append((r, b, near))
        for members in groups.values():
            b = members[0][1]
            P = (b.s[1] - b.s[0]) + 2 * self.rH
            kx, ky, _ = lattice(P, math.pi / (2 * b.h), half=False)
            rho2 = kx[:, None] ** 2 + ky[None, :] ** 2
            S = khat(self.kern, rho2) * (1.0 - taper(self.w, np.sqrt(rho2))) * spectrum(b, kx, ky) / (P[0] * P[1])
            R = np.concatenate([Y[near] - q.x0[None] for _, q, near in members])
            t = 2 * math.pi * R / P[None]
            c = np.real(fin.nufft2d2(np.ascontiguousarray(t[:, 0]), np.ascontiguousarray(t[:, 1]),
                                     np.ascontiguousarray(S), eps=1e-13, isign=1))
            k = 0
            for r, _, near in members:
                out[r, near] += c[k:k + len(near)]
                k += len(near)
        return out

    def indicator(self, idx):
        return float(np.max(self.W.ind[np.asarray(idx, int)])) if len(idx) else 0.0
