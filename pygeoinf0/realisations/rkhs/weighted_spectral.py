"""Weighted pieces in the spectral engine (EXP-031 W4, DESIGN 5.3): node rules that integrate e^{-i xi.y} w(y) dsigma(y)
for |xi| <= kmax on every kind of piece, and the rate at which the far field converges without near corrections.

    piece           nodes (Q = 16 Gauss points per panel of half-length OMEGA / kmax, OMEGA = 8 here)
    segment         panels along the segment
    curve           panels in arc length (t = t(s))
    sphere          Gauss panels in cos(theta) x trapezoid in phi, 2 kmax R + 32 points
    parallelogram   tensor panels along both edges
    triangle        the collapsed Gauss-Jacobi rule of order Q times the panels along its diameter
    disk, ball      radial panels x (trapezoid in phi | Gauss panels in cos(theta) x trapezoid in phi)

Without near corrections the omitted part int Khat (1 - T) |mu^|^2 of a piece of dimension k in R^d whose weight
jumps at its boundary falls like Xi^-p with p = 2 beta = 2s - (d - k) for k < d (the transform does not decay normal
to the piece) and p = 2s + 1 for k = d (the jump at the boundary: |mu^|^2 ~ |xi|^-(d+1) on average; DESIGN 5.3,
measured 2.5-5.3 in P5 B); for weights smooth on R^d with compact support (bumps) it falls faster than any power.
uncorrected_rate() gives the smallest p of the set.

Near corrections (EXP-031 O2): for weighted pieces whose omitted part converges slowly, the pairs closer than r_corr
(the piece with itself included) get C = E_K - E_{K_T}, E_K from the W3 routes (pairs.energies, cached per plan),
E_{K_T} by the double sum of the node rules with the tabulated tapered kernel (spectral_terms.TaperedTable, ~2e-8):
every piece of lower dimension (its transform does not decay normal to it), and pieces of full dimension whose weight
does not vanish on their boundary (sampled: a jump there). C^inf weights vanishing at the boundary (bumps) need none.
The estimate (observed_q): the two-resolution difference divided by r^p - 1 with p the smaller of the stated rate and
the rate observed between l Xi / 2, l Xi / sqrt 2 and l Xi, so that a convergence slower than stated (pre-asymptotic,
or bumps between their regimes) does not make it under-state, times a safety factor SAFETY = 2 (measured need), and
never below 10 eps (the transforms' tolerance; at that floor differences between resolutions are rounding).
"""
import math

import numpy as np

from ...quadrature import gauss01, simplex
from .spectral_terms import Q

OMEGA = 8.0     # half-panel phase: J_32(8) ~ 1e-15 (segments use 13, ~1e-9: spectral_terms); measured on the bump family


def _npan(length, kmax):
    return max(1, int(math.ceil(kmax * length / (2 * OMEGA))))


def _panels01(npan):
    t, w = gauss01(Q)
    j = np.arange(npan)
    return ((j[:, None] + t[None]) / npan).ravel(), np.tile(w / npan, npan)


def piece_nodes(piece, kmax):
    """(X (N, d), W (N,)) for the piece (its weight folded in)."""
    from .weighted import CurvePiece, SegmentPiece, SpherePiece
    from .weighted_volume import ParallelogramPiece, RoundPiece, TrianglePiece
    if isinstance(piece, SegmentPiece):
        t, w = _panels01(_npan(piece.L, kmax))
        return piece.p + (piece.L * t)[:, None] * piece.u, piece.L * w * piece.wt(piece.L * t)
    if isinstance(piece, CurvePiece):
        C = piece.c
        s, w = _panels01(_npan(C.length, kmax))
        t = C.parameter(C.length * s)
        X = C.point(t)
        return X, C.length * w * piece.w(X)
    if isinstance(piece, SpherePiece):
        R = piece.R
        ct, wt = _panels01(_npan(2 * R, kmax))
        ct, wt = 2 * ct - 1, 2 * wt
        nphi = int(2 * kmax * R + 32)
        phi = 2 * math.pi * np.arange(nphi) / nphi
        st = np.sqrt(1 - ct * ct)
        U = np.stack([(st[:, None] * np.cos(phi)).ravel(), (st[:, None] * np.sin(phi)).ravel(), np.repeat(ct, nphi)], 1)
        X = piece.c + R * U
        return X, np.repeat(wt, nphi) * 2 * math.pi / nphi * R * R * piece.w(X)
    if isinstance(piece, TrianglePiece):
        U, W = simplex(2, Q * _npan(piece.diam, kmax))
        X = piece.o + U @ piece.E
        return X, piece.det * W * piece.w(X)
    if isinstance(piece, ParallelogramPiece):
        a, wa = _panels01(_npan(np.linalg.norm(piece.E[0]), kmax))
        b, wb = _panels01(_npan(np.linalg.norm(piece.E[1]), kmax))
        A, B = (g.reshape(-1, 1) for g in np.meshgrid(a, b, indexing="ij"))
        X = piece.o + A * piece.E[0] + B * piece.E[1]
        return X, piece.det * np.outer(wa, wb).ravel() * piece.w(X)
    from .weighted_solids import TetraPiece
    from .weighted_solids import ShellPiece
    if isinstance(piece, (TetraPiece, ShellPiece)):              # their rule, of an order resolving kmax
        n = Q * max(1, _npan(piece.diam, kmax))
        return piece.rule(min(n, 64))
    from .weighted_solids import ParallelepipedPiece
    if isinstance(piece, ParallelepipedPiece):
        rules = [_panels01(_npan(np.linalg.norm(e), kmax)) for e in piece.E]
        G = np.meshgrid(*[r[0] for r in rules], indexing="ij")
        X = piece.o + np.stack([g.ravel() for g in G], 1) @ piece.E
        W = np.einsum("i,j,k->ijk", *[r[1] for r in rules]).ravel()
        return X, piece.det * W * piece.w(X)
    if isinstance(piece, RoundPiece):
        R, d = piece.R, piece.dim
        r, wr = _panels01(max(3, _npan(R, kmax)))         # at least 48 radial points: a C^inf bump needs them (P1)
        r, wr = R * r, R * wr * (R * r) ** (d - 1)
        nphi = int(2 * kmax * R + 32)
        phi = 2 * math.pi * np.arange(nphi) / nphi
        if d == 2:
            U, Wu = np.stack([np.cos(phi), np.sin(phi)], 1), np.full(nphi, 2 * math.pi / nphi)
        else:
            ct, wt = _panels01(_npan(2 * R, kmax))
            ct, wt = 2 * ct - 1, 2 * wt
            st = np.sqrt(1 - ct * ct)
            U = np.stack([(st[:, None] * np.cos(phi)).ravel(), (st[:, None] * np.sin(phi)).ravel(),
                          np.repeat(ct, nphi)], 1)
            Wu = np.repeat(wt, nphi) * 2 * math.pi / nphi
        X = piece.c + (r[:, None, None] * U[None]).reshape(-1, d)
        return X, np.outer(wr, Wu).ravel() * piece.w(X)
    raise TypeError(f"no spectral node rule for {type(piece).__name__}")


def weighted_nodes(A, idx, kmax):
    """Nodes, weights (times the atoms' coefficients) and owners of the weighted atoms idx of the atom set A."""
    X, W, own = [], [], []
    for q in idx:
        x, w = piece_nodes(A.piece[q], kmax)
        keep = w != 0
        X.append(x[keep])
        W.append(w[keep] * A.weight[q])
        own.append(np.full(int(keep.sum()), A.owner[q]))
    return np.vstack(X), np.concatenate(W), np.concatenate(own)


def split(A, kernel, segment_rate):
    """(segment atoms, weighted atoms, the rate p of the set, the corners of the weighted pieces' bounding boxes)."""
    from .atoms import Kind
    sidx, widx = np.flatnonzero(A.kind == Kind.SEGMENT), np.flatnonzero(A.kind == Kind.WEIGHTED)
    p = segment_rate if len(sidx) else math.inf
    box = np.zeros((0, A.P.shape[1]))
    if len(widx):
        p = min(p, uncorrected_rate(kernel, A, widx))
        r = A.radius[widx, None]
        box = np.vstack([A.P[widx] - r, A.P[widx] + r])
    return sidx, widx, p, box


def uncorrected_rate(kernel, A, idx):
    """The smallest p of the weighted atoms idx (module docstring)."""
    d = kernel.d
    ks = {A.piece[q].dim for q in idx}
    return min((2 * kernel.s + 1) if k == d else (2 * kernel.s - (d - k)) for k in ks)


SAFETY = 2.0    # the rate between consecutive resolutions varied by up to 2x on the bump family (N = 50, 1e3): with
                # the observed rate alone the estimate was 0.8-1.0 x the actual error at l Xi = 12-17; with this factor
                # it was 1.6-16 x (NOTES of MISSION_20261027_INFERENCES_wmeas_open_items)


def _boundary(piece, n=64):
    """Points on the relative boundary of a piece of full dimension (where a jump of the weight would be)."""
    from .weighted_volume import ParallelogramPiece, RoundPiece
    t = (np.arange(n) + 0.5) / n
    if isinstance(piece, ParallelogramPiece):
        V = piece.V
        return np.vstack([V[k] + np.outer(t, V[(k + 1) % len(V)] - V[k]) for k in range(len(V))])
    from .weighted_solids import ParallelepipedPiece
    if isinstance(piece, ParallelepipedPiece):                        # points on the six faces
        g = (np.arange(16) + 0.5) / 16
        F = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        pts = [np.insert(F, k, side, axis=1) for k in range(3) for side in (0.0, 1.0)]
        return piece.o + np.vstack(pts) @ piece.E
    if isinstance(piece, RoundPiece):
        if piece.dim == 2:
            a = 2 * math.pi * t
            return piece.c + piece.R * np.stack([np.cos(a), np.sin(a)], 1)
        U = np.random.default_rng(0).standard_normal((4 * n, 3))
        return piece.c + piece.R * U / np.linalg.norm(U, axis=1)[:, None]
    return None


def needs_correction(piece, d):
    """Pieces of lower dimension always; of full dimension when the weight does not vanish on the boundary (CHECKED
    on samples: below 1e-12 of its maximum there counts as vanishing)."""
    if piece.dim < d:
        return True
    B = _boundary(piece)
    if B is None:
        return True
    X, _ = piece.formula_nodes()
    return bool(np.max(np.abs(piece.w(B))) > 1e-12 * max(float(np.max(np.abs(piece.w(X)))), 1e-300))


def _weighted_pairs(plan):
    """(i, j, E_K) for the weighted atoms that need corrections: i <= j closer than r_corr (bounding balls)."""
    if getattr(plan, "_wpairs", None) is None:
        from . import pairs
        A, d = plan.A, plan.kernel.d
        need = np.array([needs_correction(A.piece[q], d) for q in plan.widx], bool)
        idx = plan.widx[need]
        C, R = A.P[idx], A.radius[idx]
        D = np.linalg.norm(C[:, None] - C[None], axis=-1) - R[:, None] - R[None]
        a, b = np.nonzero(np.triu(D < plan.rc))
        i, j = idx[a], idx[b]
        eK = pairs.energies(plan.kernel, A, i, A, j)[0] if len(i) else np.zeros(0)
        plan._wpairs = (i, j, eK)
        plan.stats.update(weighted_pairs=len(i), weighted_corrected=int(need.sum()))
    return plan._wpairs


def weighted_corrections(plan, L):
    """The sparse (n, n) corrections of the weighted near pairs at the resolution L (module docstring)."""
    import scipy.sparse as sps
    from .spectral_terms import TaperedTable
    i, j, eK = _weighted_pairs(plan)
    A, n = plan.A, plan.A.n
    if not len(i):
        return sps.csr_matrix((n, n))
    rmax = float(np.max(np.linalg.norm(A.P[i] - A.P[j], axis=1) + A.radius[i] + A.radius[j])) * 1.01 + 1e-12
    table = TaperedTable(plan.kernel, L.Xi, rmax)
    nodes = {q: piece_nodes(A.piece[q], 1.05 * L.Xi) for q in np.unique(np.r_[i, j])}
    eT = np.empty(len(i))
    for m, (a, b) in enumerate(zip(i, j)):
        (Xa, Wa), (Xb, Wb) = nodes[a], nodes[b]
        tot = 0.0
        for c in range(0, len(Xa), 512):
            tot += float(Wa[c:c + 512] @ table(np.linalg.norm(Xa[c:c + 512, None] - Xb[None], axis=-1)) @ Wb)
        eT[m] = tot
    e = (eK - eT) * A.weight[i] * A.weight[j]
    oa, ob, off = A.owner[i], A.owner[j], i != j
    return sps.coo_matrix((np.r_[e, e[off]], (np.r_[oa, ob[off]], np.r_[ob, oa[off]])), shape=(n, n)).tocsr()


def observed_q(plan, hi, lo, V, Eh, El):
    """r^p_eff - 1 for the two-resolution estimate, p_eff = min(stated p, observed rate) (module docstring)."""
    lo2 = plan.level(lo.lxi / math.sqrt(2.0))
    try:
        E2 = plan.product(lo2, V)
    finally:
        if lo2 is not plan.used:
            plan._drop(lo2)
    r = hi.Xi / lo.Xi
    d1 = np.linalg.norm(Eh - El, axis=0)
    d2 = np.linalg.norm(El - E2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pobs = np.where((d1 > 0) & (d2 > 0), np.log(d2 / d1) / math.log(r), plan.p)
    p = float(np.clip(np.min(pobs), 0.5, plan.p))
    plan.stats["observed_rate"] = float(np.min(pobs))
    return (r ** p - 1) / SAFETY
