"""Energies int int K dmu dnu of pairs of atoms (atoms.py), and values int K(x, .) dmu of atoms at points: one table
PAIR indexed by the kinds of the two atoms (the smaller kind first; energies are symmetric).

Every entry takes the kernel and the geometry of two equally long lists of atoms (atoms.Rows) and returns
(values, error estimates). Weighted atoms (kind WEIGHTED, EXP-031) carry their piece in Rows.piece; their entries are
generic (_weighted_one): the same piece by its same-piece route; pieces well apart by plain rules; axis-parallel
boxes, two round pieces, the same parallelogram, segments on one segment or sharing a vertex, flat pieces sharing
vertices (common edge / vertex) by their dedicated routes; any other pair by the outer rule of the piece of lower
dimension against the potential of the other; all at two levels (weighted*.py), with the floor of singular pieces. Point/point, point/segment, segment/segment, point/patch and point/sphere are exact
reductions (kernels.py, surfaces.py) and return zero errors; the other entries (and point/circle) are graded
quadratures, evaluated at two orders (surfaces.LOW, surfaces.HIGH) whose difference is the error estimate. Values at
points are the POINT rows of the table.
"""
import numpy as np

from ...core.errors import NotDefined, Unsupported
from . import kernels as KR
from . import surfaces as SF
from .atoms import Kind, Rows

ROWS = 2048            # pairs per vectorised call: keeps the temporaries in cache (50 000: 1.5 times slower serially,
                       # 6 times slower per process with 30 processes on europa, bound by memory bandwidth)
SEGMENT_ROWS = 256     # segment pairs per call (their temporaries are 8 edges x chunks x 16 nodes per pair): 30
                       # processes on europa do 7.9 pairs/us at 256 against 2.3 at 2048 (2.1 against 2.4 us serially)


def _exact(values):
    return values, np.zeros(len(values))


def _point_point(kern, a, b):
    if not np.isfinite(kern.k0):
        raise NotDefined("point evaluations need s > d/2 (the kernel is singular at 0)")
    return _exact(kern(np.linalg.norm(a.P - b.P, axis=1)))


def _point_segment(kern, a, b):
    L = np.linalg.norm(b.E1, axis=1)
    U = b.E1 / L[:, None]
    rel = a.P - b.P
    t0 = np.sum(rel * U, axis=1)
    delta = np.sqrt(np.maximum(np.sum(rel * rel, axis=1) - t0 * t0, 0.0))
    return _exact(KR.line_integral(kern, t0, delta, 0.0, L))


def _segment_segment(kern, a, b):
    return _exact(KR.segment_pair_list(kern, a.P, a.Q, b.P, b.Q))


def _point_patch(kern, a, b):
    return _exact(SF.point_patch(kern, a.P, b.P, b.E1, b.E2))


def _point_sphere(kern, a, b):
    return _exact(SF.sphere_potential(kern, np.linalg.norm(a.P - b.P, axis=1), b.radius))


def _point_circle(kern, a, b):
    r = np.linalg.norm(a.P - b.P, axis=1)
    hi, lo = np.empty(len(r)), np.empty(len(r))
    for R in np.unique(b.radius):                    # one quadrature rule per radius, vectorised over the points
        s = b.radius == R
        hi[s], lo[s] = (SF.circle_potential(kern, r[s], R, n) for n in (SF.HIGH, SF.LOW))
    return hi, np.abs(hi - lo)


def _two_orders(one):
    """A table entry from a function of one pair of atoms (geometry rows a[n], b[n]) and a quadrature order."""
    def entry(kern, a, b):
        hi = np.array([one(kern, a, b, n, SF.HIGH) for n in range(len(a.P))])
        lo = np.array([one(kern, a, b, n, SF.LOW) for n in range(len(a.P))])
        return hi, np.abs(hi - lo)
    return entry


PAIR = {
    (Kind.POINT, Kind.POINT): _point_point,
    (Kind.POINT, Kind.SEGMENT): _point_segment,
    (Kind.SEGMENT, Kind.SEGMENT): _segment_segment,
    (Kind.POINT, Kind.PATCH): _point_patch,
    (Kind.POINT, Kind.SPHERE): _point_sphere,
    (Kind.SEGMENT, Kind.PATCH): _two_orders(
        lambda kern, a, b, n, o: SF.segment_patch(kern, a.P[n], a.Q[n], b.P[n], b.E1[n], b.E2[n], o)),
    (Kind.SEGMENT, Kind.SPHERE): _two_orders(
        lambda kern, a, b, n, o: SF.segment_sphere(kern, a.P[n], a.Q[n], b.P[n], b.radius[n], o)),
    (Kind.PATCH, Kind.PATCH): _two_orders(
        lambda kern, a, b, n, o: SF.patch_patch(kern, a.P[n], a.E1[n], a.E2[n], b.P[n], b.E1[n], b.E2[n], o)),
    (Kind.PATCH, Kind.SPHERE): _two_orders(
        lambda kern, a, b, n, o: SF.patch_sphere(kern, a.P[n], a.E1[n], a.E2[n], b.P[n], b.radius[n], o)),
    (Kind.SPHERE, Kind.SPHERE): _two_orders(
        lambda kern, a, b, n, o: SF.sphere_sphere(kern, a.P[n], a.radius[n], b.P[n], b.radius[n], o)),
    (Kind.POINT, Kind.CIRCLE): _point_circle,
    (Kind.SEGMENT, Kind.CIRCLE): _two_orders(
        lambda kern, a, b, n, o: SF.segment_circle(kern, a.P[n], a.Q[n], b.P[n], b.radius[n], o)),
    (Kind.CIRCLE, Kind.CIRCLE): _two_orders(
        lambda kern, a, b, n, o: SF.circle_circle(kern, a.P[n], a.radius[n], b.P[n], b.radius[n], o)),
}


CONST_DIM = {Kind.SEGMENT: 1, Kind.PATCH: 2, Kind.SPHERE: 2, Kind.CIRCLE: 1}


def _trim(X, d):
    """Points in the piece's dimension (Rows embed 1-D flat coordinates in R^2)."""
    return X[:, :d] if X.shape[1] > d else X


def _pad(X, d):
    return np.concatenate([X, np.zeros((len(X), d - X.shape[1]))], axis=1) if X.shape[1] < d else X


def _const_nodes(kind, a, n, kern, level):
    """An outer rule (points, weights) on the constant atom n of the rows a."""
    from . import weighted as W
    m = W.ORDER[level]
    if kind == Kind.SEGMENT:
        L = float(np.linalg.norm(a.E1[n]))
        t, w = W.both_ends(m, W._cells(L, kern, 2))
        return a.P[n] + t[:, None] * a.E1[n], L * w
    if kind == Kind.PATCH:
        s, ws = W.both_ends(m, W._cells(np.linalg.norm(a.E1[n]), kern, 1), 1e-4)
        t, wt = W.both_ends(m, W._cells(np.linalg.norm(a.E2[n]), kern, 1), 1e-4)
        S, T = (g.reshape(-1, 1) for g in np.meshgrid(s, t, indexing="ij"))
        return a.P[n] + S * a.E1[n] + T * a.E2[n], np.linalg.norm(np.cross(a.E1[n], a.E2[n])) * np.outer(ws, wt).ravel()
    if kind == Kind.SPHERE:
        U, w = W.SpherePiece(a.P[n], a.radius[n], None)._rule((16, 24)[level])
        return a.P[n] + a.radius[n] * U, w
    nphi = (64, 96)[level]
    phi = 2 * np.pi * np.arange(nphi) / nphi
    R = a.radius[n]
    return a.P[n] + R * np.stack([np.cos(phi), np.sin(phi)], axis=1), np.full(nphi, 2 * np.pi * R / nphi)


def _patch_piece(a, n):
    """A constant patch atom as a parallelogram piece with w = 1."""
    from .atoms import FlatWeight
    from .weighted_volume import ParallelogramPiece
    return ParallelogramPiece(a.P[n], np.stack([a.E1[n], a.E2[n]]), FlatWeight(constant=1.0))


def _weighted_one(kern, kind, a, b, n, level):
    """The energy of the pair n (rows a of kind `kind`, rows b weighted) at one level."""
    from .weighted import SegmentPiece, same_segment, same_segment_energy, shared_vertex, vertex_energy
    pb = b.piece[n]
    d = len(pb.centre)
    if kind == Kind.SEGMENT and isinstance(pb, SegmentPiece):      # a constant segment: a segment piece with w = 1
        from .atoms import FlatWeight
        kind, a = Kind.WEIGHTED, Rows(a.P[n:n + 1], a.E1[n:n + 1], a.E2[n:n + 1], a.radius[n:n + 1],
                                      np.array([SegmentPiece(_trim(a.P[n:n + 1], d)[0], _trim(a.Q[n:n + 1], d)[0],
                                                             FlatWeight(constant=1.0))], object))
        n0, n = n, 0
        b = Rows(b.P[n0:n0 + 1], b.E1[n0:n0 + 1], b.E2[n0:n0 + 1], b.radius[n0:n0 + 1], b.piece[n0:n0 + 1])
    if kind == Kind.PATCH and hasattr(pb, "same") and pb.same(_patch_piece(a, n)):
        return pb.self_energy(kern, level, other=_patch_piece(a, n))
    if kind == Kind.WEIGHTED:
        pa = a.piece[n]
        if pa is pb:
            return pb.cached_self_energy(kern, level)
        from .weighted_volume import axis_aligned, box_pair_energy
        gap = float(np.linalg.norm(pa.centre - pb.centre)) - pa.radius - pb.radius
        if gap >= max(pa.radius, pb.radius):             # well apart (far lattice or reflection images): plain rules
            from .weighted import disjoint_energy
            e = disjoint_energy(kern, pa, pb, level)
            if e is not None:
                return e
        if axis_aligned(pa) and axis_aligned(pb):
            return box_pair_energy(kern, pa, pb, level)
        from .weighted_volume import RoundPiece
        if isinstance(pa, RoundPiece) and isinstance(pb, RoundPiece):     # overlapping, nested, touching (O5)
            from .weighted_contacts import round_pair_energy
            return round_pair_energy(kern, pa, pb, level)
        if hasattr(pa, "same") and pa.same(pb):
            return pa.self_energy(kern, level, other=pb)
        from .weighted_solids import TetraPiece
        if isinstance(pa, TetraPiece) and isinstance(pb, TetraPiece):         # the same cell, vertices reordered
            tol = 1e-12 * max(pa.diam, pb.diam)
            order = [j for i in range(4) for j in range(4) if np.linalg.norm(pa.V[i] - pb.V[j]) <= tol]
            if len(order) == 4:
                return pa.self_energy(kern, level, other=TetraPiece(*pb.V[order], pb.w))
        if isinstance(pa, SegmentPiece) and isinstance(pb, SegmentPiece):
            if same_segment(pa, pb):
                return same_segment_energy(kern, pa, pb, level)
            ij = shared_vertex(pa, pb)
            if ij is not None:
                return vertex_energy(kern, pa, pb, ij, level)
        from .weighted import disjoint_energy
        from .weighted_contacts import touches, touching_energy
        key = ("pair", id(pb), kern.s, kern.d, kern.l, level)          # pieces live with their atoms: memoise
        memo = pa.__dict__.setdefault("_memo", {})
        if key not in memo:
            from .weighted_solids import TetraPiece
            from .weighted_contacts import join_energy, overlap
            sh = _tet_shared(pa, pb) if isinstance(pa, TetraPiece) and isinstance(pb, TetraPiece) else []
            if sh and not overlap(pa, pb, [i for i, _ in sh], [j for _, j in sh]):
                memo[key] = join_energy(kern, pa, pb, [i for i, _ in sh], [j for _, j in sh], level)   # tetrahedra
            elif touches(pa, pb):                        # flat pieces sharing vertices: edge and vertex routes
                memo[key] = touching_energy(kern, pa, pb, level, lambda x, y: disjoint_energy(kern, x, y, level),
                                            lambda x, y: _generic(kern, x, y, level))
            else:
                e = disjoint_energy(kern, pa, pb, level)
                memo[key] = _generic(kern, pa, pb, level) if e is None else e
            memo[key + ("piece",)] = pb                  # keep pb alive while the key (its id) is in use
        return memo[key]
    if pb.dim < CONST_DIM[kind]:                     # the weighted piece outside, the exact potential of the atom inside
        X, Wt = pb.nodes(kern, level)[:2]
        m = len(X)
        rows = Rows(_pad(X, a.P.shape[1]), np.zeros((m, a.P.shape[1])), np.zeros((m, a.P.shape[1])), np.zeros(m))
        atom = Rows(np.repeat(a.P[n:n + 1], m, 0), np.repeat(a.E1[n:n + 1], m, 0), np.repeat(a.E2[n:n + 1], m, 0),
                    np.repeat(a.radius[n:n + 1], m))
        return float(Wt @ PAIR[(Kind.POINT, kind)](kern, rows, atom)[0])
    X, Wt = _const_nodes(kind, a, n, kern, level)
    return float(Wt @ pb.potential(kern, _trim(X, d), level))


def _tet_shared(a, b):
    """Pairs (i, j) of coinciding vertices of two tetrahedra (fewer than four: not the same cell)."""
    tol = 1e-12 * max(a.diam, b.diam)
    sh = [(i, j) for i in range(4) for j in range(4) if np.linalg.norm(a.V[i] - b.V[j]) <= tol]
    return sh if 0 < len(sh) < 4 else []


def _generic(kern, pa, pb, level):
    """The outer rule of the piece of lower dimension against the singular potential of the other."""
    outer, inner = (pa, pb) if pa.dim <= pb.dim else (pb, pa)
    X, Wt = outer.nodes(kern, level)[:2]
    return float(Wt @ inner.potential(kern, X, level))


def _weighted(kind):
    """The table entry (kind, WEIGHTED)."""
    def entry(kern, a, b):
        if kind == Kind.POINT:                       # values: one potential call per piece, two levels
            hi, lo = np.empty(len(a.P)), np.empty(len(a.P))
            ids = np.array([id(p) for p in b.piece])
            for u in np.unique(ids):
                sel = np.nonzero(ids == u)[0]
                pc = b.piece[sel[0]]
                X = _trim(a.P[sel], len(pc.centre))
                hi[sel], lo[sel] = pc.potential(kern, X, 1), pc.potential(kern, X, 0)
            return hi, np.abs(hi - lo)
        hi = np.array([_weighted_one(kern, kind, a, b, n, 1) for n in range(len(a.P))])
        lo = np.array([_weighted_one(kern, kind, a, b, n, 0) for n in range(len(a.P))])
        floor = np.array([max(getattr(b.piece[n], "floor", 0.0), getattr(a.piece[n], "floor", 0.0) if a.piece is not
                              None else 0.0) for n in range(len(a.P))])     # singular pieces: the omitted cells
        return hi, np.abs(hi - lo) + floor * np.abs(hi)
    return entry


for _k in Kind:
    PAIR[(_k, Kind.WEIGHTED)] = _weighted(_k)


def energies(kern, A, ia, B, ib, shift=None):
    """Energies and error estimates of the atom pairs (A[ia[n]], B[ib[n]] translated by shift)."""
    E, err = np.zeros(len(ia)), np.zeros(len(ia))
    ka, kb = A.kind[ia], B.kind[ib]
    lo, hi = np.minimum(ka, kb), np.maximum(ka, kb)
    codes = np.unique(lo * 16 + hi)
    missing = {(int(c) // 16, int(c) % 16) for c in codes} - {(int(x), int(y)) for x, y in PAIR}
    if missing:
        raise Unsupported(f"no energy for atoms of kinds {[(Kind(x).name, Kind(y).name) for x, y in missing]}")
    for key, entry in PAIR.items():
        sel = np.nonzero((lo == key[0]) & (hi == key[1]))[0]
        for flip in (False, True):                           # flip: the first atom has the larger kind
            s = sel[(ka[sel] != key[0]) == flip]
            rows = SEGMENT_ROWS if key == (Kind.SEGMENT, Kind.SEGMENT) else ROWS
            for c in range(0, len(s), rows):
                t = s[c:c + rows]
                ra, rb = A.rows(ia[t]), B.rows(ib[t], shift)
                E[t], err[t] = entry(kern, *((rb, ra) if flip else (ra, rb)))
    return E, err


def values(kern, F, A, shift=None):
    """(len(F), len(A.P)): the potentials of the atoms of A (translated by shift) at the flat points F."""
    V = np.zeros((len(F), len(A.P)))
    zero = np.zeros_like(F)
    for k in np.unique(A.kind):
        idx = np.nonzero(A.kind == k)[0]
        I, J = (g.ravel() for g in np.meshgrid(np.arange(len(F)), idx, indexing="ij"))
        out = np.empty(len(I))
        for c in range(0, len(I), ROWS):
            i, j = I[c:c + ROWS], J[c:c + ROWS]
            out[c:c + ROWS] = PAIR[(Kind.POINT, Kind(k))](kern, Rows(F[i], zero[i], zero[i], np.zeros(len(i))),
                                                           A.rows(j, shift))[0]
        V[:, idx] = out.reshape(len(F), len(idx))
    return V
