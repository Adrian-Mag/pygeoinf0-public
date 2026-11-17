"""Two simplices of dimension k (triangles, tetrahedra) sharing an m-face (m = 0: a vertex, 1: an edge, 2: a
triangle), in any ambient dimension (EXP-031 O7; DESIGN Q8 for k = 3).

Join coordinates. With the shared face F (vertices F_0..F_m, barycentric s in the m-simplex S_m) and the other
vertices a_1..a_(k-m) of A and b_j of B:
    x = (1 - |p|) F(s) + sum p_i a_i,   y = (1 - |q|) F(t) + sum q_j b_j,   p, q in the (k - m)-simplex S_(k-m),
    dx = J_A (1 - |p|)^m ds dp,  dy = J_B (1 - |q|)^m dt dq.
The kernel is singular only at p = q = 0, s = t; with the offset z = t - s that is the point (z, p, q) = 0, and for
fixed (z, p, q) the integral over s runs over the homothetic simplex S_m cap (S_m - z) (smooth integrand). The region
of z (the difference body S_m - S_m) is split into the cones where the size of that simplex is affine (m = 1: two
intervals; m = 2: the six triangles of the hexagon). Every factor (a cone of z, S_(k-m) for p and for q) is a
simplex with the vertex 0, written as rho * (a point of its far facet): the singular point is the corner rho = 0 of
the cube of the radial variables (rho_z, rho_p, rho_q), split into pyramids at that corner (Duffy, graded), the far
facets by collapsed Gauss rules. |x - y| >= c max(rho) for simplices that do not overlap, so the reduced integrand is
the kernel's integrable singularity times smooth functions. Checked: a cube as six Kuhn tetrahedra (every pair shares
an edge or a face) against the box route; a square as two and four triangles against the box route.

From weighted_touching.py (merged here by plan 063):
Touching flat pieces (EXP-031, DESIGN 5.2 "touching flat pieces", question 8): energies of two triangles that share
an edge or a vertex, in a plane or bent in 3-D, and of polygons (parallelograms) made of them.

Common edge E0 E1 (apices Ca, Cb): x = (1 - p)(E0 + s e) + p Ca, y = (1 - q)(E0 + t e) + q Cb, e = E1 - E0, with
dx = 2|A| (1 - p) ds dp and dy = 2|B| (1 - q) dt dq. The kernel is singular only at p = q = 0, s = t (the shared edge),
and with z = t - s that is the single point (z, p, q) = 0: so for fixed (z, p, q) the integral over s, on an interval
of length 1 - |z|, is smooth, and the cube of (|z|, p, q) is split into three pyramids at 0 (Duffy), the radial
variable graded geometrically, Jacobian rho^2. |x - y| >= c rho for triangles that do not overlap, so the reduced
integrand is k(rho ...) rho^2 times smooth functions: integrable (rho^(2 + 2s - d)) and smooth in the other variables.
Common vertex E0: x = E0 + p ((1 - s) (A1 - E0) + s (A2 - E0)), y likewise with q, t, dx = 2|A| p ds dp; singular only
at p = q = 0: the square of (p, q) is split into two triangles at 0 (Duffy), Jacobian rho; |x - y| >= c rho unless the
two angular sectors overlap.
These are the common-edge and common-vertex cases of the Sauter-Schwab classification, in relative coordinates of
our own (DESIGN Q8); the identical case is TrianglePiece.self_energy. Checked by additivity: a square split into two and
into four triangles against the box route (tests/test_weighted_routes.py).

From weighted_round.py (merged here by plan 063):
Two round pieces (disks in 2-D, balls in 3-D) of any radii and centres (EXP-031 O5): overlapping, nested, touching
or apart, and so weighted shells as a ball minus a concentric ball.

E(a, b) = int k(|z|) C(z) dz,   C(z) = int over B_a cap (B_b - z) of w_a(x) w_b(x + z) dx,

in polar (spherical) coordinates z = rho e about the singular point z = 0, the directions measured from the axis
Delta = c_b - c_a. Along a direction the intersection changes shape where |rho e - Delta| = R_a + R_b (it vanishes)
and |rho e - Delta| = |R_a - R_b| (one ball inside the other): the roots of a quadratic in rho split the rho integral
(graded at both ends of every panel, where C behaves like a power of the distance to them); the directions in which
those roots appear or merge (tangency, sin(angle) = r / |Delta|) split the angular integral the same way.
C(rho e) is integrated over the intersection: the smaller ball by spherical coordinates when it lies inside the other,
the lens otherwise in coordinates xi along the axis between the two caps and eta = H sin(theta) across (no
square-root ends). Constant weights: the closed-form area (2-D) or volume (3-D) of the intersection.

From weighted_images.py (merged here by plan 063):
Images of weighted pieces (EXP-031 O3): the piece y -> R y + t (R a vector of +-1 per flat axis: lattice translations
on tori, R = 1; signed reflection images of spectral Neumann/Dirichlet spaces on intervals and boxes, green.Image) with
the weight carried along, w'(y) = w(R (y - t)) (R is its own inverse). The energy engine adds the images' energies with
their signs (energy.ExactEngine); a piece's images are made once and kept on it.
"""


import math

import numpy as np

from ...quadrature import both_ends, gauss, graded_unit, panels, simplex
from .weighted import ORDER, _cells, kfin

ANG = (6, 8)                   # Gauss points per direction of the pyramid and s rules (per cell) at the two levels
FAC = (8, 12)                  # ... of the far-facet rules (they limit the vertex case: 4e-7 at 6, 1e-12 at 12)


def _far_facet(V, n):
    """Points (q, d) and weights on the facet opposite 0 of the simplex with vertices 0, V (j, d): y = sum lambda_i V_i,
    lambda in the standard (j - 1)-simplex parametrised by its first j - 1 coordinates: then
    dV = rho^(j - 1) |det V| drho dlambda (the weights here are those of dlambda)."""
    j = len(V)
    U, W = simplex(j - 1, n)
    lam = np.c_[U, 1 - U.sum(axis=1)]
    return lam @ V, W


def _z_cones(m):
    """The cones (vertex 0) of the difference body S_m - S_m on which the overlap size is affine."""
    if m == 0:
        return [np.zeros((0, 0))]
    if m == 1:
        return [np.array([[1.0]]), np.array([[-1.0]])]
    H = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, -1.0]])
    return [np.stack([H[i], H[(i + 1) % 6]]) for i in range(6)]


def _size(Z):
    if Z.shape[1] == 0:
        return np.ones(len(Z))
    return np.minimum(1.0, 1.0 - Z.sum(axis=1)) - np.maximum(0.0, -Z).sum(axis=1)


def _jac(F0, M):
    """sqrt(det(M^T M)) for the columns M (d, k): the k-volume factor of x = F0 + M u."""
    return math.sqrt(max(np.linalg.det(M.T @ M), 0.0))


def _inside(V, X, tol=1e-9):
    """Whether points X lie strictly inside the simplex with vertices V (of full dimension in its own span)."""
    E = (V[1:] - V[0]).T
    lam, *_ = np.linalg.lstsq(E, (X - V[0]).T, rcond=None)
    lam = lam.T
    resid = np.linalg.norm(X - V[0] - lam @ E.T, axis=1)
    b = np.c_[1 - lam.sum(1), lam]
    return (resid <= tol * np.ptp(V, axis=0).max()) & np.all(b > tol, axis=1)


def overlap(A, B, ia, ib):
    """Whether two simplices sharing the vertices A.V[ia] = B.V[ib] overlap (then join coordinates do not apply):
    points of each near the shared set and the centroids tested in the other."""
    VA, VB = np.asarray(A.V, float), np.asarray(B.V, float)
    for P, Q, sh in ((VA, VB, ia), (VB, VA, ib)):
        c = P.mean(axis=0)
        X = np.vstack([c] + [P[i] + 0.01 * (c - P[i]) for i in sh] + [P[list(sh)].mean(0) + 0.01 * (c - P[list(sh)].mean(0))])
        if np.any(_inside(Q, X)):
            return True
    return False


def join_energy(kern, A, B, ia, ib, level):
    """int int K w_A w_B for simplex pieces A, B (TrianglePiece, TetraPiece: .V (k + 1, d), .w) sharing the vertices
    A.V[ia[i]] = B.V[ib[i]] (m + 1 of them, m < k)."""
    VA, VB = np.asarray(A.V, float), np.asarray(B.V, float)
    k, m = len(VA) - 1, len(ia) - 1
    F = VA[list(ia)]
    a = VA[[i for i in range(k + 1) if i not in ia]]
    b = VB[[i for i in range(k + 1) if i not in ib]]
    F0 = F[0]
    DF = (F[1:] - F0).T                                          # (d, m)
    JA = _jac(F0, np.c_[DF, (a - F0).T])
    JB = _jac(F0, np.c_[DF, (b - F0).T])
    n, j = ORDER[level], k - m                                     # j: the dimension of the p and q simplices
    diam = max(float(np.ptp(VA, axis=0).max()), float(np.ptp(VB, axis=0).max()), 1e-300)
    cells = _cells(diam, kern, 1)
    rho, wrho, _ = graded_unit(n, cells)
    nr = 3 if m > 0 else 2                                         # radial variables: z (if m > 0), p, q
    # Duffy pyramids of the cube [0, 1]^nr at 0: the largest radial variable is tau, the others tau u
    rules = {}

    def pyramid(c):                                                # the other radial variables, c cells per axis
        if c not in rules:
            q, wq = panels(np.linspace(0, 1, c + 1), ANG[level])
            rules[c] = (np.stack([g.ravel() for g in np.meshgrid(*[q] * (nr - 1), indexing="ij")], 1),
                        np.prod(np.stack([g.ravel() for g in np.meshgrid(*[wq] * (nr - 1), indexing="ij")], 1), 1))
        return rules[c]
    Ys, Ws = _far_facet(np.eye(j), FAC[level])                     # far facet of S_j (for p and q)
    Us, Wsm = simplex(m, ANG[level] + 2)                     # the s simplex
    total = 0.0
    for C in _z_cones(m):
        if m > 0:
            Yz, Wz = _far_facet(C, FAC[level])
            detC = abs(float(np.linalg.det(C))) if m > 0 else 1.0
        else:
            Yz, Wz, detC = np.zeros((1, 0)), np.ones(1), 1.0
        for piv in range(nr):
            for t, wt in zip(rho, wrho):
                U, WU = pyramid(max(1, min(3, int(math.ceil(t * diam / kern.l)))))   # cf. weighted_touching
                R = np.ones((len(U), nr))
                cols = [c for c in range(nr) if c != piv]
                R[:, cols] = U
                R *= t                                              # the radial variables (rows of the pyramid)
                Wr = wt * WU * t ** (nr - 1)                          # Duffy Jacobian tau^(nr - 1)
                rz, rp, rq = (R[:, 0], R[:, 1], R[:, 2]) if m > 0 else (np.zeros(len(R)), R[:, 0], R[:, 1])
                # all combinations of the angular points: (radial rows) x (z facet) x (p facet) x (q facet)
                shp = (len(R), len(Yz), len(Ys), len(Ys))
                Z = np.broadcast_to(rz[:, None, None, None, None] * Yz[None, :, None, None, :], shp + (m,))
                P = np.broadcast_to(rp[:, None, None, None, None] * Ys[None, None, :, None, :], shp + (j,))
                Qv = np.broadcast_to(rq[:, None, None, None, None] * Ys[None, None, None, :, :], shp + (j,))
                W = (Wr[:, None, None, None] * Wz[None, :, None, None] * Ws[None, None, :, None]
                     * Ws[None, None, None, :] * (rz[:, None, None, None] ** (m - 1) * detC if m > 0 else 1.0)
                     * rp[:, None, None, None] ** (j - 1) * rq[:, None, None, None] ** (j - 1))
                W = np.broadcast_to(W, shp).ravel()
                Z, P, Qv = Z.reshape(W.size, m), P.reshape(W.size, j), Qv.reshape(W.size, j)
                sz = np.maximum(_size(Z), 0.0)
                keep = (sz > 0) & (P.sum(1) <= 1) & (Qv.sum(1) <= 1)
                if not keep.any():
                    continue
                Z, P, Qv, W, sz = Z[keep], P[keep], Qv[keep], W[keep], sz[keep]
                corner = np.maximum(0.0, -Z)
                S = corner[:, None, :] + sz[:, None, None] * Us[None]            # (r, s, m)
                T = S + Z[:, None, :]
                Fs = F0 + S @ DF.T if m > 0 else np.broadcast_to(F0, S.shape[:2] + (len(F0),))
                Ft = F0 + T @ DF.T if m > 0 else np.broadcast_to(F0, T.shape[:2] + (len(F0),))
                pa, qb = 1 - P.sum(1), 1 - Qv.sum(1)
                X = pa[:, None, None] * Fs + (P @ a)[:, None, :]
                Y = qb[:, None, None] * Ft + (Qv @ b)[:, None, :]
                d = X.shape[-1]
                f = kfin(kern, np.linalg.norm(X - Y, axis=-1)) * A.w(X.reshape(-1, d)).reshape(X.shape[:2]) \
                    * B.w(Y.reshape(-1, d)).reshape(Y.shape[:2])
                inner = (f @ Wsm) * sz ** m                     # the homothetic s simplex: size^m
                total += float(np.sum(W * pa ** m * qb ** m * inner))
    return JA * JB * total


# ---------------------------------------------------------------- from weighted_touching.py (plan 063)
def triangles_of(piece):
    """The triangles of a flat piece (a triangle; a parallelogram as two triangles with its weight), or None."""
    from .weighted_volume import ParallelogramPiece, TrianglePiece
    if isinstance(piece, TrianglePiece):
        return [piece]
    if isinstance(piece, ParallelogramPiece):
        V = piece.V
        return [TrianglePiece(V[0], V[1], V[2], piece.w), TrianglePiece(V[0], V[2], V[3], piece.w)]
    return None


def touches(pa, pb):
    """Whether two flat pieces share a vertex (then touching_energy applies)."""
    ta, tb = triangles_of(pa), triangles_of(pb)
    if ta is None or tb is None:
        return False
    tol = 1e-12 * max(pa.diam, pb.diam)
    return any(_shared(a, b, tol) for a in ta for b in tb)


def _shared(a, b, tol):
    """Pairs (i, j) of coinciding vertices of triangles a and b."""
    return [(i, j) for i in range(3) for j in range(3) if np.linalg.norm(a.V[i] - b.V[j]) <= tol]


ANGULAR = (8, 12)            # Gauss points per cell of the smooth (angular, along-edge) variables at the two levels


def _rule(level, cells):
    return panels(np.linspace(0, 1, cells + 1), ANGULAR[level])


def _cells_at(rho, diam, kern, cap=3):
    """Cells of the smooth variables at radius rho: the integrand varies on the kernel's scale l, i.e. over
    l / (rho diam) of the unit range; capped, because where rho diam >> l the kernel has decayed (|x - y| >= c rho)
    and that part's contribution is exponentially small."""
    return max(1, min(cap, int(np.ceil(rho * diam / kern.l))))


def edge_energy(kern, a, b, ia, ib, level):
    """int int K w_a w_b for triangles sharing the edge (a.V[ia[0]], a.V[ia[1]]) = (b.V[ib[0]], b.V[ib[1]])."""
    E0, E1 = a.V[ia[0]], a.V[ia[1]]
    Ca, Cb = a.V[3 - sum(ia)], b.V[3 - sum(ib)]
    e = E1 - E0
    diam = max(a.diam, b.diam)
    r, wr, _ = graded_unit(ORDER[level], _cells(diam, kern, 1))
    rules = {}
    total = 0.0
    for ri, wri in zip(r, wr):
        c = _cells_at(ri, diam, kern)
        if c not in rules:
            u, wu = _rule(level, c)
            s0, ws0 = _rule(level, c)
            U1, U2 = (g.ravel() for g in np.meshgrid(u, u, indexing="ij"))
            rules[c] = (U1, U2, np.outer(wu, wu).ravel(), s0, ws0)
        U1, U2, WU, s0, ws0 = rules[c]
        for sign in (1.0, -1.0):
            for perm in range(3):                               # the three pyramids of the cube of (|z|, p, q)
                C = np.roll(np.stack([np.ones_like(U1), U1, U2], axis=1), perm, axis=1) * ri
                z, p, q = sign * C[:, 0], C[:, 1], C[:, 2]
                lo, ln = np.maximum(0.0, -z), 1 - np.abs(z)
                S = lo[:, None] + ln[:, None] * s0[None]                # (m, ns)
                T = S + z[:, None]
                X = (1 - p)[:, None, None] * (E0 + S[..., None] * e) + p[:, None, None] * Ca
                Y = (1 - q)[:, None, None] * (E0 + T[..., None] * e) + q[:, None, None] * Cb
                d = X.shape[-1]
                f = kfin(kern, np.linalg.norm(X - Y, axis=-1)) * a.w(X.reshape(-1, d)).reshape(S.shape) \
                    * b.w(Y.reshape(-1, d)).reshape(S.shape)
                total += wri * ri * ri * float(np.sum(WU * (1 - p) * (1 - q) * ln * (f @ ws0)))
    return a.det * b.det * total


def vertex_energy(kern, a, b, ia, ib, level):
    """int int K w_a w_b for triangles sharing only the vertex a.V[ia] = b.V[ib]."""
    E0 = a.V[ia]
    A1, A2 = (a.V[k] for k in range(3) if k != ia)
    B1, B2 = (b.V[k] for k in range(3) if k != ib)
    diam = max(a.diam, b.diam)
    r, wr, _ = graded_unit(ORDER[level], _cells(diam, kern, 1))
    rules = {}
    total = 0.0
    for ri, wri in zip(r, wr):
        c = _cells_at(ri, diam, kern)
        if c not in rules:
            v, wv = _rule(level, c)
            S, Tt = (g.ravel() for g in np.meshgrid(v, v, indexing="ij"))
            Da = (1 - S)[:, None] * (A1 - E0) + S[:, None] * (A2 - E0)       # directions of a and b
            Db = (1 - Tt)[:, None] * (B1 - E0) + Tt[:, None] * (B2 - E0)
            rules[c] = (v, wv, np.outer(wv, wv).ravel(), Da, Db)
        v, wv, WS, Da, Db = rules[c]
        for swap in (False, True):                              # the two Duffy triangles of the square of (p, q)
            p = ri * (np.ones_like(v) if not swap else v)
            q = ri * (v if not swap else np.ones_like(v))
            X = E0 + p[:, None, None] * Da[None]
            Y = E0 + q[:, None, None] * Db[None]
            d = X.shape[-1]
            f = kfin(kern, np.linalg.norm(X - Y, axis=-1)) * a.w(X.reshape(-1, d)).reshape(X.shape[:2]) \
                * b.w(Y.reshape(-1, d)).reshape(Y.shape[:2])
            total += wri * ri * float(np.sum((wv * p * q)[:, None] * WS[None] * f))
    return a.det * b.det * total


def touching_energy(kern, pa, pb, level, disjoint, generic):
    """The energy of two flat pieces (triangles, parallelograms) that may touch: triangle by triangle, identical,
    common-edge, common-vertex or apart (disjoint(ta, tb) or, when the triangles meet otherwise, generic(ta, tb)).
    None when a piece is not flat."""
    ta, tb = triangles_of(pa), triangles_of(pb)
    if ta is None or tb is None:
        return None
    tol = 1e-12 * max(pa.diam, pb.diam)
    total = 0.0
    for a in ta:
        for b in tb:
            sh = _shared(a, b, tol)
            if 0 < len(sh) < 3:
                from .weighted_contacts import overlap
                if overlap(a, b, [i for i, _ in sh], [j for _, j in sh]):     # overlapping cells: no join route
                    total += generic(a, b)
                    continue
            if len(sh) == 3:
                total += a.self_energy(kern, level, other=b if b.same(a) else _reordered(b, a, sh))
            elif len(sh) == 2:
                total += edge_energy(kern, a, b, [i for i, _ in sh], [j for _, j in sh], level)
            elif len(sh) == 1:
                total += vertex_energy(kern, a, b, sh[0][0], sh[0][1], level)
            else:
                e = disjoint(a, b)
                total += generic(a, b) if e is None else e
    return total


def _reordered(b, a, sh):
    """b with its vertices in the order of a's (the same triangle)."""
    from .weighted_volume import TrianglePiece
    order = [j for _, j in sorted(sh)]
    return TrianglePiece(*b.V[order], b.w)


# ---------------------------------------------------------------- from weighted_round.py (plan 063)
def _const(w):
    return getattr(w, "constant", None)


def _inter_measure(d, Ra, Rb, D):
    """|B_a cap B_b| for centres D apart (closed form)."""
    D = np.asarray(D, float)
    r, R = min(Ra, Rb), max(Ra, Rb)
    out = np.zeros_like(D)
    inside = D <= R - r
    lens = (D > R - r) & (D < R + r)
    q = D[lens]
    if d == 2:
        out[inside] = math.pi * r * r
        a1 = r * r * np.arccos(np.clip((q * q + r * r - R * R) / (2 * q * r), -1, 1))
        a2 = R * R * np.arccos(np.clip((q * q + R * R - r * r) / (2 * q * R), -1, 1))
        a3 = 0.5 * np.sqrt(np.maximum((-q + r + R) * (q + r - R) * (q - r + R) * (q + r + R), 0.0))
        out[lens] = a1 + a2 - a3
    else:
        out[inside] = 4 * math.pi * r ** 3 / 3
        out[lens] = math.pi * (R + r - q) ** 2 * (q * q + 2 * q * r - 3 * r * r + 2 * q * R + 6 * r * R - 3 * R * R) / (12 * q)
    return out


def _frames(E):
    """Unit vectors perpendicular to the rows of E (m, d): (m, 1, 2) in 2-D, (m, 2, 3) in 3-D."""
    if E.shape[1] == 2:
        return np.stack([-E[:, 1], E[:, 0]], 1)[:, None, :]
    b = np.where(np.abs(E[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    f1 = np.cross(E, b)
    f1 /= np.linalg.norm(f1, axis=1)[:, None]
    return np.stack([f1, np.cross(E, f1)], 1)


def _corr(a, b, Z, level, block=512):
    """C(z) for points z (m, d)."""
    d = len(a.c)
    Dv = b.c - Z - a.c                                            # centre of B_b - z relative to c_a
    D = np.linalg.norm(Dv, axis=1)
    ca, cb = _const(a.w), _const(b.w)
    if ca is not None and cb is not None:
        return ca * cb * _inter_measure(d, a.R, b.R, D)
    Ra, Rb = a.R, b.R
    out = np.zeros(len(Z))
    ng = (16, 24)[level] if d == 2 else (8, 10)[level]
    g, gw = gauss(ng)
    nested = D <= abs(Ra - Rb)
    lens = ~nested & (D < Ra + Rb)
    if nested.any():                                              # the smaller ball inside the other
        small = a if Ra <= Rb else b
        P, Wp = _ball_rule(d, small.R, level)
        idx = np.flatnonzero(nested)
        for i in range(0, len(idx), 64):
            m = idx[i:i + 64]
            cs = small.c - (Z[m] if small is b else 0.0)
            X = np.broadcast_to(np.atleast_2d(cs)[:, None, :], (len(m), 1, d)) + P[None]
            f = a.w(X.reshape(-1, d)) * b.w((X + Z[m][:, None, :]).reshape(-1, d))
            out[m] = f.reshape(len(m), -1) @ Wp
    if lens.any():
        idx = np.flatnonzero(lens)
        for i in range(0, len(idx), block):
            m = idx[i:i + block]
            Dm = D[m]
            e = Dv[m] / Dm[:, None]
            F = _frames(e)
            x0 = (Dm ** 2 + Ra * Ra - Rb * Rb) / (2 * Dm)
            H = np.sqrt(np.maximum(Ra * Ra - x0 * x0, 0.0))
            far = (x0 >= Dm) if Rb <= Ra else (x0 <= 0)
            eta, weta = _eta_rule(H, min(Ra, Rb), far, d, ng)
            sa = np.sqrt(np.maximum(Ra * Ra - eta * eta, 0.0))
            sb = np.sqrt(np.maximum(Rb * Rb - eta * eta, 0.0))
            lo = np.maximum(-sa, Dm[:, None] - sb)
            hi = np.maximum(np.minimum(sa, Dm[:, None] + sb), lo)
            xi = 0.5 * (lo + hi)[..., None] + 0.5 * (hi - lo)[..., None] * g             # (b, ne, nx)
            wxi = 0.5 * (hi - lo)[..., None] * gw
            if d == 2:
                P = a.c + xi[..., None] * e[:, None, None, :] + eta[..., None, None] * F[:, None, None, 0, :]
                f = (a.w(P.reshape(-1, 2)) * b.w((P + Z[m][:, None, None, :]).reshape(-1, 2))).reshape(xi.shape)
                out[m] = np.sum(f * wxi * weta[..., None], axis=(1, 2))
            else:
                npsi = (12, 16)[level]
                psi = 2 * math.pi * np.arange(npsi) / npsi
                ring = np.cos(psi)[None, :, None] * F[:, None, 0] + np.sin(psi)[None, :, None] * F[:, None, 1]
                P = (a.c + xi[..., None, None] * e[:, None, None, None, :]
                     + eta[..., None, None, None] * ring[:, None, None, :, :])         # (b, ne, nx, npsi, 3)
                f = (a.w(P.reshape(-1, 3)) * b.w((P + Z[m][:, None, None, None, :]).reshape(-1, 3))).reshape(P.shape[:4])
                out[m] = np.sum(f * (wxi * (weta * eta)[..., None])[..., None], axis=(1, 2, 3)) * 2 * math.pi / npsi
    return out


def _eta_rule(H, rs, far, d, n):
    """Transverse coordinates (b, q) and weights of the intersection of two balls: eta in [0, H] (2-D: [-H, H]) up
    to the intersection circle, eta = H sin(theta); and, where the smaller ball's equator lies inside the other ball
    (far), eta in [H, rs] (2-D: both signs), eta = rs sin(phi) for phi in [asin(H / rs), pi / 2]: no square-root ends.
    The weights include d eta (and, in 3-D, the caller multiplies by eta for the ring)."""
    t, wt = gauss(n)
    t, wt = 0.5 * (t + 1), 0.5 * wt
    if d == 2:
        a = math.pi * (t - 0.5)
        e1, w1 = H[:, None] * np.sin(a)[None], H[:, None] * np.cos(a)[None] * math.pi * wt[None]
    else:
        a = 0.5 * math.pi * t
        e1, w1 = H[:, None] * np.sin(a)[None], H[:, None] * np.cos(a)[None] * 0.5 * math.pi * wt[None]
    p0 = np.arcsin(np.clip(H / rs, 0.0, 1.0))
    phi = p0[:, None] + (0.5 * math.pi - p0)[:, None] * t[None]
    e2 = rs * np.sin(phi)
    w2 = np.where(far[:, None], rs * np.cos(phi) * (0.5 * math.pi - p0)[:, None] * wt[None], 0.0)
    if d == 2:
        return np.concatenate([e1, e2, -e2], axis=1), np.concatenate([w1, w2, w2], axis=1)
    return np.concatenate([e1, e2], axis=1), np.concatenate([w1, w2], axis=1)


def _ball_rule(d, R, level):
    """Points (q, d) of the ball of radius R about 0 and their volume weights."""
    g, gw = gauss((16, 24)[level] if d == 2 else (10, 14)[level])
    r, wr = 0.5 * R * (g + 1), 0.5 * R * gw * (0.5 * R * (g + 1)) ** (d - 1)
    if d == 2:
        nphi = (32, 48)[level]
        phi = 2 * math.pi * np.arange(nphi) / nphi
        U, Wu = np.stack([np.cos(phi), np.sin(phi)], 1), np.full(nphi, 2 * math.pi / nphi)
    else:
        from .weighted_volume import _sphere_rule
        U, Wu = _sphere_rule((6, 8)[level])
    return (r[:, None, None] * U[None]).reshape(-1, d), np.outer(wr, Wu).ravel()


def _roots(ed, D2, r):
    """Positive roots rho of |rho e - Delta| = r."""
    disc = ed * ed - D2 + r * r
    if disc < 0:
        return []
    s = math.sqrt(disc)
    return [x for x in (ed - s, ed + s) if x > 0]


def _angle_rule(d, Dn, radii, level, concentric):
    """Angles from the axis and their weights: 2-D phi in [-pi, pi]; 3-D theta in [0, pi] with sin(theta). Split at
    the tangency angles asin(r / |Delta|), graded towards them."""
    lo = -math.pi if d == 2 else 0.0
    cuts = {lo, math.pi}
    if not concentric:
        for r in radii:
            if 0 < r < Dn:
                t = math.asin(r / Dn)
                cuts |= {t, math.pi - t} | ({-t, t - math.pi} if d == 2 else set())
    cuts = sorted(c for c in cuts if lo <= c <= math.pi)
    T, W = [], []
    for t0, t1 in zip(cuts[:-1], cuts[1:]):
        x, w = both_ends((12, 16)[level] if d == 2 else (8, 10)[level], 2, 1e-6) if not concentric else \
            panels(np.linspace(0, 1, 3), (8, 12)[level])      # concentric: only the weights vary with direction
        T.append(t0 + (t1 - t0) * x)
        W.append((t1 - t0) * w)
    T, W = np.concatenate(T), np.concatenate(W)
    return T, (W if d == 2 else W * np.sin(T))


def round_pair_energy(kern, a, b, level):
    """int int K w_a w_b for two disks or balls (module docstring)."""
    from .weighted_volume import _frame
    d = len(a.c)
    Dv = b.c - a.c
    Dn = float(np.linalg.norm(Dv))
    concentric = Dn <= 1e-14 * max(a.R, b.R)
    axis = np.eye(d)[-1] if concentric else Dv / Dn
    radii = (a.R + b.R, abs(a.R - b.R))
    th, wth = _angle_rule(d, Dn, radii, level, concentric)
    n = ORDER[level]
    cells = _cells(Dn + a.R + b.R, kern, 1)
    if d == 2:
        perp = np.array([-axis[1], axis[0]])
        dirs = np.cos(th)[:, None] * axis + np.sin(th)[:, None] * perp
        wdir = wth
    else:
        F = _frame(axis)
        npsi = (12, 16)[level]
        psi = 2 * math.pi * np.arange(npsi) / npsi
        ring = np.cos(psi)[:, None] * F[0] + np.sin(psi)[:, None] * F[1]
        dirs = (np.cos(th)[:, None, None] * axis + np.sin(th)[:, None, None] * ring[None]).reshape(-1, 3)
        wdir = np.repeat(wth, npsi) * 2 * math.pi / npsi
    x0, w0, _ = graded_unit(n, max(1, cells // 2))
    xe, we = both_ends(n, max(1, cells // 2), 1e-10)
    Zs, Ws = [], []
    for e, wd in zip(dirs, wdir):
        ed = float(e @ Dv)
        ends = sorted({x for r in radii for x in _roots(ed, Dn * Dn, r)})
        if not ends:
            continue
        edges = [0.0] + ends
        for k, (r0, r1) in enumerate(zip(edges[:-1], edges[1:])):
            if k == 0:                                            # graded at 0 (the kernel) and at r1 (the shape)
                xs = np.r_[0.5 * x0, 0.5 + 0.5 * xe]
                ws = np.r_[0.5 * w0, 0.5 * we]
            else:
                xs, ws = xe, we
            rho = r0 + (r1 - r0) * xs
            Zs.append(rho[:, None] * e)
            Ws.append(wd * (r1 - r0) * ws * kfin(kern, rho) * rho ** (d - 1))
    Z, W = np.vstack(Zs), np.concatenate(Ws)
    return float(W @ _corr(a, b, Z, level))


# ---------------------------------------------------------------- from weighted_images.py (plan 063)
class _Mapped:
    """w(R (y - t)): the weight of an image."""

    def __init__(self, w, R, t):
        self.w, self.R, self.t = w, R, t
        self.constant = getattr(w, "constant", None)

    def __call__(self, Y):
        return self.w((np.atleast_2d(Y) - self.t) * self.R) if self.constant is None else \
            np.full(len(np.atleast_2d(Y)), self.constant)


def transform(piece, R, t):
    """The image of a piece (the piece itself for the identity)."""
    R = np.broadcast_to(np.asarray(R, float), np.shape(piece.centre)).copy()
    t = np.broadcast_to(np.asarray(t, float), np.shape(piece.centre)).copy()
    if np.all(R == 1) and not np.any(t):
        return piece
    key = ("image", R.tobytes(), t.tobytes())
    memo = piece.__dict__.setdefault("_memo", {})
    if key not in memo:
        memo[key] = _make(piece, R, t)
    return memo[key]


def _make(piece, R, t):
    from .weighted import CurvePiece, SegmentPiece, SpherePiece
    from .weighted_volume import ParallelogramPiece, RoundPiece, TrianglePiece

    def m(y):
        return np.asarray(y, float) * R + t
    w = _Mapped(piece.w, R, t)
    if isinstance(piece, SegmentPiece):
        return SegmentPiece(m(piece.p), m(piece.q), w)
    if isinstance(piece, SpherePiece):
        return SpherePiece(m(piece.c), piece.R, w)
    if isinstance(piece, RoundPiece):
        return RoundPiece(m(piece.c), piece.R, w)
    if isinstance(piece, TrianglePiece):
        return TrianglePiece(*m(piece.V), w)
    if isinstance(piece, ParallelogramPiece):
        return ParallelogramPiece(m(piece.o), piece.E * R, w)
    from .weighted_solids import TetraPiece
    if isinstance(piece, TetraPiece):
        return TetraPiece(*m(piece.V), w)
    from .weighted_solids import ParallelepipedPiece, ShellPiece
    if isinstance(piece, ShellPiece):
        return ShellPiece(m(piece.c), piece.r, piece.R, w)
    if isinstance(piece, ParallelepipedPiece):
        return ParallelepipedPiece(m(piece.o), piece.E * R, w)
    if isinstance(piece, CurvePiece):
        from ...domains import Curve
        C = piece.c
        g, dg = C.point, C.velocity
        return CurvePiece(Curve(lambda s: g(s) * R + t, C.t0, C.t1, derivative=lambda s: dg(s) * R,
                                breaks=C.breaks, closed=C.closed), w)
    raise TypeError(f"no image of {type(piece).__name__}")
