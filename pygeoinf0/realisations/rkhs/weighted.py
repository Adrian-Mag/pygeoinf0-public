"""Weighted atoms (EXP-031 W3, DESIGN 5.2): pieces of weighted measures w dsigma whose energies and potentials need
routes of their own. An atom of kind WEIGHTED (atoms.py) carries one piece object; every piece offers

    nodes(kern, level)           an outer rule (points, weights with w folded in) graded towards its ends and edges
    potential(kern, X, level)    int K(x, y) w(y) dsigma(y) at points X, singularity-aware (x on or near the piece)
    self_energy(kern, level)     int int K w w dsigma dsigma by the same-piece route
    formula_nodes()              a high-order rule for int f w dsigma (inner products with formulas)

each at two levels (LOW = 0, HIGH = 1) whose difference is the error estimate. Same-piece routes:

    piece        route                                                              DESIGN 5.2 row
    segment      2 int_0^L k(u) A(u) du, A the autocorrelation of w (Gauss),        same flat piece (P4)
                 u graded at 0, the innermost cell from k's expansion (Matern.small)
    curve        the chart route: outer Gauss in t graded at the ends; the inner    chart pieces (P6)
                 integral split at t' = t and graded towards it, the cell next to t
                 from k's expansion at |gamma'(t)| h
    sphere       geodesic polar coordinates about each outer node (Jacobian         chart pieces / sphere (P2)
                 R^2 sin(alpha) cancels the singularity), alpha graded at 0
Two segments sharing an end point: the common-vertex Duffy split (Sauter-Schwab classification in one dimension, P4).
Pieces clearly apart (disjoint_energy): plain Gauss rules on both, of an order growing with size / gap. Other pairs: the
outer rule of the piece of lower dimension against the potential of the other (graded, two levels). Segments with a
declared singular weight: SingularSegmentPiece. The other modules: weighted_volume.py (parallelograms, patches,
triangles, disks, balls), weighted_solids.py (3-D boxes, box pairs, shells), weighted_solids.py, weighted_contacts.py
(common edges and vertices), weighted_contacts.py (pairs of disks or balls), weighted_contacts.py (lattice and reflection
images), weighted_spectral.py (the spectral engine's node rules and near corrections); the choice of route for a pair
is pairs._weighted_one.
"""
import math

import numpy as np

from ...quadrature import DEPTH, OUTER_DEPTH, RATIO, both_ends, gauss, graded_unit, panels

ORDER = (12, 16)             # Gauss points per panel at the two levels
# the geometric grading towards singular points (ratio RATIO, smallest cell DEPTH relative) and of outer rules towards
# ends and edges (OUTER_DEPTH) are the defaults of quadrature.graded_unit and quadrature.both_ends


def _cells(length, kern, factor=1.0):
    return max(1, int(math.ceil(factor * length / kern.l)))


def kfin(kern, r):
    """k(r), with 0 where r = 0 to rounding (that point is the head cell's, see head)."""
    return np.where(r > 0, kern(np.where(r > 0, r, 1.0)), 0.0)


def head(kern, h, dist, wt, scale):
    """int over the innermost cell of length h next to the nearest point (at distance dist) of w K: the expansion of
    k at 0 (Matern.small) when the point is on the piece (dist < h, or within 1e-12 of the piece's size: rounding),
    k(dist) h otherwise."""
    on = (dist < h) | (dist <= 1e-12 * scale)
    return wt * np.where(on, kern.small(h), kern(np.where(on, 1.0, dist)) * h)


class Piece:
    """The interface of a weighted piece (module docstring)."""

    dim = None
    centre, radius = None, None

    def rule(self, n):
        """A plain Gauss rule of order n per axis on the piece (weight folded in): the far field and disjoint pairs."""
        return self.formula_nodes(n)

    def formula_low(self):
        """The second, lower order of formula_nodes (3/4 of its default order): the pairings with formulas are
        computed at both and their difference is the error estimate (EXP-031 O4)."""
        import inspect
        p = inspect.signature(self.formula_nodes).parameters.get("n")
        if p is not None:
            return self.formula_nodes(max(8, int(0.75 * p.default)))
        return self.rule(18 if self.dim == 3 else 36)

    def cached_self_energy(self, kern, level):
        """self_energy, computed once per kernel and level (a piece is shared by the subsets of its atoms)."""
        key = (kern.s, kern.d, kern.l, level)
        memo = self.__dict__.setdefault("_memo", {})
        if key not in memo:
            memo[key] = self.self_energy(kern, level)
        return memo[key]


def far_split(piece, X, level, near_potential, kern, far_nodes):
    """The potential at X: for points at least one bounding radius away from the piece, by its rule far_nodes(level)
    (the integrand is smooth there; vectorised), otherwise by near_potential(kern, X, level)."""
    X = np.atleast_2d(np.asarray(X, float))
    far = np.linalg.norm(X - piece.centre, axis=1) >= 2.0 * piece.radius
    out = np.empty(len(X))
    if far.any():
        Y, W = far_nodes(level)
        F = X[far]
        for i in range(0, len(F), 512):
            r = np.linalg.norm(F[i:i + 512, None, :] - Y[None], axis=-1)
            out[np.flatnonzero(far)[i:i + 512]] = kern(r) @ W
    if (~far).any():
        out[~far] = near_potential(kern, X[~far], level)
    return out


def disjoint_energy(kern, a, b, level):
    """int int K w_a w_b for two pieces clearly apart (bounding balls disjoint, or a node gap above 5 % of their size):
    plain Gauss rules on both, of an order growing with size / gap (the near-singularity of K at the gap's scale), at
    two levels; None when the pieces may touch or overlap, or the rules would be too large."""
    size = 2 * max(a.radius, b.radius)
    g = float(np.linalg.norm(a.centre - b.centre)) - a.radius - b.radius          # bounding balls apart: a lower bound
    if g <= 0:                                  # otherwise the rules' nodes, which stay inside the pieces: pieces that
        n = 8 if max(a.dim, b.dim) == 3 else 24   # touch show a node gap of ~size / n^2, so ask for a clear margin
        Xa, _ = a.rule(n)
        Xb, _ = b.rule(n)
        g = min(float(np.min(np.linalg.norm(Xa[i:i + 512, None] - Xb[None], axis=-1))) for i in range(0, len(Xa), 512))
        if g <= 0.05 * size:
            return None
        g -= 0.01 * size
    k = max(a.dim, b.dim)
    n = int(min({1: 256, 2: 64, 3: 24}[k], 12 + 6 * size / g)) * (2 + level) // 3
    Xa, Wa = a.rule(n)
    Xb, Wb = b.rule(n)
    if len(Xa) * len(Xb) > 5e7:
        return None
    total = 0.0
    for i in range(0, len(Xa), 1024):
        total += float(Wa[i:i + 1024] @ kern(np.linalg.norm(Xa[i:i + 1024, None] - Xb[None], axis=-1)) @ Wb)
    return total


class SegmentPiece(Piece):
    """w ds on the segment from p to q (flat coordinates; a subinterval of an interval domain in one dimension)."""

    dim = 1

    def __init__(self, p, q, w):
        self.p, self.q = np.asarray(p, float), np.asarray(q, float)
        e = self.q - self.p
        self.L = float(np.linalg.norm(e))
        self.u, self.w = e / self.L, w
        self.centre, self.radius = 0.5 * (self.p + self.q), 0.5 * self.L
        self.ends = (self.p, self.q)

    def wt(self, t):
        t = np.asarray(t, float)
        return self.w((self.p + t.reshape(-1, 1) * self.u)).reshape(t.shape)

    def nodes(self, kern, level):
        x, w = both_ends(ORDER[level], _cells(self.L, kern, 2))
        t = self.L * x
        return self.p + t[:, None] * self.u, self.L * w * self.wt(t)

    def rule(self, n):
        t, w = panels([0.0, self.L], n)
        return self.p + t[:, None] * self.u, w * self.wt(t)

    def formula_nodes(self, n=128):
        """The rule for int f w ds: n points, in panels of 32 bisected where w is not resolved (plan 022 P2)."""
        if getattr(self.w, "constant", None) is not None:
            return self.rule(n)
        from ...core.adaptive import adapted_panels
        t, w, _ = adapted_panels(np.linspace(0.0, self.L, max(1, -(-n // 32)) + 1), 32, self.wt)
        return self.p + t[:, None] * self.u, w * self.wt(t)

    def potential(self, kern, X, level):
        X = np.atleast_2d(np.asarray(X, float))
        rel = X - self.p
        t0 = rel @ self.u
        delta = np.linalg.norm(rel - t0[:, None] * self.u, axis=1)          # no cancellation for points on the line
        tc = np.clip(t0, 0.0, self.L)
        dist = np.hypot(tc - t0, delta)
        x, w, h = graded_unit(ORDER[level], _cells(self.L, kern, 2))
        out = np.zeros(len(X))
        wc = self.wt(tc)
        for side, S in ((-1.0, tc), (1.0, self.L - tc)):
            T = tc[:, None] + side * S[:, None] * x[None]
            r = np.sqrt(((tc - t0)[:, None] + side * S[:, None] * x[None]) ** 2 + delta[:, None] ** 2)   # no cancellation
            out += np.sum(S[:, None] * w[None] * kfin(kern, r) * self.wt(T), axis=1)
            out += np.where(S > 0, head(kern, S * h, dist, wc, self.L), 0.0)
        return out

    def self_energy(self, kern, level):
        n, L = ORDER[level], self.L
        x, w, h = graded_unit(n, _cells(L, kern, 2))
        u, wu = L * x, L * w
        g, gw = gauss((24, 32)[level])
        g = 0.5 * (g + 1)
        T = (L - u)[:, None] * g[None]
        A = (L - u) * np.sum(0.5 * gw * self.wt(T) * self.wt(T + u[:, None]), axis=1)
        A0 = L * float(0.5 * gw @ (self.wt(L * g) ** 2))
        return 2 * (float(wu @ (kern(u) * A)) + A0 * float(kern.small(L * h)))


SING_DEPTH = 1e-15     # grading towards a declared singular end; deeper is meaningless: the weight is a function of
                       # Cartesian points, and p + t u rounds to p below t ~ 1e-16 |p|. The omitted innermost cell
                       # costs about SING_DEPTH^(1 - a) relative (3e-11 at a = 0.3, 3e-8 as a -> 1/2)


def _graded_two(n, cells, depth0, depth1):
    """Gauss panels on [0, 1] graded towards 0 (to depth0; the innermost cell [0, depth0 / 2] left out: the weight
    cannot be sampled closer to its singular point) and towards 1 (to depth1, innermost cell included)."""
    k0 = int(math.ceil(math.log(depth0) / math.log(RATIO)))
    k1 = int(math.ceil(math.log(depth1) / math.log(RATIO)))
    e = np.r_[0.5 * RATIO ** np.arange(k0, 0, -1), np.linspace(0.5 * RATIO, 1 - 0.5 * RATIO, cells + 1),
              1 - 0.5 * RATIO ** np.arange(1, k1 + 1), 1.0]
    return panels(np.unique(e), n)


class SingularSegmentPiece(SegmentPiece):
    exponent = None

    @property
    def floor(self):
        """The relative error of the omitted innermost cells, SING_DEPTH^(1 - a) (times 10): not seen by two levels."""
        return 10 * SING_DEPTH ** (1 - float(self.exponent))
    """A segment piece whose weight has a declared singularity |w| ~ t^-a (a < 1/2: w in L2 of the piece) at its end
    p (atoms split segments at their singular points): every rule is graded geometrically towards p. Samples that
    round onto the singular point itself (products of graded variables below 1e-16 |p|) count as 0: their cells
    have measure below 1e-30."""

    def wt(self, t):
        v = SegmentPiece.wt(self, t)
        return np.where(np.isfinite(v), v, 0.0)

    def nodes(self, kern, level):
        x, w = _graded_two(ORDER[level], _cells(self.L, kern, 2), SING_DEPTH, OUTER_DEPTH)
        t = self.L * x
        return self.p + t[:, None] * self.u, self.L * w * self.wt(t)

    def formula_nodes(self, n=32):
        x, w = _graded_two(n, 2, SING_DEPTH, 1e-6)
        t = self.L * x
        return self.p + t[:, None] * self.u, self.L * w * self.wt(t)

    def potential(self, kern, X, level):
        X = np.atleast_2d(np.asarray(X, float))
        rel = X - self.p
        t0 = rel @ self.u
        delta = np.linalg.norm(rel - t0[:, None] * self.u, axis=1)
        tc = np.clip(t0, 0.0, self.L)
        dist = np.hypot(tc - t0, delta)
        n, cells = ORDER[level], _cells(self.L, kern, 2)
        x, w, h = graded_unit(n, cells)
        xs, ws = _graded_two(n, cells, DEPTH, SING_DEPTH)       # [0, tc] graded at tc (kernel) and at 0 (weight)
        out = np.zeros(len(X))
        T = tc[:, None] * xs[None]                               # from 0 to tc
        r = np.sqrt((T - t0[:, None]) ** 2 + delta[:, None] ** 2)
        out += np.sum(tc[:, None] * ws[None] * kfin(kern, r) * self.wt(T), axis=1)
        S = self.L - tc
        T = tc[:, None] + S[:, None] * x[None]
        r = np.sqrt(((tc - t0)[:, None] + S[:, None] * x[None]) ** 2 + delta[:, None] ** 2)
        out += np.sum(S[:, None] * w[None] * kfin(kern, r) * self.wt(T), axis=1)
        wc = np.where(tc > 0, self.wt(np.maximum(tc, 1e-300)), 0.0)
        return out + np.where(S > 0, head(kern, S * h, dist, np.nan_to_num(wc, posinf=0.0), self.L), 0.0)

    def self_energy(self, kern, level):
        n, L = ORDER[level], self.L
        x, w = _graded_two(n, _cells(L, kern, 2), DEPTH, SING_DEPTH)   # u graded at 0 (kernel) and at L: A(u) ~
        h = 0.5 * RATIO ** int(math.ceil(math.log(DEPTH) / math.log(RATIO)))  # (L - u)^(1 - a) there
        u, wu = L * x, L * w
        g, gw = _graded_two((16, 20)[level], 2, SING_DEPTH, 1e-6)
        T = (L - u)[:, None] * g[None]
        A = (L - u) * np.sum(gw * self.wt(T) * self.wt(T + u[:, None]), axis=1)
        A0 = L * float(gw @ (self.wt(L * g) ** 2))
        a = float(self.exponent)                                 # the omitted cell [0, h L] of w^2 ~ C^2 t^(-2a)
        t1 = L * float(g.min())
        C = float(self.wt(np.array([t1]))[0]) * t1 ** a
        h0 = L * 0.5 * RATIO ** int(math.ceil(math.log(SING_DEPTH) / math.log(RATIO)))     # _graded_two's cell
        A0 += C * C * h0 ** (1 - 2 * a) / (1 - 2 * a)
        return 2 * (float(wu @ (kern(u) * A)) + A0 * float(kern.small(L * h)))


def same_segment(a, b, tol=1e-12):
    """Whether two segment pieces are the same segment (either orientation)."""
    sc = max(a.L, b.L)
    return (np.linalg.norm(a.p - b.p) <= tol * sc and np.linalg.norm(a.q - b.q) <= tol * sc) or \
        (np.linalg.norm(a.p - b.q) <= tol * sc and np.linalg.norm(a.q - b.p) <= tol * sc)


def same_segment_energy(kern, a, b, level):
    """int int K w_a w_b for two weights on one segment: int_0^L k(u) (C_ab(u) + C_ba(u)) du, C_ab(u) = int w_a(t)
    w_b(t + u) dt (Gauss), u graded at 0 with the innermost cell from k's expansion (the segment route, P4)."""
    if np.linalg.norm(a.p - b.p) > 1e-12 * a.L:                         # the same points, opposite orientations
        b = SegmentPiece(a.p, a.q, b.w)
    n, L = ORDER[level], a.L
    x, w, h = graded_unit(n, _cells(L, kern, 2))
    u, wu = L * x, L * w
    g, gw = gauss((24, 32)[level])
    g, gw = 0.5 * (g + 1), 0.5 * gw
    T = (L - u)[:, None] * g[None]
    C = (L - u) * np.sum(gw * (a.wt(T) * b.wt(T + u[:, None]) + b.wt(T) * a.wt(T + u[:, None])), axis=1)
    C0 = 2 * L * float(gw @ (a.wt(L * g) * b.wt(L * g)))
    return float(wu @ (kern(u) * C)) + C0 * float(kern.small(L * h))


def shared_vertex(a, b, tol=1e-12):
    """The common end point of two segment pieces (and the orientation of each from it), or None."""
    scale = max(a.L, b.L)
    for i, pa in enumerate(a.ends):
        for j, pb in enumerate(b.ends):
            if np.linalg.norm(pa - pb) <= tol * scale:
                return i, j
    return None


def vertex_energy(kern, a, b, ij, level):
    """int int K w_a w_b ds dt for segments sharing a vertex: s, t the distances from it, the square [0, La] x [0, Lb]
    split into two triangles at the singular corner (Duffy), the radial variable graded at 0."""
    i, j = ij
    ua, ub = (a.u if i == 0 else -a.u), (b.u if j == 0 else -b.u)
    V = a.ends[i]
    n = ORDER[level]
    sing = isinstance(a, SingularSegmentPiece) or isinstance(b, SingularSegmentPiece)
    tau, wt, _ = graded_unit(n, 1, depth=SING_DEPTH if sing else DEPTH)
    cells = _cells(math.hypot(a.L, b.L), kern, 2)
    q, wq = _graded_two(n, cells, SING_DEPTH, 1e-6) if sing else panels(np.linspace(0, 1, cells + 1), n)
    T, Q = np.meshgrid(tau, q, indexing="ij")
    W = np.outer(wt, wq) * a.L * b.L * T
    tot = 0.0
    for s, t in ((a.L * T, b.L * T * Q), (a.L * T * Q, b.L * T)):
        r = np.linalg.norm(s[..., None] * ua - t[..., None] * ub, axis=-1)
        wa = a.w((V + s.reshape(-1, 1) * ua)).reshape(s.shape)
        wb = b.w((V + t.reshape(-1, 1) * ub)).reshape(t.shape)
        if sing:                                         # samples rounded onto the singular vertex (cf. wt)
            wa, wb = np.where(np.isfinite(wa), wa, 0.0), np.where(np.isfinite(wb), wb, 0.0)
        tot += float(np.sum(W * kern(r) * wa * wb))
    return tot


class CurvePiece(Piece):
    """w ds on a curve (domains.Curve) by the chart route (module docstring)."""

    dim = 1

    def __init__(self, curve, w):
        self.c, self.w = curve, w
        self.t0, self.t1, self.closed = curve.t0, curve.t1, curve.closed
        lo, hi = np.array(curve.bounding_box).T
        self.centre, self.radius = 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo))
        self.ends = () if self.closed else (curve.point(self.t0), curve.point(self.t1))

    def a(self, t):
        """w(gamma(t)) |gamma'(t)|."""
        t = np.asarray(t, float)
        tt = self.t0 + np.mod(t - self.t0, self.t1 - self.t0) if self.closed else t
        X = self.c.point(tt.ravel())
        return (self.w(X) * self.c.speed(tt.ravel())).reshape(t.shape)

    def nodes(self, kern, level, n=None):
        n = n or ORDER[level]
        edges = np.unique(np.r_[self.t0, self.c.breaks, self.t1])
        X, W = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            cells = _cells(self.c.arclength(hi) - self.c.arclength(lo), kern, 2)
            if self.closed:
                x, w = panels(np.linspace(0, 1, cells + 1), n)
            else:
                x, w = both_ends(n, cells)
            X.append(lo + (hi - lo) * x)
            W.append((hi - lo) * w)
        t, w = np.concatenate(X), np.concatenate(W)
        return self.c.point(t), w * self.a(t), t

    def formula_nodes(self, n=128):
        edges = np.unique(np.r_[self.t0, self.c.breaks, self.t1])
        t, w = panels(edges, n)
        return self.c.point(t), w * self.a(t)

    def potential(self, kern, X, level, tstar=None):
        X = np.atleast_2d(np.asarray(X, float))
        if tstar is None:
            tstar, dist = self.c.nearest(X)
        else:
            dist = np.zeros(len(X))
        T = self.t1 - self.t0
        x, w, h = graded_unit(ORDER[level], _cells(self.c.length, kern, 2))
        c = self.c.speed(tstar)
        wc = self.w(self.c.point(tstar))
        sides = ((tstar, tstar + T),) if self.closed else ((tstar, self.t0), (tstar, self.t1))
        out = np.zeros(len(X))
        for a, b in sides:                                  # from the nearest parameter a towards b
            if self.closed:                                 # graded towards both ends of [t*, t* + T]
                parts = ((a, a + 0.5 * T), (b, b - 0.5 * T))
            else:
                parts = ((a, b),)
            for lo, hi in parts:
                S = hi - lo
                tt = lo[:, None] + S[:, None] * x[None] if np.ndim(lo) else lo + S * x
                P = self.c.point(self.t0 + np.mod(tt - self.t0, T) if self.closed else tt)
                r = np.linalg.norm(P - X[:, None, :], axis=-1)
                out += np.sum(np.abs(S)[:, None] * w[None] * kfin(kern, r) * self.a(tt), axis=1)
                out += np.where(np.abs(S) > 0, head(kern, c * np.abs(S) * h, dist, wc, self.c.length), 0.0)
        return out

    def self_energy(self, kern, level):
        X, W, t = self.nodes(kern, level)
        return float(W @ self.potential(kern, X, level, tstar=t))


class SpherePiece(Piece):
    """w dA on the sphere |x - c| = R (3-D) by geodesic polar coordinates about each point (module docstring)."""

    dim = 2

    def __init__(self, c, R, w):
        self.c, self.R, self.w = np.asarray(c, float), float(R), w
        self.centre, self.radius = self.c, self.R

    def _rule(self, nt):
        t, wt = gauss(nt)
        nphi = 2 * nt
        phi = 2 * math.pi * np.arange(nphi) / nphi
        st = np.sqrt(1 - t * t)
        U = np.stack([(st[:, None] * np.cos(phi)).ravel(), (st[:, None] * np.sin(phi)).ravel(),
                      np.repeat(t, nphi)], axis=1)
        return U, np.repeat(wt, nphi) * 2 * math.pi / nphi * self.R ** 2

    def nodes(self, kern, level):
        U, W = self._rule((16, 24)[level])
        X = self.c + self.R * U
        return X, W * self.w(X)

    def formula_nodes(self, n=64):
        U, W = self._rule(n)
        X = self.c + self.R * U
        return X, W * self.w(X)

    def potential(self, kern, X, level, block=16):
        X = np.atleast_2d(np.asarray(X, float))
        R, l = self.R, kern.l
        alpha, wa = _graded_angle(l / R, ORDER[level])
        nb = (32, 48)[level]
        beta = 2 * math.pi * np.arange(nb) / nb
        ca, sa = np.cos(alpha), np.sin(alpha)
        out = np.empty(len(X))
        for i0 in range(0, len(X), block):
            V = X[i0:i0 + block] - self.c
            r = np.linalg.norm(V, axis=1)
            U = np.where(r[:, None] > 0, V / np.where(r > 0, r, 1)[:, None], np.array([0, 0, 1.0]))
            E1 = np.cross(U, np.where(np.abs(U[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
            E1 /= np.linalg.norm(E1, axis=1)[:, None]
            E2 = np.cross(U, E1)
            dirs = np.cos(beta)[None, :, None] * E1[:, None] + np.sin(beta)[None, :, None] * E2[:, None]
            Y = self.c + R * (ca[None, :, None, None] * U[:, None, None] + sa[None, :, None, None] * dirs[:, None])
            ang = self.w(Y.reshape(-1, 3)).reshape(Y.shape[:3]).sum(axis=2) * 2 * math.pi / nb
            rho = np.sqrt((r[:, None] - R) ** 2 + 4 * r[:, None] * R * np.sin(alpha[None] / 2) ** 2)
            out[i0:i0 + block] = np.sum(kfin(kern, rho) * R * R * sa * wa * ang, axis=1)
        return out

    def self_energy(self, kern, level):
        X, W = self.nodes(kern, level)
        return float(W @ self.potential(kern, X, level))


def _graded_angle(scale, n, depth=1e-12):
    """Gauss panels on [0, pi] graded towards 0 on the kernel's angular scale."""
    cell = min(scale, 0.25)
    k = int(math.ceil(math.log(depth) / math.log(RATIO)))
    h0 = min(scale, math.pi)
    e = np.r_[0.0, h0 * RATIO ** np.arange(k, -1, -1)]
    if math.pi > h0:
        e = np.r_[e, np.linspace(h0, math.pi, int(math.ceil((math.pi - h0) / cell)) + 1)[1:]]
    return panels(np.unique(e), n)
