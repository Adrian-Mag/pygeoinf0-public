"""Axis-parallel boxes and parallelepipeds as weighted pieces (EXP-031 O3; 3-D boxes were refused after W3).

    ParallelepipedPiece   w dx on o + a E (a in [0, 1]^3) in a 3-D domain. Same piece: Duffy on the autocorrelation,
                          E = |det E|^2 int_{[-1, 1]^3} k(|z E|) A(z) dz, A(z) = int w(x(a)) w(x(a + z)) da over the
                          overlap box (closed form for constant weights, Gauss otherwise); A(-z) = A(z), so four
                          octants, each cube split into three pyramids at z = 0 (Jacobian rho^2), rho graded at 0.
                          Potentials: a tensor rule for far points; near points by spherical coordinates about x with
                          the exact entry and exit distances of each ray (slabs), the radial variable graded at 0
                          (directions have kinks across the faces' and edges' directions: algebraic in the number of
                          directions; the two levels give the estimate).
    box_pair_energy       two axis-parallel boxes (2-D or 3-D; touching, overlapping or apart): E = int k(|z|) C(z) dz
                          over z = y - x in B_b - B_a, C the cross-correlation (closed form for constant weights), the
                          z-box split at 0 and where the overlap changes regime, the parts with a corner at 0 by the
                          Duffy split there (d simplices of the cube), the others by tensor rules graded at both ends.

From weighted_tetra.py (merged here by plan 063):
Tetrahedra as weighted pieces (EXP-031 O3): the cells of a Simplicial set of dimension 3.

Same piece, in reference coordinates x = v0 + u E (u in the simplex T = {u >= 0, u1 + u2 + u3 <= 1}):
E = |det E|^2 int_{T - T} k(|z E|) C(z) dz, C(z) = int over T cap (T - z) of w(x(u)) w(x(u + z)) du, and T cap (T - z)
is the homothetic simplex with corner max(0, -z) and size s(z) = min(1, 1 - sum z) - sum max(0, -z_i) (closed form
s^3 / 6 for constant weights, the collapsed rule otherwise). s is affine on the cells of the difference body cut by
the planes z_i = 0 and sum z = 0; each cell is a convex polytope with the origin on its boundary, split into cones from
0 over its facets (ConvexHull), and every cone integrated by Duffy from its apex (rho graded at 0, the opposite
triangle by the collapsed rule): C is a polynomial on each cone for polynomial weights, so the rule converges
exponentially, and the singular point z = 0 is only ever at an apex. Built once (geometry-free) and cached.
Potentials: a tensor rule far away; near points by spherical coordinates about x with the exact entry and exit
distances of each ray (barycentric coordinates are affine along it). Pairs of tetrahedra that touch go through the
generic route (outer rule x potential): correct with its estimate, but slow and algebraic in the outer rule.
"""


import itertools

import numpy as np

from ...quadrature import both_ends, gauss, graded_unit, panels, simplex
from .weighted import ORDER, Piece, _cells, far_split, kfin

OUTER = 1e-4


def _const(w):
    return getattr(w, "constant", None)


def axis_aligned(p):
    from .weighted_volume import ParallelogramPiece
    if not isinstance(p, (ParallelogramPiece, ParallelepipedPiece)):
        return False
    E = p.E
    return E.shape[0] == E.shape[1] and np.all(E[~np.eye(len(E), dtype=bool)] == 0) and np.all(np.diag(E) > 0)


def _tensor_weights(ws):
    W = np.ones([len(w) for w in ws])
    for k, w in enumerate(ws):
        shape = [1] * len(ws)
        shape[k] = len(w)
        W = W * w.reshape(shape)
    return W.ravel()


ANGULAR = (6, 8)       # Gauss points per cell of the angular variables of the 3-D Duffy rules at the two levels


def _corner_rules(d, n, cells, diam=None, kern=None, level=1):
    """The cube [0, 1]^d as d Duffy simplices at the corner 0: points (m, d) of each and the weights. With diam and
    kern (3-D), the angular variables get cells growing with tau diam / l (capped at 3; cf. weighted_touching)."""
    tau, wt, _ = graded_unit(n, cells)
    if d == 2 or kern is None:
        q, wq = panels(np.linspace(0, 1, cells + 1), n)
        rules = [(tau, wt, q, wq)]
    else:
        rules = []
        c = np.clip(np.ceil(tau * diam / kern.l).astype(int), 1, 3)
        for cc in np.unique(c):
            q, wq = panels(np.linspace(0, 1, cc + 1), ANGULAR[level])
            rules.append((tau[c == cc], wt[c == cc], q, wq))
    out, Ws = [[] for _ in range(d)], []
    for t, w, q, wq in rules:
        grids = np.meshgrid(t, *([q] * (d - 1)), indexing="ij")
        T, U = grids[0].ravel(), [g.ravel() for g in grids[1:]]
        Ws.append(_tensor_weights([w] + [wq] * (d - 1)) * T ** (d - 1))
        for k in range(d):
            cols = [T * u for u in U]
            cols.insert(k, T)
            out[k].append(np.stack(cols, axis=1))
    return [np.vstack(o) for o in out], np.concatenate(Ws)


def box_pair_energy(kern, a, b, level):
    """See the module docstring."""
    d = len(a.o)
    n = ORDER[level]
    alo, ahi = a.o, a.o + np.diag(a.E)
    blo, bhi = b.o, b.o + np.diag(b.E)
    zlo, zhi = blo - ahi, bhi - alo
    brk = [blo - alo, bhi - ahi]
    cuts = [np.unique(np.r_[zlo[k], [x[k] for x in brk if zlo[k] < x[k] < zhi[k]],
                            [0.0] if zlo[k] < 0 < zhi[k] else [], zhi[k]]) for k in range(d)]
    cells = _cells(float(np.linalg.norm(zhi - zlo)), kern, 1)
    corner, Wc = _corner_rules(d, n, cells, float(np.linalg.norm(zhi - zlo)), kern if d == 3 else None, level)
    total = 0.0
    for box in itertools.product(*[list(zip(c[:-1], c[1:])) for c in cuts]):
        c0 = np.array([x0 if abs(x0) <= abs(x1) else x1 for x0, x1 in box])          # the corner nearest to 0
        L = np.array([(x1 - x0) * (1 if c == x0 else -1) for (x0, x1), c in zip(box, c0)])
        if not np.any(c0):
            parts = [(c0 + P * L, Wc) for P in corner]
        else:
            rules = [both_ends(n, _cells(abs(li), kern, 1), 1e-6) if d == 2 else
                     both_ends(ANGULAR[level], max(1, min(3, _cells(abs(li), kern, 1))), 1e-3) for li in L]
            G = np.meshgrid(*[r[0] for r in rules], indexing="ij")
            parts = [(c0 + np.stack([g.ravel() for g in G], 1) * L, _tensor_weights([r[1] for r in rules]))]
        vol = float(np.prod(np.abs(L)))
        for Z, w in parts:
            for i in range(0, len(Z), 8192):
                z = Z[i:i + 8192]
                total += vol * float(np.sum(w[i:i + 8192] * kfin(kern, np.linalg.norm(z, axis=1))
                                            * _cross(a, b, z, level)))
    return total


def _cross(a, b, Z, level, block=2048):
    """C(z) = int over B_a cap (B_b - z) of w_a(x) w_b(x + z) dx (boxes in d = 2 or 3)."""
    d = len(a.o)
    alo, ahi = a.o, a.o + np.diag(a.E)
    blo, bhi = b.o, b.o + np.diag(b.E)
    lo = np.maximum(alo, blo - Z)
    ln = np.maximum(np.minimum(ahi, bhi - Z) - lo, 0.0)
    ca, cb = _const(a.w), _const(b.w)
    if ca is not None and cb is not None:
        return ca * cb * np.prod(ln, axis=1)
    g, gw = gauss((12, 16)[level] if d == 2 else (6, 8)[level])
    g, gw = 0.5 * (g + 1), 0.5 * gw
    G = np.stack([x.ravel() for x in np.meshgrid(*[g] * d, indexing="ij")], 1)
    Wg = _tensor_weights([gw] * d)
    out = np.empty(len(Z))
    for i in range(0, len(Z), block):
        P = lo[i:i + block, None, :] + ln[i:i + block, None, :] * G[None]
        Qp = P + Z[i:i + block, None, :]
        f = (a.w(P.reshape(-1, d)) * b.w(Qp.reshape(-1, d))).reshape(P.shape[:2])
        out[i:i + block] = np.prod(ln[i:i + block], axis=1) * (f @ Wg)
    return out


class ParallelepipedPiece(Piece):
    """w dx on the parallelepiped o + a E, a in [0, 1]^3, in a 3-D domain (module docstring)."""

    dim = 3

    def __init__(self, o, E, w):
        self.o, self.E, self.w = np.asarray(o, float), np.asarray(E, float), w
        self.det = abs(float(np.linalg.det(self.E)))
        self.V = np.array([self.o + np.array(c) @ self.E for c in itertools.product((0, 1), repeat=3)])
        self.centre = self.o + 0.5 * self.E.sum(axis=0)
        self.radius = float(np.max(np.linalg.norm(self.V - self.centre, axis=1)))
        self.diam = 2 * self.radius
        self.Einv = np.linalg.inv(self.E)

    def _tensor(self, rules):
        G = np.meshgrid(*[r[0] for r in rules], indexing="ij")
        X = self.o + np.stack([g.ravel() for g in G], 1) @ self.E
        return X, self.det * _tensor_weights([r[1] for r in rules]) * self.w(X)

    def nodes(self, kern, level):
        n = (8, 10)[level]
        return self._tensor([both_ends(n, _cells(np.linalg.norm(e), kern, 1), OUTER) for e in self.E])

    def formula_nodes(self, n=24):
        g, w = gauss(n)
        return self._tensor([(0.5 * (g + 1), 0.5 * w)] * 3)

    def rule(self, n):
        return self.formula_nodes(n)

    def potential(self, kern, X, level):
        return far_split(self, X, level, self._near, kern, lambda lv: self.formula_nodes((12, 16)[lv]))

    def _near(self, kern, X, level):
        from .weighted_volume import _sphere_rule
        U, Wu = _sphere_rule((16, 24)[level])
        xr, wr, _ = graded_unit(ORDER[level], _cells(self.diam, kern, 1), depth=1e-12)
        Ud = U @ self.Einv                                       # ray directions in reference coordinates
        out = np.empty(len(X))
        for m, x in enumerate(X):
            a0 = (x - self.o) @ self.Einv
            inside = (a0 >= 0) & (a0 <= 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = -a0 / Ud, (1 - a0) / Ud
            par = Ud == 0
            lo = np.max(np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2)), axis=1)
            hi = np.min(np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2)), axis=1)
            rin = np.maximum(lo, 0.0)
            Ln = np.where(hi > rin, hi - rin, 0.0)
            rho = rin[:, None] + Ln[:, None] * xr[None]
            Y = x + rho[..., None] * U[:, None, :]
            f = kfin(kern, rho) * rho ** 2 * self.w(Y.reshape(-1, 3)).reshape(rho.shape)
            out[m] = float(Wu @ np.sum(f * Ln[:, None] * wr, axis=1))
        return out

    def self_energy(self, kern, level, other=None):
        """Same piece; with other (the same parallelepiped, another weight) the cross energy (symmetrised)."""
        corner, Wc = _corner_rules(3, ORDER[level], _cells(self.diam, kern, 1), self.diam, kern, level)
        total = 0.0
        for s2, s3 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):            # four octants, x 2 by A(-z) = A(z)
            for P in corner:
                Z = P * np.array([1.0, s2, s3])
                for i in range(0, len(Z), 8192):
                    z = Z[i:i + 8192]
                    C = self._auto(z, level, self) if other is None else \
                        0.5 * (self._auto(z, level, other) + other._auto(z, level, self))
                    total += 2 * float(np.sum(Wc[i:i + 8192] * kfin(kern, np.linalg.norm(z @ self.E, axis=1)) * C))
        return self.det ** 2 * total

    def _auto(self, Z, level, v):
        """int over a in [0,1]^3 with a + z in [0,1]^3 of w(x(a)) v(x(a + z)) da."""
        ln = 1 - np.abs(Z)
        c, cv = _const(self.w), _const(v.w)
        if c is not None and cv is not None:
            return c * cv * np.prod(ln, axis=1)
        lo = np.maximum(0.0, -Z)
        g, gw = gauss((6, 8)[level])                            # exact for weights of degree <= 5 (7) per axis
        g, gw = 0.5 * (g + 1), 0.5 * gw
        G = np.stack([x.ravel() for x in np.meshgrid(g, g, g, indexing="ij")], 1)
        Wg = _tensor_weights([gw] * 3)
        out = np.empty(len(Z))
        for i in range(0, len(Z), 512):
            A = lo[i:i + 512, None, :] + ln[i:i + 512, None, :] * G[None]
            P = self.o + A @ self.E
            Q = P + (Z[i:i + 512] @ self.E)[:, None, :]
            f = (self.w(P.reshape(-1, 3)) * v.w(Q.reshape(-1, 3))).reshape(A.shape[:2])
            out[i:i + 512] = f @ Wg
        return np.prod(ln, axis=1) * out

    def same(self, other, tol=1e-12):
        return isinstance(other, ParallelepipedPiece) and \
            np.allclose(other.o, self.o, atol=tol * self.diam, rtol=0) and \
            np.allclose(other.E, self.E, atol=tol * self.diam, rtol=0)


class ShellPiece(Piece):
    """w dx on the shell r <= |x - centre| <= R (3-D): the outer ball minus the inner one, with the same weight (w is
    sampled inside the inner ball too, so it must be defined there). Same piece: E(B_R) + E(B_r) - 2 E(B_R, B_r), the
    cross term for constant weights int k(rho) 4 pi rho^2 V(rho) drho with V the volume of two intersecting balls at
    distance rho (closed form), split at rho = R - r; for other weights by the round-pair route (weighted_contacts.py)."""

    dim = 3

    def __init__(self, c, r, R, w):
        from .weighted_volume import RoundPiece
        self.c, self.r, self.R, self.w = np.asarray(c, float), float(r), float(R), w
        self.cw = _const(w)
        self.outer, self.inner = RoundPiece(c, R, w), RoundPiece(c, r, w)
        self.centre, self.radius, self.diam = self.c, self.R, 2 * self.R

    def _rule(self, nr, level):
        from .weighted_volume import _sphere_rule
        g, gw = gauss(nr)
        rr = self.r + (self.R - self.r) * 0.5 * (g + 1)
        wr = 0.5 * (self.R - self.r) * gw * rr * rr
        U, Wu = _sphere_rule((12, 16)[level])
        X = self.c + (rr[:, None, None] * U[None]).reshape(-1, 3)
        return X, np.outer(wr, Wu).ravel() * self.w(X)

    def nodes(self, kern, level):
        return self._rule((16, 24)[level], level)

    def formula_nodes(self):
        return self._rule(24, 1)

    def rule(self, n):
        return self._rule(n, 1)

    def potential(self, kern, X, level):
        return self.outer.potential(kern, X, level) - self.inner.potential(kern, X, level)

    def self_energy(self, kern, level):
        if self.cw is None:
            from .weighted_contacts import round_pair_energy
            return (self.outer.self_energy(kern, level) + self.inner.self_energy(kern, level)
                    - 2 * round_pair_energy(kern, self.outer, self.inner, level))
        n, R, r = ORDER[level], self.R, self.r
        cells = _cells(2 * R, kern, 1)

        def V(rho):
            out = np.full(rho.shape, 4 * np.pi * r ** 3 / 3)
            m = rho > R - r
            q = rho[m]
            out[m] = np.pi * (R + r - q) ** 2 * (q * q + 2 * q * r - 3 * r * r + 2 * q * R + 6 * r * R - 3 * R * R) / (12 * q)
            return out
        x, w, _ = graded_unit(n, cells)
        rho1, w1 = (R - r) * x, (R - r) * w                    # [0, R - r]: V constant, graded at 0
        x2, w2 = both_ends(n, cells, 1e-10)
        rho2, w2 = (R - r) + 2 * r * x2, 2 * r * w2            # [R - r, R + r]
        cross = 4 * np.pi * (float(w1 @ (kern(rho1) * rho1 ** 2 * V(rho1))) + float(w2 @ (kern(rho2) * rho2 ** 2 * V(rho2))))
        return self.outer.self_energy(kern, level) + self.inner.self_energy(kern, level) - 2 * self.cw ** 2 * cross


# ---------------------------------------------------------------- from weighted_tetra.py (plan 063)
_CONES = None


def _cones():
    """The cones (a, b, c) of the difference body T - T from 0 (module docstring)."""
    global _CONES
    if _CONES is not None:
        return _CONES
    from scipy.spatial import ConvexHull
    planes = []                                                  # (n, c): n . z <= c
    for i in range(3):
        planes += [(np.eye(3)[i], 1.0), (-np.eye(3)[i], 1.0)]
    cones = []
    for signs in itertools.product((1, -1), repeat=4):
        H = list(planes)
        for i in range(3):
            H.append((-signs[i] * np.eye(3)[i], 0.0))            # signs[i] z_i >= 0
        H.append((-signs[3] * np.ones(3), 0.0))                   # signs[3] sum z >= 0
        neg = np.array([1.0 if s < 0 else 0.0 for s in signs[:3]])
        lin = -np.ones(3) * (signs[3] > 0) + neg                  # s(z) = [1 or 1 - sum z] + sum_{z_i < 0} z_i >= 0
        H.append((-lin, 1.0))
        V = []
        for (n1, c1), (n2, c2), (n3, c3) in itertools.combinations(H, 3):
            A = np.array([n1, n2, n3])
            if abs(np.linalg.det(A)) < 1e-12:
                continue
            z = np.linalg.solve(A, [c1, c2, c3])
            if all(n @ z <= c + 1e-12 for n, c in H):
                V.append(z)
        if len(V) < 4:
            continue
        V = np.unique(np.round(np.array(V), 12), axis=0)
        try:
            hull = ConvexHull(V)
        except Exception:                                         # a degenerate (flat) cell has no volume
            continue
        for facet, eq in zip(hull.simplices, hull.equations):
            if abs(eq[3]) > 1e-12:                                # facets whose plane misses 0: cones from 0
                cones.append(V[facet])
    _CONES = cones
    return cones


def _size(Z):
    return np.minimum(1.0, 1.0 - Z.sum(axis=1)) - np.maximum(0.0, -Z).sum(axis=1)


class TetraPiece(Piece):
    """w dx on the tetrahedron v0 v1 v2 v3 (module docstring)."""

    dim = 3

    def __init__(self, v0, v1, v2, v3, w):
        self.V = np.array([v0, v1, v2, v3], float)
        self.o, self.E, self.w = self.V[0], self.V[1:] - self.V[0], w
        self.det = abs(float(np.linalg.det(self.E)))
        self.Einv = np.linalg.inv(self.E)
        self.centre = self.V.mean(axis=0)
        self.radius = float(np.max(np.linalg.norm(self.V - self.centre, axis=1)))
        self.diam = 2 * self.radius

    def _ref(self, n):
        return simplex(3, n)

    def nodes(self, kern, level):
        U, W = self._ref((10, 14)[level] + _cells(self.diam, kern, 1))
        X = self.o + U @ self.E
        return X, self.det * W * self.w(X)

    def formula_nodes(self, n=20):
        U, W = self._ref(n)
        X = self.o + U @ self.E
        return X, self.det * W * self.w(X)

    def rule(self, n):
        return self.formula_nodes(n)

    def potential(self, kern, X, level):
        return far_split(self, X, level, self._near, kern, lambda lv: self.formula_nodes((12, 16)[lv]))

    def _near(self, kern, X, level):
        from .weighted_volume import _sphere_rule
        U, Wu = _sphere_rule((16, 24)[level])
        xr, wr, _ = graded_unit(ORDER[level], _cells(self.diam, kern, 1), depth=1e-12)
        D = U @ self.Einv                                        # u-velocity of each ray
        Db = np.c_[-D.sum(axis=1), D]                             # barycentric velocities (lambda0, u1, u2, u3)
        out = np.empty(len(X))
        for m, x in enumerate(X):
            u0 = (x - self.o) @ self.Einv
            b0 = np.r_[1 - u0.sum(), u0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = -b0[None, :] / Db                              # where lambda_i(x + rho e) = 0
            enter = np.where(Db > 0, t, np.where(Db < 0, -np.inf, np.where(b0 >= 0, -np.inf, np.inf)))
            leave = np.where(Db < 0, t, np.where(Db > 0, np.inf, np.where(b0 >= 0, np.inf, -np.inf)))
            rin = np.maximum(np.max(enter, axis=1), 0.0)
            Ln = np.where(np.min(leave, axis=1) > rin, np.min(leave, axis=1) - rin, 0.0)
            rho = rin[:, None] + Ln[:, None] * xr[None]
            Y = x + rho[..., None] * U[:, None, :]
            f = kfin(kern, rho) * rho ** 2 * self.w(Y.reshape(-1, 3)).reshape(rho.shape)
            out[m] = float(Wu @ np.sum(f * Ln[:, None] * wr, axis=1))
        return out

    def self_energy(self, kern, level, other=None):
        n = (16, 24)[level]                                  # measured: 1e-11 between these two (s = 1, l = 0.3)
        r, wr, _ = graded_unit(n, max((2, 3)[level], _cells(self.diam, kern, 1)))
        Ut, Wt = simplex(2, (12, 16)[level])
        total = 0.0
        for a, b, c in _cones():
            jac = abs(float(np.linalg.det(np.array([a, b, c]))))           # dz = rho^2 |det(a, b, c)| drho dp dq
            Y = a + Ut[:, :1] * (b - a) + Ut[:, 1:] * (c - a)               # (q, 3) on the opposite face
            Z = (r[:, None, None] * Y[None]).reshape(-1, 3)
            W = (wr[:, None] * r[:, None] ** 2 * Wt[None]).ravel() * jac
            C = self._corr(Z, level, self if other is None else other)
            total += float(np.sum(W * kfin(kern, np.linalg.norm(Z @ self.E, axis=1)) * C))
        return self.det ** 2 * total

    def _corr(self, Z, level, v, block=1024):
        s = np.maximum(_size(Z), 0.0)
        c, cv = getattr(self.w, "constant", None), getattr(v.w, "constant", None)
        if c is not None and cv is not None:
            return c * cv * s ** 3 / 6
        U, W = self._ref((6, 8)[level])
        m = np.maximum(0.0, -Z)
        out = np.empty(len(Z))
        for i in range(0, len(Z), block):
            A = m[i:i + block, None, :] + s[i:i + block, None, None] * U[None]
            P = self.o + A @ self.E
            Q = P + (Z[i:i + block] @ self.E)[:, None, :]
            out[i:i + block] = (self.w(P.reshape(-1, 3)) * v.w(Q.reshape(-1, 3))).reshape(A.shape[:2]) @ W
        return s ** 3 * out

    def same(self, other, tol=1e-12):
        return isinstance(other, TetraPiece) and np.allclose(other.V, self.V, atol=tol * self.diam, rtol=0)
