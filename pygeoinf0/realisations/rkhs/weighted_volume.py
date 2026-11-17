"""Weighted volume pieces (EXP-031 W3, DESIGN 5.2): sets of full dimension inside a 2-D or 3-D domain with a weight w.

    piece             same-piece energy                                    potential at a point x
    parallelogram     Duffy: E = |det|^2 int k(|z E|) A(z) dz over the     the fan of signed triangles (x, V_k, V_k+1) in
    (2-D: Box, Patch) difference vectors z of reference coordinates, A    polar coordinates about x (edge substitution
                      the autocorrelation of w (closed form for constant  tau = d sinh(omega)), radial variable graded at 0;
                      w, Gauss on the overlap otherwise), split at z = 0  constant w: kernels.polygon_potential (exact)
                      into triangles, the radial variable graded at 0
    disk (2-D),       the lens route: E = int_0^2R k(rho) rho^(d-1) int    polar (spherical) coordinates about x with the
    ball (3-D)        A(rho e) de drho, A the integral of w(x) w(x + z)    exact entry and exit distances of each ray;
                      over the lens D cap (D - z) (closed-form lens area    outside points through the cone of rays that
                      or volume for constant w; Gauss in lens coordinates   meet the set (a sine substitution removes the
                      otherwise)                                           square-root ends)
Every route is evaluated at two levels (weighted.ORDER); the difference is the error estimate.
"""
import math

import numpy as np

from ...quadrature import both_ends, gauss, graded_unit, panels, simplex
from .weighted import ORDER, Piece, _cells, far_split, kfin


LENS = {2: (32, 48), 3: (8, 10)}     # Gauss points per lens coordinate at the two levels (a C^inf bump
                                     # needs 48 for 1e-11 in 2-D; polynomial weights are exact with far fewer)
LENS_THETA = {2: (32, 48), 3: (12, 16)}   # the same for the angle of the eta = H sin(theta) substitution
OUTER = 1e-4                         # grading of the outer rules of volume pieces towards their boundary
FAR = ((24, 32), (12, 16))           # Gauss points per axis of the far-field rules (2-D, 3-D) at the two levels


def _const(w):
    return getattr(w, "constant", None)


class ParallelogramPiece(Piece):
    """w dx on the parallelogram o + a e1 + b e2, (a, b) in [0, 1]^2: of full dimension in a 2-D domain, or a flat
    patch (area measure) in a 3-D one; the fan works in the plane's coordinates with the out-of-plane distance h."""

    dim = 2

    def __init__(self, o, E, w):
        self.o, self.E, self.w = np.asarray(o, float), np.asarray(E, float), w
        e1, e2 = self.E
        u1 = e1 / np.linalg.norm(e1)
        u2 = e2 - (e2 @ u1) * u1
        self.F = np.stack([u1, u2 / np.linalg.norm(u2)])                  # orthonormal frame of the plane (2, d)
        self.E2 = self.E @ self.F.T                                         # the edges in plane coordinates
        self.det = abs(float(np.linalg.det(self.E2)))
        self.V = np.array([self.o, self.o + e1, self.o + e1 + e2, self.o + e2])
        self.centre = self.o + 0.5 * (e1 + e2)
        self.radius = 0.5 * max(np.linalg.norm(e1 + e2), np.linalg.norm(e1 - e2))
        self.diam = 2 * self.radius

    def nodes(self, kern, level):
        n = ORDER[level]
        a, wa = both_ends(n, _cells(np.linalg.norm(self.E[0]), kern, 1), OUTER)
        b, wb = both_ends(n, _cells(np.linalg.norm(self.E[1]), kern, 1), OUTER)
        A, B = np.meshgrid(a, b, indexing="ij")
        X = self.o + A.reshape(-1, 1) * self.E[0] + B.reshape(-1, 1) * self.E[1]
        return X, self.det * np.outer(wa, wb).ravel() * self.w(X)

    def rule(self, n):
        g, w = gauss(n)
        g, w = 0.5 * (g + 1), 0.5 * w
        A, B = np.meshgrid(g, g, indexing="ij")
        X = self.o + A.reshape(-1, 1) * self.E[0] + B.reshape(-1, 1) * self.E[1]
        return X, self.det * np.outer(w, w).ravel() * self.w(X)

    def formula_nodes(self, n=48):
        """The rule for int f w dA: n points per axis, in cells of 16 x 16 quartered where w is not resolved (plan 022
        P2)."""
        if _const(self.w) is not None:
            return self.rule(n)
        from ...core.adaptive import adapted_square

        def at(a, b):
            return self.o + a.reshape(-1, 1) * self.E[0] + b.reshape(-1, 1) * self.E[1]
        a, b, w, _ = adapted_square(16, lambda a, b: self.w(at(a, b)).reshape(np.shape(a)), cells=max(1, -(-n // 16)))
        X = at(a, b)
        return X, self.det * w * self.w(X)

    def potential(self, kern, X, level, block=64):
        X = np.atleast_2d(np.asarray(X, float))
        c = _const(self.w)
        if c is not None:
            from .kernels import polygon_potential
            x2, h = self._plane(X)
            V2 = (self.V - self.o) @ self.F.T
            if len(V2) == 3:                                    # a triangle: a quadrilateral with an empty edge
                V2 = np.vstack([V2, V2[-1:]])
            return c * polygon_potential(kern, V2[None] - x2[:, None, :], h)
        return far_split(self, X, level, self._near, kern, lambda lv: self.formula_nodes(FAR[0][lv]))

    def _near(self, kern, X, level, block=64):
        out = np.empty(len(X))
        for i in range(0, len(X), block):
            out[i:i + block] = self._fan(kern, X[i:i + block], level)
        return out

    def _plane(self, X):
        """Plane coordinates (m, 2) of the points and their distances from the plane."""
        x2 = (X - self.o) @ self.F.T
        return x2, np.linalg.norm((X - self.o) - x2 @ self.F, axis=1)

    def _fan(self, kern, X3, level):
        n = ORDER[level]
        X, hh = self._plane(X3)
        orient = np.sign(np.linalg.det(self.E2))
        V2 = (self.V - self.o) @ self.F.T
        xr, wr, h = graded_unit(n, _cells(self.diam, kern, 1), depth=1e-12)
        total = np.zeros(len(X))
        nv = len(V2)
        for k in range(nv):
            Va, Vb = V2[k] - X, V2[(k + 1) % nv] - X
            cr = Va[:, 0] * Vb[:, 1] - Va[:, 1] * Vb[:, 0]
            Ed = Vb - Va
            Lk = np.linalg.norm(Ed, axis=1)
            T = Ed / Lk[:, None]
            d = np.abs(cr) / Lk
            live = d > 1e-14 * self.diam
            dd = np.where(live, d, 1.0)
            ta, tb = np.sum(Va * T, axis=1), np.sum(Vb * T, axis=1)
            nrm = (Va - ta[:, None] * T) / dd[:, None]                     # unit vector from x to the edge line
            w0, w1 = np.arcsinh(ta / dd), np.arcsinh(tb / dd)
            chunks = max(1, int(math.ceil(float(np.max(np.abs(w1 - w0)[live], initial=0.0)) / 3.0)))
            g, gw = panels(np.linspace(0, 1, chunks + 1), n)
            W = w0[:, None] + (w1 - w0)[:, None] * g[None]                 # (m, q)
            Ww = (w1 - w0)[:, None] * gw[None] / np.cosh(W)
            rho_e = dd[:, None] * np.cosh(W)
            dirs = (dd[:, None, None] * nrm[:, None, :] + (dd[:, None] * np.sinh(W))[..., None] * T[:, None, :]) \
                / rho_e[..., None]
            rho = rho_e[..., None] * xr                                    # (m, q, r)
            Y = self.o + (X[:, None, None, :] + rho[..., None] * dirs[:, :, None, :]) @ self.F
            r = np.sqrt(rho ** 2 + hh[:, None, None] ** 2)
            f = kfin(kern, r) * rho * self.w(Y.reshape(-1, len(self.o))).reshape(rho.shape)
            inner = np.sum(f * rho_e[..., None] * wr, axis=-1)
            H = np.broadcast_to(hh[:, None], rho_e.shape)
            inner += self.w(self.o + X @ self.F)[:, None] * kern.G(H, np.sqrt((rho_e * h) ** 2 + H ** 2))
            tri = np.sum(inner * Ww, axis=1) * np.sign(cr) * orient
            total += np.where(live, tri, 0.0)
        return total

    def self_energy(self, kern, level, other=None):
        """The same-piece route; with other (the same parallelogram, another weight) the cross energy, from the
        symmetrised cross-correlation C_ab(z) + C_ba(z) on the half of the z-square used here."""
        n = ORDER[level]
        cells = _cells(self.diam, kern, 1)
        tau, wt, _ = graded_unit(n, cells)
        q, wq = panels(np.linspace(0, 1, cells + 1), n)
        T, Q = (a.ravel() for a in np.meshgrid(tau, q, indexing="ij"))
        Wz = (np.outer(wt, wq) * np.meshgrid(tau, q, indexing="ij")[0]).ravel()
        total = 0.0
        for s1, s2, mult in ((1, 1, 2.0), (1, -1, 2.0)):
            for u, v in ((T, T * Q), (T * Q, T)):
                Z = np.stack([s1 * u, s2 * v], axis=1)
                r = np.linalg.norm(Z @ self.E, axis=1)
                C = self._autocorr(Z, level) if other is None else 0.5 * (self._autocorr(Z, level, other)
                                                                          + other._autocorr(Z, level, self))
                total += mult * float(np.sum(Wz * kern(r) * C))
        return self.det ** 2 * total

    def same(self, other, tol=1e-12):
        return isinstance(other, ParallelogramPiece) and other.E.shape == self.E.shape and \
            np.allclose(other.o, self.o, rtol=0, atol=tol * self.diam) and \
            np.allclose(other.E, self.E, rtol=0, atol=tol * self.diam)

    def _autocorr(self, Z, level, other=None, block=4096):
        """A(z) = int_{a in [0,1]^2, a + z in [0,1]^2} w(x(a)) v(x(a + z)) da, v = w or the other piece's weight."""
        v = self.w if other is None else other.w
        c, cv = _const(self.w), _const(v)
        if c is not None and cv is not None:
            return c * cv * (1 - np.abs(Z[:, 0])) * (1 - np.abs(Z[:, 1]))
        g, gw = gauss((12, 16)[level])
        g, gw = 0.5 * (g + 1), 0.5 * gw
        out = np.empty(len(Z))
        for i in range(0, len(Z), block):
            z = Z[i:i + block]
            lo, ln = np.maximum(0.0, -z), 1 - np.abs(z)
            a1 = lo[:, 0, None, None] + ln[:, 0, None, None] * g[None, :, None]
            a2 = lo[:, 1, None, None] + ln[:, 1, None, None] * g[None, None, :]
            a1, a2 = np.broadcast_arrays(a1, a2)
            P = self.o + a1.reshape(-1, 1) * self.E[0] + a2.reshape(-1, 1) * self.E[1]
            Q = P + (z @ self.E).repeat(len(g) ** 2, axis=0)            # (2-D and 3-D alike)
            f = (self.w(P) * v(Q)).reshape(a1.shape)
            out[i:i + block] = ln[:, 0] * ln[:, 1] * np.einsum("mij,i,j->m", f, gw, gw)
        return out


class RoundPiece(Piece):
    """w dx on the disk (d = 2) or ball (d = 3) |x - c| <= R."""

    def __init__(self, c, R, w):
        self.c, self.R, self.w = np.asarray(c, float), float(R), w
        self.dim = len(self.c)
        self.centre, self.radius = self.c, self.R

    def _directions(self, level, axis=None):
        """Unit directions (q, d) and solid-angle weights."""
        if self.dim == 2:
            nphi = (64, 96)[level]
            phi = 2 * math.pi * np.arange(nphi) / nphi
            return np.stack([np.cos(phi), np.sin(phi)], axis=1), np.full(nphi, 2 * math.pi / nphi)
        nt = (12, 16)[level]
        t, wt = panels(np.linspace(-1, 1, 3), nt)
        nphi = 2 * nt
        phi = 2 * math.pi * np.arange(nphi) / nphi
        st = np.sqrt(1 - t * t)
        U = np.stack([(st[:, None] * np.cos(phi)).ravel(), (st[:, None] * np.sin(phi)).ravel(), np.repeat(t, nphi)], 1)
        W = np.repeat(wt, nphi) * 2 * math.pi / nphi
        if axis is not None:
            U = U @ _frame(axis)
        return U, W

    def nodes(self, kern, level, n=None):
        n = n or ORDER[level]
        r, wr = both_ends(n, _cells(self.R, kern, 1), OUTER)
        r, wr = self.R * r, self.R * wr * (self.R * r) ** (self.dim - 1)
        U, Wu = self._directions(level)
        X = self.c + (r[:, None, None] * U[None]).reshape(-1, self.dim)
        return X, np.outer(wr, Wu).ravel() * self.w(X)

    def formula_nodes(self):
        return self._formula(48 if self.dim == 2 else 24)

    def rule(self, n):
        return self._formula(n)

    def _formula(self, n):
        g, gw = gauss(n)
        r, wr = 0.5 * self.R * (g + 1), 0.5 * self.R * gw * (0.5 * self.R * (g + 1)) ** (self.dim - 1)
        U, Wu = self._directions(1)
        X = self.c + (r[:, None, None] * U[None]).reshape(-1, self.dim)
        return X, np.outer(wr, Wu).ravel() * self.w(X)

    def potential(self, kern, X, level):
        return far_split(self, X, level, self._near, kern, lambda lv: self._formula(FAR[self.dim - 2][lv]))

    def _near(self, kern, X, level):
        return np.array([self._at(kern, x, level) for x in X])

    def _at(self, kern, x, level):
        v = x - self.c
        D = float(np.linalg.norm(v))
        n, d, R = ORDER[level], self.dim, self.R
        if D <= R * (1 + 1e-14):                          # inside (or on the sphere): every direction
            U, Wu = self._directions(level, axis=v / D if D > 0 and d == 3 else None)
            a = U @ v
            rin, rout = np.zeros(len(U)), -a + np.sqrt(np.maximum(a * a + R * R - D * D, 0.0))
        else:                                             # outside: the cone of rays meeting the set
            th = math.asin(R / D)
            psi, wpsi = panels(np.linspace(0, 1, 3), (24, 32)[level])
            if d == 2:
                ang = th * np.sin(math.pi * (psi - 0.5))
                Wu = wpsi * th * math.pi * np.cos(math.pi * (psi - 0.5))
                base = math.atan2(-v[1], -v[0])
                U = np.stack([np.cos(base + ang), np.sin(base + ang)], axis=1)
            else:
                ang = th * np.sin(0.5 * math.pi * psi)
                wang = wpsi * th * 0.5 * math.pi * np.cos(0.5 * math.pi * psi) * np.sin(ang)
                nphi = (24, 32)[level]
                phi = 2 * math.pi * np.arange(nphi) / nphi
                Ul = np.stack([(np.sin(ang)[:, None] * np.cos(phi)).ravel(), (np.sin(ang)[:, None] * np.sin(phi)).ravel(),
                               np.repeat(np.cos(ang), nphi)], 1)
                U, Wu = Ul @ _frame(-v / D), np.repeat(wang, nphi) * 2 * math.pi / nphi
            a = -(U @ v)
            disc = np.sqrt(np.maximum(a * a - (D * D - R * R), 0.0))
            rin, rout = a - disc, a + disc
        c = _const(self.w)
        if c is not None and d == 2:
            return c * float(Wu @ kern.G(rin, rout))      # int_rin^rout k(rho) rho drho, closed form
        xr, wr, h = graded_unit(n, _cells(2 * R, kern, 1), depth=1e-12)
        L = rout - rin
        rho = rin[:, None] + L[:, None] * xr[None]
        Y = x + rho[..., None] * U[:, None, :]
        f = kfin(kern, rho) * rho ** (d - 1) * self.w(Y.reshape(-1, d)).reshape(rho.shape)
        inner = np.sum(f * L[:, None] * wr, axis=1)
        inner += np.where(rin == 0, self.w(x[None])[0] * (kern.G(0.0, L * h) if d == 2 else 0.0), 0.0)
        return float(Wu @ inner)

    def self_energy(self, kern, level):
        R, d, n = self.R, self.dim, ORDER[level]
        cells = _cells(2 * R, kern, 1)
        x, w, _ = graded_unit(n, cells)
        x2, w2 = both_ends(n, 2)                           # grading towards 2R too (lens area ~ (2R - rho)^(3/2))
        rho = np.r_[2 * R * 0.5 * x, R + R * x2]
        wrho = np.r_[2 * R * 0.5 * w, R * w2]
        rho, wrho = rho[rho < 2 * R], wrho[rho < 2 * R]
        c = _const(self.w)
        if c is not None:
            if d == 2:
                lens = 2 * R * R * np.arccos(rho / (2 * R)) - 0.5 * rho * np.sqrt(4 * R * R - rho * rho)
                return c * c * 2 * math.pi * float(wrho @ (kern(rho) * rho * lens))
            lens = math.pi * (4 * R + rho) * (2 * R - rho) ** 2 / 12
            return c * c * 4 * math.pi * float(wrho @ (kern(rho) * rho ** 2 * lens))
        U, Wu = self._directions(level) if d == 2 else _sphere_rule((6, 8)[level])
        half = U[:, -1] > 0 if d == 3 else np.arange(len(U)) < len(U) // 2     # A(-z) = A(z): half the directions
        U, Wu = U[half], 2 * Wu[half]
        total = 0.0
        for r0, wr0 in zip(rho, wrho):
            total += wr0 * kern(r0) * r0 ** (d - 1) * float(Wu @ self._lens(r0, U, level))
        return total

    def _lens(self, rho, U, level):
        """A(rho e) for the directions U (q, d): int over D cap (D - rho e) of w(x) w(x + rho e) dx, in lens coordinates
        x = c - rho e / 2 + xi e + eta f (f perpendicular to e), eta = H sin(theta) (no square-root ends)."""
        R, d = self.R, self.dim
        H = math.sqrt(max(R * R - rho * rho / 4, 0.0))
        g, gw = gauss(LENS[d][level])
        gt, gwt = gauss(LENS_THETA[d][level])
        if d == 2:
            th, wth = 0.5 * math.pi * gt, 0.5 * math.pi * gwt
        else:
            th, wth = 0.25 * math.pi * (gt + 1), 0.25 * math.pi * gwt
        eta, weta = H * np.sin(th), H * np.cos(th) * wth
        Xi = np.sqrt(np.maximum(R * R - eta * eta, 0.0)) - rho / 2
        xi, wxi = Xi[:, None] * g[None], Xi[:, None] * gw[None]               # (eta, xi)
        if d == 2:
            F = np.stack([-U[:, 1], U[:, 0]], axis=1)
            P = (self.c - 0.5 * rho * U)[:, None, None] + xi[None, ..., None] * U[:, None, None] \
                + eta[None, :, None, None] * F[:, None, None]                    # (q, eta, xi, 2)
            W = wxi * weta[:, None]
            vals = (self.w(P.reshape(-1, 2)) * self.w((P + rho * U[:, None, None]).reshape(-1, 2))).reshape(P.shape[:3])
            return np.einsum("qij,ij->q", vals, W)
        npsi = (12, 16)[level]
        psi = 2 * math.pi * np.arange(npsi) / npsi
        F1 = np.cross(U, np.where(np.abs(U[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        F1 /= np.linalg.norm(F1, axis=1)[:, None]
        F2 = np.cross(U, F1)
        ring = np.cos(psi)[None, :, None] * F1[:, None] + np.sin(psi)[None, :, None] * F2[:, None]   # (q, psi, 3)
        P = ((self.c - 0.5 * rho * U)[:, None, None, None] + xi[None, ..., None, None] * U[:, None, None, None]
             + eta[None, :, None, None, None] * ring[:, None, None])                                 # (q, eta, xi, psi, 3)
        vals = (self.w(P.reshape(-1, 3)) * self.w((P + rho * U[:, None, None, None]).reshape(-1, 3))).reshape(P.shape[:4])
        W = wxi * (weta * eta)[:, None] * 2 * math.pi / npsi
        return np.einsum("qijk,ij->q", vals, W)


class TrianglePiece(ParallelogramPiece):
    """w dsigma on the triangle with vertices v0, v1, v2 (a cell of a Simplicial set: of full dimension in 2-D, a
    surface piece in 3-D). Same piece: the identical-triangle case of Sauter-Schwab in relative coordinates: with
    x = v0 + a E, y = v0 + b E (a, b in the reference triangle T), E = |det|^2 int_{T - T} k(|z E|) C(z) dz, where
    T cap (T - z) is a homothetic copy of T (corner max(0, -z), size min(1, 1 - z1 - z2) - max(0, -z1) - max(0, -z2))
    and the hexagon T - T splits into six triangles with apex z = 0 on each of which the size is affine (Duffy
    there, the radial variable graded at 0). Potentials: the fan of the three edges, as for parallelograms."""

    HEX = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, -1.0]])

    def __init__(self, v0, v1, v2, w):
        v0, v1, v2 = (np.asarray(v, float) for v in (v0, v1, v2))
        super().__init__(v0, np.stack([v1 - v0, v2 - v0]), w)
        self.V = np.array([v0, v1, v2])
        self.centre = (v0 + v1 + v2) / 3
        self.radius = float(np.max(np.linalg.norm(self.V - self.centre, axis=1)))
        self.diam = 2 * self.radius

    def _ref(self, n):
        return simplex(2, n)

    def nodes(self, kern, level):
        U, W = self._ref(ORDER[level] + _cells(self.diam, kern, 1))
        X = self.o + U @ self.E
        return X, self.det * W * self.w(X)

    def formula_nodes(self, n=32):
        U, W = self._ref(n)
        X = self.o + U @ self.E
        return X, self.det * W * self.w(X)

    def self_energy(self, kern, level, other=None):
        n = ORDER[level]
        cells = _cells(self.diam, kern, 1)
        tau, wt, _ = graded_unit(n, cells)
        q, wq = panels(np.linspace(0, 1, cells + 1), n)
        T, Q = (g.ravel() for g in np.meshgrid(tau, q, indexing="ij"))
        Wz = np.outer(wt, wq).ravel() * T
        v = self if other is None else other
        total = 0.0
        for k in range(6):
            A, B = self.HEX[k], self.HEX[(k + 1) % 6]
            Z = T[:, None] * (A + Q[:, None] * (B - A))
            jac = abs(A[0] * B[1] - A[1] * B[0])
            r = np.linalg.norm(Z @ self.E, axis=1)
            total += jac * float(np.sum(Wz * kfin(kern, r) * self._tri_corr(Z, level, v)))
        return self.det ** 2 * total

    def _tri_corr(self, Z, level, v, block=4096):
        """C(z) = int_{a in T cap (T - z)} w(x(a)) v(x(a + z)) da (T the reference triangle)."""
        m = np.maximum(0.0, -Z)
        size = np.maximum(np.minimum(1.0, 1.0 - Z[:, 0] - Z[:, 1]) - m[:, 0] - m[:, 1], 0.0)
        c, cv = _const(self.w), _const(v.w)
        if c is not None and cv is not None:
            return c * cv * 0.5 * size ** 2
        U, W = self._ref((12, 16)[level])
        out = np.empty(len(Z))
        for i in range(0, len(Z), block):
            A = m[i:i + block, None, :] + size[i:i + block, None, None] * U[None]          # (b, q, 2)
            P = self.o + A.reshape(-1, 2) @ self.E
            Qp = P + np.repeat(Z[i:i + block] @ self.E, len(W), axis=0)
            f = (self.w(P) * v.w(Qp)).reshape(A.shape[:2])
            out[i:i + block] = size[i:i + block] ** 2 * (f @ W)
        return out

    def same(self, other, tol=1e-12):
        return isinstance(other, TrianglePiece) and np.allclose(other.V, self.V, rtol=0, atol=tol * self.diam)


def axis_aligned(p):
    from .weighted_solids import axis_aligned as f            # the box routes live in weighted_solids.py
    return f(p)


def box_pair_energy(kern, a, b, level):
    from .weighted_solids import box_pair_energy as f
    return f(kern, a, b, level)


def _sphere_rule(nt):
    """Directions on the unit sphere (Gauss in cos(theta) on two panels x trapezoid in phi) and their weights."""
    t, wt = panels(np.linspace(-1, 1, 3), nt)
    nphi = 2 * nt
    phi = 2 * math.pi * np.arange(nphi) / nphi
    st = np.sqrt(1 - t * t)
    U = np.stack([(st[:, None] * np.cos(phi)).ravel(), (st[:, None] * np.sin(phi)).ravel(), np.repeat(t, nphi)], 1)
    return U, np.repeat(wt, nphi) * 2 * math.pi / nphi


def _frame(axis):
    """An orthonormal frame (3, 3) whose last row is the unit vector axis."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    b = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, b)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(a, e1), a])
