"""Eigenvalue brackets of regular Sturm-Liouville operators with interfaces (plan 009 G5(e); EXP-030 DESIGN G5).

L u = w^-1(-(p u')' + q u) on (a, b), p, w > 0, Dirichlet/Neumann/Robin ends, interfaces where the coefficients jump
(operators/unbounded.py). Two computations bracket every eigenvalue:

Upper bounds (Rayleigh-Ritz). The Galerkin eigenvalues mu_n of the form on the conforming space of continuous piecewise
polynomials of degree P on cells that contain the interfaces as nodes (Lobatto shape functions; Dirichlet ends drop the
vertex function) satisfy lambda_n <= mu_n (min-max; THEOREM when the form is integrated exactly: constant and
pg.Tabulated coefficients with their nodes as cell edges, Gauss-Legendre with P + 2 points; for callable coefficients the
quadrature error is not bounded: checked).

Lower bounds (comparison). On each cell replace p by its minimum, w by its maximum and q + sigma w by its minimum
(sigma >= 0 makes q + sigma w >= 0); the Rayleigh quotients compare, so lambda_n + sigma >= nu_n, the n-th eigenvalue of
the piecewise-constant problem (min-max; THEOREM). nu_n is located by its counting function: on a cell with constants the
solution is a sine, a hyperbolic function or linear, so the zeros of the left solution and the final Prufer phase are
explicit (realisations/spectra/bessel.py describes the count N(lambda) = Z + [psi > beta_R]); nu_n is found by bisection on the
count. Exact minima and maxima: constants and tables (piecewise linear: at the cell ends and table nodes); callables are
sampled (checked).

Tails: for every n, lambda_n >= (p_min (max(n - free, 0) pi/L)^2 + min(q + sigma w)) / w_max - sigma (the comparison with
global constants and interlacing for the free ends).
"""
import math

import numpy as np
from numpy.polynomial import legendre as npleg

from ...core.claims import CHECKED, DERIVED, Claim, Claims
from ...core.errors import Unsupported
from ...quadrature import gauss
from ..series import legendre as LG


def _is_table(f):
    return hasattr(f, "nodes") and hasattr(f, "rule")


def _number(f):
    return isinstance(f, (int, float, np.number))


def _exact(f):
    return _number(f) or _is_table(f)


def _shape(P, t):
    """Lobatto shape functions on [-1, 1]: vertex (1 - t)/2, (1 + t)/2 and bubbles (P_j - P_(j-2)) / sqrt(2(2j - 1)),
    with their derivatives: arrays (len(t), P + 1)."""
    t = np.asarray(t, float)
    V = npleg.legvander(t, P)
    D = np.zeros_like(V)
    for j in range(P + 1):
        c = np.zeros(P + 1)
        c[j] = 1.0
        D[:, j] = npleg.legval(t, npleg.legder(c))
    S, dS = np.zeros((len(t), P + 1)), np.zeros((len(t), P + 1))
    S[:, 0], S[:, 1] = 0.5 * (1 - t), 0.5 * (1 + t)
    dS[:, 0], dS[:, 1] = -0.5, 0.5
    for j in range(2, P + 1):
        k = 1.0 / math.sqrt(2 * (2 * j - 1))
        S[:, j], dS[:, j] = k * (V[:, j] - V[:, j - 2]), k * (D[:, j] - D[:, j - 2])
    return S, dS


class GalerkinSpectrum:
    """Eigenvalue brackets, eigenfunctions and tail bounds of a regular Sturm-Liouville operator (module docstring)."""

    kind = "galerkin"

    def __init__(self, op, cells=96, degree=8):
        for c in (op.left, op.right):
            if c[0] == "regular":
                raise Unsupported("singular ends are realised for the per-degree radial Laplacian (bessel.py) only")
        self.op, self.P = op, int(degree)
        a, b = op.a, op.b
        knots = [a, b] + list(op.interfaces)
        for f in (op.p_src, op.q_src, op.w_src):
            if _is_table(f):
                knots += [x for x in f.nodes if a < x < b]
        knots = np.unique(knots)
        per = max(1, int(math.ceil(cells / (len(knots) - 1))))
        self.x = np.unique(np.concatenate([np.linspace(knots[i], knots[i + 1], per + 1)
                                           for i in range(len(knots) - 1)]))
        self.exact = all(_exact(f) for f in (op.p_src, op.q_src, op.w_src))
        self._cell_bounds()
        self._mu = None

    # coefficient bounds per cell ------------------------------------------------------------------------
    def _range(self, f, src, x0, x1):
        if _number(src):
            return float(src), float(src)
        pts = np.linspace(x0, x1, 3 if _is_table(src) else 65)
        if _is_table(src):
            pts = np.concatenate([pts, [t for t in src.nodes if x0 < t < x1]])
        v = np.asarray(f(pts), float)
        return float(v.min()), float(v.max())

    def _cell_bounds(self):
        op, x = self.op, self.x
        p = [self._range(op.p, op.p_src, x[i], x[i + 1]) for i in range(len(x) - 1)]
        q = [self._range(op.q, op.q_src, x[i], x[i + 1]) for i in range(len(x) - 1)]
        w = [self._range(op.w, op.w_src, x[i], x[i + 1]) for i in range(len(x) - 1)]
        self.p_lo = np.array([r[0] for r in p])
        self.w_lo, self.w_hi = np.array([r[0] for r in w]), np.array([r[1] for r in w])
        q_lo = np.array([r[0] for r in q])
        if self.p_lo.min() <= 0 or self.w_lo.min() <= 0:
            raise Unsupported("p and w must be positive on [a, b] (a regular problem)")
        self.sigma = max(0.0, float(np.max(-q_lo / np.where(q_lo < 0, self.w_lo, self.w_hi))))
        # min over the cell of q + sigma w >= q_lo + sigma w_lo
        self.r_lo = np.maximum(q_lo + self.sigma * self.w_lo, 0.0)

    # upper bounds: Galerkin -----------------------------------------------------------------------------
    def _assemble(self):
        op, P, x = self.op, self.P, self.x
        n_cells = len(x) - 1
        nq = P + 2 if self.exact else 2 * P + 8
        t, wq = gauss(nq)
        S, dS = _shape(P, t)
        ndof = n_cells * (P - 1) + n_cells + 1
        K, M = np.zeros((ndof, ndof)), np.zeros((ndof, ndof))
        self._dofs = []
        for i in range(n_cells):
            x0, x1 = x[i], x[i + 1]
            h = x1 - x0
            r = 0.5 * (x0 + x1) + 0.5 * h * t
            pv, qv, wv = np.asarray(op.p(r), float), np.asarray(op.q(r), float), np.asarray(op.w(r), float)
            dofs = np.concatenate([[i, i + 1], n_cells + 1 + i * (P - 1) + np.arange(P - 1)])
            self._dofs.append(dofs)
            Ke = (dS.T * (wq * pv * (2 / h))) @ dS + (S.T * (wq * qv * (h / 2))) @ S
            Me = (S.T * (wq * wv * (h / 2))) @ S
            K[np.ix_(dofs, dofs)] += Ke
            M[np.ix_(dofs, dofs)] += Me
        for node, c in ((0, op.left), (n_cells, op.right)):
            if c[0] == "robin":
                K[node, node] += c[1]
        keep = np.ones(ndof, bool)
        if op.left[0] == "dirichlet":
            keep[0] = False
        if op.right[0] == "dirichlet":
            keep[n_cells] = False
        self._keep = keep
        return K[np.ix_(keep, keep)], M[np.ix_(keep, keep)]

    def _solve(self):
        if self._mu is None:
            from scipy.linalg import eigh
            K, M = self._assemble()
            self._KM = (K, M)
            mu, V = eigh(0.5 * (K + K.T), 0.5 * (M + M.T))
            self._mu, self._V = mu, V
        return self._mu, self._V

    # lower bounds: piecewise-constant comparison -------------------------------------------------------
    def count_lower(self, lam):
        """The number of eigenvalues < lam of the comparison problem (shifted by sigma), vectorised over lam.

        The Prufer angle theta (tan theta = u / (p u')) is carried as K pi + t, t in [0, pi). On a cell with
        mu = (lam w - r)/p > 0 the scaled angle phi (tan phi = o p tan theta, o = sqrt(mu)) has the same multiples of pi
        and advances by exactly o h; on a cell with mu <= 0 the solution has at most one zero, found from the transfer
        of (u, p u') = (sin t, cos t)."""
        lam = np.atleast_1d(np.asarray(lam, float)) + self.sigma
        op, x = self.op, self.x
        kind, beta = op.left
        K = np.zeros(lam.shape)
        t = np.zeros(lam.shape) if kind == "dirichlet" else np.full(lam.shape, math.atan2(1.0, beta))
        for i in range(len(x) - 1):
            h, p, w, r = x[i + 1] - x[i], self.p_lo[i], self.w_hi[i], self.r_lo[i]
            mu = (lam * w - r) / p
            pos = mu > 0
            o = np.sqrt(np.abs(mu))
            if pos.any():
                op_ = o[pos] * p
                phi = np.arctan2(op_ * np.sin(t[pos]), np.cos(t[pos])) + o[pos] * h      # in [0, pi) + o h
                k = np.floor(phi / math.pi)
                f = phi - k * math.pi
                K[pos] += k
                t[pos] = np.mod(np.arctan2(np.sin(f) / op_, np.cos(f)), math.pi)
            neg = ~pos
            if neg.any():
                on = o[neg]
                u, pu = np.sin(t[neg]), np.cos(t[neg])
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = np.where(on > 0, np.cosh(on * h), 1.0)
                    sh = np.where(on > 0, np.sinh(on * h) / np.where(on > 0, on * p, 1.0), h / p)
                    u1 = u * c + pu * sh
                    pu1 = u * (on * p) * np.where(on > 0, np.sinh(on * h), 0.0) + pu * c
                cross = u1 < 0
                K[neg] += cross
                t[neg] = np.where(cross, np.arctan2(-u1, -pu1), np.arctan2(u1, pu1))
                t[neg] = np.mod(t[neg], math.pi)
        kind, beta = op.right
        target = math.pi if kind == "dirichlet" else 0.5 * math.pi + math.atan(beta)
        return (K + (t > target)).astype(int)

    def _comparison(self, n):
        """nu_k - sigma for k = 1..n by bisection on the count (all at once)."""
        lo = np.full(n, -self.sigma)
        hi = np.full(n, max(float(self._solve()[0][min(n, len(self._mu)) - 1]), 1.0) * 2 + 1.0)
        while (self.count_lower(hi) < np.arange(1, n + 1)).any():
            hi *= 2
        idx = np.arange(1, n + 1)
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            c = self.count_lower(mid)
            below = c < idx                       # fewer than k eigenvalues below mid: nu_k >= mid
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return lo

    # interface of a spectrum -----------------------------------------------------------------------------
    def eigenvalues(self, n):
        """Upper bounds of the first n eigenvalues: the Ritz values of the span of the computed Galerkin vectors (any
        n-dimensional subspace gives upper bounds, min-max) plus a bound on the rounding of the projected matrices
        (Weyl: entrywise errors <= gamma |V|^T |K| |V|, gamma = (dofs + 2) eps)."""
        mu, V = self._solve()
        if n > len(mu) // 2:
            raise Unsupported(f"{n} eigenvalues from a Galerkin space of dimension {len(mu)}: refine (cells, degree)")
        if getattr(self, "_up", None) is None or len(self._up) < n:
            from scipy.linalg import eigh
            K, M = self._KM
            W = V[:, :n]
            A, B = W.T @ K @ W, W.T @ M @ W
            g = (K.shape[0] + 2) * np.finfo(float).eps
            EA = g * np.linalg.norm(np.abs(W).T @ np.abs(K) @ np.abs(W))
            EB = g * np.linalg.norm(np.abs(W).T @ np.abs(M) @ np.abs(W))
            ritz = eigh(0.5 * (A + A.T), 0.5 * (B + B.T), eigvals_only=True)
            bmin = float(np.linalg.eigvalsh(0.5 * (B + B.T))[0]) - EB
            self._up = ritz + (EA + np.abs(ritz) * EB) / bmin
        return self._up[:n].copy()

    def brackets(self, n):
        up = self.eigenvalues(n)
        low = self._comparison(n)
        low = np.maximum(low - 1e-12 * np.maximum(np.abs(low), 1.0), self.lower_bound(np.arange(1, n + 1)))  # rounding
        return low, up

    def lower_bound(self, idx):
        """Lower bounds of lambda_n for every n (module docstring)."""
        n = np.asarray(idx, float)
        L = self.op.b - self.op.a
        m = np.maximum(n - self.op.free_ends(), 0.0)
        return (self.p_lo.min() * (m * math.pi / L) ** 2 + self.r_lo.min()) / self.w_hi.max() - self.sigma

    def tail_constants(self):
        """(A, theta, sigma) with lambda_n >= A (n - theta)^2 - sigma for every n (for tail sums)."""
        L = self.op.b - self.op.a
        return self.p_lo.min() / self.w_hi.max() * (math.pi / L) ** 2, float(self.op.free_ends()), self.sigma

    def pl(self, n, tol=None):
        """The first n Galerkin eigenfunctions, w-orthonormal, as an n-column piecewise Legendre function."""
        mu, V = self._solve()
        full = np.zeros((len(self._keep), n))
        full[self._keep] = V[:, :n]
        t, _ = gauss(self.P + 1)
        S, _ = _shape(self.P, t)
        A = LG.analysis(self.P + 1)
        coefs = [A @ (S @ full[d]) for d in self._dofs]
        return LG.PiecewiseLegendre(self.x, coefs, np.zeros(n)), 0

    def eigenfunctions(self, n):
        P, _ = self.pl(n)
        return lambda r: LG.evaluate(P, np.asarray(r, float))

    @property
    def claims(self):
        st = DERIVED if self.exact else CHECKED
        return Claims([Claim("eigenvalues bracketed: Galerkin (Rayleigh-Ritz) upper bounds, piecewise-constant "
                             "comparison lower bounds by the Prufer count", st,
                             "min-max (Courant-Fischer); exact integration and exact cell extrema for constant and "
                             "tabulated coefficients" if self.exact else "callable coefficients: quadrature and cell "
                             "extrema sampled, not bounded"),
                       Claim("eigenfunctions are the Galerkin ones (continuous piecewise polynomials)", CHECKED,
                             "their distance to the exact eigenfunctions is controlled by the bracket width and the "
                             "spectral gap, not computed")])
