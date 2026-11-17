"""Closed-form spectra of the per-degree radial Laplacian (plan 009 G5; EXP-030 DESIGN G5(c), NOTES.md section 5).

L_s u = -r^-2 (r^2 u')' + s(s + 1) r^-2 u on L2((a, b), r^2 dr), a = 0 (a ball) or a > 0 (a shell). For lambda = k^2 > 0
the solutions are the cylinder functions C(kr) = c1 j_s(kr) + c2 y_s(kr) (spherical Bessel functions); on a ball the
regular one is j_s(kr). Conditions in the form's convention (operators/unbounded.py): Dirichlet u = 0; Robin
p d_n u + beta u = 0 with p = r^2 (Neumann: beta = 0). phi_k is the solution satisfying the left condition, and the
eigenvalues are the zeros of the right condition applied to it, char(k).

Counting (Prufer). With phi = rho sin(theta), p phi' = rho cos(theta), theta(b, lambda) is continuous and increasing in
lambda, crosses multiples of pi only upwards (at zeros of phi), and the n-th eigenvalue (n = 0, 1, ...) is where
theta(b) = beta_R + n pi, beta_R = pi (Dirichlet) or pi/2 + arctan(beta) (Robin); hence the number of eigenvalues
below lambda is N(lambda) = Z + [psi > beta_R], Z the zeros of phi in (a, b) and psi = theta(b) - Z pi in (0, pi]
(Zettl 2005, ch. 4 and 10; at a limit-point centre with the principal solution j_s). The zeros of a cylinder function of
order s + 1/2 >= 1/2 are at least pi apart (Sturm comparison for x^(1/2) C(x), whose equation is
u'' + (1 - (nu^2 - 1/4)/x^2) u = 0), so Z is counted exactly by sign changes on a grid of step < pi/k. Each eigenvalue is
isolated by counting (exactly one in its interval), found by Brent's method on char, and enclosed by a sign change of
char: the bracket is derived (in floating point).

Normalisation by Lommel's integral, int C(kr)^2 r^2 dr = (r^3/2)[C_s(kr)^2 - C_(s-1)(kr) C_(s+1)(kr)] (no quadrature);
C_(-1)(x) = C_0(x)/x - C_1(x). Lower bounds for every index (tails): NOTES.md section 5(d), interlacing_lower below
(from core/sturm_liouville.py, plan 068 P3).
"""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

from ...core.claims import CHECKED, DERIVED, Claim, Claims

EPS = np.finfo(float).eps


def interlacing_lower(n, s, a, b, free, special=False):
    """Lower bounds for the eigenvalues lambda_n (n = 1, 2, ..., arrays) of L_s on (a, b) (NOTES.md section 5(d)):
    s(s + 1)/b^2 + (max(n - free, 0) pi / (b - a))^2, or with special (Neumann/Robin inside, Dirichlet outside, a > 0)
    s(s + 1)/b^2 + ((n - 1/2) pi / (b - a))^2."""
    n = np.asarray(n, float)
    L = b - a
    c = s * (s + 1) / b ** 2
    base = c + (np.maximum(n - free, 0.0) * math.pi / L) ** 2
    if special:
        base = np.maximum(base, c + ((n - 0.5) * math.pi / L) ** 2)
    return base


def _C(s, x, c1, c2, derivative=False):
    out = c1 * spherical_jn(s, x, derivative)
    return out + c2 * spherical_yn(s, x, derivative) if np.any(c2 != 0) else out


def _Cm1(s, x, c1, c2):
    """C_(s-1)(x), with the recurrence at s = 0."""
    if s >= 1:
        return _C(s - 1, x, c1, c2)
    return _C(0, x, c1, c2) / x - _C(1, x, c1, c2)


class RadialBessel:
    """The spectrum of a per-degree radial Laplacian with constant coefficients (module docstring)."""

    kind = "bessel"

    def __init__(self, op):
        self.op, self.s = op, op.radial[0]
        self.a, self.b = op.a, op.b
        self.ball = self.a == 0
        self.zero_mode = (self.s == 0 and op.right[0] == "neumann" and (self.ball or op.left[0] == "neumann"))
        self.special = (not self.ball and op.left[0] in ("neumann", "robin") and op.right[0] == "dirichlet")
        self._k = np.zeros(0)
        self._dk = np.zeros(0)

    # the left solution and the characteristic function ------------------------------------------------
    def _left(self, k):
        if self.ball:
            return np.ones_like(k), np.zeros_like(k)
        s, a = self.s, self.a
        kind, beta = self.op.left
        x = k * a
        if kind == "dirichlet":
            fj, fy = spherical_jn(s, x), spherical_yn(s, x)
        else:                                       # -p u'(a) + beta u(a) = 0, p = a^2 (outward normal -d/dr)
            fj = -a * a * k * spherical_jn(s, x, True) + beta * spherical_jn(s, x)
            fy = -a * a * k * spherical_yn(s, x, True) + beta * spherical_yn(s, x)
        h = np.hypot(fj, fy)
        return fy / h, -fj / h

    def _phi(self, k, r):
        """phi and p phi' at the radii r for one k (left solution)."""
        c1, c2 = self._left(np.asarray(k, float))
        x = k * r
        return _C(self.s, x, c1, c2), r * r * k * _C(self.s, x, c1, c2, True)

    def char(self, k):
        u, pu = self._phi(k, np.asarray(self.b))
        kind, beta = self.op.right
        return float(u) if kind == "dirichlet" else float(pu + beta * u)

    def count(self, lam):
        """The number of eigenvalues < lam (module docstring)."""
        if lam <= 0:
            return 0
        k = math.sqrt(lam)
        a0 = self.a if not self.ball else min(0.5 / k, 0.5 * self.b)        # j_s(kr) > 0 for kr < 1
        n = int(math.ceil((self.b - a0) * k / (0.45 * math.pi))) + 2
        r = np.linspace(a0, self.b, n + 1)
        u, _ = self._phi(k, r[:-1])
        if not self.ball and self.op.left[0] == "dirichlet":
            u = u[1:]                  # phi(a) = 0 (rounded to either sign): the next zero is >= a + pi/k, past r[1]
        inner = u[1:]                                                     # zeros in (a, b): sign changes on the grid
        sg = np.sign(u)
        exact = int(np.count_nonzero(inner == 0.0))
        nz = sg[sg != 0]
        Z = int(np.count_nonzero(nz[1:] != nz[:-1])) + exact
        ub, pub = self._phi(k, np.asarray(self.b))
        ub, pub = float(ub), float(pub)
        if ub == 0.0:
            psi = math.pi
        else:
            if (np.sign(nz[-1]) if len(nz) else 1.0) != np.sign(ub):      # a zero between the last node and b
                Z += 1
            psi = 0.5 * math.pi - math.atan(pub / ub)
        kind, beta = self.op.right
        target = math.pi if kind == "dirichlet" else 0.5 * math.pi + math.atan(beta)
        return Z + (1 if psi > target else 0)

    # eigenvalues --------------------------------------------------------------------------------------
    def _find(self, n):
        """The first n values of k (lambda = k^2), each isolated by counting and enclosed by a sign change."""
        L = self.b - self.a
        ks, dks = [], []
        start = 0
        if self.zero_mode:
            ks.append(0.0)
            dks.append(0.0)
            start = 1
        if n <= start:
            return np.array(ks[:n]), np.array(dks[:n])
        hi = (n + 3) * math.pi / L + math.sqrt(self.s * (self.s + 1)) / self.b
        while self.count(hi * hi) < n:
            hi *= 1.5
        grid = np.linspace(0.0, hi, 4 * n + 8)[1:]
        counts = [self.count(g * g) for g in grid]
        lo_k, lo_c = 1e-12 * hi, self.count((1e-12 * hi) ** 2)
        todo = []
        for g, c in zip(grid, counts):
            if c > lo_c:
                todo.append((lo_k, g, lo_c, c))
            lo_k, lo_c = g, c
        found = {}
        while todo:
            lo, hi_, clo, chi = todo.pop()
            if clo >= n:
                continue
            if chi - clo == 1:
                found[clo] = self._root(lo, hi_)
                continue
            mid = 0.5 * (lo + hi_)
            cm = self.count(mid * mid)
            if hi_ - lo < 1e-14 * hi_:
                raise ArithmeticError("eigenvalues closer than the resolution: not isolated")
            todo += [(lo, mid, clo, cm), (mid, hi_, cm, chi)]
        for i in range(start, n):
            ks.append(found[i][0])
            dks.append(found[i][1])
        return np.array(ks), np.array(dks)

    def _root(self, lo, hi):
        f_lo, f_hi = self.char(lo), self.char(hi)
        if f_lo * f_hi > 0:                          # no sign change on the isolating interval: bisect by counting
            c0 = self.count(lo * lo)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if self.count(mid * mid) > c0:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi), 0.5 * (hi - lo)
        k = brentq(self.char, lo, hi, xtol=1e-300, rtol=4 * EPS, maxiter=400)
        d = 8 * EPS * k
        for _ in range(40):
            if self.char(k - d) * self.char(k + d) <= 0:
                return k, d
            d *= 4
        return k, hi - lo

    def wavenumbers(self, n):
        """The first n values of k = sqrt(lambda), ascending (the zeros of the cylinder-function determinant)."""
        self.eigenvalues(n)
        return self._k[:n].copy()

    def eigenvalues(self, n):
        """The first n eigenvalues, ascending (lambda_1 <= lambda_2 <= ...)."""
        if len(self._k) < n:
            self._k, self._dk = self._find(max(n, 2 * len(self._k)))
        return self._k[:n] ** 2

    def brackets(self, n):
        """(lower, upper) enclosures of the first n eigenvalues (module docstring)."""
        self.eigenvalues(n)
        k, d = self._k[:n], self._dk[:n]
        return np.maximum(k - d, 0.0) ** 2, (k + d) ** 2

    def lower_bound(self, idx):
        """Lower bounds of lambda_n for indices n = 1, 2, ... (every n, so tails are bounded)."""
        return interlacing_lower(idx, self.s, self.a, self.b, self.op.free_ends(), self.special)

    # eigenfunctions -----------------------------------------------------------------------------------
    def eigenfunctions(self, n):
        """phi(r) -> (len(r), n): the first n eigenfunctions, orthonormal in L2((a, b), r^2 dr), positive next to a."""
        self.eigenvalues(n)
        k = self._k[:n]
        s, a, b = self.s, self.a, self.b
        c1, c2 = (np.ones(n), np.zeros(n)) if self.ball else self._left(np.where(k > 0, k, 1.0))
        norms = np.empty(n)
        sign = np.ones(n)
        for i in range(n):
            if k[i] == 0.0:
                norms[i] = math.sqrt((b ** 3 - a ** 3) / 3.0)
                continue
            F = lambda r: 0.5 * r ** 3 * (_C(s, k[i] * r, c1[i], c2[i]) ** 2               # noqa: E731
                                          - _Cm1(s, k[i] * r, c1[i], c2[i]) * _C(s + 1, k[i] * r, c1[i], c2[i]))
            norms[i] = math.sqrt(F(b) - (F(a) if a > 0 else 0.0))
            r0 = a + 1e-3 * min(b - a, math.pi / k[i]) if a > 0 else min(0.5 / k[i], 0.5 * b)
            sign[i] = 1.0 if _C(s, k[i] * r0, c1[i], c2[i]) >= 0 else -1.0

        ks = np.where(k > 0, k, 1.0)

        def phi(r):
            r = np.asarray(r, float)[..., None]
            out = np.where(k > 0, _C(s, ks * r, c1, c2), 1.0)
            return out * (sign / norms)
        return phi

    def pl(self, n, tol=1e-14):
        """The first n eigenfunctions as one n-column piecewise Legendre function on [a, b] (adaptive fit: its error
        estimate is a realisation error)."""
        from ..series import legendre as LG
        phi = self.eigenfunctions(n)
        P, u = LG.fit(lambda r: phi(r), self.a, self.b, N=n, tol=tol, max_pieces=4096)
        return P, u

    @property
    def claims(self):
        return Claims([Claim("eigenvalues: zeros of the cylinder-function determinant, each isolated by the Prufer count "
                             "and enclosed by a sign change", DERIVED, "realisations/spectra/bessel.py (Zettl 2005; Sturm "
                             "comparison for the zero spacing); floating-point evaluation of j_s, y_s (scipy)"),
                       Claim("eigenfunctions in closed form, normalised by Lommel's integral", DERIVED,
                             "int C(kr)^2 r^2 dr = (r^3/2)[C_s^2 - C_(s-1) C_(s+1)]"),
                       Claim("radial factors realised as piecewise Legendre fits", CHECKED,
                             "adaptive fit with its error estimate (a realisation error)")])
