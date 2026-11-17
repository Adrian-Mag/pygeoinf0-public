"""Spectra of the Laplacian -Delta of a domain (plan 068 P3; review B5): closed forms and the unions that separation of
variables gives. operators/unbounded.Laplacian(D, bc).spectrum() is chosen in realisations/registry.laplacian_spectrum.

    IntervalModes(a, b, left, right)  -u'' on (a, b) with separated conditions (from the function bank, plan 021 P1)
    ProductSpectrum(factors)          a box (or a flat torus): -Delta = sum_i -d_i^2 on a product, so its eigenvalues are
                                      the sums lambda_(i_1) + ... + lambda_(i_d) of the factors' (separation of variables)
    DegreeSpectrum(lap)               a ball or a shell: -Delta = (+)_s L_s (x) I_(H_s) (Schur), so its eigenvalues are
                                      those of the radial operators L_s with multiplicity 2s + 1
    CircleSpectrum(R), SphereSpectrum(R)  (m/R)^2 (m >= 0, multiplicity 2 for m >= 1) and l(l + 1)/R^2 (2l + 1)

Every spectrum has eigenvalues(n) (the first n, ascending, with multiplicity), brackets(n) (lower and upper enclosures)
and claims. Order statistics keep enclosures: if lambda_t in [l_t, u_t] for every t, the k-th smallest lambda lies
between the k-th smallest l and the k-th smallest u; the unions are truncated only where a stated lower bound shows that
nothing omitted is among the first n.

Interval modes. With x' = x - a, L = b - a and the phase phi_e at an end e (Dirichlet pi/2, Neumann 0, Robin
p d_n u + beta u = 0 with p = 1: arctan(beta / k)), u(x) = N cos(k x' - phi_a), and the g-th eigenvalue (g = 1, 2, ...)
is k^2 with k L = phi_a + phi_b + (g - 1) pi: closed when no end is Robin, else the unique root in (0, g pi / L] (the
left side increases and the right side decreases in k), found by Brent's method to machine precision and enclosed by a
sign change. N^-2 = L/2 + (sin 2 phi_a + sin 2 phi_b)/(4k) and int u = N [(-1)^(g-1) sin phi_b + sin phi_a] / k
(k = 0: the constant 1/sqrt(L), Neumann at both ends, g = 1). Courant & Hilbert 1953, V.5.
"""
import math

import numpy as np
from scipy.optimize import brentq

from ...core.claims import DERIVED, THEOREM, Claim, Claims

EPS = np.finfo(float).eps
HALF_PI = 0.5 * math.pi
# exact (sin, cos) of a phase in {0, pi/2}: math.cos(pi/2) is 6e-17, not 0
EXACT_SC = {0.0: (0.0, 1.0), HALF_PI: (1.0, 0.0)}
COURANT = "Courant & Hilbert 1953, Methods of Mathematical Physics I, V.5 and VI.2 (eigenfunctions of the Laplacian)"


# --------------------------------------------------------------------------------------------- interval modes
class Mode:
    """One interval eigenfunction N (ca cos(k x') + sa sin(k x')), x' = x - a, with ca + i sa = e^{i phi_a}; the phase
    at b is given by both sin_phib and cos_phib (each computed without cancellation by the caller: sin 2 phi_b =
    2 sin cos loses all precision if cos is recovered as sqrt(1 - sin^2) near phi_b = pi/2)."""

    def __init__(self, a, L, k, ca, sa, g, sin_phib, cos_phib, robin_residual=None):
        self.a, self.L, self.k, self.ca, self.sa, self.g = a, L, k, ca, sa, g
        self.sin_phib, self.cos_phib, self.residual = sin_phib, cos_phib, robin_residual
        if k == 0.0:
            self.N = 1.0 / math.sqrt(L)
            self.integral = self.N * L
        else:
            sin2a, sin2b = 2 * sa * ca, 2 * sin_phib * cos_phib
            self.N = 1.0 / math.sqrt(0.5 * L + (sin2a + sin2b) / (4 * k))
            self.integral = self.N * ((-1) ** (g - 1) * sin_phib + sa) / k

    def value(self, x):
        t = self.k * (np.asarray(x, float) - self.a)
        return self.N * (self.ca * np.cos(t) + self.sa * np.sin(t))

    def deriv(self, x):
        t = self.k * (np.asarray(x, float) - self.a)
        return self.N * self.k * (self.sa * np.cos(t) - self.ca * np.sin(t))


def _fixed_phase(cond):
    """The phase of a Dirichlet (pi/2) or Neumann (0) end; None for a Robin end (its phase depends on k)."""
    kind, beta = cond
    if kind == "robin" and beta > 0:
        return None
    return HALF_PI if kind == "dirichlet" else 0.0


def _phase(cond, k):
    p = _fixed_phase(cond)
    return p if p is not None else math.atan2(cond[1], k)


def _sc(cond, k):
    """(sin, cos) of the phase at an end, exactly for a fixed phase and without cancellation for a Robin one."""
    p = _fixed_phase(cond)
    if p is not None:
        return EXACT_SC[p]
    h = math.hypot(k, cond[1])
    return cond[1] / h, k / h


def mode(a, b, left, right, g):
    """The g-th (g >= 1) eigenfunction of -u'' on (a, b) with the conditions left, right ((kind, beta): "dirichlet",
    "neumann", "robin" with beta > 0; module docstring), as a Mode; for a Robin end its residual
    |p d_n u + beta u| / (N (k + beta)) is on record."""
    L = b - a
    pa, pb = _fixed_phase(left), _fixed_phase(right)
    if pa is not None and pb is not None:
        k = (pa + pb + (g - 1) * math.pi) / L
        (sa, ca), (sb, cb) = EXACT_SC[pa], EXACT_SC[pb]
        return Mode(a, L, k, ca, sa, g, sb, cb)

    def h(k):
        return k * L - _phase(left, k) - _phase(right, k) - (g - 1) * math.pi

    k = brentq(h, 0.0, g * math.pi / L, xtol=1e-300, rtol=8.9e-16)
    (sa, ca), (sb, cb) = _sc(left, k), _sc(right, k)
    m = Mode(a, L, k, ca, sa, g, sb, cb)
    res = 0.0
    for (kind, beta), x, sign in ((left, a, -1.0), (right, b, 1.0)):
        if kind == "robin" and beta > 0:
            res = max(res, abs(sign * float(m.deriv(x)) + beta * float(m.value(x))) / (m.N * (k + beta)))
    m.residual = res
    return m


class IntervalModes:
    """The spectrum of -u'' on (a, b) with separated conditions in closed form (module docstring)."""

    kind = "closed form"

    def __init__(self, a, b, left, right):
        self.a, self.b, self.left, self.right = float(a), float(b), left, right
        self.robin = _fixed_phase(left) is None or _fixed_phase(right) is None
        self._modes = {}

    def mode(self, g):
        if g not in self._modes:
            self._modes[g] = mode(self.a, self.b, self.left, self.right, g)
        return self._modes[g]

    def wavenumbers(self, n):
        return np.array([self.mode(g).k for g in range(1, n + 1)])

    def eigenvalues(self, n):
        return self.wavenumbers(n) ** 2

    def brackets(self, n):
        """Enclosures of the first n eigenvalues: the closed forms to rounding; a Robin root by a sign change of
        k L - phi_a(k) - phi_b(k) - (g - 1) pi (increasing in k) at its ends."""
        k = self.wavenumbers(n)
        lo, hi = np.empty(n), np.empty(n)
        L = self.b - self.a
        for i, kk in enumerate(k):
            g = i + 1
            d = 4 * EPS * max(kk, 1.0 / L)
            if self.robin:
                h = lambda x: x * L - _phase(self.left, x) - _phase(self.right, x) - (g - 1) * math.pi  # noqa: E731
                for _ in range(60):
                    if h(max(kk - d, 0.0)) <= 0.0 <= h(kk + d):
                        break
                    d *= 4
            lo[i], hi[i] = max(kk - d, 0.0) ** 2, (kk + d) ** 2
        return lo, hi

    def lower_bound(self, idx):
        """lambda_g >= ((g - 1) pi / L)^2 for every g (the phases are >= 0)."""
        g = np.asarray(idx, float)
        return (np.maximum(g - 1, 0.0) * math.pi / (self.b - self.a)) ** 2

    def eigenfunctions(self, n):
        """phi(x) -> (len(x), n): the first n eigenfunctions, orthonormal in L2(a, b)."""
        modes = [self.mode(g) for g in range(1, n + 1)]

        def phi(x):
            x = np.asarray(x, float)
            return np.stack([m.value(x) for m in modes], axis=-1)
        return phi

    @property
    def claims(self):
        how = "Brent's method on k L = phi_a + phi_b + (g - 1) pi, enclosed by a sign change" if self.robin else \
            "k L = phi_a + phi_b + (g - 1) pi with fixed phases"
        return Claims([Claim("eigenvalues and eigenfunctions of -u'' in closed form", DERIVED, f"{how} ({COURANT})")])


# --------------------------------------------------------------------------------------------- unions
def _smallest(values, n):
    v = np.sort(np.asarray(values, float).ravel())
    return v[:n]


class ProductSpectrum:
    """The spectrum of a sum of operators acting on separate factors (a box, a flat torus): eigenvalues are sums, one per
    factor (separation of variables; the products of the factors' eigenfunctions are complete in L2 of the product)."""

    kind = "product"

    def __init__(self, factors, what="a product of intervals"):
        self.factors, self.what = list(factors), what

    def _sums(self, n, pick):
        out = None
        for f in self.factors:
            v = pick(f, n)
            out = v if out is None else _smallest(np.add.outer(out, v), n)
        return _smallest(out, n)

    def eigenvalues(self, n):
        return self._sums(n, lambda f, m: f.eigenvalues(m))

    def brackets(self, n):
        return self._sums(n, lambda f, m: f.brackets(m)[0]), self._sums(n, lambda f, m: f.brackets(m)[1])

    @property
    def claims(self):
        return Claims([Claim(f"-Delta on {self.what} = sum of the one-dimensional operators: its eigenvalues are the "
                             "sums of theirs", THEOREM, "separation of variables: products of complete orthonormal "
                             "systems are complete in L2 of the product (Fubini)"),
                       Claim("enclosures of the first n eigenvalues from the factors' (order statistics of the sums)",
                             DERIVED, "k-th smallest of the lower and of the upper sums")])


class DegreeSpectrum:
    """The spectrum of the Laplacian of a ball or a shell with rotation-invariant conditions: the union over degrees
    s >= 0 of the spectra of L_s (lap.radial(s)), each with multiplicity 2s + 1. Degrees are added until the lower
    bound of the next degree's first eigenvalue, s(s + 1)/b^2 + ... (interlacing, realisations/spectra/bessel.py), exceeds the
    n-th smallest upper end found: nothing omitted is among the first n."""

    kind = "degrees"

    def __init__(self, lap):
        self.lap = lap

    def _union(self, n):
        lam, lo, hi = [], [], []
        s = 0
        while True:
            sp = self.lap.radial(s).spectrum()
            if len(hi) >= n and float(sp.lower_bound([1])[0]) > np.sort(hi)[n - 1]:
                break
            m = 2 * s + 1
            l_, (a, b) = sp.eigenvalues(n), sp.brackets(n)
            lam += list(np.repeat(l_, m))
            lo += list(np.repeat(a, m))
            hi += list(np.repeat(b, m))
            s += 1
        return lam, lo, hi

    def eigenvalues(self, n):
        return _smallest(self._union(n)[0], n)

    def brackets(self, n):
        _, lo, hi = self._union(n)
        return _smallest(lo, n), _smallest(hi, n)

    @property
    def claims(self):
        return Claims([Claim("-Delta = (+)_s L_s (x) I_(H_s): eigenvalues of L_s with multiplicity 2s + 1", THEOREM,
                             "it commutes with rotations (Schur's lemma; EXP-030 Theorem G2.1)"),
                       Claim("degrees beyond the last one used have no eigenvalue among the first n", DERIVED,
                             "lambda_1(L_s) >= s(s + 1)/b^2 + ... (interlacing, NOTES.md section 5(d))")])


# --------------------------------------------------------------------------------------------- closed manifolds
class _Closed:
    kind = "closed form"

    def _list(self, n):
        raise NotImplementedError

    def eigenvalues(self, n):
        return np.asarray(self._list(n)[:n], float)

    def brackets(self, n):
        lam = self.eigenvalues(n)
        return lam * (1 - 4 * EPS), lam * (1 + 4 * EPS)


class CircleSpectrum(_Closed):
    """-d^2/ds^2 on a circle of length 2 pi R (arc length s): (m/R)^2, m = 0 once and m >= 1 twice (cos and sin)."""

    def __init__(self, R):
        self.R = float(R)

    def _list(self, n):
        out, m = [0.0], 1
        while len(out) < n:
            out += [(m / self.R) ** 2] * 2
            m += 1
        return out

    @property
    def claims(self):
        return Claims([Claim(f"eigenvalues (m/R)^2, R = {self.R:g}, multiplicity 2 for m >= 1", THEOREM,
                             "Fourier series on the circle")])


class SphereSpectrum(_Closed):
    """The Laplace-Beltrami operator of a sphere of radius R: l(l + 1)/R^2 with multiplicity 2l + 1 (spherical
    harmonics)."""

    def __init__(self, R):
        self.R = float(R)

    def _list(self, n):
        out, l = [], 0
        while len(out) < n:
            out += [l * (l + 1) / self.R ** 2] * (2 * l + 1)
            l += 1
        return out

    @property
    def claims(self):
        return Claims([Claim(f"eigenvalues l(l + 1)/R^2, R = {self.R:g}, multiplicity 2l + 1", THEOREM,
                             "the spherical harmonics of degree l are its eigenfunctions and complete in L2(S^2)")])
