# Copied from experiments/EXP-024-continuum-sandbox/turf/problems.py (commit 5c4fc10), the independent referee of the
# turf comparison: closed-form Gram matrices and exact damped solutions, used by the sphere tests.
"""The referee: damped least-squares test problems on each package's own domain, with references computed
independently of every package (closed-form Gram matrices, spectrally accurate quadrature).

Every problem is
    M = L2(domain, mu)                       mu the natural measure (length, area, volume, surface area)
    (G m)_i = int k_i m dmu,  i = 1..n       smooth localised kernels k_i
    d = G m_true + e,  e ~ N(0, sigma^2 I)   sigma = 1 % of max |G m_true| (seeded)
    m_alpha = argmin  |G m - d|^2 / sigma^2 + alpha |m|^2_M
            = sum_i c_i k_i,   (K + alpha sigma^2 I) c = d,   K_ij = int k_i k_j dmu  (closed form).

A package answers with the values of its m_alpha on the referee's tensor grid of quadrature nodes (natural
coordinates, `axes`); the referee scores
    err   = |m_pkg - m_alpha|_M / |m_alpha|_M      the solver's error against the exact m_alpha
    model = |m_pkg - m_true|_M / |m_true|_M        for orientation: the inverse problem's own error
by its quadrature, which is exact to rounding for the smooth functions involved.

Kernels on lattices (torus, boxes) are products of one-dimensional factors, so the referee's sums are tensor
contractions. Kernels on the interval, the line problems and the box problems keep a distance of more than 6 widths
from the boundary, so that they and m_alpha vanish to rounding near it: then packages that pad the domain (pygeoinf's
line and plane) or impose boundary conditions solve the same problem.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import special


@dataclass
class Problem:
    name: str
    n: int
    params: dict                    # the kernels, for the adapters
    axes: list                      # referee nodes per natural axis
    axis_w: list                    # quadrature weights per axis (the measure included)
    m_true: callable                # natural coordinates (broadcast) -> values
    gram: np.ndarray
    factors: list = None            # separable kernels: per axis (q_k, n_k) factor values; members ordered 'ij'
    kern: callable = None           # otherwise: idx -> (q_total, len(idx)) kernel values on the grid
    sigma: float = 0.0
    alpha: float = 0.0
    d: np.ndarray = None
    d_clean: np.ndarray = None
    coef: np.ndarray = None
    m_alpha: np.ndarray = None      # on the grid
    info: dict = field(default_factory=dict)

    @property
    def shape(self):
        return tuple(len(a) for a in self.axes)

    @property
    def W(self):
        W = self.axis_w[0]
        for w in self.axis_w[1:]:
            W = np.multiply.outer(W, w)
        return W

    def grid(self):
        return np.meshgrid(*self.axes, indexing="ij")

    def norm(self, v):
        return float(np.sqrt(np.sum(self.W * v * v)))

    def apply(self, F):
        """(G F)_i = int k_i F for F on the grid."""
        F = self.W * F
        if self.factors is not None:
            for k, A in enumerate(self.factors):
                F = np.tensordot(F, A, axes=([0], [0]))          # contracts the leading grid axis; appends n_k
            return F.ravel()
        F = F.ravel()
        out = np.empty(self.n)
        for s in range(0, self.n, 256):
            idx = np.arange(s, min(self.n, s + 256))
            out[idx] = F @ self.kern(idx)
        return out

    def synth(self, c):
        """sum_i c_i k_i on the grid."""
        if self.factors is not None:
            C = c.reshape([A.shape[1] for A in self.factors])
            for A in self.factors:
                C = np.tensordot(C, A, axes=([0], [1]))           # contracts the leading member axis; appends q_k
            return C
        out = np.zeros(int(np.prod(self.shape)))
        for s in range(0, self.n, 256):
            idx = np.arange(s, min(self.n, s + 256))
            out += self.kern(idx) @ c[idx]
        return out.reshape(self.shape)

    def score(self, values):
        V = np.asarray(values, float).reshape(self.shape)
        mt = self.m_true(*self.grid())
        return dict(err=self.norm(V - self.m_alpha) / self.norm(self.m_alpha),
                    model=self.norm(V - mt) / self.norm(mt), finite=bool(np.all(np.isfinite(V))))

    def spec(self):
        """What an adapter receives (JSON-able apart from arrays)."""
        return dict(name=self.name, n=self.n, params=self.params, axes=self.axes, d=self.d, sigma=self.sigma,
                    alpha=self.alpha)


def discrepancy_alpha(K, d, sigma):
    """alpha with chi^2(m_alpha) = n (the discrepancy principle), by bisection in log alpha on the eigenvalues of K:
    the residual of m_alpha is alpha sigma^2 (K + alpha sigma^2)^-1 d."""
    lam, U = np.linalg.eigh(K)
    b2 = (U.T @ d) ** 2
    chi2 = lambda a: float(np.sum((a * sigma ** 2 / (np.maximum(lam, 0) + a * sigma ** 2)) ** 2 * b2) / sigma ** 2)  # noqa: E731
    lo, hi = -12.0, 12.0
    if chi2(10 ** lo) > len(d):
        return 10 ** lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if chi2(10 ** mid) > len(d) else (mid, hi)
    return 10 ** lo


def _finish(P, alpha, seed=0, noise=0.01):
    """Noisy data and the exact m_alpha; alpha=None: chosen by the discrepancy principle (then fixed and handed to
    every package)."""
    P.d_clean = P.apply(P.m_true(*P.grid()))
    P.sigma = noise * float(np.abs(P.d_clean).max())
    P.d = P.d_clean + P.sigma * np.random.default_rng(seed).standard_normal(P.n)
    P.alpha = float(discrepancy_alpha(P.gram, P.d, P.sigma) if alpha is None else alpha)
    A = P.gram + P.alpha * P.sigma ** 2 * np.eye(P.n)
    P.coef = np.linalg.solve(A, P.d)
    P.m_alpha = P.synth(P.coef)
    mt = P.m_true(*P.grid())
    P.info.update(cond_K=float(np.linalg.cond(P.gram)), cond_A=float(np.linalg.cond(A)),
                  chi2_over_n=float(np.sum((P.gram @ P.coef - P.d) ** 2) / P.sigma ** 2 / P.n),
                  model_err_of_m_alpha=P.norm(P.m_alpha - mt) / P.norm(mt))
    return P


# ------------------------------------------------------------------------------------------ 1-D building blocks

def gauss_composite(a, b, panels, order):
    t, w = np.polynomial.legendre.leggauss(order)
    e = np.linspace(a, b, panels + 1)
    h = np.diff(e)[:, None]
    return (e[:-1, None] + 0.5 * h * (t[None] + 1)).ravel(), (0.5 * h * w[None]).ravel()


def periodic_nodes(m, L=2 * math.pi):
    return np.arange(m) * L / m, np.full(m, L / m)


def gauss_1d(n, width=1.1, panels_per_width=2.0, order=12):
    """Gaussian kernels exp(-((x - c)/w)^2) on [0, 1], centres linspace(a, 1 - a, n), w = width * spacing, with a
    the smallest margin (at least 0.1) such that 6 w <= a: the kernels vanish to rounding at the boundary."""
    lo = max(0.1, 6 * width / (n - 1 + 12 * width))
    hi = 1 - lo
    c = np.linspace(lo, hi, n)
    w = width * (hi - lo) / (n - 1)
    assert 6 * w <= min(lo, 1 - hi) + 1e-12, "kernels must vanish at the boundary"
    A, B = np.meshgrid(c, c, indexing="ij")
    mid = 0.5 * (A + B)
    K = np.exp(-(A - B) ** 2 / (2 * w * w)) * w * math.sqrt(math.pi / 8) * (
        special.erf(math.sqrt(2) * (1 - mid) / w) + special.erf(math.sqrt(2) * mid / w))
    x, q = gauss_composite(0.0, 1.0, max(int(32 * panels_per_width), int(math.ceil(panels_per_width / w))), order)
    return c, w, K, x, q, np.exp(-((x[:, None] - c[None]) / w) ** 2)


def vonmises_1d(n, width=1.1, R=1.0):
    """von Mises kernels exp(kappa (cos(phi - c) - 1)) = exp(-2 kappa sin^2((phi - c)/2)) on the circle of radius R
    (measure R dphi), c equispaced, angular width s = width * spacing, kappa = 1 / s^2. They are evaluated in the second
    form everywhere: cos(phi - c) - 1 cancels, and kappa times its rounding error is 1e-11 relative at kappa = 1e5."""
    c = 2 * np.pi * np.arange(n) / n
    kap = 1.0 / (width * 2 * np.pi / n) ** 2
    # int e^{kap (cos(phi-a) + cos(phi-b) - 2)} dphi = 2 pi I_0(2 kap D) e^{-2 kap}, D = |cos((a-b)/2)|; written as
    # ive(0, 2 kap D) e^{2 kap (D - 1)} with D - 1 = -2 sin^2(delta/4), delta = (a - b) wrapped into [-pi, pi]
    # (2 kap (D - 1) computed directly cancels: 4e-12 relative error at kappa = 8e4)
    delta = np.mod(np.subtract.outer(c, c) + np.pi, 2 * np.pi) - np.pi
    D = np.cos(0.5 * delta)
    K = R * 2 * np.pi * special.ive(0, 2 * kap * D) * np.exp(-4 * kap * np.sin(0.25 * delta) ** 2)
    m = max(512, 2 * int(12 * math.sqrt(kap)) + 8 * n)
    x, q = periodic_nodes(m)
    return c, kap, K, x, q * R, np.exp(-2 * kap * np.sin(0.5 * (x[:, None] - c[None])) ** 2)


# ------------------------------------------------------------------------------------------ problems

def interval(n, alpha=None):
    """L2([0, 1]); n Gaussians with centres in [0.25, 0.75]."""
    c, w, K, x, q, F = gauss_1d(n)
    P = Problem("interval", n, dict(a=0.0, b=1.0, centres=c, width=w, kernel="exp(-((x-c)/w)^2)"),
                [x], [q], lambda x: np.sin(2 * np.pi * x) + 0.5 * x ** 2, K, factors=[F])
    return _finish(P, alpha)


def circle(n, alpha=None, R=1.0):
    """L2(S^1_R, R dphi); n von Mises kernels."""
    c, kap, K, x, q, F = vonmises_1d(n, R=R)
    P = Problem("circle", n, dict(radius=R, centres=c, kappa=kap, kernel="exp(-2 kappa sin^2((phi-c)/2))"),
                [x], [q], lambda phi: np.exp(np.cos(phi)) * np.sin(2 * phi) + 0.3 * np.cos(5 * phi), K, factors=[F])
    return _finish(P, alpha)


def torus(n_side, alpha=None):
    """L2 of the flat torus [0, 2 pi)^2 (dx dy); products of von Mises kernels on an n_side x n_side lattice."""
    c, kap, K1, x, q, F = vonmises_1d(n_side)
    P = Problem("torus", n_side ** 2, dict(period=2 * np.pi, centres_1d=c, kappa=kap, n_side=n_side,
                                           kernel="exp(-2 kappa (sin^2((x-cx)/2) + sin^2((y-cy)/2)))"),
                [x, x], [q, q], lambda x, y: np.sin(x) * np.cos(2 * y) + np.exp(np.cos(x - 1) + np.cos(y) - 2),
                np.kron(K1, K1), factors=[F, F])
    return _finish(P, alpha)


def box2(n_side, alpha=None):
    """L2([0, 1]^2); products of Gaussians on an n_side^2 lattice in [0.25, 0.75]^2."""
    c, w, K1, x, q, F = gauss_1d(n_side)
    P = Problem("box2", n_side ** 2, dict(sides=((0, 1),) * 2, centres_1d=c, width=w, n_side=n_side,
                                          kernel="exp(-|x-c|^2/w^2)"),
                [x, x], [q, q], lambda x, y: np.sin(3 * x) * np.cos(2 * y) + 0.5 * y ** 2, np.kron(K1, K1),
                factors=[F, F])
    return _finish(P, alpha)


def box3(n_side, alpha=None):
    """L2([0, 1]^3); products of Gaussians on an n_side^3 lattice in [0.25, 0.75]^3."""
    c, w, K1, x, q, F = gauss_1d(n_side, panels_per_width=0.5, order=10)   # a coarser grid: it is cubed
    P = Problem("box3", n_side ** 3, dict(sides=((0, 1),) * 3, centres_1d=c, width=w, n_side=n_side,
                                          kernel="exp(-|x-c|^2/w^2)"),
                [x, x, x], [q, q, q], lambda x, y, z: np.sin(3 * x) * np.cos(2 * y) + z ** 2,
                np.kron(np.kron(K1, K1), K1), factors=[F, F, F])
    return _finish(P, alpha)


def radial(n, alpha=None):
    """Radial functions on the unit ball: L2([0, 1], 4 pi r^2 dr); spherical shells exp(-((r - c)/w)^2) with centres
    in [a, 1 - a] (they vanish at r = 0 and r = 1 to rounding). The Gram matrix is computed by composite Gauss
    quadrature (closed forms exist but are unwieldy); its accuracy is checked against a second, finer rule."""
    c, w, _, x, q, F = gauss_1d(n)
    q = q * 4 * np.pi * x * x
    K = (F * q[:, None]).T @ F
    x2, q2 = gauss_composite(0.0, 1.0, 3 * len(x) // 12, 16)
    F2 = np.exp(-((x2[:, None] - c[None]) / w) ** 2)
    K2 = (F2 * (q2 * 4 * np.pi * x2 * x2)[:, None]).T @ F2
    P = Problem("radial", n, dict(a=0.0, b=1.0, centres=c, width=w, weight="4 pi r^2", kernel="exp(-((r-c)/w)^2)"),
                [x], [q], lambda r: np.cos(3 * r) + r ** 2, K2, factors=[F])
    P.info["gram_rule_difference"] = float(np.abs(K - K2).max() / np.abs(K2).max())
    return _finish(P, alpha)


def fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    lon = np.mod(np.pi * (1 + 5 ** 0.5) * i, 2 * np.pi)
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(lon), r * np.sin(lon), z], axis=1)


def sphere(n, alpha=None, R=1.0, width=1.1):
    """L2(S^2_R, surface measure); caps exp(kappa (u . c_i - 1)) = exp(-kappa |u - c_i|^2 / 2) at Fibonacci points
    c_i, angular width s = width * sqrt(4 pi / n), kappa = 1/s^2; evaluated in the second form (haversine on the
    grid), which does not cancel. Grid: Gauss-Legendre in cos(theta) x equispaced phi."""
    C = fibonacci(n)
    kap = 1.0 / (width * math.sqrt(4 * math.pi / n)) ** 2
    L = int(12 * math.sqrt(kap)) + 24                          # harmonic degree carrying the kernels to rounding
    mu, wmu = np.polynomial.legendre.leggauss(L + 8)
    phi, wphi = periodic_nodes(2 * L + 16)
    theta = np.arccos(mu)
    ST, CT = np.sin(theta), np.cos(theta)
    U = np.stack([np.multiply.outer(ST, np.cos(phi)), np.multiply.outer(ST, np.sin(phi)),
                  np.multiply.outer(CT, np.ones_like(phi))], axis=-1).reshape(-1, 3)
    tc, pc = np.arccos(np.clip(C[:, 2], -1, 1)), np.arctan2(C[:, 1], C[:, 0])
    TT, PP = [a.ravel() for a in np.meshgrid(theta, phi, indexing="ij")]

    def hav(idx):           # |u - c|^2 / 4 = sin^2(dtheta/2) + sin(t) sin(tc) sin^2(dphi/2)
        return (np.sin(0.5 * (TT[:, None] - tc[idx])) ** 2
                + np.sin(TT)[:, None] * np.sin(tc[idx]) * np.sin(0.5 * (PP[:, None] - pc[idx])) ** 2)
    S_ = np.linalg.norm(C[:, None, :] + C[None, :, :], axis=2)
    Dm = np.sum((C[:, None, :] - C[None, :, :]) ** 2, axis=2)
    z = kap * S_
    # int_{S^2} e^{kap u.(a+b)} dOmega = 4 pi sinh(z)/z, z = kap |a + b|; times R^2 e^{-2 kap}. The exponent z - 2 kap =
    # kap (|a+b| - 2) = -kap |a-b|^2 / (|a+b| + 2) for unit vectors, written so (the difference cancels)
    K = (R * R * 2 * np.pi * np.where(z > 1e-8, -np.expm1(-2 * z) / np.where(z > 1e-8, z, 1), 2.0)
         * np.exp(-kap * Dm / (S_ + 2)))

    def mt(theta, phi):
        x, y, zz = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
        return zz + 0.5 * x * y + np.exp(4 * (0.6 * x + 0.8 * zz - 1))
    P = Problem("sphere", n, dict(radius=R, centres=C, kappa=kap, kernel="exp(-kappa |u-c|^2 / 2)", lmax_kernels=L),
                [theta, phi], [wmu * R * R, wphi], mt, K, kern=lambda idx: np.exp(-2 * kap * hav(idx)))
    return _finish(P, alpha)


PROBLEMS = dict(interval=interval, radial=radial, circle=circle, torus=torus, box2=box2, box3=box3, sphere=sphere)


def build(name, n, **kw):
    return PROBLEMS[name](n, **kw)


if __name__ == "__main__":
    import time
    for name, n in [("interval", 50), ("radial", 50), ("circle", 50), ("torus", 8), ("box2", 8), ("box3", 5), ("sphere", 100)]:
        t = time.perf_counter()
        P = build(name, n)
        # self-check: the closed-form Gram against the referee quadrature, column by column
        E = 0.0
        for j in range(0, P.n, max(1, P.n // 7)):
            e = np.zeros(P.n); e[j] = 1
            E = max(E, np.abs(P.apply(P.synth(e)) - P.gram[:, j]).max() / np.abs(P.gram).max())
        print(name, P.n, P.shape, "gram vs quadrature %.1e" % E,
              {k: round(v, 4) for k, v in P.info.items()}, "%.2fs" % (time.perf_counter() - t), flush=True)
