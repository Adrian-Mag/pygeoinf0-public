"""Matern kernels and their integrals against measures on points, straight segments and flat polygons.

The Bessel-potential kernel of H^s_l(R^d) (norm^2 = (2 pi)^-d int (1 + l^2 |xi|^2)^s |u^|^2 dxi):

    K(x, y) = k(|x - y|),   k(r) = c (r/l)^nu K_nu(r/l),   nu = s - d/2,   c = l^-d (2 pi)^(-d/2) 2^(1-s) / Gamma(s)

(the inverse Fourier transform of (1 + l^2 |xi|^2)^-s). For s > d/2 it is the reproducing kernel (k(0) finite); for
(d-1)/2 < s <= d/2 it is singular at r = 0 but its energy integrals int int k dmu dnu of path measures are finite,
which is all the package uses. Half-integer nu gives exponential-polynomial kernels (nu = 1/2: the exponential kernel,
s = (d+1)/2).

Exact reductions used here (every remaining integral is one-dimensional and analytic in its variable):
  radial tail      F(rho) = int_rho^inf k(t) t dt = c l^2 (rho/l)^(nu+1) K_(nu+1)(rho/l)    (closed form, every nu)
  segment pairs    x = p1 + a u1, y = p2 + b u2: z = x - y = (in-plane part) + h n (n normal to u1, u2), and
                   (a, b) -> in-plane part is affine with Jacobian sin(theta) onto a parallelogram P; so
                   int int k(|x - y|) da db = (1/sin theta) int_P k(sqrt(|z|^2 + h^2)) dz. Split P into the signed
                   triangles (0, V_k, V_k+1); in polar coordinates about 0, a triangle with far edge at distance d,
                   edge parameter tau = d sinh(w), gives
                       int [F(h) - F(sqrt(d^2 cosh^2 w + h^2))] / cosh(w) dw,
                   analytic in |Im w| < pi/2 for every d > 0 (edges through the origin contribute nothing).
                   Parallel pieces: one integral in the offset c = a - b with a trapezoid weight; its odd part is F.
  point-segment    int k(sqrt(delta^2 + (t - t0)^2)) dt with t - t0 = delta sinh(w) on each side of the projection.
The w-integrals are computed with 16-point Gauss rules on chunks of length <= 3 (the strip of analyticity has half
width pi/2, so each chunk converges to rounding), so near-singular configurations cost a few more chunks, not
accuracy.
"""
import math

import numpy as np
from scipy import special

from ...quadrature import gauss

NODES, CHUNK = 16, 3.0


def _zK(m, z):
    """z^m K_m(z) for z >= 0 (finite at 0 for m > 0: 2^(m-1) Gamma(m)); closed form for half-integer m."""
    z = np.asarray(z, float)
    h = 2 * m
    if m > 0 and abs(h - round(h)) < 1e-12 and round(h) % 2 == 1:
        n = int(round(m - 0.5))                   # z^m K_m(z) = sqrt(pi/2) e^-z sum_k (n+k)!/(k!(n-k)! 2^k) z^(n-k)
        poly = np.zeros_like(z)
        for k in range(n + 1):
            poly = poly + math.factorial(n + k) / (math.factorial(k) * math.factorial(n - k)) / 2 ** k * z ** (n - k)
        return math.sqrt(math.pi / 2) * np.exp(-z) * poly
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if m == 0:                                # H^1 in 2-D, H^(3/2) in 3-D: Cephes k0, 5x faster than kv (plan 028)
            out = special.k0(z)
        elif m == 1:                              # the radial tail of m = 0; both agree with kv to 4e-14 relative
            out = z * special.k1(z)
        else:
            out = z ** m * special.kv(m, z)
    if m > 0:
        out = np.where(z < 1e-300, 2 ** (m - 1) * math.gamma(m), out)
    return np.where(z > 745, 0.0, out)


class Matern:
    """The kernel of H^s_l(R^d): k(r), the radial tail F(rho) = int_rho^inf k(t) t dt, k(0) (finite when s > d/2)."""

    def __init__(self, s, d, length_scale):
        self.s, self.d, self.l = float(s), int(d), float(length_scale)
        if self.l <= 0:
            raise ValueError("the length scale must be positive")
        self.nu = self.s - self.d / 2.0
        self.c = self.l ** -self.d * (2 * math.pi) ** (-self.d / 2.0) * 2 ** (1 - self.s) / math.gamma(self.s)
        self.k0 = self.c * 2 ** (self.nu - 1) * math.gamma(self.nu) if self.nu > 0 else math.inf

    def __call__(self, r):
        return self.c * _zK(self.nu, np.abs(np.asarray(r, float)) / self.l)

    def F(self, rho):
        return self.c * self.l ** 2 * _zK(self.nu + 1, np.asarray(rho, float) / self.l)

    def G(self, h, rho):
        """int_h^rho k(t) t dt (h <= rho), without cancellation when rho is close to h."""
        h, rho = np.broadcast_arrays(np.asarray(h, float), np.asarray(rho, float))
        out = self.F(h) - self.F(rho)
        close = (h > 0) & ((rho - h) < 1e-3 * h)            # t k(t) is smooth on [h, rho] only away from t = 0
        if np.any(close):
            g, w = gauss(8)
            a, b = h[close], rho[close]
            t = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * g[None]
            out = out.copy()
            out[close] = 0.5 * (b - a) * np.sum(self(t) * t * w, axis=1)
        return out

    def small(self, eps):
        """int_0^eps k(t) dt for eps << l, from the expansion of k at 0. For non-integer nu,
        z^nu K_nu(z) = 2^(-nu-1) Gamma(-nu) z^(2 nu) + 2^(nu-1) Gamma(nu) + O(z^(2 nu + 2), z^2), both terms integrated
        (the first is the singularity when nu < 0, the second the value k(0) when nu > 0); nu = 0: c eps (log(2 l/eps)
        + 1 - gamma); integer nu >= 1: k(0) eps. Relative error O((eps/l)^2) or better (checked against mpmath)."""
        eps = np.asarray(eps, float)
        nu, c, l = self.nu, self.c, self.l
        if nu == 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(eps > 0, c * eps * (np.log(2 * l / np.where(eps > 0, eps, 1.0)) + 1 - np.euler_gamma), 0.0)
        if nu > 0 and abs(nu - round(nu)) < 1e-12:
            return self.k0 * eps
        sing = c * 2 ** (-nu - 1) * math.gamma(-nu) * l ** (-2 * nu) * eps ** (2 * nu + 1) / (2 * nu + 1)
        return sing + c * 2 ** (nu - 1) * math.gamma(nu) * eps

    def reach(self, tol=1e-17):
        """A distance beyond which the kernel's contributions are below tol (relative): integrals stop there."""
        return self.l * (-math.log(tol) + 10 * max(self.nu + 1, 1.0))

    def __repr__(self):
        return f"Matern(s={self.s:g}, d={self.d}, length_scale={self.l:g}; nu={self.nu:g})"


# ------------------------------------------------------------------------------------------ chunked rules

def _chunked(w0, w1, f):
    """sum over items i of int_{w0_i}^{w1_i} f(w, i) dw, for arrays w0 <= w1: Gauss on chunks of length <= CHUNK.
    f(W, I) takes node arrays W (m, NODES) and item indices I (m,) and returns values (m, NODES)."""
    w0, w1 = np.asarray(w0, float).ravel(), np.asarray(w1, float).ravel()
    span = np.maximum(w1 - w0, 0.0)
    live = np.nonzero(span > 0)[0]                       # empty items contribute nothing (and are not evaluated)
    K = np.ceil(span[live] / CHUNK).astype(int)
    I = np.repeat(live, K)
    j = np.arange(len(I)) - np.repeat(np.cumsum(K) - K, K)
    h = (span[I] / np.repeat(K, K))
    a = w0[I] + j * h
    t, w = gauss(NODES)
    W = a[:, None] + 0.5 * h[:, None] * (t[None] + 1)
    vals = np.sum(f(W, I) * w[None], axis=1) * 0.5 * h
    return np.bincount(I, weights=vals, minlength=len(w0))


def line_integral(kern, t0, delta, a, b):
    """int_a^b k(sqrt(delta^2 + (t - t0)^2)) dt, elementwise over broadcast arrays (t0, delta, a, b)."""
    t0, delta, a, b = np.broadcast_arrays(*(np.asarray(x, float) for x in (t0, delta, a, b)))
    shape = t0.shape
    t0, delta, a, b = (x.ravel() for x in (t0, delta, a, b))
    R = kern.reach()
    eps = 1e-12 * kern.l
    on = delta < eps                                  # on the line: [0, eps] by the expansion of k at 0, then sinh
    de = np.maximum(delta, eps)
    reach = np.sqrt(np.maximum(R * R - delta * delta, 0.0))
    out = np.zeros(t0.size)
    for side in (1.0, -1.0):
        if side > 0:
            tau0, tau1 = np.clip(t0, a, b) - t0, np.clip(t0 + reach, a, b) - t0
        else:
            tau0, tau1 = t0 - np.clip(t0, a, b), t0 - np.clip(t0 - reach, a, b)
        tau1 = np.maximum(tau1, tau0)
        head = on & (tau0 < eps) & (tau1 > tau0)
        if np.any(head):
            top = np.minimum(tau1, eps)
            out += np.where(head, kern.small(top) - kern.small(np.minimum(tau0, top)), 0.0)
            tau0 = np.where(head, top, tau0)
        w0, w1 = np.arcsinh(tau0 / de), np.arcsinh(tau1 / de)

        def f(W, I):
            return kern(np.sqrt(delta[I, None] ** 2 + (de[I, None] * np.sinh(W)) ** 2)) * de[I, None] * np.cosh(W)
        out += _chunked(w0, w1, f)
    return out.reshape(shape)


def point_segment(kern, X, P, Q):
    """int over segment j of k(|x_i - y|) dy (arc length), for points X (m, d) and segments P -> Q (k, d): (m, k)."""
    X, P, Q = (np.atleast_2d(np.asarray(a, float)) for a in (X, P, Q))
    L = np.linalg.norm(Q - P, axis=1)
    U = (Q - P) / L[:, None]
    rel = X[:, None, :] - P[None]
    t0 = np.einsum("mkd,kd->mk", rel, U)
    delta = np.sqrt(np.maximum(np.einsum("mkd,mkd->mk", rel, rel) - t0 * t0, 0.0))
    return line_integral(kern, t0, delta, 0.0, L[None])


def point_point(kern, X, Y):
    X, Y = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Y, float))
    return kern(np.linalg.norm(X[:, None, :] - Y[None], axis=-1))


# ------------------------------------------------------------------------------------------ segment pairs

def polygon_potential(kern, V, h):
    """int over the convex polygon with vertices V (m, 4, 2) (either orientation) of k(sqrt(|z|^2 + h^2)) dz, by the
    signed triangles (0, V_k, V_k+1) and the edge substitution tau = d sinh(w)."""
    m = len(V)
    A, B = V, np.roll(V, -1, axis=1)
    orient = np.sign(np.sum(A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0], axis=1))     # polygon orientation (2 area)
    E = B - A
    Lk = np.linalg.norm(E, axis=-1)
    T = E / np.where(Lk > 0, Lk, 1.0)[..., None]
    cr = A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]                                     # 2 x signed triangle area
    d = np.abs(cr) / np.where(Lk > 0, Lk, 1.0)                                             # distance of 0 to edge line
    tauA = np.sum(A * T, axis=-1)                                                          # positions along the edge
    tauB = np.sum(B * T, axis=-1)
    scale = np.linalg.norm(V, axis=-1).max(axis=1, keepdims=True) + 1e-300
    live = (d > 1e-14 * scale) & (Lk > 0)
    lo, hi = np.minimum(tauA, tauB), np.maximum(tauA, tauB)
    dd = np.where(live, d, 1.0)
    w0, w1 = np.arcsinh(lo / dd), np.arcsinh(hi / dd)
    w0, w1 = np.where(live, w0, 0.0), np.where(live, w1, 0.0)
    hh = np.repeat(h, 4)
    dflat = dd.ravel()

    def f(W, I):
        rho = np.sqrt((dflat[I, None] * np.cosh(W)) ** 2 + hh[I, None] ** 2)
        return kern.G(np.broadcast_to(hh[I, None], rho.shape), rho) / np.cosh(W)
    tri = _chunked(w0.ravel(), w1.ravel(), f).reshape(m, 4)
    return orient * np.sum(np.sign(cr) * tri, axis=1)


def _parallel(kern, s0, delta, L1, L2):
    """int_0^L1 int_0^L2 k(sqrt(delta^2 + (a - b + s0)^2)) db da via the offset c = a - b + s0 (trapezoid weight)."""
    br = np.sort(np.stack([s0 - L2, s0 + np.minimum(L1 - L2, 0.0), s0 + np.maximum(L1 - L2, 0.0), s0 + L1], axis=1),
                 axis=1)

    def weight(c):
        return np.clip(np.minimum(L1, c - s0 + L2) - np.maximum(0.0, c - s0), 0.0, None)
    total = np.zeros(len(s0))
    for j in range(3):
        c0, c1 = br[:, j], br[:, j + 1]
        ok = c1 > c0
        g0, g1 = weight(c0), weight(c1)
        beta = np.where(ok, (g1 - g0) / np.where(ok, c1 - c0, 1.0), 0.0)
        alpha = g0 - beta * c0
        I0 = line_integral(kern, 0.0, delta, c0, np.maximum(c1, c0))
        I1 = kern.F(np.sqrt(delta ** 2 + c0 ** 2)) - kern.F(np.sqrt(delta ** 2 + c1 ** 2))     # int c k dc
        total += np.where(ok, alpha * I0 + beta * I1, 0.0)
    return total


def segment_pair_list(kern, p1, q1, p2, q2):
    """The energy integrals of the pairs (p1[i] -> q1[i], p2[i] -> q2[i]), elementwise."""
    L1, L2 = np.linalg.norm(q1 - p1, axis=1), np.linalg.norm(q2 - p2, axis=1)
    u1, u2 = (q1 - p1) / L1[:, None], (q2 - p2) / L2[:, None]
    if p1.shape[1] == 2:
        u1, u2, p1, p2 = (np.concatenate([x, np.zeros((len(x), 1))], axis=1) for x in (u1, u2, p1, p2))
    cr = np.cross(u1, u2)
    sn = np.linalg.norm(cr, axis=1)
    par = sn < 1e-8                                          # ~sqrt(eps): parallel formula error O(sn) vs eps/sn
    out = np.empty(len(L1))
    w0 = p1 - p2
    k = ~par
    if np.any(k):
        nrm = cr[k] / sn[k, None]
        h = np.abs(np.sum(w0[k] * nrm, axis=1))
        e1 = u1[k]
        e2 = np.cross(nrm, e1)

        def proj(v):
            return np.stack([np.sum(v * e1, axis=1), np.sum(v * e2, axis=1)], axis=1)
        Z0, E1, E2 = proj(w0[k]), proj(L1[k, None] * u1[k]), proj(-L2[k, None] * u2[k])
        V = np.stack([Z0, Z0 + E1, Z0 + E1 + E2, Z0 + E2], axis=1)
        out[k] = polygon_potential(kern, V, h) / sn[k]
    if np.any(par):
        s0 = np.sum(w0[par] * u1[par], axis=1)
        sign = np.sign(np.sum(u1[par] * u2[par], axis=1))
        delta = np.linalg.norm(w0[par] - s0[:, None] * u1[par], axis=1)
        s0 = np.where(sign > 0, s0, s0 + L2[par])          # antiparallel: b' = L2 - b runs along u1
        out[par] = _parallel(kern, s0, delta, L1[par], L2[par])
    return out
