"""Spherical-harmonic realisations on a sphere of radius R.

An element of L2(S_R) is realised as f(u) = sum_{l <= L} sum_m a_lm Y_lm(u), Y_lm the real spherical harmonics that are
orthonormal on the unit sphere:

    Y_l0 = p_l0(cos theta),  Y_lm^c = sqrt(2) p_lm(cos theta) cos(m phi),  Y_lm^s = sqrt(2) p_lm(cos theta) sin(m phi)

with p_lm the associated Legendre functions normalised so that 2 pi int p_lm^2 d(cos theta) = 1. The measure of S_R is
R^2 dOmega, so <f, g> = R^2 sum a_lm b_lm: inner products, norms and integrals are exact on the realisations.

Coefficients are packed degree by degree, row l^2 + j for j = 0 (m = 0), 2m - 1 (cos), 2m (sin), so the realisation
of degree L has (L + 1)^2 rows and truncation or zero-padding is slicing. Transforms use Gauss-Legendre nodes in
cos(theta) (L + 1 of them) and 2L + 2 equispaced longitudes, on which analysis and synthesis are exact for degree <= L
(the integrands are polynomials of degree <= 2L + 1 in cos(theta) and trigonometric polynomials of degree <= 2L in
phi); the functions p_lm come from the standard three-term recurrence in l, one order m at a time.

Adaptivity: L runs through 15, 23, 31, 47, 63, ... (steps of about 1.5) up to LMAX until the coefficient energy of the top eighth of the degrees is
below the tolerance relative to the column's norm; the result is then truncated to the smallest degree with the same
property. The estimated error is R times the root of the discarded energy plus a rounding term.

Compared with the theta x phi tensor basis this is the natural basis of the sphere: the same accuracy needs about a
third of the coefficients (no pole oversampling: the phi resolution of a tensor basis is the same at every latitude).
"""
import inspect
import math
import warnings
from functools import lru_cache

import numpy as np

from ...core.errors import Unsupported
from ..backends import Backend
from .legendre import EPS, TINY
from ...spaces.harmonic import assoc, order_rows, rows  # noqa: F401  (the definitions of p_lm and the packing)
from ...quadrature import gauss

LADDER = (15, 23, 31, 47, 63, 95, 127, 191, 255, 383, 511)   # steps of about 1.5
LMAX = LADDER[-1]
MAX_VALUES = 1 << 23


def _ro(a):
    a.flags.writeable = False
    return a


@lru_cache(maxsize=12)
def grid(L):
    """The transform grid of degree L: theta (L + 1 Gauss nodes in cos theta), Gauss weights, phi (2L + 2), and the
    Legendre blocks P_m (l = m..L) at the nodes."""
    mu, w = gauss(L + 1)                         # the refined rule: numpy's leggauss loses 1e-13 at 128 nodes
    mu, w = mu[::-1].copy(), w[::-1].copy()                       # theta increasing
    theta = np.arccos(mu)
    s = np.sqrt(np.maximum(0.0, 1 - mu * mu))
    phi = 2 * np.pi * np.arange(2 * L + 2) / (2 * L + 2)
    P = tuple(_ro(assoc(m, L, mu, s)) for m in range(L + 1))
    return _ro(theta), _ro(w), _ro(phi), P


# ------------------------------------------------------------------------------------------ transforms

def analyse(V, L):
    """Packed coefficients ((L+1)^2, N) of values V (L+1, 2L+2, N) on grid(L)."""
    theta, w, phi, P = grid(L)
    nlon = phi.size
    F = np.fft.rfft(V, axis=1)[:, :L + 1]                          # int f e^{-i m phi} dphi = (2 pi / nlon) F
    F = np.moveaxis(F, 1, 0) * (w * (2 * np.pi / nlon))[None, :, None]   # (m, theta, N)
    Fr, Fi = np.ascontiguousarray(F.real), np.ascontiguousarray(F.imag)  # contiguous blocks: BLAS products per m
    out = np.zeros(((L + 1) ** 2, V.shape[2]))
    for m, (rc, rs) in enumerate(order_rows(L)):
        if m == 0:
            out[rc] = P[0] @ Fr[0]
        else:
            out[rc] = math.sqrt(2) * (P[m] @ Fr[m])
            out[rs] = -math.sqrt(2) * (P[m] @ Fi[m])
    return out


def synth_rows(C, L, P_rows):
    """Per order m, the theta-profiles g_m^c, g_m^s (len(theta), N) of coefficients C (degree L), given the Legendre
    blocks P_rows[m] (l = m..L) at the theta values."""
    gc, gs = [], []
    for m, (rc, rs) in enumerate(order_rows(L)):
        gc.append(P_rows[m].T @ C[rc])
        gs.append(None if rs is None else P_rows[m].T @ C[rs])
    return gc, gs


def synthesise_grid(C, L, theta, phi):
    """Values (len(theta), len(phi), N) on the tensor grid theta x phi."""
    theta = np.asarray(theta, float)
    mu, s = np.cos(theta), np.abs(np.sin(theta))
    P = [assoc(m, L, mu, s) for m in range(L + 1)]
    gc, gs = synth_rows(C, L, P)
    phi = np.asarray(phi, float)
    out = np.einsum("tn,p->tpn", gc[0], np.ones_like(phi))
    for m in range(1, L + 1):
        out += math.sqrt(2) * (np.einsum("tn,p->tpn", gc[m], np.cos(m * phi))
                               + np.einsum("tn,p->tpn", gs[m], np.sin(m * phi)))
    return out


def synthesise_on(C, L, Lg):
    """Values on grid(Lg) (Lg >= L), by FFT in phi."""
    theta, w, phi, P = grid(Lg)
    nlon = phi.size
    Pl = [Pm[:L - m + 1] for m, Pm in enumerate(P[:L + 1])]
    gc, gs = synth_rows(C, L, Pl)
    N = C.shape[1]
    G = np.zeros((theta.size, nlon // 2 + 1, N), complex)
    G[:, 0] = gc[0]
    for m in range(1, L + 1):
        G[:, m] = (math.sqrt(2) / 2) * (gc[m] - 1j * gs[m])
    return np.fft.irfft(G, n=nlon, axis=1) * nlon


def evaluate_points(C, L, theta, phi):
    """Values (npts, N) at points (theta, phi) (NaN where theta is NaN)."""
    theta, phi = np.asarray(theta, float).ravel(), np.asarray(phi, float).ravel()
    ok = np.isfinite(theta) & np.isfinite(phi)
    out = np.full((theta.size, C.shape[1]), np.nan)
    t, p = theta[ok], phi[ok]
    mu, s = np.cos(t), np.abs(np.sin(t))
    acc = np.zeros((t.size, C.shape[1]))
    for m, (rc, rs) in enumerate(order_rows(L)):
        Pm = assoc(m, L, mu, s)
        if m == 0:
            acc += Pm.T @ C[rc]
        else:
            acc += math.sqrt(2) * ((Pm.T @ C[rc]) * np.cos(m * p)[:, None] + (Pm.T @ C[rs]) * np.sin(m * p)[:, None])
    out[ok] = acc
    return out


# ------------------------------------------------------------------------------------------ realisations

class SphericalHarmonicSeries:
    """N functions on S_R: packed spherical-harmonic coefficients C ((L+1)^2, N) and error estimates err (N,)."""

    __slots__ = ("L", "C", "err", "R")
    CAPABILITIES = ("harmonic_coefficients", "box_support", "box_rule")

    def __init__(self, L, C, err, R=1.0):
        self.L, self.R = int(L), float(R)
        self.C = np.asarray(C, float)
        self.err = np.asarray(err, float)

    @property
    def N(self):
        return self.C.shape[1]

    @property
    def size(self):
        return self.C.size

    def padded(self, L):
        if L == self.L:
            return self.C
        out = np.zeros(((L + 1) ** 2, self.N))
        out[:min(L, self.L) ** 2 + 2 * min(L, self.L) + 1] = self.C[:(min(L, self.L) + 1) ** 2]
        return out

    # capabilities (registry.CAPABILITIES; box_integrals.py) ---------------------------------------------------
    def box_support(self, H, l, u, rtol=1e-8):
        from ..bounds.box_integrals import sphere_support
        return sphere_support(self, H, l, u, rtol)

    def box_rule(self, H, Rs, l, u, weight, oversample, least):
        from ..bounds.box_integrals import SphereRule
        return SphereRule(H, Rs, l, u, weight, oversample, least)

    def harmonic_coefficients(self, L):
        return np.asarray(self.padded(L), float) if self.L != L else np.asarray(self.C, float)


def degree_energy(C, L):
    """Per degree l and column: sum over m of a_lm^2, shape (L + 1, N)."""
    return np.add.reduceat(C * C, np.arange(L + 1) ** 2, axis=0)     # degree l occupies rows l^2 .. (l+1)^2 - 1


def _resolved_degree(E, thr):
    """The smallest degree L' (per column) with sqrt(sum_{l > L'} E_l) <= thr."""
    tail = np.sqrt(np.cumsum(E[::-1], axis=0)[::-1])                 # tail[l] = sqrt(sum_{k >= l} E_k)
    tail = np.vstack([tail[1:], np.zeros((1, E.shape[1]))])          # tail[l] = sqrt(sum_{k > l} E_k)
    ok = tail <= thr[None]
    return np.argmax(ok, axis=0)


def chop(C, L, thr):
    E = degree_energy(C, L)
    Lc = int(_resolved_degree(E, thr).max()) if C.shape[1] else 0
    disc = np.sqrt(np.maximum(E[Lc + 1:].sum(axis=0), 0.0))
    return C[:(Lc + 1) ** 2].copy(), Lc, disc


def fit_sh(fun, N, R=1.0, tol=1e-14, lmax=LMAX, lmin=None):
    """Adaptive spherical-harmonic realisation of fun(theta, phi) -> (..., N). Returns (SphericalHarmonicSeries, unresolved). lmin: the
    least degree (a declared resolution's, core/fits.py); refused above lmax."""
    if lmin is not None and lmin > lmax:
        from ...core.errors import RealisationError
        raise RealisationError(f"the declared resolution needs degree >= {lmin} on the sphere's sampling grid, beyond "
                               f"the cap {lmax} of the harmonic realisation: declare a coarser resolution, or use "
                               "realisation='tensor' with breaks= at the feature's support")
    last = None
    for L in [x for x in LADDER if (lmin or 0) <= x <= lmax] or [lmax]:
        theta, w, phi, P = grid(L)
        T, Ph = np.meshgrid(theta, phi, indexing="ij")
        V = np.asarray(fun(T, Ph), float).reshape(theta.size, phi.size, N)
        if not np.all(np.isfinite(V)):
            raise ValueError("the function is not finite on the sphere's sampling grid")
        C = analyse(V, L)
        norm = np.sqrt((C * C).sum(axis=0))
        vscale = np.abs(V).reshape(-1, N).max(axis=0)
        # resolved when the tail is below the tolerance relative to the norm, or at the rounding floor of the
        # quadrature (each coefficient carries about eps * max|f| of rounding; (L + 1)^2 coefficients in all)
        thr = np.maximum(tol * norm, 2 * EPS * (L + 1) * vscale) + TINY
        E = degree_energy(C, L)
        top = max(2, (L + 1) // 8)
        tail = np.sqrt(E[L + 1 - top:].sum(axis=0))
        last = (L, C, thr, norm, vscale)
        if np.all(tail <= thr):
            break
    L, C, thr, norm, vscale = last
    unresolved = int(np.sum(np.sqrt(degree_energy(C, L)[L + 1 - max(2, (L + 1) // 8):].sum(axis=0)) > thr))
    Cc, Lc, disc = chop(C, L, thr)
    err = R * np.sqrt(disc ** 2 + (EPS * vscale) ** 2 * (Lc + 1) ** 2)
    return SphericalHarmonicSeries(Lc, Cc, err, R), unresolved


def angular_degree(R, w, ladder):
    """The least degree of the ladder whose grid's spacing on a sphere of radius R is <= w / 2 (theta: Gauss nodes in
    cos theta, at most about 1.2 pi / (L + 1) apart with the poles; phi: pi / (L + 1)), or the next degree beyond it
    (then refused by the fit)."""
    need = int(math.ceil(1.2 * math.pi * R / (0.5 * w))) - 1
    return next((L for L in ladder if L >= need), need)


def lincomb(terms):
    L = max(R.L for _, R in terms)
    C = sum(a * R.padded(L) for a, R in terms)
    err = sum(abs(a) * R.err for a, R in terms)
    return SphericalHarmonicSeries(L, C, err, terms[0][1].R)


def hstack(Rs):
    L = max(R.L for R in Rs)
    return SphericalHarmonicSeries(L, np.hstack([R.padded(L) for R in Rs]), np.concatenate([R.err for R in Rs]), Rs[0].R)


def multiply(A, B, tol=1e-14):
    """Pointwise product (exact up to the chop: the product of degrees L_A and L_B has degree L_A + L_B)."""
    if A.N != B.N and 1 not in (A.N, B.N):
        raise ValueError("products of families need equal sizes")
    L = min(A.L + B.L, LMAX)
    va, vb = synthesise_on(A.C, A.L, L), synthesise_on(B.C, B.L, L)
    C = analyse(va * vb, L)
    sa = np.abs(va).reshape(-1, A.N).max(axis=0)
    sb = np.abs(vb).reshape(-1, B.N).max(axis=0)
    Cc, Lc, disc = chop(C, L, 4 * EPS * np.maximum(np.sqrt((C * C).sum(axis=0)), TINY))
    return SphericalHarmonicSeries(Lc, Cc, sa * B.err + sb * A.err + A.R * disc, A.R)


# ------------------------------------------------------------------------------------------ backend

class SphereBackend(Backend):
    """Spherical-harmonic realisations for L2 of a sphere: surface measure sigma, or c sigma with a constant density
    c > 0 (plan 064). <f, g> = c R^2 sum_lm f_lm g_lm; self.mass = c R^2 is that factor (consumers that pair
    coefficients by hand use it), self.c the density."""

    dim = 2
    realisation_name = "harmonics"
    harmonic = "sphere"                 # realisations/series/harmonic_degrees.py: degree-wise operations, separable families

    def __init__(self, measure, tol):
        from ...measures import constant_density
        c = constant_density(measure)
        if c is None:
            raise Unsupported(f"the density of {measure} is not constant: on a sphere only constant densities are "
                              "realised (plan 064)")
        self.measure, self.tol = measure, tol
        self.domain = measure.set
        self.R = self.domain.radius
        self.c = c
        self.mass = c * self.R ** 2
        self.jac_max = self.R ** 2

    # construction ----------------------------------------------------------------------------
    def probe(self):
        t = np.array([0.7, 1.4, 2.3])
        return tuple(np.asarray(c, float) for c in self.domain.to_cartesian(t, np.array([0.5, 2.0, 4.5])))

    def _internal(self, fun):
        to_cart = self.domain.to_cartesian
        return lambda theta, phi: fun(*to_cart(theta, phi))

    def fit(self, fun, N=1, breaks=None):
        return fit_sh(self._internal(fun), N, self.R, tol=self.tol, lmin=getattr(breaks, "degree", None))

    def resolution_hint(self, breaks, w):
        """Breaks for a fit at a declared resolution w (core/fits.py): the least degree whose grid (L + 1 Gauss nodes in
        cos theta, 2L + 2 longitudes) has spacing <= w / 2 on the sphere of radius R."""
        from ..backends import NaturalBreaks
        return NaturalBreaks(tuple(breaks or (None, None)), degree=angular_degree(self.R, w, LADDER),
                             box=getattr(breaks, "box", None))

    def fit_family(self, fun_idx, N, breaks=None, sparse=None, name="F"):
        out, s = [], 0
        step = max(16, MAX_VALUES // (192 * 384))       # columns per chunk: grids up to degree 191 within the budget
        while s < N:
            idx = np.arange(s, min(N, s + step))
            R, u = fit_sh(self._internal(lambda *X, idx=idx: fun_idx(*X, idx)), len(idx), self.R, tol=self.tol,
                          lmin=getattr(breaks, "degree", None))
            if u:
                warnings.warn(f"family {name}: {u} member(s) not resolved at degree {LMAX} in columns {s}..{idx[-1]} "
                              "(a function that is not smooth on the sphere, e.g. a product f(theta) g(phi) with g not "
                              "constant and f not vanishing to high order at a pole, has slowly decaying coefficients)")
            out.append(R)
            s += len(idx)
        return out[0] if len(out) == 1 else hstack(out)

    def product_family(self, factors, params, name="K"):
        """Products f_theta(theta) f_phi(phi) (natural coordinates), realised as spherical harmonics. A product is
        continuous on the sphere only if f_phi is constant or f_theta vanishes at the poles."""
        if len(factors) != 2:
            raise ValueError("the sphere has natural coordinates (theta, phi): give 2 factors (None for 1)")
        arrays = {k: np.asarray(v, float).ravel() for k, v in params.items()}
        N = len(next(iter(arrays.values())))
        names = []
        for f in factors:
            if f is None:
                names.append(None)
                continue
            sig = inspect.signature(f).parameters
            var = any(p.kind == p.VAR_KEYWORD for p in sig.values())
            names.append(list(arrays) if var else [p for p in list(sig)[1:] if p in arrays])
        from_cart = self.domain.from_cartesian

        def fun_idx(*X_and_idx):
            *X, idx = X_and_idx
            th, ph = from_cart(*X)
            out = 1.0
            for f, nm, t in zip(factors, names, (th, ph)):
                if f is not None:
                    out = out * f(np.asarray(t)[..., None], **{k: arrays[k][idx] for k in nm})
            return np.broadcast_to(out, np.shape(th) + (len(idx),))
        return self.fit_family(fun_idx, N, name=name)

    def constant(self, values):
        values = np.atleast_1d(np.asarray(values, float))
        return SphericalHarmonicSeries(0, values[None] * math.sqrt(4 * math.pi), np.zeros(len(values)), self.R)

    def coordinate(self, k):
        R, _ = self.fit(lambda *X: X[k])
        return R

    # algebra ---------------------------------------------------------------------------------
    def lincomb(self, terms):
        return lincomb(terms)

    def combine(self, R, A):
        A = np.asarray(A, float)
        return SphericalHarmonicSeries(R.L, R.C @ A, np.abs(A).T @ R.err, R.R)

    def colsum(self, R):
        return self.combine(R, np.ones((R.N, 1)))

    def take(self, R, idx):
        sel = np.atleast_1d(idx)
        return SphericalHarmonicSeries(R.L, R.C[:, sel], R.err[sel], R.R)

    def hstack(self, Rs):
        return hstack(Rs)

    def multiply(self, A, B):
        return multiply(A, B, self.tol)

    # measure ---------------------------------------------------------------------------------
    def gram(self, A, B):
        L = min(A.L, B.L)
        S = (L + 1) ** 2
        return self.mass * (A.C[:S].T @ B.C[:S])

    def sqnorms(self, A):
        return self.mass * (A.C * A.C).sum(axis=0)

    def integral(self, R, over=None):
        if over is not None:
            raise Unsupported("integrals over parts of the sphere are not implemented yet")
        return self.mass * math.sqrt(4 * math.pi) * R.C[0]

    def lp_norm(self, R, p):
        Lg = min(2 * R.L + 8, LMAX)
        theta, w, phi, _ = grid(Lg)
        V = np.abs(synthesise_on(R.C, R.L, Lg))
        if math.isinf(p):
            return V.reshape(-1, R.N).max(axis=0)
        W = (w[:, None] * (2 * np.pi / phi.size) * self.mass)[..., None]
        return ((V ** p) * W).reshape(-1, R.N).sum(axis=0) ** (1.0 / p)


    # values and description ------------------------------------------------------------------
    def evaluate(self, R, *X):
        shape = np.broadcast(*X).shape
        th, ph = self.domain.from_cartesian(*[np.broadcast_to(np.asarray(a, float), shape) for a in X])
        return evaluate_points(R.C, R.L, th, ph).reshape(shape + (R.N,))

    def grid_values(self, R, axes_points):
        """Values on the tensor grid theta x phi: shape (m_theta, m_phi, N)."""
        return synthesise_grid(R.C, R.L, axes_points[0], axes_points[1])

    def error(self, R):
        return R.err if self.c == 1.0 else math.sqrt(self.c) * R.err      # R.err in L2(sigma); the space's norm

    def breaks(self, R):
        return None

    def merge_breaks(self, items):
        return None

    def is_sparse(self, R):
        return False

    def describe(self, R):
        return dict(kind="spherical harmonics", pieces=1, degree=(R.L,), size=R.size,
                    basis=f"real orthonormal Y_lm, l <= {R.L} ({(R.L + 1) ** 2} coefficients per function)")

    def summary(self, R):
        return f"spherical harmonics: degree {R.L}"
