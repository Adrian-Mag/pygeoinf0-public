"""The inner product of formulas in the form space of order 1 on a box (plan 025 P4): the box counterpart of
green.IntervalForm, for SobolevForm(B, 1, l).space() = Sobolev(A_a, 1) (registry: the Spectral construction of a form
of order 1 at s = 1 on a box, Neumann or Dirichlet faces):

    <f, g> = a(f, g) = int f g + l^2 sum_k int d_k f d_k g

from exact derivatives of the realisations (tensor.TensorSeries: per piece, numpy legder with the piece's scale; support-local
patches (realisations/series/local.py, PS): per patch, with the derivative of a translate type's shared coefficient array computed once,
so translates keep one Gram block per offset). This is the norm of D(A_a^(1/2)) for every element of the form space
(Kato's second representation theorem), the classic H^1 (FEM) inner product; with the space's kernel (reflection images
of the Matern kernel, the Green function of A_a) it satisfies <k_mu, f> = int f dmu (tests: Galerkin representers
converge to the kernel's energy). Dirichlet faces: every element vanishes there (check). Like IntervalForm, no error
is propagated into the inner products (the fits are converged to their tolerance; derivatives of a polynomial piece are
exact).

From box_integrals.py (merged here by plan 063):
int_S f dx of tensor realisations over a coordinate box S (EXP-032 finding 8).

On domains whose natural coordinates are the Cartesian ones with Jacobian 1 (boxes, flat tori), the integral of a
tensor realisation sum C_{a b c} phi_a(x) psi_b(y) chi_c(z) over a box S = [a_1, b_1] x ... is the contraction of C with
one row per axis, the integrals of the axis basis over [a_k, b_k]: in closed form for a Fourier basis, by Gauss-Legendre
with n points on the part of each piece in [a_k, b_k] for a piecewise Legendre basis of n points (exact: degree n - 1).
It is exact for the realisation (the realisation's own error is reported with it elsewhere), and costs one pass over
the coefficients, where evaluating the realisation at a tensor rule of q^3 points inside S costs q^3 times the
coefficients of a cell of the realisation: on a torus a single cell, the whole Fourier series (2 s against 0.06 s per
pixel cell on the box with 129^3 coefficients, measured by EXP-032).
"""


import math

import numpy as np
from numpy.polynomial import legendre as npleg

from ...core.errors import Unsupported, ViolatesBoundaryCondition
from ..series import tensor as T
from ...quadrature import gauss

CONDITION_TOL = 1e-8            # a formula violates a Dirichlet face when |f| there exceeds this times its maximum


def derivative_tr(R, k):
    """d/dxi_k of a tensor realisation (exact on each piece), with the same bases."""
    b = R.bases[k]
    if isinstance(b, T.FourierBasis):
        raise Unsupported("derivatives of Fourier axes are not needed on boxes (no periodic faces)")
    C = np.moveaxis(R.C, k, 0)
    rest = C.shape[1:]
    C = C.reshape((b.P, b.n) + rest)
    h = np.diff(np.asarray(b.breaks))
    out = np.zeros_like(C)
    if b.n > 1:
        for p in range(b.P):
            out[p, :b.n - 1] = npleg.legder(C[p], 1, scl=2.0 / h[p], axis=0)
    return T.TensorSeries(R.bases, np.moveaxis(out.reshape((b.S,) + rest), 0, k), np.zeros(R.N))


def derivative(R, k, memo=None):
    """d/dxi_k of a TensorSeries or a PS (patches share the derivative of a shared coefficient array)."""
    from ..series.local import PS
    if not isinstance(R, PS):
        return derivative_tr(R, k)
    memo = {} if memo is None else memo
    patches = []
    for c, cols, Q in R.patches:
        key = (id(Q.C), k, tuple(np.round(np.diff(np.asarray(Q.bases[k].breaks)), 15)))
        hit = memo.get(key)
        if hit is None:
            hit = memo[key] = (Q.C, derivative_tr(Q, k).C)             # keep Q.C alive: the id stays its own
        patches.append((c, cols, T.TensorSeries(Q.bases, hit[1], np.zeros(Q.N))))
    base = None if R.base is None else derivative_tr(R.base, k)
    return PS(R.N, base, patches) if patches else base


class BoxForm:
    """a(f, g) = int f g + l^2 int grad f . grad g on a box, from the L2 backend's Grams of the realisations and of
    their derivatives; dirichlet: the faces ("x-", "y+", ...) where every element vanishes."""

    def __init__(self, base, length_scale, faces, dirichlet=()):
        self.base, self.l, self.faces = base, float(length_scale), tuple(faces)
        self.dirichlet = tuple(dirichlet)
        self._memo = {}

    def check(self, fx):
        if not self.dirichlet:
            return
        ref = self.base.domain.reference
        grids = [np.linspace(lo, hi, 33) for lo, hi in ref]
        scale = np.abs(self.base.grid_values(fx, grids)).reshape(-1, fx.N).max(axis=0)
        for face in self.dirichlet:
            k, side = self.faces.index(face) // 2, face[-1]
            g = list(grids)
            g[k] = np.array([ref[k][0] if side == "-" else ref[k][1]])
            v = np.abs(self.base.grid_values(fx, g)).reshape(-1, fx.N).max(axis=0)
            if np.any(v > CONDITION_TOL * np.maximum(scale, 1e-300)):
                raise ViolatesBoundaryCondition(
                    f"a formula is {float(v.max()):.3g} on the Dirichlet face {face} (every element of this space "
                    "vanishes there), so the function is not an element (choose a space without the condition, or a "
                    "function that vanishes on the face)")

    def gram(self, F, G):
        self.check(F)
        self.check(G)
        B = self.base
        out = B.gram(F, G)                                     # dense, or scipy.sparse for many patches
        for k in range(len(B.domain.reference)):
            dF = derivative(F, k, self._memo)
            dG = dF if G is F else derivative(G, k, self._memo)
            out = out + self.l ** 2 * B.gram(dF, dG)
        return out.tocsr() if hasattr(out, "tocsr") else np.asarray(out, float)


# ---------------------------------------------------------------- from box_integrals.py (plan 063)
def axis_row(b, lo, hi):
    """(b.S,): int_lo^hi of each function of the axis basis b (lo, hi inside the axis)."""
    from ..series.tensor import FourierBasis
    if isinstance(b, FourierBasis):
        out = np.zeros(b.n)
        out[0] = hi - lo
        L = b.hi - b.lo
        k = np.arange(1, b.K + 1)
        t0, t1 = (2 * math.pi * k * (x - b.lo) / L for x in (lo, hi))
        out[1::2] = L / (2 * math.pi * k) * (np.sin(t1) - np.sin(t0))
        out[2::2] = -L / (2 * math.pi * k) * (np.cos(t1) - np.cos(t0))
        return out
    br = np.asarray(b.breaks)
    out = np.zeros((b.P, b.n))
    g, w = gauss(b.n)
    for p in range(b.P):
        x0, x1 = br[p], br[p + 1]
        a, c = max(lo, x0), min(hi, x1)
        if c <= a:
            continue
        t = 0.5 * (a + c) + 0.5 * (c - a) * g                     # Gauss points of the part, local coordinates
        out[p] = (0.5 * (c - a) * w) @ npleg.legvander((2 * t - x0 - x1) / (x1 - x0), b.n - 1)
    return out.ravel()


def _cartesian(domain, jac):
    """Whether the domain's natural coordinates are its Cartesian ones, with Jacobian 1 on every axis."""
    if any(j != "1" for j in jac) or domain.cartesian_dim != domain.dim:
        return False
    xi = [np.array([lo + f * (hi - lo) for f in (0.3, 0.5, 0.7)]) for lo, hi in domain.reference]
    return all(np.array_equal(np.asarray(c, float), x) for c, x in zip(domain.to_cartesian(*xi), xi))


def box_integral(R, S, domain, jac):
    """(N,): int_S of the N functions of the tensor realisation R, for a coordinate box S (a set with .sides) of a
    domain with Cartesian natural coordinates; None where this does not apply (the caller integrates by its rule)."""
    sides, reference = getattr(S, "sides", None), domain.reference
    if sides is None or len(sides) != len(reference) or len(R.bases) != len(sides) or not _cartesian(domain, jac):
        return None
    for s, (lo, hi) in zip(sides, reference):
        span = hi - lo
        if s.a < lo - 1e-12 * span or s.b > hi + 1e-12 * span:
            return None
    v = R.C
    for b, s, (lo, hi) in zip(R.bases, sides, reference):
        v = np.tensordot(axis_row(b, max(s.a, lo), min(s.b, hi)), v, axes=(0, 0))
    return np.asarray(v, float)
