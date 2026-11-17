"""Radial x spherical-harmonic realisations on balls and shells (plan 009 G1; EXP-030 DESIGN G1).

L2 of a ball or shell (a < r < b, a = 0 on a ball) is the Hilbert tensor product L2((a, b), r^2 dr) (x) L2(S^2), and
L2(S^2) = (+)_l H_l (Parseval), so a function is f(r, xi) = sum_{l <= L} sum_j f_lj(r) Y_lj(xi) with the package's real
orthonormal harmonics (spaces/harmonic.py) and

    <f, g> = sum_lj int_a^b f_lj(r) g_lj(r) r^2 dr                                     (exact on the realisation)

Two realisations:
    RadialHarmonicSeries   dense: radial factors f_lj, piecewise Legendre on common radial breakpoints (interfaces, kernel nodes), for
         every (l, j) with l <= L; coefficients per piece of shape (points, (L + 1)^2, N)
    SeparableRadialHarmonics   separable families k_i(r) Y_(l_i j_i): one radial factor and one harmonic per member (the representers of
         separable functionals, plan 009 G2), the radial factors stored once (col[i] picks member i's), so 8002 data
         kernels with 1240 distinct radial factors cost 1240 radial functions, not 8002 x (L + 1)^2
Inner products are exact: Gauss-Legendre with enough points per piece for the polynomial f_lj g_lj r^2. A product of
two members of different harmonics vanishes without quadrature, so the Gram matrix of an SeparableRadialHarmonics family is assembled per
harmonic (and is block diagonal by construction).

Fitting a function of the Cartesian coordinates (radial_fit.py): the angular degree from the coefficient energy at probe
radii (spherical-harmonic analysis on the Gauss x equispaced grid of spherical.py, exact for the degree of the grid),
then every radial factor adaptively per piece; the error estimate is the discarded energy (angular and radial).
"""
import math

import numpy as np
from numpy.polynomial import legendre as npleg

from ...core.errors import Unsupported
from . import legendre as LG
from ..backends import Backend
from ...spaces.harmonic import values as harmonic_values
from .legendre import EPS
from ...quadrature import gauss
from .radial_fit import resolution_hint as _resolution_hint
from .spherical import _ro

SQ4PI = math.sqrt(4 * math.pi)
MAX_VALUES = 1 << 23            # values per evaluation array


class RadialHarmonicSeries:
    """N functions: radial breaks, per piece Legendre coefficients (d, (L + 1)^2, N), error estimates (N,)."""

    __slots__ = ("L", "breaks", "coefs", "err")
    CAPABILITIES = ("radial_harmonic_coefficients", "box_support")

    def __init__(self, L, breaks, coefs, err):
        self.L, self.breaks = int(L), _ro(np.asarray(breaks, float))
        self.coefs = tuple(_ro(np.asarray(c, float)) for c in coefs)
        self.err = _ro(np.broadcast_to(np.asarray(err, float), (self.coefs[0].shape[2],)).copy())

    @property
    def N(self):
        return self.coefs[0].shape[2]

    @property
    def nlm(self):
        return (self.L + 1) ** 2

    @property
    def size(self):
        return sum(c.size for c in self.coefs)

    # capabilities (registry.CAPABILITIES; box_integrals.py) ---------------------------------------------------
    def box_support(self, H, l, u, rtol=1e-8):
        from ..bounds.box_integrals import radial_support
        return radial_support(self, H, l, u, rtol)


class SeparableRadialHarmonics:
    """N separable functions k_col[i](r) Y_lm[i]: radial factors pl (a legendre.PiecewiseLegendre), the packed harmonic row lm[i] and
    the radial column col[i] of each member, the largest degree L (module docstring)."""

    __slots__ = ("L", "pl", "lm", "col")
    CAPABILITIES = ("radial_harmonic_coefficients", "box_support")

    def __init__(self, pl, lm, col=None, L=None):
        self.pl = pl
        self.lm = _ro(np.asarray(lm, int).ravel().copy())
        self.col = _ro((np.arange(len(self.lm)) if col is None else np.asarray(col, int).ravel()).copy())
        self.L = int(math.isqrt(int(self.lm.max())) if L is None and len(self.lm) else (L or 0))

    @property
    def N(self):
        return len(self.lm)

    @property
    def breaks(self):
        return self.pl.breaks

    @property
    def size(self):
        return self.pl.size + 2 * self.N

    # capabilities (registry.CAPABILITIES; box_integrals.py) ---------------------------------------------------
    def box_support(self, H, l, u, rtol=1e-8):
        from ..bounds.box_integrals import radial_support
        return radial_support(self, H, l, u, rtol)


# ------------------------------------------------------------------------------------------------ helpers
def r2_density(a, b):
    """r^2 on [a, b] as a one-column legendre.PiecewiseLegendre (exact: m^2 + h^2/3, 2 m h, 2 h^2/3 in P_0, P_1, P_2)."""
    m, h = 0.5 * (a + b), 0.5 * (b - a)
    return LG.PiecewiseLegendre([a, b], [np.array([[m * m + h * h / 3.0], [2 * m * h], [2 * h * h / 3.0]])], [0.0])


def _refine3(R, nb):
    """R's coefficients on the finer breaks nb (exact), as a list of (d, nlm, N) arrays."""
    if len(nb) == len(R.breaks) and np.array_equal(nb, R.breaks):
        return list(R.coefs)
    mids = 0.5 * (nb[:-1] + nb[1:])
    parent = np.clip(np.searchsorted(R.breaks, mids, side="right") - 1, 0, len(R.coefs) - 1)
    out = []
    for i, k in enumerate(parent):
        C = R.coefs[k]
        d = C.shape[0]
        out.append(LG.restrict(C.reshape(d, -1), R.breaks[k], R.breaks[k + 1], nb[i], nb[i + 1]).reshape(C.shape))
    return out


def _pad(C, d, nlm):
    if C.shape[0] == d and C.shape[1] == nlm:
        return C
    out = np.zeros((d, nlm, C.shape[2]))
    out[:C.shape[0], :C.shape[1]] = C
    return out


def _common(Rs):
    """(L, breaks, [per realisation: list of (d, nlm, N) per piece]) on common breaks and degree."""
    L = max(R.L for R in Rs)
    nb = LG.union_breaks(*[R.breaks for R in Rs])
    return L, nb, [[_pad(C, C.shape[0], (L + 1) ** 2) for C in _refine3(R, nb)] for R in Rs]


def dense(R):
    """An RadialHarmonicSeries realisation of R (SeparableRadialHarmonics members expanded: memory (L + 1)^2 x N per radial coefficient)."""
    if isinstance(R, RadialHarmonicSeries):
        return R
    return combine(R, np.eye(R.N))


def lincomb(terms):
    terms = [(a, dense(R)) for a, R in terms]
    N = max(R.N for _, R in terms)
    terms = [(a, R if R.N == N else combine(R, np.ones((1, N)))) for a, R in terms]
    L, nb, cs = _common([R for _, R in terms])
    scale = max(abs(a) * max(float(np.abs(c).max()) for c in R.coefs) for a, R in terms) or 1.0
    coefs = []
    for i in range(len(nb) - 1):
        d = max(c[i].shape[0] for c in cs)
        acc = sum(a * _pad(c[i], d, (L + 1) ** 2) for (a, _), c in zip(terms, cs))
        rows = np.nonzero((np.abs(acc) > 2 * EPS * scale).any(axis=(1, 2)))[0]
        coefs.append(acc[:rows[-1] + 1] if len(rows) else acc[:1] * 0.0)
    return RadialHarmonicSeries(L, nb, coefs, sum(abs(a) * R.err for a, R in terms))


def combine(R, A):
    """Members R @ A (A of shape (R.N, M)): RadialHarmonicSeries for an RadialHarmonicSeries, RadialHarmonicSeries for an SeparableRadialHarmonics (expanded per harmonic)."""
    A = np.asarray(A, float)
    if isinstance(R, RadialHarmonicSeries):
        return RadialHarmonicSeries(R.L, R.breaks, [np.tensordot(C, A, axes=(2, 0)) for C in R.coefs], np.abs(A).T @ R.err)
    nlm, M = (R.L + 1) ** 2, A.shape[1]
    out = [np.zeros((C.shape[0], nlm, M)) for C in R.pl.coefs]
    for u in np.unique(R.lm):
        rows = np.flatnonzero(R.lm == u)
        for p, C in enumerate(R.pl.coefs):
            out[p][:, u, :] = C[:, R.col[rows]] @ A[rows]
    return RadialHarmonicSeries(R.L, R.pl.breaks, out, np.abs(A).T @ sr_error(R))


def take(R, idx):
    sel = np.atleast_1d(np.asarray(idx, int))
    if isinstance(R, SeparableRadialHarmonics):
        return SeparableRadialHarmonics(R.pl, R.lm[sel], R.col[sel], R.L)
    return RadialHarmonicSeries(R.L, R.breaks, [C[:, :, sel] for C in R.coefs], R.err[sel])


def hstack(Rs):
    if all(isinstance(R, SeparableRadialHarmonics) for R in Rs):
        pl = LG.hstack([R.pl for R in Rs])
        off = np.cumsum([0] + [R.pl.N for R in Rs])
        return SeparableRadialHarmonics(pl, np.concatenate([R.lm for R in Rs]), np.concatenate([R.col + o for R, o in zip(Rs, off)]),
                  max(R.L for R in Rs))
    Rs = [dense(R) for R in Rs]
    L, nb, cs = _common(Rs)
    coefs = []
    for i in range(len(nb) - 1):
        d = max(c[i].shape[0] for c in cs)
        coefs.append(np.concatenate([_pad(c[i], d, (L + 1) ** 2) for c in cs], axis=2))
    return RadialHarmonicSeries(L, nb, coefs, np.concatenate([R.err for R in Rs]))


def sr_error(R):
    """Estimated errors of SeparableRadialHarmonics members in the r^2 dr norm (the radial estimates are in L2(dr); r^2 <= b^2)."""
    return R.pl.err[R.col] * abs(R.pl.breaks[-1])


# ------------------------------------------------------------------------------------------------ measure
def _nodes(a, b, n):
    t, w = gauss(n)
    r = 0.5 * (a + b) + 0.5 * (b - a) * t
    return t, 0.5 * (b - a) * w * r * r


def gram(A, B):
    """<A_i, B_j> = sum_lj int A_lj B_lj r^2 dr, exact (module docstring)."""
    if isinstance(A, SeparableRadialHarmonics) and isinstance(B, SeparableRadialHarmonics):
        return _gram_sr(A, B)
    if isinstance(A, SeparableRadialHarmonics) or isinstance(B, SeparableRadialHarmonics):
        from .radial_direct import gram_sr_rh
        return gram_sr_rh(A, B) if isinstance(A, SeparableRadialHarmonics) else gram_sr_rh(B, A).T
    L, nb, (CA, CB) = _common([A, B])
    out = np.zeros((A.N, B.N))
    for i in range(len(nb) - 1):
        n = (CA[i].shape[0] + CB[i].shape[0]) // 2 + 2
        t, w = _nodes(nb[i], nb[i + 1], n)
        V = npleg.legvander(t, max(CA[i].shape[0], CB[i].shape[0]) - 1)
        VA = np.tensordot(V[:, :CA[i].shape[0]], CA[i], axes=(1, 0))            # (q, nlm, NA)
        VB = np.tensordot(V[:, :CB[i].shape[0]], CB[i], axes=(1, 0))
        out += (VA * w[:, None, None]).reshape(-1, A.N).T @ VB.reshape(-1, B.N)
    return out


def rgram(P, Q, diag=False):
    """int p_i q_j r^2 dr for the columns of two legendre.PiecewiseLegendre on one interval, with r^2 evaluated at the Gauss nodes
    (exact, and with relative accuracy near r = 0, where a refined polynomial density would only have absolute
    accuracy). diag: only int p_i q_i r^2 dr (P and Q with equal columns)."""
    nb = LG.union_breaks(P.breaks, Q.breaks)
    CP, CQ = LG.refine(P, nb), LG.refine(Q, nb)
    out = np.zeros(P.N) if diag else np.zeros((P.N, Q.N))
    for i in range(len(nb) - 1):
        t, w = _nodes(nb[i], nb[i + 1], (CP[i].shape[0] + CQ[i].shape[0]) // 2 + 2)
        VP, VQ = LG.polyval(t, CP[i]), LG.polyval(t, CQ[i])
        out += np.einsum("q,qn,qn->n", w, VP, VQ) if diag else (VP * w[:, None]).T @ VQ
    return out


def _gram_sr(A, B):
    out = np.zeros((A.N, B.N))
    for u in np.intersect1d(np.unique(A.lm), np.unique(B.lm)):
        ia, ib = np.flatnonzero(A.lm == u), np.flatnonzero(B.lm == u)
        ca, inva = np.unique(A.col[ia], return_inverse=True)
        cb, invb = np.unique(B.col[ib], return_inverse=True)
        out[np.ix_(ia, ib)] = rgram(A.pl.take(ca), B.pl.take(cb))[np.ix_(inva, invb)]
    return out


def radial_gram(A, cols_a, B, cols_b):
    """int k_a k_b r^2 dr for radial columns of two SeparableRadialHarmonics realisations (exact)."""
    return rgram(A.pl.take(np.asarray(cols_a, int)), B.pl.take(np.asarray(cols_b, int)))


def sqnorms(A):
    if isinstance(A, SeparableRadialHarmonics):
        cu, inv = np.unique(A.col, return_inverse=True)
        P = A.pl.take(cu)
        return rgram(P, P, diag=True)[inv]
    out = np.zeros(A.N)
    for i, C in enumerate(A.coefs):
        t, w = _nodes(A.breaks[i], A.breaks[i + 1], C.shape[0] + 2)
        V = np.tensordot(npleg.legvander(t, C.shape[0] - 1), C, axes=(1, 0))
        out += np.einsum("q,qln->n", w, V * V)
    return out


def integral(R):
    """int f dV = sqrt(4 pi) int f_00 r^2 dr."""
    R = dense(R) if isinstance(R, SeparableRadialHarmonics) else R
    out = np.zeros(R.N)
    for i, C in enumerate(R.coefs):
        t, w = _nodes(R.breaks[i], R.breaks[i + 1], C.shape[0] + 2)
        out += w @ (npleg.legvander(t, C.shape[0] - 1) @ C[:, 0, :])
    return SQ4PI * out


def evaluate(R, r, theta, phi):
    """Values (m, N) at points given in (r, theta, phi) (NaN where r is NaN)."""
    r, theta, phi = (np.asarray(x, float).ravel() for x in (r, theta, phi))
    ok = np.isfinite(r) & np.isfinite(theta) & np.isfinite(phi)
    out = np.full((r.size, R.N), np.nan)
    idx = np.flatnonzero(ok)
    nlm = (R.L + 1) ** 2
    step = max(1, MAX_VALUES // max(1, nlm * (R.N if isinstance(R, RadialHarmonicSeries) else 1)))
    for s in range(0, len(idx), step):
        p = idx[s:s + step]
        Y = harmonic_values(R.L, None, (theta[p], phi[p]))                         # (m, nlm)
        if isinstance(R, SeparableRadialHarmonics):
            rad = LG.evaluate(R.pl, r[p])                                          # (m, C)
            out[p] = rad[:, R.col] * Y[:, R.lm]
        else:
            pl = LG.PiecewiseLegendre(R.breaks, [C.reshape(C.shape[0], -1) for C in R.coefs], 0.0)
            rad = LG.evaluate(pl, r[p]).reshape(len(p), nlm, R.N)
            out[p] = np.einsum("pl,pln->pn", Y, rad)
    return out


def _sr_scaled(R, f):
    """The SeparableRadialHarmonics family f_i k_col[i] Y_lm[i] (one radial column per member)."""
    S = np.zeros((R.pl.N, R.N))
    S[R.col, np.arange(R.N)] = f
    return SeparableRadialHarmonics(LG.combine(R.pl, S), R.lm, None, R.L)


def band(R, keep):
    """R with every harmonic of degree l outside `keep` (a boolean array over l = 0, 1, ...) set to zero."""
    from ...spaces.harmonic import rows
    deg = rows(R.L)[0]
    mask = (deg < len(keep)) & np.asarray(keep, bool)[np.minimum(deg, len(keep) - 1)]
    if isinstance(R, SeparableRadialHarmonics):
        live = mask[R.lm]
        return R if live.all() else _sr_scaled(R, live.astype(float))
    return RadialHarmonicSeries(R.L, R.breaks, [C * mask[None, :, None] for C in R.coefs], R.err)


def scale_degrees(R, f):
    """R with the degree-l part multiplied by f[l] (a function of the angular Laplacian, operators/band.py)."""
    from ...spaces.harmonic import rows
    deg = rows(R.L)[0]
    fl = np.asarray(f, float)[deg]
    if isinstance(R, SeparableRadialHarmonics):
        return _sr_scaled(R, fl[R.lm])
    return RadialHarmonicSeries(R.L, R.breaks, [C * fl[None, :, None] for C in R.coefs], R.err * np.max(np.abs(fl)))


# ------------------------------------------------------------------------------------------------ backend
class RadialHarmonicsBackend(Backend):
    """Radial x spherical-harmonic realisation of L2 of a ball or shell (module docstring), with Lebesgue measure dx,
    c dx for a constant density c > 0, or rho(r) dx for a radial density (plan 064): <f, g> = c sum_lj int f_lj g_lj
    rho r^2 dr. rho is realised once as a piecewise Legendre series on the radial interval (self.rho, with its fit's
    error; None for constant densities), so the pairings stay exact for the realisation; self.c is the constant."""

    CAPABILITIES = ("radial_harmonic_coefficients",)
    dim = 3
    realisation_name = "radial_harmonics"
    harmonic = "radial"

    def __init__(self, measure, tol):
        from ...measures import constant_density
        self.measure, self.tol, self.domain = measure, tol, measure.set
        self.a, self.b = (float(x) for x in self.domain.reference[0])
        self.jac_max = self.b ** 2
        c = constant_density(measure)
        self.c, self.rho = (1.0, _radial_density(measure, self.a, self.b, tol)) if c is None else (c, None)

    # construction ------------------------------------------------------------------------------
    def probe(self):
        r = self.a + (self.b - self.a) * np.array([0.3, 0.5, 0.7])
        return tuple(np.asarray(c, float) for c in self.domain.to_cartesian(r, np.array([0.7, 1.4, 2.3]),
                                                                             np.array([0.5, 2.0, 4.5])))

    def _internal(self, fun):
        to_cart = self.domain.to_cartesian
        return lambda r, theta, phi: fun(*to_cart(r, theta, phi))

    def _radial_breaks(self, breaks):
        if breaks is None:
            return None
        if isinstance(breaks, tuple) and len(breaks) == 3 and not np.isscalar(breaks[0]):
            breaks = breaks[0]                    # per-axis breaks (r, theta, phi): the radial ones
        return None if breaks is None else np.asarray(breaks, float).ravel()

    def fit(self, fun, N=1, breaks=None):
        from .radial_fit import fit_rh, support_check
        R, u = fit_rh(self._internal(fun), self.a, self.b, N, self.tol, self._radial_breaks(breaks),
                      lmin=getattr(breaks, "degree", None))
        box = getattr(breaks, "box", None)              # a support's box (bank functions): checked there (plan 035)
        return (R, u) if box is None else support_check(self, fun, R, u, box)

    resolution_hint = _resolution_hint                   # a declared resolution's angular degree (core/fits.py)

    def restricted(self, R, target):
        if tuple(self.domain.centre) != tuple(target.domain.centre):
            return None
        if isinstance(R, SeparableRadialHarmonics):
            out = LG.cut(R.pl.coefs, R.pl.breaks, target.a, target.b)
            return None if out is None else SeparableRadialHarmonics(LG.PiecewiseLegendre(out[0], out[1], R.pl.err), R.lm, R.col, R.L)
        out = LG.cut(R.coefs, R.breaks, target.a, target.b)
        return None if out is None else RadialHarmonicSeries(R.L, out[0], out[1], R.err)

    def glued(self, parts):
        from .cuts import radial_harmonic
        return radial_harmonic(self, parts)

    def piecewise_linear(self, radii, coefficients):
        from .radial_direct import piecewise_linear
        return piecewise_linear(self.a, self.b, radii, coefficients)

    def fit_family(self, fun_idx, N, breaks=None, sparse=None, name="F"):
        import warnings
        out = []
        for s in range(0, N, 32):
            idx = np.arange(s, min(N, s + 32))
            R, u = self.fit(lambda *X, idx=idx: fun_idx(*X, idx), len(idx), breaks)
            if u:
                warnings.warn(f"family {name}: {u} unresolved radial piece(s) in columns {s}..{idx[-1]}")
            out.append(R)
        return out[0] if len(out) == 1 else hstack(out)

    def constant(self, values):
        v = np.atleast_1d(np.asarray(values, float))
        return RadialHarmonicSeries(0, [self.a, self.b], [SQ4PI * v.reshape(1, 1, -1)], np.zeros(len(v)))

    def coordinate(self, k):
        R, _ = self.fit(lambda *X: np.asarray(X[k], float))
        return R

    def separable(self, radial, lm, col=None, L=None):
        """The SeparableRadialHarmonics family k_col[i](r) Y_lm[i] from radial factors (a legendre.PiecewiseLegendre on [a, b]); lm packed rows."""
        if abs(radial.breaks[0] - self.a) > 1e-12 * self.b or abs(radial.breaks[-1] - self.b) > 1e-12 * self.b:
            raise ValueError(f"radial factors on [{radial.breaks[0]:g}, {radial.breaks[-1]:g}], not on the radial "
                             f"interval [{self.a:g}, {self.b:g}]")
        return SeparableRadialHarmonics(radial, lm, col, L)

    def radial_measure(self):
        """r^2 dr on the radial interval: the measure of the radial factors; c r^2 dr or rho(r) r^2 dr with a density
        (plan 064)."""
        from ...measures import DensityMeasure, LebesgueMeasure
        from ...sets import Interval
        if self.rho is None and self.c == 1.0:
            return DensityMeasure(LebesgueMeasure(Interval(self.a, self.b)), _r2, name="r^2")
        if self.rho is None:
            return DensityMeasure(LebesgueMeasure(Interval(self.a, self.b)), lambda r, c=self.c: c * _r2(r),
                                  name=f"{self.c:.6g} r^2")
        rho = self.measure.weight
        return DensityMeasure(LebesgueMeasure(Interval(self.a, self.b)), lambda r: rho.radial(r) * _r2(r),
                              name=f"{self.measure.name}(r) r^2")

    def radial_lebesgue(self):
        """r^2 dr on the radial interval: the radial part of the domain's own (Lebesgue) measure, against which kernel
        functionals integrate whatever the space's density (plan 066)."""
        from ...measures import DensityMeasure, LebesgueMeasure
        from ...sets import Interval
        return DensityMeasure(LebesgueMeasure(Interval(self.a, self.b)), _r2, name="r^2")

    def weighted(self, R):
        """rho R (exact for the realised rho; R itself without a radial density): the factor the pairings take."""
        return R if self.rho is None else times_radial(R, self.rho)

    def density_sup(self):
        """A bound of c rho on the radial interval (c without a radial density)."""
        return self.c * (1.0 if self.rho is None else float(self.rho.sup_bound()[0]))

    def radial_gram(self, A, cols_a, B, cols_b):
        """c int k_a k_b rho r^2 dr for radial columns of two SeparableRadialHarmonics realisations (exact for the realisation)."""
        P, Q = A.pl.take(np.asarray(cols_a, int)), B.pl.take(np.asarray(cols_b, int))
        return self.c * rgram(P, Q if self.rho is None else LG.multiply(Q, self.rho))

    # algebra -----------------------------------------------------------------------------------
    def lincomb(self, terms):
        return lincomb(terms)

    def combine(self, R, A):
        return combine(R, A)

    def take(self, R, idx):
        return take(R, idx)

    def hstack(self, Rs):
        return hstack(Rs)

    def colsum(self, R):
        return combine(R, np.ones((R.N, 1)))

    def multiply(self, A, B):
        A, B = dense(A), dense(B)
        if B.L == 0 or A.L == 0:                 # a radial factor: exact, radial factor by radial factor
            F, g = (A, B) if B.L == 0 else (B, A)
            if g.N != 1 and F.N != g.N:
                raise ValueError("products of families need equal sizes")
            out = []
            nb = LG.union_breaks(F.breaks, g.breaks)
            for CF, Cg in zip(_refine3(F, nb), _refine3(g, nb)):
                n = CF.shape[0] + Cg.shape[0] - 1
                t, _ = gauss(n)
                vF = np.tensordot(npleg.legvander(t, CF.shape[0] - 1), CF, axes=(1, 0))
                vg = (npleg.legvander(t, Cg.shape[0] - 1) @ Cg[:, 0, :])[:, None, :] / SQ4PI
                out.append(np.tensordot(LG.analysis(n), vF * vg, axes=(1, 0)))
            gmax = max(float(np.abs(C).sum(axis=0).max()) for C in g.coefs) / SQ4PI
            return RadialHarmonicSeries(F.L, nb, out, gmax * F.err + g.err)
        R, u = self.fit(lambda *X: self.evaluate(A, *X) * self.evaluate(B, *X), max(A.N, B.N),
                        LG.union_breaks(A.breaks, B.breaks))
        return R

    # measure -----------------------------------------------------------------------------------
    def gram(self, A, B):
        G = gram(A, self.weighted(B))
        return G if self.c == 1.0 else self.c * G

    def sqnorms(self, A):
        q = sqnorms(A) if self.rho is None else _cross_sqnorms(A, self.weighted(A))
        return q if self.c == 1.0 else self.c * q

    def integral(self, R, over=None):
        if over is not None:
            raise Unsupported("integrals over parts of a ball are not implemented in this realisation")
        v = integral(self.weighted(R))
        return v if self.c == 1.0 else self.c * v

    # values and description --------------------------------------------------------------------
    def evaluate(self, R, *X):
        shape = np.broadcast(*X).shape
        r, th, ph = self.domain.from_cartesian(*[np.broadcast_to(np.asarray(a, float), shape) for a in X])
        return evaluate(R, r, th, ph).reshape(shape + (R.N,))

    def error(self, R):
        e = sr_error(R) if isinstance(R, SeparableRadialHarmonics) else R.err                  # in L2(r^2 dr dxi)
        return e if self.rho is None and self.c == 1.0 else math.sqrt(self.density_sup()) * e

    def breaks(self, R):
        return np.asarray(R.breaks)

    def merge_breaks(self, items):
        items = [np.asarray(b, float) for b in items if b is not None]
        return np.unique(np.concatenate(items)) if items else None

    def describe(self, R):
        if isinstance(R, SeparableRadialHarmonics):
            return dict(kind="separable radial x harmonic", pieces=R.pl.npieces, degree=(R.L,), size=R.size,
                        basis=f"{R.pl.N} radial factors, one harmonic per member (l <= {R.L})")
        return dict(kind="radial x spherical harmonics", pieces=len(R.coefs),
                    degree=(max(C.shape[0] for C in R.coefs) - 1, R.L), size=R.size,
                    basis=f"radial piecewise Legendre ({len(R.coefs)} pieces) x real orthonormal Y_lj, l <= {R.L}")

    def summary(self, R):
        d = self.describe(R)
        return f"{d['kind']}: {d['pieces']} radial piece(s), degree {d['degree']}"


def _r2(r):
    r = np.asarray(r, float)
    return r * r


def _radial_density(measure, a, b, tol):
    """The radial density of measure (measures/weights.RadialWeight) realised on [a, b] as a one-column legendre.PiecewiseLegendre."""
    from ...core.errors import RealisationError
    wt = measure.weight
    pl, unresolved = LG.fit(lambda r: wt.radial(r), a, b, tol=max(tol, 1e-14))
    if unresolved or not np.all(np.isfinite(pl.err)):
        raise RealisationError(f"could not resolve the radial density {measure.name} on [{a:g}, {b:g}]")
    return pl


def times_radial(R, rho):
    """rho R for a radial factor rho (a one-column legendre.PiecewiseLegendre on the radial interval): exact on the polynomials."""
    if isinstance(R, SeparableRadialHarmonics):
        return SeparableRadialHarmonics(LG.multiply(R.pl, rho), R.lm, R.col, R.L)
    nb = LG.union_breaks(R.breaks, rho.breaks)
    CR, Cr = _refine3(R, nb), LG.refine(rho, nb)
    out = []
    for C, c in zip(CR, Cr):
        n = C.shape[0] + c.shape[0] - 1
        t, _ = gauss(n)
        v = np.tensordot(npleg.legvander(t, C.shape[0] - 1), C, axes=(1, 0)) * LG.polyval(t, c)[:, :1, None]
        out.append(np.tensordot(LG.analysis(n), v, axes=(1, 0)))
    return RadialHarmonicSeries(R.L, nb, out, float(rho.sup_bound()[0]) * R.err)      # rho's own fit: to tol, as an interval density


def _cross_sqnorms(A, B):
    """<A_j, B_j> per member for two dense RadialHarmonicSeries (or SeparableRadialHarmonics) realisations with the same members (exact)."""
    if isinstance(A, SeparableRadialHarmonics) and isinstance(B, SeparableRadialHarmonics):
        cu, inv = np.unique(A.col, return_inverse=True)
        return rgram(A.pl.take(cu), B.pl.take(cu), diag=True)[inv]
    A, B = dense(A), dense(B)
    L, nb, (CA, CB) = _common([A, B])
    out = np.zeros(A.N)
    for i in range(len(nb) - 1):
        n = (CA[i].shape[0] + CB[i].shape[0]) // 2 + 2
        t, w = _nodes(nb[i], nb[i + 1], n)
        VA = np.tensordot(npleg.legvander(t, CA[i].shape[0] - 1), CA[i], axes=(1, 0))
        VB = np.tensordot(npleg.legvander(t, CB[i].shape[0] - 1), CB[i], axes=(1, 0))
        out += np.einsum("q,qln,qln->n", w, VA, VB)
    return out
