"""Backends: the numerical realisation of elements. The contract is the abstract class Backend below.

A function space owns one backend (chosen by realisations.choose). Elements and operators never touch coefficient
arrays; they ask the backend. Every backend method is exact on the realised objects except fit(), which approximates
a function by a realisation and returns an error estimate. A realisation R always holds N columns (N functions).

IntervalBackend (here) realises functions on an interval; TensorBackend (tensor.py) on boxes, disks, cylinders,
balls, circles, tori, paths and patches; SphereBackend (series/spherical.py) on spheres; for Sobolev spaces RKHSBackend
(rkhs/backend.py: representers of measures with Matern energies, plus formulas) and SphereSobolevBackend
(series/sphere_sobolev.py).
"""
import abc
import math
import warnings

import numpy as np

from ..core.errors import NormUnavailable, Unsupported
from ..core.sampling import localised
from ..quadrature import gauss
from .series import legendre as L
from .series import sparse as S


class NaturalBreaks(tuple):
    """Breaks per natural axis (a tuple, as every backend takes them) with a hint for spectral fits: degree, the least
    angular degree whose sampling grid resolves the function's support (a bump of angular half-width delta seen from
    the centre of a ball or shell: about 2 pi / delta; BankFunction.fit_breaks, plan 035 P2), and box, a natural-coordinate
box holding the support (a posteriori checks of spectral fits there: radial_fit.support_error)."""

    def __new__(cls, axes, degree=None, box=None):
        out = super().__new__(cls, axes)
        out.degree, out.box = degree, box          # box: ((lo, hi) per natural axis) holding the support (phi may wrap)
        return out


class Backend(abc.ABC):
    """The contract between a function space and the numbers that realise its elements.

    Required (every backend implements them):
        fit(fun, N, breaks) -> (R, unresolved)   realise N functions of the Cartesian coordinates
        lincomb([(a, R), ...]), combine(R, A), take(R, idx), hstack([R, ...])     exact linear algebra
        gram(A, B) -> (A.N, B.N)                 inner products in the norm of the space
        evaluate(R, *X) -> (..., N)              canonical values at Cartesian points (NaN outside the domain)
        error(R) -> (N,)                         estimated distance to the realised functions, in the space's norm
        describe(R) -> dict, summary(R) -> str
    Derived here from those (a backend may override them for speed):
        fit_family, constant, coordinate, colsum, sqnorms, grid_values, breaks, merge_breaks, is_sparse
    Optional (raise Unsupported unless implemented; ask supports(name)):
        multiply, integral, lp_norm, measure_of_cell, product_family, to_tensor, legendre_decay_generators, riesz, to_l2
    Flags: sparse_families (fit_family may return sparse families), errors_known (False when error() may be NaN,
    meaning unknown in this norm, e.g. formulas in a restriction space)
    Every backend names its realisation (realisation_name, the value of the realisation= argument).
    """
    dim = None
    realisation_name = None
    sparse_families = False
    errors_known = True

    # required ------------------------------------------------------------------------------
    @abc.abstractmethod
    def fit(self, fun, N=1, breaks=None):
        ...

    @abc.abstractmethod
    def lincomb(self, terms):
        ...

    @abc.abstractmethod
    def combine(self, R, A):
        ...

    @abc.abstractmethod
    def take(self, R, idx):
        ...

    @abc.abstractmethod
    def hstack(self, Rs):
        ...

    @abc.abstractmethod
    def gram(self, A, B):
        ...

    @abc.abstractmethod
    def evaluate(self, R, *X):
        ...

    @abc.abstractmethod
    def error(self, R):
        ...

    @abc.abstractmethod
    def describe(self, R):
        ...

    @abc.abstractmethod
    def summary(self, R):
        ...

    # derived -------------------------------------------------------------------------------
    def fit_family(self, fun_idx, N, breaks=None, sparse=None, name="F"):
        R, u = self.fit(lambda *X: fun_idx(*X, np.arange(N)), N, breaks=breaks)
        if u:
            warnings.warn(f"family {name}: {u} unresolved piece(s)")
        return R

    def constant(self, values):
        values = np.atleast_1d(np.asarray(values, float))
        R, _ = self.fit(lambda *X: np.broadcast_to(values, np.broadcast(*X).shape + values.shape), len(values))
        return R

    def coordinate(self, k):
        R, _ = self.fit(lambda *X: np.asarray(X[k], float))
        return R

    def colsum(self, R):
        return self.combine(R, np.ones((R.N, 1)))

    def sqnorms(self, A):
        return np.diag(self.gram(A, A)).copy()

    def sqnorms_or(self, A, unavailable):
        """sqnorms(A), or `unavailable` for every member where the norm cannot be computed (NormUnavailable): the one
        place where that refusal is turned into a value (warnings and error bounds use it; norms do not)."""
        try:
            return self.sqnorms(A)
        except NormUnavailable:
            return np.full(A.N, float(unavailable))

    def grid_values(self, R, axes_points):
        """Values on the tensor grid of natural coordinates axes_points[0] x ...: shape (m_1, ..., m_d, N)."""
        D = self.domain
        G = np.meshgrid(*[np.asarray(a, float) for a in axes_points], indexing="ij")
        X = D.to_cartesian(*G) if hasattr(D, "to_cartesian") else tuple(G)
        return self.evaluate(R, *X)

    def breaks(self, R):
        return None

    def merge_breaks(self, items):
        return None

    def is_sparse(self, R):
        return False

    # optional ------------------------------------------------------------------------------
    def same_realisation(self, other):
        """Whether other realises its space the same way (the condition for Backend.restricted)."""
        return self.realisation_name == other.realisation_name

    def restricted(self, R, target):
        """R restricted exactly to target's smaller domain, a realisation of target, or None when this realisation cannot
        be cut there (the caller then evaluates and fits). R's error estimate is carried: restriction does not raise a norm.
        Here a tensor realisation, for every backend that holds one; other realisations override it or stay None."""
        from .series.cuts import restricted
        return restricted(self, R, target)

    def glued(self, parts):
        """The piecewise realisation on this domain from parts [(backend of a piece, realisation)], pieces that do not overlap:
        their coefficients as they are, zero elsewhere, a jump kept at a seam. None when this realisation cannot hold them that
        way (the caller then fits). Here a tensor realisation, as restricted."""
        from .series.cuts import glued
        return glued(self, parts)

    def _unsupported(self, op):
        raise Unsupported(f"{type(self).__name__} does not implement {op}")

    def multiply(self, A, B):
        self._unsupported("pointwise products")

    def integral(self, R, over=None):
        self._unsupported("integrals")

    def lp_norm(self, R, p):
        self._unsupported("L^p norms")

    def measure_of_cell(self, c, d):
        self._unsupported("measures of cells")

    def product_family(self, factors, params, name="K"):
        self._unsupported("product families (they are for 2-D and 3-D domains; on an interval use M.family)")

    def to_tensor(self, R):
        self._unsupported("conversion to a tensor realisation")

    def legendre_decay_generators(self, decay, modes):
        self._unsupported("the orthonormal Legendre functions of pygeoinf0.probability.legendre_decay (the 'legendre' "
                          "realisation of an interval has them)")

    def riesz(self, measures):
        self._unsupported("Riesz representers of measures")

    def to_l2(self, R):
        """The realisation of the same functions in L2 of the domain (with the L2 backend of the same realisation), or
        None when it has to be fitted (used by Inclusion)."""
        self._unsupported("conversion to L2")

    @classmethod
    def supports(cls, op):
        """Whether this backend implements (or overrides) the operation op."""
        return getattr(cls, op, None) is not getattr(Backend, op, None)


class IntervalBackend(Backend):
    """Piecewise Legendre series on a bounded interval (dense or sparse families), Lebesgue or density measure."""

    dim = 1
    realisation_name = "legendre"
    sparse_families = True

    def __init__(self, measure, tol):
        self.measure, self.tol, self.domain = measure, tol, measure.set
        self.a, self.b = measure.set.a, measure.set.b
        self.density = getattr(measure, "density", None)

    def restricted(self, R, target):
        if not isinstance(R, L.PiecewiseLegendre):                               # sparse families are fitted
            return None
        out = L.cut(R.coefs, R.breaks, target.a, target.b)
        return None if out is None else L.PiecewiseLegendre(out[0], out[1], R.err)

    def glued(self, parts):
        if not all(isinstance(R, L.PiecewiseLegendre) for _, R in parts) or len({R.N for _, R in parts}) != 1:
            return None
        N = parts[0][1].N
        from .series.cuts import joined
        out = joined(self.a, self.b, [(B.a, B.b, R.breaks, R.coefs) for B, R in parts], lambda: np.zeros((1, N)))
        return None if out is None else L.PiecewiseLegendre(out[0], out[1], np.sqrt(sum(R.err ** 2 for _, R in parts)))

    # construction ----------------------------------------------------------------------------
    def probe(self):
        return (self.a + (self.b - self.a) * np.array([0.3, 0.5, 0.7]),)

    def _scout(self, breaks, w, N=1):
        """legendre.fit's scout for samples at spacing <= w / 2 on every initial piece (never fewer than its default)."""
        init = np.array([self.a, self.b], float) if breaks is None else \
            np.unique(np.concatenate(([self.a, self.b], np.clip(breaks, self.a, self.b))))
        default = 2049 if N == 1 else int(min(2049, max(129, (1 << 19) // N)))
        return max(default, (len(init) - 1) * (int(math.ceil(float(np.diff(init).max()) / (0.5 * w))) + 2))

    def fit_at(self, fun, N=1, breaks=None, w=None):
        """fit at a declared resolution w (core/fits.py): the scout at spacing <= w / 2."""
        return L.fit(fun, self.a, self.b, N=N, breaks=breaks, tol=self.tol, scout=self._scout(breaks, w, N))

    def fit_family_at(self, fun_idx, N, breaks, w, name="F"):
        return self.fit_family(fun_idx, N, breaks=breaks, name=name, w=w)

    def fit(self, fun, N=1, breaks=None):
        return L.fit(fun, self.a, self.b, N=N, breaks=breaks, tol=self.tol)

    def fit_family(self, fun_idx, N, breaks=None, sparse=None, name="F", w=None):
        a, b = self.a, self.b                              # w: a declared resolution (core/fits.py): finer scouts
        if sparse is None:
            probe = np.unique(np.linspace(0, N - 1, min(N, 64)).astype(int))
            sparse = N >= 32 and localised(fun_idx(np.linspace(a, b, 257), probe)) and 4097 * N * 4 <= 4e8
        if sparse and self.density is not None:
            sparse = False
        if sparse:
            scout = 4097 if w is None else max(4097, int(math.ceil((b - a) / (0.5 * w))) + 2)
            spl, u = S.fit_sparse(fun_idx, a, b, N, tol=self.tol, scout=scout)
            if u:
                warnings.warn(f"family {name}: {u} unresolved piece(s)")
            return spl
        pls, n_start = [], 17
        for s in range(0, N, 4096):
            idx = np.arange(s, min(N, s + 4096))
            pl, u = L.fit(lambda x, idx=idx: fun_idx(x, idx), a, b, N=len(idx), breaks=breaks, tol=self.tol,
                          n_start=n_start, scout=None if w is None else self._scout(breaks, w, len(idx)))
            if pl.npieces == 1:
                n_start = max([n for n in L.NSEQ if n <= pl.coefs[0].shape[0]] + [17])
            if u:
                warnings.warn(f"family {name}: {u} unresolved piece(s) in columns {s}..{idx[-1]}")
            pls.append(pl)
        return pls[0] if len(pls) == 1 else L.hstack(pls)

    def constant(self, values):
        return L.PiecewiseLegendre.constant(self.a, self.b, values)

    def coordinate(self, k=0):
        return L.PiecewiseLegendre.identity(self.a, self.b)

    def legendre_decay_generators(self, decay, modes):
        """The orthonormal Legendre functions phi_k of Lebesgue measure on [a, b] (k < modes), exactly (one piece, the
        coefficients sqrt((2k + 1) / (b - a))), and the eigenvalues decay^(2k) of pygeoinf0.probability.legendre_decay."""
        k = np.arange(int(modes))
        R = L.PiecewiseLegendre([self.a, self.b], [np.diag(np.sqrt((2 * k + 1) / (self.b - self.a)))], np.zeros(len(k)))
        return R, decay ** (2.0 * k)

    # algebra ---------------------------------------------------------------------------------
    def lincomb(self, terms):
        if any(S.is_sparse(R) for _, R in terms):
            raise NotImplementedError("arithmetic on a sparse family: take its elements, or apply operators to it")
        return L.lincomb(terms)

    def combine(self, R, A):
        return S.combine(R, A)

    def colsum(self, R):
        return L.lincomb([(1.0, S.colsum(R))])

    def take(self, R, idx):
        out = S.take(R, idx)
        return L.lincomb([(1.0, out)]) if np.ndim(idx) == 0 else out

    def hstack(self, Rs):
        return L.hstack(Rs)

    def multiply(self, A, B):
        return L.multiply(A, B)

    # measure ---------------------------------------------------------------------------------
    def gram(self, A, B):
        if self.density is None:
            return S.gram(A, B)
        if S.is_sparse(A) or S.is_sparse(B):
            raise NotImplementedError("sparse families with a density measure are not implemented yet")
        return L.gram_weighted(A, B, self.density)

    def sqnorms(self, A):
        if self.density is None:
            return S.sqnorms(A)
        return np.diag(self.gram(A, A)).copy()

    def integral(self, R, over=None):
        from ..sets import Interval
        if over is None:
            c, d = self.a, self.b
        else:
            J = Interval(over) if isinstance(over, str) else over
            c, d = (J.a, J.b) if isinstance(J, Interval) else (float(J[0]), float(J[1]))
        if self.density is not None:
            if S.is_sparse(R):
                raise NotImplementedError("sparse families with a density measure are not implemented yet")
            R = L.multiply(R, self.density)
        return S.integral(R, c, d)

    def measure_of_cell(self, c, d):
        from ..sets import Interval
        return self.measure(Interval(c, d))

    # values and description ----------------------------------------------------------------
    def evaluate(self, R, x):
        if S.is_sparse(R):
            R = R.take(np.arange(R.N))
        return L.evaluate(R, x)

    def error(self, R):
        if self.density is not None:
            return R.err * math.sqrt(float(self.density.sup_bound()[0]))
        return R.err

    def breaks(self, R):
        return R.breaks

    def merge_breaks(self, items):
        return np.unique(np.concatenate([np.asarray(b, float) for b in items]))

    def lp_norm(self, R, p):
        if self.density is not None:
            raise NotImplementedError("L^p norms with a density are not implemented yet")
        if math.isinf(p):
            out = np.zeros(R.N)
            for C in R.coefs:
                t = np.linspace(-1, 1, max(2001, 20 * C.shape[0]))
                out = np.maximum(out, np.abs(L.polyval(t, C)).max(axis=0))
            return out
        out = np.zeros(R.N)
        for i, C in enumerate(R.coefs):
            t, w = gauss(2 * C.shape[0] + 64)
            out += 0.5 * (R.breaks[i + 1] - R.breaks[i]) * (w[:, None] * np.abs(L.polyval(t, C)) ** p).sum(axis=0)
        return out ** (1.0 / p)

    def is_sparse(self, R):
        return S.is_sparse(R)

    def describe(self, R):
        kind = "sparse " if S.is_sparse(R) else ""
        return dict(kind=f"{kind}Legendre", pieces=R.npieces, degree=max(R.degrees), size=R.size,
                    breaks=np.asarray(R.breaks))

    def summary(self, R):
        d = self.describe(R)
        return f"Legendre: {d['pieces']} piece{'s' * (d['pieces'] > 1)}, degree {d['degree']}"
