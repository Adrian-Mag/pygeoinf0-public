"""G G* on a data space with a diagonal metric, for separable functionals (plan 045).

With D = R^n and <x, y>_D = x . W y, W diagonal, the adjoint of G carries W, so as a map on D

    G G* = K W = W^(-1/2) Kw W^(1/2),    Kw = W^(1/2) K W^(1/2),    (G G* + s)^+ = W^(-1/2) (Kw + s)^+ W^(1/2)     (s diagonal)

with K the plain Gram matrix <g_i, g_j> of G (G.structured_gram): block diagonal, one block per harmonic, and so is Kw. The
symmetric solve is on Kw, not K: conditioning is measured where the solve happens, and on a singular K only this one is the
pseudo-inverse in D's metric (A.pseudo_inverse(rcond)), so a datum outside the range is fitted in D's norm. A.inv is the
inverse, or refused (solvers/).
"""
import numpy as np

from ..core.claims import DERIVED, Claim, Claims
from ..spaces.elements import Vector, VectorFamily
from ..spaces.base import Euclidean
from .blocks import BlockDiagonal
from .linear import Diagonal, LinearOperator


class WeightedGram(LinearOperator):
    """K W on D (module docstring), K a plain BlockDiagonal on R^n, optionally plus a diagonal in D's coordinates."""

    why_linear = "K W (+ a diagonal), with K its block-diagonal part"

    def linear_parts(self):
        """The operators whose linearity this one inherits."""
        return [self.K]

    rows = False

    def __init__(self, D, K, name="A", shift=None, claims=()):
        super().__init__(D, D, name=name)
        self.K, self.w, self.shift = K, np.asarray(D.weights, float), shift
        self._sa = (True, DERIVED, "x . W K W y + x . W s y, symmetric, in D's metric")
        self._pos = ("psd", DERIVED, "K positive semidefinite, W and the shift nonnegative")
        self._adjoint = self
        self._claims += [Claim("G G* = K W on D, K the plain Gram matrix, W D's diagonal metric", DERIVED,
                               "the adjoint of G in D's metric; block diagonal with K (operators/plain_gram.py)", key="block diagonal")]
        self._claims += list(claims)

    @property
    def n_blocks(self):
        return self.K.n_blocks

    def values(self, V):
        out = self.K._rows_values(self.w[:, None] * V)
        return out if self.shift is None else out + self.shift[:, None] * V

    def _apply(self, x):
        if isinstance(x, Vector):
            return Vector(self.codomain, self.values(x.values[:, None])[:, 0], name=f"{self.name}{x.name}")
        return VectorFamily(self.codomain, self.values(x.values.T).T, name=f"{self.name}{x.name}")

    def matrix(self):
        return self.values(np.eye(self.domain.dim))

    def __add__(self, other):
        d = _diagonal(other, self.domain)
        if d is None:
            return super().__add__(other)
        return WeightedGram(self.domain, self.K, name=f"{self.name} + {other.name}", shift=d if self.shift is None else self.shift + d)

    __radd__ = __add__

    def inverse(self, solver=None):
        from ..solvers.direct import BlockDirect
        if isinstance(solver, BlockDirect):
            return _Inverse(self, solver.rcond, solver.refuse)
        if solver is not None:
            return super().inverse(solver)
        return _Inverse(self)

    def pseudo_inverse(self, rcond):
        return _Inverse(self, rcond)

    @property
    def inv(self):
        if self._inv is None:
            self._inv = _Inverse(self)
        return self._inv


class _Inverse(LinearOperator):
    """W^(-1/2) (Kw + s)^-1 W^(1/2) (module docstring): a symmetric solve per block of Kw; with rcond, its pseudo-inverse."""

    why_linear = "an inverse, by symmetric solves, of a linear operator"

    def linear_parts(self):
        """The operators whose linearity this one inherits."""
        return [self.A]

    def __init__(self, A, rcond=None, refuse=True):
        super().__init__(A.domain, A.domain, name=f"({A.name})⁻¹" if rcond is None else f"({A.name})⁺")
        self.A = A
        K, E = A.K, Euclidean(A.domain.dim)
        self.root = np.sqrt(A.w)
        mats = []
        for b, rows in enumerate(K.partition):
            r = self.root[rows]
            mats.append(r[:, None] * K.matrices[K.shared[b]] * r[None, :])
        shift = None if A.shift is None else [A.shift[rows] for rows in K.partition]
        white = BlockDiagonal(E, mats, name=f"{A.name} whitened", order=K.order, shift=shift)
        from ..solvers.direct import BlockDirect
        self.plain = white.inv if rcond is None and refuse else white.inverse(BlockDirect(rcond, refuse))
        self._sa = (True, DERIVED, "the inverse of a self-adjoint operator")
        self._adjoint = self
        kind = "inverse" if rcond is None else f"pseudo-inverse with rcond {rcond:g}"
        self._claims.append(Claim(f"the {kind} in D's metric, by a solve per block of W^(1/2) K W^(1/2) + s", DERIVED,
                                  "similarity by W^(1/2), W diagonal", key="inverse"))

    @property
    def claims(self):
        return Claims(list(super().claims) + [c for c in self.plain.claims if c.key == "inverse"])

    def report(self):
        return self.plain.report()

    def _apply(self, y):
        E, r = self.plain.domain, self.root
        if isinstance(y, Vector):
            return Vector(self.codomain, self.plain(E(r * y.values)).values / r, name=f"{self.name}{y.name}")
        z = self.plain(VectorFamily(E, r[None, :] * y.values))
        return VectorFamily(self.codomain, z.values / r[None, :], name=f"{self.name}{y.name}")


def _diagonal(other, D):
    """The numbers of a diagonal operator on D, or None."""
    from .linear import Identity, Sum
    if isinstance(other, Identity) and other.domain == D:
        return np.ones(D.dim)
    if isinstance(other, Diagonal) and other.domain == D:
        return np.asarray(other.d, float)
    if isinstance(other, Sum) and other.domain == D:
        parts = [(a, _diagonal(B, D)) for a, B in other.terms]
        if all(d is not None for _, d in parts):
            return sum(a * d for a, d in parts)
    return None


def plain_normal(G, H):
    """G H* as a structured operator, or None: G and H the same separable functionals, D standard or with a diagonal metric."""
    if H is not G:
        return None
    D = G.codomain
    if not (D.is_standard or D.weights is not None):
        return None
    K = G.structured_gram()
    if K is None:
        return None
    K.name = f"{G.name}{G.name}*"
    K._pos = ("psd", DERIVED, f"<{G.name}{G.name}* y, y> = ||{G.name}* y||^2 >= 0")
    if D.is_standard:
        K._sa = (True, DERIVED, f"of the form {G.name} {G.name}*")
        return K
    return WeightedGram(D, K, name=K.name)
