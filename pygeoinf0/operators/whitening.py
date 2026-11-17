"""The whitened coordinates of a noise covariance (plan 068 P5; the author's decision Q1): D.whitened(C).

    W = D.whitened(C)          d -> L^-1 d with C = L L^T (any factor the metric code chooses: spaces/metrics.py), from
                               D_C = D's vectors with the inner product <x, y> = x . C^-1 y onto R^n with the dot
                               product (a DataSpace: the whitened coordinates)
    W(d), W.inv(z)             z = L^-1 d; d = L z. W accepts an element of D itself (the same vectors)
    W.adjoint                  is W.inv: W is an isometry onto, ||W d||^2 = d . C^-1 d = ||d||_(D_C)^2

If d ~ N(0, C) then W d ~ N(0, I): the whitened noise is standard. The construction does not change D (its metric is
not the noise); it names the space D_C and the map into plain coordinates that a method wanting the C^-1 inner product
uses. The data-space engine (solvers/dataspace/, plan 070 P3) takes D.whitened(C) as one of its explicit metrics
(solvers.dataspace.metric_of: the covariance C in D's coordinates).

    SameVectors(D, E)          x -> x from D onto E, the same vectors (same dimension) with another inner product: its
                               adjoint is y -> W_D^-1 W_E y. It reads a plain D with the noise N(b, C) stated in the
                               relation as D_C = D's vectors with the C^-1 inner product, where the noise is the
                               standard Gaussian (relations.kernels.AdditiveMarkovKernel.standardised; plan 071 P4, Q1
                               as amended: both routes are one model)
"""
from ..core.claims import THEOREM, Claim
from ..core.errors import SpaceMismatch
from ..spaces.data import DataSpace
from ..spaces.elements import Vector, VectorFamily
from .linear import LinearOperator


class Whitening(LinearOperator):
    """d -> L^-1 d from D_C (D's vectors, C^-1 inner product) onto the whitened coordinates (the module docstring);
    with inverse=True the inverse z -> L z, which is also its adjoint."""

    why_linear = "a triangular factor of the covariance applied to coordinates"

    def __init__(self, D, C, name="L⁻¹", _pair=None):
        if _pair is not None:                              # the inverse z -> L z of a whitening
            W = _pair
            super().__init__(W.codomain, W.domain, name=f"({W.name})⁻¹")
            self.source, self.covariance, self.inverse_of = W.source, W.covariance, W
        else:
            DC = D.standard.with_covariance(C)
            Z = DataSpace(D.dim, labels=D.labels if DC.weights is not None else None)
            super().__init__(DC, Z, name=name)
            self.source, self.covariance, self.inverse_of = D, C, None
        self._claims += [Claim("isometry onto", THEOREM, "C = L L^T: ||L^-1 d||^2 = d . C^-1 d, and L^-1 is invertible",
                               key="isometry"),
                         Claim("bounded", THEOREM, "an isometry: norm 1", key="bounded")]

    # the two directions ------------------------------------------------------------------------------
    def _forward(self, V):
        """L^-1 V for coordinates V of shape (n,) or (n, k)."""
        return self.domain.to_orthonormal(V)

    def _backward(self, V):
        """L V."""
        return self.inverse_of.domain.from_orthonormal(V)

    def _values(self, V):
        return self._forward(V) if self.inverse_of is None else self._backward(V)

    def __call__(self, x):
        if isinstance(x, (Vector, VectorFamily)) and self.inverse_of is None and x.space != self.domain \
                and x.space == self.source:
            x = (Vector if isinstance(x, Vector) else VectorFamily)(self.domain, x.values, name=x.name)   # same vectors
        return super().__call__(x)

    def _apply(self, x):
        if isinstance(x, Vector):
            return Vector(self.codomain, self._values(x.values), name=f"{self.name}{x.name}")
        if isinstance(x, VectorFamily):
            return VectorFamily(self.codomain, self._values(x.values.T).T, name=f"{self.name}{x.name}")
        raise SpaceMismatch(f"{self.name} acts on vectors of {self.domain}")

    # inverse and adjoint: one operator ---------------------------------------------------------------
    def _other(self):
        if self._inv is None:
            self._inv = self.inverse_of if self.inverse_of is not None else Whitening(None, None, _pair=self)
            self._inv._inv = self
        return self._inv

    @property
    def inv(self):
        return self._other()

    def inverse(self, solver=None):
        """The inverse z -> L z (exact: no solver)."""
        return self._other()

    def _make_adjoint(self):
        return self._other()                               # an isometry onto: W* = W^-1

    def norm_bound(self):
        """1: an isometry."""
        return 1.0

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}  [whitened coordinates]"


class SameVectors(LinearOperator):
    """x -> x from a Euclidean space D onto E, the same coordinates with another inner product (module docstring):
    <J x, y>_E = x . W_E y = <x, W_D^-1 W_E y>_D, so J* y = W_D^-1 W_E y."""

    why_linear = "the identity on coordinates"

    def __init__(self, D, E, name="J"):
        if D.dim != E.dim:
            raise SpaceMismatch(f"the same vectors: {D} and {E} have different dimensions")
        super().__init__(D, E, name=name)
        self._claims.append(Claim("bounded", THEOREM, "a linear map between finite-dimensional spaces", key="bounded"))

    def _apply(self, x):
        if isinstance(x, Vector):
            return Vector(self.codomain, x.values, name=x.name)
        if isinstance(x, VectorFamily):
            return VectorFamily(self.codomain, x.values, name=x.name)
        raise SpaceMismatch(f"{self.name} acts on vectors of {self.domain}")

    def _make_adjoint(self):
        D, E = self.domain, self.codomain

        def back(y):
            V = D.solve_metric(E.apply_metric(y.values.T)).T if isinstance(y, VectorFamily) else \
                D.solve_metric(E.apply_metric(y.values))
            return (VectorFamily if isinstance(y, VectorFamily) else Vector)(D, V, name=f"{self.name}*{y.name}")
        out = LinearOperator(E, D, apply=back, name=f"{self.name}*")
        out._claims = [Claim("linear", THEOREM, "W_D^-1 W_E applied to coordinates", key="linear")]
        out._adjoint = self
        return out

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}  [the same vectors]"
