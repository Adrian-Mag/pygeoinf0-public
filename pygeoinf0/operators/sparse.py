"""Linear maps between Euclidean spaces stored without a dense matrix (moved from linear.py for plan 008; imported from here
since plan 065, which removed the forwarder in linear.py):

SparseMatrix(A)       a scipy.sparse matrix (e.g. the Gram matrix of many localised kernels)
KroneckerMatrix(K)    G G* of a lattice product family, applied factor by factor
"""
import numpy as np

from ..core.claims import DERIVED
from ..spaces.elements import Vector, VectorFamily
from ..spaces.base import Euclidean
from .linear import LinearOperator
from ..spaces.identity import identical


class SparseMatrix(LinearOperator):
    """A linear map between Euclidean spaces given by a scipy.sparse matrix (for example the Gram matrix of
    many localised kernels). Applying it costs one sparse product; no dense matrix is ever formed unless asked."""

    why_linear = "multiplication by a fixed sparse matrix"

    def __init__(self, A, domain=None, codomain=None, name="A"):
        import scipy.sparse as sp
        A = sp.csr_matrix(A)
        domain = Euclidean(A.shape[1]) if domain is None else domain
        codomain = Euclidean(A.shape[0]) if codomain is None else codomain
        if (codomain.dim, domain.dim) != A.shape:
            raise ValueError(f"a {A.shape} matrix maps R^{A.shape[1]} to R^{A.shape[0]}")
        super().__init__(domain, codomain, name=name)
        self.A = A

    @property
    def nnz(self):
        return self.A.nnz

    def _apply(self, x):
        if isinstance(x, Vector):
            return Vector(self.codomain, self.A @ x.values, name=f"{self.name}{x.name}")
        if isinstance(x, VectorFamily):
            return VectorFamily(self.codomain, np.asarray((self.A @ x.values.T).T), name=f"{self.name}{x.name}")
        raise TypeError(type(x).__name__)

    def matrix(self):
        if self.A.shape[0] * self.A.shape[1] > 6000 ** 2:
            raise MemoryError(f"{self.name} is a {self.A.shape} sparse matrix; it is not converted to a dense one")
        return self.A.toarray()

    def diagonal(self):
        return self.A.diagonal()

    def sparse(self):
        return self.A

    def _make_adjoint(self):
        import scipy.sparse as sp
        B = self.A.T.tocsr()
        if not self.codomain.is_standard:
            if self.codomain.weights is None:
                raise NotImplementedError("sparse adjoints with a non-diagonal metric are not implemented")
            B = B @ sp.diags(self.codomain.weights)
        if not self.domain.is_standard:
            if self.domain.weights is None:
                raise NotImplementedError("sparse adjoints with a non-diagonal metric are not implemented")
            B = sp.diags(1.0 / self.domain.weights) @ B
        return SparseMatrix(B, domain=self.codomain, codomain=self.domain, name=f"{self.name}*")

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}  [sparse, {self.A.nnz} nonzeros]"


class KroneckerMatrix(LinearOperator):
    """G G* for a lattice product family: y -> K (W y), K = P^T (G_1 kron ... kron G_d) P (see separated.KronGram) and W the
    diagonal metric of the data space. Applying it costs O(N sum n_k); it is never formed as an N x N matrix."""

    why_linear = "products with fixed Kronecker factors and a diagonal"

    def __init__(self, K, weights=None, domain=None, codomain=None, name="A"):
        domain = Euclidean(K.N) if domain is None else domain
        codomain = domain if codomain is None else codomain
        super().__init__(domain, codomain, name=name)
        self.K = K
        self.weights = None if weights is None else np.asarray(weights, float)

    @classmethod
    def from_factors(cls, factors, space=None, name="K"):
        """A_1 kron A_2 kron ... on R^(n_1 n_2 ...) (C order), each A_k a symmetric matrix, e.g. a separable covariance."""
        from ..realisations.series.separated import KronGram
        factors = [np.asarray(A, float) for A in factors]
        N = int(np.prod([A.shape[0] for A in factors]))
        out = cls(KronGram(factors, np.arange(N)), domain=space, name=name)
        if all(np.allclose(A, A.T, rtol=1e-12, atol=0) for A in factors):
            out._sa = (True, DERIVED, "a Kronecker product of symmetric matrices")
        return out

    @property
    def nnz(self):
        return self.K.nnz

    def _w(self, V):
        if self.weights is None:
            return V
        return self.weights * V if V.ndim == 1 else self.weights[:, None] * V

    def _apply(self, x):
        if isinstance(x, Vector):
            return Vector(self.codomain, self.K @ self._w(x.values), name=f"{self.name}{x.name}")
        if isinstance(x, VectorFamily):
            return VectorFamily(self.codomain, (self.K @ self._w(x.values.T)).T, name=f"{self.name}{x.name}")
        raise TypeError(type(x).__name__)

    def matrix(self):
        D = self.K.toarray()
        return D if self.weights is None else D * self.weights[None, :]

    def diagonal(self):
        d = self.K.diagonal()
        return d if self.weights is None else d * self.weights

    def _make_adjoint(self):
        if identical(self.domain, self.codomain):
            return self                       # W^-1 (K W)^T W = K W
        raise NotImplementedError("adjoint of a Kronecker Gram operator between different spaces")

    def __repr__(self):
        return (f"{self.name}: {self.domain} → {self.codomain}  [Kronecker product of {len(self.K.dims)} factors of "
                f"sizes {self.K.dims}]")
