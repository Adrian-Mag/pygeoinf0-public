"""Solver backends: where and how a solver's factorisation runs (plan 063). The solver fixes the mathematics; a backend
computes it to a stated precision or refuses, and says what it costs before it runs.

    "dense"    LAPACK on the CPU (numpy, scipy): the n x n matrix and its factor, 16 n^2 bytes, n^3/3 flops for Cholesky
    "sparse"   a sparse Cholesky of A.sparse() (sparse matrices, diagonals, identities and their sums): CHOLMOD when
               scikit-sparse is installed, else SuperLU in symmetric mode with diagonal pivots; memory is the factor's
               fill-in, not known before it is computed

pg.solvers.backends() lists them with what each computes, when it is the right choice and whether it is installed. A GPU
backend is a later plan: float64 on the GPU at its real speed, or a float32 result with its precision stated.
"""
import numpy as np

from ..core.errors import Unsupported


class Backend:
    name = "backend"
    about = {}

    def available(self):
        return True, "always"

    def __repr__(self):
        return f"backend {self.name!r}"


class Dense(Backend):
    name = "dense"
    about = dict(computes="the factorisation of the n x n matrix (LAPACK)", when="n up to settings.dense_max and 16 n^2 bytes "
                 "within settings.memory_gb", error="backward stable; the condition estimate is LAPACK's", cost="n^3/3 flops "
                 "(Cholesky), 16 n^2 bytes")

    def cost(self, n, nnz=None):
        return dict(flops=n ** 3 / 3, bytes=16.0 * n * n)


class Sparse(Backend):
    name = "sparse"
    about = dict(computes="the Cholesky factor of A.sparse() (CHOLMOD, or SuperLU with symmetric diagonal pivots)",
                 when="A given as sparse matrices, diagonals and their sums, with a factor that stays sparse (local "
                 "kernels, finite elements); the dense matrix need not fit in memory",
                 error="backward stable; the condition is estimated from the factor (Higham's 1-norm estimator of the "
                 "inverse); positive definiteness is checked by the pivots", cost="the factor's fill-in, known only "
                 "after it is computed; nnz(A) to start")

    def available(self):
        try:
            import sksparse.cholmod  # noqa: F401
            return True, "CHOLMOD (scikit-sparse)"
        except ImportError:
            return True, "SuperLU (scipy); install scikit-sparse for CHOLMOD"

    def cost(self, n, nnz=None):
        return dict(flops=None, bytes=None if nnz is None else 12.0 * nnz)

    def cholesky(self, A):
        """A factorisation of the self-adjoint operator A from its sparse matrix, in the inner product of its space."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        from .direct import _digits, _Factorisation, _warn
        D = A.domain
        M = A.sparse()
        if M is None:
            raise Unsupported(f"{A.name} has no sparse form (sparse matrices, diagonals, identities and their sums); "
                              "use the dense backend or CG")
        if D.is_standard:
            S = M
        elif getattr(D, "weights", None) is not None:
            S = sp.diags(D.weights) @ M
        else:
            raise Unsupported(f"{A.name}: the sparse backend needs a diagonal metric; {D} has a full one")
        S = (0.5 * (S + S.T)).tocsc()
        anorm = float(abs(S).sum(axis=0).max()) if S.nnz else 0.0
        how = self.available()[1]
        try:
            from sksparse.cholmod import CholmodNotPositiveDefiniteError, cholesky
            try:
                F = cholesky(S)
            except CholmodNotPositiveDefiniteError:
                raise np.linalg.LinAlgError(f"sparse Cholesky failed: {A.name} is not numerically positive definite")
            solve = F
        except ImportError:
            try:
                lu = spla.splu(S, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
            except RuntimeError as e:                              # a zero pivot: not positive definite
                raise np.linalg.LinAlgError(f"sparse Cholesky failed: {A.name} is not numerically positive definite ({e})")
            if not (np.array_equal(lu.perm_r, lu.perm_c) and np.all(lu.U.diagonal() > 0)):
                raise np.linalg.LinAlgError(f"sparse Cholesky failed: {A.name} is not numerically positive definite (or "
                                            "SuperLU left the diagonal; install scikit-sparse for CHOLMOD)")
            solve = lu.solve
        n = S.shape[0]
        inv = spla.LinearOperator((n, n), matvec=solve, rmatvec=solve, dtype=float)
        cond = anorm * spla.onenormest(inv) if n else 1.0
        _warn(A.name, cond)
        return _Factorisation(A, lambda Y: solve(D.apply_metric(Y)), [f"n = {n}, nnz = {S.nnz}, sparse Cholesky ({how})",
                                                                       _digits(cond)])


_BACKENDS = {b.name: b for b in (Dense(), Sparse())}


def backend_named(backend):
    if isinstance(backend, Backend):
        return backend
    if backend not in _BACKENDS:
        raise ValueError(f"no backend {backend!r}; the backends are {', '.join(_BACKENDS)}")
    return _BACKENDS[backend]


def backends():
    """For people and agents: each backend with what it computes, when it is the right choice, its error and cost, and
    whether it is installed here."""
    return {name: dict(b.about, available=b.available()[1]) for name, b in _BACKENDS.items()}
