"""The metric of a weighted R^n or C^n (one code for both fields, plan 068 P1: the complex Euclidean space had its own,
with a dense inverse for a covariance).

A Euclidean space with an inner product other than the dot product is the standard space E = (R^n, .) with a
self-adjoint positive definite operator W on E:  <x, y>_D = x . W y. W lives on E, not on D, so nothing is
circular, and W is the adjoint of the identity map J: E -> D (<Jx, y>_D = x . W y = <x, W y>_E).

The space can be given its metric W, or its covariance C = W^-1 (the natural form for noise). What the space
needs from the operator is three things, and each is computed by the cheapest route for the operator's type:

    apply W         inner products, adjoints into D, chi^2
    solve with W    adjoints out of D, orthonormal coordinates
    a factor L      with W = L L^T, for orthonormal coordinates and for sampling the space's standard Gaussian

    type                 apply        solve                               factor
    Diagonal             O(n)         O(n)                                O(n)
    Matrix (dense)       O(n^2)       Cholesky once, then O(n^2)           the Cholesky factor
    KroneckerMatrix      O(n sum n_k) eigendecompositions of the factors   symmetric square root, same
    SparseMatrix         O(nnz)       sparse LU once, then sparse solves  not available (no sparse Cholesky)
    other operators      apply        its recommended solver (op.inv)      a dense Cholesky up to n = 4000

Self-adjointness and positive definiteness are claims: checked where it is cheap (positive diagonal, Cholesky,
eigenvalues of Kronecker factors), recorded as asserted otherwise.

On C^n (Euclidean(n, field="C")) the metric is a number, a positive vector (diagonal) or a Hermitian positive definite
matrix, given as an array: <x, y> = y^H W x, W = L L^H by Cholesky, the transposes above conjugate transposes. A
covariance C is factored and solved with, as on R^n, never inverted. Operators on C^n are not built (the operator
classes act on real coordinates), so an operator, Kronecker or sparse metric of C^n is refused (Unsupported).
"""
import numpy as np
import scipy.linalg as sla

from ..core.claims import ASSERTED, CHECKED, Claim, by_rule

DENSE_FACTOR_MAX = 4000
SYMMETRY_RTOL, SYMMETRY_ATOL = 1e-12, 1e-14      # rounding asymmetry tolerated: |A - A^H| <= ATOL max|A| + RTOL |A^H|


def symmetric_within_rounding(A, rtol=SYMMETRY_RTOL):
    """Whether a matrix (a metric or a covariance in coordinates) is symmetric (Hermitian) up to rounding: entrywise
    |A - A^H| <= SYMMETRY_ATOL max|A| + rtol |A^H| (the one tolerance of the package for metrics and noise covariances,
    plan 071 P4; a caller whose matrix was computed through a solve passes a wider rtol for that solve's rounding). Within
    it the matrix is used symmetrised, (A + A^H) / 2; beyond it, refused (never symmetrised silently)."""
    H = A.conj().T
    return bool(np.allclose(A, H, rtol=rtol, atol=SYMMETRY_ATOL * np.abs(A).max()))


def _cols(V):
    V = np.asarray(V)
    V = V if np.iscomplexobj(V) else np.asarray(V, float)
    return (V[:, None], True) if V.ndim == 1 else (V, False)


def make_metric(x, base, kind):
    """The Metric of kind "metric" (W) or "covariance" (C = W^-1) on the standard space base, from what a Euclidean
    space accepts (Euclidean's docstring): an operator on base, a number, a vector (diagonal), an n x n array or a
    scipy.sparse matrix; on C^n a number, a positive vector or a Hermitian n x n array."""
    if base.field == "C":
        return Metric(_complex_array(x, base.dim, kind), base, kind)
    from ..operators.linear import Diagonal, LinearOperator, Matrix
    from ..operators.sparse import SparseMatrix
    if isinstance(x, LinearOperator):
        op = x
    elif np.ndim(x) == 0:
        op = Diagonal(np.full(base.dim, float(x)), base)
    else:
        import scipy.sparse as sps
        if sps.issparse(x):
            op = SparseMatrix(x, base, base)
        else:
            X = np.asarray(x, float)
            if X.shape == (base.dim,):
                op = Diagonal(X, base)
            elif X.shape == (base.dim, base.dim):
                op = Matrix(X, base, base, name="W" if kind == "metric" else "C")
            else:
                raise ValueError(f"a {kind} on R^{base.dim} is a number, a vector of length {base.dim}, an "
                                 f"{base.dim} x {base.dim} matrix or an operator")
    return Metric(op, base, kind)


def _complex_array(x, n, kind):
    """The array of a metric or covariance of C^n: a positive vector (a number broadcast) or an n x n complex array."""
    import scipy.sparse as sps
    if sps.issparse(x) or not isinstance(x, (np.ndarray, list, tuple, int, float, complex, np.number)):
        from ..core.errors import Unsupported
        raise Unsupported(f"a {kind} of C^{n} is a number, a positive vector or a Hermitian {n} x {n} array: operators "
                          f"on C^n are not built (the operator classes act on real coordinates), got {type(x).__name__}")
    X = np.asarray(x)
    if X.ndim == 0:
        X = np.full(n, X)
    if X.ndim == 1:
        if X.shape != (n,) or (np.iscomplexobj(X) and np.any(X.imag != 0)):
            raise ValueError(f"a diagonal {kind} of C^{n} is {n} positive numbers")
        return np.asarray(X.real, float)
    if X.shape != (n, n):
        raise ValueError(f"a {kind} of C^{n} is a number, a vector of length {n} or a Hermitian {n} x {n} matrix")
    return np.asarray(X, complex)


def _out(V, one):
    return V[:, 0] if one else V


class _Ops:
    """apply / solve / factor routines for one operator P on the standard R^n (P is W or C)."""

    def __init__(self, op, base):
        self.op, self.base, self.n = op, base, base.dim
        self.claims = []
        if isinstance(op, np.ndarray):                    # C^n: a positive diagonal or a Hermitian matrix (make_metric)
            if op.ndim == 1:
                self._diagonal(op)
            else:
                self._dense(op)
            return
        from ..operators.linear import Diagonal, Matrix
        from ..operators.inverses import Inverse
        from ..operators.sparse import KroneckerMatrix, SparseMatrix
        if isinstance(op, Diagonal):
            self._diagonal(op.d)
        elif isinstance(op, Matrix):
            self._dense(op.A)
        elif isinstance(op, KroneckerMatrix) and op.weights is None:
            self.kind, self.K = "kronecker", op.K
            lam = [l for l, _ in self.K.eig()]
            if any(np.min(l) <= 0 for l in lam):
                raise ValueError("a Kronecker factor is not positive definite")
            self.claims.append(Claim("self-adjoint and positive definite", CHECKED,
                                     "symmetric factors with positive eigenvalues"))
        elif isinstance(op, SparseMatrix):
            import scipy.sparse as sps
            self.kind, A = "sparse", op.A
            asym = abs(A - A.T)
            if asym.nnz and asym.max() > 1e-12 * abs(A).max():
                raise ValueError("a metric or covariance must be symmetric")
            self.A, self._lu = A, None
            self._sps = sps
            self.claims.append(Claim("positive definite", ASSERTED,
                                     "a sparse matrix is not tested for positive definiteness (no sparse Cholesky in SciPy)"))
        else:
            self.kind = "operator"
            sa, pos = op.is_selfadjoint(), op.positivity()
            if sa[0] is False:
                raise ValueError(f"{op.name} is not self-adjoint, so it is not a metric or covariance")
            if sa[0] and pos[0] == "pd":
                self.claims.append(Claim("self-adjoint and positive definite", by_rule(sa[1], pos[1]),
                                         f"{sa[2]}; {pos[2]}"))
            else:
                self.claims.append(Claim("self-adjoint and positive definite", ASSERTED, "stated by the user, not verified"))
            self._inv = op.inv if not isinstance(op, Inverse) else None
            self._L = None

    def _diagonal(self, d):
        self.kind = "diagonal"
        if not (np.all(np.isfinite(d)) and np.all(d > 0)):
            raise ValueError(
                "the diagonal of a metric or covariance must be positive and finite. (A noise standard deviation "
                "sigma = 0 gives the infinite weights 1/sigma^2 and the covariance 0: the weighted data space does not "
                "exist. Noise-free data need no weighting: use Euclidean(n) and the minimum-norm solution; there is "
                "nothing to damp.)")
        self.d, self.sd = d, np.sqrt(d)
        self.claims.append(Claim("self-adjoint and positive definite", CHECKED, "a positive diagonal"))

    def _dense(self, A):
        self.kind = "dense"
        if not np.all(np.isfinite(A)):
            raise ValueError("a metric or covariance must have finite entries")
        H = A.conj().T                                     # the adjoint: A^T on R^n, A^H on C^n
        what = "Hermitian" if np.iscomplexobj(A) else "symmetric"
        if not symmetric_within_rounding(A):
            raise ValueError(f"a metric or covariance must be {what}")
        A = 0.5 * (A + H)                        # applied as factorised: self-adjoint to the last bit (EXP-036 finding 2)
        try:
            self.L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            raise ValueError("the matrix is not positive definite, so it is not a metric or covariance") from None
        self.A = A
        self.claims.append(Claim("self-adjoint and positive definite", CHECKED, f"{what}; Cholesky succeeded"))

    # P V, P^-1 V ----------------------------------------------------------------------------------------
    def apply(self, V):
        V, one = _cols(V)
        if self.kind == "diagonal":
            out = self.d[:, None] * V
        elif self.kind in ("dense", "sparse"):
            out = np.asarray(self.A @ V)
        elif self.kind == "kronecker":
            out = self.K @ V
        else:
            from .elements import VectorFamily
            out = self.op(VectorFamily(self.base, V.T)).values.T
        return _out(out, one)

    def solve(self, V):
        V, one = _cols(V)
        if self.kind == "diagonal":
            out = V / self.d[:, None]
        elif self.kind == "dense":
            out = sla.cho_solve((self.L, True), V, check_finite=False)
        elif self.kind == "kronecker":
            out = self.K.apply_fn(V, lambda l: 1.0 / l)
        elif self.kind == "sparse":
            if self._lu is None:
                from scipy.sparse.linalg import splu
                self._lu = splu(self._sps.csc_matrix(self.A))
            out = self._lu.solve(V)
        else:
            from .elements import VectorFamily
            out = self.op.inv(VectorFamily(self.base, V.T)).values.T
        return _out(out, one)

    # a factor P = L L^T: L V, L^T V, L^-1 V, L^-T V ------------------------------------------------------
    def _dense_factor(self):
        if self.kind == "dense":
            return self.L
        if self.kind == "operator":
            if self.n > DENSE_FACTOR_MAX:
                raise NotImplementedError(f"a square root of {self.op.name} on R^{self.n} is not available; "
                                          "give a Diagonal, dense Matrix or KroneckerMatrix metric to sample or use "
                                          "orthonormal coordinates")
            if self._L is None:
                A = self.op.matrix()
                self._L = np.linalg.cholesky(0.5 * (A + A.T))
            return self._L
        raise NotImplementedError(f"a square root of a {self.kind} metric is not available (no sparse Cholesky in "
                                  "SciPy): sampling and orthonormal coordinates need a Diagonal, dense or Kronecker one")

    def half(self, V, transpose, inverse):
        V, one = _cols(V)
        if self.kind == "diagonal":
            out = V / self.sd[:, None] if inverse else self.sd[:, None] * V
        elif self.kind == "kronecker":                     # symmetric square root: L = L^T
            out = self.K.apply_fn(V, (lambda l: 1.0 / np.sqrt(l)) if inverse else np.sqrt)
        else:
            L = self._dense_factor()
            if inverse:
                adj = "C" if np.iscomplexobj(L) else "T"           # L^-H on C^n
                out = sla.solve_triangular(L, V, lower=True, trans=adj if transpose else "N", check_finite=False)
            else:
                out = (L.conj().T if transpose else L) @ V
        return _out(out, one)

    def diagonal_values(self):
        return self.d if self.kind == "diagonal" else None

    def key(self):
        if self.kind == "diagonal":
            return ("diagonal", self.d.tobytes())
        if self.kind == "dense":
            return ("dense", self.A.tobytes())
        return (self.kind, id(self.op))


class Metric:
    """The metric of a weighted R^n: W given directly (kind "metric") or through its inverse C (kind "covariance")."""

    def __init__(self, op, base, kind):
        if isinstance(op, np.ndarray):                    # C^n: an array (make_metric)
            self.op, self.base, self.kind = op, base, kind
            self.ops = _Ops(op, base)
            return
        from ..operators.inverses import Inverse
        if isinstance(op, Inverse):                       # a metric C^-1 is a covariance C, and vice versa
            op, kind = op.A, ("covariance" if kind == "metric" else "metric")
        if op.domain != base or op.codomain != base:
            raise ValueError(f"the {kind} must be an operator on the standard {base}; {op.name} maps {op.domain} to "
                             f"{op.codomain}")
        self.op, self.base, self.kind = op, base, kind
        self.ops = _Ops(op, base)

    def apply(self, V):                                    # W V
        return self.ops.apply(V) if self.kind == "metric" else self.ops.solve(V)

    def solve(self, V):                                    # W^-1 V
        return self.ops.solve(V) if self.kind == "metric" else self.ops.apply(V)

    def to_orthonormal(self, V):
        """L_W^T V, W = L_W L_W^T (^H on C^n). For a covariance C = L L^T, W = C^-1 and L_W = L^-T, so this is L^-1 V."""
        if self.kind == "metric":
            return self.ops.half(V, transpose=True, inverse=False)
        return self.ops.half(V, transpose=False, inverse=True)

    def from_orthonormal(self, Z):
        """L_W^-T Z (for a covariance: L Z)."""
        if self.kind == "metric":
            return self.ops.half(Z, transpose=True, inverse=True)
        return self.ops.half(Z, transpose=False, inverse=False)

    def weights(self):
        d = self.ops.diagonal_values()
        if d is None:
            return None
        return d if self.kind == "metric" else 1.0 / d

    @property
    def operator_kind(self):
        return self.ops.kind

    @property
    def claims(self):
        return [Claim(f"the {self.kind} ({self.ops.kind}) is {c.statement}", c.status, c.reason) for c in self.ops.claims]

    def same(self, other):
        if self.weights() is not None and other.weights() is not None:
            a, b = self.weights(), other.weights()
            return a is b or np.array_equal(a, b)
        return self.kind == other.kind and self.ops.key() == other.ops.key()
