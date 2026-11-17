"""Complex spaces (plan 020 P3a; experiments/EXP-035-mineral-exploration/COMPLEX.md): the complex structure of
Euclidean(n, field="C") (plan 068 P1: one Euclidean class, one metric code), complexifications, complex matrices.

    Euclidean(n, field="C")                  C^n with <x, y> = sum_i x_i conj(y_i)
    Euclidean(n, field="C", weights=w)       W = diag(w), w > 0
    Euclidean(n, field="C", metric=W)        W a Hermitian positive definite matrix (n x n)
    Euclidean(n, field="C", covariance=C)    W = C^-1: circular complex Gaussian noise with covariance C (solved with by
                                             its Cholesky factor, not inverted: spaces/metrics.py)

The complex inner product is linear in the first argument and conjugate-linear in the second (the author's choice):
<x, y> = y^H W x. Elements are pg.Vector objects holding complex arrays (native storage, the hybrid of COMPLEX.md
answer 3). As a space for the rest of pygeoinf0 it is the real Hilbert space C^n_R with <x, y>_R = Re <x, y> and the
complex structure J x = i x, so adjoints, least squares, Tikhonov and Bayesian formulas, and pg.port.taylor_test work on it
unchanged: for a complex-linear map the real adjoint is the complex adjoint, and for a real-linear map (a real model
with complex data) it is the only adjoint there is. c.cinner(x, y) gives the complex inner product.

Real coordinates (for code that needs a real basis, e.g. the data-space solves of pg.optimisers.gauss_newton): real_dim = 2n,
to_real(v) = (Re v, Im v), from_real(z), real_basis() = (e_1, ..., e_n, i e_1, ..., i e_n).

    Complexified(H)                          H_C = H (+) H, the complexification of any real Hilbert space (plan 020 P3b):
                                             real and imaginary parts realised by H's own backend (the adaptive fitters
                                             unchanged: the hybrid of COMPLEX.md), J(u, v) = (-v, u), the same complex
                                             inner product convention; H_C(f) for a complex function f, H_C.values(z, x)
                                             its complex values, H_C.scale(a, z) for complex a, H_C.lift(A) the
                                             complex-linear extension of a real operator A: H -> K
    ComplexMatrix(A, X, Y)                   z -> A z between complex Euclidean spaces: complex-linear, adjoint
                                             W_X^-1 A^H W_Y (plan 020 P3c)
    Complexified(M), M a cell-realised L2      the same interface with native complex storage (the hybrid): one complex
                                             array of cell averages (CellsComplexified, realisations/piecewise_constant.py)
"""
import numpy as np

from ..core.claims import THEOREM, Claim, Claims
from .direct_sum import DirectSum
from .elements import Function, Vector, VectorFamily
from .base import CoordinateSpace, Euclidean, L2


class _ComplexEuclidean(CoordinateSpace):
    """The class of Euclidean(n, field="C"): C^n with a Hermitian metric (module docstring). Its constructor, elements and
    metric are those of every CoordinateSpace (spaces/base.py, spaces/metrics.py); what this class adds is what exists
    only over C: the complex structure J, the complex inner product cinner, real coordinates, and circular Gaussian
    measures. It is not an instance of Euclidean (the routes for real coordinates refuse it, as before)."""

    dtype = complex
    field = "C"

    @property
    def real_dim(self):
        return 2 * self.dim

    @property
    def metric(self):
        """W as an array (operators on C^n are not built: the operator classes act on real coordinates)."""
        return None if self._metric is None else (self.metric_matrix() if self.weights is None else
                                                  np.diag(self.weights))

    @property
    def covariance(self):
        """W^-1 as an array: the covariance of the coordinates of the space's circular standard Gaussian (a stored
        covariance as given; a metric's inverse by solving with its Cholesky factor)."""
        m = self._metric
        if m is None:
            return np.eye(self.dim, dtype=complex)
        if m.weights() is not None:
            return np.diag(1.0 / m.weights()).astype(complex)
        if m.kind == "covariance":
            return np.array(m.op, complex)
        return self.solve_metric(np.eye(self.dim, dtype=complex))

    def random(self, size=None, seed=None, name="v"):
        """Standard circular complex normal numbers (real and imaginary parts N(0, 1/2)): directions for checks."""
        rng = np.random.default_rng(seed)
        shape = (self.dim,) if size is None else (size, self.dim)
        z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        return Vector(self, z, name=name) if size is None else VectorFamily(self, z, name=name)

    def standard_gaussian(self):
        """The standard circular Gaussian CN(0, W^-1) of the coordinates (probability/circular.py): the noise whose
        covariance the metric carries (Euclidean(n, field="C", covariance=S)), as for a real Euclidean space. Another
        covariance or a mean: pygeoinf0.probability.circular.circular(X, S, mean) (X.gaussian() refuses since plan 069
        P6)."""
        from ..probability.circular import circular
        w = self.weights
        covariance = 1.0 if self.is_standard else (1.0 / w if w is not None else self.covariance)
        return circular(self, covariance, None, "CN(0, W^-1)")

    def J(self, x):
        """The complex structure: x -> i x (a real-linear isometry with J^2 = -I)."""
        return Vector(self, 1j * x.values, name=f"i·{x.name}")

    # ------------------------------------------------------------------------------------------- inner products
    def cinner(self, x, y):
        """The complex inner product <x, y> = y^H W x (linear in x, conjugate-linear in y)."""
        return complex(np.vdot(self._values(y), self.apply_metric(self._values(x))))

    def inner(self, x, y):
        """The real inner product of C^n_R: Re <x, y> (module docstring)."""
        X, Y = self._values(x), self._values(y)
        if X.ndim == 1 and Y.ndim == 1:
            return float(np.real(np.vdot(Y, self.apply_metric(X))))
        return np.real(np.atleast_2d(X).conj() @ self.apply_metric(np.atleast_2d(Y).T))

    def norm(self, x):
        if isinstance(x, VectorFamily):
            return np.sqrt(np.maximum(np.real(np.einsum("ij,ji->i", x.values.conj(), self.apply_metric(x.values.T))),
                                      0.0))
        return float(np.sqrt(max(self.inner(x, x), 0.0)))

    # ------------------------------------------------------------------------------------------- real coordinates
    def to_real(self, V):
        V = np.asarray(V)
        return np.concatenate([V.real, V.imag], axis=-1)

    def from_real(self, z):
        z = np.asarray(z, float)
        return z[..., :self.dim] + 1j * z[..., self.dim:]

    def real_basis(self):
        eye = np.eye(self.dim)
        return [Vector(self, e) for e in eye] + [Vector(self, 1j * e) for e in eye]

    @property
    def claims(self):
        cl = [Claim("Hilbert space (complete)", THEOREM, "finite-dimensional"),
              Claim("complex inner product linear in the first argument, conjugate-linear in the second",
                    THEOREM, "<x, y> = y^H W x, W Hermitian positive definite", key="convention"),
              Claim("as a real space: Re <x, y>, with J x = i x", THEOREM,
                    "the realification C^n_R = R^2n; J is a real-linear isometry with J^2 = -I",
                    key="complex structure")]
        if self._metric is not None:
            cl += self._metric.claims
        return Claims(cl)


def ComplexMatrix(A, domain=None, codomain=None, name="A"):
    """The complex-linear operator z -> A z between complex Euclidean spaces (default C^n and C^m with the standard
    metrics). Its adjoint in the spaces' inner products is w -> W_X^-1 A^H W_Y w (the complex adjoint, which is also
    the real one: <A x, y>_R = Re <A x, y>). Claims: linear and complex-linear (theorems)."""
    from ..operators.linear import LinearOperator
    A = np.asarray(A, complex)
    if A.ndim != 2:
        raise ValueError("a complex matrix is two-dimensional")
    X = domain if domain is not None else Euclidean(A.shape[1], field="C")
    Y = codomain if codomain is not None else Euclidean(A.shape[0], field="C")
    for S, k in ((X, A.shape[1]), (Y, A.shape[0])):
        if not (isinstance(S, CoordinateSpace) and S.field == "C") or S.dim != k:
            raise ValueError(f"{name} is {A.shape[0]} x {A.shape[1]}: its spaces are complex Euclidean of those sizes")
    op = LinearOperator(X, Y, lambda x: (Vector(Y, A @ x.values) if isinstance(x, Vector)
                                         else VectorFamily(Y, (A @ x.values.T).T)),
                        adjoint=lambda w: (Vector(X, X.solve_metric(A.conj().T @ Y.apply_metric(w.values)))
                                           if isinstance(w, Vector) else
                                           VectorFamily(X, X.solve_metric(A.conj().T @ Y.apply_metric(w.values.T)).T)),
                        name=name)
    op._claims = [Claim("linear", THEOREM, "a matrix", key="linear"),
                  Claim("complex-linear (commutes with J)", THEOREM, "multiplication by a complex matrix",
                        key="complex-linear")]
    op.cmatrix = A
    return op


class Complexified(DirectSum):
    """H_C = H (+) H with J(u, v) = (-v, u): the complexification of a real Hilbert space (module docstring)."""

    field = "C"

    def __new__(cls, H=None, *args, **kwargs):
        if cls is Complexified and _cell_realised(H):
            return CellsComplexified(H)               # native complex storage for cell values (the hybrid)
        return super().__new__(cls)

    def __init__(self, H):
        super().__init__({"re": H, "im": H})
        self.real_space = H

    def element(self, f, name="z"):
        """From a complex function of the coordinates, a dict {"re": u, "im": v} of elements, or a pair (u, v)."""
        H = self.real_space
        if isinstance(f, dict):
            return super().element(f, name=name)
        if isinstance(f, tuple):
            return super().element({"re": f[0], "im": f[1]}, name=name)
        if callable(f):
            return super().element({"re": H(lambda *X: np.real(f(*X))), "im": H(lambda *X: np.imag(f(*X)))}, name=name)
        raise TypeError("a complex element is a complex function, {'re': u, 'im': v} or (u, v)")

    def __call__(self, f, name="z"):
        return self.element(f, name=name)

    def parts(self, z):
        return self.part(z, "re"), self.part(z, "im")

    def J(self, z):
        u, v = self.parts(z)
        return super().element({"re": -1.0 * v, "im": u}, name=f"i·{z.name}")

    def scale(self, a, z):
        """(a + i b) z = a z + b J z."""
        a = complex(a)
        return a.real * z + a.imag * self.J(z)

    def cinner(self, x, y):
        """<x, y> = <x, y>_R - i <J x, y>_R: linear in x, conjugate-linear in y."""
        return complex(self.inner(x, y), -self.inner(self.J(x), y))

    def values(self, z, *X):
        """The complex values of z at points (the real space's own evaluation of each part)."""
        u, v = self.parts(z)
        return np.asarray(u(*X)) + 1j * np.asarray(v(*X))

    def lift(self, A, codomain=None):
        """A_C(u + i v) = A u + i A v for a real operator A: H -> K; complex-linear, its adjoint is (A*)_C."""
        from ..operators.linear import LinearOperator
        K = codomain if codomain is not None else Complexified(A.codomain)
        out = LinearOperator(self, K, lambda z: K.element({"re": A(self.part(z, "re")), "im": A(self.part(z, "im"))}),
                             adjoint=lambda w: self.element({"re": A.adjoint(K.part(w, "re")),
                                                             "im": A.adjoint(K.part(w, "im"))}),
                             name=f"{A.name}_C")
        out._claims = [Claim("linear", THEOREM, "the complexification of a linear operator", key="linear"),
                       Claim("complex-linear (commutes with J)", THEOREM, "it acts on the real and imaginary parts alike",
                             key="complex-linear")]
        return out

    @property
    def claims(self):
        return Claims(list(super().claims) + [
            Claim("complex inner product linear in the first argument, conjugate-linear in the second", THEOREM,
                  "<x, y> = <x, y>_R - i <J x, y>_R", key="convention"),
            Claim("the complexification H (+) H with J(u, v) = (-v, u)", THEOREM, "J^2 = -I, J an isometry",
                  key="complex structure")])

    def __eq__(self, other):
        return _same_complexification(self, other)

    def __hash__(self):
        return hash(("complexified", hash(self.real_space)))

    def __repr__(self):
        return f"{self.real_space} ⊗ C"


def _same_complexification(H, K):
    """H_C == K_C iff H == K, whatever realises them (decision Q4): Complexified (pairs of real elements) and
    CellsComplexified (native complex cell values) are two realisations of one complexification."""
    return isinstance(K, (Complexified, CellsComplexified)) and K.real_space == H.real_space


def _cells():
    from ..realisations import piecewise_constant as cells  # the realisation layer, imported when used (core layering)
    return cells


def _cell_realised(H):
    from ..realisations.registry import provides
    from .base import L2
    return type(H) is L2 and provides(getattr(H, "backend", None), "cell_values")      # real cell values: type(H) is L2


class CellsComplexified(L2):
    """The complexification of a cell-realised L2 with native complex storage (module docstring): the interface of
    Complexified (element from a complex function, parts, J, scale, cinner, values, lift), one complex array of cell
    averages per element, the real inner product Re sum vol f conj(g)."""

    field = "C"

    def __init__(self, H):
        from ..realisations.piecewise_constant import ComplexCellsBackend
        super().__init__(H.measure, tol=H.tol, realisation=H.backend.cells)
        self.backend = ComplexCellsBackend(H.measure, H.tol, H.backend.cells)
        self.realisation = self.backend.realisation_name
        self.real_space = H

    # ------------------------------------------------------------------------------------------- elements
    def element(self, f, *, breaks=None, name=None, resolution=None):
        """From a complex function of the coordinates, a complex number, {"re": u, "im": v} or (u, v) with u, v in
        the real space, or an element of this space. resolution= as for real spaces (core/fits.py)."""
        if isinstance(f, dict):
            f = (f["re"], f["im"])
        if isinstance(f, tuple):
            u, v = (self.real_space.element(w, resolution=resolution) if callable(w) and resolution is not None
                    else self.real_space.element(w) for w in f)
            return self._of(u.realisation.V + 1j * v.realisation.V, u.realisation.err + v.realisation.err, name or "z")
        if isinstance(f, (complex, np.complexfloating)):
            return self._of(np.full((self.backend.cells.n, 1), complex(f)), None, name or f"{f}")
        return super().element(f, breaks=breaks, name=name, resolution=resolution)

    def _cv(self, z):
        return self.element(z).realisation

    def _of(self, V, err, name, space=None):
        """The element of this space (or of the real space) with cell values V and realisation error err."""
        return Function(space or self, pl=_cells().CellValues(V, err), name=name)

    def part(self, z, key):
        R = self._cv(z)
        if key not in ("re", "im"):
            raise KeyError(f"the parts of a complex element are 're' and 'im', not {key!r}")
        V = R.V.real if key == "re" else R.V.imag
        return self._of(V.copy(), R.err, f"{key} {z.name}", self.real_space)

    def parts(self, z):
        return self.part(z, "re"), self.part(z, "im")

    def J(self, z):
        R = self._cv(z)
        return self._of(1j * R.V, R.err, f"i·{z.name}")

    def scale(self, a, z):
        R = self._cv(z)
        a = complex(a)
        return self._of(a * R.V, abs(a) * R.err, f"({a})·{z.name}")

    def cinner(self, x, y):
        """<x, y> = sum vol x conj(y): linear in x, conjugate-linear in y."""
        return complex(self.backend.cgram(self._cv(x), self._cv(y))[0, 0])

    def values(self, z, *X):
        return np.asarray(self.backend.evaluate(self._cv(z), *[np.asarray(x, float) for x in X])[..., 0])

    def lift(self, A, codomain=None):
        """A_C(u + i v) = A u + i A v for a real operator A on the real space; complex-linear, adjoint (A*)_C."""
        return Complexified.lift(self, A, codomain)

    def random(self, size=None, seed=None, name="z"):
        """Standard circular complex normal cell values (real and imaginary parts N(0, 1/2)): directions for checks
        (pg.port.taylor_test, pg.port.complex_linearity), not draws from a stated measure (complex Gaussian measures: P3c)."""
        from .elements import Family
        rng = np.random.default_rng(seed)
        n, k = self.backend.cells.n, 1 if size is None else int(size)
        V = (rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))) / np.sqrt(2)
        return self._of(V, None, name) if size is None else Family(self, _cells().CellValues(V), name=name)

    @property
    def claims(self):
        return Claims(list(super().claims) + [
            Claim("complex inner product linear in the first argument, conjugate-linear in the second", THEOREM,
                  "<x, y> = sum_i vol_i x_i conj(y_i)", key="convention"),
            Claim("the complexification of the cell-realised L2, stored natively", THEOREM,
                  "J z = i z, J^2 = -I, J an isometry", key="complex structure")])

    def __eq__(self, other):
        return _same_complexification(self, other)

    def __hash__(self):
        return hash(("complexified", hash(self.real_space)))

    def _with_realisation(self, realisation):
        return Complexified(self.real_space.realised(realisation))

    def __repr__(self):
        return f"{self.real_space} ⊗ C"
