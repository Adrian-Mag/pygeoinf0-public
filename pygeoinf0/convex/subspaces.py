"""Linear subspaces of a Hilbert space as convex sets (plan 067 P3, from relations/subspaces.py; review B1 and L5a): the
dom, ran, ker and mul of linear relations, and the subspace parts of noise sets (unknown origin times: span of the
per-event indicators).

    LinearSubspace(space)   a linear subspace (a ConvexSet): member, perp() = its orthogonal complement (closed), and
                            support(v) = 0 on the annihilator, +inf elsewhere (that of its closure); e.g. the dense
                            domain D(A) of an unbounded closed operator (relations/linear.py)
    ClosedSubspace(space)   a closed linear subspace: also project(x), the orthogonal projection, and projector()

    Whole(X)          X itself
    Zero(X)           {0}
    Span(vectors)     the one finite-dimensional subspace (plan 068 P6): the span of finitely many elements (a Family,
                      a VectorFamily or a list; with space=M also bank functions and callables), closed; its Gram,
                      coefficient space, synthesis and analysis maps (operators/galerkin.py), and the orthogonal
                      projection. Its approximation role is operators/galerkin.Discretisation(V)
    Complement(S)     the orthogonal complement of a Span: finite codimension (ker G = (span of the representers)^perp)

Every closed subspace has member(x) -> Truth (decided to rounding: UNKNOWN between 1e-10 and 1e-7 of the scale), `x in
S`, project(x) (the orthogonal projection, in the space's inner product), perp(), dim and support(v) (an Enclosure: 0
where v's component in S computes to exactly 0, +inf where it exceeds the rounding level, 1e-7 ||v||, and [0, +inf]
between, since 0 is in S: a component at the rounding level may be rounding or a true one, whose support is +inf; plan
071 P1, Mac review item 2: Span([e1]).support((1e-10, 1, 0)) was 0 EXACT). Norms of tiny elements are taken after an
exact scaling by a power of 2, so they do not underflow (Whole(R^3).support((1e-300, 0, 0)) is +inf, not 0).
Projector(S) is the projection as a self-adjoint LinearOperator (here until plan 068 places it among the operators).
"""
import math

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..core.claims import CHECKED, DERIVED, THEOREM, Claim, Claims
from ..spaces.elements import Family, Function, Vector, VectorFamily
from ..core.errors import Refused, SpaceMismatch, Unsupported
from ..core.logic import Truth, decide
from ..core.numbers import Enclosure
from ..spaces.base import Euclidean
from ..operators.galerkin import Synthesis
from ..operators.linear import LinearOperator, synthesise
from .sets import ConvexSet, exact
from ..spaces.identity import require

EPS = np.finfo(float).eps    # a Span's dimension: correlation eigenvalues at most eps k times the largest are dropped
RCOND = 1e-12                # independent generators: the correlation matrix's reciprocal condition at least this
_UNSET = object()
_TINY, _UP = 2.0 ** -500, 2.0 ** 1000     # norms below _TINY are taken of x * _UP (exact: a power of 2), not underflowed


def _single(x):
    return isinstance(x, (Function, Vector))


def _elements(space, basis):
    """Elements of the space from elements, bank functions (made elements of the space) and callables, with their
    sources (the bank functions and callables, None for given elements): Span(basis, space=M)."""
    items, sources = [], []
    for b in basis:
        if hasattr(b, "space") and not (hasattr(b, "element") and hasattr(b, "support")):
            require(b.space, space, f"{getattr(b, 'name', 'a generator')} is in {b.space}, not in {space}",
                    getattr(b, "name", None))
            items.append(b)
            sources.append(b.rep if isinstance(b, Function) and b._src is not None else None)
        elif hasattr(b, "element") and hasattr(b, "support"):            # a bank function
            items.append(b.element(space))
            sources.append(b)
        elif callable(b):
            items.append(space(b))
            sources.append(b)
        else:
            raise TypeError(f"a generator is an element of {space}, a bank function or a callable, not "
                            f"{type(b).__name__}")
    return items, sources


def as_family(space, items):
    """A Family or VectorFamily of the space from a family or a list of elements."""
    if isinstance(items, (Family, VectorFamily)):
        return items
    items = list(items)
    if isinstance(space, Euclidean):
        return space.family(np.array([space(np.asarray(getattr(v, "values", v), float)).values for v in items])
                            .reshape(len(items), space.dim), name="V")
    return space.family(items, name="V")


def _unit(space, x):
    """(x scaled by a power of 2, its norm): the norm of x, or of 2^1000 x when ||x|| < 2^-500, so that squares do not
    underflow (a positive scaling is exact and changes neither membership nor the sign of a support); (None, 0.0) when
    x is 0."""
    n = space.norm(x)
    if n < _TINY:
        x = x * _UP
        n = space.norm(x)
        if n == 0:
            return None, 0.0
    return x, n


def _zero_like(space, x):
    if isinstance(x, VectorFamily):
        return VectorFamily(space, np.zeros_like(x.values), name="0")
    if isinstance(x, Family):
        return x * 0.0
    return space.zero


class LinearSubspace(ConvexSet):
    """A linear subspace of a Hilbert space (module docstring): a convex set with 0 in it. Subclasses give member and
    perp(); a closed one is a ClosedSubspace."""

    def __init__(self, space, name="S"):
        super().__init__(space, name)

    def _check(self, x):
        if getattr(x, "space", None) != self.space:
            raise SpaceMismatch(f"{getattr(x, 'name', 'the argument')} is not an element of {self.space}")

    def perp(self):
        raise Unsupported(f"the orthogonal complement of {self.name} is not computed here")

    def support(self, v):
        """sup over the subspace of <v, x>: 0 if v is orthogonal to it, +inf otherwise. 0 only where v's component in
        the subspace (in its closure) computes to exactly 0; +inf where it exceeds the rounding level (core.logic.decide
        FALSE); between, the Enclosure [0, +inf] with the value +inf: orthogonality undecided to rounding (0 is in the
        subspace). A tolerance never reads as orthogonality (plan 071 P1)."""
        self._check(v)
        v, n = _unit(self.space, v)
        if v is None:
            return exact(0.0, "v = 0")
        r = self.space.norm(self.perp()._off(v))            # v's component in the closure of the subspace
        if r == 0:
            return exact(0.0, "v's component in the subspace computes to exactly 0: <v, x> = 0 on the subspace")
        if decide(r, n) is Truth.FALSE:
            return exact(math.inf, "v not orthogonal to the subspace: <v, t x> grows without bound")
        return Enclosure(math.inf, 0.0, math.inf, DERIVED, f"v's component in the subspace is {r / n:.3g} of ||v||, at "
                         "the rounding level: orthogonality of v undecided (0 if v is orthogonal, +inf if not)")

    def is_empty(self):
        return Truth.FALSE

    @property
    def claims(self):
        return Claims([Claim("linear subspace", THEOREM, "by construction", key="linear subspace"),
                       Claim("convex", THEOREM, "a linear subspace", key="convex")])


class ClosedSubspace(LinearSubspace):
    """A closed linear subspace of a Hilbert space (module docstring): project, perp, projector."""

    def member(self, x):
        """Truth: is x in the subspace (to rounding)?"""
        self._check(x)
        x, n = _unit(self.space, x)
        if x is None:
            return Truth.TRUE
        return decide(self.space.norm(self._off(x)), n)

    def _off(self, x):
        """x - P x, the component of x orthogonal to the subspace."""
        return x - self.project(x)

    def project(self, x):
        raise Unsupported(f"the orthogonal projection onto {self.name} is not computed here")

    def projector(self):
        return Projector(self)

    @property
    def claims(self):
        return Claims([Claim("closed linear subspace", THEOREM, self._why_closed, key="closed"),
                       Claim("convex", THEOREM, "a linear subspace", key="convex")])

    _why_closed = "by construction"


class Whole(ClosedSubspace):
    """The whole space X as a subspace (the set of the prior "none")."""

    _why_closed = "the whole space"

    def __init__(self, space):
        super().__init__(space, name=f"{space!r}")

    def member(self, x):
        return Truth.of(getattr(x, "space", None) == self.space)

    def project(self, x):
        return x

    def _off(self, x):
        return _zero_like(self.space, x)

    def perp(self):
        return Zero(self.space)

    @property
    def dim(self):
        return self.space.dim if isinstance(self.space, Euclidean) else math.inf

    def __eq__(self, other):
        return isinstance(other, Whole) and other.space == self.space

    __hash__ = None

    def __repr__(self):
        return f"Whole({self.space!r})"


class Zero(ClosedSubspace):
    """{0} in X."""

    _why_closed = "a point"

    def __init__(self, space):
        super().__init__(space, name="{0}")

    def member(self, x):
        self._check(x)
        return Truth.of(_unit(self.space, x)[0] is None)

    def project(self, x):
        return _zero_like(self.space, x)

    def _off(self, x):
        return x

    def perp(self):
        return Whole(self.space)

    dim = 0

    def __eq__(self, other):
        return isinstance(other, Zero) and other.space == self.space

    __hash__ = None

    def __repr__(self):
        return f"{{0}} in {self.space!r}"


class Span(ClosedSubspace):
    """span{v_1, ..., v_k}: the one finite-dimensional subspace (plan 068 P6), closed.

    Span(family), Span([v1, v2, ...]), or Span(basis, space=M) with bank functions and callables made elements of M.

    The set role (a grid taken as the model space): member, project (the orthogonal projection), perp, support, and the
    numbers of the span: gram G_ij = <v_i, v_j> (scipy.sparse for many local functions), loads(x) = (<v_i, x>)_i,
    coefficients(x) (c with P x = sum_j c_j v_j), and for independent generators the coefficient space
    Euclidean(k, metric=G) (c . G d = <S c, S d>: coefficient norms are norms in the space), synthesis S, analysis
    S* = G^-1 (<v_i, .>) (P = S S*), element(c) = S c and solve(b) = G^-1 b. The approximation role is
    operators/galerkin.Discretisation(V).

    Independence and dimension, both scale-free (plan 054): the correlation matrix C (the Gram of the generators
    normalised to unit length) is factored once (Cholesky, or a sparse LU) and accepted when its reciprocal condition
    estimate is at least max(RCOND, eps k): then dim = k (CHECKED). Otherwise a zero generator is dropped (with scale=,
    one of norm at most sqrt(eps k) scale: a vector computed as a cancellation, at the rounding of its scale), and
    eigenvalues of C at most eps k times the largest are dropped, the floor below which an eigenvalue of a matrix given
    to double precision has no correct digit: dependent generators, or a combination the Gram matrix does not resolve
    (span{v1, v1 + 1e-9 v2} has dimension 1; span{v1, 1e-9 v2} dimension 2). Coefficients are then those of least
    sum_j ||v_j||^2 c_j^2, and the maps that need independent generators refuse with the numerical rank. .dropped and
    the claims state both counts."""

    _why_closed = "finite-dimensional"

    def __init__(self, vectors, name="span", scale=None, space=None):
        self.sources = None
        if space is not None and not isinstance(vectors, (Family, VectorFamily)):
            vectors, self.sources = _elements(space, vectors)
        items = vectors if isinstance(vectors, (Family, VectorFamily)) else list(vectors)
        if not len(items):
            raise ValueError("a span needs at least one generator")
        own = items.space if isinstance(items, (Family, VectorFamily)) else items[0].space
        if space is not None and own != space:
            raise SpaceMismatch(f"the generators are elements of {own}, not of {space}")
        super().__init__(own, name=name)
        self.vectors = as_family(own, items)
        G = own.inner(self.vectors, self.vectors)
        if sp.issparse(G):
            self.gram = sp.csr_matrix((G + G.T) * 0.5)
        else:
            G = np.atleast_2d(np.asarray(G, float))
            self.gram = 0.5 * (G + G.T)
        self._scale, self._fac, self._eigs, self._S = scale, _UNSET, None, None

    # ---------------------------------------------------------------- independence and dimension
    def _diag(self):
        G = self.gram
        return np.clip(np.asarray(G.diagonal() if sp.issparse(G) else np.diag(G), float), 0.0, None)

    def _live(self):
        k = len(self.vectors)
        return self._diag() > (0.0 if self._scale is None else EPS * k * self._scale ** 2)

    def _factored(self):
        """(solve, why) when the correlation matrix is factored with reciprocal condition >= max(RCOND, eps k), else
        None (cached)."""
        if self._fac is _UNSET:
            self._fac = self._factorise()
        return self._fac

    def _factorise(self):
        k, G = len(self.vectors), self.gram
        if not self._live().all():
            return None
        floor = max(RCOND, EPS * k)
        dinv = 1.0 / np.sqrt(self._diag())
        if sp.issparse(G):
            C = sp.csc_matrix(sp.diags(dinv) @ G @ sp.diags(dinv))
            try:
                lu = spla.splu(C)
            except RuntimeError:
                return None
            piv = np.abs(lu.U.diagonal())
            ratio = piv.min() / piv.max()
            if not ratio >= floor:
                return None
            inner, why = lu.solve, f"sparse LU of the correlation matrix, pivot ratio {ratio:.3g} >= {floor:.3g}"
        else:
            C = G * np.outer(dinv, dinv)
            try:
                cf = sla.cho_factor(C, lower=True)
            except np.linalg.LinAlgError:
                return None
            rc = float(sla.lapack.dpocon(cf[0], np.abs(C).sum(axis=0).max(), uplo="L")[0])
            if not rc >= floor:
                return None
            inner = lambda b: sla.cho_solve(cf, b)                                     # noqa: E731
            why = (f"Cholesky of the correlation matrix, reciprocal condition estimate {rc:.3g} >= {floor:.3g} "
                   "(LAPACK dpocon, 1-norm)")

        def solve(b):
            b = np.asarray(b, float)
            d = dinv.reshape((-1,) + (1,) * (b.ndim - 1))
            return d * inner(d * b)
        return solve, why

    def _eig(self):
        """The eigen-route of the dimension (the generators not factored): (V, lam) with G-orthonormal coordinates."""
        if self._eigs is None:
            if sp.issparse(self.gram):
                raise Unsupported(f"{self.name}: the {len(self.vectors)} generators with a sparse Gram are not "
                                  "independent to the factor's floor, and their numerical rank is not computed here "
                                  "(a dense eigendecomposition): drop dependent generators")
            k = len(self.vectors)
            live = self._live()
            d = np.sqrt(self._diag()[live])
            lam, U = (np.linalg.eigh(self.gram[np.ix_(live, live)] / np.outer(d, d)) if live.any()
                      else (np.zeros(0), np.zeros((0, 0))))
            self.cut = EPS * k * max(lam.max(initial=0.0), 1e-300)
            keep = lam > self.cut
            V = np.zeros((k, int(keep.sum())))
            V[live] = U[:, keep] / d[:, None]            # G = D^(1/2) C D^(1/2): coordinates for the original generators
            self._eigs = (V, lam[keep], int((~live).sum()), int((~live).sum() + (~keep).sum()))
        return self._eigs

    @property
    def dim(self):
        return len(self.vectors) if self._factored() is not None else len(self._eig()[1])

    @property
    def zero_generators(self):
        return 0 if self._factored() is not None else self._eig()[2]

    @property
    def dropped(self):
        return 0 if self._factored() is not None else self._eig()[3]

    @property
    def claims(self):
        out = list(super().claims)
        k, fac = len(self.vectors), self._factored()
        if fac is not None:
            out.append(Claim(f"the {k} generators are independent", CHECKED, fac[1], key="rank"))
            out.append(Claim("coefficient norms are norms in the space", DERIVED, "the coefficient space carries the "
                             "Gram G as its metric: c . G d = <S c, S d>", key="metric"))
        elif self.dropped:
            z = self.zero_generators
            out.append(Claim(f"dimension {self.dim} of {k} generators: " +
                             (f"{z} zero generator(s) dropped; " if z else "") +
                             f"{self.dropped - z} correlation eigenvalue(s) (generators normalised to unit length) "
                             f"<= eps k max = {self.cut:.3g} dropped (no correct digit in double precision: dependent "
                             "generators, or a combination the Gram matrix does not resolve)", DERIVED,
                             "the rounding floor, scale-free (plan 054)", key="dimension"))
        out.append(Claim("P is the orthogonal projection onto the span", DERIVED, "P x = sum_j c_j v_j with G c = "
                         "(<v_j, x>)_j (least-norm c for dependent generators): x - P x is orthogonal to every "
                         "generator", key="projection"))
        return Claims(out)

    # ---------------------------------------------------------------- numbers of the span
    def loads(self, x):
        """(<v_i, x>)_i: shape (k,) for an element, (k, m) for a family of m."""
        self._check(x)
        b = self.space.inner(self.vectors, x)
        b = np.asarray(b.toarray() if sp.issparse(b) else b, float)
        k = len(self.vectors)
        return b.reshape(k) if _single(x) else b.reshape(k, -1)

    def coefficients(self, x):
        """c with P x = sum_j c_j v_j: G^-1 loads(x) for independent generators, else those of least
        sum_j ||v_j||^2 c_j^2. Shape (k,) for an element, (k, m) for a family of m."""
        b = self.loads(x)
        fac = self._factored()
        if fac is not None:
            return fac[0](b)
        V, lam = self._eig()[:2]
        c = V @ ((V.T @ b.reshape(len(b), -1)) / lam[:, None])
        return c.reshape(b.shape)

    def project(self, x):
        self._check(x)
        if self.dim == 0:
            return _zero_like(self.space, x)
        return synthesise(self.vectors, self.coefficients(x).reshape(len(self.vectors), -1), single=_single(x))

    def _require_independent(self, what):
        if self._factored() is None:
            k = len(self.vectors)
            try:
                r = self.dim
                why = (f"numerical rank {r}" if r < k else f"numerical rank {r}, but a correlation matrix too badly "
                       f"conditioned to solve (reciprocal condition below {max(RCOND, EPS * k):.3g})")
            except Unsupported:
                why = "a sparse Gram the factor finds singular"
            raise Refused(f"{self.name}: {what} needs independent generators; the {k} generators have {why}. Choices: "
                          "drop dependent generators, or use project and coefficients (least-norm coefficients)")

    @property
    def coefficient_space(self):
        """Euclidean(k, metric=G): <c, d> = c . G d = <S c, S d> (independent generators)."""
        self._require_independent("the coefficient space")
        if getattr(self, "_C", None) is None:
            self._C = Euclidean(len(self.vectors), metric=self.gram)
        return self._C

    @property
    def synthesis(self):
        """S: c -> sum_i c_i v_i from the coefficient space (independent generators)."""
        self._require_independent("the synthesis map")
        if self._S is None:
            self._S = Synthesis(self)
        return self._S

    @property
    def analysis(self):
        """S* = G^-1 (<v_i, .>): the coefficients of P x (independent generators)."""
        return self.synthesis.adjoint

    def solve(self, b):
        """G^-1 b for b of shape (k,) or (k, m) (independent generators)."""
        self._require_independent("solving with the Gram")
        return self._factored()[0](b)

    def element(self, c):
        """sum_i c_i v_i for a coefficient vector c (an array or a Vector of the coefficient space)."""
        c = c if isinstance(c, Vector) else self.coefficient_space(np.asarray(c, float))
        return self.synthesis(c)

    def basis(self):
        """An orthonormal basis of the span (a family of dim members)."""
        V, lam = self._eig()[:2]
        return synthesise(self.vectors, V / np.sqrt(lam), single=False)

    def matrix_basis(self):
        """Standard coordinates (n, dim) of an orthonormal basis (Euclidean spaces)."""
        if not isinstance(self.space, Euclidean):
            raise TypeError("matrix_basis is for subspaces of R^n")
        return np.atleast_2d(self.basis().values).T.reshape(self.space.dim, self.dim)

    def perp(self):
        return Complement(self)

    def __eq__(self, other):
        if not isinstance(other, Span) or other.space != self.space or other.dim != self.dim:
            return False
        return all(self.member(v) is Truth.TRUE for v in other.vectors)

    __hash__ = None

    def __repr__(self):
        kind = ", sparse Gram" if sp.issparse(self.gram) else ""
        return f"{self.name}: a span of dimension {self.dim} in {self.space!r}{kind}"


class Complement(ClosedSubspace):
    """The orthogonal complement of a Span (finite codimension), e.g. ker G = (span of G's representers)^perp."""

    _why_closed = "an orthogonal complement"

    def __init__(self, span, name="complement"):
        if not isinstance(span, Span):
            raise TypeError("Complement takes a Span (the complement of Whole is Zero and conversely)")
        super().__init__(span.space, name=name)
        self.of = span

    def project(self, x):
        self._check(x)
        return x - self.of.project(x)

    def _off(self, x):
        return self.of.project(x)

    def perp(self):
        return self.of

    @property
    def dim(self):
        return self.space.dim - self.of.dim if isinstance(self.space, Euclidean) else math.inf

    def __eq__(self, other):
        return isinstance(other, Complement) and other.of == self.of

    __hash__ = None

    def __repr__(self):
        return f"{self.name}: the orthogonal complement of a span of dimension {self.of.dim} in {self.space!r}"


def span_sum(a, b):
    """The sum of two subspaces of one space, when it is computable here."""
    if isinstance(a, Zero):
        return b
    if isinstance(b, Zero):
        return a
    if isinstance(a, Whole) or isinstance(b, Whole):
        return Whole(a.space)
    if isinstance(a, Span) and isinstance(b, Span):
        return Span(as_family(a.space, list(a.vectors) + list(b.vectors)), name=f"{a.name} + {b.name}")
    raise Unsupported(f"the sum of {a!r} and {b!r} is not implemented (finite spans only)")


def image(T, S):
    """T(S) for a bounded linear T and a subspace S of its domain, when computable: spans map to spans; Whole to ran T
    (T into R^p); the complement of a span to ran(T P), whose Gram matrix is the Schur complement
    [T T*] - [T G*][G G*]^+ [G T*] (DESIGN 2.5 (a))."""
    P = T.codomain
    if isinstance(S, Zero):
        return Zero(P)
    if isinstance(S, Span):
        return Span(T(S.vectors), name=f"{T.name}({S.name})")
    if not isinstance(P, Euclidean):
        raise Unsupported(f"the image of {S!r} under {T.name} is computed for maps into R^p")
    basis = P.family(np.eye(P.dim), name="e")
    reps = T.adjoint(basis)                                   # T* e_l: the representers, up to P's metric
    if not isinstance(S, (Whole, Complement)):
        raise Unsupported(f"the image of {S!r} under {T.name}")
    scale = float(np.max(np.atleast_1d(P.norm(T(reps))), initial=0.0))     # rank relative to T's own size
    return Span(T(S.project(reps)), name=f"{T.name}({S.name})", scale=scale)


def complement_basis(S):
    """A family whose span is S^perp, for a subspace S of R^n (W-orthonormal for a span)."""
    Y = S.space
    if not isinstance(Y, Euclidean):
        raise Unsupported("bases of complements are computed in R^n")
    if isinstance(S, Complement):
        return S.of.vectors
    if isinstance(S, Zero):
        return Y.family(np.eye(Y.dim), name="e")
    if isinstance(S, Whole):
        return Y.family(np.zeros((0, Y.dim)), name="∅")
    Q = Y.to_orthonormal(S.matrix_basis())
    Z, _ = np.linalg.qr(np.c_[Q, np.eye(Y.dim)])
    return Y.family(Y.from_orthonormal(Z[:, Q.shape[1]:Y.dim]).T, name="basis of the complement")


def basis_matrix(S):
    """Standard coordinates (n, dim S) of a basis of a subspace S of R^n."""
    Y = S.space
    if isinstance(S, Whole):
        return np.eye(Y.dim)
    if isinstance(S, Zero):
        return np.zeros((Y.dim, 0))
    if isinstance(S, Span):
        return S.matrix_basis()
    return complement_basis(S.of).values.T


def preimage(T, S):
    """T^-1(S) = {x : T x in S} = (T*(S^perp))^perp for a bounded linear T and a closed subspace S of its codomain."""
    if isinstance(S, Whole):
        return Whole(T.domain)
    V = complement_basis(S)
    return Complement(Span(T.adjoint(V), name=f"{T.name}*({S.name}^⊥)"), name=f"{T.name}⁻¹({S.name})")


class Projector(LinearOperator):
    """The orthogonal projection onto a closed subspace, as an operator on its space: self-adjoint, idempotent, norm <= 1."""

    why_linear = "an orthogonal projection"

    def __init__(self, subspace, name=None):
        super().__init__(subspace.space, subspace.space, name=name or f"P[{subspace.name}]")
        self.subspace = subspace
        self._claims = [Claim("linear", THEOREM, "an orthogonal projection", key="linear"),
                        Claim("bounded, norm <= 1", THEOREM, "||P x|| <= ||x||", key="bounded")]
        self._sa = (True, THEOREM, "an orthogonal projection")
        self._pos = ("psd", THEOREM, "<P x, x> = ||P x||^2 >= 0")

    def _apply(self, x):
        return self.subspace.project(x)

    def norm_bound(self):
        """||P|| <= 1 (an orthogonal projection; its claim "bounded"), the protocol operators/error.norm_bound reads."""
        return 1.0

    def _make_adjoint(self):
        return self
