"""Linear operators between spaces (functionals.py has the stacks of integrals).

LinearOperator(domain, codomain, apply, adjoint=None)  any linear map you can apply (linearity is asserted, not checked)
Matrix(A)                   a linear map between Euclidean spaces
Identity(space)

Writing:  G(m) or G @ m applies;  G @ H or G(H) composes;  G.adjoint;  A.inverse(solver) or A.inv (inverses.py);  A + B, 2 * A.
Results of G(m) are remembered on m (elements are immutable), so applying G to the same element again is free.
A LinearOperator is a single-valued linear relation defined on all of its domain (core/relation.py, plan 008): G(S) of
a set and G(mu) of a measure push them forward, G @ R composes with any relation, G.converse() is the fibre relation
d -> G^-1(d) (G.inv stays the operator inverse, applied with a solver).
"""
import itertools

import numpy as np

from ..core import estimable
from ..core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN, Claim, Claims, by_rule
from ..spaces.elements import Family, Function, Vector, VectorFamily
from ..core.errors import SpaceMismatch, Unsupported
from ..core.kinds import MEASURE, POINT, stated_kind
from ..core.relation import Relation
from ..spaces.base import Euclidean, HilbertSpace
from ..spaces.identity import require, require_realised


_SA = "self-adjoint"
_POS = ("positive semidefinite", "positive definite")


def class_linearity(op):
    """Why op's action is linear, as its class states it (plan 065 A3), or None. The class that defines the action
    (_apply or __call__, the first one in the method resolution order) must state why_linear itself: a subclass that
    changes the action states nothing by inheritance, and a bare subclass of LinearOperator states nothing at all."""
    for C in type(op).__mro__:
        if C is LinearOperator:
            return None
        own = C.__dict__
        if own.get("why_linear"):
            return own["why_linear"]
        if "_apply" in own or "__call__" in own:
            return None
    return None


def _record(op, keys):
    for c in getattr(op, "_claims", None) or ():
        if c.key in keys:
            return c
    return None


def _set_record(op, keys, claim):
    if not isinstance(getattr(op, "_claims", None), list):
        op._claims = list(getattr(op, "_claims", None) or ())
    op._claims[:] = [c for c in op._claims if c.key not in keys] + ([claim] if claim is not None else [])


def carry_structure(out, A, B):
    """The claims a product A B carries by its structure (plan 065 A5: one helper for every structural shortcut, so that
    Matrix, Diagonal, Kronecker or Gram products keep what Composition derives): A A* and A* A are self-adjoint and
    positive semidefinite. Records already on out are kept."""
    if B is getattr(A, "_adjoint", None) and B is not None:
        if out._sa is None:
            out._sa = (True, DERIVED, f"of the form {A.name} {A.name}*")
        if out._pos is None:
            out._pos = ("psd", DERIVED, f"<{A.name}{A.name}* y, y> = ||{A.name}* y||^2 >= 0")
    return out


class LinearOperator(Relation):
    """A linear map between two spaces: A(x), A @ B (composition), A + B, a * A, A.adjoint, A.inv, A.matrix();
    with claims about linearity, boundedness, self-adjointness and positivity. A relation of point kind (plan 008).

    Linearity: LinearOperator(X, Y, apply) asserts it for your function; a package class states why_linear, the reason
    its action is linear; a subclass of your own states nothing until you assert it (pg.assert_claim(A, "linear", why)).
    Self-adjointness and positivity are claims in the same store (_sa and _pos read and write them, plan 065 A5)."""

    kind = POINT
    why_linear = None
    _ids = itertools.count()

    def __init__(self, domain, codomain, apply=None, adjoint=None, name="A"):
        self.domain, self.codomain, self.name = domain, codomain, name
        self._fn, self._adj_fn = apply, adjoint
        self._id = next(LinearOperator._ids)
        self._adjoint = None
        self._inv = None
        self._claims = [Claim("linear", ASSERTED, "a user-supplied function", key="linear")] if apply is not None \
            else []

    @property
    def _sa(self):
        """(bool, status, reason) of self-adjointness from the claims, or None when nothing is on record."""
        c = _record(self, (_SA,))
        return None if c is None else (c.holds, c.status, c.reason)

    @_sa.setter
    def _sa(self, rec):
        _set_record(self, (_SA,), None if not rec or rec[0] is None else
                    Claim("self-adjoint" if rec[0] else "not self-adjoint", rec[1], rec[2], key=_SA, holds=bool(rec[0])))

    @property
    def _pos(self):
        """("psd" | "pd", status, reason) of positivity from the claims, or None when nothing is on record."""
        c = _record(self, _POS)
        return None if c is None else ("pd" if c.key == _POS[1] else "psd", c.status, c.reason)

    @_pos.setter
    def _pos(self, rec):
        key = None if not rec or rec[0] is None else _POS[1] if rec[0] == "pd" else _POS[0]
        _set_record(self, _POS, None if key is None else Claim(key, rec[1], rec[2], key=key))

    # application ---------------------------------------------------------------------------
    def _apply(self, x):
        if self._fn is None:
            raise NotImplementedError(f"{self.name} has no action")
        return self._fn(x)

    def __call__(self, x):
        if isinstance(x, Relation):
            return self @ x
        k = stated_kind(x)
        if k is MEASURE and hasattr(x, "pushforward"):
            return x.pushforward(self)                  # a Gaussian measure: G(mu) is its push-forward
        if k is not None:
            from ..cells import push
            return push(x, self)                        # a state (a set, an ordering, a measure): pushed forward
        if not hasattr(x, "space"):
            raise SpaceMismatch(f"{self.name} acts on elements of {self.domain}, not on a {type(x).__name__}: make one "
                                f"first, e.g. X(values) for a Euclidean space X or M(f) / M.riesz(set) for a function space M")
        require(x.space, self.domain, f"{self.name} acts on {self.domain}; {getattr(x, 'name', 'the argument')} is in "
                f"{x.space}", f"{self.name}({getattr(x, 'name', 'x')})")          # realised as its domain (Q4)
        memo = x._memo
        hit = memo.get(self._id)
        if hit is None:
            hit = memo[self._id] = estimable.mark(self._apply(x), self)    # it lies in ran A (core/estimable.py)
        return hit

    def __matmul__(self, other):
        if isinstance(other, LinearOperator):
            require(other.codomain, self.domain, f"cannot compose: {other.name} maps into {other.codomain}, {self.name} "
                    f"acts on {self.domain}", f"{self.name} @ {other.name}")
            from .error import carry_factors                   # a shortcut keeps its factors' stated errors (A6)
            out = carry_factors(carry_structure(self._compose(other), self, other), self, other)
            return estimable.within(out, self, other)                                         # ranges by construction
        if isinstance(other, Relation):
            from ..relations.algebra import compose
            return compose(self, other)
        return self(other)

    def _compose(self, other):
        return Composition(self, other)

    # adjoint and inverse ------------------------------------------------------------------------
    @property
    def adjoint(self):
        if self._adjoint is None:
            for sp in (self.domain, self.codomain):
                if not isinstance(sp, HilbertSpace):
                    raise TypeError(f"the adjoint needs Hilbert spaces; {sp} has no inner product "
                                    "(its dual is not identified with the space itself)")
            self._adjoint = self._make_adjoint()
            self._adjoint._adjoint = self
            estimable.adjoint_made(self, self._adjoint)
        return self._adjoint

    def _make_adjoint(self):
        if self._adj_fn is None:
            raise NotImplementedError(f"{self.name}: no adjoint was given")
        return LinearOperator(self.codomain, self.domain, self._adj_fn, self._fn, name=f"{self.name}*")

    def inverse(self, solver=None):
        """The inverse, applied with the given solver, or with the recommended one (see recommend_solver); an Inverse
        (operators/inverses.py)."""
        from .inverses import Inverse
        return Inverse(self, solver)

    @property
    def inv(self):
        if self._inv is None:
            from .inverses import Inverse
            self._inv = Inverse(self)
        return self._inv

    def pseudo_inverse(self, rcond):
        """A⁺, the pseudo-inverse with the threshold you choose (a mathematical choice; A.inv never makes it for you): its
        claims state the threshold and how many eigenvalues it dropped. A non-square operator is refused here, with the
        pseudo-inverses that are defined for it named (plan 065 A12; operators/inverses.py)."""
        from .inverses import pseudo_inverse
        return pseudo_inverse(self, rcond)

    def recommend_solver(self):
        """The solver A.inv would use, with the reason (solvers.recommend, through the Inverse that applies it)."""
        from .inverses import Inverse
        return Inverse(self).recommendation

    def __pow__(self, k):
        if k == -1:
            return self.inv
        if isinstance(k, int) and k >= 1:
            out = self
            for _ in range(k - 1):
                out = out @ self
            return out
        return NotImplemented

    # linear combinations -----------------------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return Sum([(1.0, self), (1.0, other)])

    def __sub__(self, other):
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return Sum([(1.0, self), (-1.0, other)])

    def __neg__(self):
        return Sum([(-1.0, self)])

    def __mul__(self, c):
        if np.ndim(c) == 0 and isinstance(c, (int, float, np.number)):
            out = Sum([(float(c), self)])
            if c > 0:                              # a positive multiple keeps self-adjointness and positivity
                out._sa = self._sa and (self._sa[0], by_rule(self._sa[1]), f"a positive multiple of {self.name}")
                out._pos = self._pos and (self._pos[0], by_rule(self._pos[1]), f"a positive multiple of {self.name}")
            return out
        return NotImplemented

    __rmul__ = __mul__

    # structure ---------------------------------------------------------------------------------
    def matrix(self):
        """The matrix in the standard coordinates (Euclidean domain and codomain only)."""
        if not (isinstance(self.domain, Euclidean) and isinstance(self.codomain, Euclidean)):
            raise TypeError(f"{self.name} is not a map between Euclidean spaces, so it has no matrix")
        _dense_allowed(self)
        return np.array(self(self.domain.family(np.eye(self.domain.dim), name="I")).values.T)

    def sparse(self):
        """A scipy.sparse matrix of this map in the standard coordinates, when its structure gives one at no cost (sparse
        matrices, diagonals, identities and their sums); None otherwise. Products are not multiplied out."""
        return None

    def is_selfadjoint(self):
        """(True/False/None, status, reason)."""
        if self._sa is not None:
            return self._sa
        if isinstance(self.domain, Euclidean) and self.domain == self.codomain:
            from ..settings import settings
            if max(self.domain.dim, self.codomain.dim) > settings.dense_max:
                return (None, UNKNOWN, f"not checked: n > settings.dense_max = {settings.dense_max} (no matrix formed)")
            S = self.domain.apply_metric(self.matrix())
            ok = bool(np.allclose(S, S.T, rtol=0, atol=1e-12 * max(1.0, np.abs(S).max())))
            self._sa = (ok, CHECKED, "W A is symmetric to 1e-12" if ok else "W A is not symmetric")
            return self._sa
        return (None, UNKNOWN, "not known")

    def positivity(self):
        return self._pos or (None, UNKNOWN, "not known")

    @property
    def claims(self):
        if isinstance(self.domain, Euclidean):
            self.is_selfadjoint()                          # checked on R^n when nothing is on record (kept as a claim)
        cl = list(self._claims)
        last = (_SA,) + _POS                               # the order of the claims view: self-adjointness, positivity last
        return Claims([c for c in cl if c.key not in last] + [c for k in last for c in cl if c.key == k])

    def claim_status(self, key):
        """(holds, status) of the property key: the claims on record, else the package's rules (operators/claim_rules.py:
        compositions, sums, finite-dimensional maps, ...), else (Truth.UNKNOWN, unknown). The spaces below operators
        read an operator's properties through this (spaces/dual.Functional.from_operator; plan 071 P2)."""
        from .claim_rules import status
        return status(self, key)

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}"


def combine_outputs(terms, codomain):
    """sum_j c_j y_j for outputs y_j of the same kind."""
    c0, y0 = terms[0]
    if isinstance(y0, (Vector, VectorFamily)):
        v = sum(c * y.values for c, y in terms)
        return (Vector if isinstance(y0, Vector) else VectorFamily)(codomain, v)
    for _, y in terms:
        require_realised(y.space, codomain, getattr(y, "name", None))    # read with codomain's backend (Q4)
    if isinstance(y0, Function):
        return Function(codomain, pl=codomain.backend.lincomb([(c, y.realisation) for c, y in terms]), name="y")
    if isinstance(y0, Family):
        return Family(codomain, codomain.backend.lincomb([(c, y.realisation) for c, y in terms]), name="Y")
    raise TypeError(type(y0).__name__)


class Composition(LinearOperator):
    """A @ B: x -> A(B(x)); its adjoint is B* A*."""

    def __init__(self, A, B):
        super().__init__(B.domain, A.codomain, name=f"{A.name}{B.name}")
        self.A, self.B = A, B
        estimable.within(self, A, B)
        ca, cb = (Claims(getattr(X, "_claims", ())).find("complex-linear") for X in (A, B))   # stored claims only:
        if ca is not None and cb is not None and ca.holds and cb.holds:          # .claims may form a matrix (P3c)
            self._claims = list(self._claims) + [Claim("complex-linear (commutes with J)", DERIVED,
                                                       f"a composition of complex-linear {A.name} and {B.name}",
                                                       key="complex-linear")]
        carry_structure(self, A, B)

    def _apply(self, x):
        return self.A(self.B(x))

    def _make_adjoint(self):
        return Composition(self.B.adjoint, self.A.adjoint)

    def matrix(self):
        """From the factors' matrices when one factor is sparse (a sparse product with the other's matrix), without the
        n x n identity the default applies; otherwise the default."""
        if isinstance(self.domain, Euclidean) and isinstance(self.codomain, Euclidean) and isinstance(self.A.domain, Euclidean):
            a, b = self.A.sparse(), self.B.sparse()
            if a is not None:
                _dense_allowed(self)
                return np.asarray(a @ self.B.matrix())
            if b is not None:
                _dense_allowed(self)
                return np.asarray((b.T @ self.A.matrix().T).T)
        return super().matrix()


class Sum(LinearOperator):
    """a_1 A_1 + ... + a_k A_k between the same spaces."""

    def __init__(self, terms):
        A0 = terms[0][1]
        for _, A in terms:
            require(A.domain, A0.domain, "operators in a sum need the same domain and codomain", A.name)
            require(A.codomain, A0.codomain, "operators in a sum need the same domain and codomain", A.name)
        names = " + ".join((f"{c:g}·" if c != 1 else "") + A.name for c, A in terms)
        super().__init__(A0.domain, A0.codomain, name=names if len(terms) > 1 else names)
        self.terms = terms
        sas = [A._sa or (A.is_selfadjoint() if isinstance(A.domain, Euclidean) else (None, UNKNOWN, ""))
               for _, A in terms]
        if all(s[0] for s in sas):
            self._sa = (True, by_rule(*(s[1] for s in sas)), "a real combination of self-adjoint operators")
        pos = [A._pos for _, A in terms]
        if all(p is not None for p in pos) and all(c > 0 for c, _ in terms):
            kind = "pd" if any(p[0] == "pd" for p in pos) else "psd"
            self._pos = (kind, by_rule(*(p[1] for p in pos)), "a positive combination of positive operators"
                         + (", one of them definite" if kind == "pd" else ""))

    def _apply(self, x):
        return combine_outputs([(c, A(x)) for c, A in self.terms], self.codomain)

    def _make_adjoint(self):
        return Sum([(c, A.adjoint) for c, A in self.terms])

    def matrix(self):
        """The dense terms summed into one array; sparse terms (diagonals, sparse matrices) added at their entries."""
        sparse = [(c, A, A.sparse()) for c, A in self.terms]
        dense = [(c, A) for c, A, S in sparse if S is None]
        if not dense:
            _dense_allowed(self)
            return self.sparse().toarray()
        c0, A0 = dense[0]
        out = np.array(A0.matrix(), dtype=float)               # a fresh array: A0.matrix() may be the operator's own
        if c0 != 1:
            out *= c0
        for c, A in dense[1:]:
            out += c * A.matrix()
        for c, _, S in sparse:
            if S is not None:
                S = S.tocoo()
                np.add.at(out, (S.row, S.col), c * S.data)
        return out

    def sparse(self):
        parts = [(c, A.sparse()) for c, A in self.terms]
        if any(S is None for _, S in parts):
            return None
        return sum(c * S for c, S in parts).tocsr()


class Identity(LinearOperator):
    """The identity of a space (self-adjoint, positive definite)."""

    why_linear = "the identity"

    def __init__(self, space):
        super().__init__(space, space, name="I")
        self._sa = (True, THEOREM, "the identity")
        self._pos = ("pd", THEOREM, "<x, x> > 0 for x != 0")

    def _apply(self, x):
        return x

    def _make_adjoint(self):
        return self

    def matrix(self):
        _dense_allowed(self)
        return np.eye(self.domain.dim)

    def sparse(self):
        import scipy.sparse as sp
        return sp.identity(self.domain.dim, format="csr")


def _dense_allowed(A):
    """Refuse to form the matrix of an operator on R^n with n > settings.dense_max (8 n^2 bytes)."""
    from ..settings import settings
    if max(A.domain.dim, A.codomain.dim) > settings.dense_max:
        raise Unsupported(f"{A.name}: no {A.codomain.dim} x {A.domain.dim} matrix above settings.dense_max = "
                          f"{settings.dense_max}; it is applied without one (raise the setting for a dense route)")


class Diagonal(LinearOperator):
    """x -> d * x (entrywise) on R^n: Diagonal(d), or Diagonal(d, space). Self-adjoint when the space's metric is
    diagonal (or the dot product); its inverse is the diagonal of 1/d, without a solver."""

    why_linear = "entrywise multiplication by fixed numbers"

    def __init__(self, d, space=None, name="Δ"):
        d = np.array(d, float).ravel()
        space = Euclidean(len(d)) if space is None else space
        if space.dim != len(d):
            raise ValueError(f"{len(d)} diagonal entries for {space}")
        super().__init__(space, space, name=name)
        d.flags.writeable = False
        self.d = d
        if space.is_standard or space.weights is not None:
            self._sa = (True, DERIVED, "a diagonal operator, and the metric is diagonal")
            if np.all(d > 0):
                self._pos = ("pd", CHECKED, "all diagonal entries are positive")

    def _apply(self, x):
        if isinstance(x, Vector):
            return Vector(self.codomain, self.d * x.values, name=f"{self.name}{x.name}")
        if isinstance(x, VectorFamily):
            return VectorFamily(self.codomain, x.values * self.d, name=f"{self.name}{x.name}")
        raise TypeError(type(x).__name__)

    def diagonal(self):
        return self.d

    def matrix(self):
        if len(self.d) > 6000:
            raise MemoryError(f"{self.name} is a diagonal of length {len(self.d)}; it is not formed densely")
        return np.diag(self.d)

    def sparse(self):
        import scipy.sparse as sp
        return sp.diags(self.d, format="csr")

    def _make_adjoint(self):
        if self._sa:
            return self
        D = self.domain
        return LinearOperator(D, D, apply=lambda x: Vector(D, D.solve_metric(self.d * D.apply_metric(x.values))),
                              adjoint=None, name=f"{self.name}*")

    def inverse(self, solver=None):
        if solver is None and np.all(self.d != 0):
            out = Diagonal(1.0 / self.d, self.domain, name=f"{self.name}⁻¹")
            out._adjoint = None
            return out
        return super().inverse(solver)

    @property
    def inv(self):
        if self._inv is None:
            self._inv = self.inverse()
        return self._inv

    def _compose(self, other):
        if isinstance(other, Diagonal):
            return Diagonal(self.d * other.d, other.domain, name=f"{self.name}{other.name}")
        return super()._compose(other)

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}  [diagonal]"


class Matrix(LinearOperator):
    """A linear map between Euclidean spaces given by its matrix in the standard coordinates."""

    why_linear = "multiplication by a fixed matrix"

    def __init__(self, A, domain=None, codomain=None, name="A"):
        A = np.array(A, float)
        if A.ndim != 2:
            raise ValueError("a matrix is 2-dimensional")
        domain = Euclidean(A.shape[1]) if domain is None else domain
        codomain = Euclidean(A.shape[0]) if codomain is None else codomain
        if (codomain.dim, domain.dim) != A.shape:
            raise ValueError(f"a {A.shape} matrix maps R^{A.shape[1]} to R^{A.shape[0]}")
        super().__init__(domain, codomain, name=name)
        A.flags.writeable = False
        self.A = A

    def _apply(self, x):
        if isinstance(x, Vector):
            return Vector(self.codomain, self.A @ x.values, name=f"{self.name}{x.name}")
        if isinstance(x, VectorFamily):
            return VectorFamily(self.codomain, x.values @ self.A.T, name=f"{self.name}{x.name}")
        raise TypeError(type(x).__name__)

    def matrix(self):
        return self.A

    def _make_adjoint(self):
        # <A x, y>_C = x . A^T W_C y = <x, W_D^-1 A^T W_C y>_D
        B = self.domain.solve_metric(self.codomain.apply_metric(self.A).T)
        return Matrix(B, domain=self.codomain, codomain=self.domain, name=f"{self.name}*")

    def _compose(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.A @ other.A, domain=other.domain, codomain=self.codomain, name=f"{self.name}{other.name}")
        return super()._compose(other)


def synthesise(generators, A, single):
    """sum_j A[j, m] psi_j for the members psi_j of a Family or VectorFamily (one column of A per output)."""
    A = np.asarray(A, float).reshape(len(generators), -1)
    sp_ = generators.space
    if isinstance(generators, VectorFamily):
        V = A.T @ generators.values
        return Vector(sp_, V[0], name="v") if single else VectorFamily(sp_, V, name="V")
    R = sp_.backend.combine(generators.realisation, A)
    if single:
        return Function(sp_, pl=R if A.shape[1] == 1 else sp_.backend.take(R, 0), name="f")
    return Family(sp_, R, name="F")


class FiniteRank(LinearOperator):
    """x -> sum_j lam_j <psi_j, x> psi_j for generators psi_j (a Family or VectorFamily) and weights lam_j >= 0:
    self-adjoint, positive semidefinite and of finite rank (hence trace class). The covariance operator of every
    finite-rank Gaussian measure in pygeoinf0; pushing it forward through A gives the generators A psi_j."""

    why_linear = "a finite sum of inner products with fixed generators times fixed weights"

    def __init__(self, generators, weights, name="C"):
        lam = np.asarray(weights, float).ravel()
        if len(lam) != len(generators):
            raise ValueError(f"{len(generators)} generators and {len(lam)} weights")
        super().__init__(generators.space, generators.space, name=name)
        self.generators, self.lam = generators, lam
        self._sa = (True, DERIVED, "a sum of lam_j psi_j (x) psi_j")
        if np.all(lam >= 0):
            self._pos = ("psd", DERIVED, "non-negative weights")
        self._claims.append(Claim("trace class", THEOREM, f"finite rank (at most {len(lam)})"))

    @property
    def rank(self):
        return len(self.lam)

    def coefficients(self, x):
        """lam_j <psi_j, x> for each generator j: shape (N,) for an element, (N, M) for a family."""
        G = np.asarray(self.domain.inner(self.generators, x), float).reshape(self.rank, -1)
        return self.lam[:, None] * G

    def _apply(self, x):
        return synthesise(self.generators, self.coefficients(x), single=isinstance(x, (Function, Vector)))

    def _make_adjoint(self):
        return self

    def trace(self):
        return float(self.lam @ np.asarray(self.domain.norm(self.generators)) ** 2)

    def pushforward(self, A):
        """A C A* as a finite-rank operator on A's codomain, with generators A psi_j."""
        return FiniteRank(A(self.generators), self.lam, name=f"{A.name}{self.name}{A.name}*")

    def matrix(self):
        if not isinstance(self.domain, Euclidean):
            raise TypeError(f"{self.name} is not a map between Euclidean spaces, so it has no matrix")
        P = self.generators.values                                    # (N, n)
        return (P.T * self.lam) @ self.domain.apply_metric(P.T).T

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.domain}  [finite rank {self.rank}]"
