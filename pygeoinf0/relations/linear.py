"""Linear relations (plan 008 decision 3; EXP-028 DESIGN.md 1.6): graphs that are linear subspaces of X x Y,

    R = {(x, G x + z) : x in dom, z in mul}

with G a bounded LinearOperator on all of X (an extension of the operator part), dom a closed subspace of X and mul a
closed subspace of Y (convex/subspaces.py). R(x) = G x + mul is an affine set (L1); R is single-valued iff
mul = {0}. The facts used (proofs in DESIGN.md):
    adjoint      R* = {(k, G* k + w) : k in mul^perp, w in dom^perp}: mul R* = dom^perp, dom R* = mul^perp (L3, L4)
    converse     R^-1: dom = ran R, mul = ker R; its operator part is the minimum-norm map (L2, the Corollary):
                 for G: X -> R^n, G.converse().operator_part()(d) is G*(G G*)^+ d, the package's minimum-norm model
    operator part R_s x = P_(mul^perp) G x, the element of least norm of R(x) (L7); the graph norm of dom R is
                 ||x||^2 + ||R_s x||^2 (domain_space())
    quotient     q o R: X -> Y / mul, single-valued (L8); Y / mul is realised as R^(n-k) in coordinates of a
                 W-orthonormal basis of mul^perp, so its norm is the quotient norm dist(y, mul)

Three objects are kept apart (AGENTS.md): R.domain (the ambient space X), R.dom (the restricted domain, a subset of
X) and R.domain_space() (dom R with the graph norm, a new Hilbert space).

ClosedOperator is the graph of a closed, densely defined operator A on X (plan 068 P3): single-valued and partial, with
dom = D(A) an OperatorDomain, a linear subspace of X that is dense and, when A is unbounded, not closed (a closed
operator with a closed domain is bounded: the closed graph theorem); its graph-norm space only through domain_space().
The operators themselves (FormOperator, SturmLiouville, Laplacian) are in operators/unbounded.py; the minimum-norm map
of the converse is operators/inverses.MinimumNorm.
"""
import numpy as np

from ..core.claims import THEOREM, UNKNOWN, Claim, Claims, by_rule
from ..core.errors import NotDefined, Refused, Unsupported
from ..core.kinds import SET, is_state
from ..core.relation import Relation
from ..core.logic import Truth
from ..spaces.base import Euclidean
from ..operators.linear import Matrix
from ..operators.inverses import MinimumNorm
from ..convex.subspaces import Complement, LinearSubspace, Projector, Span, Whole, Zero, span_sum
from ..spaces.identity import require_realised


def _finite(Y):
    if not isinstance(Y, Euclidean):
        raise Unsupported(f"this construction needs a finite-dimensional codomain (R^n); {Y!r} is not one")


class LinearRelation(Relation):
    """A linear relation {(x, G x + z) : x in dom, z in mul} (module docstring). LinearRelation(G, mul=E) is
    m -> G m + E (unknown origin times: E the span of the per-event indicators); pg.Additive(G, E) builds the same."""

    kind = SET

    def __init__(self, G=None, mul=None, dom=None, domain=None, codomain=None, name="R", claims=()):
        if G is None and (domain is None or codomain is None):
            raise ValueError("give the operator part G, or the domain and codomain of a relation without one")
        self.G = G
        self.domain = G.domain if G is not None else domain
        self.codomain = G.codomain if G is not None else codomain
        self._dom = Whole(self.domain) if dom is None else dom
        self._mul = Zero(self.codomain) if mul is None else mul
        for S, X, what in ((self._dom, self.domain, "dom"), (self._mul, self.codomain, "mul")):
            if not isinstance(S, LinearSubspace) or S.space != X:
                raise TypeError(f"{what} must be a closed subspace of {X!r} (Whole, Zero, Span, Complement)")
            require_realised(S.space, X, what)
        self.name = name
        self._claims = _graph_claims(G, self._dom, self._mul)
        self._claims.append(Claim("single-valued" if isinstance(self._mul, Zero) else "not single-valued", THEOREM,
                                  "mul R = {0}" if isinstance(self._mul, Zero) else "mul R is not {0} (L1)",
                                  key="single-valued", holds=isinstance(self._mul, Zero)))
        self._claims.append(Claim("total" if isinstance(self._dom, Whole) else "partial", THEOREM, "dom R",
                                  key="total", holds=isinstance(self._dom, Whole)))
        self._claims += list(claims)
        self._converse_of = None

    # the four subspaces ----------------------------------------------------------------------------------------
    @property
    def dom(self):
        return self._dom

    @property
    def mul(self):
        return self._mul

    @property
    def ker(self):
        """ker R = {x in dom : G x in mul} = (G*(mul^perp))^perp for a total R into R^n."""
        if self._converse_of is not None:
            return self._converse_of.mul
        if not isinstance(self._dom, Whole):
            raise Unsupported("ker of a partial linear relation is not implemented")
        _finite(self.codomain)
        Y = self.codomain
        B = self._mul.perp().project(Y.family(np.eye(Y.dim), name="e"))
        return Complement(Span(self.G.adjoint(B), name="G*(mul^⊥)"), name=f"ker {self.name}")

    @property
    def ran(self):
        """ran R = G(dom) + mul (finite-dimensional codomains)."""
        if self._converse_of is not None:
            return self._converse_of.dom
        from ..convex.subspaces import image
        return span_sum(image(self.G, self._dom), self._mul)

    @property
    def is_single_valued(self):
        return Truth.of(isinstance(self._mul, Zero))

    @property
    def claims(self):
        return Claims(self._claims)

    # values ------------------------------------------------------------------------------------------------------
    def __call__(self, x):
        from ..convex.sets import AffineSubset, EmptySet
        if is_state(x):
            from ..cells import push
            return push(x, self)
        if self._dom.member(x) is Truth.FALSE:
            return EmptySet(self.codomain, why=f"{getattr(x, 'name', 'x')} is not in dom {self.name}")
        if self.G is None:
            raise Unsupported(f"{self.name} has no realised action (a descriptor: claims, adjoint, graph-norm space)")
        return AffineSubset(self.G(x), self._mul, name=f"{self.name}({getattr(x, 'name', 'x')})")

    def operator_part(self):
        """R_s: x -> P_(mul^perp) G x, the element of least norm of R(x) (L7); defined on all of the domain, equal to
        R_s on dom."""
        if self.G is None:
            raise Unsupported(f"{self.name} has no realised operator part")
        return self.G if isinstance(self._mul, Zero) else Projector(self._mul.perp()) @ self.G

    # adjoint, converse, closure ------------------------------------------------------------------------------------
    @property
    def adjoint(self):
        """R* = {(k, G* k + w) : k in mul^perp, w in dom^perp} (L3, L4): always a closed linear relation."""
        return LinearRelation(self.G.adjoint, mul=self._dom.perp(), dom=self._mul.perp(), name=f"{self.name}*",
                              claims=[Claim("closed", THEOREM, "an adjoint is an orthogonal complement (L3)",
                                            key="closed graph")])

    def closure(self):
        """The closure: R itself (dom and mul are closed subspaces and G is bounded)."""
        st = self.G is not None and _bounded(self.G)
        if not st:
            raise Unsupported(f"closedness of {self.name} needs a bounded operator part (claim 'bounded' of {self.G})")
        return self

    def converse(self, prior=None):
        """R^-1 = {(G x + z, x)}: dom = ran R, mul = ker R, operator part the minimum-norm map (L2, Corollary)."""
        if prior is not None:
            raise Refused("a linear relation is set-valued: its converse needs no prior")
        if self._converse_of is not None:
            return self._converse_of
        if not isinstance(self._dom, Whole):
            raise Unsupported("the converse of a partial linear relation is not implemented")
        _finite(self.codomain)
        out = LinearRelation(MinimumNorm(self.G, self._mul), mul=self.ker, dom=self.ran, name=f"{self.name}⁻¹",
                             claims=[Claim("the converse: R^-1(d) = {x : d in R(x)}", THEOREM, "L2")])
        out._converse_of = self
        return out

    # quotient and graph norm --------------------------------------------------------------------------------------
    def quotient_map(self):
        """q: Y -> Y / mul R realised as R^(n-k): y -> U^T W y, U a W-orthonormal basis of mul^perp (L8). The norm of
        q(y) is dist(y, mul R), and q maps the standard Gaussian of Y to that of R^(n-k)."""
        Y = self.codomain
        _finite(Y)
        if isinstance(self._mul, Zero):
            U = Y.from_orthonormal(np.eye(Y.dim))
        else:
            V = self._mul.matrix_basis() if isinstance(self._mul, Span) else None
            if V is None:
                raise Unsupported("quotients by subspaces other than a span are not implemented")
            Q = Y.to_orthonormal(V)                            # orthonormal coordinates of mul (columns)
            Z, _ = np.linalg.qr(np.c_[Q, np.eye(Y.dim)])
            U = Y.from_orthonormal(Z[:, Q.shape[1]:Y.dim])     # a W-orthonormal basis of mul^perp
        k = U.shape[1]
        q = Matrix(Y.apply_metric(U).T, domain=Y, codomain=Euclidean(k), name="q")
        q._claims = [Claim("linear", THEOREM, "a matrix", key="linear"),
                     Claim("the quotient map onto Y / mul R, norm the distance to mul R", THEOREM, "L8")]
        return q

    def quotient(self):
        """q o R: X -> Y / mul R, single-valued (L8)."""
        q = self.quotient_map()
        return q @ self.G

    def domain_space(self):
        """dom R with the graph norm ||x||^2 + ||R_s x||^2 (L7): a Hilbert space with an inclusion into the domain."""
        X = self.domain
        if isinstance(X, Euclidean) and isinstance(self._dom, Whole):
            Rs = self.operator_part().matrix()
            W = X.metric_matrix() + Rs.T @ self.codomain.apply_metric(Rs)
            return Euclidean(X.dim, metric=0.5 * (W + W.T))
        raise Unsupported("the graph-norm space is constructed for relations on R^n and for the closed operators of "
                          "operators/unbounded.py (their domain_space()); on a function space with a bounded operator "
                          "part it is the domain with an equivalent norm")

    def __repr__(self):
        return (f"{self.name}: {self.domain!r} ⇉ {self.codomain!r}  [linear relation; mul "
                f"{'{0}' if isinstance(self._mul, Zero) else f'of dimension {self._mul.dim}'}]")


def _graph_claims(G, dom, mul):
    """The claims of the graph {(x, G x + z) : x in dom, z in mul} whose premises are on record (plan 065 A4): linear when
    G is (the subspaces are linear by their type); closed when dom and mul are closed and G is bounded (L7, L9). Without
    an operator part nothing is added: a constructor that knows the graph (ClosedOperator) states it."""
    if G is None:
        return []
    from ..operators.claim_rules import all_hold, status
    out = []
    held, st = all_hold([status(G, "linear")])
    if held is Truth.TRUE:
        out.append(Claim("linear graph", st, f"dom and mul are linear subspaces and {G.name} is linear", key="linear"))
        held, st = all_hold([status(G, "linear"), status(G, "bounded"), dom.claims.status("closed"),
                             mul.claims.status("closed")])
        if held is Truth.TRUE:
            out.append(Claim("closed graph", by_rule(st),
                             f"dom and mul are closed subspaces and {G.name} is bounded (L7, L9)", key="closed graph"))
    return out


class OperatorDomain(LinearSubspace):
    """D(A) of a closed, densely defined operator A as a subset of its ambient space X (plan 068 P3; review A4): a linear
    subspace, dense in X, closed in the graph norm of A, and not closed in X when A is unbounded. It has no orthogonal
    projection (NotDefined: its closure is X) and its orthogonal complement is {0}. member(x) is TRUE for 0 and for the
    elements of A.domain_space() (dom A with the graph norm, a different space), UNKNOWN for other elements of X: whether
    a function of X lies in D(A) is a matter of regularity and boundary conditions the package does not decide."""

    def __init__(self, A, space):
        super().__init__(space, name=f"D({getattr(A, 'name', 'A')})")
        self.A = A

    @property
    def claims(self):
        """Dense and closed in the graph norm (A closed, densely defined), with the status of A's premises
        (A.premises(); plan 071 P2); "not closed" in the ambient norm exactly when A states that it is unbounded (with
        that claim's status), with the reason; otherwise nothing is claimed about closedness."""
        st0, note = premises_of(self.A)
        cl = [Claim("linear subspace", THEOREM, "the domain of a linear operator", key="linear subspace"),
              Claim("dense", st0, self.A.dense_reason() + note, key="dense"),
              Claim("closed in the graph norm of A", st0, "A is closed" + note, key="closed in the graph norm"),
              Claim("convex", THEOREM, "a linear subspace", key="convex")]
        held, st = self.A.claims.status("unbounded")
        if held is Truth.TRUE and st != UNKNOWN:
            cl.append(Claim("not closed in the ambient norm", st, f"{self.A.name} is unbounded on {self.space!r} "
                            "and closed, and a closed operator with a closed domain is bounded (the closed graph "
                            "theorem)", key="closed", holds=False))
        return Claims(cl)

    def member(self, x):
        try:
            V = self.A.domain_space()
        except (Unsupported, NotDefined):
            V = None
        if V is not None and getattr(x, "space", None) == V:
            return Truth.TRUE
        return Truth.TRUE if Zero(self.space).member(x) is Truth.TRUE else Truth.UNKNOWN

    def project(self, x):
        raise NotDefined(f"{self.name} is dense in {self.space!r}: there is no orthogonal projection onto it",
                         needs=("the projection onto its closure, the whole space (the identity)",))

    def perp(self):
        return Zero(self.space)                               # dense: its orthogonal complement is {0}


def premises_of(A):
    """A.premises() of a ClosedOperator (status, note), (THEOREM, "") for other operators with a domain."""
    get = getattr(A, "premises", None)
    return get() if callable(get) else (THEOREM, "")


class ClosedOperator(LinearRelation):
    """The graph of a closed, densely defined linear operator A on a Hilbert space X (module docstring; plan 068 P3):
    a single-valued, partial linear relation with dom = D(A) (an OperatorDomain), mul = {0} and no realised operator
    part (A is a descriptor: its claims, its adjoint, its graph-norm space, its spectrum). Subclasses
    (operators/unbounded.py) set name and give ambient() (X, built when first used), operator_claims() (with
    "unbounded" and "self-adjoint" stated with their reasons), dense_reason() and domain_space()."""

    G = None
    _converse_of = None

    def ambient(self):
        raise NotImplementedError

    def dense_reason(self):
        return "the domain of the operator of a densely defined closed form (Kato, Perturbation Theory, VI.2.1)"

    def operator_claims(self):
        return []

    def premises(self):
        """(status, note) of the premises under which A is closed, densely defined and what operator_claims() says:
        THEOREM when they hold by construction (the default) or were decided exactly, CHECKED when checked on samples
        (SturmLiouville with callable coefficients; plan 071 P2). note is appended to the reasons."""
        return THEOREM, ""

    @property
    def domain(self):
        X = self.__dict__.get("_X")
        if X is None:
            X = self.__dict__["_X"] = self.ambient()
        return X

    @property
    def codomain(self):
        return self.domain

    @property
    def _dom(self):
        D = self.__dict__.get("_D")
        if D is None:
            D = self.__dict__["_D"] = OperatorDomain(self, self.domain)
        return D

    @property
    def _mul(self):
        return Zero(self.domain)

    @property
    def claims(self):
        st, note = self.premises()
        return Claims([Claim("linear graph", THEOREM, "the graph of a linear operator", key="linear"),
                       Claim("closed graph", st, "a closed operator" + note, key="closed graph"),
                       Claim("densely defined", st, self.dense_reason() + note, key="densely defined"),
                       Claim("single-valued", THEOREM, "mul R = {0}", key="single-valued"),
                       Claim("partial", st, "dom R = D(A), not the whole space (A is unbounded)" + note, key="total",
                             holds=False)]
                      + list(self.operator_claims()))

    @property
    def adjoint(self):
        """A* = A for an operator stated self-adjoint; otherwise not built here."""
        held, _ = self.claims.status("self-adjoint")
        if held is Truth.TRUE:
            return self
        raise Unsupported(f"the adjoint of {self.name} is not built (it is not stated self-adjoint)")

    def closure(self):
        return self                                           # closed: its claim

    def converse(self, prior=None):
        raise Unsupported(f"the converse of {self.name} (its inverse: a bounded operator where A >= c > 0) is not built")

    def operator_part(self):
        raise Unsupported(f"{self.name} has no realised action; its claims, its spectrum and its graph-norm space "
                          "(domain_space()) are what the package computes")

    @property
    def ran(self):
        raise Unsupported(f"ran {self.name} is not computed here")

    def domain_space(self):
        raise Unsupported(f"the graph-norm space of {self.name} (D(A) with ||u||^2 + ||A u||^2) is not built")

    def __repr__(self):
        return f"{self.name}: {self.domain!r} ⇉ {self.domain!r}  [closed, densely defined]"


def _bounded(G):
    from ..operators.claim_rules import status
    held, st = status(G, "bounded")
    return held is Truth.TRUE and st != UNKNOWN
