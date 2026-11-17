"""Convex sets of a space with their support functions (plan 008, P2; plan 067 P3): ConvexSet(Subset) is the
computable subfamily of the subsets of a space (sets/subsets.py, the one Set root), with three-valued membership and the
support function sigma_S(v) = sup over x in S of <v, x>, in the inner product of the space, returned as an Enclosure
whose lower and upper ends are certified in their directions (ConvexSet's docstring; review A7).

    NormBall(X, radius, centre=None)        {x : ||x - c|| <= r} in any Hilbert space (Gram-exact supports)
    Ellipsoid(D, radius, covariance=S)      {e : (e - c)^T S^-1 (e - c) <= r^2} in R^n (standard coordinates);
    Ellipsoid(X, radius, shape=A)           {m : <A (m - c), m - c> <= r^2} in any Hilbert space (plan 009 P5)
    Slab(D, normal=a, half_width=w)         {e : |<a, e - c>| <= w}: the noise region of one fixed estimator
    CoordinateBox(D, lo, hi)                {e : lo <= e <= hi} in R^n (pg.Box is the 3-D domain)
    Polytope(D, A, b)                       {e : A e <= b} in R^n (supports by linear programming, optimisers/lp.py)
    AffineSubset(point, direction)          point + a closed subspace (convex/subspaces.py): fibres m + ker G
    EmptySet(X)                             the empty set (a finding, not an error), the root's EmptySet of X

S + K, x + S, c * S, S & K, S | K are the lazy Minkowski sum, translate, scaling, intersection and union
(convex/algebra.py); T(S) the image under a linear operator. Sets in R^p have interval(k) and outer_polytope(Q).
A ball of D = Euclidean(n, covariance=C) is the noise ellipsoid of C: NormBall(D, r) == Ellipsoid(D.standard, r, C).
"""
import math

import numpy as np

from ..core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, Claim, by_rule
from ..core.errors import Refused, SpaceMismatch
from ..core.kinds import SET, stated_kind
from ..core.logic import Truth, decide
from ..core.numbers import EXACT, Enclosure
from ..spaces.base import Euclidean
from ..sets import EmptySet as RootEmptySet
from ..sets.subsets import Subset
from ..spaces.identity import require


def as_element(space, v):
    """v as an element of the space: arrays are standard coordinates of a vector of a Euclidean space."""
    if isinstance(space, Euclidean) and not hasattr(v, "space"):
        return space(np.asarray(v, float).ravel())
    if getattr(v, "space", None) != space:
        raise SpaceMismatch(f"{getattr(v, 'name', 'the direction')} is not an element of {space}")
    return v


def coordinate_direction(space, k):
    """The element v of R^p with <v, p> = p_k (W^-1 e_k for the metric W)."""
    e = np.zeros(space.dim)
    e[k] = 1.0
    return space(space.solve_metric(e))


class ConvexSet(Subset):
    """A convex subset of a space with a support function: the computable subfamily of Subset (sets/subsets.py; the
    author's amendment 1: convexity is what is computable, not a kind of state).

    support(v) is an Enclosure of sigma_S(v) = sup over S of <v, x> (core/numbers.py; plan 067 P3, review A7): its lower
    end is a lower bound and its upper end an upper bound of sigma_S(v), each certified in that direction (an end not
    known is infinite: how is OUTER when only one end is informative); its value is the computed approximation (for a
    feasible set the solver's value, which may sit at either end). The status says how the interval is known: derived
    for a closed form evaluated in floating point (its ends agree to rounding: how EXACT), checked where a numerical
    step enters (a linear programme, a witness checked feasible to the tolerance), asserted where a premise is the
    caller's. Readers that need a direction read that end: an upper bound of sigma reads .upper."""

    convex = True

    # sets in R^p ----------------------------------------------------------------------------------------
    def interval(self, k=0):
        """The certified outer interval [l, u] of the coordinate p_k over the set (a set in R^p): l is the lower end of
        inf p_k = -sigma(-e_k) (minus the upper end of sigma(-e_k)), u the upper end of sup p_k = sigma(e_k)."""
        e = coordinate_direction(self.space, k)
        return -float(support_of(self, e * -1.0).upper), float(support_of(self, e).upper)

    def outer_polytope(self, directions):
        """(Q, b) with the set inside {p : Q p <= b}: b_j the certified upper end of sigma(q_j) (rows q_j of
        directions)."""
        Q = np.atleast_2d(np.asarray(directions, float))
        b = np.array([float(support_of(self, self.space(self.space.solve_metric(q))).upper) for q in Q])
        return Q, b

    def polytope(self, directions, approx="outer"):
        """The outer Polytope {Q p <= b} of outer_polytope(directions), marked outer of this set; approx="inner" the
        hull of feasible points found in those directions (convex/polytope.py inner_polytope), marked inner."""
        if approx == "inner":
            from .polytope import inner_polytope
            return inner_polytope(self, directions)
        Q, b = self.outer_polytope(directions)
        return Polytope(self.space, Q, b, f"outer({self.name})", "outer", self)

    def plot(self, ax=None, directions=180, **kw):
        """Draw the set in R^1, R^2 or R^3 (plotting/sets.py): a polytope as itself, other sets by their outer
        polygon of `directions` supports, labelled outer."""
        from ..plotting.sets import plot_set
        return plot_set(self, ax=ax, directions=directions, **kw)

    # algebra ------------------------------------------------------------------------------------------------
    def __add__(self, other):
        from .algebra import minkowski
        return minkowski(self, other)

    __radd__ = __add__

    def __rmul__(self, c):
        from .algebra import Scale
        return Scale(self, c) if np.ndim(c) == 0 else NotImplemented

    def _meet(self, other):
        from .algebra import Intersection
        return Intersection([self, other])

    def _join(self, other):
        from .algebra import Union
        return Union([self, other])


def support_of(S, v):
    """S.support(v) as an Enclosure; a plain number is refused (A7: a support states which end is certified)."""
    s = S.support(v)
    if not isinstance(s, Enclosure):
        raise TypeError(f"the support of {S.name} gave a plain number ({type(s).__name__}): a support function returns "
                        "an Enclosure whose lower end is a lower bound and upper end an upper bound of sigma (plan 067 "
                        "P3, A7), so that the set algebra can keep the direction of every end")
    return s


def exact(value, why, status=DERIVED):
    """The Enclosure of a support value given by a closed form evaluated in floating point: both ends at the value
    (how EXACT: they agree to rounding)."""
    v = float(value)
    return Enclosure(v, v, v, status, why, how=EXACT)


def _closed_convex(why, bounded=None):
    cl = [Claim("closed", THEOREM, why, key="closed"), Claim("convex", THEOREM, why, key="convex")]
    if bounded is not None:
        cl.append(Claim("bounded" if bounded else "not bounded", THEOREM, why, key="bounded", holds=bounded))
    return cl


class EmptySet(ConvexSet, RootEmptySet):
    """The empty set: support -inf. An empty feasible set is a finding (prior, relation and datum disagree). It is the
    one empty set of the Set root (sets.EmptySet, plan 067 P2) with the space as its ambient."""

    def __init__(self, space, why="empty", name="∅"):
        super().__init__(space, name, _closed_convex("the empty set", True) + [Claim("empty", THEOREM, why, key="empty")])
        self._ambient, self.why = space, why

    def member(self, x):
        return Truth.FALSE

    def support(self, v):
        return exact(-math.inf, "the support of the empty set", THEOREM)

    def is_empty(self):
        return Truth.TRUE

    __and__, __rand__, __or__, __ror__ = (RootEmptySet.__and__, RootEmptySet.__rand__, RootEmptySet.__or__,
                                          RootEmptySet.__ror__)
    __eq__, __hash__, __repr__ = RootEmptySet.__eq__, RootEmptySet.__hash__, RootEmptySet.__repr__


class NormBall(ConvexSet):
    """{x : ||x - c|| <= r} in a Hilbert space: support <v, c> + r ||v||, exact from pairings (inf:1001-1050)."""

    def __init__(self, space, radius, centre=None, name=None):
        r = float(radius)
        if not r >= 0:
            raise ValueError("the radius must be >= 0")
        self.radius, self.centre = r, centre
        super().__init__(space, name or f"B({r:g})",
                         _closed_convex("a closed ball of a Hilbert space", True)
                         + [Claim("weakly compact", THEOREM, "bounded, closed and convex in a Hilbert space",
                                  key="weakly compact")])

    def member(self, x):
        x = as_element(self.space, x)
        d = x if self.centre is None else x - self.centre
        n = self.space.norm(d)
        return decide(max(n - self.radius, 0.0), max(self.radius, n, 1e-300), tol=0.0)     # tolerance is not truth

    def support(self, v):
        v = as_element(self.space, v)
        c = 0.0 if self.centre is None else self.space.inner(v, self.centre)
        return exact(c + self.radius * self.space.norm(v), "<v, c> + r ||v|| (Cauchy-Schwarz), from pairings")

    def is_empty(self):
        return Truth.FALSE


class Ellipsoid(ConvexSet):
    """{e : (e - c)^T S^-1 (e - c) <= r^2} in R^n, S symmetric positive definite (a number, a diagonal or a matrix) in
    standard coordinates. Its support in D's inner product is <mu, c> + r sqrt(a^T S a), a = W mu.

    In any Hilbert space (plan 009 P5, G8 (a)): Ellipsoid(X, r, shape=A, inverse=A^-1, centre=c) is
    {m : <A (m - c), m - c> <= r^2} for A self-adjoint and bounded below (coercive; ASSERTED as given), with
    sigma(q) = <q, c> + r <q, A^-1 q>^(1/2) (THEOREM: the substitution m - c = A^(-1/2) u maps it to a ball). Give A,
    A^-1 or both (operators on X; A^-1 is the middle operator of BGP's data-space decomposition).

    One interface for both parameterisations (plan 065 P4, A8): radius; shape_operator A and inverse A^-1 (operators on
    the space; on R^n, A = W^-1 S^-1 and A^-1 = S W for the metric W); and on R^n the coordinates shape (S = (W A)^-1),
    factor (S = L L^T) and centre_values (the centre's standard coordinates, zeros without one). An ellipsoid given by
    its form computes the coordinates from A^-1 on first use and refuses them off R^n or when A^-1 is not
    self-adjoint and positive definite to rounding. (One class with one parameterisation is Plan B.)"""

    def __new__(cls, space, radius, covariance=1.0, centre=None, name=None, shape=None, inverse=None):
        if cls is Ellipsoid and (shape is not None or inverse is not None or not isinstance(space, Euclidean)):
            return super().__new__(OperatorEllipsoid)
        return super().__new__(cls)

    def __init__(self, space, radius, covariance=1.0, centre=None, name=None, shape=None, inverse=None):
        n = space.dim
        S = np.asarray(covariance, float)
        S = S * np.eye(n) if S.ndim == 0 else np.diag(S) if S.ndim == 1 else S
        if S.shape != (n, n):
            raise ValueError(f"the shape matrix of an ellipsoid in R^{n} is {n} x {n}")
        self.radius, self.shape = float(radius), 0.5 * (S + S.T)
        self.centre = np.zeros(n) if centre is None else np.asarray(getattr(centre, "values", centre), float)
        self.factor = np.linalg.cholesky(self.shape)          # S = L L^T: whitening by L^-1
        super().__init__(space, name or f"E({self.radius:g})", _closed_convex("an ellipsoid with S positive definite", True))

    @property
    def centre_values(self):
        return np.array(self.centre, float)

    @property
    def shape_operator(self):
        """A = W^-1 S^-1: {e : <A (e - c), e - c> <= r^2} in the inner product of the space."""
        from ..operators.linear import Matrix
        Si = np.linalg.inv(self.factor).T @ np.linalg.inv(self.factor)          # S^-1 = L^-T L^-1
        return Matrix(self.space.solve_metric(Si), self.space, self.space, name=f"A({self.name})")

    @property
    def inverse(self):
        """A^-1 = S W."""
        from ..operators.linear import Matrix
        return Matrix(self.space.apply_metric(self.shape).T, self.space, self.space, name=f"A({self.name})^-1")

    def support(self, mu):
        a = self.space.apply_metric(as_element(self.space, mu).values)
        return exact(a @ self.centre + self.radius * math.sqrt(max(a @ self.shape @ a, 0.0)),
                     "<mu, c> + r sqrt(a^T S a), a = W mu (the image of a ball)")

    def member(self, x):
        z = np.linalg.solve(self.factor, as_element(self.space, x).values - self.centre)
        return decide(max(float(np.linalg.norm(z)) - self.radius, 0.0), max(self.radius, 1e-300))

    def is_empty(self):
        return Truth.FALSE


class OperatorEllipsoid(Ellipsoid):
    """{m : <A (m - c), m - c> <= r^2} in a Hilbert space (Ellipsoid's docstring); pg.Ellipsoid(X, r, shape=A) builds
    one."""

    operator = True

    def __init__(self, space, radius, covariance=1.0, centre=None, name=None, shape=None, inverse=None):
        if shape is None and inverse is None:
            raise TypeError("an ellipsoid in a function space needs shape=A or inverse=A^-1 (an operator); "
                            "NormBall is the case A = I")
        for A in (shape, inverse):
            if A is not None:
                require(A.domain, space, f"the shape operator {A.name} does not act on {space}", A.name)
                require(A.codomain, space, f"the shape operator {A.name} does not act on {space}", A.name)
        self.radius, self.centre = float(radius), centre
        if inverse is None:
            from ..operators.inverses import Inverse
            inverse = Inverse(shape)
        self._A, self._Ainv, self._coords = shape, inverse, None
        ConvexSet.__init__(self, space, name or f"E({self.radius:g})",
                           _closed_convex("the image of a closed ball under the bounded bijection A^(-1/2)", True)
                           + [Claim("A self-adjoint and bounded below", ASSERTED, "as given (shape= / inverse=)",
                                    key="coercive shape")])

    shape_operator = property(lambda self: self._A)
    inverse = property(lambda self: self._Ainv)

    def _coordinates(self):
        """(S, L, c) in standard coordinates of R^n: S = A^-1 W^-1 from the matrix of A^-1 (Ellipsoid's docstring)."""
        if self._coords is None:
            D = self.space
            if not isinstance(D, Euclidean):
                raise Refused(f"{self.name} is an ellipsoid of {D}, which has no standard coordinates: read its form "
                              "through shape_operator (A) and inverse (A^-1), or realise the space in R^n first",
                              needs=["shape_operator", "inverse"])
            S = D.solve_metric(self._Ainv.matrix().T).T                         # A^-1 W^-1 (W symmetric)
            asym = float(np.linalg.norm(S - S.T))
            if decide(asym, max(float(np.linalg.norm(S)), 1e-300)) is not Truth.TRUE:
                raise Refused(f"the form of {self.name} is not self-adjoint in {D} to rounding (|S - S^T| = {asym:.3g}): "
                              "the ellipsoid {<A e, e> <= r^2} needs A self-adjoint and positive definite; give the "
                              "self-adjoint part (A + A*)/2 if that is the set meant", needs=["shape"])
            S = 0.5 * (S + S.T)                                                  # the difference is rounding only
            try:
                L = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                raise Refused(f"the form of {self.name} is not positive definite in {D}: an ellipsoid needs A "
                              "bounded below (coercive)", needs=["shape"]) from None
            c = np.zeros(D.dim) if self.centre is None else as_element(D, self.centre).values.astype(float)
            self._coords = (S, L, c)
        return self._coords

    shape = property(lambda self: self._coordinates()[0])
    factor = property(lambda self: self._coordinates()[1])
    centre_values = property(lambda self: self._coordinates()[2].copy())

    def support(self, v):
        v = as_element(self.space, v)
        c = 0.0 if self.centre is None else self.space.inner(v, self.centre)
        return exact(c + self.radius * math.sqrt(max(self.space.inner(v, self.inverse(v)), 0.0)),
                     "<q, c> + r <q, A^-1 q>^(1/2), with A^-1 as given or applied (its accuracy is the operator's; "
                     "A self-adjoint and coercive as asserted)", by_rule(ASSERTED))

    def member(self, x):
        x = as_element(self.space, x)
        d = x if self.centre is None else x - self.centre
        if self.shape_operator is None:
            return Truth.UNKNOWN
        q = self.space.inner(self.shape_operator(d), d)
        return decide(max(math.sqrt(max(q, 0.0)) - self.radius, 0.0), max(self.radius, 1e-300))


class Slab(ConvexSet):
    """{e : |<a, e - c>| <= w}: the noise region of one fixed estimator (inf:400-407, inf:436-443)."""

    def __init__(self, space, normal, half_width, centre=None, name="slab"):
        self.normal, self.half_width = as_element(space, normal), float(half_width)
        self.centre = centre
        super().__init__(space, name, _closed_convex("a slab between two parallel hyperplanes", space.dim == 1))

    def _value(self, x):
        d = x if self.centre is None else x - self.centre
        return self.space.inner(self.normal, d)

    def member(self, x):
        return decide(max(abs(self._value(as_element(self.space, x))) - self.half_width, 0.0),
                      max(self.half_width, 1e-300))

    def support(self, mu):
        mu = as_element(self.space, mu)
        a = self.normal
        tau = self.space.inner(mu, a) / self.space.inner(a, a)
        r = self.space.norm(mu - a * float(tau))
        c = 0.0 if self.centre is None else self.space.inner(mu, self.centre)
        along = float(c + abs(tau) * self.half_width)       # attained by points of the slab: a lower bound always
        t = decide(r, max(self.space.norm(mu), 1e-300))
        if t is Truth.FALSE:
            return exact(math.inf, "a component along which the slab is unbounded")
        if t is Truth.UNKNOWN:                               # undecided to rounding: +inf, at least the value along a
            return Enclosure(math.inf, along, math.inf, DERIVED, "the component off the normal is at the rounding of "
                             "the direction: sigma is <mu, c> + |tau| w (orthogonal) or +inf")
        return exact(along, "<mu, c> + |tau| w for mu = tau a (to rounding)")

    def is_empty(self):
        return Truth.FALSE


class CoordinateBox(ConvexSet):
    """{e : lo <= e <= hi} in R^n (componentwise, standard coordinates; infinite bounds allowed)."""

    def __init__(self, space, lo, hi, name="box"):
        lo, hi = (np.broadcast_to(np.asarray(v, float), (space.dim,)).copy() for v in (lo, hi))
        if np.any(lo > hi):
            raise ValueError("a box needs lo <= hi")
        self.lo, self.hi = lo, hi
        super().__init__(space, name, _closed_convex("a product of closed intervals",
                                                      bool(np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)))))

    def member(self, x):
        v = as_element(self.space, x).values
        excess = float(np.max(np.maximum(self.lo - v, v - self.hi), initial=0.0))
        return decide(max(excess, 0.0), max(float(np.max(np.abs(v), initial=0.0)), 1.0))

    def support(self, mu):
        a = self.space.apply_metric(as_element(self.space, mu).values)
        with np.errstate(invalid="ignore"):
            terms = np.where(a > 0, a * self.hi, np.where(a < 0, a * self.lo, 0.0))
        return exact(np.sum(terms), "sum of a_i hi_i (a_i > 0) and a_i lo_i (a_i < 0), a = W mu")

    def is_empty(self):
        return Truth.FALSE


class Polytope(ConvexSet):
    """{e : A e <= b} in R^n (standard coordinates); supports by linear programming (optimisers/lp.py), vertices in R^1-3
    (convex/polytope.py). approx says what it is of the set `of`: "exact" (it is the set: pg.Polytope(space, A, b)),
    "outer" (it contains `of`: half-spaces from support values) or "inner" (it lies in `of`: the hull of points of
    `of`). The package's builders always state it; reporting one as the other understates the set."""

    APPROX = ("exact", "outer", "inner")

    def __init__(self, space, A, b, name="polytope", approx="exact", of=None):
        self.A, self.b = np.atleast_2d(np.asarray(A, float)), np.asarray(b, float).ravel()
        if self.A.shape != (len(self.b), space.dim):
            raise ValueError(f"A must be ({len(self.b)}, {space.dim})")
        if approx not in self.APPROX or (approx == "exact") != (of is None):
            raise ValueError('approx is "exact" (no of=) or "outer" / "inner" with the set it approximates as of=')
        self.approx, self.of = approx, of
        cl = _closed_convex("an intersection of closed half-spaces")
        if of is not None:
            cl.append(Claim(f"{approx} approximation of {of.name}", DERIVED,
                            "half-spaces from certified support upper ends" if approx == "outer" else
                            "the hull of points of the set (each checked feasible to rounding)", key=approx))
        super().__init__(space, name, cl)

    def member(self, x):
        v = as_element(self.space, x).values
        excess = float(np.max(self.A @ v - self.b, initial=0.0))
        return decide(max(excess, 0.0), max(float(np.max(np.abs(self.b), initial=0.0)), 1.0))

    def support(self, mu):
        from ..optimisers.lp import lp_max
        return exact(lp_max(self.space.apply_metric(as_element(self.space, mu).values), self.A, self.b)[0],
                     "a linear programme (optimisers/lp.py): optimal to the solver's tolerance", CHECKED)

    def is_empty(self):
        return Truth.of(self.support(self.space.zero) == -math.inf)

    def vertices(self):
        """The vertices (k, n) from the exact intersection of all half-spaces, in R^1-3 (convex/polytope.py); refused
        for an empty, unbounded or lower-dimensional polytope."""
        from .polytope import vertices
        return vertices(self.A, self.b)

    def _like(self, A, b, of):
        return Polytope(self.space, A, b, self.name, self.approx if of is not None else "exact", of)

    def __add__(self, x):                                    # a translate keeps the relation to the translated set
        if stated_kind(x) is SET:
            return super().__add__(x)
        xe = as_element(self.space, x)
        return self._like(self.A, self.b + self.A @ xe.values, None if self.of is None else self.of + xe)

    __radd__ = __add__

    def __rmul__(self, c):
        if np.ndim(c) != 0 or c <= 0:
            return super().__rmul__(c)
        return self._like(self.A, c * self.b, None if self.of is None else c * self.of)

    def __and__(self, other):
        """A polytope: outer (inner) of S when both are outer (inner) of the same S, else exact (itself)."""
        if not isinstance(other, Polytope):
            return super().__and__(other)
        same = self.approx == other.approx and self.of is other.of
        return Polytope(self.space, np.r_[self.A, other.A], np.r_[self.b, other.b], f"{self.name} ∩ {other.name}",
                        self.approx if same else "exact", self.of if same else None)


class AffineSubset(ConvexSet):
    """point + direction, with direction a closed subspace: the fibre m + ker G of exact data, G x + mul R."""

    def __init__(self, point, direction, name="affine set", least_norm=False):
        self.point, self.direction, self._least_norm = point, direction, least_norm
        super().__init__(direction.space, name, _closed_convex("a translate of a closed subspace",
                                                               bool(getattr(direction, "dim", 1) == 0)))

    def member(self, x):
        return self.direction.member(x - self.point)

    def support(self, v):
        v = as_element(self.space, v)
        s = support_of(self.direction, v)
        return s + float(self.space.inner(v, self.point))   # [l, u] + <v, p>: +inf stays +inf

    def summary(self):
        """The element of least norm (a summary, not a state: no guarantee, el:533)."""
        return self.point if self._least_norm else self.direction.perp().project(self.point)

    def is_empty(self):
        return Truth.FALSE

    def __repr__(self):
        return f"{self.name}: a point + {self.direction!r}"


def singleton(x):
    """{x} as an affine set."""
    from .subspaces import Zero
    return AffineSubset(x, Zero(x.space), name=f"{{{getattr(x, 'name', 'x')}}}")
