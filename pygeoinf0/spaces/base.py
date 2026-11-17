"""The hierarchy of spaces, each level with its contract (plan 068 P1; the architecture review, section 2, L3), and
the basic spaces.

    VectorSpace(Set)   a vector space over its field ("R" or "C"): X.field, X.zero, X.linear_combination(c, xs),
                       X.member(x) (sets/base.py: TRUE for an element of this very space)
    NormedSpace        + X.norm(x), the dual space X.dual() (spaces/dual.py) and its elements X.functional(...)
    BanachSpace        + completeness (a theorem about the space, recorded as a claim; no code can check it)
    HilbertSpace       + X.inner(x, y) and the Riesz map X.riesz(...): the representer in X of a bounded functional

A level's operations exist only where the level does. A Banach space that is not a Hilbert space (L^p for p != 2,
C(K)) has no inner product and no Riesz map, so it has no `inner` or `riesz` attribute at all: there is nothing to call
(plan 065 made them refuse; plan 068 P1 took them out of the classes). A complex space (field "C": Euclidean(n,
field="C"), pg.Complexified) is, for the rest of the package, the real Hilbert space X_R with <x, y>_R = Re <x, y>;
its complex inner product is X.cinner.

L2(mu)          classes of square-integrable functions on a domain (interval, box, disk, cylinder, ball)
Lp(mu, p)       the Banach space for p >= 1; Lp(mu, 2) is L2(mu)
Euclidean(n)    R^n with the dot product, with positive weights, or with a symmetric positive definite metric;
                Euclidean(n, field="C") is C^n with a Hermitian metric (the complex structure: spaces/complex.py)

Function spaces have no basis and no dimension. Their elements carry an explicit realisation
(a backend chosen from the domain, see realisations/), automatic and replaceable; the space itself
only knows its measure. A space is the same whatever realises its elements (plan 069 P1, decision Q4): X == Y by the
measure and the norm, X.realised(r) == X is the space with its elements realised by r, and elements realised
differently are combined only after a conversion, or refused with RealisationMismatch (spaces/identity.py).
"""
import math

import numpy as np

from ..core.claims import ASSERTED, CHECKED, THEOREM, Claim, Claims
from ..core import fits, lazy
from ..core.numbers import field_scalar
from ..core.logic import Truth
from .identity import require
from .elements import Family, Function, Vector, VectorFamily, as_array_function
from .elements import _name_of, hints
from .labels import LabelledCoordinates, Labels
from ..measures import CountingMeasure, DensityMeasure, LebesgueMeasure, Measure, check_space_measure, measure_of
from ..sets import Interval, Set
from ..core.errors import PointEvaluationUndefined, PriorUnstated, SpaceMismatch, Unsupported  # noqa: E402

FIELDS = ("R", "C")             # the scalars of a vector space: the real or the complex numbers

PRIOR_CONSTRUCTORS = (     # the named priors (plan 069 P6, decision Q2: no space picks a prior for the caller)
    "pygeoinf0.probability.legendre_decay(X, decay, modes=None): on L2 of an interval, C phi_k = decay^(2k) phi_k for the "
    "orthonormal Legendre functions phi_k",
    "pygeoinf0.probability.random_features(X, length_scale, features, seed): on L2 of a box, sphere or ball, the "
    "finite-rank Gaussian of random Fourier features (approximating the squared-exponential covariance)",
    "pygeoinf0.probability.cylindrical(X, alpha) or pg.CylindricalGaussian(X, alpha): N(0, I/alpha) on a Hilbert space "
    "(not a measure on an infinite-dimensional one: exact variances of bounded functionals, no samples)",
    "X.standard_gaussian(): the standard Gaussian of a finite-dimensional inner product (Euclidean, data and complex "
    "coordinate spaces, direct sums of them)",
    "pygeoinf0.probability.circular.circular(X, covariance, mean): a circular complex Gaussian of the coordinates",
    "pg.GaussianMeasure(X, covariance=C, mean=m): a covariance of your own")


def no_prior(X, what="X.gaussian()", given=None):
    """The refusal of a prior nobody named (plan 069 P6): PriorUnstated naming the constructors."""
    kw = f" with {', '.join(f'{k}={v!r}' for k, v in given.items())}" if given else ""
    return PriorUnstated(
        f"{what}{kw} on {X}: no space chooses a prior for the caller (decision Q2; until plan 069 P6 it named four "
        "different objects); name one: legendre_decay, random_features, cylindrical, standard_gaussian, circular, or "
        "a GaussianMeasure with a covariance of your own", needs=PRIOR_CONSTRUCTORS)


class VectorSpace(Set):
    """A vector space over its field (X.field: "R" or "C"): a Set with a zero and linear combinations.

    Its elements know their space (sets/base.py); write x + y and a * x, or X.linear_combination(coefficients,
    elements). member(x) is TRUE for an element of this
    space and FALSE otherwise: an element of another space belongs here only after a conversion (an inclusion
    operator), and an array is a vector's coordinates, not a vector. An element of X.realised(r) is an element of X
    (decision Q4: a space is the same whatever realises its elements; spaces/identity.py)."""

    field = "R"

    def member(self, x):
        return Truth.of(getattr(x, "space", None) == self)

    @property
    def zero(self):
        """The zero vector (each space builds its own)."""
        raise NotImplementedError(f"{type(self).__name__} does not build its zero vector")

    def linear_combination(self, coefficients, elements):
        """sum_i c_i x_i for scalars c_i of the field and elements x_i of this space (an element of another space is
        refused: SpaceMismatch; a complex coefficient on a real space is refused by the element's own product)."""
        cs, xs = list(coefficients), list(elements)
        if len(cs) != len(xs):
            raise ValueError(f"{len(cs)} coefficients for {len(xs)} elements")
        for x in xs:
            if self.member(x) is not Truth.TRUE:
                raise SpaceMismatch(f"{getattr(x, 'name', x)!s} is not an element of {self}")
            require(x.space, self, "", getattr(x, "name", None))       # realised as this space (RealisationMismatch)
        if not xs:
            return self.zero
        out = cs[0] * xs[0]
        for c, x in zip(cs[1:], xs[1:]):
            out = out + c * x
        return out

    def gaussian(self, *args, **kw):
        """Refused (plan 069 P6, decision Q2): no space picks a prior; PriorUnstated names the constructors."""
        raise no_prior(self, given=kw)

    def standard_gaussian(self):
        """The standard Gaussian of a finite-dimensional inner product (CoordinateSpace, finite direct sums); refused
        here: on an infinite-dimensional Hilbert space the identity is not trace class (white noise is not an element),
        so a prior has to be named."""
        raise no_prior(self, "X.standard_gaussian()")

    @property
    def claims(self):
        return Claims()


class NormedSpace(VectorSpace):
    """A vector space with a norm, its dual space X.dual() and its functionals X.functional(...) (abstract).

    X' = X.dual() holds the bounded linear functionals with the operator norm; X.functional(...) builds an element of
    X' (each space says from what)."""

    def norm(self, x):
        raise NotImplementedError

    def distance(self, x, y):
        return self.norm(x - y)

    def dual(self):
        """The dual space X' (spaces/dual.py): a Banach space whatever X is, with X as its primal."""
        from .dual import DualSpace
        if self.__dict__.get("_dual") is None:
            self.__dict__["_dual"] = DualSpace(self)
        return self.__dict__["_dual"]

    def functional(self, *args, **kwargs):
        """A bounded linear functional on X, a spaces.dual.Functional of X.dual(); each space says from what it builds
        one."""
        raise Unsupported(f"the bounded linear functionals of {self} are not built here")

    def _dual_claims(self):
        """What is known about X' beyond its completeness (the claims of X.dual()); each space adds its theorems."""
        return []


class BanachSpace(NormedSpace):
    """A complete normed space. Completeness is a theorem about the space, recorded as a claim; no
    code can check it."""


class HilbertSpace(BanachSpace):
    """A complete inner-product space with its Riesz map; the norm is the square root of the inner product.

    The Riesz map takes a bounded functional (as each space describes it: a measure, a set, a point, coefficients) to its representer r in X, with
    l(x) = <x, r>; X.functional(spec) is that functional, an element of X.dual()."""

    def inner(self, x, y):
        raise NotImplementedError

    def norm(self, x):
        return math.sqrt(max(self.inner(x, x), 0.0))

    def riesz(self, spec):
        """The Riesz representer of the bounded functional spec (each space says how a functional is described)."""
        raise NotImplementedError(f"{type(self).__name__} does not build Riesz representers")

    def functional(self, spec, name=None):
        """The bounded linear functional given by spec, x -> <x, X.riesz(spec)>: a Functional of X.dual() (its norm is
        the representer's, by Cauchy-Schwarz). One functional; X.riesz([...]) gives several representers at once."""
        from .dual import Functional
        r = self.riesz(spec)
        if isinstance(r, (Family, VectorFamily)):
            raise ValueError("one functional per call: X.riesz([...]) gives the representers of several, and "
                             "pg.Functionals(X.riesz([...])) the operator into R^n")
        return Functional.from_representer(r, name=name)

    def _dual_claims(self):
        conj = " (conjugate-linear: X is complex)" if self.field == "C" else ""
        return [Claim(f"isometric to {self} by the Riesz map{conj}", THEOREM, "the Riesz representation theorem")]


# ------------------------------------------------------------------------------------------ functions
class _FunctionSpace:
    """What every function space shares (L2, Lp, Sobolev): a measure on a domain, a backend that realises elements
    (chosen by realisations.choose), and ways to make elements and families."""

    def _setup(self, measure, tol, realisation=None, sobolev=None):
        self._measure_given = isinstance(measure, Measure)     # False: a set was given, Lebesgue measure assumed
        mu = measure_of(measure)
        if not isinstance(mu, (LebesgueMeasure, DensityMeasure)):
            raise TypeError("function spaces are built here on Lebesgue measure, or a density, on a domain")
        check_space_measure(mu)
        if sobolev is not None and isinstance(mu, DensityMeasure) and not isinstance(mu.set, Interval):
            from ..core.errors import Unsupported
            raise Unsupported(f"Sobolev spaces with a density as their measure on {mu.set!r} are not defined here "
                              "(plan 064 realises L2 with constant and radial densities; a Sobolev space needs the "
                              "density's role in its derivative terms decided)")
        if isinstance(mu.set, Interval) and not mu.set.is_bounded:
            raise NotImplementedError("unbounded intervals have no realisation yet (Legendre series need a bounded interval)")
        self.measure, self.domain, self.tol = mu, mu.set, tol
        from ..realisations import choose
        self.backend = choose(mu, tol, realisation, sobolev)
        self.realisation = self.backend.realisation_name

    def realised(self, realisation):
        """This space (X.realised(r) == X) with its elements realised by r: a name of realisations/registry.py
        ("local", "tensor", "legendre", ...) or a pg.realisations.Cells partition (plan 069 P1, decision Q4). Elements
        made in X and in X.realised(r) are elements of one space, combined only within one realisation
        (spaces/identity.py: RealisationMismatch otherwise). X itself when r is X's realisation."""
        if getattr(realisation, "name", realisation) == self.realisation:
            return self
        return self._with_realisation(realisation)

    def _with_realisation(self, realisation):
        from ..core.errors import Unsupported
        raise Unsupported(f"{self} has one realisation here ({self.realisation})")

    def _given_measure(self):
        """What the space was built on: the measure when one was given, else the domain (Lebesgue by convention)."""
        return self.measure if self._measure_given else self.domain

    def restrict_to(self, D, realisation=None, label=None):
        """pg.Restrict(X, D), u -> u|_D into a new space X(D) = R.codomain (plan 035 P3; Sobolev: P4); label names the
        piece in the operator's name."""
        from ..operators.pieces import Restrict
        return Restrict(self, D, realisation, label=label)

    def restrictions(self, partition, realisation=None):
        """R = (R_k)_k: X(Omega) -> (+)_k X(Omega_k), u -> (u|_(Omega_k))_k over the leaves of a partition of the
        domain, into the broken pg.PiecewiseSpace(P, X, broken=True) (operators/pieces.Restrictions: for L2 an isometry
        onto it, which is also the glued space; for a Sobolev space injective onto the glued subspace); R.glue is its
        inverse on L2. One operator per partition (cached)."""
        from ..operators.pieces import restrictions
        cache = self.__dict__.setdefault("_restrictions", [])
        for P, r, R in cache:
            if P is partition and r == realisation:
                return R
        R = restrictions(self, partition, realisation)
        cache.append((partition, realisation, R))
        return R

    @property
    def interval(self):
        if not isinstance(self.domain, Interval):
            raise AttributeError(f"{self} is not a space on an interval; use .domain")
        return self.domain

    @property
    def domain_dim(self):
        """The dimension of the domain (the space itself is infinite-dimensional)."""
        return self.backend.dim

    # elements --------------------------------------------------------------------------
    def element(self, f, *, breaks=None, name=None, support=None, resolution=None):
        """The element with representative f, a function of the Cartesian coordinates (x), (x, y) or (x, y, z),
        vectorised or not; or a constant. support=((lo, hi), ...): f vanishes outside that box (realisation="local").
        resolution=w: f's narrowest feature width; the fit samples at w/2 and its claims state w met (core/fits.py).
        A fit with neither breaks=, support= nor resolution= states the resolution it got, with a ResolutionWarning."""
        given = breaks
        if support is not None:
            from ..realisations.series.local import with_support
            from ..realisations.registry import provides
            f, breaks = with_support(f, breaks, support)
            if isinstance(self.domain, Interval) and not provides(self.backend, "breaks_per_axis"):
                breaks = tuple(breaks[0])                   # an interval's own backends take flat breaks
        if isinstance(f, Function):
            require(f.space, self, f"{f.name} is in {f.space}, not in {self}", f.name)   # realised as this space
            return f if resolution is None else f.with_resolution(resolution)
        if isinstance(f, (int, float, np.number)) and np.ndim(f) == 0:
            return self.constant(float(f))
        if not callable(f):
            raise TypeError("give a function of the coordinates (or a number)")
        member = self._membership(f)
        g, _ = as_array_function(f, self.backend.probe())
        name = name or _name_of(f)
        bank = callable(getattr(f, "fit_breaks", None)) and hasattr(f, "support")
        rec = fits.record_of(f, given, support, resolution, name, bank)
        breaks, expect = hints(f, self, breaks)          # bank functions: natural breaks, no silent zero (plan 035)
        if bank and breaks is None and resolution is None and support is None:
            fits.bank_record(rec, getattr(f, "feature_width", lambda: None)())    # its width as resolution (fix2)
        out = Function(self, source=g, name=name, breaks=breaks, expect=expect, res=rec)
        if member is not None:
            out._memo["membership"] = member
        return out

    def _membership(self, f):
        """The membership claim of a function that carries a Regularity (a bank function), by the rules of
        spaces/membership.py; refused when they put it outside this space; None for a plain callable or a space without
        a rule (plan 065 A1: f.element(M) and M(f) give one answer)."""
        if not callable(getattr(getattr(f, "regularity", None), "sobolev_bound", None)):
            return None
        from .membership import check, has_rule, refusal
        if not has_rule(self):
            return None
        ok, reason, claim = check(f, self)
        if not ok:
            raise refusal(f, self, reason)
        return claim

    def _element_claim(self, f, ok):
        """The first claim of an element f given by a function (Function.claims): f is in this space."""
        how = "its realisation converged with finite norm" + ("" if getattr(self, "has_classes", True)
                                                               else " and has no jump")
        return Claim(f"{f.name} is in {self}", CHECKED if ok else "unknown",
                     how if ok else "the realisation did not converge everywhere")

    def __call__(self, f, **kw):
        return self.element(f, **kw)

    def coordinate(self, k):
        """The k-th Cartesian coordinate function (0: x, 1: y, 2: z)."""
        name = "xyz"[k]
        src = lambda *X, k=k: np.asarray(X[k], float)  # noqa: E731
        return Function(self, pl=self.backend.coordinate(k), source=src, name=name, exact=src)

    @property
    def x(self):
        """The coordinate function x, so that np.sin(2 * np.pi * M.x) is an element."""
        return self.coordinate(0)

    def natural(self, name):
        """A natural coordinate as an element: "r", "phi" (disk, cylinder, ball), "theta" (ball, colatitude), "z"."""
        D = self.domain
        names = getattr(D, "internal_names", ())
        if name in names:
            k = names.index(name)
            src = lambda *X: D.from_cartesian(*X)[k]  # noqa: E731
        else:
            raise AttributeError(f"{D!r} has no natural coordinate {name!r} (it has {', '.join(names) or 'x'})")
        return Function(self, source=src, name=name, res=fits.Record("formula", name=name))

    @property
    def r(self):
        return self.natural("r")

    @property
    def phi(self):
        return self.natural("phi")

    @property
    def theta(self):
        return self.natural("theta")

    @property
    def y(self):
        return self.coordinate(1)

    @property
    def z(self):
        return self.coordinate(2)

    def constant(self, c):
        src = lambda *X, c=c: np.full(np.broadcast(*X).shape, c)  # noqa: E731
        return Function(self, pl=self.backend.constant([c]), source=src, name=f"{c:g}", exact=src)

    @property
    def zero(self):
        return self.constant(0.0)

    def family(self, f, *, name="F", breaks=None, sparse=None, resolution=None, **params):
        """Many elements at once.

        M.family(lambda x, a: np.sin(a * x), a=np.linspace(1, 30, 100_000))   one formula, many parameters
        M.family([f, g, h])                                                     a list of elements or callables

        On an interval, localised families (each member negligible on most of the domain, like narrow data
        kernels) are stored sparsely; sparse=None decides from a sample, True or False forces the choice.
        resolution=w: the members' narrowest feature width, as for element (core/fits.py).
        """
        return fits.family_of(self, f, name, breaks, sparse, resolution, params)

    def product_family(self, factors, *, name="K", **params):
        """Many product functions k_i = f_1(xi_1) ... f_d(xi_d) in the natural coordinates of the domain
        (Box: x, y, z; Disk: r, phi; Cylinder: r, phi, z; Ball: r, theta, phi), one per parameter value:

            K = M.product_family([lambda r, rc: np.exp(-((r - rc) / 0.05) ** 2),
                                  lambda theta, tc: np.exp(-((theta - tc) / 0.1) ** 2),
                                  None],                                  # None: constant in phi
                                 rc=radii, tc=colatitudes)

        Each factor receives only the parameters its signature names. Distinct factors are realised once per
        axis (1-D, dense or sparse), so a lattice of 100,000 kernels costs a few hundred 1-D functions. This is
        the fast route for many localised kernels; M.family is the general (dense tensor) one."""
        return Family(self, self.backend.product_family(list(factors), params, name=name), name=name)

    def sum(self, elements):
        """The sum of many elements, computed in one pass (Family or iterable of elements)."""
        if isinstance(elements, Family):
            return elements.sum()
        items = [self.element(e) for e in elements]
        B = self.backend
        return Function(self, pl=B.colsum(B.hstack([e.realisation for e in items])), name=f"sum of {len(items)} elements")

    # measures and integrals ----------------------------------------------------------------
    def _pl_of(self, u):
        if isinstance(u, (Function, Family)):
            require(u.space, self, f"{u.name} is in {u.space}, not in {self}", u.name)    # realised as this space
            return u.realisation
        raise TypeError(f"expected an element or family of {self}, got {type(u).__name__}")

    def integrate(self, f, over=None):
        if lazy.is_lazy(f):                              # by linearity over the terms (core/lazy.py, plan 022 P7)
            return sum(c * self.integrate(t, over) for c, t in f._terms)
        v = self.backend.integral(self._pl_of(f), over)
        return float(v[0]) if isinstance(f, Function) else v

    def measure_of_cell(self, c, d):
        return self.backend.measure_of_cell(c, d)

    def _realisation_error(self, pl):
        return self.backend.error(pl)

    def _same_measure(self, other):
        """The same kind of space on equal measures (up to null sets), whatever realises their elements (plan 069 P1,
        decision Q4): the realisation is not part of a space's identity (spaces/identity.py)."""
        return type(other) is type(self) and self.measure.same_up_to_null_set(other.measure)

    def _mu_repr(self):
        if isinstance(self.measure, DensityMeasure):
            if isinstance(self.domain, Interval):
                return f"{self.domain}, {self.measure.name}(x) dx"
            return f"{self.domain}, {repr(self.measure).rsplit(' on ', 1)[0]}"         # c dσ, rho(r) dx (plan 064)
        return f"{self.domain}, Lebesgue"

    def _measure_claim(self):
        if self._measure_given:
            return Claim(f"measure: {self.measure}", ASSERTED, "given by the user")
        return Claim(f"measure: {self.measure}", ASSERTED,
                     "default: a set was given, not a measure; a function space on a domain means "
                     "Lebesgue measure on it by convention")


class _LebesgueBase(_FunctionSpace):
    """What L2 and Lp add to a function space: indicators of subintervals (elements of L^p, not of H^s for
    s >= 1/2), and the refusals of point evaluation and of draws without a named prior (plan 069 P6). (The Riesz map is L2's: L^p
    for p != 2 has none.)"""

    def indicator(self, J, name=None):
        """The indicator function of a subinterval J (its endpoints are breakpoints, so it is exact)."""
        J = Interval(J) if isinstance(J, str) else J
        I = self.interval
        brk = [p for p in (J.a, J.b) if I.a < p < I.b]
        return Function(self, source=lambda t: np.asarray(J.contains(np.asarray(t, float)[..., None]), float),
                        name=name or f"1_{J}", breaks=brk, res=fits.Record("breaks", name=name or f"1_{J}"))

    def random(self, size=None, seed=None, name="m", **kw):
        """Refused (plan 069 P6): a draw needs a measure, and no function space picks one (white noise is not square
        integrable). Draw from a named prior: pygeoinf0.probability.legendre_decay(X, decay).sample(size, seed), or
        random_features(X, length_scale, features, seed).sample(size, seed)."""
        raise no_prior(self, "X.random()", given=kw)

    def evaluation(self, x0):
        raise PointEvaluationUndefined(
            f"point evaluation f -> f({x0}) is not defined on {self}: its elements are classes of functions "
            f"equal almost everywhere and a point has measure zero, and no bounded linear functional on {self} "
            f"agrees with it on continuous functions. Use a local average, or a space of continuous or Sobolev "
            f"functions.")


class _FunctionHilbertSpace(_FunctionSpace, HilbertSpace):
    """Inner products and norms of a Hilbert function space, from its backend (L2, Sobolev)."""

    def average(self, S, weight=None):
        """The bounded functional u -> int_S u w dsigma / int_S w dsigma over a set S in the closure of the domain (w = 1
        by default; on L2(rho dx) the average for the space's measure): the Functional of the normalised measure."""
        from .representers import average
        return average(self, S, weight)

    def inner(self, f, g):
        lin = lazy.by_terms(self, f, g, _FunctionHilbertSpace.inner)
        if lin is not None:
            return lin
        G = self.backend.gram(self._pl_of(f), self._pl_of(g))
        if (isinstance(f, Function) or isinstance(g, Function)) and hasattr(G, "toarray"):
            G = G.toarray()     # one row or column; local Grams are scipy.sparse beyond realisations.series.local.DENSE_MAX
        if isinstance(f, Function) and isinstance(g, Function):
            return float(G[0, 0])
        if isinstance(f, Function):
            return G[0]
        if isinstance(g, Function):
            return G[:, 0]
        return G

    def norm(self, f):
        sq = self.backend.sqnorms(self._pl_of(f))
        r = np.sqrt(np.maximum(sq, 0.0))
        return float(r[0]) if isinstance(f, Function) else r


class L2(_LebesgueBase, _FunctionHilbertSpace):
    """L2(mu): classes of square-integrable functions, <f, g> = int f g dmu.

    L2(mu) takes a measure. L2(D) for a domain D (an Interval, or "[0, 1]", a Box, Disk, Cylinder or Ball) is a
    shortcut for L2(LebesgueMeasure(D)): the convention behind writing L2(0, 1). The default is shown when the
    space is printed and recorded in M.claims. Spaces whose measures differ only on a null set are equal.
    """

    def __init__(self, measure, tol=1e-14, realisation=None):
        self._setup(measure, tol, realisation)

    def _with_realisation(self, realisation):
        return L2(self._given_measure(), self.tol, realisation)

    def riesz(self, mu):
        """The Riesz representer of f -> int f dmu: a measure, a set (its natural measure) or a point, or a list of
        them (a Family). In L2(lambda) it exists only for measures with a square-integrable density with respect to
        lambda (spaces/dual.py): lambda itself (representer 1), a subinterval (its indicator), and densities w dx on
        sets of full dimension (sub-boxes, disks, balls, the domain): d mu / d lambda, kept symbolic
        (spaces/representers.py). Lower-dimensional measures (points, paths, surfaces) are not bounded on L2."""
        from .representers import riesz_l2
        return riesz_l2(self, mu)

    def inner(self, f, g):                       # symbolic representers (spaces/representers.py) pair exactly
        lin = lazy.by_terms(self, f, g, L2.inner)
        if lin is not None:
            return lin
        from .representers import inner
        return inner(self, f, g, super().inner)

    def norm(self, f):
        from .representers import norm
        return norm(self, f, super().norm)

    @property
    def claims(self):
        return Claims([self._measure_claim(),
                       Claim("Hilbert space (complete)", THEOREM, "Riesz-Fischer"),
                       Claim("separable", THEOREM, "polynomials are dense in L2 of a finite measure on a bounded domain"),
                       Claim("point evaluation is not defined", THEOREM, "elements are classes modulo null sets"),
                       *getattr(self.backend, "claims", ())])   # a realisation's own claims (cells: the tiling, plan 065)

    def __eq__(self, other):
        return self._same_measure(other)

    def __hash__(self):
        return hash(("L2", repr(self.domain)))

    def __repr__(self):
        return f"L2({self._mu_repr()})"


class LpSpace(_LebesgueBase, BanachSpace):
    """L^p(mu) for 1 <= p <= inf, p != 2: a Banach space whose norm does not come from an inner product."""

    def __init__(self, measure, p, tol=1e-14, realisation=None):
        p = float(p)
        if p < 1:
            raise ValueError("for p < 1 the triangle inequality fails, so ||.||_p is not a norm and L^p is not a Banach space")
        self.p = p
        self._setup(measure, tol, realisation)

    def _with_realisation(self, realisation):
        return LpSpace(self._given_measure(), self.p, self.tol, realisation)

    def norm(self, f):
        out = self.backend.lp_norm(self._pl_of(f), self.p)
        return float(out[0]) if isinstance(f, Function) else out

    # no riesz and no inner: L^p for p != 2 has no inner product (the parallelogram law fails), so no Riesz map
    def _conjugate(self):
        return "∞" if self.p == 1 else "1" if math.isinf(self.p) else f"{self.p / (self.p - 1):g}"

    def functional(self, nu, scale=1.0, name=None):
        """u -> scale * int u dnu (nu a measure or a set of full dimension), bounded on L^p(mu) iff d nu / d mu is in
        L^p' (1/p + 1/p' = 1; the rule of spaces/dual.py, refused otherwise): a Functional of the dual, which is L^p'
        for p < inf, with norm ||d nu / d mu||_p' (computed for Lebesgue measure on a set)."""
        from .dual import Functional
        return Functional.from_measure(self, nu, scale, name)

    def average(self, S):
        """u -> (1/|S|) int_S u dmu over a set S of full dimension inside the domain."""
        from ..measures import measure_of
        mu = measure_of(Interval(S) if isinstance(S, str) else S)
        return self.functional(mu.normalised(), name=f"average over {mu.set}")

    def _dual_claims(self):
        if math.isinf(self.p):
            return [Claim(f"contains L^1 isometrically, properly (the bounded finitely additive measures)", THEOREM,
                          "the dual of L^∞ (Hildebrandt, Fichtenholz-Kantorovich)")]
        return [Claim(f"isometric to L^{self._conjugate()} of the same measure (1/p + 1/p' = 1)", THEOREM,
                      "the Riesz representation theorem for L^p, 1 <= p < ∞, of a σ-finite measure")]

    @property
    def claims(self):
        return Claims([self._measure_claim(),
                       Claim("Banach space (complete)", THEOREM, "Riesz-Fischer, p >= 1"),
                       Claim("not a Hilbert space", THEOREM, "the parallelogram law fails for indicators of disjoint sets"),
                       Claim("norm computed approximately", CHECKED,
                             "|f|^p is not a polynomial; Gauss-Legendre quadrature of high order on each piece")])

    def __eq__(self, other):
        return self._same_measure(other) and self.p == other.p

    def __hash__(self):
        return hash(("Lp", self.p, repr(self.domain)))

    def __repr__(self):
        p = "∞" if math.isinf(self.p) else f"{self.p:g}"
        return f"L{p}({self._mu_repr()})"


def Lp(measure, p, tol=1e-14, realisation=None):
    """L^p(mu): L2 (a Hilbert space) when p == 2, otherwise a Banach space."""
    return L2(measure, tol, realisation) if p == 2 else LpSpace(measure, p, tol, realisation)


# ------------------------------------------------------------------------------------------ R^n

class CoordinateSpace(LabelledCoordinates, HilbertSpace):
    """R^n or C^n with coordinates and a metric: the code Euclidean(n) and Euclidean(n, field="C") share.

    The inner product is <x, y> = y^H W x, W a self-adjoint positive definite operator on the standard space
    (spaces/metrics.py). Build one with Euclidean(n, ...) (R^n) or
    Euclidean(n, field="C", ...) (C^n: spaces/complex.py). C^n is not an instance of Euclidean: the routes written
    for real coordinates (dense matrices of operators, solvers, Gaussian measures, data-space decompositions) test
    isinstance(X, Euclidean) and keep refusing C^n, whose real dimension is 2n."""

    dtype = float

    def __init__(self, n, weights=None, metric=None, covariance=None, labels=None):
        if int(n) != n or n < 1:
            raise ValueError("the dimension must be a positive integer")
        if sum(x is not None for x in (weights, metric, covariance)) > 1:
            raise ValueError("give one of weights, metric or covariance")
        self.dim = int(n)
        self._metric = None
        self._standard = None
        self._gaussian = None
        if weights is not None:
            self._metric = self._make(np.asarray(weights, float), "metric")
        elif metric is not None:
            self._metric = self._make(metric, "metric")
        elif covariance is not None:
            self._metric = self._make(covariance, "covariance")
        self._labels = None if labels is None else Labels(labels, self.dim)

    def _like(self):
        """A standard space of the same dimension and field (no metric, no labels)."""
        return Euclidean(self.dim, field=self.field)

    def _make(self, x, kind):
        from .metrics import make_metric
        return make_metric(x, self.standard, kind)

    @property
    def standard(self):
        """The same vectors with the dot product (the space the metric operator acts on)."""
        if self._metric is None:
            return self
        if self._standard is None:
            self._standard = self._like()
            self._standard._labels = self._labels
        return self._standard

    def with_metric(self, W):
        if not self.is_standard:
            raise ValueError("with_metric starts from the standard space; use .standard first")
        out = self._like()
        out._standard, out._labels = self, self._labels
        out._metric = out._make(W, "metric")
        return out

    def with_covariance(self, C):
        if not self.is_standard:
            raise ValueError("with_covariance starts from the standard space; use .standard first")
        out = self._like()
        out._standard, out._labels = self, self._labels
        out._metric = out._make(C, "covariance")
        return out

    # metric helpers ---------------------------------------------------------------------------
    @property
    def is_standard(self):
        return self._metric is None

    @property
    def weights(self):
        """The diagonal of W when W is diagonal (for a covariance C: 1 / diag C); else None."""
        return None if self._metric is None else self._metric.weights()

    @property
    def metric(self):
        """The metric W as an operator on the standard space (None for the dot product)."""
        if self._metric is None:
            return None
        m = self._metric
        return m.op if m.kind == "metric" else m.op.inv

    @property
    def covariance(self):
        """W^-1 as an operator on the standard space: the covariance of the space's standard Gaussian."""
        if self._metric is None:
            from ..operators.linear import Identity
            return Identity(self)
        m = self._metric
        return m.op if m.kind == "covariance" else m.op.inv

    def apply_metric(self, V):
        """W @ V for V of shape (n,) or (n, k)."""
        return V if self._metric is None else self._metric.apply(V)

    def solve_metric(self, V):
        """W^-1 @ V for V of shape (n,) or (n, k)."""
        return V if self._metric is None else self._metric.solve(V)

    def metric_matrix(self):
        if self.dim > 6000:
            raise MemoryError(f"the metric of {self!r} is not formed as a dense matrix")
        eye = np.eye(self.dim, dtype=self.dtype)
        return eye if self._metric is None else self._metric.apply(eye)

    def to_orthonormal(self, V):
        """Coordinates in an orthonormal basis: L^H v, where W = L L^H."""
        return V if self._metric is None else self._metric.to_orthonormal(V)

    def from_orthonormal(self, Z):
        return Z if self._metric is None else self._metric.from_orthonormal(Z)

    # elements ---------------------------------------------------------------------------------
    def element(self, values, name="v"):
        return Vector(self, values, name=name)

    def __call__(self, values, name="v"):
        return Vector(self, values, name=name)

    def family(self, values, name="V"):
        return VectorFamily(self, values, name=name)

    @property
    def zero(self):
        return Vector(self, np.zeros(self.dim, dtype=self.dtype), name="0")

    def basis(self, i):
        e = np.zeros(self.dim, dtype=self.dtype)
        e[i] = 1.0
        return Vector(self, e, name=f"e{i}")

    def linear_combination(self, coefficients, elements):
        """sum_i c_i x_i, formed on the coordinates in one product (VectorSpace.linear_combination)."""
        cs, xs = list(coefficients), list(elements)
        if len(cs) != len(xs) or not xs:
            return super().linear_combination(cs, xs)
        V = np.array([self._values(x) for x in xs])
        c = np.array([field_scalar(self, a) for a in cs])
        return Vector(self, c @ V, name="Σ")

    def standard_gaussian(self):
        """The standard Gaussian measure of this inner product: covariance operator the identity, coordinates
        N(0, W^-1). On Euclidean(n, covariance=C) its coordinates are N(0, C): the noise model of the data space.
        (X.gaussian() refuses since plan 069 P6: it named four different objects.)"""
        if getattr(self, "_gaussian", None) is None:
            from ..probability import GaussianMeasure
            self._gaussian = GaussianMeasure(self, name="N(0, I)")
        return self._gaussian

    def random(self, size=None, seed=None, name="v"):
        """A draw from the standard Gaussian measure of this inner product (coordinates N(0, W^-1)): the one law meant
        on a finite-dimensional inner product space (X.standard_gaussian())."""
        out = self.standard_gaussian().sample(size, seed)
        out.name = name
        return out

    # geometry ---------------------------------------------------------------------------------
    def _values(self, u):
        if isinstance(u, (Vector, VectorFamily)):
            if u.space != self:
                raise SpaceMismatch(f"{u.name} is in {u.space}, not in {self}")
            return u.values
        raise TypeError(f"expected an element of {self}")

    def inner(self, x, y):
        """<x, y> = x . W y (for C^n the real inner product Re y^H W x of C^n_R: spaces/complex.py)."""
        X, Y = self._values(x), self._values(y)
        if X.ndim == 1 and Y.ndim == 1:
            return float(X @ self.apply_metric(Y))
        return np.atleast_2d(X) @ self.apply_metric(np.atleast_2d(Y).T)

    def norm(self, x):
        if isinstance(x, VectorFamily):
            return np.sqrt(np.einsum("ij,ji->i", x.values, self.apply_metric(x.values.T)))
        return math.sqrt(max(self.inner(x, x), 0.0))

    def riesz(self, c):
        """The Riesz representer r of the functional x -> c . x = sum_i c_i x_i (the coefficients c in the standard
        dual basis): <x, r> = c . x, so r = W^-1 c (for C^n, r = W^-1 conj(c): <x, r> = r^H W x). c of shape (n,)
        gives an element, (m, n) a family of m representers."""
        c = np.asarray(c)
        if c.shape[-1:] != (self.dim,) or c.ndim > 2:
            raise ValueError(f"a functional on {self!r} is given by n = {self.dim} coefficients (or an m x {self.dim} "
                             f"array of them), not an array of shape {c.shape}")
        if np.iscomplexobj(c) and self.field == "R":
            raise TypeError(f"complex coefficients of a functional on the real space {self!r}")
        c = np.conj(c) if self.field == "C" else np.asarray(c, float)
        if c.ndim == 1:
            return Vector(self, self.solve_metric(c), name="r")
        return VectorFamily(self, self.solve_metric(c.T).T, name="r")

    @property
    def measure(self):
        if self._metric is not None and self.weights is None:
            raise AttributeError("a non-diagonal metric does not come from a measure")
        return CountingMeasure(range(self.dim), self.weights)

    @property
    def claims(self):
        cl = [Claim("Hilbert space", THEOREM, "finite-dimensional inner-product spaces are complete")]
        if self._metric is not None:
            cl += self._metric.claims
        return Claims(cl)

    def __eq__(self, other):
        if other is self:
            return True
        if not isinstance(other, CoordinateSpace) or other.dim != self.dim or other.field != self.field:
            return False
        la, lb = self._labels, getattr(other, "_labels", None)
        if la is not None and lb is not None and la is not lb and la.items != lb.items:
            return False
        if self.is_standard or other.is_standard:
            m = other._metric if self.is_standard else self._metric
            return m is None or (m.weights() is not None and np.all(m.weights() == 1.0))
        return self._metric.same(other._metric)

    def __hash__(self):
        return hash(("R" if self.field == "R" else "C", self.dim))

    def _symbol(self):
        return f"{self.field}^{self.dim}"

    def __repr__(self):
        lab = "" if self._labels is None else " labelled"
        if self._metric is None:
            return f"{self._symbol()}{lab}"
        if self.weights is not None:
            return f"{self._symbol()} (weighted{lab})"
        return f"{self._symbol()} ({self._metric.kind}: {self._metric.operator_kind}{lab})"


class Euclidean(CoordinateSpace):
    """R^n with an inner product x . W y, or C^n with field="C" (see spaces/metrics.py for W):

    Euclidean(n)                    the dot product: the standard space
    Euclidean(n, weights=w)         W = diag(w)  (L2 of the counting measure with weights w)
    Euclidean(n, metric=W)          W an operator on Euclidean(n) (Diagonal, Matrix, KroneckerMatrix, SparseMatrix, any
                                    self-adjoint positive definite LinearOperator), or an array or scipy.sparse matrix
    Euclidean(n, covariance=C)      W = C^-1, the natural form for noise; C a number, a vector (diagonal), an array,
                                    a sparse matrix or an operator (solved with, never inverted)
    E.with_metric(W), E.with_covariance(C)   the same, from a standard space E
    Euclidean(n, labels=...)        labelled coordinates; E.select(...), E.groups(...) (spaces/labels.py)
    Euclidean(n, field="C", ...)    C^n: <x, y> = y^H W x (linear in x, conjugate-linear in y), W a number, a positive
                                    vector or a Hermitian positive definite matrix (as weights=, metric= or covariance=);
                                    the same metric code; for the rest of the package the real space C^n_R with
                                    Re <x, y>, with J x = i x, X.cinner and real coordinates (spaces/complex.py). It is
                                    a CoordinateSpace with field "C", not an instance of Euclidean (see CoordinateSpace)

    Elements are NumPy arrays wrapped with their space: v.values. The metric is applied, solved with and factored by
    the cheapest route for its type; no n x n matrix is formed unless the metric is one. Labels name coordinates; two
    spaces with different labels are different spaces (unlabelled spaces compare by dimension, field and metric only).
    """

    def __new__(cls, n=None, weights=None, metric=None, covariance=None, labels=None, field="R"):
        if field not in FIELDS:
            raise ValueError(f"the field of a Euclidean space is 'R' (the real numbers) or 'C' (the complex numbers), "
                             f"not {field!r}")
        if field == "C":
            from .complex import _ComplexEuclidean       # C^n: its complex structure lives there
            out = CoordinateSpace.__new__(_ComplexEuclidean)
            out.__init__(n, weights, metric, covariance, labels)
            return out
        return super().__new__(cls)

    def __init__(self, n, weights=None, metric=None, covariance=None, labels=None, field="R"):
        super().__init__(n, weights, metric, covariance, labels)

    def whitened(self, C, name="L⁻¹"):
        """The whitened coordinates of a noise covariance C on these vectors (plan 068 P5, the author's decision Q1): the
        isometry d -> L^-1 d (C = L L^T) from the same vectors with the C^-1 inner product onto the plain R^n of the
        whitened coordinates (a DataSpace, labelled like this space when C is diagonal). C is what covariance= takes (a
        number, a positive vector, a symmetric positive definite array or operator). This space's own metric is not the
        noise and plays no part. An operators.whitening.Whitening: W(d), W.inv(z) = L z = W.adjoint(z)."""
        from ..operators.whitening import Whitening
        return Whitening(self, C, name=name)
