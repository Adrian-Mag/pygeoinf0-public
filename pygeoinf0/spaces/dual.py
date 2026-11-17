"""The dual space X' of a normed space X and its elements, the bounded linear functionals (plan 068: P1 named X', P2
built its one element type here, absorbing operators/functionals.LinearForm and MeasureForm, port/derivatives.Covector
and spaces/boundedness.py).

    X.dual()                    X', the bounded linear functionals on X with the operator norm ||l|| = sup |l(x)| / ||x||
    X'.primal                   X
    X'.member(l)                TRUE for a Functional on X, FALSE otherwise
    X'.zero, X'.linear_combination(c, ls), a * l + b * m     the vector space X' (its scalars X's field)
    X'.bounded(mu)              the decision for u -> int u dmu on X (each space's rule, below)
    X'.integrals_bounded(k)     integrals over k-dimensional sets (bounded weights; k = 0: points) are in X': a Truth
    X'.claims                   complete, and what each space knows of its dual (Riesz for a Hilbert space, L^p' for
                                L^p with p < inf, the finite signed measures for C(K): Riesz-Markov)

    Functional                  l in X' (l.space = X): l(x), l.norm(), l.gradient() (its Riesz representer, on a Hilbert
                                space only), l.claims; built from
      Functional.from_representer(r)        x -> <x, r> on a Hilbert space, norm ||r|| (Cauchy-Schwarz)
      Functional.from_measure(X, mu)        u -> int u dmu where X's rule says it is bounded (refused otherwise)
      Functional.from_operator(A)           x -> (A x)_1 for a linear map A into a one-dimensional coordinate space
                                            (a derivative DJ of a scalar map: a covector needs no inner product);
                                            linear and bounded as A is (A.claim_status), nothing stated otherwise
      Functional(X, f)                      a callable f: linear by the caller's word (ASSERTED, plan 071 P2: a
                                            callable's linearity is not checked), bounded only if a claim says so
    gradient(J, theta)          the Frechet gradient of a scalar map J at theta (or of a Functional): the Riesz
                                representer of its derivative in the inner product of its space; it changes with the
                                inner product (L2, H^1, a weighted space), the derivative does not

Every space builds its functionals through these (X.functional, X.evaluation, X.average, pg.Functionals(...)[i],
pg.Integrals(...)[i], Coordinates.functional(g)); the bidual X'' is not built.

WHICH MEASURES DEFINE BOUNDED FUNCTIONALS (the one decision, EXP-031 DESIGN 3.3; formerly spaces/boundedness.py):

    bounded(mu, space) -> Decision(bounded, reason, needs, claims)    bounded a Truth (UNKNOWN: no rule decides it)
    require_bounded(mu, space)      the same decision, raising when it is negative (FALSE; UNKNOWN is not refused)
    l2_density(nu, mu, kernels, taper=None)    f -> int k_i f dnu on L2(mu) (pg.Functionals(M, kernels, against=nu)):
                                    (dnu/dmu, claims), raising when a representer k_i dnu/dmu is not in L2(mu)
    trace_excess(s, d, k)           s - (d - k)/2: THE TRACE RULE, once (integrals over k-dimensional sets of a
                                    d-dimensional domain are bounded on H^s iff it is > 0)

A space is recognised by its class (Sobolev, L2 or L^p, C), never by its attributes. For mu = w sigma_S (S of dimension
k in a domain of dimension d) and the term of a signed measure one by one:

    space                    bounded iff (sufficient conditions are said to be sufficient)
    H^s_l (d-dim domain)     beta = trace_excess(s, d, k) > 0 (trace theorem; Proposition 2, 3(d)) and w in the class
                             L^p(sigma) of Proposition 3(c): bounded weights always; |w| ~ dist^(-a) at points of S
                             with a < k - max(0, d/2 - s) (p = 2k/(2k - d + 2s) for s < d/2, p > 1 at s = d/2, L^1
                             for s > d/2); points (k = 0) iff s > d/2
    L^p(rho dx)              k = d and d mu / d lambda = w / rho in L^p'(lambda) (Riesz; DESIGN 2.3)
    C(K)                     every finite signed measure (Riesz-Markov): w in L^1(sigma), a < k

THE LOCAL RULE (_point_rule; the one rule behind the L^p rows and l2_density). Near a point x0 of a k-dimensional set
where d mu / d lambda ~ dist^o and the space's density rho ~ dist^s,

    int |d mu / d lambda|^p' d lambda ~ int dist^(p' o + s) dsigma < inf   iff   margin = p' o + s + k > 0

(p' = inf: o >= 0). On intervals the orders o and s are exact (measures/densities.py: polynomial densities, tables,
declared singular_at=/exponent=); elsewhere they are the declared exponents of the weights (|w| ~ dist^(-a): o = -a - s).
With rho = r^2 on (0, R) (s = 2), mu = k(r) dr and k ~ r^p (o = p - 2), L2 (p' = 2, k = 1): 2p - 1 > 0, plan 009's G13
condition p > 1/2. l2_density applies the rule to each representer k_i dnu/dmu (o = p_i + p_taper + order of
dnu/dmu), with the kernels' orders exact (tables) or estimated from samples (then a margin of 0.1 is required).
"""
import collections

import numpy as np

from ..core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN, Claim, Claims, by_rule
from ..core.errors import NormUnavailable, Refused, SpaceMismatch, Unsupported
from ..core.logic import Truth
from .base import BanachSpace, CoordinateSpace, HilbertSpace, NormedSpace
from .elements import Family, Function, VectorFamily
from .identity import mismatch, same_realisation


class DualSpace(BanachSpace):
    """X' for a normed space X (module docstring)."""

    def __init__(self, primal):
        if not isinstance(primal, NormedSpace):
            raise TypeError(f"a dual space is the dual of a normed space, not of {primal!r}")
        self.primal = primal

    @property
    def field(self):
        return self.primal.field

    def member(self, x):
        return Truth.of(isinstance(x, Functional) and x.space == self.primal)

    def norm(self, l):
        self._check(l)
        return float(l.norm())

    def _check(self, l):
        if self.member(l) is not Truth.TRUE:
            raise SpaceMismatch(f"{getattr(l, 'name', l)!s} is not a functional on {self.primal} (an element of {self} "
                                "is a pg.Functional with that space)")

    @property
    def zero(self):
        return Functional(self.primal, lambda x: 0.0, name="0", norm=0.0, route="zero",
                          claims=[Claim("bounded, norm 0", THEOREM, "the zero functional")])

    def linear_combination(self, coefficients, elements):
        """sum_i c_i l_i (c_i in the field of X): one Functional. On a Hilbert space with representers it is the
        functional of sum_i conj(c_i) r_i (exact norm); measures on C or L^p combine into one signed measure; otherwise
        it acts by linearity, and its norm is refused (NormUnavailable: the triangle inequality only bounds it)."""
        cs, ls = list(coefficients), list(elements)
        if len(cs) != len(ls):
            raise ValueError(f"{len(cs)} coefficients for {len(ls)} functionals")
        for l in ls:
            self._check(l)
        if not ls:
            return self.zero
        X = self.primal
        if all(l.representer is not None for l in ls):
            return Functional.from_representer(X.linear_combination([np.conj(c) for c in cs],
                                                                    [l.representer for l in ls]))
        if all(l.measure is not None and l.route == "measure" for l in ls) and all(np.isreal(c) for c in cs) \
                and _kind(X) in ("continuous", "lebesgue"):
            from ..measures.base import SignedMeasure
            mu = SignedMeasure([(float(np.real(c)) * l.scale, l.measure) for c, l in zip(cs, ls)])
            return Functional.from_measure(X, mu)
        bound = None
        try:
            bound = sum(abs(c) * l.norm() for c, l in zip(cs, ls))
        except Unsupported:
            pass
        name = " + ".join(f"({c})·{l.name}" for c, l in zip(cs, ls))
        held = [l.claims.status("bounded") for l in ls]
        cl = [Claim("bounded", by_rule(*[st for _, st in held]), "a finite combination of bounded functionals",
                    key="bounded")] if all(h is Truth.TRUE and st != UNKNOWN for h, st in held) else []
        return Functional(X, lambda x, cs=cs, ls=ls: sum(c * l(x) for c, l in zip(cs, ls)), name=name,
                          route="combination", terms=list(zip(cs, ls)), bound=bound, claims=cl)

    def bounded(self, mu):
        """The decision for u -> int u dmu on X (bounded(mu, X))."""
        return bounded(mu, self.primal)

    def integrals_bounded(self, k):
        """Whether u -> int_S u w dsigma over k-dimensional sets S with bounded weights w is in X' (k = 0: point values):
        on H^s iff s > (d - k)/2 (the trace rule), on L2 and L^p iff k = d, on C always; UNKNOWN on other spaces."""
        return integrals_bounded(self.primal, k)

    def functional(self, *args, **kwargs):
        raise Unsupported(f"functionals on {self} (the bidual) are not built here")

    @property
    def claims(self):
        return Claims([Claim("Banach space (complete)", THEOREM,
                             "the bounded linear functionals into a complete field are complete in the operator norm")]
                      + list(self.primal._dual_claims()))

    def __eq__(self, other):
        return isinstance(other, DualSpace) and other.primal == self.primal

    def __hash__(self):
        return hash(("dual", hash(self.primal)))

    def __repr__(self):
        return f"({self.primal!r})'"


# ------------------------------------------------------------------------------------------ the functionals
class Functional:
    """A bounded linear functional l on a normed space X: an element of X.dual() (module docstring).

    l.space is X; l.representer its Riesz representer where it was given by one (else None); l.measure, l.scale for
    u -> scale * int u dmu (else None, 1); l.operator the map A into a one-dimensional space (else None)."""

    def __init__(self, space, apply, *, name="ℓ", norm=None, route="given", representer=None, measure=None, scale=1.0,
                 operator=None, terms=None, bound=None, claims=()):
        self.space, self.name, self._apply, self._norm = space, name, apply, norm
        self.route, self.representer, self.measure, self.scale = route, representer, measure, scale
        self.operator, self._terms, self._bound, self._claims = operator, terms, bound, list(claims)
        self._scaled = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_representer(cls, r, name=None):
        """x -> <x, r> on the Hilbert space of r (complex spaces: linear in x), with norm ||r||."""
        X = getattr(r, "space", None)
        if not isinstance(X, HilbertSpace):
            raise TypeError(f"a functional given by a representer needs a Hilbert space; {X} has no inner product "
                            "(Functional.from_measure or from_operator describe functionals there)")
        if isinstance(r, (Family, VectorFamily)):
            raise ValueError("one functional per representer: pg.Functionals(family) is the operator into R^n")
        if X.field == "C" and hasattr(X, "cinner"):
            def apply(x):
                return X.cinner(x, r)
        else:
            def apply(x):
                return X.inner(r, x)
        return cls(X, apply, name=name or f"<·, {getattr(r, 'name', 'r')}>", route="representer", representer=r,
                   claims=[Claim("bounded", THEOREM, "|<x, r>| <= ||r|| ||x|| (Cauchy-Schwarz), with equality at x = r")])

    @classmethod
    def from_measure(cls, space, mu, scale=1.0, name=None):
        """u -> scale * int u dmu (mu a measure, a set: its natural measure, or a point: its Dirac measure), where the
        space's rule says it is bounded: on a Hilbert function space through its Riesz representer (refused where
        mu has none: UnboundedFunctional), on C(K) for every finite measure (Riesz-Markov), on L^p(lambda) iff
        d mu / d lambda is in L^p'."""
        from ..measures import as_measures
        X = space
        kind = _kind(X)
        if kind is None and isinstance(X, HilbertSpace) and not isinstance(X, CoordinateSpace):
            out = cls.from_representer(X.riesz(mu) if scale == 1.0 else X.linear_combination([scale], [X.riesz(mu)]),
                                       name=name or f"∫ · d({mu})")      # a direct sum, pieces: the space's own riesz
            out.measure, out.scale, out.route = mu, float(scale), "measure"
            return out
        if kind is None:
            raise Unsupported(f"functionals given by measures on {X} are not built here"
                              + (": its functionals are X.functional(coefficients)" if isinstance(X, CoordinateSpace)
                                 else ""))
        items, many = as_measures(mu, 1 if kind == "continuous" else X.domain.cartesian_dim)
        if many:
            raise ValueError("one measure per functional; X.riesz([...]) or X.integrals([...]) give several")
        m, scale = items[0], float(scale)
        nm = name or f"∫ · d({m})"
        if kind in ("sobolev", "l2"):
            r = X.riesz(m)                                       # the space's rule, and its representer
            if scale == 1.0:
                out = cls.from_representer(r, name=nm)
            else:
                out = cls(X, lambda u: scale * X.inner(r, u), name=nm, norm=lambda: abs(scale) * X.norm(r),
                          claims=[Claim("bounded", THEOREM, "a measure with a Riesz representer in the space")])
                out._scaled = r
            out.measure, out.scale, out.route = m, scale, "measure"
            return out
        if kind == "continuous":
            return cls(X, lambda f: _first(scale * X.integrate_against(f, [m]), f), name=nm, route="measure",
                       measure=m, scale=scale, norm=lambda: abs(scale) * _total_variation(m, X),
                       claims=[Claim("bounded", THEOREM, "|int f dmu| <= |mu|(K) max |f| (Riesz-Markov: the dual of "
                                                         "C(K) is the finite signed measures)")])
        from .representers import _check_support, moments
        d = require_bounded(m, X)                                 # the L^p rule: d mu / d lambda in L^p'
        _check_support(X, m)
        return cls(X, lambda u: _first(scale * moments(X, [m], u)[0], u), name=nm, route="measure", measure=m,
                   scale=scale, norm=lambda: abs(scale) * _lp_dual_norm(m, X),
                   claims=[Claim("bounded", DERIVED, f"d mu / d lambda in L^{_conjugate_text(X)} (Hölder)")]
                   + list(d.claims))

    @classmethod
    def from_operator(cls, A, name=None, claims=()):
        """x -> (A x)_1 for a linear map A from X into a one-dimensional coordinate space (a derivative DJ of a scalar
        map J; its codomain's metric changes the adjoint, not the number). Bounded as A is."""
        D = getattr(A, "codomain", None)
        if not (isinstance(D, CoordinateSpace) and D.dim == 1) or not callable(A):
            raise SpaceMismatch(f"a functional is given by a linear map into a one-dimensional coordinate space; "
                                f"{getattr(A, 'name', A)} maps into {D}")

        def apply(x):
            return np.asarray(A(x).values).ravel()[0].item()
        cl = list(claims)
        if Claims(cl).find("bounded") is None:                   # bounded as A is: A's claims and the package's rules
            held, st = _map_status(A, "bounded")
            if held is Truth.TRUE and st != UNKNOWN:
                cl.append(Claim("bounded", st, f"{getattr(A, 'name', 'A')} is bounded ({st})", key="bounded"))
        return cls(A.domain, apply, name=name or getattr(A, "name", "ℓ"), route="operator", operator=A, claims=cl)

    # ------------------------------------------------------------------ the functional
    def __call__(self, x):
        S = getattr(x, "space", None)
        if S is not None and S is not self.space and S == self.space and not same_realisation(S, self.space):
            raise mismatch(S, self.space, f"{self.name}({getattr(x, 'name', 'x')})")      # decision Q4
        return self._apply(x)

    def norm(self):
        """||l|| = sup |l(x)| / ||x||: exact where the space's rule gives it (a representer's norm, the total variation
        of a measure on C, ||d mu / d lambda||_p' on L^p, the representer of a derivative); otherwise NormUnavailable."""
        if self._norm is not None:
            return float(self._norm() if callable(self._norm) else self._norm)
        if self.representer is not None:
            return float(self.space.norm(self.representer))
        if isinstance(self.space, HilbertSpace) and self.route in ("operator", "combination"):
            return float(self.space.norm(self.gradient()))
        bound = "" if self._bound is None else f" (the triangle inequality bounds it by {self._bound:.6g})"
        raise NormUnavailable(f"the norm of {self.name} on {self.space} is not computed here{bound}")

    def gradient(self):
        """The Riesz representer: the element r with <r, h> = l(h) for every h, in the inner product of the space."""
        X = self.space
        if not isinstance(X, HilbertSpace):
            raise Refused(f"a gradient is the Riesz representer of {self.name}, and {X} has no inner product: the "
                          "functional itself is the derivative", needs=("a Hilbert space",))
        if self.representer is not None:
            return self.representer
        if self.route == "zero":
            return X.zero
        if self._scaled is not None:                             # scale times a measure's representer
            return X.linear_combination([self.scale], [self._scaled])
        if self.route == "operator":
            D = self.operator.codomain
            r = self.operator.adjoint(D(D.solve_metric(np.array([1.0]))))
            r.name = f"∇{self.name}"
            return r
        if self.route == "combination":
            return X.linear_combination([np.conj(c) for c, _ in self._terms], [l.gradient() for _, l in self._terms])
        raise Unsupported(f"the representer of {self.name} is not built here")

    @property
    def dual(self):
        """The space this functional is an element of: X.dual()."""
        return self.space.dual()

    @property
    def claims(self):
        """The claims of the functional, with its linearity as its construction gives it (plan 071 P2, review item 1):
        THEOREM for a representer, a measure and the zero functional; for x -> (A x)_1 as A is linear (A's claims and
        the package's rules); for a combination the weakest of its terms; ASSERTED for Functional(X, f), the caller's
        word (a callable's linearity is not checked)."""
        own = Claims(self._claims)
        lin = None if own.find("linear") is not None else self._linearity()
        return Claims(self._claims + ([] if lin is None else [lin]))

    def _linearity(self):
        if self.route in ("representer", "measure", "zero"):
            why = {"representer": "x -> <x, r> is linear in x", "measure": "u -> int u dmu is linear in u",
                   "zero": "the zero functional"}[self.route]
            return Claim("linear", THEOREM, why, key="linear")
        if self.route == "operator":
            held, st = _map_status(self.operator, "linear")
            return Claim("linear", st, f"{getattr(self.operator, 'name', 'A')} is linear ({st})", key="linear") \
                if held is Truth.TRUE and st != UNKNOWN else None
        if self.route == "combination":
            held = [l.claims.status("linear") for _, l in self._terms]
            if all(h is Truth.TRUE and st != UNKNOWN for h, st in held):
                return Claim("linear", by_rule(*[st for _, st in held]), "a finite combination of linear functionals",
                             key="linear")
            return None
        return Claim("linear", ASSERTED, "the caller's word: Functional(X, f) takes f to be linear, and a callable's "
                     "linearity is not checked (X.functional, Functional.from_representer, from_measure and "
                     "from_operator state it from their construction)", key="linear")

    # ------------------------------------------------------------------ the vector space X'
    def __add__(self, other):
        if not isinstance(other, Functional):
            return NotImplemented
        return self.dual.linear_combination([1.0, 1.0], [self, other])

    def __sub__(self, other):
        if not isinstance(other, Functional):
            return NotImplemented
        return self.dual.linear_combination([1.0, -1.0], [self, other])

    def __neg__(self):
        return self.dual.linear_combination([-1.0], [self])

    def __mul__(self, c):
        if np.ndim(c) != 0 or isinstance(c, Functional):
            return NotImplemented
        return self.dual.linear_combination([c], [self])

    __rmul__ = __mul__

    def __repr__(self):
        return f"{self.name}: {self.space} → {self.space.field}"


def _map_status(A, key):
    """(holds, status) of the property key of a map A into R^1 (linear, bounded): A's own claim_status (an operator's
    claims and the package's rules, operators/claim_rules.py), else its claims on record, else unknown."""
    rule = getattr(A, "claim_status", None)
    if callable(rule):
        return rule(key)
    return Claims(getattr(A, "claims", ()) or ()).status(key)


def _first(v, u):
    """A number for an element, the row of values for a family."""
    v = np.asarray(v)
    return float(v.ravel()[0]) if isinstance(u, Function) else v


def gradient(J, theta=None):
    """The Frechet gradient at theta of a scalar map J with a derivative (J.derivative(theta)), or of a Functional,
    in the inner product of its space: the Riesz representer of the derivative (refused without an inner product)."""
    if isinstance(J, Functional):
        return J.gradient()
    if not hasattr(J, "derivative"):
        raise Refused(f"{getattr(J, 'name', J)} has no derivative", needs=("a derivative provider (pg.port.External)",))
    DJ = J.derivative(theta)
    return Functional.from_operator(DJ, name=f"D{J.name}", claims=list(getattr(DJ, "claims", ()))).gradient()


# ------------------------------------------------------------------------------------------ the rules
Decision = collections.namedtuple("Decision", "bounded reason needs claims")
Decision.__doc__ = ("bounded (a Truth: TRUE, FALSE, or UNKNOWN where no rule decides it; plan 067 P1, was a bool), "
                    "reason (str), needs (what would make it bounded), claims (the Claims behind it).")

SAMPLED_MARGIN = 0.1        # the margin required when a local order is estimated from samples (a check, not a proof)


def trace_excess(s, d, k):
    """beta = s - (d - k)/2: integration over k-dimensional sets (k = 0: points) of a d-dimensional domain is bounded on
    H^s iff beta > 0 (the trace theorem: Jonsson-Wallin 1984, Adams-Hedberg 1996). The one copy of the rule."""
    return float(s) - (int(d) - int(k)) / 2


def integrals_bounded(space, k):
    """X.dual().integrals_bounded(k) (DualSpace)."""
    kind = _kind(space)
    if kind == "sobolev":
        return Truth.of(trace_excess(space.s, space.domain.dim, k) > 0)
    if kind in ("l2", "lebesgue"):
        return Truth.of(int(k) == int(space.domain.dim))
    if kind == "continuous":
        return Truth.TRUE
    return Truth.UNKNOWN


def _point_rule(o, s, q=2.0, k=1, exact=True):
    """(ok, margin): d mu / d lambda ~ dist^o in L^q(lambda) near a point, lambda ~ dist^s, on a k-dimensional set."""
    if np.isinf(q):
        return o >= 0, o
    margin = q * o + s + k
    return margin > (0.0 if exact else SAMPLED_MARGIN), margin


def _refused(reason, needs):
    """The one refusal of the decision: pg.Refused with the reason and what would make the functional bounded."""
    return Refused(reason, needs)


def _kind(space):
    """'sobolev' (H^s), 'l2' (L2), 'lebesgue' (L^p, p != 2), 'continuous' (C) or None: by class, never by attributes."""
    from .base import L2, LpSpace
    from .continuous import ContinuousSpace
    from .sobolev import SobolevSpace
    if isinstance(space, SobolevSpace):
        return "sobolev"
    if isinstance(space, ContinuousSpace):
        return "continuous"
    if isinstance(space, L2):
        return "l2"
    if isinstance(space, LpSpace):
        return "lebesgue"
    return None


def _conjugate(space):
    p = float(getattr(space, "p", 2.0))
    return np.inf if p == 1 else 1.0 if np.isinf(p) else p / (p - 1)            # p'


def _conjugate_text(space):
    q = _conjugate(space)
    return "∞" if np.isinf(q) else f"{q:g}"


def _total_variation(mu, space):
    """|mu|(K), the norm of u -> int u dmu on C(K): exact for one term; a signed measure of several terms is refused
    (the sum of |c_t| mu_t(K) only bounds it where the terms overlap)."""
    from ..measures import terms_of, total_variation
    terms = [t for t in terms_of(mu) if t[0] != 0]
    if len(terms) > 1:
        raise NormUnavailable(f"the norm |mu|(K) of integration against {mu} on {space} is not computed here: its terms "
                              f"may overlap, and sum |c_t| mu_t(K) = {total_variation(mu):.6g} only bounds it")
    return total_variation(mu)


def _lp_dual_norm(mu, space):
    """||d mu / d lambda||_(L^p'(lambda)): exact for mu = c times Lebesgue measure on a set S with lambda Lebesgue
    measure (|c| |S|^(1/p')); otherwise NormUnavailable."""
    from ..measures import LebesgueMeasure, terms_of, total_variation
    terms = [t for t in terms_of(mu) if t[0] != 0]
    if not terms:
        return 0.0
    if len(terms) == 1 and type(terms[0][1]) is LebesgueMeasure and type(space.measure) is LebesgueMeasure:
        c, m = terms[0]
        size, q = float(total_variation(m)), _conjugate(space)
        return abs(c) * (1.0 if np.isinf(q) else size ** (1.0 / q))
    raise NormUnavailable(f"||d mu / d lambda||_(L^{_conjugate_text(space)}) of {mu} on {space} is not computed here "
                          "(it is finite: the functional is bounded by the rule of L^p)")


def _declared(term):
    """(points (m, d) or None, exponent a) of a term's weight (None, 0) for undeclared or unweighted terms."""
    wt = getattr(term, "weight", None)
    if wt is None or getattr(wt, "singular_at", None) is None:
        return None, 0.0
    return wt.singular_at, wt.exponent


def bounded(mu, space):
    """The decision for u -> int u dmu on space (see the module docstring)."""
    from ..measures import measure_dim, terms_of
    kind = _kind(space)
    claims, undecided = [], None
    for c, term in terms_of(mu):
        if c == 0:
            continue
        d = _ok(term, space, kind, measure_dim(term), claims)
        if d.bounded is Truth.FALSE:
            return d
        if d.bounded is Truth.UNKNOWN:
            undecided = undecided or d
    if undecided is not None:                           # no term is unbounded, and one is not decided
        return undecided
    return Decision(Truth.TRUE, "every term is bounded", (), tuple(claims))


def _ok(term, space, kind, k, claims):
    if kind is None:
        return Decision(Truth.UNKNOWN, "no rule for this space: not decided here", (), ())
    d = int(space.domain.dim)
    pts, a = _declared(term)
    if kind == "sobolev":
        s = float(space.s)
        beta = trace_excess(s, d, k)
        if beta <= 0:
            return Decision(Truth.FALSE, f"it needs s > (d - k)/2 = {(d - k) / 2:g} (trace theorem; the energy int int "
                                         "K dmu dmu is infinite)", (f"s > {(d - k) / 2:g}",), ())
        claims.append(Claim(f"trace excess beta = s - (d - k)/2 = {beta:g} > 0", THEOREM,
                            "Proposition 2-3 (DESIGN 2.2): bounded weights in L2(sigma) give bounded functionals",
                            key="bounded"))
        if pts is not None and a > 0:
            amax = k - max(0.0, d / 2 - s)
            cls = "L^1" if s > d / 2 else "L^p, p > 1" if s == d / 2 else f"L^p, p = {2 * k / (2 * k - d + 2 * s):.4g}"
            if not a < amax:
                return Decision(Truth.FALSE, f"the declared singularity |w| ~ dist^(-{a:g}) is outside the class {cls} "
                                       f"of Proposition 3(c) (a < {amax:g}); this condition is sufficient, not "
                                       "necessary, and no other is decided here", (f"exponent a < {amax:g}",), ())
            claims.append(Claim(f"weight in {cls}", ASSERTED, f"declared exponent {a:g} < {amax:g} (Proposition 3(c))",
                                key="weight class"))
        return Decision(Truth.TRUE, "", (), ())
    if kind in ("l2", "lebesgue"):
        if k < d:
            return Decision(Truth.FALSE, "the set has measure zero, and an element of L2 is a class of functions equal "
                                   f"almost everywhere. It is bounded on the Sobolev space pg.Sobolev({space.domain!r},"
                                   f" s, length_scale=l) for s > (d - k)/2 = {(d - k) / 2:g} (trace theorem).",
                            ("a Sobolev space", f"s > {(d - k) / 2:g}"), ())
        return _lebesgue(term, space, k, pts, a, claims)
    if pts is not None and not a < k:                   # C(K)
        return Decision(Truth.FALSE, f"|w| ~ dist^(-{a:g}) is not integrable on a {k}-dimensional set (a < {k} needed)",
                        (f"exponent a < {k}",), ())
    claims.append(Claim("a finite signed measure", THEOREM, "Riesz-Markov: the dual of C(K)", key="bounded"))
    return Decision(Truth.TRUE, "", (), ())


def _lebesgue(term, space, k, pts, a, claims):
    """L^p(lambda), lambda = rho dx: d mu / d lambda = w / rho in L^p'(lambda) near every point where it may blow up."""
    from ..sets import Interval
    q = _conjugate(space)
    lam = space.measure
    if isinstance(term.set, Interval) and isinstance(lam.set, Interval):
        return _interval_rule(term, lam, q, claims)
    rho_pts, rho_a = _declared(lam)                     # rho ~ dist^(-rho_a): s = -rho_a
    points = [] if pts is None else [tuple(x) for x in pts]
    rpoints = [] if rho_pts is None else [tuple(x) for x in rho_pts]
    for x in sorted(set(points) | set(rpoints)):
        ax = a if x in points else 0.0
        s = -rho_a if x in rpoints else 0.0
        ok, margin = _point_rule(-ax - s, s, q, k)
        if not ok:
            need = f"p' o + s + k > 0 with p' = {q:g}, o = {-ax - s:g}, s = {s:g}, k = {k}"
            return Decision(Truth.FALSE, f"d mu / d lambda = w / rho is not in L^{q:g}(lambda) at {x}: it needs {need} "
                                   "(Riesz representation; DESIGN 2.3)", (need,), ())
        claims.append(Claim(f"w / rho in L^{q:g} near {x}", ASSERTED, f"declared exponents, margin {margin:g} > 0",
                            key="bounded"))
    wt = getattr(lam, "weight", None)
    if wt is not None and not rpoints:
        from ..measures.weights import sample_points
        X = sample_points(term.set)
        if X is not None:
            m = float(np.min(wt.values(X)))
            if not m > 0:
                return Decision(Truth.FALSE, f"the density of {space} vanishes on {term.set} (sampled minimum {m:.3g}) "
                                       "and its order there is not declared",
                                ("declare it: singular_at=points, exponent=-order on the space's density",), ())
            claims.append(Claim("the space's density is bounded below on the set", CHECKED, f"sampled min {m:.3g}"))
    claims.append(Claim("d mu / d lambda exists", DERIVED, "mu has a density on a set of full dimension"))
    return Decision(Truth.TRUE, "", (), ())


def _interval_rule(term, lam, q, claims):
    """The local rule at the points of an interval where d mu / d lambda may blow up, with exact orders."""
    from ..measures.densities import declared_points, space_order
    rho = term.density_against(lam)
    if not rho.absolutely_continuous:                   # a support leaving the domain: SupportOutsideDomain, later
        return Decision(Truth.UNKNOWN, "the support leaves the space's domain: not decided here (SupportOutsideDomain "
                                       "where the functional is built)", (), ())
    asserted = set(declared_points(term)) | set(declared_points(lam))
    for x0 in sorted(x for x, o in rho.orders.items() if o < 0):
        o, s = rho.order(x0), space_order(lam, x0, rho)
        ok, margin = _point_rule(o, s, q, 1)
        if not ok:
            van = f"; the density of the space vanishes there (order {s:g})" if s > 0 else ""
            need = f"d mu / d lambda vanishing to an order o with {q:g} o + {s:g} + 1 > 0 at {x0:g}"
            return Decision(Truth.FALSE, f"d mu / d lambda is not in L^{q:g}(lambda) at {x0:g}: it behaves like "
                                   f"|x - {x0:g}|^{o:g}{van}, so int |d mu / d lambda|^{q:g} d lambda diverges (Riesz "
                                   "representation; DESIGN 2.3)", (need,), ())
        claims.append(Claim(f"d mu / d lambda in L^{q:g} near {x0:g} (margin {margin:.3g} > 0)",
                            ASSERTED if x0 in asserted else DERIVED,
                            "declared exponents" if x0 in asserted else "exact local orders of the densities",
                            key="bounded"))
    claims.append(Claim("d mu / d lambda exists", DERIVED, "mu has a density on a set of full dimension"))
    return Decision(Truth.TRUE, "", (), ())


def require_bounded(mu, space):
    """bounded(mu, space), raising when negative: UnboundedFunctional (PointEvaluationUndefined for points) for the
    dimension rules, with the wording of errors.unbounded, and Refused for a weight outside its class. Both are
    pg.Refused, with .reason and .needs."""
    from ..core.errors import unbounded
    from ..measures import measure_dim, terms_of
    kind = _kind(space)
    for c, term in terms_of(mu):
        if c == 0:
            continue
        k = measure_dim(term)
        d = _ok(term, space, kind, k, [])
        if d.bounded is not Truth.FALSE:                # UNKNOWN (no rule decides it) is not refused here, as before
            continue
        if d.reason.startswith(("it needs s >", "the set has measure zero")):
            raise unbounded(mu, k, space, d.reason)
        raise _refused(f"u -> int u d({term}) is not bounded on {space} (decided): {d.reason}", d.needs)
    return bounded(mu, space)


def l2_density(nu, mu, kernels, taper=None, label="functional"):
    """The decision for f -> int k f dnu on L2(mu), k each of the kernels (plan 009 G13, formerly in core/densities.py):
    (dnu/dmu, claims), or pg.Refused when nu is not absolutely continuous with respect to mu or a representer
    k (dnu/dmu) (times the taper) is not square integrable near a point (the local rule, module docstring). The Density
    returned carries .representer {x0: (least order of k dnu/dmu, least margin, s)} for the realisation."""
    from ..measures.densities import kernel_order, space_order
    needs = ["a declared modification that vanishes fast enough at the singular point (taper=...)",
             "another reading of the kernels (against=..., e.g. a density against the space's own measure)"]
    if not hasattr(nu, "density_against"):
        raise _refused(f"{nu} offers no density against {mu} (a measure for against= has density_against(base))",
                       ["a measure with density_against(base)"])
    rho = nu.density_against(mu)
    reason, claims = None, []
    if not rho.absolutely_continuous:
        reason, needs = rho.reason, ["a measure absolutely continuous with respect to the space's measure"]
    span = (rho.support[1] - rho.support[0]) if rho.support else 1.0
    rho.representer = {}
    for x0 in sorted(x for x, o in rho.orders.items() if o < 0) if reason is None else ():
        inside = 1.0 if rho.support is None or x0 <= rho.support[0] + 1e-12 * abs(span) else -1.0
        s = space_order(mu, x0, rho, inside)
        pt, exact = (0.0, True) if taper is None else kernel_order(taper, x0, inside, span)
        worst, least, all_exact = np.inf, np.inf, exact
        for i, k in enumerate(kernels):
            p, e = kernel_order(k, x0, inside, span)
            all_exact = all_exact and e
            o = p + pt + rho.order(x0)                         # the order of k (dnu/dmu) (times the taper)
            ok, margin = _point_rule(o, s, 2.0, 1, exact and e)
            worst, least = min(worst, margin), min(least, o)
            if not ok:
                reason = (f"{label} {i}: the kernel behaves like |x - {x0:g}|^{p:g}"
                          + (f" (times a taper of order {pt:g})" if taper is not None else "")
                          + f" against a density of order {rho.order(x0) + s:g} on a space whose measure has order "
                          f"{s:g} there, so k dnu/dmu is not in L2(mu): int (k dnu/dmu)^2 dmu ~ int |x - x0|^"
                          f"{2 * o + s:g} dx diverges, and the functional is not bounded (Riesz; EXP-030 NOTES.md "
                          "section 2)")
                break
        if reason is not None:
            break
        rho.representer[float(x0)] = (least, worst, s)
        claims.append(Claim(f"bounded near {x0:g}: the representers k dnu/dmu are square integrable there (smallest "
                            f"margin 2p + 2q - s + 1 = {worst:.3g} > 0)", DERIVED if all_exact else CHECKED,
                            "local orders " + ("exact (tables and polynomial densities)" if all_exact else
                                               "estimated from samples near the point (a check, not a proof)")))
    if reason is not None:
        raise _refused(reason, needs)
    claims.append(Claim(f"representers k d({getattr(nu, 'name', 'nu')})/dmu of the {label}s (Radon-Nikodym)", DERIVED,
                        "int k f dnu = <k dnu/dmu, f>_(L2(mu)) (Riesz)"))
    return rho, claims
