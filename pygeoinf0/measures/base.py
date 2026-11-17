"""The Measure root (plan 067 P5; architecture review 2026-11-13, layer L2 and B2). A measure carries its set and its
sigma-algebra, so it is the whole measure space (X, Sigma, mu):

    mu.set, mu.sigma_algebra       the set it lives on and the sets it measures (a sentence: "Borel subsets", ...)
    mu.claims                      what is known, with the three properties of a measure under the keys
                                   "finite" (mu(S) < inf), "probability" (non-negative with mu(S) = 1) and "signed"
                                   (takes negative values on some set), each a Truth: mu.claims.status("signed")
    mu.mass()                      (mu(S), its estimated error): exact (error 0.0) or by quadrature at two orders
    mu.integrate(f)                (int f dmu, its estimated error), for the integrands the measure states (a function
                                   of points for the measures of a domain, a bounded linear functional for a Gaussian)
    mu.restricted(D, side)         mu|_D (measures/restrict.py), or None when mu does not charge D
    mu(A)                          the measure of a set A, where it is evaluated (never required: a Gaussian measure
                                   has mass and integrals but evaluates no sets)

c * mu, mu + nu and mu - nu of measures of a domain are SignedMeasure combinations (whether one is signed is a claim, not
the class). Probability measures on spaces (pygeoinf0.probability: GaussianMeasure, Dirac, DiscreteMeasure) share this
root, not the classes of the base measures of a domain (their roles differ: a base measure defines L2 and the
functionals, a probability measure is a state of knowledge).
"""
import math

import numpy as np

from ..core.claims import CHECKED, DERIVED, THEOREM, UNKNOWN, Claim, Claims, weakest
from ..core.errors import Unsupported
from ..core.logic import Truth
from ..sets import Interval

ROUNDING = 16 * np.finfo(float).eps      # relative rounding allowed when a computed mass is compared with 1

STATEMENTS = {   # (when it holds, when it fails): the claim's words say which, its holds says it to the cells
    "finite": ("finite: mu(S) < inf", "not finite: mu(S) = inf"),
    "probability": ("a probability measure: non-negative with mu(S) = 1", "not a probability measure"),
    "signed": ("signed: mu takes negative values on some set", "not signed: mu is non-negative")}


def _as_interval(I):
    return Interval(I) if isinstance(I, str) else I


class Measure:
    """The root of every measure: its set, sigma-algebra, claims, mass, integrals and restriction (module docstring).

    Subclasses state their set, sigma-algebra, mass and integrals; the three properties default to what the mass and
    the sign say."""

    set = None
    sigma_algebra = None
    __array_ufunc__ = None              # numpy scalars defer to the products below (np.float64(2) * mu)

    def __call__(self, A):
        raise Unsupported(f"{self!r}: the measure of a set is not evaluated here (mass() and integrate(f) are)")

    def mass(self):
        """(mu(S), estimated error) for the set S carrying mu."""
        raise Unsupported(f"the mass of {self!r} is not computed")

    def integrate(self, f):
        """(int f dmu, estimated error)."""
        raise Unsupported(f"integration against {self!r} is not implemented")

    # -------------------------------------------------------------------------------- the three properties
    def _sign(self):
        """(holds, status, reason) of "signed"; a measure is non-negative unless a subclass says otherwise."""
        return Truth.FALSE, THEOREM, "a non-negative measure"

    def _properties(self):
        """{key: (holds, status, reason)} for finite, probability and signed, from the mass and the sign."""
        m, err = self.mass()
        how = DERIVED if err == 0 else CHECKED
        why = f"its mass {m:.6g}" + ("" if err == 0 else f" (estimated error {err:.1g}, by quadrature)")
        sig = self._sign()
        fin = (Truth.of(math.isfinite(m)), how, why)
        if sig[0] is Truth.TRUE:
            prob = (Truth.FALSE, sig[1], "it takes negative values")
        elif abs(m - 1.0) <= err + ROUNDING * max(1.0, abs(m)):
            prob = (Truth.TRUE, weakest(CHECKED, sig[1]), f"{why} is 1 to its error and rounding, and {sig[2]}") \
                if sig[0] is Truth.FALSE else (Truth.UNKNOWN, UNKNOWN, f"{why} is 1; its sign is not decided")
        else:
            prob = (Truth.FALSE, how, why + " is not 1")
        return {"finite": fin, "probability": prob, "signed": sig}

    def property_claims(self):
        """The claims finite, probability and signed (keys of the same names; an undecided one is left out, so its
        status reads UNKNOWN)."""
        out = []
        for key, (holds, status, reason) in self._properties().items():
            if holds is not Truth.UNKNOWN:
                yes = holds is Truth.TRUE
                out.append(Claim(STATEMENTS[key][0 if yes else 1], status, reason, key=key, holds=yes))
        return out

    def _own_claims(self):
        return []

    @property
    def claims(self):
        return Claims(self.property_claims() + self._own_claims() + _restriction_claims(self))

    def restricted(self, D, side=None):
        """mu|_D: f dmu -> f|_D dmu|_D, the measure A -> mu(A cap D), or None when mu does not charge D (plan 035 P1).
        Volume and density measures restrict to the exact set S cap D (domains/pieces.py), paths are cut at the
        spheres of a ball or shell, point and sphere masses on the boundary of D need side= ("below"/"above"), signed
        measures restrict termwise (measures/restrict.py)."""
        from .restrict import restrict
        return restrict(self, D, side)

    # linear combinations: c * mu, mu + nu, mu - nu are signed measures (functionals u -> sum c_t int u dmu_t)
    def __mul__(self, c):
        if np.ndim(c) == 0 and isinstance(c, (int, float, np.number)):
            return SignedMeasure([(float(c), self)])
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return SignedMeasure([(-1.0, self)])

    def __add__(self, other):
        if isinstance(other, Measure):
            return SignedMeasure([(1.0, self), (1.0, other)])
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self                                  # so that sum([mu, nu]) works
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Measure):
            return SignedMeasure([(1.0, self), (-1.0, other)])
        return NotImplemented


class SignedMeasure(Measure):
    """c_1 mu_1 + ... + c_m mu_m for finite measures mu_t and real numbers c_t: a finite signed measure (a measure when
    every c_t >= 0; claims key "signed"). Integration against it is the linear functional u -> sum_t c_t int u dmu_t,
    so its Riesz representer is the same combination of theirs; it is bounded on a space iff every term is (the trace
    theorem for the term of lowest dimension). Built by arithmetic: -2.0 * mu, mu + nu, c1 * DiracMeasure(x1) +
    c2 * DiracMeasure(x2). Nested combinations are flattened; terms keep their order (one measure may appear more than
    once)."""

    sigma_algebra = "Borel subsets"

    def __init__(self, terms):
        from .convert import as_measure
        flat = []
        for c, mu in terms:
            if isinstance(mu, SignedMeasure):
                flat += [(float(c) * a, m) for a, m in mu.terms]
            else:
                flat.append((float(c), as_measure(mu)))
        if not flat or not all(np.isfinite(c) for c, _ in flat):
            raise ValueError("a signed measure needs at least one term, with finite coefficients")
        self.terms = tuple(flat)

    @property
    def set(self):
        """The set of the first term (the terms of a combination live on one domain)."""
        return self.terms[0][1].set

    def __call__(self, A):
        return float(sum(c * mu(A) for c, mu in self.terms))

    def mass(self):
        """(sum_t c_t mu_t(S_t), sum_t |c_t| err_t)."""
        parts = [(c, mu.mass()) for c, mu in self.terms]
        return float(sum(c * m for c, (m, _) in parts)), float(sum(abs(c) * e for c, (_, e) in parts))

    def integrate(self, f):
        parts = [(c, mu.integrate(f)) for c, mu in self.terms]
        return float(sum(c * v for c, (v, _) in parts)), float(sum(abs(c) * e for c, (_, e) in parts))

    def __mul__(self, c):
        if np.ndim(c) == 0 and isinstance(c, (int, float, np.number)):
            return SignedMeasure([(float(c) * a, m) for a, m in self.terms])
        return NotImplemented

    __rmul__ = __mul__

    def _sign(self):
        """Not signed when every coefficient and term is non-negative; signed when it is one negative multiple of a
        non-negative measure of positive mass; otherwise undecided (terms of both signs may cancel)."""
        signs = [mu._sign() for _, mu in self.terms]
        if all(c >= 0 for c, _ in self.terms) and all(s[0] is Truth.FALSE for s in signs):
            return Truth.FALSE, weakest(*(s[1] for s in signs)), "a combination of non-negative measures with " \
                "non-negative coefficients"
        if len(self.terms) == 1 and self.terms[0][0] < 0 and signs[0][0] is Truth.FALSE:
            m, err = self.terms[0][1].mass()
            if m > err:
                return Truth.TRUE, weakest(DERIVED, signs[0][1]), "a negative multiple of a non-negative measure of " \
                    "positive mass"
        return Truth.UNKNOWN, UNKNOWN, "terms of both signs: they may cancel"

    def _own_claims(self):
        pos = all(c >= 0 for c, _ in self.terms)
        return [Claim("a measure" if pos else "a finite signed measure", THEOREM,
                      "a combination of finite measures" + (" with non-negative coefficients" if pos else ""))]

    def __repr__(self):
        parts = [f"{c:+.4g}·({mu})" for c, mu in self.terms[:3]]
        return " ".join(parts) + (f" … ({len(self.terms)} terms)" if len(self.terms) > 3 else "")


def terms_of(mu):
    """((c_t, mu_t), ...): the terms of a signed measure, or ((1, mu),) for a measure."""
    return mu.terms if isinstance(mu, SignedMeasure) else ((1.0, mu),)


def _restriction_claims(mu):
    r = getattr(mu, "_restriction", None)
    return [] if r is None else [r]
