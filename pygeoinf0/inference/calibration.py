"""Calibration: the guarantee stated for a reported result (el:535-567). It is recorded, not computed; check(problem)
returns what the library can say about its hypotheses (the rest stays unknown).

    Deterministic()                         holds for every model in the prior set and every noise in the noise set
    Frequentist(alpha, fixed_before_data)   coverage >= 1 - alpha over repeated data at a fixed model
    Bayesian(level)                         posterior mass under the stated prior and noise law
    Hardened(alpha, beta)                   a hardened prior set (mass >= 1 - beta) and confidence regions of level
                                            1 - alpha: coverage >= 1 - alpha - beta over prior and noise (plan 009 G7)
    Unstated()                              the author states none: recorded as such (el:539)

Problem.run (plan 070 P5, review D9) refuses a calibration its method does not declare (Method.calibrations,
inference/methods.py) and returns the result with problem.calibration_claims() added (carry): to its claims, on a
tuple of ends as a ClaimedTuple, on an element as a CalibratedVector or CalibratedFunction. The numbers are unchanged.
"""
from ..core.claims import ASSERTED, DERIVED, THEOREM, UNKNOWN, Claim, Claims
from ..core.kinds import MEASURE, SET, stated_kind
from ..core.logic import Truth
from ..spaces.elements import Function, Vector


class Calibration:
    """A stated guarantee (module docstring)."""

    kind = "unstated"

    def check(self, problem):
        return Claims([Claim(f"calibration: {self.kind}", ASSERTED, "stated by the author", key="calibration")])

    def __repr__(self):
        return f"{type(self).__name__}()"


class Unstated(Calibration):
    """No guarantee stated: the computation runs, its result is not calibrated (el:539)."""


class Deterministic(Calibration):
    """A bound for every model in the declared prior set and every error in the declared noise set (el:545)."""

    kind = "deterministic"

    def check(self, problem):
        out = list(super().check(problem))
        ok = stated_kind(problem.prior) is SET and getattr(problem.mdr, "kind", None) is SET
        out.append(Claim("a set prior and a set-valued MDR: the bound holds for every admissible case", DERIVED
                         if ok else UNKNOWN, "el:545" if ok else "needs a prior set and a set-valued relation"))
        return Claims(out)


class Frequentist(Calibration):
    """Coverage >= 1 - alpha over repeated data at each fixed admissible model (el:547-552): needs acceptance regions
    of probability 1 - alpha chosen before the data (inv:203-248)."""

    kind = "frequentist"

    def __init__(self, alpha, fixed_before_data=False):
        self.alpha, self.fixed_before_data = float(alpha), bool(fixed_before_data)

    def check(self, problem):
        out = list(super().check(problem))
        st = problem.mdr.claims.status("acceptance") if problem.mdr is not None else (Truth.UNKNOWN, UNKNOWN)
        out.append(Claim(f"acceptance regions of probability {1 - self.alpha:g}", st[1] if st[0] is Truth.TRUE else UNKNOWN,
                         "mdr.acceptance(alpha)" if st[0] is Truth.TRUE else "the noise set is not an acceptance region on record"))
        out.append(Claim("the region was chosen before the data", ASSERTED if self.fixed_before_data else UNKNOWN,
                         "stated" if self.fixed_before_data else "not stated (a region tuned on d has no coverage)"))
        return Claims(out)

    def __repr__(self):
        return f"Frequentist(alpha={self.alpha:g})"


class Bayesian(Calibration):
    """Posterior probability under the stated prior measure and noise law (el:554-558)."""

    kind = "Bayesian"

    def __init__(self, level=None):
        self.level = level

    def check(self, problem):
        out = list(super().check(problem))
        mu = problem.prior
        ok = stated_kind(mu) is MEASURE
        out.append(Claim("a prior measure on the model space", DERIVED if ok else UNKNOWN,
                         "the prior is a measure" if ok else "needs a prior measure (el:594)"))
        if ok and getattr(mu, "is_cylindrical", False):
            out.append(Claim("the prior is cylindrical: posterior probabilities of bounded functionals only", DERIVED,
                             "EXP-028 DESIGN 1.9"))
        return Claims(out)


class Hardened(Calibration):
    """Confidence-set inversion under a hardened Gaussian prior (the thesis's appendix H, inv:696-827; plan 009 G7).

    M ~ mu, S = mu.harden(form, 1 - beta) with mu(S) >= 1 - beta, and C_alpha(D) the confidence set whose acceptance
    regions A_alpha(m) have P(D in A_alpha(m) | M = m) >= 1 - alpha for every m. For K(D) = S & C_alpha(D):
        P(M in K(D)) >= 1 - alpha - beta                 the union bound: P(M not in S) + P(D not in A(M))
        P(M in S and D in A_alpha(M)) >= (1 - alpha)(1 - beta)
                                                         int_S P(D in A(m) | m) dmu(m): the prior-predictive event
    Both are theorems given their hypotheses; check(problem) reports which hypotheses are on record."""

    kind = "hardened"

    def __init__(self, alpha, beta):
        self.alpha, self.beta = float(alpha), float(beta)
        if not (0 < self.alpha < 1 and 0 < self.beta < 1):
            raise ValueError("0 < alpha, beta < 1")

    def coverage(self):
        """The two guaranteed lower bounds: union 1 - alpha - beta and joint (1 - alpha)(1 - beta)."""
        return dict(union=1 - self.alpha - self.beta, joint=(1 - self.alpha) * (1 - self.beta))

    def check(self, problem):
        out = list(super().check(problem))
        c = self.coverage()
        out.append(Claim(f"P(M in K(D)) >= 1 - alpha - beta = {c['union']:.6g}", THEOREM,
                         "the union bound over the prior event and the noise event (appendix H)", key="coverage"))
        out.append(Claim(f"P(M in S and D in A(M)) >= (1 - alpha)(1 - beta) = {c['joint']:.6g}", THEOREM,
                         "Fubini: the acceptance probability is >= 1 - alpha for every m in S (appendix H)",
                         key="joint coverage"))
        prior = getattr(problem, "prior", None)
        m = prior.claims.find("mass") if hasattr(prior, "claims") else None
        out.append(Claim(f"the prior set has mass >= 1 - beta = {1 - self.beta:g}", m.status if m is not None else UNKNOWN,
                         m.reason if m is not None else "needs S = mu.harden(form, 1 - beta) as the prior",
                         key="prior mass"))
        mdr = getattr(problem, "mdr", None)
        st = mdr.claims.status("acceptance") if mdr is not None else (Truth.UNKNOWN, UNKNOWN)
        out.append(Claim(f"acceptance regions of probability {1 - self.alpha:g}", st[1] if st[0] is Truth.TRUE else UNKNOWN,
                         "mdr.acceptance(alpha)" if st[0] is Truth.TRUE else "the noise set is not an acceptance region on record"))
        return Claims(out)

    def __repr__(self):
        return f"Hardened(alpha={self.alpha:g}, beta={self.beta:g})"


# ------------------------------------------------------------------ carrying the claims on a result (Problem.run)
class ClaimedTuple(tuple):
    """A tuple of results (oCSOLA's ends (lower, upper), or arrays of them) with .claims: equal to the plain tuple."""

    def __new__(cls, values, claims=()):
        out = tuple.__new__(cls, values)
        out.claims = Claims(claims)
        return out


def _calibrated(base):
    """The subclass of an element class (Vector, Function) that carries claims besides its own (a slot)."""

    class Calibrated(base):
        __slots__ = ("_carried",)
        same_values = True              # its values are the base's: arithmetic on it stays lazy (core/lazy.py)

        @property
        def claims(self):
            own = base.claims.__get__(self) if isinstance(getattr(base, "claims", None), property) else ()
            return Claims(list(own) + list(self._carried))

    Calibrated.__name__ = Calibrated.__qualname__ = f"Calibrated{base.__name__}"
    Calibrated.__doc__ = f"A {base.__name__} returned by Problem.run: the element, with the problem's calibration claims."
    return Calibrated


def _element_classes():
    return {Vector: CalibratedVector, Function: CalibratedFunction}


def _slots(cls):
    return [s for k in cls.__mro__ for s in getattr(k, "__slots__", ()) if s != "__weakref__"]


def _new(claims, have):
    seen = {(c.statement, c.status, c.reason, c.key) for c in have}
    return [c for c in claims if (c.statement, c.status, c.reason, c.key) not in seen]


def carry(result, claims):
    """The result with the claims added (module docstring): appended to a list of claims it keeps (states, sets),
    set as .claims on a record, a ClaimedTuple for a plain tuple, a Calibrated element for an element (its values the
    same objects). A result that cannot carry claims is refused (TypeError)."""
    claims = list(claims)
    if type(result) is tuple:
        return ClaimedTuple(result, claims)
    keep = getattr(result, "_claims", None)
    if isinstance(keep, list):
        keep.extend(_new(claims, keep))
        return result
    for base, sub in _element_classes().items():
        if type(result) is base:
            out = object.__new__(sub)
            for s in _slots(base):
                if hasattr(result, s):
                    v = getattr(result, s)
                    object.__setattr__(out, s, dict(v) if s == "_memo" else v)
            out._carried = tuple(claims)
            return out
    if hasattr(result, "__dict__") and not isinstance(getattr(type(result), "claims", None), property):
        have = list(getattr(result, "claims", None) or ())
        result.claims = Claims(have + _new(claims, have))
        return result
    raise TypeError(f"a result of type {type(result).__name__} cannot carry the calibration claims")


CalibratedVector, CalibratedFunction = _calibrated(Vector), _calibrated(Function)
