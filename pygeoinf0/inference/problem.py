"""The inference problem: the seven elements of the thesis (el:7-16) and the observed datum, every one optional
(plan 008; EXP-028 DESIGN.md 3.9). A method runs whenever the objects it needs exist (the SYNTHESIS addendum): no fixed
order, and the model-data relation is a relation, not a G.

    pb = pg.Problem(M=M, D=D)                                   any subset of M, D, mdr, P, mpr, prior, calibration,
    pb = pb.with_(mdr=pg.Additive(G, D.standard_gaussian()), datum=d)    datum; with_ returns a new record
    pb.methods()                                                which methods run, and what the others need
    pb.run("least_squares", gamma=0.1)                          a method (pygeoinf0.inference.methods); a gamma
                                                                chosen from the datum: pg.meta.discrepancy(pb); the
                                                                result carries the calibration's claims
    pb.update(), pb.infer()                                     the backward cell, then the forward one through the MPR
    pb.decomposition(middle)                                    the data-space decomposition of the MDR (plan 009 G3),
                                                                cached: the methods pb.run calls share it

An unstated prior (None) differs from the stated value none (pg.NoPrior(M)): update() on an unstated prior uses none
and records that the element was unstated.
"""
from ..core.claims import ASSERTED, Claim
from ..core.errors import SpaceMismatch
from ..probability import is_gaussian
from . import calibration as _calibration
from . import methods as _methods
from ..spaces.identity import require

FIELDS = ("M", "D", "mdr", "P", "mpr", "prior", "calibration", "datum")
ELEMENTS = {"M": "the model space M", "D": "the data space D", "mdr": "the model-data relation",
            "P": "the property space P", "mpr": "the model-property relation", "prior": "the prior state",
            "calibration": "the calibration statement"}      # the seven elements (el:7-16): FIELDS without the datum


class Problem:
    """An inference problem: the seven elements and the datum, all optional (module docstring)."""

    def __init__(self, **fields):
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise TypeError(f"unknown elements {sorted(unknown)}; the elements are {', '.join(FIELDS)}")
        self._f = {k: fields.get(k) for k in FIELDS}
        f = self._f
        for rel, a, b in (("mdr", "M", "D"), ("mpr", "M", "P")):
            R = f[rel]
            if R is None:
                continue
            for side, sp in ((R.domain, f[a]), (R.codomain, f[b])):
                if sp is not None and side != sp:
                    raise SpaceMismatch(f"the {rel} acts between {R.domain!r} and {R.codomain!r}, not {sp!r}")
        if f["prior"] is not None and f["M"] is not None:
            require(getattr(f["prior"], "space", f["M"]), f["M"], "the prior is not a state on the model space", "the prior")
        if f["datum"] is not None and f["mdr"] is not None:
            require(f["datum"].space, getattr(f["mdr"], "G", f["mdr"]).codomain,
                    "the datum is not an element of the data space of the MDR", "the datum")
        self._cache = {}               # data-space decompositions, keyed by (G, noise covariance, middle): decomposition.py

    def __getattr__(self, name):
        if name in FIELDS:
            return self._f[name]
        raise AttributeError(name)

    def with_(self, **changes):
        """A new problem with some elements added or replaced (it shares the cache of decompositions: their keys are
        the identities and values of the objects, so a replaced relation or prior gets its own)."""
        out = Problem(**{**self._f, **changes})
        out._cache = self._cache
        return out

    def methods(self):
        """[(name, runnable, what is missing, what it produces)] for every registered method."""
        out = []
        for name, m in _methods.METHODS.items():
            miss = m.missing(self._f)
            out.append((name, not miss, miss, m.produces))
        return out

    def run(self, name, **options):
        """Run a method on this problem's elements: refused with what is missing, for a calibration the method does not
        declare (Method.calibrations) and for an option it does not take; the result carries calibration_claims()
        (inference/calibration.py carry: plan 070 P5, review D9)."""
        if name in _methods.METHODS:
            _methods.METHODS[name].supports(self.calibration)
        out = _methods.run(name, {k: v for k, v in self._f.items() if k not in ("M", "D", "P", "calibration")},
                           **{"cache": self._cache, **options})
        return _calibration.carry(out, self.calibration_claims())

    def decomposition(self, middle=None):
        """The data-space decomposition of the MDR's operator and noise (solvers/dataspace/decomposition.py), from the cache:
        middle None (the model metric: SOLA, a norm-ball BGP), "prior" (the prior covariance: PLI, the evidence) or an
        operator on M (an ellipsoid's A^-1)."""
        if self.mdr is None:
            from ..core.errors import Refused
            raise Refused("a decomposition needs the model-data relation", needs=["mdr"])
        if isinstance(middle, str):
            if middle != "prior" or not is_gaussian(self.prior):
                raise ValueError('middle is None, "prior" (with a Gaussian prior) or an operator on M')
            middle = self.prior.covariance
        return _methods.decomposition_of(self.mdr, middle, self._cache)

    def update(self, rule=None, policy="strict"):
        """The updated state on the model space: the backward cell of (prior, mdr, datum)."""
        from ..cells import update
        from ..states.base import NoPrior
        prior = self.prior
        unstated = prior is None
        if unstated:
            prior = NoPrior(self.mdr.domain)
        out = update(prior, self.mdr, self.datum, rule=rule, policy=policy)
        if unstated and isinstance(getattr(out, "_claims", None), list):
            out._claims.append(Claim("prior element unstated (treated as none)", ASSERTED, "Problem.update"))
        return out

    def infer(self, rule=None, policy="strict"):
        """The output on the property space: the updated state pushed through the MPR."""
        from ..cells import push
        return push(self.update(rule, policy), self.mpr, policy=policy)

    def calibration_claims(self):
        """The claims of the calibration element (its check of this problem); unstated (None): the claim that no
        guarantee is stated."""
        if self.calibration is None:
            from ..core.claims import Claims
            return Claims([Claim("calibration: unstated", ASSERTED, "the problem has no calibration element (None): the "
                                 "result is not calibrated (el:539)", key="calibration")])
        return self.calibration.check(self)

    def __repr__(self):
        stated = [k for k in FIELDS if self._f[k] is not None]
        return f"Problem({', '.join(stated) or 'nothing stated'})"
