"""The Covariance root (plan 068 P4, architecture review B6): a positive, self-adjoint, trace-class operator C: X -> X
with a bracketed trace, and the rules its implementations share.

    C.trace()                     an Enclosure containing tr C (lower 0 and upper inf when nothing is known)
    C.selfadjoint, C.positive, C.trace_class
                                  Truth (core/logic.py): TRUE where a claim on record holds, FALSE where one records the
                                  property fails, UNKNOWN where nobody derived, checked or stated it
    C.holds(key)                  (Truth, status) of "self-adjoint", "positive semidefinite" (implied by "positive
                                  definite"), "positive definite" or "trace class"
    pg.Covariance(A, trace=...)   the stated covariance of an operator (probability/operator_covariance.py)

The implementations: OperatorCovariance (an operator stated or built to be a covariance: spectral representations,
congruences, normalisations; probability/operator_covariance.py, and GluedCovariance in probability/degrees.py),
PosteriorCovariance (Joseph's form, probability/posterior_covariance.py), FactoredCovariance (Phi V Phi^T + diag on
R^p, probability/lazy_covariance.py) and LowRankCovariance (solvers/dataspace/factored.py). FiniteRank (operators/linear.py)
stays a general operator: a sum of lam_j psi_j (x) psi_j is a covariance only for lam_j >= 0.

The properties come from the construction's claims: derived by a named rule from premises with their statuses
(posterior_records: a posterior covariance is self-adjoint and positive semidefinite when the prior and the noise
covariances are; core.claims.by_rule never promotes), checked numerically on R^n with the precision rule
(core.logic.decide), stated by the caller (asserted), or not on record (UNKNOWN). Nothing is set by hand after the
construction (plan 068 P4: inference/methods.py set them on a Matrix).

Mathematics (Conway, A Course in Functional Analysis, IX.2; Reed-Simon I, VI.6): a positive operator with a finite trace
is trace class; on a finite-dimensional space every operator is; the trace is basis independent (the sum of the
eigenvalues), so a matrix's trace in any coordinates is the operator's.
"""
import math

import numpy as np

from ..core.numbers import Enclosure
from ..core.claims import DERIVED, ORDER, THEOREM, UNKNOWN, Claim, Claims, derived, status_of
from ..core.logic import Truth
from ..operators.linear import FiniteRank, LinearOperator

SELF_ADJOINT, PSD, PD, TRACE_CLASS = "self-adjoint", "positive semidefinite", "positive definite", "trace class"

CONGRUENCE = "Conway, A Course in Functional Analysis, IX.2 (trace-class operators: tr(B C B*) <= ||B||^2 tr C)"


def _weakest(*statuses):
    return max(statuses, key=ORDER.index)


def _once(f):
    """f, called at most once (None stays None)."""
    if f is None:
        return None
    box = []

    def g():
        if not box:
            box.append(f())
        return box[0]
    return g


REFINE_MODES = 4    # a refinement keeps this many times the eigenpairs (realisations/spectra/laplacian_covariance.py)


def _matrix_trace(A):
    """tr A from the matrix of an operator on R^n (n <= settings.dense_max), or None."""
    from ..spaces.base import Euclidean
    from ..settings import settings
    if isinstance(A.domain, Euclidean) and A.domain == A.codomain and A.domain.dim <= settings.dense_max:
        M = A.matrix()
        t = float(np.trace(M))
        e = 1e-13 * float(np.abs(np.diag(M)).sum())
        return Enclosure(t, t - e, t + e, DERIVED, "the trace of the matrix (basis independent)")
    return None


def _trace_of(A):
    """An Enclosure for tr A from how A was built, or None."""
    from ..operators.blocks import BlockDiagonal
    if isinstance(A, Covariance):
        return A.trace()
    if isinstance(A, FiniteRank):
        t = A.trace()
        return Enclosure(t, t * (1 - 1e-13), t * (1 + 1e-13), DERIVED, "finite rank: sum lam_j ||psi_j||^2")
    t = _matrix_trace(A)
    if t is not None:
        return t
    if isinstance(A, BlockDiagonal) and not A.rows:
        parts = [None if B is None else _trace_of(B) for B in A.blocks]
        if any(p is None for p, B in zip(parts, A.blocks) if B is not None):
            return None
        live = [p for p in parts if p is not None]
        return Enclosure(sum(float(p) for p in live), sum(p.lower for p in live), sum(p.upper for p in live),
                         _weakest(*(p.status for p in live)) if live else DERIVED,
                         "the sum of the traces of the blocks")
    return None




def finite_dimensional(space):
    """Whether the space is finite-dimensional here: a Euclidean space, or a direct sum of such spaces."""
    from .gaussian import _finite_dimensional
    return _finite_dimensional(space)


class Covariance(LinearOperator):
    """The root of covariance operators: positive, self-adjoint, trace class, with a bracketed trace (module docstring).

    Covariance(space, name, selfadjoint=rec, positive=rec, claims=()) for an implementation: rec a (value, status,
    reason) record from the construction's rule (core.claims.derived), None where the rule has no premise on record
    (then nothing is claimed: UNKNOWN). pg.Covariance(A, trace=...) is the stated covariance of an operator A (an
    OperatorCovariance)."""

    why_linear = None
    rest = None                       # the omitted part of a truncated covariance (OperatorCovariance), None: exact
    finite = None                     # (generators, weights, orthonormal) of a known finite-rank part, or None

    def __new__(cls, *args, **kwargs):
        if cls is Covariance:                                   # pg.Covariance(A): the stated covariance of A
            from .operator_covariance import OperatorCovariance
            cls = OperatorCovariance
        return LinearOperator.__new__(cls)

    def __init__(self, space, name="C", selfadjoint=None, positive=None, claims=()):
        super().__init__(space, space, name=name)
        self._claims += list(claims)
        self.conclude(selfadjoint, positive)

    def conclude(self, selfadjoint=None, positive=None):
        """Record the conclusions of a construction's rule ((value, status, reason) records, core.claims.derived; e.g.
        posterior_records), each where it is stronger than the record on file: None, or an UNKNOWN status, records
        nothing (plan 065 A5: a missing premise is not promoted). Returns the covariance."""
        def stronger(new, old):
            return new and new[1] != UNKNOWN and (not old or ORDER.index(new[1]) < ORDER.index(old[1]))
        if stronger(selfadjoint, self._sa):
            self._sa = selfadjoint
        if stronger(positive, self._pos):
            self._pos = positive
        return self

    # the stated covariances' constructors (pg.Covariance.spectral, .angular, .glued: probability/operator_covariance.py)
    @classmethod
    def spectral(cls, *args, **kwargs):
        """OperatorCovariance.spectral: sum_n lam_n phi_n (x) phi_n plus an omitted part with stated bounds."""
        from .operator_covariance import OperatorCovariance
        return OperatorCovariance.spectral(*args, **kwargs)

    @classmethod
    def angular(cls, space, spectrum, name="C"):
        """The isotropic covariance with angular power spectrum {l: c_l} on a sphere space (probability/degrees.py)."""
        from .degrees import angular
        return angular(space, spectrum, name)

    @classmethod
    def glued(cls, M, parts, name="C"):
        """A covariance on a direct sum from degree-structured parts (probability/degrees.py)."""
        from .degrees import glued
        return glued(M, parts, name)

    # the contract ---------------------------------------------------------------------------------
    def trace(self):
        """tr C as an Enclosure: from the matrix on R^n (n <= settings.dense_max), else lower 0 and upper inf, UNKNOWN."""
        t = _matrix_trace(self)
        return t if t is not None else Enclosure(math.nan, 0.0, math.inf, UNKNOWN, "no trace on record")

    def _known_trace(self):
        """The trace where it costs no matrix (the claims ask for it): trace() off finite-dimensional spaces, else None;
        an implementation that keeps its trace returns it."""
        return None if finite_dimensional(self.domain) else self.trace()

    def _trace_class_claim(self):
        """The trace-class claim: from a finite trace bound, from finite dimension, else None (UNKNOWN)."""
        T = self._known_trace()
        if T is not None and math.isfinite(T.upper) and T.status != UNKNOWN:
            return Claim(f"the covariance is trace class, trace in [{T.lower:.10g}, {T.upper:.10g}]", T.status, T.reason,
                         key=TRACE_CLASS)
        if finite_dimensional(self.domain):
            return Claim("the covariance is trace class", THEOREM, "finite dimension", key=TRACE_CLASS)
        return None

    def holds(self, key):
        """(Truth, status) of one of the defining properties (module docstring)."""
        cl = self.claims
        if key == PSD:
            a = cl.status(PSD)
            if a[0] is not Truth.TRUE:
                b = cl.status(PD)
                if b[0] is Truth.TRUE:
                    return b
            return a
        return cl.status(key)

    @property
    def selfadjoint(self):
        """Truth: C is self-adjoint."""
        return self.holds(SELF_ADJOINT)[0]

    @property
    def positive(self):
        """Truth: C is positive semidefinite."""
        return self.holds(PSD)[0]

    @property
    def trace_class(self):
        """Truth: C is trace class."""
        return self.holds(TRACE_CLASS)[0]

    @property
    def claims(self):
        cl = [c for c in super().claims if c.key != TRACE_CLASS]
        c = self._trace_class_claim()
        return Claims(cl + ([c] if c is not None else []))

    def __repr__(self):
        return f"{self.name}: covariance on {self.domain}, tr {self.trace()!r}"


def posterior_records(prior, noise, what, S=None):
    """(self-adjoint, positive semidefinite) records of a posterior covariance (C - C G* S^-1 G C, or a congruence
    T (...) T* of it) by the rule: Joseph's form (I - K G) C (I - K G)* + K C_e K* is self-adjoint and positive
    semidefinite for every K when C and C_e are, and the operator applied is that form when S^-1 is self-adjoint
    (probability/posterior_covariance.py). Premises: the prior's and the noise's covariance statements
    (probability.gaussian.premise: the covariance's record, else the measure's statement, named in the reason), and S's
    self-adjointness where S is given. None for a conclusion whose premise is unknown."""
    from .gaussian import premise
    (s1, n1), (s2, n2) = premise(prior, "sa"), premise(noise, "sa")
    (p1, m1), (p2, m2) = premise(prior, "psd"), premise(noise, "psd")
    extra = () if S is None else (status_of(S._sa),)
    return (derived(True, f"{what}: Joseph's form, self-adjoint when C, C_e and S are{n1}{n2}", s1, s2, *extra),
            derived("psd", f"{what}: Joseph's form (I - K G) C (I - K G)* + K C_e K*, a sum of congruences of positive "
                    f"semidefinite operators{m1}{m2}", p1, p2, *extra))
