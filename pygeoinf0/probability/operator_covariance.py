"""The stated covariance of an operator, with its trace bookkeeping (plan 009 G6; EXP-030 DESIGN G6, NOTES.md section 6):
an implementation of the Covariance root (probability/covariance.py; plan 068 P4).

    C = pg.Covariance(A)                          a positive self-adjoint operator; its trace is computed when the
                                                  structure allows (finite rank, R^n, blocks), else given: trace=(lo, hi)
    C = pg.Covariance.spectral(lam, phi, tail=t, tail_max=m)
                                                  C = sum_n lam_n phi_n (x) phi_n with N known orthonormal eigenpairs and
                                                  an omitted part R >= 0 with tr R <= t and ||R|| <= m (a KL truncation:
                                                  the Sturm-Liouville constructions of plan 009 G5 give t and m)
    C.trace()                                     an Enclosure: value, lower <= tr C <= upper
    C.normalised(trace=1.0)                       (alpha / tr C) C, the relative scale error carried as a bracket
    C.congruence(B)                               B C B*: positivity, self-adjointness and trace class are kept,
                                                  tr(B C B*) = sum lam_n ||B phi_n||^2 + tr(B R B*) <= ... + ||B||^2 t
    C.compressed(P)                               P C P for an orthogonal projection P (a band projection)
    C.variance(u)                                 <C u, u> as an Enclosure

Mathematics (Conway, A Course in Functional Analysis, IX.2; Reed-Simon I, VI.6): a congruence B C B* of a positive
trace-class C by a bounded B is positive and trace class with tr(B C B*) <= ||B||^2 tr C; with the spectral
representation, tr(B C B*) = sum_n lam_n ||B phi_n||^2 (Parseval in the eigenbasis). An orthogonal projection has
norm <= 1. The trace is basis independent: on a direct sum the trace of a block operator is the sum of the traces of its
blocks, whatever the weights of the summands.

What is applied: the known part (all of C when there is no omitted part). With an omitted part, C u is realised as the
known part's action, and <C u, u> lies in [<C_N u, u>, <C_N u, u> + m ||P_N^perp u||^2] (P_N the projection onto the
known eigenvectors; after a congruence the cruder m' ||u||^2 with m' = ||B||^2 m). Samples of a Gaussian measure with
such a covariance are drawn from the known part, which is stated in the claims.
"""
import math

import numpy as np

from ..core.numbers import Enclosure
from ..core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN, Claim, Claims, by_rule
from ..core.errors import Refused, Unsupported
from ..core.logic import Truth, decide
from ..operators.linear import FiniteRank
from .covariance import CONGRUENCE, TRACE_CLASS, Covariance, _once, _trace_of, _weakest
from ..spaces.identity import identical, require_realised

def norm_bound(B):
    """An upper bound on ||B|| with its status, from how B was built (None when nothing is known)."""
    from ..operators.band import BandProjection
    from ..operators.linear import Identity, Sum
    if hasattr(B, "norm_bound"):
        c = B.claims.find("bounded")
        return float(B.norm_bound()), (c.status if c is not None else DERIVED)
    if isinstance(B, Identity):
        return 1.0, THEOREM
    if isinstance(B, BandProjection):
        return (1.0 if B.degrees else 0.0), THEOREM
    if isinstance(B, Sum):
        parts = [norm_bound(A) for _, A in B.terms]
        if any(p is None for p in parts):
            return None
        return sum(abs(c) * p[0] for (c, _), p in zip(B.terms, parts)), _weakest(*(p[1] for p in parts))
    from ..operators.linear import Composition
    if isinstance(B, Composition):
        a, b = norm_bound(B.A), norm_bound(B.B)
        return None if a is None or b is None else (a[0] * b[0], _weakest(a[1], b[1]))
    from ..spaces.base import Euclidean
    from ..settings import settings
    if hasattr(B, "kernels") and isinstance(B.codomain, Euclidean) and B.codomain.dim <= settings.dense_max:
        return _functionals_norm(B)
    if isinstance(B.domain, Euclidean) and isinstance(B.codomain, Euclidean):
        from ..settings import settings
        if max(B.domain.dim, B.codomain.dim) <= settings.dense_max:
            from scipy.linalg import eigh
            A = B.matrix()                  # ||B||^2 = the largest lambda with A^T W_Y A v = lambda W_X v
            WX, WY = B.domain.apply_metric(np.eye(B.domain.dim)), B.codomain.apply_metric(np.eye(B.codomain.dim))
            top = float(eigh(A.T @ WY @ A, 0.5 * (WX + WX.T), eigvals_only=True)[-1])
            return math.sqrt(max(top, 0.0)) * (1 + 1e-12), DERIVED
    return None


def _functionals_norm(G):
    """||G|| for functionals G into R^n: ||G||^2 = ||G G*|| = the largest eigenvalue of the Gram operator in orthonormal
    coordinates of R^n, plus the Gram's stated error (its kernels' realisations; plan 054 P3), or None when that is not
    stated finite."""
    from ..solvers.direct import orthonormal_form, stated_error
    A = G @ G.adjoint
    M = orthonormal_form(A, A.matrix())
    top = float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])
    st = stated_error(A)
    e = 0.0 if st is None else float(st[0])
    if not math.isfinite(e):
        return None
    return math.sqrt(max(top + e, 0.0)) * (1 + 1e-12), (DERIVED if st is None else _weakest(DERIVED, st[1]))


def _checked_positivity(A):
    """A positivity record of an operator on R^n checked with the precision rule (n <= settings.dense_max): the
    eigenvalues of the symmetric part of its Gram form W A against the rounding level of the largest (core.logic.decide:
    TRUE "psd" checked, FALSE refused by the caller, UNKNOWN nothing); None off R^n or above dense_max."""
    from ..spaces.base import Euclidean
    from ..settings import settings
    X = A.domain
    if not (isinstance(X, Euclidean) and X == A.codomain and X.dim <= settings.dense_max):
        return None
    K = X.apply_metric(A.matrix())
    lam = np.linalg.eigvalsh(0.5 * (K + K.T))
    top = float(np.abs(lam).max(initial=0.0))
    low = float(lam.min(initial=0.0))
    t = decide(max(-low, 0.0), top) if top > 0 else Truth.TRUE
    if t is Truth.TRUE:
        return ("psd", CHECKED, f"the Gram form's smallest eigenvalue {low:.3g} is within the rounding level of the "
                f"largest {top:.3g} (core.logic.decide)")
    if t is Truth.FALSE:
        return (False, CHECKED, f"the Gram form has an eigenvalue {low:.3g} below the rounding level of the largest "
                f"{top:.3g}")
    return None


class OperatorCovariance(Covariance):
    """A covariance given by an operator it applies, with its trace and omitted part (module docstring): pg.Covariance(A).

    Self-adjointness and positivity: the records given (selfadjoint=, positive=: the conclusions of the caller's
    construction rule), else the operator's own records, else on R^n (n <= settings.dense_max) checked numerically with
    the precision rule (core.logic.decide; FALSE is refused: not a covariance), else asserted: stated as a covariance."""

    why_linear = "the operator it applies"

    def linear_parts(self):
        """The operators whose linearity this one inherits."""
        return [self.realised]

    def __init__(self, operator, trace=None, name=None, selfadjoint=None, positive=None, _finite=None, _rest=None,
                 _claims=()):
        if operator.domain != operator.codomain:
            raise ValueError("a covariance maps a space to itself")
        require_realised(operator.domain, operator.codomain, getattr(operator, "name", None))
        nm = name or getattr(operator, "name", "C")
        self.realised = operator              # what is applied
        self.finite = _finite                 # (generators, weights, orthonormal) of the known part, or None
        self.rest = _rest                     # None (exact) or dict(norm=m, trace=t, status=, reason=, orthogonal=bool)
        sa = selfadjoint or operator._sa or operator.is_selfadjoint()
        if sa and sa[0] is False:
            raise Refused(f"{getattr(operator, 'name', 'the operator')} is not self-adjoint ({sa[1]}: {sa[2]}): it is not "
                          "a covariance", needs=("a self-adjoint operator, e.g. (A + A*) / 2 if that is what you mean",))
        pos = positive or operator._pos or _checked_positivity(operator)
        if pos and pos[0] is False:
            raise Refused(f"{getattr(operator, 'name', 'the operator')} is not positive semidefinite ({pos[1]}: "
                          f"{pos[2]}): it is not a covariance", needs=("a positive semidefinite operator",))
        claims = list(_claims) + [c for c in getattr(operator, "claims", ()) if c.key == "rotation invariant"
                                  and not any(d.key == c.key for d in _claims)]
        super().__init__(operator.domain, nm, claims=claims,
                         selfadjoint=(True, sa[1], sa[2]) if sa and sa[0] else (True, ASSERTED, "stated as a covariance"),
                         positive=pos or ("psd", ASSERTED, "stated as a covariance"))
        if trace is None:
            tr = _trace_of(operator) if _rest is None else None
        elif isinstance(trace, Enclosure):
            tr = trace
        elif np.ndim(trace) == 0:
            tr = Enclosure(float(trace), float(trace), float(trace), ASSERTED, "the trace given")
        else:
            lo, hi = (float(x) for x in trace)
            tr = Enclosure(0.5 * (lo + hi), lo, hi, ASSERTED, "the trace bracket given")
        self._trace = tr
        self._refine = None                   # a callable: the same covariance with more of it known (sharper_error)

    # constructors -------------------------------------------------------------------------------
    @classmethod
    def spectral(cls, eigenvalues, eigenvectors, tail=0.0, tail_max=None, status=DERIVED, reason="given",
                 name="C", claims=(), refine=None):
        """C = sum_n lam_n phi_n (x) phi_n + R from N orthonormal eigenvectors phi_n (a Family or VectorFamily) and the
        bounds tr R <= tail, ||R|| <= tail_max (default: tail) on the omitted part (status and reason of the bounds).
        refine: a callable returning the same covariance with more of it known (more eigenpairs, a smaller omitted
        part), for the a-posteriori error a later inverse uses (sharper_error; plan 054)."""
        lam = np.asarray(eigenvalues, float).ravel()
        if np.any(lam < 0):
            raise ValueError("the eigenvalues of a covariance are >= 0")
        F = FiniteRank(eigenvectors, lam, name=name)
        tail = float(tail)
        if tail < 0:
            raise ValueError("the bound on the trace of the omitted part is >= 0")
        known = float(lam.sum())
        rel = max(1e-13, 2 * lam.size * np.finfo(float).eps)        # the rounding of the sum of N eigenvalues
        if tail == 0:
            tr = Enclosure(known, known * (1 - rel), known * (1 + rel), DERIVED, "the sum of the eigenvalues")
            rest = None
        else:
            tr = Enclosure(known + 0.5 * tail, known * (1 - rel), known * (1 + rel) + tail, status,
                           f"the known eigenvalues plus the omitted part's trace bound ({reason})")
            rest = dict(norm=float(tail if tail_max is None else tail_max), trace=tail, status=status, reason=reason,
                        orthogonal=True)
        out = cls(F, tr, name=name, selfadjoint=(True, THEOREM, "a spectral representation with real eigenvalues"),
                  positive=("psd", THEOREM, "eigenvalues >= 0"), _finite=(eigenvectors, lam, True), _rest=rest,
                  _claims=claims)
        out._refine = _once(refine)
        return out

    # trace ------------------------------------------------------------------------------------------
    def trace(self):
        """tr C as an Enclosure (lower and upper may be 0 and inf when nothing is known)."""
        if self._trace is None:
            return Enclosure(math.nan, 0.0, math.inf, UNKNOWN, "no trace on record (give trace=(lo, hi))")
        return self._trace

    def normalised(self, trace=1.0, at="value", name=None):
        """(alpha / t) C with t the value (or the lower or upper end, at=) of tr C: the result has trace alpha up to the
        relative bracket [t / upper, t / lower], recorded in its claims."""
        T = self.trace()
        if not np.isfinite(float(T)) or T.upper == math.inf:
            raise Unsupported(f"{self.name}: the trace is not known well enough to normalise ({T!r})")
        t = {"value": float(T), "lower": T.lower, "upper": T.upper}[at]
        c = float(trace) / t
        rel = Enclosure(1.0, t / T.upper, t / T.lower if T.lower > 0 else math.inf, T.status,
                        "exact scale alpha / tr C relative to the scale used")
        out = self._scaled(c, name or f"{float(trace):g}/tr·{self.name}")
        out._trace = Enclosure(float(trace) * float(T) / t, c * T.lower, c * T.upper, T.status,
                               f"normalised with t = the {at} of tr {self.name}")
        out._claims.append(rel.claim(f"normalising scale relative to alpha / tr {self.name}", key="normalisation"))
        out.scale_bracket = rel
        return out

    def scaled(self, c, name=None):
        """c C for c >= 0 (the trace, the omitted part's bounds and any degree structure scale with it)."""
        if not float(c) >= 0:
            raise ValueError("a covariance is scaled by a number >= 0")
        return self._scaled(float(c), name or f"{float(c):g}·{self.name}")

    def _scaled(self, c, name):
        op = c * self.realised
        fin = None if self.finite is None else (self.finite[0], c * self.finite[1], self.finite[2])
        if fin is not None:
            op = FiniteRank(fin[0], fin[1], name=name)
        rest = None if self.rest is None else dict(self.rest, norm=c * self.rest["norm"], trace=c * self.rest["trace"])
        sa, pos = self._sa, self._pos
        out = OperatorCovariance(op, self.trace().scaled(c), name=name, _finite=fin, _rest=rest, _claims=self._inherited(),
                                 selfadjoint=(True, by_rule(sa[1]), f"c {self.name} with c >= 0 and {self.name} "
                                              f"self-adjoint"),
                                 positive=("psd", by_rule(pos[1]), f"c {self.name} with c >= 0 and {self.name} positive "
                                           "semidefinite"))
        if self._refine is not None:
            out._refine = _once(lambda: self._refine()._scaled(c, name))
        parts = getattr(self, "degree_parts", None)
        if parts is not None:
            out.degree_parts = {l: (p[0], p[1], c * p[2]) + tuple(p[3:]) for l, p in parts.items()}
            out.quadrature = getattr(self, "quadrature", None)
        spec = getattr(self, "angular_spectrum", None)
        if spec is not None:
            out.angular_spectrum = {l: c * v for l, v in spec.items()}
        return out

    def _inherited(self):
        return [c for c in self._claims if c.key == "rotation invariant"]

    # algebra ----------------------------------------------------------------------------------------
    def congruence(self, B, name=None, quadrature=None, tail_modes=4):
        """B C B* (_congruence); with C's refinement carried over (sharper_error)."""
        out = self._congruence(B, name, quadrature, tail_modes)
        if self._refine is not None and out.rest is not None:
            out._refine = _once(lambda: self._refine().congruence(B, name, quadrature, tail_modes))
        return out

    def sharper_error(self):
        """(e, status, how): an a-posteriori bound on ||C~ - C|| in the domain's norm (plan 054, the general rule), C~
        the operator applied and C the covariance meant, from C' = the same covariance with more of it known (its
        refinement, taken pg.settings.verification times, at least once: each refinement's own refinement, while there is
        one and it omits something): ||C~ - C|| <= ||C~ - C'~|| + ||C'~ - C'|| + ||C' - C|| <= ||C~ - C'~||_2 + e' +
        ||R'||, R' the omitted part of the deepest C' and e' the stated error of what it applies. A bound where the
        omitted parts' bounds are. None where there is no refinement, the domain has no dense matrix (above
        settings.dense_max), or it is not smaller than the first-order statement ||R||."""
        from ..core.claims import weakest
        from ..spaces.base import Euclidean
        from ..operators.refinement import depth
        from ..settings import settings
        from ..solvers.direct import orthonormal_form, stated_error
        D, r = self.domain, self.rest
        if self._refine is None or r is None or not isinstance(D, Euclidean) or D.dim > settings.dense_max:
            return None
        chain, C2, asked, why = [], self, depth(), ""
        while len(chain) < asked:
            if C2._refine is None or C2.rest is None:
                why = "nothing omitted" if C2.rest is None else "no further refinement"
                break
            try:
                Cn = C2._refine()
            except (Unsupported, ArithmeticError, MemoryError) as exc:   # the refinement cannot be realised here
                why = f"the next refinement cannot be realised here: {str(exc)[:90]}"
                break
            chain.append(float(np.linalg.norm(orthonormal_form(self, C2.realised.matrix() - Cn.realised.matrix()), 2)))
            C2 = Cn
        if not chain:
            return None
        e2 = 0.0 if C2.rest is None else float(C2.rest["norm"])
        inner2 = stated_error(C2.realised, True)
        diff = float(np.linalg.norm(orthonormal_form(self, self.realised.matrix() - C2.realised.matrix()), 2))
        e = diff + e2 + (inner2[0] if inner2 else 0.0)
        inner = stated_error(self.realised)
        if not e < float(r["norm"]) + (inner[0] if inner else 0.0):
            return None
        st = weakest(r["status"], *((C2.rest["status"],) if C2.rest else ()), *(inner2[1:2] if inner2 else ()))
        lev = f"verification level {len(chain)}" + ("" if len(chain) == asked else f" of the {asked} asked ({why})")
        return (e, st, f"the covariance applied as its known part; a posteriori: ||C~ - C'|| = {diff:.3g}, C' the same "
                f"covariance with more of it known, plus its omitted part ||R'|| <= {e2:.3g}" +
                (f" and its stated error {inner2[0]:.3g}" if inner2 else "") +
                f" (the triangle inequality; {lev}, ||C^(j-1) - C^(j)|| = {', '.join(f'{x:.2g}' for x in chain)}; the "
                f"first-order statement ||R|| <= {r['norm']:.3g})")

    _sharper_error = sharper_error            # the protocol of solvers.stated_error(A, sharp=True)

    def _congruence(self, B, name=None, quadrature=None, tail_modes=4):
        """B C B* on B's codomain (module docstring). A covariance built degree by degree keeps that structure under a
        radial multiplication (probability/degrees.py; quadrature=: a declared rule for the radial integrals of the
        trace, a realisation option; tail_modes: the omitted part's trace summed explicitly to that many times the modes
        kept, 0 for the plain ||B||^2 bound)."""
        if B.domain != self.domain:
            raise ValueError(f"{B.name} acts on {B.domain}, not on {self.domain}")
        require_realised(B.domain, self.domain, B.name)
        if getattr(self, "degree_parts", None) is not None and getattr(B, "radial_factor", None) is not None \
                and B.radial_factor() is not None:
            from .degrees import congruence_radial
            return congruence_radial(self, B, quadrature, name, tail_modes)
        nm = name or f"{B.name}{self.name}{B.name}*"
        nb = norm_bound(B)
        claims = []
        records = dict(selfadjoint=(True, by_rule(self._sa[1]), "a congruence B C B* of a self-adjoint operator"),
                       positive=("psd", by_rule(self._pos[1]), "a congruence of a positive semidefinite operator"))
        if self.finite is not None:
            gens, lam, _ = self.finite
            op = FiniteRank(B(gens), lam, name=nm)
            known = float(op.trace())
            if self.rest is None:
                rest, tr = None, Enclosure(known, known * (1 - 1e-13), known * (1 + 1e-13), DERIVED,
                                           "sum lam_n ||B phi_n||^2 (finite rank)")
            else:
                r = self.rest
                if nb is None:
                    rest = dict(r, norm=math.inf, trace=math.inf, status=UNKNOWN, orthogonal=False,
                                reason="no bound on ||B||")
                else:
                    st = _weakest(r["status"], nb[1])
                    rest = dict(norm=nb[0] ** 2 * r["norm"], trace=nb[0] ** 2 * r["trace"], status=st,
                                reason=f"||B||^2 times the omitted part's bounds ({r['reason']}); {CONGRUENCE}",
                                orthogonal=False)
                tr = Enclosure(known + 0.5 * rest["trace"] if np.isfinite(rest["trace"]) else known, known,
                               known + rest["trace"], rest["status"],
                               "sum lam_n ||B phi_n||^2 over the known eigenpairs plus ||B||^2 tr R")
            out = OperatorCovariance(op, tr, name=nm, _finite=(op.generators, lam, False), _rest=rest, **records)
        else:
            op = B @ self.realised @ B.adjoint
            tr = _trace_of(op)
            if tr is None:
                T = self.trace()
                hi = math.inf if nb is None else nb[0] ** 2 * T.upper
                tr = Enclosure(math.nan if not np.isfinite(hi) else 0.5 * hi, 0.0, hi,
                               UNKNOWN if nb is None else _weakest(T.status, nb[1]), f"tr(B C B*) <= ||B||^2 tr C; "
                               f"{CONGRUENCE}")
            out = OperatorCovariance(op, tr, name=nm, **records)
        a, b = self.claims.find("rotation invariant"), B.claims.find("rotation invariant")
        if a is not None and b is not None and a.holds and b.holds:
            claims.append(Claim("commutes with rotations", _weakest(a.status, b.status), "a congruence of "
                                "rotation-invariant operators (EXP-030 Theorem G2.1)", key="rotation invariant"))
        out._claims += claims
        return out

    @classmethod
    def angular(cls, space, spectrum, name="C"):
        """The isotropic covariance with angular power spectrum {l: c_l} on a sphere space (probability/degrees.py)."""
        from .degrees import angular
        return angular(space, spectrum, name)

    @classmethod
    def glued(cls, M, parts, name="C"):
        """A covariance on the direct sum M from degree-structured parts on its summands or on groups of summands that
        tile one domain (probability/degrees.py)."""
        from .degrees import glued
        return glued(M, parts, name)

    def radial_variance(self, r):
        """Var u(r, xi), u ~ N(0, C), for a covariance built degree by degree (probability/degrees.py)."""
        if getattr(self, "degree_parts", None) is None:
            raise Unsupported(f"{self.name}: the pointwise variance needs the radial structure per degree")
        from .degrees import radial_variance
        return radial_variance(self, r)

    def degree_spectrum(self):
        """(eigenvalues, multiplicities, tail, tail_max) of a covariance built degree by degree on one space."""
        parts = getattr(self, "degree_parts", None)
        if parts is None:
            raise Unsupported(f"{self.name}: no degree structure on record")
        from .degrees import _interval, part_spectrum
        a, b = _interval(self.domain)
        lam, mult = [], []
        for l, p in parts.items():
            ev = part_spectrum(p[1], p[2], p[3] if len(p) > 3 else None, a, b, getattr(self, "quadrature", None))
            lam.append(ev)
            mult.append(np.full(len(ev), 2 * l + 1.0))
        r = self.rest
        return np.concatenate(lam), np.concatenate(mult), (r["trace"] if r else 0.0), (r["norm"] if r else None)

    def compressed(self, P, name=None):
        """P C P, the compression of C to the range of an orthogonal projection P (a band projection, or an operator
        claimed to be one)."""
        from ..operators.band import BandProjection
        ok = isinstance(P, BandProjection) or any("orthogonal projection" in c.statement for c in P.claims)
        if not ok:
            raise ValueError(f"{P.name} is not known to be an orthogonal projection (P C P is a compression only then)")
        out = self.congruence(P, name or f"{P.name}{self.name}{P.name}")
        out._claims.append(Claim(f"supported on the range of {P.name}", THEOREM, "P C P with P an orthogonal "
                                 "projection: zero on ker P, ||P|| <= 1 keeps the bounds", key="compressed"))
        return out

    def variance(self, u):
        """<C u, u> as an Enclosure (an element u of the space)."""
        v = float(self.domain.inner(self.realised(u), u))
        if self.rest is None:
            e = 1e-13 * abs(v)
            return Enclosure(v, v - e, v + e, DERIVED, "the covariance applied exactly")
        r = self.rest
        if r["orthogonal"] and self.finite is not None:
            gens = self.finite[0]
            c = np.asarray(self.domain.inner(gens, u), float).ravel()
            perp = max(float(self.domain.inner(u, u)) - float(c @ c), 0.0)
        else:
            perp = float(self.domain.inner(u, u))
        return Enclosure(v, v, v + r["norm"] * perp, r["status"], "the known part plus ||R|| ||P^perp u||^2")

    def form_bracket(self, F, G=None):
        """(<C f_i, g_j>, E): the known part's matrix for families F, G (G = F by default) and entrywise bounds E on the
        omitted part's contribution, E_ij = ||R|| ||P^perp f_i|| ||P^perp g_j|| (Cauchy-Schwarz; ||f_i|| for P^perp
        after a congruence). E is 0 when nothing is omitted."""
        G = F if G is None else G
        K = np.atleast_2d(np.asarray(self.domain.inner(self.realised(F), G), float))
        if self.rest is None:
            return K, np.zeros_like(K)

        def perp(H):
            sq = np.atleast_1d(np.asarray(self.domain.norm(H), float)) ** 2
            if self.rest["orthogonal"] and self.finite is not None:
                c = np.atleast_2d(np.asarray(self.domain.inner(self.finite[0], H), float))
                sq = sq - np.sum(c * c, axis=0)
            return np.sqrt(np.maximum(sq, 0.0))
        pf = perp(F)
        pg = pf if G is F else perp(G)
        return K, self.rest["norm"] * np.outer(pf, pg)

    # operator protocol ----------------------------------------------------------------------------
    def _apply(self, x):
        return self.realised(x)

    def _make_adjoint(self):
        return self

    def matrix(self):
        return self.realised.matrix()

    def degree_scalars(self, space):
        """The degree structure of the realised operator, when the covariance is applied exactly (operators/
        equivariance.py)."""
        if self.rest is not None:
            return None
        from ..operators.equivariance import degree_scalars
        return degree_scalars(self.realised, space)

    def degree_operators(self, space):
        """Per degree the radial operator of the known part (operators/equivariance.degree_operators), when the
        covariance was built degree by degree (Laplacian.bessel_potential), else None."""
        parts = getattr(self, "degree_parts", None)
        if parts is None or not identical(space, self.domain):
            return None
        return [lambda l: parts.get(int(l))]

    def pushforward(self, A):
        return self.congruence(A)

    def _known_trace(self):
        return self.trace()

    def _trace_class_claim(self):
        """The root's claim; else asserted: the operator was stated to be a covariance (pg.Covariance(A))."""
        c = super()._trace_class_claim()
        if c is None:
            T = self.trace()
            c = Claim("the covariance is trace class", ASSERTED if T.status == UNKNOWN else T.status,
                      "no finite trace bound on record", key=TRACE_CLASS)
        return c

    @property
    def claims(self):
        cl = list(super().claims)
        if self.rest is not None:
            r = self.rest
            parts = getattr(self, "degree_parts", None)
            rank = len(self.finite[1]) if self.finite is not None else \
                sum((2 * l + 1) * len(p[2]) for l, p in parts.items()) if parts else "?"
            cl.append(Claim(f"applied and sampled as its known part (rank {rank}); the omitted part "
                            f"R >= 0 has tr R <= {r['trace']:.6g}, ||R|| <= {r['norm']:.6g}", r["status"], r["reason"],
                            key="truncation"))
        return Claims(cl)

    def __repr__(self):
        return f"{self.name}: covariance on {self.domain}, tr {self.trace()!r}"

