"""The lazy covariance of a property family on R^p, C = Phi V Phi^T + diag(extra), never formed (solvers/dataspace/factored.py;
plan 009 P1 review: no dense p x p covariance for map-sized families). A Covariance (probability/covariance.py; plan 068
P4: from operators/families.py, so that operators/ does not import probability/; the families are in properties/ since
plan 070 P1).

Its claims from its structure: self-adjoint (V is symmetrised, the metric of R^p the dot product); positive semidefinite
where V is (checked: V's eigenvalues against the rounding level of the largest, core.logic.decide; V is B x B) and the
extra diagonal is >= 0; otherwise nothing is claimed (a posterior's derivation from its premises is added by its
construction, probability.covariance.posterior_records). Its trace sum_p (Phi V Phi^T)_pp + extra_p, with the rounding of
the diagonal and of the matrix's own diagonal bounded by (B + p + 2) eps sum_p (|Phi| |V| |Phi|^T)_pp, twice
(the two routes), plus the same for the extra diagonal.
"""
import numpy as np

from ..core.claims import CHECKED, DERIVED, Claim, by_rule
from ..core.errors import Unsupported
from ..core.logic import Truth, decide
from ..core.numbers import Enclosure
from ..spaces.elements import Vector, VectorFamily
from ..solvers.dataspace.quadratic import rowquad
from .covariance import Covariance

EPS = np.finfo(float).eps


def _extra_nonnegative(extra):
    return extra is None or bool(np.all(np.asarray(extra) >= 0))


def _factored_trace(Phi, V, extra):
    """tr(Phi V Phi^T + diag(extra)) as an Enclosure (module docstring)."""
    d = rowquad(Phi, V)
    t = float(d.sum()) + (0.0 if extra is None else float(np.sum(extra)))
    size = float(rowquad(np.abs(Phi), np.abs(V)).sum()) + (0.0 if extra is None else float(np.abs(extra).sum()))
    e = 2.0 * (V.shape[0] + Phi.shape[0] + 2) * EPS * size
    return Enclosure(t, t - e, t + e, DERIVED, "sum_p Phi_p V Phi_p^T + extra_p, with the rounding of both routes to the "
                     "diagonal bounded by (B + p + 2) eps sum_p (|Phi| |V| |Phi|^T)_pp each")


def _factored_positivity(V, extra):
    """("psd", CHECKED, reason) where V is positive semidefinite to the rounding level and extra >= 0, else None."""
    lam = np.linalg.eigvalsh(V)
    top = float(np.abs(lam).max(initial=0.0))
    low = float(lam.min(initial=0.0))
    ok = Truth.TRUE if top == 0 else decide(max(-low, 0.0), top)
    if ok is Truth.TRUE and _extra_nonnegative(extra):
        return ("psd", CHECKED, f"Phi V Phi^T + diag(extra): V's smallest eigenvalue {low:.3g} is within the rounding "
                f"level of the largest {top:.3g} (core.logic.decide) and extra >= 0")
    return None


class FactoredCovariance(Covariance):
    """C = Phi V Phi^T + diag(extra) on R^p, never formed: products, the diagonal, blocks between chosen properties
    (module docstring)."""

    why_linear = "Phi V Phi^T + diag(extra), fixed matrices"

    def __init__(self, P, Phi, V, extra=None, name="C", claims=()):
        V = np.asarray(V, float)
        self.Phi, self.V, self.extra = Phi, 0.5 * (V + V.T), extra
        sa = (True, DERIVED, "Phi V Phi^T with V symmetric, plus a diagonal") if getattr(P, "is_standard", False) \
            else None
        super().__init__(P, name=name, selfadjoint=sa, positive=_factored_positivity(self.V, extra), claims=claims)

    def diagonal(self):
        d = rowquad(self.Phi, self.V)
        return d if self.extra is None else d + self.extra

    def trace(self):
        return _factored_trace(self.Phi, self.V, self.extra)

    def _known_trace(self):
        return self.trace()

    def block(self, rows, cols=None):
        """The covariance between the properties rows and cols (indices): a small dense matrix."""
        rows = np.atleast_1d(rows)
        cols = rows if cols is None else np.atleast_1d(cols)
        out = self.Phi[rows] @ self.V @ self.Phi[cols].T
        if self.extra is not None:
            out = out + np.where(rows[:, None] == cols[None, :], self.extra[rows][:, None], 0.0)
        return out

    def _apply(self, x):
        Y = np.atleast_2d(x.values)
        out = (Y @ self.Phi) @ self.V @ self.Phi.T + (0.0 if self.extra is None else Y * self.extra)
        return Vector(self.codomain, out[0]) if isinstance(x, Vector) else VectorFamily(self.codomain, out)

    def _make_adjoint(self):
        return self

    def matrix(self):
        from ..settings import settings
        if self.domain.dim > settings.dense_max:
            raise Unsupported(f"{self.name}: no {self.domain.dim} x {self.domain.dim} matrix above settings.dense_max = "
                              f"{settings.dense_max}; it is applied without one (raise the setting for a dense route)")
        return self.block(np.arange(self.domain.dim))

    def __repr__(self):
        return f"{self.name}: lazy covariance on R^{self.domain.dim} (Phi V Phi^T, {self.V.shape[0]} blocks)"


class LowRankCovariance(Covariance):
    """C = a^2 T S T* - a^4 B^T diag(1/s) B on R^p, applied without forming it (many properties of a plain map): a
    Covariance (probability/covariance.py; plan 070 P3: from solvers/dataspace/factored.py, whose closed form
    solvers/dataspace/factored.posterior_many returns it). Self-adjoint from its form (T S T* given self-adjoint, the
    metric of R^p the dot product); positivity is not claimed here (the posterior's rule adds it from its premises:
    inference/methods.py); its trace from the variances returned with it (trace())."""

    why_linear = "a fixed matrix expression applied to coordinates"

    def __init__(self, P, TT, beta, wts, a2, diag, name="C_post"):
        from ..solvers.dataspace.factored import COVARIANCE_DENSE
        rec = TT._sa
        sa = None if not getattr(P, "is_standard", False) or (rec and rec[0] is False) else \
            (True, by_rule(rec[1]) if rec else DERIVED, "a posterior covariance T (C - C G* S^-1 G C) T*: a^2 T S T* "
             "(T S T* built as one, solvers/dataspace/decomposition.py) minus a^4 B^T diag(w) B, the dot product on R^p")
        super().__init__(P, name=name, selfadjoint=sa, claims=[Claim(
            "covariance kept lazy (variances returned)", DERIVED, f"p = {P.dim} > COVARIANCE_DENSE = {COVARIANCE_DENSE}")])
        self.TT, self.beta, self.w, self.a2, self._diag = TT, beta, wts, a2, diag

    def trace(self):
        """sum_p of the variances returned (clipped at 0), widened by the rounding of the two terms of each,
        64 (B + p) eps (a^2 (T S T*)_pp + a^4 (B^T diag(w) B)_pp), and below by what the clipping removed (at most the
        second term where a variance is 0). Premise: the diagonal <t_p, S t_p> and T S T* are one Gram by two routes, equal
        to rounding (solvers/dataspace/decomposition._many_targets)."""
        d = np.asarray(self._diag, float)
        second = self.a2 * self.a2 * ((self.beta * self.beta).T @ np.asarray(self.w, float))
        first = d + second                                   # an upper bound of a^2 (T S T*)_pp (the clipping)
        t = float(d.sum())
        e = 64.0 * (self.beta.shape[0] + d.size) * EPS * float(np.sum(np.abs(first) + np.abs(second)))
        clip = float(second[d <= 0].sum())
        return Enclosure(t, t - e - clip, t + e, DERIVED, "the sum of the variances returned, widened by their rounding "
                         "and the clipping at 0")

    def _known_trace(self):
        return self.trace()

    def diagonal(self):
        return self._diag

    def _apply(self, x):
        Y = np.atleast_2d(x.values)
        first = self.a2 * self.TT(self.domain.family(Y)).values
        second = ((Y @ self.beta.T) * self.w[None, :]) @ self.beta
        out = first - self.a2 * self.a2 * second
        return Vector(self.codomain, out[0]) if isinstance(x, Vector) else VectorFamily(self.codomain, out)

    def _make_adjoint(self):
        return self
