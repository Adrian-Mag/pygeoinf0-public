"""The covariance of a linear Gaussian posterior, applied without subtracting the prior's variance from the data's
reduction of it (plan 054, pre-existing bugs, item 3).

    C_post = C - C G* S^-1 G C,    S = G C G* + C_e

is a difference of nearly equal operators along the directions the data resolve well under a wide prior: applied as
written, <t, C_post t> has an absolute error of about eps <t, C t>, and the posterior std lost 6.5e-3 to all of its
digits at a = 1e6 (prior a^2 I) already at cond 1e2, with no warning (/disks/data/tmp/mission-054-report.md, P4). The
same operator in Joseph's form,

    C_post = (I - K G) C (I - K G)* + K C_e K*,    K = C G* S^-1,

is a sum of positive terms; the subtraction moves into u = (I - K G)* v = v - G* S^-1 G C v, a difference of elements
whose error is about eps ||v||, so <v, C_post v> keeps about half the digits the first form loses (its error is about
eps ||v|| ||C^1/2|| ||C_post^1/2 v|| instead of eps <v, C v>). K enters at second order (Joseph's form is stationary in
K), so the error of the solve does too. Applied as

    z = S^-1 G C v,   u = v - G* z,   C_post v = C u - C G* S^-1 (G C u - C_e z)

it costs two solves with S per application instead of one.

The posterior mean m + C G* z, S z = r = d - G m - E e (plan 054, wrong rows S2/P2), is refined: z += S^-1 rho with the
residual rho = r - G (C G* z) - C_e z formed by applying the operators, not the factorised matrix. The factorised S carries
an error of about eps ||S||, and its solve an error of about cond(S) eps in the mean (measured: 2.3e-7 to 4.4e-7 relative at
cond(S) 1e10, below P1's warning); the operators' residual has its large rounding along the directions S^-1 damps, so the
refined mean is left with about eps sqrt(cond(S)) (measured 5e-10 there; /disks/data/tmp/mission-054-report.md, "Wrong
rows"). REFINE steps. The stated error, in the model space's norm:
    SAFETY ||dx_last|| + eps ||C G*||_HS ||z|| + eps ||mean||,
the first an a-posteriori estimate (the size of the next correction), the second the rounding of the sum C G* z, which
the residual does not see where G is weak (measured: the true error reached 0.15 of it, while the first term alone was
exceeded up to 22 x), with the Hilbert-Schmidt norm over an orthonormal basis of the data space (a bound of the operator
norm) on a Euclidean model space, else a power-iteration estimate counted SAFETY times. Where it exceeds TOL ||mean|| a
PrecisionWarning, where it reaches ||mean|| a refusal with the choices named (the rule of solvers/dataspace/cancellation.py).
Only where S.inv is an inverse to rounding: an iterative solve (CG, above settings.dense_max or on function spaces)
is a stated approximation to settings.accuracy and is left as it is, without .mean_error (the author's decision 2).
Cost: REFINE more solves with the factorised S and applications of G, C G* and C_e, and C G* on n basis vectors (a
Euclidean model space) or 8 applications of C G* and G C (measured: pg.update at n = 1500, m = 3000 0.26 -> 0.33 s)."""
import warnings

import numpy as np

from ..core.claims import UNKNOWN, by_rule
from ..core.errors import PrecisionWarning, Refused
from ..core.numbers import Enclosure
from ..operators.linear import combine_outputs
from .covariance import Covariance, _matrix_trace, _trace_of, posterior_records

EPS = np.finfo(float).eps
REFINE = 2          # steps; the second step's correction states the error of the first
SAFETY = 2.0        # the next correction is an estimate of the error, not a bound (measured: module docstring)


def refined_mean(space, C, K, G, Ce, Sinv, r):
    """(C G* z, its stated error in the norm of space, without the eps ||mean|| term) for S z = r, refined with the
    operators' residual (module docstring); (C G* z, None) unrefined where Sinv is not an inverse to rounding (an
    iterative solve to settings.accuracy, a stated approximation that stays as it is, or a pseudo-inverse)."""
    z = Sinv(r)
    x = K(z)
    if not _to_rounding(Sinv):
        return x, None
    last = 0.0
    for _ in range(REFINE):
        rho = combine_outputs([(1.0, r), (-1.0, G(x)), (-1.0, Ce(z))], G.codomain)
        dz = Sinv(rho)
        dx = K(dz)
        z, x = combine_outputs([(1.0, z), (1.0, dz)], G.codomain), combine_outputs([(1.0, x), (1.0, dx)], space)
        last = float(space.norm(dx))
    return x, SAFETY * last + EPS * _norm_bound(space, K, lambda w: G(C(w)), G.codomain, z) * float(G.codomain.norm(z))


def _to_rounding(Sinv):
    """True for an inverse computed to rounding (a direct factorisation, or an inverse without a solver)."""
    from ..solvers import CG
    solver = getattr(Sinv, "solver", None)
    return not isinstance(solver, CG) and getattr(solver, "rcond", None) is None


def _norm_bound(space, K, Ks, D, v):
    """||K||_HS over an orthonormal basis of D (a bound of the operator norm) where the model space is Euclidean (K on
    that basis is a matrix product), else a power-iteration estimate of ||K|| from v (Ks = K* = G C, C self-adjoint),
    counted SAFETY times (measured: the true error reached 0.20 of the stated one on that route, 0.16 with HS)."""
    from ..spaces.elements import VectorFamily
    from ..spaces.base import Euclidean
    try:
        if not isinstance(space, Euclidean):
            raise TypeError("not a coordinate space")
        E = VectorFamily(D, np.asarray(D.from_orthonormal(np.eye(D.dim)), float).T, name="e")
        F = K(E)
        return float(np.sqrt(max(np.trace(np.atleast_2d(np.asarray(space.inner(F, F), float))), 0.0)))
    except (AttributeError, NotImplementedError, TypeError, MemoryError):
        est = 0.0
        for _ in range(8):
            nv = float(D.norm(v))
            if nv == 0:
                break
            w = K(combine_outputs([(1.0 / nv, v)], D))
            est, v = float(space.norm(w)), Ks(w)
        return SAFETY * est


def judge_mean(name, norm, err):
    """The general rule for the mean's stated error: a PrecisionWarning where err > TOL norm, a refusal where err reaches
    norm (a mean with no correct digit)."""
    from ..core.precision import DIGITS, TOL
    if not err > TOL * norm:
        return
    msg = (f"{name}: the posterior mean holds fewer than {DIGITS} correct digits in the model space's norm under the stated "
           f"rounding (relative error {err / norm if norm > 0 else np.inf:.2g}; probability/posterior_covariance.py)")
    choices = ["a narrower prior or a larger noise covariance: a mathematical choice", "the operators in higher precision"]
    if err >= norm:
        raise Refused(msg, needs=choices)
    warnings.warn(msg + ". " + "; ".join(choices), PrecisionWarning, stacklevel=4)


class PosteriorCovariance(Covariance):
    """C - C G* S^-1 G C in Joseph's form (module docstring), a Covariance (probability/covariance.py): self-adjoint and
    positive semidefinite by posterior_records from the prior's and the noise's statements (prior=, noise=: the
    measures; S=: G C G* + C_e, whose self-adjointness makes the operator applied Joseph's form); nothing claimed
    without them. Its trace: from the matrix on R^n, else trace_bound()."""

    why_linear = "sums and products of C, G, C_e and S^-1"

    def linear_parts(self):
        """The operators whose linearity this one inherits."""
        return [self.C, self.G, self.Ce, self.Sinv]

    def __init__(self, C, G, Ce, Sinv, prior=None, noise=None, S=None):
        sa, pos = (None, None) if prior is None or noise is None else \
            posterior_records(prior, noise, "C - C G* S^-1 G C", S)
        super().__init__(C.domain, name="C - C G* S^-1 G C", selfadjoint=sa, positive=pos)
        self.C, self.G, self.Ce, self.Sinv, self.Gs = C, G, Ce, Sinv, G.adjoint
        self.form = "Joseph: (I - K G) C (I - K G)* + K C_e K*, K = C G* S^-1 (probability/posterior_covariance.py)"

    def trace(self):
        """tr C_post: the matrix's on R^n (n <= settings.dense_max; formed once, on the first call), else
        trace_bound()."""
        if getattr(self, "_tr", None) is None:
            t = _matrix_trace(self)
            self._tr = t if t is not None else self.trace_bound()
        return self._tr

    def trace_bound(self):
        """0 <= tr C_post <= tr C without a matrix: 0 <= C_post (positive semidefinite) and C_post <= C (C G* S^-1 G C
        is positive semidefinite where S is), for the posterior of the exact solve; status by the rule from the prior
        trace's and the positivity's (UNKNOWN where either is not on record)."""
        T = _trace_of(self.C)
        pos = self._pos
        if T is None or not np.isfinite(T.upper) or not pos or T.status == UNKNOWN:
            return Enclosure(np.nan, 0.0, np.inf, UNKNOWN, "no trace bound: the prior's trace or the posterior's "
                             "positivity is not on record")
        return Enclosure(0.5 * T.upper, 0.0, T.upper, by_rule(T.status, pos[1]),
                         "0 <= C - C G* S^-1 G C <= C: the posterior's trace is at most the prior's")

    def _apply(self, v):
        C, G, Gs, Sinv = self.C, self.G, self.Gs, self.Sinv
        z = Sinv(G(C(v)))
        u = combine_outputs([(1.0, v), (-1.0, Gs(z))], self.domain)
        Cu = C(u)
        w = combine_outputs([(1.0, G(Cu)), (-1.0, self.Ce(z))], G.codomain)
        return combine_outputs([(1.0, Cu), (-1.0, C(Gs(Sinv(w))))], self.codomain)

    def _make_adjoint(self):
        return self
