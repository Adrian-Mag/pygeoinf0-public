"""Hardening a Gaussian measure to a set of prescribed mass (plan 009 G7; EXP-030 DESIGN G7, NOTES.md section 7; the
thesis inv:696-827, appendix D).

    S = mu.harden(B, mass=0.95)                 {m : ||B (m - c)||^2 <= t}, P_mu(S) >= mass, c the mean of mu
    S = mu.harden(A, mass=0.95, kind="quadratic")   {m : <A (m - c), m - c> <= t}, A self-adjoint, >= 0
    S = mu.harden(B, mass, law=pg.WeightedChiSquare(...))   with the law of the form supplied by the caller
    law_of_form(mu, B, kind)                    the law of the form under mu: a WeightedChiSquare, and the bound on the
                                                mean of the part of mu that is not known spectrally

For x ~ N(0, C), Q = ||B x||^2 = ||Y||^2 with Y = B x ~ N(0, B C B*), so Q = sum_j lam_j(B C B*) Z_j^2 (the Karhunen-Loeve
expansion of Y): a weighted chi-square law, which needs B C B* trace class. For <A x, x> the weights are those of
C^(1/2) A C^(1/2). t is the upper end of the bracket of the mass-quantile, so P(S) >= mass is proved by the bracket
(derived when the law's bracket is).

Computed laws. On R^n: Q = x^T K x with x ~ N(0, S) in standard coordinates, K = B^T W_Y B (or W A), weights the
eigenvalues of L^T K L, S = L L^T. On a function space the covariance needs a known spectral part (pg.Covariance.spectral
or a finite-rank covariance): x = sum_n sqrt(lam_n) xi_n psi_n + x_R, and the known part gives the weights of
Lam^(1/2) G Lam^(1/2), G_ij = <B psi_i, B psi_j> (or <A psi_i, psi_j>). An omitted part x_R (independent, E Q(x_R) <=
||B||^2 tr R) is not independent of the known part inside Q; with Q(a + b) <= (1 + e) Q(a) + (1 + 1/e) Q(b) (any
e > 0) and Markov,
    P(Q > t) <= P(Q_N > (t - s) / (1 + e)) + (1 + 1/e) E Q(x_R) / s,
so t = (1 + e) q_N(1 - beta_1) + (1 + 1/e) E Q(x_R) / beta_2 with beta_1 + beta_2 = 1 - mass (minimised over e and
the split) gives P(S) >= mass.

The Cameron-Martin ball {||C^(-1/2) m|| <= r} has mass 0 under mu in infinite dimension (C^(-1/2) C C^(-1/2) = I is not
trace class): hardening to it is refused with that reason.
"""
import math

import numpy as np

from ..core.numbers import Enclosure
from ..core.claims import ASSERTED, DERIVED, ORDER, THEOREM, Claim
from ..core.errors import Refused, Unsupported
from ..core.logic import Truth, decide
from .chisquare import WeightedChiSquare
from ..spaces.identity import require_realised
from ..sets.subsets import Subset


def _cameron_martin(mu, A):
    from ..operators.inverses import Inverse
    C = mu.covariance
    return isinstance(A, Inverse) and (A.A is C or A.A is getattr(C, "realised", None))


def law_of_form(mu, form, kind="norm"):
    """(WeightedChiSquare of the known part, bound on E Q(x_R) for the omitted part, its status) (module docstring)."""
    from ..spaces.base import Euclidean
    if kind not in ("norm", "quadratic"):
        raise ValueError("kind is 'norm' (||B m||^2) or 'quadratic' (<A m, m>)")
    if form.domain != mu.space:
        raise ValueError(f"the form acts on {form.domain}, not on {mu.space}")
    require_realised(form.domain, mu.space, form.name)
    if not isinstance(mu.space, Euclidean) and (_cameron_martin(mu, form) and kind == "quadratic"):
        raise Refused("the Cameron-Martin ball of a Gaussian measure on an infinite-dimensional space has measure 0 (the "
                      "form <C^-1 m, m> has the law of sum Z_j^2 over infinitely many j, which is infinite almost "
                      "surely): it cannot be hardened to a set of positive mass (inv:696-827)",
                      needs=["a form whose operator C^(1/2) A C^(1/2) is trace class"])
    if getattr(mu, "is_cylindrical", False):
        raise Refused("a cylindrical measure has no samples in the space: the law of a form on it is that of the "
                      "push-forward, give B into R^n")
    if isinstance(mu.space, Euclidean):
        n = mu.space.dim
        C = getattr(mu.covariance, "realised", mu.covariance)
        S = mu.space.solve_metric(C.matrix().T).T                       # coordinate covariance (gaussian.py)
        S = 0.5 * (S + S.T)
        if kind == "norm":
            M = form.matrix()
            K = M.T @ form.codomain.apply_metric(M)
        else:
            K = mu.space.apply_metric(form.matrix())
        K = 0.5 * (K + K.T)
        w, V = np.linalg.eigh(S)
        R = V * np.sqrt(np.maximum(w, 0.0))
        lam = np.linalg.eigvalsh(R.T @ K @ R)
        e = 64 * np.finfo(float).eps * n * max(float(np.abs(lam).max(initial=0.0)), 1e-300)
        return WeightedChiSquare(lam[lam > e], name="Q"), 0.0, DERIVED
    cov = mu.covariance
    fin = getattr(cov, "finite", None)
    from ..operators.linear import FiniteRank
    if fin is None and isinstance(cov, FiniteRank):
        fin, rest = (cov.generators, cov.lam, False), None
    else:
        rest = getattr(cov, "rest", None)
    if fin is None:
        raise Unsupported(f"the law of a quadratic form under {mu.name} needs a known spectral part of its covariance "
                          "(pg.Covariance.spectral, a finite-rank covariance), or law=pg.WeightedChiSquare(...) given")
    gens, lam, _ = fin
    if kind == "norm":
        P = form(gens)
        G = np.atleast_2d(form.codomain.inner(P, P))
    else:
        G = np.atleast_2d(mu.space.inner(form(gens), gens))
    G = 0.5 * (G + G.T)
    s = np.sqrt(lam)
    w = np.linalg.eigvalsh(s[:, None] * G * s[None, :])
    e = 64 * np.finfo(float).eps * len(w) * max(float(np.abs(w).max(initial=0.0)), 1e-300)
    law = WeightedChiSquare(w[w > e], name="Q_N")
    if rest is None:
        return law, 0.0, DERIVED
    from .operator_covariance import norm_bound
    nb = norm_bound(form)
    if nb is None:
        return law, math.inf, ASSERTED
    c = nb[0] ** 2 if kind == "norm" else nb[0]
    st = DERIVED if rest["status"] in (THEOREM, DERIVED) and nb[1] in (THEOREM, DERIVED) else ASSERTED
    return law, c * rest["trace"], st


def threshold(law, rest, mass, method="auto"):
    """t with P(Q <= t) >= mass, for Q the known law plus an omitted part with E <= rest (module docstring)."""
    beta = 1.0 - float(mass)
    if not 0 < beta < 1:
        raise ValueError("0 < mass < 1")
    if rest == 0:
        q = law.quantile(1 - beta, method=method)
        return q.upper, q
    if not np.isfinite(rest):
        raise Unsupported("no bound on the omitted part's contribution to the form: give a bounded form")
    best = math.inf
    for frac in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5):
        b2 = frac * beta
        q = law.quantile(1 - (beta - b2), method=method).upper
        for e in np.geomspace(1e-4, 1.0, 13):
            t = (1 + e) * q + (1 + 1 / e) * rest / b2
            best = min(best, t)
    return best, law.quantile(1 - beta, method=method)


class HardenedSet(Subset):
    """{m : Q(m - c) <= t} with P_mu >= mass (module docstring): a subset of the space (sets/subsets.py; plan 067 P3)
    with membership, the threshold and the mass claim; convex and closed as claimed, but not a ConvexSet: its support
    function is not computed here. On R^n harden returns a pg.Ellipsoid with the same attributes, which also has a
    support function."""

    convex = True

    def __init__(self, space, form, kind, t, centre, claims, name="S"):
        super().__init__(space, name, claims)
        self.kind_of_form, self.form, self.t, self.centre = kind, form, float(t), centre

    def value(self, m):
        d = m if self.centre is None else m - self.centre
        if self.kind_of_form == "norm":
            y = self.form(d)
            return float(self.form.codomain.inner(y, y))
        return float(self.space.inner(self.form(d), d))

    def member(self, m):
        return decide(self.value(m) - self.t, max(self.t, 1e-300))

    def support(self, v):
        raise Unsupported("the support function of a quadratic sublevel set in a function space needs the inverse of "
                          "its form (plan 009 P5, G8, generalises pg.Ellipsoid to function spaces)")

    def is_empty(self):
        return Truth.FALSE

    def __repr__(self):
        return f"{self.name} = {{Q(m - c) <= {self.t:.6g}}} ⊆ {self.space!r}"


def harden(mu, form, mass, kind="norm", law=None, method="auto", name=None):
    """The set {m : Q(m - c) <= t} with P_mu(S) >= mass (module docstring); c is mu's mean."""
    from ..spaces.base import Euclidean
    if law is None:
        law, rest, st = law_of_form(mu, form, kind)
        how = "the law of the form under the measure, computed"
    else:
        rest, st = 0.0, ASSERTED
        how = "the law of the form supplied by the caller"
    t, q = threshold(law, rest, mass, method)
    mass_status = max((st, q.status), key=ORDER.index)                 # the weaker of the law's and the quantile's
    m_br = law.cdf(t, method=method) if rest == 0 else Enclosure(float(mass), float(mass), 1.0, mass_status,
                                                                 "the omitted part's bound")
    claims = [Claim(f"prior mass >= {float(mass):g}", mass_status, f"{how}; t = {t:.12g} is the upper end of the "
                    f"bracket of its {float(mass):g}-quantile" + (" with the omitted part's bound" if rest else ""),
                    key="mass"),
              Claim("closed", THEOREM, "a sublevel set of a continuous quadratic form", key="closed"),
              Claim("convex", THEOREM, "a sublevel set of a positive semidefinite quadratic form", key="convex")]
    centre = None if mu.zero_mean else mu.mean
    nm = name or f"S^{1 - float(mass):g}"
    if isinstance(mu.space, Euclidean):
        from ..convex.sets import Ellipsoid
        M = form.matrix()
        K = M.T @ form.codomain.apply_metric(M) if kind == "norm" else mu.space.apply_metric(M)
        K = 0.5 * (K + K.T)
        if np.linalg.eigvalsh(K)[0] > 0:
            E = Ellipsoid(mu.space, math.sqrt(t), np.linalg.inv(K), centre=centre, name=nm)
            E._claims += claims
            out = E
        else:
            out = HardenedSet(mu.space, form, kind, t, centre, claims, nm)
    else:
        out = HardenedSet(mu.space, form, kind, t, centre, claims, nm)
    out.threshold, out.law, out.mass, out.quantile = t, law, m_br, q
    out.omitted = rest
    return out
