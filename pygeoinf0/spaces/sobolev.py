"""Sobolev spaces H^s_l(X), defined through the Bessel potential (1 - l^2 Laplacian)^(s/2).

    H^s_l(R^d)  = { u : ||u||^2 = (2 pi)^-d int (1 + l^2 |xi|^2)^s |u^(xi)|^2 dxi < inf }
    H^s_l(X)    = { U|_X : U in H^s_l(R^d) },  ||u|| = inf { ||U|| : U|_X = u }
                  for X an interval, box, disk, cylinder or ball (restriction space, quotient norm: McLean, Strongly
                  Elliptic Systems, 2000, ch. 3); in the flat coordinates of X for paths (arc length) and flat patches
    H^s_l(M)    = { u : sum_k (1 + l^2 lambda_k)^s |u_k|^2 < inf } on flat tori and circles (lambda_k the eigenvalues
                  of -Laplacian); by Poisson summation its kernel is the lattice sum of the free-space kernel

The reproducing (Bessel-potential) kernel is the Matern kernel K(x, y) = k(|x - y|) of realisations/rkhs/kernels.py; the
restriction of a reproducing-kernel space to a subset has the restricted kernel, so

    the functional u -> int u dmu (mu a finite measure supported in the closure of X with finite energy) has the Riesz
    representer k_mu = int K(., y) dmu(y), and <k_mu, k_nu> = int int K dmu dnu exactly.

A ray integral (mu = arc length on a segment) is bounded iff s > (d-1)/2; point evaluation iff s > d/2 (trace /
energy theorem: Jonsson-Wallin 1984, Adams-Hedberg 1996). s = 0 is L2. Elements are either combinations of
representers (exact norms) or formulas (inner products with representers exact; their own norms are the quotient
norm, which is not computed: NormUnavailable). pg.Inclusion(H, L2) maps elements to L2.
"""
import numpy as np

from ..core.claims import ASSERTED, CHECKED, THEOREM, Claim, Claims
from .constructions import Spectral, default_construction
from ..core.errors import Unsupported
from .elements import Family, Function
from ..measures import Measure, as_measures
from .base import L2, _FunctionHilbertSpace


class SobolevSpace(_FunctionHilbertSpace):
    """H^s_l(X) for s > 0 with a construction (spaces/constructions.py: restriction, intrinsic form, spectral power of a
    form's operator); build it with Sobolev(X, s, length_scale=l), a.space() or Sobolev(a.operator(), s)."""

    def __init__(self, construction, tol=1e-14, realisation=None, measure=None):
        self.construction = construction
        self.s, self.length_scale = construction.s, construction.length_scale
        if self.length_scale <= 0:
            raise ValueError("the length scale must be positive")
        self._setup(construction.domain if measure is None else measure, tol, realisation, sobolev=construction)

    def _with_realisation(self, realisation):
        return SobolevSpace(self.construction, self.tol, realisation, self.measure if self._measure_given else None)

    # ---------------------------------------------------------------- functionals
    def riesz(self, mu):
        """The Riesz representer of u -> int u dmu, for a measure, a set (its natural measure: arc length on a
        Segment or Polyline, area on a Patch or Sphere, length of a subinterval) or a point (its Dirac measure); a list
        gives a Family. A measure on a Dirichlet part of a constrained space defines the zero functional (every element
        vanishes there): its representer is exactly 0 (variances there are 0; pg.Functionals refuses it as a datum)."""
        from .dual import require_bounded
        items, many = as_measures(mu, self.domain.cartesian_dim)
        for m in items:
            require_bounded(m, self)          # the trace theorem and the weight class (spaces/dual.py)
        zero = np.array([self.construction.vanishes(m) for m in items])    # on a Dirichlet part: the zero functional
        R = self.backend.riesz(items, zero=zero) if zero.any() else self.backend.riesz(items)
        out = Family(self, R, name="k") if many else Function(self, pl=R, name=f"k[{items[0]}]")
        out._memo["riesz measures"] = tuple(items)      # what it represents (restriction to pieces, plan 035 P4)
        return out

    def evaluation(self, x0):
        """The bounded functional u -> u(x0) (s > d/2): a Functional with representer K(x0, .)."""
        from .dual import Functional
        r = self.riesz(np.atleast_1d(np.asarray(x0, float)))
        r = r[0] if isinstance(r, Family) else r                 # on an interval (1,) reads as one point in a list
        return Functional.from_representer(r, name=f"evaluation at {x0}")

    # ---------------------------------------------------------------- elements
    def _element_claim(self, f, ok):
        """f is in H^s (Function.claims; plan 065 A1). A combination of Riesz representers is, by construction. A
        function given as a formula (a plain callable, or arithmetic on one) is ASSERTED: its fit converged, and on
        intervals and paths it was checked to have no jump at its breaks in the derivative orders m + 1/2 <= s
        (realisations/series/sup.check_sobolev, refused otherwise), but samples do not prove membership. A bank function's
        regularity gives the theorem instead (spaces/membership.py)."""
        if not ok:
            return Claim(f"{f.name} is in {self}", "unknown", "the realisation did not converge everywhere")
        R = f.realisation
        fx = getattr(R, "fx", None)
        if getattr(R, "kx", None) is not None and fx is None:
            return Claim(f"{f.name} is in {self}", CHECKED, "a combination of Riesz representers of bounded functionals")
        orders = None if fx is None else getattr(self.backend, "jumps_checked", lambda fx: None)(fx)
        if orders is None:
            seen = "jumps of a formula are not checked in this realisation"
        elif not orders:
            seen = "no jump is excluded at this order (s < 1/2: piecewise smooth functions with jumps are elements)"
        else:
            m = max(orders)
            seen = ("its realisation has no jump at its breaks" + ("" if m == 0 else
                    f", nor its derivatives of order <= {m}") + " (checked on the realisation: realisations/series/sup.py)")
        return Claim(f"{f.name} is in {self}", ASSERTED,
                     f"given by a formula: its realisation converged; {seen}; samples of a function do not prove "
                     "membership (a bank function's regularity, f.element(M), gives the theorem; or "
                     "pg.assert_claim)")

    # ---------------------------------------------------------------- identity
    @property
    def claims(self):
        """The measure, the construction's claims (definition, completeness, boundary conditions), the reproducing
        kernel and the trace theorem: integrals over k-dimensional sets are bounded iff s > (d - k)/2 for every
        construction here (each is a closed subspace of H^s(D) with an equivalent norm, or H^s(D) itself, away from
        the Lions-Magenes borderline orders)."""
        bounded = self.dual().integrals_bounded           # the one trace rule (spaces/dual.py)
        rk = (Claim("reproducing kernel: point evaluation bounded", THEOREM, "s > d/2 (Sobolev embedding; Aronszajn)")
              if bounded(0) else
              Claim("not a reproducing-kernel space (point evaluation unbounded)", THEOREM, "s <= d/2"))
        trace = Claim(f"integrals over k-dimensional sets are bounded iff s > (d - k)/2: points {bool(bounded(0))}, "
                      f"paths {bool(bounded(1))}", THEOREM, "trace theorem (Jonsson-Wallin 1984)")
        return Claims([self._measure_claim()] + list(self.construction.claims) + [rk, trace]
                      + list(getattr(self.backend, "kernel_claims", ())))

    def __eq__(self, other):
        return (isinstance(other, SobolevSpace) and self._same_measure(other)
                and self.construction.key() == other.construction.key())

    def __hash__(self):
        return hash(("H", self.s, self.length_scale, self.measure))

    def __repr__(self):
        c = self.construction
        tag = "" if c.key() == default_construction(self.domain, self.s, self.length_scale).key() else f"; {c.label()}"
        return f"H^{self.s:g}({self.domain}, l={self.length_scale:g}{tag})"


def Sobolev(domain, s, *args, length_scale=None, tol=1e-14, realisation=None):
    """H^s_l(X): the Sobolev space of order s >= 0 with length scale l on a domain X (see spaces/sobolev.py).
    Sobolev(X, 0) is L2(X).

    Sobolev(A, s) for the operator A = a.operator() of a SobolevForm a is the spectral space D(A^(s/2k)) with
    ||A^(s/2k) u|| (spaces/constructions.py; its length scale is the form's): Sobolev(a.operator(), k) is a.space()."""
    if args:
        raise TypeError(f"give the length scale by name: Sobolev(X, {s:g}, length_scale={args[0]!r}) (a third "
                        "positional number would be ambiguous with the tolerance)")
    from ..operators.unbounded import FormOperator
    if isinstance(domain, FormOperator):
        if length_scale is not None and float(length_scale) != domain.length_scale:
            raise ValueError(f"the length scale of Sobolev(A, s) is the form's, l = {domain.length_scale:g}")
        if s == 0:
            return L2(domain.form.domain, tol, realisation)
        return SobolevSpace(Spectral(domain, s), tol, realisation)
    if s == 0:
        return L2(domain, tol, realisation)
    if s < 0:
        raise Unsupported("negative orders (dual spaces): use measures and M.riesz; H^-s is not a space of functions")
    if length_scale is None:
        raise TypeError("a Sobolev space of positive order needs its length scale: Sobolev(X, s, length_scale=l)")
    if float(length_scale) <= 0:
        raise ValueError("the length scale must be positive")
    mu = domain if isinstance(domain, Measure) else None
    return SobolevSpace(default_construction(domain.set if mu else domain, s, length_scale), tol, realisation, mu)
