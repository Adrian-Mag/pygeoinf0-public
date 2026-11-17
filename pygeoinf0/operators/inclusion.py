"""Inclusion(A, B): u -> u between function spaces on one domain. Inclusion(H, L2) embeds a Sobolev space H^s_l(X)
(s >= 0) into L2(X); Inclusion(C, Lp) and Inclusion(H, C) (s > 1/2, on intervals) connect C([a, b]) (class docstring).

H^s into L2 is bounded with norm <= 1, because (1 + l^2 |xi|^2)^s >= 1 and restriction does not increase L2 norms. It
is the one route from H^s to L2 (L2 norms of differences, comparison with a true model, arithmetic as functions): formula parts
are carried over exactly; combinations of Riesz representers are realised in L2 by the adaptive fit (they have kinks
across their supports, so their L2 realisation converges algebraically and carries its own error estimate). That fit
is made to the tolerance tol of the inclusion (default: the L2 space's tolerance, or 1e-4 when that is the 1e-14
default, because a fit to 1e-14 of a function with kinks along every ray costs minutes); u.error of the image reports
what was reached.
"""
import numpy as np

from ..core.claims import CHECKED, THEOREM, Claim
from ..core import fits
from ..spaces.elements import Family, Function
from ..core.errors import RealisationMismatch, SpaceMismatch, Unsupported
from ..measures import LebesgueMeasure
from .linear import LinearOperator


DEFAULT_TOL = 1e-4      # fit tolerance when the L2 space has the 1e-14 default (kinked representers: minutes at 1e-14)


class Inclusion(LinearOperator):
    """u -> u between function spaces on one domain:

    Inclusion(H, L2)    H^s_l(X) into L2(X): norm <= 1; representers are fitted in L2 to the relative tolerance tol
    Inclusion(C, Lp)    C([a, b]) into L^p([a, b]) (Lebesgue measure, p = 2 gives L2): norm <= (b - a)^(1/p); the
                        realisation is carried over, its error estimate converted
    Inclusion(H, C)     H^s_l([a, b]) into C([a, b]) for s > 1/2 (the Sobolev embedding): norm <= sqrt(k(0)), because
                        |u(x)| = |<u, k_x>| <= ||u|| sqrt(k(x, x)); representers are fitted in C to tol

    L^p into C does not exist: an element of L^p is a class, and most classes have no continuous member."""

    why_linear = "u -> u"

    def __init__(self, H, L, tol=None):
        from ..spaces.continuous import Continuous, ContinuousSpace
        from ..spaces.sobolev import SobolevSpace
        from ..spaces.base import L2, LpSpace
        lebesgue = (L2, LpSpace)
        if isinstance(H, lebesgue) and isinstance(L, ContinuousSpace):
            raise SpaceMismatch(f"there is no inclusion of {H} into {L}: its elements are classes of functions equal "
                                "almost everywhere, and most have no continuous member (a jump, 1/x^(1/3)); "
                                f"{L}.element(f) checks whether a given function is continuous")
        if isinstance(H, ContinuousSpace) and isinstance(L, lebesgue):
            self._route = "C->Lp"
        elif isinstance(H, SobolevSpace) and isinstance(L, ContinuousSpace):
            if not H.dual().integrals_bounded(0):            # the trace rule (spaces/dual.py)
                raise SpaceMismatch(f"{H} is not contained in {L}: for s <= d/2 = {H.domain.dim / 2:g} it has "
                                    "discontinuous and unbounded elements (the Sobolev embedding needs s > d/2)")
            self._route = "H->C"
        elif isinstance(H, SobolevSpace) and isinstance(L, L2):
            self._route = "H->L2"
        else:
            raise SpaceMismatch("Inclusion(A, B) is implemented for H^s into L2, C into L^p and H^s into C")
        if not H.domain.same_up_to_null_set(L.domain):
            raise SpaceMismatch(f"{H} and {L} live on different domains")
        if H.realisation != L.realisation:                 # the inclusion carries realisations over (decision Q4)
            raise RealisationMismatch(f"the inclusion of {H} into {L} carries its realisations over, and these are "
                                      f"realised {H.realisation} and {L.realisation}",
                                      needs=(f"one realisation for both: H.realised(r) or L.realised(r), r = "
                                             f"{H.realisation!r} or {L.realisation!r}",))
        if self._route == "C->Lp" and not isinstance(L.measure, LebesgueMeasure):
            raise Unsupported(f"the inclusion of {H} into L^p of a density measure is not implemented")
        super().__init__(H, L, name="ι")
        self._claims.append(Claim("linear", THEOREM, "the identity on functions"))
        if self._route == "C->Lp":
            self._claims.append(Claim(f"bounded, norm <= (b - a)^(1/p) = {L.domain.length ** (1 / getattr(L, 'p', 2)):g}",
                                      THEOREM, "||f||_p^p = int |f|^p dx <= (b - a) max |f|^p"))
            return
        if self._route == "H->C":                    # on an interval the fit at the space's own tolerance is cheap
            self.tol = float(tol) if tol is not None else L.tol
            self._fit_space = L if self.tol == L.tol else Continuous(L.domain, self.tol)
            kmax, exact = H.backend.diagonal_max()
            self._claims.append(Claim(f"bounded, norm <= sqrt(max K(x, x)) = {kmax ** 0.5:.6g}", THEOREM if exact else
                                      CHECKED, "|u(x)| = |<u, k_x>| <= ||u|| ||k_x||, ||k_x||^2 = K(x, x) (Sobolev "
                                      "embedding)" + ("; K(x, x) = k(0) (stationary kernel)" if exact else
                                                      "; the maximum of K(x, x) on 4097 points")))
        else:
            self.tol = float(tol) if tol is not None else (DEFAULT_TOL if L.tol <= 1e-14 else L.tol)
            self._fit_space = L if self.tol == L.tol else L2(L.measure, self.tol, L.realisation)
            self._claims.append(Claim("bounded, norm <= 1", THEOREM,
                                      "(1 + l^2 |xi|^2)^s >= 1; restriction is a contraction"))

    def _apply(self, u):
        L = self.codomain
        if self._route == "C->Lp":
            R = self.domain.backend.to_lp(u.realisation, getattr(L, "p", 2.0))
            return Function(L, pl=R, name=u.name) if isinstance(u, Function) else Family(L, R, name=u.name)
        R = self.domain.backend.to_l2(u.realisation) if self._route == "H->L2" else None
        if isinstance(u, Function):
            if R is None:                                      # fit in the target space at the inclusion's tolerance
                brk = self.domain.backend.breaks(u.realisation)           # kinks of representers (intervals)
                R = fits.refit(self._fit_space.element(lambda *X: u(*X), name=u.name, breaks=brk), u).realisation
            return Function(L, pl=R, name=u.name)
        if isinstance(u, Family):
            if R is not None:
                return Family(L, R, name=u.name)
            return L.family([self._apply(u[i]) for i in range(len(u))], name=u.name)
        raise TypeError(type(u).__name__)

    def _make_adjoint(self):
        raise Unsupported("the adjoint of the inclusion (a convolution with the kernel over the domain) is not "
                          "implemented")
