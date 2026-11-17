"""C([a, b]): the continuous functions on a closed bounded interval, with the sup norm ||f|| = max |f(x)|.

A Banach space whose elements are functions, not classes: f(x) is the value of f at x. Point evaluation f -> f(x0) is
bounded with norm 1, and so is every integral f -> int f dmu against a finite measure mu, with norm mu([a, b]). By the
Riesz-Markov theorem the finite signed measures are the whole dual space, so a functional on C is a measure, not a
Riesz representer. C has no inner product: the parallelogram law fails (f = 1, g = x on [0, 1] give
||f + g||^2 + ||f - g||^2 = 5, while 2 ||f||^2 + 2 ||g||^2 = 4). So an operator G: C -> R^n has no adjoint into C (G*
maps R^n into the measures), there is no G G* and no least squares in C. pg.Inclusion(X, pg.L2(I)) and, for s > 1/2,
pg.Inclusion(pg.Sobolev(I, s, length_scale=l), X) (the Sobolev embedding) connect C with the Hilbert spaces.

Elements are realised as in L2 (piecewise Legendre series), with the differences listed in realisations/series/sup.py: a
function whose realisation jumps is refused (NotContinuous; continuity is checked on the realisation, it cannot be
proved from a callable), and errors and norms are in the sup norm: X.norm(f) is the maximum of the realised
polynomial (to rounding), f.error the estimated sup distance between f and its realisation.
"""
import numpy as np

from ..core.claims import CHECKED, THEOREM, Claim, Claims
from .elements import Function
from ..core.errors import SupportOutsideDomain, Unsupported
from ..measures import CountingMeasure, DensityMeasure, DiracMeasure, LebesgueMeasure, SignedMeasure
from ..sets import Interval
from .base import BanachSpace, _FunctionSpace


class ContinuousSpace(_FunctionSpace, BanachSpace):
    """C([a, b]) with the sup norm (see the module docstring); build it with Continuous("[a, b]")."""

    has_classes = False             # elements are functions (Function.info, Function.redefine_at read this)
    membership_hint = "If the function is not continuous on the interval it is not an element of {space}"

    def __init__(self, interval, tol=1e-14):
        I = Interval(interval) if isinstance(interval, str) else interval
        if not isinstance(I, Interval):
            raise Unsupported(f"C(K) is implemented on closed bounded intervals, and {I!r} is not one")
        if not I.is_bounded or not (I.left_closed and I.right_closed):
            raise ValueError(f"C({I}) is not a normed space: on an interval that is not closed and bounded a continuous "
                             f"function can be unbounded (1/x on (0, 1]). Use the closed interval "
                             f"[{I.a:g}, {I.b:g}]" + ("" if I.is_bounded else " (bounded)"))
        self._measure_given = False
        self.measure = LebesgueMeasure(I)             # only for X.integrate(f) = int_a^b f(x) dx
        self.domain, self.tol = I, tol
        from ..realisations import choose
        self.backend = choose(self.measure, tol, sup=True)
        self.realisation = self.backend.realisation_name

    # elements: realised at once, so that a discontinuous function is refused where it is given ----------------
    def element(self, f, *, breaks=None, name=None, resolution=None):
        u = super().element(f, breaks=breaks, name=name, resolution=resolution)
        u.realisation
        return u

    # geometry -----------------------------------------------------------------------------------------------
    def norm(self, f):
        """max |f(x)| over [a, b]: of the realisation, to rounding (f.error is its estimated distance from f)."""
        r = self.backend.sup_norms(self._pl_of(f))
        return float(r[0]) if isinstance(f, Function) else r

    # no inner and no riesz: the sup norm fails the parallelogram law (f = 1, g = x on [0, 1]: 5 against 4), so C has
    # no inner product and no Riesz map; its dual is the finite signed measures (Riesz-Markov) and a functional on C is
    # the measure itself: X.functional(mu), X.evaluation(x0), X.integrals([mu_1, ...]); pg.Inclusion(X, pg.L2(I))
    # gives the L2 inner product of continuous functions
    def _dual_claims(self):
        return [Claim("isometric to the finite signed Borel measures on the interval, with the total variation norm",
                      THEOREM, "Riesz-Markov")]

    def random(self, *args, **kw):
        raise Unsupported("random elements of C need a stated Gaussian measure on C (for instance Wiener measure); "
                          "none is implemented yet")

    # functionals: measures ----------------------------------------------------------------------------------
    def integrate_against(self, f, measures):
        """(len(measures),) for an element, (len(measures), N) for a family: int f dmu_i, for points (Dirac
        measures), subintervals (Lebesgue measure), weighted points (CountingMeasure), densities on the interval and
        signed combinations of these (termwise)."""
        R = self._pl_of(f)
        B, I = self.backend, self.domain
        out = []
        for mu in measures:
            if isinstance(mu, SignedMeasure):                 # termwise (the split of a measure over pieces)
                out.append(sum(c * np.reshape(self.integrate_against(f, [m]), -1) for c, m in mu.terms))
            elif isinstance(mu, DiracMeasure):
                x = np.atleast_1d(mu.x0)
                self._inside(x, mu)
                out.append(B.evaluate(R, x)[0])
            elif isinstance(mu, CountingMeasure):
                x = np.asarray(mu.set.points, float).ravel()
                self._inside(x, mu)
                out.append(np.asarray(mu.weights, float) @ B.evaluate(R, x))
            elif isinstance(mu, LebesgueMeasure) and isinstance(mu.set, Interval):
                J = mu.set
                self._inside(np.array([J.a, J.b]), mu)
                out.append(B.integral(R, J))
            elif isinstance(mu, DensityMeasure) and mu.set.same_up_to_null_set(I) and mu.density is not None:
                out.append(B.integral(B.multiply(R, mu.density), None))
            elif isinstance(mu, DensityMeasure) and isinstance(mu.set, Interval):
                out.append(self._density_integral(R, mu))
            else:
                raise Unsupported(f"integrals against {mu} on {self} are not implemented (points, subintervals, "
                                  "weighted points and densities on subintervals are)")
        v = np.array(out, float)
        return v[:, 0] if isinstance(f, Function) else v

    def _density_integral(self, R, mu):
        """int_J f w dx for a density on a subinterval J: Gauss-Legendre with 128 points on every piece common to the
        realisation of f and the density (exact for their polynomial pieces up to degree 255), graded towards
        declared singular points of w (an integrable singularity: the boundedness decision checks w in L^1)."""
        from .dual import require_bounded
        from ..measures.weights import integrate_on
        J = mu.set
        self._inside(np.array([J.a, J.b]), mu)
        require_bounded(mu, self)
        wt, B = mu.weight, self.backend
        brk = list(np.ravel(B.breaks(R))) + ([] if mu.density is None else list(mu.density.breaks))
        pts = [] if wt.singular_at is None else [float(p[0]) for p in wt.singular_at]
        return integrate_on(J, lambda X: B.evaluate(R, X[:, 0]).reshape(len(X), -1) * wt.values(X)[:, None], 128,
                            points=pts, breaks=brk, exponent=wt.exponent)

    def _inside(self, x, mu):
        I = self.domain
        if np.any(x < I.a) or np.any(x > I.b):
            raise SupportOutsideDomain(f"{mu} is not supported in {I}")

    def functional(self, mu, scale=1.0, name=None):
        """The bounded linear functional f -> scale * int f dmu (mu a measure, a point or a subinterval): a Functional
        of the dual, the finite signed measures (spaces/dual.py)."""
        from .dual import Functional
        return Functional.from_measure(self, mu, scale, name)

    def evaluation(self, x0):
        """f -> f(x0): bounded, with norm 1."""
        return self.functional(float(x0), name=f"evaluation at {float(x0):g}")

    def average(self, J):
        """f -> (1/|J|) int_J f dx over a subinterval J."""
        J = Interval(J) if isinstance(J, str) else J
        return self.functional(LebesgueMeasure(J), 1.0 / J.length, name=f"average over {J}")

    def split_functional(self, partition, mu, side=None):
        """{piece: f -> int f d(mu|K_k) on C(K_k)} over the pieces of a partition that mu charges, a seam mass once
        (side=); called on u, sum_k l_k(R_k u) = int u dmu (Partition.functionals; operators/pieces_continuous.py)."""
        from ..operators.pieces_continuous import functionals
        return functionals(self, partition, mu, side)

    def integrals(self, measures, into=None, name="G"):
        """G f = (int f dmu_1, ..., int f dmu_n): pg.Integrals(X, measures, into)."""
        from ..operators.functionals import Integrals
        return Integrals(self, measures, into=into, name=name)

    # identity -----------------------------------------------------------------------------------------------
    @property
    def claims(self):
        return Claims([
            Claim("Banach space (complete)", THEOREM, "a uniform limit of continuous functions is continuous"),
            Claim("not a Hilbert space", THEOREM, "the parallelogram law fails: f = 1, g = x on [0, 1]"),
            Claim("separable", THEOREM, "polynomials are dense (Weierstrass)"),
            Claim("dual space: the finite signed Borel measures", THEOREM, "Riesz-Markov"),
            Claim("point evaluation bounded, norm 1", THEOREM, "|f(x0)| <= max |f|"),
            Claim("elements are continuous", CHECKED, "on their realisations: no gap at a breakpoint above 1e-6 of the "
                                                      "function's size and 10 local errors; a callable cannot be "
                                                      "proved continuous"),
            Claim("norm: the maximum of the realisation", CHECKED, "a grid in arccos t refined by bisection on p' "
                                                                   "(Bernstein's inequality); plus f.error")])

    def __eq__(self, other):
        return isinstance(other, ContinuousSpace) and other.domain == self.domain

    def __hash__(self):
        return hash(("C", repr(self.domain)))

    def __repr__(self):
        return f"C({self.domain})"


def Continuous(interval, tol=1e-14):
    """C([a, b]): continuous functions on a closed bounded interval with the sup norm (spaces/continuous.py)."""
    return ContinuousSpace(interval, tol)
