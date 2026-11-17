"""Lebesgue (Hausdorff) measure on a Region and the measures with a density against it (EXP-031): LebesgueMeasure,
DensityMeasure (w dsigma, w a Weight: measures/weights.py), PieceDensity (w_j ds on the segments of a polyline), and the
rules for the measure of a function space (check_space_measure, constant_density, space_factor). Plan 067 P5: moved
from core/measures.py into the Measure root's package.
"""
import math

import numpy as np

from ..core.claims import THEOREM, Claim
from ..core.logic import Truth
from ..sets import EmptySet, FiniteSet, Interval, Region
from .base import Measure, _as_interval

TWO_ORDERS = (32, 24)        # Gauss orders of the two quadratures whose difference is the stated error


def carries_hausdorff(S):
    """Whether S is a Region that carries its Hausdorff measure here (plan 067 P2: the Region test replaces the tuple
    (Interval, Domain, Simplicial, Surface)): a dimension >= 1 that the region states (finite sets carry the counting
    measure; regions known by membership only state no dimension) and charts through which that measure is integrated
    (plan 069 P2: the rule is the numerical column's, pygeoinf0.quadrature.rule; R, a LayeredBall or a Partition's
    union, which state none, are refused as before)."""
    return isinstance(S, Region) and S.dim is not None and S.dim >= 1 and S.charts() is not None


class LebesgueMeasure(Measure):
    """Lebesgue measure on an interval or on a domain (Box, Disk, Cylinder, Ball):
    LebesgueMeasure(Interval("[0, 1]")), LebesgueMeasure("[0, 1]"), LebesgueMeasure(Ball(radius=1)).
    On a ball, in spherical coordinates, it is r^2 sin(theta) dr dtheta dphi; on a disk or cylinder r dr dphi (dz)."""

    sigma_algebra = "Lebesgue-measurable subsets"

    def __init__(self, domain):
        domain = _as_interval(domain)
        if not carries_hausdorff(domain):
            raise TypeError("Lebesgue (Hausdorff) measure is defined here on a Region of stated dimension >= 1 with a "
                            "quadrature: an Interval, a Domain (Box, Disk, Cylinder, Ball, paths, surfaces), a Simplicial "
                            "set, a parametric Surface (a finite set carries the counting measure, pg.CountingMeasure)")
        self.set = domain

    def __call__(self, A):
        A = _as_interval(A)
        if isinstance(A, FiniteSet):
            return 0.0
        if isinstance(A, Interval) and isinstance(self.set, Interval):
            J = A & self.set                        # an Interval, a point or empty: H^1 of it
            return 0.0 if J.dim == 0 or isinstance(J, EmptySet) else J.volume
        if A is self.set or A == self.set:
            return self.set.volume
        raise TypeError(f"cannot measure {A!r} with {self!r}")

    def mass(self):
        """(|S|, 0.0): the Hausdorff measure of the set (exact)."""
        return float(self.set.volume), 0.0

    def integrate(self, f):
        """(int_S f dsigma, the difference of two Gauss orders) for f a function of the Cartesian coordinates, f(x) on an
        interval, f(x, y) or f(x, y, z) (vectorised or not, as for weights)."""
        from .weights import _call, integrate_on
        hi, lo = (integrate_on(self.set, lambda X: _call(f, X), n) for n in TWO_ORDERS)
        return hi, float(np.max(np.abs(np.asarray(hi) - lo)))

    def with_density(self, w, breaks=None, name="w", **kw):
        """The measure w dsigma on the same set (an interval, a domain used as a sub-domain, a path, a surface): w a
        number, a callable of the Cartesian coordinates, an element of a space on the set, or one value per piece; on a
        polyline an array is the piecewise-constant PieceDensity (w_j >= 0 ds on the j-th segment; negative values are
        refused there as before: signed piece values go through pg.DensityMeasure(pg.LebesgueMeasure(polyline), w)).
        Signed weights give a finite signed measure (DensityMeasure, said in its claims). Keywords singular_at=,
        exponent=, sup= declare the weight's behaviour (measures/weights.py); radial=True on a ball or shell: w is a
        callable of the distance r from the centre (plan 064; L2 of it is realised in radial x harmonic factors)."""
        from ..domains import Polyline
        if isinstance(self.set, Polyline) and not callable(w) and np.ndim(w) == 1 and not kw:
            return PieceDensity(self.set, w, name=name)
        return DensityMeasure(self, w, breaks=breaks, name=name, **kw)

    def normalised(self):
        """1/|S| times this measure: the average over its set (a DensityMeasure with a constant weight)."""
        return DensityMeasure(self, 1.0 / self.mass()[0], name=f"1/|{self.set}|")

    def same_up_to_null_set(self, other):
        return isinstance(other, LebesgueMeasure) and type(other.set) is type(self.set) \
            and self.set.same_up_to_null_set(other.set)

    def density_against(self, base):
        """d(self)/d(base): the interface of against= (plan 009 G13, EXP-031 DESIGN 8), a measures.densities.Density with
        its local orders where it vanishes or blows up and .weight, the derivative as a Weight. Refused
        (UnboundedFunctional) when self lives on a set of lower dimension than base's."""
        from .densities import density_against
        return density_against(self, base)

    def __eq__(self, other):
        return isinstance(other, LebesgueMeasure) and self.set == other.set

    def __hash__(self):
        return hash(("Lebesgue", self.set))

    def __repr__(self):
        return f"Lebesgue measure on {self.set}"


class DensityMeasure(Measure):
    """w dsigma: a measure with density w with respect to the Lebesgue (Hausdorff) measure sigma of a set S, an
    interval, a domain used as a sub-domain (Box, Disk, Ball, ...), a path or a surface (EXP-031, DESIGN 3.1):

        pg.LebesgueMeasure(S).with_density(w)      pg.DensityMeasure(pg.LebesgueMeasure(S), w)     (the same object)

    w is a Weight (measures/weights.py: a number, a callable of the Cartesian coordinates, one value per piece, an element
    of a space on S). A signed w gives a finite signed measure (claims say which); function spaces accept only a
    non-negative density as their own measure. The measure holds its weight (and the breaks given); constructing it fits
    nothing (plan 069 P2, review C2). On an interval self.density is the weight's piecewise Legendre series, made on
    first use by the numerical column (realisations/series/legendre.fit_density; RealisationError then if it is not resolved)
    and kept; elsewhere, and for a declared singular weight, it is None."""

    sigma_algebra = "Lebesgue-measurable subsets"

    def __init__(self, base, w, breaks=None, name="w", singular_at=None, exponent=None, sup=None, at=None,
                 radial=False):
        from .weights import as_weight
        if not isinstance(base, LebesgueMeasure):
            raise TypeError("densities are taken with respect to Lebesgue (Hausdorff) measure on a set: "
                            "pg.LebesgueMeasure(S).with_density(w)")
        S = base.set
        if isinstance(S, Interval) and not S.is_bounded:
            raise TypeError("densities are supported with respect to Lebesgue measure on a bounded interval")
        self.base, self.set, self.name = base, S, name
        self.w, self.breaks = w, tuple(breaks or ())
        self.weight = as_weight(w, S, at=at, singular_at=singular_at, exponent=exponent, sup=sup, radial=radial)
        self._density = None
        self.weight.sign()                              # sampled now, so that a non-finite weight is refused here

    @property
    def density(self):
        """The piecewise Legendre series of the weight on an interval, fitted on first use by the numerical column
        (realisations/series/legendre.fit_density) and kept; None on other sets and for a declared singular weight."""
        if self._density is None and isinstance(self.set, Interval) and not self.weight.singular:
            from ..realisations.series.legendre import fit_density
            self._density = fit_density(self.weight, self.set, self.breaks, self.name)
        return self._density

    @property
    def signed(self):
        """True unless the weight is non-negative (then this is a measure, not only a finite signed measure)."""
        return not self.weight.nonnegative

    def mass(self):
        """(int w dsigma, estimated error): exact for constant and per-piece weights; for interval densities the
        integral of their piecewise Legendre fit p, with the fit's error |int (w - p)| <= sqrt(b - a) ||w - p||_L2
        (Cauchy-Schwarz, the fit's L2 estimate; plan 065 A10: it was stated 0.0); else Gauss quadrature on S at two
        orders."""
        from .weights import integrate_on, size
        wt = self.weight
        if wt.kind == "constant" or (wt.kind == "pieces" and hasattr(self.set, "_L")):
            return (wt.c * size(self.set) if wt.kind == "constant" else float(wt.c @ self.set._L)), 0.0
        if self.density is not None:
            from ..realisations.series import legendre as L
            S = self.set
            return float(L.integral(self.density)[0]), math.sqrt(S.b - S.a) * float(self.density.err[0])
        pts = () if wt.singular_at is None else [p[0] for p in wt.singular_at] if isinstance(self.set, Interval) else ()
        hi, lo = (integrate_on(self.set, wt.values, n, points=pts, exponent=wt.exponent) for n in TWO_ORDERS)
        return hi, abs(hi - lo)

    def integrate(self, f):
        """(int f w dsigma, the difference of two Gauss orders) for f a function of the Cartesian coordinates (as for
        LebesgueMeasure.integrate), graded towards the weight's declared singular points on an interval."""
        from .weights import _call, integrate_on
        wt = self.weight
        pts = () if wt.singular_at is None else [p[0] for p in wt.singular_at] if isinstance(self.set, Interval) else ()

        def g(X):
            return wt.values(X) * _call(f, X)
        hi, lo = (integrate_on(self.set, g, n, points=pts, exponent=wt.exponent) for n in TWO_ORDERS)
        return hi, float(np.max(np.abs(np.asarray(hi) - lo)))

    def _sign(self):
        """From the weight's sign: proved for exact kinds, CHECKED on samples for callables, or declared."""
        word, status, reason = self.weight.sign()
        holds = {"non-negative": Truth.FALSE, "zero": Truth.FALSE, "signed": Truth.TRUE,
                 "non-positive": Truth.TRUE}.get(word, Truth.UNKNOWN)
        return holds, status, f"the density {self.name} is {word} ({reason})"

    def variation(self):
        """|mu|(S) = int |w| dsigma (the norm of the functional on C(K)); equal to the mass for non-negative weights."""
        from .weights import integrate_on
        if not self.signed:
            return self.mass()[0]
        wt = self.weight
        if wt.kind in ("constant", "pieces"):
            return DensityMeasure(self.base, np.abs(wt.c) if wt.kind == "pieces" else abs(wt.c)).mass()[0]
        brk = ()
        if isinstance(self.set, Interval):               # break at the sign changes of the weight (bisection)
            t = np.linspace(self.set.a, self.set.b, 4097)
            v = wt.values(t[:, None])
            j = np.flatnonzero(np.sign(v[1:]) != np.sign(v[:-1]))
            lo, hi, slo = t[j], t[j + 1], np.sign(v[j])
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                left = np.sign(wt.values(mid[:, None])) == slo
                lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
            brk = 0.5 * (lo + hi)
        # elsewhere |w| has a kink along the zero set of w: Gauss converges algebraically there (C(K) exists only on
        # intervals, so this is a diagnostic, not a norm the package uses)
        return integrate_on(self.set, lambda X: np.abs(wt.values(X)), 64, breaks=brk)

    def normalised(self):
        """w / int w dsigma: the weighted average over S (refused when the total mass is zero)."""
        from ..core.errors import Refused
        Z, err = self.mass()
        if Z == 0 or abs(Z) <= max(10 * err, 1e-12 * (self.variation() if self.signed else abs(Z))):
            raise Refused(f"the total mass of {self} is zero (to its estimated error {err:.1g}), so it has no "
                          "normalisation", needs=("a weight with non-zero integral",))
        out = DensityMeasure(self.base, self.weight.scaled(1.0 / Z), name=f"{self.name}/{Z:.6g}")
        out._mass_estimate = err / abs(Z)
        return out

    def density_against(self, base):
        """d(self)/d(base): the interface of against= (plan 009 G13, EXP-031 DESIGN 8), a measures.densities.Density with
        its local orders where it vanishes or blows up and .weight, the derivative as a Weight. Refused
        (UnboundedFunctional) when self lives on a set of lower dimension than base's."""
        from .densities import density_against
        return density_against(self, base)

    def __call__(self, A):
        A = _as_interval(A)
        if isinstance(A, FiniteSet):
            return 0.0
        if A is self.set or A == self.set:
            return self.mass()[0]
        if not (isinstance(A, Interval) and self.density is not None):
            raise TypeError(f"cannot measure {A!r} with {self!r}")
        from ..realisations.series import legendre as L
        J = A & self.set
        return 0.0 if J is None else float(L.integral(self.density, J.a, J.b)[0])

    def _own_claims(self):
        word, status, reason = self.weight.sign()
        pos = word in ("non-negative", "zero")
        out = [Claim("a measure" if pos else "a finite signed measure", status,
                     f"the density {self.name} is {word} ({reason})", key="measure"),
               Claim(f"{self.name} >= 0", status, reason, holds=pos)]
        out += [c for c in self.weight.claims if c.key != "sign"]
        if getattr(self, "_mass_estimate", 0):
            out.append(Claim("normalised: total mass 1", "checked",
                             f"divided by a mass computed by quadrature (relative estimate {self._mass_estimate:.1g})"))
        return out

    def _same_weight(self, other):
        """The same density: the same object, or two constants of equal value (plan 064: L2(c dsigma) built twice is
        one space)."""
        if other.w is self.w:
            return True
        a, b = self.weight, other.weight
        return a.kind == "constant" and b.kind == "constant" and a.c == b.c

    def same_up_to_null_set(self, other):
        return isinstance(other, DensityMeasure) and self._same_weight(other) and self.set.same_up_to_null_set(other.set)

    def __eq__(self, other):
        return isinstance(other, DensityMeasure) and self._same_weight(other) and self.set == other.set

    def __hash__(self):
        key = ("c", self.weight.c) if self.weight.kind == "constant" else id(self.w)
        return hash(("density", key, self.set))

    def _label(self):
        """The density's name; a constant left with the default name shows its value (plan 064)."""
        if self.weight.kind == "constant" and self.name == "w":
            return f"{self.weight.c:.6g}"
        return self.name + ("(r)" if self.weight.kind == "radial" else "")

    def __repr__(self):
        if isinstance(self.set, Interval):
            return f"{self.name}(x) dx on {self.set}"
        return f"{self._label()} dσ on {self.set}" if self.set.dim < getattr(self.set, "cartesian_dim", self.set.dim) \
            else f"{self._label()} dx on {self.set}"


class PieceDensity(DensityMeasure):
    """w_j ds on the j-th segment of a polyline: arc length with a piecewise-constant density w >= 0 (one value per
    piece), e.g. the slowness-squared weights of a linearised travel time along a ray. A DensityMeasure whose weight is
    the per-piece kind; negative values are refused here (signed piece values: pg.DensityMeasure)."""

    sigma_algebra = "Borel subsets"

    def __init__(self, polyline, w, name="w"):
        from .weights import PieceWeight
        w = np.asarray(w, float)
        if w.shape != (len(polyline._L),):
            raise ValueError(f"a piecewise density on {polyline} needs one value per piece ({len(polyline._L)}), "
                             f"not shape {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError(f"the density {name} must be finite and non-negative, so w ds is a measure")
        self.set, self.weights, self.name = polyline, w, name
        self.base, self.w, self._density = LebesgueMeasure(polyline), w, None
        self.weight = PieceWeight(polyline, w)

    def __call__(self, A):
        if A is self.set or A == self.set:
            return float(self.weights @ self.set._L)
        raise TypeError(f"cannot measure {A!r} with {self!r}")

    def _own_claims(self):
        return [Claim("a measure", THEOREM, "a non-negative density with respect to arc length")]

    def __repr__(self):
        return f"{self.name} ds on {self.set} (piecewise constant)"


def check_space_measure(mu):
    """The measure of a function space: Lebesgue measure; a non-negative density on an interval; a constant density
    c > 0 on any other set with an L2 realisation (plan 064 stage 1: L2(c sigma) is L2(sigma) with every pairing times
    c); a radial density rho(r) >= 0 on a ball or shell (plan 064 stage 2, realisation "radial_harmonics"). Other
    densities on 2-D/3-D sets (a full Gram in the realisation) are refused, naming that remaining scope."""
    from ..core.errors import Unsupported
    if not isinstance(mu, DensityMeasure):
        return
    if mu.signed:
        raise ValueError(f"the density {mu.name} takes negative values, so w(x) dx is not a measure")
    if isinstance(mu.set, Interval):
        return
    kind = mu.weight.kind
    if kind == "constant":
        if not mu.weight.c > 0:
            raise ValueError(f"the constant density {mu.weight.c:g} is not positive: L2 of the zero measure is not a "
                             "Hilbert space of functions (every norm vanishes)")
        return
    from ..domains import Ball, Shell
    if kind == "radial" and isinstance(mu.set, (Ball, Shell)):
        return
    raise Unsupported(f"function spaces with the density {mu.name} ({kind}) as their measure on {mu.set!r}: plan 064 "
                      "realises constant densities on every domain and radial densities rho(r) on balls and shells "
                      "(with_density(f, radial=True)); other densities on 2-D/3-D domains (a full Gram in the "
                      "realisation) remain its refused stage 3; a density that is not one of these can be written "
                      "into the functionals instead")


def constant_density(mu):
    """c when mu = c sigma with sigma Lebesgue (Hausdorff) measure on its set (1.0 for Lebesgue measure itself), else
    None: the factor by which a realisation of L2(sigma) realises L2(mu) (plan 064)."""
    if isinstance(mu, LebesgueMeasure):
        return 1.0
    if isinstance(mu, DensityMeasure) and mu.weight.kind == "constant":
        return float(mu.weight.c)
    return None


def space_factor(H):
    """The constant density of the measure of a function space H on a set other than an interval (plan 064): the
    factor its backend applies on top of the realisation of L2(sigma); 1 for Lebesgue measure, for spaces without a
    measure (R^n) and on intervals (whose densities are realised in the backend itself). None for a non-constant
    density (a radial density on a ball or shell): callers that rebuild the measure by hand refuse it or apply it."""
    mu = getattr(H, "measure", None)
    if mu is None or isinstance(getattr(mu, "set", None), Interval):
        return 1.0
    return constant_density(mu)


