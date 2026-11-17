"""Pointwise convex sets of function spaces (plan 009 P5, G8; EXP-030 DESIGN G8 (b), NOTES section 8):

    B = PointwiseBox(X, lower=l, upper=u)          {m : l <= m <= u a.e.} in L2(mu) (on a direct sum: per summand)
    B = PointwiseBox(X, envelope=A, scale=a)       the weighted L-infinity ball {|m| <= a A a.e.}
    B = PointwiseBox(X, lower=0)                   the positivity cone
    B.support(q, rtol=None)                        an Enclosure of sigma_B(q) = int (u q+ - l q-) dmu: its lower and
                                                   upper ends certified (equal where the integral is exact); the value
                                                   is the upper end (plan 067 P3: support_bracket is gone)
    B.projected(P)                                 P(B) for a bounded operator P: sigma_{P B}(q) = sigma_B(P* q)

THEOREM (NOTES 8): B is closed and convex in L2(mu); bounded when l, u are in L2 (finite measure and finite bounds);
sigma_B(q) = int (u q+ - l q-) dmu, attained by u on {q > 0} and l on {q < 0} (+inf when the bound is infinite on a set
of positive measure where q has that sign). On a direct sum with weights w_k, sigma_B(q) = sum_k w_k sigma_{B_k}(q_k).
For any bounded P, sigma_{P B}(q) = sup <q, P m> = sigma_B(P* q); an orthogonal projection has P* = P.

The integral is the realisation's capability box_support (membership box_member; realisations/registry.py, the code
in realisations/bounds/box_integrals.py; plan 069 P3), per realisation (the claims say which):
    interval L2 (Lebesgue), piecewise Legendre    exact: the roots of q per piece, the bounds constant or elements
                                                  (piecewise polynomial) of the same space
    the sphere (harmonics)                        certified L1 bracket (realisations/bounds/sphere_l1.py), constant bounds;
                                                  it converges with rtol (local remainders, plan 061); times c R^2
                                                  for L2(c sigma) (plan 064)
    balls and shells (radial x harmonics)         certified radial x angular bracket of int A |q| r^2 dr dxi,
                                                  constant bounds per summand (split summands at envelope zones);
                                                  c rho(r) dx: the bracket of c |rho q| (rho >= 0, plan 064)
    R^n                                           exact

Varying bounds (elements of the space) by route. Support, bracket and membership: intervals and R^n; the sphere and
balls / shells refuse them (one constant per summand: split the space into zones where the envelope changes). The
feasible-set cell route (optimisers/dual/nonsmooth.py interval_cells, plan 057): intervals only, with per-cell bounds certified
(lo_c >= sup l, hi_c <= inf u: chord and curvature bounds, a Bernstein enclosure where they are loose; the cells
include the bounds' breaks); on the sphere and balls the cell route needs cells= from the caller even for constant
bounds, and varying bounds there are refused one layer lower, in the support itself. A function is not a bound: give
X(f), whose realisation is the bound.
"""
import numpy as np

from ..core.claims import DERIVED, THEOREM, Claim
from ..core.errors import Unsupported
from ..core.logic import Truth
from ..core.numbers import EXACT, Enclosure
from .sets import ConvexSet, _closed_convex
from ..spaces.identity import require_realised


def _bound(v, scale):
    """None (infinite), a float, or an element's realisation, times scale."""
    if v is None:
        return None
    if callable(v) and not hasattr(v, "space"):
        raise Unsupported("a bound given as a function: give it as an element of the space, X(f) (its realisation is "
                          "then the bound, and supports and cell bounds are exact on it); it is not sampled")
    if np.ndim(v) == 0 and not hasattr(v, "space"):
        return float(v) * scale
    return (v * scale) if scale != 1.0 else v


class PointwiseBox(ConvexSet):
    """{m : l <= m <= u a.e.} (module docstring). lower, upper, envelope: a number, an element of the space, None
    (no bound) or, on a direct sum, a dict {summand: bound} (missing summands are unbounded); envelope=A gives
    l = -A, u = A; scale multiplies every bound. rtol: the relative width that support() asks of a certified bracket
    where the integral is not exact (the sphere, balls and shells); support() returns the bracket as an Enclosure with
    its value at the upper end, so rtol is the precision of every bound computed from this box (plan 061)."""

    def __init__(self, space, lower=None, upper=None, envelope=None, scale=1.0, name=None, rtol=1e-8):
        from ..spaces.direct_sum import DirectSum
        self.scale = float(scale)
        if not rtol > 0:
            raise ValueError("rtol is a relative width > 0")
        self.rtol = float(rtol)
        if envelope is not None:
            if lower is not None or upper is not None:
                raise ValueError("give envelope= or lower=/upper=, not both")
            neg = {k: (None if v is None else -v) for k, v in envelope.items()} if isinstance(envelope, dict) \
                else (None if envelope is None else -envelope)
            lower, upper = neg, envelope
        self.summed = isinstance(space, DirectSum)
        keys = space.names if self.summed else [None]

        def per(spec):
            if isinstance(spec, dict):
                if not self.summed:
                    raise ValueError("a dict of bounds needs a direct sum")
                return [_bound(spec.get(k), self.scale) for k in keys]
            return [_bound(spec, self.scale) for _ in keys]
        self.lower, self.upper = per(lower), per(upper)
        for H, l, u in zip(space.summands if self.summed else [space], self.lower, self.upper):
            for b in (l, u):
                if hasattr(b, "space"):
                    require_realised(b.space, H, getattr(b, "name", "a bound"))    # read with H's backend (Q4)
        for l, u in zip(self.lower, self.upper):
            if isinstance(l, float) and isinstance(u, float) and l > u:
                raise ValueError("a box needs lower <= upper")
        bounded = all(l is not None and u is not None for l, u in zip(self.lower, self.upper))
        super().__init__(space, name or ("box" if bounded else "pointwise cone"),
                         _closed_convex("an intersection of closed half-spaces {m : <m, 1_E> <= int_E u} (NOTES 8)",
                                        bounded)
                         + [Claim("support int (u q+ - l q-) dmu, attained by u 1{q>0} + l 1{q<0}", THEOREM,
                                  "NOTES 8 (EXP-030)", key="support")])

    # ------------------------------------------------------------------------------------------------ support
    def _parts(self, q):
        """[(space_k, weight_k, realisation of q_k or None, l_k, u_k)] (one member)."""
        X = self.space
        if hasattr(q, "space"):
            require_realised(q.space, X, getattr(q, "name", "the direction"))     # read with X's backend (Q4)
        pl = q.realisation if hasattr(q, "realisation") else None
        if not self.summed:
            return [(X, 1.0, pl if pl is not None else np.asarray(getattr(q, "values", q), float), self.lower[0],
                     self.upper[0])]
        out = []
        for k, H in enumerate(X.summands):
            part = pl.parts[k]
            R = None if part is None else part[0]
            out.append((H, X.weights[k], R, self.lower[k], self.upper[k]))
        return out

    def support(self, q, rtol=None):
        """The Enclosure of sigma_B(q) with certified lower and upper ends, equal where the integral is exact (module
        docstring); its value is the upper end. rtol: the relative width asked of a certified bracket (default the
        box's)."""
        rtol = self.rtol if rtol is None else rtol
        lo = hi = 0.0
        for H, w, R, l, u in self._parts(q):
            if R is None:
                continue
            a, b = part_bracket(H, R, l, u, rtol)
            lo, hi = lo + w * a, hi + w * b
        return _enclosure(lo, hi)

    def member(self, m):
        verdict = Truth.TRUE
        for H, w, R, l, u in self._parts(m):
            t = part_member(H, R, l, u)
            if t is Truth.FALSE:
                return t
            if t is Truth.UNKNOWN:
                verdict = Truth.UNKNOWN
        return verdict

    def is_empty(self):
        return Truth.FALSE

    def projected(self, P, name=None):
        return ProjectedBox(self, P, name)


class ProjectedBox(ConvexSet):
    """P(B) for a pointwise box B and a bounded operator P (module docstring): sigma(q) = sigma_B(P* q)."""

    def __init__(self, box, P, name=None):
        self.box, self.P = box, P
        cl = [Claim("convex", DERIVED, "the image of a convex set under a linear map", key="convex"),
              Claim("support sigma_B(P* q)", THEOREM, "<q, P m> = <P* q, m>", key="support")]
        if box.claims.status("bounded")[0] is Truth.TRUE:
            cl.append(Claim("bounded", DERIVED, "the image of a bounded set under a bounded map", key="bounded"))
        super().__init__(P.codomain, name or f"P({box.name})", cl)

    def support(self, q, rtol=None):
        return self.box.support(self.P.adjoint(q), rtol)

    def is_empty(self):
        return Truth.FALSE


def _enclosure(lo, hi):
    """The support's Enclosure: [lo, hi] certified (exact integrals give lo == hi), value hi."""
    how = EXACT if lo == hi else None
    return Enclosure(hi, lo, hi, DERIVED, "int (u q+ - l q-) dmu: exact per piece, or a certified L1 bracket "
                     "(module docstring)", how=how)


# ------------------------------------------------------------------------------------------------ per realisation
# The integrals are capabilities of the realisation (realisations/registry.py: box_support, box_member; the code in
# realisations/bounds/box_integrals.py); R^n summands are computed here.
def _euclidean(H):
    from ..spaces.base import Euclidean
    return isinstance(H, Euclidean)


def _supported(H, R):
    from ..realisations.registry import require
    return require(R, "box_support", f"pointwise boxes in {H}: no L1 integral for its realisation")


def part_bracket(H, R, l, u, rtol=1e-8):
    """(lower, upper) of int (u q+ - l q-) dmu for one summand H, q realised by R (module docstring)."""
    if _euclidean(H):
        v = H.apply_metric(np.asarray(R, float).ravel())
        lo = -np.inf if l is None else np.broadcast_to(np.asarray(getattr(l, "values", l), float), v.shape)
        hi = np.inf if u is None else np.broadcast_to(np.asarray(getattr(u, "values", u), float), v.shape)
        with np.errstate(invalid="ignore"):
            t = np.where(v > 0, v * hi, np.where(v < 0, v * lo, 0.0))
        s = float(np.sum(t))
        return s, s
    return _supported(H, R).box_support(H, l, u, rtol)


def part_member(H, R, l, u):
    """Truth of l <= m <= u a.e. for one summand (exact for interval and R^n realisations, otherwise from the
    realisation's sup bound when that decides, else UNKNOWN)."""
    if R is None:
        return Truth.TRUE if (l is None or (isinstance(l, float) and l <= 0)) and \
            (u is None or (isinstance(u, float) and u >= 0)) else Truth.UNKNOWN
    if _euclidean(H):
        v = np.asarray(R, float).ravel()
        ok = (l is None or np.all(v >= np.asarray(getattr(l, "values", l)) - 1e-12 * max(1.0, np.abs(v).max()))) and \
             (u is None or np.all(v <= np.asarray(getattr(u, "values", u)) + 1e-12 * max(1.0, np.abs(v).max())))
        return Truth.of(bool(ok))
    from ..realisations.registry import provides
    if provides(_supported(H, R), "box_member"):
        return R.box_member(H, l, u)
    if l is not None and u is not None and isinstance(l, float) and isinstance(u, float):
        sb = getattr(R, "sup_bound", None)
        if callable(sb) and float(np.max(sb())) <= min(-l, u):
            return Truth.TRUE
    return Truth.UNKNOWN
