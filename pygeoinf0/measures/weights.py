"""Weights: the densities w of weighted measures w dsigma (EXP-031, DESIGN 3.2). One protocol, built by as_weight from
what users have, and shared with the weighted functionals of plan 009 (G6, G13):

    given w                                         kind        exact   what a realisation sees on piece j
    a number c                                      constant    yes     c
    an array, one value per piece of the set        pieces      yes     c_j (segments of a polyline, cells)
    an element of a function space on the set       function    yes*    its realisation (*exact for the realisation)
    a callable of the Cartesian coordinates         callable    no      samples (or a fit, with its error)
    a callable of r, radial=True (ball or shell)    radial      no      samples of r; a radial fit (plan 064)
    pg.Tabulated (plan 009, G10) on an interval     tabulated   yes     its linear pieces (breakpoints become breaks)

A weight knows its set, its sign (proved for exact kinds, CHECKED on samples for callables, or declared) and its
integrability class: bounded (assumed for callables: ASSERTED, with a sampled maximum CHECKED) or singular as declared,
singular_at=points with exponent=a meaning |w(x)| ~ dist(x, points)^(-a) near the points (a < 0: w vanishes there
to order -a). spaces/dual.py decides from this class which spaces the measure w dsigma is bounded on.
Weights are continuum objects: no basis, mesh or quadrature is attached to them; samples are taken only for checks.
"""
import numpy as np

from ..core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, Claim, Claims

SAMPLES_1D = 4097            # points at which a callable is sampled on an interval (the check DensityMeasure had)
SAMPLES_PER_AXIS = 24        # Gauss points per axis on other sets


def sample_points(S, n=None):
    """Sample points (m, d) of a set (Cartesian coordinates); None when the set offers no sampling."""
    from ..sets import Interval
    if isinstance(S, Interval):
        return np.linspace(S.a, S.b, n or SAMPLES_1D)[:, None]
    if not callable(getattr(S, "charts", None)) or S.charts() is None:
        return None
    from ..quadrature import rule
    X, _ = rule(S, n or SAMPLES_PER_AXIS)
    return np.asarray(X, float)


def _call(f, X):
    """f at the points X (m, d), vectorised or not, as a float array (m,)."""
    cols = [X[:, k] for k in range(X.shape[1])]
    try:
        v = np.asarray(f(*cols), float)
    except (TypeError, ValueError):
        v = None
    if v is None or v.shape != (len(X),):
        v = np.broadcast_to(np.asarray(v, float), (len(X),)) if v is not None and v.ndim == 0 else \
            np.array([float(f(*p)) for p in X])
    return v


class Weight:
    """A real weight on a set (see the module docstring). values(X) gives w at Cartesian points X (m, d)."""

    kind, exact = None, False

    def __init__(self, S, singular_at=None, exponent=None, sup=None):
        self.set = S
        self.singular_at = None if singular_at is None else np.atleast_2d(np.asarray(singular_at, float))
        if self.singular_at is not None and self.singular_at.shape[0] == 1 and self.singular_at.shape[1] > 1 \
                and getattr(S, "cartesian_dim", 1) == 1:
            self.singular_at = self.singular_at.T                    # points of an interval given as a list
        if (singular_at is None) != (exponent is None):
            raise ValueError("a declared singular behaviour needs both singular_at=points and exponent=a "
                             "(|w(x)| ~ dist(x, points)^(-a))")
        self.exponent = None if exponent is None else float(exponent)
        self.sup = None if sup is None else float(sup)          # a declared bound sup |w| (ASSERTED)
        self._sign = None

    # ---------------------------------------------------------------- values
    def values(self, X):
        raise NotImplementedError

    def on(self, j):
        """The representation of w on piece j of the set (a number for constant and per-piece kinds, else self)."""
        return self

    @property
    def breaks(self):
        """Points of the set where w may lose smoothness (piece boundaries, declared singular points)."""
        return () if self.singular_at is None else tuple(map(tuple, self.singular_at))

    def scaled(self, c):
        return ScaledWeight(self, float(c))

    # ---------------------------------------------------------------- sign and class
    def sign(self):
        """(word, status, reason): word "non-negative", "non-positive", "signed" or "zero"."""
        if self._sign is None:
            self._sign = self._find_sign()
        return self._sign

    @property
    def nonnegative(self):
        return self.sign()[0] in ("non-negative", "zero")

    def _find_sign(self):
        X = sample_points(self.set)
        if X is None:
            return "unknown", "unknown", f"{self.set!r} offers no sample points"
        v = self.values(X)
        if not np.all(np.isfinite(v)) and self.singular_at is None:
            raise ValueError(f"the weight is not finite at sample points of {self.set!r}: declare its singular "
                             "behaviour (singular_at=points, exponent=a)")
        v = v[np.isfinite(v)]
        return _word(v), CHECKED, f"sampled at {len(X)} points of the set; a sample is not a proof"

    @property
    def singular(self):
        return self.singular_at is not None and self.exponent is not None and self.exponent > 0

    def bound(self):
        """(sup |w|, status, reason) for bounded weights; None for declared singular ones."""
        if self.singular:
            return None
        if self.sup is not None:
            return self.sup, ASSERTED, "declared by the user (sup=)"
        X = sample_points(self.set)
        if X is None:
            return np.nan, "unknown", f"{self.set!r} offers no sample points"
        return float(np.max(np.abs(self.values(X)))), CHECKED, \
            f"the maximum of |w| at {len(X)} sample points; a callable is assumed bounded on the set (ASSERTED)"

    def class_claim(self):
        if self.singular_at is not None:
            return Claim(f"|w| ~ dist^({-self.exponent:g}) near {len(self.singular_at)} declared point(s)", ASSERTED,
                         "declared with singular_at= and exponent=", key="weight class")
        if self.exact:
            return Claim("bounded weight", DERIVED, f"a {self.kind} weight is bounded on the set", key="weight class")
        b = self.bound()
        return Claim("bounded weight", ASSERTED, f"a callable is assumed bounded on the set; sampled max |w| = "
                                                 f"{b[0]:.3g} ({b[1]})", key="weight class")

    @property
    def claims(self):
        word, status, reason = self.sign()
        return Claims([Claim(f"weight: {self.kind}" + (" (exact)" if self.exact else ""), DERIVED,
                             "how it was given", key="weight kind"),
                       Claim(f"weight {word}", status, reason, key="sign"), self.class_claim()])


def _word(v):
    if v.size == 0 or np.all(v == 0):
        return "zero"
    if np.all(v >= 0):
        return "non-negative"
    return "non-positive" if np.all(v <= 0) else "signed"


class ConstantWeight(Weight):
    kind, exact = "constant", True

    def __init__(self, S, c, **kw):
        super().__init__(S, **kw)
        self.c = float(c)
        if not np.isfinite(self.c):
            raise ValueError("a constant weight must be finite")

    def values(self, X):
        return np.full(len(X), self.c)

    def on(self, j):
        return self.c

    def _find_sign(self):
        return _word(np.array([self.c])), THEOREM, "a constant"

    def bound(self):
        return abs(self.c), THEOREM, "a constant"

    def scaled(self, c):
        return ConstantWeight(self.set, self.c * c)


class PieceWeight(Weight):
    """One value per piece of the set (the segments of a polyline; the cells of a Simplicial set)."""

    kind, exact = "pieces", True

    def __init__(self, S, values, **kw):
        super().__init__(S, **kw)
        v = np.asarray(values, float).ravel()
        n = _npieces(S)
        if n is not None and v.shape != (n,):
            raise ValueError(f"a piecewise-constant weight on {S!r} needs one value per piece ({n}), not shape {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("piece values must be finite")
        self.c = v

    def values(self, X):
        idx = _piece_index(self.set, X)
        return self.c[idx]

    def on(self, j):
        return float(self.c[j])

    def _find_sign(self):
        return _word(self.c), THEOREM, "one value per piece"

    def bound(self):
        return float(np.max(np.abs(self.c))), THEOREM, "one value per piece"

    def scaled(self, c):
        return PieceWeight(self.set, self.c * c)


def _npieces(S):
    L = getattr(S, "_L", None)              # Polyline: segment lengths
    if L is not None:
        return len(L)
    cells = getattr(S, "cells", None)       # Simplicial (W2)
    return None if cells is None else len(cells)


def _piece_index(S, X):
    """The piece of the set containing each point (polyline: the nearest segment)."""
    if getattr(S, "_L", None) is not None:
        s = S.from_cartesian(*[X[:, k] for k in range(X.shape[1])])[0]
        return np.clip(np.searchsorted(S._cum, s, side="right") - 1, 0, len(S._L) - 1)
    locate = getattr(S, "locate", None)
    if locate is None:
        raise TypeError(f"{S!r} has no pieces to index a piecewise weight")
    return locate(X)


class CallableWeight(Weight):
    kind, exact = "callable", False

    def __init__(self, S, f, **kw):
        super().__init__(S, **kw)
        if not callable(f):
            raise TypeError("a weight is a number, one value per piece, an element or a callable")
        self.f = f

    def values(self, X):
        return _call(self.f, np.asarray(X, float))


class RadialWeight(Weight):
    """A weight rho(r) of the distance r from the centre of a ball or shell (plan 064): f a callable of r, sampled on the
    radial interval for its sign and bound (a callable is assumed bounded: ASSERTED, as for CallableWeight). It is
    what L2(rho dx) on a ball or shell realises in radial x harmonic factors (realisations/series/radial_harmonic.py)."""

    kind, exact = "radial", False

    def __init__(self, S, f, **kw):
        from ..domains import Ball, Shell
        super().__init__(S, **kw)
        if not isinstance(S, (Ball, Shell)):
            raise ValueError(f"a radial weight is a function of the distance from the centre of a ball or shell, not a "
                             f"weight on {S!r}")
        if not callable(f):
            raise TypeError("a radial weight is a callable of r")
        if self.singular_at is not None:
            raise ValueError("a radial weight with a declared singularity is not supported (plan 064)")
        self.f, self.centre = f, np.asarray(S.centre, float)
        self.interval = tuple(float(x) for x in S.reference[0])

    def radial(self, r):
        """rho at radii r (an array), as floats."""
        r = np.asarray(r, float)
        v = self.f(r)
        return np.broadcast_to(np.asarray(v, float), r.shape) if np.ndim(v) == 0 else np.asarray(v, float)

    def values(self, X):
        X = np.asarray(X, float)
        return self.radial(np.linalg.norm(X - self.centre[None, :], axis=1))

    def _samples(self):
        a, b = self.interval
        return self.radial(np.linspace(a, b, SAMPLES_1D))

    def _find_sign(self):
        v = self._samples()
        if not np.all(np.isfinite(v)):
            raise ValueError("the radial weight is not finite at sample radii")
        return _word(v), CHECKED, f"sampled at {SAMPLES_1D} radii of {self.interval}; a sample is not a proof"

    def bound(self):
        if self.sup is not None:
            return self.sup, ASSERTED, "declared by the user (sup=)"
        return float(np.max(np.abs(self._samples()))), CHECKED, \
            f"the maximum of |rho| at {SAMPLES_1D} sample radii; a callable is assumed bounded (ASSERTED)"

    def scaled(self, c):
        return RadialWeight(self.set, lambda r, f=self.f, c=float(c): c * np.asarray(f(r), float), sup=None
                            if self.sup is None else abs(c) * self.sup)


class FunctionWeight(Weight):
    """An element of a function space on the set (L2(Sphere), L2(Polyline), L2(Interval), ...): its realisation is the
    weight (exact for the realisation; its own error is the element's)."""

    kind, exact = "function", True

    def __init__(self, S, u, **kw):
        super().__init__(S, **kw)
        dom = u.space.domain
        if not (dom == S or getattr(dom, "same_up_to_null_set", lambda o: False)(S)):
            raise ValueError(f"the element {u.name} lives on {dom!r}, not on {S!r}")
        self.u = u

    def values(self, X):
        return np.asarray(self.u(*[X[:, k] for k in range(X.shape[1])]), float).reshape(len(X))

    def bound(self):
        X = sample_points(self.set)
        return float(np.max(np.abs(self.values(X)))), CHECKED, f"the maximum at {len(X)} sample points"


class RestrictedWeight(Weight):
    """A weight w on a part J of its set (mu.restricted(D), plan 035 P1): the same values, bound to J; its declared
    singular behaviour and bound carry over."""

    kind, exact = "callable", False

    def __init__(self, J, w):
        super().__init__(J, w.singular_at, w.exponent, w.sup)
        self.base = w
        self.exact = w.exact

    def values(self, X):
        return self.base.values(X)


class ScaledWeight(Weight):
    """c w, for a weight w (normalisations)."""

    def __init__(self, w, c):
        super().__init__(w.set, None if w.singular_at is None else w.singular_at, w.exponent,
                         None if w.sup is None else abs(c) * w.sup)
        self.base, self.c = w, float(c)
        self.kind, self.exact = w.kind, w.exact

    def values(self, X):
        return self.c * self.base.values(X)

    def on(self, j):
        v = self.base.on(j)
        return self.c * v if np.ndim(v) == 0 and not isinstance(v, Weight) else self

    @property
    def breaks(self):
        return self.base.breaks

    def _find_sign(self):
        word, status, reason = self.base.sign()
        flip = {"non-negative": "non-positive", "non-positive": "non-negative"}
        return (flip.get(word, word) if self.c < 0 else word if self.c > 0 else "zero"), status, reason

    def bound(self):
        b = self.base.bound()
        return None if b is None else (abs(self.c) * b[0], b[1], b[2])


class TabulatedWeight(Weight):
    """A pg.Tabulated (plan 009, G10) read as a weight on an interval inside its domain: exactly piecewise linear, so its
    sign and bound come from the values at the ends of its pieces (THEOREM) and its local orders are exact (order_at)."""

    kind, exact = "tabulated", True

    def __init__(self, S, tab, **kw):
        from ..sets import Interval
        super().__init__(S, **kw)
        D = tab.domain
        if not isinstance(S, Interval) or S.a < D.a or S.b > D.b:
            raise ValueError(f"{tab!r} is a weight on intervals inside {D}, not on {S!r} (on other sets give a "
                             "callable of the Cartesian coordinates)")
        self.tab = tab

    def values(self, X):
        return np.asarray(self.tab(np.asarray(X, float)[:, 0]), float).reshape(len(X))

    def _ends(self):
        t = self.tab
        lo, hi = np.maximum(t._x0, self.set.a), np.minimum(t._x1, self.set.b)
        keep = hi > lo
        s0 = (lo - t._x0) / (t._x1 - t._x0)
        s1 = (hi - t._x0) / (t._x1 - t._x0)
        dv = t._v1 - t._v0
        return np.r_[(t._v0 + dv * s0)[keep], (t._v0 + dv * s1)[keep]]

    def _find_sign(self):
        return _word(self._ends()), THEOREM, "piecewise linear: the extremes are at the ends of its pieces"

    def bound(self):
        return float(np.max(np.abs(self._ends()))), THEOREM, "piecewise linear: the extremes are at the ends of its pieces"

    @property
    def breaks(self):
        inner = [(float(b),) for b in self.tab.breaks if self.set.a < b < self.set.b]
        return tuple(inner) + super().breaks

    def order_at(self, x0):
        return self.tab.order_at(x0)


class QuotientWeight(Weight):
    """w / rho: the Radon-Nikodym derivative of w dx with respect to rho dx (measures.DensityMeasure.density_against)."""

    kind, exact = "quotient", False

    def __init__(self, w, rho):
        super().__init__(w.set)                  # the integrability of w / rho is decided from w and rho (boundedness)
        self.num, self.den = w, rho

    def values(self, X):
        return self.num.values(X) / self.den.values(X)


def size(S):
    """The Lebesgue (Hausdorff) measure of a set: length, area or volume."""
    return float(S.volume)                         # the Region's Hausdorff measure (plan 067 P2; an interval's length)


def integrate_on(S, f, n=32, points=(), breaks=(), exponent=None):
    """int_S f dsigma for f a function of Cartesian points (m, d) -> (m,) (or (m, N)): see nodes_on. A rule, not a
    claim: callers evaluate it at two orders and report the difference."""
    X, W = nodes_on(S, n, points, breaks, exponent)
    v = W @ f(X)
    return float(v) if np.ndim(v) == 0 else v


def nodes_on(S, n=32, points=(), breaks=(), exponent=None):
    """Nodes (m, d) and weights (m,) for the Lebesgue (Hausdorff) measure of S, from the numerical column (plan 069
    P2): on an interval pygeoinf0.quadrature.graded (Gauss-Legendre with n points per piece, pieces split at breaks,
    graded geometrically towards the points, where integrands may be singular, the innermost cell by Gauss-Jacobi for a
    declared exponent); on other sets pygeoinf0.quadrature.rule(S, n), through the set's charts."""
    from ..quadrature import graded, rule
    from ..sets import Interval
    if not isinstance(S, Interval):
        X, W = rule(S, n)
        return np.asarray(X, float), np.asarray(W, float)
    x, w = graded(S.a, S.b, n, points, breaks, exponent)
    return x[:, None], w


def as_weight(w, S, at=None, singular_at=None, exponent=None, sup=None, radial=False):
    """The Weight of the set S given by w (see the module docstring); a Weight passes through (bound to S).
    radial=True: w is a callable of the distance r from the centre of a ball or shell (RadialWeight, plan 064); a
    number stays a constant weight."""
    from ..spaces.elements import Function
    from ..core.tabulated import Tabulated
    kw = dict(singular_at=singular_at, exponent=exponent, sup=sup)
    if isinstance(w, Weight):
        if w.set is not S and w.set != S:
            raise ValueError(f"the weight is defined on {w.set!r}, not on {S!r}")
        return w
    if radial and not (np.ndim(w) == 0 and isinstance(w, (int, float, np.number))):
        return RadialWeight(S, w, **kw)
    if isinstance(w, Tabulated):                     # the one tabulated class of the package (plan 009 G10)
        return TabulatedWeight(S, w, **kw)
    if at == "vertices":
        raise NotImplementedError("vertex weights (piecewise linear on a Simplicial set) arrive with pg.Simplicial")
    if isinstance(w, Function):
        return FunctionWeight(S, w, **kw)
    if np.ndim(w) == 0 and isinstance(w, (int, float, np.number)):
        return ConstantWeight(S, w, **kw)
    if callable(w):
        return CallableWeight(S, w, **kw)
    return PieceWeight(S, w, **kw)
