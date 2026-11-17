"""Bank functions: functions with metadata and claims, not elements (plan 021; EXP-037 DESIGN.md 1, 3).

    BankFunction(family, domain, params, fn, support, regularity, closed=None, gradient=None, name=None)
    Support(kind, description, centre=None, radius=None, contains=None, breaks=None)   kind: interior, touches, full
    Regularity(order, piecewise=False, vanishes_at_boundary=None)

regularity.order: math.inf for C-infinity, k for C^k with bounded derivatives on the closure, 0 for continuous, -1 for
bounded with jumps. piecewise=True: smooth away from finitely many smooth interfaces, where the derivatives up to
`order` are continuous (then H^s for s < order + 3/2). vanishes_at_boundary: None (no claim), "all orders" (every
derivative vanishes at the boundary, e.g. interior support), or an integer j (derivatives of order < j vanish there).
"""
import math
import warnings
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..core.claims import CHECKED, DERIVED, THEOREM, UNKNOWN, Claim, Claims
from ..core.errors import AveragingWarning, Refused
from . import geometry as geo

FLAT = ("interval", "box", "disk", "ball", "shell", "cylinder")
AVERAGE_NEGATIVE = 1e-12        # normalised(): a value below -this * max |f| is a sign change
AVERAGE_TINY = 1e-8             # normalised(): |int f| below this * int |f| is a near-zero integral


@dataclass(frozen=True)
class Support:
    kind: str                       # "interior", "touches" or "full"
    description: str
    centre: tuple = None            # with radius: the support lies in the closed geodesic ball (centre, radius)
    radius: float = None
    contains: object = None         # optional: (*X) -> bool array, the support set itself (for non-ball supports)
    box: tuple = None               # optional: ((lo, hi), ...) per axis, a coordinate box holding the support
    breaks: tuple = None            # radial functions: the distances from the centre where derivatives jump inside the
    #                                 ball (() for none); integrals are then taken in polar coordinates, split there

    def __post_init__(self):
        if self.kind not in ("interior", "touches", "full"):
            raise ValueError(f"support kind {self.kind!r} (interior, touches, full)")


@dataclass(frozen=True)
class Regularity:
    order: float                    # math.inf, an integer k >= 0, or -1 (bounded with jumps)
    piecewise: bool = False
    vanishes_at_boundary: object = None     # None, "all orders", or an integer j

    def sobolev_bound(self):
        """(s_max, strict): the function lies in H^s for s < s_max (strict) or s <= s_max (not strict)."""
        if self.order == math.inf:
            return math.inf, False
        if self.piecewise or self.order < 0:
            return self.order + 1.5, True
        return float(self.order), False

    def describe(self):
        if self.order == math.inf:
            r = "C-infinity"
        elif self.order < 0:
            r = "bounded, piecewise smooth with jumps"
        else:
            r = f"C^{int(self.order)}" + (", piecewise smooth" if self.piecewise else "")
        v = self.vanishes_at_boundary
        if v == "all orders":
            r += "; every derivative vanishes at the boundary"
        elif v is not None:
            r += f"; derivatives of order < {int(v)} vanish at the boundary"
        return r


def _plain(v):
    """numpy scalars (also inside tuples) as Python numbers, so that names and params read plainly."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, tuple):
        return tuple(_plain(x) for x in v)
    return v


POSITIONS = ("centre", "index", "nodes", "count", "knots", "lower", "upper")    # parameters that place a function
WIDTHS = ("width", "radius", "half_width", "edge", "size", "sizes", "radii")     # parameters that are feature widths


class BankFunction:
    """A function of the bank (module docstring)."""

    def __init__(self, family, domain, params, fn, support, regularity, closed=None, gradient=None, name=None,
                 reasons=None, breaks=None):
        self.family, self.domain, self.fn = family, domain, fn
        self.breaks = breaks            # per axis, Cartesian: where derivatives (or values) jump inside the support
        self.params = MappingProxyType({k: _plain(v) for k, v in params.items()})
        self.support, self.regularity = support, regularity
        self.closed = MappingProxyType(dict(closed or {}))          # name -> (value, status, reason)
        self.gradient = gradient
        self.reasons = dict(reasons or {})                          # "support", "regularity" -> reason text
        self.name = name or f"{family}({', '.join(f'{k}={v!r}' for k, v in self.params.items())})"
        self.kind = geo.kind(domain)

    def __call__(self, *X):
        if len(X) == 1 and geo.ambient(self.domain) > 1 and np.ndim(X[0]) >= 1 and np.shape(X[0])[-1] == geo.ambient(
                self.domain):
            X = tuple(np.moveaxis(np.asarray(X[0], float), -1, 0))         # f(P) with P of shape (..., d)
        if len(X) != geo.ambient(self.domain):
            raise TypeError(f"{self.name} takes {geo.ambient(self.domain)} Cartesian coordinates, got {len(X)}")
        shape = np.broadcast(*[np.asarray(x) for x in X]).shape
        out = np.asarray(self.fn(*X), float)
        return np.broadcast_to(out, shape) if out.shape != shape else out

    # ------------------------------------------------------------------------------------------- values
    def integral(self):
        """(value, status, reason): the closed form if the family derived one, else the domain's quadrature with its
        estimated error, status unknown: a rule's value is not a claim."""
        if "integral_refused" in self.reasons:
            raise Refused(self.reasons["integral_refused"])
        if "integral" in self.closed:
            return self.closed["integral"]
        return self._rule(24)

    def _rule(self, n):
        """The domain's quadrature rule with n points per axis, with the difference from n/2 + 4 as its estimated error."""
        rule = [float(w @ self(*X.T)) for X, w in (geo.quadrature(self.domain, m) for m in (n, n // 2 + 4))]
        return rule[0], UNKNOWN, (f"the domain's quadrature rule ({n} points per axis); estimated error {abs(rule[0] - rule[1]):.1g}, "
                                  "from a coarser rule, not a bound")

    def scaled(self, c, name=None):
        c = float(c)
        closed = {}
        for key, (v, st, why) in self.closed.items():
            v = c * v if key in ("integral", "mass") else c * c * v if key == "l2_norm_sq" else abs(c) * v
            closed[key] = (v, st, f"{why}; scaled by {c:g}")
        grad = None if self.gradient is None else (lambda *X: c * self.gradient(*X))
        return BankFunction(self.family, self.domain, {**self.params, "multiplier": c}, lambda *X: c * self.fn(*X),
                            self.support, self.regularity, closed, grad, name or f"{c:g}·{self.name}", self.reasons, self.breaks)

    def normalised(self, against=None):
        """The multiple with integral 1 over the domain, or with int f dmu = 1 for a measure `against` on the domain
        (e.g. a weighted space's M.measure); refused when the integral is 0. Without an exact integral it divides by a quadrature
        rule and claims nothing: integral() then reports the coarser rule's value and its estimated error. A function that is not a localized average
        gets an AveragingWarning: negative values somewhere, or an integral tiny against the integral of |f|, both read
        from a quadrature rule on the support, which can miss a narrow feature."""
        lebesgue = against is None or (type(against).__name__ == "LebesgueMeasure" and against.set == self.domain)
        if lebesgue:
            v, status, why = self.integral()
            if status == UNKNOWN:           # no exact integral: a finer rule, and no claim of 1 (see integral())
                v = self._rule(32)[0]
        else:
            mu = self.measure(against)
            v, err = mu.mass()
            status, why = CHECKED, f"int f dmu over {mu.set}, mu = {against} (estimated error {err:.1g})"
        if abs(v) < 1e-300:
            raise Refused(f"{self.name} has integral 0: it cannot be normalised to unit mass")
        self._warn_if_not_an_average(v if lebesgue else None)
        out = self.scaled(1.0 / v, name=f"normalised {self.name}")
        if lebesgue:
            if status != UNKNOWN:
                out._with_unit_integral(status, f"normalised: {why}")
        else:
            out.closed = MappingProxyType({**out.closed, "mass": (1.0, status, f"normalised against {against}: {why}")})
        return out

    def _with_unit_integral(self, status, why):
        self.closed = MappingProxyType({**self.closed, "integral": (1.0, status, why)})

    def _warn_if_not_an_average(self, integral):
        X, w = geo.quadrature(self.support_set(), 24)
        y = self(*X.T)
        size = float(np.max(np.abs(y)))
        if size > 0 and float(np.min(y)) < -AVERAGE_NEGATIVE * size and float(np.max(y)) > AVERAGE_NEGATIVE * size:
            warnings.warn(f"{self.name} changes sign: normalised to integral 1, it is not a localized average",
                          AveragingWarning, stacklevel=3)
        whole = float(w @ np.abs(y))
        if integral is not None and abs(integral) < AVERAGE_TINY * whole:
            warnings.warn(f"{self.name} has an integral {abs(integral) / whole:.1e} of the integral of its absolute value: "
                          "normalising it blows up its scale", AveragingWarning, stacklevel=3)

    def support_set(self):
        """A set containing the support, on which integrals against f are taken: the support's ball as an Interval,
        Disk or Ball on interval, box, disk, ball, shell and cylinder domains, else the domain itself. A rule on the
        support resolves a narrow function that a rule over the whole domain misses."""
        from ..sets import Interval
        from ..domains import Ball, Disk
        s, D = self.support, self.domain
        if s.kind == "full" or s.centre is None or geo.kind(D) not in FLAT:
            return D
        c, r = [float(x) for x in s.centre], float(s.radius)
        if len(c) > 1 and geo.room(D, c) < r * (1 - 1e-12):    # a ball cut by the domain (bump(clip=True))
            from ..domains import Box
            return Box(*[Interval(a, b) for a, b in self.support_box()]) if geo.kind(D) == "box" else D
        if len(c) == 1:
            return Interval(max(c[0] - r, D.a), min(c[0] + r, D.b))
        return Disk(r, centre=tuple(c)) if len(c) == 2 else Ball(r, centre=tuple(c))

    def measure(self, against=None):
        """f dmu as a measure on support_set(): mu the Lebesgue measure of the domain, or `against` (its Lebesgue
        measure or a density measure on it, e.g. a weighted space's M.measure). M.riesz([f.measure(M.measure)]) is the
        representer of m -> int f m dmu in M: the SOLA target functional with averaging kernel f."""
        from ..measures import DensityMeasure, LebesgueMeasure
        S = self.support_set()
        if against is None or (isinstance(against, LebesgueMeasure) and against.set == self.domain):
            return LebesgueMeasure(S).with_density(self.fn, name=self.name)
        if not (isinstance(against, DensityMeasure) and against.set == self.domain):
            raise Refused(f"{self.name}: a measure against {against} needs the Lebesgue measure of {self.domain} or "
                          "a density measure on it")
        w = against.weight

        def density(*X):
            P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
            return self.fn(*X) * w.values(P.reshape(-1, P.shape[-1])).reshape(P.shape[:-1])
        return LebesgueMeasure(S).with_density(density, name=f"{self.name}·{against.name}")

    # ------------------------------------------------------------------------------------------- claims
    @property
    def claims(self):
        cl = [Claim(f"support: {self.support.description}", DERIVED,
                    self.reasons.get("support", f"the definition of {self.family}"), key="support"),
              Claim(f"regularity: {self.regularity.describe()}", THEOREM if "regularity" in self.reasons else DERIVED,
                    self.reasons.get("regularity", f"the definition of {self.family}"), key="regularity")]
        for key, (v, status, why) in self.closed.items():
            cl.append(Claim(f"{key} = {v:.15g}", status, why, key=key))
        return Claims(cl)

    def fit_breaks(self, D=None):
        """Breaks for a realisation on a box or interval D (default: the function's domain), per axis: the edges of the
        support's bounding box and the stated kinks, strictly inside D. On a ball or shell D, natural-coordinate breaks
        from a support ball B(c, rho) (round_breaks: r = d -+ rho, the cap theta_c -+ delta, phi_c -+ its half-width,
        delta = asin(rho / d)) with the least angular degree 2 pi / delta for spectral fits (plan 035 P2). None
        elsewhere.

        Without them the fit samples the whole domain and refines strips (plan 022 P8, EXP-038 p8_local.py): a bump of
        radius 0.02 met no starting node and was realised as 0 with error 0; a P1 hat of a 40 x 40 grid took 20.8 s and
        2.2e6 coefficients (64 with its kinks as breaks, exact)."""
        D = self.domain if D is None else D
        k = geo.kind(D)
        if k in ("ball", "shell"):
            return self.round_breaks(D)
        if k not in ("interval", "box") or geo.kind(self.domain) != k:
            return None
        axes = [(float(D.a), float(D.b))] if k == "interval" else [(float(s.a), float(s.b)) for s in D.sides]
        out = [set() for _ in axes]
        sup = self.support
        if sup.box is not None:                              # a stated box (box indicators, FEM): its own edges
            for j, (lo, hi) in enumerate(sup.box[:len(axes)]):
                out[j] |= {float(lo), float(hi)}
        elif sup.centre is not None and sup.radius is not None:
            for j, c in enumerate(np.atleast_1d(np.asarray(sup.centre, float))[:len(axes)]):
                out[j] |= {c - sup.radius, c + sup.radius}
        for j, b in enumerate(self.breaks or ()):
            out[j] |= {float(x) for x in b}
        brk = tuple(tuple(sorted(x for x in o if a + 1e-12 * (b - a) < x < b - 1e-12 * (b - a)))
                    for o, (a, b) in zip(out, axes))
        if not any(brk):
            return None
        if k == "interval":
            return list(brk[0])
        box = self.support_box()
        out = tuple(b or None for b in brk)
        if box is None:
            return out
        from ..realisations.series.local import Supported         # realisation="local" fits on the box only (plan 025)
        return Supported(out, box, self.translation())

    def round_breaks(self, D):
        """NaturalBreaks (r, theta, phi) on the ball or shell D from the support ball (fit_breaks), or None when the
        support is not a ball of R^3 or is the whole domain."""
        from ..realisations.backends import NaturalBreaks
        sup = self.support
        if sup.kind == "full" or sup.centre is None or sup.radius is None or np.size(sup.centre) != 3:
            return None
        a, b = (0.0, D.radius) if geo.kind(D) == "ball" else (D.inner, D.outer)
        c = np.asarray(sup.centre, float) - np.asarray(D.centre, float)
        d, rho = float(np.linalg.norm(c)), float(sup.radius)

        def inside(xs, lo, hi):
            return tuple(sorted(float(x) for x in xs if lo + 1e-12 * (hi - lo) < x < hi - 1e-12 * (hi - lo)))
        rb = inside((d - rho, d + rho), a, b) or None
        rbox = (max(a, d - rho), min(b, d + rho))
        if d <= rho:                                     # the support holds the centre: every direction
            return NaturalBreaks((rb, None, None), box=(rbox, (0.0, math.pi), (0.0, 2 * math.pi)))
        delta = math.asin(rho / d)
        th, ph = math.acos(max(-1.0, min(1.0, c[2] / d))), math.atan2(c[1], c[0]) % (2 * math.pi)
        tb = inside((th - delta, th + delta), 0.0, math.pi) or None
        pb, pbox = None, (0.0, 2 * math.pi)
        if th - delta > 0 and th + delta < math.pi:      # a cap off the poles: its longitudes
            w = math.asin(min(1.0, math.sin(delta) / math.sin(th)))
            pbox = ((ph - w) % (2 * math.pi), (ph + w) % (2 * math.pi))
            pb = inside(pbox, 0.0, 2 * math.pi) or None
        box = (rbox, (max(0.0, th - delta), min(math.pi, th + delta)), pbox)
        return NaturalBreaks((rb, tb, pb), degree=int(math.ceil(2 * math.pi / delta)), box=box)

    def nonzero_on(self, D):
        """Why f is not zero on D (a reason, or None when that is not known): a positive closed norm or a non-zero
        closed integral, and a support that meets D in a set of positive measure (decided exactly where possible)."""
        v = self.closed.get("l2_norm_sq", self.closed.get("integral", (0.0,)))[0]
        if not v or D is None:
            return None
        if D != self.domain:
            from ..core.errors import Unsupported
            from ..domains.pieces import meet
            try:
                if meet(self.support_set(), D) is None:
                    return None
            except (Unsupported, TypeError):
                return None
        return f"{self.name} is not zero there (closed {'l2_norm_sq' if 'l2_norm_sq' in self.closed else 'integral'} " \
               f"{v:.3g}; support {self.support.description})"

    def translation(self):
        """(key, origin): functions of one family with equal key are translates of one another by the difference of
        their origins (the lower corner of the support box), as far as the parameters show: the family, the parameters
        that are not positions, the box's size and the kinks relative to it. realisation="local" then shifts an earlier
        fit instead of fitting again, after checking the values at sample points (local.LocalBackend.fit). None
        without a support box."""
        box = self.support_box()
        if box is None or self.kind != "box":
            return None
        lo = tuple(float(a) for a, _ in box)
        size = tuple(round(b - a, 12) for a, b in box)
        rel = tuple(tuple(round(float(x) - a, 12) for x in (b or ())) for b, a in zip(self.breaks or [()] * len(box), lo))
        shape = tuple((k, v) for k, v in sorted(self.params.items()) if k not in POSITIONS)
        return (self.family, shape, size, rel), lo

    def support_box(self):
        """((lo, hi), ...) per axis: a coordinate box holding the support (Support.box, or the ball's bounding box on
        a box or interval), None when the support is the whole domain or not a box or ball."""
        sup = self.support
        if sup.kind == "full":
            return None
        if sup.box is not None:
            return tuple((float(a), float(b)) for a, b in sup.box)
        if sup.centre is not None and sup.radius is not None and geo.kind(self.domain) in ("interval", "box"):
            D = self.domain
            sides = [(D.a, D.b)] if geo.kind(D) == "interval" else [(s.a, s.b) for s in D.sides]
            return tuple((max(c - sup.radius, a), min(c + sup.radius, b))
                         for c, (a, b) in zip(np.atleast_1d(np.asarray(sup.centre, float)), sides))
        return None

    def feature_width(self):
        """The narrowest feature width its parameters state (plan 054 fix2): the smallest positive of its width-like
        parameters (WIDTHS: width, radius, half_width, edge, sizes, radii) and of radius - inner; None when it states
        none (polynomials, eigenfunctions, harmonics: their degree, not a width, sets what a fit must resolve)."""
        vals = [np.asarray(self.params[k], float).ravel() for k in WIDTHS if k in self.params
                and self.params[k] is not None]
        if self.params.get("inner") is not None and self.params.get("radius") is not None:
            vals.append(np.atleast_1d(float(self.params["radius"]) - float(self.params["inner"])))
        pos = [float(v) for a in vals for v in a if np.isfinite(v) and v > 0]
        return min(pos) if pos else None

    def element(self, M, name=None):
        """M(f) when the claims show f is a member of M; refused with the reason otherwise (spaces/membership.py). The
        realisation starts from the support's edges and the kinks (fit_breaks). Where those give no breaks (a support
        that is the whole domain, such as a Gaussian's) the feature width its parameters state (feature_width) is the
        fit's declared resolution (core/fits.py): it samples at w/2 and its claim says w met, or a RealisationWarning
        says it is not (also where the realisation refuses it, such as a spherical degree above its cap)."""
        from ..spaces.membership import check, refusal
        ok, reason, claim = check(self, M)
        if not ok:
            raise refusal(self, M, reason)
        brk = self.fit_breaks(getattr(M, "domain", None))
        w = self.feature_width() if brk is None else None
        from ..core.fits import as_bank, bank_width
        with as_bank():                                    # a bank function: not a bare callable (core/fits.py)
            if w is not None:
                el = bank_width(M(self.fn, name=name or self.name, resolution=w))
            else:
                el = M(self.fn, name=name or self.name) if brk is None else M(self.fn, name=name or self.name,
                                                                                breaks=brk)
        if self.regularity.order >= 0 and getattr(el, "_exact", 1) is None and el._src is not None:
            el._exact = el._src             # continuous: the formula is the canonical representative (plan 022 P6)
        el._memo["membership"] = claim                       # its first claim (Function.claims, plan 065 A1)
        self.last_membership = claim
        return el

    def __repr__(self):
        return f"{self.name} on {self.domain!r} [{self.regularity.describe()}; support {self.support.description}]"
