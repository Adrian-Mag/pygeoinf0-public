"""Curves and surfaces given by callables (EXP-031, DESIGN 4.4; question 13).

    Curve(gamma, t0, t1, derivative=None, breaks=(), closed=False)
        the image gamma([t0, t1]) of gamma: [t0, t1] -> R^d, injective (or closed: gamma(t0) = gamma(t1), injective on
        [t0, t1)), C^1 with |gamma'| > 0 between the breaks. Its measure is arc length ds = |gamma'(t)| dt (the area
        formula: the Hausdorff measure of the image, independent of the parametrisation). A Domain of dimension 1 whose
        natural coordinate is arc length s in [0, length], like Polyline and Arc: L2 and Sobolev spaces on it are the
        interval realisations in s (intrinsic in arc length, as on paths), and its measures are functionals on 2-D and
        3-D domains.
    Surface(chi, (u0, u1), (v0, v1), derivatives=None)
        the image of a chart chi(u, v) in R^3 with area element |chi_u x chi_v| du dv: a support of measures (a Region, not
        a Domain: function spaces on it would need its area measure, which is not a product in (u, v), and the
        Laplace-Beltrami operator for intrinsic H^s).

gamma takes an array of parameters (m,) and returns points (m, d), or a tuple of d coordinate arrays (m,); chi takes
(u, v) arrays and returns the same. |gamma'| comes from `derivative` when it is given (checked against the fit at
sample points: CHECKED, a sample is not a proof), otherwise from a piecewise Legendre fit of the components,
differentiated exactly; the fit's error is recorded. No curve is ever replaced by a polyline here (question 14).
"""
import numpy as np
from numpy.polynomial import legendre as npleg

from .base import Domain, Parametrisation, _fmt
from .flat import ON
from ..core.claims import ASSERTED, CHECKED, THEOREM, Claim, Claims
from ..quadrature import gauss, rule
from ..sets import Region
from ..sets.regions import Chart

PANEL_NODES = 32           # Gauss points per panel of the arc-length integral
PANELS = 4                 # panels per piece of the fit
SAMPLES = 4097             # parameter samples for the projection's starting points and the bounding box


def _points(f, *t):
    """f(*t) as points (m, d): tuples/lists of coordinate arrays are stacked, arrays must already be (m, d)."""
    v = f(*t)
    m = np.size(t[0])
    if isinstance(v, (tuple, list)):
        return np.stack([np.broadcast_to(np.asarray(c, float), np.shape(t[0])).ravel() for c in v], axis=-1)
    v = np.asarray(v, float)
    return v.reshape(m, -1)


def _derivative(pl, r=1):
    """The r-th derivative of a piecewise Legendre realisation (exact on its polynomials)."""
    from ..realisations.series import legendre as L
    coefs = []
    for k, C in enumerate(pl.coefs):
        h = pl.breaks[k + 1] - pl.breaks[k]
        coefs.append(npleg.legder(C, r, scl=2.0 / h, axis=0) if C.shape[0] > r else np.zeros((1, C.shape[1])))
    return L.PiecewiseLegendre(pl.breaks, coefs, np.zeros(pl.N))


class Curve(Domain):
    """A curve given by a callable (module docstring)."""

    dim = 1
    jacobian = ("1",)
    internal_names = ("s",)

    def __init__(self, gamma, t0, t1, derivative=None, breaks=(), closed=False, name=None):
        from ..realisations.series import legendre as L
        t0, t1 = float(t0), float(t1)
        if not t0 < t1:
            raise ValueError("a curve needs t0 < t1")
        self.gamma, self.derivative, self.t0, self.t1 = gamma, derivative, t0, t1
        self.breaks = tuple(sorted(float(b) for b in breaks if t0 < float(b) < t1))
        self.closed, self.name = bool(closed), name
        ends = _points(gamma, np.array([t0, t1]))
        self.ambient_dim = ends.shape[1]
        if self.ambient_dim not in (2, 3):
            raise ValueError("a curve lies in R^2 or R^3")
        fit, unresolved = L.fit(lambda t: _points(gamma, np.asarray(t, float)), t0, t1, N=self.ambient_dim,
                                breaks=list(self.breaks) or None)
        if unresolved or not np.all(np.isfinite(fit.err)):
            raise L.RealisationError("could not resolve the curve's parametrisation (pass breaks= at its corners)")
        self._fit, self._d1, self._d2 = fit, _derivative(fit, 1), _derivative(fit, 2)
        scale = float(np.max(np.abs(ends))) + 1.0
        if self.closed and np.linalg.norm(ends[0] - ends[1]) > 1e-10 * scale:
            raise ValueError("a closed curve needs gamma(t0) = gamma(t1)")
        # panels of the arc-length integral: every piece of the fit split in PANELS
        e = np.asarray(fit.breaks, float)
        self._panels = np.unique(np.concatenate([np.linspace(a, b, PANELS + 1) for a, b in zip(e[:-1], e[1:])]))
        g, gw = gauss(PANEL_NODES)
        self._g, self._gw = g, gw
        a, b = self._panels[:-1], self._panels[1:]
        lens = [0.5 * (b - a) * (self.speed(0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * g) @ gw)]
        g2, gw2 = gauss(PANEL_NODES + 16)
        lens2 = 0.5 * (b - a) * (self.speed(0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * g2) @ gw2)
        self._cum = np.concatenate([[0.0], np.cumsum(lens[0])])
        self.length = float(self._cum[-1])
        self.length_error = float(abs(np.sum(lens2) - self.length))
        if not self.length > 0:
            raise ValueError("the curve has zero length")
        self.reference = ((0.0, self.length),)
        self.periodic = (self.closed,)
        self.reference_breaks = (tuple(float(v) for v in self.arclength(np.array(self.breaks))),) if self.breaks \
            else None
        self._ts = np.linspace(t0, t1, SAMPLES)
        self._xs = _points(gamma, self._ts)
        self._speed_check = None
        if derivative is not None:
            d = _points(derivative, self._ts)
            ref = L.evaluate(self._d1, self._ts)
            self._speed_check = float(np.max(np.linalg.norm(d - ref, axis=1)) / max(np.max(np.linalg.norm(ref, axis=1)),
                                                                                     1e-300))
            if self._speed_check > 1e-6:
                raise ValueError(f"the derivative given does not match gamma (relative difference {self._speed_check:.1g} "
                                 "at sample points)")

    # ---------------------------------------------------------------- the parametrisation
    def velocity(self, t):
        """gamma'(t) (points (m, d)): the derivative given, or the fit's."""
        from ..realisations.series import legendre as L
        t = np.asarray(t, float)
        if self.derivative is not None:
            return _points(self.derivative, t.ravel()).reshape(t.shape + (self.ambient_dim,))
        return L.evaluate(self._d1, t)

    def speed(self, t):
        """|gamma'(t)|."""
        return np.linalg.norm(self.velocity(t), axis=-1)

    def parametrisation(self):
        return Parametrisation(lambda t: _points(self.gamma, np.asarray(t, float).ravel()), self.velocity,
                               self.t0, self.t1, self.breaks, self.closed)

    def arclength(self, t):
        """s(t) = int_t0^t |gamma'|: Gauss-Legendre on the panel containing t."""
        t = np.clip(np.asarray(t, float), self.t0, self.t1)
        k = np.clip(np.searchsorted(self._panels, t, side="right") - 1, 0, len(self._panels) - 2)
        a = self._panels[k]
        nodes = 0.5 * (a + t)[..., None] + 0.5 * (t - a)[..., None] * self._g
        return self._cum[k] + 0.5 * (t - a) * (self.speed(nodes) @ self._gw)

    def parameter(self, s):
        """t(s): the inverse of the arc length (bracketed Newton on the panel containing s)."""
        s = np.clip(np.asarray(s, float), 0.0, self.length)
        k = np.clip(np.searchsorted(self._cum, s, side="right") - 1, 0, len(self._panels) - 2)
        lo, hi = self._panels[k].copy(), self._panels[k + 1].copy()
        t = lo + (hi - lo) * (s - self._cum[k]) / np.maximum(self._cum[k + 1] - self._cum[k], 1e-300)
        for _ in range(60):
            r = self.arclength(t) - s
            lo, hi = np.where(r < 0, t, lo), np.where(r > 0, t, hi)
            v = self.speed(t)
            step = np.where(v > 0, r / np.where(v > 0, v, 1.0), np.inf)
            tn = t - step
            bad = ~((tn > lo) & (tn < hi))
            tn = np.where(bad, 0.5 * (lo + hi), tn)
            if np.all(np.abs(tn - t) <= 4e-16 * (abs(self.t1) + abs(self.t0))):
                t = tn
                break
            t = tn
        return t

    # ---------------------------------------------------------------- Domain interface (arc-length coordinate)
    def to_cartesian(self, s):
        s = np.asarray(s, float)
        X = _points(self.gamma, self.parameter(s).ravel())
        return tuple(X[:, k].reshape(s.shape) for k in range(self.ambient_dim))

    def point(self, t):
        """gamma(t), shape t.shape + (d,) (the callable given)."""
        t = np.asarray(t, float)
        return _points(self.gamma, t.ravel()).reshape(t.shape + (self.ambient_dim,))

    def nearest(self, X):
        """(t, distance) of the nearest point of the curve to each point of X (m, d): the nearest of 4097 samples,
        then Newton on (gamma(t) - x) . gamma'(t) = 0 with the fit's derivatives."""
        from ..realisations.series import legendre as L
        X = np.atleast_2d(np.asarray(X, float))
        T, D = np.empty(len(X)), np.empty(len(X))
        for c in range(0, len(X), 256):
            P = X[c:c + 256]
            i = np.argmin(np.linalg.norm(P[:, None, :] - self._xs[None], axis=-1), axis=1)
            t = self._ts[i]
            for _ in range(30):
                g0 = L.evaluate(self._fit, t) - P
                g1, g2 = L.evaluate(self._d1, t), L.evaluate(self._d2, t)
                f, df = np.sum(g0 * g1, axis=1), np.sum(g1 * g1, axis=1) + np.sum(g0 * g2, axis=1)
                tn = np.clip(t - f / np.where(df > 0, df, np.inf), self.t0, self.t1)
                done = np.all(np.abs(tn - t) <= 1e-15 * (abs(self.t1) + abs(self.t0) + 1))
                t = tn
                if done:
                    break
            T[c:c + 256], D[c:c + 256] = t, np.linalg.norm(_points(self.gamma, t) - P, axis=1)
        return T, D

    def from_cartesian(self, *x):
        """The arc length of the nearest point of the curve, for points within 1e-8 of the length of it (else NaN)."""
        X = np.stack(np.broadcast_arrays(*[np.asarray(a, float) for a in x]), axis=-1)
        t, dist = self.nearest(X.reshape(-1, self.ambient_dim))
        out = np.where(dist <= ON * self.length, self.arclength(t), np.nan)
        return (out.reshape(X.shape[:-1]),)

    def _inside(self, *x):
        return np.isfinite(self.from_cartesian(*x)[0])

    @property
    def bounding_box(self):
        return [(float(lo), float(hi)) for lo, hi in zip(self._xs.min(axis=0), self._xs.max(axis=0))]

    @property
    def volume(self):
        return self.length

    @property
    def claims(self):
        speed = (Claim("speed |gamma'| from the derivative given", CHECKED,
                       f"against the fit's derivative at {SAMPLES} points: relative difference {self._speed_check:.1g}")
                 if self.derivative is not None else
                 Claim("speed |gamma'| from a piecewise Legendre fit of gamma, differentiated exactly", CHECKED,
                       f"fit error {float(np.max(self._fit.err)):.1g} (L2 in t)"))
        return Claims([Claim("measure: arc length", THEOREM, "the area formula (Hausdorff measure of the image)"),
                       speed, Claim(f"length {self.length:.15g}", CHECKED, f"Gauss on {len(self._panels) - 1} panels; "
                                                                           f"a second order differs by {self.length_error:.1g}"),
                       Claim("injective" + (" on [t0, t1)" if self.closed else ""), ASSERTED, "not checked")])

    def subpiece(self, lo, hi):
        """The sub-curve between arc lengths lo[0] < hi[0]: the same gamma on [t(lo), t(hi)] (exact up to the inversion
        of the arc length, Newton to rounding)."""
        a, b = (float(self.parameter(np.array(float(np.ravel(v)[0])))) for v in (lo, hi))
        return Curve(self.gamma, a, b, derivative=self.derivative, breaks=self.breaks, name=None)

    def key(self):
        return ("Curve", id(self.gamma), self.t0, self.t1, self.breaks, self.closed)

    def __repr__(self):
        return self.name or f"Curve(γ on [{_fmt(self.t0)}, {_fmt(self.t1)}] in R^{self.ambient_dim}, length " \
                            f"{self.length:.6g}{', closed' if self.closed else ''})"


class Surface(Region):
    """A parametric surface in R^3 (module docstring): a support of measures, a Region whose membership is not decided
    here (member(x) is Truth.UNKNOWN, contains refuses). Its chart is (u, v) (pygeoinf0.quadrature.rule: a tensor Gauss rule in
    (u, v)) with the area element |chi_u x chi_v| (from `derivatives` = (chi_u, chi_v) when given, else central
    differences of fourth order with a step 2e-4 of the parameter range: about 1e-12 relative, CHECKED)."""

    dim, ambient_dim = 2, 3
    cartesian_dim = 3

    def __init__(self, chi, u, v, derivatives=None, name=None):
        (u0, u1), (v0, v1) = (tuple(map(float, u)), tuple(map(float, v)))
        if not (u0 < u1 and v0 < v1):
            raise ValueError("a surface needs u0 < u1 and v0 < v1")
        self.chi, self.derivatives, self.u, self.v, self.name = chi, derivatives, (u0, u1), (v0, v1), name
        P = _points(chi, np.array([u0, u1]), np.array([v0, v1]))
        if P.shape[1] != 3:
            raise ValueError("a surface lies in R^3")
        self._box = None

    def tangents(self, U, V):
        U, V = np.asarray(U, float).ravel(), np.asarray(V, float).ravel()
        if self.derivatives is not None:
            return _points(self.derivatives[0], U, V), _points(self.derivatives[1], U, V)
        out = []
        for k, (lo, hi) in enumerate((self.u, self.v)):
            h = 2e-4 * (hi - lo)
            def at(d, k=k):
                return _points(self.chi, U + d * (k == 0), V + d * (k == 1))
            out.append((8 * (at(h) - at(-h)) - (at(2 * h) - at(-2 * h))) / (12 * h))
        return out[0], out[1]

    def area_element(self, U, V):
        cu, cv = self.tangents(U, V)
        return np.linalg.norm(np.cross(cu, cv), axis=1)

    def charts(self):
        """One chart (sets/regions.Chart): (u, v) on its parameter box, with the area element |chi_u x chi_v| as the
        density (pygeoinf0.quadrature.rule: a tensor Gauss rule in (u, v))."""
        return (Chart((self.u, self.v), (False, False), None, self._to_cartesian, None, self.area_element),)

    def _to_cartesian(self, U, V):
        P = _points(self.chi, np.ravel(U), np.ravel(V))
        return tuple(P[:, k].reshape(np.shape(U)) for k in range(3))

    @property
    def volume(self):
        return float(np.sum(rule(self, 48)[1]))

    @property
    def bounding_box(self):
        if self._box is None:
            t = np.linspace(0, 1, 129)
            U, V = np.meshgrid(self.u[0] + t * (self.u[1] - self.u[0]), self.v[0] + t * (self.v[1] - self.v[0]))
            P = _points(self.chi, U.ravel(), V.ravel())
            self._box = [(float(lo), float(hi)) for lo, hi in zip(P.min(axis=0), P.max(axis=0))]
        return self._box

    decides_membership = False      # a projection onto the surface is not implemented: member is UNKNOWN

    def pieces(self):
        return (self,)

    @property
    def claims(self):
        how = ("from the derivatives given" if self.derivatives is not None
               else "from central differences of fourth order (step 2e-4 of the parameter range)")
        return Claims([Claim("measure: area |chi_u x chi_v| du dv", THEOREM, "the area formula"),
                       Claim("area element " + how, CHECKED if self.derivatives is None else ASSERTED, "not proved")])

    def key(self):
        return ("Surface", id(self.chi), self.u, self.v)

    def same_up_to_null_set(self, other):
        return getattr(other, "key", lambda: None)() == self.key()

    def __eq__(self, other):
        return isinstance(other, Surface) and other.key() == self.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return self.name or f"Surface(χ on [{_fmt(self.u[0])}, {_fmt(self.u[1])}] × [{_fmt(self.v[0])}, " \
                            f"{_fmt(self.v[1])}])"

