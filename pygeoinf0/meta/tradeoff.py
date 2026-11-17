"""Trade-off curves, knee rules and operating points in closed form (plan 009 G11 (1); EXP-030 DESIGN.md G11, S19,
decision 11).

    tc = pg.meta.TradeOff(dec, T, datum=None)        dec = pg.decompose(G, noise) (SOLA's; or with middle=prior: the
                                                     PLI estimator family with gamma = nu^2 / a^2)
    tc.at(gammas)                                    sigma, variance, mismatch, trm, delivered, their derivatives and
                                                     (with a datum) the estimates for every gamma: one dec.sola call
    k = pg.meta.Knee(rule="chord", axes=("trm", "log sigma"), rescale="unit", side="convex", refine="continuous")
    k.find(tc, lo, hi)                               a KneeResult: the knee gamma, its point, the distance
    pg.meta.OperatingPoints(tc, lo, hi, knee=k, factor=10.0)    the knee and its flanks knee / 10, knee * 10

The curve. For every gamma > 0, X_gamma = T S G* (G S G* + gamma C)^-1 minimises the SOLA objective
||t - G* X*||^2_S + gamma X C X* over all linear estimators X (a convex quadratic in X, whose stationarity condition is
X (G S G* + gamma C) = T S G*), so (variance, mismatch) is on the Pareto front of the two criteria. In the eigenbasis
(NOTES.md section 4) variance = sum beta^2 / (Lam + gamma)^2 and mismatch = ||t||^2 - 2 sum beta^2 / (Lam + gamma) +
sum Lam beta^2 / (Lam + gamma)^2, with d variance / d gamma = -2 sum beta^2 / (Lam + gamma)^3 <= 0 and d mismatch /
d gamma = -gamma d variance / d gamma >= 0: sigma decreases and the TRM increases along the curve. The TRM (target
resolution misfit) is the relative mismatch sqrt(mismatch / ||t||^2), as in the thesis (sola:43-80). sigma and the TRM
do not depend on the datum, so neither does a knee chosen from them.

The knee rule "chord" (the thesis's rule r3, sweep/sola_pareto_sweep.py:275-339; EXP-030 p4a_sola.py): in the plane of
the declared axes (default (TRM, log10 sigma)), each axis rescaled to [0, 1] over the curve (rescale="unit"), the knee
is the point of largest distance from the chord joining the curve's ends, among the points on the declared side:
"convex" is the side of the point (chord midpoint) - (1, 1), towards smaller values of both axes (r3 literally);
"concave" the other; "either" the largest |distance|. When no point is strictly on the declared side r3 takes the
largest |distance| (stated in the claims).
Two realisations of the rule (refine=):
  "continuous"   decision 11: the maximiser over gamma in [lo, hi] of the same distance on the closed-form curve, by a
                 scan of `samples` points in log gamma and Brent's method in the bracket of the best sample; rescaled
                 over [lo, hi] (for the TRM and sigma the ends of the range: both are monotone in gamma, above)
  (points, f)    the thesis's grid rule: `grid` points geometric in [lo, hi], the knee among them, `points` more over
                 [knee / f, knee * f] merged into the grid, the knee again (rescaled over the rows of the grid); the
                 flanks of OperatingPoints are then snapped to the grid (S19, the thesis realisation)
"""
import math

import numpy as np

from ..core.claims import ASSERTED, CHECKED, DERIVED, Claim
from .record import Choice, Record, column, inherited

AXES = ("trm", "log trm", "mismatch", "sigma", "log sigma", "variance", "log variance")


class TradeOff(Record):
    """The closed-form trade-off curve of a decomposition and a property map (module docstring). T: a property map
    into R^p (or a model element, or pairings=(cross, tt)); datum: optional, only for the estimates."""

    kind = "trade-off curve"
    axes_use_datum = False          # sigma and the TRM never depend on the datum

    def __init__(self, dec, T=None, datum=None, pairings=None):
        self.dec, self.T, self.datum, self.pairings = dec, T, datum, pairings
        self._record(datum is not None, "the datum enters the estimates only; sigma, the TRM and the mismatch do not "
                     "depend on it" if datum is not None else "no datum given: sigma, TRM and mismatch only", [
                         Claim("each gamma gives the minimiser of mismatch + gamma variance over linear estimators: a "
                               "point of the Pareto front; d mismatch / d variance = -gamma", DERIVED,
                               "SOLA's objective is a convex quadratic in X (module docstring; NOTES.md section 4)"),
                         Claim("sigma decreases and the TRM increases with gamma", DERIVED,
                               "d variance / d gamma = -2 sum beta^2 / (Lam + gamma)^3 <= 0 and d mismatch / d gamma = "
                               "-gamma d variance / d gamma"),
                         Claim("from one data-space decomposition", DERIVED, dec.describe())])

    def at(self, gamma, fields=False):
        """dec.sola at gamma (a number or an array) with sigma (= std) and the TRM sqrt(mismatch / ||t||^2) added:
        arrays of shape (len(gamma), p), or (p,) for one gamma."""
        r = self.dec.sola(self.T, gamma, self.datum, fields=fields, pairings=self.pairings)
        r.sigma = r.std
        r.trm = np.sqrt(np.maximum(r.mismatch, 0.0) / np.asarray(r.target, float))
        r.claims = type(r.claims)(list(self.claims) + list(r.claims))
        return r

    def values(self, gamma, column_index=0):
        """The curve's quantities (trm, mismatch, sigma, variance) of one property at an array of gamma."""
        r = self.at(np.atleast_1d(np.asarray(gamma, float)))
        return {k: column(getattr(r, k), column_index) for k in ("trm", "mismatch", "sigma", "variance")}

    def knee(self, lo, hi, rule=None, column=0):
        """The knee of one property by a declared rule (default Knee()): a KneeResult."""
        return (rule or Knee()).find(self, lo, hi, column)

    def operating_points(self, lo, hi, knee=None, factor=10.0, column=0):
        return OperatingPoints(self, lo, hi, knee=knee, factor=factor, column=column)

    def _show(self):
        return []


def _axis(vals, name):
    if name.startswith("log "):
        return np.log10(np.asarray(vals[name[4:]], float))
    return np.asarray(vals[name], float)


class Knee:
    """A declared knee rule (module docstring): rule="chord" (r3), axes, rescale ("unit" or None), side ("convex",
    "concave", "either"), refine ("continuous", decision 11, or (points, factor): the thesis's grid rule), grid (the
    coarse grid's size for the grid rule), samples (the scan of the continuous rule). Its choices are claims."""

    def __init__(self, rule="chord", axes=("trm", "log sigma"), rescale="unit", side="convex", refine="continuous",
                 grid=53, samples=2001):
        if rule != "chord":
            raise ValueError("the knee rule is 'chord' (the thesis's r3); others are not implemented")
        if len(axes) != 2 or any(a not in AXES for a in axes):
            raise ValueError(f"axes: two of {AXES}")
        if rescale not in ("unit", None):
            raise ValueError("rescale is 'unit' (each axis to [0, 1] over the curve) or None")
        if side not in ("convex", "concave", "either"):
            raise ValueError("side is 'convex', 'concave' or 'either'")
        if refine != "continuous" and not (isinstance(refine, tuple) and len(refine) == 2 and refine[1] > 1):
            raise ValueError("refine is 'continuous' or (points, factor > 1)")
        self.rule, self.axes, self.rescale, self.side, self.refine = rule, tuple(axes), rescale, side, refine
        self.grid, self.samples = int(grid), int(samples)

    @property
    def claims(self):
        how = ("continuous: the maximiser over gamma on the closed-form curve (decision 11)" if self.refine ==
               "continuous" else f"the grid rule: {self.grid} points, then {self.refine[0]} over a factor "
               f"{self.refine[1]:g} either side of the coarse knee (the thesis realisation, S19)")
        return [Claim(f"knee rule '{self.rule}' in the axes ({self.axes[0]}, {self.axes[1]}), rescale {self.rescale}, "
                      f"side {self.side}; {how}", ASSERTED, "a declared meta-layer choice (el:59; the thesis's r3, "
                      "sola:70-80)", key="knee rule")]

    def __repr__(self):
        return (f"Knee({self.rule}, axes={self.axes}, rescale={self.rescale}, side={self.side}, "
                f"refine={self.refine})")

    def _uv(self, curve, g, j):
        vals = curve.values(g, j) if hasattr(curve, "values") else curve(np.asarray(g, float))
        return _axis(vals, self.axes[0]), _axis(vals, self.axes[1])

    def _side_sign(self, p0, p1):
        q = 0.5 * (p0 + p1) - 1.0
        s = np.sign(_signed(q[0], q[1], p0, p1))
        return -s if self.side == "concave" else s

    def _rows(self, u, v):
        """The index of the knee among rows sorted by gamma (r3, rescaled over the rows)."""
        x, y = (_unit(u), _unit(v)) if self.rescale else (u, v)
        p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
        s = _signed(x, y, p0, p1)
        if self.side == "either":
            return int(np.argmax(np.abs(s))), True
        conv = np.where(np.sign(s) == self._side_sign(p0, p1), np.abs(s), -np.inf)
        if np.isfinite(conv).any() and conv.max() > 0:
            return int(np.argmax(conv)), True
        return int(np.argmax(np.abs(s))), False

    def find(self, curve, lo, hi, column=0, family=None):
        """The knee of the curve over [lo, hi] (curve: a TradeOff, or a callable gamma array -> dict of the axes'
        base quantities): a KneeResult. family: the hyperparameter family searched (pg.meta.knee passes the problem's;
        default a damping gamma in [lo, hi] of the trade-off curve)."""
        from .families import damping
        lo, hi = float(lo), float(hi)
        if not 0 < lo < hi:
            raise ValueError("0 < lo < hi")
        family = damping("the trade-off curve", lo=lo, hi=hi) if family is None else family
        if self.refine == "continuous":
            return self._continuous(curve, lo, hi, column, family)
        coarse = np.geomspace(lo, hi, self.grid)
        u, v = self._uv(curve, coarse, column)
        kc = coarse[self._rows(u, v)[0]]
        pts, f = self.refine
        seen = set(coarse)
        grid = np.sort(np.r_[coarse, [x for x in np.geomspace(kc / f, kc * f, int(pts)) if x not in seen]])
        u, v = self._uv(curve, grid, column)
        i, on_side = self._rows(u, v)
        x, y = (_unit(u), _unit(v)) if self.rescale else (u, v)
        dist = abs(float(_signed(x[i], y[i], np.array([x[0], y[0]]), np.array([x[-1], y[-1]]))))
        return KneeResult(self, curve, float(grid[i]), (u[i], v[i]), dist, lo, hi, column, grid=grid,
                          on_side=on_side, how=f"the grid rule on {len(grid)} rows ({self.grid} coarse + "
                          f"{len(grid) - self.grid} refined)", status=DERIVED, family=family)

    def _continuous(self, curve, lo, hi, j, family):
        from ..optimisers.minimise import minimise_scalar
        g = np.geomspace(lo, hi, self.samples)
        u, v = self._uv(curve, g, j)
        if self.rescale:
            box = (min(u.min(), u[0], u[-1]), max(u.max(), u[0], u[-1]), min(v.min(), v[0], v[-1]),
                   max(v.max(), v[0], v[-1]))
        else:
            box = (0.0, 1.0, 0.0, 1.0)

        def sc(a, b):
            return (a - box[0]) / (box[1] - box[0]), (b - box[2]) / (box[3] - box[2])
        p0, p1 = np.array(sc(u[0], v[0])), np.array(sc(u[-1], v[-1]))
        sgn = self._side_sign(p0, p1)
        s = _signed(*sc(u, v), p0, p1)
        on_side = self.side == "either" or bool(np.max(sgn * s) > 0)
        score = (lambda d: np.abs(d)) if (self.side == "either" or not on_side) else (lambda d: sgn * d)
        sv = score(s)
        i = int(np.argmax(sv))
        lg = np.log10(g)

        def f(t):
            a, b = self._uv(curve, np.array([10.0 ** t]), j)
            return float(score(_signed(*sc(a[0], b[0]), p0, p1)))
        a, b = lg[max(i - 1, 0)], lg[min(i + 1, len(g) - 1)]
        best, fbest = lg[i], float(sv[i])
        if b > a:
            r = minimise_scalar(lambda t: -f(t), (a, b), 1e-12)
            if -r.fun >= fbest:
                best, fbest = float(r.x), float(-r.fun)
        gam = float(10.0 ** best)
        uk, vk = self._uv(curve, np.array([gam]), j)
        return KneeResult(self, curve, gam, (uk[0], vk[0]), abs(fbest), lo, hi, j, grid=None, on_side=on_side,
                          how=f"a scan of {self.samples} points in log gamma, then Brent's method in the bracket of the "
                          "best sample (a global maximum unless the distance has another peak narrower than the scan "
                          "step)", status=CHECKED, family=family)


def _unit(a):
    a = np.asarray(a, float)
    return (a - a.min()) / (a.max() - a.min())


def _signed(x, y, p0, p1):
    """The signed distance of (x, y) from the line through p0 and p1 (positive to the left of p0 -> p1)."""
    ch = p1 - p0
    return (ch[0] * (y - p0[1]) - ch[1] * (x - p0[0])) / math.hypot(*ch)


class KneeResult(Choice):
    """The knee of a curve by a declared rule (a choice rule's result, meta/record.py Choice): gamma, point (the two
    axis values), distance (from the chord, rescaled units), the range, grid (the rows of the grid rule; None for the
    continuous rule), the family and the claims."""

    kind = "knee"

    def __init__(self, rule, curve, gamma, point, distance, lo, hi, column, grid, on_side, how, status, family):
        self.point, self.distance = tuple(map(float, point)), distance
        self.lo, self.hi, self.column, self.grid = lo, hi, column, grid
        used = bool(getattr(curve, "axes_use_datum", False))
        cl = list(rule.claims) + [Claim(f"knee gamma = {gamma:.12g} over [{lo:g}, {hi:g}]", status, how)]
        if not on_side:
            cl.append(Claim(f"no point strictly on the {rule.side} side of the chord: the largest |distance| was "
                            "taken (as r3 does)", DERIVED, "the rule's fallback"))
        self._choose("the knee rule", family, {"gamma": gamma}, used, "the knee is chosen from the curve's axes, which "
                     "do not depend on the datum" if not used else "the curve's axes depend on the datum",
                     cl + inherited(getattr(curve, "claims", [])))
        self.rule = rule

    def snap(self, g):
        """The grid row nearest to g in log gamma (the grid rule's flanks); g itself for the continuous rule."""
        if self.grid is None:
            return float(g)
        return float(self.grid[np.argmin(np.abs(np.log10(self.grid) - math.log10(g)))])

    def _show(self):
        return ["gamma", "distance"]


class OperatingPoints(Record):
    """The knee and its flanks knee / factor (the resolution flank: smaller TRM, larger sigma) and knee * factor (the
    variance flank), snapped to the grid under the grid rule (sola:70-80, S19), evaluated on the curve: gamma, sigma,
    trm (each a dict by "resolution", "knee", "variance") and, with a datum, estimate."""

    kind = "operating points"
    NAMES = ("resolution", "knee", "variance")

    def __init__(self, tradeoff, lo, hi, knee=None, factor=10.0, column=0):
        if factor <= 1:
            raise ValueError("the flank factor is > 1")
        k = knee if isinstance(knee, KneeResult) else (knee or Knee()).find(tradeoff, lo, hi, column)
        self.knee, self.factor, self.column = k, float(factor), column
        g = np.array([k.snap(k.gamma / factor), k.gamma, k.snap(k.gamma * factor)])
        r = tradeoff.at(g)
        self.gamma = dict(zip(self.NAMES, map(float, g)))
        self.sigma = dict(zip(self.NAMES, column_values(r.sigma, column)))
        self.trm = dict(zip(self.NAMES, column_values(r.trm, column)))
        self.estimate = None if r.estimate is None else dict(zip(self.NAMES, column_values(r.estimate, column)))
        self._record(tradeoff.datum is not None, "the points are chosen without the datum; the estimates at them use "
                     "it" if tradeoff.datum is not None else "the points are chosen without the datum",
                     [Claim(f"flanks knee / {factor:g} and knee * {factor:g}" + (", snapped to the grid" if k.grid is not
                                                                                 None else ""), ASSERTED,
                            "a declared meta-layer choice (sola:70-80)", key="flanks")] + inherited(k.claims))

    def _show(self):
        return ["gamma"]


def column_values(a, j):
    return [float(x) for x in column(a, j)]
