"""oCSOLA with a pointwise-box prior (plan 062): BGP over the box B and the slab of a fixed estimator X, per property.

Property j: t_j = T* e_j, the estimator's row x_j (X e = x_j . e), s_j = x_j . (d - b) (b the noise mean),
w_j = z_{1 - alpha/2} sigma_j (sigma_j^2 = x_j C_e x_j), a_j = G* X* e_j, the slab S_j = {m : |<a_j, m> - s_j| <= w_j}.
For EVERY real tau (THEOREM, weak duality: sup over B and S_j of <t_j, m> <= sup_B <t_j - tau a_j, m> + sup over
|g - s_j| <= w_j of tau g):

    sup_{B and S_j} <t_j, m>  <=  phi_j(tau) = tau s_j + |tau| w_j + sigma_B(t_j - tau a_j)

and the lower end is minus the same with -t_j. So only the returned tau needs a certified support: the end is
phi_j(tau*) with sigma_B replaced by the upper end of its certified bracket (convex/pointwise.py), a valid bound
whatever tau* is; tau* decides only how tight it is (inf:427-465).

The search (no certified support): sigma_B(q) = int (u q+ - l q-) dmu is replaced by a fixed quadrature
sum_i om_i (u_i q_i+ - l_i q_i-), q_i = t(x_i) - tau a(x_i). The approximate phi is convex and piecewise linear in
tau, with breakpoints t(x_i) / a(x_i) (slope jump om_i |a(x_i)| (u_i - l_i)) and 0 (jump 2 w); its exact minimiser
is the breakpoint where the cumulated slope changes sign (a weighted median), found for all properties at once by
sorting. The rule, per summand of the box's space (weights w_k of a direct sum):
    L2 of an interval (Lebesgue)    Gauss-Legendre per piece of the common breaks of t, a and the bounds,
                                    oversample x (degree + 1) nodes; bounds that are elements evaluated at the nodes
    L2 of a sphere (harmonics)      the transform grid of degree max(oversample (L + 1), 64) - 1 (Gauss in cos theta x
                                    equispaced longitudes), R^2 times its weights; constant bounds
    R^n                             the coordinates of W q (exact: sigma_B is linear there per sign)
Balls and shells (radial x harmonic realisations) are refused here (no rule is built for them: the rule is the
realisation's capability box_rule, realisations/registry.py and box_integrals.py).

Refinement (refine= k): a bounded scalar search of the certified phi in the window where the approximation stays
within twice its observed error (|certified - approximate| at tau*) of its minimum, every trial certified and the
best kept; the window moves outward when the best trial is at its edge. refine=None: where sigma_B is exact for every
summand (intervals, R^n) the search runs to convergence (at most 100 trials), otherwise k = 0, one certified support
per end.

Stated realisation errors of T's or G's kernels (Functionals with errors): each end moves outwards by
(e_t + |tau| e_a) rho, rho = sup_B ||m|| <= (sum_k w_k |mu_k| max(sup |l_k|, sup |u_k|)^2)^(1/2) (Cauchy-Schwarz),
e_a = sum_i |(W y)_i| e_i as in the ball's route.
"""
import math

import numpy as np

from ..core.claims import DERIVED, THEOREM, Claim, Claims
from ..core.errors import Refused, Unsupported

CHUNK = 1 << 22                  # nodes x properties per block of the search
SPHERE_MIN = 64                  # the least grid degree + 1 on a sphere: |q|'s kinks need the resolution at low L


class BoxOCSOLA(tuple):
    """oCSOLA ends with a box prior: the tuple (lower, upper) (floats for one property, arrays (P,) for several) with

        tau_lower, tau_upper       the tau of each end (the lower end's tau belongs to the problem with -t)
        lower_bracket, upper_bracket   (P, 2): the certified bracket of the dual value at that tau; the returned end
                                   is its outer end (the lower end of lower_bracket, the upper end of upper_bracket)
        approx_lower, approx_upper     the quadrature approximation of the end at that tau (search only)
        supports                   (P, 2) certified support calls per end (lower, upper); 1 + refinements
        widened                    (P, 2) the outward move for stated realisation errors (0 where none stated)
        nodes                      the quadrature's size; claims
    """

    def __new__(cls, lower, upper, **info):
        one = info.pop("one")
        out = tuple.__new__(cls, (float(lower[0]), float(upper[0])) if one else (lower, upper))
        for k, v in info.items():
            setattr(out, k, v)
        return out

    def interval(self, k=0):
        return (self[0], self[1]) if np.ndim(self[0]) == 0 else (float(self[0][k]), float(self[1][k]))

    def __repr__(self):
        lo, hi = (np.atleast_1d(v) for v in self[:2])
        return f"oCSOLA box ends of {len(lo)} properties: " + ", ".join(
            f"[{a:.6g}, {b:.6g}]" for a, b in list(zip(lo, hi))[:6]) + (" ..." if len(lo) > 6 else "")


# ------------------------------------------------------------------------------------------------ quadrature
class Rule:
    """The fixed quadrature of the search (module docstring) for the families fams on the box's space: om, lo, hi
    (n,) and values(fam) (n, N). exact: True when sigma_B is exact on every summand used (intervals, R^n)."""

    def __init__(self, box, fams, oversample=4):
        from ..realisations.registry import capability
        from ..spaces.base import Euclidean
        if oversample < 1:
            raise ValueError("oversample >= 1")
        M = box.space
        spaces = M.summands if box.summed else [M]
        weights = M.weights if box.summed else [1.0]
        self.parts, om, lo, hi = [], [], [], []
        self.exact, self.rho2 = True, 0.0
        for k, H in enumerate(spaces):
            Rs = [self._part(f, k, box) for f in fams]
            if all(R is None for R, _ in Rs):
                continue
            R0 = next(R for R, _ in Rs if R is not None)
            rule = None if isinstance(H, Euclidean) else capability(
                R0, "box_rule", f"oCSOLA with a box on {H}: no search rule for this realisation; the routes are "
                "intervals, spheres and R^n (or bgp with the slab, optimisers/dual/slabs.py)")
            l, u = box.lower[k], box.upper[k]
            if l is None or u is None:
                raise Unsupported(f"oCSOLA with a box needs finite lower and upper bounds on every summand ({H} has an "
                                  "infinite one): sigma_B is +inf for most directions there")
            wk = float(weights[k])
            if rule is not None:                        # the realisation's rule (realisations/bounds/box_integrals.py)
                r = rule(H, Rs, l, u, wk, oversample, SPHERE_MIN)
                w, lv, uv = r.w, r.lo, r.hi
                if not r.exact:
                    self.exact = False
                self.rho2 += r.rho2
                self.parts.append((r.kind, k, r))
            else:
                n = H.dim
                w = np.full(n, wk)
                lv = np.broadcast_to(np.asarray(getattr(l, "values", l), float), (n,)).copy()
                uv = np.broadcast_to(np.asarray(getattr(u, "values", u), float), (n,)).copy()
                Wm = np.asarray(H.apply_metric(np.eye(n)), float)
                self.rho2 += wk * float(np.linalg.eigvalsh(0.5 * (Wm + Wm.T)).max()) * float(
                    np.sum(np.maximum(lv * lv, uv * uv)))
                self.parts.append(("euclidean", k, H))
            om.append(w)
            lo.append(lv)
            hi.append(uv)
        self.box = box
        self.om, self.lo, self.hi = (np.concatenate(v) if v else np.zeros(0) for v in (om, lo, hi))
        self.size = self.om.size

    @staticmethod
    def _part(fam, k, box):
        R = fam.realisation
        if not box.summed:
            return R, None
        p = R.parts[k]
        return (None, None) if p is None else p

    def values(self, fam):
        """(n, N): the members of fam at the nodes (W q on R^n summands)."""
        N = len(fam)
        out = []
        for kind, k, info in self.parts:
            R, idx = self._part(fam, k, self.box)
            cols = np.arange(N) if idx is None else np.asarray(idx, int)
            if kind != "euclidean":
                V = info.values(R, cols, N, CHUNK)
            else:
                H = info
                V = np.zeros((H.dim, N))
                if R is not None:
                    A = np.asarray(R, float).reshape(H.dim, -1)
                    V[:, cols] = np.column_stack([np.asarray(H.apply_metric(A[:, i]), float) for i in range(A.shape[1])])
            out.append(V)
        return np.vstack(out) if out else np.zeros((0, N))


def approx_value(tv, av, tau, s, w, rule):
    """The quadrature approximation of phi at tau ((n, p) values, (p,) tau, s, w): (p,)."""
    q = tv - av * tau[None, :]
    lo, hi = rule.lo[:, None], rule.hi[:, None]
    return tau * s + np.abs(tau) * w + rule.om @ (np.where(q > 0, hi * q, lo * q))


def minimise(tv, av, s, w, rule):
    """The exact minimiser of the approximate phi for every column (module docstring): (tau, value, unbounded), the
    last True where the approximation decreases without bound (then the extreme breakpoint; the feasible set may be
    empty)."""
    om, lo, hi = rule.om[:, None], rule.lo[:, None], rule.hi[:, None]
    ap, an = np.maximum(av, 0.0), np.maximum(-av, 0.0)
    slope0 = s - w + (om * (lo * an - hi * ap)).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(av != 0, tv / av, 0.0)
    jump = om * np.abs(av) * (hi - lo)
    b = np.vstack([b, np.zeros((1, b.shape[1]))])
    jump = np.vstack([jump, 2.0 * np.asarray(w, float)[None, :]])
    order = np.argsort(b, axis=0, kind="stable")
    bs = np.take_along_axis(b, order, 0)
    cum = slope0[None, :] + np.cumsum(np.take_along_axis(jump, order, 0), axis=0)
    ok = cum >= 0
    k = np.where(ok.any(axis=0), ok.argmax(axis=0), b.shape[0] - 1)
    tau = bs[k, np.arange(b.shape[1])]
    unbounded = (slope0 > 0) | ~ok.any(axis=0)
    return tau, approx_value(tv, av, tau, s, w, rule), unbounded


def window(tcol, acol, s, w, rule, tau0, level):
    """[lo, hi] in tau where the approximate phi (one property) is <= level, widened by one breakpoint each side: phi at
    every breakpoint from the cumulated slopes (O(n log n))."""
    om, lo, hi = rule.om, rule.lo, rule.hi
    nz = acol != 0
    b = np.r_[tcol[nz] / acol[nz], 0.0]
    jump = np.r_[om[nz] * np.abs(acol[nz]) * (hi[nz] - lo[nz]), 2.0 * w]
    order = np.argsort(b, kind="stable")
    b, jump = b[order], jump[order]
    slope0 = s - w + float(om @ (lo * np.maximum(-acol, 0.0) - hi * np.maximum(acol, 0.0)))
    after = slope0 + np.cumsum(jump)                          # the slope right of each breakpoint
    v0 = float(approx_value(tcol[:, None], acol[:, None], np.array([b[0]]), np.array([s]), np.array([w]), rule)[0])
    vals = v0 + np.r_[0.0, np.cumsum(after[:-1] * np.diff(b))]
    inside = np.nonzero(vals <= level)[0]
    if not inside.size:
        i0 = i1 = int(np.clip(np.searchsorted(b, tau0), 0, b.size - 1))
    else:
        i0, i1 = int(inside[0]), int(inside[-1])
    a, c = b[max(i0 - 1, 0)], b[min(i1 + 1, b.size - 1)]
    if a == c:
        r = max(abs(tau0), 1.0) * 1e-3
        a, c = tau0 - r, tau0 + r
    return min(a, tau0), max(c, tau0)


# ------------------------------------------------------------------------------------------------ certification
def _end(box, t, a, sign, tau0, s, w, refine, exact, approx0, tcol, acol, rule, rtol):
    """One end: (tau, (lo, hi) of phi at tau, calls). phi's upper end is the returned bound (module docstring)."""
    calls = [0]
    best = [None]

    def phi(tau):
        calls[0] += 1
        sb = box.support(t * sign - a * tau, rtol)
        lo, hi = sb.lower, sb.upper
        lin = tau * s + abs(tau) * w
        v = (lin + lo, lin + hi)
        if best[0] is None or v[1] < best[0][1][1]:
            best[0] = (tau, v)
        return v[1]

    v0 = phi(tau0)
    k = (100 if exact else 0) if refine is None else int(refine)
    if k > 0:
        from ..optimisers.minimise import minimise_scalar
        e = abs(v0 - approx0)
        lev = approx0 + 2.0 * max(e, 1e-12 * max(1.0, abs(v0)))
        lo, hi = window(tcol, acol, s, w, rule, tau0, lev)
        total = 1 + k

        def f(tau):
            if calls[0] >= total:
                raise StopIteration                      # the budget of certified trials is spent
            return phi(float(tau))
        for _ in range(12):
            try:
                minimise_scalar(f, (lo, hi), xatol=1e-12 * max(1.0, abs(lo), abs(hi)), maxiter=k)
            except StopIteration:
                break
            tb = best[0][0]
            span = hi - lo
            if tb - lo <= 1e-6 * span:
                lo, hi = lo - span, lo + 1e-6 * span
            elif hi - tb <= 1e-6 * span:
                lo, hi = hi - 1e-6 * span, hi + span
            else:
                break
            if calls[0] >= total:
                break
    tau, v = best[0]
    return tau, v, calls[0]


def _rho(rule):
    return math.sqrt(max(rule.rho2, 0.0))


def box_ocsola(box, G, noise, T, datum, X, alpha=0.05, refine=None, oversample=4, rtol=None):
    """oCSOLA ends of every property of T with the box prior (module docstring): a BoxOCSOLA."""
    from scipy.stats import norm
    from ..optimisers.dual.exactness import kernel_errors
    from ..execution import map_blocks
    P = T.codomain
    p = P.dim
    if X.codomain.dim != p:
        raise Refused(f"the estimator X gives {X.codomain.dim} values for {p} properties: give one row per property")
    E = P.family(np.eye(p))
    tf = T.adjoint(E)
    yf = X.adjoint(X.codomain.family(np.eye(p)))
    af = G.adjoint(yf)
    z = norm.ppf(1 - alpha / 2)
    cov = np.atleast_2d(np.asarray(X(noise).covariance.matrix(), float))
    w = z * np.sqrt(np.maximum(np.diag(cov), 0.0))
    s = np.asarray(X(datum).values, float).ravel()
    if not noise.zero_mean:
        s = s - np.asarray(X(noise.mean).values, float).ravel()
    rule = Rule(box, [tf, af], oversample)
    tv, av = rule.values(tf), rule.values(af)
    taus, approx = np.zeros((p, 2)), np.zeros((p, 2))
    flagged = np.zeros((p, 2), bool)
    step = max(1, CHUNK // max(1, rule.size))
    for col, sign in ((0, -1.0), (1, 1.0)):
        for i in range(0, p, step):
            sl = slice(i, i + step)
            ta, va, un = minimise(sign * tv[:, sl], av[:, sl], s[sl], w[sl], rule)
            taus[sl, col], approx[sl, col], flagged[sl, col] = ta, va, un

    def one(j, col):
        sign = -1.0 if col == 0 else 1.0
        return _end(box, tf[j], af[j], sign, float(taus[j, col]), float(s[j]), float(w[j]), refine, rule.exact,
                    float(approx[j, col]), sign * tv[:, j], av[:, j], rule, rtol)
    res = dict(map_blocks(one, (), [(j, c) for j in range(p) for c in (0, 1)]))
    brk = np.zeros((p, 2, 2))
    calls = np.zeros((p, 2), int)
    for (j, c), (tau, v, n) in res.items():
        taus[j, c], brk[j, c], calls[j, c] = tau, v, n
    widen = np.zeros((p, 2))
    te, ge = kernel_errors(T), kernel_errors(G)
    if te is not None or ge is not None:
        rho = _rho(rule)
        et = np.zeros(p) if te is None else np.broadcast_to(np.asarray(te, float), (p,))
        if ge is None:
            ea = np.zeros(p)
        else:
            D = G.codomain
            Y = np.column_stack([np.asarray(D.apply_metric(yf[j].values), float) for j in range(p)])
            ea = np.abs(Y).T @ np.asarray(ge, float)
        widen = (et[:, None] + np.abs(taus) * ea[:, None]) * rho
    lower = -(brk[:, 0, 1] + widen[:, 0])
    upper = brk[:, 1, 1] + widen[:, 1]
    lower_bracket = np.c_[-brk[:, 0, 1], -brk[:, 0, 0]] - widen[:, :1]
    upper_bracket = brk[:, 1] + widen[:, 1:]
    kinds = sorted({k for k, _, _ in rule.parts})
    cl = [Claim("each end is a valid bound at its tau, whatever tau is: sup over B and the slab of <t, m> <= tau s + "
                "|tau| w + sigma_B(t - tau a) (weak duality), with sigma_B's certified upper end", THEOREM,
                "inf:427-465; convex/pointwise.py", key="valid ends"),
          Claim(f"tau minimises a quadrature approximation of the dual ({rule.size} nodes on {', '.join(kinds)}); the "
                "end may exceed the dual optimum inf over tau by up to the approximation's error near it (not "
                "certified)", DERIVED, "inference/ocsola_box.py"),
          Claim(f"the slab |x . (d - b) - <a, m>| <= z sigma holds with probability {1 - alpha:g} for each property "
                "under the Gaussian noise; the intervals of several properties are not simultaneous", THEOREM,
                "inf:427-465")]
    if refine is None and rule.exact:
        cl.append(Claim("sigma_B is exact on this space: tau refined by a bounded scalar search on the certified dual "
                        "(every trial a valid bound, the least kept)", DERIVED, "inference/ocsola_box.py"))
    if flagged.any():
        cl.append(Claim(f"{int(flagged.sum())} ends: the approximation decreases without bound (the box and the slab "
                        "may not meet); the extreme breakpoint was certified instead", DERIVED,
                        "inference/ocsola_box.py"))
    if widen.any():
        cl.append(Claim("ends moved outwards by (e_t + |tau| e_a) sup_B ||m|| for the kernels' stated realisation "
                        "errors", DERIVED, "Cauchy-Schwarz (module docstring)"))
    return BoxOCSOLA(lower, upper, one=(p == 1), tau_lower=taus[:, 0], tau_upper=taus[:, 1],
                     lower_bracket=lower_bracket, upper_bracket=upper_bracket, approx_lower=-approx[:, 0],
                     approx_upper=approx[:, 1], supports=calls, widened=widen, nodes=rule.size, claims=Claims(cl))
