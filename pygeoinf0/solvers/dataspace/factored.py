"""Closed forms for property families T = Phi tau (properties/families.py) and for many properties at once (plan 009 G4;
EXP-030 DESIGN G4, NOTES.md section 4).

With the eigen coordinates beta_tau = U^T F <g, S tau*> of the B block functionals (an n_eff x B matrix) every
closed form of solvers/dataspace/closed_forms.py becomes a synthesis: for property p, beta(p) = beta_tau Phi_p^T, so

    estimate(p)  = Phi_p . (beta_tau^T x)                       x = delta / (Lam + gamma)  (SOLA), etc.
    variance(p)  = Phi_p Q Phi_p^T                              Q = beta_tau^T diag(w) beta_tau  (B x B)

computed for all p without a p x p matrix (diagonal Q, as for block-diagonal problems with one block per harmonic:
Phi^2 diag(Q)). Covariances between properties stay lazy (FactoredCovariance: Phi V Phi^T plus the exact diagonal).
The target norms ||T_p||_S^2 come from the family (exact from the weight where known).

Properties of a plain map T with more than COVARIANCE_DENSE rows: the posterior returns variances and a lazy
covariance operator (LowRankCovariance), not a dense p x p matrix.

ellipsoid_intervals gives the BGP bounds of every property at once through optimisers/dual/spectral.py (vectorised over
directions, each one certified as before).
"""
import numpy as np

from ...core.claims import CHECKED, DERIVED, Claim, Claims
from ...spaces.base import Euclidean
from ...operators.linear import LinearOperator
from .quadratic import rowquad

COVARIANCE_DENSE = 2000     # above this many properties a posterior covariance stays lazy (variances are returned)
BATCH = 4096                # BGP directions per batch of intervals (n_eff x BATCH doubles: 262 MB at n_eff = 8002)


class FactoredTargets:
    """The pairings of a factored family: cross (n x B) and tt (B x B) of its blocks, Phi (p x B), the target norms."""

    def __init__(self, cross, tt, Phi, norms2, T, claims=()):
        self.cross, self.tt, self.Phi, self.T = cross, tt, Phi, T
        self.tt_diag = np.asarray(norms2, float)
        self.beta, self.claims = None, list(claims)

    def diag(self):
        return self.tt_diag


def factored_targets(dec, T, S):
    Phi, tau = T.factors
    sub = dec.targets(tau)
    nrm = T.target_norms2(S)
    if nrm is None:
        n2, cl = rowquad(Phi, sub.tt), None
    else:
        n2, cl = nrm
    out = FactoredTargets(sub.cross, np.asarray(sub.tt if not isinstance(sub.tt, LinearOperator) else sub.tt.matrix(),
                                                float), Phi, n2, T, [] if cl is None else [cl])
    from ...optimisers.dual.exactness import target_errors
    se = target_errors(sub)                                   # the blocks' stated errors through Phi (exactness.py)
    out.err = None if se is None else np.ravel(np.asarray(abs(Phi) @ se, float))
    return out


def _claims(dec, tg, what):
    return dec._result_claims(what, "synthesis over the blocks of a factored family (plan 009 G4, EXP-030 DESIGN G4)") \
        + [Claim(f"{tg.Phi.shape[0]} properties from {tg.Phi.shape[1]} block functionals: no p x p matrix", DERIVED,
                 "T = Phi tau")] + list(tg.claims)


def sola(dec, tg, gamma, datum):
    from .closed_forms import Result
    g = np.asarray(gamma, float)
    scalar = g.ndim == 0
    g = np.atleast_1d(g).ravel()
    if np.any(~(g > 0)):
        raise ValueError("gamma must be positive")
    B = tg.cross.shape[1]
    cols = tg.cross if datum is None else np.c_[tg.cross, dec._delta(datum)]
    lam, C, _ = dec._spec(cols)
    beta = C[:, :B]
    Phi = tg.Phi
    R = 1.0 / (lam[:, None] + g[None, :])                                      # (k, G)
    diag = not np.any(beta.T @ beta - np.diag(np.einsum("kb,kb->b", beta, beta)))
    est = None if datum is None else ((C[:, B][:, None] * R).T @ beta) @ Phi.T   # (G, p)

    def quad(w):                                                              # (G, p): Phi_p (beta^T diag(w_g) beta) Phi_p
        if diag:
            return (w.T @ (beta * beta)) @ (Phi * Phi).T
        return np.array([rowquad(Phi, beta.T @ (w[:, i][:, None] * beta)) for i in range(w.shape[1])])
    var = quad(R * R)
    S1 = quad(R)
    dlv = quad(R * R * lam[:, None])
    S3 = quad(R * R * R)
    mis = tg.tt_diag[None, :] - 2.0 * S1 + dlv
    pick = (lambda a: a[0]) if scalar else (lambda a: a)
    return Result("SOLA", _claims(dec, tg, "SOLA in closed form in gamma, for a factored family"),
                  gamma=g[0] if scalar else g, estimate=None if est is None else pick(est), std=np.sqrt(pick(var)),
                  variance=pick(var), mismatch=pick(mis), delivered=pick(dlv), target=tg.tt_diag, cross=pick(S1),
                  d_variance=pick(-2.0 * S3), d_mismatch=pick(2.0 * g[:, None] * S3), components=None)


def posterior(dec, tg, datum, a, nu, prior_mean):
    """The property posterior of a factored family: means, variances, lazy covariances (one per lane, when one)."""
    from ...probability.lazy_covariance import FactoredCovariance       # the engine's one import from above (its results)
    from .closed_forms import Result
    A, N = np.broadcast_arrays(np.asarray(a, float), np.asarray(nu, float))
    shape = A.shape
    A, N = A.ravel(), N.ravel()
    B = tg.cross.shape[1]
    lam, C, _ = dec._spec(np.c_[tg.cross, dec._delta(datum, prior_mean)])
    beta, delta = C[:, :B], C[:, B]
    Phi = tg.Phi
    P = tg.T.codomain if tg.T is not None else Euclidean(Phi.shape[0])
    S = (A ** 2)[None, :] * lam[:, None] + (N ** 2)[None, :]
    mean = (A ** 2)[:, None] * ((delta[:, None] / S).T @ beta) @ Phi.T
    if prior_mean is not None and tg.T is not None:
        mean = mean + tg.T(prior_mean).values[None, :]
    trunc = rowquad(Phi, tg.tt)
    extra = np.maximum(tg.tt_diag - trunc, 0.0)
    covs, var = [], []
    for i in range(len(A)):
        Q = beta.T @ (beta / S[:, i][:, None])
        V = A[i] ** 2 * tg.tt - A[i] ** 4 * Q
        var.append(A[i] ** 2 * tg.tt_diag - A[i] ** 4 * rowquad(Phi, Q))
        covs.append(FactoredCovariance(P, Phi, V, A[i] ** 2 * extra if np.any(extra) else None, name="C_post",
                                       claims=[_lazy_claim(extra)]))
    var = np.maximum(np.array(var), 0.0)
    out = Result("posterior", _claims(dec, tg, "the property posterior of a factored family for each (a, nu)"),
                 a=A.reshape(shape), nu=N.reshape(shape), mean=mean.reshape(shape + (Phi.shape[0],)),
                 covariance=covs[0] if len(A) == 1 else None, variance=var.reshape(shape + (Phi.shape[0],)),
                 prior_variance=((A ** 2)[:, None] * tg.tt_diag[None, :]).reshape(shape + (Phi.shape[0],)))
    out.covariances = covs
    out.prior_covariance = FactoredCovariance(P, Phi, A[0] ** 2 * tg.tt, A[0] ** 2 * extra if np.any(extra) else None,
                                              name="C_prior", claims=[_lazy_claim(extra)])
    out.std = np.sqrt(out.variance)
    out.T = tg.T
    if getattr(dec, "error", None) is not None:          # the randomised route: Phi_p (beta^T diag(1/S^2) beta) Phi_p^T
        quad = np.array([rowquad(Phi, beta.T @ (beta / (S[:, i] ** 2)[:, None])) for i in range(len(A))])
        _bound(dec, out, (A ** 6)[:, None] * quad, shape, A / N)
    return out


def _bound(dec, out, quad, shape, rho):
    """variance_bound = a^6 ||W - W_k|| ||R w||^2 / (1 - rho^2 ||W - W_k||) per property (closed_forms.posterior; plan
    005 P5, two-sided since plan 054 P3): the exact variance lies in [variance - variance_bound, variance +
    variance_bound]."""
    from .partial import std_error, two_sided
    fac = two_sided(dec.error, np.asarray(rho, float) ** 2)
    out.variance_bound = (dec.error * fac[:, None] * quad).reshape(out.variance.shape)
    out.std_bound = std_error(out.variance, out.variance_bound)
    out.claims = Claims(list(out.claims) + [Claim(
        f"variances from the rank-{len(dec.lam)} Nystrom approximation: each within variance_bound of the exact one, "
        f"on either side (largest std bound {float(np.max(out.std_bound, initial=0.0)):.3g})", CHECKED,
        f"||W - W_k|| <= {dec.error:.3g} with probability >= 1 - 1e-10; |R - (W + I)^-1| <= ||E|| ||R w||^2 / (1 - ||E||) "
        "per property (the resolvent identity; W_k <= W only to rounding in double, plan 054 P3), vectorised over the "
        "family by its factors")])


def _lazy_claim(extra):
    tail = "" if not np.any(extra) else (" plus a diagonal for the degrees beyond the factors (the exact variances; the "
                                         "neglected off-diagonal part is bounded by sqrt(e_p e_q), Cauchy-Schwarz)")
    return Claim("covariance kept lazy: Phi V Phi^T" + tail, DERIVED, "T = Phi tau; no p x p matrix is formed")


def ellipsoid_intervals(dec, tg, a, r, datum, centre=None):
    """[lower, upper] of every property over {||m - m0||_(S^-1) <= a} and the noise ball of radius r, certified
    direction by direction (optimisers/dual/spectral.py). lower and upper are the computed ends; lower_bracket and
    upper_bracket ((p, 2) arrays) the certified brackets of each end, widened by WIDEN x E where the general rule made
    them BRACKET (optimisers/dual/exactness.py), and error_lower, error_upper that E (nan where the rule did not judge)."""
    from .closed_forms import Result
    prob = dec.ball_problem(dec._delta(datum, centre), a, r, centre=centre)
    p = tg.Phi.shape[0] if isinstance(tg, FactoredTargets) else tg.cross.shape[1]
    Tm0 = tg.T(centre).values if (centre is not None and tg.T is not None) else np.zeros(p)
    lo, hi, st_lo, st_hi, redo, br = intervals(prob, dec._beta(tg), tg, Tm0)
    return Result("ellipsoid intervals", interval_claims(dec, p, redo), lower=lo, upper=hi, status_lower=st_lo,
                  status_upper=st_hi, **br)


def interval_claims(dec, p, redo):
    return dec._result_claims(
        f"BGP bounds of {p} properties, vectorised over directions; every bound certified (primal witness and dual value "
        f"evaluated exactly); {redo} of {2 * p} directions solved again one by one", "optimisers/dual/spectral.py")


def intervals(prob, beta, tg, Tm0, how=None):
    """(lower, upper, statuses, statuses, re-solved count, brackets) of every property (the coordinate directions of P,
    standard metric) for a problem in the eigenbasis (optimisers/dual/spectral.SpectralBall); how: their estimability
    (core/estimable.py). brackets: lower_bracket, upper_bracket ((p, 2): the certified bracket of each end, widened
    where the general rule made it BRACKET) and error_lower, error_upper (E of the rule, optimisers/dual/exactness.py)."""
    from ...optimisers.dual.spectral import Directions, bounds, signed_bounds
    from ...execution import in_worker, workers
    p = len(Tm0)
    cc = tg.diag()
    from ...optimisers.dual.exactness import target_errors
    te = target_errors(tg)                                       # the targets' stated errors (optimisers/dual/exactness.py)
    lo, hi, st_lo, st_hi, redo = np.empty(p), np.empty(p), np.empty(p, object), np.empty(p, object), 0
    lb, ub, el, eu = np.empty((p, 2)), np.empty((p, 2)), np.empty(p), np.empty(p)

    def put(sl, up, down):
        """up: the bound along +e (the upper end), down: along -e (the lower end, negated)."""
        v, a, b, st, k, E = up
        hi[sl], ub[sl, 0], ub[sl, 1], st_hi[sl], eu[sl] = v, a, b, st, E
        v, a, b, st, k2, E = down
        lo[sl], lb[sl, 0], lb[sl, 1], st_lo[sl], el[sl] = -v, -b, -a, st, E
        return k + k2
    if workers() <= 1 or in_worker():                             # serial: no shared buffer, no pool (as before it)
        for s in range(0, p, BATCH):
            sl = slice(s, min(p, s + BATCH))
            Bm = beta @ tg.Phi[sl].T if isinstance(tg, FactoredTargets) else beta[:, sl]
            ts = None if te is None else te[sl]
            up, down = (bounds(prob, Bm, cc[sl], Tm0[sl], how, True, ts),
                        bounds(prob, -Bm, cc[sl], -Tm0[sl], how, True, ts))
            redo += put(sl, up, down)
    else:
        with Directions(prob, beta.shape[0], min(p, BATCH)) as pool:  # one worker pool for every batch and both signs
            for s in range(0, p, BATCH):
                sl = slice(s, min(p, s + BATCH))
                Bm = beta @ tg.Phi[sl].T if isinstance(tg, FactoredTargets) else beta[:, sl]
                redo += put(sl, *signed_bounds(prob, Bm, cc[sl], Tm0[sl], directions=pool, how=how, error=True,
                                               terr=None if te is None else te[sl]))
    return lo, hi, st_lo, st_hi, redo, dict(lower_bracket=lb, upper_bracket=ub, error_lower=el, error_upper=eu)


def direction(dec, tg, w):
    """(eigen coordinates, <c, S c>) of the property direction w (in P's coordinates, metric applied)."""
    if isinstance(tg, FactoredTargets):
        z = tg.Phi.T @ w
        extra = np.maximum(tg.tt_diag - rowquad(tg.Phi, tg.tt), 0.0)
        return dec._beta(tg) @ z, float(z @ tg.tt @ z + extra @ (w * w))
    ww = float(w @ tg.tt @ w) if tg.tt is not None else float(w @ tg.tt_op(tg.T.codomain(w)).values)
    return dec._beta(tg) @ w, ww


def posterior_many(dec, tg, datum, a, nu, prior_mean):
    """The posterior of a plain map with more than COVARIANCE_DENSE properties: means and variances for every (a, nu),
    a lazy covariance (LowRankCovariance) for a single lane."""
    from ...probability.lazy_covariance import LowRankCovariance        # the engine's one import from above (its results)
    from .closed_forms import Result
    A, N = np.broadcast_arrays(np.asarray(a, float), np.asarray(nu, float))
    shape = A.shape
    A, N = A.ravel(), N.ravel()
    p = tg.cross.shape[1]
    lam, C, _ = dec._spec(np.c_[tg.cross, dec._delta(datum, prior_mean)])
    beta, delta = C[:, :p], C[:, p]
    S = (A ** 2)[None, :] * lam[:, None] + (N ** 2)[None, :]
    mean = (A ** 2)[:, None] * (beta.T @ (delta[:, None] / S)).T
    if prior_mean is not None and tg.T is not None:
        mean = mean + tg.T(prior_mean).values[None, :]
    var = (A ** 2)[:, None] * tg.tt_diag[None, :] - (A ** 4)[:, None] * ((1.0 / S).T @ (beta * beta))
    var = np.maximum(var, 0.0)
    P = tg.T.codomain
    cov = LowRankCovariance(P, tg.tt_op, beta, 1.0 / S[:, 0], A[0] ** 2, var[0]) if len(A) == 1 else None
    out = Result("posterior", dec._result_claims(
        f"the property posterior for each (a, nu): {p} properties, variances and a lazy covariance (p > "
        f"COVARIANCE_DENSE = {COVARIANCE_DENSE})", "Gaussian conditioning in the eigenbasis, NOTES.md section 4"),
        a=A.reshape(shape), nu=N.reshape(shape), mean=mean.reshape(shape + (p,)), covariance=cov,
        variance=var.reshape(shape + (p,)), prior_variance=((A ** 2)[:, None] * tg.tt_diag[None, :]).reshape(shape + (p,)))
    out.std, out.T = np.sqrt(out.variance), tg.T
    if getattr(dec, "error", None) is not None:
        _bound(dec, out, (A ** 6)[:, None] * ((1.0 / S ** 2).T @ (beta * beta)), shape, A / N)
    return out


def fields(dec, tg, gamma, beta, G, T):
    """Per-summand SOLA components from the structured Grams of separable functionals (plan 009 P1 open issue): the
    summand Gram K_f is block diagonal (one block per harmonic), the target pairings come harmonic by harmonic; for a
    factored family the components of every property are quadratic forms in the B block coordinates. None when G is
    not separable."""
    if not hasattr(G, "structured_gram") or not hasattr(G, "layout"):
        return None
    factored = isinstance(tg, FactoredTargets)
    tau = T.factors[1] if factored else T
    if not hasattr(tau, "layout"):
        return None
    M = G.domain
    names = getattr(M, "names", [0])
    extra = np.maximum(tg.tt_diag - rowquad(tg.Phi, tg.tt), 0.0) if factored else None
    live = [k for k in range(len(names)) if k in G.layout or k in tau.layout]
    homes = [k for k in live if k in tau.layout]
    scalar = np.ndim(gamma) == 0
    out = {}
    for k in live:
        Kf = G.structured_gram(None, {k})
        pair = G.structured_cross(tau, None, {k})
        if Kf is None or pair is None:
            return None
        cf, tt = pair
        tt = np.asarray(tt if isinstance(tt, np.ndarray) else tt.matrix(), float)
        dl, ms, tf = [], [], None
        for gv in np.atleast_1d(gamma):
            x = dec._back(beta / (dec.lam[:, None] + gv))
            Kx = Kf._rows_values(x)
            if factored:
                d = rowquad(tg.Phi, x.T @ Kx)
                c = x.T @ cf
                tf = rowquad(tg.Phi, tt) + (extra if len(homes) == 1 and k == homes[0] else 0.0)
                ms.append(tf - 2.0 * rowquad(tg.Phi, 0.5 * (c + c.T)) + d)
            else:
                d = np.einsum("ip,ip->p", x, Kx)
                tf = np.diag(tt)
                ms.append(tf - 2.0 * np.einsum("ip,ip->p", cf, x) + d)
            dl.append(d)
        out[names[k]] = dict(delivered=dl[0] if scalar else np.array(dl), mismatch=ms[0] if scalar else np.array(ms),
                             target=tf)
    return out
