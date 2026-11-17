"""Two closed-form quantities that are differences of nearly equal numbers (plan 054, pre-existing bugs, item 3), with
their stated error, a second route that does not subtract, and the general rule (optimisers/dual/exactness.py) for what is left.

    SOLA's delivered mismatch   q(x) = ||t - G* w||_S^2 = tt - 2 x.beta + x^T Lam x,    x = (Lam + g)^-1 beta
    the posterior variance      a^2 v,  v = tt - beta^T (Lam + s)^-1 beta = q(x) + s ||x||^2,  x = (Lam + s)^-1 beta,
                                s = nu^2 / a^2

(eigen coordinates of the decomposition; w = F^T U x the data weights). For a property the data resolve well, q and v
are much smaller than tt, and the Gram route's subtraction leaves an absolute error of about eps tt: measured (P4)
relative std errors 1.3e-6 at a = 1e3 and 6.5e-3 to 1.0 at a = 1e6 already at cond(W) 1e2, and mismatch errors of 1e3
relative, with no warning.

The Gram route's stated error (the input errors of optimisers/dual/exactness.py, per block b: ||dW_b|| <= C eps n_b L_b,
||d beta_b|| <= C eps n_b sqrt(L_b tt), |d tt| <= C eps tt; and the subtraction's own rounding):
    e_gram(x) = C eps tt + sum_b C eps n_b (L_b ||x_b||^2 + 2 sqrt(L_b tt) ||x_b||) + C eps (tt + 2 |x.beta| + x^T Lam x)
for v (stationary in x: the multiplier's own error enters at second order), and for q, which is not stationary
(dq/dx = -2 g x), also 2 g ||x|| ||dx||, ||dx|| <= sum_b C eps n_b (sqrt(L_b tt) + L_b ||x_b||) / (lam_min + g)
(dx_bound). Nothing in it can be made smaller from the Gram: tt's own rounding is eps tt.

The residual route does not subtract numbers: the residual element u = R* pi, R = T - X G (X the SOLA operator at g, or
at s for the posterior), is formed as an element of the model space and q = <u, S u>, v = q + s ||x||^2, a sum of
positive terms. The subtraction is then of elements, with an error of about eps (||t|| + the terms of G* w) in u, so
    e_res = 2 C eps n (||t||_S + sqrt(L) ||x||) ||u||_S + (C eps n (||t||_S + sqrt(L) ||x||))^2    (n data, L = max Lam;
G* w sums terms up to sqrt(L) ||x||, much larger than G* w itself where the data are ill-conditioned: measured, with
||G* w|| there, the error was understated up to 8 x),
plus the multiplier's error, a posteriori (x_error): the residual of its equation r = <g_k, S u> - g x, formed from u,
gives dx = (Lam + g)^-1 r, so q gains 2 g |x . dx| (first order) and v the excess r^T (Lam + g)^-1 r of J(x) = ||u||^2 +
s ||x||^2 over its minimum (second order: v is stationary in x), each with the rounding of r and counted SAFETY = 2
times (estimates: measured, the true error reached 1.0 x them); and the rounding of s ||x||^2. It costs one SOLA
operator and p adjoint applications of T and G per gamma (or per (a, nu)), so it runs only where e_gram > TOL |value|
(TOL = core.precision.TOL), and only where the operators are at hand (a property map T into a standard R^p, G, a complete
decomposition). Measured against mpmath (P4's problems, cond(W) 1e2 .. 1e14, 20 problems): largest relative errors
mismatch 7.5e10 -> 0.035 (gamma 1e-10), 1e3 -> 3.6e-6 (1e-6); std 1.3e-6 -> 4.2e-13 (a = 1e3), 1.0 -> 9.5e-5 (1e6);
every entry with fewer than 9 digits warned of, each within its stated error (at most 0.51 of it).

What is left after the better route: where the stated error exceeds TOL |value| a PrecisionWarning gives the count
and the largest relative error, and the Result carries the errors (mismatch_error; variance_error, std_error). Where
the error reaches the value itself (no correct digit) the default refuses (core.errors.Refused) with the choices named;
refuse=False returns the numbers with the warning; refuse=None (a caller that does not deliver the mismatch, such as
oCSOLA) keeps the closed form's values with their stated errors and judges nothing. The constants are those of
optimisers/dual/exactness.py (C = 1).

Cost (plan 054 fix2). SOLA's mismatch needs only the diagonal <u_i, S u_i>, so its residual route forms u
RESIDUAL_BATCH properties at a time (no p x p identity family: memory bounded by a batch of model elements), and keeps
the result for the next call with the same pairings and gamma (the mismatch does not depend on the datum: a run over
many data costs one residual route; dec._residual_cache, identical pairings to the bit). It still forms every property
once any needs it: the route also compares the Gram with the elements, and on EXP-030 the two differ by 3e-3 of the
mismatch where the Gram route alone states 1e-9 (the kernels' elements and the Gram they were paired into disagree;
/disks/data/tmp/mission-054-report.md, "Fix2"), so keeping the Gram route's values where it alone passes would hide
that. Measured on EXP-030's 119 coefficient functionals: the route is one application of G to the residual elements per
gamma, about 0.3 s per property there. Beyond RESIDUAL_MAX properties in one call it is not run: the Gram route's
values stand with their stated errors, and the warning (or refusal) says so and names the choices; at
pg.settings.verification = 0 ("quick") it is not run either, and the warning says the values were not verified (no
refusal, as for an inverse at that level). The posterior's covariance needs the full u-Gram and is formed as before
(p <= COVARIANCE_DENSE there)."""
import math
import warnings

import numpy as np

from ...core.errors import PrecisionWarning, Refused, Unsupported

EPS = np.finfo(float).eps
C = 1.0
RESIDUAL_BATCH = 64     # properties per batch of the residual route (SOLA's mismatch)
RESIDUAL_MAX = 1024     # beyond this many properties in one call the residual route is not run (stated)
CACHE = 16              # residual routes kept per decomposition
SAFETY = 2.0        # the a-posteriori terms of the multiplier's error are first-order estimates, not bounds: measured,
# the true error reached 1.0 x the estimate (m_cancel.py, /disks/data/tmp/mission-054-report.md), so they count twice


def _blocks(dec, n_eff):
    """(starts, n_b, L_b) of the decomposition's blocks; a decomposition without blocks: componentwise."""
    lam_b = getattr(dec, "_lam_b", None)
    if lam_b is not None and hasattr(dec, "offsets") and not dec.partial and int(dec.offsets[-1]) == n_eff:
        starts = np.asarray(dec.offsets[:-1], int)
        nb = np.array([len(b) for b in lam_b], float)
        L = np.array([float(np.abs(b).max(initial=0.0)) for b in lam_b])
        return starts, nb, L
    lam = np.abs(np.asarray(dec.lam, float))
    lam = np.r_[lam, np.zeros(n_eff - len(lam))] if n_eff > len(lam) else lam[:n_eff]
    return np.arange(n_eff), np.ones(n_eff), lam


def _terms(lam, X):
    """sqrt(L) ||x||: a bound of the size of the terms that G* w = G* F^T U x sums."""
    return math.sqrt(float(np.max(np.abs(lam), initial=0.0))) * np.sqrt(np.sum(X * X, axis=0))


def _lmin(lam):
    pos = lam[lam > 0]
    return float(pos.min()) if len(pos) else 0.0


def gram_error(dec, lam, X, beta, tt):
    """e_gram (module docstring) for the multipliers X (n_eff, k) with pairings beta (n_eff, k) and tt (k)."""
    starts, nb, L = _blocks(dec, len(lam))
    nx = np.sqrt(np.add.reduceat(X * X, starts, axis=0))
    w = (C * EPS * nb)[:, None]
    tt = np.maximum(np.asarray(tt, float), 0.0)
    rt = np.sqrt(tt)[None, :]
    sL = np.sqrt(L)[:, None]
    e = C * EPS * tt + np.sum(w * (L[:, None] * nx ** 2 + 2.0 * sL * rt * nx), axis=0)
    return e + C * EPS * (tt + 2.0 * np.abs(np.sum(X * beta, axis=0)) + np.sum(lam[:, None] * X * X, axis=0))


def residuals(dec, tg, g, cols=None):
    """(u-Gram <u_i, S u_j> (p x p), ||t_i||_S, ||G* w_i||_S, <g_k, S u_i> in eigen coordinates) for the SOLA operator
    at g, or None where the residual route is not at hand (module docstring). cols: only these properties, in batches of
    RESIDUAL_BATCH, with the diagonal <u_i, S u_i> (k,) in place of the u-Gram."""
    from ...operators.linear import Sum
    if not residuals_at_hand(dec, tg):
        return None
    T, G, S = tg.T, dec.G, dec.middle
    P, M = T.codomain, G.domain
    try:
        X = dec.sola_operator(T, g)
        R = Sum([(1.0, T), (-1.0, X @ G)])

        def gram(A):
            SA = A if S is None else S(A)
            return np.atleast_2d(np.asarray(M.inner(A, SA), float))

        def form(E):
            Ut, Uw, U = T.adjoint(E), G.adjoint(X.adjoint(E)), R.adjoint(E)
            nt, nw = np.sqrt(np.maximum(np.diag(gram(Ut)), 0.0)), np.sqrt(np.maximum(np.diag(gram(Uw)), 0.0))
            GSu = np.asarray(G(U if S is None else S(U)).values, float)
            return gram(U), nt, nw, dec._coef(GSu.T if GSu.ndim == 2 else GSu[:, None])
        if cols is None:
            Gu, nt, nw, ru = form(P.family(np.eye(P.dim), name="I"))
            if Gu.shape != (P.dim, P.dim):
                return None
            return 0.5 * (Gu + Gu.T), nt, nw, ru
        parts = []
        for a in range(0, len(cols), RESIDUAL_BATCH):
            c = np.asarray(cols[a:a + RESIDUAL_BATCH], int)
            E = np.zeros((len(c), P.dim))
            E[np.arange(len(c)), c] = 1.0
            Gu, nt, nw, ru = form(P.family(E, name="I"))
            if Gu.shape != (len(c), len(c)):
                return None
            parts.append((np.diag(Gu), nt, nw, ru))
    except (Unsupported, NotImplementedError, TypeError, AttributeError, MemoryError):
        return None
    return (np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts]),
            np.concatenate([q[2] for q in parts]), np.concatenate([q[3] for q in parts], axis=1))


def _cached_residuals(dec, tg, g, cols, beta, tt):
    """residuals(dec, tg, g, cols), kept on dec for the next call with the same pairings (beta, tt to the bit) and g:
    the mismatch does not depend on the datum (module docstring, "Cost")."""
    import hashlib
    parts = [np.ascontiguousarray(beta[:, cols]), np.ascontiguousarray(tt[cols]), np.float64(g),
             np.asarray(cols, np.int64)]
    key = hashlib.sha1(b"".join(np.asarray(v).tobytes() for v in parts)).hexdigest()
    cache = dec.__dict__.setdefault("_residual_cache", {})
    if key not in cache:
        res = residuals(dec, tg, g, cols)
        if res is None:
            return None
        while len(cache) >= CACHE:
            cache.pop(next(iter(cache)))
        cache[key] = res
    return cache[key]


def x_error(dec, lam, X, ru, g, nu):
    """The multiplier's error a posteriori, from the residual of its equation (Lam + g) x = beta in eigen coordinates,
    r = <g_k, S u> - g x (ru: the first term, formed from the residual element u): dx = (Lam + g)^-1 r to first order in
    the decomposition's error. Returns (sum r^2 / (Lam + g) with the rounding of r, x . dx with it): the excess of the
    posterior variance's J(x) over its minimum (x^T (Lam + g) x is J's Hessian), and the first-order term of the
    mismatch. The rounding of r: C eps n (sqrt(L) ||u|| + g ||x||), taken in the smallest direction."""
    r = ru - g * X
    lmin, L = _lmin(lam), float(np.max(np.abs(lam), initial=0.0))
    dr = C * EPS * dec.n * (np.sqrt(L) * nu + g * np.sqrt(np.sum(X * X, axis=0)))
    h = np.sqrt(np.sum(r * r / (lam[:, None] + g), axis=0)) + dr / np.sqrt(lmin + g)
    xd = np.abs(np.sum(X * r / (lam[:, None] + g), axis=0)) + np.sqrt(np.sum(X * X, axis=0)) * dr / (lmin + g)
    return h * h, xd


def residual_error(n, nt, nw, nu):
    """e_res (module docstring) for norms ||t||_S = nt, nw (a bound of the terms of G* w: sqrt(L) ||x||, since G* w can
    be much smaller than its terms where the data are ill-conditioned) and ||u||_S = nu."""
    a = C * EPS * n * (nt + nw)
    return 2.0 * a * nu + a * a


def dx_bound(dec, lam, X, tt):
    """sum_b C eps n_b (sqrt(L_b tt) + L_b ||x_b||): a bound of ||d beta - dW x|| under the stated input errors."""
    starts, nb, L = _blocks(dec, len(lam))
    nx = np.sqrt(np.add.reduceat(X * X, starts, axis=0))
    w = (C * EPS * nb)[:, None]
    return np.sum(w * (np.sqrt(L)[:, None] * np.sqrt(np.maximum(tt, 0.0))[None, :] + L[:, None] * nx), axis=0)


def _route_choice(tg, dec):
    if residuals_at_hand(dec, tg):
        return "the model in higher precision (not available in pygeoinf0 yet)"
    return ("the property map T into a standard R^p, G and a complete decomposition (not pairings, a factored family or "
            "a partial decomposition), so that the residual is formed as an element")


def residuals_at_hand(dec, tg):
    from ...spaces.base import Euclidean
    from ...operators.linear import LinearOperator
    T = tg.T
    return (isinstance(T, LinearOperator) and dec.G is not None and not dec.partial and
            getattr(T, "factors", None) is None and isinstance(T.codomain, Euclidean) and T.codomain.is_standard)


def judge(name, value, err, refuse, what, choices, note=""):
    """The rule for what is left (module docstring): a PrecisionWarning where err > TOL |value|, a refusal where err
    reaches |value| (unless refuse=False). note: why the better route was not run, if it was not."""
    from ...core.precision import assess, fewer
    if refuse is None:                                  # a caller that does not deliver these values
        return
    lost, none, worst = assess(value, err)
    if not lost.any():
        return
    msg = (f"{name}: {fewer(int(lost.sum()), lost.size, what)} under the stated rounding "
           f"(the largest relative error {worst:.2g}; {int(none.sum())} with no correct digit); "
           "solvers/dataspace/cancellation.py" + note)
    if none.any() and refuse:
        raise Refused(msg, needs=list(choices) + [f"{name}(..., refuse=False): the numbers, with the errors stated "
                                                  "and a warning"])
    warnings.warn(msg + ". " + "; ".join(choices), PrecisionWarning, stacklevel=4)


def sola_mismatch(dec, tg, g, lam, beta, tt, mis, refuse):
    """SOLA's mismatch for every gamma (rows) and property (columns), with its stated error: the Gram route's values
    where they hold TOL, else the residual route's where at hand (module docstring). (mismatch, error)."""
    from ...core.precision import TOL
    from ...settings import settings
    mis = np.array(mis, float, copy=True)
    err = np.empty_like(mis)
    n, lmin = dec.n, _lmin(lam)
    note, choices = "", ["a larger gamma, at which the mismatch is resolved", _route_choice(tg, dec)]
    for i, gi in enumerate(g):
        X = beta / (lam[:, None] + gi)
        xterm = 2.0 * gi * np.sqrt(np.sum(X * X, axis=0)) * dx_bound(dec, lam, X, tt) / (lmin + gi)
        err[i] = gram_error(dec, lam, X, beta, tt) + xterm
        if refuse is None or not np.any(err[i] > TOL * np.abs(mis[i])) or not residuals_at_hand(dec, tg):
            continue
        cols = np.arange(mis.shape[1])          # every property: the route also checks the Gram against the elements
        if settings.verification == 0:
            note = ("; the residual route was not run (pg.settings.verification = 0, quick): the Gram route's values, "
                    "not verified; verification >= 1 forms them again")
            refuse = False
            continue
        if len(cols) > RESIDUAL_MAX:
            note = (f"; the residual route was not run: {len(cols)} properties need it, more than RESIDUAL_MAX = "
                    f"{RESIDUAL_MAX} in one call (about one application of G per property): the Gram route's values")
            choices.append(f"at most {RESIDUAL_MAX} properties per call, so that the residual route runs")
            continue
        res = _cached_residuals(dec, tg, gi, cols, beta, tt)
        if res is not None:
            q, nt, nw, ru = res
            q = np.maximum(q, 0.0)
            xd = x_error(dec, lam, X, ru, gi, np.sqrt(q))[1]
            nw = np.maximum(nw, _terms(lam, X))
            mis[i], err[i] = q, residual_error(n, nt, nw, np.sqrt(q)) + SAFETY * 2.0 * gi * xd
    judge("SOLA", mis, err, refuse, "delivered mismatches", choices, note)
    return mis, err


def posterior_covariance(dec, tg, A, N, lam, beta, cov, refuse):
    """The posterior covariance for every (a, nu) (A, N flattened), with the stated error of its diagonal: the Gram
    route where it holds TOL, else a^2 (<u, S u> + s x^T x) by the residual route where at hand. (cov, variance_error)."""
    from ...core.precision import TOL
    tt = tg.diag()
    cov = np.array(cov, float, copy=True)
    err = np.empty(cov.shape[:2])
    n = dec.n
    for h, (a, nu) in enumerate(zip(A, N)):
        s = (nu / a) ** 2
        X = beta / (lam[:, None] + s)
        err[h] = a * a * gram_error(dec, lam, X, beta, tt)
        if np.any(err[h] > TOL * np.abs(np.diagonal(cov[h]))):
            res = residuals(dec, tg, s)
            if res is not None:
                Gu, nt, nw, ru = res
                cov[h] = a * a * (Gu + s * (X.T @ X))
                nu_ = np.sqrt(np.maximum(np.diag(Gu), 0.0))
                second = SAFETY * x_error(dec, lam, X, ru, s, nu_)[0]
                nw = np.maximum(nw, _terms(lam, X))
                err[h] = a * a * (residual_error(n, nt, nw, nu_) + C * EPS * n * s * np.sum(X * X, axis=0) + second)
    var = np.diagonal(cov, axis1=-2, axis2=-1)
    judge("posterior", var, err, refuse, "posterior variances", (
        "a narrower prior (a smaller a): a mathematical choice", _route_choice(tg, dec)))
    return cov, err


def std_error(var, err):
    """The error of sqrt(var) for a variance error err: sqrt(var + err) - sqrt(var), and sqrt(err) at var <= err."""
    var = np.maximum(np.asarray(var, float), 0.0)
    return np.where(var > 0, err / (np.sqrt(var + err) + np.sqrt(var)), np.sqrt(err))
