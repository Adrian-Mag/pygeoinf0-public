"""The span reduction and the solvers of the reduced problems (the solver part of convex/dual.py, moved by plan 070
P4; convex/feasible.py states the problem and its dual, convex/forms.py turns the noise set into a form). Support
functions of feasible sets K(d) = S ∩ {m : d - G m in N} and of their images T(K(d)): BGP (inf:476-1092;
EXP-028 DESIGN.md B5, B6 and 2.5), exact in Gram terms for a norm-ball prior on the model space or none.

Span reduction (LAND EXP-016; inf:1028). With m = m0 + u and the prior ball ||u|| <= r, an extremal u lies in the span
of the direction and the data representers: removing the orthogonal part changes neither objective nor data and does
not increase the norm. The joint Gram matrix J of [targets, data representers] = U diag(lam) U^T gives coordinates
y in R^k of that span, so the problem becomes: maximise a . y with ||y|| <= r and e - B y in N (e = d - G m0, B the data
map in coordinates). Noise sets reduce further:
    ball      a NormBall of D, an Ellipsoid, either plus a span of free nuisances (origin times): ||d~ - B~ y|| <= rho
              after whitening by the set's shape and projecting out the nuisances (DESIGN 2.5 (e))
    exact     {0} or a span: B~ y = d~ (exact data, with or without nuisances; strong duality needs a strictly
              feasible model, DESIGN 2.5 (c))
    slab      the region of one fixed estimator: |s - beta . y| <= w (oCSOLA's set, inf:427-465)
    general   other sets with support functions (boxes, polytopes): a primal witness by SLSQP where the set is linear,
              a certificate by minimising the dual numerically
Ball-ball problems are solved by the nested bisection of inf:1001-1072 (inner trust-region solve, outer data
multiplier, both monotone). Every Bound carries the value, a lower bound from a feasible witness and an upper bound
from a certificate lambda evaluated by the general dual formula through the sets' own support functions,
    phi(lambda) = <c, m0> + <lambda, e>_D + sigma_S(c - G* lambda) + sigma_N(-lambda)   (inf:874-895),
a second route independent of the solver. status: EXACT (they agree to 1e-9), BRACKET, OUTER (certificate only),
UNBOUNDED (+inf: a non-estimable direction without a prior), EMPTY (with a Farkas certificate: phi(lambda) < 0 at
direction 0, DESIGN 2.5 (d)).
"""
import math

import numpy as np

from ...core.errors import Refused
from ..scalar import bisect, trs


EPS = np.finfo(float).eps


def structural_zeros(n, G):
    """How many of the n data directions are exactly dependent by structure: at least n - dim M when the model space M
    is finite-dimensional (rank G G* <= dim M), else 0."""
    from ...spaces.base import Euclidean
    M = getattr(G, "domain", None)
    return max(0, n - M.dim) if isinstance(M, Euclidean) else 0


def floor_refusal(where, k, n, how):
    """The refusal of a bound that depends on data directions at the rounding floor (plan 054 fix round): k of n
    directions there, not exactly zero and not explained by structure, cannot be told from exact dependence in double
    precision, and the value depends on which (measured: 8 to 160 % wrong, a false EMPTY, a finite value where the
    bound is +inf; tests/test_error_account_fix.py). An instance of the general rule (optimisers/dual/exactness.py): the stated
    input error (rounding) admits both exactly dependent data and independent data whose bound differs without limit
    (E = inf); declaring the dependence (K.dependent) states a smaller input error, under which the bound is exact."""
    return Refused(f"{where}: {k} of {n} data directions are at the rounding floor ({how}) and not exactly zero: in "
                   "double precision they cannot be told from exactly dependent data, and this bound depends on which "
                   "(the datum along them is imposed, or not)",
                   needs=["declare the data dependent: K.dependent = True on the feasible set K (pg.update's result, or "
                          "FeasibleSet(..., dependent=True)): those directions are dropped, exact if the data are exactly "
                          "dependent there, and the Bound's note says so",
                          "regularise: a noise ball instead of exact data, or a prior ball where there is none (a "
                          "mathematical choice)",
                          "a Gram matrix in higher precision (not available in pygeoinf0 yet)"])


# ------------------------------------------------------------------------------------------------ reduction
def gram_blocks(G, T=None):
    """<g_i, g_j>, <g_i, t_l>, <t_l, t_m> (standard coordinates of D and P; representers in the model space)."""
    D = G.codomain
    GG = D.solve_metric((G @ G.adjoint).matrix().T).T
    if T is None:
        return GG, None, None
    P = T.codomain
    GT = P.solve_metric(np.atleast_2d((G @ T.adjoint).matrix()).T).T
    TT = P.solve_metric(np.atleast_2d((T @ T.adjoint).matrix()).T).T
    return GG, GT, TT


class Reduction(tuple):
    """(J, d, lam, U, keep, n_dirs) of reduction.reduce, with cerr: the stated realisation errors of J's columns (the
    directions', then the data representers'; None: none stated beyond rounding; optimisers/dual/exactness_reduce.py)."""
    cerr = None


def reduce(J, n_dirs, info=None):
    """Coordinates (k, q) of the representers whose Gram matrix is J: the first n_dirs columns are the directions.
    Rank is decided on the correlations, so it does not depend on the representers' scales.

    The cut (plan 054 P2): eigenvalues of the q x q correlation matrix at most eps q times its largest are dropped, the
    floor below which an eigenvalue of a matrix given to double precision has no correct digit (exactly dependent
    representers, e.g. an estimable target, land there). Dropping one moves each representer by at most about
    sqrt(eps q) of its norm, so a support value by about r ||c|| sqrt(eps q): the size of the rounding of J itself.
    Measured against exact 2-D intervals (tests/test_error_account_cuts.py): the former cut 1e-13 dropped directions
    with correct digits and gave intervals up to 1e-2 of their width too narrow. With info (a dict), info["dropped"]
    and info["size"] report the count, info["reduction"] what the general rule needs (optimisers/dual/exactness_reduce.py)."""
    J = 0.5 * (J + J.T)
    d = np.sqrt(np.clip(np.diag(J), 0.0, None))
    d = np.where(d > 0.0, d, 1.0)
    lam, U = np.linalg.eigh(J / np.outer(d, d))
    keep = lam > EPS * len(lam) * max(lam.max(initial=0.0), 1e-300)
    if info is not None:
        info["dropped"], info["size"] = int((~keep).sum()), len(lam)
        info["reduction"] = Reduction((J, d, lam, U, keep, n_dirs))   # for the general rule (optimisers/dual/exactness_reduce.py)
    Y = np.sqrt(lam[keep])[:, None] * U[:, keep].T * d
    return Y[:, :n_dirs], Y[:, n_dirs:].T                       # (k, n_dirs) directions, (n, k) data map B


def whiten(form, e, B):
    """d~ = Z^T Phi (e - c), B~ = Z^T Phi B and Z, with Z an orthonormal basis of (Phi F)^perp. A column of the QR of
    Phi F with |R_jj| at most eps n max |R_jj| is dropped: a nuisance direction already in the span of the others to
    rounding (plan 054 P2; the former cut 1e-12 dropped nuisance directions with correct digits and moved intervals by
    8 % of their width, tests/test_error_account_cuts.py). A Span's own basis has its cut first
    (convex/subspaces.Span, at its own floor)."""
    Phi, F = form["Phi"], form["F"]
    n = Phi.shape[0]
    if F.shape[1]:
        Q, R = np.linalg.qr(Phi @ F)
        Q = Q[:, np.abs(np.diag(R)) > EPS * n * max(np.abs(np.diag(R)).max(), 1e-300)] if Q.size else Q
        Z = np.linalg.qr(np.c_[Q, np.eye(n)])[0][:, Q.shape[1]:n]
    else:
        Z = np.eye(n)
    M = Z.T @ Phi
    return M @ (e - form["centre"]), M @ B, M


# ------------------------------------------------------------------------------------------------ solvers
def data_svd(B):
    """The SVD of the data map and its rank: singular values at most eps max(n, k) s_max count as zero, the floor below
    which a singular value of a matrix given to double precision has no correct digit (plan 054 P2). ball_exact and
    none_ball lose the datum along a dropped direction: exact if the data are exactly dependent there, otherwise the
    constraint along it is not imposed. The former cut 1e-12 dropped singular values with correct digits (errors up to
    50 % of the value, status EXACT; tests/test_error_account_cuts.py). ball_ball does not use the rank."""
    U, s, Vt = np.linalg.svd(B, full_matrices=True)
    k = B.shape[1]
    s2 = np.zeros(k)
    s2[:len(s)] = s ** 2
    rank = int(np.sum(s > EPS * max(B.shape) * max(s.max(initial=0.0), 1e-300)))
    return U, s, Vt.T, s2, rank


def residual_misfit(bb, s2, dd, res=None):
    """y -> ||d - B y||^2 in the right singular basis. From the residual, ||dp - s y||^2 + f0 (res = (dp, f0)): its
    rounding is eps ||dp|| ||d - B y||, against eps ||d||^2 for the expansion dd - 2 bb.y + s2.y^2, which is all that
    is left without res. Where ||d|| / rho is large (small noise) the expansion's rounding is far above the 1e-10 at
    which the witness's misfit is judged (SpectralBall._finish): the root it locates then sits off the noise ball by an
    amount set by the linear-algebra library (plan 054: the Mac's Accelerate against europa's OpenBLAS, 4e-7 relative
    at ||d|| / rho = 6e4, PRIMAL or BRACKET by library; tests/test_error_account_blas.py). Columns of y: directions."""
    if res is None:
        return lambda y: dd - 2.0 * (bb @ y) + s2 @ (y * y)
    dp, f0 = np.asarray(res[0], float), float(res[1])
    s = np.sqrt(s2)

    def misfit(y):
        e = (dp - s * y) if y.ndim == 1 else (dp[:, None] - s[:, None] * y)
        return np.einsum("k...,k...->...", e, e) + f0
    return misfit


def ball_ball(a, B, d, r, rho):
    """max a.y s.t. ||y|| <= r, ||d - B y|| <= rho: (value, y, lam, status) with lam the certificate in d's space."""
    U, s, V, s2, rank = data_svd(B)
    ab, bb, dd = V.T @ a, V.T @ (B.T @ d), float(d @ d)
    ud = U.T @ d
    dp = np.zeros(B.shape[1])
    dp[:len(s)] = ud[:len(s)]
    case, val, yb, om = ball_ball_spectral(ab, bb, s2, dd, r, rho, (dp, float(ud[len(s):] @ ud[len(s):])))
    if case == "EMPTY":
        e = d - B @ (V @ yb)
        return None, V @ yb, -e / np.linalg.norm(e), "EMPTY"
    if case == "TOUCH":                                        # touching: the feasible set is a face of the ball
        return ball_exact(a, B, B @ (V @ yb), r)
    if case in ("ZERO", "PRIOR"):
        return val, V @ yb, np.zeros_like(d), "EXACT"
    y = V @ yb
    return val, y, -om * (d - B @ y), "EXACT"


def ball_ball_spectral(ab, bb, s2, dd, r, rho, res=None):
    """ball_ball in the right singular basis of the data map B = U diag(s) V^T: ab = V^T a, bb = V^T B^T d, s2 = s^2
    (zero where B has no singular value), dd = ||d||^2. Returns (case, value, yb, omega) with yb in that basis and case
    EMPTY (yb the least-misfit point of the ball), TOUCH (the same point; the feasible set is a face of the ball),
    ZERO (a = 0), PRIOR (the prior alone decides, omega = 0) or EXACT (omega the data multiplier: the certificate is
    -omega (d - B y)). One evaluation costs O(len(ab)): with a data-space decomposition the basis is known in advance
    and nothing is factorised per direction (solvers/dataspace/decomposition.py). res = (dp, f0), the datum in that basis
    (bb = sqrt(s2) dp) and its part the data map does not reach: the misfit is then formed from the residual
    (residual_misfit)."""
    misfit = residual_misfit(bb, s2, dd, res)

    yls, _ = trs(bb, s2, r)                                   # the least misfit over the ball
    mmin = misfit(yls)
    if mmin > rho ** 2 * (1 + 1e-10) + 1e-300:
        return "EMPTY", None, yls, None
    if mmin >= rho ** 2 * (1 - 1e-10):
        return "TOUCH", None, yls, None
    na = np.linalg.norm(ab)
    if na == 0:
        return "ZERO", 0.0, yls, 0.0
    yf = r * ab / na
    if misfit(yf) <= rho ** 2:                                 # the prior alone decides (inf:1054)
        return "PRIOR", r * na, yf, 0.0

    def y_of(om):
        return trs(ab + om * bb, om * s2, r)[0]
    hi = 1.0
    while misfit(y_of(hi)) > rho ** 2 and hi < 1e300:
        hi *= 4.0
    lo = hi / 4.0
    while lo > 1e-300 and misfit(y_of(lo)) <= rho ** 2:
        lo /= 4.0
    om = bisect(lambda o: misfit(y_of(o)) - rho ** 2, lo, hi)
    yb = y_of(om)
    return "EXACT", float(ab @ yb), yb, om


def _at_floor(where, s, rank, n, info, dependent):
    """The rank to use: refuse (floor_refusal) when singular values of B lie in (0, floor], or info["data_dependent"]
    counts data representers dependent to rounding (FeasibleSet.dependency), unless the data are declared dependent;
    declared, the rank is at most info["data_rank"] (the data's own rank under the declaration: the reduction leaves
    singular values of size sqrt(eps) along the dependency, above B's floor)."""
    info = {} if info is None else info
    amb = max(int(np.sum(s[rank:] > 0)), int(info.get("data_dependent", 0)))   # one dependency shows in both
    if amb and not dependent:
        raise floor_refusal(where, amb, n, "singular values of the data map <= eps max(n, k) s_max, or correlation "
                            "eigenvalues of the data Gram <= eps n max")
    if amb:
        rank = min(rank, int(info.get("data_rank", rank)))
    info["data_dropped"], info["declared"] = len(s) - rank, bool(amb)
    return rank


def ball_exact(a, B, d, r, info=None, dependent=False):
    """max a.y s.t. ||y|| <= r (r = inf: no prior), B y = d (exact data). Refused (floor_refusal) where a singular value
    of B is at the rounding floor but not exactly zero (plan 054 fix round: the dropped equation's value has no correct
    digit; measured 37 % wrong with status EXACT before), unless dependent=True declares the data exactly dependent
    there. info (a dict): info["data_dropped"], the singular values of B at the rounding floor (data_svd)."""
    U, s, V, s2, rank = data_svd(B)
    rank = _at_floor("exact data", s, rank, B.shape[0], info, dependent)
    Ur, sr, Vr = U[:, :rank], s[:rank], V[:, :rank]
    dpar = Ur @ (Ur.T @ d)
    dperp = d - dpar
    if np.linalg.norm(dperp) > 1e-10 * max(np.linalg.norm(d), 1e-300):
        return None, None, -dperp, "EMPTY"                     # d is not attainable: B^T dperp = 0
    Bp = Vr @ np.diag(1.0 / sr) @ Ur.T                         # B^+
    y0 = Bp @ d
    n0 = np.linalg.norm(y0)
    if n0 > r * (1 + 1e-12):
        return None, y0, -(Bp.T @ y0), "EMPTY"                 # phi(lam, 0) = n0 (r - n0) < 0
    p = a - Vr @ (Vr.T @ a)                                    # the part of a the data do not see
    if np.linalg.norm(p) <= 1e-12 * max(np.linalg.norm(a), 1e-300):
        return float(a @ y0), y0, Bp.T @ a, "EXACT"            # estimable: one value on the fibre
    if math.isinf(r):
        return math.inf, None, None, "UNBOUNDED"
    rem = math.sqrt(max(r * r - n0 * n0, 0.0))
    y = y0 + rem * p / np.linalg.norm(p)
    if rem == 0:
        return float(a @ y0), y0, None, "EXACT"                # a single feasible model; no finite multiplier
    mu = np.linalg.norm(p) / rem
    return float(a @ y), y, Bp.T @ (a - mu * y), "EXACT"


def none_ball(a, B, d, rho, info=None, dependent=False):
    """max a.y s.t. ||d - B y|| <= rho, no prior: finite iff a is in ran B^T (estimable). Refused at the rounding floor
    and info as for ball_exact."""
    U, s, V, s2, rank = data_svd(B)
    rank = _at_floor("no prior", s, rank, B.shape[0], info, dependent)
    Ur, sr, Vr = U[:, :rank], s[:rank], V[:, :rank]
    if np.linalg.norm(a - Vr @ (Vr.T @ a)) > 1e-12 * max(np.linalg.norm(a), 1e-300):
        return math.inf, None, None, "UNBOUNDED"
    mu0 = Ur @ ((Vr.T @ a) / sr)                               # B^T mu0 = a, mu0 in ran B
    dpar = Ur @ (Ur.T @ d)
    dperp = d - dpar
    rp2 = rho ** 2 - dperp @ dperp
    if rp2 < -1e-12 * max(rho ** 2, 1e-300):
        return None, None, -dperp / np.linalg.norm(dperp), "EMPTY"
    rp = math.sqrt(max(rp2, 0.0))
    nm = np.linalg.norm(mu0)
    z = dpar + (rp * mu0 / nm if nm > 0 else 0.0)
    y = Vr @ ((Ur.T @ z) / sr)
    lam = mu0 - (nm / rp) * dperp if rp > 0 else None
    return float(mu0 @ dpar + rp * nm), y, lam, "EXACT"


def ball_slab(a, beta, s, w, r):
    """max a.y s.t. ||y|| <= r, |s - beta.y| <= w (closed form; the certificate comes from the 1-D dual)."""
    nb, na = np.linalg.norm(beta), np.linalg.norm(a)
    if s + w < -r * nb or s - w > r * nb:
        return None, None, "EMPTY"
    yf = r * a / na if na > 0 else np.zeros_like(a)
    t = beta @ yf
    if s - w <= t <= s + w:
        return r * na, yf, "EXACT"
    h = s + w if t > s + w else s - w
    y0 = h * beta / nb ** 2
    pa = a - (a @ beta) * beta / nb ** 2
    pa = pa - (pa @ beta) * beta / nb ** 2                     # twice: pa _|_ beta to rounding, so y stays on the face
    rem = math.sqrt(max(r * r - h * h / nb ** 2, 0.0))
    npa = np.linalg.norm(pa)
    y = y0 + (rem * pa / npa if npa > 0 else 0.0)
    # a . y = a . y0 + rem ||pa||: written so, not as a @ y, whose a . pa / ||pa|| has no correct digit when a is
    # parallel to beta to rounding (an estimable target resolved by the slab's estimator): it returned values far
    # outside the certificate's bound (plan 054 P2, tests/test_error_account_cuts.py)
    return float(a @ y0) + rem * float(npa), y, "EXACT"


def slab_dual(a, beta, s, w, r):
    """inf over tau of tau s + |tau| w + r ||a - tau beta|| (oCSOLA, inf:446-452): (value, tau)."""
    from ..minimise import minimise_scalar
    if not np.any(beta):                                        # no slab direction: tau = 0
        return float(r * np.linalg.norm(a)), 0.0

    def phi(t):
        return t * s + abs(t) * w + r * np.linalg.norm(a - t * beta)
    scale = np.linalg.norm(a) / max(np.linalg.norm(beta), 1e-300)   # the size of tau; no absolute floor, which made
    scale = scale if scale > 0 else 1.0                              # the tolerance depend on X's units (plan 037)
    best = min((minimise_scalar(phi, (lo, hi), xatol=1e-13 * scale)
                for lo, hi in ((-100 * scale, 0.0), (0.0, 100 * scale))), key=lambda res: res.fun)
    return float(best.fun), float(best.x)


def general(a, B, r, cons, phi, starts=()):
    """Other noise sets: a primal witness by SLSQP when the set is given by smooth constraints (cons: SLSQP
    inequalities in y), a certificate by minimising phi from the given starts (each start is already a certificate,
    so the result is never worse than the best of them)."""
    from ..minimise import minimise
    k, n = B.shape[1], B.shape[0]
    lower, y = -math.inf, None
    if cons is not None:
        cons = list(cons) + ([dict(type="ineq", fun=lambda y: r * r - y @ y)] if math.isfinite(r) else [])
        res = minimise(lambda y: -(a @ y), np.zeros(k), "SLSQP", dict(maxiter=1000, ftol=1e-15), jac=lambda y: -a,
                       constraints=cons)
        if all(np.all(np.atleast_1d(c["fun"](res.x)) >= -1e-9) for c in cons):
            lower, y = float(a @ res.x), res.x
    best_x, best_f = None, math.inf
    for x0 in list(starts) + [np.zeros(n)]:
        f0 = phi(x0)
        if f0 < best_f:
            best_x, best_f = x0, f0
    for method, opts in (("Nelder-Mead", dict(xatol=1e-12, fatol=1e-14, maxiter=40000, maxfev=80000)),
                         ("BFGS", dict(gtol=1e-12))):
        res = minimise(phi, best_x, method, opts)
        if res.fun < best_f:
            best_x, best_f = res.x, float(res.fun)
    return lower, y, best_x, best_f
