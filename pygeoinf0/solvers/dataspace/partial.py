"""Partial data-space decompositions above settings.dense_max (plan 005 P5; plan 009 G3): the leading part of the
whitened Gram W = F G S G* F^T from products with G S G* only, and the solves that use it.

    randomised   (default) a Nystrom approximation W_k = Y (Q^T Y)^+ Y^T, Y = W Q, from a block randomised range finder
                 (Halko-Martinsson-Tropp 2011, Algorithm 4.2 by blocks; the stable shifted Nystrom of Tropp et al. 2017),
                 with adaptive power iterations (0-3). In exact arithmetic 0 <= W_k <= W + nu I (nu the stable shift);
                 in double W - W_k has eigenvalues of both signs (plan 054 P3: down to -0.8 nu, and to -147 nu when the
                 basis stalled), so E = W - W_k is treated as symmetric, not positive. ||E|| is estimated a
                 posteriori from r = 10 Gaussian probes of a power of E: ||E||^q = ||E^q|| <= 10 sqrt(2/pi) max_i
                 ||E^q omega_i|| with probability at least 1 - 10^-10 (Halko et al., Lemma 4.1, for the matrix E^q;
                 q = 2 while the basis grows, q = 3 for the stated value: the power concentrates the probes on the
                 largest eigenvalue, where q = 1 would measure a Frobenius-like sum). The basis grows until that
                 estimate is at most `tail` (default settings.accuracy, in whitened units) or the rank cap.
    lanczos      the leading eigenpairs by ARPACK until the next would be below `tail`; the remainder is not bounded.

Either way the rest of the spectrum is treated as 0 (exact for W_k). The closed forms of the posterior carry a
two-sided bound (plan 054 P3; it was one-sided, from E >= 0, which double precision does not deliver): for w the
whitened image of a functional, R = (W_k + I)^-1 and T = (W + I)^-1, the resolvent identity R - T = R E T gives
|w^T (R - T) w| <= ||E|| ||R w|| ||T w||, and T w = R w - R E T w with ||R|| <= 1 gives ||T w|| <= ||R w|| / (1 - ||E||):
so |w^T (R - T) w| <= ||E|| ||R w||^2 / (1 - ||E||) for ||E|| < 1 (two_sided below; ClosedForms.posterior states it per
functional: the leading directions, where R is small, contribute little). Measured (plan 054 P3, an exact mpmath
reference for the stored W): the former one-sided claim var_k <= var failed in 4 of 6 spectra, by up to 1.5e-7 of the
prior variance where the tail was met, and the two-sided bound held in all of them.

pcg: block conjugate gradients on (rho^2 W + I) z = b in whitened coordinates for many right-hand sides at once,
preconditioned by (rho^2 W_k + I)^-1 = I - U diag(rho^2 Lam / (1 + rho^2 Lam)) U^T (deflation: the preconditioned
operator's spectrum lies in [1, 1 + rho^2 ||E||]).
"""
import math

import numpy as np

from ...core.claims import CHECKED, Claim
from ...core.errors import Unsupported

BLOCK = 64          # columns per block of the range finder
PROBES = 10         # Gaussian probes of the residual E (failure probability 10^-PROBES)
KMAX = 4000         # rank cap (n k doubles; 4000 at n = 10^6 is 32 GB)
LEMMA = 10.0 * math.sqrt(2.0 / math.pi)


def operator(dec):
    """(W as a function of an (n, m) array, the whitening): products with G S G* only (the decomposition's operator)."""
    from ...settings import settings
    from .whitening import _White
    D, noise = dec.D, dec.noise
    if noise.kind not in ("standard", "diag"):
        raise Unsupported(f"above settings.dense_max = {settings.dense_max} the noise covariance must be diagonal or "
                          "the data space's own (no n x n factorisation)")
    wt = _White("metric", None, D) if noise.kind == "standard" else _White("diag", 1.0 / np.sqrt(noise.data))
    A = dec.normal if dec.normal is not None else (
        dec.G @ dec.G.adjoint if dec.middle is None else dec.G @ dec.middle @ dec.G.adjoint)

    def K(V):                                       # K V = A(W_D^-1 V): A y = K W_D y in coordinates
        V2 = V if V.ndim == 2 else V[:, None]
        out = A(D.family(D.solve_metric(V2).T)).values.T
        return out if V.ndim == 2 else out[:, 0]

    def W(V):
        return wt.fwd(K(wt.back(V)))
    return W, wt


def lanczos(dec, tail):
    """The leading eigenpairs of W by Lanczos (ARPACK), products with G S G* only (module docstring)."""
    from scipy.sparse.linalg import LinearOperator as SLO, eigsh
    from ...settings import settings
    n = dec.n
    W, wt = operator(dec)
    tail = settings.accuracy if tail is None else float(tail)
    kmax = min(n - 1, KMAX)
    k = min(kmax, 64)
    op = SLO((n, n), matvec=W, matmat=W, dtype=float)
    while True:
        lam, U = eigsh(op, k=k, which="LA", tol=1e-12)
        if lam.min() <= tail or k >= kmax:
            break
        k = min(2 * k, kmax)
    R = W(U) - U * lam[None, :]
    res = float(np.linalg.norm(R, axis=0).max())
    dec.partial, dec.tail = True, tail
    dec._claims.append(Claim(
        f"Lanczos partial decomposition: the {k} leading eigenpairs of the {n} x {n} whitened Gram matrix (ARPACK), "
        f"largest residual ||W u - Lam u|| = {res:.2g}; the rest of the spectrum treated as 0", CHECKED,
        f"smallest computed Lam = {lam.min():.3g} (tail {tail:g}, settings.accuracy unless given; kmax {kmax}); "
        "the leading pairs converge first (Lanczos), not proven for the remainder"))
    dec._finish([np.arange(n)], [wt], [(lam, U, res / max(float(lam.max()), 1e-300))], "lanczos", 1)
    dec.route = "lanczos"


def two_sided(E, rho2=1.0):
    """1 / (1 - rho2 E) where rho2 E < 1, else inf (module docstring): the factor that turns the Nystrom error E ||R w||^2
    into a bound on |w^T [R - (rho2 W + I)^-1] w| for a symmetric E = W - W_k of either sign (plan 054 P3)."""
    x = np.asarray(rho2, float) * float(E)
    with np.errstate(divide="ignore"):
        return np.where(x < 1.0, 1.0 / np.maximum(1.0 - x, 1e-300), np.inf)


def std_error(var, vb):
    """The larger side of the std error from a two-sided variance error vb: the true std lies in
    [sqrt(max(var - vb, 0)), sqrt(var + vb)], so it is within this of sqrt(var)."""
    var = np.maximum(np.asarray(var, float), 0.0)
    std = np.sqrt(var)
    with np.errstate(invalid="ignore"):
        return np.maximum(np.sqrt(var + vb) - std, std - np.sqrt(np.maximum(var - vb, 0.0)))


def _orth(V, Q):
    """Orthonormal columns spanning V with Q's span removed (twice, for stability); columns with |R_jj| at most 1e-12
    times the largest dropped. Bookkeeping, measured (plan 054 P2, tests/test_error_account_cuts.py): it keeps the basis
    orthonormal; any orthonormal Q gives 0 <= W_k <= W + nu I in exact arithmetic, and dec.error is measured a posteriori on W - W_k, so the stated
    bound does not depend on the cut (1e-12, eps n and 1e-6 gave the same rank and bound; keeping every column made
    the basis worse)."""
    for _ in range(2):
        if Q.shape[1]:
            V = V - Q @ (Q.T @ V)
    Qv, R = np.linalg.qr(V)
    keep = np.abs(np.diag(R)) > 1e-12 * max(float(np.abs(np.diag(R)).max(initial=0.0)), 1e-300)
    return Qv[:, keep]


def nystrom(dec, tail=None, seed=0, kmax=None):
    """The randomised Nystrom route (module docstring): sets the decomposition's eigenpairs of W_k and dec.error, the
    probabilistic bound on ||W - W_k||."""
    from ...settings import settings
    n = dec.n
    W, wt = operator(dec)
    tail = settings.accuracy if tail is None else float(tail)
    kmax = min(n, KMAX if kmax is None else int(kmax))
    rng = np.random.default_rng(seed)
    Q, Y = np.zeros((n, 0)), np.zeros((n, 0))
    q, est, prev, products = 0, math.inf, math.inf, 0
    lam = U = None
    while True:
        b = min(BLOCK, kmax - Q.shape[1])
        if b <= 0:
            break
        V = _orth(rng.standard_normal((n, b)), Q)
        for _ in range(q):                                  # power iterations against slow spectral decay
            V = _orth(W(V), Q)
            products += V.shape[1]
        if not V.shape[1]:
            break
        Q = np.c_[Q, V]
        Y = np.c_[Y, W(V)]
        products += V.shape[1]
        lam, U = _stable_nystrom(Q, Y)
        est = _norm(W, U, lam, n, rng, 2)                   # fresh probes of E = W - W_k
        products += 2 * PROBES
        if est <= tail or Q.shape[1] >= kmax:
            break
        if est > 0.5 * prev and q < 3:                      # slow decay: one more power iteration from now on
            q += 1
        prev = est
    if lam is None:
        raise Unsupported("the randomised route found no range (W = 0?)")
    est = _norm(W, U, lam, n, rng, 3)                       # the stated value: one probe set, so its probability is
    products += 3 * PROBES                                  # 1 - 1e-10 (a minimum of two would be 1 - 2e-10)
    dec.partial, dec.tail, dec.error = True, tail, est
    dec._claims.append(Claim(
        f"randomised Nystrom partial decomposition of the {n} x {n} whitened Gram matrix: rank {len(lam)} "
        f"({products} products with G S G*, power iterations up to {q}); ||W - W_k|| <= {est:.3g}, of either sign "
        f"(tail {tail:g}" + (", the rank cap was reached first" if est > tail else "") + ")", CHECKED,
        f"{PROBES} Gaussian probes of W - W_k: the bound holds with probability >= 1 - 1e-{PROBES} (Halko, Martinsson "
        "and Tropp 2011, Lemma 4.1); W_k <= W only to rounding (plan 054 P3), so the bounds that use it are two-sided; "
        "the rest of the spectrum treated as 0 (exact for W_k)"))
    dec._finish([np.arange(n)], [wt], [(lam, U, est / max(float(lam.max()), 1e-300))], "randomised", 1)
    dec.route = "randomised"


def _norm(W, U, lam, n, rng, q):
    """||W - W_k|| <= (10 sqrt(2/pi) max_i ||(W - W_k)^q omega_i||)^(1/q) with probability >= 1 - 10^-PROBES."""
    V = rng.standard_normal((n, PROBES))
    for _ in range(q):
        V = W(V) - U @ (lam[:, None] * (U.T @ V))
    return float((LEMMA * np.linalg.norm(V, axis=0).max()) ** (1.0 / q))


def _stable_nystrom(Q, Y):
    """(Lam, U) with W_k = U diag(Lam) U^T = Y (Q^T Y)^+ Y^T, by the shifted Cholesky form (Tropp et al. 2017)."""
    nu = math.sqrt(Y.shape[0]) * np.finfo(float).eps * max(float(np.linalg.norm(Y, 2)), 1e-300)
    Yn = Y + nu * Q
    M = Q.T @ Yn
    M = 0.5 * (M + M.T)
    try:
        C = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:                           # rounding: a clipped eigendecomposition instead
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, nu)
        C = V * np.sqrt(w)[None, :]
    B = np.linalg.solve(C, Yn.T).T                          # Yn C^-T
    Ub, s, _ = np.linalg.svd(B, full_matrices=False)
    lam = np.maximum(s * s - nu, 0.0)
    keep = lam > 0
    order = np.argsort(lam[keep])
    return lam[keep][order], Ub[:, keep][:, order]


def pcg(W, U, lam, B, rho2=1.0, tol=1e-10, maxiter=500):
    """Block CG on (rho2 W + I) Z = B (columns), deflation-preconditioned by the decomposition (module docstring).
    Returns (Z, iterations, largest relative residual)."""
    shrink = rho2 * lam / (1.0 + rho2 * lam)

    def A(V):
        return rho2 * W(V) + V

    def P(V):
        return V - U @ (shrink[:, None] * (U.T @ V))
    nb = np.linalg.norm(B, axis=0)
    Z = P(B)                                                # the low-rank solution as the start
    R = B - A(Z)
    Zr = P(R)
    Pd = Zr.copy()
    rz = np.einsum("ij,ij->j", R, Zr)
    it = 0
    live = np.linalg.norm(R, axis=0) > tol * np.maximum(nb, 1e-300)
    while live.any() and it < maxiter:
        AP = A(Pd[:, live])
        alpha = rz[live] / np.einsum("ij,ij->j", Pd[:, live], AP)
        Z[:, live] += alpha[None, :] * Pd[:, live]
        R[:, live] -= alpha[None, :] * AP
        Zr_l = P(R[:, live])
        rz_new = np.einsum("ij,ij->j", R[:, live], Zr_l)
        Pd[:, live] = Zr_l + (rz_new / rz[live])[None, :] * Pd[:, live]
        rz[live] = rz_new
        it += 1
        live = np.linalg.norm(R, axis=0) > tol * np.maximum(nb, 1e-300)
    rel = float((np.linalg.norm(R, axis=0) / np.maximum(nb, 1e-300)).max(initial=0.0))
    return Z, it, rel
