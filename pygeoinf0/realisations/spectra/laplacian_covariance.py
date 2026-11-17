"""Bessel-potential covariances of the Laplacian on balls and shells, degree by degree (plan 009 G5 with G6; EXP-030
DESIGN G5, NOTES.md section 5).

    C = pg.Laplacian(B, bc={...}).bessel_potential(kappa, q, degrees=range(2, 15, 2), modes=256)

C = (+)_(s in S) c_s (1 + kappa^-2 L_s)^(-q/2) (x) I_(H_s) on L2 of the ball or shell (functional calculus of the
self-adjoint -Delta with rotation-invariant conditions, restricted to the degrees S), a pg.Covariance.spectral whose
known part is the KL truncation to N_s eigenpairs per degree: generators phi_(s,n)(r) Y_(s,t)(xi), orthonormal in
L2(dV), stored as one separable family (radial factors once per (s, n): realisations/series/radial_harmonic.SeparableRadialHarmonics), and weights
c_s f(lambda_(s,n)), f(lambda) = (1 + lambda/kappa^2)^(-q/2). Its omitted part R >= 0 has, with g_s(n) the lower bounds
of bessel.RadialBessel.lower_bound (valid for every n, so for the omitted ones),

    tr R <= sum_s c_s (2s + 1) [sum_(n = N_s + 1)^M f(g_s(n)) + (L/pi) kappa^q y_M^(1 - q)/(q - 1)],
                                                            y_M = (M - theta) pi / L, the integral comparison
    ||R|| <= max_s c_s f(g_s(N_s + 1))

(f decreasing, g_s(n) >= c + ((n - theta) pi/L)^2, theta the interlacing shift). On a finite band the trace is finite
iff q > 1 (lambda_n ~ (n pi / L)^2, Weyl): q <= 1 is refused. The Gram entries of separable functionals follow from
C.variance / C.form_bracket with the bound ||R|| ||P^perp g|| ||P^perp g'||.
"""
import math

import numpy as np

from ...core.claims import CHECKED, DERIVED, THEOREM, Claim, by_rule
from ...core.errors import Refused, Unsupported


def matern_tail(f, lower, N, A, theta, sigma, kappa, q, M_extra=200_000):
    """sum_(n > N) f(lambda_n) for f(lambda) = (1 + lambda/kappa^2)^(-q/2) decreasing and lambda_n >= lower(n) >=
    A (n - theta)^2 - sigma for every n: the explicit sum of f(lower(n)) to M, then the integral comparison
    int_M^inf (kappa^2 / (A' (x - theta)^2))^(q/2) dx = (kappa^2/A')^(q/2) (M - theta)^(1 - q)/(q - 1),
    A' = A - sigma/(M - theta)^2 (module docstring)."""
    M = N + M_extra
    n = np.arange(N + 1, M + 1, dtype=float)
    head = float(np.sum(f(np.maximum(lower(n), -kappa ** 2 * (1 - 1e-15)))))
    Ap = A - sigma / (M - theta) ** 2
    return head + (kappa ** 2 / Ap) ** (0.5 * q) * (M - theta) ** (1 - q) / (q - 1)


def bessel_potential(lap, kappa, q, degrees, modes=None, rtol=None, weights=None, space=None, name="C"):
    from ...spaces.elements import Family
    from ...spaces.base import L2
    from ...probability.covariance import Covariance
    from ..series import legendre as LG
    if not q > 1:
        raise Refused(f"(1 + L/kappa^2)^(-q/2) with q = {q:g} is not trace class on a band of degrees: its eigenvalues "
                      "decay like n^(-q) (Weyl: lambda_n ~ (n pi/L)^2), and sum n^(-q) diverges for q <= 1",
                      needs=["q > 1"])
    if modes is None and rtol is None:
        raise ValueError("give modes (eigenpairs per degree) or rtol (omitted trace <= rtol times the known trace)")
    S = sorted({int(s) for s in degrees})
    if not S:
        raise ValueError("at least one degree")
    X = space if space is not None else L2(lap.region, realisation="radial_harmonics")
    if getattr(X.backend, "harmonic", None) != "radial" or X.domain != lap.region:
        raise Unsupported(f"the covariance is assembled in the radial x harmonic realisation of L2({lap.region!r}); "
                          f"{X} is realised {getattr(X, 'realisation', '?')}")
    if X.backend.c != 1.0 or X.backend.rho is not None:
        raise Unsupported(f"the Laplacian's eigenfunctions are orthonormal in L2 with Lebesgue measure; on {X} (a density, "
                          "plan 064) the Bessel potential as sum lam phi (x) phi is not derived here")
    c = {s: 1.0 for s in S} if weights is None else {s: float(weights(s) if callable(weights) else weights[s])
                                                     for s in S}
    if any(v < 0 for v in c.values()):
        raise ValueError("degree weights are >= 0")
    f = lambda lam: (1.0 + np.asarray(lam, float) / kappa ** 2) ** (-0.5 * q)   # noqa: E731
    L = lap.b - lap.a
    pls, lams, lms, cols, tails, tops, orth = [], [], [], [], [], [], 0.0
    spectra = {}
    col0, parts = 0, {}
    for s in S:
        op = lap.radial(s)
        sp = op.spectrum()
        theta = 0.5 if sp.special else float(op.free_ends())
        if modes is not None:
            N = int(modes)
        else:
            N = 8
            while True:
                lam = sp.eigenvalues(N)
                known = float(np.sum(f(lam)))
                if matern_tail(f, sp.lower_bound, N, (math.pi / L) ** 2, theta, 0.0, kappa, q, 20_000) <= \
                        rtol * known or N > 4096:
                    break
                N *= 2
        lam = sp.eigenvalues(N)
        tails.append(c[s] * (2 * s + 1) * matern_tail(f, sp.lower_bound, N, (math.pi / L) ** 2, theta, 0.0, kappa, q))
        tops.append(c[s] * float(f(sp.lower_bound(np.array([N + 1.0])))[0]))
        P, unresolved = sp.pl(N)
        if unresolved:
            raise ArithmeticError(f"the radial eigenfunctions of degree {s} were not resolved")
        orth = max(orth, float(np.abs(LG.gram_weighted(P, P, _r2_pl(lap.a, lap.b)) - np.eye(N)).max()))
        pls.append(P)
        parts[s] = ("radial", P, c[s] * f(lam))
        spectra[s] = dict(spectrum=sp, c=c[s], f=f, N=N, theta=theta, L=L, kappa=kappa, q=q)
        for t in range(2 * s + 1):
            lms.append(s * s + t + np.zeros(N, int))
            cols.append(col0 + np.arange(N))
            lams.append(c[s] * f(lam))
        col0 += N
    radial = LG.hstack(pls) if len(pls) > 1 else pls[0]
    R = X.backend.separable(radial, np.concatenate(lms), np.concatenate(cols), L=max(S))
    F = Family(X, R, name="φY")
    tail, top = float(sum(tails)), float(max(tops))
    reason = ("lower bounds lambda_n >= s(s + 1)/b^2 + ((n - theta) pi/L)^2 for every n (NOTES.md section 5(d)), an "
              "explicit sum and the integral comparison (realisations/spectra/laplacian_covariance.py)")
    claims = [Claim("commutes with rotations", THEOREM, "a function of -Delta with rotation-invariant conditions, "
                    "restricted to a band of degrees: (+)_s c_s f(L_s) (x) I (Schur; EXP-030 Theorem G2.1)",
                    key="rotation invariant"),
              Claim(f"(+)_s c_s (1 + kappa^-2 L_s)^(-{q:g}/2) (x) I over degrees {S}, kappa = {kappa:g}; {lap.label()}",
                    DERIVED, "functional calculus degree by degree", key="definition"),
              Claim(f"the generators are orthonormal to {orth:.1e} (realised radial factors)", CHECKED,
                    "the Gram matrix of the fitted eigenfunctions in L2(r^2 dr)", key="orthonormality")]
    from ...probability.covariance import REFINE_MODES

    def refine():                             # the same covariance with more of it known (Covariance.sharper_error)
        return bessel_potential(lap, kappa, q, S, modes=None if modes is None else REFINE_MODES * int(modes),
                                rtol=None if modes is not None else rtol / REFINE_MODES ** (q - 1), weights=weights,
                                space=X, name=name)
    out = Covariance.spectral(np.concatenate(lams), F, tail=tail, tail_max=top, status=DERIVED, reason=reason,
                              name=name, claims=claims, refine=refine)
    out.degree_parts = parts                  # per degree (P_s, c_s f(lambda)): structured Grams (equivariance.py)
    out.degree_spectra = spectra              # closed-form spectra: sharper tails of congruences (degrees.py)
    return out


def _r2_pl(a, b):
    from ..series.radial_harmonic import r2_density
    return r2_density(a, b)


def interval_bessel_potential(op, kappa, q, modes, space=None, name="C"):
    """(1 + kappa^-2 L)^(-q/2) for a regular Sturm-Liouville operator L on (a, b), as a pg.Covariance.spectral on
    L2((a, b), w dx) (the given space, or built: pg.L2 of the interval, with the density w unless w = 1); the known part
    is the first `modes` Galerkin eigenpairs, the omitted part bounded through the tail constants of the spectrum.
    Eigenvalues enter through their upper (Galerkin) values; the bracket of each is in op.spectrum().brackets(n)."""
    from ...spaces.elements import Family
    from ...measures import DensityMeasure, LebesgueMeasure
    from ...spaces.base import L2
    from ...probability.covariance import Covariance
    if not q > 1:
        raise Refused(f"(1 + L/kappa^2)^(-q/2) with q = {q:g} is not trace class on an interval (eigenvalues ~ n^2)",
                      needs=["q > 1"])
    sp = op.spectrum()
    if space is None:
        I = op.interval
        space = L2(I) if isinstance(op.w_src, (int, float)) and float(op.w_src) == 1.0 else \
            L2(DensityMeasure(LebesgueMeasure(I), op.w, name="w"))
    f = lambda lam: (1.0 + np.asarray(lam, float) / kappa ** 2) ** (-0.5 * q)   # noqa: E731
    N = int(modes)
    low, up = sp.brackets(N)
    P, _ = sp.pl(N)
    F = Family(space, P, name="φ")
    A, theta, sigma = sp.tail_constants()
    tail = matern_tail(f, sp.lower_bound, N, A, theta, sigma, kappa, q)
    top = float(f(max(float(sp.lower_bound(np.array([N + 1.0]))[0]), -kappa ** 2 * (1 - 1e-15))))
    # the known weights use the Galerkin values; their exact values are larger by at most f(low) - f(up)
    gap = float(np.sum(f(low) - f(up)))
    st = CHECKED                       # R >= 0 needs exact eigenfunctions; the Galerkin ones are approximations
    sa = op.claims.status("self-adjoint")[1]          # the functional calculus of L: as L is self-adjoint (plan 071 P2)
    claims = [Claim(f"(1 + kappa^-2 L)^(-{q:g}/2), kappa = {kappa:g}; {op.label()}", by_rule(sa), "functional calculus "
                    f"of the self-adjoint L ({sa})", key="definition"),
              Claim(f"eigenvalues used at their Galerkin upper bounds; the exact weights differ by <= {gap:.3g} in sum",
                    st, "the eigenvalue brackets of the spectrum (sl_galerkin.py)", key="eigenvalue brackets")]
    from ...probability.covariance import REFINE_MODES
    return Covariance.spectral(f(up), F, tail=tail + gap, tail_max=max(top, float(np.max(f(low) - f(up)))),
                               status=st, reason="Galerkin eigenpairs; tail by the comparison lower bounds and the "
                               "integral comparison (realisations/spectra/laplacian_covariance.py)", name=name, claims=claims,
                               refine=lambda: interval_bessel_potential(op, kappa, q, REFINE_MODES * N, space, name))
