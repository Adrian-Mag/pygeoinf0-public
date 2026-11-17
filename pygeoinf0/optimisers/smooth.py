"""Smooth optimisers in a Hilbert space (plan 019 F5; moved from nonlinear/ by plan 063): Levenberg-Marquardt for
nonlinear least squares and L-BFGS, both written only against the port (a map with .derivative and a declared dom) and
the spaces' inner products; their result, NonlinearResult.

Gauss-Newton with a trust region (Levenberg-Marquardt) for nonlinear least squares in a Hilbert space (plan 019 F5,
experiments/EXP-034-external-implementations/DESIGN.md section 3).

    gauss_newton(G, d, theta, alpha=0.0, theta0=None, tol=1e-8, maxiter=50)  ->  NonlinearResult

minimises Phi(theta) = 1/2 ||G(theta) - d||_D^2 + alpha/2 ||theta - theta0||_M^2 over dom G, for a map G : M -> D with a
derivative (pg.port.External with a derivative provider, or any map with .derivative(theta)) and a data space D of finite
dimension k. Each step solves the linearised problem with a Levenberg-Marquardt term mu/2 ||delta||_M^2 in the data
space: with A = DG(theta), r = d - G(theta), u = theta0 - theta and lambda = alpha + mu,

    delta = (alpha / lambda) u + A* z,    (A A* + lambda I) z = r - (alpha / lambda) A u      (a k x k system)

which is the normal equation (A* A + lambda I) delta = A* r + alpha u rewritten through A* (A A* + lambda)^-1 =
(A* A + lambda)^-1 A*. The ratio rho of the actual to the predicted reduction of Phi sets mu (rho > 3/4: mu / 3;
rho < 1/4: mu * 2); steps with rho <= 1e-4, or that leave dom, are rejected. The stationarity measure is the norm in M
of grad Phi = A* (G(theta) - d) + alpha (theta - theta0): it depends on M's inner product, and so does every step.

Near a minimum Phi - Phi* ~ 1/2 g' H^-1 g falls below the rounding of Phi (~ eps Phi): no step can then show a
decrease, and the gradient cannot be pushed below ~ sqrt(eps Phi ||H||). When the first (least damped) trial step of an
iteration predicts a decrease below ROUNDING * |Phi|, the run stops there as a stationary point to rounding, and the
claim states that floor and the gradient reached (wsl#1's diagnosis of a europa failure, 2026-10-30). The least damped
step is the one with the initial mu: a mu driven up by rejected steps (e.g. steps that leave dom) also shrinks the
predicted decrease and must not pass for the floor (mac#2's report from the Moon problem, 2026-10-30); a run whose mu
is exhausted that way ends as not converged, with the reason.

The result is a stationary point to the stated tolerance or to rounding (a checked claim), never a global minimum.

L-BFGS in the inner product of a Hilbert space (plan 019 F5, experiments/EXP-034-external-implementations/DESIGN.md
section 3).

    lbfgs(J, theta, alpha=0.0, theta0=None, tol=1e-8, maxiter=200, memory=8)  ->  NonlinearResult

minimises Phi(theta) = J(theta) + alpha/2 ||theta - theta0||_M^2 for a scalar map J : M -> R with a derivative (its
codomain a 1-dimensional Euclidean space), over J's declared dom. Gradients are Riesz representers in M (pg.gradient),
and the two-loop recursion uses M's inner product, so the method is L-BFGS preconditioned by M's Riesz map: changing
M's inner product changes every step (and is the honest way to precondition). Backtracking line search with the Armijo
condition; curvature pairs with <s, y> <= 0 are skipped. When no step decreases Phi in floating point, the run ends:
as a stationary point to rounding if the predicted decrease |<grad, p>| is below ROUNDING * |Phi| (the floor is stated
in the claim), otherwise as not converged (stalled). The result is a stationary point to the stated tolerance (a
checked claim), never a global minimum.
"""
from dataclasses import dataclass, field

import numpy as np

from ..core.claims import CHECKED, DERIVED, Claim, Claims
from ..core.errors import Refused
from ..core.logic import Truth
from ..spaces.dual import gradient


@dataclass
class NonlinearResult:
    theta: object               # the final point (an element of the model space)
    value: float                # the objective there
    gradient_norm: float        # the stationarity measure, in the model space's norm
    iterations: int
    converged: bool
    history: list               # one dict per iteration
    claims: Claims = field(default_factory=Claims)

    def __repr__(self):
        return (f"NonlinearResult(value {self.value:.6g}, ||grad|| {self.gradient_norm:.3e}, {self.iterations} "
                f"iterations, {'converged' if self.converged else 'not converged'})\n{self.claims!r}")


ROUNDING = 64 * np.finfo(float).eps          # a predicted decrease below ROUNDING * |Phi| cannot be measured


def _in_dom(G, theta):
    dom = getattr(G, "_dom", None)
    return dom is None or dom.member(theta) is not Truth.FALSE


def _real_coordinates(D):
    """(basis, to_real, from_real) of the data space as a real space: R^k, or C^k as R^2k (plan 020 P3a)."""
    if hasattr(D, "real_basis"):
        return D.real_basis(), D.to_real, (lambda z: D(D.from_real(z)))
    return [D(e) for e in np.eye(D.dim)], (lambda v: np.asarray(v, float)), (lambda z: D(z))


def _gram(A):
    """The matrix of A A* in the data space's real coordinates, column by column."""
    basis, to_real, _ = _real_coordinates(A.codomain)
    return np.array([to_real(A(A.adjoint(b)).values) for b in basis]).T


def _lm_step(st, mu, alpha, theta0, M, D):
    """The Levenberg-Marquardt step for damping mu at the iteration's state st, and its predicted decrease of Phi."""
    lam = alpha + mu
    A, K = st["A"], st["K"]
    b = -st["to_real"](st["res"].values) - (alpha / lam) * st["Au"]
    z = st["from_real"](np.linalg.solve(K + lam * np.eye(K.shape[0]), b))
    delta = A.adjoint(z) + (alpha / lam) * st["u"] if alpha else A.adjoint(z)
    step = A(delta)
    model = 0.5 * D.norm(st["res"] + step) ** 2 + 0.5 * alpha * M.norm(st["theta"] + delta - theta0) ** 2
    return delta, st["value"] - model


def gauss_newton(G, d, theta, alpha=0.0, theta0=None, tol=1e-8, maxiter=50, mu=None):
    """Levenberg-Marquardt for 1/2 ||G(theta) - d||^2 + alpha/2 ||theta - theta0||^2 (module docstring)."""
    M, D = G.domain, G.codomain
    if not hasattr(G, "derivative"):
        raise Refused(f"{getattr(G, 'name', 'G')} has no derivative", needs=("a derivative provider (pg.port.External)",))
    if not _in_dom(G, theta):
        raise Refused("the starting point is not in the map's declared dom")
    theta0 = M.zero if theta0 is None else theta0
    alpha = float(alpha)

    def phi(t, Gt):
        return 0.5 * D.norm(Gt - d) ** 2 + 0.5 * alpha * M.norm(t - theta0) ** 2

    Gt = G(theta)
    value = phi(theta, Gt)
    history, g0, mu0 = [], None, None
    converged, floor = False, None
    for it in range(maxiter + 1):
        A = G.derivative(theta)
        res = Gt - d
        grad = A.adjoint(res) + alpha * (theta - theta0) if alpha else A.adjoint(res)
        gnorm = M.norm(grad)
        g0 = gnorm if g0 is None else g0
        history.append({"iteration": it, "phi": value, "gradient_norm": gnorm, "mu": mu})
        if gnorm <= tol * max(g0, 1e-300) or gnorm == 0.0:
            converged = True
            break
        if it == maxiter:
            break
        K = _gram(A)
        if mu is None:
            mu = 1e-3 * max(np.trace(K) / K.shape[0], 1e-300)
        mu0 = mu if mu0 is None else mu0                       # the least damping of the run: the floor test's step
        u = theta0 - theta
        _, to_real, from_real = _real_coordinates(D)
        Au = to_real(A(u).values) if alpha else 0.0
        mu_max = 1e16 * max(np.trace(K), 1.0)

        it_state = {"A": A, "K": K, "res": res, "Au": Au, "u": u, "to_real": to_real, "from_real": from_real,
                    "theta": theta, "value": value}
        _, predicted0 = _lm_step(it_state, min(mu, mu0), alpha, theta0, M, D)
        if predicted0 <= ROUNDING * max(abs(value), 1e-300):
            converged, floor = True, predicted0                # below what Phi can show: stationary to rounding
            history[-1]["rounding"] = floor
            break
        while mu <= mu_max:
            delta, predicted = _lm_step(it_state, mu, alpha, theta0, M, D)
            trial = theta + delta
            if not _in_dom(G, trial):
                mu *= 4.0
                history[-1].setdefault("rejected", []).append("left dom")
                continue
            Gtrial = G(trial)
            new = phi(trial, Gtrial)
            rho = (value - new) / predicted if predicted > 0 else -np.inf
            if rho > 0.75:
                mu /= 3.0
            elif rho < 0.25:
                mu *= 2.0
            if rho > 1e-4:
                theta, Gt, value = trial, Gtrial, new
                history[-1].update(rho=rho, step_norm=M.norm(delta))
                break
            history[-1].setdefault("rejected", []).append(f"rho {rho:.2e}")
        if "rho" not in history[-1]:
            break                                              # no acceptable step: mu exhausted
    why = f"||grad Phi||_M = {gnorm:.3e} after {it} steps (initial {g0:.3e}, tol {tol:.0e} relative)"
    stuck = history[-1].get("rejected", [])
    if not converged and stuck and "rho" not in history[-1]:
        left = sum(r == "left dom" for r in stuck)
        why += (f"; no acceptable step: {len(stuck)} trials rejected ({left} left dom), damping exhausted")
    if floor is not None:
        why += (f"; stopped at rounding: the predicted decrease {floor:.1e} is below {ROUNDING:.1e} |Phi| = "
                f"{ROUNDING * abs(value):.1e}")
    claims = [Claim(("a stationary point of Phi to rounding" if floor is not None else "a stationary point of Phi")
                    if converged else "not converged", CHECKED, why, key="stationary", holds=converged),
              Claim("a local result: no claim of a global minimum", DERIVED,
                    "Gauss-Newton converges to stationary points of a nonconvex Phi", key="global minimum",
                    holds=False),
              Claim("steps and the stationarity measure in the inner product of the model space", DERIVED,
                    f"A* is the adjoint in {M}", key="metric")]
    return NonlinearResult(theta, value, gnorm, it, converged, history, Claims(claims))


def lbfgs(J, theta, alpha=0.0, theta0=None, tol=1e-8, maxiter=200, memory=8):
    """L-BFGS in M's inner product for J + alpha/2 ||theta - theta0||^2 (module docstring)."""
    M = J.domain
    if not hasattr(J, "derivative"):
        raise Refused(f"{getattr(J, 'name', 'J')} has no derivative", needs=("a derivative provider (pg.port.External)",))
    if not _in_dom(J, theta):
        raise Refused("the starting point is not in the map's declared dom")
    theta0 = M.zero if theta0 is None else theta0
    alpha = float(alpha)

    def phi(t):
        return float(J(t).values[0]) + (0.5 * alpha * M.norm(t - theta0) ** 2 if alpha else 0.0)

    def grad(t):
        g = gradient(J, t)
        return g + alpha * (t - theta0) if alpha else g

    value, g = phi(theta), grad(theta)
    g0 = M.norm(g)
    pairs, history, converged, floor = [], [], False, None
    for it in range(maxiter + 1):
        gnorm = M.norm(g)
        history.append({"iteration": it, "phi": value, "gradient_norm": gnorm})
        if gnorm <= tol * max(g0, 1e-300) or gnorm == 0.0:
            converged = True
            break
        if it == maxiter:
            break
        q, coef = g, []                                     # two-loop recursion in M's inner product
        for s, y, rho in reversed(pairs):
            a = rho * M.inner(s, q)
            coef.append(a)
            q = q - a * y
        if pairs:
            s, y, _ = pairs[-1]
            q = (M.inner(s, y) / M.inner(y, y)) * q
        else:
            q = (1.0 / gnorm) * q
        for (s, y, rho), a in zip(pairs, reversed(coef)):
            b = rho * M.inner(y, q)
            q = q + (a - b) * s
        p = -1.0 * q
        slope = M.inner(g, p)
        if slope >= 0:                                       # not a descent direction: restart along -g
            pairs, p, slope = [], (-1.0 / gnorm) * g, -gnorm
        step, accepted = 1.0, False
        for _ in range(60):
            trial = theta + step * p
            if _in_dom(J, trial):
                new = phi(trial)
                if new < value and new <= value + 1e-4 * step * slope:
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            if abs(slope) <= ROUNDING * max(abs(value), 1e-300):
                converged, floor = True, abs(slope)
                history[-1]["rounding"] = floor
            else:
                history[-1]["rejected"] = "no decrease of Phi in floating point: stalled"
            break
        g_new = grad(trial)
        s, y = trial - theta, g_new - g
        sy = M.inner(s, y)
        if sy > 0:
            pairs.append((s, y, 1.0 / sy))
            pairs = pairs[-memory:]
        theta, value, g = trial, new, g_new
        history[-1]["step"] = step
    why = f"||grad Phi||_M = {gnorm:.3e} after {it} steps (initial {g0:.3e}, tol {tol:.0e} relative)"
    if floor is not None:
        why += f"; stopped at rounding: the predicted decrease {floor:.1e} is below {ROUNDING:.1e} |Phi|"
    claims = [Claim(("a stationary point of Phi to rounding" if floor is not None else "a stationary point of Phi")
                    if converged else "not converged", CHECKED, why, key="stationary", holds=converged),
              Claim("a local result: no claim of a global minimum", DERIVED,
                    "quasi-Newton methods converge to stationary points", key="global minimum", holds=False),
              Claim("gradients and curvature pairs in the inner product of the model space", DERIVED,
                    f"Riesz representers in {M}", key="metric")]
    return NonlinearResult(theta, value, gnorm, it, converged, history, Claims(claims))
