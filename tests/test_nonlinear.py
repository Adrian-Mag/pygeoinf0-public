"""The core nonlinear drivers (plan 019 P4, EXP-034 DESIGN.md section 3): Levenberg-Marquardt against the closed form of
a linear Tikhonov problem with metrics on both sides, the Rosenbrock problem for both drivers, the two drivers agreeing
on a nonlinear problem in a Sobolev space (and that problem's minimiser changing with the space's inner product), steps
kept inside dom, and the refusals."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.sets import PredicateSubset
from pygeoinf0.core.claims import CHECKED

I = pg.Interval("[0, 1]")
X = np.array([0.15, 0.4, 0.55, 0.9])


def test_gauss_newton_solves_a_linear_tikhonov_problem_with_metrics():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4, 3))
    w, c, alpha = np.array([1.0, 2.0, 0.5]), np.array([0.1, 0.2, 0.3, 0.4]), 0.5
    M, D = pg.Euclidean(3, weights=w), pg.Euclidean(4, covariance=c)
    G = pg.port.External(M, D, lambda x: A @ x, jacobian=lambda x: A, name="A")
    d = D(rng.standard_normal(4))
    res = pg.optimisers.gauss_newton(G, d, M.zero, alpha=alpha, tol=1e-12)
    exact = np.linalg.solve(A.T @ (A / c[:, None]) + alpha * np.diag(w), A.T @ (d.values / c))
    assert np.allclose(res.theta.values, exact, atol=1e-9)
    assert res.claims.status("stationary") == (pg.Truth.TRUE, CHECKED)


def rosenbrock_residual(x):
    return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def rosenbrock_jacobian(x):
    return np.array([[-20 * x[0], 10.0], [-1.0, 0.0]])


def test_both_drivers_on_rosenbrock():
    M = pg.Euclidean(2)
    start = M(np.array([-1.2, 1.0]))
    G = pg.port.External(M, pg.Euclidean(2), rosenbrock_residual, jacobian=rosenbrock_jacobian, name="r")
    gn = pg.optimisers.gauss_newton(G, pg.Euclidean(2).zero, start, tol=1e-12, maxiter=100)
    assert gn.converged and np.allclose(gn.theta.values, [1.0, 1.0], atol=1e-8)
    J = pg.port.External(M, pg.Euclidean(1), lambda x: np.array([0.5 * np.sum(rosenbrock_residual(x) ** 2)]),
                    jacobian=lambda x: (rosenbrock_jacobian(x).T @ rosenbrock_residual(x))[None, :], name="J")
    lb = pg.optimisers.lbfgs(J, start, tol=1e-10, maxiter=500)
    assert lb.converged and np.allclose(lb.theta.values, [1.0, 1.0], atol=1e-6)
    assert all(h["phi"] >= lb.history[i + 1]["phi"] for i, h in enumerate(lb.history[:-1]))   # monotone


def squares_problem(M):
    """G(m) = (m(x_i)^2), data b, through nodal coordinates; and J = 1/2 ||G(m) - b||^2 for L-BFGS."""
    C = pg.port.Coordinates.nodal(M, X)
    b = np.array([0.3, 1.2, 0.8, 0.1])
    G = pg.port.External(M, pg.Euclidean(4), lambda x: x ** 2, coords=(C, None), jacobian=lambda x: np.diag(2 * x), name="G")
    J = pg.port.External(M, pg.Euclidean(1), lambda x: np.array([0.5 * np.sum((x ** 2 - b) ** 2)]), coords=(C, None),
                    jacobian=lambda x: ((x ** 2 - b) * 2 * x)[None, :], name="J")
    return G, J, pg.Euclidean(4)(b)


def test_drivers_agree_in_a_sobolev_space_and_the_minimiser_depends_on_the_metric():
    alpha, out = 1e-2, []
    for s in (1, 2):
        M = pg.Sobolev(I, s, length_scale=0.2)
        G, J, b = squares_problem(M)
        start = M(lambda t: 0.5 + 0 * t)
        gn = pg.optimisers.gauss_newton(G, b, start, alpha=alpha, tol=1e-10)
        lb = pg.optimisers.lbfgs(J, start, alpha=alpha, tol=1e-8, maxiter=500)
        assert gn.converged and lb.converged
        assert M.distance(gn.theta, lb.theta) < 1e-6 * M.norm(gn.theta)
        stat = pg.gradient(J, gn.theta) + alpha * gn.theta               # grad Phi in M, computed independently
        assert M.norm(stat) < 1e-6 * M.norm(gn.theta)                    # the rounding floor here is ~2e-8
        out.append(np.array([gn.theta(t) for t in np.linspace(0, 1, 21)]))
    assert np.max(np.abs(out[0] - out[1])) > 1e-2                         # H^1 and H^2 regularise differently


def test_steps_stay_in_dom():
    M = pg.Euclidean(1)
    dom = PredicateSubset(M, lambda x: bool(x.values[0] > 0), name="x > 0")
    G = pg.port.External(M, pg.Euclidean(1), lambda x: np.log(x), dom=dom, jacobian=lambda x: np.array([[1 / x[0]]]),
                    name="log")
    res = pg.optimisers.gauss_newton(G, pg.Euclidean(1)(np.array([np.log(0.01)])), M(np.array([1.0])), tol=1e-12, maxiter=100)
    assert res.converged and abs(res.theta.values[0] - 0.01) < 1e-10
    assert any("left dom" in r for h in res.history for r in h.get("rejected", []))   # the first full step overshoots
    with pytest.raises(pg.Refused, match="starting point"):
        pg.optimisers.gauss_newton(G, pg.Euclidean(1).zero, M(np.array([-1.0])))


def test_a_tolerance_below_rounding_ends_at_the_stated_floor():
    """Near the minimum Phi - Phi* falls below the rounding of Phi: both drivers stop there and say so (wsl#1's
    diagnosis of a europa failure: the gradient stalls near sqrt(eps Phi ||H||))."""
    M = pg.Sobolev(I, 2, length_scale=0.2)
    G, J, b = squares_problem(M)
    start = M(lambda t: 0.5 + 0 * t)
    for res in (pg.optimisers.lbfgs(J, start, alpha=1e-2, tol=1e-15, maxiter=500),
                pg.optimisers.gauss_newton(G, b, start, alpha=1e-2, tol=1e-15, maxiter=200)):
        assert res.converged and res.iterations < 200
        c = res.claims.find("stationary")
        assert c.holds and "to rounding" in c.statement and "stopped at rounding" in c.reason
        assert res.gradient_norm < 1e-6 * res.history[0]["gradient_norm"]


def test_drivers_need_a_derivative():
    M = pg.Euclidean(2)
    plain = pg.Map(M, pg.Euclidean(2), lambda x: x)
    with pytest.raises(pg.Refused, match="no derivative"):
        pg.optimisers.gauss_newton(plain, pg.Euclidean(2).zero, M.zero)
    with pytest.raises(pg.Refused, match="no derivative"):
        pg.optimisers.lbfgs(pg.Map(M, pg.Euclidean(1), lambda x: pg.Euclidean(1)(np.array([0.0]))), M.zero)


def test_dom_rejections_do_not_pass_for_the_rounding_floor():
    """mac#2's report from the Moon problem: steps rejected for leaving dom drive mu up, and a floor test at that mu
    claimed stationarity with the gradient still large. The minimum of (x - 2)^2 lies outside dom = {x < 1}."""
    M = pg.Euclidean(1)
    dom = PredicateSubset(M, lambda x: bool(x.values[0] < 1.0), name="x < 1")
    G = pg.port.External(M, pg.Euclidean(1), lambda x: x, dom=dom, jacobian=lambda x: np.eye(1), name="id")
    res = pg.optimisers.gauss_newton(G, pg.Euclidean(1)(np.array([2.0])), M(np.array([0.5])), tol=1e-12, maxiter=60)
    c = res.claims.find("stationary")
    assert not res.converged and not c.holds and "left dom" in c.reason, c
    assert res.gradient_norm > 0.5                                       # far from stationary, and said so
