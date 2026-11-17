"""Derivatives through the port (plan 019 P2, EXP-034 DESIGN.md 2.3-2.4 and checks 5.1, 5.2, 5.5): the three derivative
providers, adjoints in the spaces' inner products, functionals and gradients (the same derivative, a pg.Functional, different gradients in
different spaces: plan 068 P2 made the covector a pg.Functional of spaces/dual.py), Taylor rates with kinks and wrong derivatives detected, and the refusals."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.claims import CHECKED

I = pg.Interval("[0, 1]")
X = np.array([0.15, 0.4, 0.55, 0.9])


def H(s):
    return pg.Sobolev(I, s, length_scale=0.2)


def g(x):
    return np.array([x[0] * x[1], np.sum(x ** 2)])


def jvp(x, v):
    return np.array([v[0] * x[1] + x[0] * v[1], 2 * x @ v])


def vjp(x, w):
    out = 2 * w[1] * x
    out[0] += w[0] * x[1]
    out[1] += w[0] * x[0]
    return out


def jac(x):
    J = np.zeros((2, len(x)))
    J[0, 0], J[0, 1] = x[1], x[0]
    J[1] = 2 * x
    return J


D2 = pg.Euclidean(2, covariance=np.array([0.1, 0.4]))


def domains():
    """(space, coordinates, a point) for the three kinds of domain of check 5.1."""
    L = pg.L2(pg.LebesgueMeasure(I))
    cells = [pg.Interval(f"[{a}, {b}]") for a, b in ((0, 0.2), (0.2, 0.5), (0.5, 0.7), (0.7, 1))]
    E = pg.Euclidean(4, weights=np.array([1.0, 2.0, 3.0, 4.0]))
    return [(H(2), pg.port.Coordinates.nodal(H(2), X), H(2)(lambda t: 1 + np.sin(2 * t))),
            (L, pg.port.Coordinates.averages(L, cells), L(lambda t: 1 + t ** 2)),
            (E, pg.port.Coordinates.identity(E), E(np.array([1.0, -0.5, 2.0, 0.3])))]


@pytest.mark.parametrize("case", range(3))
@pytest.mark.parametrize("route", ["jvp", "jacobian"])
def test_taylor_rate_two_and_adjoint_identity(case, route):
    M, C, theta = domains()[case]
    kw = {"jvp": jvp, "vjp": vjp} if route == "jvp" else {"jacobian": jac}
    G = pg.port.External(M, D2, g, coords=(C, None), name="G", **kw)
    res = pg.port.taylor_test(G, theta, directions=4)
    assert all(r > 1.9 for r in res.rates), res
    assert res.adjoint_gap < 1e-10, res
    assert res.claims.status("derivative") == (pg.Truth.TRUE, CHECKED)
    assert res.claims.status("adjoint") == (pg.Truth.TRUE, CHECKED)


def test_jvp_and_jacobian_routes_agree():
    M, C, theta = domains()[0]
    A = pg.port.External(M, D2, g, coords=(C, None), jvp=jvp, vjp=vjp).derivative(theta)
    B = pg.port.External(M, D2, g, coords=(C, None), jacobian=jac).derivative(theta)
    h = M(lambda t: np.cos(5 * t))
    y = D2(np.array([0.3, -1.2]))
    assert np.allclose(A(h).values, B(h).values, atol=1e-13)
    assert M.distance(A.adjoint(y), B.adjoint(y)) < 1e-12 * M.norm(A.adjoint(y))


# ------------------------------------------------------------------------- covectors and gradients (check 5.2)
B = np.array([0.5, 1.0, 0.2, -0.3])


def misfit(M):
    """J(m) = 1/2 sum_i (m(x_i) - b_i)^2 through nodal coordinates, its derivative as a Jacobian."""
    C = pg.port.Coordinates.nodal(M, X)
    return pg.port.External(M, pg.Euclidean(1), lambda x: np.array([0.5 * np.sum((x - B) ** 2)]), coords=(C, None),
                       jacobian=lambda x: (x - B)[None, :], name="J"), C


def test_same_covector_different_gradients():
    f = lambda t: np.sin(3 * t)
    h = lambda t: np.exp(t) - t ** 3
    dJ_h = np.sum((f(X) - B) * h(X))                              # DJ(f) h, by hand
    grads = []
    for s in (1, 2):
        M = H(s)
        J, C = misfit(M)
        theta, hh = M(f), M(h)
        r = pg.gradient(J, theta)
        assert abs(M.inner(r, hh) - dJ_h) < 1e-10 * abs(dJ_h)      # the Riesz property in this space
        w = pg.Functional.from_operator(J.derivative(theta))
        assert abs(w(hh) - dJ_h) < 1e-12 * abs(dJ_h)              # the covector does not depend on the space
        r2 = C.functional(f(X) - B).gradient()                       # what a tool's coordinate gradient means here
        assert M.distance(r, r2) < 1e-10 * M.norm(r)
        grads.append(np.array([r(t) for t in np.linspace(0, 1, 11)]))
    assert np.max(np.abs(grads[0] - grads[1])) > 1e-2                # different spaces, different gradients


def test_gradient_needs_an_inner_product():
    Lp = pg.Lp(pg.LebesgueMeasure(I), 3)
    w = pg.Functional.from_operator(pg.LinearOperator(Lp, pg.Euclidean(1), lambda f: pg.Euclidean(1)(np.array([f.integral()]))))
    with pytest.raises(pg.Refused, match="no inner product"):
        w.gradient()


def test_functional_from_a_measure():
    M = H(1)
    w = pg.Functional.from_measure(M, pg.LebesgueMeasure(pg.Interval("[0.2, 0.6]")))
    m = M(lambda t: t)
    assert abs(w(m) - (0.6 ** 2 - 0.2 ** 2) / 2) < 1e-12
    assert abs(M.inner(w.gradient(), m) - w(m)) < 1e-10


# ------------------------------------------------------------------------- the geometry route
def test_derivative_given_as_measures():
    """G(m)_i = (int_{J_i} m)^2 through cell averages; its derivative at m is the measure 2 (int_{J_i} m) dx on J_i."""
    L = pg.L2(pg.LebesgueMeasure(I))
    edges = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
    cells = [pg.Interval(f"[{a}, {b}]") for a, b in edges]
    widths = np.array([b - a for a, b in edges])
    C = pg.port.Coordinates.averages(L, cells)
    G = pg.port.External(L, pg.Euclidean(3), lambda x: (widths * x) ** 2, coords=(C, None), name="G",
                    measures=lambda x: [pg.LebesgueMeasure(c).with_density(2 * w * v)
                                        for c, w, v in zip(cells, widths, x)])
    res = pg.port.taylor_test(G, L(lambda t: 1 + t), directions=3)
    assert all(r > 1.9 for r in res.rates), res
    assert res.adjoint_gap < 1e-10
    assert G.derivative(L(lambda t: 1 + t)).claims.status("derivative")[0]


# ------------------------------------------------------------------------- kinks and wrong derivatives (5.5)
def test_kink_and_wrong_derivative_are_reported():
    M, C, theta = domains()[0]
    c = C.numbers(theta)[0]
    kink = pg.port.External(M, pg.Euclidean(1), lambda x: np.array([abs(x[0] - c)]), coords=(C, None),
                       jacobian=lambda x: np.r_[np.sign(x[0] - c), np.zeros(3)][None, :], name="K")
    res = pg.port.taylor_test(kink, theta, directions=3)
    assert res.claims.status("derivative") == (pg.Truth.FALSE, CHECKED)
    assert all(abs(r - 1) < 0.2 for r in res.rates)
    wrong = pg.port.External(M, D2, g, coords=(C, None), jacobian=lambda x: 1.1 * jac(x), name="W")
    res = pg.port.taylor_test(wrong, theta, directions=3)
    assert res.claims.status("derivative") == (pg.Truth.FALSE, CHECKED)


# ------------------------------------------------------------------------- refusals and failures
def test_providers_refusals_and_failures():
    M, C, theta = domains()[0]
    with pytest.raises(ValueError, match="one derivative provider"):
        pg.port.External(M, D2, g, coords=(C, None), jvp=jvp, jacobian=jac)
    with pytest.raises(ValueError, match="give jvp too"):
        pg.port.External(M, D2, g, coords=(C, None), vjp=vjp)
    with pytest.raises(pg.Refused, match="no derivative provider"):
        pg.port.External(M, D2, g, coords=(C, None)).derivative(theta)
    forward = pg.port.External(M, D2, g, coords=(C, None), jvp=jvp).derivative(theta)
    forward(M(lambda t: t))
    with pytest.raises(NotImplementedError, match="no adjoint"):
        forward.adjoint(D2(np.array([1.0, 1.0])))
    from pygeoinf0.sets import PredicateSubset
    dom = PredicateSubset(M, lambda m: bool(C.numbers(m)[0] > 0), name="x_1 > 0")
    G = pg.port.External(M, D2, g, coords=(C, None), dom=dom, jacobian=jac, name="G")
    with pytest.raises(pg.Refused, match="no derivative there"):
        G.derivative(M(lambda t: -1 + 0 * t))
    broken = pg.port.External(M, D2, g, coords=(C, None), jacobian=lambda x: 1 / 0, code="solver", name="S")
    with pytest.raises(pg.ExternalError, match="solver failed computing the Jacobian of S"):
        broken.derivative(theta)


# ------------------------------------------------------------------------- linearise and precision (plan 020 P2)
def test_linearise_is_called_once_per_derivative():
    M, C, theta = domains()[0]
    calls = []

    def linearise(x):
        calls.append(1)
        J = jac(x)                                         # the shared state: computed once per model
        return (lambda v: J @ v), (lambda w: J.T @ w)

    G = pg.port.External(M, D2, g, coords=(C, None), linearise=linearise, name="G")
    A = G.derivative(theta)
    for k in range(5):
        A(M(lambda t, k=k: np.cos(k * t)))
        A.adjoint(D2(np.array([1.0, k])))
    assert len(calls) == 1
    res = pg.port.taylor_test(G, theta, directions=3)
    assert all(r > 1.9 for r in res.rates) and res.adjoint_gap < 1e-10
    forward_only = pg.port.External(M, D2, g, coords=(C, None), linearise=lambda x: (lambda v: jac(x) @ v, None))
    with pytest.raises(NotImplementedError, match="no adjoint"):
        forward_only.derivative(theta).adjoint(D2(np.array([1.0, 1.0])))
    with pytest.raises(ValueError, match="one derivative provider"):
        pg.port.External(M, D2, g, coords=(C, None), jacobian=jac, linearise=linearise)


def test_single_precision_derivative_is_stated_and_cited():
    M, C, theta = domains()[0]
    lin = lambda x: (lambda v: (jac(x).astype(np.float32) @ v.astype(np.float32)).astype(float),
                     lambda w: (jac(x).astype(np.float32).T @ w.astype(np.float32)).astype(float))
    G = pg.port.External(M, D2, g, coords=(C, None), linearise=lin, derivative_precision="float32", name="G32")
    assert G.claims.status("precision") == (pg.Truth.FALSE, "asserted")
    res = pg.port.taylor_test(G, theta, directions=2)
    failed = [c for c in res.claims if not c.holds]                   # the rate's median may survive; the adjoint not
    assert failed and all("the provider states: derivative in float32" in c.reason for c in failed)
