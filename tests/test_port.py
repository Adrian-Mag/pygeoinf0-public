"""The port (plan 019 P1, EXP-034 DESIGN.md 2.1-2.2): coordinates of spaces for external code (boundedness enforced,
the canonical import, adjoints in the spaces' inner products) and maps given by external code (dom, failures, claims)."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.sets import PredicateSubset
from pygeoinf0.core.claims import ASSERTED, DERIVED, THEOREM, UNKNOWN

I = pg.Interval("[0, 1]")
X = np.array([0.15, 0.4, 0.55, 0.9])


def H(s):
    return pg.Sobolev(I, s, length_scale=0.2)


def adjoint_gap(A, x, y):
    """|<A x, y> - <x, A* y>| relative to the size of the terms."""
    lhs = A.codomain.inner(A(x), y)
    rhs = A.domain.inner(x, A.adjoint(y))
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)


@pytest.mark.parametrize("s", [1, 2])
def test_nodal_coordinates_on_sobolev(s):
    M = H(s)
    C = pg.port.Coordinates.nodal(M, X)
    m = M(lambda t: np.sin(3 * t) + t ** 2)
    assert np.allclose(C.numbers(m), np.sin(3 * X) + X ** 2, atol=1e-12)
    x = np.array([1.0, -2.0, 0.5, 3.0])
    assert np.allclose(C.numbers(C.element(x)), x, atol=1e-10)                         # E P = I
    p = C.element(C.numbers(m))
    assert np.allclose(C.numbers(C.element(C.numbers(p))), C.numbers(p), atol=1e-10)   # P E idempotent
    assert C.error(C.element(x)) < 1e-8 * M.norm(C.element(x))                         # nothing lost on P's range
    assert C.error(m) > 1e-3                                                           # a generic element is not
    R = C.export.codomain
    assert adjoint_gap(C.export, m, R(x)) < 1e-12
    assert adjoint_gap(C.imp, R(x), m) < 1e-10
    assert C.claims.status("bounded") == (pg.Truth.TRUE, DERIVED)
    assert C.claims.status("E P = I") == (pg.Truth.TRUE, DERIVED)


def test_nodal_coordinates_refused_on_l2():
    L = pg.L2(pg.LebesgueMeasure(I))
    with pytest.raises(pg.PointEvaluationUndefined):
        pg.port.Coordinates.nodal(L, X)


def test_average_coordinates_on_l2_and_the_piecewise_constant_import():
    L = pg.L2(pg.LebesgueMeasure(I))
    a, b = np.array([0.0, 0.25, 0.5]), np.array([0.25, 0.5, 1.0])
    cells = [pg.Interval(f"[{lo}, {hi}]") for lo, hi in np.c_[a, b]]
    C = pg.port.Coordinates.averages(L, cells)
    m = L(lambda t: np.sin(3 * t))
    means = (np.cos(3 * a) - np.cos(3 * b)) / (3 * (b - a))
    assert np.allclose(C.numbers(m), means, atol=1e-12)
    x = np.array([2.0, -1.0, 0.5])
    p = C.element(x)
    assert np.allclose(C.numbers(p), x, atol=1e-10)
    mids = [0.1, 0.4, 0.8]                       # the least-norm element with given means is constant on each cell
    assert np.allclose([p(t) for t in mids], x, atol=1e-8)
    assert adjoint_gap(C.export, m, C.export.codomain(x)) < 1e-12


def test_same_numbers_different_adjoints_in_different_spaces():
    """The numbers of an element do not depend on the inner product; the adjoint of the export does (the gradient of
    a function of the numbers will differ between H^1 and H^2: DESIGN 2.3)."""
    C1, C2 = pg.port.Coordinates.nodal(H(1), X), pg.port.Coordinates.nodal(H(2), X)
    f = lambda t: np.cos(2 * t)
    assert np.allclose(C1.numbers(H(1)(f)), C2.numbers(H(2)(f)), atol=1e-12)
    c = np.array([1.0, 0.0, -1.0, 0.5])
    a1, a2 = C1.export.adjoint(C1.export.codomain(c)), C2.export.adjoint(C2.export.codomain(c))
    t = np.linspace(0, 1, 9)
    assert np.max(np.abs(np.array([a1(u) for u in t]) - np.array([a2(u) for u in t]))) > 1e-2


def test_identity_coordinates_with_a_metric():
    D = pg.Euclidean(3, covariance=np.array([1.0, 4.0, 9.0]))
    C = pg.port.Coordinates.identity(D)
    v = D(np.array([1.0, 2.0, 3.0]))
    c = np.array([0.5, -1.0, 2.0])
    assert np.allclose(C.numbers(v), v.values)
    assert np.allclose(C.element(c).values, c)
    assert np.allclose(C.export.adjoint(C.export.codomain(c)).values, c * np.array([1.0, 4.0, 9.0]))  # W^-1 c
    assert adjoint_gap(C.export, v, C.export.codomain(c)) < 1e-14
    assert adjoint_gap(C.imp, C.export.codomain(c), v) < 1e-14
    assert C.claims.status("E P = I") == (pg.Truth.TRUE, THEOREM)
    assert C.error(v) == 0.0


def test_dependent_coordinates_are_refused():
    with pytest.raises(pg.Refused, match="linearly dependent"):
        pg.port.Coordinates.nodal(H(1), [0.3, 0.3, 0.7])


def test_numbers_of_the_wrong_length_are_refused():
    C = pg.port.Coordinates.nodal(H(1), X)
    with pytest.raises(ValueError, match="4 coordinates"):
        C.element([1.0, 2.0])


# ------------------------------------------------------------------------------------------------- external maps
def code(x):
    """A stand-in for external code: two numbers from the nodal values."""
    return np.array([x[0] * x[1], np.sum(x ** 2)])


def test_external_map_through_coordinates():
    M, D = H(1), pg.Euclidean(2, covariance=0.1)
    G = pg.port.External(M, D, code, coords=(pg.port.Coordinates.nodal(M, X), None), name="G")
    m = M(lambda t: 1 + t)
    v = 1 + X
    y = G(m)
    assert y.space == D
    assert np.allclose(y.values, [v[0] * v[1], np.sum(v ** 2)], atol=1e-12)
    assert isinstance(G, pg.Map)
    assert G.claims.status("single-valued") == (pg.Truth.TRUE, THEOREM)
    assert G.claims.status("realisation") == (pg.Truth.TRUE, UNKNOWN)


def test_external_map_dom_failures_and_claims():
    M, D = H(1), pg.Euclidean(2)
    C = pg.port.Coordinates.nodal(M, X)
    dom = PredicateSubset(M, lambda m: bool(np.all(C.numbers(m) > 0)), name="u > 0 at the nodes")

    def picky(x):
        if x[0] > 5:
            raise RuntimeError("no first arrival")
        return code(x)

    G = pg.port.External(M, D, picky, coords=(C, None), dom=dom, name="T", code="tracer",
                    realises="travel times of the model whose nodal values are given", continuous="piecewise smooth")
    assert G.claims.status("realisation") == (pg.Truth.TRUE, ASSERTED)
    assert G.claims.status("total") == (pg.Truth.FALSE, THEOREM)                       # a partial map, by construction
    with pytest.raises(pg.Refused, match="not in dom"):
        G(M(lambda t: t - 0.5))
    with pytest.raises(pg.ExternalError, match="tracer failed inside its declared domain of T") as err:
        G(M(lambda t: 10 + t))
    assert isinstance(err.value.__cause__, RuntimeError)
    bad = pg.port.External(M, D, lambda x: x, coords=(C, None), name="B", code="shape")
    with pytest.raises(pg.ExternalError, match=r"shape \(4,\)"):
        bad(M(lambda t: t))


def test_external_map_needs_coordinates_of_function_spaces():
    with pytest.raises(ValueError, match="no default coordinates"):
        pg.port.External(H(1), pg.Euclidean(2), code)


# ------------------------------------------------------------------------------------------------- direct sums
NODES = X


def mixed_space():
    """H^1 of [0, 1] (+) a weighted R^3 (the shape of a velocity model with source terms), summand weights 2 and 1."""
    return pg.DirectSum({"v": H(1), "src": pg.Euclidean(3, weights=np.array([1.0, 4.0, 0.5]))}, weights={"v": 2.0})


def test_direct_sum_coordinates_round_trips_and_adjoints():
    X = mixed_space()
    C = pg.port.Coordinates.direct_sum(X, {"v": pg.port.Coordinates.nodal(X.summand("v"), NODES), "src":
                                      pg.port.Coordinates.identity(X.summand("src"))})
    x = X({"v": H(1)(lambda t: np.cos(2 * t)), "src": X.summand("src")(np.array([0.5, -1.0, 2.0]))})
    assert np.allclose(C.numbers(x), np.r_[np.cos(2 * NODES), [0.5, -1.0, 2.0]], atol=1e-12)
    c = np.arange(1.0, 8.0)
    assert np.allclose(C.numbers(C.element(c)), c, atol=1e-10)                         # E P = I
    R = C.export.codomain
    assert adjoint_gap(C.export, x, R(c)) < 1e-12                                       # in X's weighted inner product
    assert adjoint_gap(C.imp, R(c), x) < 1e-10
    assert C.claims.status("E P = I") == (pg.Truth.TRUE, DERIVED) and C.claims.status("bounded") == (pg.Truth.TRUE, DERIVED)
    only_v = pg.port.Coordinates.direct_sum(X, {"v": pg.port.Coordinates.nodal(X.summand("v"), NODES)})
    assert only_v.dim == 4 and X.summand("src").norm(X.part(only_v.element(c[:4]), "src")) == 0.0   # imported as 0


def test_drivers_on_a_direct_sum():
    X = mixed_space()
    C = pg.port.Coordinates.direct_sum(X, {"v": pg.port.Coordinates.nodal(X.summand("v"), NODES), "src":
                                      pg.port.Coordinates.identity(X.summand("src"))})
    A = np.random.default_rng(5).standard_normal((5, 7))
    G = pg.port.External(X, pg.Euclidean(5), lambda z: A @ np.tanh(z), coords=(C, None),
                    jacobian=lambda z: A * (1 - np.tanh(z) ** 2)[None, :], name="G")
    theta = C.element(np.linspace(-0.5, 0.5, 7))
    assert min(pg.port.taylor_test(G, theta, directions=3).rates) > 1.9
    fit = pg.optimisers.gauss_newton(G, pg.Euclidean(5)(np.array([0.1, -0.2, 0.3, 0.0, 0.2])), X.zero, alpha=0.1)
    assert fit.converged


def test_subset_from_a_predicate_on_numbers():
    """A dom given by external code on the numbers (plan 020 P5b: 'a first arrival exists')."""
    M = H(1)
    C = pg.port.Coordinates.nodal(M, NODES)
    S = C.subset(lambda x: bool(np.all(x > 0)), name="positive at the nodes")
    assert S.member(M(lambda t: 1 + t)) is pg.Truth.TRUE
    assert S.member(M(lambda t: t - 0.5)) is pg.Truth.FALSE
