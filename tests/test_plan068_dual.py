"""Plan 068 P2: the dual (spaces/dual.py). One Functional, an element of X', with from_representer, from_measure and
from_operator, absorbing operators/functionals.LinearForm and MeasureForm and port/derivatives.Covector; each space's
boundedness rule and the one trace rule there (absorbing spaces/boundedness.py and SobolevSpace.bounded); the Frechet
gradient there (optimisers/ no longer import port/); the dual's algebra (its zero, linear combinations) and L^p's
functionals."""
import ast
import importlib
import math
import os

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import PointEvaluationUndefined, Refused, SpaceMismatch, Unsupported
from pygeoinf0.core.logic import Truth
from pygeoinf0.spaces import dual
from pygeoinf0.spaces.dual import DualSpace, Functional

I = pg.Interval("[0, 1]")
J = pg.Interval("[0.2, 0.6]")
PKG = os.path.dirname(pg.__file__)


# ------------------------------------------------------------------ the clean break
@pytest.mark.parametrize("path", ["pygeoinf0.spaces.boundedness", "pygeoinf0.port.derivatives"])
def test_absorbed_modules_are_gone(path):
    with pytest.raises(ImportError):
        importlib.import_module(path)


@pytest.mark.parametrize("name", ["LinearForm", "MeasureForm", "Covector"])
def test_absorbed_names_are_gone(name):
    assert not hasattr(pg, name)
    assert not hasattr(pg.operators, name) and not hasattr(pg.port, name)
    assert not hasattr(importlib.import_module("pygeoinf0.operators.functionals"), name)


def test_the_new_names():
    assert pg.Functional is Functional and pg.DualSpace is DualSpace and pg.gradient is dual.gradient
    for name in ("bounded", "require_bounded", "l2_density", "Decision", "trace_excess"):
        assert callable(getattr(dual, name)), name
    assert not hasattr(pg.Sobolev(I, 1, length_scale=0.2), "bounded")      # the trace rule lives once, in the dual


def test_optimisers_do_not_import_port():
    for root, _, files in os.walk(os.path.join(PKG, "optimisers")):
        for f in files:
            if f.endswith(".py"):
                for node in ast.walk(ast.parse(open(os.path.join(root, f)).read())):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        assert "port" not in node.module.split("."), (f, node.module)


# ------------------------------------------------------------------ one Functional, three constructors
def test_from_representer_on_a_hilbert_space():
    M = pg.L2(I)
    r = M(lambda t: t)
    l = Functional.from_representer(r)
    u = M(lambda t: 1.0 + t)
    assert l.space == M and l.representer is r and l in M.dual()
    assert abs(l(u) - M.inner(u, r)) < 1e-14 and abs(l.norm() - M.norm(r)) < 1e-14
    assert l.gradient() is r
    with pytest.raises(TypeError):
        Functional.from_representer(pg.Continuous("[0, 1]")(lambda t: t))      # C has no inner product


def test_from_measure_on_every_kind_of_space():
    mu = pg.LebesgueMeasure(J)
    exact = (0.6 ** 2 - 0.2 ** 2) / 2                                       # int_J t dt
    for X in (pg.L2(I), pg.Sobolev(I, 1, length_scale=0.2), pg.Continuous("[0, 1]"), pg.Lp(I, 3), pg.Lp(I, 1),
              pg.Lp(I, np.inf)):
        l = Functional.from_measure(X, mu)
        assert isinstance(l, Functional) and l.space == X and l.measure is mu and X.dual().member(l) is Truth.TRUE
        assert abs(l(X(lambda t: t)) - exact) < 1e-10, X
    assert abs(Functional.from_measure(pg.Continuous("[0, 1]"), mu, scale=-2.0).norm() - 0.8) < 1e-14


def test_lp_has_its_functionals():
    """u -> int u dnu on L^p(mu) is bounded iff d nu / d mu is in L^p' (1/p + 1/p' = 1), with norm ||d nu / d mu||_p'."""
    for p in (1.0, 1.5, 3.0, np.inf):
        X = pg.Lp(I, p)
        l = X.functional(pg.LebesgueMeasure(J))
        q = np.inf if p == 1 else 1.0 if np.isinf(p) else p / (p - 1)
        expect = 1.0 if np.isinf(q) else 0.4 ** (1.0 / q)                  # ||1_J||_q
        assert isinstance(l, Functional) and abs(l.norm() - expect) < 1e-12, (p, l.norm(), expect)
        u = X(lambda t: t ** 2)
        assert abs(l(u)) <= l.norm() * X.norm(u) * (1 + 1e-9)
        with pytest.raises(PointEvaluationUndefined):
            X.functional(0.3)                                                  # a point has measure zero
    with pytest.raises(PointEvaluationUndefined):
        pg.Lp(I, 3).evaluation(0.3)


def test_from_operator_absorbs_the_covector():
    R1 = pg.Euclidean(1)
    X = pg.Euclidean(3, weights=[1.0, 2.0, 3.0])
    g = np.array([1.0, -1.0, 2.0])
    A = pg.Matrix(g[None, :], domain=X, codomain=R1)
    l = Functional.from_operator(A)
    x = X(np.array([0.5, 1.0, -2.0]))
    assert l.space == X and l.operator is A and abs(l(x) - g @ x.values) < 1e-14
    r = l.gradient()                                                      # <r, h>_X = l(h): r = W^-1 g
    assert np.allclose(r.values, g / np.array([1.0, 2.0, 3.0])) and abs(X.inner(r, x) - l(x)) < 1e-14
    assert abs(l.norm() - X.norm(r)) < 1e-14
    W = pg.Euclidean(1, weights=[4.0])                                    # a weighted R^1: the number is the coordinate
    l4 = Functional.from_operator(pg.Matrix(g[None, :], domain=X, codomain=W))
    assert abs(l4(x) - g @ x.values) < 1e-14 and np.allclose(l4.gradient().values, r.values)
    with pytest.raises(SpaceMismatch):
        Functional.from_operator(pg.Matrix(np.eye(3), domain=X, codomain=pg.Euclidean(3)))


def test_a_gradient_needs_an_inner_product():
    Lp = pg.Lp(pg.LebesgueMeasure(I), 3)
    R1 = pg.Euclidean(1)
    l = Functional.from_operator(pg.LinearOperator(Lp, R1, lambda f: R1(np.array([f.integral()]))))
    with pytest.raises(Refused, match="no inner product"):
        l.gradient()
    with pytest.raises(Refused, match="no inner product"):
        Lp.functional(pg.LebesgueMeasure(J)).gradient()


def test_coordinates_give_the_functional_of_a_tools_gradient():
    M = pg.Sobolev(I, 1, length_scale=0.2)
    X = np.linspace(0.1, 0.9, 5)
    C = pg.port.Coordinates.nodal(M, X)
    g = np.array([1.0, 0.0, -2.0, 0.5, 0.0])
    l = C.functional(g)
    h = M(lambda t: np.exp(t))
    assert isinstance(l, Functional) and l.space == M and abs(l(h) - g @ np.exp(X)) < 1e-10
    assert abs(M.inner(l.gradient(), h) - l(h)) < 1e-10 * abs(l(h))


# ------------------------------------------------------------------ the dual's algebra
def test_the_dual_has_its_zero_and_linear_combinations():
    for X, x in ((pg.L2(I), None), (pg.Continuous("[0, 1]"), None), (pg.Lp(I, 3), None),
                 (pg.Euclidean(2), np.array([1.0, -2.0]))):
        D = X.dual()
        u = X(x) if x is not None else X(lambda t: np.cos(t))
        if isinstance(X, pg.Euclidean):
            a, b = X.functional(np.array([1.0, 0.0])), X.functional(np.array([0.5, 3.0]))
        else:
            a, b = X.functional(pg.LebesgueMeasure(J)), X.functional(pg.LebesgueMeasure(pg.Interval("[0.5, 1]")))
        z = D.zero
        assert isinstance(z, Functional) and D.member(z) is Truth.TRUE and z(u) == 0.0 and z.norm() == 0.0
        c = D.linear_combination([2.0, -0.5], [a, b])
        assert isinstance(c, Functional) and D.member(c) is Truth.TRUE
        assert abs(c(u) - (2.0 * a(u) - 0.5 * b(u))) < 1e-12
        assert abs((2.0 * a - 0.5 * b)(u) - c(u)) < 1e-12 and abs((a + b)(u) - (a(u) + b(u))) < 1e-12
        assert abs((-a)(u) + a(u)) < 1e-14
    M = pg.L2(I)
    a = M.functional(pg.LebesgueMeasure(J))
    with pytest.raises(SpaceMismatch):
        M.dual().linear_combination([1.0, 1.0], [a, pg.L2("[0, 2]").functional(pg.LebesgueMeasure(J))])


def test_a_combination_of_representers_has_the_exact_norm():
    M = pg.L2(I)
    a, b = M.functional(pg.LebesgueMeasure(J)), M.functional(pg.LebesgueMeasure(pg.Interval("[0.4, 1]")))
    c = a - b                                       # 1_[0.2, 0.4) - 1_(0.6, 1]: norm sqrt(0.2 + 0.4)
    assert abs(c.norm() - math.sqrt(0.6)) < 1e-12 and c.representer is not None


def test_the_complex_dual_is_conjugate_linear_in_the_representer():
    E = pg.Euclidean(2, field="C")
    a = E.functional(np.array([1.0 + 1j, 2.0]))
    x = E(np.array([0.5, 1j]))
    c = E.dual().linear_combination([1j], [a])
    assert abs(c(x) - 1j * a(x)) < 1e-14


# ------------------------------------------------------------------ every functional the package builds
def test_built_functionals_are_functionals_of_the_dual():
    H = pg.Sobolev(I, 1, length_scale=0.2)
    C = pg.Continuous("[0, 1]")
    M = pg.L2(I)
    G = pg.Functionals([M(lambda t: t), M(lambda t: t ** 2)])
    built = [(H, H.evaluation(0.3)), (H, H.functional(0.3)), (H, H.average(J)), (M, M.average(J)),
             (M, M.functional(pg.LebesgueMeasure(J))), (C, C.evaluation(0.3)), (C, C.average(J)),
             (C, C.integrals([0.2, 0.5])[1]), (M, G[0]), (pg.Lp(I, 3), pg.Lp(I, 3).functional(pg.LebesgueMeasure(J)))]
    for X, l in built:
        assert isinstance(l, Functional) and l.space == X and X.dual().member(l) is Truth.TRUE, (X, l)
    assert G[0].representer.space == M


def test_functionals_take_functionals():
    M = pg.L2(I)
    a, b = M.functional(pg.LebesgueMeasure(J)), M.functional(pg.LebesgueMeasure(pg.Interval("[0.5, 1]")))
    G = pg.Functionals([a, b])
    u = M(lambda t: t)
    assert np.allclose(G(u).values, [a(u), b(u)])


# ------------------------------------------------------------------ the boundedness rules and the one trace rule
def test_the_trace_rule_is_one_function():
    assert dual.trace_excess(1.0, 1, 0) == 0.5 and dual.trace_excess(1.0, 3, 1) == 0.0
    for s, d in ((0.5, 1), (1.0, 1), (1.0, 2), (2.0, 3), (1.4, 3)):
        H = pg.Sobolev(pg.Interval("[0, 1]") if d == 1 else pg.Box([(0, 1)] * d), s, length_scale=0.3) \
            if d < 3 else None
        for k in range(d + 1):
            expect = s > (d - k) / 2
            assert (dual.trace_excess(s, d, k) > 0) == expect
            if H is not None:
                assert bool(H.dual().integrals_bounded(k)) == expect, (s, d, k)


def test_integrals_bounded_on_each_kind_of_space():
    assert H1().dual().integrals_bounded(0) is Truth.TRUE
    assert pg.Sobolev(I, 0.5, length_scale=0.2).dual().integrals_bounded(0) is Truth.FALSE
    assert pg.L2(I).dual().integrals_bounded(0) is Truth.FALSE and pg.L2(I).dual().integrals_bounded(1) is Truth.TRUE
    assert pg.Continuous("[0, 1]").dual().integrals_bounded(0) is Truth.TRUE
    assert pg.Euclidean(2).dual().integrals_bounded(0) is Truth.UNKNOWN


def H1():
    return pg.Sobolev(I, 1, length_scale=0.2)


def test_the_dual_decides_boundedness():
    H, M = H1(), pg.L2(I)
    assert H.dual().bounded(pg.DiracMeasure(0.3)).bounded is Truth.TRUE
    assert M.dual().bounded(pg.DiracMeasure(0.3)).bounded is Truth.FALSE
    assert dual.bounded(pg.DiracMeasure(0.3), M).bounded is Truth.FALSE
    with pytest.raises(PointEvaluationUndefined):
        dual.require_bounded(pg.DiracMeasure(0.3), M)


def test_no_space_is_duck_typed():
    """The rule recognises a space by its class, not by its attributes (the review's B4: _kind duck-typed spaces)."""
    class LooksSobolev:
        construction, s, domain = None, 1.0, I
    assert dual.bounded(pg.DiracMeasure(0.3), LooksSobolev()).bounded is Truth.UNKNOWN


def test_from_measure_on_a_direct_sum_uses_its_riesz():
    X = pg.DirectSum({"a": pg.L2(I), "b": H1()})
    l = Functional.from_measure(X, {"b": pg.DiracMeasure(0.3)})
    x = X.element({"a": lambda t: t, "b": lambda t: t ** 2})
    assert X.dual().member(l) is Truth.TRUE and abs(l(x) - 0.09) < 1e-12 and l.measure is not None
    with pytest.raises(Unsupported):
        Functional.from_measure(pg.Euclidean(2), pg.DiracMeasure(0.3))   # R^n describes its functionals by numbers
