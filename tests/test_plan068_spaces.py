"""Plan 068 P1: the space contracts per level and the spaces out of core into pygeoinf0.spaces (layer L3).

- Each level's operations exist exactly at that level (VectorSpace: field, zero, linear combinations; NormedSpace: norm,
  dual(), functional(); HilbertSpace: inner, riesz). L^p for p != 2 and C have no inner or riesz attribute at all.
- Euclidean(n, field="R" | "C") is the one constructor: C^n is Euclidean(n, field="C") (ComplexEuclidean is gone), a
  CoordinateSpace like R^n (not an instance of Euclidean, so the routes for real coordinates refuse it as before), its
  metric applied by the one metric code of R^n (spaces/metrics.py), a covariance solved with, never inverted.
- The space modules live in pygeoinf0.spaces; their old paths in pygeoinf0.core are gone (a clean break).
- Operations the mathematics does not define raise NotDefined (a Refused), not Unsupported (not built).
"""
import importlib
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import NotDefined, Refused, Unsupported
from pygeoinf0.core.logic import Truth
from pygeoinf0.spaces import BanachSpace, CoordinateSpace, DualSpace, Euclidean, HilbertSpace, NormedSpace, VectorSpace

I = pg.Interval("[0, 1]")
MOVED = ("spaces", "elements", "sobolev", "constructions", "direct_sum", "layered", "continuous", "complex",
         "boundedness", "representers", "membership", "metrics", "labels")
NEW = {"spaces": ("base", "dual")}


# ------------------------------------------------------------------ the move (a clean break)
@pytest.mark.parametrize("name", MOVED)
def test_old_core_paths_are_gone(name):
    with pytest.raises(ImportError):
        importlib.import_module(f"pygeoinf0.core.{name}")


# spaces/layered.py was removed by plan 069 P7 (pg.PiecewiseSpace in spaces/piecewise.py replaces Layered)
@pytest.mark.parametrize("name", [m for m in MOVED if m not in ("spaces", "boundedness", "layered")] + list(NEW["spaces"]))
def test_new_paths_import(name):
    assert importlib.import_module(f"pygeoinf0.spaces.{name}").__name__ == f"pygeoinf0.spaces.{name}"


def test_core_keeps_its_foundations_and_the_numerics_of_plan_069():
    import pygeoinf0.core as core
    import os
    left = sorted(f[:-3] for f in os.listdir(os.path.dirname(core.__file__)) if f.endswith(".py") and f != "__init__.py")
    assert left == sorted(["adaptive", "claims", "errors", "estimable", "fits", "kinds", "lazy", "logic", "numbers",
                           "precision", "relation", "sampling", "tabulated"]), left


def test_pg_spaces_is_the_package_and_the_catalogue_is_pg_catalogue():
    import pygeoinf0.spaces as S
    assert pg.spaces is S and pg.spaces.Euclidean is pg.Euclidean
    assert callable(pg.catalogue.how) and callable(pg.catalogue.spaces.how)
    assert not hasattr(pg, "ComplexEuclidean")


# ------------------------------------------------------------------ level contracts
def _two(X):
    if isinstance(X, CoordinateSpace):
        dt = complex if X.field == "C" else float
        return X(np.arange(1, X.dim + 1, dtype=dt)), X(np.ones(X.dim, dtype=dt))
    return X(lambda t: t), X(lambda t: 1.0 - t)


@pytest.mark.parametrize("X", [pg.Euclidean(3), pg.Euclidean(2, field="C"), pg.L2(I), pg.Lp(I, 3), pg.Lp(I, 1),
                               pg.Continuous("[0, 1]"), pg.Sobolev(I, 1, length_scale=0.2)], ids=repr)
def test_vector_space_contract(X):
    assert isinstance(X, VectorSpace) and X.field in ("R", "C")
    x, y = _two(X)
    assert X.member(X.zero) is Truth.TRUE
    z = X.linear_combination([3.0, -1.0], [x, y])
    assert X.norm(z - (3.0 * x - y)) <= 1e-12 * (1 + X.norm(z))
    with pytest.raises(ValueError):
        X.linear_combination([1.0], [x, y])
    with pytest.raises(pg.SpaceMismatch):
        X.linear_combination([1.0], [pg.Euclidean(7).zero])


@pytest.mark.parametrize("X", [pg.Lp(I, 3), pg.Lp(I, 1), pg.Lp(I, np.inf), pg.Continuous("[0, 1]")], ids=repr)
def test_banach_spaces_have_no_inner_product_or_riesz_map(X):
    assert isinstance(X, BanachSpace) and not isinstance(X, HilbertSpace)
    assert not hasattr(X, "inner") and not hasattr(X, "riesz")
    assert not hasattr(type(X), "inner") and not hasattr(type(X), "riesz")
    assert callable(X.norm) and callable(X.dual) and callable(X.functional)


@pytest.mark.parametrize("X", [pg.Euclidean(3), pg.Euclidean(2, field="C"), pg.L2(I), pg.Sobolev(I, 1, length_scale=0.2),
                               pg.DirectSum({"a": pg.Euclidean(2), "b": pg.L2(I)}), pg.Complexified(pg.L2(I))], ids=repr)
def test_hilbert_spaces_have_inner_and_riesz(X):
    assert isinstance(X, HilbertSpace) and callable(X.inner) and callable(X.riesz)


def test_the_abstract_levels():
    assert issubclass(NormedSpace, VectorSpace) and issubclass(BanachSpace, NormedSpace)
    assert issubclass(HilbertSpace, BanachSpace)
    assert not hasattr(NormedSpace, "inner") and not hasattr(BanachSpace, "riesz")
    assert hasattr(HilbertSpace, "inner") and hasattr(HilbertSpace, "riesz")
    assert hasattr(VectorSpace, "linear_combination") and not hasattr(VectorSpace, "norm")
    assert hasattr(NormedSpace, "dual") and hasattr(NormedSpace, "functional")


# ------------------------------------------------------------------ duals and functionals
def test_the_dual_space():
    X = pg.L2(I)
    D = X.dual()
    assert isinstance(D, DualSpace) and isinstance(D, BanachSpace) and D.primal == X and D == pg.L2(I).dual()
    assert D.field == "R" and "complete" in str(D.claims).lower()
    assert repr(D) == f"({X!r})'"
    assert "L^1.5" in str(pg.Lp(I, 3).dual().claims)
    assert "measures" in str(pg.Continuous("[0, 1]").dual().claims)
    assert pg.Lp(I, np.inf).dual() != pg.Lp(I, 1).dual()


def test_functionals_are_elements_of_the_dual():
    X = pg.L2(I)
    phi = X.functional(pg.LebesgueMeasure("[0, 0.5]"))
    f = X(lambda t: t)
    assert abs(phi(f) - 0.125) < 1e-12
    assert X.dual().member(phi) is Truth.TRUE and pg.Lp(I, 3).dual().member(phi) is Truth.FALSE
    assert abs(X.dual().norm(phi) - math.sqrt(0.5)) < 1e-12
    C = pg.Continuous("[0, 1]")
    ev = C.functional(0.3)
    assert abs(ev(C(lambda t: t ** 2)) - 0.09) < 1e-12 and C.dual().member(ev) is Truth.TRUE
    lp = pg.Lp(I, 3).functional(pg.LebesgueMeasure("[0, 0.5]"))       # plan 068 P2: L^p has its functionals
    assert pg.Lp(I, 3).dual().member(lp) is Truth.TRUE and abs(lp(pg.Lp(I, 3)(lambda t: t)) - 0.125) < 1e-12
    with pytest.raises(pg.PointEvaluationUndefined):  # point evaluation is not bounded on L2
        X.functional(0.3)


def test_euclidean_riesz_and_functional():
    E = pg.Euclidean(3, weights=[1.0, 2.0, 4.0])
    c = np.array([1.0, -2.0, 0.5])
    r = E.riesz(c)
    x = E(np.array([0.3, 1.0, -2.0]))
    assert abs(E.inner(x, r) - c @ x.values) < 1e-14
    assert abs(E.functional(c)(x) - c @ x.values) < 1e-14
    R = E.riesz(np.eye(3))
    assert np.allclose(np.ravel(E.inner(R, x)), x.values)
    Cn = pg.Euclidean(2, field="C", metric=np.array([[2.0, 0.5j], [-0.5j, 1.0]]))
    z = Cn(np.array([1.0 + 2j, -1j]))
    a = np.array([0.5 - 1j, 2.0])
    assert abs(Cn.cinner(z, Cn.riesz(a)) - a @ z.values) < 1e-14


# ------------------------------------------------------------------ Euclidean(n, field="C")
def test_complex_euclidean_is_euclidean_with_field_c():
    C3 = pg.Euclidean(3, field="C")
    assert isinstance(C3, CoordinateSpace) and C3.field == "C" and C3.dtype is complex and C3.real_dim == 6
    assert not isinstance(C3, Euclidean)               # the routes for real coordinates refuse C^n, as before
    assert repr(C3) == "C^3" and C3 == pg.Euclidean(3, field="C") and C3 != pg.Euclidean(3)
    assert pg.Euclidean(3).field == "R" and not hasattr(pg.Euclidean(3), "cinner") and not hasattr(pg.Euclidean(3), "J")
    with pytest.raises(ValueError):
        pg.Euclidean(3, field="Q")
    one, i = pg.Euclidean(1, field="C")(np.array([1.0])), pg.Euclidean(1, field="C")(np.array([1j]))
    assert np.isclose(pg.Euclidean(1, field="C").cinner(one, i), -1j)
    with pytest.raises(TypeError):
        pg.Euclidean(2)(np.array([1j, 0.0]))


def test_complex_metric_and_covariance_are_one_space():
    W = np.array([[2.0, 0.5 + 0.5j], [0.5 - 0.5j, 1.0]])
    A, B = pg.Euclidean(2, field="C", metric=W), pg.Euclidean(2, field="C", covariance=np.linalg.inv(W))
    x, y = np.array([1.0 + 1j, -2.0]), np.array([0.5j, 3.0])
    assert np.isclose(A.cinner(A(x), A(y)), y.conj() @ W @ x) and np.isclose(B.cinner(B(x), B(y)), y.conj() @ W @ x)
    assert np.allclose(B.solve_metric(x), np.linalg.inv(W) @ x)
    D = pg.Euclidean(3, field="C", covariance=np.array([1.0, 4.0, 0.5]))
    assert np.allclose(D.weights, [1.0, 0.25, 2.0])
    S = np.array([[1.0, 0.2j], [-0.2j, 2.0]])
    E = pg.Euclidean(2, field="C", covariance=S)
    assert np.allclose(E.standard_gaussian().coordinate_covariance, S)
    with pytest.raises(ValueError):
        pg.Euclidean(2, field="C", metric=np.array([[1.0, 1j], [1j, 1.0]]))       # not Hermitian
    with pytest.raises(Unsupported):
        pg.Euclidean(2, field="C", metric=pg.Identity(pg.Euclidean(2)))            # operators on C^n: not built


def test_complex_matrix_and_real_coordinates():
    A = np.array([[1.0 + 1j, 2.0], [0.0, -1j], [3.0, 1.0]])
    G = pg.ComplexMatrix(A)
    assert G.domain == pg.Euclidean(2, field="C") and G.codomain == pg.Euclidean(3, field="C")
    X = pg.Euclidean(2, field="C")
    z = X(np.array([1.0 - 1j, 2j]))
    assert np.allclose(X.from_real(X.to_real(z.values)), z.values) and len(X.real_basis()) == 4
    assert np.allclose(X.J(z).values, 1j * z.values)


# ------------------------------------------------------------------ NotDefined: the mathematics has no such object
def test_draws_of_a_cylindrical_gaussian_are_not_defined():
    X = pg.DirectSum({"a": pg.Euclidean(2), "b": pg.L2(I)})
    mu = pg.probability.cylindrical(X)
    with pytest.raises(NotDefined):
        mu.sample(seed=1)
    assert issubclass(NotDefined, Refused) and not issubclass(NotDefined, Unsupported)


def test_projection_onto_a_dense_domain_is_not_defined():
    Ar = pg.SobolevForm("[0, 1]", 1, length_scale=0.1).operator()             # a relation (plan 068 P3)
    with pytest.raises(NotDefined):
        Ar.dom.project(Ar.domain.zero)


def test_an_unbounded_polytope_has_no_vertex_list():
    from pygeoinf0.convex.polytope import vertices
    with pytest.raises(NotDefined):
        vertices([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    with pytest.raises(Unsupported):                       # not built (n > 3), a meaningful object
        vertices(np.eye(4), np.ones(4))


def test_the_taylor_check_on_a_direct_sum_takes_other_directions():
    from pygeoinf0.port.taylor import _random_in
    X = pg.DirectSum({"a": pg.Euclidean(2), "b": pg.L2(I)})
    h = X.element({"a": np.array([0.5, -1.0]), "b": lambda t: t})
    assert _random_in(object(), X, 0, lambda u: u, h) is h       # X.random has no draws (NotDefined): y = DG h
