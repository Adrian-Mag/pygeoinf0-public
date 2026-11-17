"""Plan 068 P6 (review B10): finite subspaces and their two roles.

The set role: one finite subspace, pg.Span (convex/subspaces.py), a ClosedSubspace of finite dimension with its
generators, Gram matrix, coefficient space (the Gram as metric), synthesis and analysis maps and orthogonal projection;
a grid taken as the model space is a Span used as a set. The approximation role: pg.Discretisation
(operators/galerkin.py), built on a Span, with its claim of what it approximates and how well, stated on the function
space and never stronger than what was checked. port/ keeps coordinates, fields, plugins and Taylor checks.
"""
import importlib
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN
from pygeoinf0.core.errors import Refused, SpaceMismatch
from pygeoinf0.core.logic import Truth

warnings.simplefilter("ignore", pg.RealisationWarning)
I = pg.Interval("[0, 1]")
B = pg.Box((0, 1), (0, 1))


# ------------------------------------------------------------------ moved paths and refusals
def test_discretisation_left_the_port():
    with pytest.raises(ImportError):
        importlib.import_module("pygeoinf0.port.discretisation")
    from pygeoinf0.operators.galerkin import Analysis, Discretisation, Synthesis  # noqa: F401
    assert pg.Discretisation is Discretisation
    assert not hasattr(pg.port, "Discretisation") and "Discretisation" not in pg.port.__all__


def test_the_old_signature_is_refused_with_its_replacement():
    M = pg.L2(I)
    with pytest.raises(TypeError, match=r"pg\.Span\(basis, space=M\)"):
        pg.Discretisation(M, F.bases.hats(I, 5))
    with pytest.raises(TypeError, match="finite subspace"):
        pg.Discretisation(M)


# ------------------------------------------------------------------ the set role: Span
def test_a_span_of_bank_functions_and_callables():
    M = pg.L2(I)
    V = pg.Span(F.bases.legendre(I, 3), space=M)
    assert V.dim == 4 and len(V.vectors) == 4 and np.allclose(V.gram, np.eye(4), atol=1e-13)
    W = pg.Span([lambda x: 1.0 + 0 * x, lambda x: x], space=M)
    assert W.dim == 2 and W.member(M(lambda x: 3.0 - 2.0 * x)) is Truth.TRUE
    assert W.member(M(lambda x: x ** 2)) is Truth.FALSE
    with pytest.raises(TypeError, match="bank function or a callable"):
        pg.Span(["not a function"], space=M)
    with pytest.raises(SpaceMismatch):
        pg.Span([pg.L2(B)(lambda x, y: x)], space=M)


def test_the_coefficient_space_carries_the_gram():
    M = pg.L2(I)
    V = pg.Span(F.bases.hats(I, 9), space=M)
    C = V.coefficient_space
    c = C(np.random.default_rng(0).standard_normal(len(V.vectors)))
    f = M(lambda x: np.exp(x))
    S = V.synthesis
    assert C.norm(c) == pytest.approx(S(c).norm(), rel=1e-12)                    # coefficient norms are M's norms
    assert M.inner(S(c), f) == pytest.approx(C.inner(c, S.adjoint(f)), rel=1e-12)
    assert V.analysis.domain == M and V.analysis.codomain == C
    cf = V.coefficients(f)
    assert cf.shape == (len(V.vectors),) and np.allclose(cf, S.adjoint(f).values, rtol=1e-12)
    assert V.coefficients(M.family([f, f * 2.0])).shape == (len(V.vectors), 2)
    assert (V.element(c.values) - S(c)).norm() <= 1e-14
    assert np.allclose(V.loads(f), M.inner(V.vectors, f))
    assert V.claims.find("rank").status == CHECKED
    assert V.claims.find("projection").status == DERIVED


def test_many_local_functions_span_with_a_sparse_gram():
    """6400 hats: the Gram is scipy.sparse (realisations/series/local.py), which the Span of plan 067 could not take (its
    dimension was an eigendecomposition of a dense array)."""
    M = pg.L2(B, realisation="local")
    V = pg.Span(F.bases.hats(B, 82), space=M)
    assert sp.issparse(V.gram) and V.dim == 6400 and V.claims.find("rank").status == CHECKED
    f = M(lambda x, y: np.sin(np.pi * x) * y)
    Pf = V.project(f)
    assert (V.project(Pf) - Pf).norm() <= 1e-10 * Pf.norm()
    assert V.member(Pf) is Truth.TRUE and V.member(f) is Truth.FALSE


def test_dependent_generators_span_but_have_no_coefficient_space():
    M = pg.L2(B)
    a, b = M(lambda x, y: x), M(lambda x, y: y)
    V = pg.Span([a, b, a + 2.0 * b])
    assert V.dim == 2 and V.member(a - b) is Truth.TRUE
    assert (V.project(a + b) - (a + b)).norm() <= 1e-12                         # minimum-norm coefficients
    with pytest.raises(Refused, match="rank 2"):
        V.coefficient_space
    with pytest.raises(Refused, match="rank 2"):
        V.synthesis
    with pytest.raises(Refused, match="rank 2"):
        pg.Discretisation(V)


def test_independence_is_decided_without_the_generators_scales():
    """span{e1, 1e-9 e2}: independent (plan 054: the dimension is scale-free); the Gram's condition number 1e18 refused
    a Discretisation of the same functions before (Cholesky of G itself); the factor is now of the correlation matrix."""
    X = pg.Euclidean(3)
    V = pg.Span([X(np.array([1.0, 0.0, 0.0])), X(np.array([0.0, 1e-9, 0.0]))])
    assert V.dim == 2 and V.claims.find("rank").status == CHECKED
    assert np.allclose(V.coefficients(X(np.array([2.0, 3.0, 4.0]))), [2.0, 3e9], rtol=1e-12)
    assert V.coefficient_space.dim == 2


def test_a_cancelled_generator_is_not_called_independent():
    """scale= marks generators at the rounding level of a cancellation (convex/subspaces.image): they are dropped, never
    normalised into an independent direction by the correlation factor."""
    X = pg.Euclidean(3)
    V = pg.Span([X(np.array([1.0, 0.0, 0.0])), X(np.array([0.0, 1e-17, 0.0]))], scale=1.0)
    assert V.dim == 1 and V.zero_generators == 1
    with pytest.raises(Refused, match="rank 1"):
        V.coefficient_space


# ------------------------------------------------------------------ the approximation role: Discretisation
def test_the_discretisation_states_its_role_and_never_more():
    M = pg.L2(I)
    D = pg.Discretisation(pg.Span(F.bases.hats(I, 9), space=M))
    assert D.space == M and len(D) == 7 and D.subspace.dim == 7
    u = D.claims.find("uniform approximation")
    assert u.status == THEOREM and u.holds is False                 # a unit element of M orthogonal to the span exists
    assert D.claims.find("approximation").status == UNKNOWN
    f = M(lambda x: np.sin(np.pi * x))
    (e,) = D.check([f])
    assert e.status == CHECKED and e.upper >= D.residual(f) > 0
    assert D.claims.find("approximation").status == UNKNOWN          # a check of one element says nothing of a class
    chk = D.claims.find("checked approximation")
    assert chk.status == CHECKED and "1 element" in chk.statement
    D.state(0.05, "the functions with ||f''||_L2 <= 1 that vanish at 0 and 1", how="a cited interpolation bound")
    assert D.claims.find("approximation").status == ASSERTED
    assert "||f - P f||" in D.claims.find("approximation").statement
    with pytest.raises(ValueError):
        D.state(-1.0, "anything")
    with pytest.raises(SpaceMismatch):
        D.state(0.1, pg.NormBall(pg.Euclidean(2), 1.0))
    D.state(0.5, pg.NormBall(M, 1.0, centre=M(lambda x: x)))           # a subset of M is a class too
    assert "B(1)" in D.claims.find("approximation").statement


def test_a_full_span_of_a_euclidean_space_approximates_it_uniformly():
    X = pg.Euclidean(2)
    D = pg.Discretisation(pg.Span([X(np.array([1.0, 1.0])), X(np.array([1.0, -1.0]))]))
    u = D.claims.find("uniform approximation")
    assert u.holds is True and u.status == THEOREM
    D1 = pg.Discretisation(pg.Span([X(np.array([1.0, 1.0]))]))
    assert D1.claims.find("uniform approximation").holds is False


def test_check_accounts_for_the_realisation_errors():
    M = pg.L2(B)
    D = pg.Discretisation(pg.Span(F.bases.legendre(B, 2), space=M))
    f = M(lambda x, y: np.exp(x) * np.cos(2 * y))
    (e,) = D.check(f)
    r = D.residual(f)
    assert e.lower <= r <= e.upper and e.upper - r >= f.error
    fam = M.family([f, f * 3.0])
    es = D.check(fam)
    assert len(es) == 2 and es[1].upper == pytest.approx(3 * es[0].upper, rel=1e-9)


def test_galerkin_operators_between_two_discretisations():
    M = pg.L2(I)
    D = pg.Discretisation(pg.Span(F.bases.legendre(I, 3), space=M))
    T = pg.Discretisation(pg.Span(F.bases.legendre(I, 2), space=M))
    K = D.galerkin(test=T)
    assert K.shape == (3, 4) and np.allclose(K, np.eye(3, 4), atol=1e-13)
    A = D.operator(test=T)
    assert A.domain == D.subspace.coefficient_space and A.codomain == T.subspace.coefficient_space
    assert D.operator().claims.find("galerkin").status == DERIVED


def test_a_grid_used_as_the_model_space_is_a_set():
    """The set role: the span of a pixel grid is a subset of L2 (a ClosedSubspace, a state of kind set)."""
    from pygeoinf0.convex.subspaces import ClosedSubspace
    from pygeoinf0.core.kinds import SET
    M = pg.L2(B, realisation="local")
    V = pg.Span(F.bases.pixels(B, 4), space=M)
    assert isinstance(V, ClosedSubspace) and V.kind == SET and V.dim == 16
    assert np.allclose(V.gram, np.eye(16) / 16, atol=1e-16)
    assert V.support(M(lambda x, y: x)) == float("inf")
