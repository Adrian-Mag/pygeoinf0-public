"""Law tests: each checks a mathematical property, not just an example output."""
import math
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg

TOL = 1e-12


@pytest.fixture(scope="module")
def M():
    return pg.L2(pg.LebesgueMeasure(pg.Interval("[0, 1]")))


@pytest.fixture(scope="module")
def G(M):
    ks = [np.exp(-((M.x - c) / 0.05) ** 2) for c in np.linspace(0.05, 0.95, 20)]
    return pg.Functionals(ks, name="G")


# ---------------------------------------------------------------- sets and measures

def test_interval_reading_and_membership():
    I = pg.Interval("[0, 1)")
    assert 0 in I and 0.5 in I and 1 not in I
    assert pg.Interval("(-inf, 2pi]").b == pytest.approx(2 * math.pi)
    assert list(pg.Interval(0, 1).contains(np.array([-1, 0, 1, 2])[:, None])) == [False, True, True, False]
    with pytest.raises(ValueError):
        pg.Interval("[0, inf]")
    with pytest.raises(ValueError):
        pg.Interval(1, 0)


def test_lebesgue_measure_ignores_endpoints():
    mu = pg.LebesgueMeasure("[0, 1]")
    assert mu(pg.Interval("[0.2, 0.5)")) == pytest.approx(0.3)
    assert mu(pg.FiniteSet([0.1, 0.3])) == 0.0
    assert pg.L2("[0, 1]") == pg.L2("(0, 1)")          # same space: the endpoints are a null set
    assert pg.L2("[0, 1]") != pg.L2("[0, 2]")


def test_default_measure_is_visible():
    shortcut, explicit = pg.L2("[0, 1]"), pg.L2(pg.LebesgueMeasure("[0, 1]"))
    assert shortcut == explicit and repr(shortcut) == "L2([0, 1], Lebesgue)"
    assert "default" in shortcut.claims[0].reason and explicit.claims[0].reason == "given by the user"
    assert repr(pg.Lp("[0, 1]", 1)) == "L1([0, 1], Lebesgue)"


# ---------------------------------------------------------------- elements are classes

def test_inner_products_against_closed_forms(M):
    f, g = M(np.sin), M(np.cos)
    assert f @ f == pytest.approx(0.5 - math.sin(2) / 4, abs=TOL)
    assert f @ g == pytest.approx(math.sin(1) ** 2 / 2, abs=TOL)
    assert M.x.norm() == pytest.approx(1 / math.sqrt(3), abs=TOL)


def test_equality_is_equality_almost_everywhere(M):
    f = M(np.sin)
    g = f.redefine_at([0.5, 0.25], [10.0, -3.0])
    assert f == g                                       # same element
    assert g.rep(0.5) == 10.0 and f.rep(0.5) == pytest.approx(math.sin(0.5))
    assert g(0.5) == pytest.approx(math.sin(0.5), abs=1e-14)   # canonical value depends only on the class


def test_canonical_value_at_a_jump_is_the_mean_of_the_limits(M):
    s = M(lambda x: (x > 0.4) * 1.0)
    assert s(0.4) == pytest.approx(0.5)
    assert s.integral() == pytest.approx(0.6, abs=1e-14)
    t = M(lambda x: (x > 1 / math.pi) * 1.0)            # jump at an irrational point, located automatically
    assert t.integral() == pytest.approx(1 - 1 / math.pi, abs=1e-13)


def test_vector_space_laws(M):
    f, g, h = M(np.sin), M(np.exp), np.cos(5 * M.x)
    assert (f + g) + h == f + (g + h)
    assert f + g == g + f
    assert 2.5 * (f + g) == 2.5 * f + 2.5 * g
    assert f - f == M.zero
    assert sum([f, g, h]) == f + g + h


def test_ufuncs_and_coordinate(M):
    assert np.sin(M.x) == M(np.sin)
    assert np.exp(-M.x ** 2) == M(lambda x: np.exp(-x ** 2))
    taylor = sum(M.x ** k * (1 / math.factorial(k)) for k in range(20))
    assert taylor == M(np.exp)


def test_cauchy_schwarz_and_parallelogram(M):
    F = pg.probability.legendre_decay(M).sample(size=6, seed=3)
    for i in range(5):
        f, g = F[i], F[i + 1]
        assert abs(f @ g) <= f.norm() * g.norm() + 1e-14
        assert (f + g).norm() ** 2 + (f - g).norm() ** 2 == pytest.approx(2 * f.norm() ** 2 + 2 * g.norm() ** 2)


def test_lp_spaces():
    L1 = pg.Lp("[0, 1]", 1)
    assert isinstance(pg.Lp("[0, 1]", 2), pg.HilbertSpace)
    assert isinstance(L1, pg.BanachSpace) and not isinstance(L1, pg.HilbertSpace)
    assert L1.x.norm() == pytest.approx(0.5, abs=1e-12)
    a, b = L1.indicator("[0, 0.5]"), L1.indicator("[0.5, 1]")     # parallelogram law fails in L1
    assert (a + b).norm() ** 2 + (a - b).norm() ** 2 != pytest.approx(2 * a.norm() ** 2 + 2 * b.norm() ** 2)
    with pytest.raises(ValueError):
        pg.Lp("[0, 1]", 0.5)


def test_weighted_measure():
    mu = pg.LebesgueMeasure("[0, 1]").with_density(lambda x: 1 + x ** 2)
    W = pg.L2(mu)
    one = W.constant(1.0)
    assert one @ one == pytest.approx(4 / 3, abs=1e-13)
    assert W.x @ W.x == pytest.approx(1 / 3 + 1 / 5, abs=1e-13)


def test_point_evaluation_is_refused(M):
    with pytest.raises(pg.PointEvaluationUndefined):
        M.evaluation(0.3)
    avg = M.average("[0.2, 0.4]")
    assert avg(M.x) == pytest.approx(0.3, abs=1e-14)


def test_not_square_integrable_is_flagged(M):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f = M(lambda x: x ** -0.5)
        f.realisation
    assert any(issubclass(x.category, pg.RealisationWarning) for x in w)


# ---------------------------------------------------------------- Euclidean spaces

def test_euclidean_metrics():
    D = pg.Euclidean(3, weights=[1, 2, 3])
    x, y = D([1, 1, 1]), D([1, 0, 2])
    assert x @ y == pytest.approx(7)
    W = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 1.0]])
    E = pg.Euclidean(3, metric=W)
    assert E([1, 0, 0]) @ E([0, 1, 0]) == pytest.approx(1)
    with pytest.raises(ValueError):
        pg.Euclidean(2, metric=[[1, 2], [2, 1]])        # not positive definite
    V = pg.Euclidean(4, metric=np.diag([1, 2, 3, 4.0])).random(size=20000, seed=0).values
    assert np.cov(V.T) == pytest.approx(np.diag([1, 1 / 2, 1 / 3, 1 / 4]), abs=0.02)  # N(0, W^-1)


# ---------------------------------------------------------------- operators

@pytest.mark.parametrize("weights", [None, np.linspace(1, 3, 20)])
def test_adjoint_identity(M, weights):
    ks = [np.exp(-((M.x - c) / 0.05) ** 2) for c in np.linspace(0.05, 0.95, 20)]
    D = pg.Euclidean(20, weights=weights)
    G = pg.Functionals(ks, into=D)
    m = pg.probability.legendre_decay(M).sample(seed=5)
    y = D.random(seed=6)
    assert G(m) @ y == pytest.approx(m @ G.adjoint(y), rel=1e-12, abs=1e-14)


def test_gram_and_claims(G):
    A = G @ G.adjoint
    K = A.matrix()
    assert np.allclose(K, K.T, atol=1e-15) and np.linalg.eigvalsh(K).min() > 0
    assert A.is_selfadjoint()[1] == "derived"
    assert isinstance(A.recommend_solver().solver, pg.Cholesky)


def test_minimum_norm_solution(M, G):
    m_true = np.sin(2 * np.pi * M.x) + 0.5 * M.x ** 2
    d = G(m_true)
    m = G.adjoint((G @ G.adjoint).inv(d))
    assert np.linalg.norm((G(m) - d).values) <= 1e-13 * np.linalg.norm(d.values)
    assert m.norm() <= m_true.norm()
    null = m_true - m                                   # in the null space of G
    assert np.abs(G(null).values).max() < 1e-13
    assert abs(m @ null) < 1e-13                        # m is orthogonal to the null space


def test_ill_conditioned_gram_is_refused_and_has_an_explicit_pseudo_inverse(M):
    ks = [np.exp(-((M.x - c) / 0.05) ** 2) for c in np.linspace(0.05, 0.95, 120)]
    G = pg.Functionals(ks)
    rec = (G @ G.adjoint).recommend_solver()
    assert isinstance(rec.solver, pg.Direct)
    m_true = np.sin(2 * np.pi * M.x)
    d = G(m_true)
    with pytest.raises(pg.errors.Refused):
        (G @ G.adjoint).inv(d)
    m = G.adjoint((G @ G.adjoint).pseudo_inverse(1e-12)(d))
    assert np.linalg.norm((G(m) - d).values) <= 1e-8 * np.linalg.norm(d.values)
    assert m.norm() <= m_true.norm() * (1 + 1e-9)


def test_family_equals_individual(M, G):
    F = M.family(lambda x, a: np.sin(a * x), a=np.linspace(1, 30, 50))
    Y = G(F)
    for i in (0, 17, 49):
        assert np.allclose(Y[i].values, G(F[i]).values, rtol=0, atol=1e-14)
    assert G(F) is Y                                    # remembered on the family


def test_cg_on_a_function_space(M, G):
    alpha = 1e-3
    A = G.adjoint @ G + alpha * pg.Identity(M)
    rec = A.recommend_solver()
    assert isinstance(rec.solver, pg.CG)
    d = G(M(np.cos))
    m_cg = A.inverse(pg.CG(rtol=1e-12))(G.adjoint(d))
    m_dual = G.adjoint((G @ G.adjoint + alpha * pg.Identity(G.codomain)).inv(d))
    assert m_cg.distance(m_dual) < 1e-9 * m_dual.norm()


def test_realisations_are_immutable(M):
    f = M(np.sin)
    with pytest.raises(ValueError):
        f.realisation.coefs[0][0, 0] = 1.0
    v = pg.Euclidean(3)([1, 2, 3])
    with pytest.raises(ValueError):
        v.values[0] = 5


def test_adjoint_needs_hilbert_space():
    L1 = pg.Lp("[0, 1]", 1)
    T = pg.LinearOperator(L1, pg.Euclidean(1), apply=lambda f: pg.Euclidean(1)([f.integral()]))
    with pytest.raises(TypeError):
        T.adjoint
