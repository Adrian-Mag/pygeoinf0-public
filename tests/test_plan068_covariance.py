"""Plan 068 P4 (architecture review B6): one Covariance root in probability/.

A Covariance is positive, self-adjoint and trace class, with a bracketed trace (an Enclosure). The stated covariance of
an operator (pg.Covariance(A), probability/operator_covariance.py), the posterior covariance in Joseph's form
(probability/posterior_covariance.py), the lazy covariances of property families (probability/lazy_covariance.py, from
operators/families.py) and of many properties (LowRankCovariance, probability/lazy_covariance.py since plan 070 P3) implement it; FiniteRank stays a general
operator. The properties are Truth-valued and come from the construction's claims: derived from premises, checked on
R^n with the precision rule, asserted by the caller, or UNKNOWN; inference/methods.py sets no _sa or _pos by hand.
"""
import importlib
import inspect
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN
from pygeoinf0.core.errors import Refused
from pygeoinf0.core.logic import Truth
from pygeoinf0.core.numbers import Enclosure


def root():
    return importlib.import_module("pygeoinf0.probability.covariance").Covariance


# ------------------------------------------------------------------------------------------------ the root
def test_the_implementations_share_one_root():
    Cov = root()
    from pygeoinf0.probability.operator_covariance import OperatorCovariance
    from pygeoinf0.probability.posterior_covariance import PosteriorCovariance
    from pygeoinf0.probability.lazy_covariance import FactoredCovariance
    from pygeoinf0.probability.lazy_covariance import LowRankCovariance
    from pygeoinf0.probability.degrees import GluedCovariance
    for C in (OperatorCovariance, PosteriorCovariance, FactoredCovariance, LowRankCovariance, GluedCovariance):
        assert issubclass(C, Cov), C
    assert not issubclass(pg.FiniteRank, Cov)                      # FiniteRank stays a general operator
    assert pg.Covariance is Cov and pg.probability.Covariance is Cov


def test_pg_covariance_of_an_operator_is_the_stated_covariance():
    from pygeoinf0.probability.operator_covariance import OperatorCovariance
    C = pg.Covariance(pg.Matrix(np.diag([1.0, 2.0])))
    assert type(C) is OperatorCovariance and isinstance(C, root())
    assert float(C.trace()) == pytest.approx(3.0)


def test_the_factored_covariance_left_operators():
    import pygeoinf0.properties.families as fam                    # plan 070 P1: the families are in properties/
    assert not hasattr(fam, "FactoredCovariance")
    with pytest.raises(ImportError):
        exec("from pygeoinf0.operators.families import FactoredCovariance", {})


# ------------------------------------------------------------------------------------------------ Truth-valued
def test_properties_are_truth_valued_from_the_claims():
    X = pg.Euclidean(3)
    phi = X.family(np.eye(3)[:2])
    C = pg.Covariance.spectral([2.0, 1.0], phi)
    assert C.selfadjoint is Truth.TRUE and C.positive is Truth.TRUE and C.trace_class is Truth.TRUE
    assert C.holds("positive semidefinite") == (Truth.TRUE, THEOREM)
    assert C.holds("self-adjoint")[1] == THEOREM
    assert C.holds("trace class")[0] is Truth.TRUE


def test_a_property_nobody_derived_or_checked_is_unknown():
    """A covariance on a function space built from an operator without records: nothing derived, nothing checked; the
    root says UNKNOWN (the stated covariance of pg.Covariance(A) says ASSERTED: the caller stated it)."""
    from pygeoinf0.probability.posterior_covariance import PosteriorCovariance
    M = pg.L2("[0, 1]")
    A = pg.LinearOperator(M, M, lambda x: x, adjoint=lambda x: x, name="A")
    G = pg.LinearOperator(M, pg.Euclidean(1), lambda x: pg.Euclidean(1)(np.zeros(1)), adjoint=lambda y: M.zero,
                          name="G")
    P = PosteriorCovariance(A, G, pg.Identity(pg.Euclidean(1)), pg.Identity(pg.Euclidean(1)))
    assert P.positive is Truth.UNKNOWN and P.selfadjoint is Truth.UNKNOWN
    assert P.trace_class is Truth.UNKNOWN
    S = pg.Covariance(A)
    assert S.positive is Truth.TRUE and S.holds("positive semidefinite")[1] == ASSERTED


def test_stated_covariance_on_rn_checks_positivity():
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    C = pg.Covariance(pg.Matrix(S))
    assert C.holds("positive semidefinite") == (Truth.TRUE, CHECKED)
    with pytest.raises(Refused):                                   # base: accepted, "positive semidefinite, asserted"
        pg.Covariance(pg.Matrix(np.array([[1.0, 2.0], [2.0, 1.0]])))


# ------------------------------------------------------------------------------------------------ the posteriors
def _problem(stated=False):
    """A prior whose covariance is a spectral representation (positive semidefinite, a theorem), or a matrix only
    asserted to be a covariance by building the measure (stated=True); the noise the standard Gaussian of R^2."""
    E, D = pg.Euclidean(3), pg.Euclidean(2)
    C = pg.Matrix(np.diag([1.0, 2.0, 3.0]), domain=E, codomain=E) if stated else \
        pg.Covariance.spectral([1.0, 2.0, 3.0], E.family(np.eye(3)))
    prior = pg.GaussianMeasure(E, covariance=C)
    G = pg.Matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), domain=E, codomain=D)
    return E, D, prior, G


def test_the_posterior_covariance_is_a_covariance_with_derived_claims():
    E, D, prior, G = _problem()
    post = prior.condition(G, D(np.zeros(2)), D.standard_gaussian())
    C = post.covariance
    assert isinstance(C, root())
    assert C.holds("positive semidefinite") == (Truth.TRUE, DERIVED)
    assert C.holds("self-adjoint") == (Truth.TRUE, DERIVED)
    M = C.matrix()
    assert C.trace().contains(float(np.trace(M)))
    b = C.trace_bound()                                             # the route without a matrix: 0 <= C_post <= C
    assert b.lower == 0.0 and b.contains(float(np.trace(M))) and b.upper >= 6.0
    asserted = _problem(stated=True)[2].condition(G, D(np.zeros(2)), D.standard_gaussian()).covariance
    assert asserted.holds("positive semidefinite") == (Truth.TRUE, ASSERTED)     # never stronger than the premise


def test_methods_set_no_records_by_hand():
    from pygeoinf0.inference import methods
    src = inspect.getsource(methods)
    assert "._sa =" not in src and "._pos =" not in src


def test_the_property_posterior_is_a_covariance():
    E, D, prior, G = _problem()
    P = pg.Euclidean(1)
    T = pg.Matrix(np.array([[1.0, 1.0, 0.0]]), domain=E, codomain=P)
    out = pg.inference.Problem(M=E, D=D, P=P, prior=prior, mdr=pg.Additive(G, D.standard_gaussian()), mpr=T,
                               datum=D(np.array([0.5, -0.2]))).run("pli")
    C = out.covariance
    assert isinstance(C, root())
    assert C.holds("positive semidefinite") == (Truth.TRUE, DERIVED)
    assert C.trace().contains(float(np.trace(C.matrix())))


def test_factored_covariance_trace_and_checked_positivity():
    from pygeoinf0.probability.lazy_covariance import FactoredCovariance
    rng = np.random.default_rng(1)
    Phi, R = rng.normal(size=(7, 3)), rng.normal(size=(3, 3))
    P = pg.Euclidean(7)
    C = FactoredCovariance(P, Phi, R @ R.T, np.abs(rng.normal(size=7)))
    assert isinstance(C, root())
    assert C.holds("positive semidefinite") == (Truth.TRUE, CHECKED)
    assert C.trace().contains(float(np.trace(C.matrix())))
    bad = FactoredCovariance(P, Phi, R @ R.T, -10.0 * np.ones(7))   # a negative diagonal: not a covariance
    assert bad.positive is not Truth.TRUE


def test_low_rank_covariance_trace_contains_the_computed_trace():
    from pygeoinf0.probability.lazy_covariance import LowRankCovariance
    rng = np.random.default_rng(2)
    p, k = 6, 2
    T = rng.normal(size=(p, 4))
    TT = pg.Matrix(T @ T.T)
    beta, w, a2 = rng.normal(size=(k, p)), np.array([0.3, 0.1]), 2.0
    diag = np.maximum(a2 * np.diag(T @ T.T) - a2 * a2 * ((beta * beta).T @ w), 0.0)
    C = LowRankCovariance(pg.Euclidean(p), TT, beta, w, a2, diag)
    assert isinstance(C, root())
    assert C.trace().contains(float(np.trace(C.matrix())))
    assert C.trace_class is Truth.TRUE                             # finite dimension


def test_trace_is_an_enclosure_everywhere():
    E, D, prior, G = _problem()
    for C in (pg.Covariance(pg.Matrix(np.eye(2))), prior.condition(G, D(np.zeros(2)), D.standard_gaussian()).covariance):
        assert isinstance(C.trace(), Enclosure)
    M = pg.L2("[0, 1]")
    A = pg.LinearOperator(M, M, lambda x: x, adjoint=lambda x: x, name="A")
    t = pg.Covariance(A).trace()
    assert isinstance(t, Enclosure) and t.status == UNKNOWN and t.upper == math.inf
