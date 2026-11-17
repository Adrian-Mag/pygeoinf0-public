"""Plan 065 P3 (architecture review 2026-11-13, A6, A12): operators. Each test failed on main a45553c (the false
statement) and passes with the repair."""
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import solvers
from pygeoinf0.core.claims import CHECKED, UNKNOWN
from pygeoinf0.core.errors import PrecisionWarning, RealisationWarning, Refused
from pygeoinf0.operators.inverses import MinimumNorm

CS = np.linspace(0.1, 0.9, 6)


def _G(tol):
    M = pg.L2(pg.Interval(0, 1), tol=tol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)
        return pg.Functionals([M(lambda x, c=c: np.exp(-(x - c) ** 2 / 0.02)) for c in CS])


# ------------------------------------------------------------------------------------------------ A6
def test_a6_a_composition_keeps_the_stated_error_of_its_factors():
    G = _G(1e-5)
    D = G.codomain
    mn = MinimumNorm(G, pg.Span(D.family(np.eye(6)[:1])))
    A = mn.Ainv.A                                              # P G G* P, a Composition
    st = solvers.stated_error(A)
    assert st is not None and st[1] == CHECKED                 # main: None, "nothing beyond rounding"
    Gt = _G(1e-11)                                             # the same kernels realised far more accurately
    P = mn.P.matrix()
    true = P @ (G @ G.adjoint).matrix() @ P - P @ (Gt @ Gt.adjoint).matrix() @ P
    assert np.linalg.norm(true, 2) <= st[0]                    # the stated bound holds


def test_a6_minimum_norm_states_what_its_solve_inherits():
    G = _G(1e-5)
    mn = MinimumNorm(G, pg.Span(G.codomain.family(np.eye(6)[:1])))
    c = mn.claims.find("inherited error")
    assert c is not None and c.status == UNKNOWN               # main: only the rounding claim
    assert "pseudo-inverse" in c.statement


def test_a6_no_rule_gives_unknown_not_rounding_only():
    G = _G(1e-5)
    GG = G @ G.adjoint
    D = G.codomain
    U = pg.LinearOperator(D, D, lambda y: y, name="U")         # no norm on record
    st = solvers.stated_error(U @ GG)
    assert st is not None and st[1] == UNKNOWN                 # main: None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        c = GG.pseudo_inverse(1e-12).claims.find("inherited error")
    assert c is not None and c.status == UNKNOWN               # main: skipped for any rcond solver


# ------------------------------------------------------------------------------------------------ A12
def test_a12_a_pseudo_inverse_is_named_and_states_its_threshold():
    A = pg.Matrix(np.diag([1.0, 1e-3, 1e-14]))
    P = A.pseudo_inverse(1e-10)
    assert "⁺" in P.name and "⁻¹" not in P.name                 # main: "(A)⁻¹"
    c = P.claims.find("pseudo-inverse")
    assert c is not None and "1 of 3" in c.statement and "1e-10" in c.statement
    x = P(pg.Euclidean(3)(np.ones(3))).values
    assert np.allclose(x, [1.0, 1e3, 0.0])


def test_a12_a_non_square_pseudo_inverse_is_refused_when_built():
    G = pg.Matrix(np.ones((2, 3)))
    with pytest.raises(Refused) as e:                          # main: built as "(A)⁻¹: R^2 -> R^3", fails on first use
        G.pseudo_inverse(1e-12)
    assert "G.adjoint" in str(e.value.needs) or "adjoint" in " ".join(e.value.needs)
