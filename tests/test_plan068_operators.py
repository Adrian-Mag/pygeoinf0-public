"""Plan 068 P3 (architecture review 2026-11-13, B5/A4 types, A6, A12): the operators.

- operators/unbounded.py: FormOperator, SturmLiouville and one Laplacian(D, bc) with spectrum(), closed densely defined
  operators as single-valued linear relations (relations.linear.ClosedOperator) whose dom is D(A), dense and not
  closed (relations.linear.OperatorDomain), the graph-norm space only through domain_space();
- operators/inverses.py: Inverse, PseudoInverse (A⁺, its threshold and what it dropped) and MinimumNorm together;
- operators/error.py: the operator-error claim ||A~ - A|| <= e and its rules for sums, compositions (structural
  shortcuts included) and inverses, UNKNOWN where no rule applies;
- the function bank's Laplacian eigenfunctions as views on the Laplacian's spectrum, with contract tests against
  independent realisations (realisations/spectra/bessel.py, the Galerkin brackets of realisations/spectra/sl_galerkin.py).
"""
import importlib
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions  # noqa: F401  (pg.functions)
from pygeoinf0.core.claims import THEOREM, UNKNOWN
from pygeoinf0.core.errors import Refused, Unsupported
from pygeoinf0.core.logic import Truth
from pygeoinf0.convex import subspaces as csub
from pygeoinf0.sets import Interval


def galerkin(op):
    """The Galerkin brackets of a regular Sturm-Liouville operator (realisations/spectra/sl_galerkin.py), on a small mesh: an
    independent computation of the spectrum."""
    from pygeoinf0.realisations.spectra.sl_galerkin import GalerkinSpectrum
    return GalerkinSpectrum(op, cells=12, degree=6)


def unbounded():
    return importlib.import_module("pygeoinf0.operators.unbounded")


# ------------------------------------------------------------------------------------------------- moved paths
@pytest.mark.parametrize("module, name", [
    ("pygeoinf0.spaces.constructions", "FormOperator"),
    ("pygeoinf0.relations.linear", "_OperatorDomain"),
    ("pygeoinf0.operators.linear", "PseudoInverse"),
    ("pygeoinf0.operators.linear", "Inverse"),
    ("pygeoinf0.relations", "MinimumNorm"),
])
def test_moved_names_are_gone_from_their_old_modules(module, name):
    assert not hasattr(importlib.import_module(module), name)


def test_minimum_norm_is_defined_in_operators():
    from pygeoinf0.operators import inverses
    assert inverses.MinimumNorm.__module__ == "pygeoinf0.operators.inverses"


def test_core_sturm_liouville_is_gone():
    with pytest.raises(ImportError):
        importlib.import_module("pygeoinf0.core.sturm_liouville")


def test_new_homes():
    ub = unbounded()
    inv = importlib.import_module("pygeoinf0.operators.inverses")
    from pygeoinf0.relations.linear import ClosedOperator, LinearRelation, OperatorDomain
    for cls in (ub.FormOperator, ub.SturmLiouville, ub.Laplacian):
        assert issubclass(cls, ClosedOperator) and issubclass(cls, LinearRelation)
    assert issubclass(OperatorDomain, csub.LinearSubspace) and not issubclass(OperatorDomain, csub.ClosedSubspace)
    assert pg.Laplacian is ub.Laplacian
    assert pg.operators.PseudoInverse is inv.PseudoInverse and pg.operators.MinimumNorm is inv.MinimumNorm
    assert pg.Inverse is inv.Inverse and issubclass(inv.PseudoInverse, inv.Inverse)
    assert not hasattr(LinearRelation, "of_form_operator")


# ------------------------------------------------------------------------------------------------- unbounded
def test_form_operator_is_its_relation():
    A = pg.SobolevForm("[0, 1]", 1, length_scale=0.1).operator()
    assert not hasattr(A, "as_relation")
    assert A.domain == pg.L2(Interval("[0, 1]")) and A.dom.space == A.domain
    V = A.domain_space()
    assert type(V).__name__ == "SobolevSpace" and V != A.domain and V == pg.Sobolev(A, 2)
    assert A.dom.member(V(lambda t: np.cos(np.pi * t))) is Truth.TRUE
    assert A.adjoint is A                                         # self-adjoint
    assert A.closure() is A
    assert A.claims.status("self-adjoint") == (Truth.TRUE, THEOREM)
    with pytest.raises(Unsupported):
        A.converse()
    with pytest.raises(Unsupported):
        A(A.domain(lambda t: t))                                 # a descriptor: no realised action
    assert A == pg.SobolevForm("[0, 1]", 1, length_scale=0.1).operator() and hash(A) == hash(A.key())


def test_sturm_liouville_is_a_relation_on_its_weighted_l2():
    ub = unbounded()
    L = ub.SturmLiouville("[0, 2]", 1.0, 0.0, 3.0, left="dirichlet", right="neumann")
    X = L.domain
    assert isinstance(X, pg.L2) and X.inner(X(lambda t: np.ones_like(t), resolution=1.0),
                                              X(lambda t: np.ones_like(t), resolution=1.0)) == \
        pytest.approx(6.0, rel=1e-12)                                                  # int 1 * 3 dx over [0, 2]
    with pytest.raises(Unsupported) as e:
        L.domain_space()
    assert "graph" in str(e.value)
    assert L.dom.member(X.zero) is Truth.TRUE                                          # 0 is in every D(A)
    assert L.dom.member(X(lambda t: t, resolution=0.1)) is Truth.UNKNOWN              # not decided here


@pytest.mark.parametrize("bc, closed", [
    ({"left": "dirichlet", "right": "dirichlet"}, lambda n: (n * math.pi / 2.0) ** 2),
    ({"left": "neumann", "right": "neumann"}, lambda n: ((n - 1) * math.pi / 2.0) ** 2),
    ({"left": "dirichlet", "right": "neumann"}, lambda n: ((n - 0.5) * math.pi / 2.0) ** 2),
])
def test_laplacian_on_an_interval_closed_forms(bc, closed):
    Lap = pg.Laplacian(Interval("[1, 3]"), bc=bc)
    lam = Lap.spectrum().eigenvalues(6)
    ref = np.array([closed(n) for n in range(1, 7)])
    np.testing.assert_allclose(lam, ref, rtol=1e-14, atol=1e-14)
    lo, hi = Lap.spectrum().brackets(6)
    assert np.all(lo <= ref) and np.all(ref <= hi)


def test_laplacian_on_an_interval_robin_against_galerkin_brackets():
    ub = unbounded()
    bc = {"left": ("robin", 1.5), "right": "dirichlet"}
    lam = pg.Laplacian(Interval("[0, 1]"), bc=bc).spectrum().eigenvalues(5)
    G = galerkin(ub.SturmLiouville("[0, 1]", 1.0, 0.0, 1.0, left=("robin", 1.5), right="dirichlet"))
    lo, hi = G.brackets(5)
    assert np.all(lo <= lam * (1 + 1e-12)) and np.all(lam <= hi * (1 + 1e-12)), (lo, lam, hi)


def test_laplacian_on_a_box_is_the_sum_of_the_axes():
    B = pg.Box(Interval("[0, 1]"), Interval("[0, 2]"))
    Lap = pg.Laplacian(B, bc="dirichlet")
    lam = Lap.spectrum().eigenvalues(8)
    ref = sorted((i * math.pi) ** 2 + (j * math.pi / 2) ** 2 for i in range(1, 9) for j in range(1, 9))[:8]
    np.testing.assert_allclose(lam, ref, rtol=1e-13)
    lo, hi = Lap.spectrum().brackets(8)
    assert np.all(lo <= np.array(ref) * (1 + 1e-13)) and np.all(np.array(ref) <= hi * (1 + 1e-13))
    mixed = pg.Laplacian(B, bc={"x-": "dirichlet", "x+": "dirichlet", "y-": "neumann", "y+": "neumann"})
    assert mixed.spectrum().eigenvalues(1)[0] == pytest.approx(math.pi ** 2, rel=1e-14)    # (1, 0): pi^2 + 0
    with pytest.raises(ValueError):
        pg.Laplacian(B, bc={"x-": "dirichlet"})                                             # every face stated


def test_laplacian_on_a_ball_merges_the_degrees():
    Lap = pg.Laplacian(pg.Ball(1.0), bc={"boundary": "dirichlet"}, centre="regular")
    lam = Lap.spectrum().eigenvalues(10)
    x0, x1, x2 = math.pi, 4.493409457909064, 5.763459196894550                     # zeros of j_0, j_1, j_2
    ref = [x0 ** 2] + [x1 ** 2] * 3 + [x2 ** 2] * 5 + [(2 * math.pi) ** 2]
    np.testing.assert_allclose(lam, ref, rtol=1e-12)
    lo, hi = Lap.spectrum().brackets(10)
    assert np.all(lo <= np.array(ref) * (1 + 1e-12)) and np.all(np.array(ref) <= hi * (1 + 1e-12))
    with pytest.raises(Refused):                                     # degree 0 at the centre needs its condition
        pg.Laplacian(pg.Ball(1.0), bc={"boundary": "dirichlet"}).spectrum().eigenvalues(1)


@pytest.mark.parametrize("make, ref", [
    (lambda: pg.Circle(2.0), [0.0, 0.25, 0.25, 1.0, 1.0, 2.25, 2.25]),
    (lambda: pg.Sphere(0.5), [0.0] + [8.0] * 3 + [24.0] * 5),
    (lambda: pg.Torus(Interval("[0, 1]"), Interval("[0, 2]")),
     sorted((2 * math.pi * i) ** 2 + (math.pi * j) ** 2 for i in range(-4, 5) for j in range(-4, 5))[:9]),
])
def test_laplacian_on_closed_manifolds(make, ref):
    Lap = pg.Laplacian(make())
    np.testing.assert_allclose(Lap.spectrum().eigenvalues(len(ref)), ref, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError):
        pg.Laplacian(make(), bc={"boundary": "dirichlet"})                   # a closed manifold has no boundary


def test_laplacian_elsewhere_refuses_with_the_choices():
    with pytest.raises(Unsupported) as e:
        pg.Laplacian(pg.Disk(1.0), bc={"boundary": "dirichlet"})
    assert "laplace_eigen" in str(e.value)


def test_laplacian_claims():
    Lap = pg.Laplacian(pg.Shell(0.5, 1.0), bc={"inner": "neumann", "outer": "dirichlet"})
    assert Lap.claims.status("unbounded") == (Truth.TRUE, THEOREM)
    assert Lap.claims.status("rotation invariant") == (Truth.TRUE, THEOREM)
    assert Lap.radial(2).spectrum().eigenvalues(1)[0] > 6.0                         # >= s(s + 1)/b^2
    assert Lap.adjoint is Lap


# ------------------------------------------------------------------------------------------------- inverses
def test_pseudo_inverse_and_minimum_norm_live_together():
    inv = importlib.import_module("pygeoinf0.operators.inverses")
    X = pg.Euclidean(3)
    A = pg.Matrix(np.diag([1.0, 1e-14, 2.0]), X, X)
    P = A.pseudo_inverse(1e-10)
    assert isinstance(P, inv.PseudoInverse) and P.name.endswith("⁺")
    assert "1 of 3 eigenvalue(s) dropped" in P.claims.find("pseudo-inverse").statement
    G = pg.Matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), X, pg.Euclidean(2))
    mn = pg.LinearRelation(G).converse().G
    assert isinstance(mn, inv.MinimumNorm)
    d = pg.Euclidean(2)(np.array([1.0, 2.0]))
    np.testing.assert_allclose(mn(d).values, np.linalg.pinv(G.A) @ d.values, atol=1e-14)


# ------------------------------------------------------------------------------------------------- errors
def test_stated_error_api_and_claim():
    from pygeoinf0.operators import error as operror
    from pygeoinf0.solvers import stated_error
    X = pg.Euclidean(2)
    A = pg.Matrix(np.eye(2), X, X, name="A")
    operror.state(A, 1e-3, "measured")
    assert stated_error(A)[0] == 1e-3 and A.claims.find("operator error").status == "asserted"
    c = operror.error_claim(stated_error(3.0 * A))
    assert c.key == "operator error" and "0.003" in c.statement
    u = operror.error_claim((math.nan, UNKNOWN, "no rule"))
    assert u.status == UNKNOWN and "unknown" in u.statement
    with pytest.raises(ValueError):
        operror.state(A, -1.0, "negative")


def test_matrix_products_keep_the_factors_error():
    """A structural shortcut (Matrix @ Matrix multiplied out) keeps the composition rule (review A6)."""
    from pygeoinf0.operators import error as operror
    from pygeoinf0.solvers import stated_error
    X = pg.Euclidean(2)
    A, B = pg.Matrix(np.eye(2) * 2, X, X, name="A"), pg.Matrix(np.eye(2) * 3, X, X, name="B")
    operror.state(A, 0.1, "planted")
    AB = A @ B
    assert type(AB) is pg.Matrix
    e = stated_error(AB)
    assert e[0] == pytest.approx(0.1 * 3.0)                              # e_A ||B||, B exact


def test_inverse_rule_and_its_refusal_to_guess():
    from pygeoinf0.operators import error as operror
    from pygeoinf0.solvers import stated_error
    X = pg.Euclidean(2)
    A = pg.Matrix(np.diag([4.0, 2.0]), X, X, name="A")
    operror.state(A, 0.5, "planted")
    e = stated_error(A.inverse(pg.Direct()))
    assert e[0] == pytest.approx(0.25 * 0.5 / (1 - 0.25), rel=1e-12)       # n = 1/2: n^2 e / (1 - n e)
    B = pg.Matrix(np.diag([4.0, 0.25]), X, X, name="B")
    operror.state(B, 0.5, "planted")
    e = stated_error(B.inverse(pg.Direct()))
    assert math.isnan(e[0]) and e[1] == UNKNOWN and "singular" in e[2]


def test_operator_error_without_rounding_only():
    from pygeoinf0.solvers import stated_error
    X = pg.Euclidean(2)
    assert stated_error(pg.Matrix(np.eye(2), X, X) @ pg.Matrix(np.eye(2), X, X)) is None    # nothing stated: None


def test_norm_bound_protocol_without_type_tests_of_convex():
    from pygeoinf0.operators.error import norm_bound
    X = pg.Euclidean(3)
    P = csub.Span([X(np.array([1.0, 0.0, 0.0]))]).projector()
    assert norm_bound(P) == 1.0
    assert norm_bound(pg.Identity(X)) == 1.0


# ------------------------------------------------------------------------------------------------- the bank
@pytest.mark.parametrize("bc, beta, n", [("dirichlet", None, 3), ("neumann", None, 2), ("dn", None, 2),
                                         ("nd", None, 4), ("robin", (1.5, 0.7), 3)])
def test_bank_interval_mode_inside_independent_galerkin_brackets(bc, beta, n):
    """The bank's closed form (a view on the Laplacian's spectrum) against the Galerkin brackets of the Sturm-Liouville
    realisation (sl_galerkin.py), an independent computation."""
    ub = unbounded()
    I = Interval("[0, 2]")
    f = pg.functions.interval.laplace_eigen(I, n=n, bc=bc, beta=beta)
    ends = {"dirichlet": ("dirichlet", "dirichlet"), "neumann": ("neumann", "neumann"),
            "dn": ("dirichlet", "neumann"), "nd": ("neumann", "dirichlet")}.get(bc)
    if bc == "robin":
        ends = (("robin", beta[0]), ("robin", beta[1]))
    idx = n + 1 if bc == "neumann" else n
    lam = f.closed["eigenvalue"][0]
    assert lam == pg.Laplacian(I, bc={"left": ends[0], "right": ends[1]}).spectrum().eigenvalues(idx)[-1]
    lo, hi = galerkin(ub.SturmLiouville(I, 1.0, 0.0, 1.0, left=ends[0], right=ends[1])).brackets(idx)
    assert lo[-1] <= lam * (1 + 1e-12) + 1e-12 and lam <= hi[-1] * (1 + 1e-12) + 1e-12, (lo[-1], lam, hi[-1])


@pytest.mark.parametrize("bc, l, n", [("dirichlet", 0, 2), ("dirichlet", 3, 1), ("neumann", 0, 0), ("neumann", 2, 2)])
def test_bank_ball_eigenvalue_is_a_view_on_the_laplacian(bc, l, n):
    f = pg.functions.ball.laplace_eigen(pg.Ball(1.5), n=n, l=l, j=0, bc=bc)
    Lap = pg.Laplacian(pg.Ball(1.5), bc={"boundary": bc}, centre="regular")
    idx = n + 1 if (bc == "neumann" and l == 0) else n
    assert f.closed["eigenvalue"][0] == pytest.approx(Lap.radial(l).spectrum().eigenvalues(idx)[-1], rel=1e-14, abs=0)
