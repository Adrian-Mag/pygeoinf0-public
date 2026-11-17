"""Plan 069 P6, named priors (architecture review 2026-11-13, A13 and E3; decision Q2: default priors on L2 removed, named
prior constructors only, chosen by the caller).

- No space picks a prior: X.gaussian() refuses on every space class (core.errors.PriorUnstated, a Refused) and names
  the constructors; a function space's X.random() refuses too (it drew from the default prior).
- pygeoinf0.probability.priors: legendre_decay(X, decay) on L2 of an interval, random_features(X, length_scale,
  features, seed) on L2 of a box, sphere or ball (with what it approximates claimed), cylindrical(H, alpha) on a Hilbert
  space; pg.CylindricalGaussian(H, alpha); D.standard_gaussian() on a finite-dimensional space.
- Function space first: each prior is stated on the function space (an operator on X with its claims through the
  Covariance root), never in coefficient coordinates.
"""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.claims import DERIVED, THEOREM, UNKNOWN
from pygeoinf0.core.errors import PriorUnstated, Refused
from pygeoinf0.core.logic import Truth
from pygeoinf0.probability import CylindricalGaussian, GaussianMeasure
from pygeoinf0.probability.priors import cylindrical, legendre_decay, random_features

I01 = pg.Interval("[0, 1]")
NAMES = ("legendre_decay", "random_features", "cylindrical", "standard_gaussian")


def _cells_l2():
    cells = pg.realisations.Cells.tensor(np.linspace(0, 1, 5), np.linspace(0, 2, 3))
    return pg.L2(pg.Box([(0, 1), (0, 2)]), realisation=cells)


def _planet():
    return pg.LayeredBall([0.3, 0.6, 1.0], names=["core", "mantle", "crust"], interfaces=["cmb", "moho"])


# every concrete space class: a constructor (test_every_space_class_has_a_case keeps the table complete)
SPACES = {
    "L2": lambda: pg.L2(I01),
    "LpSpace": lambda: pg.Lp(I01, 3),
    "SobolevSpace": lambda: pg.Sobolev(I01, 1, length_scale=0.2),
    "ContinuousSpace": lambda: pg.Continuous("[0, 1]"),
    "Euclidean": lambda: pg.Euclidean(3, weights=[1.0, 2.0, 3.0]),
    "_ComplexEuclidean": lambda: pg.Euclidean(2, field="C"),
    "DataSpace": lambda: pg.DataSpace(3, labels={"station": ["a", "b", "c"]}),
    "DirectSum": lambda: pg.DirectSum([pg.L2(I01), pg.Euclidean(2)]),
    "Complexified": lambda: pg.Complexified(pg.Sobolev(I01, 1, length_scale=0.2)),
    "CellsComplexified": lambda: pg.Complexified(_cells_l2()),
    "PiecewiseSpace": lambda: pg.PiecewiseSpace(pg.Partition(pg.Ball(1.0), [pg.Ball(0.5), pg.Shell(0.5, 1.0)],
                                                          names=["in", "out"]), {"vp": pg.L2}),
    "DualSpace": lambda: pg.L2(I01).dual(),
}
ABSTRACT = {"VectorSpace", "NormedSpace", "BanachSpace", "HilbertSpace", "_FunctionHilbertSpace", "CoordinateSpace"}
FINITE = {"Euclidean", "_ComplexEuclidean", "DataSpace"}


def _space_classes():
    from pygeoinf0.spaces.base import VectorSpace
    seen, todo = set(), [VectorSpace]
    while todo:
        for s in todo.pop().__subclasses__():
            if s not in seen and s.__module__.startswith("pygeoinf0."):
                seen.add(s)
                todo.append(s)
    return {c.__name__ for c in seen}


def test_every_space_class_has_a_case():
    found = _space_classes() - ABSTRACT
    assert found == set(SPACES), f"space classes without a case: {sorted(found - set(SPACES))}; stale: " \
                                 f"{sorted(set(SPACES) - found)}"


@pytest.mark.parametrize("name", sorted(SPACES))
def test_no_space_builds_a_prior_unless_asked(name):
    X = SPACES[name]()
    assert type(X).__name__ == name
    with pytest.raises(PriorUnstated) as e:
        X.gaussian()
    assert isinstance(e.value, Refused)
    assert all(n in str(e.value) for n in NAMES), str(e.value)
    assert e.value.needs
    if name in FINITE:                      # exactly one object is meant: the standard Gaussian of the inner product
        mu = X.standard_gaussian()
        assert isinstance(mu, GaussianMeasure)
        assert X.random(seed=1).space == X
    else:
        with pytest.raises(PriorUnstated):
            X.standard_gaussian()


@pytest.mark.parametrize("make", [lambda: pg.L2(I01), lambda: pg.Lp(I01, 3), lambda: pg.L2(pg.Box([(0, 1), (0, 1)]))])
def test_a_function_space_draws_nothing_without_a_named_prior(make):
    X = make()
    with pytest.raises(PriorUnstated, match="legendre_decay"):
        X.random(seed=0)


def test_the_hook_that_chose_the_default_is_gone():
    from pygeoinf0.realisations.backends import Backend
    assert not hasattr(Backend, "gaussian_generators")


# ------------------------------------------------------------------------------------------------ legendre_decay
def test_legendre_decay_is_stated_on_the_function_space():
    X = pg.L2(I01)
    mu = legendre_decay(X, 0.7)
    assert isinstance(mu, GaussianMeasure) and mu.space == X
    C = mu.covariance
    assert isinstance(C, pg.Covariance)
    for key in ("self-adjoint", "positive semidefinite", "trace class"):
        assert C.holds(key) == (Truth.TRUE, THEOREM), (key, C.claims)
    K = C.finite[1].size
    tail = 0.7 ** (2 * K) / (1 - 0.7 ** 2)                       # sum_{k >= K} decay^(2k)
    T = C.trace()
    full = 1.0 / (1 - 0.49)                                      # tr C = sum_k decay^(2k), every mode
    assert T.lower <= full <= T.upper and T.upper - T.lower >= tail * (1 - 1e-12)
    assert C.rest["trace"] >= tail * (1 - 1e-12) and C.rest["norm"] >= 0.7 ** (2 * K) * (1 - 1e-12)
    trunc = [c for c in mu.claims if c.key == "truncation"]
    assert trunc and trunc[0].status == THEOREM
    # the variance of a bounded functional: sum_k decay^(2k) <phi_k, u>^2 with phi_k the orthonormal Legendre functions
    u = X(lambda t: np.exp(t))
    k = np.arange(4)
    phi = [X(lambda t, j=j: np.sqrt(2 * j + 1) * np.polynomial.legendre.Legendre.basis(j)(2 * t - 1)) for j in k]
    head = sum(0.49 ** j * X.inner(p, u) ** 2 for j, p in zip(k, phi))
    assert head <= mu.variance(u) <= head + 0.49 ** 4 / 0.51 * X.norm(u) ** 2


def test_legendre_decay_keeps_the_old_default_numbers():
    X = pg.L2(I01)
    mu = legendre_decay(X, 0.8)
    assert mu.covariance.finite[1].size == int(math.ceil(math.log(1e-16) / math.log(0.8))) + 1
    a, b = mu.sample(seed=3), legendre_decay(pg.L2(I01), 0.8).sample(seed=3)
    assert X.norm(a - b) == 0.0


def test_legendre_decay_modes_and_refusals():
    X = pg.L2(I01)
    mu = legendre_decay(X, 0.5, modes=6)
    assert mu.covariance.finite[1].size == 6
    with pytest.raises(ValueError):
        legendre_decay(X, 1.0)
    with pytest.raises(Refused, match="interval"):
        legendre_decay(pg.L2(pg.Box([(0, 1), (0, 1)])), 0.5)
    with pytest.raises(Refused, match="L2"):
        legendre_decay(pg.Lp(I01, 3), 0.5)
    weighted = pg.L2(pg.DensityMeasure(pg.LebesgueMeasure(I01), lambda x: 1.0 + x))
    with pytest.raises(Refused, match="orthonormal"):                  # phi_k are not orthonormal for (1 + x) dx
        legendre_decay(weighted, 0.5)


# ------------------------------------------------------------------------------------------------ random_features
def test_random_features_claims_what_it_approximates():
    X = pg.L2(pg.Box([(0, 1), (0, 1)]))
    mu = random_features(X, length_scale=0.3, features=16, seed=2)
    C = mu.covariance
    assert isinstance(C, pg.Covariance)
    for key in ("self-adjoint", "positive semidefinite"):
        assert C.holds(key) == (Truth.TRUE, THEOREM), (key, C.claims)
    assert C.holds("trace class") == (Truth.TRUE, DERIVED)                 # finite rank: sum lam_j ||psi_j||^2
    approx = {c.key: c for c in mu.claims if c.key and c.key.startswith("approximat")}
    assert approx["approximates"].status == THEOREM                       # in expectation over the features, Hoeffding
    assert "exp(-|x - y|^2 / (2 l^2))" in approx["approximates"].statement
    assert "2 exp(-F eps^2 / 8)" in approx["approximates"].statement
    assert approx["approximation error"].status == UNKNOWN                # this draw's error: not derived
    # the covariance function of the draw: E u(x) u(y) = (2/F) sum_j psi_j(x) psi_j(y) for the realised features
    rng = np.random.default_rng(2)
    W = rng.standard_normal((16, 2)) / 0.3
    b = rng.uniform(0, 2 * math.pi, 16)
    x, y = np.array([0.2, 0.7]), np.array([0.6, 0.1])
    kF = 2.0 / 16 * float(np.cos(W @ x + b) @ np.cos(W @ y + b))
    psi = C.finite[0]
    B = X.backend
    px = np.asarray(B.evaluate(psi.realisation, *[np.array([c]) for c in x])).ravel()
    py = np.asarray(B.evaluate(psi.realisation, *[np.array([c]) for c in y])).ravel()
    got = 2.0 / 16 * float(px @ py)
    assert got == pytest.approx(kF, abs=1e-8)


def test_random_features_refuses_an_interval_and_lp():
    with pytest.raises(Refused, match="random_features"):
        random_features(pg.L2(I01), 0.2)
    with pytest.raises(Refused, match="L2"):
        random_features(pg.Lp(pg.Box([(0, 1), (0, 1)]), 3), 0.2)


# ------------------------------------------------------------------------------------------------ cylindrical
def test_cylindrical_and_the_class_constructor():
    H = pg.Sobolev(I01, 1, length_scale=0.2)
    g = cylindrical(H, 2.0)
    assert isinstance(g, CylindricalGaussian) and not isinstance(g, GaussianMeasure)
    h = CylindricalGaussian(H, 2.0)
    u = H(lambda t: t * (1 - t))
    assert h.variance(u) == pytest.approx(H.norm(u) ** 2 / 2.0, rel=1e-13) == g.variance(u)
    assert isinstance(CylindricalGaussian(H), CylindricalGaussian)
    D = pg.Euclidean(3)
    assert isinstance(cylindrical(D, 2.0), GaussianMeasure)               # finite dimension: a measure
    with pytest.raises(Refused, match="standard_gaussian"):
        CylindricalGaussian(D, 2.0)
    with pytest.raises(ValueError):
        cylindrical(H, 0.0)


def test_standard_gaussian_on_finite_dimensional_spaces():
    D = pg.Euclidean(3, weights=[1.0, 2.0, 4.0])
    mu = D.standard_gaussian()
    assert isinstance(mu, GaussianMeasure) and isinstance(mu.covariance, pg.Identity)
    V = D.random(size=40000, seed=0).values
    np.testing.assert_allclose(V.var(axis=0), [1.0, 0.5, 0.25], rtol=0.05)
    S = pg.DirectSum([pg.Euclidean(2), pg.Euclidean(1)])
    assert isinstance(S.standard_gaussian(), GaussianMeasure)
    Z = pg.Euclidean(2, field="C").standard_gaussian()
    assert any(c.key == "circular" for c in Z.claims)


def test_the_refusal_names_the_realisation_that_has_the_prior():
    with pytest.raises(PriorUnstated) as e:
        pg.L2(I01).gaussian(decay=0.5)
    assert "legendre_decay" in str(e.value) and "decay" in str(e.value)


def test_no_package_module_calls_a_space_default_prior():
    """Inside the package every prior is named: no module calls X.gaussian(...) (a call of an attribute named
    gaussian), so no code path builds a prior the caller did not name."""
    import ast
    import pathlib
    root = pathlib.Path(pg.__file__).parent
    bad = []
    for p in root.rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "gaussian":
                bad.append(f"{p.relative_to(root)}:{node.lineno}")
    assert not bad, bad
