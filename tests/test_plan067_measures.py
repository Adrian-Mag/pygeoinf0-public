"""Plan 067 P5, the Measure root (architecture review 2026-11-13, B2; decisions Q11, Q12 and the author's amendment 3).

- A package measures/ (layer L2, in the block of core and sets until Plan D splits core) holds the root: Measure(set,
  sigma_algebra) with claims (finite, probability, signed: keys of its claims, each a Truth), mass, integration and
  restriction; the measures of core/measures.py (Lebesgue, density, counting, Dirac, signed combinations), the
  weights and Radon-Nikodym densities (core/weights.py, core/densities.py) and the restriction of measures
  (sets/restrict.py) move in: a clean break, no alias modules.
- ProbabilityMeasure(Measure) on a set or a space: GaussianMeasure is one; pg.Dirac(x) is the named point mass on a
  space (the degenerate Gaussian N(x, 0), which stays a valid Dirac) and the cells build it; laws on finite sets are
  DiscreteMeasure (the dict laws are refused). Base measures of a domain and probability measures share the root, not
  the class.
- CylindricalGaussian (Q11) is its own class: it speaks about bounded functionals only and is not a Measure on its
  space; the cells check it by class, not by attribute.
"""
import importlib
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.logic import Truth
from pygeoinf0.core.kinds import MEASURE
from pygeoinf0.measures import (CountingMeasure, DensityMeasure, DiracMeasure, LebesgueMeasure, Measure, PieceDensity,
                                SignedMeasure)
from pygeoinf0.probability import CylindricalGaussian, Dirac, DiscreteMeasure, GaussianMeasure, ProbabilityMeasure

KEYS = ("finite", "probability", "signed")


# ------------------------------------------------------------------------------------------------ the package
@pytest.mark.parametrize("old", ["pygeoinf0.core.measures", "pygeoinf0.core.weights", "pygeoinf0.core.densities",
                                 "pygeoinf0.sets.restrict"])
def test_the_moved_modules_are_gone(old):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)


def test_measures_is_a_package_at_its_layer():
    from pygeoinf0.measures.densities import Density, density_against          # noqa: F401
    from pygeoinf0.measures.restrict import point_in, restrict                  # noqa: F401
    from pygeoinf0.measures.weights import Weight, as_weight                    # noqa: F401
    import test_structure as ts
    assert "measures" in ts.RANK
    assert ts.RANK["sets"] <= ts.RANK["measures"] < ts.RANK["realisations"] < ts.RANK["probability"]
    assert pg.Measure is Measure and pg.ProbabilityMeasure is ProbabilityMeasure
    assert pg.Dirac is Dirac and pg.DiscreteMeasure is DiscreteMeasure and pg.CylindricalGaussian is CylindricalGaussian


# ------------------------------------------------------------------------------------------------ one root
def base_measures():
    I = pg.Interval("[0, 1]")
    pl = pg.Polyline([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)])
    return {
        "lebesgue [0, 1]": LebesgueMeasure(I),
        "lebesgue [0, 2]": LebesgueMeasure("[0, 2]"),
        "lebesgue disk": LebesgueMeasure(pg.Disk(1.0)),
        "density 2x": LebesgueMeasure(I).with_density(lambda x: 2.0 * x, name="2x"),
        "density x - 1/2": LebesgueMeasure(I).with_density(lambda x: x - 0.5, name="x - 1/2"),
        "piece density": PieceDensity(pl, [1.0, 0.5]),
        "counting": CountingMeasure([0.1, 0.4, 0.9]),
        "counting half": CountingMeasure([0.1, 0.4], weights=[0.5, 0.5]),
        "dirac": DiracMeasure(0.3),
        "dirac 2-d": DiracMeasure((0.2, 0.3)),
        "signed": 3.0 * DiracMeasure(0.2) - LebesgueMeasure(I),
        "negative": -1.0 * LebesgueMeasure(I),
        "positive combination": 2.0 * LebesgueMeasure(I) + DiracMeasure(0.5),
    }


def probability_measures():
    X = pg.Euclidean(2)
    x = X([1.0, -2.0])
    return {
        "gaussian": X.standard_gaussian(),
        "dirac(x)": Dirac(x),
        "degenerate gaussian": GaussianMeasure(X, 0.0 * pg.Identity(X), mean=x),
        "discrete": DiscreteMeasure({0.0: 0.25, 1.0: 0.75}),
    }


def test_every_measure_has_the_one_root():
    for name, mu in {**base_measures(), **probability_measures()}.items():
        assert isinstance(mu, Measure), name
        assert mu.set is not None and isinstance(mu.sigma_algebra, str), name
    for name, mu in base_measures().items():
        assert not isinstance(mu, ProbabilityMeasure), name               # the root is shared, not the class
    for name, mu in probability_measures().items():
        assert isinstance(mu, ProbabilityMeasure), name
        assert mu.space is mu.set, name                                   # a probability measure on its space or set


# ------------------------------------------------------------------------------------------------ claims
WANT = {   # (finite, probability, signed)
    "lebesgue [0, 1]": (Truth.TRUE, Truth.TRUE, Truth.FALSE),
    "lebesgue [0, 2]": (Truth.TRUE, Truth.FALSE, Truth.FALSE),
    "lebesgue disk": (Truth.TRUE, Truth.FALSE, Truth.FALSE),
    "density 2x": (Truth.TRUE, Truth.TRUE, Truth.FALSE),
    "density x - 1/2": (Truth.TRUE, Truth.FALSE, Truth.TRUE),
    "piece density": (Truth.TRUE, Truth.FALSE, Truth.FALSE),
    "counting": (Truth.TRUE, Truth.FALSE, Truth.FALSE),
    "counting half": (Truth.TRUE, Truth.TRUE, Truth.FALSE),
    "dirac": (Truth.TRUE, Truth.TRUE, Truth.FALSE),
    "dirac 2-d": (Truth.TRUE, Truth.TRUE, Truth.FALSE),
    "signed": (Truth.TRUE, Truth.FALSE, Truth.UNKNOWN),                    # terms may cancel: not decided
    "negative": (Truth.TRUE, Truth.FALSE, Truth.TRUE),
    "positive combination": (Truth.TRUE, Truth.FALSE, Truth.FALSE),
    "gaussian": (Truth.TRUE, Truth.TRUE, Truth.FALSE),
    "dirac(x)": (Truth.TRUE, Truth.TRUE, Truth.FALSE),
    "degenerate gaussian": (Truth.TRUE, Truth.TRUE, Truth.FALSE),
    "discrete": (Truth.TRUE, Truth.TRUE, Truth.FALSE),
}


def test_claims_say_finite_probability_signed():
    for name, mu in {**base_measures(), **probability_measures()}.items():
        got = tuple(mu.claims.status(k)[0] for k in KEYS)
        assert got == WANT[name], (name, got)


def test_the_claims_agree_with_the_mass():
    """probability TRUE => mass 1 within its error and not signed; signed FALSE => mass >= 0; finite TRUE => finite."""
    for name, mu in {**base_measures(), **probability_measures()}.items():
        m, err = mu.mass()
        fin, prob, sig = (mu.claims.status(k)[0] for k in KEYS)
        if fin is Truth.TRUE:
            assert math.isfinite(m), name
        if prob is Truth.TRUE:
            assert abs(m - 1.0) <= err + 1e-14 and sig is Truth.FALSE, name
        if prob is Truth.FALSE and sig is Truth.FALSE and err == 0:
            assert m != 1.0, name
        if sig is Truth.FALSE:
            assert m >= -err, name


# ------------------------------------------------------------------------------------------------ mass, integration
def test_integration_of_one_is_the_mass():
    for name, mu in base_measures().items():
        v, e = mu.integrate(lambda *x: np.ones_like(x[0]))
        m, em = mu.mass()
        assert v == pytest.approx(m, rel=1e-12, abs=1e-12), name
        assert e >= 0 and em >= 0, name


def test_integrals_of_base_measures():
    I = pg.Interval("[0, 1]")
    x2 = lambda x: x ** 2                                                    # noqa: E731
    assert LebesgueMeasure(I).integrate(x2)[0] == pytest.approx(1.0 / 3.0, rel=1e-14)
    assert LebesgueMeasure(I).with_density(lambda x: 2.0 * x).integrate(lambda x: x)[0] == \
        pytest.approx(2.0 / 3.0, rel=1e-12)
    assert DiracMeasure(0.3).integrate(x2) == (pytest.approx(0.09, rel=1e-15), 0.0)
    assert DiracMeasure((0.2, 0.3)).integrate(lambda x, y: x * y) == (pytest.approx(0.06, rel=1e-15), 0.0)
    assert CountingMeasure([0.1, 0.4], weights=[2.0, 3.0]).integrate(x2)[0] == pytest.approx(2 * 0.01 + 3 * 0.16)
    mu = 2.0 * DiracMeasure(0.2) - LebesgueMeasure(I)
    assert mu.integrate(lambda x: x)[0] == pytest.approx(0.4 - 0.5, rel=1e-13)
    disk = LebesgueMeasure(pg.Disk(1.0))
    assert disk.integrate(lambda x, y: x ** 2 + y ** 2)[0] == pytest.approx(math.pi / 2, rel=1e-12)


def test_gaussian_mass_and_integration_of_bounded_linear_functionals():
    X = pg.Euclidean(3)
    m = X([1.0, 2.0, -1.0])
    mu = GaussianMeasure(X, mean=m)
    u = X([0.5, 0.0, 2.0])
    assert mu.mass() == (1.0, 0.0)
    assert mu.integrate(u) == (pytest.approx(float(X.inner(m, u)), rel=1e-15), 0.0)       # E <x, u> = <m, u>
    with pytest.raises(pg.errors.Unsupported, match="Monte Carlo"):
        mu.integrate(lambda x: float(X.norm(x)) ** 2)
    with pytest.raises(pg.errors.Unsupported):
        mu(pg.FiniteSet([0.0]))                                          # mu(A) is not evaluated: mass and integrals


def test_restriction_is_on_the_root():
    from pygeoinf0.measures.restrict import restrict
    mu = LebesgueMeasure("[0, 2]")
    r = mu.restricted(pg.Interval("[0, 1]"))
    assert isinstance(r, Measure) and r.mass()[0] == pytest.approx(1.0)
    assert restrict(DiracMeasure(1.5), pg.Interval("[0, 1]")) is None


# ------------------------------------------------------------------------------------------------ Dirac
def test_dirac_is_named_and_the_degenerate_gaussian_stays_one():
    X = pg.Euclidean(2)
    x = X([1.0, -2.0])
    d = Dirac(x)
    assert isinstance(d, GaussianMeasure) and isinstance(d, ProbabilityMeasure) and d.space is X
    u = X([0.3, 0.7])
    assert d.variance(u) == 0.0 and d.integrate(u)[0] == pytest.approx(float(X.inner(x, u)))
    assert np.array_equal(d.sample().values, x.values)
    assert np.array_equal(d.sample(3).values, np.tile(x.values, (3, 1)))
    G = pg.Matrix(np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 0.0]]), X, pg.Euclidean(3))
    Gd = G(d)                                                            # the degenerate Gaussian N(G x, 0): still a Dirac
    assert np.allclose(Gd.mean.values, G(x).values) and np.allclose(Gd.covariance.matrix(), 0.0)
    old = GaussianMeasure(X, 0.0 * pg.Identity(X), mean=x)               # the zero-covariance spelling stays valid
    assert old.variance(u) == 0.0 and np.array_equal(old.mean.values, x.values)
    assert Dirac(space=X).zero_mean                                       # delta_0 by its space
    assert any("Dirac" in c.statement for c in d.claims)


def test_the_cells_use_the_named_dirac():
    X, D = pg.Euclidean(2), pg.Euclidean(3)
    G = pg.Matrix(np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 0.0]]), X, D)
    x0 = X([1.0, 1.0])
    post = pg.update(x0, pg.Additive(G, D.standard_gaussian()), G(x0))
    assert isinstance(post, Dirac) and np.array_equal(post.mean.values, x0.values) and post.cell == "B4"
    from pygeoinf0.relations import as_kind
    K = as_kind(G, MEASURE)
    assert isinstance(K.noise, Dirac) and K.noise.space is D


# ------------------------------------------------------------------------------------------------ discrete laws
def test_discrete_measure_on_a_finite_set():
    mu = DiscreteMeasure({0.0: 0.25, 1.0: 0.75})
    assert mu(pg.FiniteSet([1.0])) == 0.75 and mu(pg.FiniteSet([0.0, 1.0])) == 1.0 and mu(pg.Interval("[2, 3]")) == 0
    assert mu.at(1.0) == 0.75 and mu.at(5.0) == 0.0
    assert dict(mu.items()) == {0.0: 0.25, 1.0: 0.75}
    assert mu.integrate(lambda y: y ** 2) == (0.75, 0.0)
    assert isinstance(mu.set, pg.FiniteSet) and len(mu.set) == 2
    with pytest.raises(ValueError, match="non-negative"):
        DiscreteMeasure({0.0: -0.5, 1.0: 1.5})
    with pytest.raises(ValueError, match="total mass"):
        DiscreteMeasure({0.0: 0.5, 1.0: 0.25})
    with pytest.raises(ValueError, match="not a point"):
        DiscreteMeasure({0.0: 0.5, 7.0: 0.5}, on=pg.FiniteSet([0.0, 1.0]))


def test_kernels_on_finite_sets_have_discrete_laws():
    from pygeoinf0.relations import MarkovKernel, Map, as_kind, compose
    X, Y, Z = pg.FiniteSet(range(3)), pg.FiniteSet(range(2)), pg.FiniteSet(range(2))
    P, Q = np.array([[0.2, 0.8], [0.5, 0.5], [1.0, 0.0]]), np.array([[0.9, 0.1], [0.3, 0.7]])
    K = MarkovKernel(X, Y, lambda a: DiscreteMeasure({float(b): P[int(a), j] for j, b in enumerate(Y.points)}, on=Y))
    L = MarkovKernel(Y, Z, lambda a: DiscreteMeasure({float(b): Q[int(a), j] for j, b in enumerate(Z.points)}, on=Z))
    LK = compose(L, K)
    for i, x in enumerate(X.points):
        law = LK(x)
        assert isinstance(law, DiscreteMeasure)
        assert [law.at(z) for z in Z.points] == pytest.approx((P @ Q)[i], abs=1e-15)
    f = Map(X, Y, lambda x: float(int(x) % 2), name="f")
    delta = as_kind(f, MEASURE)
    assert isinstance(delta(2.0), DiscreteMeasure) and delta(2.0).at(0.0) == 1.0
    bad = MarkovKernel(X, Y, lambda a: {0.0: 1.0})
    with pytest.raises(TypeError, match="DiscreteMeasure"):
        bad(0.0)


# ------------------------------------------------------------------------------------------------ cylindrical (Q11)
def test_the_cylindrical_gaussian_is_its_own_class():
    H = pg.Sobolev("[0, 1]", 1.0, length_scale=0.2)
    g = pg.probability.cylindrical(H, alpha=2.0)
    assert isinstance(g, CylindricalGaussian)
    assert not isinstance(g, Measure) and not isinstance(g, ProbabilityMeasure) and not isinstance(g, GaussianMeasure)
    assert pg.state_kind(g) is MEASURE
    with pytest.raises(pg.errors.NotDefined, match="cylindrical"):
        g.sample()
    D = pg.Euclidean(2)
    G = pg.Functionals(H.riesz([0.3, 0.7]), into=D)
    Gg = G(g)                                                            # a Gaussian measure on R^n
    assert isinstance(Gg, GaussianMeasure) and not isinstance(Gg, CylindricalGaussian)
    assert Gg.sample(5, seed=0).values.shape == (5, 2)
    post = g.condition(G, D([0.1, -0.2]), D.standard_gaussian())
    assert isinstance(post, CylindricalGaussian) and "cylindrical" in repr(post.claims)
    assert isinstance(pg.push(g, G), GaussianMeasure) and "cylindrical" in repr(pg.push(g, G).claims)
    v = g.variance(H.riesz(0.3))
    assert v == pytest.approx(float(H.riesz(0.3)(0.3)) / 2.0, rel=1e-10)    # Var u(x0) = k(x0, x0) / alpha


def test_cylindrical_constructions_stay_cylindrical_and_finite_dimension_gives_a_measure():
    H = pg.Sobolev("[0, 1]", 1.0, length_scale=0.2)
    g = pg.probability.cylindrical(H)
    assert isinstance(g + g, CylindricalGaussian)
    X = pg.DirectSum([H, pg.Euclidean(2)])
    assert isinstance(pg.probability.cylindrical(X), CylindricalGaussian)
    assert isinstance(GaussianMeasure.product([g, pg.Euclidean(2).standard_gaussian()]), CylindricalGaussian)
    E = pg.Euclidean(3)
    e = GaussianMeasure.cylindrical(E, 2.0)                               # N(0, I/2) on R^3: trace class, a measure
    assert isinstance(e, GaussianMeasure) and not isinstance(e, CylindricalGaussian)
    assert not any("not trace class" in c.statement for c in e.claims)
    assert e.sample(4, seed=1).values.shape == (4, 3)
    F = pg.DirectSum([pg.Euclidean(1), pg.Euclidean(2)])
    assert isinstance(F.standard_gaussian(), GaussianMeasure)                      # a finite direct sum: a measure too


# ------------------------------------------------------------------------------------------------ states by class
def test_measure_states_are_recognised_by_class():
    X = pg.Euclidean(2)
    for mu in (X.standard_gaussian(), Dirac(X([1.0, 0.0])), DiscreteMeasure({0.0: 1.0})):
        assert pg.state_kind(mu) is MEASURE
    with pytest.raises(TypeError, match="pg.Dirac"):
        pg.state_kind(DiracMeasure(0.3))                                 # a base measure of a domain is not a state

    class LooksGaussian:
        space, covariance = X, None

        def sample(self):
            return None
    with pytest.raises(TypeError):
        pg.state_kind(LooksGaussian())                                   # attributes do not make a measure
