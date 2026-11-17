"""C([a, b]): continuous functions with the sup norm; functionals are measures; inclusions into L^p and from H^s."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import NotContinuous, SpaceMismatch, SupportOutsideDomain, Unsupported


def test_construction_and_claims():
    X = pg.Continuous("[0, 1]")
    assert repr(X) == "C([0, 1])" and X == pg.Continuous(pg.Interval(0, 1))
    for bad in ("(0, 1]", "[0, 1)", "(0, 1)"):
        with pytest.raises(ValueError, match="not a normed space"):
            pg.Continuous(bad)
    text = str(X.claims)
    assert "Banach space" in text and "not a Hilbert space" in text and "Riesz-Markov" in text


def test_elements_are_functions_with_values():
    X = pg.Continuous("[0, 2]")
    f = X(np.sin)
    assert 0 < f.error < 1e-13 and abs(f(0.3) - math.sin(0.3)) <= f.error   # values to the estimated sup error
    assert np.abs(f(np.array([0.0, 1.0, 2.0])) - np.sin([0.0, 1.0, 2.0])).max() <= 2 * f.error
    assert "a function: f(x) is its value" in f.info() and "equivalence class" not in f.info()
    with pytest.raises(ValueError, match="determined by its values"):
        f.redefine_at([0.5], [3.0])
    g = np.exp(-f ** 2) + 2 * X.x                       # arithmetic and composition stay in C
    assert g.space == X and abs(g(1.0) - math.exp(-math.sin(1.0) ** 2) - 2) <= g.error + 1e-15


def test_discontinuous_functions_are_refused():
    with pytest.raises(NotContinuous, match=r"x = 0\.4"):
        pg.Continuous("[0, 1]")(lambda x: (x > 0.4).astype(float))
    with pytest.raises(NotContinuous, match="x = 0"):
        pg.Continuous("[-1, 1]")(np.sign)
    assert pg.L2("[0, 1]")(lambda x: (x > 0.4).astype(float)).norm() == pytest.approx(math.sqrt(0.6), rel=1e-14)
    steep = pg.Continuous("[0, 1]")(lambda x: np.tanh((x - 0.5) / 1e-3))     # steep but continuous: accepted
    assert abs(steep.norm() - math.tanh(500)) <= 2 * steep.error


@pytest.mark.parametrize("f, interval, want", [
    (lambda x: np.sin(2 * np.pi * x), "[0, 1]", 1.0),
    (lambda x: x * (1 - x), "[0, 1]", 0.25),                                 # interior maximum
    (np.exp, "[0, 2]", math.exp(2)),                                         # at an end
    (lambda x: np.abs(x - 0.3), "[0, 1]", 0.7),                              # kink inside, maximum at an end
    (lambda x: np.cos(20 * np.arccos(2 * x - 1)), "[0, 1]", 1.0),            # Chebyshev T_20: 21 equal maxima
    (lambda x: 3 * np.exp(-((x - 0.3) / 0.01) ** 2) - 1, "[0, 1]", 2.0),     # narrow bump
    (lambda x: (x - 0.5) ** 2 * np.cos(40 * x) + 0.1 * x, "[0, 1]", None),   # checked against a fine grid
])
def test_sup_norm_is_the_maximum(f, interval, want):
    X = pg.Continuous(interval)
    u = X(f)
    if want is None:
        I = X.domain
        x = np.linspace(I.a, I.b, 2_000_001)
        want = np.abs(f(x)).max()
        assert u.norm() >= want * (1 - 1e-12) and u.norm() <= want * (1 + 1e-9)
    else:
        assert abs(u.norm() - want) <= 2 * u.error + 1e-15 * want        # |max |p| - max |f|| <= ||p - f||
    assert u.error <= 1e-6 * max(1.0, u.norm())


def test_sup_errors_follow_arithmetic():
    X = pg.Continuous("[0, 1]")
    f, g = X(np.exp), X(np.cos)
    assert 0 < f.error <= 1e-13 and (f + 2 * g).error <= f.error + 2 * g.error + 1e-30
    rough = X(lambda x: np.abs(x - 1 / 3) ** 0.5)                           # a cusp: slow convergence, estimated error
    assert rough.error < 1e-6 and rough.norm() == pytest.approx((2 / 3) ** 0.5, rel=1e-6)


def test_no_inner_product():
    X = pg.Continuous("[0, 1]")
    one, x = X(1.0), X.x
    lhs, rhs = (one + x).norm() ** 2 + (one - x).norm() ** 2, 2 * one.norm() ** 2 + 2 * x.norm() ** 2
    assert lhs == pytest.approx(5.0) and rhs == pytest.approx(4.0)                # the parallelogram law fails
    assert not hasattr(X, "inner") and not hasattr(X, "riesz")        # not defined, so not there (plan 068 P1)
    assert "Riesz-Markov" in str(X.dual().claims)
    with pytest.raises(Unsupported, match="Wiener"):
        X.random()


def test_functionals_are_measures():
    X = pg.Continuous("[0, 1]")
    f = X(lambda x: x ** 2)
    ev = X.evaluation(0.3)
    assert ev(f) == pytest.approx(0.09, abs=1e-14) and ev.norm() == 1.0
    assert X.average("[0.2, 0.6]")(f) == pytest.approx((0.6 ** 3 - 0.2 ** 3) / 3 / 0.4, rel=1e-14)
    w = pg.CountingMeasure([0.1, 0.5], weights=[2.0, 3.0])
    dens = pg.LebesgueMeasure(pg.Interval(0, 1)).with_density(lambda x: 1 + x)
    G = pg.Integrals(X, [0.2, "[0.1, 0.5]", w, dens])
    want = [0.04, (0.125 - 0.001) / 3, 2 * 0.01 + 3 * 0.25, 1 / 3 + 1 / 4]
    assert np.allclose(G(f).values, want, rtol=1e-13, atol=1e-14)
    assert np.allclose(G.norms(), [1.0, 0.4, 5.0, 1.5], rtol=1e-12)
    F = X.family(lambda x, a: x ** a, a=[1.0, 2.0])
    assert G(F).values.shape == (2, 4) and np.allclose(G(F).values[1], want, rtol=1e-13, atol=1e-14)
    assert G[0](f) == pytest.approx(0.04, abs=1e-14)
    assert np.allclose(pg.Integrals(X, [0.25, "[0, 1]"])(f).values, [0.0625, 1 / 3], rtol=1e-13)   # numbers and strings
    with pytest.raises(TypeError, match="dual"):
        G.adjoint
    with pytest.raises(SupportOutsideDomain):
        X.evaluation(1.5)(f)
    with pytest.raises(TypeError, match="continuous"):
        pg.Integrals(pg.L2("[0, 1]"), [0.2])


def test_inclusions():
    X, L = pg.Continuous("[0, 1]"), pg.L2("[0, 1]")
    f = X(lambda x: np.sin(3 * x))
    g = pg.Inclusion(X, L)(f)
    assert g.space == L and g(0.4) == pytest.approx(f(0.4), abs=1e-14)
    assert g.norm() <= f.norm() and g.norm() == pytest.approx(math.sqrt(0.5 - math.sin(6) / 12), rel=1e-13)
    assert pg.Inclusion(X, pg.Lp("[0, 1]", 1))(f).norm() == pytest.approx((1 - math.cos(3)) / 3, rel=1e-12)
    with pytest.raises(SpaceMismatch, match="no continuous member"):
        pg.Inclusion(L, X)
    H = pg.Sobolev("[0, 1]", 1, length_scale=0.2)
    k = H.riesz(0.3)                                     # the representer of evaluation at 0.3: a kink at 0.3
    i = pg.Inclusion(H, X)
    u = i(k)
    assert u.space == X and u(0.3) == pytest.approx(k(0.3), rel=1e-13) and u.error < 1e-12  # breaks at the kink
    bound = H.backend.kernel.k0 ** 0.5
    assert u.norm() <= bound * k.norm() * (1 + 1e-10) and "Sobolev embedding" in str(i.claims)
    with pytest.raises(SpaceMismatch, match="s > d/2"):
        pg.Inclusion(pg.Sobolev("[0, 1]", 0.4, length_scale=0.2), X)
