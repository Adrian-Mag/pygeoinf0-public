"""The reconciliation of plan 009 P2 (density functionals, against=, measures/densities.py) with EXP-031 (weighted measures,
spaces/dual.py, formerly spaces/boundedness.py): one decision and one refusal path, DensityMeasure with P2's density_against interface, pg.Tabulated
as an exact weight. References: scipy.integrate.quad and closed forms."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

import pygeoinf0 as pg
from pygeoinf0.spaces.dual import bounded, l2_density, require_bounded
from pygeoinf0.measures.densities import Density


def q(f, a, b, points=()):
    return quad(f, a, b, points=[p for p in points if a < p < b] or None, epsabs=1e-15, epsrel=1e-15, limit=500)[0]


def radial_space(b=2.0):
    I = pg.Interval(0.0, b)
    return pg.L2(pg.LebesgueMeasure(I).with_density(lambda r: r ** 2, name="r^2")), pg.LebesgueMeasure(I)


class Power:
    """r^p with its exact order at 0 (as a Tabulated gives orders 0, 1, inf)."""

    def __init__(self, p):
        self.p = p

    def __call__(self, r):
        return np.asarray(r, float) ** self.p

    def order_at(self, x0):
        return self.p if x0 == 0 else 0.0


def test_tabulated_is_an_exact_weight():
    I = pg.Interval(0.0, 2.0)
    k = pg.Tabulated(I, [0.0, 0.5, 1.0, 2.0], [0.0, 2.0, 1.0, 3.0])
    mu = pg.LebesgueMeasure(I).with_density(k)
    wt = mu.weight
    assert wt.kind == "tabulated" and wt.exact and not mu.signed
    assert wt.sign()[1] == "theorem" and wt.bound()[0] == 3.0
    assert mu.mass()[0] == pytest.approx(k.integral(), rel=1e-14)
    assert mu(pg.Interval(0.25, 1.5)) == pytest.approx(k.integral(over=(0.25, 1.5)), rel=1e-13)
    s = pg.LebesgueMeasure(I).with_density(pg.Tabulated(I, [0.0, 2.0], [-1.0, 1.0]))
    assert s.signed and s.weight.sign()[0] == "signed"
    with pytest.raises(ValueError):
        pg.LebesgueMeasure(pg.Interval(0.0, 3.0)).with_density(k)        # beyond the table's domain


def test_density_measures_offer_p2s_interface_on_every_set():
    M, dr = radial_space()
    k = pg.Tabulated("[0, 2]", [0.0, 1.0, 2.0], [0.0, 1.0, 0.5])
    nu = dr.with_density(k)
    rho = nu.density_against(M.measure)
    assert isinstance(rho, Density) and rho.order(0.0) == -1.0           # r / r^2, exact (order_at of the table)
    assert rho(np.array([0.5]))[0] == pytest.approx(0.5 / 0.25)
    box = pg.LebesgueMeasure(pg.Box((0.2, 0.6), (0.2, 0.6))).with_density(lambda x, y: 1 + x)
    r2 = box.density_against(pg.LebesgueMeasure(pg.Box((0, 1), (0, 1))))
    assert r2.weight is box.weight and r2(np.array([[0.3, 0.4]]))[0] == pytest.approx(1.3)
    with pytest.raises(pg.errors.UnboundedFunctional):
        pg.LebesgueMeasure(pg.Segment((0, 0), (1, 1))).density_against(pg.LebesgueMeasure(pg.Box((0, 1), (0, 1))))


@pytest.mark.parametrize("p", [0.4, 0.5, 0.6, 1.0, 1.5])
def test_one_rule_for_measures_and_for_density_kernels(p):
    """bounded(k dr, L2(r^2 dr)) and l2_density(dr, r^2 dr, [k]) decide the same, k ~ r^p: bounded iff p > 1/2."""
    M, dr = radial_space()
    mu = dr.with_density(lambda r: r ** p, singular_at=[0.0], exponent=-p)
    a = bounded(mu, M).bounded
    try:
        l2_density(dr, M.measure, [Power(p)])
        b = True
    except pg.Refused:
        b = False
    assert a is pg.Truth.of(b) and b == (p > 0.5)


def test_against_a_density_measure_given_by_a_table():
    M, dr = radial_space()
    w = pg.Tabulated("[0, 2]", [0.0, 2.0], [1.0, 3.0])                   # nu = (1 + r) dr
    nu = dr.with_density(w)
    kern = pg.Tabulated("[0, 2]", [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])      # vanishes linearly at 0: p = 1 > 1/2
    with pytest.raises(pg.errors.Unsupported):                            # bounded, singular representer on an
        pg.Functionals(M, [kern], against=nu)                             # interval space (P2's rule, unchanged)
    G = pg.Functionals(M, [lambda r: np.cos(r)], against=nu, taper=lambda r: r * r)
    f = M(lambda r: np.exp(-r))
    ref = q(lambda r: math.cos(r) * r * r * math.exp(-r) * (1 + r), 0, 2)
    assert G(f).values[0] == pytest.approx(ref, rel=1e-11)


def test_a_measure_away_from_the_zero_of_the_space_density_is_accepted():
    """Bug fix: the orders of the space's density were checked at points outside the support of nu, so a functional
    on [1, 2] of L2(r^2 dr) on [0, 2] was refused for the zero of r^2 at 0."""
    M, _ = radial_space()
    nu = pg.LebesgueMeasure(pg.Interval(1.0, 2.0))
    G = pg.Functionals(M, [lambda r: np.ones_like(r)], against=nu)
    f = M(lambda r: np.sin(r))
    assert G(f).values[0] == pytest.approx(math.cos(1.0) - math.cos(2.0), rel=1e-12)
    assert bounded(nu, M).bounded is pg.Truth.TRUE
    require_bounded(pg.LebesgueMeasure(pg.Interval(0.0, 1.0)).with_density(lambda r: r * r), M)


def test_refusals_come_from_one_place():
    M, dr = radial_space()
    with pytest.raises(pg.Refused, match="not in L"):
        require_bounded(dr, M)                                            # dr on L2(r^2 dr): 1/r^2 not in L2
    with pytest.raises(pg.Refused, match="not in L2"):
        l2_density(dr, M.measure, [lambda r: np.ones_like(r)])
