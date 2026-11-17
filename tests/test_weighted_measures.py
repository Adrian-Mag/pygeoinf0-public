"""Weighted measures w dsigma (EXP-031 phase W1): construction on every kind of set, the boundedness decision against
Propositions 2-5 of the design, symbolic L2 representers against scipy.integrate, densities on subintervals of C(I)
and averages."""
import math
import warnings

import numpy as np
import pytest
import scipy.integrate as si

import pygeoinf0 as pg
from pygeoinf0.spaces.dual import bounded, require_bounded
from pygeoinf0.measures.densities import Density
from pygeoinf0.measures import measure_dim, total_mass, total_variation


SETS = [(pg.Interval(0.2, 0.7), 0.5, 1), (pg.Box((0.1, 0.5), (0.2, 0.9)), 0.28, 2),
        (pg.Disk(0.3, (0.1, 0.2)), math.pi * 0.09, 2), (pg.Ball(0.5), 4 / 3 * math.pi * 0.125, 3),
        (pg.Cylinder(0.4, (0, 0.5)), math.pi * 0.16 * 0.5, 3), (pg.Shell(0.3, 0.6), 4 / 3 * math.pi * (0.216 - 0.027), 3),
        (pg.Sphere(0.7), 4 * math.pi * 0.49, 2), (pg.Circle(0.4), 2 * math.pi * 0.4, 1),
        (pg.Segment((0, 0), (0.3, 0.4)), 0.5, 1), (pg.Patch((0, 0, 0), (1, 0, 0), (0, 2, 0)), 2.0, 2),
        (pg.Polyline([[0, 0], [0.3, 0.4], [0.3, 1.4]]), 1.5, 1), (pg.Arc(pg.Sphere(1.0), (1, 0, 0), (0, 1, 0)), math.pi / 2, 1)]


@pytest.mark.parametrize("S, vol, k", SETS, ids=[repr(s[0])[:24] for s in SETS])
def test_construction_on_every_kind_of_set(S, vol, k):
    mu = pg.LebesgueMeasure(S).with_density(2.5) if not isinstance(S, pg.Polyline) else \
        pg.DensityMeasure(pg.LebesgueMeasure(S), 2.5)
    assert isinstance(mu, pg.DensityMeasure) and measure_dim(mu) == k
    assert mu.mass() == (pytest.approx(2.5 * vol, rel=1e-13), 0.0)
    assert total_mass(mu.normalised()) == pytest.approx(1.0, rel=1e-13)
    nu = pg.DensityMeasure(pg.LebesgueMeasure(S), lambda *x: 1.0 + 0 * x[0])     # a callable: by quadrature
    m, err = nu.mass()
    assert m == pytest.approx(vol, rel=1e-12) and err < 1e-10
    assert nu.claims.find("measure").statement == "a measure"


def test_callable_weight_mass_against_a_closed_form():
    D = pg.Disk(0.5, (0.2, -0.1))
    mu = pg.LebesgueMeasure(D).with_density(lambda x, y: (x - 0.2) ** 2 + (y + 0.1) ** 2)   # int r^2 r dr dphi
    m, err = mu.mass()
    assert m == pytest.approx(2 * math.pi * 0.5 ** 4 / 4, rel=1e-13) and err < 1e-12
    S = pg.Sphere(2.0)
    nu = pg.LebesgueMeasure(S).with_density(lambda x, y, z: z ** 2)            # int z^2 dA = 4 pi R^4 / 3
    assert nu.mass()[0] == pytest.approx(4 * math.pi * 16 / 3, rel=1e-12)


def test_signed_weights_are_finite_signed_measures():
    mu = pg.LebesgueMeasure(pg.Box((0, 1), (0, 1))).with_density(lambda x, y: x - 0.25)
    assert mu.signed and mu.claims.find("measure").statement == "a finite signed measure"
    assert mu.mass()[0] == pytest.approx(0.25, rel=1e-13)
    assert total_variation(mu) == pytest.approx(0.25 ** 2 / 2 + 0.75 ** 2 / 2, rel=1e-3)   # a kink: algebraic
    z = pg.LebesgueMeasure(pg.Interval(0, 1)).with_density(lambda x: x - 0.5)
    with pytest.raises(pg.Refused):                                           # no normalisation of zero mass
        z.normalised()
    with pytest.raises(ValueError, match="takes negative values"):            # not a space's measure
        pg.L2(pg.LebesgueMeasure(pg.Interval(0, 1)).with_density(lambda x: x - 0.5))


def test_domains_no_longer_raise_attribute_errors_and_space_densities_are_stated():
    mu = pg.LebesgueMeasure(pg.Disk(0.15, (0.3, 0.65))).with_density(lambda x, y: np.exp(-x))
    assert mu.density is None and mu.claims.find("weight kind").statement == "weight: callable"
    with pytest.raises(pg.errors.Unsupported, match="plan 064"):     # a general density (plan 064's refused stage 3)
        pg.L2(pg.LebesgueMeasure(pg.Disk(1.0)).with_density(lambda x, y: 1 + x * x))
    assert "2 dx" in repr(pg.L2(pg.LebesgueMeasure(pg.Disk(1.0)).with_density(2.0)))   # constants: plan 064 stage 1


def test_piece_density_is_a_density_measure_with_its_refusal():
    P = pg.Polyline([[0, 0], [0.3, 0.4], [0.3, 1.4]])
    mu = pg.LebesgueMeasure(P).with_density([1.0, 3.0])
    assert isinstance(mu, pg.PieceDensity) and isinstance(mu, pg.DensityMeasure)
    assert mu(mu.set) == pytest.approx(0.5 + 3.0) and mu.weight.kind == "pieces"
    with pytest.raises(ValueError):
        pg.LebesgueMeasure(P).with_density([1.0, -3.0])
    nu = pg.DensityMeasure(pg.LebesgueMeasure(P), [1.0, -3.0])               # signed piece values: the general class
    assert nu.signed and nu.mass()[0] == pytest.approx(0.5 - 3.0)
    H = pg.Sobolev(pg.Box((-1, 1), (-1, 2)), 1.5, length_scale=0.3)
    a, b = H.riesz(nu), H.riesz(pg.LebesgueMeasure(P).with_density([1.0, 3.0]))
    f = H(lambda x, y: np.cos(x) + y)
    assert H.inner(a, f) == pytest.approx(H.inner(b, f) - 6.0 * H.inner(H.riesz(pg.Segment((0.3, 0.4), (0.3, 1.4))), f),
                                          rel=1e-12)


def test_trace_thresholds_on_sobolev_spaces():
    for d, D in ((2, pg.Box((0, 1), (0, 1))), (3, pg.Box((0, 1), (0, 1), (0, 1)))):
        sets = {0: pg.DiracMeasure((0.5,) * d), 1: pg.Segment((0.2,) * d, (0.6,) * d),
                d: pg.LebesgueMeasure(pg.Box(*[(0.2, 0.6)] * d))}
        if d == 3:
            sets[2] = pg.Patch((0.2, 0.2, 0.5), (0.3, 0, 0), (0, 0.3, 0))
        for k, mu in sets.items():
            for s in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75):
                H = pg.Sobolev(D, s, length_scale=0.2)
                want = s - (d - k) / 2 > 0
                assert bounded(pg.measures.as_measure(mu), H).bounded is pg.Truth.of(want), (d, k, s)
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), 1.0, length_scale=0.2)            # s = d/2: points unbounded
    with pytest.raises(pg.PointEvaluationUndefined) as e:
        H.riesz((0.5, 0.5))
    assert isinstance(e.value, pg.Refused) and "it needs s > (d - k)/2" in e.value.reason


def test_declared_singular_weights_against_proposition_3c():
    D = pg.Box((0, 1), (0, 1))
    seg = pg.LebesgueMeasure(pg.Segment((0.1, 0.5), (0.9, 0.5)))
    for s, amax in ((0.75, 0.75), (1.0, 1.0), (1.5, 1.0)):                   # a < k - max(0, d/2 - s), k = 1, d = 2
        H = pg.Sobolev(D, s, length_scale=0.2)
        for a in (amax - 0.05, amax + 0.05):
            mu = seg.with_density(lambda x, y: np.abs(x - 0.5) ** -a + 0 * y, singular_at=[(0.5, 0.5)], exponent=a)
            assert bounded(mu, H).bounded is pg.Truth.of(a < amax), (s, a)
    H = pg.Sobolev(D, 0.75, length_scale=0.2)
    mu = seg.with_density(lambda x, y: np.abs(x - 0.5) ** -0.8 + 0 * y, singular_at=[(0.5, 0.5)], exponent=0.8)
    with pytest.raises(pg.Refused, match="Proposition 3"):
        require_bounded(mu, H)


def test_l2_rule_and_the_g13_condition():
    rho = pg.LebesgueMeasure(pg.Interval(0, 1)).with_density(lambda r: r ** 2, singular_at=[0.0], exponent=-2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M = pg.L2(rho)
    for q, ok in ((0.6, True), (0.5, False), (0.4, False), (1.0, True)):    # k ~ r^q admissible iff q > 1/2
        mu = pg.LebesgueMeasure(pg.Interval(0, 1)).with_density(lambda r, q=q: r ** q, singular_at=[0.0], exponent=-q)
        assert bounded(mu, M).bounded is pg.Truth.of(ok), q
    with pytest.raises(pg.Refused, match="not in L"):
        M.riesz(pg.LebesgueMeasure(pg.Interval(0, 1)).with_density(lambda r: r ** 0.4, singular_at=[0.0],
                                                                    exponent=-0.4))
    mu = pg.LebesgueMeasure(pg.Interval(0, 1)).with_density(lambda r: r ** 0.75, singular_at=[0.0], exponent=-0.75)
    k = M.riesz(mu)                                   # k_mu = r^0.75 / r^2; ||k||^2 = int r^1.5 / r^2 dr = 2
    assert k.norm() ** 2 == pytest.approx(2.0, rel=1e-10)
    rho = mu.density_against(M.measure)              # plan 009's interface: a Density with orders, and the Weight
    assert isinstance(rho, Density) and isinstance(rho.weight, pg.measures.weights.QuotientWeight)
    assert rho.order(0.0) == pytest.approx(0.75 - 2.0) and rho(np.array([0.5]))[0] == pytest.approx(0.5 ** -1.25)
    with pytest.raises(pg.Refused, match="vanishes"):                        # undeclared vanishing density refused
        L = pg.L2(pg.LebesgueMeasure(pg.Interval(0, 1)).with_density(lambda r: r ** 2))
        L.riesz(pg.LebesgueMeasure(pg.Interval(0, 0.5)))


def test_volume_measures_are_bounded_everywhere():                         # Proposition 5
    D = pg.Box((0, 1), (0, 1))
    mu = pg.LebesgueMeasure(pg.Disk(0.2, (0.5, 0.5))).with_density(lambda x, y: 1 + x)
    for s in (0.1, 0.5, 1.0, 2.0):
        assert bounded(mu, pg.Sobolev(D, s, length_scale=0.1)).bounded is pg.Truth.TRUE
    assert bounded(mu, pg.L2(D)).bounded is pg.Truth.TRUE
    with pytest.raises(pg.errors.UnboundedFunctional):
        pg.L2(D).riesz(pg.Circle(0.2, (0.5, 0.5)))


def test_l2_pairings_against_scipy():
    D = pg.Box((0, 1), (0, 1))
    M = pg.L2(D)
    sub, disk = pg.Box((0.1, 0.6), (0.3, 0.9)), pg.Disk(0.15, (0.3, 0.65))
    w = lambda x, y: 1 + x * y                                               # noqa: E731
    mu = pg.LebesgueMeasure(sub).with_density(w)
    nu = pg.LebesgueMeasure(disk).with_density(lambda x, y: np.cos(x - y))
    f = M(lambda x, y: np.exp(x) * y)
    ref = si.dblquad(lambda y, x: w(x, y) * np.exp(x) * y, 0.1, 0.6, 0.3, 0.9, epsabs=1e-14)[0]
    k = M.riesz(mu)
    assert M.inner(k, f) == pytest.approx(ref, rel=1e-12)
    assert k.norm() ** 2 == pytest.approx(si.dblquad(lambda y, x: w(x, y) ** 2, 0.1, 0.6, 0.3, 0.9)[0], rel=1e-12)
    h = M.riesz(nu)
    ref = si.dblquad(lambda r, t: np.cos(0.3 + r * np.cos(t) - 0.65 - r * np.sin(t))
                     * np.exp(0.3 + r * np.cos(t)) * (0.65 + r * np.sin(t)) * r, 0, 2 * np.pi, 0, 0.15, epsabs=1e-14)[0]
    assert M.inner(h, f) == pytest.approx(ref, rel=1e-12)
    ref = si.dblquad(lambda r, t: np.cos(0.3 + r * np.cos(t) - 0.65 - r * np.sin(t))
                     * w(0.3 + r * np.cos(t), 0.65 + r * np.sin(t)) * r, 0, 2 * np.pi, 0, 0.15, epsabs=1e-14)[0]
    assert M.inner(k, h) == pytest.approx(ref, rel=1e-12)                    # the disk lies inside the sub-box
    G = pg.Functionals(M.riesz([mu, nu]))
    assert np.allclose(G(f).values, [M.inner(k, f), M.inner(h, f)], rtol=1e-14)
    A = (G @ G.adjoint).A if hasattr(G @ G.adjoint, "A") else None
    assert A is not None and A[0, 1] == pytest.approx(M.inner(k, h), rel=1e-14)
    g = G.adjoint(pg.Euclidean(2)([1.0, -2.0]))
    assert g(0.3, 0.65) == pytest.approx(w(0.3, 0.65) - 2 * np.cos(-0.35), rel=1e-14)
    with pytest.raises(pg.Refused, match="not representable"):               # a box and a disk that cross
        M.inner(M.riesz(pg.Box((0.3, 0.6), (0.3, 0.7))), h)
    with pytest.raises(pg.Refused, match="kept symbolic"):                   # no fit across the boundary of a disk
        h.realisation


def test_l2_representers_on_coordinate_boxes_realise_exactly():
    D = pg.Box((0, 1), (0, 1))
    M = pg.L2(D)
    k = M.riesz(pg.LebesgueMeasure(pg.Box((0.2, 0.6), (0.3, 0.7))).with_density(lambda x, y: 1 + x))
    u = k + M(lambda x, y: 0 * x)                                            # needs the realisation: breaks at faces
    assert u.norm() ** 2 == pytest.approx(k.norm() ** 2, rel=1e-12)


def test_interval_densities_in_weighted_l2():
    rho = pg.LebesgueMeasure(pg.Interval(0, 1)).with_density(lambda x: 1 + x ** 2)
    M = pg.L2(rho)
    mu = pg.LebesgueMeasure(pg.Interval(0.2, 0.5)).with_density(lambda x: np.sin(3 * x))
    k = M.riesz(mu)                                                         # sin(3x) 1_J / rho
    f = M(np.exp)
    assert M.inner(k, f) == pytest.approx(si.quad(lambda x: np.sin(3 * x) * np.exp(x), 0.2, 0.5, epsabs=1e-15)[0],
                                          rel=1e-13)
    assert k.norm() ** 2 == pytest.approx(si.quad(lambda x: np.sin(3 * x) ** 2 / (1 + x * x), 0.2, 0.5,
                                                  epsabs=1e-15)[0], rel=1e-13)
    assert k(0.3) == pytest.approx(np.sin(0.9) / 1.09) and k(0.6) == 0


def test_densities_on_subintervals_of_c():
    X = pg.Continuous("[0, 1]")
    mu = pg.LebesgueMeasure(pg.Interval(0.2, 0.5)).with_density(lambda x: np.cos(5 * x))
    G = pg.Integrals(X, [mu])
    assert G(X(np.exp)).values[0] == pytest.approx(si.quad(lambda x: np.exp(x) * np.cos(5 * x), 0.2, 0.5,
                                                           epsabs=1e-15)[0], rel=1e-13)
    ref = si.quad(lambda x: abs(np.cos(5 * x)), 0.2, 0.5, points=[np.pi / 10], epsabs=1e-15)[0]
    assert G.norms()[0] == pytest.approx(ref, rel=1e-10) and X.functional(mu).norm() == pytest.approx(ref, rel=1e-10)
    sing = pg.LebesgueMeasure(pg.Interval(0.0, 0.5)).with_density(lambda x: x ** -0.5, singular_at=[0.0],
                                                                  exponent=0.5)
    assert pg.Integrals(X, [sing])(X(lambda x: 1 + 0 * x)).values[0] == pytest.approx(2 * math.sqrt(0.5), rel=1e-10)
    bad = pg.LebesgueMeasure(pg.Interval(0.0, 0.5)).with_density(lambda x: x ** -1.0, singular_at=[0.0], exponent=1.0)
    with pytest.raises(pg.Refused, match="not integrable"):
        pg.Integrals(X, [bad])(X(np.exp))


def test_averages_on_l2_and_sobolev():
    D = pg.Box((0, 1), (0, 1))
    M = pg.L2(D)
    f = M(lambda x, y: x + y ** 2)
    sub = pg.Box((0.2, 0.6), (0.3, 0.7))
    want = (0.4 * (0.36 - 0.04) / 2 + 0.4 * (0.343 - 0.027) / 3) / 0.16
    assert M.average(sub)(f) == pytest.approx(want, rel=1e-13)
    wavg = M.average(sub, weight=lambda x, y: x)
    ref = si.dblquad(lambda y, x: x * (x + y * y), 0.2, 0.6, 0.3, 0.7)[0] / si.dblquad(lambda y, x: x, 0.2, 0.6, 0.3,
                                                                                     0.7)[0]
    assert wavg(f) == pytest.approx(ref, rel=1e-12)
    H = pg.Sobolev(D, 1.5, length_scale=0.2)
    seg = pg.Segment((0.1, 0.2), (0.7, 0.9))
    T = H.average(seg)
    assert T.norm() == pytest.approx(H.riesz(seg).norm() / seg.volume, rel=1e-13)
    I = pg.L2("[0, 1]")
    assert I.average("[0.2, 0.4]")(I(lambda x: x)) == pytest.approx(0.3, rel=1e-14)
