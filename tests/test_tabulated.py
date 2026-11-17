"""Plan 009 P2, G10 and G13: tabulated functions read by declared rules and integrated exactly; density functionals,
their Radon-Nikodym conversion and the boundedness check at singular ends (EXP-030 DESIGN G10, G13; NOTES.md section 2).
References: scipy.integrate.quad, np.interp per segment (the thesis code's own reading), explicit formulas."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

import pygeoinf0 as pg


def q(f, a, b, points=()):
    return quad(f, a, b, points=[p for p in points if a < p < b] or None, epsabs=1e-15, epsrel=1e-15, limit=500)[0]


def test_linear_rule_with_segments_and_flat_extrapolation_is_the_thesis_reading():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.5, 9.5, 25))
    v = rng.normal(size=25)
    seg = [3.0, 6.5]
    k = pg.Tabulated("[0, 10]", x, v, rule="linear", segments=seg, extrapolation="flat")
    edges = [0.0, 3.0, 6.5, 10.0]
    t = np.linspace(0, 10, 2001)
    ref = np.empty_like(t)
    for i in range(3):                                   # sola_forward_operator.py's segment-wise np.interp
        lo, hi = edges[i], edges[i + 1]
        qm = (t >= (-np.inf if i == 0 else lo)) & ((t <= hi) if i == 2 else (t < hi))
        dm = (x >= lo) & ((x <= hi) if i == 2 else (x < hi))
        ref[qm] = np.interp(t[qm], x[dm], v[dm], left=v[dm][0], right=v[dm][-1])
    assert np.allclose(k(t), ref, rtol=0, atol=1e-14)
    assert k.integral() == pytest.approx(q(k, 0, 10, list(k.breaks)), rel=1e-13, abs=1e-14)


def test_constant_and_midpoint_rules():
    edges = np.array([0.0, 1.0, 1.0, 2.5, 4.0])          # a zero-width cell (a marker) with value 0
    vals = np.array([2.0, 0.0, -1.0, 3.0])
    c = pg.Tabulated("[0, 4]", edges, vals, rule="constant")
    for i in range(4):
        if edges[i + 1] > edges[i]:
            assert c.integral(over=(edges[i], edges[i + 1])) == pytest.approx(vals[i] * (edges[i + 1] - edges[i]))
    m = pg.Tabulated("[0, 4]", edges, vals, rule="midpoint-linear")
    mids = np.array([0.5, 1.75, 3.25])
    assert np.allclose(m(mids), [2.0, -1.0, 3.0])
    assert m(0.1) == 2.0 and m(3.9) == 3.0                # flat beyond the first and last midpoint
    assert m.integral() == pytest.approx(q(m, 0, 4, [0.5, 1.75, 3.25]), rel=1e-14)
    with pytest.raises(ValueError):
        pg.Tabulated("[0, 4]", edges, [2.0, 1.0, -1.0, 3.0], rule="midpoint-linear")   # zero-width cell, nonzero value


def test_exact_integrals_against_polynomial_weights():
    from pygeoinf0.realisations.series.radial_harmonic import r2_density
    k = pg.Tabulated("[0, 3]", [0.2, 1.0, 1.7, 2.9], [1.0, -2.0, 0.5, 4.0], extrapolation="linear")
    w = r2_density(0.0, 3.0)
    assert k.integral(weight=w) == pytest.approx(q(lambda r: k(r) * r * r, 0, 3, list(k.breaks)), rel=1e-13)
    g = pg.Tabulated("[0, 3]", [0.0, 1.5, 3.0], [1.0, 0.0, 2.0])
    assert k.inner(g, w) == pytest.approx(q(lambda r: k(r) * g(r) * r * r, 0, 3, list(k.breaks) + [1.5]), rel=1e-13)
    z = pg.Tabulated("[0, 3]", [1.0, 2.0], [1.0, 2.0], extrapolation="zero")
    assert z(0.5) == 0.0 and z(2.5) == 0.0


def test_trapezoid_realisation_states_its_error():
    k = pg.Tabulated("[0, 1]", [0.0, 0.3337, 1.0], [0.0, 1.0, 0.0])
    val, claim = k.integral(quadrature=pg.realisations.Trapezoid(64))
    assert claim.status == "checked" and "trapezoid" in claim.statement
    assert abs(val - k.integral()) > 0 and f"{abs(val - k.integral()):.3g}" in claim.statement


def test_orders_at_points():
    k = pg.Tabulated("[0, 2]", [0.0, 1.0, 2.0], [0.0, 1.0, 1.0])
    assert k.order_at(0.0) == 1.0 and k.order_at(1.0) == 0.0
    z = pg.Tabulated("[0, 2]", [0.0, 1.0, 2.0], [0.0, 0.0, 1.0])
    assert z.order_at(0.0) == math.inf


# ---------------------------------------------------------------------------------------------- G13
def radial_space(b=2.0):
    I = pg.Interval(0.0, b)
    return pg.L2(pg.LebesgueMeasure(I).with_density(lambda r: r ** 2, name="r^2")), pg.LebesgueMeasure(I)


def test_refusal_of_a_kernel_that_does_not_vanish_at_the_centre():
    M, dr = radial_space()
    k = pg.Tabulated("[0, 2]", [0.0, 0.5, 1.0, 2.0], [1.0, 2.0, 0.0, 1.0])
    with pytest.raises(pg.Refused) as e:
        pg.Functionals(M, [k], against=dr)
    assert "taper" in " ".join(e.value.needs)
    with pytest.raises(pg.Refused):
        pg.Functionals(M, [lambda r: np.sqrt(r)], against=dr)       # p = 1/2: the borderline diverges


def test_acceptance_with_a_taper_and_the_representer_identity():
    M, dr = radial_space()
    k = pg.Tabulated("[0, 2]", [0.0, 0.5, 1.0, 2.0], [1.0, 2.0, 0.0, 1.0])
    eps = 2e-3
    taper = lambda r: r ** 2 / (r ** 2 + eps ** 2)                  # noqa: E731
    G = pg.Functionals(M, [k], against=dr, taper=taper)
    f = M(lambda r: np.cos(r) + r)
    ref = q(lambda r: k(r) * taper(r) * (math.cos(r) + r), 0, 2, [eps, 0.5, 1.0])
    assert G(f).values[0] == pytest.approx(ref, rel=1e-12)
    assert any(c.status == "asserted" and "taper" in c.statement for c in G.claims)
    assert any(c.status in ("derived", "checked") and "bounded near 0" in c.statement for c in G.claims)


def test_kernels_that_vanish_fast_enough_need_no_taper():
    """k ~ r and k ~ r^(3/4) against dr: bounded on L2(r^2 dr) without a taper (p > 1/2), representers k / r^2
    singular but square integrable at the centre: realised as radial factors on a ball."""
    M = pg.L2(pg.Ball(2.0), realisation="radial_harmonics")
    k = pg.Tabulated("[0, 2]", [0.0, 1.0, 2.0], [0.0, 1.0, -0.5])
    G = pg.Functionals.separable(M, [k, lambda r: r ** 0.75], [(0, 0), (1, 1)], against=pg.LebesgueMeasure("[0, 2]"))
    f = M(lambda x, y, z: np.exp(-np.sqrt(x * x + y * y + z * z)) * (1 + x))
    v = G(f).values
    assert v[0] == pytest.approx(math.sqrt(4 * math.pi) * q(lambda r: k(r) * math.exp(-r), 0, 2, [1.0]), rel=1e-11)
    assert v[1] == pytest.approx(math.sqrt(4 * math.pi / 3) * q(lambda r: r ** 1.75 * math.exp(-r), 0, 2), rel=1e-10)
    assert any(c.status == "checked" and "set to zero within" in c.statement for c in G.claims)
    I, dr = radial_space()
    with pytest.raises(pg.errors.Unsupported):
        pg.Functionals(I, [k], against=dr)                     # singular representers on an interval space


def test_a_kernel_given_against_the_space_measure_is_its_own_representer():
    M, dr = radial_space()
    mu = M.measure
    k = pg.Tabulated("[0, 2]", [0.0, 1.0, 2.0], [1.0, 2.0, 0.5], against=mu)   # the volume-consistent reading
    G = pg.Functionals(M, [k], against=mu)
    f = M(lambda r: np.sin(r))
    assert G(f).values[0] == pytest.approx(q(lambda r: k(r) * math.sin(r) * r * r, 0, 2, [1.0]), rel=1e-12)


def test_against_accepts_any_measure_with_density_against():
    M, dr = radial_space()

    class Weighted:                                       # a measure defined elsewhere (EXP-031's interface)
        name = "r^3 dr"

        def density_against(self, base):
            from pygeoinf0.measures.densities import Density
            return Density(lambda r: r, (0.0, 2.0), {0.0: 1.0}, (0.0, 2.0))          # r^3 dr against r^2 dr
    G = pg.Functionals(M, [lambda r: np.ones_like(r)], against=Weighted())
    f = M(lambda r: np.exp(r))
    assert G(f).values[0] == pytest.approx(q(lambda r: math.exp(r) * r ** 3, 0, 2), rel=1e-12)
