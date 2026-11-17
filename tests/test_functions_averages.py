"""An averaging kernel is f.normalised() (plan 034): no second version of a family for it. GOLDEN holds the value at the domain's
centre of the removed sola_target (main 4ab2f8d), per shape and domain; the guards of normalised()."""
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0 import AveragingWarning
from _bank_averages import cap_average, depth_window, radius_window, target
from pygeoinf0.functions import geometry as geo
from pygeoinf0.functions.registry import entries

GOLDEN = [
    ('box', 'pg.Interval("[0, 1]")', 0.08, 0.0, 6.25),
    ('box', 'pg.Interval("[0, 1]")', 0.08, 0.03, 5.2631578947368425),
    ('box', 'pg.Box((0.0, 1.0), (0.0, 1.0))', 0.08, 0.0, 39.0625),
    ('box', 'pg.Box((0.0, 1.0), (0.0, 1.0))', 0.08, 0.03, 27.700831024930746),
    ('box', 'pg.Disk(1.0)', 0.08, 0.0, 39.0625),
    ('box', 'pg.Disk(1.0)', 0.08, 0.03, 27.700831024930746),
    ('box', 'pg.Ball(1.0)', 0.08, 0.0, 244.14062499999997),
    ('box', 'pg.Ball(1.0)', 0.08, 0.03, 145.79384749963552),
    ('box', 'pg.Shell(0.5, 1.0)', 0.08, 0.0, 244.14062499999997),
    ('box', 'pg.Shell(0.5, 1.0)', 0.08, 0.03, 145.79384749963552),
    ('box', 'pg.Cylinder(1.0, (0.0, 1.0))', 0.08, 0.0, 244.14062499999997),
    ('box', 'pg.Cylinder(1.0, (0.0, 1.0))', 0.08, 0.03, 145.79384749963552),
    ('box', 'pg.Torus((0.0, 2.0), (0.0, 2.0))', 0.08, 0.0, 39.0625),
    ('box', 'pg.Torus((0.0, 2.0), (0.0, 2.0))', 0.08, 0.03, 27.700831024930746),
    ('gaussian', 'pg.Interval("[0, 1]")', 0.15, 0.0, 2.661899433592045),
    ('gaussian', 'pg.Box((0.0, 1.0), (0.0, 1.0))', 0.15, 0.0, 7.0857085945576515),
    ('gaussian', 'pg.Disk(1.0)', 0.15, 0.0, 7.073553027886433),
    ('gaussian', 'pg.Ball(1.0)', 0.15, 0.0, 18.812929188545347),
    ('gaussian', 'pg.Shell(0.5, 1.0)', 0.15, 0.0, 20.801114328558878),
    ('gaussian', 'pg.Sphere(2.0)', 0.15, 0.0, 7.07355302630646),
    ('gaussian', 'pg.Circle(1.5)', 0.15, 0.0, 2.6562759831643286),
    ('gaussian', 'pg.Cylinder(1.0, (0.0, 1.0))', 0.15, 0.0, 18.829086798414192),
    ('bump', 'pg.Interval("[0, 1]")', 0.2, 0.0, 4.142844199345526),
    ('bump', 'pg.Box((0.0, 1.0), (0.0, 1.0))', 0.2, 0.0, 19.714344492816934),
    ('bump', 'pg.Disk(1.0)', 0.2, 0.0, 19.714344492816934),
    ('bump', 'pg.Ball(1.0)', 0.2, 0.0, 104.25320490469169),
    ('bump', 'pg.Shell(0.5, 1.0)', 0.2, 0.0, 104.25320490469169),
    ('bump', 'pg.Sphere(2.0)', 0.2, 0.0, 19.722932518895778),
    ('bump', 'pg.Circle(1.5)', 0.2, 0.0, 4.142844199345526),
    ('bump', 'pg.Cylinder(1.0, (0.0, 1.0))', 0.2, 0.0, 104.25320490469169),
    ('bump', 'pg.Torus((0.0, 2.0), (0.0, 2.0))', 0.2, 0.0, 19.714344492816934),
    ('cap', 'pg.Interval("[0, 1]")', 0.2, 0.0, 2.5),
    ('cap', 'pg.Interval("[0, 1]")', 0.2, 0.1, 3.3333333333333335),
    ('cap', 'pg.Box((0.0, 1.0), (0.0, 1.0))', 0.2, 0.0, 7.957747154594767),
    ('cap', 'pg.Box((0.0, 1.0), (0.0, 1.0))', 0.2, 0.1, 13.97985289338123),
    ('cap', 'pg.Disk(1.0)', 0.2, 0.0, 7.957747154594767),
    ('cap', 'pg.Disk(1.0)', 0.2, 0.1, 13.97985289338123),
    ('cap', 'pg.Ball(1.0)', 0.2, 0.0, 29.841551829730378),
    ('cap', 'pg.Ball(1.0)', 0.2, 0.1, 68.28468281529241),
    ('cap', 'pg.Shell(0.5, 1.0)', 0.2, 0.0, 29.841551829730378),
    ('cap', 'pg.Shell(0.5, 1.0)', 0.2, 0.1, 68.28468281529241),
    ('cap', 'pg.Sphere(2.0)', 0.2, 0.0, 7.964381927601134),
    ('cap', 'pg.Sphere(2.0)', 0.2, 0.1, 13.986797331624183),
    ('cap', 'pg.Circle(1.5)', 0.2, 0.0, 2.5),
    ('cap', 'pg.Circle(1.5)', 0.2, 0.1, 3.3333333333333335),
    ('cap', 'pg.Cylinder(1.0, (0.0, 1.0))', 0.2, 0.0, 29.841551829730378),
    ('cap', 'pg.Cylinder(1.0, (0.0, 1.0))', 0.2, 0.1, 68.28468281529241),
    ('cap', 'pg.Torus((0.0, 2.0), (0.0, 2.0))', 0.2, 0.0, 7.957747154594767),
    ('cap', 'pg.Torus((0.0, 2.0), (0.0, 2.0))', 0.2, 0.1, 13.97985289338123),
]


WINDOWS = [
    ('radius', 'pg.Ball(1.0)', 0.2, 0.7, 0.1, (0.5, 0.0, 0.0), 0.9205986050711864),
    ('radius', 'pg.Ball(1.0)', 0.1, 0.9, 0.3, (0.2, 0.0, 0.0), 0.10625172741752209),
    ('radius', 'pg.Disk(1.0)', 0.2, 0.7, 0.1, (0.5, 0.0), 0.8841941282883078),
    ('radius', 'pg.Shell(0.5, 1.0)', 0.6, 0.9, 0.1, (0.7, 0.0, 0.0), 0.7028538884157692),
    ('radius', 'pg.Ball(1.0)', 0.2, 0.7, 0.0, (0.5, 0.0, 0.0), 0.7126340735458001),
    ('radius', 'pg.Shell(0.5, 1.0)', 0.6, 0.9, 0.0, (0.7, 0.0, 0.0), 0.46536533067805647),
    ('depth', 'pg.Interval("[0, 1]")', 0.2, 0.7, 0.1, (0.5,), 2.5),
    ('depth', 'pg.Box((0.0, 1.0), (0.0, 2.0))', 0.2, 0.7, 0.1, (0.5, 1.5), 2.5),
    ('depth', 'pg.Cylinder(1.0, (0.0, 1.0))', 0.2, 0.7, 0.1, (0.1, 0.1, 0.5), 0.7957747154594768),
    ('depth', 'pg.Ball(1.0)', 0.2, 0.7, 0.1, (0.5, 0.0, 0.0), 0.6293644181990953),
    ('depth', 'pg.Shell(0.5, 1.0)', 0.1, 0.4, 0.05, (0.75, 0.0, 0.0), 0.5606261863032346),
    ('depth', 'pg.Disk(1.0)', 0.2, 0.7, 0.1, (0.5, 0.0), 0.7234315595086154),
    ('depth', 'pg.Interval("[0, 1]")', 0.2, 0.7, 0.0, (0.5,), 2.0000000000000004),
    ('depth', 'pg.Box((0.0, 1.0), (0.0, 2.0))', 0.2, 0.7, 0.0, (0.5, 1.5), 2.0000000000000004),
    ('depth', 'pg.Cylinder(1.0, (0.0, 1.0))', 0.2, 0.7, 0.0, (0.1, 0.1, 0.5), 0.6366197723675815),
    ('depth', 'pg.Ball(1.0)', 0.2, 0.7, 0.0, (0.5, 0.0, 0.0), 0.49223178275843926),
    ('cap', 'pg.Sphere(2.0)', 0.7, None, 0.0, (0.0, 0.0, 2.0), 0.16920014399483005),
    ('cap', 'pg.Sphere(2.0)', 0.7, None, 0.3, (0.0, 0.0, 2.0), 0.2679236764757994),
    ('cap', 'pg.Sphere(1.0)', 2.0, None, 1.0, (0.0, 0.0, 1.0), 0.17109566439889753),
]


@pytest.mark.parametrize("shape, domain, scale, edge, value", GOLDEN)
def test_each_removed_target_is_a_plain_family_normalised(shape, domain, scale, edge, value):
    D = eval(domain)
    f = target(D, shape=shape, scale=scale, edge=edge)
    assert f(*geo.centre(D)) == pytest.approx(value, rel=1e-12) and f.integral()[0] == 1.0


@pytest.mark.parametrize("kind, domain, a, b, edge, point, value", WINDOWS)
def test_each_removed_window_is_a_plain_family_normalised(kind, domain, a, b, edge, point, value):
    D = eval(domain)
    if kind == "radius":
        f = radius_window(D, r1=a, r2=b, edge=edge)
    elif kind == "depth":
        f = depth_window(D, d1=a, d2=b, edge=edge)
    else:
        f = cap_average(D, angle=a, edge=edge)
    assert f(*point) == pytest.approx(value, rel=1e-12) and f.integral()[0] == 1.0


def test_the_windows_are_gone():
    assert not {"cap_average", "radius_window", "depth_window", "sola_target"} & set(entries())


def test_a_rule_is_not_a_claim_of_one():
    """No exact integral (an oblique slab in a cylinder, an oblique smooth step): status unknown, the estimated error stated, and the
    normalised function's integral is what a coarser rule gives, with nothing claimed."""
    C = pg.Cylinder(1.0, (0.0, 1.0))
    for f in (F.on(C).slab_indicator(normal=(0.6, 0.0, 0.8)), F.on(C).step_across_plane(normal=(0.6, 0.0, 0.8), point=(0.1, 0.0, 0.5), width=0.05)):
        v, status, why = f.integral()
        assert status == "unknown" and "estimated error" in why and "integral" not in f.closed
        n = f.normalised()
        assert "integral" not in n.closed and "integral" not in getattr(n, "exact", {})
        v, status, why = n.integral()
        assert status == "unknown" and "estimated error" in why and v != 1.0 and abs(v - 1.0) < 0.05


@pytest.mark.parametrize("make", [
    lambda: F.on(pg.Ball(1.0)).dyke(normal=(0, 0, 1), thickness=0.3, width=0.05),
    lambda: F.on(pg.Disk(1.0)).dyke(normal=(0.6, 0.8), thickness=0.3, width=0.05, point=(0.2, 0.0)),
    lambda: F.on(pg.Box((0.0, 1.0), (0.0, 1.0))).dyke(normal=(0.6, 0.8), thickness=0.2, width=0.05),
    lambda: F.on(pg.Ball(1.0)).step_across_plane(normal=(0, 0, 1), point=(0, 0, 0.2), width=0.05),
    lambda: F.on(pg.Disk(1.0)).step_across_plane(normal=(0.6, 0.8), point=(0.2, 0.0), width=0.1),
    lambda: F.on(pg.Box((0.0, 1.0), (0.0, 1.0))).partition_of_unity(index=(1, 1), count=3, overlap=1.2),
    lambda: F.on(pg.Interval("[0, 1]")).partition_of_unity(index=1, count=4, overlap=0.75, k=2.0),
])
def test_the_former_rule_only_families_have_a_checked_integral(make):
    f = make()
    v, status, _ = f.integral()
    assert status == "checked" and "integral" in f.closed
    n = {1: 20000, 2: 1500}.get(geo.dimension(f.domain), 300)
    X, w = geo.quadrature(f.domain, n)
    assert w @ f(*X.T) == pytest.approx(v, rel=2e-5)
    assert f.normalised().integral()[0] == 1.0


def test_smooth_band_is_an_ordinary_family():
    B = pg.Ball(1.0)
    f = F.on(B).smooth_band(lower=0.2, upper=0.6, edge=0.1)
    assert f(0.4, 0.0, 0.0) == 1.0 and f(0.1, 0.0, 0.0) == 0.0 and f.params["coordinate"] == "radius"
    depth = F.on(B).smooth_band(coordinate="depth", lower=0.4, upper=0.8, edge=0.1)
    X, _ = geo.quadrature(B, 9)
    assert np.allclose(f(*X.T), depth(*X.T), atol=1e-12)                 # radius 0.2..0.6 is depth 0.4..0.8
    for kw, msg in (({"edge": 0.0}, "sharp band"), ({"edge": 0.5, "lower": 0.2, "upper": 0.6}, "no plateau"), ({"lower": 0.6, "upper": 0.2}, "lower < upper"),
                    ({"coordinate": "angle"}, "radius"), ({"lower": "a"}, "number")):
        with pytest.raises(pg.Refused, match=msg):
            F.on(B).smooth_band(**kw)
    with pytest.raises(pg.Refused, match="depth"):
        F.on(pg.Interval("[0, 1]")).smooth_band(coordinate="radius")


def test_no_family_carries_sola_in_its_name():
    assert not [name for name in entries() if "sola" in name]
    assert not [m for m in {e.module for e in entries().values()} if "sola" in m]


def test_the_plateau_is_an_ordinary_family_normalised_can_scale():
    B = pg.Ball(1.0)
    p = F.on(B).plateau(half_width=(0.1, 0.2, 0.15), edge=0.05)
    assert p(0.0, 0.0, 0.0) == 1.0 and p.integral()[0] == pytest.approx(0.25 * 0.45 * 0.35, rel=1e-14)
    assert p.normalised()(0.0, 0.0, 0.0) == pytest.approx(1 / (0.25 * 0.45 * 0.35), rel=1e-14)
    sharp = F.on(B).plateau(half_width=0.1)
    assert sharp.integral()[0] == pytest.approx(0.008, rel=1e-14) and sharp.normalised().integral()[0] == 1.0


def test_normalised_keeps_the_gaussians_exact_integral_on_a_curved_boundary():
    B = pg.Ball(1.0)
    g = F.on(B).gaussian(width=0.03)
    full = (2 * np.pi * 0.03 ** 2) ** 1.5
    assert g.integral()[0] == pytest.approx(full, rel=1e-12) and g.integral()[1] == "checked"
    assert g.normalised()(0.0, 0.0, 0.0) == pytest.approx(1 / full, rel=1e-12)


def test_a_gaussian_almost_outside_the_domain_is_refused_when_its_integral_is_asked_for():
    f = F.on(pg.Ball(1.0)).gaussian(centre=(2.5, 0.0, 0.0), width=0.1)
    assert f(1.0, 0.0, 0.0) > 0
    with pytest.raises(pg.Refused, match="full-space mass"):
        f.normalised()


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AveragingWarning)
        return f.normalised()


@pytest.mark.parametrize("make", [lambda D: F.on(D).bump(), lambda D: F.on(D).gaussian(), lambda D: F.on(D).indicator(radius=0.3),
                                  lambda D: F.on(D).smooth_indicator(), lambda D: F.on(D).plateau(half_width=0.1)])
def test_a_localized_average_gives_no_warning(make):
    assert _quiet(make(pg.Ball(1.0))).integral()[0] == pytest.approx(1.0, abs=1e-13)


def test_a_function_that_changes_sign_warns():
    B = pg.Ball(1.0)
    g = F.on(B).gaussian(width=0.2).scaled(-1.0)
    assert _quiet(g)(0.0, 0.0, 0.0) > 0                       # one sign throughout is still an average, once normalised
    blobs = F.on(pg.Box((0.0, 1.0), (0.0, 1.0))).blobs(centres=((0.3, 0.3), (0.7, 0.7)), sizes=(0.1, 0.1), amplitudes=(1.0, -0.5))
    with pytest.warns(AveragingWarning, match="changes sign"):
        blobs.normalised()


def test_a_near_zero_integral_warns_and_an_exact_zero_is_refused():
    box = pg.Box((0.0, 1.0), (0.0, 1.0))
    near = F.on(box).blobs(centres=((0.3, 0.3), (0.7, 0.7)), sizes=(0.1, 0.1), amplitudes=(1.0, -1.0 + 1e-10))
    with pytest.warns(AveragingWarning, match="blows up"):
        near.normalised()
    with pytest.raises(pg.Refused, match="integral 0"):
        F.on(pg.Sphere(1.0)).spherical_harmonic(degree=4, j=4).normalised()


def test_sharp_function_normalises_against_a_measure():
    I = pg.Interval("[0, 1]")
    w = pg.LebesgueMeasure(I).with_density(lambda r: r ** 2, name="r^2")
    t = F.on(I).plateau(centre=0.5, half_width=0.2).normalised(against=w)
    M = pg.L2(w)
    assert t.closed["mass"][0] == 1.0
    assert abs(pg.Functionals(M.riesz([t.measure(w)]))(M(lambda r: 1 + 0 * r)).values[0] - 1) < 1e-8
