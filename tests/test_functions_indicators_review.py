"""Review fixes of the indicator families (plan 021 WP-C, second pass): each test pins one corrected claim.

    slab_indicator on a box   the exact integral does not lose digits on elongated boxes with nearly axial normals
    smooth_indicator          the integral and squared norm stay accurate for a narrow transition (inner near radius)
    indicator, smooth_*       the centre on a closed manifold must lie on it
    full / touches / interior decided by exact comparison: nothing is the constant 1 unless it is, to the last bit
    the quadrature fallback is no longer called "checked"
"""
import itertools
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.functions import geometry as geo
from pygeoinf0.functions.families.indicators import _psi
from pygeoinf0.spaces.membership import check


def build(name, D, **params):
    return F.entries()[name].builder(D, **params)


# ------------------------------------------------------------------------------------------ box slab accuracy
def polytope_volume(sides, n, a, b):
    """Volume of {x in the box, a <= n . x <= b} by scipy (halfspace intersection and convex hull): independent of the
    Irwin-Hall formula."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    d = len(sides)
    A, c = [], []
    for i, (lo, hi) in enumerate(sides):
        e = np.zeros(d)
        e[i] = 1.0
        A += [-e, e]
        c += [lo, -hi]                                       # A x + c <= 0
    A += [-np.asarray(n), np.asarray(n)]
    c += [a, -b]
    A, c = np.array(A), np.array(c)
    norms = np.linalg.norm(A, axis=1)
    res = linprog(np.r_[np.zeros(d), -1.0], A_ub=np.c_[A, norms], b_ub=-c, bounds=[(None, None)] * d + [(0, None)])
    hs = HalfspaceIntersection(np.c_[A, c], res.x[:d])
    return ConvexHull(hs.intersections).volume


@pytest.mark.parametrize("sides, v, lower, upper", [
    (((0, 0.01), (0, 0.01), (0, 10)), (1.5e-3, 1.5e-3, 1.0), 3.0, 6.0),                  # the reviewer's 3.6e-5 case
    (((0, 0.1), (0, 0.1), (0, 100)), (1.1e-3, 1.1e-3, 1.0), 30.0, 60.0),
    (((0, 1), (0, 1), (0, 1000)), (1.1e-3, 1.1e-3, 1.0), 300.0, 600.0),
    (((0, 1), (0, 1), (0, 1)), (1.1e-3, 1.1e-3, 1.0), 0.3, 0.6),
    (((0, 1), (0, 1), (0, 1)), (1e-5, 2e-5, 1.0), 0.3, 0.6),                              # below the old 1e-3 guard
    (((0, 1), (0, 1)), (1e-4, 1.0), 0.3, 0.6),
])
def test_box_slab_does_not_lose_digits(sides, v, lower, upper):
    pytest.importorskip("scipy")
    D = pg.Box(*sides)
    n = np.asarray(v, float) / np.linalg.norm(v)
    f = build("slab_indicator", D, normal=tuple(n), lower=lower, upper=upper)
    val, status, _ = f.integral()
    assert status == "derived"
    assert val == pytest.approx(polytope_volume(sides, n, lower, upper), rel=1e-9)


def test_box_slab_random_boxes_and_normals():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(5)
    for _ in range(25):
        dim = int(rng.integers(2, 4))
        sides = tuple((float(lo), float(lo + w)) for lo, w in zip(rng.uniform(-2, 2, dim), 10 ** rng.uniform(-2, 2, dim)))
        v = rng.normal(size=dim) * 10 ** rng.uniform(-3, 0, dim)
        n = v / np.linalg.norm(v)
        D = pg.Box(*sides)
        smin = sum(min(x * lo, x * hi) for x, (lo, hi) in zip(n, sides))
        smax = sum(max(x * lo, x * hi) for x, (lo, hi) in zip(n, sides))
        a, b = np.sort(rng.uniform(smin, smax, 2))
        f = build("slab_indicator", D, normal=tuple(n), lower=float(a), upper=float(b))
        assert f.integral()[0] == pytest.approx(polytope_volume(sides, n, a, b), rel=1e-9)


def test_box_slab_partition_of_an_elongated_box_is_exact():
    """Slabs that tile the box add up to its volume to rounding, however skewed the normal."""
    sides = ((0, 0.01), (0, 0.01), (0, 10))
    D = pg.Box(*sides)
    v = np.array([1.5e-3, 1.5e-3, 1.0])
    n = v / np.linalg.norm(v)
    smin, smax = 0.0, float(n @ [0.01, 0.01, 10])
    t = np.linspace(smin, smax, 9)
    vols = [build("slab_indicator", D, normal=tuple(n), lower=float(a), upper=float(b)).integral()[0]
            for a, b in itertools.pairwise(t)]
    assert sum(vols) == pytest.approx(0.01 * 0.01 * 10, rel=1e-13)


# ------------------------------------------------------------------------------------------ narrow transitions
def reference_mass(D, radius, inner, power):
    """vol(ball(inner)) + width int_0^1 psi^p(t) density(radius - width t) dt by scipy quad; psi from first
    principles, not from the module."""
    from scipy.integrate import quad

    def f(s):
        return math.exp(-1.0 / s) if s > 0 else 0.0

    def psi(t):
        return f(t) / (f(t) + f(1 - t))
    w = radius - inner
    k = geo.kind(D)
    if k == "ball":
        def dens(r):
            return 4 * math.pi * r * r

        def vol(r):
            return 4 * math.pi * r ** 3 / 3
    elif k == "interval":
        def dens(r):
            return 2.0

        def vol(r):
            return 2 * r
    else:                                                                                # sphere
        R = D.radius

        def dens(r):
            return 2 * math.pi * R * math.sin(r / R)

        def vol(r):
            return 2 * math.pi * R * R * (1 - math.cos(r / R))
    val, _ = quad(lambda t: psi(t) ** power * dens(radius - w * t), 0, 1, epsabs=1e-15, epsrel=1e-13, limit=200,
                  points=[0.5])
    return vol(inner) + w * val


@pytest.mark.parametrize("D, radius, inner", [
    (pg.Ball(1.0), 0.9, 0.85), (pg.Ball(1.0), 0.9, 0.89), (pg.Ball(1.0), 0.9, 0.899), (pg.Ball(1.0), 0.9, 0.8999),
    (pg.Interval("[0, 1]"), 0.4, 0.399), (pg.Sphere(1.7), 1.0, 0.999), (pg.Ball(1.0), 0.6, 0.2), (pg.Ball(1.0), 0.6, 0.0),
])
def test_smooth_indicator_narrow_transition(D, radius, inner):
    pytest.importorskip("scipy")
    f = build("smooth_indicator", D, radius=radius, inner=inner)
    v, status, _ = f.integral()
    assert status == "derived"
    assert v == pytest.approx(reference_mass(D, radius, inner, 1), rel=1e-11)
    assert f.closed["l2_norm_sq"][0] == pytest.approx(reference_mass(D, radius, inner, 2), rel=1e-11)


def test_smooth_indicator_mass_matches_radial_mass_on_wide_transitions():
    for D in (pg.Disk(1.0), pg.Ball(1.0), pg.Sphere(1.0), pg.Circle(1.0), pg.Interval("[0, 1]")):
        rad = 0.4 if geo.kind(D) == "interval" else 0.8
        inner = 0.3 * rad
        f = build("smooth_indicator", D, radius=rad, inner=inner)
        ref = geo.radial_mass(D, lambda r, rad=rad, inner=inner: _psi((rad - np.asarray(r, float)) / (rad - inner)), rad)
        assert f.integral()[0] == pytest.approx(ref, rel=1e-12)


# ------------------------------------------------------------------------------------------ closed-manifold centres
@pytest.mark.parametrize("name", ["indicator", "smooth_indicator"])
@pytest.mark.parametrize("D, centre", [
    (pg.Sphere(1.0), (0.0, 0.0, 0.0)), (pg.Sphere(1.0), (0.0, 0.0, 2.0)), (pg.Sphere(1.0), (0.5, 0.0, 0.0)),
    (pg.Circle(1.0), (2.0, 0.0)), (pg.Circle(1.0), (0.0, 0.0)),
    (pg.Torus((0, 2 * math.pi), (0, 2 * math.pi)), (20.0, 20.0)),
    (pg.Torus((0, 2 * math.pi), (0, 2 * math.pi)), (10.0, 10.0)),
    (pg.Torus((0, 2 * math.pi), (0, 2 * math.pi)), (-0.1, 1.0)),
])
def test_centre_off_a_closed_manifold_is_refused(name, D, centre):
    with pytest.raises(pg.Refused, match="outside the domain"):
        build(name, D, centre=centre, radius=0.5)


def test_centre_on_a_closed_manifold_is_accepted_and_projected():
    S = pg.Sphere(2.0, (0.0, 0.0, 1.0))
    f = build("indicator", S, centre=(2.0 * (1 + 1e-12), 0.0, 1.0), radius=0.5)             # on the sphere to 1e-12
    assert np.linalg.norm(np.array(f.support.centre) - [2.0, 0.0, 1.0]) < 1e-9
    assert f.integral()[0] == pytest.approx(2 * math.pi * 4 * (1 - math.cos(0.25)), rel=1e-12)
    T = pg.Torus((0, 2 * math.pi), (0, 2 * math.pi))
    g = build("indicator", T, centre=(0.0, 2 * math.pi), radius=1.0)                       # the closed cell's corner
    assert g.integral()[0] == pytest.approx(math.pi, rel=1e-14)
    X = (np.arange(1200) + 0.5) * (2 * math.pi / 1200)
    P = np.stack(np.meshgrid(X, X), -1).reshape(-1, 2)
    area = g(*P.T).mean() * (2 * math.pi) ** 2
    assert area == pytest.approx(math.pi, rel=0.01)                                        # wraps around, as claimed


# ------------------------------------------------------------------------------------------ exact 'full'
def test_a_ball_slightly_smaller_than_the_domain_is_not_the_constant_one():
    D = pg.Ball(1.0)
    f = build("indicator", D, radius=1 - 3e-10)
    assert f.support.kind == "interior" and f.regularity.order == -1 and f.regularity.piecewise
    assert f(1.0, 0.0, 0.0) == 0.0 and f(0.5, 0, 0) == 1.0
    assert not check(f, pg.Sobolev(D, 2, length_scale=0.2))[0]
    g = build("indicator", D, radius=1 - 1e-13)                                            # below the old slack as well
    assert g.support.kind == "interior" and g.regularity.order == -1 and g(1.0, 0, 0) == 0.0
    for r in (1.0, 1 + 1e-13):                                                              # the domain itself
        h = build("indicator", D, radius=r)
        assert h.support.kind == "full" and h.regularity.order == math.inf and h(1.0, 0, 0) == 1.0
        assert h.integral()[0] == pytest.approx(4 * math.pi / 3, rel=1e-12)
    assert build("indicator", D, radius=1 + 1e-13).params["radius"] == 1.0                  # reduced onto the domain
    assert build("indicator", pg.Interval("[0, 1]"), radius=0.5 - 1e-13).support.kind == "interior"
    assert build("indicator", pg.Disk(1.0), radius=1 - 1e-12).support.kind == "interior"
    assert build("indicator", pg.Interval("[0, 1]"), radius=0.5).support.kind == "full"


def test_box_indicator_does_not_snap_inside_the_slack():
    D = pg.Box((0, 1), (0, 1))
    f = build("box_indicator", D, lower=(1e-13, 0.0), upper=(1.0, 1.0))
    assert f.support.kind == "touches" and f.regularity.order == -1          # touches the other three sides
    assert f(5e-14, 0.5) == 0.0 and f(1e-13, 0.5) == 1.0
    g = build("box_indicator", D, lower=(0.0, 0.0), upper=(1.0, 1 - 1e-13))
    assert g.support.kind == "touches" and g(0.5, 1.0) == 0.0
    h = build("box_indicator", D, lower=(-1e-13, 0.0), upper=(1.0 + 1e-13, 1.0))          # outside by the slack: onto the domain
    assert h.support.kind == "full" and h.regularity.order == math.inf and h(0.0, 0.0) == 1.0
    assert h.integral()[0] == 1.0
    with pytest.raises(pg.Refused, match="leaves the domain"):
        build("box_indicator", D, lower=(-1e-6, 0.0), upper=(1.0, 1.0))


def test_band_and_slab_do_not_snap_inside_the_slack():
    D = pg.Disk(1.0)
    f = build("band_indicator", D, lower=1e-13, upper=1.0)
    assert f.support.kind == "touches" and f.regularity.order == -1 and f(0.0, 0.0) == 0.0 and f(0.1, 0.0) == 1.0
    assert build("band_indicator", D, lower=0.0, upper=1 - 1e-13).support.kind == "interior"
    assert build("band_indicator", D, lower=-1e-13, upper=1.0 + 1e-13).support.kind == "full"
    S = pg.Sphere(1.0)
    g = build("band_indicator", S, lower=-math.pi / 2 + 1e-13, upper=math.pi / 2)
    assert g.support.kind == "interior" and g.regularity.order == -1
    assert build("band_indicator", S, lower=-math.pi / 2, upper=math.pi / 2).support.kind == "full"
    B = pg.Ball(1.0)
    s = build("slab_indicator", B, normal=(0, 0, 1), lower=-1.0 + 1e-13, upper=2.0)
    assert s.support.kind == "touches" and s.regularity.order == -1 and s(0, 0, -1.0) == 0.0
    assert build("slab_indicator", B, normal=(0, 0, 1), lower=-1.0, upper=2.0).support.kind == "full"


# ------------------------------------------------------------------------------------------ documented defaults, fallback
def test_smooth_indicator_default_radius_is_half_of_min_room_and_injectivity():
    f = build("smooth_indicator", pg.Sphere(1.5))
    assert f.params["radius"] == pytest.approx(0.5 * math.pi * 1.5, rel=1e-14)
    g = build("smooth_indicator", pg.Ball(1.0))
    assert g.params["radius"] == pytest.approx(0.5, rel=1e-14)
    T = pg.Torus((0, 2 * math.pi), (0, 4))
    assert build("smooth_indicator", T).params["radius"] == pytest.approx(0.5 * 2.0, rel=1e-14)


def test_quadrature_fallback_is_not_called_checked():
    D = pg.Cylinder(1.0, (0.0, 1.0))
    f = build("slab_indicator", D, normal=(0.0, 0.6, 0.8), lower=0.1, upper=0.6)
    _v, status, why = f.integral()
    assert status == "unknown" and "not bounded" in why
    assert all(c.status != "checked" for c in f.claims if c.key == "integral")
