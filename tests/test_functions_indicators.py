"""Indicator families of the bank (plan 021 WP-C): what the contract cannot see.

The contract test cannot check the exact integrals of 0/1 functions (a quadrature rule is O(h) accurate at a jump), so
they are checked here: by partitions of the domain (the pieces add up to the domain's volume), by an independent
one-dimensional integral, by known values, and by Monte Carlo with a 5-sigma tolerance. Also: the values on and off the
boundary of the set, the kinds of support, the regularity claims and the refusals, and the smooth step's identities
(psi(t) + psi(1 - t) = 1, flatness at the ends, the r = 0 limit)."""
import itertools
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.functions import geometry as geo
from pygeoinf0.functions.families.indicators import _psi
from pygeoinf0.spaces.membership import check

JUMPY = ("indicator", "box_indicator", "band_indicator", "slab_indicator")


def build(name, D, **params):
    return F.entries()[name].builder(D, **params)


def sample(D, n, rng):
    """(points (n, ambient), measure of the region they are uniform in, indicator of the domain on them)."""
    k = geo.kind(D)
    if k == "sphere":
        v = rng.normal(size=(n, 3))
        return np.asarray(D.centre) + D.radius * v / np.linalg.norm(v, axis=1)[:, None], geo.volume(D), np.ones(n, bool)
    if k == "circle":
        a = rng.uniform(0, 2 * np.pi, n)
        return np.asarray(D.centre) + D.radius * np.stack([np.cos(a), np.sin(a)], 1), geo.volume(D), np.ones(n, bool)
    box = [(D.a, D.b)] if k == "interval" else D.bounding_box
    lo, hi = np.array([b[0] for b in box]), np.array([b[1] for b in box])
    P = lo + (hi - lo) * rng.uniform(size=(n, len(box)))
    ins = np.ones(n, bool) if k in ("interval", "box", "torus") else np.asarray(D.contains(P), bool)
    return P, float(np.prod(hi - lo)), ins


def monte_carlo(f, D, n=300_000, seed=7):
    P, V, ins = sample(D, n, np.random.default_rng(seed))
    y = f(*P.T) * ins
    return V * y.mean(), V * y.std() / math.sqrt(n)


def cases():
    out = []
    for name in JUMPY:
        e = F.entries()[name]
        for k in e.kinds:
            for i, D in enumerate(geo.standard(k)):
                for j, params in enumerate(e.example_params(D)):
                    out.append(pytest.param(name, D, params, id=f"{name}-{k}{i}-{j}"))
    return out


# ------------------------------------------------------------------------------------------ exact integrals
@pytest.mark.parametrize("name, D, params", cases())
def test_exact_integral_against_monte_carlo(name, D, params):
    f = build(name, D, **params)
    if "integral" not in f.exact:                                   # oblique slab in a cylinder: quadrature fallback
        v, status, why = f.integral()
        assert status == "unknown" and "O(h)" in why                # error not bounded: not "checked"
        return
    v, status, _ = f.integral()
    assert status == "derived" and v == f.exact["integral"][0]
    est, sigma = monte_carlo(f, D)
    assert abs(est - v) <= 5 * sigma + 1e-12, (est, v, sigma)
    assert f.exact["l2_norm_sq"][0] == v                             # 0/1 valued: int f^2 = int f
    assert "integral" not in f.closed                                # kept out of `closed`: the contract cannot check it


def test_known_values():
    R = 1.0
    assert build("indicator", pg.Ball(R), radius=0.5).integral()[0] == pytest.approx(4 / 3 * math.pi * 0.125, rel=1e-14)
    assert build("indicator", pg.Disk(R), radius=0.5).integral()[0] == pytest.approx(math.pi * 0.25, rel=1e-14)
    assert build("indicator", Interval01(), radius=0.25).integral()[0] == pytest.approx(0.5, rel=1e-14)
    assert build("indicator", pg.Circle(2.0), radius=0.3).integral()[0] == pytest.approx(0.6, rel=1e-14)
    # a cap of the unit sphere of angular radius pi/3 is a quarter of the sphere: 2 pi (1 - cos(pi/3)) = pi
    assert build("indicator", pg.Sphere(1.0), radius=math.pi / 3).integral()[0] == pytest.approx(math.pi, rel=1e-14)
    assert build("box_indicator", pg.Box((0, 2), (0, 3)), lower=(0.5, 1), upper=(1.5, 3)).integral()[0] == 2.0
    # the unit ball cut by |z| <= 1/2: pi (1 - 1/12)
    assert build("slab_indicator", pg.Ball(1.0), normal=(0, 0, 1), lower=-0.5, upper=0.5).integral()[0] == \
        pytest.approx(11 * math.pi / 12, rel=1e-14)
    # the northern hemisphere, and a zone: 2 pi R^2 (sin(upper) - sin(lower))
    assert build("band_indicator", pg.Sphere(1.0), lower=0.0, upper=math.pi / 2).integral()[0] == \
        pytest.approx(2 * math.pi, rel=1e-14)
    assert build("band_indicator", pg.Sphere(2.0), lower=-0.3, upper=0.9).integral()[0] == \
        pytest.approx(8 * math.pi * (math.sin(0.9) + math.sin(0.3)), rel=1e-14)
    # an annulus and a spherical shell
    assert build("band_indicator", pg.Disk(2.0), lower=0.5, upper=1.5).integral()[0] == pytest.approx(2 * math.pi, rel=1e-14)
    assert build("band_indicator", pg.Ball(2.0), lower=1.0, upper=2.0).integral()[0] == pytest.approx(28 * math.pi / 3, rel=1e-14)


def Interval01():
    return pg.Interval("[0, 1]")


def cuts(lo, hi):
    return np.r_[lo, np.sort(np.random.default_rng(3).uniform(lo, hi, 5)), hi]


def test_partitions_add_up_to_the_volume():
    """Consecutive slabs (bands) tile the domain, so their exact integrals add up to its volume."""
    domains = [pg.Box((0, 1), (0, 2)), pg.Box((0, 1), (-1, 1), (0, 3)), pg.Disk(1.3, (0.2, -0.1)), pg.Ball(0.8, (0.1, 0, -0.2)),
               pg.Cylinder(1.0, (0.0, 2.0), (0.3, 0.1))]
    for D in domains:
        k = geo.kind(D)
        dim = geo.ambient(D)
        normals = [np.eye(dim)[0], np.eye(dim)[-1], np.ones(dim) / math.sqrt(dim)]
        if dim == 2:
            normals.append(np.array([0.6, -0.8]))
        if dim == 3:
            normals.append(np.array([1.0, -2.0, 2.0]) / 3.0)
        for n in normals:
            if k == "cylinder" and n[2] != 0 and (n[0] != 0 or n[1] != 0):
                continue                                              # no closed form for oblique normals
            smin, smax = _range(D, n)
            t = cuts(smin, smax)
            vols = [build("slab_indicator", D, normal=tuple(n), lower=a, upper=b).integral()[0] for a, b in itertools.pairwise(t)]
            assert sum(vols) == pytest.approx(geo.volume(D), rel=1e-11), (D, n)
    for D in (pg.Disk(1.3), pg.Ball(0.8), pg.Shell(0.4, 1.1)):
        lo, hi = (D.inner, D.outer) if geo.kind(D) == "shell" else (0.0, D.radius)
        t = cuts(lo, hi)
        vols = [build("band_indicator", D, lower=a, upper=b).integral()[0] for a, b in itertools.pairwise(t)]
        assert sum(vols) == pytest.approx(geo.volume(D), rel=1e-12)
    S = pg.Sphere(1.7)
    t = cuts(-math.pi / 2, math.pi / 2)
    assert sum(build("band_indicator", S, lower=a, upper=b).integral()[0] for a, b in itertools.pairwise(t)) == \
        pytest.approx(4 * math.pi * 1.7 ** 2, rel=1e-12)


def _range(D, n):
    from pygeoinf0.functions.families.indicators import _extent
    return _extent(D, np.asarray(n, float))


def test_oblique_box_slab_against_an_independent_integral():
    """Area of {a <= n . x <= b} in the unit square, as the integral over x of the length of the admissible y-interval."""
    n = np.array([0.6, 0.8])
    x = (np.arange(400_000) + 0.5) / 400_000
    for a, b in ((0.2, 0.7), (0.0, 0.5), (0.45, 1.3), (-1.0, 0.9)):
        lo = np.clip((a - n[0] * x) / n[1], 0, 1)
        hi = np.clip((b - n[0] * x) / n[1], 0, 1)
        ref = np.mean(np.maximum(hi - lo, 0))
        f = build("slab_indicator", pg.Box((0, 1), (0, 1)), normal=(0.6, 0.8), lower=a, upper=b)
        assert f.integral()[0] == pytest.approx(ref, abs=1e-9)
    # three dimensions: a slab through the cube [0,1]^3 with normal (1,1,1)/sqrt 3 splits it in halves at the midplane
    s = math.sqrt(3)
    h = build("slab_indicator", pg.Box((0, 1), (0, 1), (0, 1)), normal=(1 / s, 1 / s, 1 / s), lower=-1.0, upper=1.5 / s)
    assert h.integral()[0] == pytest.approx(0.5, rel=1e-13)
    # P(sum of three uniforms <= 1) = 1/6
    h = build("slab_indicator", pg.Box((0, 1), (0, 1), (0, 1)), normal=(1 / s, 1 / s, 1 / s), lower=-1.0, upper=1 / s)
    assert h.integral()[0] == pytest.approx(1 / 6, rel=1e-13)


def test_disk_slab_against_the_chord_integral():
    u = np.linspace(-0.5, 0.5, 400_001)
    ref = np.trapezoid(2 * np.sqrt(1 - u ** 2), u)
    f = build("slab_indicator", pg.Disk(1.0), normal=(1.0, 0.0), lower=-0.5, upper=0.5)
    assert f.integral()[0] == pytest.approx(ref, rel=1e-10)


# ------------------------------------------------------------------------------------------ values and supports
def test_values_on_and_off_the_boundary_of_the_set():
    f = build("indicator", pg.Disk(1.0), radius=0.5)
    assert f(0.5, 0.0) == 1.0 and f(0.0, -0.5) == 1.0 and f(0.0, 0.0) == 1.0     # the ball is closed
    assert f(0.5 + 1e-9, 0.0) == 0.0 and f(0.9, 0.0) == 0.0
    P = np.array([[0.1, 0.2], [0.9, 0.0], [0.0, 0.5]])
    assert np.array_equal(f(P), [1.0, 0.0, 1.0])                                  # f(P) with P of shape (..., d)
    g = build("box_indicator", pg.Box((0, 1), (0, 1)), lower=(0.2, 0.2), upper=(0.6, 0.8))
    assert g(0.2, 0.2) == 1.0 and g(0.6, 0.8) == 1.0 and g(0.61, 0.5) == 0.0 and g(0.4, 0.81) == 0.0
    s = build("slab_indicator", pg.Box((0, 1), (0, 1)), normal=(0.6, 0.8), lower=0.3, upper=0.7)
    assert s(0.0, 0.5) == 1.0 and s(0.5, 0.0) == 1.0 and s(0.0, 0.2) == 0.0 and s(1.0, 1.0) == 0.0
    b = build("band_indicator", pg.Shell(0.5, 1.0), lower=0.6, upper=0.8)
    assert b(0.6, 0, 0) == 1.0 and b(0, 0.8, 0) == 1.0 and b(0.59, 0, 0) == 0.0 and b(0.9, 0, 0) == 0.0
    z = build("band_indicator", pg.Sphere(2.0, (0, 0, 1)), lower=0.0, upper=math.pi / 6)
    assert z(2.0, 0, 1.0) == 1.0 and z(0, 0, 3.0) == 0.0 and z(0, 0, -1.0) == 0.0


def test_support_kinds_and_regularity():
    D = pg.Disk(1.0)
    inside = build("indicator", D, radius=0.4)
    assert inside.support.kind == "interior" and inside.regularity.vanishes_at_boundary == "all orders"
    assert inside.regularity.order == -1 and inside.regularity.piecewise
    assert inside.support.centre == (0.0, 0.0) and inside.support.radius == 0.4
    touching = build("indicator", D, centre=(0.5, 0.0), radius=0.5)
    assert touching.support.kind == "touches" and touching.regularity.vanishes_at_boundary is None
    whole = build("indicator", D, radius=1.0)                                       # the ball is the domain: constant 1
    assert whole.support.kind == "full" and whole.regularity.order == math.inf and whole(0.3, 0.3) == 1.0
    assert build("box_indicator", pg.Box((0, 1), (0, 1)), lower=(0.2, 0.2), upper=(0.5, 0.5)).support.kind == "interior"
    assert build("box_indicator", pg.Box((0, 1), (0, 1)), lower=(0.0, 0.2), upper=(0.5, 0.5)).support.kind == "touches"
    assert build("box_indicator", pg.Box((0, 1), (0, 1)), lower=(0, 0), upper=(1, 1)).support.kind == "full"
    assert build("band_indicator", pg.Ball(1.0), lower=0.0, upper=0.5).support.kind == "interior"      # r = 0 is no boundary
    assert build("band_indicator", pg.Ball(1.0), lower=0.5, upper=1.0).support.kind == "touches"
    assert build("band_indicator", pg.Shell(0.5, 1.0), lower=0.5, upper=0.7).support.kind == "touches"
    assert build("band_indicator", pg.Shell(0.5, 1.0), lower=0.6, upper=0.7).support.kind == "interior"
    assert build("band_indicator", pg.Shell(0.5, 1.0), lower=0.5, upper=1.0).support.kind == "full"
    assert build("band_indicator", pg.Sphere(1.0), lower=-0.2, upper=0.2).support.kind == "interior"
    assert build("slab_indicator", pg.Ball(1.0), normal=(0, 0, 1), lower=-0.2, upper=0.2).support.kind == "touches"
    cover = build("slab_indicator", pg.Ball(1.0), normal=(0, 0, 1), lower=-1.0, upper=2.0)
    assert cover.support.kind == "full" and cover.regularity.order == math.inf
    # the support set is exactly where the function is 1
    X, _ = geo.quadrature(D, 20)
    assert np.array_equal(inside.support.contains(*X.T), inside(*X.T) == 1.0)


def test_membership_claims():
    for D in (pg.Disk(1.0), pg.Box((0, 1), (0, 1)), pg.Sphere(1.0)):
        f = build("indicator", D, radius=0.3)
        assert check(f, pg.L2(D))[0]
        ok, reason, _ = check(f, pg.Sobolev(D, 1, length_scale=0.2))
        assert not ok and "H^s" in reason
    I = Interval01()
    f = build("indicator", I, radius=0.2)
    assert not check(f, pg.Continuous("[0, 1]"))[0]
    with pytest.raises(pg.Refused, match="not shown to be in"):
        f.element(pg.Continuous("[0, 1]"))
    assert f.element(pg.L2(I)) is not None
    whole = build("indicator", I, radius=0.5)                                        # the constant 1 is continuous
    assert check(whole, pg.Continuous("[0, 1]"))[0]


def test_claims_report_the_exact_forms():
    f = build("slab_indicator", pg.Ball(1.0), normal=(0, 0, 1), lower=-0.5, upper=0.5)
    keys = {c.key: c for c in f.claims}
    assert keys["integral"].status == "derived" and keys["l2_norm_sq"].status == "derived" and keys["max"].status == "derived"
    assert all(c.status in ("theorem", "derived", "checked") for c in f.claims)
    g = f.normalised()
    assert g.integral()[0] == pytest.approx(1.0, rel=1e-14) and g(0.0, 0.0, 0.0) == pytest.approx(12 / (11 * math.pi))
    h = f.scaled(3.0)
    assert h.integral()[0] == pytest.approx(3 * 11 * math.pi / 12, rel=1e-14)
    assert h.exact["l2_norm_sq"][0] == pytest.approx(9 * 11 * math.pi / 12, rel=1e-14)


def test_cylinder_oblique_slab_has_no_closed_form():
    D = pg.Cylinder(1.0, (0.0, 1.0))
    f = build("slab_indicator", D, normal=(0.0, 0.6, 0.8), lower=0.1, upper=0.6)
    assert "integral" not in f.exact
    v, status, why = f.integral()
    assert status == "unknown" and "O(h)" in why                # error not bounded: not "checked"
    est, _sigma = monte_carlo(f, D)
    assert abs(est - v) < 0.05 * v                                                 # the fallback is rough, not wrong
    ax = build("slab_indicator", D, normal=(0.0, 0.0, -1.0), lower=-0.5, upper=0.25)      # z in [-0.25, 0.5]
    assert ax.integral()[0] == pytest.approx(math.pi * 0.5, rel=1e-13)


# ------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("call, match", [
    (lambda: F.on(pg.Ball(1.0)).indicator(radius=1.5), "leaves the domain"),
    (lambda: F.on(pg.Ball(1.0)).indicator(radius=0.0), "must be positive"),
    (lambda: F.on(pg.Ball(1.0)).indicator(radius=float("nan")), "must be positive"),
    (lambda: F.on(pg.Ball(1.0)).indicator(centre=(2, 0, 0), radius=0.1), "outside the domain"),
    (lambda: F.on(pg.Ball(1.0)).indicator(centre=(0, 0), radius=0.1), "Cartesian coordinates"),
    (lambda: F.on(pg.Sphere(1.0)).indicator(radius=math.pi), "injectivity radius"),
    (lambda: F.on(pg.Torus((0, 1), (0, 1))).indicator(radius=0.6), "injectivity radius"),
    (lambda: F.on(pg.Ball(1.0)).smooth_indicator(radius=0.5, inner=0.5), "inner < radius"),
    (lambda: F.on(pg.Ball(1.0)).smooth_indicator(radius=0.5, inner=-0.1), "inner < radius"),
    (lambda: F.on(pg.Ball(1.0)).smooth_indicator(radius=1.0), "leaves the domain"),
    (lambda: F.on(pg.Box((0, 1), (0, 1))).box_indicator(lower=(0.2, 0.2), upper=(0.2, 0.9)), "lower < upper"),
    (lambda: F.on(pg.Box((0, 1), (0, 1))).box_indicator(lower=(-0.1, 0.2), upper=(0.5, 0.9)), "leaves the domain"),
    (lambda: F.on(pg.Box((0, 1), (0, 1))).box_indicator(lower=(0.1,), upper=(0.5, 0.9)), "needs 2 finite"),
    (lambda: F.on(pg.Disk(1.0)).band_indicator(lower=0.5, upper=0.5), "lower < upper"),
    (lambda: F.on(pg.Disk(1.0)).band_indicator(lower=0.5, upper=1.5), "must lie in"),
    (lambda: F.on(pg.Shell(0.5, 1.0)).band_indicator(lower=0.2, upper=0.7), "must lie in"),
    (lambda: F.on(pg.Sphere(1.0)).band_indicator(lower=-2.0, upper=0.0), "radians"),
    (lambda: F.on(pg.Sphere(1.0)).band_indicator(lower=0.3, upper=0.1), "lower < upper"),
    (lambda: F.on(pg.Ball(1.0)).slab_indicator(normal=(0, 0, 2), lower=0, upper=0.5), "unit vector"),
    (lambda: F.on(pg.Ball(1.0)).slab_indicator(normal=(0, 1), lower=0, upper=0.5), "3 finite"),
    (lambda: F.on(pg.Ball(1.0)).slab_indicator(normal=(0, 0, 1), lower=0.5, upper=0.2), "lower < upper"),
    (lambda: F.on(pg.Ball(1.0)).slab_indicator(normal=(0, 0, 1), lower=1.0, upper=2.0), "does not meet the domain"),
    (lambda: F.ball.band_indicator(pg.Box((0, 1), (0, 1))), "not defined on a box"),
    (lambda: F.box.box_indicator(pg.Disk(1.0)), "not defined on a disk"),
])
def test_refusals(call, match):
    with pytest.raises(pg.Refused, match=match):
        call()


def test_registered_names_and_kinds():
    reg = F.entries()
    assert reg["indicator"].kinds == ("interval", "box", "disk", "ball", "shell", "sphere", "circle", "cylinder", "torus")
    assert reg["box_indicator"].kinds == ("interval", "box")
    assert reg["band_indicator"].kinds == ("disk", "ball", "shell", "sphere")
    assert reg["slab_indicator"].kinds == ("box", "disk", "ball", "cylinder")
    assert set(reg["smooth_indicator"].kinds) == set(reg["indicator"].kinds)


# ------------------------------------------------------------------------------------------ smooth_indicator
def test_psi_identities():
    t = np.linspace(-0.5, 1.5, 2001)
    p = _psi(t)
    assert np.all(p[t <= 0] == 0) and np.all(p[t >= 1] == 1)
    assert np.all(np.diff(p) >= 0) and _psi(0.5) == pytest.approx(0.5, abs=1e-15)
    tt = np.linspace(0, 1, 1001)
    assert np.max(np.abs(_psi(tt) + _psi(1 - tt) - 1)) < 1e-15                      # sum to one
    assert _psi(0.01) < 1e-40 and 1 - _psi(0.99) < 1e-40                            # flat to all orders at the ends
    assert _psi(np.array(0.3)).shape == () and _psi(np.array([[0.2, 0.9]])).shape == (1, 2)


def test_smooth_indicator_values_and_flat_ends():
    D = pg.Ball(1.0)
    f = build("smooth_indicator", D, radius=0.6, inner=0.2)
    assert f(0, 0, 0) == 1.0 and f(0.2, 0, 0) == 1.0 and f(0, 0.1, 0) == 1.0 and f(0.6, 0, 0) == 0.0 and f(0.7, 0, 0) == 0.0
    assert f(0.4, 0, 0) == pytest.approx(0.5, abs=1e-15)                              # the middle radius
    r = np.linspace(0.2, 0.6, 41)
    v = f(r, 0 * r, 0 * r)
    assert np.all(np.diff(v) <= 0) and np.all(np.diff(v[5:-5]) < 0)                    # decreasing on the transition
    assert np.max(np.abs(v + v[::-1] - 1)) < 1e-14                                     # point-symmetric about the middle
    assert f(0.2 + 1e-3, 0, 0) > 1 - 1e-12 and f(0.6 - 1e-3, 0, 0) < 1e-12
    assert f.support.kind == "interior" and f.regularity.order == math.inf
    assert f.regularity.vanishes_at_boundary == "all orders"


def test_smooth_indicator_r0_limit_and_gradient():
    D = pg.Disk(1.0)
    f = build("smooth_indicator", D, radius=0.5, inner=0.0)                           # no plateau: peak only at the centre
    g = f.gradient(np.array([0.0, 0.3]), np.array([0.0, 0.1]))
    assert np.all(np.isfinite(g)) and np.all(g[0] == 0.0) and f(0.0, 0.0) == 1.0     # the gradient at r = 0 is 0, not NaN
    h = build("smooth_indicator", D, radius=0.5, inner=0.2)
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.19, 0.0]])
    assert np.all(h.gradient(*X.T) == 0.0)                                              # on the plateau
    assert np.all(h.gradient(0.5, 0.0) == 0.0) and np.all(h.gradient(0.7, 0.1) == 0.0)  # at and beyond the edge
    assert h.gradient(0.2 + 0.15, 0.0)[0] < 0                                           # falling outwards


def test_smooth_indicator_integral_between_the_two_balls():
    for D in (pg.Disk(1.0), pg.Ball(1.0), pg.Sphere(1.0), pg.Circle(1.0), Interval01()):
        k = geo.kind(D)
        rad = {"disk": 0.5, "ball": 0.5, "interval": 0.4}.get(k, 0.9)
        inner = 0.5 * rad
        f = build("smooth_indicator", D, radius=rad, inner=inner)
        lo = build("indicator", D, radius=inner).integral()[0]
        hi = build("indicator", D, radius=rad).integral()[0]
        assert lo < f.integral()[0] < hi
        # the middle radius has psi = 1/2 and the profile is point-symmetric about it: the integral is exact in the mean
        f2 = build("smooth_indicator", D, radius=rad, inner=inner * 0.999)
        assert abs(f2.integral()[0] - f.integral()[0]) < 0.01 * hi
        est, sigma = monte_carlo(f, D)
        assert abs(est - f.integral()[0]) <= 5 * sigma + 1e-12


def test_smooth_indicator_tends_to_the_sharp_one():
    D = pg.Ball(1.0)
    sharp = build("indicator", D, radius=0.5).integral()[0]
    for eps, tol in ((1e-1, 0.3), (1e-2, 0.05), (1e-3, 0.006)):
        v = build("smooth_indicator", D, radius=0.5, inner=0.5 * (1 - eps)).integral()[0]
        assert 0 < sharp - v < tol * sharp


def test_normalised_smooth_indicator_has_unit_mass():
    D = pg.Disk(1.0)
    f = build("smooth_indicator", D, radius=0.6, inner=0.2).normalised()
    X, w = geo.quadrature(D, 200)
    assert w @ f(*X.T) == pytest.approx(1.0, rel=1e-6)
