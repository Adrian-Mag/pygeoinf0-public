"""Curve, cylinder and torus families of the bank (plan 021 P1, work package I): what the contract cannot see, with
independent references (scipy quadrature, direct sums, the functions' own values)."""
import math

import numpy as np
import pytest
from scipy import integrate

import pygeoinf0 as pg
from pygeoinf0 import quadrature
import pygeoinf0.functions as F
from pygeoinf0.domains.curves import Curve

TWO_PI = 2 * math.pi


def bump1(r, radius, k):
    t = abs(r) / radius
    return math.exp(-k / (1 - t * t)) if t < 1 else 0.0


def curves():
    return {"segment": pg.Segment((0.0, 0.0), (1.0, 0.5)),
            "polyline": pg.Polyline([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
            "arc": pg.Arc(pg.Sphere(1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            "spatial polyline": pg.Polyline([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])}


def circle_curve():
    return Curve(lambda t: (np.cos(t), np.sin(t)), 0.0, TWO_PI, derivative=lambda t: (-np.sin(t), np.cos(t)),
                 closed=True)


def on_curve(D, s):
    return D.to_cartesian(np.asarray(s, float))


# ----------------------------------------------------------------------------------------------- curve_bump
@pytest.mark.parametrize("name", list(curves()))
def test_curve_bump_is_the_arclength_profile(name):
    D = curves()[name]
    L = D.volume
    f = F.on(D).curve_bump(centre=0.5 * L, radius=0.3 * L, k=1.5)       # the polyline's vertex is inside the support
    s = np.linspace(0.0, L, 301)
    expected = np.array([bump1(v - 0.5 * L, 0.3 * L, 1.5) for v in s])
    assert np.allclose(f(*on_curve(D, s)), expected, rtol=0, atol=1e-13)
    assert f.support.kind == "interior" and f.regularity.order == math.inf
    # the maximum exp(-k) at the centre; symmetric in s about it
    assert np.isclose(f(*on_curve(D, [0.5 * L]))[0], math.exp(-1.5), rtol=1e-13)
    d = np.linspace(0.0, 0.29 * L, 9)
    assert np.allclose(f(*on_curve(D, 0.5 * L + d)), f(*on_curve(D, 0.5 * L - d)), rtol=1e-12)
    assert f.closed["max"][0] == pytest.approx(math.exp(-1.5))
    assert np.max(f(*on_curve(D, s))) <= f.closed["max"][0] * (1 + 1e-13)


@pytest.mark.parametrize("name", list(curves()))
def test_curve_bump_closed_forms_against_scipy(name):
    D = curves()[name]
    L = D.volume
    for centre, radius, k in ((0.5 * L, 0.45 * L, 0.7), (0.3 * L, 0.2 * L, 2.0)):
        f = F.on(D).curve_bump(centre=centre, radius=radius, k=k)
        I = integrate.quad(lambda r, radius=radius, k=k: bump1(r, radius, k), -radius, radius, epsabs=1e-15, epsrel=1e-13)[0]
        I2 = integrate.quad(lambda r, radius=radius, k=k: bump1(r, radius, k) ** 2, -radius, radius, epsabs=1e-15, epsrel=1e-13)[0]
        assert f.closed["integral"][0] == pytest.approx(I, rel=1e-12)
        assert f.closed["l2_norm_sq"][0] == pytest.approx(I2, rel=1e-12)
        # ... and against the domain's own quadrature of the function (arc-length measure, vertices as breaks)
        X, w = quadrature.rule(D, 400)
        assert w @ f(*X.T) == pytest.approx(I, rel=1e-8)


def test_curve_bump_support_and_off_curve_points():
    D = curves()["polyline"]
    f = F.on(D).curve_bump(centre=0.5, radius=0.4)
    s = np.array([0.0, 0.05, 0.09, 0.11, 0.9, 1.0, 1.3, 1.99])
    assert np.all(f(*on_curve(D, s))[(s < 0.1) | (s > 0.9)] == 0.0)
    assert np.all(f(*on_curve(D, s))[(s > 0.1) & (s < 0.9)] > 0)
    # a point off the curve: undefined, not zero
    assert np.isnan(f(0.5, 0.2))
    assert np.array_equal(f.support.contains(*on_curve(D, s)), np.abs(s - 0.5) <= 0.4)
    # the support centre is the Cartesian point of the centre
    assert np.allclose(f.support.centre, np.ravel(on_curve(D, [0.5])))


def test_curve_bump_defaults_and_arc_length_on_a_sphere():
    A = curves()["arc"]                                                  # length pi/2
    f = F.on(A).curve_bump()
    assert f.params["centre"] == pytest.approx(math.pi / 4) and f.params["radius"] == pytest.approx(math.pi / 8)
    # the point at arc length s of the unit-sphere arc is (cos s, sin s, 0)
    s0 = math.pi / 4 + 0.1
    assert f(math.cos(s0), math.sin(s0), 0.0) == pytest.approx(bump1(0.1, math.pi / 8, 1.0), rel=1e-12)
    assert f(math.cos(0.3), math.sin(0.3), 0.0) == 0.0                   # s = 0.3 is outside |s - pi/4| < pi/8


def test_curve_bump_on_a_closed_curve_wraps_through_the_seam():
    C = circle_curve()                                                   # length 2 pi, s = angle
    f = F.on(C).curve_bump(centre=0.2, radius=1.0, k=1.0)                # the support crosses s = 0 = 2 pi
    for ang in (-0.5, -0.1, 0.0, 0.3, 0.9, 1.1, 2.5):
        s = ang % TWO_PI
        assert f(math.cos(ang), math.sin(ang)) == pytest.approx(bump1(ang - 0.2, 1.0, 1.0), abs=1e-12), s
    I = integrate.quad(lambda r: bump1(r, 1.0, 1.0), -1.0, 1.0, epsabs=1e-15)[0]
    assert f.closed["integral"][0] == pytest.approx(I, rel=1e-12)
    X, w = quadrature.rule(C, 1024)                                            # equispaced; the edge is C-infinity, not analytic
    assert w @ f(*X.T) == pytest.approx(I, rel=1e-8)
    with pytest.raises(pg.Refused, match="leaves the curve"):
        F.on(C).curve_bump(centre=1.0, radius=3.2)                       # a bump wider than half the period


@pytest.mark.parametrize("bad, match", [
    ({"radius": 0.0}, "radius must be positive"), ({"radius": -1.0}, "radius must be positive"),
    ({"radius": 0.6}, "leaves the curve"), ({"centre": 0.1, "radius": 0.2}, "leaves the curve"),
    ({"k": 0.0}, "k must be positive"), ({"k": float("nan")}, "k must be positive and finite"),
    ({"radius": "wide"}, "radius must be a number"), ({"centre": 1.5}, "arc length in"),
    ({"centre": -0.1}, "arc length in"), ({"radius": float("inf")}, "radius must be positive and finite")])
def test_curve_bump_refusals(bad, match):
    D = pg.Segment((0.0, 0.0), (1.0, 0.0))
    params = {"centre": 0.5, "radius": 0.3, **bad}
    with pytest.raises(pg.Refused, match=match):
        F.on(D).curve_bump(**params)


def test_curve_families_take_keywords_only():
    D = pg.Segment((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(TypeError):
        F.curve.curve_bump(D, 0.5, 0.2)
    with pytest.raises(TypeError):
        F.curve.curve_gaussian(D, 0.5, 0.2)
    with pytest.raises(TypeError):
        F.curve.curve_bump(D, width=0.1)                                 # no silent acceptance of foreign parameters
    with pytest.raises(pg.Refused, match="not defined on a ball"):
        F.entries()["curve_bump"].builder(pg.Ball(1.0))


# ------------------------------------------------------------------------------------------- curve_gaussian
@pytest.mark.parametrize("name", list(curves()))
def test_curve_gaussian_values_and_closed_forms(name):
    D = curves()[name]
    L = D.volume
    for centre, width in ((0.5 * L, 0.1 * L), (0.0, 0.2 * L), (L, 0.05 * L), (0.7 * L, 3.0 * L)):
        f = F.on(D).curve_gaussian(centre=centre, width=width)
        s = np.linspace(0.0, L, 101)
        assert np.allclose(f(*on_curve(D, s)), np.exp(-(s - centre) ** 2 / (2 * width ** 2)), rtol=1e-13)
        I = integrate.quad(lambda u, centre=centre, width=width: math.exp(-(u - centre) ** 2 / (2 * width ** 2)), 0, L, epsabs=1e-15,
                           epsrel=1e-13, points=[centre])[0]
        I2 = integrate.quad(lambda u, centre=centre, width=width: math.exp(-(u - centre) ** 2 / width ** 2), 0, L, epsabs=1e-15, epsrel=1e-13,
                            points=[centre])[0]
        assert f.closed["integral"][0] == pytest.approx(I, rel=1e-11)
        assert f.closed["l2_norm_sq"][0] == pytest.approx(I2, rel=1e-11)
    assert f.support.kind == "full" and f.regularity.vanishes_at_boundary is None


def test_curve_gaussian_normalised_has_unit_mass():
    D = curves()["polyline"]
    g = F.on(D).curve_gaussian(centre=0.8, width=0.2).normalised()
    X, w = quadrature.rule(D, 300)
    assert w @ g(*X.T) == pytest.approx(1.0, rel=1e-9)
    assert g.integral()[0] == pytest.approx(1.0)


def test_curve_gaussian_on_a_closed_curve_is_the_periodic_sum():
    C = circle_curve()
    w, c = 0.7, 0.4                                                      # the images across the seam matter
    f = F.on(C).curve_gaussian(centre=c, width=w)
    ang = np.linspace(-3.0, 3.0, 25)
    ref = sum(np.exp(-((ang % TWO_PI) - c + n * TWO_PI) ** 2 / (2 * w * w)) for n in range(-6, 7))
    assert np.allclose(f(np.cos(ang), np.sin(ang)), ref, rtol=1e-12)
    # periodic: the same value at s = 0 and s = 2 pi, and the mass of the full line
    assert f(1.0, 0.0) == pytest.approx(ref.dtype.type(sum(math.exp(-(-c + n * TWO_PI) ** 2 / (2 * w * w))
                                                          for n in range(-6, 7))), rel=1e-12)
    assert f.closed["integral"][0] == pytest.approx(w * math.sqrt(TWO_PI), rel=1e-12)
    overlap = 1 + 2 * sum(math.exp(-(m * TWO_PI) ** 2 / (4 * w * w)) for m in range(1, 4))     # images overlap
    assert f.closed["l2_norm_sq"][0] == pytest.approx(w * math.sqrt(math.pi) * overlap, rel=1e-12)
    X, wq = quadrature.rule(C, 200)
    assert wq @ f(*X.T) == pytest.approx(w * math.sqrt(TWO_PI), rel=1e-10)
    # a width of the order of the period: still the sum over all images (checked directly)
    g = F.on(C).curve_gaussian(centre=1.0, width=5.0)
    gref = sum(math.exp(-(2.0 - 1.0 + n * TWO_PI) ** 2 / 50.0) for n in range(-40, 41))
    assert g(math.cos(2.0), math.sin(2.0)) == pytest.approx(gref, rel=1e-12)
    assert g.closed["integral"][0] == pytest.approx(5.0 * math.sqrt(TWO_PI), rel=1e-6)


@pytest.mark.parametrize("bad, match", [({"width": 0.0}, "width must be positive"), ({"width": -1.0}, "positive"),
                                         ({"width": float("nan")}, "finite"), ({"centre": 2.0}, "arc length in"),
                                         ({"width": None, "centre": "mid"}, "centre must be a number")])
def test_curve_gaussian_refusals(bad, match):
    with pytest.raises(pg.Refused, match=match):
        F.on(pg.Segment((0.0, 0.0), (1.0, 0.0))).curve_gaussian(**bad)
    with pytest.raises(pg.Refused, match="exceeds 10"):
        F.on(circle_curve()).curve_gaussian(width=100.0)


def test_curve_functions_are_members_of_l2_and_the_claims_are_stated():
    D = curves()["polyline"]
    f = F.on(D).curve_bump(centre=1.0, radius=0.5)
    assert f.element(pg.L2(D)) is not None
    assert f.claims.find("regularity").status == "theorem" and "Hormander" in f.claims.find("regularity").reason
    H2 = pg.Sobolev(D, 2, length_scale=0.2)
    assert f.element(H2) is not None                                     # C-infinity, support away from the ends


# ------------------------------------------------------------------------------------------ cylinder_product
def cyl():
    return pg.Cylinder(2.0, (-1.0, 3.0), centre=(0.5, -0.5))


def test_cylinder_product_is_the_product_of_the_two_profiles():
    D = cyl()
    f = F.on(D).cylinder_product(centre=(1.0, -0.5, 1.5), radius=0.9, half_height=1.2, k_radial=2.0, k_axial=0.6)
    rng = np.random.default_rng(3)
    x, y, z = 1.0 + 2.5 * rng.uniform(-1, 1, 200), -0.5 + 2.5 * rng.uniform(-1, 1, 200), rng.uniform(-1, 3, 200)
    ref = np.array([bump1(math.hypot(a - 1.0, b + 0.5), 0.9, 2.0) * bump1(c - 1.5, 1.2, 0.6)
                    for a, b, c in zip(x, y, z)])
    assert np.allclose(f(x, y, z), ref, rtol=1e-12, atol=1e-15)
    assert f(1.0, -0.5, 1.5) == pytest.approx(math.exp(-2.6), rel=1e-13)       # the maximum exp(-kr - kz)
    assert f.closed["max"][0] == pytest.approx(math.exp(-2.6))


def test_cylinder_product_is_rotationally_symmetric_about_its_axis_and_vanishes_outside():
    D = cyl()
    f = F.on(D).cylinder_product(centre=(1.0, 0.0, 1.0), radius=0.8, half_height=0.9)
    ph = np.linspace(0, TWO_PI, 17)
    for r, z in ((0.3, 1.2), (0.7, 0.4), (0.79, 1.8)):
        vals = f(1.0 + r * np.cos(ph), 0.0 + r * np.sin(ph), np.full_like(ph, z))
        assert np.ptp(vals) <= 1e-9 * vals.max() and vals.max() > 0
    assert f(1.8, 0.0, 1.0) == 0.0 and f(1.81, 0.0, 1.0) == 0.0 and f(1.0, 0.0, 1.9) == 0.0 and f(1.0, 0.0, 0.05) == 0.0
    assert f.support.contains(1.7, 0.0, 1.5) and not f.support.contains(1.9, 0.0, 1.0)
    assert not f.support.contains(1.0, 0.0, 2.0)


def test_cylinder_product_support_stays_in_the_cylinder_and_vanishes_to_all_orders_at_the_boundary():
    D = cyl()
    f = F.on(D).cylinder_product(centre=(1.2, -0.4, 0.5), radius=1.25, half_height=1.4)   # nearly touches the wall
    th = np.linspace(0, TWO_PI, 200)
    wall = (D.centre[0] + 2.0 * np.cos(th), D.centre[1] + 2.0 * np.sin(th))
    for z in (-1.0, 0.0, 1.5, 3.0):
        assert np.all(f(*wall, np.full_like(th, z)) == 0.0)
    assert f(1.0, 0.0, -1.0) == 0.0 and f(1.0, 0.0, 3.0) == 0.0
    # the profile and its first derivatives are tiny just inside the edge (all orders vanish): checked for orders 0-1
    e = 1e-3
    near = f(1.2 + 1.25 * (1 - e), -0.4, 0.5)
    assert near < 1e-100 and f.regularity.vanishes_at_boundary == "all orders"
    assert np.all(np.abs(f.gradient(1.2 + 1.25 * (1 - e), -0.4, 0.5)) < 1e-90)


def test_cylinder_product_closed_forms_by_cylindrical_quadrature_of_the_function():
    D = cyl()
    for params in ({"centre": (1.0, -0.5, 1.5), "radius": 0.9, "half_height": 1.2, "k_radial": 2.0, "k_axial": 0.6},
                   {"radius": 1.7, "half_height": 1.8}, {"centre": (0.0, 0.3, 2.0), "radius": 0.5, "half_height": 0.9}):
        f = F.on(D).cylinder_product(**params)
        cx, cy, zc = f.params["centre"]
        rho, h = f.params["radius"], f.params["half_height"]
        t, wt = np.polynomial.legendre.leggauss(160)
        r, wr = 0.5 * rho * (t + 1), 0.5 * rho * wt
        zz, wz = zc + h * t, h * wt
        R, Z = np.meshgrid(r, zz, indexing="ij")
        ph = np.linspace(0, TWO_PI, 6, endpoint=False)               # f is symmetric in phi: any phi samples agree
        vals = f(cx + R * np.cos(ph[1]), cy + R * np.sin(ph[1]), Z)
        vals0 = f(cx + R * np.cos(ph[3]), cy + R * np.sin(ph[3]), Z)
        assert np.allclose(vals, vals0, rtol=1e-12)
        I = TWO_PI * np.einsum("i,ij,j->", wr * r, vals, wz)
        I2 = TWO_PI * np.einsum("i,ij,j->", wr * r, vals ** 2, wz)
        assert f.closed["integral"][0] == pytest.approx(I, rel=1e-9)
        assert f.closed["l2_norm_sq"][0] == pytest.approx(I2, rel=1e-9)
        assert f.closed["max"][0] == pytest.approx(f(cx, cy, zc))


def test_cylinder_product_gradient_off_axis():
    D = cyl()
    f = F.on(D).cylinder_product(centre=(1.0, -0.5, 1.5), radius=0.9, half_height=1.2, k_radial=2.0, k_axial=0.6)
    P = np.array([[1.3, -0.2, 1.0], [0.7, -0.9, 2.1], [1.0, -0.5, 1.5], [1.5, -0.5, 1.4]])
    G = f.gradient(*P.T)
    h = 1e-6
    for i in range(3):
        E = np.zeros(3)
        E[i] = h
        fd = (f(*(P + E).T) - f(*(P - E).T)) / (2 * h)
        assert np.allclose(fd, G[:, i], atol=1e-8)
    assert np.allclose(f.gradient(1.0, -0.5, 1.5), 0.0, atol=1e-14)       # a critical point at the centre (r = 0 limit)
    assert np.all(f.gradient(3.0, 0.0, 1.0) == 0.0)


@pytest.mark.parametrize("bad, match", [
    ({"radius": 2.0}, "leaves the cylinder"), ({"half_height": 2.0}, "half_height 2 leaves"),
    ({"centre": (0.5, -0.5, 2.9), "half_height": 0.5}, "half_height 0.5 leaves"),
    ({"centre": (2.4, -0.5, 1.0), "radius": 0.5}, "leaves the cylinder"), ({"centre": (3.0, 0.0, 1.0)}, "not inside"),
    ({"centre": (0.5, -0.5, 3.5)}, "not inside"), ({"centre": (0.5, -0.5)}, "3 finite Cartesian"),
    ({"centre": (0.5, float("nan"), 1.0)}, "finite Cartesian"), ({"radius": -0.1}, "radius must be positive"),
    ({"half_height": 0.0}, "half_height must be positive"), ({"k_radial": 0.0}, "k_radial must be positive"),
    ({"k_axial": -1.0}, "k_axial must be positive"), ({"k_axial": "x"}, "k_axial must be a number")])
def test_cylinder_product_refusals(bad, match):
    with pytest.raises(pg.Refused, match=match):
        F.on(cyl()).cylinder_product(**bad)


def test_cylinder_product_is_an_element_of_l2_and_h2():
    D = pg.Cylinder(1.0, (0.0, 1.0))
    f = F.on(D).cylinder_product(radius=0.6, half_height=0.3)
    assert f.element(pg.L2(D)) is not None
    assert f.regularity.order == math.inf and f.claims.find("regularity").status == "theorem"


# ---------------------------------------------------------------------------------------------- torus_gaussian
def torus3():
    return pg.Torus((-1.0, 3.0), (0.0, 2.0), (1.0, 4.0))


def theta(w, L, c, x, N=8):
    return sum(math.exp(-((x - c) + n * L) ** 2 / (2 * w * w)) for n in range(-N, N + 1))


def test_torus_gaussian_is_periodic_and_peaks_at_the_centre():
    D = torus3()
    c = (0.5, 1.7, 2.0)
    f = F.on(D).torus_gaussian(centre=c, width=0.4)
    rng = np.random.default_rng(5)
    P = np.stack([rng.uniform(-1, 3, 50), rng.uniform(0, 2, 50), rng.uniform(1, 4, 50)], axis=1)
    L = np.array([4.0, 2.0, 3.0])
    base = f(*P.T)
    for i in range(3):
        shift = np.zeros(3)
        shift[i] = L[i]
        assert np.allclose(f(*(P + shift).T), base, rtol=1e-14, atol=1e-16)
        assert np.allclose(f(*(P - 2 * shift).T), base, rtol=1e-14, atol=1e-16)
    ref = np.array([theta(0.4, 4.0, 0.5, p[0]) * theta(0.4, 2.0, 1.7, p[1]) * theta(0.4, 3.0, 2.0, p[2]) for p in P])
    assert np.allclose(base, ref, rtol=1e-13)
    fc = f(*c)
    peak = theta(0.4, 4.0, 0, 0) * theta(0.4, 2.0, 0, 0) * theta(0.4, 3.0, 0, 0)
    assert fc == pytest.approx(peak, rel=1e-13) and np.all(base <= peak * (1 + 1e-13))


def test_torus_gaussian_wraps_through_the_seam_and_reduces_the_centre():
    D = pg.Torus((0.0, TWO_PI), (0.0, TWO_PI))
    f = F.on(D).torus_gaussian(centre=(0.05, TWO_PI - 0.05), width=0.3)
    # the points x = 6.2 and x = 0.05 + (0.05 - ...) are at the same minimum-image distance from the centre
    assert f(TWO_PI - 0.1, 0.1) == pytest.approx(f(0.2, TWO_PI - 0.2), rel=1e-12)
    d = 0.15      # across the seam in x
    assert f(TWO_PI - 0.1, TWO_PI - 0.05) == pytest.approx(math.exp(-d * d / (2 * 0.09)) * theta(0.3, TWO_PI, 0, 0),
                                                           rel=1e-12)
    g = F.on(D).torus_gaussian(centre=(0.05 + TWO_PI, -0.05 - 2 * TWO_PI), width=0.3)
    assert g.params["centre"] == pytest.approx((0.05, TWO_PI - 0.05), abs=1e-12)
    assert np.allclose(g(1.0, 2.0), F.on(D).torus_gaussian(centre=(0.05, TWO_PI - 0.05), width=0.3)(1.0, 2.0))


def test_torus_gaussian_mass_is_the_full_space_gaussian_integral():
    D = torus3()
    for w in (0.1, 0.4, 0.9):
        f = F.on(D).torus_gaussian(centre=(0.5, 1.7, 2.0), width=w)
        assert f.closed["integral"][0] == pytest.approx((TWO_PI) ** 1.5 * w ** 3, rel=1e-12)
        if w <= 0.1:        # the images of the squares overlap otherwise (exp(-L^2/(4 w^2)) with L = 2)
            assert f.closed["l2_norm_sq"][0] == pytest.approx(math.pi ** 1.5 * w ** 3, rel=1e-12)
    # a width that sees the images: the sum over all images integrates to the same full-space value
    f = F.on(D).torus_gaussian(centre=(0.5, 1.7, 2.0), width=1.5)
    assert f.closed["integral"][0] == pytest.approx(TWO_PI ** 1.5 * 1.5 ** 3, rel=1e-9)
    # and against the torus quadrature of the function itself (equispaced: exact for the trigonometric part)
    g = F.on(D).torus_gaussian(centre=(0.5, 1.7, 2.0), width=0.5)
    X, wq = quadrature.rule(D, 48)
    assert wq @ g(*X.T) == pytest.approx(g.closed["integral"][0], rel=1e-10)
    assert wq @ (g(*X.T) ** 2) == pytest.approx(g.closed["l2_norm_sq"][0], rel=1e-10)


def test_torus_gaussian_normalised_and_gradient_in_three_dimensions():
    D = torus3()
    f = F.on(D).torus_gaussian(centre=(0.5, 1.7, 2.0), width=0.6)
    g = f.normalised()
    X, wq = quadrature.rule(D, 48)
    assert wq @ g(*X.T) == pytest.approx(1.0, rel=1e-10)
    rng = np.random.default_rng(8)
    P = np.stack([rng.uniform(-1, 3, 12), rng.uniform(0, 2, 12), rng.uniform(1, 4, 12)], axis=1)
    G = f.gradient(*P.T)
    h = 1e-6
    for i in range(3):
        E = np.zeros(3)
        E[i] = h
        fd = (f(*(P + E).T) - f(*(P - E).T)) / (2 * h)
        assert np.allclose(fd, G[:, i], atol=1e-8)
    assert np.allclose(f.gradient(0.5, 1.7, 2.0), 0.0, atol=1e-13)


def test_torus_gaussian_defaults_and_claims():
    D = pg.Torus((0.0, 2.0), (0.0, 3.0))
    f = F.on(D).torus_gaussian()
    assert f.params["width"] == pytest.approx(0.2) and f.params["centre"] == pytest.approx((1.0, 1.5))
    assert f.support.kind == "full" and f.regularity.order == math.inf
    assert f.element(pg.L2(D)) is not None
    assert "integral" in f.closed and "l2_norm_sq" in f.closed and f.closed["integral"][1] == "derived"


@pytest.mark.parametrize("bad, match", [
    ({"width": 0.0}, "width must be positive"), ({"width": float("inf")}, "finite"), ({"width": 21.0}, "exceeds 10"),
    ({"centre": (1.0,)}, "2 finite Cartesian"), ({"centre": (1.0, float("nan"))}, "finite Cartesian"),
    ({"width": "thin"}, "width must be a number")])
def test_torus_gaussian_refusals(bad, match):
    with pytest.raises(pg.Refused, match=match):
        F.on(pg.Torus((0.0, 2.0), (0.0, 3.0))).torus_gaussian(**bad)
    with pytest.raises(pg.Refused, match="not defined on a disk"):
        F.entries()["torus_gaussian"].builder(pg.Disk(1.0))


def test_torus_bump_is_the_bump_family_and_the_catalogue_lists_the_new_families():
    D = pg.Torus((0.0, TWO_PI), (0.0, TWO_PI))
    assert "torus" in F.entries()["bump"].kinds and "torus_bump" not in F.entries()
    cat = {n: kinds for n, kinds, _ in F.catalogue()}
    assert cat["curve_bump"] == ("curve",) and cat["curve_gaussian"] == ("curve",)
    assert cat["cylinder_product"] == ("cylinder",) and cat["torus_gaussian"] == ("torus",)
    assert F.on(D).bump(radius=1.0).family == "bump"


# ------------------------------------------------------------------ review fixes: steep edge, full periodic sum
def graded_reference(k, power=1, moment=0):
    """int_0^1 t^moment exp(-power k/(1-t^2)) dt by scipy quad panel by panel in u = 1 - t, panels [2^-(j+1), 2^-j]."""
    edges = [1.0, 0.5] + [2.0 ** -j for j in range(2, 61)]
    return sum(integrate.quad(lambda u: (1 - u) ** moment * math.exp(-power * k / (u * (2 - u))), lo, hi,
                              epsabs=0, epsrel=2.3e-14, limit=200)[0] for lo, hi in zip(edges[1:], edges[:-1]))


@pytest.mark.parametrize("k", [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 5.0, 50.0, 250.0])
@pytest.mark.parametrize("name", ["segment", "polyline", "arc"])
def test_curve_bump_closed_forms_resolve_the_steep_edge_for_small_k(name, k):
    D = curves()[name]
    L = D.volume
    f = F.on(D).curve_bump(centre=0.5 * L, radius=0.3 * L, k=k)
    assert f.closed["integral"][0] == pytest.approx(2 * 0.3 * L * graded_reference(k), rel=1e-12)
    assert f.closed["l2_norm_sq"][0] == pytest.approx(2 * 0.3 * L * graded_reference(k, power=2), rel=1e-12)
    assert f.closed["max"][0] == pytest.approx(math.exp(-k), rel=1e-15)


def test_curve_bump_small_k_pins_values_that_the_uniform_rule_got_wrong():
    # the previous uniform 8 x 64 Gauss-Legendre rule gave relative errors 2.3e-7 at k = 1e-3 and 1.7e-6 at k = 1e-6
    f = F.on(curves()["segment"]).curve_bump(centre=0.5 * curves()["segment"].volume, radius=0.3 * curves()["segment"].volume,
                                             k=1e-6)
    ref = 2 * 0.3 * curves()["segment"].volume * graded_reference(1e-6)
    assert abs(f.closed["integral"][0] / ref - 1) < 1e-13
    # for k -> 0 the profile tends to the indicator of the open ball: the integral to 2 radius
    g = F.on(pg.Segment((0.0, 0.0), (1.0, 0.0))).curve_bump(centre=0.5, radius=0.3, k=1e-12)
    assert g.closed["integral"][0] == pytest.approx(0.6, rel=1e-10)


def test_curve_bump_closed_forms_are_not_stated_where_exp_of_minus_2k_underflows():
    D = pg.Segment((0.0, 0.0), (1.0, 0.0))
    f = F.on(D).curve_bump(centre=0.5, radius=0.3, k=400.0)
    assert "integral" not in f.closed and "l2_norm_sq" not in f.closed and f.closed["max"][0] == pytest.approx(math.exp(-400.0))
    g = F.on(D).curve_bump(centre=0.5, radius=0.3, k=300.0)
    assert "integral" in g.closed and "l2_norm_sq" in g.closed
    assert g.closed["l2_norm_sq"][0] == pytest.approx(0.6 * graded_reference(300.0, power=2), rel=1e-12)


def test_cylinder_product_closed_forms_resolve_the_steep_edges_for_small_k():
    D = cyl()
    for kr, kz in ((1e-3, 1e-3), (1e-5, 0.5), (3.0, 1e-4), (200.0, 250.0)):
        f = F.on(D).cylinder_product(radius=0.5, half_height=0.3, k_radial=kr, k_axial=kz)
        for key, p in (("integral", 1), ("l2_norm_sq", 2)):
            ref = (2 * math.pi * 0.5 ** 2 * graded_reference(kr, power=p, moment=1)) * (2 * 0.3 * graded_reference(kz, power=p))
            assert f.closed[key][0] == pytest.approx(ref, rel=1e-12)
    g = F.on(D).cylinder_product(radius=0.5, half_height=0.3, k_radial=400.0)
    assert "integral" not in g.closed and "l2_norm_sq" not in g.closed and "max" in g.closed


def periodic_gaussian(x, c, w, period, N=200):
    return sum(math.exp(-(x - c + n * period) ** 2 / (2 * w * w)) for n in range(-N, N + 1))


@pytest.mark.parametrize("w", [0.3, 0.7, 2.0, 5.0, 60.0])
def test_closed_curve_gaussian_closed_forms_are_those_of_the_full_periodic_sum(w):
    C = circle_curve()
    c = 0.4
    f = F.on(C).curve_gaussian(centre=c, width=w)
    # periodic trapezoid of the FULL sum (many images, spectrally accurate) as an independent reference
    s = np.linspace(0.0, TWO_PI, 2000, endpoint=False)
    full = np.array([periodic_gaussian(v, c, w, TWO_PI, N=300) for v in s])
    h = TWO_PI / s.size
    assert f.closed["integral"][0] == pytest.approx(h * full.sum(), rel=1e-13)
    assert f.closed["integral"][0] == pytest.approx(w * math.sqrt(TWO_PI), rel=1e-15)
    assert f.closed["l2_norm_sq"][0] == pytest.approx(h * (full ** 2).sum(), rel=1e-12)
    # the evaluation is the full sum (the arc length recovered from the point costs ~1e-15 of rounding)
    ang = np.linspace(0.0, TWO_PI, 41)[:-1] + 0.01
    ref = np.array([periodic_gaussian(a, c, w, TWO_PI, N=300) for a in ang])
    assert np.allclose(f(np.cos(ang), np.sin(ang)), ref, rtol=1e-12, atol=0)
    # and the truncation itself, as stated, is below 1e-17 absolute (in rounding terms: equal to the full sum)
    from pygeoinf0.functions.families.curves import wrapped_images, wrapped_sum
    N = wrapped_images(w, TWO_PI)
    pts = np.linspace(0.0, TWO_PI, 97)
    full_pts = np.array([periodic_gaussian(v, c, w, TWO_PI, N=300) for v in pts])
    assert np.max(np.abs(wrapped_sum(pts, c, w, TWO_PI, N) - full_pts)) <= 4e-16 * full_pts.max()


def test_closed_curve_gaussian_seam_jump_is_below_the_stated_truncation_and_regularity_claim_is_for_the_full_sum():
    C = circle_curve()
    f = F.on(C).curve_gaussian(centre=0.1, width=1.0)
    eps = 1e-9
    below = f(math.cos(TWO_PI - eps), math.sin(TWO_PI - eps))
    above = f(math.cos(eps), math.sin(eps))
    assert abs(below - above) < 1e-8                              # the function itself is continuous (no extra jump)
    # the truncation error at the seam for the stated N is below 1e-17 absolute
    N = max(1, math.ceil(9.0 * 1.0 / TWO_PI))
    full = periodic_gaussian(0.0, 0.1, 1.0, TWO_PI, N=60)
    trunc = sum(math.exp(-(0.0 - 0.1 + n * TWO_PI) ** 2 / 2.0) for n in range(-N, N + 1))
    assert abs(full - trunc) < 1e-17
    reason = f.claims.find("regularity").reason
    assert "all images" in reason and "1e-17" in reason and "truncat" in reason
    assert f.regularity.order == math.inf and "finite sum" not in reason


def test_torus_gaussian_closed_forms_are_those_of_the_full_periodic_sum_and_the_seam_claim_is_honest():
    D = pg.Torus((0.0, TWO_PI), (0.0, 3.0))
    w, c = 1.3, (0.1, 2.9)
    f = F.on(D).torus_gaussian(centre=c, width=w)
    n = 600
    xs = np.linspace(0.0, TWO_PI, n, endpoint=False)
    ys = np.linspace(0.0, 3.0, n, endpoint=False)
    fx = np.array([periodic_gaussian(v, c[0], w, TWO_PI, N=100) for v in xs])
    fy = np.array([periodic_gaussian(v, c[1], w, 3.0, N=100) for v in ys])
    ix, iy = (TWO_PI / n) * fx.sum(), (3.0 / n) * fy.sum()
    lx, ly = (TWO_PI / n) * (fx ** 2).sum(), (3.0 / n) * (fy ** 2).sum()
    assert f.closed["integral"][0] == pytest.approx(ix * iy, rel=1e-13)
    assert f.closed["integral"][0] == pytest.approx(TWO_PI * w * w, rel=1e-15)
    assert f.closed["l2_norm_sq"][0] == pytest.approx(lx * ly, rel=1e-12)
    # values: the full sum to 1e-17 per axis (absolute), and the seam jump of the evaluation is of that size only
    P = np.array([[0.0, 0.0], [TWO_PI - 1e-12, 2.999999999999], [1.0, 1.0], [5.0, 2.5]])
    ref = np.array([periodic_gaussian(p[0], c[0], w, TWO_PI) * periodic_gaussian(p[1], c[1], w, 3.0) for p in P])
    assert np.allclose(f(*P.T), ref, rtol=0, atol=1e-15)
    assert abs(f(1e-14, 1.0) - f(TWO_PI - 1e-14, 1.0)) < 1e-14
    reason = f.claims.find("regularity").reason
    assert "all images" in reason and "1e-17" in reason and "finite sum" not in reason
    assert "invisible" not in reason
    doc = F.entries()["torus_gaussian"].builder.__doc__
    assert "exactly C-infinity (a finite sum" not in doc and "full sum" in doc


def test_wrapped_l2_cut_is_accurate_for_wide_gaussians():
    from pygeoinf0.functions.families.curves import wrapped_l2
    for w, period in ((0.2, 1.0), (1.0, 1.0), (10.0, 1.0), (3.0, 0.7)):
        full = w * math.sqrt(math.pi) * sum(math.exp(-(m * period) ** 2 / (4 * w * w)) for m in range(-2000, 2001))
        assert wrapped_l2(w, period) == pytest.approx(full, rel=1e-15)
