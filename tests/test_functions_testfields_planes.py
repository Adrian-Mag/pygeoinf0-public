"""Steps and dykes across a plane (plan 021 P1, work package G): values, the volumes of a cut domain against independent
constructions (cross-section integrals, polygon clipping, Irwin-Hall, Monte Carlo), the closed forms of the sharp and
smooth versions, supports and refusals."""
import itertools
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.functions import geometry as geo
from pygeoinf0.functions.families.testfields import _below, _unit
from pygeoinf0.spaces.membership import check

T, W = np.polynomial.legendre.leggauss(60)


def sobolev(D, s):
    return pg.Sobolev(D, s, length_scale=0.2)


def gl(a, b, panels=24):
    """Composite Gauss-Legendre nodes and weights on [a, b]."""
    e = np.linspace(a, b, panels + 1)
    x = np.concatenate([0.5 * (q - p) * (T + 1) + p for p, q in itertools.pairwise(e)])
    w = np.concatenate([0.5 * (q - p) * W for p, q in itertools.pairwise(e)])
    return x, w


def clip_area(poly, n, d):
    """Area of the convex polygon (vertices in order) on the side n . x <= d (Sutherland-Hodgman, shoelace)."""
    out, m = [], len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        fa, fb = n @ a - d, n @ b - d
        if fa <= 0:
            out.append(a)
        if fa * fb < 0:
            out.append(a + (b - a) * fa / (fa - fb))
    if len(out) < 3:
        return 0.0
    x, y = np.array(out).T
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ------------------------------------------------------------------------------------------------ cut volumes
def test_cut_volume_of_a_ball_and_a_disk_against_cross_section_integrals():
    for R in (1.0, 2.5):
        B, D = pg.Ball(R), pg.Disk(R)
        for d in (-0.9 * R, -0.3 * R, 0.0, 0.4 * R, 0.95 * R):
            s, w = gl(-R, d)
            assert _below(B, np.array([0.0, 0.0, 1.0]), d) == pytest.approx(float(np.sum(w * math.pi * (R * R - s * s))),
                                                                              rel=1e-12)
            phi, wp = gl(-math.pi / 2, math.asin(d / R))                       # s = R sin(phi): 2 sqrt(R^2 - s^2) ds
            assert _below(D, np.array([1.0, 0.0]), d) == pytest.approx(float(np.sum(wp * 2 * R * R * np.cos(phi) ** 2)),
                                                                         rel=1e-12)
        assert _below(B, np.array([0.0, 0.0, 1.0]), R) == B.volume and _below(D, np.array([0.0, 1.0]), -R) == 0.0


def test_cut_volume_of_a_box_in_2d_against_polygon_clipping():
    B = pg.Box((0.0, 2.0), (0.0, 1.0))
    sq = [np.array(p, float) for p in ((0, 0), (2, 0), (2, 1), (0, 1))]
    rng = np.random.default_rng(2)
    for _ in range(40):
        n = _unit(B, rng.normal(size=2))
        d0 = float(rng.uniform(-0.9, 0.9)) * 0.5 * (abs(n[0]) * 2 + abs(n[1]))
        assert _below(B, n, d0) == pytest.approx(clip_area(sq, n, float(n @ np.array([1.0, 0.5])) + d0), abs=1e-12)


def test_cut_volume_of_a_box_in_3d_against_irwin_hall_and_axes():
    B = pg.Box((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    n = np.ones(3) / math.sqrt(3.0)
    for t in (0.4, 0.9, 1.0, 1.3, 1.9, 2.0, 2.4, 2.9):             # P(sum of three uniforms <= t)
        irwin = sum((-1) ** k * math.comb(3, k) * max(t - k, 0.0) ** 3 for k in range(4)) / 6.0
        assert _below(B, n, (t - 1.5) / math.sqrt(3.0)) == pytest.approx(irwin, abs=1e-12)
    # a coordinate normal, and a normal with one zero component (a prism over the 2d cut)
    assert _below(B, np.array([0.0, 1.0, 0.0]), -0.2) == pytest.approx(0.3)
    sq = [np.array(p, float) for p in ((0, 0), (1, 0), (1, 1), (0, 1))]
    m = np.array([1.0, 2.0, 0.0]) / math.sqrt(5.0)
    assert _below(B, m, 0.1) == pytest.approx(clip_area(sq, m[:2], float(m[:2] @ [0.5, 0.5]) + 0.1))
    assert _below(B, np.array([1.0, 1e-9, 0.0]), 0.1) == pytest.approx(0.6, abs=1e-9)    # a tiny tilt is still computed


def exact_box_below(box, n, d):
    """Inclusion-exclusion of the volume of box below n . (x - centre) <= d in rational arithmetic (a reference written
    here, on the floats n and d, as two lists of Fractions)."""
    from fractions import Fraction as Q
    L = [Q(s.b) - Q(s.a) for s in box.sides]
    m = [abs(Q(float(x))) for x in n]
    idx = [i for i in range(len(m)) if m[i] != 0]
    t = Q(float(d)) + sum(m[i] * L[i] for i in idx) / 2
    total = Q(0)
    for mask in range(1 << len(idx)):
        sel = [idx[j] for j in range(len(idx)) if mask >> j & 1]
        total += (-1) ** len(sel) * max(t - sum(m[i] * L[i] for i in sel), Q(0)) ** len(idx)
    free = Q(1)
    for i in range(len(m)):
        if i not in idx:
            free *= L[i]
    den = math.factorial(len(idx))
    for i in idx:
        den *= m[i]
    return free * total / den


@pytest.mark.parametrize("tilt", [0.0, 1e-14, 1e-9, 1e-7, 1e-6, 1.01e-6, 1e-5, 1e-4, 1e-2])
def test_box_cut_volume_stays_accurate_for_nearly_axis_aligned_normals_in_3d(tilt):
    """The reviewer's case: the float alternating sum lost 4 to 5 digits at a tilt of 1e-6."""
    B = pg.Box((0.0, 1.0), (0.0, 2.0), (-1.0, 0.5))
    n = _unit(B, (1.0, tilt, tilt))
    for d in (-0.45, 0.1, 0.4):
        exact = exact_box_below(B, n, d)
        assert _below(B, n, d) == pytest.approx(float(exact), rel=1e-13, abs=0)
    centre = np.array(geo.centre(B))
    point = centre + 0.1 * n
    dyke = F.on(B).dyke(normal=(1.0, tilt, tilt), thickness=0.2, point=tuple(point))
    p, h = float(n @ (point - centre)), 0.1
    vs = exact_box_below(B, n, p + h) - exact_box_below(B, n, p - h)
    assert dyke.closed["integral"][0] == pytest.approx(float(vs), rel=1e-13)
    assert dyke.closed["l2_norm_sq"][0] == pytest.approx(float(vs), rel=1e-13)
    step = F.on(B).step_across_plane(normal=(1.0, tilt, tilt), lower=0.0, upper=1.0, point=tuple(point))
    assert step.closed["integral"][0] == pytest.approx(float(geo.volume(B) - exact_box_below(B, n, p)), rel=1e-13)


def test_box_cut_volume_is_continuous_in_the_tilt():
    B = pg.Box((0.0, 1.0), (0.0, 2.0), (-1.0, 0.5))
    ref = _below(B, np.array([1.0, 0.0, 0.0]), 0.1)
    for tilt in (1e-12, 1e-9, 1e-6, 1e-4):
        n = _unit(B, (1.0, tilt, tilt))
        assert abs(_below(B, n, 0.1) - ref) < 10 * tilt


def test_cut_volumes_against_monte_carlo():
    rng = np.random.default_rng(7)
    N = 600_000
    cases = [(pg.Ball(1.5), 3), (pg.Disk(1.5), 2), (pg.Box((0.0, 2.0), (-1.0, 1.0), (0.0, 0.5)), 3),
             (pg.Cylinder(1.0, (0.0, 2.0)), 3), (pg.Cylinder(1.0, (0.0, 2.0)), 3)]
    for D, d in cases:
        lo, hi = np.array([[a, b] for a, b in D.bounding_box]).T
        P = rng.uniform(lo, hi, size=(N, d))
        inside = np.asarray(D._inside(*P.T), bool)
        P = P[inside]
        vol = geo.volume(D)
        for _ in range(4):
            n = _unit(D, rng.normal(size=d))
            if geo.kind(D) == "cylinder":                         # the elementary cuts: parallel and orthogonal to the axis
                n = _unit(D, np.array([n[0], n[1], 0.0]) if rng.uniform() < 0.5 else [0.0, 0.0, 1.0])
            c0 = geo.centre(D)
            dd = float(rng.uniform(-0.8, 0.8)) * 0.5 * (n @ n) * min(1.0, vol ** (1 / d))
            exact = _below(D, n, dd)
            assert exact is not None
            frac = np.mean((P - c0) @ n <= dd)
            assert exact == pytest.approx(frac * vol, abs=4 * vol * math.sqrt(0.25 / len(P)) + 1e-3 * vol)


def test_oblique_cuts_of_a_cylinder_have_no_closed_form_and_fall_back_to_the_quadrature():
    C = pg.Cylinder(1.0, (0.0, 1.0))
    f = F.on(C).step_across_plane(normal=(1.0, 0.0, 1.0), point=(0.1, 0.0, 0.5), lower=1.0, upper=2.0)
    assert "integral" not in f.closed and f.integral()[1] == "unknown" and "estimated error" in f.integral()[2]
    g = F.on(C).step_across_plane(normal=(1.0, 0.0, 1.0), lower=1.0, upper=2.0)           # through the centre
    assert g.closed["integral"][0] == pytest.approx(1.5 * C.volume)


# ------------------------------------------------------------------------------------------------ steps
def test_sharp_step_values_and_closed_forms_in_every_kind():
    cases = [(pg.Box((0.0, 2.0), (0.0, 1.0)), (1.0, 1.0), None), (pg.Disk(1.0), (0.0, 1.0), (0.0, 0.3)),
             (pg.Ball(1.0), (0.0, 0.0, 1.0), (0.0, 0.0, -0.4)), (pg.Cylinder(1.0, (0.0, 2.0)), (0.0, 0.0, 1.0), (0, 0, 0.7)),
             (pg.Cylinder(1.0, (0.0, 2.0)), (0.0, 1.0, 0.0), (0.0, 0.5, 1.0))]
    for D, normal, point in cases:
        f = F.on(D).step_across_plane(normal=normal, point=point, lower=-1.0, upper=3.0)
        n, p = np.array(f.params["normal"]), np.array(f.params["point"])
        assert np.linalg.norm(n) == pytest.approx(1.0)
        assert f(*(p + 0.1 * n)) == 3.0 and f(*(p - 0.1 * n)) == -1.0 and f(*p) == 1.0
        vm = _below(D, n, float(n @ (p - geo.centre(D))))
        V = geo.volume(D)
        assert f.closed["integral"][0] == pytest.approx(-vm + 3 * (V - vm))
        assert f.closed["l2_norm_sq"][0] == pytest.approx(vm + 9 * (V - vm))
        # independent: Monte Carlo of the values
        rng = np.random.default_rng(1)
        lo, hi = np.array([[a, b] for a, b in D.bounding_box]).T
        P = rng.uniform(lo, hi, size=(400_000, len(lo)))
        P = P[np.asarray(D._inside(*P.T), bool)]
        assert V * np.mean(f(*P.T)) == pytest.approx(f.closed["integral"][0], abs=0.02 * V)
        assert f.regularity.order == -1 and not check(f, sobolev(D, 1))[0] and check(f, pg.L2(D))[0]


def test_sharp_step_with_a_zero_side_has_the_half_space_as_support():
    B = pg.Ball(1.0)
    f = F.on(B).step_across_plane(normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.2), lower=0.0, upper=2.0)
    assert f.support.kind == "touches"
    P = np.random.default_rng(4).uniform(-0.8, 0.8, size=(300, 3))
    out = ~f.support.contains(*P.T)
    assert out.any() and np.all(f(*P[out].T) == 0.0)
    g = F.on(B).step_across_plane(normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.2), lower=-1.0, upper=0.0)
    assert np.all(g(*P[~g.support.contains(*P.T)].T) == 0.0)
    assert F.on(B).step_across_plane(normal=(0.0, 0.0, 1.0), lower=0.0, upper=2.0, width=0.1).support.kind == "full"


def test_smooth_step_axis_closed_form_against_gauss_in_the_normal_direction():
    B = pg.Box((0.0, 2.0), (0.0, 1.0), (0.0, 3.0))
    for normal, axis in (((1.0, 0.0, 0.0), 0), ((0.0, -1.0, 0.0), 1), ((0.0, 0.0, 2.0), 2)):
        f = F.on(B).step_across_plane(normal=normal, point=(0.7, 0.3, 2.2), lower=1.0, upper=4.0, width=0.17)
        a, b = B.sides[axis].a, B.sides[axis].b
        x, w = gl(a, b)
        sign = math.copysign(1.0, normal[axis])
        prof = 1.0 + 3.0 * 0.5 * (1 + np.tanh(sign * (x - (0.7, 0.3, 2.2)[axis]) / 0.17))
        others = geo.volume(B) / (b - a)
        assert f.closed["integral"][0] == pytest.approx(others * float(np.sum(w * prof)), rel=1e-12)
    C = pg.Cylinder(1.0, (0.0, 2.0))
    g = F.on(C).step_across_plane(normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, 1.5), lower=0.0, upper=1.0, width=0.2)
    z, w = gl(0.0, 2.0)
    assert g.closed["integral"][0] == pytest.approx(math.pi * float(np.sum(w * 0.5 * (1 + np.tanh((z - 1.5) / 0.2)))),
                                                    rel=1e-12)


def test_smooth_step_through_the_centre_integral_by_symmetry_and_limits():
    B = pg.Ball(1.0)
    f = F.on(B).step_across_plane(normal=(1.0, 2.0, -1.0), lower=2.0, upper=5.0, width=0.2)
    X, w = geo.quadrature(B, 60)
    assert f.closed["integral"][0] == pytest.approx(3.5 * B.volume) and w @ f(*X.T) == pytest.approx(3.5 * B.volume, rel=1e-6)
    assert "l2_norm_sq" not in f.closed
    g = F.on(B).step_across_plane(normal=(1.0, 0.0, 0.0), point=(0.2, 0.0, 0.0), width=0.2)
    assert g.closed["integral"][1] == "checked"                            # a ball cut off the centre: through the plane
    # narrow smooth step -> sharp step
    box = pg.Box((0.0, 1.0), (0.0, 1.0))
    s = F.on(box).step_across_plane(normal=(1.0, 0.0), point=(0.3, 0.5), lower=0.0, upper=1.0)
    t = F.on(box).step_across_plane(normal=(1.0, 0.0), point=(0.3, 0.5), lower=0.0, upper=1.0, width=1e-4)
    assert s.closed["integral"][0] == pytest.approx(0.7) and t.closed["integral"][0] == pytest.approx(0.7, abs=1e-6)


def test_smooth_step_regularity_and_gradient():
    D = pg.Disk(1.0)
    f = F.on(D).step_across_plane(normal=(1.0, 1.0), point=(0.1, 0.0), lower=-1.0, upper=1.0, width=0.3)
    assert f.regularity.order == math.inf and check(f, sobolev(D, 2))[0]
    assert f(0.1, 0.0) == 0.0 and f(5.0, 5.0) == pytest.approx(1.0) and f(-5.0, -5.0) == pytest.approx(-1.0)
    P = np.random.default_rng(1).uniform(-0.7, 0.7, size=(10, 2))
    h = 1e-6
    for i in range(2):
        E = np.zeros(2)
        E[i] = h
        assert np.allclose((f(*(P + E).T) - f(*(P - E).T)) / (2 * h), f.gradient(*P.T)[:, i], atol=1e-8)
    assert np.all(np.isfinite(f.gradient(np.array([1e3]), np.array([1e3]))))       # no overflow far from the plane


@pytest.mark.parametrize("params, message", [
    ({"normal": (0.0, 0.0, 0.0)}, "nonzero"), ({"normal": (1.0, 0.0)}, "3 coordinates"),
    ({"normal": (1.0, float("nan"), 0.0)}, "finite nonzero"), ({"normal": "x"}, "vector"),
    ({"normal": (0.0, 0.0, 1.0), "point": (0.0, 0.0, 2.0)}, "misses"),
    ({"normal": (0.0, 0.0, 1.0), "point": (0.0, 0.0, 1.0)}, "misses"),
    ({"normal": (0.0, 0.0, 1.0), "point": (0.0, 0.0)}, "point needs 3"),
    ({"normal": (0.0, 0.0, 1.0), "lower": 1.0, "upper": 1.0}, "no step"),
    ({"normal": (0.0, 0.0, 1.0), "width": 0.0}, "width must be positive"),
    ({"normal": (0.0, 0.0, 1.0), "width": float("inf")}, "width must be positive"),
    ({"normal": (0.0, 0.0, 1.0), "upper": float("nan")}, "upper must be finite"),
    ({"normal": (1.0, 0.0, 0.0), "point": "ab"}, "point needs 3"), ({"normal": (1.0, 0.0, 0.0), "lower": "x"}, "lower"),
    ({"normal": (1.0, 0.0, 0.0), "upper": None}, "upper"), ({"normal": (1.0, 0.0, 0.0), "width": "x"}, "width"),
    ({"normal": (1.0, 0.0, 0.0), "lower": [1.0, 2.0]}, "lower")])
def test_step_refusals(params, message):
    with pytest.raises(pg.Refused, match=message):
        F.on(pg.Ball(1.0)).step_across_plane(**params)
    with pytest.raises(TypeError):
        F.on(pg.Ball(1.0)).step_across_plane((0.0, 0.0, 1.0))


# ------------------------------------------------------------------------------------------------ dykes
def test_sharp_dyke_values_and_axis_aligned_closed_forms():
    B = pg.Box((0.0, 2.0), (0.0, 1.0))
    f = F.on(B).dyke(normal=(1.0, 0.0), thickness=0.5, point=(1.0, 0.5), amplitude=3.0, background=-1.0)
    assert f(1.0, 0.3) == 2.0 and f(1.24, 0.3) == 2.0 and f(1.3, 0.3) == -1.0 and f(0.5, 0.3) == -1.0
    assert f(1.25, 0.3) == pytest.approx(0.5)                                        # the face: the mean
    V, Vs = 2.0, 0.5
    assert f.closed["integral"][0] == pytest.approx(-1.0 * V + 3.0 * Vs)
    assert f.closed["l2_norm_sq"][0] == pytest.approx(1.0 * (V - Vs) + 4.0 * Vs)
    assert f.support.kind == "full" and f.regularity.order == -1
    g = F.on(B).dyke(normal=(1.0, 0.0), thickness=0.5, point=(1.0, 0.5))
    assert g.support.kind == "touches" and not g.support.contains(0.5, 0.2) and g.support.contains(1.1, 0.2)
    X = np.random.default_rng(0).uniform(0, 1, size=(200, 2)) * [2.0, 1.0]
    assert np.all(g(*X.T)[~g.support.contains(*X.T)] == 0.0)


def test_sharp_dyke_in_a_ball_and_a_disk_against_cross_sections():
    R = 1.3
    B, D = pg.Ball(R), pg.Disk(R)
    for off, h in ((0.0, 0.2), (0.4, 0.3), (-0.9, 0.35), (1.0, 0.5)):
        s, w = gl(max(off - h, -R), min(off + h, R))
        p3, p2 = (0.0, 0.0, off), (0.0, off)
        f = F.on(B).dyke(normal=(0.0, 0.0, 1.0), thickness=2 * h, point=p3)
        assert f.closed["integral"][0] == pytest.approx(float(np.sum(w * math.pi * (R * R - s * s))), rel=1e-12)
        g = F.on(D).dyke(normal=(0.0, 1.0), thickness=2 * h, point=p2, amplitude=2.0, background=1.0)
        area = float(np.sum(w * 2 * np.sqrt(R * R - s * s)))              # fine for these offsets (smooth integrand)
        assert g.closed["integral"][0] == pytest.approx(D.volume + 2.0 * area, rel=1e-6)


def test_dyke_oblique_in_a_box_against_clipping():
    B = pg.Box((0.0, 2.0), (0.0, 1.0))
    sq = [np.array(p, float) for p in ((0, 0), (2, 0), (2, 1), (0, 1))]
    n = np.array([1.0, 1.0]) / math.sqrt(2.0)
    f = F.on(B).dyke(normal=(1.0, 1.0), thickness=0.4, point=(1.0, 0.7))
    c = np.array([1.0, 0.5])
    d0 = float(n @ (np.array([1.0, 0.7]) - c))
    area = clip_area(sq, n, float(n @ c) + d0 + 0.2) - clip_area(sq, n, float(n @ c) + d0 - 0.2)
    assert f.closed["integral"][0] == pytest.approx(area, abs=1e-12)


def test_smooth_dyke_closed_form_limits_and_peak():
    B = pg.Box((0.0, 2.0), (0.0, 1.0), (0.0, 1.0))
    f = F.on(B).dyke(normal=(0.0, 1.0, 0.0), thickness=0.4, point=(1.0, 0.5, 0.5), amplitude=2.0, background=0.5, width=0.05)
    y, w = gl(0.0, 1.0)
    g = 0.5 * (np.tanh((y - 0.5 + 0.2) / 0.05) - np.tanh((y - 0.5 - 0.2) / 0.05))
    assert f.closed["integral"][0] == pytest.approx(2.0 * (0.5 * 1.0 + 2.0 * float(np.sum(w * g))), rel=1e-12)
    assert f(1.0, 0.5, 0.5) == pytest.approx(0.5 + 2.0 * math.tanh(0.2 / 0.05)) and f.regularity.order == math.inf
    sharp = F.on(B).dyke(normal=(0.0, 1.0, 0.0), thickness=0.4, point=(1.0, 0.5, 0.5), amplitude=2.0, background=0.5)
    thin = F.on(B).dyke(normal=(0.0, 1.0, 0.0), thickness=0.4, point=(1.0, 0.5, 0.5), amplitude=2.0, background=0.5,
                        width=1e-4)
    assert thin.closed["integral"][0] == pytest.approx(sharp.closed["integral"][0], abs=1e-6)
    C = pg.Cylinder(1.0, (0.0, 2.0))
    c = F.on(C).dyke(normal=(0.0, 0.0, 1.0), thickness=0.5, point=(0.0, 0.0, 0.7), width=0.1)
    z, wz = gl(0.0, 2.0)
    assert c.closed["integral"][0] == pytest.approx(math.pi * float(np.sum(
        wz * 0.5 * (np.tanh((z - 0.7 + 0.25) / 0.1) - np.tanh((z - 0.7 - 0.25) / 0.1)))), rel=1e-12)
    D = pg.Disk(1.0)
    off = F.on(D).dyke(normal=(1.0, 0.0), thickness=0.4, width=0.1)
    assert off.closed["integral"][1] == "checked"                                     # not elementary on a disk: through the plane
    P = np.random.default_rng(2).uniform(-0.7, 0.7, size=(10, 2))
    h = 1e-6
    for i in range(2):
        E = np.zeros(2)
        E[i] = h
        assert np.allclose((off(*(P + E).T) - off(*(P - E).T)) / (2 * h), off.gradient(*P.T)[:, i], atol=1e-8)


@pytest.mark.parametrize("params, message", [
    ({"normal": (0.0, 0.0, 1.0), "thickness": 0.0}, "thickness must be positive"),
    ({"normal": (0.0, 0.0, 1.0), "thickness": -1.0}, "thickness must be positive"),
    ({"normal": (0.0, 0.0, 1.0), "thickness": 0.5, "point": (0.0, 0.0, 1.5)}, "misses"),
    ({"normal": (0.0, 0.0, 1.0), "thickness": 0.5, "point": (0.0, 0.0, -1.25)}, "misses"),
    ({"normal": (0.0, 0.0, 1.0), "thickness": 0.5, "amplitude": 0.0}, "amplitude must be nonzero"),
    ({"normal": (0.0, 0.0, 1.0), "thickness": 0.5, "width": -0.1}, "width must be positive"),
    ({"normal": (0.0, 0.0), "thickness": 0.5}, "nonzero"),
    ({"normal": (0.0, 0.0, 1.0), "thickness": 0.5, "background": float("nan")}, "background must be finite"),
    ({"normal": (0.0, 0.0, 1.0), "thickness": "x"}, "thickness"), ({"normal": (0.0, 0.0, 1.0), "thickness": 0.5,
                                                                    "amplitude": "x"}, "amplitude"),
    ({"normal": (0.0, 0.0, 1.0), "thickness": 0.5, "point": "ab"}, "point needs 3")])
def test_dyke_refusals(params, message):
    with pytest.raises(pg.Refused, match=message):
        F.on(pg.Ball(1.0)).dyke(**params)
    with pytest.raises(TypeError):
        F.on(pg.Ball(1.0)).dyke((0.0, 0.0, 1.0), thickness=0.5)


def test_dyke_that_covers_the_whole_domain_is_allowed():
    f = F.on(pg.Box((0.0, 1.0), (0.0, 1.0))).dyke(normal=(1.0, 0.0), thickness=5.0, amplitude=2.0)
    assert f.closed["integral"][0] == pytest.approx(2.0) and f(0.3, 0.3) == 2.0
