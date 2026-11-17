"""The SOLA families of the bank (plan 021 P1, work package H): unit mass against an independent adaptive quadrature
(scipy, not the families' own Gauss-Legendre), the special values the contract cannot see, closed forms of the
indicator masses, the C-infinity step, relations between families, claims, and refusals."""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import quad

import pygeoinf0 as pg
from _bank_averages import cap_average, depth_window, radius_window, target
import pygeoinf0.functions as F
from pygeoinf0.core.claims import DERIVED
from pygeoinf0.functions import geometry as geo
from pygeoinf0.functions.families.band_common import STEP_REFERENCE, dstep, gl, step

E = F.entries()


def build(name, D, **params):
    return E[name].builder(D, **params)


def centre_of(D):
    return np.asarray(D.centre, float)


# ------------------------------------------------------------------------------------------- independent masses
def zonal_mass(f, D, c, rmax, points=()):
    """The integral of a function that depends on the geodesic distance from c only (cap, bump, window about c)."""
    k, d = geo.kind(D), geo.ambient(D)
    c = np.asarray(c, float)
    if k in ("sphere", "circle"):
        o, R = centre_of(D), D.radius
        u = (c - o) / R
        v = (
            np.cross(u, [0.0, 1.0, 0.0] if abs(u[1]) < 0.9 else [1.0, 0.0, 0.0])
            if k == "sphere"
            else np.array([-u[1], u[0]])
        )
        v = v / np.linalg.norm(v)

        def g(r):
            return float(f(*(o + R * (math.cos(r / R) * u + math.sin(r / R) * v)))) * (
                2 * math.pi * R * math.sin(r / R) if k == "sphere" else 2.0
            )
    else:
        e = np.zeros(d)
        e[0] = 1.0

        def g(r):
            return float(f(*(c + r * e))) * geo.SPHERE_AREA[d] * r ** (d - 1)

    return quad(g, 0, rmax, points=list(points) or None, limit=500, epsabs=1e-13, epsrel=1e-13)[0]


def axis_mass(f, D, window_kind, edges, a, b):
    """Radius/depth window: the integral from the radial / vertical line through the centre."""
    k, d = geo.kind(D), geo.ambient(D)
    if k in ("disk", "ball", "shell"):
        o, e = centre_of(D), np.eye(d)[0]
        return quad(
            lambda r: float(f(*(o + r * e))) * geo.SPHERE_AREA[d] * r ** (d - 1),
            0,
            D.outer if k == "shell" else D.radius,
            points=edges,
            limit=500,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
    mid = geo.centre(D)
    cross = (
        1.0
        if k == "interval"
        else math.pi * D.radius**2
        if k == "cylinder"
        else float(np.prod([s.b - s.a for s in D.sides[:-1]]))
    )

    def g(z):
        p = mid.copy()
        p[-1] = z
        return float(f(*p))

    return cross * quad(g, a, b, points=edges, limit=500, epsabs=1e-13, epsrel=1e-13)[0]


# ------------------------------------------------------------------------------------------- the step
def test_step_properties():
    u = np.linspace(-0.5, 1.5, 2001)
    S = step(u)
    assert np.all(S[u <= 0] == 0) and np.all(S[u >= 1] == 1) and np.all(np.diff(S) >= 0)
    assert np.allclose(step(u) + step(1 - u), 1.0, atol=1e-15)
    assert step(0.5) == pytest.approx(0.5) and step(np.array([0.5])).shape == (1,) and step(0.3).shape == ()
    assert gl(step, [0.0, 1.0]) == pytest.approx(0.5, abs=1e-12)  # int_0^1 S = 1/2
    h = 1e-6
    t = np.linspace(0.05, 0.95, 19)
    assert np.allclose((step(t + h) - step(t - h)) / (2 * h), dstep(t), rtol=1e-6, atol=1e-9)
    assert np.all(np.isfinite(dstep(np.array([0.0, 1e-300, 1 - 1e-16, 1.0]))))


# ------------------------------------------------------------------------------------------- radius and depth windows
RADIAL = [(pg.Ball(1.0), 0.0, 1.0), (pg.Disk(1.0), 0.0, 1.0), (pg.Shell(0.5, 1.0), 0.5, 1.0)]


@pytest.mark.parametrize("D, lo, hi", RADIAL, ids=["ball", "disk", "shell"])
@pytest.mark.parametrize(
    "r1, r2, edge",
    [
        (0.2, 0.7, 0.0),
        (0.55, 0.95, 0.0),
        (0.1, 0.9, 0.05),
        (0.3, 1.0, 0.1),
        (0.0, 0.6, 0.2),
        (0.0, 1.0, 0.5),
        (0.6, 0.8, 0.003),
    ],
)
def test_radius_window_unit_mass(D, lo, hi, r1, r2, edge):
    r1, r2, edge = lo + r1 * (hi - lo), lo + r2 * (hi - lo), edge * (hi - lo)  # fractions of the radial range
    f = radius_window(D, r1=r1, r2=r2, edge=edge)
    assert f.integral()[0] == 1.0
    pts = sorted({r1, r2, r1 + edge, r2 - edge})
    assert axis_mass(f, D, "radius", pts, 0, 0) == pytest.approx(1.0, abs=1e-11)
    assert f.closed["max"][0] == pytest.approx(f(*(centre_of(D) + 0.5 * (r1 + r2) * np.eye(geo.ambient(D))[0])))


@pytest.mark.parametrize(
    "D, hi",
    [
        (pg.Interval("[0, 2]"), 2.0),
        (pg.Box((0.0, 1.0), (0.0, 3.0)), 3.0),
        (pg.Box((0.0, 2.0), (0.0, 1.0), (-1.0, 0.5)), 1.5),
        (pg.Cylinder(0.7, (0.0, 2.0)), 2.0),
        (pg.Disk(1.0), 1.0),
        (pg.Ball(2.0), 2.0),
        (pg.Shell(0.5, 1.5), 1.0),
    ],
)
@pytest.mark.parametrize(
    "fr", [(0.1, 0.5, 0.0), (0.0, 0.4, 0.0), (0.2, 1.0, 0.0), (0.1, 0.9, 0.2), (0.0, 1.0, 0.5), (0.3, 0.6, 0.01)]
)
def test_depth_window_unit_mass(D, hi, fr):
    d1, d2, edge = fr[0] * hi, fr[1] * hi, fr[2] * hi
    f = depth_window(D, d1=d1, d2=d2, edge=edge)
    k = geo.kind(D)
    if k in ("disk", "ball", "shell"):
        top = D.outer if k == "shell" else D.radius
        pts = sorted({top - d for d in (d1, d2, d1 + edge, d2 - edge)})
        assert axis_mass(f, D, "depth", pts, 0, 0) == pytest.approx(1.0, abs=1e-11)
    else:
        top = D.b if k == "interval" else D.sides[-1].b if k == "box" else D.z.b
        pts = sorted({top - d for d in (d1, d2, d1 + edge, d2 - edge)})
        assert axis_mass(f, D, "depth", pts, top - hi, top) == pytest.approx(1.0, abs=1e-11)
    assert f.integral()[0] == 1.0 and f.integral()[1] == DERIVED


def test_window_special_values_and_indicator_masses():
    B, S, Dk = pg.Ball(1.0), pg.Shell(0.5, 1.0), pg.Disk(1.0)
    f = radius_window(B, r1=0.2, r2=0.7)
    assert f(0.5, 0.0, 0.0) == pytest.approx(3 / (4 * math.pi * (0.7**3 - 0.2**3)), rel=1e-14)
    assert f(0.1, 0.0, 0.0) == 0.0 and f(0.8, 0.0, 0.0) == 0.0 and f(0.2, 0.0, 0.0) > 0 and f(0.7, 0.0, 0.0) > 0
    assert np.allclose(
        np.unique(f(*np.random.default_rng(1).uniform(-1, 1, (3, 400)) * 0.5)), [0.0, f.closed["max"][0]]
    )
    g = radius_window(Dk, r1=0.1, r2=0.6)
    assert g(0.3, 0.0) == pytest.approx(1 / (math.pi * (0.36 - 0.01)), rel=1e-14)
    h = radius_window(S, r1=0.6, r2=0.9)
    assert h(0.0, 0.75, 0.0) == pytest.approx(3 / (4 * math.pi * (0.9**3 - 0.6**3)), rel=1e-14)
    box = pg.Box((0.0, 2.0), (0.0, 3.0), (0.0, 1.0))
    d = depth_window(box, d1=0.25, d2=0.75)
    assert d(0.3, 2.0, 0.5) == pytest.approx(1 / (2 * 3 * 0.5), rel=1e-14) and d(0.3, 2.0, 0.9) == 0.0
    assert d(1.9, 0.1, 0.5) == d(0.1, 2.9, 0.5)  # constant across the cross-section
    assert d.support.kind == "touches" and d.regularity.vanishes_at_boundary is None
    assert depth_window(pg.Interval("[0, 1]"), d1=0.2, d2=0.4).support.kind == "interior"


def test_radius_limit_at_the_centre():
    B = pg.Ball(1.0)
    f = radius_window(B, r1=0.0, r2=0.5)  # the origin is inside the window
    assert f(0.0, 0.0, 0.0) == pytest.approx(f.closed["max"][0]) and f(0.0, 0.0, 0.0) == pytest.approx(
        3 / (4 * math.pi * 0.125)
    )
    g = radius_window(B, r1=0.0, r2=0.8, edge=0.3)  # C-infinity at r = 0: S(r/edge) flat
    assert (
        g(0.0, 0.0, 0.0) == 0.0
        and np.all(np.isfinite(g.gradient(0.0, 0.0, 0.0)))
        and np.allclose(g.gradient(0.0, 0.0, 0.0), 0.0)
    )
    assert np.all(np.isfinite(g(*np.zeros((3, 4)))))
    t = depth_window(B, d1=0.5, d2=1.0, edge=0.2)  # contains the centre (depth = R)
    assert np.allclose(t.gradient(0.0, 0.0, 0.0), 0.0) and t(0.0, 0.0, 0.0) == 0.0


def test_window_partition_of_unity_for_sharp_tilings():
    B = pg.Ball(1.0)
    cuts = [0.0, 0.3, 0.55, 1.0]
    fs = [radius_window(B, r1=a, r2=b) for a, b in itertools.pairwise(cuts)]
    masses = [4 * math.pi * (b**3 - a**3) / 3 for a, b in itertools.pairwise(cuts)]
    X = np.random.default_rng(2).uniform(-0.57, 0.57, (3, 500))
    X = X[:, np.linalg.norm(X, axis=0) < 1.0]
    total = sum(m * f(*X) for m, f in zip(masses, fs))
    on_cut = np.min(np.abs(np.linalg.norm(X, axis=0)[:, None] - np.array(cuts[1:-1])), axis=1) < 1e-9
    assert np.allclose(total[~on_cut], 1.0, atol=1e-13)
    assert all(m * f.closed["max"][0] == pytest.approx(1.0) for m, f in zip(masses, fs))
    box = pg.Box((0.0, 1.0), (0.0, 1.0))
    ds = [depth_window(box, d1=a, d2=b) for a, b in ((0.0, 0.25), (0.25, 1.0))]
    x, y = np.random.default_rng(3).uniform(0, 1, (2, 300))
    assert np.allclose(0.25 * ds[0](x, y) + 0.75 * ds[1](x, y), 1.0)  # off the cut at depth 0.25


def test_depth_is_a_reparametrised_radius():
    for D in (pg.Ball(1.0), pg.Shell(0.5, 1.0), pg.Disk(2.0)):
        top = D.outer if isinstance(D, pg.Shell) else D.radius
        r1, r2 = top - 0.45, top - 0.1
        a = depth_window(D, d1=0.1, d2=0.45, edge=0.1)
        b = radius_window(D, r1=r1, r2=r2, edge=0.1)
        X = np.random.default_rng(4).uniform(-0.9, 0.9, (geo.ambient(D), 200))
        assert np.allclose(a(*X), b(*X), rtol=1e-12, atol=1e-14)


def test_smooth_window_mass_lies_between_plateau_and_indicator():
    B = pg.Ball(1.0)
    sharp = 1 / radius_window(B, r1=0.2, r2=0.9).closed["max"][0]
    plateau = 4 * math.pi * (0.7**3 - 0.4**3) / 3  # [r1 + edge, r2 - edge]
    smooth = 1 / radius_window(B, r1=0.2, r2=0.9, edge=0.2).closed["max"][0]
    assert plateau < smooth < sharp
    masses = [
        1 / radius_window(B, r1=0.2, r2=0.9, edge=e).closed["max"][0] for e in (1e-3, 1e-2, 0.1, 0.3, 0.35)
    ]
    assert masses[0] == pytest.approx(sharp, rel=2e-2) and np.all(np.diff(masses) < 0)


def test_window_gradient_against_differences():
    cases = [
        (pg.Ball(1.0), "smooth_band", {"coordinate": "radius", "lower": 0.2, "upper": 0.8, "edge": 0.2}),
        (pg.Box((0.0, 1.0), (0.0, 1.0)), "smooth_band", {"coordinate": "depth", "lower": 0.1, "upper": 0.9, "edge": 0.3}),
        (pg.Shell(0.5, 1.0), "smooth_band", {"coordinate": "depth", "lower": 0.05, "upper": 0.45, "edge": 0.1}),
    ]
    for D, name, p in cases:
        f = build(name, D, **p)
        P = np.random.default_rng(5).uniform(0.45, 0.8, (30, geo.ambient(D)))
        G, h = f.gradient(*P.T), 1e-6
        for i in range(P.shape[1]):
            e = np.zeros(P.shape[1])
            e[i] = h
            assert np.allclose(
                (f(*(P + e).T) - f(*(P - e).T)) / (2 * h), G[:, i], rtol=1e-5, atol=1e-6 * np.max(np.abs(G))
            )


# ------------------------------------------------------------------------------------------- caps
CAP_DOMAINS = {
    "interval": pg.Interval("[0, 1]"),
    "box2": pg.Box((0.0, 1.0), (0.0, 1.0)),
    "box3": pg.Box(*[(0.0, 1.0)] * 3),
    "disk": pg.Disk(1.0),
    "ball": pg.Ball(1.0),
    "shell": pg.Shell(0.5, 1.0),
    "sphere": pg.Sphere(2.0),
    "circle": pg.Circle(1.5),
    "cylinder": pg.Cylinder(1.0, (0.0, 1.0)),
    "torus": pg.Torus((0.0, 2.0), (0.0, 2.0)),
}


@pytest.mark.parametrize("key", sorted(CAP_DOMAINS))
@pytest.mark.parametrize("scale, edge", [(0.3, 0.0), (0.25, 0.25), (0.35, 0.1), (0.2, 0.01)])
def test_cap_target_unit_mass(key, scale, edge):
    D = CAP_DOMAINS[key]
    c = geo.centre(D)
    scale = min(scale, 0.9 * min(geo.room(D, c), geo.injectivity(D)))
    edge = min(edge, scale)
    f = target(D, shape="cap", scale=scale, edge=edge)
    rho = f.support.radius
    assert f.integral()[0] == 1.0
    assert zonal_mass(f, D, c, rho, [rho - edge] if edge else [rho]) == pytest.approx(1.0, abs=1e-10)


def test_sharp_cap_closed_masses():
    for D, vol in (
        (pg.Interval("[0, 1]"), lambda r: 2 * r),
        (pg.Disk(1.0), lambda r: math.pi * r * r),
        (pg.Ball(1.0), lambda r: 4 * math.pi * r**3 / 3),
        (pg.Circle(1.5), lambda r: 2 * r),
        (pg.Sphere(2.0), lambda r: 2 * math.pi * 4 * (1 - math.cos(r / 2.0))),
        (pg.Torus((0.0, 2.0), (0.0, 2.0)), lambda r: math.pi * r * r),
    ):
        f = target(D, shape="cap", scale=0.4)
        assert f(*geo.centre(D)) == pytest.approx(1 / vol(0.4), rel=1e-13)
        assert f.closed["max"][0] == pytest.approx(1 / vol(0.4), rel=1e-13) and f.exact["l2_norm_sq"][
            0
        ] == pytest.approx(1 / vol(0.4))


def test_cap_average_equals_the_cap_target_and_is_rotation_invariant():
    S = pg.Sphere(2.0)
    a = cap_average(S, angle=0.7, edge=0.3)
    b = target(S, shape="cap", scale=1.4, edge=0.6)
    X = np.random.default_rng(6).normal(size=(3, 300))
    X = 2.0 * X / np.linalg.norm(X, axis=0)
    assert np.allclose(a(*X), b(*X), rtol=1e-12, atol=1e-14)
    n = np.array([0.6, 0.0, 0.8])  # rotate the north pole to n
    Rm = 2 * np.outer(m := (np.array([0, 0, 1.0]) + n) / np.linalg.norm(np.array([0, 0, 1.0]) + n), m) - np.eye(3)
    off = cap_average(S, centre=tuple(2.0 * n), angle=0.7, edge=0.3)
    assert np.allclose(off(*(Rm @ X)), a(*X), rtol=1e-11, atol=1e-13)
    assert zonal_mass(off, S, 2.0 * n, 1.4, [1.4 - 0.6]) == pytest.approx(1.0, abs=1e-10)


def test_sharp_cap_average_sphere_special_values():
    S = pg.Sphere(1.0)
    f = cap_average(S, angle=math.pi / 2)  # the hemisphere
    assert f(0.0, 0.0, 1.0) == pytest.approx(1 / (2 * math.pi)) and f(0.0, 0.0, -1.0) == 0.0 and f(1.0, 0.0, 0.0) > 0
    pole = cap_average(S, angle=0.3, edge=0.3)  # edge = angle: S(1 - theta/angle)
    assert pole(0.0, 0.0, 1.0) == pytest.approx(pole.closed["max"][0]) and pole.regularity.order == math.inf


# ------------------------------------------------------------------------------------------- bump, gaussian, box
@pytest.mark.parametrize("key", ["interval", "box2", "disk", "ball", "shell", "sphere", "circle", "cylinder", "torus"])
def test_bump_target_is_the_normalised_bump(key):
    D = CAP_DOMAINS[key]
    rho = min(0.3, 0.9 * geo.room(D, geo.centre(D)))
    base = build("bump", D, radius=rho, k=2.0)
    f = target(D, shape="bump", scale=rho, k=2.0)
    X, _ = geo.quadrature(D, 7)
    assert np.allclose(f(*X.T) * base.closed["integral"][0], base(*X.T), rtol=1e-13, atol=1e-300)
    assert zonal_mass(f, D, geo.centre(D), rho) == pytest.approx(1.0, abs=1e-10)
    assert f.regularity.order == math.inf and f.closed["integral"][1] == DERIVED


@pytest.mark.parametrize(
    "key, width",
    [
        ("interval", 0.1),
        ("box2", 0.12),
        ("box3", 0.2),
        ("disk", 0.2),
        ("ball", 0.25),
        ("shell", 0.2),
        ("sphere", 0.5),
        ("circle", 0.4),
        ("cylinder", 0.25),
    ],
)
def test_gaussian_target_unit_mass(key, width):
    D = CAP_DOMAINS[key]
    f = target(D, shape="gaussian", scale=width)
    n = {1: 3000, 2: 300}.get(geo.dimension(D), 90)
    X, w = geo.quadrature(D, n)
    assert w @ f(*X.T) == pytest.approx(1.0, abs=1e-9)
    base = build("gaussian", D, width=width)
    assert np.allclose(f(*X[::97].T) / base(*X[::97].T), f(*geo.centre(D)) / 1.0, rtol=1e-12)
    exact = key in ("interval", "box2", "box3", "sphere", "circle")
    assert f.closed["integral"][1] == (DERIVED if exact else "checked")


def test_gaussian_target_off_centre_and_narrow():
    B = pg.Ball(1.0)
    f = target(B, shape="gaussian", centre=(0.4, -0.2, 0.3), scale=0.3)
    X, w = geo.quadrature(B, 90)
    assert w @ f(*X.T) == pytest.approx(1.0, abs=1e-9)
    narrow = target(B, shape="gaussian", scale=0.01)  # no longer refused: polar mass about the centre
    assert narrow(0.0, 0.0, 0.0) == pytest.approx(1 / (2 * math.pi * 1e-4) ** 1.5, rel=1e-12)


@pytest.mark.parametrize("key", ["interval", "box2", "box3", "disk", "ball", "cylinder", "torus"])
@pytest.mark.parametrize("edge", [0.0, 0.05, 0.1])
def test_box_target_unit_mass_and_values(key, edge):
    D = CAP_DOMAINS[key]
    d = geo.ambient(D)
    h = np.array([0.08, 0.12, 0.1][:d])
    c = geo.centre(D)
    f = target(D, shape="box", scale=tuple(h), edge=edge)
    M = float(np.prod(2 * h + edge))
    assert f(*c) == pytest.approx(1 / M, rel=1e-13) and f.closed["max"][0] == pytest.approx(1 / M, rel=1e-13)
    assert f(*(c + (h + edge + 1e-3) * np.eye(d)[0])) == 0.0
    ext = h + edge
    prod = np.prod(
        [
            quad(
                lambda x, i=i: float(f(*np.where(np.arange(d) == i, x, c))),
                c[i] - ext[i],
                c[i] + ext[i],
                points=[c[i] - h[i], c[i] + h[i]],
                limit=400,
                epsabs=1e-14,
                epsrel=1e-13,
            )[0]
            for i in range(d)
        ]
    )
    assert prod * (1 / f(*c)) ** (d - 1) == pytest.approx(1.0, abs=1e-11)
    assert (f.regularity.order == math.inf) == (edge > 0)


def test_box_target_default_scalar_scale_and_gradient():
    D = pg.Box((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    f = target(D, shape="box", centre=(0.45, 0.5, 0.55), scale=0.1, edge=0.08)
    assert f.params["half_width"] == (0.1, 0.1, 0.1)
    P = np.random.default_rng(7).uniform(0.3, 0.7, (40, 3))
    G, h = f.gradient(*P.T), 1e-6
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        assert np.allclose((f(*(P + e).T) - f(*(P - e).T)) / (2 * h), G[:, i], rtol=1e-5, atol=1e-6 * np.max(np.abs(G)))
    d = target(D, shape="box")  # the default fits in the room
    assert d.support.radius == pytest.approx(0.25) and d.support.kind == "interior"


# ------------------------------------------------------------------------------------------- claims
def test_claims_and_sharp_function_bookkeeping():
    B = pg.Ball(1.0)
    sharp = radius_window(B, r1=0.2, r2=0.6)
    smooth = radius_window(B, r1=0.2, r2=0.6, edge=0.1)
    for f in (sharp, smooth, target(B, shape="cap", scale=0.3)):
        assert f.integral()[0] == 1.0 and f.integral()[1] == DERIVED
        c = f.claims.find("integral")
        assert c is not None and c.status == DERIVED
        assert all(cl.status in ("theorem", "derived", "checked") for cl in f.claims)
    assert "integral" not in sharp.closed and "integral" in smooth.closed
    assert (
        smooth.claims.find("regularity").status == "theorem"
        and STEP_REFERENCE in smooth.claims.find("regularity").reason
    )
    assert sharp.regularity.order == -1 and sharp.claims.find("regularity").status == DERIVED
    s2 = sharp.scaled(3.0)
    assert s2.integral()[0] == 3.0 and s2(0.4, 0, 0) == pytest.approx(3 * sharp(0.4, 0, 0))
    n = s2.normalised()
    assert n.integral()[0] == 1.0 and n(0.4, 0, 0) == pytest.approx(sharp(0.4, 0, 0))
    assert sharp.element(pg.L2(B)) is not None
    with pytest.raises(pg.Refused):
        sharp.element(pg.Sobolev(B, 1, length_scale=0.2))
    assert smooth.element(pg.Sobolev(B, 2, length_scale=0.2)) is not None


def test_catalogue_and_kinds():
    kinds = {n: e.kinds for n, e in E.items() if n in ("plateau", "bump", "smooth_band")}
    assert set(kinds["smooth_band"]) == {"interval", "box", "disk", "ball", "shell", "cylinder"}
    assert set(kinds["plateau"]) == {"interval", "box", "disk", "ball", "shell", "cylinder", "torus"}
    assert "sola_target" not in E
    for n in kinds:
        assert (
            E[n].references
            and E[n].summary
            and all(1 <= len(E[n].example_params(D)) <= 4 for k in E[n].kinds for D in geo.standard(k))
        )
    B = pg.Ball(1.0)
    assert F.on(B).smooth_band(lower=0.1, upper=0.5).family == "smooth_band"
    assert F.ball.plateau(B, half_width=0.1).family == "plateau"


# ------------------------------------------------------------------------------------------- refusals
def test_refusals():
    B, S, box = pg.Ball(1.0), pg.Sphere(1.0), pg.Box((0.0, 1.0), (0.0, 1.0))

    torus = pg.Torus((0.0, 4.0), (0.0, 4.0))
    cases = [
        (lambda: F.on(B).indicator(radius=-1.0), "positive"),
        (lambda: F.on(B).indicator(radius=float("nan")), "positive"),
        (lambda: F.on(B).indicator(radius=1.2), "leaves the domain"),
        (lambda: F.on(B).smooth_indicator(radius=0.3, inner=-0.1), "inner"),
        (lambda: F.on(B).smooth_indicator(radius=0.3, inner=0.3), "inner"),
        (lambda: F.on(B).indicator(radius="wide"), "number"),
        (lambda: F.on(B).indicator(centre=(0.0, 0.0)), "3|centre|point"),
        (lambda: F.on(S).indicator(centre=(0.0, 0.0, 2.0)), "sphere|on the"),
        (lambda: F.on(box).plateau(half_width=(0.1, 0.2, 0.3)), "numbers"),
        (lambda: F.on(box).plateau(half_width=-0.1), "positive"),
        (lambda: F.on(box).plateau(half_width=0.45), "circumscribed"),
        (lambda: F.on(box).plateau(edge=-0.1), ">= 0"),
        (lambda: F.on(pg.Torus((0.0, 1.0), (0.0, 1.0))).plateau(centre=(0.1, 0.5), half_width=0.2), "circumscribed|leaves the torus"),
        (lambda: F.on(torus).plateau(centre=(0.3, 2.0), half_width=0.5), "leaves the torus"),
        (lambda: F.on(torus).plateau(centre=(5.0, 2.0), half_width=0.5), "outside the torus"),
    ]
    for call, msg in cases:
        with pytest.raises(pg.Refused, match=msg):
            call()
