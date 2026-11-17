"""SOLA targets, review fixes (plan 021 P1, work package H): the box on the shell, the Gaussian of any width on the
curved-boundary domains (against an independent Cartesian adaptive quadrature, scipy), the claim bookkeeping of a
normalised sharp window, the size of a target kept under `shape_scale`, and non-numeric inputs refused."""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import dblquad, quad

import pygeoinf0 as pg
from _bank_averages import cap_average, depth_window, radius_window, target
import pygeoinf0.functions as F
from pygeoinf0.core.claims import CHECKED, DERIVED
from pygeoinf0.functions import geometry as geo

E = F.entries()


def build(name, D, **params):
    return E[name].builder(D, **params)


def ref_mass(D, c, sig):
    """int_D exp(-|x - c|^2/(2 sig^2)) by Cartesian adaptive quadrature: the z-integral in closed form (error
    functions), the (x, y) integral by scipy's dblquad with the Gaussian's centre and the shell's hole as break points."""
    k = geo.kind(D)
    r2 = sig * math.sqrt(2.0)
    o = np.asarray(D.centre, float)
    G = lambda z, cz: sig * math.sqrt(math.pi / 2) * math.erf((z - cz) / r2)
    if k == "disk":
        a = D.radius

        def g(y, x):
            return math.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * sig * sig))

        return dblquad(
            g,
            -a,
            a,
            lambda x: -math.sqrt(max(a * a - x * x, 0)),
            lambda x: math.sqrt(max(a * a - x * x, 0)),
            epsabs=1e-14,
            epsrel=1e-11,
        )[0]
    if k == "cylinder":
        a = D.radius
        zf = G(D.z.b, c[2]) - G(D.z.a, c[2])
        return zf * ref_mass(pg.Disk(a, D.centre), c[:2], sig)
    a = D.radius if k == "ball" else D.outer
    hole = D.inner if k == "shell" else 0.0

    def zfac(y, x):
        rho2 = (x - o[0]) ** 2 + (y - o[1]) ** 2
        zo = math.sqrt(max(a * a - rho2, 0.0))
        w = G(o[2] + zo, c[2]) - G(o[2] - zo, c[2])
        if rho2 < hole * hole:
            zi = math.sqrt(hole * hole - rho2)
            w -= G(o[2] + zi, c[2]) - G(o[2] - zi, c[2])
        return w * math.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * sig * sig))

    def inner(x):
        h = math.sqrt(max(a * a - (x - o[0]) ** 2, 0.0))
        pts = [c[1]] if abs(c[1] - o[1]) < h else []
        if hole and abs(x - o[0]) < hole:
            e = math.sqrt(hole * hole - (x - o[0]) ** 2)
            pts += [o[1] - e, o[1] + e]
        return quad(
            lambda y: zfac(y, x), o[1] - h, o[1] + h, points=pts or None, epsabs=1e-15, epsrel=1e-12, limit=400
        )[0]

    return quad(
        inner,
        o[0] - a,
        o[0] + a,
        points=[c[0]] if abs(c[0] - o[0]) < a else None,
        epsabs=1e-14,
        epsrel=1e-10,
        limit=400,
    )[0]


GAUSS_CASES = [
    (pg.Shell(0.5, 1.0), None, None),  # the documented default, refused before
    (pg.Shell(0.5, 1.0), (0.75, 0.0, 0.0), 0.05),
    (pg.Shell(0.5, 1.0), (0.8, 0.0, 0.0), 0.02),
    (pg.Shell(0.5, 1.0), (0.8, 0.0, 0.0), 0.1),
    (pg.Shell(0.5, 1.0), (0.0, 0.0, 0.9), 0.15),
    (pg.Shell(0.5, 1.0), (0.3, 0.2, 0.0), 0.2),  # the centre in the hole
    (pg.Shell(0.5, 1.0), (0.97, 0.0, 0.0), 0.05),  # next to the outer boundary
    (pg.Disk(1.0), (0.2, 0.0), 0.05),
    (pg.Disk(1.0), (0.2, 0.0), 0.02),
    (pg.Disk(1.0), (0.95, 0.1), 0.1),
    (pg.Disk(1.0), None, 0.4),
    (pg.Ball(1.0), (0.9, 0.0, 0.0), 0.1),
    (pg.Ball(1.0), (0.2, -0.3, 0.1), 0.03),
    (pg.Ball(1.0), (0.4, -0.2, 0.3), 0.3),
    (pg.Cylinder(1.0, (0.0, 1.0)), (0.5, 0.1, 0.5), 0.1),
    (pg.Cylinder(1.0, (0.0, 1.0)), (0.9, 0.0, 0.95), 0.08),
    (pg.Cylinder(1.0, (0.0, 1.0)), None, 0.03),
]


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("D, centre, width", GAUSS_CASES, ids=lambda v: None)
def test_gaussian_target_mass_of_any_width_against_cartesian_quadrature(D, centre, width):
    f = target(D, shape="gaussian", centre=centre, scale=width)
    sig = f.params["width"]
    c = np.asarray(f.params["centre"], float)
    base = build("gaussian", D, centre=tuple(c), width=sig)
    mass = 1.0 / float(f(*c))  # f = base / mass and base(c) = 1
    P = c + 0.7 * sig * np.random.default_rng(3).normal(size=(20, c.size))
    assert np.allclose(base(*P.T) / f(*P.T), mass, rtol=1e-12)
    assert mass == pytest.approx(ref_mass(D, c, sig), rel=2e-9)
    status = f.closed["integral"]
    assert status[0] == 1.0 and status[1] == CHECKED


def test_gaussian_default_width_on_the_standard_shell_and_reviewer_widths():
    S = pg.Shell(0.5, 1.0)
    assert target(S, shape="gaussian").closed["integral"][0] == 1.0
    for c in ((0.75, 0.0, 0.0), (0.8, 0.0, 0.0)):
        for w in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
            assert target(S, shape="gaussian", centre=c, scale=w).integral()[0] == 1.0
    for D, c in ((pg.Disk(1.0), (0.2, 0.0)), (pg.Ball(1.0), (0.2, 0.0, 0.0)), (pg.Cylinder(1.0, (0.0, 1.0)), None)):
        for w in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
            assert target(D, shape="gaussian", centre=c, scale=w).integral()[0] == 1.0
    assert target(pg.Ball(1.0), shape="gaussian", centre=(0.9, 0.0, 0.0), scale=0.1)


def test_gaussian_far_from_the_boundary_is_the_full_space_mass():
    B = pg.Ball(1.0)
    f = target(B, shape="gaussian", scale=0.04)
    full = (2 * math.pi * 0.04**2) ** 1.5
    base = build("gaussian", B, width=0.04)
    assert f(0.0, 0.0, 0.0) == pytest.approx(1 / full, rel=1e-12) and base(0.0, 0.0, 0.0) == 1.0


def test_gaussian_mass_negligible_in_the_domain_is_refused():
    with pytest.raises(pg.Refused, match="full-space mass"):
        target(pg.Ball(1.0), shape="gaussian", centre=(2.5, 0.0, 0.0), scale=0.2)


def test_gaussian_centre_outside_the_domain_still_has_the_right_mass():
    D, c = pg.Disk(1.0), np.array([1.1, 0.0])
    f = target(D, shape="gaussian", centre=tuple(c), scale=0.3)
    base = build("gaussian", D, centre=tuple(c), width=0.3)
    assert base(1.0, 0.0) / f(1.0, 0.0) == pytest.approx(ref_mass(D, c, 0.3), rel=2e-9)


# ------------------------------------------------------------------------------------------- box on the shell
@pytest.mark.parametrize("edge", [0.0, 0.03])
def test_box_target_on_the_shell(edge):
    S = pg.Shell(0.5, 1.0)
    c = (0.75, 0.0, 0.0)
    h = (0.05, 0.1, 0.12)
    f = target(S, shape="box", centre=c, scale=h, edge=edge)
    assert "shell" in E["plateau"].kinds
    mass = float(np.prod(2 * np.array(h) + edge))
    assert f.integral() == (1.0, DERIVED, f.integral()[2])
    assert f(*c) == pytest.approx(1 / mass, rel=1e-13)
    # the support lies in the shell: the corners of the circumscribed box are inside 0.5 < |x| < 1
    ext = np.array(h) + edge
    corners = np.array(c) + np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * ext
    assert np.all((np.linalg.norm(corners, axis=1) > 0.5) & (np.linalg.norm(corners, axis=1) < 1.0))
    # tensor Gauss-Legendre with breaks at the plateau and the edges gives the unit mass independently
    t, w = np.polynomial.legendre.leggauss(40)
    axes = []
    for i in range(3):
        br = sorted({c[i] - ext[i], c[i] - h[i], c[i] + h[i], c[i] + ext[i]})
        pieces = [(0.5 * (q - p) * (t + 1) + p, 0.5 * (q - p) * w) for p, q in itertools.pairwise(br)]
        axes.append((np.concatenate([n for n, _ in pieces]), np.concatenate([x for _, x in pieces])))
    (X, WX), (Y, WY), (Z, WZ) = axes
    gx, gy, gz = np.meshgrid(X, Y, Z, indexing="ij")
    wgt = WX[:, None, None] * WY[None, :, None] * WZ[None, None, :]
    assert float(np.sum(wgt * f(gx, gy, gz))) == pytest.approx(1.0, abs=1e-12)


def test_box_target_on_the_shell_default_and_refusals():
    S = pg.Shell(0.5, 1.0)
    d = target(S, shape="box")
    assert d.support.radius < 0.25 and d.integral()[0] == 1.0
    with pytest.raises(pg.Refused, match="circumscribed"):
        target(S, shape="box", scale=0.2)  # corner radius 0.35 > room 0.25
    assert "sphere" not in E["plateau"].kinds


# ------------------------------------------------------------------------------------------- claim bookkeeping
def test_normalised_sharp_window_states_the_integral_once():
    B = pg.Ball(1.0)
    n = radius_window(B, r1=0.2, r2=0.6).normalised()
    assert "integral" not in n.closed and n.exact["integral"][0] == 1.0
    assert sum(1 for cl in n.claims if cl.key == "integral") == 1
    m = radius_window(B, r1=0.2, r2=0.6).scaled(4.0).normalised()
    assert sum(1 for cl in m.claims if cl.key == "integral") == 1 and m.integral()[0] == 1.0
    assert m(0.4, 0, 0) == pytest.approx(radius_window(B, r1=0.2, r2=0.6)(0.4, 0, 0))
    smooth = radius_window(B, r1=0.2, r2=0.6, edge=0.1).normalised()
    assert sum(1 for cl in smooth.claims if cl.key == "integral") == 1 and "integral" in smooth.closed


# ------------------------------------------------------------------------------------------- non-numeric inputs
@pytest.mark.parametrize(
    "make, msg",
    [
        (lambda D: F.on(D).plateau(half_width="a"), "numbers"),
        (lambda D: F.on(D).plateau(half_width=("a", "b")), "numbers"),
        (lambda D: F.on(D).plateau(centre=object()), "centre must be numbers"),
        (lambda D: F.on(D).indicator(centre="ab"), "centre is a point"),
        (lambda D: F.on(D).gaussian(centre=("x", "y")), "numbers"),
        (lambda D: F.on(D).gaussian(width="w"), "number"),
        (lambda D: F.on(D).smooth_indicator(inner="e"), "number"),
        (lambda D: F.on(D).bump(k="k"), "number"),
    ],
)
def test_non_numeric_inputs_are_refused_not_raised(make, msg):
    with pytest.raises(pg.Refused, match=msg):
        make(pg.Disk(1.0))
