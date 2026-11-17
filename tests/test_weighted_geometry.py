"""Geometry for weighted measures (EXP-031 phase W2): pieces, flat pieces of full dimension, simplices and simplicial
sets (with a sphere chart), curves and surfaces given by callables, the curve interface of the existing paths, and
exact closure checks."""
import math
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import quadrature
from pygeoinf0.domains import inside_closure

ICO_V = np.array([[-1, 1.618033988749895, 0], [1, 1.618033988749895, 0], [-1, -1.618033988749895, 0],
                  [1, -1.618033988749895, 0], [0, -1, 1.618033988749895], [0, 1, 1.618033988749895],
                  [0, -1, -1.618033988749895], [0, 1, -1.618033988749895], [1.618033988749895, 0, -1],
                  [1.618033988749895, 0, 1], [-1.618033988749895, 0, -1], [-1.618033988749895, 0, 1]])
ICO_F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7],
         [9, 8, 1]]


def test_closed_forms_of_curve_lengths():
    R = 0.7
    circle = pg.Curve(lambda t: (R * np.cos(t), R * np.sin(t)), 0, 2 * np.pi, closed=True)
    assert circle.length == pytest.approx(2 * np.pi * R, rel=1e-14)
    helix = pg.Curve(lambda t: (np.cos(t), np.sin(t), 0.3 * t), 0, 4 * np.pi)
    assert helix.length == pytest.approx(4 * np.pi * math.sqrt(1.09), rel=1e-14)
    cycloid = pg.Curve(lambda t: (t - np.sin(t), 1 - np.cos(t)), 0, 2 * np.pi)       # speed 0 at both ends
    assert cycloid.length == pytest.approx(8.0, rel=1e-13)
    given = pg.Curve(lambda t: np.stack([t - np.sin(t), 1 - np.cos(t)], axis=-1), 0, 2 * np.pi,
                     derivative=lambda t: (1 - np.cos(t), np.sin(t)))
    assert given.length == pytest.approx(8.0, rel=1e-13)
    with pytest.raises(ValueError, match="does not match"):
        pg.Curve(lambda t: (t, t * t), 0, 1, derivative=lambda t: (1 + 0 * t, t))
    with pytest.raises(ValueError, match="closed"):
        pg.Curve(lambda t: (t, t * t), 0, 1, closed=True)


@pytest.mark.parametrize("curve", ["helix", "cycloid", "parabola"])
def test_arc_length_coordinates_round_trip(curve):
    C = {"helix": pg.Curve(lambda t: (np.cos(t), np.sin(t), 0.3 * t), 0, 4 * np.pi),
         "cycloid": pg.Curve(lambda t: (t - np.sin(t), 1 - np.cos(t)), 0, 2 * np.pi),
         "parabola": pg.Curve(lambda t: (t, 0.5 + 0.15 * np.sin(2 * np.pi * t)), 0.1, 0.9)}[curve]
    s = np.linspace(0, C.length, 17)
    X = C.to_cartesian(s)
    assert np.max(np.abs(C.from_cartesian(*X)[0] - s)) <= 1e-12 * C.length
    assert np.max(np.abs(C.arclength(C.parameter(s)) - s)) <= 1e-13 * C.length   # t(s) itself is only sqrt(eps)
                                                                                   # accurate where the speed vanishes
    off = np.array([x[3] for x in X]) + 1e-3                                       # off the curve: NaN
    assert np.isnan(C.from_cartesian(*off)[0]) and not C.contains(np.asarray(off, float))


def test_curve_from_polyline_pieces_equals_the_polyline():
    P = np.array([[0, 0], [0.3, 0.4], [0.3, 1.4], [1.0, 1.0]])
    pl = pg.Polyline(P)
    par = pl.parametrisation()
    C = pg.Curve(par.gamma, par.t0, par.t1, derivative=par.derivative, breaks=par.breaks)
    assert C.length == pytest.approx(pl.volume, rel=1e-15)
    assert len(pl.pieces()) == 3 and pl.pieces()[1] == pg.Segment((0.3, 0.4), (0.3, 1.4))
    A, B = pg.L2(pl), pg.L2(C)
    f, g = (lambda x, y: np.sin(3 * x) + y ** 2), (lambda x, y: np.exp(x - y))
    assert B.inner(B(f), B(g)) == pytest.approx(A.inner(A(f), A(g)), rel=1e-13)
    H, K = pg.Sobolev(pl, 1.0, length_scale=0.3), pg.Sobolev(C, 1.0, length_scale=0.3)
    x0 = np.array([0.3, 0.9])
    assert K.riesz(x0).norm() == pytest.approx(H.riesz(x0).norm(), rel=1e-13)


def test_curve_interface_of_existing_paths():
    seg, arc, circ = pg.Segment((0, 0, 0), (1, 2, 2)), pg.Arc(pg.Sphere(2.0), (2, 0, 0), (0, 2, 0)), pg.Circle(0.5)
    for D, speed in ((seg, 1.0), (arc, 1.0), (circ, 0.5)):
        p = D.parametrisation()
        t = np.linspace(p.t0, p.t1, 7)
        assert np.allclose(np.linalg.norm(p.derivative(t), axis=-1), speed, rtol=1e-14)
        C = pg.Curve(p.gamma, p.t0, p.t1, derivative=p.derivative, closed=p.closed)
        assert C.length == pytest.approx(D.volume, rel=1e-14)
    p = circ.parametrisation()
    H1 = pg.Sobolev(circ, 1.0, length_scale=0.2)
    H2 = pg.Sobolev(pg.Curve(p.gamma, 0, 2 * np.pi, derivative=p.derivative, closed=True), 1.0, length_scale=0.2)
    assert H2.riesz((0.5, 0.0)).norm() == pytest.approx(H1.riesz((0.5, 0.0)).norm(), rel=1e-13)


def test_flat_pieces_of_full_dimension():
    par = pg.Patch((0.1, 0.2), (0.5, 0.1), (0.2, 0.6))                          # a parallelogram in R^2
    assert par.dim == par.cartesian_dim == 2 and par.volume == pytest.approx(0.5 * 0.6 - 0.1 * 0.2, rel=1e-14)
    X, W = quadrature.rule(par, 6)
    assert W.sum() == pytest.approx(par.volume, rel=1e-14) and np.all(par.contains(X))
    box = pg.domains.Flat((0, 0, 0), [(1, 0, 0), (0.5, 2, 0), (0.1, 0.2, 3)])     # a parallelepiped
    assert box.volume == pytest.approx(6.0, rel=1e-14) and repr(box).startswith("Flat(")
    mu = pg.LebesgueMeasure(box).with_density(lambda x, y, z: z)
    assert mu.mass()[0] == pytest.approx(6.0 * 1.5, rel=1e-12)
    with pytest.raises(ValueError, match="linearly independent"):
        pg.domains.Flat((0, 0, 0), [(1, 0, 0), (0, 1, 0), (1, 1, 0)])
    assert pg.Box((0, 1), (0, 1)).pieces() == (pg.Box((0, 1), (0, 1)),)


def test_simplex_rules_reach_their_degree():
    tri = pg.Simplex((0, 0), (1, 0), (0, 1))
    for n in (2, 3, 4):
        X, W = quadrature.rule(tri, n)
        for a in range(2 * n):
            b = 2 * n - 1 - a                                                       # total degree 2n - 1: exact
            exact = math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)
            assert W @ (X[:, 0] ** a * X[:, 1] ** b) == pytest.approx(exact, rel=1e-13)
        X, W = quadrature.rule(tri, n)
        assert abs(W @ X[:, 0] ** (2 * n + 1) - 2 / math.factorial(2 * n + 3) * math.factorial(2 * n + 1)) > 1e-16
    tet = pg.Simplex((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    X, W = quadrature.rule(tet, 3)
    exact = math.factorial(2) * math.factorial(1) * math.factorial(2) / math.factorial(5 + 3)
    assert W @ (X[:, 0] ** 2 * X[:, 1] * X[:, 2] ** 2) == pytest.approx(exact, rel=1e-13)
    skew = pg.Simplex((0.2, 0.1, 0.0), (1.0, 0.3, 0.2), (0.1, 0.9, 0.4))         # a triangle in R^3: its area
    e1, e2 = np.array([0.8, 0.2, 0.2]), np.array([-0.1, 0.8, 0.4])
    assert skew.volume == pytest.approx(0.5 * np.linalg.norm(np.cross(e1, e2)), rel=1e-14)
    assert quadrature.rule(skew, 5)[1].sum() == pytest.approx(skew.volume, rel=1e-14)


def test_simplicial_sets_and_their_pieces():
    V = [[0, 0], [1, 0], [1, 1], [0, 1]]
    sq = pg.Simplicial(V, [[0, 1, 2], [0, 2, 3]])
    assert sq.volume == pytest.approx(1.0) and len(sq.pieces()) == 2 and sq.dim == 2
    assert list(sq.locate(np.array([[0.8, 0.1], [0.1, 0.8]]))) == [0, 1]
    assert sq.contains(np.array([0.5, 0.5])) and not sq.contains(np.array([1.5, 0.5]))
    mu = pg.DensityMeasure(pg.LebesgueMeasure(sq), [2.0, -1.0])                  # one value per cell
    assert mu.mass()[0] == pytest.approx(0.5, rel=1e-14) and mu.signed
    with pytest.raises(ValueError, match="degenerate"):
        pg.Simplicial([[0, 0], [1, 0], [2, 0]], [[0, 1, 2]])


def test_simplicial_with_a_sphere_chart_is_the_sphere():
    S = pg.Sphere(1.3, (0.1, 0.0, 0.0))
    ico = pg.Simplicial(0.5 * ICO_V + np.array([0.1, 0, 0]), ICO_F, chart=S)
    assert ico.volume == pytest.approx(4 * np.pi * 1.3 ** 2, rel=1e-14)
    mu = pg.LebesgueMeasure(ico).with_density(lambda x, y, z: z ** 2)
    assert mu.mass()[0] == pytest.approx(4 * np.pi * 1.3 ** 4 / 3, rel=1e-13)
    flat = pg.Simplicial(0.5 * ICO_V, ICO_F)                                    # the polyhedron: a different set
    assert flat.volume < 4 * np.pi * (0.5 * np.linalg.norm(ICO_V[0])) ** 2 * 0.95
    assert ico.contains(np.array(S.to_cartesian(0.3, 1.0))) and not ico.contains(np.array([0.1, 0.0, 0.0]))


def test_parametric_surfaces_as_supports():
    R, r = 1.0, 0.3
    chi = lambda u, v: ((R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v))  # noqa
    torus = pg.Surface(chi, (0, 2 * np.pi), (0, 2 * np.pi))
    assert torus.volume == pytest.approx(4 * np.pi ** 2 * R * r, rel=1e-11)                 # differences: CHECKED
    exact = pg.Surface(chi, (0, 2 * np.pi), (0, 2 * np.pi), derivatives=(
        lambda u, v: (-(R + r * np.cos(v)) * np.sin(u), (R + r * np.cos(v)) * np.cos(u), 0 * u),
        lambda u, v: (-r * np.sin(v) * np.cos(u), -r * np.sin(v) * np.sin(u), r * np.cos(v))))
    assert exact.volume == pytest.approx(4 * np.pi ** 2 * R * r, rel=1e-14)
    mu = pg.LebesgueMeasure(exact).with_density(1.5)
    assert pg.measures.measure_dim(mu) == 2 and mu.mass()[0] == pytest.approx(1.5 * exact.volume, rel=1e-14)


def test_exact_closure_checks():
    B, D = pg.Box((0, 1), (0, 1)), pg.Disk(1.0)
    assert inside_closure(pg.Disk(0.2, (0.5, 0.5)), B) is pg.Truth.TRUE
    assert inside_closure(pg.Disk(0.2, (0.9, 0.5)), B) is pg.Truth.FALSE
    assert inside_closure(pg.Box((0.1, 0.5), (0.2, 0.3)), D) is pg.Truth.TRUE
    assert inside_closure(pg.Box((0.1, 0.9), (0.2, 0.9)), D) is pg.Truth.FALSE
    assert inside_closure(pg.Disk(0.5, (0.5, 0)), D) is pg.Truth.TRUE
    assert inside_closure(pg.Disk(0.5, (0.51, 0)), D) is pg.Truth.FALSE
    assert inside_closure(pg.Simplex((0, 0), (1, 0), (0, 1)), B) is pg.Truth.TRUE
    assert inside_closure(pg.Curve(lambda t: (t, t * t), 0, 1), B) is pg.Truth.UNKNOWN     # sampled, not decided
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M = pg.L2(B)
    with pytest.raises(pg.errors.SupportOutsideDomain, match="exactly"):
        M.riesz(pg.Disk(0.2, (0.9, 0.5)))
