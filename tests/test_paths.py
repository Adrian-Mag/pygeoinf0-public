"""Paths and flat pieces as sets, as supports of measures and as domains of L2; refusals of unbounded functionals."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import quadrature
from pygeoinf0.core.errors import PointEvaluationUndefined, UnboundedFunctional


def test_segment_patch_polyline_arc_geometry():
    s = pg.Segment((0, 0, 0), (3, 4, 0))
    assert s.volume == pytest.approx(5.0) and s.dim == 1 and s.cartesian_dim == 3
    x, w = quadrature.rule(s, 8)
    assert w.sum() == pytest.approx(5.0, rel=1e-14)
    assert np.dot(w, x[:, 0]) == pytest.approx(5 * 1.5, rel=1e-14)               # int x ds = 5 * mean(x)
    assert np.isfinite(s.from_cartesian(1.5, 2.0, 0.0)[0]) and np.isnan(s.from_cartesian(1.5, 2.1, 0.0)[0])
    P = pg.Patch((0, 0, 1), (2, 0, 0), (1, 1, 0))                                  # 45 degrees: area 2 * sqrt2 * sin 45
    x, w = quadrature.rule(P, 6)
    assert w.sum() == pytest.approx(P.volume, rel=1e-14) and P.volume == pytest.approx(2.0, rel=1e-14)
    L = pg.Polyline([(0, 0), (1, 0), (1, 2)])
    assert L.volume == pytest.approx(3.0) and L.reference_breaks == ((1.0,),)
    assert np.allclose(np.column_stack(L.to_cartesian(np.array([0.5, 2.0]))), [[0.5, 0], [1, 1]])
    S = pg.Sphere(2.0)
    A = pg.Arc(S, (2, 0, 0), (0, 2, 0))
    assert A.volume == pytest.approx(math.pi, rel=1e-14)                           # quarter of a great circle
    mid = np.column_stack(A.to_cartesian(np.array([A.volume / 2])))[0]
    assert np.allclose(mid, [math.sqrt(2), math.sqrt(2), 0])
    with pytest.raises(ValueError):
        pg.Arc(S, (2, 0, 0), (-2, 0, 0))                                          # antipodal


def test_L2_on_paths():
    seg = pg.Segment((0, 0), (1, 1))
    M = pg.L2(seg)
    f = M(lambda x, y: x * y)                                                     # along the segment: (s/sqrt2)^2
    assert f.integral() == pytest.approx(math.sqrt(2) ** 3 / 6, rel=1e-13)        # int_0^sqrt2 s^2/2 ds
    L = pg.Polyline([(0, 0), (1, 0), (1, 1)])
    g = pg.L2(L)(lambda x, y: np.abs(x - y))                                       # kink at the vertex
    assert g.integral() == pytest.approx(0.5 + 0.5, rel=1e-13)
    assert g.error < 1e-12
    A = pg.Arc(pg.Sphere(1.0), (1, 0, 0), (0, 1, 0))
    h = pg.L2(A)(lambda x, y, z: x)
    assert h.integral() == pytest.approx(1.0, rel=1e-13)                           # int_0^{pi/2} cos s ds


def test_unbounded_functionals_are_refused_in_L2():
    M = pg.L2(pg.Disk(1.0))
    with pytest.raises(UnboundedFunctional, match=r"s > \(d - k\)/2 = 0.5"):
        M.riesz(pg.Segment((-0.5, 0), (0.5, 0)))
    with pytest.raises(PointEvaluationUndefined):
        M.riesz((0.1, 0.2))
    one = M.riesz(pg.Disk(1.0))
    assert one(0.3, 0.1) == pytest.approx(1.0)
    I = pg.L2("[0, 1]")
    k = I.riesz(pg.Interval("[0.2, 0.5]"))
    assert I(np.sin) @ k == pytest.approx(math.cos(0.2) - math.cos(0.5), rel=1e-13)


def test_dirac_in_Rd_and_plots():
    d = pg.DiracMeasure((0.5, 0.25))
    assert d(pg.Disk(1.0)) == 1.0 and d(pg.Disk(0.1, centre=(2, 2))) == 0.0
    import matplotlib
    matplotlib.use("Agg")
    pg.L2(pg.Segment((0, 0), (1, 1)))(lambda x, y: x).plot()
    pg.L2(pg.Patch((0, 0, 0), (1, 0, 0), (0, 1, 0)))(lambda x, y, z: x * y).plot()
    pg.L2(pg.Circle(1.0))(lambda x, y: x).plot()


def test_clip_is_the_exact_chord():
    D = pg.Disk(1.0)
    c = pg.clip(pg.Segment((-2, 0.6), (2, 0.6)), D)
    assert np.allclose(c.endpoints, [(-0.8, 0.6), (0.8, 0.6)], atol=1e-15)
    assert pg.clip(pg.Segment((-2, 1.5), (2, 1.5)), D) is None
    B = pg.Box("[0, 1]", "[0, 2]")
    c = pg.clip(pg.Segment((-1, -1), (3, 3)), B)
    assert np.allclose(c.endpoints, [(0, 0), (1, 1)], atol=1e-15)
    C = pg.Cylinder(1.0, (0, 2))
    c = pg.clip(pg.Segment((0, 0, -1), (0, 0, 3)), C)
    assert np.allclose(c.endpoints, [(0, 0, 0), (0, 0, 2)], atol=1e-15)
    c = pg.clip(pg.Segment((-3, 0, 0.5), (3, 0, 0.5)), pg.Ball(1.0))
    assert c.volume == pytest.approx(2 * math.sqrt(0.75), rel=1e-14)
    H = pg.Sobolev(D, 1.5, length_scale=0.2)
    with pytest.raises(pg.errors.SupportOutsideDomain, match="pg.clip"):
        H.riesz(pg.Segment((-2, 0.6), (2, 0.6)))
    with pytest.raises(pg.errors.SupportOutsideDomain, match="does not meet"):
        H.riesz(pg.Segment((-2, 1.5), (2, 1.5)))
    H.riesz(pg.clip(pg.Segment((-2, 0.6), (2, 0.6)), D))


def test_usability_messages():
    D = pg.Disk(1.0)
    H = pg.Sobolev(D, 1.5, length_scale=0.2)
    with pytest.raises(TypeError, match="length_scale="):
        pg.Sobolev(D, 1.5, 0.2)
    with pytest.raises(pg.errors.UnboundedFunctional, match="length_scale"):
        pg.L2(D).riesz(pg.Segment((-1, 0), (1, 0)))
    G = pg.Functionals(H.riesz([pg.Segment((-1, 0), (1, 0))]), into=pg.Euclidean(1))
    with pytest.raises(pg.SpaceMismatch, match="make one first"):
        G(np.zeros(3))
    with pytest.raises(pg.SpaceMismatch, match="normal operator"):
        pg.solvers.damping_for_misfit(G, pg.Euclidean(1)(np.ones(1)), 1)
