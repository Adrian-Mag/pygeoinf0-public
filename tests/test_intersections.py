"""L2 pairings of symbolic representers of overlapping sets whose bounding boxes are degenerate in a coordinate (EXP-032
finding 1: sub-arcs in z = 0 and sub-segments parallel to an axis were paired as disjoint, giving 0), and the zero
pairing with a non-zero estimate (EXP-032 finding 3). References: closed forms and scipy.integrate.quad, computed
from the parametrisations, not from the library's intersection."""
import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

import pygeoinf0 as pg
from pygeoinf0.sets.intersections import intersection


def gram(M, *measures):
    k = [M.riesz(m) for m in measures]
    return np.array([[M.inner(a, b) for b in k] for a in k])


def on_circle(t):                          # the unit circle in z = 0
    return (math.cos(t), math.sin(t), 0.0)


def test_overlapping_sub_arcs_in_the_plane_z_0():
    S2 = pg.Sphere(1.0)
    D = pg.Arc(S2, (1, 0, 0), (0, 1, 0))
    M = pg.L2(D)
    A, B = pg.Arc(S2, on_circle(0.1), on_circle(0.9)), pg.Arc(S2, on_circle(0.5), on_circle(1.3))
    assert np.ptp(np.array(A.bounding_box)[2]) == 0          # the degenerate coordinate of the finding
    G = gram(M, pg.LebesgueMeasure(A), pg.LebesgueMeasure(B))
    assert G[0, 1] == pytest.approx(0.4, rel=1e-13) and G[1, 0] == pytest.approx(0.4, rel=1e-13)
    assert G[0, 0] == pytest.approx(0.8, rel=1e-13)
    wa, wb = (lambda x, y, z: 1 + x), (lambda x, y, z: y * y)
    G = gram(M, pg.LebesgueMeasure(A).with_density(wa), pg.LebesgueMeasure(B).with_density(wb))
    ref = quad(lambda t: (1 + math.cos(t)) * math.sin(t) ** 2, 0.5, 0.9, epsabs=1e-15, epsrel=1e-15)[0]
    assert G[0, 1] == pytest.approx(ref, rel=1e-12)
    C, E = pg.Arc(S2, on_circle(0.3), on_circle(0.6)), pg.Arc(S2, on_circle(0.9), on_circle(1.4))
    G = gram(M, pg.LebesgueMeasure(A), pg.LebesgueMeasure(C), pg.LebesgueMeasure(E))
    assert G[0, 1] == pytest.approx(0.3, rel=1e-13)             # nested
    assert G[0, 2] == 0.0 and G[1, 2] == 0.0                     # touching at one point, and apart


@pytest.mark.parametrize("p, u", [((0.0, 0.3), (1.0, 0.0)), ((0.2, 0.5, 0.0), (0.0, 0.0, 1.0)),
                                  ((0.2, 0.0, 0.7), (0.0, 1.0, 0.0))])
def test_overlapping_axis_parallel_sub_segments(p, u):
    p, u = np.array(p), np.array(u)
    D = pg.Segment(p, p + u)
    M = pg.L2(D)
    A, B = pg.Segment(p + 0.1 * u, p + 0.6 * u), pg.Segment(p + 0.4 * u, p + 0.9 * u)
    G = gram(M, pg.LebesgueMeasure(A), pg.LebesgueMeasure(B))
    assert G[0, 1] == pytest.approx(0.2, rel=1e-13)
    j = int(np.argmax(u))                                         # the coordinate along the segment

    def w(*x):
        return 1 + np.asarray(x[j]) ** 2
    G = gram(M, pg.LebesgueMeasure(A).with_density(w), pg.LebesgueMeasure(B))
    lo, hi = p[j] + 0.4, p[j] + 0.6
    assert G[0, 1] == pytest.approx((hi - lo) + (hi ** 3 - lo ** 3) / 3, rel=1e-13)
    C = pg.Segment(p + 0.6 * u, p + 0.8 * u)                       # touching A at one end
    assert gram(M, pg.LebesgueMeasure(A), pg.LebesgueMeasure(C))[0, 1] == 0.0


def test_sub_polylines_overlapping_across_a_vertex_in_z_0():
    D = pg.Polyline([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    M = pg.L2(D)
    A = pg.Polyline([(0.5, 0, 0), (1, 0, 0), (1, 0.5, 0)])
    B = pg.Segment((1, 0.2, 0), (1, 0.9, 0))
    E = pg.Polyline([(0.2, 0, 0), (1, 0, 0), (1, 0.1, 0)])
    G = gram(M, pg.LebesgueMeasure(A), pg.LebesgueMeasure(B), pg.LebesgueMeasure(E))
    assert G[0, 1] == pytest.approx(0.3, rel=1e-13)
    assert G[0, 2] == pytest.approx(0.6, rel=1e-13)               # 0.5 along x and 0.1 along y
    assert G[1, 2] == 0.0


def test_overlapping_sub_curves_in_the_plane_z_0():
    def gamma(t):
        t = np.asarray(t, float)
        return np.stack([t, t * t, 0 * t], axis=-1)
    D = pg.Curve(gamma, 0.0, 1.2)
    M = pg.L2(D)
    A, B = pg.Curve(gamma, 0.1, 0.7), pg.Curve(gamma, 0.4, 1.0)
    G = gram(M, pg.LebesgueMeasure(A), pg.LebesgueMeasure(B))
    ref = quad(lambda t: math.sqrt(1 + 4 * t * t), 0.4, 0.7, epsabs=1e-15, epsrel=1e-15)[0]
    assert G[0, 1] == pytest.approx(ref, rel=1e-10)


def test_overlapping_patches_in_the_plane_z_0():
    D = pg.Patch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    M = pg.L2(D)
    A = pg.Patch((0.1, 0.1, 0), (0.5, 0, 0), (0, 0.5, 0))
    B = pg.Patch((0.4, 0.3, 0), (0.5, 0, 0), (0, 0.6, 0))
    G = gram(M, pg.LebesgueMeasure(A).with_density(lambda x, y, z: 1 + x), pg.LebesgueMeasure(B).with_density(
        lambda x, y, z: y))
    assert G[0, 1] == pytest.approx(0.3 * 0.135, rel=1e-13)      # int_0.4^0.6 (1 + x) dx  int_0.3^0.6 y dy
    C = pg.Patch((0.6, 0.0, 0), (0.3, 0, 0), (0, 0.3, 0))          # touching A along x = 0.6
    assert gram(M, pg.LebesgueMeasure(A), pg.LebesgueMeasure(C))[0, 1] == 0.0
    tilted = pg.Patch((0.2, 0.2, 0), (0.3, 0.3, 0), (-0.3, 0.3, 0))   # not a coordinate rectangle of D: refused
    with pytest.raises(pg.Refused, match="not representable"):
        intersection(A, tilted, D)


def test_full_dimensional_sets_keep_their_rules():
    D = pg.Box((0, 1), (0, 1))
    left, right = pg.Box((0, 0.5), (0, 1)), pg.Box((0.5, 1), (0, 1))
    assert intersection(left, right, D) is None                    # touching boxes: a null intersection
    assert intersection(left, pg.Box((0.25, 0.75), (0.5, 1)), D) == pg.Box((0.25, 0.5), (0.5, 1))
    d1, d2 = pg.Disk(0.25, (0.25, 0.5)), pg.Disk(0.25, (0.75, 0.5))
    assert intersection(d1, d2, D) is None                         # touching disks
    assert intersection(d1, pg.Disk(0.1, (0.9, 0.9)), D) is None    # apart
    assert intersection(pg.Disk(0.1, (0.3, 0.5)), left, D) == pg.Disk(0.1, (0.3, 0.5))
    with pytest.raises(pg.Refused):
        intersection(d1, pg.Disk(0.25, (0.5, 0.5)), D)             # overlapping disks: not representable


def test_a_zero_pairing_with_a_nonzero_estimate_warns_without_dividing_by_zero():
    D = pg.Segment((0.0, 0.0), (1.0, 0.0))
    M = pg.L2(D)
    g = 0.5 + 0.5 * np.polynomial.legendre.leggauss(48)[0]          # the nodes of the lower order only

    def spikes(x, y):
        x = np.atleast_1d(np.asarray(x, float))
        return (np.min(np.abs(x[:, None] - g[None, :]), axis=1) < 1e-12).astype(float)
    k1 = M.riesz(pg.LebesgueMeasure(D).with_density(spikes, sup=1.0))
    k2 = M.riesz(pg.LebesgueMeasure(D).with_density(2.0))           # symbolic too: the pairing is pair()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = M.inner(k1, k2)
    assert v == 0.0
    assert any("relative error inf" in str(x.message) for x in w)
