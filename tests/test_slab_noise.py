"""Slab-type noise in box BGP (plan 060): one slab by the exact 1-D dual against a closed form and against the hand
bathtub of plan 057 (symmetric and asymmetric boxes); a slab against thin ellipsoids inside it; two slabs and a
paired polytope against the coordinate-box route, also through a rotation; emptiness; refusals; SOLA's slab family;
the inner polytope from the exact route's witnesses."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported

M = pg.L2(pg.Interval(0, 1))
ONE = M(lambda x: 1.0 + 0 * x, breaks=[0, 1])
X = M(lambda x: x + 0 * x, breaks=[0, 1])


def one_datum(d, w, box=None):
    D = pg.Euclidean(1)
    G = pg.Functionals(M.family([ONE]), into=D)
    return pg.FeasibleSet(box or pg.PointwiseBox(M, envelope=1.0), G, pg.Slab(D, [1.0], w), D([d])), D, G


def test_one_slab_is_exact_against_the_closed_form():
    # sup int x m over |m| <= 1, |0.2 - int m| <= 0.05: m = -1 on [0, s), 1 on [s, 1], 1 - 2 s = 0.25
    K, D, G = one_datum(0.2, 0.05)
    b = K.bound(X)
    s = 0.375
    assert b.status == "EXACT" and b.lower == pytest.approx(0.5 - s * s, abs=1e-14)
    assert b.upper == pytest.approx(0.5 - s * s, abs=1e-14) and b.certificate[0] == pytest.approx(s, abs=1e-12)
    m = b.witness                                                     # an element, in the box and in the slab
    assert K.prior.member(m) is pg.Truth.TRUE and abs(0.2 - M.inner(ONE, m)) <= 0.05
    inactive, _, _ = one_datum(0.9, 0.2)                              # the box's own extremal field fits the slab
    assert inactive.bound(X).lower == pytest.approx(0.5, abs=1e-14) and inactive.bound(X).status == "EXACT"


@pytest.mark.parametrize("asymmetric", [False, True])
def test_one_slab_equals_the_hand_bathtub_with_a_varying_box(asymmetric):
    import test_varying_box as vb
    lower = vb.L_ASYM if asymmetric else None
    M2, D, box, G, c, N, d = vb.problem(lower)
    b = pg.FeasibleSet(box, G, pg.Slab(D, [1.0], vb.W), d).bound(c)
    assert b.status == "EXACT" and b.lower == pytest.approx(vb.hand_sup(lower), abs=1e-12)
    via_box = pg.FeasibleSet(box, G, N, d).bound(c)                   # the same set, by the cell route
    assert via_box.lower - 1e-12 <= b.lower <= via_box.upper + 1e-12


def test_a_slab_bounds_the_thin_ellipsoids_inside_it():
    D = pg.Euclidean(2)
    G = pg.Functionals(M.family([ONE, X]), into=D)
    box = pg.PointwiseBox(M, envelope=M(lambda x: 1.0 + 0.3 * x, breaks=[0, 1]))
    d, w = D([0.2, 0.1]), 0.05
    c = M(lambda x: np.cos(3 * x), resolution=0.05)
    slab = pg.FeasibleSet(box, G, pg.Slab(D, [1.0, 0.0], w), d).bound(c)
    assert slab.status == "EXACT"
    for L in (1.0, 10.0, 1e3):
        K = pg.FeasibleSet(box, G, pg.Ellipsoid(D, 1.0, covariance=[w * w, L * L]), d)
        K.sub = 32
        e = K.bound(c)
        assert e.lower <= slab.upper + 1e-12                          # the ellipsoid lies in the slab
    assert e.upper >= slab.lower - 1e-6                               # ... and fills it as L grows


def test_two_slabs_and_a_paired_polytope_are_the_coordinate_box_route():
    D = pg.Euclidean(2)
    G = pg.Functionals(M.family([ONE, X]), into=D)
    box = pg.PointwiseBox(M, envelope=1.0)
    d, w = D([0.2, 0.1]), np.array([0.05, 0.02])
    c = M(lambda x: np.sin(4 * x), resolution=0.05)
    ref = pg.FeasibleSet(box, G, pg.CoordinateBox(D, -w, w), d).bound(c)
    two = pg.FeasibleSet(box, G, pg.Slab(D, [1.0, 0.0], w[0]) & pg.Slab(D, [0.0, 1.0], w[1]), d).bound(c)
    poly = pg.FeasibleSet(box, G, pg.Polytope(D, [[1, 0], [-1, 0], [0, 1], [0, -1]], np.repeat(w, 2)), d).bound(c)
    for b in (two, poly):
        assert b.status == ref.status and b.upper == pytest.approx(ref.upper, abs=1e-10)
        assert b.lower == pytest.approx(ref.lower, abs=1e-8)
    # rotated slabs: the box route on the transformed data (G' = A G, d' = A d) gives the same bound
    A = np.array([[1.0, 1.0], [1.0, -2.0]])
    rot = pg.FeasibleSet(box, G, pg.Slab(D, A[0], w[0]) & pg.Slab(D, A[1], w[1]), d).bound(c)
    D2 = pg.Euclidean(2)
    G2 = pg.Functionals(M.family([ONE * A[0, 0] + X * A[0, 1], ONE * A[1, 0] + X * A[1, 1]]), into=D2)
    ref2 = pg.FeasibleSet(box, G2, pg.CoordinateBox(D2, -w, w), D2(A @ d.values)).bound(c)
    assert rot.upper == pytest.approx(ref2.upper, abs=1e-9) and rot.status == ref2.status


def test_an_unreachable_slab_is_empty_with_a_certificate():
    K, D, G = one_datum(5.0, 0.05)
    assert K.is_empty() is pg.Truth.TRUE and K.emptiness.status == "EMPTY"
    assert K.bound(X).status == "EMPTY"


def test_slab_noise_refusals():
    D = pg.Euclidean(2)
    G = pg.Functionals(M.family([ONE, X]), into=D)
    box = pg.PointwiseBox(M, envelope=1.0)
    d = D([0.2, 0.1])
    bad = [(pg.Polytope(D, [[1, 0], [-1, 0], [0, 1]], [1, 1, 1]), "opposite"),
           (pg.Slab(D, [0.0, 0.0], 0.1), "zero normal"),
           (pg.Slab(D, [1.0, 0.0], -0.1), "half-width")]
    for N, why in bad:
        with pytest.raises(Unsupported, match=why):
            pg.FeasibleSet(box, G, N, d).bound(X)


def test_sola_slab_family():
    rng = np.random.default_rng(2)
    D = pg.Euclidean(3, covariance=[0.1, 0.2, 0.3])
    Xc = rng.standard_normal((3, 2))
    slabs = pg.inference.slab_family(D, Xc, sigma=[0.5, 2.0], mass=3.0)
    e = D(rng.standard_normal(3))
    for j, S in enumerate(slabs):
        assert D.inner(S.normal, e) == pytest.approx(Xc[:, j] @ e.values, rel=1e-12)
        assert S.half_width == pytest.approx(3.0 * [0.5, 2.0][j])


def test_the_inner_polytope_takes_the_exact_routes_witnesses():
    K, D, G = one_datum(0.2, 0.05, pg.PointwiseBox(M, envelope=M(lambda x: 1.0 + 0.5 * x, breaks=[0, 1])))
    P = K.image(pg.Functionals(M.family([X, M(lambda x: np.cos(5 * x), resolution=0.05)])))
    a = 2 * np.pi * np.arange(16) / 16
    Q = np.c_[np.cos(a), np.sin(a)]
    O, I = P.polytope(Q), P.polytope(Q, approx="inner")
    assert np.all(O.A @ I.vertices().T <= O.b[:, None] + 1e-9)
    assert all(P.bound(q).status == "EXACT" for q in Q)
    assert O.approx == "outer" and I.approx == "inner"
