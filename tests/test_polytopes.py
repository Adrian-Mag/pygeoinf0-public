"""Polytopes with vertices and an outer / inner flag, and plots of sets (plan 058): vertices of known polytopes
(a square, a triangle, a redundant half-space), refusals (unbounded, empty, flat, R^4), the support polygon of a disc
and an ellipse against the analytic areas, the flag kept by translation, scaling and intersection and set by
K.image(T).polytope(Q), the inner polytope of a box route inside the outer one, and plots drawn with Agg."""
import math

import matplotlib
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported
from pygeoinf0.plotting.sets import directions_for

matplotlib.use("Agg")
R2 = pg.Euclidean(2)
SQUARE = ([[1, 0], [-1, 0], [0, 1], [0, -1]], [1, 1, 1, 1])


def area(V):
    x, y = V[:, 0], V[:, 1]
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def same_points(V, W):
    return len(V) == len(W) and all(np.min(np.linalg.norm(W - v, axis=1)) < 1e-12 for v in V)


def test_vertices_of_known_polytopes_from_all_half_spaces():
    V = pg.Polytope(R2, *SQUARE).vertices()
    assert same_points(V, np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]))
    (a, b), (c, d) = V[1] - V[0], V[2] - V[1]
    assert a * d - b * c > 0                                       # counter-clockwise
    tri = pg.Polytope(R2, [[-1, 0], [0, -1], [1, 1]], [0, 0, 1])
    assert same_points(tri.vertices(), np.array([[0, 0], [1, 0], [0, 1]]))
    # a redundant half-space changes nothing (the consecutive-line polygon would cut a corner off at x + y = 5)
    red = pg.Polytope(R2, [[-1, 0], [0, -1], [1, 1], [1, 0], [1, 1]], [0, 0, 1, 5, 5])
    assert same_points(red.vertices(), tri.vertices())
    seg = pg.Polytope(pg.Euclidean(1), [[1], [-1], [2]], [2, 1, 8])
    assert np.allclose(seg.vertices().ravel(), [-1, 2])
    cube = pg.Polytope(pg.Euclidean(3), np.r_[np.eye(3), -np.eye(3)], np.ones(6))
    assert same_points(cube.vertices(), np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]))


@pytest.mark.parametrize("A, b, why, error", [
    ([[1, 0], [0, 1]], [1, 1], "unbounded", pg.errors.NotDefined),       # no finite vertex list exists (plan 068 P1)
    ([[1, 0], [-1, 0], [0, 1], [0, -1]], [-1, 0, 1, 1], "empty", Unsupported),
    ([[1, 0], [-1, 0], [0, 1], [0, -1]], [0, 0, 1, 1], "no interior", Unsupported),
])
def test_vertices_are_refused_for_unbounded_empty_and_flat_polytopes(A, b, why, error):
    with pytest.raises(error, match=why):
        pg.Polytope(R2, A, b).vertices()


def test_vertices_are_refused_above_three_dimensions():
    with pytest.raises(Unsupported, match="R\\^4"):
        pg.Polytope(pg.Euclidean(4), np.r_[np.eye(4), -np.eye(4)], np.ones(8)).vertices()


@pytest.mark.parametrize("k", [12, 90, 720])
def test_the_support_polygon_of_a_disc_and_an_ellipse_contains_them(k):
    disc = pg.NormBall(R2, 2.0).polytope(directions_for(2, k))
    assert disc.approx == "outer"
    assert area(disc.vertices()) == pytest.approx(4 * k * math.tan(math.pi / k), rel=1e-12)   # circumscribed k-gon
    E = pg.Ellipsoid(pg.Euclidean(2), 1.0, covariance=[4.0, 1.0])                          # semi-axes 2 and 1
    a = area(E.polytope(directions_for(2, k)).vertices())
    assert a >= 2 * math.pi and a - 2 * math.pi <= 2 * math.pi * 4 * math.tan(math.pi / k) ** 2


def test_the_flag_is_required_and_kept_where_it_holds():
    disc = pg.NormBall(R2, 1.0)
    with pytest.raises(ValueError):
        pg.Polytope(R2, *SQUARE, approx="outer")                    # outer of what
    O = disc.polytope(directions_for(2, 8))
    assert (O + np.array([1.0, 0.0])).approx == "outer" and (2.0 * O).approx == "outer"
    assert (2.0 * O).member(np.array([0.0, 1.9])) is pg.Truth.TRUE
    O2 = disc.polytope(directions_for(2, 5))
    assert (O & O2).approx == "outer" and (O & O2).of is disc
    assert (O & pg.Polytope(R2, *SQUARE)).approx == "exact"         # the relation to the disc is not claimed
    assert O.claims.status("outer")[0]


def test_a_feasible_image_gives_its_outer_and_inner_polytopes():
    import test_varying_box as vb
    M, D, box, G, c, N, d = vb.problem(None)
    K = pg.FeasibleSet(box, G, N, d)
    P = K.image(pg.Functionals(M.family([c, M(lambda x: 1.0 - x, breaks=[0, 1])])))
    Q = directions_for(2, 24)
    O, I = P.polytope(Q), P.polytope(Q, approx="inner")
    assert O.approx == "outer" and I.approx == "inner" and O.of is P and I.of is P
    assert np.array_equal(O.b, P.outer_polytope(Q)[1])
    assert np.all(O.A @ I.vertices().T <= O.b[:, None] + 1e-9)       # inner inside outer
    assert area(I.vertices()) <= area(O.vertices())
    with pytest.raises(Unsupported, match="feasible points"):
        pg.NormBall(R2, 1.0).polytope(Q, approx="inner")


def test_sets_are_drawn_in_one_two_and_three_dimensions():
    ax = pg.NormBall(R2, 1.0).plot(directions=60)
    pg.Polytope(R2, *SQUARE).plot(ax=ax)
    assert len(ax.patches) == 2 and "outer" in ax.patches[0].get_label()
    pg.Polytope(pg.Euclidean(1), [[1], [-1]], [2, 1]).plot()
    pg.NormBall(pg.Euclidean(3), 1.0).plot(directions=40)
    with pytest.raises(ValueError):
        pg.Polytope(pg.Euclidean(4), np.r_[np.eye(4), -np.eye(4)], np.ones(8)).plot()


def test_a_bgp_result_draws_a_pair_of_properties():
    rng = np.random.default_rng(0)
    X, D = pg.Euclidean(4), pg.Euclidean(2)
    G = pg.Matrix(rng.standard_normal((2, 4)), X, D)
    K = pg.FeasibleSet(pg.NormBall(X, 1.0), G, pg.NormBall(D, 0.1), D(np.array([0.3, -0.2])))
    res = pg.inference.methods.BGP(K.image(pg.Matrix(np.eye(3, 4), X, pg.Euclidean(3))), None, None)
    ax = res.plot(pair=(0, 2), directions=36)
    assert "outer" in ax.patches[0].get_label()
