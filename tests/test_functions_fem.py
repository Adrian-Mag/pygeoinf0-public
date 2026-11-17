"""The P1 hat function (plan 021 P1, work package F): special values, partition of unity, closed forms against exact
cell-by-cell integration (the contract test's whole-domain quadrature cannot resolve the kinks of a box grid), the
gradient, the claims, and the refusals."""
import itertools
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.sets import Interval

I01 = Interval("[0, 1]")
NODES = (0.0, 0.15, 0.4, 0.7, 1.0)


def cell_integral(g, grids, n=3):
    """The integral of g over the box of the grids, by n-point Gauss-Legendre on every cell (exact for g piecewise
    polynomial of degree 2n-1 per axis on the cells)."""
    t, w = np.polynomial.legendre.leggauss(n)
    pts, wts = [], []
    for grid in grids:
        p, q = [], []
        for a, b in itertools.pairwise(grid):
            p.append(0.5 * (b - a) * (t + 1) + a)
            q.append(0.5 * (b - a) * w)
        pts.append(np.concatenate(p))
        wts.append(np.concatenate(q))
    G = np.meshgrid(*pts, indexing="ij")
    W = np.ones_like(G[0])
    for j, q in enumerate(wts):
        shape = [1] * len(wts)
        shape[j] = -1
        W = W * q.reshape(shape)
    return float(np.sum(W * g(*G)))


def test_special_values_on_a_non_uniform_grid():
    x = np.array(NODES)
    for i in range(5):
        f = F.interval.hat(I01, index=i, nodes=NODES)
        assert np.allclose(f(x), np.eye(5)[i], atol=1e-15)
    f = F.interval.hat(I01, index=2, nodes=NODES)
    # linear between the nodes: (x - x_1) / (x_2 - x_1) on the left cell, (x_3 - x) / (x_3 - x_2) on the right
    assert f(0.275) == pytest.approx(0.5)
    assert f(0.55) == pytest.approx(0.5)
    assert f(0.2) == pytest.approx((0.2 - 0.15) / 0.25) and f(0.6) == pytest.approx(0.1 / 0.3)
    assert f(0.1) == 0.0 and f(0.9) == 0.0
    # the end hats have one side only
    assert F.interval.hat(I01, index=0, nodes=NODES)(0.075) == pytest.approx(0.5)
    assert F.interval.hat(I01, index=4, nodes=NODES)(0.85) == pytest.approx(0.5)


def test_uniform_grid_matches_the_formula_on_a_shifted_interval():
    D = Interval(-2.0, 3.0)
    f = F.interval.hat(D, index=3, count=6)             # nodes -2, -1, 0, 1, 2, 3
    assert f.params["count"] == 6
    x = np.linspace(-2, 3, 1001)
    assert np.allclose(f(x), np.maximum(0.0, 1.0 - np.abs(x - 1.0)), atol=1e-13)


@pytest.mark.parametrize("D, n", [(Interval(-1.0, 2.0), 7), (pg.Box((0.0, 2.0), (-1.0, 1.0)), 4),
                                  (pg.Box((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 3)])
def test_hats_sum_to_one_on_the_closed_domain(D, n):
    d = 1 if isinstance(D, Interval) else len(D.sides)
    rng = np.random.default_rng(3)
    sides = [(D.a, D.b)] if d == 1 else [(s.a, s.b) for s in D.sides]
    P = np.stack([rng.uniform(a, b, 200) for a, b in sides])
    P[:, :2] = np.array([[a] for a, _ in sides])                      # the corner of the lower faces
    P[:, 2:4] = np.array([[b] for _, b in sides])                     # and of the upper faces
    total = 0.0
    for idx in itertools.product(range(n), repeat=d):
        f = F.on(D).hat(index=idx[0] if d == 1 else idx, count=n)
        total = total + f(*P)
    assert np.allclose(total, 1.0, atol=1e-13)


def test_sum_to_one_on_a_non_uniform_box_grid():
    D = pg.Box((0.0, 1.0), (0.0, 1.0))
    gx, gy = (0.0, 0.1, 0.5, 1.0), (0.0, 0.7, 1.0)
    x, y = np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, 1, 37), indexing="ij")
    total = sum(F.box.hat(D, index=(i, j), nodes=(gx, gy))(x, y) for i in range(4) for j in range(3))
    assert np.allclose(total, 1.0, atol=1e-13)


def test_tensor_hat_is_the_product_of_axis_hats():
    D = pg.Box((0.0, 1.0), (0.0, 2.0))
    f = F.box.hat(D, index=(1, 2), nodes=((0.0, 0.3, 1.0), (0.0, 0.5, 1.5, 2.0)))
    fx = F.interval.hat(Interval(0.0, 1.0), index=1, nodes=(0.0, 0.3, 1.0))
    fy = F.interval.hat(Interval(0.0, 2.0), index=2, nodes=(0.0, 0.5, 1.5, 2.0))
    x, y = np.meshgrid(np.linspace(0, 1, 23), np.linspace(0, 2, 29), indexing="ij")
    assert np.allclose(f(x, y), fx(x) * fy(y), atol=1e-15)
    assert f(0.3, 1.5) == 1.0


GRIDS_2D = [((0.0, 0.3, 1.0), (0.0, 0.2, 0.9, 1.0)), ((0.0, 0.5, 1.0), (0.0, 1.0))]
GRIDS_3D = [((0.0, 0.3, 1.0), (0.0, 0.2, 0.9, 1.0), (0.0, 0.4, 1.0))]


@pytest.mark.parametrize("grids", GRIDS_2D + GRIDS_3D)
def test_closed_forms_on_kinked_box_grids(grids):
    d = len(grids)
    D = pg.Box(*[(0.0, 1.0)] * d)
    for idx in itertools.product(*[range(len(g)) for g in grids]):
        f = F.box.hat(D, index=idx, nodes=grids)
        gs = [np.array(g) for g in grids]
        assert f.closed["integral"][0] == pytest.approx(cell_integral(f, gs), rel=1e-13, abs=1e-15)
        assert f.closed["l2_norm_sq"][0] == pytest.approx(cell_integral(lambda *X, f=f: f(*X) ** 2, gs), rel=1e-13)
        assert f.closed["max"][0] == 1.0 == f(*[g[i] for g, i in zip(grids, idx)])
        assert f.integral()[0] == f.closed["integral"][0]


def test_normalised_hat_has_unit_mass():
    D = pg.Box((0.0, 1.0), (0.0, 1.0))
    f = F.box.hat(D, index=(1, 1), nodes=GRIDS_2D[0]).normalised()
    assert f.integral()[0] == 1.0
    assert cell_integral(f, [np.array(g) for g in GRIDS_2D[0]]) == pytest.approx(1.0, rel=1e-13)


def test_interval_integral_is_the_triangle_area():
    f = F.interval.hat(I01, index=2, nodes=NODES)
    assert f.closed["integral"][0] == pytest.approx(0.5 * (0.7 - 0.15))
    assert F.interval.hat(I01, index=0, nodes=NODES).closed["integral"][0] == pytest.approx(0.5 * 0.15)
    assert F.interval.hat(I01, index=4, nodes=NODES).closed["l2_norm_sq"][0] == pytest.approx(0.3 / 3)


@pytest.mark.parametrize("grids", GRIDS_2D + GRIDS_3D)
def test_gradient_against_finite_differences(grids):
    d = len(grids)
    D = pg.Box(*[(0.0, 1.0)] * d)
    rng = np.random.default_rng(5)
    P = rng.uniform(0.01, 0.99, (60, d))
    h = 1e-7
    for idx in itertools.product(*[range(len(g)) for g in grids]):
        f = F.box.hat(D, index=idx, nodes=grids)
        G = f.gradient(*P.T)
        assert G.shape == (60, d)
        for ax in range(d):
            E = np.zeros(d)
            E[ax] = h
            fd = (f(*(P + E).T) - f(*(P - E).T)) / (2 * h)
            near_kink = np.any([np.abs(P[:, ax] - g) < 1e-5 for g in grids[ax]], axis=0)
            assert np.allclose(fd[~near_kink], G[~near_kink, ax], atol=1e-5)


def test_support_and_boundary_claims():
    f = F.interval.hat(I01, index=2, nodes=NODES)
    assert f.support.kind == "interior" and f.regularity.vanishes_at_boundary == "all orders"
    assert f.support.contains(np.array([0.15, 0.4, 0.7, 0.71, 0.14])).tolist() == [True, True, True, False, False]
    first = F.interval.hat(I01, index=0, nodes=NODES)
    assert first.support.kind == "touches" and first.regularity.vanishes_at_boundary is None and first(0.0) == 1.0
    second = F.interval.hat(I01, index=1, nodes=NODES)               # support [0, 0.4]: f(0) = 0 but f'(0) != 0
    assert second.support.kind == "touches" and second.regularity.vanishes_at_boundary == 1 and second(0.0) == 0.0
    assert second.gradient(0.05)[0] == pytest.approx(1 / 0.15)
    # a box hat vanishes at the boundary only if every axis hat does
    B = pg.Box((0.0, 1.0), (0.0, 1.0))
    assert F.box.hat(B, index=(2, 1), nodes=(NODES, NODES)).regularity.vanishes_at_boundary == 1
    assert F.box.hat(B, index=(2, 2), nodes=(NODES, NODES)).regularity.vanishes_at_boundary == "all orders"
    assert F.box.hat(B, index=(2, 0), nodes=(NODES, NODES)).regularity.vanishes_at_boundary is None
    g = F.box.hat(B, index=(2, 1), nodes=(NODES, NODES))
    edge = np.linspace(0, 1, 11)
    for face in (g(edge, 0 * edge), g(edge, 0 * edge + 1), g(0 * edge, edge), g(0 * edge + 1, edge)):
        assert np.all(face == 0.0)


def test_regularity_is_c0_piecewise_and_affine_for_two_nodes():
    kinked = F.interval.hat(I01, index=1, count=3)
    assert kinked.regularity.order == 0 and kinked.regularity.piecewise
    assert kinked.regularity.sobolev_bound() == (1.5, True)
    affine = F.interval.hat(I01, index=0, count=2)                    # 1 - x
    assert affine.regularity.order == math.inf and not affine.regularity.piecewise
    assert affine(0.25) == 0.75 and affine.gradient(0.3)[0] == -1.0
    mixed = F.box.hat(pg.Box((0.0, 1.0), (0.0, 1.0)), index=(0, 1), count=(2, 3))
    assert mixed.regularity.order == 0 and mixed.regularity.piecewise


def test_membership_follows_the_regularity():
    f = F.interval.hat(I01, index=1, count=4)
    assert f.element(pg.L2(I01)) is not None
    assert f.element(pg.Sobolev(I01, 1, length_scale=0.2)) is not None
    with pytest.raises(pg.Refused, match="H\\^s for s < 1.5"):
        f.element(pg.Sobolev(I01, 2, length_scale=0.2))
    C = F.interval.hat(I01, index=1, count=4)
    assert C.claims is not None and all(c.status in ("theorem", "derived", "checked") for c in C.claims)


def test_catalogue_lists_the_family():
    row = {n: (k, s) for n, k, s in F.catalogue()}["hat"]
    assert row[0] == ("interval", "box") and "P1" in row[1]


@pytest.mark.parametrize("kwargs, match", [
    ({"index": 5, "count": 5}, "index 5 on axis 0 is outside 0..4"),
    ({"index": -1, "count": 5}, "outside 0..4"),
    ({"index": 1.0, "count": 5}, "index must be an integer"),
    ({"index": True, "count": 5}, "index must be an integer"),
    ({"index": 1}, "give exactly one of nodes, count"),
    ({"index": 1, "count": 5, "nodes": NODES}, "give exactly one of nodes, count"),
    ({"index": 0, "count": 1}, "at least 2 nodes"),
    ({"index": 0, "count": 2.5}, "count must be an integer"),
    ({"index": 0, "nodes": (0.0, 0.5, 0.5, 1.0)}, "strictly increasing"),
    ({"index": 0, "nodes": (0.0, 0.7, 0.5, 1.0)}, "strictly increasing"),
    ({"index": 0, "nodes": (0.1, 0.5, 1.0)}, "must run from 0 to 1"),
    ({"index": 0, "nodes": (0.0, 0.5)}, "must run from 0 to 1"),
    ({"index": 0, "nodes": (0.0, float("nan"), 1.0)}, "finite"),
    ({"index": 0, "nodes": ((0.0, 1.0),)}, "one-dimensional"),
    ({"count": 3}, "index"),
])
def test_interval_refusals(kwargs, match):
    with pytest.raises((pg.Refused, TypeError), match=match):
        F.interval.hat(I01, **kwargs)


def test_box_refusals_and_other_domains():
    B = pg.Box((0.0, 1.0), (0.0, 1.0))
    with pytest.raises(pg.Refused, match="one entry per axis"):
        F.box.hat(B, index=1, count=3)
    with pytest.raises(pg.Refused, match="one entry per axis"):
        F.box.hat(B, index=(1, 1, 1), count=3)
    with pytest.raises(pg.Refused, match="axis 1 is outside 0..2"):
        F.box.hat(B, index=(1, 3), count=3)
    with pytest.raises(pg.Refused, match="at least 2 nodes"):
        F.box.hat(B, index=(0, 0), count=(3, 1))
    with pytest.raises(pg.Refused, match="must run from 0 to 1"):
        F.box.hat(B, index=(0, 0), nodes=((0.0, 1.0), (0.0, 0.5)))
    with pytest.raises(pg.Refused, match="hat is not defined on a ball"):
        F.entries()["hat"].builder(pg.Ball(1.0), index=0, count=3)
