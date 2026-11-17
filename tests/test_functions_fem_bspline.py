"""The B-spline family (plan 021 P1, work package F): known values, partition of unity of a clamped basis, agreement
with the hat at degree 1, the integral (t_{i+p+1} - t_i)/(p+1) against exact piecewise-polynomial integration, the
derivative, continuity C^(p-m) at knots of multiplicity m, vanishing orders at the domain ends, and the refusals."""
import itertools
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.sets import Interval

I01 = Interval("[0, 1]")


def piece_integral(f, breaks, n):
    """The integral of f by n-point Gauss-Legendre on every span between the breakpoints (exact for degree < 2n)."""
    t, w = np.polynomial.legendre.leggauss(n)
    total = 0.0
    for a, b in itertools.pairwise(sorted(set(breaks))):
        total += float(np.sum(0.5 * (b - a) * w * f(0.5 * (b - a) * (t + 1) + a)))
    return total


def test_uniform_cubic_and_quadratic_values():
    cubic = F.interval.bspline(Interval(0.0, 4.0), degree=3, index=0, knots=(0.0, 1.0, 2.0, 3.0, 4.0))
    assert cubic(2.0) == pytest.approx(2 / 3, abs=1e-15)
    assert cubic(1.0) == pytest.approx(1 / 6, abs=1e-15) and cubic(3.0) == pytest.approx(1 / 6, abs=1e-15)
    assert cubic(0.0) == 0.0 and cubic(4.0) == 0.0 and cubic(0.5) == pytest.approx(0.5 ** 3 / 6)
    quad = F.interval.bspline(Interval(0.0, 3.0), degree=2, index=0, knots=(0.0, 1.0, 2.0, 3.0))
    assert quad(1.5) == pytest.approx(0.75, abs=1e-15)
    assert quad(1.0) == pytest.approx(0.5, abs=1e-15) and quad(2.0) == pytest.approx(0.5, abs=1e-15)
    assert quad.closed["integral"][0] == pytest.approx(1.0)                      # (3 - 0) / 3
    assert cubic.closed["integral"][0] == pytest.approx(1.0)                     # (4 - 0) / 4


@pytest.mark.parametrize("p, n", [(1, 2), (1, 6), (2, 3), (2, 7), (3, 4), (3, 9), (4, 8), (5, 6)])
def test_clamped_basis_is_a_partition_of_unity_on_the_closed_domain(p, n):
    D = Interval(-1.0, 2.0)
    x = np.r_[np.linspace(-1, 2, 301), -1.0, 2.0]
    funcs = [F.interval.bspline(D, degree=p, index=i, count=n) for i in range(n)]
    assert np.allclose(sum(f(x) for f in funcs), 1.0, atol=1e-13)
    assert all(np.all(f(x) >= 0) for f in funcs)
    assert funcs[0](-1.0) == 1.0 and funcs[-1](2.0) == 1.0           # clamped ends: the first and last functions are 1
    if n > p + 1:
        assert funcs[0](2.0) == 0.0 and funcs[-1](-1.0) == 0.0
    assert sum(f.integral()[0] for f in funcs) == pytest.approx(sum(piece_integral(f, np.linspace(-1, 2, max(2, n - p + 1)), p + 1)
                                                                  for f in funcs), rel=1e-13)


def test_unclamped_given_knots_sum_to_one_where_all_functions_overlap():
    D = Interval(0.0, 10.0)
    knots = tuple(float(k) for k in range(11))
    p = 3
    funcs = [F.interval.bspline(D, degree=p, index=i, knots=knots) for i in range(len(knots) - p - 1)]
    x = np.linspace(3.0, 7.0, 101)                                       # the interior spans t_3 .. t_7
    assert np.allclose(sum(f(x) for f in funcs), 1.0, atol=1e-13)


@pytest.mark.parametrize("n", [2, 5, 8])
def test_degree_one_is_the_hat(n):
    D = Interval(0.0, 3.0)
    x = np.linspace(0, 3, 401)
    for i in range(n):
        b = F.interval.bspline(D, degree=1, index=i, count=n)
        h = F.interval.hat(D, index=i, count=n)
        assert np.allclose(b(x), h(x), atol=1e-13)
        assert np.allclose(b.gradient(x[::7])[:, 0], h.gradient(x[::7])[:, 0], atol=1e-10)
        assert b.closed["integral"][0] == pytest.approx(h.closed["integral"][0], rel=1e-13)


@pytest.mark.parametrize("p, knots", [
    (2, (0.0, 0.1, 0.1, 0.4, 0.7, 1.0, 1.0)), (3, (0.0, 0.0, 0.3, 0.3, 0.8, 1.0, 1.0)), (4, (0.2, 0.25, 0.5, 0.5, 0.5, 0.9, 1.0)),
    (1, (0.0, 0.0, 0.6, 1.0, 1.0)), (5, (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0))])
def test_integral_closed_form_on_non_uniform_knots(p, knots):
    t = np.array(knots)
    for i in range(len(t) - p - 1):
        if t[i + p + 1] <= t[i]:
            continue
        f = F.interval.bspline(I01, degree=p, index=i, knots=knots)
        want = piece_integral(f, t[i:i + p + 2], p + 2)
        assert f.closed["integral"][0] == pytest.approx(want, rel=1e-13)
        assert f.closed["integral"][0] == pytest.approx((t[i + p + 1] - t[i]) / (p + 1), rel=1e-15)
        assert f.closed["integral"][1] == "theorem" and "de Boor" in f.closed["integral"][2]


def test_derivative_against_finite_differences():
    knots = (0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.9, 1.0, 1.0)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.02, 0.98, 80)
    h = 1e-6
    for p in (2, 3):
        for i in range(len(knots) - p - 1):
            f = F.interval.bspline(I01, degree=p, index=i, knots=knots)
            fd = (f(x + h) - f(x - h)) / (2 * h)
            far = np.min(np.abs(x[:, None] - np.array(knots)[None, :]), axis=1) > 1e-4
            assert np.allclose(f.gradient(x)[far, 0], fd[far], atol=1e-6)


def test_continuity_at_knots_of_multiplicity_m_is_c_p_minus_m():
    t = (0.0, 0.2, 0.5, 0.5, 0.8, 1.0)           # p = 3, window of index 0 .. : a double knot at 0.5
    f = F.interval.bspline(I01, degree=3, index=0, knots=t + (1.0,))
    assert f.regularity.order == 1 and f.regularity.piecewise                    # C^(3-2); the knot 0.2 is simple
    e = 1e-9
    assert abs(f(0.5 + e) - f(0.5 - e)) < 1e-8
    assert abs(f.gradient(0.5 + e)[0] - f.gradient(0.5 - e)[0]) < 1e-7           # f' continuous
    ddl = (f.gradient(0.5 - 1e-4)[0] - f.gradient(0.5 - 2e-4)[0]) / 1e-4
    ddr = (f.gradient(0.5 + 2e-4)[0] - f.gradient(0.5 + 1e-4)[0]) / 1e-4
    assert abs(ddr - ddl) > 1.0                                                  # f'' jumps
    simple = F.interval.bspline(I01, degree=3, index=0, knots=(0.0, 0.2, 0.5, 0.8, 1.0))
    assert simple.regularity.order == 2
    ddl = (simple.gradient(0.5 - 1e-4)[0] - simple.gradient(0.5 - 2e-4)[0]) / 1e-4
    ddr = (simple.gradient(0.5 + 2e-4)[0] - simple.gradient(0.5 + 1e-4)[0]) / 1e-4
    assert abs(ddr - ddl) < 1e-2                                                 # f'' continuous for simple knots
    assert simple.regularity.sobolev_bound() == (3.5, True)


def test_single_polynomial_piece_is_c_infinity():
    f = F.interval.bspline(I01, degree=3, index=2, count=4)                      # the Bernstein polynomial 3 x^2 (1-x)
    assert f.regularity.order == math.inf and not f.regularity.piecewise
    x = np.linspace(0, 1, 11)
    assert np.allclose(f(x), 3 * x ** 2 * (1 - x), atol=1e-14)
    assert f.element(pg.Sobolev(I01, 2, length_scale=0.2)) is not None


@pytest.mark.parametrize("p", [2, 3, 4])
def test_vanishing_order_at_the_clamped_ends(p):
    n = 2 * p + 4                                    # enough functions that the first p + 1 do not reach the upper end
    e = 1e-3
    for i in range(p + 1):
        left = F.interval.bspline(I01, degree=p, index=i, count=n)
        right = F.interval.bspline(I01, degree=p, index=n - 1 - i, count=n)
        assert left.regularity.vanishes_at_boundary == (i if i >= 1 else None) == right.regularity.vanishes_at_boundary
        # B_i ~ (x - a)^i at the left end, mirrored at the right end
        assert left(e) > 0 and right(1 - e) > 0
        assert left(2 * e) / left(e) == pytest.approx(2.0 ** i, rel=5e-2)
        assert right(1 - 2 * e) / right(1 - e) == pytest.approx(2.0 ** i, rel=5e-2)


def test_a_function_touching_both_ends_vanishes_to_the_lesser_order():
    f = F.interval.bspline(I01, degree=4, index=4, count=8)          # knots 0 (simple), ..., 1 (double): orders 4 and 3
    assert f.support.kind == "touches" and f.regularity.vanishes_at_boundary == 3
    e = 1e-3
    assert f(1 - 2 * e) / f(1 - e) == pytest.approx(2.0 ** 3, rel=5e-2)


def test_interior_function_vanishes_to_all_orders():
    f = F.interval.bspline(I01, degree=3, index=4, count=10)
    assert f.support.kind == "interior" and f.regularity.vanishes_at_boundary == "all orders"
    g = F.interval.bspline(I01, degree=3, index=0, knots=(0.0, 0.2, 0.5, 0.8, 1.0))
    assert g.support.kind == "touches" and g.regularity.vanishes_at_boundary == 3        # simple knots at both ends: j = p
    assert g(0.0) == 0.0 and g(1.0) == 0.0


def test_tensor_product_of_mixed_degrees():
    B = pg.Box((0.0, 2.0), (-1.0, 1.0))
    f = F.box.bspline(B, degree=(2, 3), index=(1, 2), count=(5, 6))
    fx = F.interval.bspline(Interval(0.0, 2.0), degree=2, index=1, count=5)
    fy = F.interval.bspline(Interval(-1.0, 1.0), degree=3, index=2, count=6)
    x, y = np.meshgrid(np.linspace(0, 2, 31), np.linspace(-1, 1, 29), indexing="ij")
    assert np.allclose(f(x, y), fx(x) * fy(y), atol=1e-15)
    assert f.closed["integral"][0] == pytest.approx(fx.closed["integral"][0] * fy.closed["integral"][0], rel=1e-15)
    assert f.regularity.order == 1 and f.regularity.piecewise                     # min(2 - 1, 3 - 1)
    G = f.gradient(x, y)
    assert G.shape == (31, 29, 2)
    assert np.allclose(G[..., 0], fx.gradient(x)[..., 0] * fy(y), atol=1e-13)
    assert np.allclose(G[..., 1], fx(x) * fy.gradient(y)[..., 0], atol=1e-13)
    # a single integer applies to every axis, and the tensor basis sums to one
    funcs = [F.box.bspline(B, degree=2, index=(i, j), count=4) for i in range(4) for j in range(4)]
    assert np.allclose(sum(g(x, y) for g in funcs), 1.0, atol=1e-13)
    assert funcs[0].params["degree"] == (2, 2)


def test_three_dimensional_tensor_integral():
    B = pg.Box((0.0, 1.0), (0.0, 1.0), (0.0, 2.0))
    f = F.box.bspline(B, degree=2, index=(0, 1, 0), knots=((0, 0, 0, 0.5, 1), (0, 0.3, 0.6, 1, 1), (0.0, 1.0, 1.5, 2.0, 2.0)))
    want = 1.0
    for ks, i in zip(((0, 0, 0, 0.5, 1), (0, 0.3, 0.6, 1, 1), (0.0, 1.0, 1.5, 2.0, 2.0)), (0, 1, 0)):
        want *= (ks[i + 3] - ks[i]) / 3
    assert f.closed["integral"][0] == pytest.approx(want)
    assert f.support.contains(0.2, 0.5, 1.5) and not f.support.contains(0.9, 0.5, 1.5)


def test_membership_follows_the_regularity():
    f1 = F.interval.bspline(I01, degree=1, index=1, count=4)
    f2 = F.interval.bspline(I01, degree=2, index=1, count=5)
    H1, H2 = pg.Sobolev(I01, 1, length_scale=0.2), pg.Sobolev(I01, 2, length_scale=0.2)
    assert f1.element(H1) is not None and f2.element(H2) is not None
    with pytest.raises(pg.Refused):
        f1.element(H2)
    assert pg.Sobolev(I01, 2, length_scale=0.2) is not None and f2.regularity.sobolev_bound() == (2.5, True)


@pytest.mark.parametrize("kwargs, match", [
    ({"degree": 0, "index": 0, "count": 3}, "degree must be an integer from 1 to 10"),
    ({"degree": 11, "index": 0, "count": 30}, "degree must be an integer from 1 to 10"),
    ({"degree": 2.0, "index": 0, "count": 4}, "degree must be an integer"),
    ({"degree": True, "index": 0, "count": 4}, "degree must be an integer"),
    ({"degree": 2, "index": 4, "count": 4}, "outside 0..3"),
    ({"degree": 2, "index": -1, "count": 4}, "outside 0..3"),
    ({"degree": 3, "index": 0, "count": 3}, "at least degree \\+ 1 = 4"),
    ({"degree": 2, "index": 0}, "give exactly one of knots, count"),
    ({"degree": 2, "index": 0, "count": 4, "knots": (0, 0, 0, 1, 1, 1)}, "give exactly one of knots, count"),
    ({"degree": 2, "index": 0, "knots": (0.0, 0.5, 1.0)}, "at least 4 knots"),
    ({"degree": 1, "index": 0, "knots": (0.0, 0.7, 0.5, 1.0)}, "nondecreasing"),
    ({"degree": 1, "index": 0, "knots": (-0.1, 0.5, 1.0)}, "must lie in the domain"),
    ({"degree": 1, "index": 0, "knots": (0.0, 0.5, 1.1)}, "must lie in the domain"),
    ({"degree": 1, "index": 0, "knots": (0.0, 0.5, 0.5, 0.5, 1.0)}, "multiplicity 2 strictly inside"),
    ({"degree": 2, "index": 0, "knots": (0.3, 0.3, 0.3, 0.6, 1.0)}, "multiplicity 3 strictly inside"),
    ({"degree": 2, "index": 0, "knots": (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)}, "are all equal"),
    ({"degree": 2, "index": 0, "knots": (0.0, 0.0, 0.0, float("inf"), 1.0)}, "finite"),
    ({"index": 0, "count": 4}, "degree"),
    ({"degree": 2, "count": 4}, "index"),
])
def test_refusals(kwargs, match):
    with pytest.raises((pg.Refused, TypeError), match=match):
        F.interval.bspline(I01, **kwargs)


def test_box_refusals():
    B = pg.Box((0.0, 1.0), (0.0, 1.0))
    with pytest.raises(pg.Refused, match="one entry per axis"):
        F.box.bspline(B, degree=2, index=0, count=4)
    with pytest.raises(pg.Refused, match="one entry per axis"):
        F.box.bspline(B, degree=(2, 2, 2), index=(0, 0), count=4)
    with pytest.raises(pg.Refused, match="axis 1 has 2"):
        F.box.bspline(B, degree=2, index=(0, 0), count=(4, 2))
    with pytest.raises(pg.Refused, match="degree must be an integer from 1 to 10"):
        F.box.bspline(B, degree=(2, 0), index=(0, 0), count=4)
