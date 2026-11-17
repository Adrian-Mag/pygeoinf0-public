"""The smooth partition of unity (plan 021 P1, work package F): the sum of the members is 1 (the claim the family exists
for, checked numerically), non-negativity, locality, the explicit formula, symmetry, the gradient, the support and
boundary claims, membership of H^2, and the refusals."""
import itertools
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.sets import Interval

I01 = Interval("[0, 1]")


def members_1d(D, **kw):
    n = len(kw["centres"]) if "centres" in kw else kw["count"]
    return [F.interval.partition_of_unity(D, index=i, **kw) for i in range(n)]


@pytest.mark.parametrize("D", [I01, Interval(-3.0, 5.0)])
@pytest.mark.parametrize("n, overlap, k", [(2, 1.0, 1.0), (5, 1.0, 1.0), (5, 0.6, 1.0), (9, 2.5, 0.3), (7, 1.5, 10.0)])
def test_uniform_members_sum_to_one(D, n, overlap, k):
    x = np.r_[np.linspace(D.a, D.b, 1001), D.a, D.b]
    funcs = members_1d(D, count=n, overlap=overlap, k=k)
    total = sum(f(x) for f in funcs)
    assert np.allclose(total, 1.0, atol=1e-13)
    assert all(np.all((f(x) >= 0) & (f(x) <= 1)) for f in funcs)
    assert F.interval.partition_of_unity(D, index=0, count=n, overlap=overlap, k=k).params["k"] == k


def test_given_centres_and_radii_sum_to_one():
    c, r = (0.0, 0.2, 0.55, 0.6, 1.0), (0.3, 0.25, 0.3, 0.2, 0.5)
    funcs = members_1d(I01, centres=c, radii=r)
    x = np.linspace(0, 1, 2001)
    assert np.allclose(sum(f(x) for f in funcs), 1.0, atol=1e-13)


@pytest.mark.parametrize("shape", [(3, 2), (4, 3)])
def test_box_members_sum_to_one(shape):
    B = pg.Box((0.0, 2.0), (-1.0, 1.0))
    x, y = np.meshgrid(np.linspace(0, 2, 41), np.linspace(-1, 1, 33), indexing="ij")
    funcs = [F.box.partition_of_unity(B, index=idx, count=shape, overlap=(1.0, 1.4))
             for idx in itertools.product(*[range(n) for n in shape])]
    assert np.allclose(sum(f(x, y) for f in funcs), 1.0, atol=1e-13)


def test_three_dimensional_members_sum_to_one():
    B = pg.Box((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    rng = np.random.default_rng(2)
    P = rng.uniform(0, 1, (3, 300))
    funcs = [F.box.partition_of_unity(B, index=idx, count=3, overlap=1.0) for idx in itertools.product(range(3), repeat=3)]
    assert np.allclose(sum(f(*P) for f in funcs), 1.0, atol=1e-13)


def test_explicit_formula_and_locality():
    c, r, k = (0.0, 0.5, 1.0), (0.6, 0.4, 0.6), 2.0
    f1 = F.interval.partition_of_unity(I01, index=1, centres=c, radii=r, k=k)
    x = np.linspace(0, 1, 101)
    def bump(ci, ri):
        t = (x - ci) / ri
        out = np.zeros_like(x)
        out[np.abs(t) < 1] = np.exp(-k / (1 - t[np.abs(t) < 1] ** 2))
        return out

    psi = [bump(ci, ri) for ci, ri in zip(c, r)]
    assert np.allclose(f1(x), psi[1] / sum(psi), atol=1e-15)
    assert f1(0.5) == pytest.approx(math.exp(-k) / (math.exp(-k) + 2 * math.exp(-k / (1 - (0.5 / 0.6) ** 2))), rel=1e-13)
    assert np.all(f1(x[(x <= 0.1) | (x >= 0.9)]) == 0.0)               # exactly 0 outside [0.1, 0.9]
    assert np.all(f1(x[(x > 0.1) & (x < 0.9)]) > 0.0)


def test_symmetry_of_a_uniform_cover():
    n = 6
    funcs = members_1d(I01, count=n, overlap=1.3)
    x = np.linspace(0, 1, 77)
    for i in range(n):
        assert np.allclose(funcs[i](x), funcs[n - 1 - i](1 - x), atol=1e-14)


def test_gradient_against_finite_differences_and_gradients_sum_to_zero():
    rng = np.random.default_rng(4)
    x = rng.uniform(0.01, 0.99, 100)
    funcs = members_1d(I01, count=5, overlap=1.2, k=0.7)
    h = 1e-6
    G = [f.gradient(x)[:, 0] for f in funcs]
    for f, g in zip(funcs, G):
        assert np.allclose(g, (f(x + h) - f(x - h)) / (2 * h), atol=1e-6)
    assert np.allclose(sum(G), 0.0, atol=1e-11)
    B = pg.Box((0.0, 1.0), (0.0, 1.0))
    P = rng.uniform(0.02, 0.98, (2, 40))
    boxf = [F.box.partition_of_unity(B, index=idx, count=3, overlap=1.1) for idx in itertools.product(range(3), repeat=2)]
    assert np.allclose(sum(f.gradient(*P) for f in boxf), 0.0, atol=1e-11)
    f = boxf[4]
    for ax in range(2):
        E = np.zeros((2, 1))
        E[ax] = h
        assert np.allclose(f.gradient(*P)[:, ax], (f(*(P + E)) - f(*(P - E))) / (2 * h), atol=1e-6)


def test_support_and_boundary_claims():
    mid = F.interval.partition_of_unity(I01, index=2, count=5, overlap=1.0)       # [0.25, 0.75]
    assert mid.support.kind == "interior" and mid.regularity.vanishes_at_boundary == "all orders"
    assert mid.support.contains(np.array([0.25, 0.75, 0.2, 0.8])).tolist() == [True, True, False, False]
    assert mid.regularity.order == math.inf and not mid.regularity.piecewise
    reg = next(c for c in mid.claims if c.key == "regularity")
    assert reg.status == "theorem" and "Hormander" in reg.reason
    first = F.interval.partition_of_unity(I01, index=0, count=5, overlap=1.0)     # centred at 0: reaches outside
    assert first.support.kind == "touches" and first.regularity.vanishes_at_boundary is None and first(0.0) > 0.5
    # a support that ends exactly at the boundary is C-infinity-flat there
    flush = F.interval.partition_of_unity(I01, index=1, centres=(0.0, 0.5, 1.0), radii=(0.9, 0.5, 0.9))
    assert flush.support.kind == "touches" and flush.regularity.vanishes_at_boundary == "all orders"
    assert flush(0.0) == 0.0 and flush(1.0) == 0.0
    B = pg.Box((0.0, 1.0), (0.0, 1.0))
    corner = F.box.partition_of_unity(B, index=(0, 1), count=3, overlap=1.0)
    assert corner.support.kind == "touches" and corner.regularity.vanishes_at_boundary is None
    centre = F.box.partition_of_unity(B, index=(1, 1), count=3, overlap=0.9)
    assert centre.support.kind == "interior" and centre.regularity.vanishes_at_boundary == "all orders"
    assert "x" in centre.support.description


def test_membership_in_smooth_spaces():
    f = F.interval.partition_of_unity(I01, index=2, count=5, overlap=1.0)
    assert f.element(pg.L2(I01)) is not None
    assert f.element(pg.Sobolev(I01, 2, length_scale=0.2)) is not None
    assert f.element(pg.Sobolev(I01, 1, length_scale=0.2)) is not None


def test_no_closed_form_but_a_checked_integral_per_axis():
    f = F.interval.partition_of_unity(I01, index=2, count=5, overlap=1.0)
    assert set(f.closed) == {"integral"}
    value, status, _ = f.integral()
    assert status == "checked"
    assert 0.2 < value < 0.3                                       # an interior member of 5 on [0, 1]: about 1/4
    total = sum(F.interval.partition_of_unity(I01, index=i, count=5, overlap=1.0).integral()[0] for i in range(5))
    assert total == pytest.approx(1.0, rel=1e-11)                  # the members sum to 1: an independent check


@pytest.mark.parametrize("kwargs, match", [
    ({"index": 0}, "give count and overlap, or centres and radii"),
    ({"index": 0, "count": 4}, "count needs overlap"),
    ({"index": 0, "overlap": 1.0}, "give count and overlap, or centres and radii"),
    ({"index": 0, "count": 4, "overlap": 1.0, "centres": (0.0, 1.0), "radii": (1.0, 1.0)}, "not both"),
    ({"index": 0, "centres": (0.0, 1.0)}, "give count and overlap, or centres and radii"),
    ({"index": 0, "centres": (0.0, 1.0), "radii": (1.0, 1.0), "overlap": 1.0}, "give count and overlap"),
    ({"index": 4, "count": 4, "overlap": 1.0}, "outside 0..3"),
    ({"index": -1, "count": 4, "overlap": 1.0}, "outside 0..3"),
    ({"index": 0.5, "count": 4, "overlap": 1.0}, "index must be an integer"),
    ({"index": 0, "count": 1, "overlap": 1.0}, "count >= 2"),
    ({"index": 0, "count": 4, "overlap": 0.0}, "overlap > 0"),
    ({"index": 0, "count": 4, "overlap": -1.0}, "overlap > 0"),
    ({"index": 0, "count": 4, "overlap": 0.5}, "do not cover"),            # needs 0.9 r >= h / 2, i.e. overlap >= 0.556
    ({"index": 0, "count": 4, "overlap": 1.0, "k": 0.0}, "k must be in"),
    ({"index": 0, "count": 4, "overlap": 1.0, "k": 51.0}, "k must be in"),
    ({"index": 0, "count": 4, "overlap": 1.0, "k": float("nan")}, "k must be"),
    ({"index": 0, "centres": (0.0, 1.0), "radii": (0.4, 0.4)}, "do not cover"),       # gap in the middle
    ({"index": 0, "centres": (0.7, 1.0), "radii": (0.5, 0.5)}, "do not cover"),       # the point 0 is uncovered
    ({"index": 0, "centres": (0.0, 1.0), "radii": (0.5,)}, "as many positive radii as centres"),
    ({"index": 0, "centres": (0.0, 1.0), "radii": (0.5, -0.5)}, "as many positive radii as centres"),
    ({"index": 0, "centres": (0.0, 1.0), "radii": (0.5, 0.0)}, "as many positive radii as centres"),
])
def test_interval_refusals(kwargs, match):
    with pytest.raises(pg.Refused, match=match):
        F.interval.partition_of_unity(I01, **kwargs)


def test_just_enough_overlap_is_accepted_and_still_sums_to_one():
    funcs = members_1d(I01, count=4, overlap=0.5556)                   # every point within 0.9 r of a centre
    x = np.linspace(0, 1, 1001)
    assert np.allclose(sum(f(x) for f in funcs), 1.0, atol=1e-13)


def test_box_refusals():
    B = pg.Box((0.0, 1.0), (0.0, 1.0))
    with pytest.raises(pg.Refused, match="one entry per axis"):
        F.box.partition_of_unity(B, index=0, count=3, overlap=1.0)
    with pytest.raises(pg.Refused, match="axis 1 do not cover"):
        F.box.partition_of_unity(B, index=(0, 0), count=3, overlap=(1.0, 0.4))
    with pytest.raises(pg.Refused, match="axis 1 is outside 0..2"):
        F.box.partition_of_unity(B, index=(0, 3), count=3, overlap=1.0)
    with pytest.raises(pg.Refused, match="count >= 2"):
        F.box.partition_of_unity(B, index=(0, 0), count=(3, 1), overlap=1.0)
    with pytest.raises(pg.Refused, match="as many positive radii"):
        F.box.partition_of_unity(B, index=(0, 0), centres=((0.0, 1.0), (0.0, 1.0)), radii=((1.0, 1.0), (1.0,)))
    ok = F.box.partition_of_unity(B, index=(0, 1), centres=((0.0, 1.0), (0.0, 0.5, 1.0)), radii=((0.7, 0.7), (0.4, 0.4, 0.4)))
    assert ok(0.0, 0.5) > 0
