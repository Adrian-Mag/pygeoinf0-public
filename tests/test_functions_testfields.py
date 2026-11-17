"""Checkerboards, layered radial profiles and blobs of the function bank (plan 021 P1, work package G): the special values
and properties the contract cannot see, with integrals checked by panel-wise integration that does not use the families'
closed forms (the contract's quadrature cannot resolve jumps and kinks)."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.functions import geometry as geo
from pygeoinf0.spaces.membership import check

GL_T, GL_W = np.polynomial.legendre.leggauss(40)


def gl(a, b):
    return 0.5 * (b - a) * (GL_T + 1) + a, 0.5 * (b - a) * GL_W


def sobolev(D, s):
    return pg.Sobolev(D, s, length_scale=0.2)


# ------------------------------------------------------------------------------------------------ checkerboard
def test_checkerboard_values_and_boundaries():
    f = F.on(pg.Box((0.0, 1.0), (0.0, 1.0))).checkerboard(cells=(2, 2), amplitude=3.0)
    assert f(0.25, 0.25) == 3.0 and f(0.75, 0.25) == -3.0 and f(0.25, 0.75) == -3.0 and f(0.75, 0.75) == 3.0
    assert f(0.5, 0.25) == -3.0                      # a cell boundary belongs to the upper cell
    assert f(1.0, 1.0) == 3.0 and f(0.0, 0.0) == 3.0  # the top face belongs to the last cell
    assert f(np.array([0.1, 0.9]), np.array([0.1, 0.1])).tolist() == [3.0, -3.0]


def test_checkerboard_integral_per_cell_parity():
    """The closed form against a cell-by-cell Gauss rule: odd cell counts give L/n per axis, even ones 0."""
    B = pg.Box((0.0, 2.0), (-1.0, 1.0), (0.0, 3.0))
    for cells in [(3, 3, 3), (3, 1, 5), (2, 3, 3), (4, 2, 2), (1, 1, 3)]:
        f = F.on(B).checkerboard(cells=cells, amplitude=-1.5)
        total = 0.0
        edges = [np.linspace(s.a, s.b, n + 1) for s, n in zip(B.sides, cells)]
        t, w = np.polynomial.legendre.leggauss(3)
        for idx in np.ndindex(*cells):
            axes = [(0.5 * (e[i + 1] - e[i]) * (t + 1) + e[i], 0.5 * (e[i + 1] - e[i]) * w) for e, i in zip(edges, idx)]
            G = np.meshgrid(*[a for a, _ in axes], indexing="ij")
            W = axes[0][1][:, None, None] * axes[1][1][None, :, None] * axes[2][1][None, None, :]
            total += float(np.sum(W * f(*G)))
        assert f.closed["integral"][0] == pytest.approx(total, abs=1e-12), cells
    f = F.on(pg.Box((0.0, 1.0), (0.0, 1.0))).checkerboard(cells=(3, 3))
    assert f.integral()[0] == pytest.approx(1 / 9)


def test_checkerboard_other_closed_forms_and_membership():
    B = pg.Box((0.0, 2.0), (0.0, 1.0))
    f = F.on(B).checkerboard(cells=4, amplitude=-2.0)           # an integer: four cells per axis
    assert f.params["cells"] == (4, 4)
    assert f.closed["l2_norm_sq"][0] == pytest.approx(4.0 * 2.0)
    assert f.closed["max"][0] == 2.0 and np.max(f(*geo.quadrature(B, 20)[0].T)) == 2.0
    X, w = geo.quadrature(B, 30)
    assert np.sum(w * f(*X.T) ** 2) == pytest.approx(f.closed["l2_norm_sq"][0])
    assert f.regularity.order == -1
    assert check(f, pg.L2(B))[0] and not check(f, sobolev(B, 1))[0]
    assert check(f, pg.L2(B))[2].status == "theorem"


@pytest.mark.parametrize("params, message", [
    ({"cells": (2,)}, "one entry per axis"), ({"cells": (2, 0)}, "positive integer"),
    ({"cells": (2, 2.5)}, "positive integer"), ({"cells": (True, 2)}, "positive integer"),
    ({"cells": (1, 1)}, "one cell is a constant"), ({"cells": 2.5}, "integer or a tuple"),
    ({"cells": (2, 2), "amplitude": 0.0}, "nonzero"), ({"cells": (2, 2), "amplitude": float("nan")}, "finite"),
    ({"cells": (2, 2), "amplitude": "x"}, "real number"), ({"cells": (2, 2), "amplitude": None}, "real number")])
def test_checkerboard_refusals(params, message):
    with pytest.raises(pg.Refused, match=message):
        F.on(pg.Box((0.0, 1.0), (0.0, 1.0))).checkerboard(**params)
    with pytest.raises(TypeError):
        F.on(pg.Box((0.0, 1.0), (0.0, 1.0))).checkerboard((2, 2))      # keyword-only


# ------------------------------------------------------------------------------------------------ layered profile
def ray_mass(f, D, layers, kind):
    """Integral of f over a ball, shell, disk or interval by radial Gauss rules per layer: only the values of f are used."""
    total = 0.0
    for a, b in layers:
        r, w = gl(a, b)
        if kind == "interval":
            total += float(np.sum(w * (f(r + D.a + 0.5 * (D.b - D.a)) + f(D.a + 0.5 * (D.b - D.a) - r))))
        elif kind == "disk":
            total += float(np.sum(w * 2 * math.pi * r * f(r, 0 * r)))
        else:
            total += float(np.sum(w * 4 * math.pi * r ** 2 * f(r, 0 * r, 0 * r)))
    return total


@pytest.mark.parametrize("D, kind, variable, breaks, coefficients, layers", [
    (pg.Ball(2.0), "ball", "radius", (0.5, 1.2), ((1.0, 2.0), (0.3,), (5.0, 0.0, -1.0)), [(0, .5), (.5, 1.2), (1.2, 2)]),
    (pg.Ball(2.0), "ball", "depth", (0.4,), ((3.0, 1.0), (1.0, 0.0, 0.0, 1.0)), [(0, 1.6), (1.6, 2.0)]),
    (pg.Shell(0.5, 1.0), "shell", "radius", (0.8,), ((1.0, 1.0), (2.0, -1.0)), [(.5, .8), (.8, 1.0)]),
    (pg.Shell(0.5, 1.0), "shell", "depth", (0.1,), ((2.0,), (1.0, 4.0)), [(0.9, 1.0), (0.5, 0.9)]),
    (pg.Disk(1.5), "disk", "radius", (1.0,), ((1.0, 0.0, 1.0), (7.0,)), [(0, 1.0), (1.0, 1.5)]),
    (pg.Disk(1.5), "disk", "depth", (0.25, 1.0), ((1.0,), (-2.0, 3.0), (0.5, 0.0, 0.5)), [(1.25, 1.5), (0.5, 1.25), (0, .5)]),
    (pg.Interval("[0, 2]"), "interval", "radius", (0.3,), ((1.0, 1.0), (4.0,)), [(0, .3), (.3, 1.0)]),
])
def test_layered_closed_forms_against_panelwise_integration(D, kind, variable, breaks, coefficients, layers):
    f = F.on(D).layered_profile(variable=variable, breaks=breaks, coefficients=coefficients)
    if kind == "interval":
        total = ray_mass(lambda x: f(x), D, layers, kind)
        sq = ray_mass(lambda x: f(x) ** 2, D, layers, kind)
    else:
        total = ray_mass(lambda *X: f(*X), D, layers, kind)
        sq = ray_mass(lambda *X: f(*X) ** 2, D, layers, kind)
    assert f.closed["integral"][0] == pytest.approx(total, rel=1e-12, abs=1e-12)
    assert f.closed["l2_norm_sq"][0] == pytest.approx(sq, rel=1e-12, abs=1e-12)


def test_layered_interval_depth_and_values():
    f = F.on(pg.Interval("[1, 3]")).layered_profile(variable="depth", breaks=(0.5,), coefficients=((1.0, 2.0), (-1.0,)))
    assert f(3.0) == 1.0 and f(2.75) == pytest.approx(1.5)        # depth b - x
    assert f(2.5) == -1.0 and f(2.0) == -1.0                      # the break belongs to the upper layer
    assert f.closed["integral"][0] == pytest.approx(0.75 - 1.5)


def test_layered_origin_has_no_singularity_and_exact_value():
    B = pg.Ball(1.0)
    f = F.on(B).layered_profile(variable="radius", breaks=(), coefficients=((2.0, 3.0, 0.0, 0.5),))
    assert f(0.0, 0.0, 0.0) == 2.0 and np.isfinite(f(*np.zeros((3, 4)))).all()
    assert f(0.0, 0.0, 0.5) == pytest.approx(2.0 + 1.5 + 0.5 * 0.125)
    d = F.on(pg.Disk(1.0)).layered_profile(variable="depth", breaks=(), coefficients=((0.0, 1.0),))
    assert d(0.0, 0.0) == 1.0 and d(0.6, 0.8) == pytest.approx(0.0)      # depth at the centre is the radius


def test_layered_breakpoint_convention_is_right_continuous():
    f = F.on(pg.Ball(1.0)).layered_profile(variable="radius", breaks=(0.5,), coefficients=((1.0,), (2.0,)))
    assert f(0.5, 0, 0) == 2.0 and f(0.4999999, 0, 0) == 1.0 and f(0, 0, -0.5) == 2.0


@pytest.mark.parametrize("D, variable, breaks, coefficients, order", [
    (pg.Ball(1.0), "radius", (0.5,), ((1.0,), (2.0,)), -1),                                  # jump
    (pg.Ball(1.0), "radius", (0.5,), ((1.0, 0.0, 0.0), (0.5, 1.0, 0.0)), 0),                  # kink (value agrees)
    (pg.Ball(1.0), "radius", (0.5,), ((0.0, 0.0, 1.0), (-0.25, 1.0, 0.0)), 1),                # derivative 2r = 1 agrees
    (pg.Ball(1.0), "radius", (0.5,), ((1.0,), (1.0, 0.0)), math.inf),                          # the same polynomial
    (pg.Ball(1.0), "radius", (), ((1.0, 1.0),), 0),                                           # a cone at the centre
    (pg.Ball(1.0), "radius", (), ((1.0, 0.0, 2.0, 1.0),), 2),                                  # r^3
    (pg.Ball(1.0), "radius", (), ((1.0, 0.0, 2.0, 0.0, 5.0),), math.inf),                      # even: smooth at the centre
    (pg.Ball(1.0), "depth", (), ((0.0, 1.0),), 0),                                             # R - r: a cone
    (pg.Ball(1.0), "depth", (), ((0.0, 0.0, 1.0),), 0),                                        # (1 - r)^2 = 1 - 2r + r^2
    (pg.Shell(0.5, 1.0), "radius", (), ((1.0, 1.0),), math.inf),                               # no centre in a shell
    (pg.Interval("[0, 1]"), "depth", (), ((1.0, 5.0),), math.inf),                             # b - x is linear in x
    (pg.Interval("[0, 1]"), "radius", (), ((1.0, 5.0),), 0),                                   # |x - 1/2|
])
def test_layered_regularity_follows_the_coefficients(D, variable, breaks, coefficients, order):
    f = F.on(D).layered_profile(variable=variable, breaks=breaks, coefficients=coefficients)
    assert f.regularity.order == order
    if order == -1:
        assert not check(f, sobolev(D, 1))[0]
    elif order == math.inf:
        assert check(f, sobolev(D, 2))[0]
    else:
        assert check(f, sobolev(D, 1))[0] and check(f, pg.L2(D))[0]
    assert (f.gradient is None) == (order < 0)


@pytest.mark.parametrize("coefficients", [
    ((1e-16,), (2e-15,)), ((6.7e-11,), (6.700067e-11,)), ((0.0, 0.0), (1e-15,)), ((1e-30,), (3e-30,)),
    ((1e-20, 1e-20), (1e-20, 2e-20))])
def test_layered_jumps_of_tiny_profiles_are_not_taken_for_continuity(coefficients):
    """The tolerance is relative to the coefficients: there is no absolute floor of 16 eps."""
    f = F.on(pg.Ball(1.0)).layered_profile(variable="radius", breaks=(0.5,), coefficients=coefficients)
    assert f.regularity.order == -1 and f.regularity.piecewise
    assert not check(f, sobolev(pg.Ball(1.0), 1))[0]
    assert "jump" in f.reasons["regularity"] and "C-infinity" not in f.reasons["regularity"]


@pytest.mark.parametrize("scale", [1e-30, 1e-14, 1.0, 1e12])
def test_layered_continuity_is_decided_relative_to_the_scale(scale):
    """The same shapes at every magnitude: a glueing exact to rounding keeps its order, a relative 1e-5 jump does not."""
    ball = pg.Ball(1.0)
    same = F.on(ball).layered_profile(variable="radius", breaks=(0.5,), coefficients=((scale,), (scale,)))
    assert same.regularity.order == math.inf
    kink = F.on(ball).layered_profile(variable="radius", breaks=(0.5,), coefficients=((scale, 0.0), (0.5 * scale, scale)))
    assert kink.regularity.order == 0
    jump = F.on(ball).layered_profile(variable="radius", breaks=(0.5,), coefficients=((scale,), (scale * (1 + 1e-5),)))
    assert jump.regularity.order == -1


def test_layered_gradient_matches_finite_differences():
    B = pg.Ball(1.0)
    # glued to order 1: the finite differences are tested away from the interface
    f = F.on(B).layered_profile(variable="radius", breaks=(0.5,), coefficients=((0.0, 0.0, 1.0), (-0.25, 1.0, 0.0)))
    rng = np.random.default_rng(3)
    P = rng.uniform(-0.6, 0.6, size=(20, 3))
    P = P[np.abs(np.linalg.norm(P, axis=1) - 0.5) > 1e-3]
    h = 1e-6
    for i in range(3):
        E = np.zeros(3)
        E[i] = h
        fd = (f(*(P + E).T) - f(*(P - E).T)) / (2 * h)
        assert np.allclose(fd, f.gradient(*P.T)[:, i], atol=1e-7)


@pytest.mark.parametrize("params, message", [
    ({"variable": "height", "breaks": (), "coefficients": ((1.0,),)}, "radius' or 'depth"),
    ({"variable": "radius", "breaks": (0.5,), "coefficients": ((1.0,),)}, "need 2"),
    ({"variable": "radius", "breaks": (), "coefficients": ((),)}, "non-empty"),
    ({"variable": "radius", "breaks": (1.5,), "coefficients": ((1.0,), (2.0,))}, "inside"),
    ({"variable": "radius", "breaks": (0.0,), "coefficients": ((1.0,), (2.0,))}, "inside"),
    ({"variable": "radius", "breaks": (0.6, 0.4), "coefficients": ((1.0,), (2.0,), (3.0,))}, "increase strictly"),
    ({"variable": "radius", "breaks": (0.4, 0.4), "coefficients": ((1.0,), (2.0,), (3.0,))}, "increase strictly"),
    ({"variable": "radius", "breaks": (), "coefficients": ((float("nan"),),)}, "finite"),
    ({"variable": "radius", "breaks": (), "coefficients": (1.0,)}, "sequence")])
def test_layered_refusals(params, message):
    with pytest.raises(pg.Refused, match=message):
        F.on(pg.Ball(1.0)).layered_profile(**params)


def test_layered_shell_breaks_are_relative_to_its_range():
    S = pg.Shell(0.5, 1.0)
    with pytest.raises(pg.Refused, match="inside"):
        F.on(S).layered_profile(variable="radius", breaks=(0.4,), coefficients=((1.0,), (2.0,)))
    F.on(S).layered_profile(variable="radius", breaks=(0.75,), coefficients=((1.0,), (2.0,)))
    with pytest.raises(pg.Refused, match="inside"):
        F.on(S).layered_profile(variable="depth", breaks=(0.75,), coefficients=((1.0,), (2.0,)))      # depth range (0, 0.5)


# ------------------------------------------------------------------------------------------------ blobs
def test_blobs_single_bump_value_and_closed_forms():
    B = pg.Ball(1.0)
    f = F.on(B).blobs(centres=((0.1, 0.0, 0.0),), sizes=(0.4,), amplitudes=(-2.0,))
    m = F.on(B).bump(centre=(0.1, 0.0, 0.0), radius=0.4)
    assert f(0.1, 0.0, 0.0) == pytest.approx(-2.0 * math.exp(-1.0))
    X = np.random.default_rng(1).uniform(-0.6, 0.6, size=(50, 3))
    assert np.allclose(f(*X.T), -2.0 * m(*X.T), atol=1e-15)
    assert f.closed["integral"][0] == pytest.approx(-2.0 * m.closed["integral"][0])
    assert f.closed["l2_norm_sq"][0] == pytest.approx(4.0 * m.closed["l2_norm_sq"][0])
    assert f.regularity.order == math.inf and f.regularity.vanishes_at_boundary == "all orders"
    assert f.support.kind == "interior" and f.support.radius == pytest.approx(0.4)


def test_blobs_sum_of_separated_bumps_integral_independent_of_closed_forms():
    B = pg.Box((0.0, 1.0), (0.0, 1.0))
    C, R, A = ((0.25, 0.25), (0.7, 0.6), (0.3, 0.75)), (0.15, 0.2, 0.1), (1.0, -0.5, 2.0)
    f = F.on(B).blobs(centres=C, sizes=R, amplitudes=A)
    t, w = np.polynomial.legendre.leggauss(80)
    total = 0.0
    for c, r in zip(C, R):                                       # the supports are disjoint: one fine rule per ball
        for ix in range(2):
            for iy in range(2):
                xs, wx = 0.5 * r * (t + 1) + c[0] - r + ix * r, 0.5 * r * w
                ys, wy = 0.5 * r * (t + 1) + c[1] - r + iy * r, 0.5 * r * w
                G = np.meshgrid(xs, ys, indexing="ij")
                total += float(np.sum(np.outer(wx, wy) * f(*G)))
    assert f.closed["integral"][0] == pytest.approx(total, rel=1e-8)
    assert "l2_norm_sq" not in f.closed                           # the cross terms are not derived
    X = np.array([[0.25, 0.25], [0.9, 0.1], [0.7, 0.6]])
    assert np.allclose(f(*X.T), [A[0] * math.exp(-1), 0.0, A[1] * math.exp(-1)])
    assert not f.support.contains(0.9, 0.1) and f.support.contains(0.25, 0.3)
    assert f.support.radius is None or f.support.radius > 0


def test_blobs_gradient_is_the_sum_of_the_members_gradients():
    B = pg.Box((0.0, 1.0), (0.0, 1.0))
    f = F.on(B).blobs(shape="gaussian", centres=((0.3, 0.4), (0.6, 0.5)), sizes=(0.1, 0.2), amplitudes=(1.0, -3.0))
    P, h = np.array([[0.31, 0.42], [0.5, 0.5], [0.9, 0.2]]), 1e-6
    G = f.gradient(*P.T)
    for i in range(2):
        E = np.zeros(2)
        E[i] = h
        assert np.allclose((f(*(P + E).T) - f(*(P - E).T)) / (2 * h), G[:, i], atol=1e-6)
    assert f.regularity.vanishes_at_boundary is None and f.support.kind == "full"


def test_blobs_seeded_is_reproducible_and_explicit_in_the_parameters():
    B = pg.Ball(1.0)
    f, g = F.on(B).blobs(seed=11, count=4), F.on(B).blobs(seed=11, count=4)
    h = F.on(B).blobs(seed=12, count=4)
    X = np.random.default_rng(0).uniform(-0.7, 0.7, size=(30, 3))
    assert np.array_equal(f(*X.T), g(*X.T)) and not np.allclose(f(*X.T), h(*X.T))
    assert len(f.params["centres"]) == len(f.params["sizes"]) == len(f.params["amplitudes"]) == 4
    # the resolved parameters rebuild the same function in the explicit form
    e = F.on(B).blobs(centres=f.params["centres"], sizes=f.params["sizes"], amplitudes=f.params["amplitudes"])
    assert np.array_equal(f(*X.T), e(*X.T)) and e.closed["integral"][0] == pytest.approx(f.closed["integral"][0])
    assert all(-1.0 <= a <= 1.0 for a in f.params["amplitudes"])
    lo = F.on(B).blobs(seed=3, count=5, size_range=(0.1, 0.1), amplitude_range=(2.0, 2.0))
    scale = geo.volume(B) ** (1 / 3)
    assert np.allclose(lo.params["sizes"], 0.1 * scale) and np.allclose(lo.params["amplitudes"], 2.0)


@pytest.mark.parametrize("kind", ["interval", "box", "disk", "ball", "shell", "sphere", "circle", "cylinder", "torus"])
def test_blobs_seeded_bumps_are_inside_every_domain(kind):
    for D in geo.standard(kind):
        for seed in range(6):
            f = F.entries()["blobs"].builder(D, seed=seed, count=3)
            for c, r in zip(f.params["centres"], f.params["sizes"]):
                assert r < geo.room(D, np.array(c)) and r < geo.injectivity(D)
            if kind in ("interval", "box", "disk", "ball", "shell", "cylinder"):
                P, _ = geo.quadrature(D, 8)
                edge = np.array([min(geo.room(D, p), 1.0) for p in P]) < 1e-9
                assert not edge.any() or np.all(f(*P[edge].T) == 0.0)


def test_blobs_gaussian_members_and_the_torus():
    D = pg.Circle(1.0)
    f = F.on(D).blobs(shape="gaussian", seed=4, count=2)
    assert f.support.kind == "full" and "integral" in f.closed
    with pytest.raises(pg.Refused, match="blob 0: gaussian is not defined on a torus"):
        F.entries()["blobs"].builder(pg.Torus((0.0, 1.0), (0.0, 1.0)), shape="gaussian", seed=1, count=1)


@pytest.mark.parametrize("params, message", [
    ({"seed": 1}, "positive integer"), ({"seed": 1, "count": 0}, "positive integer"),
    ({"seed": -1, "count": 1}, "non-negative integer"), ({"seed": 1.5, "count": 1}, "non-negative integer"),
    ({"seed": True, "count": 1}, "non-negative integer"),
    ({"seed": 1, "count": 1, "sizes": (0.1,)}, "not both"),
    ({"count": 2}, "seeded form"), ({"size_range": (0.1, 0.2)}, "seeded form"),
    ({}, "either a seed"), ({"centres": ((0, 0, 0),), "sizes": (0.1,)}, "either a seed"),
    ({"shape": "cone", "seed": 1, "count": 1}, "'bump' or 'gaussian'"),
    ({"centres": ((0, 0, 0),), "sizes": (0.1, 0.2), "amplitudes": (1.0,)}, "one length"),
    ({"centres": ((0, 0),), "sizes": (0.1,), "amplitudes": (1.0,)}, "3 finite coordinates"),
    ({"centres": ((0, 0, 0),), "sizes": (0.1,), "amplitudes": (float("inf"),)}, "finite"),
    ({"centres": ((0, 0, 0),), "sizes": (float("inf"),), "amplitudes": (1.0,)}, "size and amplitude must be finite"),
    ({"centres": ((0, 0, 0),), "sizes": (1.5,), "amplitudes": (1.0,)}, "blob 0: bump: radius"),
    ({"centres": ((0, 0, 0), (0.0, 0.0, 0.9)), "sizes": (0.2, 0.3), "amplitudes": (1.0, 1.0)}, "blob 1"),
    ({"centres": ((0, 0, 0),), "sizes": (-0.1,), "amplitudes": (1.0,)}, "must be positive"),
    ({"seed": 1, "count": 1, "size_range": (0.3, 0.1)}, "increasing pair"),
    ({"seed": 1, "count": 1, "size_range": (0.0, 0.1)}, "above 0"),
    ({"seed": 1, "count": 1, "size_range": (5.0, 6.0)}, "does not fit")])
def test_blobs_refusals(params, message):
    with pytest.raises(pg.Refused, match=message):
        F.on(pg.Ball(1.0)).blobs(**params)


def test_blobs_gaussian_with_an_infinite_or_nan_width_is_refused():
    B = pg.Box((0.0, 1.0), (0.0, 1.0))
    for size in (float("inf"), float("nan")):
        with pytest.raises(pg.Refused, match="finite"):
            F.on(B).blobs(shape="gaussian", centres=((0.5, 0.5),), sizes=(size,), amplitudes=(1.0,))


def test_catalogue_lists_the_package():
    cat = {n: kinds for n, kinds, _ in F.catalogue()}
    assert cat["checkerboard"] == ("box",)
    assert cat["layered_profile"] == ("ball", "shell", "disk", "interval")
    assert set(cat["blobs"]) == {"interval", "box", "disk", "ball", "shell", "sphere", "circle", "cylinder", "torus"}
    assert set(cat["step_across_plane"]) == set(cat["dyke"]) == {"box", "disk", "ball", "cylinder"}
