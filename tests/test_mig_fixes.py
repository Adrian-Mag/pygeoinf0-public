"""Defects found by the migration to pygeoinf0 (2026-11-10), each against an exact reference with a planted control:
support= on an interval (it raised), bank bumps that touch or cross the boundary (refused), and separable functionals
whose kernels vanish identically next to a singular end of the radial measure (realised as zero)."""
import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

import pygeoinf0 as pg
from pygeoinf0.core.errors import Refused
from pygeoinf0.functions.families.smooth import bump

mp.mp.dps = 30


# ------------------------------------------------------------------------------------------------ support= on an interval
@pytest.mark.parametrize("support", [(0.2, 0.6), pg.Interval(0.2, 0.6), ((0.2, 0.6),)])
def test_support_on_an_interval(support):
    L = pg.L2(pg.Interval(0, 1))
    exact = float(mp.quad(mp.sin, [0.2, 0.6]))
    f = L(np.sin, support=support)
    assert abs(L.inner(f, L.constant(1.0)) - exact) < 1e-14
    assert float(f(0.1)) == 0.0 and float(f(0.7)) == 0.0
    whole = L(np.sin, breaks=(0.5,))                       # planted: support ignored gives int_0^1 sin, far from exact
    assert abs(L.inner(whole, L.constant(1.0)) - exact) > 0.1


def test_support_with_flat_breaks_and_on_a_sobolev_space():
    L = pg.L2(pg.Interval(0, 1))
    f = L(np.sin, support=(0.2, 0.6), breaks=(0.4,))
    assert abs(L.inner(f, L.constant(1.0)) - float(mp.quad(mp.sin, [0.2, 0.6]))) < 1e-14
    H = pg.Sobolev(pg.Interval(0, 1), 1, length_scale=0.2)
    hump = lambda x: np.sin(np.pi * (np.asarray(x) - 0.2) / 0.4)      # noqa: E731  0 at the support's ends
    g = H(hump, support=pg.Interval(0.2, 0.6))
    assert abs(float(g(0.3)) - hump(0.3)) < 1e-12 and float(g(0.1)) == 0.0
    from pygeoinf0.core.errors import NotAnElement
    with pytest.raises(NotAnElement):              # sin cut at 0.2 jumps by sin(0.2): not in H^1 (plan 065 A1)
        H.norm(H(np.sin, support=pg.Interval(0.2, 0.6)))


def test_support_box_needs_lo_below_hi():
    with pytest.raises(ValueError):
        pg.L2(pg.Interval(0, 1))(np.sin, support=(0.6, 0.2))


# ------------------------------------------------------------------------------------------------ bumps at the boundary
def _profile_integral(c, r, k, lo, hi, power=1):
    f = lambda x: mp.exp(-k / (1 - ((x - c) / r) ** 2)) ** power if abs((x - c) / r) < 1 else mp.mpf(0)
    a, b = max(lo, c - r), min(hi, c + r)
    return float(mp.quad(f, [a, c, b] if a < c < b else [a, b]))


def test_a_bump_may_touch_the_boundary():
    D = pg.Interval(0, 1)
    b = bump(D, centre=(0.3,), radius=0.3)
    assert b.support.kind == "touches" and b.regularity.vanishes_at_boundary == "all orders"
    assert abs(b.closed["integral"][0] - _profile_integral(0.3, 0.3, 1.0, 0, 1)) < 1e-15


@pytest.mark.parametrize("c, r", [(0.0, 0.2), (1.0, 0.2), (0.1, 0.3), (0.5, 0.7)])
def test_clipped_bump_on_an_interval(c, r):
    D, X = pg.Interval(0, 1), pg.L2(pg.Interval(0, 1))
    b = bump(D, centre=(c,), radius=r, k=1.0, clip=True)
    exact, exact2 = _profile_integral(c, r, 1.0, 0, 1), _profile_integral(c, r, 1.0, 0, 1, 2)
    assert abs(b.closed["integral"][0] - exact) < 1e-14 and abs(b.closed["l2_norm_sq"][0] - exact2) < 1e-14
    assert abs(X.inner(X(b), X.constant(1.0)) - exact) < 1e-13
    assert b.regularity.vanishes_at_boundary is None
    assert "not zero on its boundary" in " ".join(str(cl) for cl in b.claims)
    full = _profile_integral(c, r, 1.0, -9, 9)                 # planted: the uncut ball's integral differs
    assert abs(full - exact) > 1e-3


def test_a_bump_leaving_the_domain_is_refused_with_its_choices():
    with pytest.raises(Refused, match="clip=True"):
        bump(pg.Interval(0, 1), centre=(0.1,), radius=0.3)
    with pytest.raises(Refused, match="clip=True"):
        bump(pg.Interval(0, 1), centre=(0.0,), radius=0.2)


def test_clipped_bump_on_a_box_is_cut_by_the_box():
    B = pg.Box(pg.Interval(0, 1), pg.Interval(0, 1))
    b = bump(B, centre=(0.0, 0.5), radius=0.3, clip=True)
    assert b.support_box() == ((0.0, 0.3), (0.2, 0.8))
    assert "integral" not in b.closed                         # no closed form for a disk cut by a box
    Y = pg.L2(B)
    t, w = np.polynomial.legendre.leggauss(200)
    x, y = 0.15 * (t + 1), 0.5 + 0.3 * t
    ref = float(0.15 * 0.3 * w @ b(x[:, None] + 0 * y[None, :], 0 * x[:, None] + y[None, :]) @ w)
    assert abs(Y.inner(Y(b), Y.constant(1.0)) - ref) < 1e-6 * ref


# ------------------------------------------------------------------------------------------------ zero next to a singular end
NODES, VALS = [0.05, 0.3, 0.7, 1.0], [1.0, 2.0, -1.0, 0.5]


def _k(r):
    r = mp.mpf(r)
    for (x0, v0), (x1, v1) in zip(zip(NODES, VALS), zip(NODES[1:], VALS[1:])):
        if x0 <= r <= x1:
            return v0 + (v1 - v0) * (r - x0) / (x1 - x0)
    return mp.mpf(0)


def _setup():
    X = pg.L2(pg.Ball(1.0), realisation="radial_harmonics")
    Y = pg.L2(pg.Shell(0.5, 1.0), realisation="radial_harmonics")
    k = pg.Tabulated((0.0, 1.0), np.array(NODES), np.array(VALS), rule="linear", extrapolation="zero")
    leb = {"ball": pg.LebesgueMeasure(pg.Interval(0.0, 1.0)), "shell": pg.LebesgueMeasure(pg.Interval(0.5, 1.0))}
    return X, pg.DirectSum({"ball": X, "shell": Y}), k, leb


REF_BALL = float(mp.quad(lambda r: _k(r) ** 2 / r ** 2, [0.05, 0.3, 0.7, 1.0]))    # ||k dr/d(r^2 dr)||^2 in L2(r^2 dr)
REF_SHELL = float(mp.quad(lambda r: _k(r) ** 2 / r ** 2, [0.5, 0.7, 1.0]))


def test_kernel_zero_next_to_the_centre_of_a_ball():
    X, M, k, leb = _setup()
    G = pg.Functionals.separable(X, [k], np.array([[2, 0]]), against=leb["ball"], into=pg.Euclidean(1))
    assert abs(float(np.asarray(G.gram())[0, 0]) - REF_BALL) < 1e-13 * REF_BALL


def test_kernel_zero_next_to_the_centre_on_a_direct_sum():
    X, M, k, leb = _setup()
    G = pg.Functionals.separable(M, {"ball": [k], "shell": [k]}, np.array([[2, 0]]), against=leb, into=pg.Euclidean(1))
    assert abs(float(np.asarray(G.gram())[0, 0]) - REF_BALL - REF_SHELL) < 1e-13 * (REF_BALL + REF_SHELL)


def test_planted_hidden_zero_end_states_no_bound():
    """Without the kernels' zero end the factor is fitted from the singular end and does not converge: the realisation
    then states no bound (inf) instead of a small error on a representer that came out as zero."""
    X, M, k, leb = _setup()
    k.zero_end = lambda inward: None
    with pytest.warns(UserWarning, match="did not converge"):
        G = pg.Functionals.separable(M, {"ball": [k], "shell": [k]}, np.array([[2, 0]]), against=leb,
                                     into=pg.Euclidean(1))
    assert any("largest estimated error inf" in str(c) for c in G.claims)
    assert abs(float(np.asarray(G.gram())[0, 0]) - REF_BALL - REF_SHELL) > 0.5 * REF_BALL
