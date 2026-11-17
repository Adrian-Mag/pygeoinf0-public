"""Varying bounds in the box's cell route on L2(0, 1) (plan 057): the certified bracket of sup <c, m> against a hand
solution (one datum, a bathtub), the witness checked inside the box by independent evaluation, the cells cut at the
bound's own break, the refusals, the certified extremum of a Legendre series, and outer_polytope taking the
certified end of a BRACKET."""
import numpy as np
import pytest
from numpy.polynomial import Polynomial as P
from numpy.polynomial import legendre as npleg
from scipy.optimize import brentq

import pygeoinf0 as pg
from pygeoinf0.optimisers.dual import interval_cells
from pygeoinf0.core.errors import Unsupported
from pygeoinf0.realisations.bounds.extrema import min_bracket

A = 0.37                                                       # u's break: not a break of c or of the datum
U1, U2 = P([1.5, 0.0, 0.5]), P([0.9, 1.0])                     # u = U1 on [0, A), U2 on [A, 1] (a jump down at A)
L_ASYM = P([-0.5, -0.2])
D_OBS, W = 0.6, 0.05                                           # |int m - 0.6| <= 0.05: the upper side is active


def u_np(x):
    return np.where(x < A, U1(x), U2(x))


def integral(p, a, b):
    q = p.integ()
    return q(b) - q(a)


def pieces(lower):
    """[(a, b, l, u)] polynomial pieces of the box on [0, 1]."""
    lp = (lambda U: -U) if lower is None else (lambda U: lower)
    return [(0.0, A, lp(U1), U1), (A, 1.0, lp(U2), U2)]


def hand_sup(lower):
    """sup int x m over l <= m <= u, int m <= D_OBS + W: m = l on [0, s), u on [s, 1] (x - lambda changes sign once)."""
    pcs = pieces(lower)

    def part(s, f):                                            # int_0^s f(l) + int_s^1 f(u), f a weight polynomial
        return sum(integral(f * l, a, min(b, s)) * (s > a) + integral(f * u, max(a, s), b) * (s < b)
                   for a, b, l, u in pcs)
    one, x = P([1.0]), P([0.0, 1.0])
    s = brentq(lambda s: part(s, one) - (D_OBS + W), 0.0, 1.0, xtol=1e-15)
    return part(s, x)


def problem(lower):
    M = pg.L2(pg.Interval(0, 1))
    D = pg.Euclidean(1)
    u = M(u_np, breaks=[0, A, 1])
    box = pg.PointwiseBox(M, envelope=u) if lower is None else \
        pg.PointwiseBox(M, lower=M(lambda x: lower(x), breaks=[0, 1]), upper=u)
    G = pg.Functionals(M.family([M(lambda x: 1.0 + 0.0 * x, breaks=[0, 1])]), into=D)
    c = M(lambda x: x + 0.0 * x, breaks=[0, 1])
    return M, D, box, G, c, pg.CoordinateBox(D, [-W], [W]), D([D_OBS])


@pytest.mark.parametrize("lower", [None, L_ASYM], ids=["symmetric", "asymmetric"])
def test_a_varying_box_brackets_the_hand_solution_and_its_witness_lies_in_the_box(lower):
    M, D, box, G, c, N, d = problem(lower)
    truth = hand_sup(lower)
    gaps = []
    for sub in (4, 64):
        K = pg.FeasibleSet(box, G, N, d)
        K.sub = sub
        b = K.bound(c)
        assert b.status in ("EXACT", "BRACKET") and b.lower <= truth + 1e-12 and truth <= b.upper + 1e-12
        gaps.append(b.upper - b.lower)
        # the witness re-verified by independent code: its cells (each piece of [0, 1] cut into sub), its values
        # against l and u evaluated densely, its datum against the noise interval, its target value
        edges = np.unique(np.r_[np.linspace(0, A, sub + 1), np.linspace(A, 1, sub + 1)])
        v = b.witness
        assert len(v) == len(edges) - 1
        for (a, e), vc in zip(zip(edges[:-1], edges[1:]), v):
            x = np.linspace(a, e, 203)[1:-1]                         # the open cell (bounds hold a.e.)
            lo = -u_np(x) if lower is None else lower(x)
            assert np.all(lo <= vc + 1e-14) and np.all(vc <= u_np(x) + 1e-14)
        h = np.diff(edges)
        assert abs(h @ v - D_OBS) <= W * (1 + 1e-9)
        assert 0.5 * (edges[:-1] + edges[1:]) * h @ v == pytest.approx(b.lower, abs=1e-12)
        assert K.is_empty() is pg.Truth.FALSE
    assert gaps[1] < gaps[0] / 4                                 # the witness side converges as the cells shrink


def test_the_cells_are_cut_at_the_bounds_own_break():
    M, D, box, G, c, N, d = problem(None)
    cp = interval_cells(box, G, c, 4)
    assert cp.size == 8                                          # [0, A] and [A, 1], four cells each
    x = np.linspace(A, A + 0.63 / 4, 2001)                       # the first cell right of the jump: inf u = U2(A)
    assert cp.hi[4] <= U2(A) and cp.hi[4] == pytest.approx(u_np(x).min(), abs=1e-13)
    assert "certified" in cp.note


def test_varying_bounds_are_refused_where_no_cell_constant_fits_or_as_functions():
    M, D, box, G, c, N, d = problem(None)
    with pytest.raises(Unsupported, match="element of the space"):
        pg.PointwiseBox(M, upper=lambda x: 1 + x)
    tight = pg.PointwiseBox(M, lower=M(lambda x: x + 0.0 * x, breaks=[0, 1]),
                            upper=M(lambda x: x + 1e-3, breaks=[0, 1]))
    with pytest.raises(Unsupported, match="touch or cross"):
        pg.FeasibleSet(tight, G, N, d).bound(c)


def test_the_certified_minimum_of_a_legendre_series():
    rng = np.random.default_rng(3)
    t = np.linspace(-1, 1, 400001)
    for n in (1, 2, 5, 12, 25):
        for _ in range(10):
            cs = rng.standard_normal(n) * np.exp(-0.3 * np.arange(n))
            lo, up = min_bracket(cs)
            ref = npleg.legval(t, cs).min()                      # >= the true minimum
            assert lo <= ref and lo <= up <= ref + 1e-12
            assert up - lo <= 1e-9 * max(1.0, np.abs(cs).sum())
    lo, up = min_bracket(npleg.poly2leg([0, 0, 0, 0, 1.0]))      # x^4: a flat minimum, 0
    assert lo <= 0.0 and up - lo < 1e-13


def test_outer_polytope_takes_the_certified_end_of_a_bracket():
    M, D, box, G, c, N, d = problem(None)
    K = pg.FeasibleSet(box, G, N, d)
    T = pg.Functionals(M.family([c]))
    img = K.image(T)
    up, dn = img.bound(np.array([1.0])), img.bound(np.array([-1.0]))
    assert up.status == "BRACKET" and up.value < up.upper
    Q, b = img.outer_polytope([[1.0], [-1.0]])
    assert b[0] == up.upper and b[1] == dn.upper and b[0] >= hand_sup(None) - 1e-12
