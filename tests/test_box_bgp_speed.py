"""Box BGP's fast paths (plan 059) against the routes they replace: the certificate side and the statuses of the
interval-cell route unchanged against a reference cell problem built the old way (the indicator family's Grams and
the Bernstein enclosure per cell, given as cells=); cell integrals against the family's Grams; essential breaks; the
vectorised certified minimum against the Bernstein one and against dense sampling; the memo keyed on K.sub; the
cell problem's memory linear in the cells."""
import math
import tracemalloc

import numpy as np
import pytest
from numpy.polynomial import legendre as npleg

import pygeoinf0 as pg
from pygeoinf0.optimisers.dual import CellProblem, interval_cells
from pygeoinf0.spaces.elements import Family
from pygeoinf0.realisations.piecewise_constant import cell_integrals, essential_breaks
from pygeoinf0.realisations.bounds.extrema import min_bracket, min_brackets
from pygeoinf0.realisations.series.legendre import PiecewiseLegendre, refine, union_breaks


def problem(n=8):
    M = pg.L2(pg.Interval(0, 1))
    D = pg.Euclidean(n, covariance=0.05 ** 2)
    G = pg.Functionals(M.family([M(lambda x, k=k: np.cos((k + 1) * math.pi * x), resolution=0.05)
                                 for k in range(n)]), into=D)
    d = D(G(M(lambda x: 0.8 * np.sin(4 * x), resolution=0.05)).values + np.random.default_rng(1).normal(0, 0.05, n))
    N = pg.NormBall(D, math.sqrt(n + 2 * math.sqrt(2 * n)))
    box = pg.PointwiseBox(M, envelope=M(lambda x: 1.2 + 0.2 * np.cos(2 * math.pi * x), resolution=0.05))
    return M, D, G, d, N, box


def reference_cells(box, G, c, sub):
    """The cell problem as plan 057 built it: the indicator family's Grams, the Bernstein enclosure per cell."""
    M, D = box.space, G.codomain
    rnd = G.adjoint(D(np.random.default_rng(0).standard_normal(D.dim)))
    u = box.upper[0]
    nb = union_breaks(essential_breaks(rnd.realisation), essential_breaks(c.realisation), u.realisation.breaks)
    edges = np.unique(np.concatenate([np.linspace(a, b, sub + 1) for a, b in zip(nb[:-1], nb[1:])]))
    m = len(edges) - 1
    E = np.eye(m)
    F = Family(M, PiecewiseLegendre(edges, [E[i:i + 1] for i in range(m)], np.zeros(m)), name="1")
    hi = np.array([min_bracket(C[:, 0])[0] for C in refine(u.realisation, edges)])
    return CellProblem(-hi, hi, np.atleast_1d(M.inner(c, F)), H=np.atleast_2d(G(F).values))


@pytest.mark.parametrize("sub", [4, 16])
def test_the_certificate_and_status_are_those_of_the_reference_cells(sub):
    M, D, G, d, N, box = problem()
    for c in (M(lambda x: ((x > 0.2) & (x < 0.45)) / 0.25 + 0 * x, breaks=[0, 0.2, 0.45, 1]),
              M(lambda x: -x + 0 * x, breaks=[0, 1])):
        K = pg.FeasibleSet(box, G, N, d)
        K.sub = sub
        fast = K.bound(c)
        R = pg.FeasibleSet(box, G, N, d)
        R.cells = reference_cells(box, G, c, sub)
        ref = R.bound(c)
        assert fast.status == ref.status and fast.cells.size == ref.cells.size
        assert fast.upper == pytest.approx(ref.upper, rel=1e-9, abs=1e-12)          # the certificate side
        assert fast.lower == pytest.approx(ref.lower, rel=1e-7, abs=1e-9)           # the witness, cell bounds 1e-9
        assert K.bound(c) is fast                                                    # remembered per direction
        K.sub = sub + 1
        assert K.bound(c) is not fast                                                # ... and per K.sub


def test_cell_integrals_equal_the_indicator_familys_grams():
    M, D, G, d, N, box = problem()
    edges = np.linspace(0, 1, 37)
    m = len(edges) - 1
    E = np.eye(m)
    F = Family(M, PiecewiseLegendre(edges, [E[i:i + 1] for i in range(m)], np.zeros(m)), name="1")
    from pygeoinf0.optimisers.dual.nonsmooth import _representers
    H = cell_integrals(_representers(G, {}, "r"), edges)
    assert np.allclose(H, np.atleast_2d(G(F).values), rtol=1e-12, atol=1e-14)
    q = M(lambda x: np.exp(x) * np.sin(9 * x), resolution=0.05)
    assert np.allclose(cell_integrals(q.realisation, edges)[:, 0], np.atleast_1d(M.inner(q, F)), atol=1e-14)


def test_essential_breaks_keep_only_where_the_pieces_differ():
    M = pg.L2(pg.Interval(0, 1))
    fam = M.family([M(lambda x, a=a: ((x > a) & (x < a + 0.1)) + 0 * x, breaks=[0, a, a + 0.1, 1])
                    for a in (0.1, 0.4, 0.7)])
    assert len(fam.realisation.breaks) == 8
    one = fam[1].realisation
    assert np.allclose(essential_breaks(one), [0, 0.4, 0.5, 1])
    smooth = M(lambda x: x ** 2, breaks=[0, 0.3, 0.6, 1]).realisation
    assert np.allclose(essential_breaks(smooth), [0, 1])


def test_the_vectorised_certified_minimum():
    rng = np.random.default_rng(5)
    t = np.linspace(-1, 1, 200001)
    C = rng.standard_normal((60, 9)) * np.exp(-0.5 * np.arange(9))
    lo, up = min_brackets(C)
    ref = np.array([npleg.legval(t, c).min() for c in C])          # >= the true minimum
    bern = np.array([min_bracket(c)[0] for c in C])
    assert np.all(lo <= ref) and np.all(lo <= up) and np.all(up <= ref + 1e-12)
    assert np.all(np.abs(lo - bern) <= 1e-9 * np.maximum(1.0, np.abs(C).sum(axis=1)))


def test_the_cell_problem_is_linear_in_the_cells():
    M, D, G, d, N, box = problem()
    c = M(lambda x: x + 0 * x, breaks=[0, 1])
    tracemalloc.start()
    cp = interval_cells(box, G, c, 400)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert cp.size >= 400 and peak < 50e6                           # a dense indicator family would be cells^2
