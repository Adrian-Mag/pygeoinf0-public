"""The cell realisation of L2 (plan 020 P2; EXP-035 NOTES.md): partitions and point location, exact fits of
cell-constant functions and the stated projection error of others, inner products, pg.port.Coordinates.cells (diagonal maps,
adjoints), refusals, and the scale it is for (1e5 cells)."""
import time

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.realisations import Cells

NODES = [np.array([0.0, 0.1, 0.3, 0.6, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([-1.0, -0.4, 0.0])]
BOX = pg.Box([(0, 1), (0, 1), (-1, 0)])


def space(cells=None):
    return pg.L2(pg.LebesgueMeasure(BOX), realisation=cells or Cells.tensor(*NODES))


def test_partition_order_and_location():
    C = Cells.tensor(*NODES)
    assert C.n == 16 and np.isclose(C.volumes.sum(), 1.0)
    centres = 0.5 * (C.lo + C.hi)
    assert np.array_equal(C.locate(centres), np.arange(16))                   # x fastest, then y, then z
    assert np.allclose(centres[1] - centres[0], [0.15, 0, 0])
    general = Cells([pg.Box(list(zip(lo, hi))) for lo, hi in zip(C.lo, C.hi)])
    assert np.array_equal(general.locate(centres), np.arange(16))
    assert C.locate(np.array([[2.0, 0.5, -0.5]]))[0] == -1


def test_fits_inner_products_and_projection_error():
    L = space()
    C = L.backend.cells
    step = L(lambda x, y, z: np.where(x < 0.3, 2.0, -1.0))                     # constant on every cell
    assert step.realisation.err[0] < 1e-14                                      # quadrature rounding only
    assert np.isclose(L.norm(step) ** 2, 0.3 * 4 + 0.7 * 1)
    f = L(lambda x, y, z: x)
    with pytest.warns(pg.RealisationWarning, match="realised by its averages on the cells"):
        L.norm(f)
    var = (C.hi[:, 0] - C.lo[:, 0]) ** 2 / 12                                   # the exact projection error of x
    assert np.isclose(f.realisation.err[0], np.sqrt(C.volumes @ var), rtol=1e-12)
    g = L(lambda x, y, z: 1.0 + 0 * x)
    assert np.isclose(L.inner(step, g), 0.3 * 2 - 0.7 * 1)
    M = pg.L2(pg.LebesgueMeasure(BOX))
    assert L == M and L.realisation != M.realisation          # one space, two realisations (plan 069 P1, decision Q4)
    with pytest.raises(pg.RealisationMismatch):
        L.inner(g, M.constant(1.0))


def test_cell_coordinates_are_diagonal_and_exact():
    L = space()
    K = pg.port.Coordinates.cells(L)
    c = np.arange(1.0, 17.0)
    assert np.allclose(K.numbers(K.element(c)), c)
    R = K.export.codomain
    f = L(lambda x, y, z: np.where(y < 0.5, 1.0, 3.0) + np.where(z < -0.4, 0.5, 0.0))
    for A, x, y in ((K.export, f, R(c)), (K.imp, R(c), f)):
        lhs, rhs = A.codomain.inner(A(x), y), A.domain.inner(x, A.adjoint(y))
        assert abs(lhs - rhs) < 1e-13 * abs(lhs)
    assert np.allclose(K.export.adjoint(R(c)).realisation.V[:, 0], c / L.backend.cells.volumes)
    assert K.claims.status("E P = I") == (pg.Truth.TRUE, "derived") and K.claims.status("bounded") == (pg.Truth.TRUE, "derived")


@pytest.mark.parametrize("N", [3, 16])                     # 16 = the number of cells: a transpose would go unnoticed
def test_cell_coordinates_map_families_as_their_members(N):
    """E, E*, P and P* on a family equal the members one by one (a Gram assembly or a posterior passes families;
    plan 020 P4 found P* reading only the first member, and E* and P building the cell values transposed)."""
    L = space()
    K = pg.port.Coordinates.cells(L)
    R = K.export.codomain
    V = np.random.default_rng(1).normal(size=(N, K.dim))
    fam = pg.spaces.elements.VectorFamily(R, V)
    funcs = K.imp(fam)
    for A, x in ((K.imp, fam), (K.export.adjoint, fam), (K.export, funcs), (K.imp.adjoint, funcs)):
        out = A(x)
        assert len(out) == N
        for i in range(N):
            one = A(x[i])
            got, want = (out[i].values, one.values) if hasattr(one, "values") else \
                (out.realisation.V[:, i], one.realisation.V[:, 0])
            assert np.allclose(got, want, rtol=1e-14, atol=0)


def test_taylor_test_checks_the_adjoint_into_a_cell_space():
    """A cell-realised L2 has no default Gaussian measure; the adjoint identity draws its random element through the
    map's codomain coordinates (plan 020 P4: pointwise petrophysical laws on cells had no adjoint check)."""
    L = space()
    K = pg.port.Coordinates.cells(L)
    G = pg.port.External(L, L, lambda x: x ** 3, coords=(K, K), name="cube", jvp=lambda x, v: 3 * x ** 2 * v,
                    vjp=lambda x, w: 3 * x ** 2 * w)
    res = pg.port.taylor_test(G, K.element(np.linspace(0.5, 2.0, K.dim)), directions=2)
    assert res.adjoint_gap < 1e-12 and res.claims.status("adjoint")[0] is pg.Truth.TRUE, res
    bad = pg.port.External(L, L, lambda x: x ** 3, coords=(K, K), name="cube", jvp=lambda x, v: 3 * x ** 2 * v,
                      vjp=lambda x, w: 3 * x ** 2 * w * K.space.backend.cells.volumes)    # a dot-product adjoint
    assert pg.port.taylor_test(bad, K.element(np.linspace(0.5, 2.0, K.dim)), directions=2).claims.status("adjoint")[0] \
        is pg.Truth.FALSE


def test_refusals():
    with pytest.raises(ValueError, match="not a partition"):
        pg.L2(pg.LebesgueMeasure(BOX), realisation=Cells([pg.Box([(0, 0.5), (0, 1), (-1, 0)])]))
    with pytest.raises(pg.errors.Unsupported, match="realise L2 only"):
        pg.Sobolev(BOX, 1, length_scale=0.2, realisation=Cells.tensor(*NODES))
    with pytest.raises(NotImplementedError, match="density"):                   # refused before the cells are read
        pg.L2(pg.LebesgueMeasure(BOX).with_density(lambda x, y, z: 1 + x), realisation=Cells.tensor(*NODES))
    with pytest.raises(pg.SpaceMismatch, match="cell coordinates need"):
        pg.port.Coordinates.cells(pg.L2(pg.LebesgueMeasure(BOX)))


def test_scale_1e5_cells():
    nodes = [np.linspace(0, 1, 65), np.linspace(0, 1, 65), np.linspace(-1, 0, 26)]
    L = pg.L2(pg.LebesgueMeasure(BOX), realisation=Cells.tensor(*nodes))
    t = time.time()
    K = pg.port.Coordinates.cells(L)
    x = np.random.default_rng(0).normal(size=K.dim)
    m = K.element(x)
    ok = np.allclose(K.numbers(m), x) and np.isclose(L.norm(m) ** 2, L.backend.cells.volumes @ x ** 2)
    assert K.dim == 102400 and ok and time.time() - t < 5.0
