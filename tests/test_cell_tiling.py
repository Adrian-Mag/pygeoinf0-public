"""Plan 065 P4, A11 (architecture review 2026-11-13): a cell realisation of L2 is an orthogonal projection only when
its cells tile the domain. Before the repair Cells(boxes) took any boxes whose volumes added up, overlapping ones
included, and recorded nothing. Now the tiling is a claim of the cells: THEOREM for a tensor mesh (by construction),
CHECKED for boxes whose interiors were checked pairwise disjoint (and whose volumes the space checks against its
domain), ASSERTED when the caller declines the check or it is over its budget; overlapping cells are refused
(NotATiling)."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.realisations import Cells, piecewise_constant as cells_module

SQUARE = pg.Box([(0, 1), (0, 1)])
OVERLAP = [pg.Box([(0, 0.6), (0, 1)]), pg.Box([(0.4, 1), (0, 0.5)]), pg.Box([(0.6, 1), (0.5, 0.75)])]   # 0.6+0.3+0.1


def tiling(C):
    return C.claims.find("tiling")


def test_overlapping_cells_whose_volumes_add_up_are_refused():
    assert sum(float(np.prod([s.b - s.a for s in B.sides])) for B in OVERLAP) == pytest.approx(1.0)
    with pytest.raises(pg.NotATiling, match="overlap"):
        pg.L2(pg.LebesgueMeasure(SQUARE), realisation=Cells(OVERLAP))


def test_the_tiling_is_recorded():
    T = Cells.tensor(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
    assert tiling(T).status == "theorem"
    general = Cells([pg.Box(list(zip(lo, hi))) for lo, hi in zip(T.lo, T.hi)])
    assert tiling(general).status == "checked"
    vouched = Cells([pg.Box(list(zip(lo, hi))) for lo, hi in zip(T.lo, T.hi)], check=False)
    assert tiling(vouched).status == "asserted" and "caller" in tiling(vouched).reason
    X = pg.L2(pg.LebesgueMeasure(SQUARE), realisation=general)
    c = X.backend.claims.find("tiling")
    assert c.status == "checked" and "volumes" in c.reason
    assert X.backend.cells.on(pg.Box([(0, 1), (0, 0.5)]))[0].claims.find("tiling").status == "checked"


def test_a_check_over_its_budget_is_asserted_not_skipped_silently(monkeypatch):
    T = Cells.tensor(np.linspace(0, 1, 6), np.linspace(0, 1, 6))
    boxes = [pg.Box(list(zip(lo, hi))) for lo, hi in zip(T.lo, T.hi)]
    monkeypatch.setattr(cells_module, "PAIR_BUDGET", 3)
    c = tiling(Cells(boxes))
    assert c.status == "asserted" and "budget" in c.reason


@pytest.mark.parametrize("route", ["grid", "sweep"])
def test_both_checks_find_every_overlap(route, monkeypatch):
    if route == "sweep":
        monkeypatch.setattr(cells_module, "_grid_cover", lambda *a: None)
    rng = np.random.default_rng(0)
    for _ in range(30):
        lo = rng.uniform(0, 1, size=(12, 2))
        hi = lo + rng.uniform(0.01, 0.3, size=(12, 2))
        boxes = [pg.Box(list(zip(a, b))) for a, b in zip(lo, hi)]
        ov = np.minimum(hi[:, None], hi[None]) - np.maximum(lo[:, None], lo[None])
        brute = any(np.all(ov[i, j] > 0) for i in range(12) for j in range(i + 1, 12))
        if brute:
            with pytest.raises(pg.NotATiling):
                Cells(boxes)
        else:
            assert tiling(Cells(boxes)).status == "checked"


def test_the_tiling_reaches_the_space_claims():
    """Integration of plan 065 (p4's open item 3): the space's claims carry its realisation's tiling, so a caller of
    X.claims sees whether the cells' projection is an orthogonal projection on record (was absent from X.claims)."""
    T = Cells.tensor(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
    boxes = [pg.Box(list(zip(lo, hi))) for lo, hi in zip(T.lo, T.hi)]
    for cells, status in [(T, "theorem"), (Cells(boxes), "checked"), (Cells(boxes, check=False), "asserted")]:
        X = pg.L2(pg.LebesgueMeasure(SQUARE), realisation=cells)
        c = X.claims.find("tiling")
        assert c is not None and c.status == status and "tile" in c.statement
    assert pg.L2(pg.Interval(0, 1)).claims.find("tiling") is None        # no cells, no tiling claim
