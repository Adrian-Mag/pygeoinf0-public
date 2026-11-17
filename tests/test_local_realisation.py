"""realisation="local" on boxes (plan 025 P1; realisations/series/local.py, local_patches.py): functions with a stated support
box are fitted on it and combinations keep their patches; every number equals the realisation="tensor" route to its
error.

The author's case (EXP-038 p6_sum_plot.py; plan 025 NOTES, europa): 100 bank bumps of radius 0.09 on the unit square.
L2: the merged realisation of the sum 25.6 s and its norm 0.31 s with "tensor"; 0.003 s and 0.12 s with "local" (one
translate type, 1936 overlapping pairs, 25 distinct offsets), the squared norms equal to 1.6e-15 (EXP-039 p1_local.py).
"""
import pickle
import time
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.core.errors import Unsupported
from pygeoinf0.realisations.series.local import PS, Supported

warnings.simplefilter("ignore", pg.RealisationWarning)
B = pg.Box((0, 1), (0, 1))


def bumps(M, n=10, R=0.09):
    t = np.linspace(R + 0.01, 1 - R - 0.01, n)                   # supports inside the square, overlapping
    return [F.on(B).bump(centre=(x, y), radius=R).element(M) for y in t for x in t]


@pytest.fixture(scope="module")
def spaces():
    return pg.L2(B, realisation="local"), pg.L2(B, realisation="tensor")


def test_a_sum_of_translates_is_kept_as_patches_and_its_norm_is_the_tensor_one(spaces):
    L, Tn = spaces
    loc, ten = bumps(L, 5, 0.18), bumps(Tn, 5, 0.18)
    t0 = time.perf_counter()
    s = sum(loc)
    R = s.realisation
    n2 = s.norm() ** 2
    seconds = time.perf_counter() - t0
    assert isinstance(R, PS) and len(R.patches) == 25 and R.base is None
    assert len(set(R.types)) == 1                       # translates share one coefficient array (canonical)
    assert all(Q.C is R.patches[0][2].C for _, _, Q in R.patches)
    st = sum(ten)
    assert n2 == pytest.approx(st.norm() ** 2, rel=1e-13)
    assert s.integral() == pytest.approx(st.integral(), rel=1e-13)
    assert s.error <= 10 * st.error + 1e-14 and s.error > 0
    assert seconds < 5


def test_values_grid_values_and_products_equal_the_tensor_route(spaces):
    L, Tn = spaces
    a, b = bumps(L, 3, 0.3)[:2]
    at, bt = bumps(Tn, 3, 0.3)[:2]
    x = np.linspace(0, 1, 41)
    X, Y = np.meshgrid(x, x, indexing="ij")
    s, st = a - 2.5 * b, at - 2.5 * bt
    s.realisation
    assert np.max(np.abs(L.backend.evaluate(s.realisation, X, Y) - Tn.backend.evaluate(st.realisation, X, Y))) < 1e-13
    g = L.backend.grid_values(s.realisation, [x, x])[..., 0]
    assert np.max(np.abs(g - Tn.backend.grid_values(st.realisation, [x, x])[..., 0])) < 1e-13
    assert (a * b).integral() == pytest.approx(at.inner(bt), rel=1e-12)
    assert a.inner(b) == pytest.approx(at.inner(bt), rel=1e-13)


def test_patches_mix_with_functions_on_the_whole_box(spaces):
    L, Tn = spaces
    out = []
    for M in (L, Tn):
        f = M(lambda x, y: np.sin(3 * x) * np.cos(2 * y))
        e = F.on(B).bump(centre=(0.4, 0.55), radius=0.2).element(M)
        h = f + 3.0 * e
        h.realisation
        out.append((h.norm() ** 2, h.inner(f), h(0.41, 0.5), (f * e).integral(), h.integral()))
    assert np.allclose(out[0], out[1], rtol=1e-12, atol=0)
    assert isinstance((F.on(B).bump(centre=(0.4, 0.55), radius=0.2).element(L) + L(lambda x, y: x)).realisation, PS)


def test_pixels_by_support_are_the_same_element_on_both_realisations(spaces):
    # M(f, support=box) is f times the indicator of the box on every realisation; values on shared faces come from
    # one pixel (half-open boxes, closed at the domain's faces)
    rows = []
    for M in spaces:
        px = [M(lambda x, y: 1.0 + 0 * x, support=((i / 4, (i + 1) / 4), (0, 1))) for i in range(4)]
        s = px[0] + 2 * px[1] + 3 * px[2] + 4 * px[3]
        rows.append(np.concatenate([s(np.array([0.2, 0.5, 0.7, 1.0, 0.0]), 0.5), [s.norm() ** 2, s.integral()],
                                    [p.inner(px[1]) for p in px]]))
    assert np.allclose(rows[0], rows[1], rtol=1e-12, atol=1e-13)
    assert rows[0][3] == pytest.approx(4.0, rel=1e-14) and rows[0][5] == pytest.approx(7.5, rel=1e-13)


def test_narrow_bump_and_hat_from_their_support(spaces):
    L, _ = spaces
    b = F.on(B).bump(centre=(0.37, 0.52), radius=0.02)
    e = b.element(L)
    assert isinstance(e.realisation, PS) and e.realisation.lo.tolist() == [[0.35, 0.5]]
    assert e.norm() ** 2 == pytest.approx(b.closed["l2_norm_sq"][0], rel=1e-12)
    hat = F.on(B).hat(count=(41, 41), index=(10, 17))
    h = hat.element(L)
    assert h.norm() ** 2 == pytest.approx(hat.closed["l2_norm_sq"][0], rel=1e-12)
    assert int(L.backend.describe(h.realisation)["size"]) <= 100


def test_functionals_in_H1_equal_the_tensor_route():
    ms = [pg.LebesgueMeasure(pg.Box((0.2, 0.4), (0.3, 0.6))), pg.LebesgueMeasure(pg.Segment((0, 0.2), (1, 0.7)))]
    vals = []
    for real in ("local", "tensor"):
        M = pg.Sobolev(B, 1, length_scale=0.1, realisation=real)
        T = pg.Functionals(M, M.riesz(ms))
        vals.append(T(sum(bumps(M, 3, 0.2))).values)
    assert np.allclose(vals[0], vals[1], rtol=1e-13, atol=0)


def test_families_of_patches_gram_combine_take(spaces):
    L, Tn = spaces
    fl = L.family(bumps(L, 4, 0.2))
    ft = Tn.family(bumps(Tn, 4, 0.2))
    G = np.asarray(L.inner(fl, fl))
    assert np.allclose(G, np.asarray(Tn.inner(ft, ft)), rtol=1e-12, atol=1e-16)
    assert np.count_nonzero(np.abs(G) > 0) < G.size              # bumps whose boxes do not meet are orthogonal
    c = np.linspace(-1, 1, 16)
    assert L.inner(fl, fl).shape == (16, 16) and c @ G @ c > 0


def test_functions_without_support_and_other_domains():
    L = pg.L2(B, realisation="local")
    f = L(lambda x, y: np.exp(x * y))
    assert not isinstance(f.realisation, PS)                      # as realisation="tensor"
    assert f.norm() == pytest.approx(pg.L2(B, realisation="tensor")(lambda x, y: np.exp(x * y)).norm(), rel=1e-14)
    with pytest.raises(Unsupported):
        pg.L2(pg.Disk(), realisation="local")
    s = pickle.loads(pickle.dumps(Supported((None, (0.3,)), ((0, 0.5), (0, 1)))))
    assert s.box == ((0.0, 0.5), (0.0, 1.0)) and tuple(s) == (None, (0.3,))


def test_local_is_the_default_on_boxes_and_translates_are_shifted_not_fitted():
    # the author, 2026-10-31: "local" the default on boxes, and no fit of a translated copy; 100 bumps: fits 2.6 s ->
    # 0.1 s (europa), the norm of their sum unchanged
    M = pg.L2(B)
    assert M.realisation == "local" and pg.L2(pg.Disk()).realisation == "tensor"
    t0 = time.perf_counter()
    els = bumps(M)
    for e in els:
        e.realisation
    assert time.perf_counter() - t0 < 5
    assert len(M.backend._shapes) == 1                               # one fit; 99 shifted copies of it
    R = sum(els).realisation
    assert all(Q.C is R.patches[0][2].C for _, _, Q in R.patches)
    ref = sum(bumps(pg.L2(B, realisation="tensor"), 4)).norm() ** 2
    assert sum(bumps(pg.L2(B), 4)).norm() ** 2 == pytest.approx(ref, rel=1e-13)


def test_a_wrong_translation_key_is_caught_by_the_value_check(monkeypatch):
    from pygeoinf0.functions.base import BankFunction
    monkeypatch.setattr(BankFunction, "translation",
                        lambda self: (("same",), tuple(a for a, _ in self.support_box())))   # claims everything shifts
    M = pg.L2(B)
    a = F.on(B).bump(centre=(0.3, 0.3), radius=0.1)
    b = F.on(B).bump(centre=(0.6, 0.6), radius=0.1, k=2.0)                 # same box size, another shape
    ea, eb = a.element(M), b.element(M)
    ea.realisation
    assert eb.norm() ** 2 == pytest.approx(b.closed["l2_norm_sq"][0], rel=1e-12)   # fitted, not shifted
    assert ea.norm() ** 2 == pytest.approx(a.closed["l2_norm_sq"][0], rel=1e-12)
