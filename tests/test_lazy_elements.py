"""Values without a realisation, combinations kept as terms (plan 022 P6, P7; core/lazy.py), and bank functions that give
the fit their support and kinks (P8a; functions/base.BankFunction.fit_breaks).

The author's case (EXP-038 p6_sum_plot.py, europa): 100 overlapping bank bumps of radius 0.09 in H^1 (l = 0.1), sum and
plot: 34 s for the sum (100 fits and the merge) and 2.5 s for the plot before; 0.002 s and 0.21 s after, the values
equal to a NumPy sum to 4e-16 (1.3e-13 before, from the realisation), functionals 32.5 s -> 4.3 s by linearity.
"""
import time
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F

warnings.simplefilter("ignore", pg.RealisationWarning)
B = pg.Box((0, 1), (0, 1))
TICKS = np.linspace(0.1, 0.9, 10)
R = 0.09


def bump_np(X, Y, c, radius=R):
    r2 = ((X - c[0]) ** 2 + (Y - c[1]) ** 2) / radius ** 2
    out = np.zeros_like(r2)
    m = r2 < 1
    out[m] = np.exp(-1.0 / (1.0 - r2[m]))
    return out


@pytest.fixture(scope="module")
def case():
    M = pg.Sobolev(B, 1, length_scale=0.1)
    cs = [(x, y) for y in TICKS for x in TICKS]
    return M, cs, [F.on(B).bump(centre=c, radius=R).element(M) for c in cs]


def test_the_sum_of_100_bumps_evaluates_and_plots_from_the_formulas(case):
    M, cs, els = case
    t0 = time.perf_counter()
    total = sum(els)
    g = np.linspace(0, 1, 301)
    X, Y = np.meshgrid(g, g, indexing="ij")
    V = total(X, Y)
    seconds = time.perf_counter() - t0
    ref = sum(bump_np(X, Y, c) for c in cs)
    assert np.max(np.abs(V - ref)) <= 1e-14 * np.max(ref)
    assert seconds < 5                                  # 0.03 s on europa (NumPy 0.015 s); 36 s before (fits + merge)
    assert total._pl is None and all(e._pl is None for e in els)          # nothing was fitted or merged
    assert "combination of 100 terms" in repr(total)
    import matplotlib
    matplotlib.use("Agg")
    total.plot()
    assert total._pl is None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_functionals_of_a_combination_by_linearity_equal_the_merged_realisation(case):
    M, cs, els = case
    ms = [pg.LebesgueMeasure(pg.Box((0.2, 0.4), (0.3, 0.6))), pg.LebesgueMeasure(pg.Segment((0, 0.2), (1, 0.7)))]
    T = pg.Functionals(M, M.riesz(ms))
    few = els[:6]
    lazy = 2.0 * few[0] - few[1] + sum(few[2:])
    v = T(lazy).values
    assert lazy._pl is None                            # the terms were realised one by one, not merged
    merged = 2.0 * few[0] - few[1] + sum(few[2:])
    merged.realisation                                 # the merged realisation, as eager arithmetic made it
    w = T(merged).values
    assert np.allclose(v, w, rtol=1e-12, atol=0)


def test_inner_products_and_integrals_by_linearity_in_L2():
    M = pg.L2(B)
    a, b = (F.on(B).bump(centre=c, radius=0.1).element(M) for c in ((0.3, 0.4), (0.45, 0.5)))
    g = F.on(B).bump(centre=(0.4, 0.45), radius=0.2).element(M)
    s = a + 3.0 * b
    assert s.inner(g) == pytest.approx(a.inner(g) + 3.0 * b.inner(g), rel=1e-14)
    assert s.integral() == pytest.approx(a.integral() + 3.0 * b.integral(), rel=1e-14)
    assert s._pl is None
    n2 = s.norm() ** 2                                 # <s, s> uses the merged realisation (expanded, it cancels)
    assert n2 == pytest.approx(a.norm() ** 2 + 6.0 * a.inner(b) + 9.0 * b.norm() ** 2, rel=1e-12)


def test_elements_without_a_formula_combine_eagerly():
    M = pg.L2(B)
    f = M(lambda x, y: np.sin(x) * y)                  # a callable: no claim that it is the canonical representative
    h = f + f
    assert h._terms is None and h._pl is not None      # as before: iterations do not accumulate terms
    assert h(0.3, 0.4) == pytest.approx(2 * np.sin(0.3) * 0.4, rel=1e-12)


def test_formula_values_are_the_realisation_values_to_its_error():
    M = pg.L2(B)
    e = F.on(B).bump(centre=(0.37, 0.52), radius=0.1).element(M)
    x = np.linspace(0.28, 0.46, 7)
    v = e(x, 0.5)
    r = M.backend.evaluate(e.realisation, x, np.full_like(x, 0.5))[..., 0]
    assert np.max(np.abs(v - r)) <= 1e-12
    assert M.x(0.25, 0.5) == 0.25 and M.constant(2.0)(0.1, 0.9) == 2.0


def test_a_narrow_bump_is_realised_from_its_support():
    # before P8a the fit met no node of the support: realised as 0 with error 0 (EXP-038 NOTES.md, P8)
    M = pg.L2(B)
    b = F.on(B).bump(centre=(0.37, 0.52), radius=0.02)
    e = b.element(M)
    assert e.norm() ** 2 == pytest.approx(b.closed["l2_norm_sq"][0], rel=1e-12)
    assert b.fit_breaks() == ((0.35, 0.39), (0.5, 0.54)) or np.allclose(b.fit_breaks(), ((0.35, 0.39), (0.5, 0.54)))


def test_a_hat_is_realised_on_its_cells():
    # 20.8 s and 2.2e6 coefficients without its kinks as breaks (EXP-038 p8_local.py); exact on 16 pieces with them
    M = pg.L2(B)
    n, i, j = 40, 10, 17
    hat = F.on(B).hat(count=(n + 1, n + 1), index=(i, j))
    t0 = time.perf_counter()
    e = hat.element(M)
    R_ = e.realisation
    assert time.perf_counter() - t0 < 5
    assert int(M.backend.describe(R_)["size"]) <= 100
    assert e.norm() ** 2 == pytest.approx(hat.closed["l2_norm_sq"][0], rel=1e-12)


def test_a_calibrated_function_with_a_formula_stays_lazy():
    """Plan 070 P7: Problem.run returns a CalibratedFunction (inference/calibration.py), a Function with the problem's
    claims and the same values; arithmetic on it keeps the terms (core/lazy.py: the subclass declares that it keeps its
    base's values), and the values are those of the plain element."""
    from pygeoinf0.core.claims import ASSERTED, Claim
    from pygeoinf0.inference.calibration import CalibratedFunction, carry
    M = pg.L2(B)
    a, b = (F.on(B).bump(centre=c, radius=0.1).element(M) for c in ((0.3, 0.4), (0.45, 0.5)))
    ca = carry(a, [Claim("planted", ASSERTED, "test")])
    assert type(ca) is CalibratedFunction
    for s, plain in ((ca + b, a + b), (b + ca, b + a), (2.0 * ca, 2.0 * a), (ca - 3.0 * b, a - 3.0 * b)):
        assert s._pl is None and s._terms is not None             # kept as terms, nothing fitted
        X, Y = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 7), indexing="ij")
        assert np.array_equal(s(X, Y), plain(X, Y))
