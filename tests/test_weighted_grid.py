"""The grid-spectral route for weighted measures in 2-D (plan 022 P3; realisations/rkhs/weighted_grid.py).

Smooth weights vanishing smoothly at the boundary of their piece (C^inf bumps) get their energies from uniform-grid
samples and a Fourier sum, with cross energies against points and segments from their potentials. References, independent
of the package (experiments/EXP-038-weighted-energies/p3_grid_reference.py, digits copied): the Hankel route for the
energies of radial bumps and their potentials integrated along segments (480 Gauss nodes; the value has stopped
changing). The package's own exact disk-to-segment route is about 1e-8 off (its estimates say so), so the exact routes
serve only as loose consistency checks. The author's case (25 bumps in H^1 on the unit square) took 72 min for the Gram
on the whole-box form and 60 s on the bumps' disks before this route; here a second and a half.
"""
import time
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F

warnings.simplefilter("ignore", pg.RealisationWarning)
RADIUS = 0.09
CENTRES = [(0.3, 0.7), (0.5, 0.5), (0.62, 0.41)]
SEGMENTS = [((0.0, 0.3), (0.8, 0.9)), ((0.45, 1.0), (0.4, 0.1)), ((1.0, 0.7), (0.15, 0.55))]
GRAM = np.array([[0.000216910652129085, 1.0701832361428343e-05, 1.975420779507671e-06],
                 [1.0701832361428343e-05, 0.000216910652129085, 5.397502619089662e-05],
                 [1.975420779507671e-06, 5.397502619089662e-05, 0.000216910652129085]])
CROSS = np.array([[0.004873701564610769, 0.005113796113514789, 0.005299118778702953],
                  [0.004873701564610767, 0.009081383418159252, 0.006498635786623132],
                  [0.0011382345436667153, 0.002533052531562249, 0.002160542632810347]])


def bump(cx, cy, radius=RADIUS):
    def f(x, y):
        r2 = ((np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2) / radius ** 2
        out = np.zeros(np.broadcast(r2).shape)
        inside = r2 < 1
        out[inside] = np.exp(-1.0 / (1.0 - r2[inside]))
        return out
    return f


def setup(measures=None, s=1):
    B = pg.Box((0, 1), (0, 1))
    M = pg.Sobolev(B, s, length_scale=0.1)
    if measures is None:
        measures = [F.on(B).bump(centre=c, radius=RADIUS).measure() for c in CENTRES]
    return B, M, M.riesz(measures), M.backend.engine.grid


def rays():
    return [pg.LebesgueMeasure(pg.Segment(s, e)) for s, e in SEGMENTS]


def test_gram_of_bumps_against_the_hankel_reference():
    _, M, K, grid = setup()
    G = pg.Functionals(M, K).gram()
    assert grid.report["route"] == "grid" and grid.report["weights"] == 3
    assert np.max(np.abs(G - GRAM)) <= 1e-10 * np.max(GRAM)


def test_the_error_estimates_dominate_the_actual_errors():
    _, M, K, _ = setup()
    G = pg.Functionals(M, K).gram()
    est = K.realisation.kx.atoms.self_energy[1]          # stated, not zero (the exact routes state theirs too)
    assert np.all(est > 0) and np.all(est >= np.abs(G - GRAM))


def test_cross_energies_with_segments_against_the_hankel_reference():
    _, M, K, _ = setup()
    T = pg.Functionals(M, K)
    G = pg.Functionals(M, M.riesz(rays()))
    TG = (T @ G.adjoint).matrix()
    assert np.max(np.abs(TG - CROSS)) <= 1e-9 * np.max(CROSS)


def test_cross_energies_with_points():
    pts = [(0.31, 0.72), (0.9, 0.1)]
    _, M, K, grid = setup(s=1.5)                         # point values are bounded for s > d / 2 = 1
    T = pg.Functionals(M, K)
    kp = M.riesz([pg.DiracMeasure(p) for p in pts])
    TP = (T @ pg.Functionals(M, kp).adjoint).matrix()
    assert grid.report["route"] == "grid" and TP.shape == (3, 2) and TP[0, 0] > 0
    # the potential of the bump at a point is the representer's value there: <k_bump, k_p> = k_bump(p)
    assert TP[0, 0] == pytest.approx(float(K[0](*pts[0])), rel=1e-6)


def test_the_original_whole_box_form_and_the_disk_form_agree():
    B = pg.Box((0, 1), (0, 1))
    forms = {"disk": [pg.LebesgueMeasure(pg.Disk(RADIUS, centre=c)).with_density(bump(*c)) for c in CENTRES],
             "box": [pg.LebesgueMeasure(B).with_density(bump(*c)) for c in CENTRES]}
    for name, ms in forms.items():
        _, M, K, grid = setup(ms)
        G = pg.Functionals(M, K).gram()
        assert grid.report["route"] == "grid", name
        assert np.max(np.abs(G - GRAM)) <= 1e-10 * np.max(GRAM), name


def test_gram_is_symmetric_positive_definite_and_fast():
    B = pg.Box((0, 1), (0, 1))
    ticks = np.linspace(0.1, 0.9, 5)
    cs = [(x, y) for y in ticks for x in ticks]
    M = pg.Sobolev(B, 1, length_scale=0.1)
    ms = [F.on(B).bump(centre=c, radius=RADIUS).measure() for c in cs]
    t0 = time.perf_counter()
    G = pg.Functionals(M, M.riesz(ms)).gram()
    assert time.perf_counter() - t0 < 30            # 0.6 s here; the whole-box form took 72 min
    assert np.allclose(G, G.T, rtol=0, atol=1e-18)
    assert np.linalg.eigvalsh(G).min() > 0


def test_a_weight_that_does_not_settle_goes_to_the_exact_routes():
    B = pg.Box((0, 1), (0, 1))
    jump = lambda x, y: (1.0 + np.asarray(x)) * (np.asarray(x) < 0.5)           # a jump at x = 0.5
    ms = [pg.LebesgueMeasure(B).with_density(bump(0.3, 0.7)), pg.LebesgueMeasure(B).with_density(jump)]
    _, M, K, grid = setup(ms)
    G = pg.Functionals(M, K).gram()
    rep = grid.report
    assert rep["route"] == "grid" and rep["weights"] == 1             # the bump on the grid, the jump refused
    assert G[0, 0] == pytest.approx(GRAM[0, 0], rel=1e-10)            # the bump: the Hankel reference
    with pg.settings.using(weighted_grid="off"):
        _, M2, K2, g2 = setup([pg.LebesgueMeasure(B).with_density(bump(0.3, 0.7)),
                                pg.LebesgueMeasure(B).with_density(jump)])
        G2 = pg.Functionals(M2, K2).gram()
    assert g2.report is None
    # the jump and its pairing with the bump are the exact routes in both runs: identical. (The old whole-box route's
    # value for the bump alone, G2[0, 0], is 2.3 % off the reference: the inaccuracy this route removes.)
    assert G[1, 1] == pytest.approx(G2[1, 1], rel=1e-12) and G[0, 1] == pytest.approx(G2[0, 1], rel=1e-12)
    assert grid.claim() is not None and "grid-spectral" in grid.claim().statement


def test_settings_switch_and_claim():
    with pytest.raises(ValueError):
        pg.settings.weighted_grid = "sometimes"
    _, M, K, grid = setup()
    assert grid.claim() is None                        # nothing computed yet
    pg.Functionals(M, K).gram()
    c = [c for c in M.backend.kernel_claims if getattr(c, "key", "") == "weighted grid"]
    assert c and c[0].status == "checked"


def mixed(B):
    return [F.on(B).bump(centre=CENTRES[0], radius=RADIUS).measure(), pg.DiracMeasure((0.5, 0.5)),
            pg.LebesgueMeasure(pg.Segment((0.0, 0.3), (0.8, 0.9))), pg.LebesgueMeasure(pg.Box((0.1, 0.4), (0.2, 0.5)))]


def test_mixed_sets_agree_with_the_exact_routes():
    B = pg.Box((0, 1), (0, 1))
    _, M, K, grid = setup(mixed(B), s=1.5)
    G = pg.Functionals(M, K).gram()
    assert grid.report["route"] == "grid" and grid.report["weights"] == 1
    with pg.settings.using(weighted_grid="off"):
        _, M2, K2, _ = setup(mixed(B), s=1.5)
        G2 = pg.Functionals(M2, K2).gram()
    assert np.max(np.abs(G - G2)) <= 1e-6 * np.max(np.abs(G2))


def test_a_disk_indicator_is_refused_and_its_estimate_covers_its_error():
    """Plan 022 acceptance: the indicator of a disk, given as a density (a jump), is never computed 1e-3 wrong with a
    1e-7 estimate (EXP-038 P1: the uniform-grid shell estimate said 1.1e-7, the error was 1.2e-3). The grid route
    refuses it, and the exact route's stated estimate dominates its actual error against the disk itself."""
    import re
    B = pg.Box((0, 1), (0, 1))
    M = pg.Sobolev(B, 1, length_scale=0.1)
    c, R = (0.45, 0.55), 0.12
    ind = lambda x, y: (((np.asarray(x) - c[0]) ** 2 + (np.asarray(y) - c[1]) ** 2) < R ** 2).astype(float)
    S = pg.Box((c[0] - R, c[0] + R), (c[1] - R, c[1] + R))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        K = M.riesz([pg.LebesgueMeasure(S).with_density(ind), pg.LebesgueMeasure(pg.Disk(R, centre=c))])
        G = pg.Functionals(M, K).gram()
    assert M.backend.engine.grid.report == {"route": "exact", "why": "no weight's indicator settled"}
    stated = [float(m.group(1)) for w in caught if (m := re.search(r"errors up to ([0-9.eE+-]+)", str(w.message)))]
    actual = abs(G[0, 0] - G[1, 1]) / G[1, 1]                 # the disk itself: an exact piece
    assert stated and max(stated) >= actual and actual < 1e-2
