"""Weighted pieces in the spectral engine (EXP-031 phase W4): node rules per kind of piece, the far field without
near corrections, the rate p = 2s + 1 of pieces of full dimension with jumps, and the two-resolution estimate against
the actual error of products (exact energies from the W3 routes)."""
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.realisations.rkhs import spectral as S
from pygeoinf0.realisations.rkhs.weighted_spectral import piece_nodes

warnings.simplefilter("ignore", pg.RealisationWarning)
needs_finufft = pytest.mark.skipif(S._finufft() is None, reason="finufft is not installed (pip install finufft)")


def test_node_rules_integrate_the_weight_on_every_kind_of_piece():
    from pygeoinf0.realisations.rkhs.atoms import FlatWeight
    from pygeoinf0.realisations.rkhs import weighted as W, weighted_volume as WV
    w2 = FlatWeight(lambda X: 1 + X[:, 0] * X[:, 1])
    C = pg.Curve(lambda t: (t, 0.5 + 0.15 * np.sin(2 * np.pi * t)), 0.1, 0.9)
    pieces = [W.SegmentPiece([0.1, 0.2], [0.8, 0.7], w2), W.CurvePiece(C, w2),
              WV.ParallelogramPiece([0.2, 0.3], np.diag([0.4, 0.3]), w2), WV.RoundPiece([0.4, 0.5], 0.2, w2),
              WV.TrianglePiece([0.1, 0.1], [0.6, 0.2], [0.3, 0.7], w2)]
    for p in pieces:
        X, Wt = piece_nodes(p, 200.0)
        Xr, Wr = p.formula_nodes()
        assert Wt.sum() == pytest.approx(Wr.sum(), rel=1e-12)
        xi = np.array([150.0, -60.0])                                    # a frequency inside kmax: the transform
        assert np.sum(Wt * np.exp(-1j * X @ xi)) == pytest.approx(np.sum(Wr * np.exp(-1j * Xr @ xi)), abs=1e-9 * Wt.sum())


@pytest.mark.slow(reason="63 s on europa at load 26")
@needs_finufft
def test_weighted_boxes_with_jumps_are_corrected_and_the_estimate_holds():
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), 2, length_scale=0.1)
    mus = [pg.LebesgueMeasure(pg.Box(a, b)).with_density(lambda x, y: (1 + x) * (2 - y))
           for a, b in (((0.1, 0.3), (0.1, 0.4)), ((0.35, 0.6), (0.2, 0.3)), ((0.5, 0.9), (0.55, 0.85)))]
    K = H.riesz(mus)
    A = K.realisation.kx.atoms
    assert S.unsupported(H.backend.engine.exact, A) is None
    E = H.backend.energy(A, A)
    errs = []
    for lxi in (8, 12):
        eng = S.SpectralEngine(H.backend.engine.exact, lxi=lxi, eps=1e-13, reach_tol=1e-14)
        A.spectral = None
        V = eng.plan(A).probes()                                            # the vectors the estimate uses
        err = np.max(np.linalg.norm(eng.apply(A, V) - E @ V, axis=0) / np.linalg.norm(E @ V, axis=0))
        assert A.spectral.stats["weighted_corrected"] == 3 and eng.estimate(A) >= err
        errs.append(err)
    assert errs[1] < 1e-7 and errs[1] < errs[0] / 5      # 2.4e-7, 2.6e-8; uncorrected (W4): 1.4e-5, 1.7e-6
    assert "weighted pieces by their node rules" in str(eng.claims(A))


@pytest.mark.slow(reason="43 s on europa at load 26")
@needs_finufft
def test_near_corrections_for_curves_and_jump_weights():
    """Lower-dimensional pieces (a weighted segment, a curve) and boxes with jumps, corrected on their near pairs:
    products converge to ~1e-11 (without corrections the curve alone converges like (l Xi)^-2 beta, beta = 3/2), and
    the estimate does not under-state."""
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), 2, length_scale=0.1)
    mus = [pg.LebesgueMeasure(pg.Box((0.1, 0.3), (0.1, 0.4))).with_density(lambda x, y: (1 + x) * (2 - y)),
           pg.LebesgueMeasure(pg.Box((0.3, 0.6), (0.2, 0.3))).with_density(lambda x, y: 1 + 0 * x),
           pg.LebesgueMeasure(pg.Segment((0.1, 0.9), (0.9, 0.5))).with_density(lambda x, y: 1 + x),
           pg.LebesgueMeasure(pg.Curve(lambda t: (t, 0.5 + 0.15 * np.sin(2 * np.pi * t)), 0.1, 0.9)).with_density(1.0)]
    A = H.riesz(mus).realisation.kx.atoms
    E = H.backend.energy(A, A)
    for lxi, tol in ((12, 2e-8), (17, 1e-9)):
        eng = S.SpectralEngine(H.backend.engine.exact, lxi=lxi, eps=1e-13, reach_tol=1e-14)
        A.spectral = None
        P = eng.plan(A)
        V = P.probes()
        err = np.max(np.linalg.norm(eng.apply(A, V) - E @ V, axis=0) / np.linalg.norm(E @ V, axis=0))
        assert err < tol and eng.estimate(A) >= err
        assert P.stats["weighted_corrected"] == 4 and P.stats["weighted_pairs"] >= 6
