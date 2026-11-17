"""Plan 005, P2: above settings.dense_max the same user code runs matrix-free (G G* applied, CG, Lanczos for the
discrepancy principle), and agrees with the dense exact route to the accuracy it is asked for."""
import numpy as np
import pytest

import ladder
import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported


def _run(P):
    G, alpha, post = ladder.solve(P)
    return alpha, G(post.mean).values, post


@pytest.mark.parametrize("name, n, kw", [("L1", 150, {}), ("L2", 150, {}), ("L3", 80, {"segments": 2})])
def test_matrix_free_route_agrees_with_the_dense_one(name, n, kw):
    P = ladder.build(name, n, **kw)
    a_dense, pred_dense, _ = _run(P)
    assert "exactly" in str(a_dense.claims)
    with pg.settings.using(dense_max=40, accuracy=1e-6):
        a_free, pred_free, post = _run(P)
        assert "Lanczos" in str(a_free.claims)
        assert post.std(P.measures[:2]).shape == (2,)
    assert a_free == pytest.approx(a_dense, rel=1e-6)
    assert np.sqrt(np.mean((pred_free - pred_dense) ** 2)) <= 1e-5 * P.sigma        # predicted data, in noise units


def test_default_accuracy_is_one_percent_of_the_noise():
    P = ladder.build("L2", 150)
    _, pred_dense, _ = _run(P)
    with pg.settings.using(dense_max=40):
        alpha, pred, _ = _run(P)
    chi2 = P.D.norm(P.D(pred) - P.d) ** 2
    assert abs(chi2 - P.n) <= 0.05 * np.sqrt(2 * P.n)                  # chi^2 = n to a fraction of its spread
    assert np.sqrt(np.mean((pred - pred_dense) ** 2)) <= 0.03 * P.sigma


def test_operators_stay_operators_above_the_budget():
    P = ladder.build("L1", 60)
    G = pg.Functionals(P.H.riesz(P.measures), into=P.D)
    with pg.settings.using(dense_max=20):
        A = G @ G.adjoint
        assert not isinstance(A, pg.Matrix) and "without forming" in str(A.claims)
        assert A.is_selfadjoint()[0] is True
        with pytest.raises(Unsupported, match="dense_max"):
            A.matrix()
        rec = (A + 1.0 * pg.Identity(P.D)).recommend_solver()
        assert isinstance(rec.solver, pg.CG) and rec.solver.atol == pytest.approx(0.01 * np.sqrt(60))
    assert isinstance(G @ G.adjoint, pg.Matrix)                          # back to dense below the budget


def test_settings_are_scoped_and_validated():
    old = pg.settings.dense_max
    with pg.settings.using(dense_max=5, accuracy=0.1):
        assert pg.settings.dense_max == 5 and pg.settings.accuracy == 0.1
    assert pg.settings.dense_max == old
    with pytest.raises(AttributeError):
        with pg.settings.using(nonsense=1):
            pass
