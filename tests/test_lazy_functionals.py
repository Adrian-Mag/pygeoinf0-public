"""Functionals by measures do no energy work until an operation needs it (plan 022 P1).

Creating pg.Functionals(M, M.riesz(measures)) and evaluating it on a formula used to assemble the self-energies of
every weighted or surface measure (for an error estimate and a zero check); the author's 25 bumps took 60-90 s before
the first value. Now the estimate and the norms are computed on first use, the zero check reads the atoms, and the
Gram, norms and errors are the same numbers as before when asked for."""
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.realisations.rkhs import energy

warnings.simplefilter("ignore", pg.RealisationWarning)


def bump(cx, cy, r=0.1):
    def f(x, y):
        r2 = ((np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2) / r ** 2
        out = np.zeros(np.broadcast(r2).shape)
        inside = r2 < 1
        out[inside] = np.exp(-1.0 / (1.0 - r2[inside]))
        return out
    return f


@pytest.fixture
def counted(monkeypatch):
    calls = []
    orig = energy.ExactEngine.measure_energy

    def spy(self, A, B):
        calls.append(1)
        return orig(self, A, B)
    monkeypatch.setattr(energy.ExactEngine, "measure_energy", spy)
    return calls


def setup():
    M = pg.Sobolev(pg.Box((0, 1), (0, 1)), 1, length_scale=0.1)
    centres = [(0.3, 0.7), (0.6, 0.4)]
    mus = [pg.LebesgueMeasure(pg.Disk(0.1, centre=c)).with_density(bump(*c)) for c in centres]
    return M, mus


def test_creation_and_evaluation_do_no_energy_work(counted):
    M, mus = setup()
    T = pg.Functionals(M, M.riesz(mus))
    y = T(M(lambda x, y: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)))
    assert counted == []
    assert np.all(np.isfinite(y.values))
    assert np.all(np.isnan(y.err))                 # a formula in H^1: its error is unknown, as before


def test_gram_norms_and_errors_on_first_use(counted):
    M, mus = setup()
    K = M.riesz(mus)
    T = pg.Functionals(M, K)
    G = T.gram()
    # the energies are assembled now: by the exact engine, or by the grid route (plan 022 P3)
    assert counted or M.backend.engine.grid.report
    assert np.allclose(np.sqrt(np.diag(G)), T._knorm, rtol=1e-12)
    assert np.allclose(T._kerr, K.errors()) and np.all(T._kerr >= 0)
    assert np.allclose(G, (T @ T.adjoint).matrix(), rtol=1e-12)


def test_value_on_a_representer_is_a_gram_row():
    M, mus = setup()
    K = M.riesz(mus)
    T = pg.Functionals(M, K)
    y = T(K[0])
    assert np.allclose(y.values, T.gram()[:, 0], rtol=1e-10)
    assert np.all(np.isfinite(y.err))


def test_zero_check_reads_the_atoms():
    M, mus = setup()
    R = M.riesz(mus).realisation
    assert not np.any(M.backend.exact_zeros(R))
    assert M.backend.errors_pending(R)
