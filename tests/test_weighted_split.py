"""The far/near split of the grid-spectral route (plan 022 P4; realisations/rkhs/weighted_grid.py).

References, independent of the package (experiments/EXP-038-weighted-energies/p4_reference.py, digits copied; two
resolutions agree to 4e-12): radial bumps of radius R and power k, energies by the Hankel route, cross energies with
segments by their potentials integrated along the segment. Measured on europa (EXP-038 NOTES.md, p4_cases.py): the
author's 25 bumps 0.32 s (P3 0.97 s), 25 bumps of radius 0.02 0.24 s (P3: 22 % off, stated), 400 bumps 4.1 s.
"""
import math
import time
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.realisations.rkhs import weighted_grid as WS

warnings.simplefilter("ignore", pg.RealisationWarning)
CENTRES = [(0.3, 0.7), (0.5, 0.5), (0.62, 0.41)]
SEGMENTS = [((0.0, 0.3), (0.8, 0.9)), ((0.45, 1.0), (0.4, 0.1)), ((1.0, 0.7), (0.15, 0.55))]
SETS = {"narrow": [(0.02, 1.0)] * 3, "mixed": [(0.09, 1.0), (0.02, 1.0), (0.02, 1.0)],
        "powers": [(0.05, 1.0), (0.05, 2.0), (0.05, 0.5)]}
GRAM = {
    "narrow": [[1.285904661149921e-06, 2.3616582426360484e-08, 4.3593177402156196e-09],
               [2.3616582426360484e-08, 1.285904661149921e-06, 1.1911171006125093e-07],
               [4.3593177402156196e-09, 1.1911171006125093e-07, 1.285904661149921e-06]],
    "mixed": [[0.00021691065212908525, 5.027332355004636e-07, 9.279809722455771e-08],
              [5.027332355004637e-07, 1.285904661149921e-06, 1.1911171006125093e-07],
              [9.279809722455773e-08, 1.1911171006125093e-07, 1.285904661149921e-06]],
    "powers": [[3.1474923285313747e-05, 2.3869647497595396e-07, 3.864144185836464e-07],
               [2.3869647497595396e-07, 2.1939666436130874e-06, 2.6580946152151316e-06],
               [3.864144185836466e-07, 2.6580946152151316e-06, 0.00014231490959814216]]}
CROSS = {
    "narrow": [[0.00022894880742772115, 0.00024022757776536737, 0.00024893336383406364],
               [0.000228948807427721, 0.0004266212275487884, 0.0003052823184862561],
               [5.3470126943717e-05, 0.00011899361267091893, 0.00010149444987983846]],
    "powers": [[0.0014506487705614039, 0.0015221124942977333, 0.0015772734623727602],
               [0.0003652106652925096, 0.0006805303949451131, 0.0004869750573896484],
               [0.000748035355265998, 0.0016646945578181179, 0.0014198851061909949]]}
B = pg.Box((0, 1), (0, 1))


def setup(name, centres=CENTRES):
    M = pg.Sobolev(B, 1, length_scale=0.1)
    ms = [F.on(B).bump(centre=c, radius=R, k=k).measure() for c, (R, k) in zip(centres, SETS[name])]
    return M, M.riesz(ms), M.backend.engine.grid


@pytest.mark.parametrize("name", sorted(SETS))
def test_gram_against_the_hankel_reference(name):
    M, K, grid = setup(name)
    G = pg.Functionals(M, K).gram()
    ref = np.array(GRAM[name])
    assert grid.report["route"] == "grid" and grid.report["weights"] == 3
    assert np.max(np.abs(G - ref)) <= 1e-10 * np.max(ref)
    est = K.realisation.kx.atoms.self_energy[1]
    assert np.all(est > 0) and np.all(est >= np.abs(G - ref))         # stated, never zero, dominating


@pytest.mark.parametrize("name", ["narrow", "powers"])
def test_cross_energies_with_segments_against_the_reference(name):
    M, K, _ = setup(name)
    TG = (pg.Functionals(M, K) @ pg.Functionals(M, M.riesz([pg.LebesgueMeasure(pg.Segment(s, e))
                                                            for s, e in SEGMENTS])).adjoint).matrix()
    ref = np.array(CROSS[name])
    assert np.max(np.abs(TG - ref)) <= 1e-9 * np.max(ref)


def test_a_bump_narrower_than_the_coarse_spacing_is_not_judged_on_no_samples():
    # P3 passed it at h = 0.025 (no lattice point inside: indicator 1.0, every energy 22 % off, EXP-038 NOTES.md)
    M, K, grid = setup("narrow")
    pg.Functionals(M, K).gram()
    assert min(grid.report["levels"]) >= 4 and grid.report["indicator"] <= WS.IND_MAX


def test_translated_copies_share_their_spectra_and_the_narrow_grid_is_fast():
    ticks = np.linspace(0.1, 0.9, 5)
    cs = [(x, y) for y in ticks for x in ticks]
    M = pg.Sobolev(B, 1, length_scale=0.1)
    ms = [F.on(B).bump(centre=c, radius=0.02).measure() for c in cs]
    t0 = time.perf_counter()
    G = pg.Functionals(M, M.riesz(ms)).gram()
    seconds = time.perf_counter() - t0
    rep = M.backend.engine.grid.report
    assert rep["far_types"] == 1 and rep["near_groups"] == 1             # one weight's work for 25 translates
    assert seconds < 30                                                  # 0.24 s on europa; P4 acceptance: 5 s
    D = np.linalg.norm(np.array(cs)[:, None] - np.array(cs)[None], axis=-1)
    assert np.allclose(np.diag(G), GRAM["narrow"][0][0], rtol=1e-10, atol=0)
    i, j = np.unravel_index(np.argmin(np.where(D > 0, D, np.inf)), D.shape)
    assert G[i, j] > 0 and np.linalg.eigvalsh(G).min() > 0
    assert np.allclose(G, G.T, rtol=0, atol=1e-20)


def test_the_general_route_and_the_offsets_route_agree(monkeypatch):
    """Translates take offsets (one spectrum per type); many different weights take one product over absolute spectra
    (the 'powers' set above, against Hankel). The same weights through both routes: the same numbers."""
    ticks = np.linspace(0.2, 0.8, 3)
    cs = [(x, y) for y in ticks for x in ticks]

    def gram():
        M = pg.Sobolev(B, 1, length_scale=0.1)
        G = pg.Functionals(M, M.riesz([F.on(B).bump(centre=c, radius=0.05).measure() for c in cs])).gram()
        return G, M.backend.engine.grid.report
    G1, rep = gram()
    assert rep["far_types"] == 1
    monkeypatch.setattr(WS, "SHARE", 1e9)                              # every set counts as "many different weights"
    G2, _ = gram()
    assert np.max(np.abs(G1 - G2)) <= 1e-12 * np.max(np.abs(G1))     # rounding of sums of ~2e5 terms (1.05e-13 seen)


def test_the_split_kernel_is_short_range():
    """|K_H(r)| < 1e-13 K(l) beyond r_H = RANGE / w (the constant the near pairs rest on), checked at one w."""
    from scipy import special
    ell, w = 0.1, WS.RANGE / 0.2
    rho, wr = np.polynomial.legendre.leggauss(40)
    edges = np.linspace(0.0, (WS.ALPHA + 9) * w, 400)
    x = (0.5 * (edges[1:] + edges[:-1])[:, None] + 0.5 * np.diff(edges)[:, None] * rho).ravel()
    q = (0.5 * np.diff(edges)[:, None] * wr).ravel() * x / (1 + ell * ell * x * x) * WS.taper(w, x) / (2 * math.pi)
    r = np.linspace(0.2, 1.0, 81)
    KT = special.j0(np.outer(r, x)) @ q
    K = special.k0(r / ell) / (2 * math.pi * ell * ell)
    assert np.max(np.abs(K - KT)) < 1e-13 * special.k0(1.0) / (2 * math.pi * ell * ell)
