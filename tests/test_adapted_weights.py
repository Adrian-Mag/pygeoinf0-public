"""Weights decide the resolution of their own rules (plan 022 P2).

A density measure's weight used to be sampled on rules fixed in advance (24-64 Gauss points per piece for pairings
with formulas, kernel-sized panels for energies), so a narrow weight lost mass and energy: the unit-mass bump of radius
0.1 on [0, 1] integrated to 0.9996 (L2) and 0.9999 (H^1), its H^1 self-energy was 1.8 % off (18 % at radius 0.05) with
warnings up to 0.49, and the author's 25 bumps on the unit square gave values 1 % off. Now the pairings use rules
bisected (quartered in 2-D) where the weight is not resolved (core/adaptive.py), and a weighted segment or
subinterval whose weight needs it is split into pieces on a partition shared by every such measure on it.

References: the mass 1 (normalised by scipy.integrate.quad); interval energies from the Fourier route of
experiments/EXP-038-weighted-energies/p2_interval_reference.py (two resolutions agree to 1.4e-14; digits copied);
values on the square from the Hankel transform of the radial bump."""
import math
import warnings

import numpy as np
import pytest
from scipy import integrate, special

import pygeoinf0 as pg
from pygeoinf0.core.adaptive import adapted_edges, adapted_panels


def bump1(c, r):
    return lambda x: np.where(np.abs(np.asarray(x) - c) < r,
                              np.exp(-1.0 / np.maximum(1.0 - ((np.asarray(x) - c) / r) ** 2, 1e-300)), 0.0)


def bump2(cx, cy, r=0.1):
    def f(x, y):
        r2 = ((np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2) / r ** 2
        out = np.zeros(np.broadcast(r2).shape)
        inside = r2 < 1
        out[inside] = np.exp(-1.0 / (1.0 - r2[inside]))
        return out
    return f


SPECS = [(0.37, 0.1), (0.6, 0.05), (0.5, 0.3)]
E_REF = np.array([[0.006553408902829811, 0.0005449042049696771, 0.010199441540611568],
                  [0.0005449042049696769, 0.001985661690977436, 0.0057073336472474884],
                  [0.010199441540611568, 0.0057073336472474884, 0.03271847918818867]])


@pytest.mark.parametrize("space", ["L2", "H1"])
def test_narrow_bump_keeps_its_mass(space):
    c, r = 0.37, 0.1
    Z = integrate.quad(lambda x: float(bump1(c, r)(x)), c - r, c + r, epsabs=1e-16, epsrel=1e-15, limit=400)[0]
    I = pg.Interval("[0, 1]")
    M = pg.L2(I) if space == "L2" else pg.Sobolev(I, 1, length_scale=0.1)
    F = pg.Functionals(M, M.riesz([pg.LebesgueMeasure(I).with_density(lambda x: bump1(c, r)(x) / Z)]))
    assert F(M(lambda x: 1 + 0 * np.asarray(x))).values[0] == pytest.approx(1.0, abs=1e-12)


def test_interval_energies_of_narrow_and_overlapping_bumps():
    I = pg.Interval("[0, 1]")
    M = pg.Sobolev(I, 1, length_scale=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pg.RealisationWarning)       # no estimate above the warning threshold
        E = pg.Functionals(M, M.riesz([pg.LebesgueMeasure(I).with_density(bump1(c, r)) for c, r in SPECS])).gram()
    assert np.allclose(E, E_REF, rtol=1e-12, atol=0)


def hankel_value(cx, cy, R=0.1):
    """int b_c m for m = sin(A x) cos(A y): b^(A sqrt 2) sin(A cx) cos(A cy), b^ the radial transform."""
    A = 2 * math.pi
    rho = A * math.sqrt(2.0)
    bh = 2 * math.pi * integrate.quad(lambda r: math.exp(-1.0 / (1.0 - (r / R) ** 2)) * special.j0(rho * r) * r,
                                      0, R, epsabs=1e-17, epsrel=1e-14, limit=400)[0]
    return bh * math.sin(A * cx) * math.cos(A * cy)


@pytest.mark.parametrize("space", ["L2", "H1"])
def test_bump_densities_on_the_whole_square(space):
    dom = pg.Box((0, 1), (0, 1))
    M = pg.L2(dom) if space == "L2" else pg.Sobolev(dom, 1, length_scale=0.1)
    centres = [(0.1, 0.1), (0.3, 0.7), (0.9, 0.5)]                  # (0.1, 0.1) touches two faces
    T = pg.Functionals(M, M.riesz([pg.LebesgueMeasure(dom).with_density(bump2(*c)) for c in centres]))
    v = T(M(lambda x, y: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))).values
    assert np.allclose(v, [hankel_value(*c) for c in centres], rtol=0, atol=1e-14)


def test_the_adapted_rule_follows_the_weight():
    w = bump1(0.37, 0.02)
    x, W, ok = adapted_panels(np.linspace(0, 1, 5), 16, w)
    Z = integrate.quad(lambda t: float(w(t)), 0.35, 0.39, epsabs=1e-17, epsrel=1e-15, limit=400)[0]
    assert ok and W @ w(x) == pytest.approx(Z, rel=1e-12)
    A, B, _ = adapted_edges(np.linspace(0, 1, 5), 16, w)
    assert np.any(B - A < 0.25)                                       # refined near the bump
    assert np.all((B - A)[(B < 0.3) | (A > 0.45)] == 0.25)            # but not where w vanishes
