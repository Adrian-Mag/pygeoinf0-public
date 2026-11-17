"""Plan 065 P4, A10 (architecture review 2026-11-13): the mass of a density measure on an interval is the integral of
the density's piecewise Legendre fit, so its stated error is the fit's: |int (w - p)| <= sqrt(b - a) ||w - p||_L2 with
the fit's L2 estimate (Cauchy-Schwarz). Before the repair mass() stated 0.0, presenting a statement about the
realisation as exact for the measure. Constant and per-piece constant weights stay exact (error 0)."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg

CASES = [(np.exp, math.e - 1, (0.0, 1.0)),
         (lambda x: np.exp(-((x - 0.5) / 0.05) ** 2), 0.05 * math.sqrt(math.pi) * math.erf(10.0), (0.0, 1.0)),
         (lambda x: np.sqrt(x + 1e-3), (2 / 3) * ((3 + 1e-3) ** 1.5 - 1e-3 ** 1.5), (0.0, 3.0))]


@pytest.mark.parametrize("w, exact, ab", CASES)
def test_a_fitted_mass_states_the_fits_error(w, exact, ab):
    mu = pg.LebesgueMeasure(pg.Interval(*ab)).with_density(w)
    m, e = mu.mass()
    assert e > 0.0                                                   # not exact: a fit
    assert e == pytest.approx(math.sqrt(ab[1] - ab[0]) * float(mu.density.err[0]), rel=1e-12)
    assert abs(m - exact) <= e
    assert e < 1e-10                                                 # and still small: the fit is good


def test_the_normalised_measure_says_its_mass_was_estimated():
    mu = pg.LebesgueMeasure(pg.Interval(0.0, 1.0)).with_density(np.exp).normalised()
    c = [c for c in mu.claims if "normalised" in c.statement]
    assert c and c[0].status == "checked"


def test_constant_weights_stay_exact():
    I = pg.Interval(0.0, 2.0)
    assert pg.LebesgueMeasure(I).with_density(3.0).mass() == (6.0, 0.0)
