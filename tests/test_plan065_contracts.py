"""Plan 065 P5 (architecture review 2026-11-13, B5): contract tests between the two implementations of the ball's
Laplacian spectrum, the function bank's eigenfunctions (functions/families/eigenfunctions.py: ball_wavenumber,
_ball_norm, laplace_eigen) and the closed-form radial spectrum of realisations/spectra/bessel.py (RadialBessel, through
pg.Laplacian(...).radial(l)). Plan 068 P3: the bank's wavenumbers are a view on the Laplacian's spectrum, so the
eigenvalues are checked against an independent reference here (the zeros of j_l and j_l' bracketed on a fine grid and
found by Brent's method), and the bank's normalisation (Lommel) against bessel.py's."""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import spherical_jn

import pygeoinf0 as pg
from pygeoinf0.functions.families.eigenfunctions import _ball_norm, ball_wavenumber
from pygeoinf0.realisations.spectra.bessel import RadialBessel

R = 1.7
NMODES = 5


def _spectrum(bc, l):
    lap = pg.Laplacian(pg.Ball(R), bc={"boundary": bc}, centre="regular")
    return RadialBessel(lap.radial(l))


def _bank_index(bc, l, i):
    """The bank's index n of the i-th (0-based) radial mode: Neumann l = 0 starts at the constant n = 0."""
    return i if (bc == "neumann" and l == 0) else i + 1


@pytest.mark.parametrize("bc", ["dirichlet", "neumann"])
@pytest.mark.parametrize("l", [0, 1, 2, 5])
def test_b5_eigenvalues_agree(bc, l):
    lam = _spectrum(bc, l).eigenvalues(NMODES)
    bank = np.array([ball_wavenumber(R, l, _bank_index(bc, l, i), bc) ** 2 for i in range(NMODES)])
    assert np.allclose(lam, bank, rtol=1e-14, atol=0)                       # the bank is a view on the spectrum
    ref = np.array(_reference_zeros(l, NMODES, bc)) ** 2 / R ** 2
    assert np.allclose(lam, ref, rtol=1e-13, atol=1e-13)


def _reference_zeros(l, n, bc):
    """The first n zeros of j_l (dirichlet) or j_l' (neumann; 0 first for l = 0, the constant), independently: sign
    changes on a fine grid, each refined by Brent's method."""
    f = (lambda x: spherical_jn(l, x)) if bc == "dirichlet" else (lambda x: spherical_jn(l, x, True))
    x = np.linspace(1e-6, (n + l + 2) * np.pi, 200000)
    v = f(x)
    idx = np.nonzero(np.sign(v[1:]) != np.sign(v[:-1]))[0]
    out = ([0.0] if (bc == "neumann" and l == 0) else []) + [brentq(f, x[i], x[i + 1], xtol=1e-300, rtol=8.9e-16)
                                                             for i in idx]
    return out[:n]


@pytest.mark.parametrize("bc", ["dirichlet", "neumann"])
@pytest.mark.parametrize("l", [0, 1, 3])
def test_b5_radial_eigenfunctions_agree_up_to_sign(bc, l):
    """|N j_l(k r)| (the bank, L2-normalised with the real orthonormal Y_lj) equals |phi(r)| (bessel.py, orthonormal in
    L2((0, R), r^2 dr)): the same functions with the same normalisation."""
    sp = _spectrum(bc, l)
    phi = sp.eigenfunctions(NMODES)
    r = np.linspace(0.05, R, 41)
    P = phi(r)
    for i in range(NMODES):
        k = ball_wavenumber(R, l, _bank_index(bc, l, i), bc)
        bank = _ball_norm(R, l, k, bc) * (spherical_jn(l, k * r) if k > 0 else np.ones_like(r) * (1.0 if l == 0 else 0.0))
        assert np.allclose(np.abs(P[:, i]), np.abs(bank), rtol=1e-11, atol=1e-12 * np.abs(bank).max())


def test_b5_the_bank_element_is_the_spectrum_s_mode():
    """laplace_eigen(n, l, j) states the spectrum's eigenvalue and unit norm, and its values are the radial mode
    times Y_lj."""
    import pygeoinf0.functions as F
    D = pg.Ball(R)
    sp = _spectrum("dirichlet", 1)
    lam = sp.eigenvalues(2)[1]
    f = F.on(D).laplace_eigen(n=2, l=1, j=0)
    assert f.closed["eigenvalue"][0] == pytest.approx(lam, rel=1e-13)
    assert f.closed["l2_norm_sq"][0] == 1.0
    z = np.linspace(0.1, R, 9)                                   # on the axis: Y_10 = sqrt(3 / (4 pi)) there
    vals = f(np.zeros_like(z), np.zeros_like(z), z)
    radial = sp.eigenfunctions(2)(z)[:, 1] * np.sqrt(3 / (4 * np.pi))
    assert np.allclose(np.abs(vals), np.abs(radial), rtol=1e-11)


# ------------------------------------------------------------------------------------------------ D2
def test_d2_the_judges_share_one_threshold():
    """Plan 065 P5 (D2): core/precision.py is the rule's one home; the judges warn above TOL of the value and refuse at
    the value itself, each with its own mathematics for the error."""
    import warnings

    from pygeoinf0.core import precision
    from pygeoinf0.core.errors import PrecisionWarning, Refused
    from pygeoinf0.solvers.dataspace import cancellation
    from pygeoinf0.probability import input_error, posterior_covariance
    v = np.array([1.0, 2.0])
    lost, none, worst = precision.assess(v, np.array([2e-9, 1e-10]))
    assert lost.tolist() == [True, False] and not none.any() and worst == pytest.approx(2e-9)
    judges = [lambda e, r: input_error.judge("value", "x", v, e, "how", refuse=r),
              lambda e, r: cancellation.judge("x", v, e, r, "values", ["c"]),
              lambda e, r: posterior_covariance.judge_mean("x", float(v[0]), float(e[0]))]
    for j in judges:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PrecisionWarning)
            j(0.5 * precision.TOL * v, True)                       # below the threshold: silent
        with pytest.warns(PrecisionWarning, match=f"fewer than {precision.DIGITS} correct digits"):
            j(2 * precision.TOL * v, True)
        with pytest.raises(Refused):
            j(2 * v, True)                                         # no correct digit
    from pygeoinf0.optimisers.dual import exactness, exactness_reduce
    assert exactness.TOL == exactness_reduce.TOL == precision.TOL
