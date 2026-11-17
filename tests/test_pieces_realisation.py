"""Realisation of localised densities on balls and shells (plan 035 P2): bank functions give natural-coordinate breaks
and a least angular degree from their Support; a fit that is zero for a function known not to be raises; the radial x
harmonic realisation is checked on the support and refuses what its grid cannot resolve; ZeroFunctional names the
realisation. References: the bank's closed forms (l2_norm_sq, integral)."""
import math
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.core.errors import RealisationError, ZeroFunctional
from pygeoinf0.realisations.backends import NaturalBreaks

SHELL = pg.Shell(0.55, 0.8)
U = np.array([0.3, 0.2, 0.7]) / np.linalg.norm([0.3, 0.2, 0.7])
NARROW = [(0.035, "axis"), (0.035, "off"), (0.02, "axis"), (0.02, "off")]      # 3.0 and 1.7 degrees at r = 0.675


def _bump(radius, where, D=SHELL):
    return F.on(D).bump(centre=(0, 0, 0.675) if where == "axis" else tuple(0.675 * U), radius=radius)


def test_bank_supports_give_natural_breaks():
    f = _bump(0.035, "off")
    nb = f.fit_breaks(SHELL)
    assert isinstance(nb, NaturalBreaks) and len(nb) == 3
    r, th, ph = SHELL.from_cartesian(*(0.675 * U))
    delta = math.asin(0.035 / 0.675)
    assert nb[0] == pytest.approx((0.64, 0.71), abs=1e-14)
    assert nb[1] == pytest.approx((th - delta, th + delta), abs=1e-14)
    w = math.asin(math.sin(delta) / math.sin(th))
    assert sorted(nb[2]) == pytest.approx(sorted([(ph - w) % (2 * math.pi), (ph + w) % (2 * math.pi)]), abs=1e-14)
    assert nb.degree == math.ceil(2 * math.pi / delta) and nb.box[0] == pytest.approx((0.64, 0.71))
    on_axis = _bump(0.035, "axis").fit_breaks(SHELL)
    assert on_axis[1] == pytest.approx((delta,)) and on_axis[2] is None              # a polar cap: no longitudes
    whole = F.on(pg.Ball(1.0)).bump(centre=(0, 0, 0.1), radius=0.3).fit_breaks(pg.Ball(1.0))
    assert whole[0] == pytest.approx((0.4,)) and whole[1] is None and whole.degree is None   # holds the centre
    assert F.on(pg.Box((0, 1), (0, 1))).bump(centre=(0.5, 0.5), radius=0.1).fit_breaks() is not None   # boxes as before


@pytest.mark.parametrize("radius,where", NARROW + [(0.05, "off"), (0.12, "off")])
def test_tensor_fits_narrow_and_off_axis_bumps(radius, where):
    f = _bump(radius, where)
    M = pg.L2(SHELL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pg.RealisationWarning)
        u = M(f)
        n2 = M.norm(u) ** 2
    assert n2 == pytest.approx(f.closed["l2_norm_sq"][0], rel=1e-9)
    assert u.integral() == pytest.approx(f.closed["integral"][0], rel=1e-9)


@pytest.mark.parametrize("radius,where", NARROW + [(0.05, "axis")])
def test_radial_harmonics_refuses_what_its_grid_cannot_resolve(radius, where):
    M = pg.L2(SHELL, realisation="radial_harmonics")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pg.RealisationWarning)
        with pytest.raises(RealisationError, match="radial x harmonic"):
            M.norm(M(_bump(radius, where)))


@pytest.mark.parametrize("where", ["axis", "off"])
def test_radial_harmonics_at_the_cap_states_its_error(where):
    """A 10 degree bump at the degree cap: the norm is near, and the stated error covers the difference (before plan
    035 the fit claimed 1e-13 against an actual 2e-3 of the norm, without a warning about the cap)."""
    f = _bump(0.12, where)
    M = pg.L2(SHELL, realisation="radial_harmonics")
    with pytest.warns(pg.RealisationWarning, match="did not fully converge"):
        u = M(f)
        n = M.norm(u)
    exact = math.sqrt(f.closed["l2_norm_sq"][0])
    assert abs(n - exact) <= u.error and u.error > 1e-6 * exact


def test_a_zero_fit_of_a_function_known_not_to_be_zero_raises():
    M = pg.L2(SHELL)
    f = _bump(0.035, "off")
    with pytest.raises(RealisationError, match="fit is zero"):        # radial breaks alone: the samples miss it
        M.norm(M(f, breaks=([0.6], None, None)))
    outside = F.on(pg.Ball(1.0)).bump(centre=(0, 0, 0.2), radius=0.1)  # genuinely zero on the shell: no refusal
    assert M.norm(M(outside)) == 0.0


def test_density_on_a_whole_piece_is_realised_with_its_support_breaks():
    """LebesgueMeasure(shell).with_density(bank bump) on a direct sum is realised by the tensor fit (the set is the
    summand's domain), now with the bump's breaks: exact where the fit sampled the whole shell and returned 0."""
    P = pg.LayeredBall([0.55, 0.8, 1.0], names=["a", "b", "c"])
    M = pg.DirectSum({k: pg.L2(P[k]) for k in P.names})
    bump = F.on(P.ball).bump(centre=tuple(0.675 * U), radius=0.03)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pg.RealisationWarning)
        k = M.riesz({"b": pg.LebesgueMeasure(P["b"]).with_density(bump)})
        got = M.inner(k, M({"b": lambda x, y, z: 1.0 + 0 * x}))
    assert got == pytest.approx(bump.closed["integral"][0], rel=1e-9)


def test_zero_functional_names_the_realisation():
    M = pg.DirectSum({"b": pg.L2(SHELL)})
    f = _bump(0.035, "off")
    plain = pg.LebesgueMeasure(SHELL).with_density(lambda x, y, z: f(x, y, z))     # no support known: nothing to check
    with pytest.raises(ZeroFunctional, match="realised as zero by the direct sum realisation .*tensor"):
        pg.Functionals(M.riesz([{"b": plain}]))
