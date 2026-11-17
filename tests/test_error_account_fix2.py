"""Plan 054 fix2 (the author's decisions of 2026-11-06 on the EXP-030 rerun and the narrow-bumps round), each against an
exact answer or the per-direction route, with a planted-defect control:

    the vectorised BGP route returns the bracket of every end (widened by WIDEN x E where the rule made it BRACKET) and
    E, as a Bound does for one direction (optimisers/dual/spectral.py, solvers/dataspace/factored.py, convex/feasible.intervals)
    the rule's re-solves run as arrays on perturbed decompositions built once per decomposition (optimisers/dual/exactness.py)
    SOLA's residual route in batches, kept for the next call with the same pairings and gamma, not run beyond
    RESIDUAL_MAX properties or at verification 0, each stated (solvers/dataspace/cancellation.py)
    bank functions fitted with no breaks take the feature width their parameters state as resolution (core/fits.py)"""
import math
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.optimisers.dual import exactness
from pygeoinf0.optimisers.dual import spectral as SM
from pygeoinf0.core.errors import PrecisionWarning, RealisationWarning
from pygeoinf0.inference import methods
from pygeoinf0.solvers.dataspace import cancellation
from test_error_account_rule import EXACT, _problem


def test_vectorised_bounds_return_the_widened_bracket_and_E():
    """P2's Gaussian kernels (exact ends at 60 digits): the vectorised route's ends are BRACKET where the per-direction
    Bounds are, and its brackets are theirs (both widened by 2 E) and hold the exact ends; E is returned. Control: the
    computed values alone (the former return) are not within TOL by their own stated error, so a BRACKET status came
    without a bracket that holds the end."""
    lo_ex, hi_ex = EXACT[(0.15, 20, "block")]
    P = _problem(0.15, 20, "block")
    lo, hi, sl, su, cl, br = P.intervals(brackets=True)
    b_lo, b_hi = P.bounds(0)
    assert (sl[0], su[0]) == ("BRACKET", "BRACKET")
    for got, b, ref, E in ((br["lower_bracket"][0], b_lo, lo_ex, br["error_lower"][0]),
                           (br["upper_bracket"][0], b_hi, hi_ex, br["error_upper"][0])):
        assert got[0] <= ref <= got[1]
        assert got[1] - got[0] >= 4 * E * (1 - 1e-6) > 4e-9 * max(1.0, abs(ref))  # widened by WIDEN x E each side
        if math.isfinite(b.lower):                    # the per-direction bracket (its E from scalar re-solves)
            assert np.allclose(got, (b.lower, b.upper), rtol=0, atol=0.05 * (got[1] - got[0]))
        else:                                         # PRIMAL there (no feasible witness): the vectorised one is inside
            assert got[1] <= b.upper + 0.05 * (got[1] - got[0])
        assert b.error is not None and math.isclose(E, b.error, rel_tol=0.5)
    B = methods.BGP(P, None, None, arrays=(lo, hi, sl, su, cl, br))
    assert B.bracket(0) == (tuple(br["lower_bracket"][0]), tuple(br["upper_bracket"][0]))
    # control: the value alone (the former return) is not within its own stated error of TOL: E > TOL. Whether it
    # misses the exact end is the rounding's (by library; tests/test_error_account_blas.py)
    assert all(E > 1e-9 * max(1.0, abs(v)) for v, E in ((lo[0], br["error_lower"][0]), (hi[0], br["error_upper"][0])))


def test_resolves_as_arrays_match_one_by_one(monkeypatch):
    """The re-solves on the perturbed problems as arrays (spread_many) against one by one (spread): the same change of
    each bound to well within the widening (a tenth of E; the bound is widened by WIDEN = 2 E): the re-solves round
    differently as arrays, by up to 2 % of E on the Mac (Accelerate; this problem's values are 3e-8 from the exact ones
    there) and identically on europa (OpenBLAS). The perturbed decompositions are built once per decomposition (a second
    problem on it reuses them). Control: rebuilding them per problem gives other objects."""
    P = _problem(0.15, 20, "block")
    sp = P.K._spectral()
    dec = sp.dec
    tg = dec.targets(P.T)
    beta = dec._beta(tg)
    cc, const = tg.diag(), np.zeros(1)
    B = np.c_[beta, -beta]
    cc2, c2 = np.r_[cc, cc], np.r_[const, const]
    v = SM.values(sp, B, cc2, c2)
    many = exactness.spread_many(sp, B, cc2, c2, v)
    one = np.array([exactness.spread(sp, B[:, j], float(cc2[j]), float(c2[j]), float(v[j])) for j in range(2)])
    assert np.all(np.abs(many - one) <= 1e-2 * 1e-9 * np.maximum(1.0, np.abs(v)) + 0.1 * one)
    first = exactness._perturbed_decompositions(dec)
    sp2 = dec.ball_problem(sp.delta * 1.01, sp.r, sp.rho) if hasattr(dec, "ball_problem") else None
    if sp2 is not None:
        assert exactness._perturbed(sp2)[0][0].dec is first[0][0]
    dec._perturbed_decs = None
    assert exactness._perturbed_decompositions(dec)[0][0] is not first[0][0]      # control


def test_sola_residual_route_in_batches_cached_and_budgeted(monkeypatch):
    """SOLA's residual route formed RESIDUAL_BATCH properties at a time gives the full route's mismatch (and error), is
    kept for a second call with another datum (one route), and beyond RESIDUAL_MAX properties is not run, which the
    warning says. Control: a batch of the whole family (the former route) gives the same numbers."""
    P = _problem(0.1, 10, "estimable")
    dec = P.K._spectral().dec
    T = P.T
    if not cancellation.residuals_at_hand(dec, dec.targets(T)):
        pytest.skip("the residual route is not at hand for this problem")
    calls = []
    real = cancellation.residuals

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(cancellation, "residuals", counted)
    gam = 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = dec.sola(T, gam, refuse=False)
        r2 = dec.sola(T, gam, refuse=False)
    assert np.array_equal(np.asarray(r1.mismatch), np.asarray(r2.mismatch))
    assert len(calls) <= 1
    monkeypatch.setattr(cancellation, "RESIDUAL_BATCH", 10 ** 6)                 # control: one batch
    dec._residual_cache = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r3 = dec.sola(T, gam, refuse=False)
    assert np.allclose(np.asarray(r3.mismatch), np.asarray(r1.mismatch), rtol=1e-12, atol=0)
    if len(calls) == 2:                                                          # the route ran: now the budget
        monkeypatch.setattr(cancellation, "RESIDUAL_MAX", 0)
        dec._residual_cache = {}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dec.sola(T, gam, refuse=False)
        assert any("residual route was not run" in str(x.message) for x in w if x.category is PrecisionWarning)
        with pg.settings.using(verification=0):
            monkeypatch.setattr(cancellation, "RESIDUAL_MAX", 1024)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dec.sola(T, gam)                                                 # no refusal at the quick level
            assert any("not verified" in str(x.message) for x in w if x.category is PrecisionWarning)


def test_batched_residual_diagonal_equals_the_full_gram():
    """residuals(..., cols) in batches: the diagonal of the full u-Gram and the same norms, to the bit or rounding."""
    P = _problem(0.1, 10, "block")
    dec = P.K._spectral().dec
    tg = dec.targets(P.T)
    if not cancellation.residuals_at_hand(dec, tg):
        pytest.skip("the residual route is not at hand for this problem")
    full = cancellation.residuals(dec, tg, 1e-6)
    part = cancellation.residuals(dec, tg, 1e-6, cols=np.arange(full[0].shape[0]))
    assert np.allclose(np.diag(full[0]), part[0], rtol=1e-13, atol=0)
    for a, b in zip(full[1:], part[1:]):
        assert np.allclose(a, b, rtol=1e-13, atol=0)


@pytest.mark.parametrize("w", [1e-4, 3e-5])
def test_narrow_bank_gaussian_is_resolved_by_its_width(w, monkeypatch):
    """A bank Gaussian of width 1e-4 on [0, 1] (support: the whole domain, so no breaks): fitted at its width it holds
    the closed-form integral to 1e-7 and says so in its claims. Control: without the width (the former exemption) its
    default sampling misses it by percents, silently."""
    M = pg.L2("[0, 1]")
    g = F.on(M.domain).gaussian(centre=(0.37123,), width=w)
    exact = float(g.integral()[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", pg.errors.ResolutionWarning)
        el = M(g)
        val = float(M.inner(el, M.constant(1.0)))
    assert abs(val - exact) <= 1e-7 * exact
    assert any("feature width its parameters state" in c.statement for c in el.claims if c.key == "resolution")
    monkeypatch.setattr(type(g), "feature_width", lambda self: None)
    bare = M(g)
    try:
        wrong = abs(float(M.inner(bare, M.constant(1.0))) - exact) / exact
    except pg.errors.RealisationError:                         # missed entirely: caught as a zero fit
        wrong = 1.0
    assert wrong > 1e-3


def test_bank_width_beyond_the_realisation_warns_not_refuses():
    """A bank Gaussian of width 0.01 on the unit sphere needs degree 753, above the harmonic cap 511: fitted as before,
    with a RealisationWarning and a claim that the width is not met (never a refusal)."""
    M = pg.L2(pg.Sphere(1.0))
    c = (math.sin(1.1) * math.cos(0.7), math.sin(1.1) * math.sin(0.7), math.cos(1.1))
    g = F.on(M.domain).gaussian(centre=c, width=0.01)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        el = M(g)
        el.realisation
    assert any(x.category is RealisationWarning and "feature width its parameters state" in str(x.message) for x in w)
    assert any("is not met" in cl.statement for cl in el.claims if cl.key == "resolution")


def test_feature_width_from_parameters():
    """The smallest width-like parameter, radius - inner, none for a polynomial."""
    B = pg.Box((0, 1), (0, 1))
    fb = F.on(B)
    assert fb.gaussian(width=0.02).feature_width() == 0.02
    assert fb.legendre(index=(2, 3)).feature_width() is None
