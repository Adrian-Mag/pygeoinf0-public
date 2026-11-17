"""Plan 054, narrow bumps, option C (the author's decision of 2026-11-06; core/fits.py): every fit from samples of a
callable, and every pointwise re-fit, states the resolution its samples achieved; resolution=w makes the fit sample at
w/2 and its claim states w met; a bare fit warns (ResolutionWarning) and is refused under
settings.resolution_policy = "refuse".

References are exact: the L2 error of a fit against the function by mpmath quadrature (30 digits) with the bump's
centre and support as points, or closed-form Gaussian integrals. Each test has a planted-defect control: the fit at the
declared resolution with its sampling put back to the default (the bump is then lost, and the claim must say that the
resolution was not met), or the claim of a bare fit checked against a bump wider than its stated resolution."""
import math
import warnings

import mpmath as mp
import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as FB
from pygeoinf0.core.errors import ResolutionUnstated, ResolutionWarning
from pygeoinf0.realisations.backends import IntervalBackend

C0 = 0.3141


def _bump(c, w):
    return lambda x: np.sin(3 * x) + np.exp(-((x - c) / w) ** 2)


def _true_error(M, F, c, w):
    """||F - f||_L2 on [0, 1] by mpmath, the bump's centre and its support among the quadrature points."""
    mp.mp.dps = 30
    B, R = M.backend, F.realisation
    fr = lambda x: float(B.evaluate(R, np.array([float(x)]))[0, 0])                       # noqa: E731
    f_mp = lambda x: mp.sin(3 * x) + mp.e ** (-((x - c) / w) ** 2)                           # noqa: E731
    pts = sorted({0.0, c - 8 * w, c - w, c, c + w, c + 8 * w, 1.0} | set(R.breaks.tolist()))
    return float(mp.sqrt(mp.quad(lambda x: (f_mp(x) - fr(x)) ** 2, pts)))


def _size(w):
    return math.sqrt(math.sqrt(math.pi / 2) * w)                                           # ||exp(-((x - c)/w)^2)||


@pytest.mark.parametrize("w", [1e-4, 1e-5, 1e-6])
def test_a_declared_resolution_catches_a_narrow_bump(w, monkeypatch):
    M = pg.L2(pg.Interval(0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # neither a ResolutionWarning nor any other
        F = M(_bump(C0, w), resolution=w)
        F.realisation
    assert _true_error(M, F, C0, w) < 1e-6 * _size(w)                         # (F4: the scout accepts 1e-8 misfits)
    claim = F.claims.find("resolution")
    assert claim.status == "checked" and "met" in claim.statement and f"{w:g}" in claim.statement
    # control: the same declaration with the default sampling (the defect P3 found) loses the bump, and says so
    monkeypatch.setattr(IntervalBackend, "_scout", lambda self, breaks, w, N=1: 2049)
    with pytest.warns(pg.RealisationWarning, match="not met"):
        Fd = M(_bump(C0, w), resolution=w)
        Fd.realisation
    assert _true_error(M, Fd, C0, w) > 1e-3 * _size(w)
    assert Fd.claims.find("resolution").status == "unknown"


@pytest.mark.parametrize("w", [1e-4, 1e-5, 1e-6])
def test_a_bare_fit_states_a_resolution_that_says_the_bump_can_be_lost(w):
    M = pg.L2(pg.Interval(0, 1))
    with pytest.warns(ResolutionWarning, match="resolution 0.000977"):
        F = M(_bump(C0, w))
        F.realisation
    claim = F.claims.find("resolution")
    r = float(claim.statement.split("resolution is ")[1].split(" ")[0])
    assert r == pytest.approx(2 / 2048, rel=1e-3) and r > w                      # stated: a bump narrower than r
    err = _true_error(M, F, C0, w)
    assert err > 1e-3 * _size(w)                                                # ... and this one was lost (in part)
    # control: a bump wider than the stated resolution is caught without any declaration
    W = 4 * r
    with pytest.warns(ResolutionWarning):
        G = M(_bump(C0, W))
        G.realisation
    assert _true_error(M, G, C0, W) < 1e-6 * _size(W)


def test_pointwise_arithmetic_is_covered():
    """np.exp(-((M.x - c) / w) ** 2) is re-fitted from samples: it warns and states its resolution; with_resolution
    declares one and meets it."""
    M, w = pg.L2(pg.Interval(0, 1)), 1e-4          # (at 1e-5 the realised ((x - c)/w)^2 carries 6e-8 of rounding:
    exact = math.sqrt(math.pi / 2) * w             # its re-fit does not converge, and says so: RealisationWarning)
    with pytest.warns(ResolutionWarning, match="pointwise arithmetic"):
        g = np.exp(-((M.x - C0) / w) ** 2)
        lost = g.norm() ** 2
    assert "resolution is 0.000977" in g.claims.find("resolution").statement
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResolutionWarning)
        h = np.exp(-((M.x - C0) / w) ** 2).with_resolution(w)
        caught = h.norm() ** 2
    assert abs(caught - exact) < 1e-7 * exact
    assert h.claims.find("resolution").status == "checked"
    assert abs(lost - exact) > 1e-3 * exact                                      # control: the bare re-fit lost some


def test_the_policy_refuses_bare_fits_and_names_the_choices():
    M = pg.L2(pg.Interval(0, 1))
    f = lambda x: np.sin(3 * x)                                                  # noqa: E731
    with pg.settings.using(resolution_policy="refuse"):
        with pytest.raises(ResolutionUnstated) as e:
            M(f).realisation
        assert any("resolution=" in n for n in e.value.needs) and isinstance(e.value, pg.Refused)
        with pytest.raises(ResolutionUnstated):
            np.exp(M.x).realisation
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResolutionWarning)
            for u in (M(f, resolution=0.1), M(f, breaks=[0.5]), np.exp(M.x).with_resolution(1.0),
                      M.x, M.constant(2.0), FB.on(pg.Interval(0, 1)).gaussian(centre=0.3, width=1e-4).element(M)):
                assert np.isfinite(u.norm())
            P = pg.L2(pg.Box((0, 1), (0, 1)))
            assert np.isfinite(P(lambda x, y: x * y, support=((0.0, 0.5), (0.0, 0.5))).norm())
    with pytest.raises(ValueError):
        pg.settings.resolution_policy = "silent"
    assert pg.settings.resolution_policy == "warn"
    with pytest.warns(ResolutionWarning, match="will become a refusal"):        # the default: warn, and fit
        assert M(f).norm() > 0


def test_a_two_dimensional_bump_is_caught_at_its_declared_resolution(monkeypatch):
    """A Gaussian of width w = 1e-3 in the unit square, between the default samples: its integral pi w^2 (closed form,
    up to e^-(0.3/w)^2) is found with resolution=w, lost without, and the bare claim's stated resolution exceeds w."""
    M = pg.L2(pg.Box((0, 1), (0, 1)))
    w, c = 1e-3, (0.3141, 0.2718)
    f = lambda x, y: np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / w ** 2)       # noqa: E731
    exact = math.pi * w * w
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResolutionWarning)
        F = M(f, resolution=w)
        assert abs(F.integral() - exact) < 1e-8 * exact
    assert F.claims.find("resolution").status == "checked"
    with pytest.warns(ResolutionWarning):
        F0 = M(f)
        lost = F0.integral()
    r = float(F0.claims.find("resolution").statement.split("resolution is ")[1].split(" ")[0])
    assert r > w and abs(lost - exact) > 0.5 * exact                             # stated could be lost; it was
    # control: the lattice check disabled (the declared fit is the bare one): the claim must not say met
    from pygeoinf0.core import sampling
    monkeypatch.setattr(sampling, "lattice_misses", lambda *a, **k: None)
    with pytest.warns(pg.RealisationWarning, match="not met"):
        Fd = M(f, resolution=w)
        Fd.realisation
    assert Fd.claims.find("resolution").status == "unknown"


def test_internal_constructions_do_not_repeat_the_warning():
    """The parts of X({...}) give one warning; resolution= passes to them; an inclusion and a restriction inherit."""
    iv = pg.Interval(0, 1)
    X = pg.DirectSum({"a": pg.L2(iv), "b": pg.L2(iv)})
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        X({"a": lambda x: np.sin(x), "b": lambda x: np.cos(x)})
    res = [g for g in got if issubclass(g.category, ResolutionWarning)]
    assert len(res) == 1 and "2 part(s)" in str(res[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResolutionWarning)
        x = X({"a": lambda x: np.sin(x), "b": lambda x: np.cos(x)}, resolution=0.5)
    assert x["a"].claims.find("resolution").status == "checked"
    M = pg.L2(iv)
    u = M(lambda x: np.sin(3 * x))
    with warnings.catch_warnings(record=True) as got:                 # a restriction is a piece of u: u's one warning
        warnings.simplefilter("always")
        v = M.restrict_to(pg.Interval(0, 0.5))(u)
        v.realisation, u.realisation
    assert sum(issubclass(g.category, ResolutionWarning) for g in got) == 1
    assert v.claims.find("resolution") is not None


def test_a_family_at_a_declared_resolution():
    M, w = pg.L2(pg.Interval(0, 1)), 1e-5
    c = np.array([0.1234, 0.5432, 0.7777])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResolutionWarning)
        F = M.family(lambda x, c: np.exp(-((x - c) / w) ** 2), c=c, resolution=w)
    assert np.allclose(np.asarray(M.norm(F)) ** 2, math.sqrt(math.pi / 2) * w, rtol=1e-9)
    assert F[1].claims.find("resolution").status == "checked"
    with pytest.warns(ResolutionWarning):
        F0 = M.family(lambda x, c: np.exp(-((x - c) / w) ** 2), c=c)
    assert not np.allclose(np.asarray(M.norm(F0)) ** 2, math.sqrt(math.pi / 2) * w, rtol=1e-3)      # control: lost
    assert "may be lost" in str(F0[1].claims.find("resolution"))
