"""Plan 006 follow-ups (EXP-026 NOTES §14-§15): the reach of reflection images tied to a proven tail bound and to
pg.settings.images, and products of image sums by the spectral engine on the doubled (quadrupled on mixed axes) torus
under settings.engine = "spectral", against the exact image route and independent eigen-series."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.realisations.rkhs import green
from pygeoinf0.realisations.rkhs import spectral as S
from pygeoinf0.realisations.rkhs.kernels import Matern
from test_sobolev_boxes import series

needs_finufft = pytest.mark.skipif(S._finufft() is None, reason="finufft is not installed (pip install finufft)")


def chords(d, n, sides, seed=1, lo=0.03):
    """n straight rays between points on two different faces of the box [lo, side - lo]^d."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        fa, fb = rng.choice(2 * d, 2, replace=False)
        ends = []
        for f in (fa, fb):
            p = lo + (np.array(sides) - 2 * lo) * rng.random(d)
            p[f // 2] = lo if f % 2 == 0 else sides[f // 2] - lo
            ends.append(tuple(p))
        out.append(pg.Segment(*ends))
    return out


def space(dirichlet, d=2, s=2.0, l=0.15):
    sides = (1.0, 0.7, 0.8)[:d]
    D = pg.Box(*[f"[0, {x}]" for x in sides])
    return pg.Sobolev(pg.SobolevForm(D, 1, length_scale=l, dirichlet=dirichlet).operator(), s), sides


@pytest.mark.parametrize("d, s, sides, l", [(1, 0.8, (1.0,), 0.3), (2, 2.0, (1.0, 0.7), 0.1), (2, 1.0, (0.3, 1.0), 1.0),
                                            (3, 1.2, (1.0, 0.5, 0.8), 0.3)])
def test_tail_bound_dominates_the_omitted_images(d, s, sides, l):
    k = Matern(s, d, l)
    refl = green.Reflections([0.0] * d, sides, [1.0] * d, [1.0] * d)
    rng = np.random.default_rng(0)
    X = np.r_[rng.random((60, d)) * sides, np.zeros((1, d))]
    Y = np.r_[rng.random((60, d)) * sides, np.zeros((1, d))]
    for R in (l, 3 * l, 8 * l):
        imgs = refl.transforms(R + 25 * l)
        Rm, T = np.array([g.R for g in imgs]), np.array([g.t for g in imgs])
        dist = np.linalg.norm(X[:, None] - (Y[:, None] * Rm[None] + T[None]), axis=-1)
        omitted = np.sum(np.where(dist >= R, k(np.maximum(dist, R)), 0.0), axis=1)
        assert omitted.max() <= green.tail_bound(k, sides, R)


def test_reach_is_proven_and_shorter_than_the_old_margin():
    for d, s in ((2, 2.0), (3, 2.0), (3, 1.5)):
        k = Matern(s, d, 0.2)
        sides = np.ones(d)
        R = green.image_reach(k, sides, green.ROUNDING)
        assert green.tail_bound(k, sides, R) <= green.ROUNDING * k.k0 and R < k.reach()
        assert green.image_reach(k, sides, 5e-4) < 0.35 * R


def test_settings_images_validates_and_restores():
    assert pg.settings.images == "exact"
    with pytest.raises(ValueError):
        pg.settings.images = "fast"
    with pg.settings.using(images="accuracy"):
        assert pg.settings.images == "accuracy" and "images='accuracy'" in repr(pg.settings)
    assert pg.settings.images == "exact"


@pytest.mark.parametrize("dirichlet, conds", [((), ("NN", "NN")), (("x-", "y+"), ("DN", "ND"))])
def test_accuracy_mode_within_its_stated_bound_and_against_the_series(dirichlet, conds):
    H, sides = space(dirichlet, s=2.5, l=0.15)
    P = np.array([[0.1, 0.2], [0.5, 0.35], [0.93, 0.66], [0.3, 0.05]])
    rays = [pg.Segment((0.1, 0.1), (0.9, 0.6)), pg.Segment((0.0, 0.3), (0.6, 0.0))]
    mus = [pg.DiracMeasure(tuple(p)) for p in P] + rays
    mass = np.r_[np.ones(len(P)), [np.hypot(0.8, 0.5), np.hypot(0.6, 0.3)]]
    G = {}
    for mode in ("exact", "accuracy"):
        with pg.settings.using(images=mode):
            k = H.riesz(mus)
            G[mode] = np.asarray(H.inner(k, k))
            claim = [str(c) for c in H.claims if "reflection images" in str(c)][0]
            assert f"settings.images = '{mode}'" in claim
    R, tau, bound, _ = H.backend.engine.images.cut(H.backend.kernel)
    with pg.settings.using(images="accuracy"):
        R, tau, bound, _ = H.backend.engine.images.cut(H.backend.kernel)
    assert tau == pytest.approx(0.01 * 0.05) and bound <= tau
    k0 = H.backend.kernel.k0
    assert np.all(np.abs(G["accuracy"] - G["exact"]) <= bound * k0 * np.outer(mass, mass))
    ref = series(2.5, 0.15, sides, conds, P, P)
    assert np.allclose(np.diag(G["accuracy"])[:4], ref, rtol=2e-6 + bound * k0 / ref.min())
    assert np.allclose(np.diag(G["exact"])[:4], ref, rtol=2e-6)


@needs_finufft
@pytest.mark.parametrize("d, dirichlet", [(2, ()), (2, "all"), (2, ("x-",)), (3, ("y-", "z+"))])
def test_spectral_products_of_image_sums_against_the_exact_route(d, dirichlet):
    H, sides = space(dirichlet, d=d)
    k = H.riesz(chords(d, 10, sides))
    A, energies = k.realisation.kx.atoms, H.backend.engine
    E = energies.exact.energy(A, A)
    A.self_energy = None
    V = np.random.default_rng(3).standard_normal((A.n, 4))
    with pg.settings.using(engine="spectral", dense_max=1):
        name, eng, why = energies.route(A)
        assert name == "spectral" and "signed copies" in why
        W = energies.apply(A, V)
        est = eng.estimate(A, V)
        assert "signed copies of every segment" in energies.describe(A)
        mixed = [dirichlet != "all" and ((f"{a}-" in dirichlet) != (f"{a}+" in dirichlet)) for a in "xyz"[:d]]
        assert len(eng.G0) == np.prod([4 if m else 2 for m in mixed])
    err = np.linalg.norm(W - E @ V, axis=0) / np.linalg.norm(E @ V, axis=0)
    assert err.max() < 3e-3 and err.max() <= 2 * est             # the estimate on V itself (not a bound)


@needs_finufft
def test_spectral_route_for_images_falls_back_for_points():
    H, sides = space(())
    k = H.riesz([pg.DiracMeasure((0.2, 0.3)), pg.DiracMeasure((0.5, 0.5))])
    A = k.realisation.kx.atoms
    with pg.settings.using(engine="spectral", dense_max=1):
        name, _, why = H.backend.engine.route(A)
    assert name == "hmatrix" and "atoms other than segments" in why


def test_copies_of_mixed_axes_are_antiperiodic():
    from pygeoinf0.realisations.rkhs.spectral_images import copies
    refl = green.Reflections([0.0, 0.0], [1.0, 0.5], [-1.0, 1.0], [1.0, 1.0])     # x: Dirichlet-Neumann; y: Neumann
    G0, P, origin = copies(refl)
    assert len(G0) == 8 and np.allclose(P, [4.0, 1.0]) and np.allclose(origin, [-1.0, -0.5])
    assert sorted(g[0] for g in G0) == [-1.0] * 4 + [1.0] * 4
    assert math.isclose(sum(g[0] for g in G0), 0.0)
