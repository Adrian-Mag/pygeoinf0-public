"""Restriction of realised elements on the realisation (plan 043): a cut is exact and calls nothing, and every other case is
fitted as before. Sources of several origins: fits, combinations, products, piecewise-linear tables, families."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.operators.pieces import _cut
from pygeoinf0.realisations import Cells
from pygeoinf0.realisations.series.radial_direct import harmonic_element

rng = np.random.default_rng(43)


def realised(X, f):
    """An element with no formula to fall back on: only its realisation."""
    u = 1.0 * X.element(f) + 0.0 * X.element(f)
    u.realisation
    assert u._src is None
    return u


def ball_points(a, b, n=60):
    r, th, ph = rng.uniform(a, b, n), rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n)
    return r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)


def box_points(D, n=60):
    return tuple(rng.uniform(s.a, s.b, n) for s in D.sides)


def sphere_points(t0, t1, n=60):
    th, ph = rng.uniform(t0, t1, n), rng.uniform(0, 2 * np.pi, n)
    return np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)


smooth = lambda x: np.sin(3 * x) + x ** 2
kink = lambda x: np.abs(x - 0.5) + np.sin(x)
box_f = lambda x, y, z: np.sin(x) * np.cos(y) + z * x
ball_f = lambda x, y, z: np.exp(z) * x + y ** 2
sphere_f = lambda x, y, z: z * x + y
BOX, PIECE = pg.Box([(0, 1), (0, 2), (0, 1)]), pg.Box([(0.2, 0.7), (0, 2), (0.3, 1)])
BALL, SHELL = pg.Ball(1.0), pg.Shell(0.4, 0.8)
J, K = pg.Interval(0, 1), pg.Interval(0.2, 0.73)

CUTS = {
    "interval": (pg.L2(J), K, smooth, (rng.uniform(0.2, 0.73, 60),)),
    "a break inside": (pg.L2(J), K, kink, (rng.uniform(0.2, 0.73, 60),)),
    "density": (pg.L2(pg.DensityMeasure(pg.LebesgueMeasure(J), lambda x: 1.0 + x)), K, smooth, (rng.uniform(0.2, 0.73, 60),)),
    "Lp": (pg.LpSpace(pg.LebesgueMeasure(J), 3.0), K, smooth, (rng.uniform(0.2, 0.73, 60),)),
    "box local": (pg.L2(BOX, realisation="local"), PIECE, box_f, box_points(PIECE)),
    "box tensor": (pg.L2(BOX, realisation="tensor"), PIECE, box_f, box_points(PIECE)),
    "box legendre": (pg.L2(BOX, realisation="legendre"), PIECE, box_f, box_points(PIECE)),
    "ball tensor": (pg.L2(BALL), SHELL, ball_f, ball_points(0.4, 0.8)),
    "ball legendre": (pg.L2(BALL, realisation="legendre"), SHELL, ball_f, ball_points(0.4, 0.8)),
    "ball harmonics": (pg.L2(BALL, realisation="radial_harmonics"), SHELL, ball_f, ball_points(0.4, 0.8)),
    "smaller ball": (pg.L2(BALL, realisation="radial_harmonics"), pg.Ball(0.8), ball_f, ball_points(0.0, 0.8)),
    "shell to shell": (pg.L2(pg.Shell(0.2, 1.0), realisation="radial_harmonics"), SHELL, ball_f, ball_points(0.4, 0.8)),
    "sphere tensor": (pg.L2(pg.Sphere(1.0), realisation="tensor"), pg.Sphere(1.0).band(0.6, 1.4), sphere_f, sphere_points(0.6, 1.4)),
}


@pytest.mark.parametrize("case", list(CUTS))
def test_a_cut_is_the_source_on_the_piece(case):
    X, D, f, pts = CUTS[case]
    u = realised(X, f)
    R = pg.Restrict(X, D)
    assert _cut(X, R.codomain, u.realisation) is not None
    v = R(u)
    assert v._src is None and v._pl is not None                # no formula: nothing to refit
    assert np.asarray(v(*pts), float) == pytest.approx(np.asarray(u(*pts), float), abs=1e-12)
    assert v.norm() == pytest.approx(R.codomain.element(f).norm(), rel=1e-11)
    assert np.all(R.codomain.backend.error(v.realisation) <= X.backend.error(u.realisation) * (1 + 1e-12))


@pytest.mark.parametrize("case", ["interval", "ball harmonics"])
def test_a_product_of_realisations_is_cut(case):
    X, D, f, pts = CUTS[case]
    u = X.element(f) * X.element(f)
    u.realisation
    v = pg.Restrict(X, D)(u)
    assert v._src is None or u._src is not None
    assert np.asarray(v(*pts), float) == pytest.approx(np.asarray(f(*pts), float) ** 2, abs=1e-11)


def test_a_cut_calls_no_evaluation_of_the_source(monkeypatch):
    X, D, f, _ = CUTS["ball harmonics"]
    u = realised(X, f)
    R = pg.Restrict(X, D)
    calls = []
    original = type(X.backend).evaluate
    monkeypatch.setattr(type(X.backend), "evaluate", lambda *a: calls.append(1) or original(*a))
    fit = type(X.backend).fit
    monkeypatch.setattr(type(X.backend), "fit", lambda *a, **k: calls.append(1) or fit(*a, **k))
    R(u).realisation
    assert not calls


def test_the_parts_of_a_partition_add_up_and_glue_back():
    planet = pg.LayeredBall([0.3, 0.55, 1.0], names=["core", "mid", "top"])
    P = planet.partition
    X = pg.L2(BALL, realisation="radial_harmonics")
    u = realised(X, ball_f)
    parts = X.restrictions(P)(u)
    S = X.restrictions(P).codomain
    assert sum(S.part(parts, k).norm() ** 2 for k in S.names) == pytest.approx(u.norm() ** 2, rel=1e-12)
    pts = ball_points(0.0, 1.0)
    assert np.asarray(P.glue(X)(parts)(*pts), float) == pytest.approx(np.asarray(u(*pts), float), abs=1e-10)


def test_on_pieces_split_uses_the_cut():
    P = pg.LayeredBall([0.3, 0.55, 1.0], names=["core", "mid", "top"]).partition
    M = pg.PiecewiseSpace(P, {"a": pg.L2(BALL, realisation="radial_harmonics")})
    x = M.split({"a": realised(M.fields["a"], ball_f)})
    for name in M.names:
        assert M.part(x, name)._src is None


def test_a_table_cut_inside_a_linear_segment_stays_on_the_segment():
    """A piecewise-linear table, cut between its nodes: the straddling segment is re-expressed, not dropped."""
    X = pg.L2(BALL, realisation="radial_harmonics")
    radii = np.array([0.1, 0.5, 0.9])
    coef = np.zeros((3, 9))
    coef[:, 0] = [1.0, 3.0, -1.0]
    coef[:, 5] = [0.5, 0.0, 2.0]
    u = harmonic_element(X, radii, coef)
    D = pg.Shell(0.3, 0.7)
    v = pg.Restrict(X, D)(u)
    pts = ball_points(0.3, 0.7)
    assert np.asarray(v(*pts), float) == pytest.approx(np.asarray(u(*pts), float), abs=1e-13)
    assert len(pg.Restrict(X, D).codomain.backend.breaks(v.realisation)) == 3            # 0.3, 0.5, 0.7


@pytest.mark.parametrize("case", ["interval", "box tensor", "ball tensor", "ball harmonics"])
def test_a_family_is_cut_as_its_members(case):
    X, D, f, pts = CUTS[case]
    F = X.family([X.element(lambda *a, c=c: c * np.asarray(f(*a), float) + c ** 2) for c in (1.0, 2.0, 3.0)])
    R = pg.Restrict(X, D)
    assert _cut(X, R.codomain, F.realisation) is not None
    G = R(F)
    for i in range(3):
        assert np.asarray(G[i](*pts), float) == pytest.approx(np.asarray(F[i](*pts), float), abs=1e-11)


def test_a_separable_family_stays_separable():
    from pygeoinf0.realisations.series.radial_harmonic import SeparableRadialHarmonics
    X = pg.L2(BALL, realisation="radial_harmonics")
    from pygeoinf0.realisations.series import legendre as LG
    pl, _ = LG.fit(lambda r: np.c_[np.sin(3 * r), r ** 2], 0.0, 1.0, N=2)
    from pygeoinf0.spaces.elements import Family
    F = Family(X, SeparableRadialHarmonics(pl, [0, 4, 5], [0, 1, 1]), name="k")
    G = pg.Restrict(X, SHELL)(F)
    assert isinstance(G.realisation, SeparableRadialHarmonics) and list(G.realisation.lm) == [0, 4, 5]
    pts = ball_points(0.4, 0.8)
    for i in range(3):
        assert np.asarray(G[i](*pts), float) == pytest.approx(np.asarray(F[i](*pts), float), abs=1e-13)


def test_a_piece_with_another_realisation_is_fitted_as_it_asked():
    X = pg.L2(BALL, realisation="radial_harmonics")                # this route raised IndexError before plan 043
    u = realised(X, ball_f)
    R = pg.Restrict(X, SHELL, realisation="tensor")
    assert _cut(X, R.codomain, u.realisation) is None and R.codomain.realisation == "tensor"
    pts = ball_points(0.4, 0.8)
    assert np.asarray(R(u)(*pts), float) == pytest.approx(np.asarray(u(*pts), float), abs=1e-10)


def test_harmonics_to_a_band_are_fitted():
    S = pg.Sphere(1.0)
    X = pg.L2(S)
    u = realised(X, sphere_f)
    R = pg.Restrict(X, S.band(0.6, 1.4))
    assert _cut(X, R.codomain, u.realisation) is None
    pts = sphere_points(0.6, 1.4)
    assert np.asarray(R(u)(*pts), float) == pytest.approx(np.asarray(u(*pts), float), abs=1e-11)


def test_another_chart_is_not_cut():
    from pygeoinf0.realisations.series.cuts import same_chart
    assert not same_chart(pg.Ball(1.0), pg.Ball(0.5, centre=(0.1, 0.0, 0.0)))
    assert not same_chart(pg.Ball(0.5), pg.Ball(1.0))
    X = pg.L2(BALL, realisation="radial_harmonics")
    off = pg.L2(pg.Ball(0.5, centre=(0.1, 0.0, 0.0)), realisation="radial_harmonics")
    assert X.backend.restricted(realised(X, ball_f).realisation, off.backend) is None


def test_cells_are_cut_when_whole_cells_tile_the_piece():
    n = np.linspace(0.0, 1.0, 9)
    cells = Cells.tensor(n, n)
    X = pg.L2(pg.LebesgueMeasure(pg.Box([(0, 1), (0, 1)])), realisation=cells)
    f = lambda x, y: x + 2 * y
    u = 1.0 * X.element(f) + 0.0 * X.element(f)
    D = pg.Box([(0.25, 0.75), (0, 0.5)])
    R = pg.Restrict(X, D)
    v = R(u)
    assert v._src is None and v.realisation.V.shape[0] == 16
    pts = (np.array([0.3, 0.6]), np.array([0.1, 0.45]))
    assert np.asarray(v(*pts), float) == pytest.approx(np.asarray(u(*pts), float), abs=1e-14)
    with pytest.raises(pg.Refused, match="cells"):
        pg.Restrict(X, pg.Box([(0.3, 0.7), (0, 0.5)]))


def test_the_claim_says_cut():
    R = pg.Restrict(pg.L2(J), K)
    assert "cut exactly" in R.claims.find("realisation").statement


# the mirror: extension and gluing join realised parts (plan 043 P2) ----------------------------------------------------
def test_glue_joins_the_coefficients_and_is_the_inverse():
    P = pg.LayeredBall([0.3, 0.55, 1.0], names=["core", "mid", "top"]).partition
    X = pg.L2(BALL, realisation="radial_harmonics")
    u = realised(X, ball_f)
    g = P.glue(X)(X.restrictions(P)(u))
    assert g._src is None and g._pl is not None
    pts = ball_points(0.0, 1.0)
    assert np.asarray(g(*pts), float) == pytest.approx(np.asarray(u(*pts), float), abs=1e-13)
    assert g.norm() == pytest.approx(u.norm(), rel=1e-12)


def test_a_jump_at_a_seam_stays_a_jump_and_no_part_means_zero():
    P = pg.LayeredBall([0.3, 0.55, 1.0], names=["core", "mid", "top"]).partition
    X = pg.L2(BALL, realisation="radial_harmonics")
    M = pg.PiecewiseSpace(P, {"a": X}, where={"a": ["core", "top"]})
    core, top = M.summand(M.key("a", "core")), M.summand(M.key("a", "top"))
    c = np.zeros((2, 9))
    c[:, 0] = 1.0
    x = M({M.key("a", "core"): harmonic_element(core, [0.0, 0.3], c), M.key("a", "top"): harmonic_element(top, [0.55, 1.0], 3.0 * c)})
    g = M.glue(x, "a")
    assert g._src is None
    z = np.array([0.0, 0.0, 0.0])
    y = 1.0 / np.sqrt(4 * np.pi)
    assert np.asarray(g(np.array([0.29]), z[:1], z[:1]), float) == pytest.approx(y, rel=1e-13)
    assert np.asarray(g(np.array([0.4]), z[:1], z[:1]), float) == pytest.approx(0.0, abs=1e-15)         # no part: zero
    assert np.asarray(g(np.array([0.56]), z[:1], z[:1]), float) == pytest.approx(3 * y, rel=1e-13)
    energy = (0.3 ** 3 / 3.0) + 9.0 * (1.0 - 0.55 ** 3) / 3.0
    assert g.norm() ** 2 == pytest.approx(energy, rel=1e-12)


def test_parts_with_different_degrees_are_padded():
    X = pg.L2(BALL, realisation="radial_harmonics")
    a, b = pg.L2(pg.Ball(0.5), realisation="radial_harmonics"), pg.L2(pg.Shell(0.5, 1.0), realisation="radial_harmonics")
    low, high = a.element(lambda x, y, z: 1.0 + 0.0 * x), b.element(lambda x, y, z: x * y * z + np.exp(z))
    low, high = 1.0 * low + 0.0 * low, 1.0 * high + 0.0 * high
    assert low.realisation.L != high.realisation.L
    g = pg.Restrict(X, pg.Ball(0.5)).extension(low) + pg.Restrict(X, pg.Shell(0.5, 1.0)).extension(high)
    pts = ball_points(0.5, 1.0)
    assert np.asarray(g(*pts), float) == pytest.approx(np.asarray(high(*pts), float), abs=1e-12)


@pytest.mark.parametrize("case", ["interval", "a break inside", "density", "ball tensor", "ball harmonics", "sphere tensor"])
def test_the_extension_is_the_adjoint_of_the_cut(case):
    X, D, f, _ = CUTS[case]
    R = pg.Restrict(X, D)
    u = realised(X, f)
    v = R(u)
    Ev = R.extension(v)
    assert Ev._src is None
    assert X.inner(u, Ev) == pytest.approx(R.codomain.inner(R(u), v), rel=1e-11)
    assert Ev.norm() == pytest.approx(v.norm(), rel=1e-11)


def test_cells_glue_back_and_a_box_cut_on_two_axes_is_a_formula():
    n = np.linspace(0.0, 1.0, 9)
    X = pg.L2(pg.LebesgueMeasure(pg.Box([(0, 1), (0, 1)])), realisation=Cells.tensor(n, n))
    f = lambda x, y: x + 2 * y
    u = 1.0 * X.element(f) + 0.0 * X.element(f)
    R = pg.Restrict(X, pg.Box([(0.25, 0.75), (0, 0.5)]))
    Ev = R.extension(R(u))
    assert Ev._src is None and Ev.realisation.V.shape[0] == 64 and Ev.norm() == pytest.approx(R(u).norm(), rel=1e-13)
    Y = pg.L2(pg.Box([(0, 1), (0, 1), (0, 1)]), realisation="tensor")
    one = pg.Restrict(Y, pg.Box([(0, 1), (0, 1), (0.3, 0.8)]))
    w = realised(Y, box_f)
    joined = one.extension(one(w))
    assert joined._src is None and Y.inner(w, joined) == pytest.approx(one(w).norm() ** 2, rel=1e-11)
    R3 = pg.Restrict(Y, pg.Box([(0.2, 0.7), (0, 1), (0.3, 1)]))
    assert R3.extension(R3(w))._src is not None                 # two axes differ: the piecewise formula, as before


def test_a_formula_part_keeps_its_formula():
    R = pg.Restrict(pg.L2(J), K)
    assert R.extension(R.codomain.element(smooth))._src is not None
