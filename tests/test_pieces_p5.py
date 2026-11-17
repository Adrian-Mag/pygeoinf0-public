"""Plan 035 P5: C(K) on closed pieces (restriction, the max over pieces, the dual split with seam masses once), sphere
partitions into latitude bands and polar caps (exact tiling, L2 restriction, extension by zero, glue, measures cut at
the parallels, Sobolev on the sphere glued only), and (field, piece) sums of mixed kinds. References: closed forms on
the unit sphere and on intervals (P4's H^1_l energies)."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import SupportOutsideDomain, Unsupported
from pygeoinf0.measures import DensityMeasure, DiracMeasure, LebesgueMeasure, SignedMeasure, terms_of

F = lambda x, y, z: 1 + z + x * y / 2                       # noqa: E731
F_NORM2 = 81 * math.pi / 15                                  # int_S (1 + z + xy/2)^2 = 4 pi + 4 pi/3 + pi/15


@pytest.fixture(scope="module")
def line():
    C = pg.Continuous("[0, 1]")
    P = pg.Partition("[0, 1]", ["[0, 0.4]", "[0.4, 1]"], names=["left", "right"])
    return C, P, C.element(lambda x: np.sin(3 * x) + x ** 2)


@pytest.fixture(scope="module")
def globe():
    S = pg.Sphere(1.0)
    P = pg.Partition(S, {"north": S.cap(0.6), "belt": S.band(0.6, 2.0), "south": S.cap(2.0, pole="south")})
    return S, P


# ---------------------------------------------------------------------------------------------- C(K)
def test_continuous_restriction_norm_is_the_max_over_closed_pieces(line):
    C, P, u = line
    R = {n: C.restrict_to(P[n]) for n in P.names}
    assert R["left"].codomain == pg.Continuous("[0, 0.4]")
    norms = [R[n].codomain.norm(R[n](u)) for n in P.names]
    assert max(norms) == pytest.approx(C.norm(u), rel=1e-14) and norms[0] < norms[1]
    x = np.linspace(0, 0.4, 9)
    assert np.max(np.abs(R["left"](u)(x) - (np.sin(3 * x) + x ** 2))) < 1e-13
    cl = R["left"].claims
    for key in ("linear", "bounded", "pieces", "adjoint"):
        assert cl.find(key).status == "theorem"
    assert cl.find("realisation").status == "checked"


def test_continuous_dual_split_counts_a_seam_mass_once(line):
    C, P, u = line
    seam = DiracMeasure(0.4)
    with pytest.raises(ValueError, match="side="):
        P.functionals(C, seam)
    for side, piece in (("below", "left"), ("above", "right")):
        l = P.functionals(C, seam, side=side)
        assert list(l) == [piece] and l(u) == pytest.approx(u(0.4), rel=1e-13)
    lam = LebesgueMeasure(pg.Interval(0.2, 0.9))
    mu = SignedMeasure([(1.0, lam), (2.0, seam)])
    l = P.functionals(C, mu, side="above")
    assert list(l) == ["left", "right"] and l.claims.find("functionals").status == "theorem"
    assert l(u) == pytest.approx(C.functional(lam)(u) + 2 * u(0.4), rel=1e-13)
    R = C.restrict_to(P["left"])
    assert R.dual(DiracMeasure(0.1))(u) == pytest.approx(R.codomain.evaluation(0.1)(R(u)), rel=1e-14)
    with pytest.raises(SupportOutsideDomain):
        R.dual(DiracMeasure(0.7))


def test_continuous_refusals_by_name(line):
    C, P, u = line
    R = C.restrict_to(P["left"])
    for attr, need in (("extension", "Tietze"), ("adjoint", "R.dual")):
        with pytest.raises(pg.Refused) as e:
            getattr(R, attr)
        assert any(need in n for n in e.value.needs)
    for make in (lambda: C.restrictions(P), lambda: pg.PiecewiseSpace(P, {"c": C})):
        with pytest.raises(pg.Refused) as e:
            make()
        assert any("max norm" in n for n in e.value.needs)
    with pytest.raises(pg.Refused):
        P.functionals(pg.L2("[0, 1]"), DiracMeasure(0.2))
    with pytest.raises(ValueError, match="closed"):
        C.restrict_to("(0, 0.4)")


# ---------------------------------------------------------------------------------------------- sphere bands
def test_sphere_partitions_into_bands_are_decided_exactly(globe):
    S, P = globe
    assert [s.at for s in P.seams] == [0.6, 2.0] and (P.seam("north", "belt").below == "north")
    assert sum(P[n].volume for n in P.names) == pytest.approx(4 * math.pi, rel=1e-15)
    with pytest.raises(pg.NotATiling):
        pg.Partition(S, [S.cap(0.6), S.cap(2.0, pole="south")])                          # a gap
    with pytest.raises(pg.NotATiling):
        pg.Partition(S, [S.cap(0.7), S.cap(0.6, pole="south")])                          # an overlap
    with pytest.raises(Unsupported):
        pg.Partition(S, [S.cap(0.6), pg.Sphere(2.0).cap(0.6, pole="south")])             # another sphere
    Q = P.refine("belt", [S.band(0.6, 1.2), S.band(1.2, 2.0)], names=["b1", "b2"])
    assert Q.flat().names == ("north", "b1", "b2", "south")


@pytest.mark.parametrize("real", [None, "tensor"])
def test_sphere_l2_restriction_is_a_co_isometry_and_the_glue_is_unitary(globe, real):
    S, P = globe
    X = pg.L2(S, realisation=real)
    u, R = X.element(F), X.restrictions(P)
    assert R.parts["belt"].codomain.realisation == "tensor"
    assert sum(R.parts[n].codomain.norm(R.parts[n](u)) ** 2 for n in P.names) == pytest.approx(F_NORM2, rel=1e-14)
    v = R.codomain({"north": 1.0, "belt": lambda x, y, z: x, "south": -2.0})
    assert R.codomain.inner(R(u), v) == pytest.approx(X.inner(u, R.adjoint(v)), rel=1e-12)
    G = R.glue(R(u))
    pts = np.array(S.from_angles(np.array([0.3, 1.0, 2.5]), np.array([0.2, 1.0, 4.0])))
    assert np.max(np.abs(G(*pts) - F(*pts))) < 1e-13
    if real == "tensor":                              # a piecewise formula with the seams as breaks: exact
        assert (G - u).norm() < 1e-12
    else:                                             # harmonics take no breaks: the measures u dA on the bands, kept
        assert X.norm(G) ** 2 == pytest.approx(F_NORM2, rel=1e-14)          # symbolic (pairs exactly, bands meet in
        with pytest.raises(pg.Refused):                                      # null seams), not fitted across the seams
            G - u


def test_sphere_measures_restricted_to_bands(globe):
    S, P = globe
    M = pg.PiecewiseSpace(P, {"u": pg.L2})
    x = M.split({"u": F})
    a, b = 0.3, 1.5                                  # a band across the seam 0.6
    band = LebesgueMeasure(S.band(a, b))
    ref = 2 * math.pi * (math.cos(a) - math.cos(b)) + math.pi * (math.sin(b) ** 2 - math.sin(a) ** 2)
    assert set(M.parts("u", band)) == {"u_north", "u_belt"}
    assert M.functional("u", band)(x) == pytest.approx(ref, rel=1e-14)
    g = DensityMeasure(LebesgueMeasure(S), lambda x, y, z: 1 + x * y)
    assert M.functional("u", g)(x) == pytest.approx(4 * math.pi + 4 * math.pi / 30, rel=1e-14)
    seam = DiracMeasure(np.array(S.from_angles(0.6, 1.0)).ravel())
    with pytest.raises(ValueError, match="side="):
        P.split(seam)
    assert list(P.split(seam, side="below")) == ["north"] and list(P.split(seam, side="above")) == ["belt"]
    assert list(P.split(DiracMeasure(np.array([0.0, 0.0, 1.0])))) == ["north"]      # the pole: inside the cap


def test_great_circle_arcs_are_cut_at_the_parallels(globe):
    S, P = globe
    A = pg.Arc(S, np.array(S.from_angles(0.2, 0.3)).ravel(), np.array(S.from_angles(2.4, 1.2)).ravel())
    parts = P.split(LebesgueMeasure(A))
    assert list(parts) == ["north", "belt", "south"]
    assert sum(c * m.set.volume for p in parts.values() for c, m in terms_of(p)) == pytest.approx(A.volume, rel=1e-14)
    end = parts["north"].set.q
    assert P["north"].colatitude(*end) == pytest.approx(0.6, abs=1e-12)
    Q = pg.Partition(S, {"n": S.cap(math.pi / 2), "s": S.cap(math.pi / 2, pole="south")})
    equator = LebesgueMeasure(pg.Arc(S, np.array([1.0, 0, 0]), np.array([0, 1.0, 0])))
    with pytest.raises(ValueError, match="side="):
        Q.split(equator)
    assert list(Q.split(equator, side="above")) == ["s"]


def test_sobolev_on_the_sphere_is_glued_only(globe):
    S, P = globe
    H = pg.Sobolev(S, 2, length_scale=0.3)
    for make in (lambda: H.restrict_to(S.band(0.6, 2.0)), lambda: pg.PiecewiseSpace(P, {"u": H}, broken=True)):
        with pytest.raises(pg.Refused) as e:
            make()
        assert any("glued" in n for n in e.value.needs)
    M = pg.PiecewiseSpace(P, {"u": H, "v": pg.L2})
    assert M.names == ("u", "v_north", "v_belt", "v_south")
    A = LebesgueMeasure(pg.Arc(S, np.array(S.from_angles(0.2, 0.3)).ravel(),
                               np.array(S.from_angles(2.4, 1.2)).ravel()))
    k, r = H.riesz(A), M.riesz(M.parts("u", A))
    assert M.inner(r, r) == pytest.approx(H.inner(k, k), rel=1e-14)
    reps = [M.riesz(M.parts("u", m)) for m in P.split(A).values()]
    G = np.array([[M.inner(p, q) for q in reps] for p in reps])
    assert G.sum() == pytest.approx(H.inner(k, k), rel=1e-12) and np.trace(G) < 0.8 * G.sum()     # cross terms
    u = H.element(F)
    x = M.split({"u": u})
    assert sum(l(x) for l in M.piece_functionals("u", A).values()) == pytest.approx(H.inner(k, u), rel=1e-12)
    seam = DiracMeasure(np.array(S.from_angles(0.6, 1.0)).ravel())
    parts = M.parts("u", seam)                                     # no side= on the glued space
    assert parts.claims.find("seam sides").status == "theorem"
    assert M.functional("u", seam)(x) == pytest.approx(u(*seam.x0), rel=1e-10)


# ---------------------------------------------------------------------------------------------- mixed kinds
def test_on_pieces_with_fields_of_mixed_kinds_works_per_field():
    L = 0.2
    P = pg.Partition("[0, 1]", ["[0, 0.4]", "[0.4, 1]"], names=["a", "b"])
    H = pg.Sobolev("[0, 1]", 1, length_scale=L)
    M = pg.PiecewiseSpace(P, {"vp": pg.L2, "rho": H, "vs": H}, broken={"vs"})
    assert M.names == ("vp_a", "vp_b", "rho", "vs_a", "vs_b") and M.glued_fields == {"rho"}
    lam = LebesgueMeasure(pg.Interval(0.2, 0.6))
    reps = [M.riesz(M.parts(f, lam)) for f in M.fields]
    G = np.array([[M.inner(p, q) for q in reps] for p in reps])
    ref = [0.4, 0.4 - L * (1 - np.exp(-2)), 2 * (0.2 - L * (1 - np.exp(-1)))]   # L2; glued H^1; broken H^1 (P4)
    assert np.allclose(G, np.diag(ref), rtol=1e-13, atol=1e-15)
    x = M.split({"vp": lambda t: t, "rho": H.element(lambda t: np.sin(3 * t)), "vs": lambda t: 1.0 + 0 * t})
    got = [M.functional(f, lam)(x) for f in M.fields]
    assert np.allclose(got, [0.16, (np.cos(0.6) - np.cos(1.8)) / 3, 0.4], rtol=1e-13)
    seam = DiracMeasure(0.4)
    assert list(M.parts("rho", seam)) == ["rho"]                   # glued: side-free
    with pytest.raises(ValueError, match="side="):
        M.parts("vs", seam)                                       # broken: side= required
    assert list(M.parts("vs", seam, side="above")) == ["vs_b"]
    keys = {c.key for c in M.claims}
    assert {"on pieces", "glued", "broken", "additive"} <= keys
