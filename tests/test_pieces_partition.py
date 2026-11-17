"""Plan 035 P3: partitions (tiling, seams, refinement), restriction of elements to pieces, extension by zero, the glue,
and (field, piece) sums. References: the whole-domain norms and pairings, P1's T(vp) numbers (independent quadrature
about the bump's axis, scratch t3.py), closed forms on intervals."""
import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.core.errors import SupportOutsideDomain, Unsupported

VP = lambda x, y, z: 1.0 + z + 0.5 * x * y          # noqa: E731
VS = lambda x, y, z: 3.0 - x                          # noqa: E731
U = np.array([0.3, 0.2, 0.7]) / np.linalg.norm([0.3, 0.2, 0.7])


@pytest.fixture(scope="module")
def planet():
    P = pg.LayeredBall([0.55, 0.8, 1.0], names=["a", "b", "c"])
    X = pg.L2(P.ball)
    return P, P.partition, X


# ---------------------------------------------------------------------------------------------- tiling
def test_tilings_are_decided_and_own_their_seams():
    I = pg.Interval("[0, 1]")
    P = pg.Partition(I, ["[0.4, 1]", "[0, 0.4]"], names=["r", "l"])
    assert P.order == ("l", "r") and [(s.below, s.above, s.at) for s in P.seams] == [("l", "r", 0.4)]
    assert P.claims.find("tiling").status == "derived"
    B = pg.Box((0, 1), (0, 2))
    PB = pg.Partition(B, [pg.Box((0, 0.5), (0, 2)), pg.Box((0.5, 1), (0, 1)), pg.Box((0.5, 1), (1, 2))],
                      names=["w", "se", "ne"])
    assert {(s.below, s.above, s.at) for s in PB.seams} == {("w", "se", (0, 0.5)), ("w", "ne", (0, 0.5)),
                                                            ("se", "ne", (1, 1.0))}
    S = pg.Shell(0.3, 1.0)
    PS = pg.Partition(S, {"lo": pg.Shell(0.3, 0.6), "hi": pg.Shell(0.6, 1.0)})
    assert PS.seam("hi", "lo").set == pg.Sphere(0.6)


@pytest.mark.parametrize("pieces, word", [(["[0, 0.4]", "[0.5, 1]"], "gap"), (["[0, 0.6]", "[0.4, 1]"], "overlap"),
                                          (["[0, 0.4]"], "gap"), (["[0, 0.4]", "[0.4, 1.2]"], "leaves")])
def test_gaps_overlaps_and_pieces_outside_are_refused(pieces, word):
    with pytest.raises(pg.NotATiling, match=word) as e:
        pg.Partition("[0, 1]", pieces)
    assert e.value.needs and isinstance(e.value, ValueError)


def test_boxes_and_balls_refuse_gaps_overlaps_and_what_is_not_decided():
    B = pg.Box((0, 1), (0, 2))
    with pytest.raises(pg.NotATiling, match="gap"):
        pg.Partition(B, [pg.Box((0, 0.5), (0, 2)), pg.Box((0.5, 1), (0, 1))])
    with pytest.raises(pg.NotATiling, match="overlap"):
        pg.Partition(B, [pg.Box((0, 0.6), (0, 2)), pg.Box((0.5, 1), (0, 2))])
    with pytest.raises(pg.NotATiling, match="gap"):
        pg.Partition(pg.Ball(1.0), [pg.Ball(0.5), pg.Shell(0.6, 1.0)])
    with pytest.raises(Unsupported, match="concentric"):
        pg.Partition(pg.Ball(1.0), [pg.Ball(0.5, (0.1, 0, 0)), pg.Shell(0.5, 1.0)])
    with pytest.raises(Unsupported, match="not decided"):
        pg.Partition(pg.Disk(1.0), [pg.Disk(0.5)])
    with pytest.raises(Unsupported, match="not decided"):
        pg.Partition(pg.Ball(1.0), [pg.Box((0, 1), (0, 1), (0, 1))])


def test_refinement_is_a_tree_and_flattens_to_the_leaves(planet):
    P, Q, _ = planet
    T = Q.refine("b", [pg.Shell(0.55, 0.7), pg.Shell(0.7, 0.8)], names=["b1", "b2"])
    assert Q.is_flat and Q.names == ("a", "b", "c") and not T.is_flat         # Q is not changed
    assert T["b1"] == pg.Shell(0.55, 0.7) and T.names == ("a", "b", "c")
    F_ = T.flat()
    assert F_.names == ("a", "b1", "b2", "c") and [s.at for s in F_.seams] == [0.55, 0.7, 0.8]
    with pytest.raises(pg.NotATiling):
        Q.refine("b", [pg.Shell(0.55, 0.7)])
    with pytest.raises(ValueError, match="twice"):
        Q.refine("b", [pg.Shell(0.55, 0.7), pg.Shell(0.7, 0.8)], names=["a", "b2"]).flat()


def test_layered_ball_exposes_its_radial_partition_and_split_agrees(planet):
    P, Q, _ = planet
    assert Q.domain == P.ball and Q.names == P.names and [s.set for s in Q.seams] == list(P.interfaces)
    bump = F.on(P.ball).bump(centre=tuple(0.675 * U), radius=0.2)
    mu = bump.measure()
    a, b = Q.split(mu), P.split(mu)
    assert list(a) == list(b) == ["a", "b", "c"] and all(a[k] == b[k] for k in a)
    assert a.claims.find("split").status == "derived"
    seam = pg.DiracMeasure([0, 0, 0.8])
    with pytest.raises(ValueError, match="side="):
        Q.split(seam)
    assert list(Q.split(seam, side="above")) == ["c"] and list(Q.split(seam, side="below")) == ["b"]
    assert list(Q.split(pg.DiracMeasure([0, 0, 1.0]))) == ["c"]                 # the outer surface: one piece


# ---------------------------------------------------------------------------------------------- L2 on a ball
def test_restrict_and_glue_are_inverse_on_l2_of_a_ball(planet):
    P, Q, X = planet
    R, G = X.restrictions(Q), Q.glue(X)
    assert R.adjoint is G and G.adjoint is R and R.claims.find("isometry").status == "theorem"
    u = X(VP)
    x = R(u)
    assert sum(R.codomain.part(x, n).norm() ** 2 for n in Q.names) == pytest.approx(u.norm() ** 2, rel=1e-13)
    assert (G(x) - u).norm() < 1e-12 * u.norm()
    w = u * X(lambda x, y, z: np.exp(x))                                         # a realised element
    assert w._pl is not None and w._src is None
    assert (G(R(w)) - w).norm() < 1e-12 * w.norm()
    y = R.codomain({"a": VS, "b": VP, "c": lambda x, y, z: x * z})              # jumps at both seams
    assert (R(G(y)) - y).norm() < 1e-12 * R.codomain.norm(y)
    pts = np.array([[0.1, 0.2, 0.3], [0.0, 0.6, 0.1], [0.5, 0.5, 0.5]])
    assert np.allclose(G(y)(pts), [VS(*pts[0]), VP(*pts[1]), pts[2, 0] * pts[2, 2]], rtol=1e-12)


def test_the_adjoint_of_a_restriction_is_extension_by_zero(planet):
    P, Q, X = planet
    R = X.restrict_to(P["b"])
    assert R.codomain == pg.L2(pg.Shell(0.55, 0.8)) and X == pg.L2(P.ball)       # a new space; X is unchanged
    assert R.claims.find("adjoint").status == "theorem"
    u, v = X(VP), R.codomain(VS)
    assert R.codomain.inner(R(u), v) == pytest.approx(X.inner(u, R.adjoint(v)), rel=1e-13)
    Ev = R.adjoint(v)
    assert Ev.norm() == pytest.approx(v.norm(), rel=1e-13)
    assert Ev([0, 0, 0.3]) == 0.0 and Ev([0, 0, 0.9]) == 0.0 and Ev([0.6, 0, 0]) == pytest.approx(VS(0.6, 0, 0))
    assert (R(Ev) - v).norm() < 1e-12 * v.norm()                               # R R* = I


@pytest.mark.filterwarnings("ignore:the L2 pairing")       # X.norm(k): pg.Ball's sub-ball rule (preexisting, P1 notes)
def test_symbolic_representers_restrict_exactly(planet):
    P, Q, X = planet
    R, G = X.restrictions(Q), Q.glue(X)
    bump = F.on(P.ball).bump(centre=tuple(0.675 * U), radius=0.2).normalised()
    k = X.riesz(bump.measure())
    xk = R(k)
    f = X(VP)
    ref = 1.62212197104364                                                      # P1, independent quadrature
    assert R.codomain.inner(xk, R(f)) == pytest.approx(ref, rel=1e-13)
    assert X.inner(G(xk), f) == pytest.approx(ref, rel=1e-13)
    assert R.codomain.norm(xk) == pytest.approx(X.norm(k), rel=1e-6)            # pg.Ball's sub-ball rule: ~1e-8


# ---------------------------------------------------------------------------------------------- weighted L2, L^p
@pytest.fixture(scope="module")
def weighted():
    I = pg.Interval("[0, 1]")
    W = pg.L2(pg.LebesgueMeasure(I).with_density(lambda x: 1 + x ** 2))
    return W, pg.Partition(I, {"l": "[0, 0.4]", "r": "[0.4, 1]"})


def test_weighted_l2_restricts_its_weight_and_tiles(weighted):
    W, P = weighted
    R, G = W.restrictions(P), P.glue(W)
    Wl = R.codomain.summand("l")
    assert isinstance(Wl.measure, pg.DensityMeasure) and Wl.measure.w is W.measure.w
    u = W(np.sin)
    # int_0^1 sin^2 (1 + x^2) dx in closed form
    exact = 0.5 + 1 / 6 - (np.sin(2) / 4 + (2 * np.sin(2) + 2 * np.cos(2) - np.sin(2)) / 8)
    assert u.norm() ** 2 == pytest.approx(exact, rel=1e-13)
    assert R.codomain.norm(R(u)) ** 2 == pytest.approx(exact, rel=1e-13)
    v = R.codomain({"l": np.cos, "r": lambda t: t ** 2})
    assert R.codomain.inner(R(u), v) == pytest.approx(W.inner(u, G(v)), rel=1e-13)
    assert (G(R(u)) - u).norm() < 1e-13 and (R(G(v)) - v).norm() < 1e-13
    ur = W(np.sin) * W(np.exp)
    assert (G(R(ur)) - ur).norm() < 1e-13 * ur.norm()


def test_lp_restriction_adds_pth_powers_and_refuses_what_needs_dual_spaces():
    I = pg.Interval("[0, 1]")
    L3 = pg.Lp(I, 3)
    Rl, Rr = L3.restrict_to("[0, 0.4]"), L3.restrict_to("[0.4, 1]")
    g = L3(np.sin)
    assert Rl(g).norm() ** 3 + Rr(g).norm() ** 3 == pytest.approx(g.norm() ** 3, rel=1e-12)
    assert "L^1.5(D)" in Rl.claims.find("adjoint").statement
    with pytest.raises(pg.Refused, match="dual") as e:
        Rl.adjoint
    assert any("extension" in n for n in e.value.needs)
    with pytest.raises(pg.Refused, match="l\\^p") as e:
        L3.restrictions(pg.Partition(I, ["[0, 0.4]", "[0.4, 1]"]))
    E = Rl.extension
    assert E(Rl(g)).norm() == pytest.approx(Rl(g).norm(), rel=1e-12) and E(Rl(g))(0.7) == 0.0


def test_boxes_tile_in_l2():
    B = pg.Box((0, 1), (0, 2))
    P = pg.Partition(B, [pg.Box((0, 0.5), (0, 2)), pg.Box((0.5, 1), (0, 1)), pg.Box((0.5, 1), (1, 2))])
    X = pg.L2(B)
    u = X(lambda x, y: np.sin(x) * y)
    R = X.restrictions(P)
    assert R.codomain.norm(R(u)) ** 2 == pytest.approx(u.norm() ** 2, rel=1e-13)
    assert (P.glue(X)(R(u)) - u).norm() < 1e-12


# ---------------------------------------------------------------------------------------------- refusals by name
def test_sobolev_continuous_products_and_sums_are_refused_by_name(planet):
    P, Q, X = planet
    H = pg.SobolevForm("[0, 1]", 1, length_scale=0.1).space()          # Sobolev pieces: P4 (test_pieces_sobolev.py)
    with pytest.raises(pg.Refused) as e:
        H.restrict_to("[0, 0.5]")
    assert any("glued" in n for n in e.value.needs)
    C = pg.Continuous("[0, 1]")                                          # C(K) pieces: P5 (test_pieces_p5.py)
    assert C.restrict_to("[0, 0.5]").codomain == pg.Continuous("[0, 0.5]")
    with pytest.raises(pg.Refused) as e:
        C.restrictions(pg.Partition("[0, 1]", ["[0, 0.5]", "[0.5, 1]"]))
    assert any("max norm" in n for n in e.value.needs)
    with pytest.raises(pg.Refused):
        pg.Restrict(pg.DirectSum({"a": X}), P["a"])
    with pytest.raises(SupportOutsideDomain):
        pg.Restrict(pg.L2(P["a"]), P["c"])
    with pytest.raises(SupportOutsideDomain):
        pg.Restrict(pg.L2(P["b"]), pg.Shell(0.7, 0.9))


# ---------------------------------------------------------------------------------------------- (field, piece) sums
def test_on_pieces_reproduces_the_target_on_the_vp_pieces(planet):
    P, Q, X = planet
    M = pg.PiecewiseSpace(Q, {"vp": pg.L2, "vs": pg.L2})
    assert M.names == ("vp_a", "vp_b", "vp_c", "vs_a", "vs_b", "vs_c") and M.key("vs", "b") == "vs_b"
    x = M.split({"vp": VP, "vs": VS})
    for c, rho, ref in [((0, 0, 0.55), 0.15, 1.55), ((0, 0, 0.675), 0.2, 1.675),
                        (tuple(0.8 * U), 0.15, 1.74216845313656), (tuple(0.675 * U), 0.2, 1.62212197104364)]:
        g = F.on(P.ball).bump(centre=c, radius=rho).normalised()
        assert M.functional("vp", g.measure())(x) == pytest.approx(ref, rel=2e-14)
    g = F.on(P.ball).bump(centre=tuple(0.675 * U), radius=0.2).normalised()
    assert M.functional("vp", g.measure())(M.split({"vs": VS})) == 0.0
    mus = [g.measure(), F.on(P.ball).bump(centre=(0, 0, 0.55), radius=0.15).normalised().measure()]
    T = M.functional("vp", mus)                 # a list: the bounded pg.Functionals into R^n, as on main (plan 071 P3)
    assert isinstance(T, pg.Functionals) and T.codomain.dim == 2
    assert np.allclose(T(x).values, [1.62212197104364, 1.55], rtol=2e-14)
    long = pg.Functionals(M.riesz([M.parts("vp", m) for m in mus]))
    assert np.array_equal(T(x).values, long(x).values)
    assert np.array_equal(T(x).values, [M.functional("vp", m)(x) for m in mus])
    assert isinstance(M.functional("vp", tuple(mus)), pg.Functionals)


def test_on_pieces_split_and_glue_round_trip_and_where(planet):
    P, Q, X = planet
    M = pg.PiecewiseSpace(Q, {"vp": X, "vs": pg.L2}, where={"vs": ["b", "c"]}, weights={"vp": 2.0, "vs_c": 3.0})
    assert M.names == ("vp_a", "vp_b", "vp_c", "vs_b", "vs_c")
    assert dict(zip(M.names, M.weights)) == {"vp_a": 2.0, "vp_b": 2.0, "vp_c": 2.0, "vs_b": 1.0, "vs_c": 3.0}
    u = X(VP) * X(lambda x, y, z: np.cos(y))
    x = M.split({"vp": u, "vs": VS})
    back = M.glue(x)
    assert (back["vp"] - u).norm() < 1e-12 * u.norm()
    assert back["vs"]([0, 0, 0.3]) == 0.0 and back["vs"]([0, 0, 0.9]) == pytest.approx(3.0)
    with pytest.raises(SupportOutsideDomain, match="vs"):
        M.functional("vs", F.on(P.ball).bump(centre=(0, 0, 0.55), radius=0.15).measure())


def test_covariance_glued_records_the_partition_it_tiles():
    lap = pg.Laplacian(pg.Ball(1.0), bc={"boundary": "dirichlet"})
    C = lap.bessel_potential(3.0, 2.0, degrees=[2], modes=8)
    V = pg.PiecewiseSpace.lebesgue(pg.LayeredBall([0.4, 1.0], names=("a", "b")), realisation="radial_harmonics")
    M = pg.DirectSum({"a": V.summand("a"), "b": V.summand("b")})
    K = pg.Covariance.glued(M, {("a", "b"): C})
    assert K.partitions[(0, 1)].order == ("a", "b")
    with pytest.raises(ValueError, match="in order"):
        pg.Covariance.glued(M, {("b", "a"): C})
    with pytest.raises(ValueError):
        pg.Covariance.glued(M, {("a",): C})
