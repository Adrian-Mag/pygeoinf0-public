"""Plan 069 P7 (review B9): one PiecewiseSpace(partition, fields, glued | broken) in spaces/ replaces spaces/layered.Layered
and operators/pieces.OnPieces, and X.restrictions(P) maps into its broken form. The laws (also in tests/test_laws.py,
with Hypothesis): X.restrictions(P) is an isometry of L2 onto the broken space, which for L2 is the glued one; the norm
of a broken space is the sum of its pieces' norms; elements of a glued H^s (1/2 < s < 3/2) have matching traces on the
seams, a broken one does not require them, and gluing a broken element checks them. References: closed forms of
polynomials on intervals, and the exponential kernel of H^1_l, K = e^(-|x-y|/l) / (2l)."""
import importlib

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.logic import Truth
from pygeoinf0.measures import DiracMeasure, LebesgueMeasure

L = 0.2
U = lambda x: 1.0 + x - 2.0 * x ** 2 + 0.5 * x ** 3          # noqa: E731


@pytest.fixture(scope="module")
def line():
    P = pg.Partition("[0, 1]", ["[0, 0.3]", "[0.3, 0.7]", "[0.7, 1]"], names=["a", "b", "c"])
    return P, pg.L2("[0, 1]"), pg.Sobolev("[0, 1]", 1, length_scale=L)


# ---------------------------------------------------------------------------------------------- the clean break
def test_one_class_replaces_layered_and_on_pieces():
    assert isinstance(pg.PiecewiseSpace, type) and issubclass(pg.PiecewiseSpace, pg.DirectSum)
    assert not hasattr(pg, "Layered") and not hasattr(pg, "OnPieces")
    assert hasattr(pg, "LayeredBall")                                       # the domain stays
    with pytest.raises(ImportError):
        importlib.import_module("pygeoinf0.spaces.layered")
    from pygeoinf0.operators import pieces
    assert not hasattr(pieces, "OnPieces")
    from pygeoinf0.spaces import piecewise
    assert piecewise.PiecewiseSpace is pg.PiecewiseSpace


def test_a_layered_ball_gives_its_radial_partition_and_no_planet_vocabulary():
    B = pg.LayeredBall([0.3, 0.6, 1.0], names=["core", "mantle", "crust"])
    V, W = pg.PiecewiseSpace.lebesgue(B), pg.PiecewiseSpace.lebesgue(B.partition)
    assert V == W and V.partition == B.partition and V.names == ("core", "mantle", "crust")
    assert not hasattr(V, "planet")
    S = pg.PiecewiseSpace.sobolev(B, 2, length_scale=[0.15, 0.2], pieces=["mantle", "crust"])
    assert S.names == ("mantle", "crust") and S.summand("crust") == pg.Sobolev(B["crust"], 2, length_scale=0.2)
    x = V({"core": 1.0, "mantle": 2.0, "crust": 3.0})
    on = np.array([0.0, 0.6, 0.0])
    assert V.evaluate(x, np.array([[0.1, 0, 0], [0, 0.45, 0], [0, 0, 0.95]])) == pytest.approx([1.0, 2.0, 3.0])
    assert V.evaluate(x, on, side="below") == pytest.approx(2.0) and V.evaluate(x, on, side="above") == pytest.approx(3.0)
    with pytest.raises(ValueError, match="side="):
        V.evaluate(x, on)


# ---------------------------------------------------------------------------------------------- X.restrictions(P)
def test_l2_restrictions_are_an_isometry_onto_the_broken_space_which_is_the_glued_one(line):
    P, X, _ = line
    R = X.restrictions(P)
    M = R.codomain
    assert isinstance(M, pg.PiecewiseSpace) and M.partition == P and M.names == ("a", "b", "c")
    assert M == pg.PiecewiseSpace(P, X, broken=True) == pg.PiecewiseSpace(P, X)          # L2: glued = broken
    assert M.claims.find("glued is broken").status == "theorem"
    u = X.element(U, resolution=1.0)
    sq = np.polynomial.Polynomial([1.0, 1.0, -2.0, 0.5]) ** 2
    want = sq.integ()(1.0) - sq.integ()(0.0)                                # int_0^1 U^2, exactly
    assert X.norm(u) ** 2 == pytest.approx(want, rel=1e-13)
    assert M.norm(R(u)) ** 2 == pytest.approx(want, rel=1e-13)
    assert R.claims.find("isometry").status == "theorem"
    v = R.glue(R(u))
    t = np.linspace(0, 1, 11)
    assert np.allclose(v(t), U(t), rtol=0, atol=1e-13)


def test_sobolev_restrictions_map_into_the_broken_space_and_gluing_gives_the_glued_one(line):
    P, _, H = line
    R = H.restrictions(P)
    Mb, Mg = R.codomain, pg.PiecewiseSpace(P, H)
    assert Mb == pg.PiecewiseSpace(P, H, broken=True) and Mb != Mg
    assert Mb.broken_fields == {"value"} and Mg.glued_fields == {"value"} and Mg.names == ("value",)
    assert Mb.glued() == Mg and Mg.glued() is Mg
    k = H.riesz(0.5)                                     # K(., 0.5) in H^1(0, 1): not additive over pieces
    assert Mb.norm(R(k)) ** 2 > 1.05 * H.norm(k) ** 2
    assert Mg.claims.find("additive").holds is False


# ---------------------------------------------------------------------------------------------- the broken norm
def test_the_norm_of_a_broken_space_is_the_sum_of_the_pieces_norms(line):
    P, X, H = line
    for M in (pg.PiecewiseSpace(P, H, broken=True), pg.PiecewiseSpace(P, X), pg.PiecewiseSpace.sobolev(P, 1, L)):
        y = M.element({n: (lambda t, c=c: c + t ** 2) for c, n in enumerate(M.names)}, resolution=1.0)
        parts = sum(M.summand(n).norm(M.part(y, n)) ** 2 for n in M.names)
        assert M.norm(y) ** 2 == pytest.approx(parts, rel=1e-13)
        assert M.claims.find("broken norm").status == "theorem"


# ---------------------------------------------------------------------------------------------- traces on the seams
def test_glued_elements_have_matching_traces_and_broken_ones_need_not(line):
    P, _, H = line
    Mg, Mb = pg.PiecewiseSpace(P, H), pg.PiecewiseSpace(P, H, broken=True)
    for x in (Mg.split({"value": U}, resolution=1.0), Mg.split({"value": H.riesz(0.3)})):
        J = Mg.seam_jumps(x)
        assert set(J) == {"a/b", "b/c"} and J.match is Truth.TRUE, dict(J)
    y = Mb({"a": 1.0, "b": 2.0, "c": 2.0})                 # a jump of 1 at 0.3: an element of the broken space
    assert Mb.member(y) is Truth.TRUE
    J = Mb.seam_jumps(y)
    assert J["a/b"] == pytest.approx(1.0, rel=1e-12) and J["b/c"] <= 1e-12 and J.match is Truth.FALSE
    with pytest.raises(pg.Refused, match="trace"):
        Mb.glue(y)
    g = Mb.glue(Mb.split({"value": U}, resolution=1.0))                    # matching traces: an element of H^1(0, 1)
    assert g.space == H
    t = np.linspace(0, 1, 13)
    assert np.allclose(g(t), U(t), rtol=0, atol=1e-10)


def test_gluing_states_its_conditions_by_order():
    P = pg.Partition("[0, 1]", ["[0, 0.5]", "[0.5, 1]"], names=["a", "b"])
    low = pg.PiecewiseSpace(P, pg.Sobolev("[0, 1]", 0.25, length_scale=L), broken=True)
    y = low({"a": 1.0, "b": 2.0})
    assert low.glue(y).space == low.fields["value"]           # s < 1/2: no trace, every broken element glues
    for s in (0.5, 1.5, 2):
        M = pg.PiecewiseSpace(P, pg.Sobolev("[0, 1]", s, length_scale=L), broken=True)
        with pytest.raises(pg.Refused, match="trace|derivative"):
            M.glue(M.split({"value": U}, resolution=1.0))
    c = pg.PiecewiseSpace(P, pg.Sobolev("[0, 1]", 2, length_scale=L)).claims.find("interface conditions")
    assert "normal derivative" in c.statement and c.status == "asserted"
    assert "trace" in pg.PiecewiseSpace(P, pg.Sobolev("[0, 1]", 1, length_scale=L)).claims.find(
        "interface conditions").statement


# ---------------------------------------------------------------------------------------------- the (field, piece) form
def test_fields_on_pieces_keep_the_keys_of_on_pieces(line):
    P, X, H = line
    M = pg.PiecewiseSpace(P, {"vp": pg.L2, "s": H}, where={"vp": ["b", "c"]})
    assert M.names == ("vp_b", "vp_c", "s") and M.glued_fields == {"s"}
    assert M.key("vp", "c") == "vp_c" and M.key("s", "a") == "s" and M.pieces_of("vp") == ["b", "c"]
    seg = LebesgueMeasure(pg.Interval(0.2, 0.6))
    assert M.functional("s", seg).norm() ** 2 == pytest.approx(H.riesz(seg).norm() ** 2, rel=1e-13)
    T = M.functional("s", [seg, DiracMeasure(0.5)])     # a list: pg.Functionals into R^n, as on main (plan 071 P3)
    G = pg.Functionals(M.riesz([M.parts("s", m) for m in (seg, DiracMeasure(0.5))]))
    assert isinstance(T, pg.Functionals) and T.codomain.dim == G.codomain.dim == 2
    x = M.split({"vp": U, "s": U}, resolution=1.0)
    assert M.glue(x, "s").norm() == pytest.approx(H.element(U, resolution=1.0).norm(), rel=1e-12)
