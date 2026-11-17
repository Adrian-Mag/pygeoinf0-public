"""Plan 035 P4: Sobolev spaces on pieces. The restriction construction restricted to a piece (co-isometry, the
minimal-norm extension as adjoint, no extension by zero), the broken sum over a partition, glued and broken (field,
piece) sums, seam masses, thresholds, and the refusals for the intrinsic and spectral constructions. References: closed
forms of the exponential (Matern 1/2) kernel of H^1_l on intervals, K = e^(-|x-y|/l) / (2l), and whole-domain
energies."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import PointEvaluationUndefined, UnboundedFunctional
from pygeoinf0.measures import DiracMeasure, LebesgueMeasure

L = 0.2


@pytest.fixture(scope="module")
def line():
    X = pg.Sobolev("[0, 1]", 1, length_scale=L)
    P = pg.Partition("[0, 1]", ["[0, 0.4]", "[0.4, 1]"], names=["a", "b"])
    return X, P, X.restrict_to(P["a"])


@pytest.fixture(scope="module")
def ball():
    X = pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.3)
    P = pg.LayeredBall([0.55, 0.8, 1.0], names=["a", "b", "c"]).partition
    seg = LebesgueMeasure(pg.Segment([0.1, 0.0, 0.3], [0.2, 0.1, 0.95]))        # crosses both seams
    return X, P, seg


def seg_energy(n):                     # int int K over a segment of length n, K = e^(-|x-y|/l) / (2l)
    return n - L * (1 - np.exp(-n / L))


# ---------------------------------------------------------------------------------------------- one piece
def test_the_piece_space_is_the_same_construction_and_r_is_a_co_isometry(line):
    X, P, R = line
    XD, k0 = R.codomain, 1 / (2 * L)
    assert XD == pg.Sobolev(P["a"], 1, length_scale=L) and XD.construction.kind == "restriction"
    for mu in (DiracMeasure(0.3), DiracMeasure(0.4), LebesgueMeasure(pg.Interval(0.1, 0.35))):
        assert XD.riesz(mu).norm() ** 2 == pytest.approx(X.riesz(mu).norm() ** 2, rel=1e-14)    # same energies
    k = X.riesz(0.3)
    assert R(k).norm() ** 2 == pytest.approx(k0, rel=1e-14)
    # y outside the piece: R k_y = e^(-(y - b)/l) k_b on [a, b] (the kernel is Markov), the quotient norm
    ky = X.riesz(0.7)
    assert R(ky).norm() ** 2 == pytest.approx(np.exp(-2 * 0.3 / L) * k0, rel=1e-12)
    assert R(ky).norm() < ky.norm()                                       # a co-isometry, not an isometry
    cl = R.claims
    for key in ("co-isometry", "adjoint", "piece space", "measures"):
        assert cl.find(key).status == "theorem"
    assert cl.find("extension by zero").holds is False and cl.find("realisation").status == "checked"


def test_the_adjoint_is_the_minimal_norm_extension(line):
    X, P, R = line
    XD = R.codomain
    vs = [XD.riesz(DiracMeasure(0.2)), XD.riesz(LebesgueMeasure(pg.Interval(0.05, 0.4))), XD.riesz(0.4) * 2.0
          - XD.riesz(0.1)]
    us = [X.riesz(0.7), X.riesz(LebesgueMeasure(pg.Interval(0.2, 0.6))), X(lambda x: np.sin(3 * x)), X.riesz(0.25)]
    for v in vs:
        Ev = R.adjoint(v)
        assert Ev.space == X and Ev.norm() == pytest.approx(v.norm(), rel=1e-13)       # an isometry
        assert (R(Ev) - v).norm() <= 1e-12 * v.norm()                                   # R R* = I
        for u in us:
            assert XD.inner(R(u), v) == pytest.approx(X.inner(u, Ev), rel=1e-11, abs=1e-13)
    # R* k^D_mu = k^Omega_mu: the same kernel function, also off the piece
    assert R.adjoint(vs[0])(np.array([0.9])) == pytest.approx(np.exp(-0.7 / L) / (2 * L), rel=1e-13)
    with pytest.raises(pg.Refused, match="exterior problem"):
        R.adjoint(XD(lambda x: x))
    assert R.adjoint.adjoint is R


def test_extension_by_zero_is_refused(line):
    X, P, R = line
    with pytest.raises(pg.Refused, match="not in H\\^s") as e:
        R.extension
    assert any("minimal-norm" in n for n in e.value.needs)
    R3 = pg.Sobolev("[0, 1]", 0.3, length_scale=L).restrict_to("[0, 0.4]")
    with pytest.raises(pg.Refused, match="not R\\*"):
        R3.extension


# ---------------------------------------------------------------------------------------------- over a partition
def test_the_broken_restriction_is_injective_bounded_and_not_an_isometry(line):
    X, P, _ = line
    R, k0 = X.restrictions(P), 1 / (2 * L)
    assert R.codomain.names == ("a", "b")
    y = R(X.riesz(0.7))
    assert R.codomain.norm(y) ** 2 == pytest.approx(k0 * (1 + np.exp(-2 * 0.3 / L)), rel=1e-12)   # != k0 = ||u||^2
    cl = R.claims
    assert cl.find("isometry").holds is False and cl.find("injective").status == "theorem"
    assert cl.find("lower bound").status == "asserted" and "Lions-Magenes" in cl.find("lower bound").reason
    x = R.codomain.riesz({"a": 0.1, "b": LebesgueMeasure(pg.Interval(0.5, 0.9))})
    for u in (X.riesz(0.7), X(lambda t: t * t)):
        assert R.codomain.inner(R(u), x) == pytest.approx(X.inner(u, R.adjoint(x)), rel=1e-11)
    with pytest.raises(pg.Refused, match="match across the seams"):
        P.glue(X)
    half = pg.Sobolev("[0, 1]", 0.5, length_scale=L)
    assert half.restrictions(P).claims.find("lower bound").holds is False


def test_glued_grams_keep_cross_terms_and_broken_ones_drop_them(line):
    X, P, _ = line
    seg = LebesgueMeasure(pg.Interval(0.2, 0.6))
    Mg, Mb = pg.PiecewiseSpace(P, {"vp": X}), pg.PiecewiseSpace(P, {"vp": X}, broken=True)
    assert Mg.names == ("vp",) and Mg.key("vp", "b") == "vp" and Mb.names == ("vp_a", "vp_b")
    glued, broken = Mg.functional("vp", seg).norm() ** 2, Mb.functional("vp", seg).norm() ** 2
    assert glued == pytest.approx(seg_energy(0.4), rel=1e-13) == X.riesz(seg).norm() ** 2      # 0.227067...
    assert broken == pytest.approx(2 * seg_energy(0.2), rel=1e-13)                             # 0.147152...
    k = [f.representer for f in Mg.piece_functionals("vp", seg).values()]
    G = np.array([[Mg.inner(a, b) for b in k] for a in k])
    assert G.sum() == pytest.approx(glued, rel=1e-13) and np.trace(G) == pytest.approx(broken, rel=1e-13)
    assert G[0, 1] > 0.03                                                                   # the cross term
    u = X.riesz(0.7)
    assert Mg.functional("vp", seg)(Mg.split({"vp": u})) == pytest.approx(X.inner(X.riesz(seg), u), rel=1e-14)
    assert Mg.glue(Mg.split({"vp": u}))["vp"].norm() == pytest.approx(u.norm(), rel=1e-14)
    assert Mg.claims.find("glued").status == "theorem" and Mb.claims.find("broken").status == "asserted"


def test_seam_masses_glued_any_side_broken_side_required(line):
    X, P, _ = line
    Mg, Mb = pg.PiecewiseSpace(P, {"vp": X}), pg.PiecewiseSpace(P, {"vp": X}, broken=True)
    d = DiracMeasure(0.4)
    lg = Mg.functional("vp", d)
    c = lg.claims.find("seam sides")
    assert c.status == "checked" and "both sides" in c.statement
    u = X.riesz(0.7)
    assert lg(Mg.split({"vp": u})) == pytest.approx(np.exp(-0.3 / L) / (2 * L), rel=1e-14)
    for side in ("below", "above"):
        assert Mg.functional("vp", d, side=side)(Mg.split({"vp": u})) == pytest.approx(lg(Mg.split({"vp": u})))
    with pytest.raises(ValueError, match="side="):
        Mb.functional("vp", d)
    x = Mb({"vp_a": Mb.summand("vp_a").riesz(0.1)})                 # a broken element: jumps at the seam
    below, above = Mb.functional("vp", d, side="below"), Mb.functional("vp", d, side="above")
    assert below(x) == pytest.approx(np.exp(-0.3 / L) / (2 * L), rel=1e-13) and above(x) == 0.0


# ---------------------------------------------------------------------------------------------- a ball cut at two radii
def test_ball_glued_and_broken_grams_and_seam_masses(ball):
    X, P, seg = ball
    Mg, Mb = pg.PiecewiseSpace(P, {"vp": X}), pg.PiecewiseSpace(P, {"vp": X}, broken=True)
    whole = X.riesz(seg).norm() ** 2
    glued, broken = Mg.functional("vp", seg).norm() ** 2, Mb.functional("vp", seg).norm() ** 2
    k = [f.representer for f in Mg.piece_functionals("vp", seg).values()]
    G = np.array([[Mg.inner(a, b) for b in k] for a in k])
    assert len(k) == 3 and glued == pytest.approx(whole, rel=1e-14) and G.sum() == pytest.approx(whole, rel=1e-12)
    assert broken == pytest.approx(np.trace(G), rel=1e-12) and broken < 0.6 * glued       # 0.1747 against 0.3518
    for mu in (DiracMeasure(np.array([0.0, 0.0, 0.55])), LebesgueMeasure(pg.Sphere(0.8))):
        assert Mg.functional("vp", mu).claims.find("seam sides").status == "checked"
        assert Mg.functional("vp", mu).norm() == pytest.approx(X.riesz(mu).norm(), rel=1e-14)
        with pytest.raises(ValueError, match="side="):
            Mb.functional("vp", mu)
    R = X.restrict_to(P["b"])
    v = R.codomain.riesz(DiracMeasure(np.array([0.3, 0.2, 0.6])))
    w = X.riesz(DiracMeasure(np.array([0.0, 0.1, 0.65])))
    assert R(w).norm() == pytest.approx(w.norm(), rel=1e-14)
    for u in (w, X(lambda x, y, z: 1 + z + x * y)):
        assert R.codomain.inner(R(u), v) == pytest.approx(X.inner(u, R.adjoint(v)), rel=1e-12)


def test_a_shell_cut_in_two(ball):
    S = pg.Shell(0.3, 1.0)
    X = pg.Sobolev(S, 2, length_scale=0.3)
    P = pg.Partition(S, {"lo": pg.Shell(0.3, 0.6), "hi": pg.Shell(0.6, 1.0)})
    seg = LebesgueMeasure(pg.Segment([0.0, 0.1, 0.35], [0.1, 0.0, 0.9]))
    Mg, Mb = pg.PiecewiseSpace(P, {"vp": X}), pg.PiecewiseSpace(P, {"vp": X}, broken=True)
    assert Mg.functional("vp", seg).norm() == pytest.approx(X.riesz(seg).norm(), rel=1e-14)
    assert Mb.functional("vp", seg).norm() < Mg.functional("vp", seg).norm()
    inner = pg.Segment([0.0, 0.0, 0.35], [0.0, 0.0, 0.55])
    R = X.restrict_to(P["lo"])
    assert R(X.riesz(inner)).norm() == pytest.approx(X.riesz(inner).norm(), rel=1e-14)


# ---------------------------------------------------------------------------------------------- thresholds
def test_measures_are_bounded_on_pieces_by_the_same_threshold():
    I = pg.Partition("[0, 1]", ["[0, 0.4]", "[0.4, 1]"])
    half = pg.Sobolev("[0, 1]", 0.5, length_scale=L)
    with pytest.raises(PointEvaluationUndefined):
        half.restrict_to("[0, 0.4]").codomain.riesz(0.2)
    with pytest.raises(PointEvaluationUndefined):
        pg.PiecewiseSpace(I, {"v": half}).functional("v", DiracMeasure(0.4))
    ray = pg.Segment([0.0, 0.0, 0.6], [0.0, 0.0, 0.9])
    R = pg.Sobolev(pg.Ball(1.0), 1.2, length_scale=0.3).restrict_to(pg.Shell(0.5, 1.0))
    assert "points False, surfaces True, paths True" in R.claims.find("measures").statement
    with pytest.raises(PointEvaluationUndefined):
        R.codomain.riesz(DiracMeasure(np.array([0.0, 0.0, 0.7])))                     # s = 1.2 <= 3/2
    assert R(R.domain.riesz(ray)).norm() == pytest.approx(R.domain.riesz(ray).norm(), rel=1e-14)    # 1.2 > 1
    with pytest.raises(UnboundedFunctional):                                           # the borderline s = 1
        pg.Sobolev(pg.Ball(1.0), 1, length_scale=0.3).restrict_to(pg.Shell(0.5, 1.0)).codomain.riesz(ray)
    with pytest.raises(UnboundedFunctional):
        pg.Sobolev(pg.Ball(1.0), 0.5, length_scale=0.3).restrict_to(pg.Shell(0.5, 1.0)).codomain.riesz(
            LebesgueMeasure(pg.Sphere(0.7)))


# ---------------------------------------------------------------------------------------------- other constructions
def test_intrinsic_and_spectral_spaces_glue_and_refuse_piece_spaces(line):
    _, P, _ = line
    a = pg.SobolevForm("[0, 1]", 1, length_scale=L)
    forms = {"form": a.space(), "power": pg.Sobolev(a.operator(), 2), "fractional": pg.Sobolev(a.operator(), 1.5)}
    for name, X in forms.items():
        with pytest.raises(pg.Refused, match="quotient") as e:
            X.restrict_to(P["a"])
        assert any("glued" in n for n in e.value.needs)
        with pytest.raises(pg.Refused, match="broken model"):
            pg.PiecewiseSpace(P, {"v": X}, broken=True)
        M = pg.PiecewiseSpace(P, {"v": X})
        seg = LebesgueMeasure(pg.Interval(0.2, 0.6))
        assert M.functional("v", seg).norm() == pytest.approx(X.riesz(seg).norm(), rel=1e-14)
        assert M.functional("v", DiracMeasure(0.4)).claims.find("seam sides").status == "theorem"
    assert M.claims.find("additive").holds is False                                 # fractional
    assert pg.PiecewiseSpace(P, {"v": forms["form"]}).claims.find("additive").holds is True
    assert pg.PiecewiseSpace(P, {"v": forms["power"]}).claims.find("additive").holds is True
    assert pg.PiecewiseSpace(P, {"v": pg.Sobolev("[0, 1]", 1.5, length_scale=L)}).claims.find("additive").holds is False


def test_mixed_fields_and_glued_fields_on_some_pieces(line):
    X, P, _ = line
    M = pg.PiecewiseSpace(P, {"vp": X, "rho": pg.L2})
    assert M.names == ("vp", "rho_a", "rho_b") and M.pieces_of("vp") == ["a", "b"]
    assert M.restriction("vp", "b").codomain == pg.Sobolev(P["b"], 1, length_scale=L)
    with pytest.raises(pg.Refused, match="some pieces only"):
        pg.PiecewiseSpace(P, {"vp": X}, where={"vp": ["a"]})
    Mb = pg.PiecewiseSpace(P, {"vp": X}, where={"vp": ["a"]}, broken={"vp"})
    assert Mb.names == ("vp_a",)
