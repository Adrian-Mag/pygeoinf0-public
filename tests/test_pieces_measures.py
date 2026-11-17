"""Measures carried to the pieces of a layered ball (plan 035 P1): mu.restricted(D) on the exact set S cap D, splits of
volume, density and bank measures that add up to the whole, side= on interfaces, and functionals on direct sums of L2
shells that keep their symbolic parts symbolic (the T(vp) target). References: closed forms of the bank, and Gauss
quadrature in polar coordinates about the bump's own axis (independent of the pieces' rule)."""
import math
import warnings

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

import pygeoinf0 as pg
from pygeoinf0 import quadrature
import pygeoinf0.functions as F
from pygeoinf0.core.errors import SupportOutsideDomain, UnboundedFunctional, Unsupported
from pygeoinf0.measures import terms_of
from pygeoinf0.spaces.representers import DensityRepresenter
from pygeoinf0.measures.weights import size
from pygeoinf0.domains.pieces import BallPiece, meet

LAYERS = {"a": (0.0, 0.55), "b": (0.55, 0.8), "c": (0.8, 1.0)}
U = np.array([0.3, 0.2, 0.7]) / np.linalg.norm([0.3, 0.2, 0.7])
CASES = [("one cut, axis", (0, 0, 0.55), 0.15), ("two cuts, axis", (0, 0, 0.675), 0.2),
         ("one cut, off axis", tuple(0.8 * U), 0.15), ("two cuts, off axis", tuple(0.675 * U), 0.2),
         ("narrow, off axis", tuple(0.8 * U), 0.02), ("narrow inside b, off axis", tuple(0.675 * U), 0.03)]


def _planet():
    return pg.LayeredBall([0.55, 0.8, 1.0], names=["a", "b", "c"])


def _bump_values(c, rho, X):
    s = np.linalg.norm(X - np.asarray(c, float), axis=-1) / rho
    return np.where(s < 1, np.exp(-1.0 / np.maximum(1 - s * s, 1e-300)), 0.0)


def _reference(c, rho, f, a, b, n=160, m=64):
    """int over a <= |x| <= b of bump(c, rho) f: Gauss in r and in the angle from the axis 0 -> c, trapezoid about it
    (on the axis this is the axisymmetric rule; off it, the same rule in the rotated frame)."""
    c = np.asarray(c, float)
    d = np.linalg.norm(c)
    ax = c / d
    e = np.eye(3)[np.argmin(np.abs(ax))]
    e1 = e - (e @ ax) * ax
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    lo, hi = max(a, d - rho), min(b, d + rho)
    if hi <= lo:
        return 0.0
    tmax = math.asin(min(1.0, rho / d))
    t, w = leggauss(n)
    r, wr = 0.5 * (lo + hi) + 0.5 * (hi - lo) * t, 0.5 * (hi - lo) * w
    th, wt = 0.5 * tmax * (t + 1), 0.5 * tmax * w
    ph = 2 * np.pi * np.arange(m) / m
    R, TH, PH = np.meshgrid(r, th, ph, indexing="ij")
    X = R[..., None] * (np.cos(TH)[..., None] * ax + np.sin(TH)[..., None]
                        * (np.cos(PH)[..., None] * e1 + np.sin(PH)[..., None] * e2))
    W = np.einsum("i,j->ij", wr, wt)[..., None] * (2 * np.pi / m) * R ** 2 * np.sin(TH)
    return float(np.sum(W * _bump_values(c, rho, X) * f(X[..., 0], X[..., 1], X[..., 2])))


def _vp(x, y, z):
    return 1.0 + z + 0.5 * x * y


def _vs(x, y, z):
    return 3.0 - x


# ------------------------------------------------------------------------------------------ sets and restriction
def test_meet_is_exact_and_keeps_both_descriptions():
    P = _planet()
    S = pg.Ball(0.15, (0, 0, 0.55))
    J = meet(S, P["b"])
    assert isinstance(J, BallPiece) and J.inside(P["b"]) is pg.Truth.TRUE and J.parts == (S, P["b"])
    assert meet(S, P["c"]) is None                                   # apart
    assert meet(pg.Ball(0.1, (0, 0, 0.675)), P["b"]) == pg.Ball(0.1, (0, 0, 0.675))     # inside: the ball itself
    assert meet(pg.Ball(1.0), P["b"]) == P["b"]                    # the layer inside the ball: the layer
    assert meet(pg.Ball(0.7), P["b"]) == pg.Shell(0.55, 0.7)       # concentric
    vols = [meet(S, P[k]) for k in "ab"]
    assert sum(v.volume for v in vols) == pytest.approx(S.volume, rel=1e-14)
    for k in "ab":                                                    # the rule integrates 1 exactly
        X, W = quadrature.rule(vols["ab".index(k)], 16)
        assert W.sum() == pytest.approx(vols["ab".index(k)].volume, rel=1e-13)
        assert np.all(vols["ab".index(k)].contains(X)) and np.all(P[k].contains(X))
    with pytest.raises(Unsupported, match="spheres about different centres"):
        meet(meet(pg.Ball(0.1, (0, 0, 0.56)), P["b"]), pg.Ball(0.1, (0, 0.05, 0.56)))


def test_restricted_volume_measures_live_on_the_exact_piece():
    P = _planet()
    bump = F.on(P.ball).bump(centre=(0, 0, 0.55), radius=0.15)
    for mu in (bump.measure(), pg.LebesgueMeasure(P.ball).with_density(bump)):
        parts = P.split(mu)
        assert set(parts) == {"a", "b"}
        for k, m in parts.items():
            assert isinstance(m.set, BallPiece) and m.set.inside(P[k]) is pg.Truth.TRUE  # narrowed to its support
            assert m.claims.find("restriction").status == "derived"
        assert sum(m.mass()[0] for m in parts.values()) == pytest.approx(bump.closed["integral"][0], rel=1e-10)
    assert pg.LebesgueMeasure(P.ball).restricted(P["b"]) == pg.LebesgueMeasure(P["b"])
    assert pg.LebesgueMeasure(P.ball).restricted(pg.Ball(2.0)) == pg.LebesgueMeasure(P.ball)
    inside = F.on(P["b"]).bump(centre=(0, 0, 0.675), radius=0.1).measure()
    assert inside.restricted(P["b"]) is inside and inside.restricted(P["a"]) is None
    # the restricted measure passes the support check of the piece exactly, and fails it on the whole ball's other side
    k = pg.L2(P["b"]).riesz(bump.measure().restricted(P["b"]))
    assert pg.L2(P["b"]).norm(k) > 0
    with pytest.raises(SupportOutsideDomain):
        pg.L2(P["b"]).riesz(bump.measure())


def test_signed_sums_restrict_termwise():
    P = _planet()
    mu = F.on(P.ball).bump(centre=(0, 0, 0.55), radius=0.15).measure()
    nu = -2.0 * mu + pg.DiracMeasure([0, 0, 0.1]) + 0.5 * pg.LebesgueMeasure(pg.Segment([0, 0, -0.9], [0, 0, 0.9]))
    parts = P.split(nu)
    assert set(parts) == {"a", "b", "c"}
    total = 0.0                                      # total masses: -2 * mass + 1 + 0.5 * 1.8
    for m in parts.values():
        for c, t in terms_of(m):
            total += c * (1.0 if isinstance(t, pg.DiracMeasure) else (t.mass()[0] if hasattr(t, "mass") else size(t.set)))
    assert total == pytest.approx(-2 * mu.mass()[0] + 1 + 0.9, rel=1e-10)
    assert mu.restricted(P["c"]) is None
    assert (-2.0 * mu).restricted(P["c"]) is None


def test_points_and_spheres_on_an_interface_need_a_side():
    P = _planet()
    d = pg.DiracMeasure([0, 0, 0.55])
    with pytest.raises(ValueError, match="side="):
        d.restricted(P["b"])
    assert d.restricted(P["b"], side="above") is d and d.restricted(P["a"], side="above") is None
    assert d.restricted(P["a"], side="below") is d and d.restricted(P["b"], side="below") is None
    assert pg.DiracMeasure([0, 0, 0.7]).restricted(P["b"]) is not None
    assert pg.DiracMeasure([0, 0, 0.7]).restricted(P["c"]) is None
    S = pg.LebesgueMeasure(P.interface(0))
    with pytest.raises(ValueError, match="side="):
        P.split(S)
    assert list(P.split(S, side="above")) == ["b"] and list(P.split(S, side="below")) == ["a"]
    with pytest.raises(UnboundedFunctional):              # a seam measure is no L2 functional anyway
        pg.PiecewiseSpace.lebesgue(P).riesz(S, side="above")
    with pytest.raises(ValueError, match="side="):              # the broken sum keeps side= required (author, D4)
        pg.PiecewiseSpace.lebesgue(P).riesz(pg.DiracMeasure([0, 0, 0.8]))


# ------------------------------------------------------------------------------------------ pieces sum to the whole
@pytest.mark.parametrize("label,c,rho", CASES, ids=[c[0] for c in CASES])
def test_pieces_sum_to_the_whole(label, c, rho):
    P = _planet()
    V = pg.PiecewiseSpace.lebesgue(P)
    bump = F.on(P.ball).bump(centre=c, radius=rho)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pg.RealisationWarning)
        parts = P.split(bump.measure())
        x = V({k: _vp for k in P.names})
        k = V.riesz(bump.measure())
        for name, m in parts.items():
            got = V.summand(name).inner(V.part(k, name), x.part(name) if hasattr(x, "part") else V.part(x, name))
            assert got == pytest.approx(_reference(c, rho, _vp, *LAYERS[name]), rel=1e-12, abs=1e-18)
        total = V.inner(k, x)
    assert total == pytest.approx(sum(_reference(c, rho, _vp, *LAYERS[n]) for n in P.names), rel=1e-12)
    assert V.norm(k) ** 2 == pytest.approx(bump.closed["l2_norm_sq"][0], rel=1e-9)
    assert sum(m.mass()[0] for m in parts.values()) == pytest.approx(bump.closed["integral"][0], rel=1e-10)


# ------------------------------------------------------------------------------------------ the T(vp) target
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("label,c,rho", [CASES[1], CASES[2], CASES[4]], ids=[CASES[i][0] for i in (1, 2, 4)])
def test_target_on_one_field_of_a_flat_field_and_shell_sum(label, c, rho, weighted):
    """T(m) = int g vp dV, g a unit-mass bump straddling cuts, on (+)_(field, shell) L2; zero on the vs pieces; the
    Gram G G* = sum_k gram(l_k) / w_k."""
    P = _planet()
    keys = [f"{f}_{k}" for f in ("vp", "vs") for k in P.names]
    w = {k: (2.0 if k.startswith("vp") else 1.0) * (1 + i) for i, k in enumerate(keys)} if weighted else None
    M = pg.DirectSum({k: pg.L2(P[k.split("_")[1]]) for k in keys}, weights=w)
    bump = F.on(P.ball).bump(centre=c, radius=rho)
    g = bump.normalised()
    with warnings.catch_warnings():
        warnings.simplefilter("error", pg.RealisationWarning)
        parts = P.split(g.measure())
        G = pg.Functionals(M.riesz([{f"vp_{k}": m for k, m in parts.items()}]))
        x = M({k: (_vp if k.startswith("vp") else _vs) for k in keys})
        val = float(G(x).values[0])
        on_vs = float(G(M({k: _vs for k in keys if k.startswith("vs")})).values[0])
        GG = float((G @ G.adjoint).matrix()[0, 0])
    Z = bump.closed["integral"][0]
    assert val == pytest.approx(sum(_reference(c, rho, _vp, *LAYERS[k]) for k in P.names) / Z, rel=1e-12)
    assert on_vs == 0.0
    wk = {k: 1.0 if w is None else w[f"vp_{k}"] for k in P.names}
    one = lambda x, y, z: np.ones_like(x)          # noqa: E731
    sq = {k: _reference(c, rho, lambda x, y, z: _bump_values(c, rho, np.stack([x, y, z], -1)), *LAYERS[k]) / Z ** 2
          for k in parts}
    assert GG == pytest.approx(sum(sq[k] / wk[k] for k in parts), rel=1e-9)
    if w is None:
        assert GG == pytest.approx(g.closed["l2_norm_sq"][0], rel=1e-9)
    assert sum(_reference(c, rho, one, *LAYERS[k]) for k in P.names) / Z == pytest.approx(1.0, rel=1e-12)


def test_symbolic_parts_combine_and_refuse_a_fit():
    P = _planet()
    M = pg.PiecewiseSpace.lebesgue(P)
    mu = F.on(P.ball).bump(centre=tuple(0.8 * U), radius=0.05).measure()
    k = M.riesz(mu)
    x = M({n: _vp for n in P.names})
    two = 2.0 * k - k                                                  # stays symbolic: the signed measure's representer
    assert M.inner(two, x) == pytest.approx(M.inner(k, x), rel=1e-13)
    assert isinstance(M.part(k, "b"), DensityRepresenter)
    assert M.norm(k) ** 2 == pytest.approx(sum(M.summand(n).norm(M.part(k, n)) ** 2 for n in ("b", "c")), rel=1e-13)
    with pytest.raises(pg.Refused, match="kept symbolic"):              # adding a formula needs a fit across S cap D
        M.norm(k + x)
    F2 = M.riesz([mu, pg.LebesgueMeasure(P["b"])])                      # a family with a realisable member
    G = M.inner(F2, F2)
    assert G[1, 1] == pytest.approx(P["b"].volume, rel=1e-13)
    assert G[0, 1] == pytest.approx(mu.restricted(P["b"]).mass()[0], rel=1e-10)


def test_overlapping_round_sets_pair_exactly_or_refuse():
    """Two bumps overlapping inside a shell pair on their lens (a BallPiece, intersections rule 2b); two bumps cut by
    the same interface and overlapping in one piece (three spheres) are refused, not approximated."""
    S = pg.Shell(0.55, 0.8)
    M = pg.L2(S)
    f1, f2 = (F.on(S).bump(centre=c, radius=r) for c, r in (((0, 0, 0.675), 0.1), ((0, 0.05, 0.675), 0.08)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pg.RealisationWarning)
        G = M.inner(*(2 * [M.riesz([f1.measure(), f2.measure()])]))
    X, W = quadrature.rule(pg.Ball(0.1, (0, 0, 0.675)), 96)
    assert G[0, 1] == pytest.approx(W @ (f1(*X.T) * f2(*X.T)), rel=1e-9)
    P = _planet()
    g1, g2 = (F.on(P.ball).bump(centre=c, radius=r).measure() for c, r in (((0, 0, 0.55), 0.15), ((0, 0.05, 0.62), 0.12)))
    with pytest.raises(pg.Refused, match="not representable"):
        pg.PiecewiseSpace.lebesgue(P).inner(*(2 * [pg.PiecewiseSpace.lebesgue(P).riesz([g1, g2])]))
