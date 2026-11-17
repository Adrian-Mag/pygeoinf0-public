"""Layered balls (plan 007, P2): shells, layers, points and sides, rays split at the interfaces into measures on the
closed layers, crossings with sides and incidence. References are analytic or by quadrature on the original polylines."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import quadrature
from pygeoinf0.measures import terms_of
from pygeoinf0.quadrature import gauss


def _planet():
    return pg.LayeredBall([0.3, 0.6, 0.9, 1.0], names=["core", "lower", "upper", "crust"],
                          interfaces=["cmb", "mid", "moho"])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _integral(f, mu, n=48):
    """int f dmu for arc length or a piecewise density on a polyline, or a signed combination: Gauss on each segment."""
    t, w = gauss(n)
    out = 0.0
    for c, m in terms_of(mu):
        P = m.set._P
        dens = getattr(m, "weights", np.ones(len(P) - 1))
        for a, b, d in zip(P[:-1], P[1:], dens):
            X = a + 0.5 * (t[:, None] + 1) * (b - a)
            out += c * d * 0.5 * np.linalg.norm(b - a) * (w @ f(*X.T))
    return out


def _f(x, y, z):
    return np.exp(0.7 * x) * (1 + y * z) + np.sin(3 * z)


def test_shell_domain():
    S = pg.Shell(0.5, 1.0)
    assert S.contains(np.array([0.75, 0, 0])) and S.contains(np.array([0.5, 0, 0])) and S.contains(np.array([0, 0, 1 + 1e-12]))
    assert not S.contains(np.array([0.4, 0, 0])) and not S.contains(np.array([0, 0, 1.01]))
    r, th, ph = S.from_cartesian(*np.array([0.3, -0.4, 0.6]))
    assert np.allclose(S.to_cartesian(r, th, ph), [0.3, -0.4, 0.6], atol=1e-15)
    X, w = quadrature.rule(S, 8)
    assert w.sum() == pytest.approx(4 / 3 * math.pi * (1 - 0.125), rel=1e-13)


def test_layers_points_and_sides():
    P = _planet()
    assert P["crust"] == pg.Shell(0.9, 1.0) and P.layer(0) == pg.Ball(0.3)
    assert P.interface("moho").radius == 0.9 and P.interface(("upper", "crust")) == P.interface(2)
    pts = np.array([[0.1, 0, 0], [0.45, 0, 0], [0, 0.95, 0], [0, 0, 1.0], [0, 0, 1.2]])
    assert list(P.layer_of(pts)) == [0, 1, 3, 3, -1]
    on = np.array([0.0, 0.6, 0.0])
    with pytest.raises(ValueError, match="side"):
        P.layer_of(on)
    assert P.layer_of(on, side="below") == 1 and P.layer_of(on, side="above") == 2
    assert P.layer_of(np.array([0.0, 0.0, 1.0])) == 3                                # the surface is no interface


def test_split_restricts_a_ray_to_the_closed_layers():
    P = _planet()
    pts = np.array([[-0.98, 0.1, 0.0], [0.0, 0.2, 0.1], [0.7, -0.6, 0.2]])
    mu = pg.LebesgueMeasure(pg.Polyline(pts)).with_density([2.0, 0.5])
    parts = P.split(mu)
    assert list(parts) == ["core", "lower", "upper", "crust"]
    assert isinstance(parts["crust"], pg.SignedMeasure) and len(parts["crust"].terms) == 2      # entered twice
    assert sum(_integral(_f, m) for m in parts.values()) == pytest.approx(_integral(_f, mu), rel=1e-13)
    for name, m in parts.items():
        lay = P[name]
        lo, hi = (0.0, lay.radius) if isinstance(lay, pg.Ball) else (lay.inner, lay.outer)
        for _, piece in terms_of(m):
            X = piece.set._P
            r = np.linalg.norm(X, axis=1)
            assert np.all(r >= lo - 1e-12) and np.all(r <= hi + 1e-12)
            rm = np.linalg.norm(0.5 * (X[1:] + X[:-1]), axis=1)
            assert np.all((rm > lo) & (rm < hi))
    sm = P.split(-2.0 * mu + pg.DiracMeasure(np.array([0.0, 0.0, 0.5])))
    assert _integral(_f, sm["upper"]) == pytest.approx(-2.0 * _integral(_f, parts["upper"]), rel=1e-13)
    assert sm["lower"].terms[-1][1].x0[2] == 0.5


def test_crossings_with_sides_and_incidence():
    P = _planet()
    b = 0.45                                                     # a straight chord with impact parameter b
    h = math.sqrt(1 - b * b) * (1 - 1e-12)
    cr = P.crossings([[-h, b, 0.0], [h, b, 0.0]])
    assert [c.interface for c in cr] == ["moho", "mid", "mid", "moho"]
    for c in cr:
        rj = P.radii[c.index]
        assert np.linalg.norm(c.point) == pytest.approx(rj, rel=1e-14)
        assert c.cosines == pytest.approx((math.sqrt(1 - (b / rj) ** 2),) * 2, rel=1e-12)
    assert [c.sides for c in cr] == [("above", "below"), ("above", "below"), ("below", "above"), ("below", "above")]
    q = 0.3 * _unit([1.0, 1.0, 0.0])                             # a vertex on the cmb, both pieces above: a reflection
    refl = P.crossings([[0.0, 0.8, 0.0], q, [0.8, 0.0, 0.0]])
    assert [(c.interface, c.sides) for c in refl if c.interface == "cmb"] == [("cmb", ("above", "above"))]
