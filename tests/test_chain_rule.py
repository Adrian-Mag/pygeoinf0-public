"""The chain rule of composed point maps (plan 020 P4: forward models built from plugins' parts, e.g. gravity after a
petrophysical law): S @ R has the derivative DS(R(x)) DR(x) when both parts have one, a LinearOperator being its own;
nested compositions; a part without a derivative refused with its name; the inner map's dom kept."""
import numpy as np
import pytest

import pygeoinf0 as pg

R2, R3 = pg.Euclidean(2), pg.Euclidean(3)
A = pg.Matrix(np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]]), R2, R3)
F = pg.port.External(R3, R3, lambda x: np.sin(x) * x[::-1], name="F",
                jvp=lambda x, v: np.cos(x) * x[::-1] * v + np.sin(x) * v[::-1],
                vjp=lambda x, w: np.cos(x) * x[::-1] * w + (np.sin(x) * w)[::-1])
H = pg.port.External(R2, R2, lambda x: np.r_[x[0] ** 2, np.exp(x[1])], name="H",
                jvp=lambda x, v: np.r_[2 * x[0] * v[0], np.exp(x[1]) * v[1]],
                vjp=lambda x, w: np.r_[2 * x[0] * w[0], np.exp(x[1]) * w[1]])


@pytest.mark.parametrize("which", ["F∘A", "A∘H", "F∘A∘H", "B∘F∘A"])
def test_compositions_have_the_chain_rule(which):
    B = pg.Matrix(np.ones((1, 3)), R3, pg.Euclidean(1))
    G = {"F∘A": F @ A, "A∘H": A @ H, "F∘A∘H": F @ (A @ H), "B∘F∘A": B @ (F @ A)}[which]
    res = pg.port.taylor_test(G, G.domain(np.r_[0.3, -0.7]), directions=[G.domain(np.r_[1.0, 0.4]),
                                                                      G.domain(np.r_[-0.2, 1.0])])
    assert all(np.isinf(r) or r > 1.9 for r in res.rates), res
    assert res.adjoint_gap < 1e-12, res


def test_a_part_without_a_derivative_is_named():
    plain = pg.Map(R3, R3, lambda x: x, name="plain")
    with pytest.raises(pg.Refused, match="plain has no derivative"):
        (plain @ A).derivative(R2(np.r_[1.0, 2.0]))


def test_the_inner_dom_is_kept():
    pos = pg.port.Coordinates.identity(R2).subset(lambda x: bool(np.all(x > 0)), name="x > 0")
    L = pg.port.External(R2, R2, lambda x: np.log(x), name="log", dom=pos, jvp=lambda x, v: v / x, vjp=lambda x, w: w / x)
    G = A @ L
    with pytest.raises(pg.Refused, match="not in dom"):
        G(R2(np.r_[-1.0, 1.0]))
    assert G.claims.status("total")[0] is pg.Truth.FALSE
