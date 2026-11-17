"""The construction of a Sobolev space is part of its identity (plan 006, P0): what pg.Sobolev(D, s, length_scale=l)
means on domains (restriction) and closed manifolds (spectral), equality and hashing by the inner product, the
canonical keys (restriction of H^1 on an interval = the Robin form with beta = l; A_(a_2) = A_(a_1)^2 on a flat torus),
claims and refusals of the mathematical objects."""
import math

import pytest

import pygeoinf0 as pg
from pygeoinf0.operators.unbounded import FormOperator
from pygeoinf0.spaces.constructions import RestrictionSpace, Spectral, interval_restriction_form
from pygeoinf0.core.errors import Unsupported


def test_default_constructions():
    I = pg.Interval("[0, 1]")
    H = pg.Sobolev(I, 1, length_scale=0.1)
    assert isinstance(H.construction, RestrictionSpace) and repr(H) == "H^1([0, 1], l=0.1)"
    B = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.1)
    assert B.construction.kind == "restriction" and "McLean" in repr(B.claims)
    T = pg.Torus("[0, 1]", "[0, 1]")
    HT = pg.Sobolev(T, 2, length_scale=0.2)
    assert isinstance(HT.construction, Spectral)
    assert HT.construction.operator == pg.SobolevForm(T, 1, length_scale=0.2).operator()
    S = pg.Sphere(1.0)
    HS = pg.Sobolev(S, 2, length_scale=0.2)
    assert HS == pg.Sobolev(pg.SobolevForm(S, 1, length_scale=0.2).operator(), 2) and repr(HS).endswith("l=0.2)")
    assert "Laplace-Beltrami" in repr(HS.claims)


def test_closed_manifolds_are_spectral_instances():
    T = pg.Torus("[0, 1]", "[0, 2]")
    a1 = pg.SobolevForm(T, 1, length_scale=0.3)
    assert a1.space() == pg.Sobolev(T, 1, length_scale=0.3)              # second representation theorem
    assert pg.Sobolev(a1.operator(), 2.5) == pg.Sobolev(T, 2.5, length_scale=0.3)
    a2 = pg.SobolevForm(T, 2, length_scale=0.3)                            # flat: A_(a_2) = A_(a_1)^2
    assert a2.space() == pg.Sobolev(T, 2, length_scale=0.3) != pg.Sobolev(T, 2, length_scale=0.31)
    assert hash(a2.space()) == hash(pg.Sobolev(T, 2, length_scale=0.3))
    C = pg.Circle(2.0)
    assert pg.SobolevForm(C, 1, length_scale=0.5).space() == pg.Sobolev(C, 1, length_scale=0.5)
    with pytest.raises(Unsupported, match="curvature"):
        pg.SobolevForm(pg.Sphere(1.0), 2, length_scale=0.2)


def test_keys_follow_the_inner_product():
    I = pg.Interval("[0, 1]")
    l = 0.1
    res = RestrictionSpace(I, 1, l)
    robin = Spectral(interval_restriction_form(I, l).operator(), 1)      # a_1 + l [u(0)^2 + u(1)^2]
    assert res.key() == robin.key()
    assert res.key() != Spectral(pg.SobolevForm(I, 1, length_scale=l, robin=0.5 * l).operator(), 1).key()
    neumann = pg.SobolevForm(I, 1, length_scale=l)
    assert Spectral(neumann.operator(), 1).key() != res.key()             # same functions, other inner product
    assert Spectral(neumann.operator(), 2).key() != RestrictionSpace(I, 2, l).key()
    assert neumann == pg.SobolevForm(I, 1, length_scale=l, robin={"left": 0.0})     # beta = 0 is Neumann
    assert pg.SobolevForm(I, 1, length_scale=l, dirichlet="all").dirichlet == ("left", "right")
    assert RestrictionSpace(I, 2, l).key() != RestrictionSpace(I, 2, 2 * l).key()


def test_conditions_and_claims_of_forms_and_operators():
    I = pg.Interval("[0, 1]")
    l = 0.2
    a = pg.SobolevForm(I, 1, length_scale=l, dirichlet="left", robin={"right": 0.3})
    assert a.conditions("left") == [((1.0,), True)]
    assert a.conditions("right") == [((0.3, l * l), False)]                # l^2 d_n u + beta u = 0
    A = a.operator()
    assert isinstance(A, FormOperator) and A.order == 2
    c = A.power_conditions(2, "left")                                      # D(A^2): u = 0 and (1 - l^2 d^2) u = 0
    assert c[0] == ((1.0,), True) and c[1][0] == pytest.approx((1.0, 0.0, -l * l)) and not c[1][1]
    a2 = pg.SobolevForm(I, 2, length_scale=l)
    assert a2.conditions("right") == [((0.0, 0.0, 1.0), False), ((0.0, 2 * l * l, 0.0, -l ** 4), False)]
    sp = Spectral(pg.SobolevForm(I, 1, length_scale=l).operator(), 2)      # spectral Neumann, s = 2: d_n u = 0
    assert [c for c, _ in sp.element_conditions("left")] == [(0.0, l * l)]
    assert Spectral(pg.SobolevForm(I, 1, length_scale=l).operator(), 1.4).element_conditions("left") == []
    sd = Spectral(pg.SobolevForm(I, 1, length_scale=l, dirichlet="all").operator(), 0.75)
    assert [c for c, _ in sd.element_conditions("right")] == [(1.0,)]     # u = 0 for s >= 1/2
    assert "Kato" in repr(Spectral(a.operator(), 1).claims) and "Kato" in repr(A.claims)
    assert "Robin 0.3 at right" in repr(a)
    for bad, err in [(dict(k=1.5), ValueError), (dict(k=1, dirichlet="top"), ValueError),
                     (dict(k=1, robin={"left": -1.0}), ValueError), (dict(k=2, dirichlet="left"), Unsupported),
                     (dict(k=1, dirichlet="left", robin={"left": 1.0}), ValueError)]:
        k = bad.pop("k")
        with pytest.raises(err):
            pg.SobolevForm(I, k, length_scale=l, **bad)
    with pytest.raises(ValueError, match="the form's"):
        pg.Sobolev(a.operator(), 1, length_scale=2 * l)
    assert math.isclose(Spectral(a.operator(), 3).length_scale, l)
