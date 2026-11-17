"""Space identity without realisation (plan 069 P1; the architecture review's C1; decision Q4).

A function space is the same space whatever realises its elements: X == Y by the measure (up to null sets) and the
norm construction, never by the backend. Elements realised differently are elements of one space; mixing them
numerically is refused with RealisationMismatch naming both realisations, never a SpaceMismatch and never silently.
"""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import RealisationMismatch, SpaceMismatch
from pygeoinf0.core.logic import Truth
from pygeoinf0.realisations import Cells

B = pg.Box((0, 1), (0, 1))


def _pair(make, a, b):
    X = make(a)
    Y = make(b)
    f = X(lambda x, y: x * y, resolution=0.5)
    g = Y(lambda x, y: x + y, resolution=0.5)
    return X, Y, f, g


def _refused(fn, *names):
    with pytest.raises(RealisationMismatch) as e:
        fn()
    assert not isinstance(e.value, SpaceMismatch)
    for n in names:
        assert n in str(e.value), (n, str(e.value))
    return e.value


def test_elements_realised_differently_belong_to_one_space():
    X, Y, f, g = _pair(lambda r: pg.L2(B, realisation=r), "local", "tensor")
    assert X == Y and Y == X and hash(X) == hash(Y)
    assert X.realisation == "local" and Y.realisation == "tensor"
    assert X.member(g) is Truth.TRUE and Y.member(f) is Truth.TRUE
    assert g in X and f in Y


def test_realised_is_a_view_of_the_same_space():
    X = pg.L2(B)
    Y = X.realised("tensor")
    assert Y == X and Y.realisation == "tensor" and X.realisation == "local"
    assert X.realised("local") is X
    C = X.realised(Cells.tensor(np.linspace(0, 1, 3), np.linspace(0, 1, 4)))
    assert C == X and C.realisation.startswith("cells(")
    H = pg.Sobolev(B, 1, length_scale=0.3)
    assert H.realised("tensor") == H and H.realised("tensor").realisation == "tensor"
    P = pg.Lp(B, 3)
    assert P.realised("tensor") == P and P.realised("tensor").p == 3
    with pytest.raises(ValueError):
        X.realised("harmonics")                    # not a realisation of a box: refused by the registry


@pytest.mark.parametrize("make", [lambda r: pg.L2(B, realisation=r), lambda r: pg.Lp(B, 3, realisation=r),
                                  lambda r: pg.Sobolev(B, 1, length_scale=0.3, realisation=r)],
                         ids=["L2", "L3", "H1"])
def test_equality_is_independent_of_realisation(make):
    X, Y = make("local"), make("tensor")
    assert X == Y and hash(X) == hash(Y)
    assert X.dual() == Y.dual()


def test_unequal_spaces_still_mismatch():
    X = pg.L2(B)
    Z = pg.L2(pg.Box((0, 2), (0, 1)), realisation="tensor")
    f = X(lambda x, y: x * y, resolution=0.5)
    g = Z(lambda x, y: x + y, resolution=0.5)
    with pytest.raises(SpaceMismatch):
        f + g
    assert X != pg.Lp(B, 3) and X != pg.Sobolev(B, 1, length_scale=0.3)


def test_mixing_realisations_is_refused_naming_both():
    X, Y, f, g = _pair(lambda r: pg.L2(B, realisation=r), "local", "tensor")
    for fn in (lambda: f + g, lambda: g - f, lambda: X.inner(f, g), lambda: X.norm(g), lambda: X.element(g),
               lambda: X.linear_combination([1.0, 2.0], [f, g]), lambda: X.integrate(g),
               lambda: X.family([f, g]), lambda: X.sum([f, g]), lambda: f == g):
        e = _refused(fn, "local", "tensor")
        assert e.needs, "the refusal names the choices"
    F = Y.family([g, 2 * g])
    _refused(lambda: F + f, "local", "tensor")


def test_operators_refuse_elements_realised_differently():
    X, Y, f, g = _pair(lambda r: pg.L2(B, realisation=r), "local", "tensor")
    A = pg.Identity(X) if hasattr(pg, "Identity") else None
    if A is None:
        from pygeoinf0.operators.linear import Identity
        A = Identity(X)
    assert A(f) is not None
    _refused(lambda: A(g), "local", "tensor")
    AY = A.__class__(Y)
    _refused(lambda: A @ AY, "local", "tensor")
    ell = X.average(B)
    assert np.isfinite(ell(f))
    _refused(lambda: ell(g), "local", "tensor")


def test_same_realisation_still_combines():
    X, Y, f, g = _pair(lambda r: pg.L2(B, realisation=r), "tensor", "tensor")
    assert X == Y
    assert abs(X.inner(f, g) - 1 / 3) < 1e-12            # int_0^1 int_0^1 xy (x + y) = 1/3


def test_direct_sums_and_complexifications():
    X, Y = pg.L2(B), pg.L2(B, realisation="tensor")
    S, T = pg.DirectSum([X, X]), pg.DirectSum([Y, X])
    assert S == T and hash(S) == hash(T)
    C1 = pg.Complexified(X)
    P = Cells.tensor(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    C2 = pg.Complexified(pg.L2(B, realisation=P))
    assert type(C1) is not type(C2)
    assert C1 == C2 and C2 == C1 and hash(C1) == hash(C2)
    Q = Cells.tensor(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
    C3 = pg.Complexified(pg.L2(B, realisation=Q))
    assert C2 == C3
    z2 = C2(lambda x, y: x + 1j * y)
    z3 = C3(lambda x, y: x + 1j * y)
    _refused(lambda: z2 + z3, P.name, Q.name)


# ---------------------------------------------------------------- symbolic representers (plan 071 P3, review item 10)
# A symbolic representer (X.riesz(measure)) is a measure, not a realisation: <X.riesz(mu), g> = int g dmu is read from
# g's own realisation, so it is the value of the same pairing in g's space whichever realisation X has (never an
# AttributeError from X's backend reading g's coefficients).
SUB = pg.Box((0, 0.5), (0, 0.5))


@pytest.mark.parametrize("a, b", [("local", "cells"), ("cells", "local"), ("local", "tensor"), ("tensor", "cells")])
def test_symbolic_representer_pairs_with_an_element_of_any_realisation(a, b):
    def make(r):
        return pg.L2(B, realisation=Cells.tensor(np.linspace(0, 1, 3), np.linspace(0, 1, 4)) if r == "cells" else r)
    A, Y = make(a), make(b)
    g = Y(lambda x, y: x * y, resolution=0.5)
    want = Y.inner(Y.riesz(SUB), g)                     # the pairing in g's own space
    assert A.inner(A.riesz(SUB), g) == want and A.inner(g, A.riesz(SUB)) == want
    if b != "cells":
        assert want == pytest.approx(1 / 64, rel=1e-12)  # int_0^1/2 int_0^1/2 xy = 1/64
    fam = Y.family([g, 2 * g])
    two = A.inner(A.riesz([SUB, B]), fam)
    assert np.array_equal(two, Y.inner(Y.riesz([SUB, B]), fam))
    assert A.backend.last_pairing_error is not None     # the estimate is kept where the caller reads it (M.backend)


def test_two_realised_elements_still_refuse():
    A, C = pg.L2(B), pg.L2(B, realisation=Cells.tensor(np.linspace(0, 1, 3), np.linspace(0, 1, 4)))
    f, g = A(lambda x, y: x * y, resolution=0.5), C(lambda x, y: x * y, resolution=0.5)
    _refused(lambda: A.inner(f, g), "local", C.realisation)
