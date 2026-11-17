"""Plan 054, families keep the estimable mark (the author's decision of 2026-11-06): arithmetic on a Family or a
VectorFamily keeps the record of how it was made under the same rules as elements (core/estimable.py): a sum keeps what
all its terms share, a multiple keeps it, anything else drops it.

References are exact: for c = G* y over the noise ball RHO with no prior the bound is y.d + RHO |y| (mpmath, 60
digits); a target not built from the data functionals has the infinite bracket. Each test has a planted-defect control
(the rule as it was before: family arithmetic dropping the record)."""
import math

import mpmath as mp
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.convex.feasible import FeasibleSet
from pygeoinf0.core import estimable
from pygeoinf0.spaces import elements

TOL, RHO = 1e-9, 0.1


def _problem(seed=2):
    n, m = 8, 20
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    V = np.linalg.qr(rng.standard_normal((m, m)))[0]
    G = (U * np.geomspace(1, 0.5, n)) @ V[:, :n].T
    d = G @ rng.standard_normal(m) + 0.3 * RHO * rng.standard_normal(n) / math.sqrt(n)
    X, D = pg.Euclidean(m), pg.Euclidean(n)
    return X, D, pg.Matrix(G, X, D), d, rng


def _exact(coef, d):
    with mp.workdps(60):
        c = [mp.mpf(float(x)) for x in coef]
        return float(mp.fsum(ci * mp.mpf(float(di)) for ci, di in zip(c, d)) + RHO * mp.sqrt(mp.fsum(ci * ci for ci in c)))


def _bound(Gop, D, d, c):
    return FeasibleSet(None, Gop, pg.NormBall(D, RHO), D(d)).bound(c)


def _drop_family_records(monkeypatch):
    """The planted defect: family arithmetic as before this change (a new family with no record)."""
    monkeypatch.setattr(elements.VectorFamily, "_made", lambda self, values, other=None:
                        elements.VectorFamily(self.space, values, name=self.name))
    monkeypatch.setattr(elements.Family, "_broadcast", lambda self, other, a, b: elements.Family(
        self.space, self.space.backend.lincomb([(a, self._pl), (b, other.realisation)]), name=self.name))
    mul = lambda self, c: elements.Family(self.space, self.space.backend.lincomb([(float(c), self._pl)]),  # noqa: E731
                                          name=self.name)
    monkeypatch.setattr(elements.Family, "__mul__", mul)
    monkeypatch.setattr(elements.Family, "__rmul__", mul)


def test_vector_family_arithmetic_keeps_the_mark(monkeypatch):
    """F = G*(Y) for a family Y of data coefficients: members of 2F - 0.5 G*(Y2) + G*(y), -F and F + F are estimable by
    construction, EXACT at y.d + RHO |y|; F + an unmarked vector (G* y's numbers remade) is not (the infinite bracket). Control: with the
    family records dropped (the rule before), the combined members get the infinite bracket."""
    X, D, Gop, d, rng = _problem()
    Y, Y2, y = rng.standard_normal((3, 8)), rng.standard_normal((3, 8)), rng.standard_normal(8)
    F = Gop.adjoint(D.family(Y))
    cases = [(2.0 * F - 0.5 * Gop.adjoint(D.family(Y2)) + Gop.adjoint(D(y)), 2 * Y - 0.5 * Y2 + y), (-F, -Y), (F + F, 2 * Y)]
    for fam, coefs in cases:
        for i in range(3):
            b = _bound(Gop, D, d, fam[i])
            ref = _exact(coefs[i], d)
            assert b.status == "EXACT" and abs(b.value - ref) <= TOL * max(1.0, abs(ref)) and "by construction" in b.note
    mixed = F + X(np.array(Gop.adjoint(D(y)).values))                  # the same numbers, remade: unmarked
    b = _bound(Gop, D, d, mixed[0])
    assert b.status == "BRACKET" and math.isinf(b.upper) and "neither declared" in b.note
    _drop_family_records(monkeypatch)
    F = Gop.adjoint(D.family(Y))
    b0 = _bound(Gop, D, d, (2.0 * F)[0])
    assert b0.status == "BRACKET" and math.isinf(b0.upper)


def test_function_family_arithmetic_keeps_the_mark(monkeypatch):
    """Kernels in L2[0, 1] (Functionals of Gaussian bumps): F = G*(Y); 2F, F + F, F - 0.5 F2, F.sum() and F.mean() keep
    G*'s record, so their members are estimable by construction; F + 1 and F + x (unmarked) drop it, as for elements.
    Control: the family records dropped, 2F's members lose it."""
    M = pg.L2(pg.LebesgueMeasure(pg.Interval(0, 1)))
    centres = np.linspace(0.1, 0.9, 8)
    k = M.family(lambda x, c: np.exp(-((x - c) / 0.1) ** 2), c=centres)
    Gf = pg.Functionals(k)
    rng = np.random.default_rng(0)
    Dd = Gf.codomain
    F, F2 = Gf.adjoint(Dd.family(rng.standard_normal((3, 8)))), Gf.adjoint(Dd.family(rng.standard_normal((3, 8))))
    key = Gf.adjoint._id
    for v in (2.0 * F, F + F, F - 0.5 * F2, (2.0 * F)[1], F.sum(), F.mean(), F + Gf.adjoint(Dd(np.ones(8)))):
        assert key in estimable.of(v)
    for v in (F + 1.0, F + M(lambda x: x, breaks=[]), F - M(lambda x: x, breaks=[])):
        assert key not in estimable.of(v)
    _drop_family_records(monkeypatch)
    assert key not in estimable.of(2.0 * F) and key not in estimable.of((F + F)[0])


@pytest.mark.parametrize("route", [None, False])
def test_a_family_target_through_a_feasible_set(route, monkeypatch):
    """The same in both routes (decomposition and reduction): the member of a combined family built from G* is EXACT
    at the exact value; with the records dropped it is the infinite bracket."""
    X, D, Gop, d, rng = _problem(seed=5)
    Y = rng.standard_normal((2, 8))
    fam = 3.0 * Gop.adjoint(D.family(Y)) - Gop.adjoint(D.family(Y[::-1]))
    coefs = 3.0 * Y - Y[::-1]
    K = FeasibleSet(None, Gop, pg.NormBall(D, RHO), D(d), decomposition=route)
    b = K.bound(fam[0])
    ref = _exact(coefs[0], d)
    assert b.status == "EXACT" and abs(b.value - ref) <= TOL * max(1.0, abs(ref))
    _drop_family_records(monkeypatch)
    fam = 3.0 * Gop.adjoint(D.family(Y)) - Gop.adjoint(D.family(Y[::-1]))
    K = FeasibleSet(None, Gop, pg.NormBall(D, RHO), D(d), decomposition=route)
    b = K.bound(fam[0])
    assert b.status == "BRACKET" and math.isinf(b.upper)
