"""Plan 067 P3: subsets of a space under the one Set root (architecture review 2026-11-13, B1, A7 and the author's
amendment 1).

    Subset(Set) with its space (sets/subsets.py): `x in S` is bool(S.member(x)), raising on UNKNOWN; & and | with the
        one EmptySet; subsets of different spaces refused
    ConvexSet(Subset) (convex/): support(v) is an Enclosure whose lower end is a lower bound and whose upper end an
        upper bound of sigma_S(v) (A7); support_bracket is gone; the set algebra composes the ends in their direction
        and refuses a plain number; interval(k) and FeasibleImage.interval report the certified ends
    LinearSubspace(ConvexSet), ClosedSubspace (project, perp) in convex/subspaces.py (relations/subspaces.py is gone)
    HardenedSet is a Subset; NormBall.member takes arrays like Ellipsoid; pg.Ball(space, r) points to pg.NormBall
"""
import importlib
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import MembershipUndecided, SpaceMismatch
from pygeoinf0.core.logic import Truth
from pygeoinf0.core.numbers import Enclosure
from pygeoinf0.sets import EmptySet as RootEmpty, Set

subsets = importlib.import_module("pygeoinf0.sets.subsets")
from pygeoinf0.convex import ConvexSet  # noqa: E402
from pygeoinf0.sets import PredicateSubset, Subset  # noqa: E402
from pygeoinf0.convex import subspaces as csub  # noqa: E402
from pygeoinf0.convex.algebra import CartesianProduct, Image  # noqa: E402
from pygeoinf0.optimisers.dual import Bound  # noqa: E402
from pygeoinf0.convex.pointwise import PointwiseBox  # noqa: E402

D2 = pg.Euclidean(2)
D3 = pg.Euclidean(3, covariance=np.diag([0.5, 1.0, 2.0]))


# ------------------------------------------------------------------ the root
def test_subset_is_a_set_with_its_space_as_ambient():
    assert subsets.Subset is Subset and issubclass(Subset, Set)
    assert issubclass(ConvexSet, Subset) and ConvexSet.__module__ == "pygeoinf0.convex.sets"
    assert PredicateSubset.__module__ == "pygeoinf0.sets.subsets"
    B = pg.NormBall(D2, 1.0)
    assert B.ambient is D2 and isinstance(B, Set)


def test_in_is_member_true_and_raises_on_unknown():
    B = pg.NormBall(D2, 1.0)
    assert D2([0.3, 0.4]) in B and D2([1.0, 1.0]) not in B
    S = PredicateSubset(D2, lambda x: Truth.UNKNOWN)
    with pytest.raises(MembershipUndecided):
        D2([0.0, 0.0]) in S
    assert pg.Ellipsoid(D2, 1.0, [1.0, 4.0]).member(np.array([0.0, 1.9])) is Truth.TRUE


def test_normball_member_takes_arrays_like_ellipsoid():
    B, E = pg.NormBall(D2, 1.0), pg.Ellipsoid(D2, 1.0)
    for p in ([0.3, 0.4], [0.8, 0.8], [1.0, 0.0]):
        assert B.member(np.array(p)) is E.member(np.array(p)) is B.member(D2(p))
    Bc = pg.NormBall(D3, 2.0, centre=D3([1.0, 0.0, 0.0]))
    assert Bc.member(np.array([1.0, 0.0, 0.0])) is Truth.TRUE


def test_and_or_with_the_one_empty_set_and_space_mismatch():
    B, C = pg.NormBall(D2, 1.0), pg.CoordinateBox(D2, -0.5, 0.5)
    E = pg.EmptySet(D2)
    assert isinstance(E, RootEmpty) and isinstance(E, ConvexSet)
    assert (B & E) is E and (E & B) is E and (B | E) is B and (E | B) is B
    K = B & C
    assert isinstance(K, Subset) and K.space is D2 and K.member(D2([0.1, 0.1])) is Truth.TRUE
    assert K.member(D2([0.9, 0.0])) is Truth.FALSE
    U = B | C
    assert isinstance(U, Subset) and U.member(D2([0.8, 0.8])) is Truth.FALSE and D2([0.45, 0.45]) in U
    P = PredicateSubset(D2, lambda x: x.values[0] > 0)
    assert isinstance(P & P, Subset) and (P & E) is E and (P | E) is P and isinstance(P & B, Subset)
    with pytest.raises(SpaceMismatch):
        B & pg.NormBall(pg.Euclidean(3), 1.0)
    with pytest.raises(SpaceMismatch):
        P | pg.NormBall(pg.Euclidean(3), 1.0)


# ------------------------------------------------------------------ A7: supports are Enclosures with directed ends
class _Bracketed(ConvexSet):
    """A ball of D2 whose support is stated as a bracket [s - 0.1, s + 0.2] with its value at the lower end (as a
    witness end): the composition must keep each end in its direction."""

    def __init__(self, r=1.0):
        super().__init__(D2, "bracketed")
        self.B = pg.NormBall(D2, r)

    def support(self, v):
        s = float(self.B.support(v))
        return Enclosure(s - 0.1, s - 0.1, s + 0.2, "derived", "a stated bracket")


def _contains(e, exact, tol=1e-12):
    if math.isinf(exact):
        return e.lower <= exact <= e.upper
    return e.lower - tol * max(1.0, abs(exact)) <= exact <= e.upper + tol * max(1.0, abs(exact))


def _kinds():
    X = pg.L2(pg.Interval(0, 1))
    T = pg.Matrix(np.array([[1.0, 2.0], [0.0, 1.0], [1.0, -1.0]]), D2, pg.Euclidean(3))
    B, C = pg.NormBall(D2, 1.5, centre=D2([0.2, -0.1])), pg.CoordinateBox(D2, [-1, -2], [1, 0.5])
    v = D2([0.6, -0.8])
    a = np.array([0.6, -0.8])
    out = [
        ("ball", B, v, 0.2 * 0.6 + 0.1 * 0.8 + 1.5),
        ("ellipsoid", pg.Ellipsoid(D2, 2.0, [1.0, 4.0]), v, 2.0 * math.sqrt(0.36 + 4 * 0.64)),
        ("box", C, v, 0.6 + 0.8 * 2),
        ("slab", pg.Slab(D2, D2([1.0, 0.0]), 0.5), D2([2.0, 0.0]), 1.0),
        ("polytope", pg.Polytope(D2, np.r_[np.eye(2), -np.eye(2)], np.ones(4)), v, 1.4),
        ("empty", pg.EmptySet(D2), v, -math.inf),
        ("translate", B + D2([1.0, 1.0]), v, 0.2 * 0.6 + 0.1 * 0.8 + 1.5 + 0.6 - 0.8),
        ("scale", 2.0 * C, v, 2 * (0.6 + 1.6)),
        ("minkowski", B + C, v, 0.2 * 0.6 + 0.1 * 0.8 + 1.5 + 2.2),
        ("union", B | C, v, max(0.2 * 0.6 + 0.1 * 0.8 + 1.5, 2.2)),
        ("image", Image(T, pg.NormBall(D2, 1.0)), pg.Euclidean(3)([1.0, 0.0, 1.0]), math.sqrt(4 + 1)),
        ("product", CartesianProduct([pg.NormBall(pg.Euclidean(1), 1.0), pg.NormBall(pg.Euclidean(1), 2.0)], D2,
                                     [(0, 1), (1, 2)]), v, 0.6 + 1.6),
        ("whole", csub.Whole(D2), v, math.inf),
        ("zero", csub.Zero(D2), v, 0.0),
        ("span", csub.Span([D2([1.0, 1.0])]), D2([1.0, -1.0]), 0.0),
        ("span off", csub.Span([D2([1.0, 1.0])]), v, math.inf),
        ("affine", pg.AffineSubset(D2([1.0, 2.0]), csub.Span([D2([1.0, 1.0])])), D2([1.0, -1.0]), -1.0),
        ("interval box", PointwiseBox(X, lower=-1.0, upper=2.0), X(lambda x: x - 0.25), 2 * 0.28125 + 0.03125),
        ("metric", pg.NormBall(D3, 1.0), D3([1.0, 0.0, 1.0]), math.sqrt(0.5 + 2.0)),
    ]
    return out


@pytest.mark.parametrize("name, S, v, exact", _kinds(), ids=[k[0] for k in _kinds()])
def test_every_support_is_an_enclosure_of_sigma(name, S, v, exact):
    e = S.support(v)
    assert isinstance(e, Enclosure), (name, type(e))
    assert _contains(e, exact, 1e-9), (name, e, exact)


def test_support_bracket_is_gone():
    X = pg.L2(pg.Interval(0, 1))
    B = PointwiseBox(X, envelope=1.0)
    assert not hasattr(B, "support_bracket") and not hasattr(B.projected(pg.Identity(X)), "support_bracket")
    e = B.support(X(lambda x: x - 0.5), rtol=1e-6)
    assert isinstance(e, Enclosure) and e.lower == e.upper == pytest.approx(0.25, abs=1e-12)


def test_the_algebra_keeps_the_direction_of_every_end():
    v = D2([0.6, -0.8])
    s = 1.0                                               # sigma of the unit ball in a unit direction
    W, B = _Bracketed(), pg.NormBall(D2, 2.0)
    m = (W + B).support(v)
    assert m.lower <= s - 0.1 + 2.0 <= s + 2.0 <= m.upper and m.upper >= s + 0.2 + 2.0
    t = (W + D2([1.0, 0.0])).support(v)
    assert t.lower <= s - 0.1 + 0.6 and t.upper >= s + 0.2 + 0.6
    c = (3.0 * W).support(v)
    assert c.lower <= 3 * (s - 0.1) and c.upper >= 3 * (s + 0.2)
    u = (W | pg.NormBall(D2, 0.5)).support(v)
    assert u.lower <= s - 0.1 and u.upper >= s + 0.2
    P = CartesianProduct([W, W], pg.Euclidean(4), [(0, 2), (2, 4)])
    p = P.support(pg.Euclidean(4)([0.6, -0.8, 0.0, 1.0]))
    assert p.lower <= 2 * (s - 0.1) and p.upper >= 2 * (s + 0.2)


def test_a_plain_number_support_is_refused_by_the_algebra():
    class Plain(ConvexSet):
        def support(self, v):
            return 1.0
    with pytest.raises(TypeError, match="Enclosure"):
        (Plain(D2) + pg.NormBall(D2, 1.0)).support(D2([1.0, 0.0]))


def test_interval_reports_certified_ends():
    W = _Bracketed()
    lo, hi = W.interval(0)
    assert lo == pytest.approx(-1.2) and hi == pytest.approx(1.2)        # the outer ends, not the witness values
    X, G, d = _bgp()
    P = pg.push(pg.update(pg.NormBall(X, 1.0), pg.Additive(G, pg.NormBall(G.codomain, 0.05)), d),
                pg.Matrix(np.array([[1.0, 0.0, 0.0, 0.0]]), X, pg.Euclidean(1)))
    brk = {1.0: Bound(0.4, 0.4, 0.5, "BRACKET"), -1.0: Bound(0.1, 0.1, 0.3, "BRACKET")}
    P.bound = lambda q: brk[float(np.ravel(getattr(q, "values", q))[0])]
    assert P.interval(0) == (-0.3, 0.5)
    assert isinstance(P.support(np.array([1.0])), Enclosure)


def _bgp():
    X = pg.Euclidean(4)
    G = pg.Matrix(np.array([[1.0, 0.5, 0.0, 0.2], [0.0, 1.0, 1.0, -0.3]]), X, pg.Euclidean(2))
    return X, G, pg.Euclidean(2)([0.3, -0.2])


def test_feasible_supports_are_bounds_with_a_stated_status():
    X, G, d = _bgp()
    K = pg.update(pg.NormBall(X, 1.0), pg.Additive(G, pg.NormBall(G.codomain, 0.05)), d)
    e = K.support(X([1.0, 0.0, 0.0, 0.0]))
    assert isinstance(e, Bound) and e.how == "EXACT" and e.enclosure.status == "checked"
    assert e.lower - 1e-12 <= e.upper           # each end to the rounding of its evaluation: EXACT ends may cross by it
    box = PointwiseBox(pg.L2(pg.Interval(0, 1)), envelope=1.0)
    assert box.support(box.space(lambda x: x)).status != "unknown"


# ------------------------------------------------------------------ subspaces in convex/
def test_linear_and_closed_subspaces_live_in_convex():
    with pytest.raises(ImportError):
        importlib.import_module("pygeoinf0.relations.subspaces")
    assert issubclass(csub.LinearSubspace, ConvexSet) and issubclass(csub.ClosedSubspace, csub.LinearSubspace)
    for cls in (csub.Whole, csub.Zero, csub.Span, csub.Complement):
        assert issubclass(cls, csub.ClosedSubspace)
    S = csub.Span([D2([1.0, 1.0])])
    x = D2([2.0, 0.0])
    assert S.project(x).values == pytest.approx([1.0, 1.0]) and S.perp().project(x).values == pytest.approx([1, -1])
    assert D2([3.0, 3.0]) in S and D2([3.0, 0.0]) not in S
    assert (S & pg.EmptySet(D2)) == pg.EmptySet(D2)
    assert pg.Span is csub.Span and pg.Whole is csub.Whole and pg.Complement is csub.Complement
    from pygeoinf0.relations.linear import OperatorDomain
    assert issubclass(OperatorDomain, csub.LinearSubspace) and not issubclass(OperatorDomain, csub.ClosedSubspace)


# ------------------------------------------------------------------ HardenedSet and pg.Ball
def test_hardened_set_is_a_subset():
    M = pg.L2("[0, 1]")
    n = np.arange(1, 11)
    phi = M.family(lambda x, k: np.sqrt(2) * np.cos(k * np.pi * x), k=n)
    mu = pg.GaussianMeasure(M, pg.Covariance.spectral(1.0 / n ** 2, phi, tail=0.1))
    S = mu.harden(pg.Identity(M), 0.9)
    assert isinstance(S, Subset) and not isinstance(S, ConvexSet) and S.ambient is M
    assert M.zero in S
    assert (S & pg.EmptySet(M)) == pg.EmptySet(M)


def test_ball_of_a_space_points_to_normball():
    with pytest.raises(TypeError, match="NormBall"):
        pg.Ball(pg.Euclidean(2), 1)
    with pytest.raises(TypeError, match="NormBall"):
        pg.Ball(pg.L2(pg.Interval(0, 1)))
    assert pg.Ball(2.0).radius == 2.0


@pytest.mark.parametrize("cx", [-5.0, 5.0])
def test_general_noise_route_lower_end_includes_the_prior_centre(cx):
    """Found by the A7 law: the general noise route (a box of noise) formed its witness end without <c, m0>, the
    prior centre's part, so with a centred prior the lower end was off by it (5.1 reported EXACT for sup m_0 = 0.1)."""
    X = pg.Euclidean(2)
    G = pg.Matrix(np.eye(2), X, pg.Euclidean(2))
    D = G.codomain
    K = pg.update(pg.NormBall(X, 10.0, centre=X([cx, 0.0])), pg.Additive(G, pg.CoordinateBox(D, -0.1, 0.1)),
                  D([0.0, 0.0]))
    b = K.support(X([1.0, 0.0]))
    assert b.lower <= 0.1 + 1e-9 <= b.upper + 2e-9 and b.lower == pytest.approx(0.1, abs=1e-7)
