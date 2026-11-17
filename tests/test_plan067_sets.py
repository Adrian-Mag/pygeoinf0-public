"""Plan 067 P2, the Set root (architecture review 2026-11-13, B1 for Set and Region, B7, decision Q3).

- A package sets/ (layer L1) holds the root: Set (member(x) -> Truth; `x in S` is bool(member) and raises on UNKNOWN;
  & and |; one EmptySet(ambient) that is the identity of | and absorbing for &), Region(Set) (a subset of R^d with its
  dimension, its Hausdorff measure `volume` and contains(points) on arrays of shape (..., d)), Interval, RealLine,
  FiniteSet, the intersections and the restriction of measures (from core/sets.py, core/intersections.py,
  core/restrict.py: a clean break, no alias modules).
- One point convention for every Region: points are arrays of shape (..., d) and the answer has shape (...), d = 1
  included (n points of R^1 are (n, 1); shape (1,) is one point; a number is one point; a flat array of n >= 2 numbers
  is refused). Domain.contains(*x) is gone.
- & never returns None: Interval & Interval and Domain & Domain give a Region (exact where it is constructed, else the
  intersection known by membership) or the EmptySet of the common ambient.
- Domains, Surface, Simplicial and LayeredBall are Regions; a Partition's union is a Region; VectorSpace is a Set.
"""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import quadrature
from pygeoinf0.core.logic import Truth
from pygeoinf0.core.errors import SpaceMismatch
from pygeoinf0.sets import EmptySet, FiniteSet, Interval, R, RealLine, RealSpace, Region, Set


def regions():
    """(name, region, sample points (n, d) and points that are surely outside (m, d))."""
    S2 = pg.Sphere(1.0)
    ball = pg.Ball(1.0)
    piece = ball & pg.Ball(0.6, centre=(0.8, 0.0, 0.0))
    out = []

    def add(name, D, P, far=None):
        P = np.asarray(P, float)
        d = P.shape[1]
        far = np.full((3, d), 50.0) * np.array([1.0, -1.0, 2.0])[:, None] if far is None else far
        out.append((name, D, P, np.asarray(far, float)))

    I = Interval("[0, 1)")
    add("interval", I, np.array([[0.0], [0.3], [0.999], [1.0], [-0.1]]))
    add("real line", R, np.array([[0.0], [-3.0], [1e300]]), far=np.array([[np.inf], [-np.inf], [np.nan]]))
    add("finite set", FiniteSet([0.1, 0.5]), np.array([[0.1], [0.5], [0.2]]))
    add("finite set in R^2", FiniteSet([(0.2, 0.3), (1.0, 0.0)]), np.array([[0.2, 0.3], [1.0, 0.0], [0.3, 0.2]]))
    for name, D in (("box", pg.Box((0, 1), (0, 2))), ("box3", pg.Box((0, 1), (0, 1), (0, 1))), ("disk", pg.Disk(1.0)),
                    ("ball", ball), ("shell", pg.Shell(0.5, 1.0)), ("cylinder", pg.Cylinder(1.0)),
                    ("sphere", S2), ("circle", pg.Circle(2.0)), ("torus", pg.Torus((0, 1), (0, 1))),
                    ("segment", pg.Segment((0, 0), (1, 1))), ("polyline", pg.Polyline([(0, 0), (1, 0), (1, 1)])),
                    ("patch", pg.Patch((0, 0, 0), (1, 0, 0), (0, 1, 0))),
                    ("arc", pg.Arc(S2, (1, 0, 0), (0, 1, 0))), ("band", S2.band(0.5, 1.0)),
                    ("ball piece", piece), ("curve", pg.Circle(1.0).arc(0.0, 1.0)),
                    ("simplicial", pg.Simplex((0, 0), (1, 0), (0, 1)))):
        X = np.asarray(quadrature.rule(D, 3)[0], float)
        # the flat torus is R^d modulo its periods: every point of R^d is one of its points, none is outside
        add(name, D, np.vstack([X, X.mean(axis=0) + 0.7]), far=np.empty((0, X.shape[1])) if name == "torus" else None)
    L = pg.LayeredBall([0.5, 1.0])
    add("layered ball", L, np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.0, 1.2]]))
    return out


IDS = [r[0] for r in regions()]


# ------------------------------------------------------------------ the package
def test_the_set_root_lives_in_sets():
    import importlib
    import pygeoinf0.sets as S
    assert issubclass(Region, Set) and all(issubclass(c, Region) for c in (Interval, RealLine, FiniteSet, RealSpace))
    assert pg.Interval is S.Interval and pg.FiniteSet is S.FiniteSet and pg.R is S.R and pg.RealLine is S.RealLine
    for gone in ("pygeoinf0.core.sets", "pygeoinf0.core.intersections", "pygeoinf0.core.restrict"):
        with pytest.raises(ImportError):
            importlib.import_module(gone)                                       # a clean break: no alias modules
    from pygeoinf0.sets.intersections import intersection                      # noqa: F401
    from pygeoinf0.measures.restrict import restrict                                 # noqa: F401
    for D in (pg.Box((0, 1), (0, 1)), pg.Sphere(), pg.Simplex((0, 0), (1, 0), (0, 1)), pg.LayeredBall([0.5, 1.0]),
              pg.Surface(lambda u, v: np.stack(np.broadcast_arrays(u, v, 0 * u), -1), (0, 1), (0, 1))):
        assert isinstance(D, Region)
    assert isinstance(pg.L2("[0, 1]"), Set) and isinstance(pg.Euclidean(3), Set)


# ------------------------------------------------------------------ membership
@pytest.mark.parametrize("name, D, P, far", regions(), ids=IDS)
def test_in_is_member_true(name, D, P, far):
    """x in S <=> S.member(x) is TRUE, for points inside, on the boundary and outside."""
    for p in np.vstack([P, far]):
        t = D.member(p)
        assert isinstance(t, Truth)
        assert (p in D) == (t is Truth.TRUE)
    for p in far:
        assert D.member(p) is Truth.FALSE


@pytest.mark.parametrize("name, D, P, far", regions(), ids=IDS)
def test_one_point_convention(name, D, P, far):
    """contains(points): arrays of shape (..., d), the answer of shape (...); agrees with member point by point."""
    X = np.vstack([P, far])
    d = X.shape[1]
    r = D.contains(X)
    assert r.shape == (len(X),) and r.dtype == bool
    assert [bool(v) for v in r] == [D.member(p) is Truth.TRUE for p in X]
    Y = np.stack([X[:2], X[2:4]])                                            # (2, 2, d)
    assert D.contains(Y).shape == (2, 2) and np.array_equal(D.contains(Y), r[:4].reshape(2, 2))
    assert D.contains(X[:1]).shape == (1,)
    assert isinstance(D.contains(X[0]), bool)                                 # one point of R^d: shape (d,)
    with pytest.raises(ValueError):
        D.contains(X[:, :-1])                                                 # not points of R^d
    if d > 1:
        with pytest.raises((TypeError, ValueError)):
            D.contains(*X[0])                                                 # Domain.contains(*x) is gone


def test_points_of_the_real_line():
    """R^1 has the convention of every R^d: n points are an array of shape (n, 1), shape (1,) is one point; a number is
    one point; a flat array of n >= 2 numbers is refused (never read silently as n points)."""
    I = Interval("[0, 1]")
    assert I.contains(0.5) is True and I.contains(2.0) is False
    assert I.contains(np.array([-1, 0, 1, 2])[:, None]).tolist() == [False, True, True, False]
    assert I.contains(np.array([0.5])) is True                                # shape (1,): one point
    assert I.contains(np.array([[0.5], [2.0]])).tolist() == [True, False]
    assert I.contains(np.zeros((2, 3, 1))).shape == (2, 3)
    with pytest.raises(ValueError, match=r"t\[:, None\]"):
        I.contains(np.array([0.1, 0.2]))                                      # two numbers: t[:, None] says so
    assert I.member(0.5) is Truth.TRUE and I.member(np.array([0.5])) is Truth.TRUE
    with pytest.raises(ValueError):
        I.member(np.array([0.1, 0.2]))                                        # two numbers are not one point
    assert 0 in I and 0.5 in I and 2 not in I


def test_in_raises_on_unknown():
    S = pg.Surface(lambda u, v: np.stack(np.broadcast_arrays(np.cos(u), np.sin(u), v), -1), (0, 1), (0, 1))
    p = np.array([1.0, 0.0, 0.5])
    assert S.member(p) is Truth.UNKNOWN
    with pytest.raises(TypeError):
        p in S                                                                # noqa: B015
    both = S & pg.Ball(2.0)
    assert both.member(p) is Truth.UNKNOWN and both.member(np.array([5.0, 0, 0])) is Truth.FALSE
    with pytest.raises(TypeError):
        p in both                                                             # noqa: B015


def test_vector_spaces_are_sets():
    M, E = pg.L2("[0, 1]"), pg.Euclidean(2)
    f, v = M(np.sin), E(np.array([1.0, 2.0]))
    assert M.member(f) is Truth.TRUE and M.member(v) is Truth.FALSE and E.member(v) is Truth.TRUE
    assert f in M and v not in M and v in E
    assert not hasattr(M, "contains")                                         # member is the protocol


# ------------------------------------------------------------------ & and |
def test_interval_and_interval_is_never_none():
    a, b = Interval("[0, 1]"), Interval("[0.5, 2)")
    assert a & b == Interval("[0.5, 1]")
    assert Interval("[0, 1)") & Interval("(0.2, 3]") == Interval("(0.2, 1)")
    t = Interval("[0, 1]") & Interval("[1, 2]")
    assert t == FiniteSet([1.0]) and t.dim == 0 and t.volume == 1                # a point: H^0 counts it
    for x, y in (("[0, 1)", "[1, 2]"), ("[0, 1]", "(1, 2]"), ("[0, 1]", "[3, 4]")):
        e = Interval(x) & Interval(y)
        assert isinstance(e, EmptySet) and e.ambient == R and e.volume == 0.0


def test_domain_and_domain_is_never_none():
    B1, B2 = pg.Box((0, 1), (0, 1)), pg.Box((0.5, 2), (0.5, 2))
    assert B1 & B2 == pg.Box((0.5, 1), (0.5, 1))
    assert isinstance(B1 & pg.Box((2, 3), (0, 1)), EmptySet)
    face = B1 & pg.Box((1, 2), (0, 1))                                          # touching: the segment x = 1
    assert isinstance(face, Region) and face.member(np.array([1.0, 0.5])) is Truth.TRUE
    assert face.member(np.array([0.5, 0.5])) is Truth.FALSE
    b, c = pg.Ball(1.0), pg.Ball(1.0, centre=(1.5, 0, 0))
    lens = b & c
    assert isinstance(lens, Region) and lens.contains(np.array([[0.75, 0, 0], [0, 0, 0]])).tolist() == [True, False]
    assert isinstance(b & pg.Ball(1.0, centre=(3, 0, 0)), EmptySet)
    mixed = b & pg.Box((0, 1), (0, 1), (0, 1))                                  # not constructed: known by membership
    assert isinstance(mixed, Region)
    assert mixed.contains(np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [-0.1, 0, 0]])).tolist() == [True, False, False]
    for S, T in ((pg.Disk(1.0), pg.Disk(0.5)), (pg.Sphere(), pg.Ball(2.0)), (pg.Shell(0.5, 1.0), pg.Ball(0.7))):
        assert (S & T) is not None and isinstance(S & T, (Region, EmptySet))
    with pytest.raises(SpaceMismatch):
        pg.Disk(1.0) & pg.Ball(1.0)                                            # R^2 and R^3


@pytest.mark.parametrize("name, D, P, far", regions(), ids=IDS)
def test_empty_set_is_the_identity_of_union_and_absorbs(name, D, P, far):
    E = EmptySet(D.ambient)
    for e in (D & E, E & D):
        assert isinstance(e, EmptySet) and e.ambient == D.ambient
    for u in (D | E, E | D):
        assert u is D
    assert E.member(P[0]) is Truth.FALSE and P[0] not in E


def test_union_and_intersection_follow_membership():
    a, b = Interval("[0, 1]"), Interval("[2, 3]")
    u = a | b
    assert isinstance(u, Region) and u.contains(np.array([[0.5], [1.5], [2.5]])).tolist() == [True, False, True]
    assert u.member(1.5) is Truth.FALSE and 2.5 in u
    with pytest.raises(SpaceMismatch):
        EmptySet(RealSpace(2)) & a
    assert RealSpace(1) == R and EmptySet(R) == EmptySet(RealSpace(1)) and EmptySet(R) != EmptySet(RealSpace(2))


# ------------------------------------------------------------------ the geometric protocol
def test_volume_is_the_hausdorff_measure():
    assert Interval(0, 2).volume == 2.0 and Interval(0, 2).dim == 1 and R.volume == math.inf
    F = FiniteSet([0.1, 0.4, 0.9])
    assert F.dim == 0 and F.volume == 3
    L = pg.LayeredBall([0.5, 1.0])
    assert L.dim == 3 and L.volume == pytest.approx(4 * math.pi / 3, rel=1e-14)
    assert pg.LebesgueMeasure(Interval(0, 2))(Interval(1, 5)) == 1.0
    assert pg.LebesgueMeasure(Interval(0, 2))(Interval(2, 5)) == 0.0             # a point: Lebesgue-null


def test_partition_union_is_a_region():
    P = pg.Partition("[0, 1]", ["[0, 0.4]", "(0.4, 1]"])
    U = P.union
    assert isinstance(U, Region) and U.dim == 1 and U.volume == pytest.approx(1.0)
    assert U.contains(np.array([[0.0], [0.4], [0.7], [1.2]])).tolist() == [True, True, True, False]
    Q = pg.LayeredBall([0.5, 1.0]).partition
    assert isinstance(Q.union, Region) and Q.union.volume == pytest.approx(4 * math.pi / 3, rel=1e-14)
    gap = pg.Partition("[0, 1]", ["[0, 0.4)", "(0.4, 1]"]).union            # tiles up to a null set
    assert 0.4 not in gap and 0.3 in gap


def test_dirac_and_counting_measures_use_the_point_convention():
    """B1: DiracMeasure((.2, .3))(FiniteSet([(.2, .3)])) raised TypeError (the Domain signature on a core set)."""
    assert pg.DiracMeasure((0.2, 0.3))(FiniteSet([(0.2, 0.3)])) == 1.0
    assert pg.DiracMeasure((0.2, 0.3))(pg.Box((0, 1), (0, 1))) == 1.0
    assert pg.DiracMeasure((2.2, 0.3))(pg.Disk(1.0)) == 0.0
    assert pg.DiracMeasure(0.5)(Interval("[0, 1]")) == 1.0
    assert pg.CountingMeasure([0.1, 0.5, 2.0])(Interval("[0, 1]")) == 2.0
    assert pg.CountingMeasure([(0.1, 0.1), (0.5, 0.5), (2.0, 0.0)])(pg.Box((0, 1), (0, 1))) == 2.0
