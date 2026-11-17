"""Law tests (plan 023, from EXP-036's property-based study): algebraic laws of operators, metrics, Gaussian
conditioning, claims, enclosures and sets (plan 067 P2: membership, the point convention, & and the empty set; P3:
subsets of a space and the direction of every support's ends), checked by Hypothesis on generated valid objects with
rounding-error tolerances; measures (plan 067 P5: the probability claim agrees with the mass, integration is linear
with integral of 1 the mass); states (plan 067 P4: every state declares its kind on its class; the prior none is the
whole space, updated by one route to the exact fibre); spaces (plan 068 P1: a level's operations exist exactly at
that level; the complex inner product of Euclidean(n, field="C") is Hermitian through the one metric code); the dual
(plan 068 P2: every functional the package builds is a Functional of the dual of its space, linear and bounded by its
norm, and the dual's zero and linear combinations are its elements); covariances (plan 068 P4: what a Covariance
claims is true of its realisation by the precision rule, and its trace Enclosure contains the trace of its matrix);
operators (plan 068 P3: D(A) of an unbounded operator is dense and not closed; the stated error of an operator built
from others bounds the true error by the rules of operators/error.py, and is UNKNOWN where no rule applies); the data
space (plan 068 P5: a direct sum of Euclideans is a Euclidean with block labels and the sum's inner product; the
column operator is the stacked matrix; D.whitened(C) is an isometry onto the whitened coordinates); finite subspaces
(plan 068 P6: the projection of a Span is the orthogonal projection; a Discretisation's approximation claim is never
stronger than what was checked, and its error is stated in the norm of the function space); space identity (plan 069
P1, decision Q4: a function space is the same whatever realises its elements, and mixing realisations is refused with
RealisationMismatch naming both); quadrature (plan 069 P5: every rule of pygeoinf0.quadrature integrates exactly the
polynomials of its stated degree, and no more; P2: the rule of a region, through its charts, carries its measure);
named priors (plan 069 P6, decision Q2: no space builds a prior unless asked by name; a named prior's covariance
claims are true of it on the function space; legendre_decay states its truncation); piecewise spaces (plan 069 P7,
review B9: X.restrictions(P) is an isometry of L2 onto the broken PiecewiseSpace, which is the glued one; the norm of
a broken space is the sum of its pieces' norms; elements of a glued H^s have matching traces on the seams, a broken
one need not, and gluing checks them); choice rules (plan 070 P2,
review D3: every data-dependent choice rule of pg.meta records whether it used the datum, and its hyperparameter
family as a named record); Problem.run (plan 070 P5, review D9: every result carries the calibration claims of its
problem; a method refuses a calibration it does not declare); the data-space engine (plan 070 P3, review D4, Q1: it
never infers a metric).

Every law has a planted-defect control: the same check run with a known defect monkeypatched in must fail. A law
whose control passes is vacuous (EXP-036 found one: G.adjoint.adjoint is cached, so an involution test on it could
not fail). test_every_law_has_a_control keeps the pairing complete.

Hypothesis is an optional dev dependency: without it this module is skipped.
"""
import contextlib
import math
import os
import re
import sys
from fractions import Fraction

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, example, given, reject, settings, strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

import pygeoinf0 as pg  # noqa: E402
from pygeoinf0.spaces import metrics  # noqa: E402
from pygeoinf0.spaces import base as spaces_base  # noqa: E402
from pygeoinf0.core import numbers  # noqa: E402
from pygeoinf0.core.numbers import Enclosure  # noqa: E402
from pygeoinf0.core.claims import DERIVED, ORDER  # noqa: E402
from pygeoinf0.core.logic import Truth  # noqa: E402
from pygeoinf0.operators import linear  # noqa: E402
from pygeoinf0.probability import gaussian  # noqa: E402
from pygeoinf0 import sets as sets_pkg  # noqa: E402
from pygeoinf0.sets import EmptySet, FiniteSet, Interval, Region  # noqa: E402
from pygeoinf0.sets import intervals as intervals_mod, regions as regions_mod  # noqa: E402
from pygeoinf0.sets import PredicateSubset, Subset  # noqa: E402
from pygeoinf0.sets import subsets as subsets_mod  # noqa: E402
from pygeoinf0.convex import algebra as calgebra, subspaces as csub  # noqa: E402
from pygeoinf0.convex.sets import ConvexSet, support_of  # noqa: E402
from pygeoinf0.measures import base as measures_base  # noqa: E402
from pygeoinf0.core import kinds  # noqa: E402
from pygeoinf0.states import base as states_base  # noqa: E402
from pygeoinf0.cells import backward as cells_backward  # noqa: E402

EPS = np.finfo(float).eps
RANK = {s: i for i, s in enumerate(ORDER)}                  # 0 = strongest
LAW = settings(max_examples=int(os.environ.get("PYGEOINF0_LAW_EXAMPLES", 200)), deadline=None, derandomize=True,
               suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])


# ------------------------------------------------------------------ generators: valid objects by construction
finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


@st.composite
def spd(draw, n, max_cond=1e4):
    """An SPD n x n matrix Q diag(10^lam) Q^T with condition number at most max_cond, exactly symmetric."""
    raw = draw(hnp.arrays(float, (n, n), elements=st.floats(-1, 1, allow_nan=False)))
    Q, _ = np.linalg.qr(raw + 3 * np.eye(n))
    lam = draw(hnp.arrays(float, n, elements=st.floats(0, np.log10(max_cond))))
    S = (Q * 10.0 ** lam) @ Q.T
    return (S + S.T) / 2


@st.composite
def operator_problem(draw):
    n, k = draw(st.integers(1, 6)), draw(st.integers(1, 6))
    return (draw(hnp.arrays(float, (k, n), elements=finite)), draw(spd(n)), draw(spd(k)),
            draw(hnp.arrays(float, n, elements=finite)), draw(hnp.arrays(float, k, elements=finite)))


@st.composite
def square_problem(draw):
    """Two operators on one weighted R^n, a coefficient, and two vectors."""
    n = draw(st.integers(2, 5))
    A = draw(hnp.arrays(float, (n, n), elements=st.floats(-10, 10)))
    B = draw(hnp.arrays(float, (n, n), elements=st.floats(-10, 10)))
    return A, B, draw(st.floats(-5, 5)), draw(spd(n, max_cond=100)), \
        draw(hnp.arrays(float, n, elements=st.floats(-10, 10))), draw(hnp.arrays(float, n, elements=st.floats(-10, 10)))


def _tol(scale, cond=1.0, n=1):
    """A backward-error tolerance: a hundred ulps times dimension, conditioning and the magnitudes involved."""
    return 100 * n * cond * EPS * float(scale) + 1e-300


def _adjoint_gap(G, x, y, scale, cond, n):
    lhs, rhs = G(x) @ y, x @ G.adjoint(y)
    assert abs(lhs - rhs) <= _tol(scale, cond, n), f"<G x, y> - <x, G* y> = {lhs - rhs:.3g}"


# ------------------------------------------------------------------ the laws
@LAW
@given(operator_problem())
def law_matrix_adjoint(prob):
    """<G x, y>_N = <x, G* y>_M for a matrix between Euclidean spaces with arbitrary SPD metrics M, N."""
    A, M, N, m, d = prob
    Dm, Dd = pg.Euclidean(len(m), metric=M), pg.Euclidean(len(d), metric=N)
    G = pg.Matrix(A, domain=Dm, codomain=Dd)
    Bs = G.adjoint.matrix()
    scale = np.abs(A) @ np.abs(m) @ np.abs(N) @ np.abs(d) + np.abs(m) @ np.abs(M) @ np.abs(Bs) @ np.abs(d)
    _adjoint_gap(G, Dm(m), Dd(d), scale, np.linalg.cond(M), len(m) + len(d))


@LAW
@given(operator_problem())
def law_adjoint_involution(prob):
    """(G*)* = G, with the second adjoint rebuilt as a fresh Matrix: G.adjoint.adjoint is G by caching, a vacuous test."""
    A, M, N, *_ = prob
    Dm, Dd = pg.Euclidean(len(M), metric=M), pg.Euclidean(len(N), metric=N)
    G = pg.Matrix(A, domain=Dm, codomain=Dd)
    B = pg.Matrix(G.adjoint.matrix(), domain=Dd, codomain=Dm).adjoint.matrix()
    assert np.allclose(B, A, rtol=0, atol=_tol(np.abs(A).max() + 1, np.linalg.cond(M) * np.linalg.cond(N), 8))


@LAW
@given(square_problem())
# A B = 0 exactly while B* A* keeps rounding from W's off-diagonal entries: the adjoint side's magnitudes must come from
# the factors, since the computed product has cancelled (found on the Mac, 2026-11-06)
@example((np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]]), 0.0,
          np.array([[1.0, 6.08283541e-17], [6.08283541e-17, 1.0]]), np.ones(2), np.ones(2)))
def law_composition_adjoint(prob):
    """(A B)* = B* A* on a weighted R^n: the adjoint identity of a composition."""
    A, B, _, W, u, v = prob
    E = pg.Euclidean(len(u), metric=W)
    # explicitly a Composition: A @ B of two matrices folds into one Matrix, and the law then never reaches the
    # composition's adjoint (its planted-defect control found this)
    GA, GB = pg.Matrix(A, domain=E, codomain=E), pg.Matrix(B, domain=E, codomain=E)
    G = linear.Composition(GA, GB)
    # |u|^T |W| |B*| |A*| |v|: the rounding of B* A* scales with its factors, not with their (possibly cancelled) product
    adjoint_side = np.abs(u) @ np.abs(W) @ np.abs(GB.adjoint.matrix()) @ np.abs(GA.adjoint.matrix()) @ np.abs(v)
    scale = (np.abs(A) @ np.abs(B) @ np.abs(u)) @ np.abs(W) @ np.abs(v) + adjoint_side
    _adjoint_gap(G, E(u), E(v), scale, np.linalg.cond(W), 3 * len(u))


@LAW
@given(square_problem())
def law_sum_adjoint(prob):
    """(A + c B)* = A* + c B* on a weighted R^n."""
    A, B, c, W, u, v = prob
    E = pg.Euclidean(len(u), metric=W)
    GA, GB = pg.Matrix(A, domain=E, codomain=E), pg.Matrix(B, domain=E, codomain=E)
    G = GA + c * GB
    # per term, as for the composition: A* + c B* can cancel
    adjoint_side = np.abs(u) @ np.abs(W) @ (np.abs(GA.adjoint.matrix()) + abs(c) * np.abs(GB.adjoint.matrix())) @ np.abs(v)
    scale = ((np.abs(A) + abs(c) * np.abs(B)) @ np.abs(u)) @ np.abs(W) @ np.abs(v) + adjoint_side
    _adjoint_gap(G, E(u), E(v), scale, np.linalg.cond(W), 3 * len(u))


@LAW
@given(st.integers(2, 5), st.data())
def law_inner_product_symmetric(n, data):
    """Every metric the package accepts gives a symmetric inner product (to the rounding of one pairing), also one
    that is symmetric only within the acceptance tolerance (EXP-036 finding 2)."""
    S = data.draw(spd(n, max_cond=10))
    skew = data.draw(hnp.arrays(float, (n, n), elements=st.floats(-1, 1)))
    W = S + 5e-15 * np.abs(S).max() * (skew - skew.T)
    try:
        E = pg.Euclidean(n, metric=W)
    except ValueError:
        reject()
    i, j = data.draw(st.integers(0, n - 1)), data.draw(st.integers(0, n - 1))
    x, y = E(np.eye(n)[i]), E(np.eye(n)[j])
    assert (x @ y) == (y @ x), f"<e{i}, e{j}> - <e{j}, e{i}> = {(x @ y) - (y @ x):.3g}"


@LAW
@given(st.integers(1, 5), st.integers(1, 4), st.data())
def law_finite_rank_covariance(n, r, data):
    """A finite-rank covariance on a weighted R^n is self-adjoint, positive semidefinite, and its trace is
    sum lam_j ||psi_j||^2."""
    W = data.draw(spd(n))
    E = pg.Euclidean(n, metric=W)
    P = data.draw(hnp.arrays(float, (r, n), elements=st.floats(-10, 10)))
    lam = data.draw(hnp.arrays(float, r, elements=st.floats(0, 10)))
    C = linear.FiniteRank(E.family(P), lam)
    K = W @ C.matrix()
    assert np.allclose(K, K.T, atol=_tol(np.abs(K).max() + 1, n=n))
    assert np.linalg.eigvalsh((K + K.T) / 2).min() >= -_tol(np.abs(K).max() + 1, n=n)
    assert C.trace() == pytest.approx(float(sum(w * p @ W @ p for w, p in zip(lam, P))), rel=1e-10, abs=1e-12)


@LAW
@given(st.integers(1, 4), st.integers(1, 4), st.data())
def law_posterior_covariance(n, k, data):
    """The posterior covariance equals the information form (C^-1 + A^T R^-1 A)^-1, is symmetric and is <= C."""
    C = data.draw(spd(n, max_cond=1e3))
    R = data.draw(spd(k, max_cond=1e3))
    A = data.draw(hnp.arrays(float, (k, n), elements=st.floats(-3, 3)))
    E, D = pg.Euclidean(n), pg.Euclidean(k, covariance=R)
    prior = pg.GaussianMeasure(E, covariance=pg.Matrix(C, domain=E, codomain=E))
    P = prior.condition(pg.Matrix(A, domain=E, codomain=D), D(np.zeros(k))).covariance.matrix()
    info = np.linalg.inv(np.linalg.inv(C) + A.T @ np.linalg.solve(R, A))
    big = np.abs(C).max()
    assert np.allclose(P, info, rtol=1e-7, atol=1e-9 * big)
    assert np.allclose(P, P.T, atol=1e-9 * big)
    assert np.linalg.eigvalsh(C - (P + P.T) / 2).min() >= -1e-9 * big


LEAF_CLAIMS = st.sampled_from([s for s in ORDER])


@LAW
@given(st.integers(1, 4), st.data())
def law_claims_never_promote(n, data):
    """A conclusion drawn by a rule is never stronger than its premises or than DERIVED (plan 023; the rule table's
    "CHECKED never promotes to DERIVED"): sums, positive multiples, inverses, block-diagonal operators and metrics."""
    E = pg.Euclidean(n)
    leaves = []
    for _ in range(2):
        A = pg.Matrix(data.draw(spd(n)), domain=E, codomain=E)
        st_sa, st_pos = data.draw(LEAF_CLAIMS), data.draw(LEAF_CLAIMS)
        A._sa, A._pos = (True, st_sa, "drawn"), ("pd", st_pos, "drawn")
        leaves.append(A)
    A, B = leaves

    def check(got, *premises):
        floor = max(RANK[DERIVED], *(RANK[p] for p in premises))
        assert RANK[got] >= floor, f"{got!r} from premises {premises}"

    check((A + B).is_selfadjoint()[1], A._sa[1], B._sa[1])
    check((A + B).positivity()[1], A._pos[1], B._pos[1])
    check((2.0 * A).is_selfadjoint()[1], A._sa[1])
    check((2.0 * A).positivity()[1], A._pos[1])
    check(A.inv.positivity()[1], A._pos[1])
    X = pg.DirectSum({"a": E, "b": E})
    check(pg.BlockDiagonal(X, [A, B]).is_selfadjoint()[1], A._sa[1], B._sa[1])
    m = metrics.Metric(2.0 * A, E, "metric")            # an operator, not a Matrix: the claim comes from the rule
    check(m.ops.claims[-1].status, A._sa[1], A._pos[1])


truths = st.sampled_from(list(Truth))


@LAW
@given(truths, truths, truths)
def law_truth_kleene(a, b, c):
    """De Morgan, involution, distributivity and absorption of Kleene's three-valued logic."""
    assert ~(a & b) is (~a | ~b) and ~(a | b) is (~a & ~b)
    assert ~~a is a
    assert (a & (b | c)) is ((a & b) | (a & c))
    assert (a | (a & b)) is a


@LAW
@example(0.0, math.inf, math.inf, 0.0)   # 0 * inf: the unguarded scaling's value is nan (its control needs this case)
@given(st.floats(allow_nan=False), st.floats(allow_nan=False), st.floats(allow_nan=False),
       st.floats(allow_nan=False, allow_infinity=False))
def law_enclosure_scaling(lo, x, hi, c):
    """c times an enclosure contains c times its value, and lower <= value <= upper holds (EXP-036 finding 3)."""
    lo, x, hi = sorted((lo, x, hi))
    s = Enclosure(x, lo, hi).scaled(c)
    assert s.lower <= float(s) <= s.upper
    if np.isfinite(c * x):
        assert s.contains(c * x)


real = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)


def _enclosure(t):
    lo, x, hi = sorted(t)
    return Enclosure(x, lo, hi)


enclosures = st.tuples(real, real, real).map(_enclosure)


def _ends_kept(out, lower, upper):
    """out's lower end is at most the exact lower (a rational), its upper end at least the exact upper, and its value
    lies between them: each end keeps its direction (plan 067 P1)."""
    assert Fraction(out.lower) <= lower and upper <= Fraction(out.upper), (out, lower, upper)
    assert out.lower <= float(out) <= out.upper


@LAW
@example(Enclosure(0.1, 0.1, 0.1), Enclosure(0.2, 0.2, 0.2))
@given(enclosures, enclosures)
def law_enclosure_sum(a, b):
    """The sum of two enclosures encloses every sum of their points: [l_a + l_b, u_a + u_b], rounded outward."""
    _ends_kept(a + b, Fraction(a.lower) + Fraction(b.lower), Fraction(a.upper) + Fraction(b.upper))


@LAW
@example(Enclosure(0.1, 0.1, 0.1), 3.0)
@given(enclosures, st.floats(1e-6, 1e6))
def law_enclosure_positive_scaling(a, c):
    """c > 0 times an enclosure: [c l, c u], rounded outward."""
    _ends_kept(a.scaled(c), Fraction(c) * Fraction(a.lower), Fraction(c) * Fraction(a.upper))


@LAW
@given(enclosures, enclosures)
def law_enclosure_max(a, b):
    """The max of two enclosures encloses every max of their points: [max(l_a, l_b), max(u_a, u_b)]."""
    _ends_kept(numbers.maximum(a, b), max(Fraction(a.lower), Fraction(b.lower)),
               max(Fraction(a.upper), Fraction(b.upper)))


# ------------------------------------------------------------------ sets (plan 067 P2)
halves = st.integers(-6, 6).map(lambda k: k / 2.0)          # endpoints on a grid of halves: touching is common


@st.composite
def intervals(draw):
    a = draw(st.integers(-6, 5))
    b = draw(st.integers(a + 1, 6))
    return Interval(a / 2.0, b / 2.0, draw(st.sampled_from(["both", "left", "right", "neither"])))


def _surface():
    return pg.Surface(lambda u, v: np.stack(np.broadcast_arrays(u, v, 0 * u), -1), (0, 1), (0, 1))


@st.composite
def set_and_point(draw):
    """A set of one of the kinds (decided or not) and a point of its R^d (on a grid, so boundaries are hit)."""
    kind = draw(st.sampled_from(["interval", "finite", "box", "ball", "union", "meet", "undecided"]))
    if kind == "interval":
        S = draw(intervals())
    elif kind == "finite":
        S = FiniteSet(draw(st.lists(halves, min_size=1, max_size=4)))
    elif kind == "union":
        S = draw(intervals()) | draw(intervals())
    elif kind == "meet":
        S = draw(intervals()) & draw(intervals())
    elif kind == "box":
        S = pg.Box(*[(c, c + draw(st.integers(1, 3)) / 2.0) for c in (draw(halves), draw(halves))])
    elif kind == "ball":
        S = pg.Ball(draw(st.integers(1, 4)) / 2.0, centre=(draw(halves), 0.0, 0.0))
    else:
        S = _surface() & pg.Ball(1.0) if draw(st.booleans()) else _surface()
    d = int(S.ambient.cartesian_dim) if isinstance(S, EmptySet) else int(S.cartesian_dim)
    return S, np.array([draw(halves) for _ in range(d)])


@LAW
@given(set_and_point())
def law_in_is_member_true(sp):
    """x in S <=> S.member(x) is TRUE; `in` raises on UNKNOWN (decision Q3)."""
    S, x = sp
    t = S.member(x)
    try:
        answer = x in S
    except TypeError:
        answer = None                                                       # refused: undecided
    assert answer == (None if t is Truth.UNKNOWN else t is Truth.TRUE), (S, x, t, answer)


@st.composite
def region_and_points(draw):
    kind = draw(st.sampled_from(["interval", "finite", "box", "ball", "disk", "sphere"]))
    S = {"interval": lambda: draw(intervals()), "finite": lambda: FiniteSet(draw(st.lists(halves, min_size=1))),
         "box": lambda: pg.Box((0, 1), (0, 0.5)), "ball": lambda: pg.Ball(1.0), "disk": lambda: pg.Disk(1.0),
         "sphere": lambda: pg.Sphere(1.0)}[kind]()
    d = int(S.cartesian_dim)
    shape = tuple(draw(st.lists(st.integers(1, 3), min_size=1, max_size=3)))
    X = draw(hnp.arrays(float, shape + (d,), elements=st.sampled_from([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])))
    return S, X


@LAW
@given(region_and_points())
def law_one_point_convention(rp):
    """Every Region: points of shape (..., d) give an answer of shape (...), equal point by point to member."""
    S, X = rp
    r = S.contains(X)
    assert np.shape(r) == X.shape[:-1], (S, X.shape, np.shape(r))
    flat = np.reshape(r, -1)
    for p, v in zip(X.reshape(-1, X.shape[-1]), flat):
        assert bool(v) == (S.member(p) is Truth.TRUE)


@LAW
@given(intervals(), intervals(), halves)
def law_interval_meet_is_exact(I, J, x):
    """I & J is never None: a Region or the EmptySet of R, with x in I & J <=> x in I and x in J."""
    K = I & J
    assert isinstance(K, (Region, EmptySet)), (I, J, K)
    assert (x in K) == (x in I and x in J), (I, J, K, x)
    if isinstance(K, EmptySet):
        assert K.ambient == I.ambient


@st.composite
def regions_any(draw):
    return draw(st.sampled_from([lambda: draw(intervals()), lambda: FiniteSet([0.0, 1.0]), lambda: pg.Disk(1.0),
                                 lambda: pg.Box((0, 1), (0, 1), (0, 1)), lambda: pg.Sphere(1.0),
                                 lambda: pg.LayeredBall([0.5, 1.0]), _surface]))()


@LAW
@given(regions_any())
def law_empty_set_identity(S):
    """EmptySet(A) is the identity of | and absorbing for &, with the ambient of the set."""
    E = EmptySet(S.ambient)
    assert (S | E) is S and (E | S) is S
    for e in (S & E, E & S):
        assert isinstance(e, EmptySet) and e.ambient == S.ambient


# ------------------------------------------------------------------ subsets of a space (plan 067 P3)
X2 = pg.Euclidean(2)
grid = st.integers(-4, 4).map(lambda k: k / 4.0)


class _Bracketed(ConvexSet):
    """A ball whose support is stated as the bracket [s - dl, s + du] with its value at the lower end (a witness end,
    as a BGP bound in a BRACKET reports it)."""

    def __init__(self, r, dl, du):
        super().__init__(X2, "bracketed")
        self.r, self.dl, self.du = r, dl, du

    def support(self, v):
        s = self.r * float(np.linalg.norm(v.values))
        return Enclosure(s - self.dl, s - self.dl, s + self.du, "derived", "a stated bracket")


@st.composite
def convex_part(draw):
    """(set, sigma) of a part in R^2 with the identity metric: sigma(v) by an independent closed form."""
    kind = draw(st.sampled_from(["ball", "box", "ellipsoid", "bracketed"]))
    if kind == "ball":
        r, c = draw(st.integers(1, 4)) / 2.0, np.array([draw(grid), draw(grid)])
        return pg.NormBall(X2, r, centre=X2(c)), lambda a: float(a @ c) + r * math.hypot(*a)
    if kind == "box":
        lo = np.array([draw(grid), draw(grid)])
        hi = lo + np.array([draw(st.integers(0, 4)), draw(st.integers(0, 4))]) / 2.0
        return pg.CoordinateBox(X2, lo, hi), lambda a: float(np.sum(np.where(a > 0, a * hi, a * lo)))
    if kind == "ellipsoid":
        s = np.array([draw(st.integers(1, 4)), draw(st.integers(1, 4))]) / 2.0
        return pg.Ellipsoid(X2, 1.0, s), lambda a: math.sqrt(float(a @ (s * a)))
    r, dl, du = draw(st.integers(1, 4)) / 2.0, draw(st.integers(1, 3)) / 4.0, draw(st.integers(1, 3)) / 4.0
    return _Bracketed(r, dl, du), lambda a: r * math.hypot(*a)


@st.composite
def convex_composite(draw):
    """A combination of parts and its exact support (each part's closed form, combined by the rule)."""
    how = draw(st.sampled_from(["part", "sum", "translate", "scale", "union", "sum of three"]))
    A, sa = draw(convex_part())
    if how == "part":
        return A, sa
    if how == "translate":
        x = np.array([draw(grid), draw(grid)])
        return A + X2(x), lambda a: sa(a) + float(a @ x)
    if how == "scale":
        c = draw(st.integers(1, 6)) / 2.0
        return c * A, lambda a: c * sa(a)
    B, sb = draw(convex_part())
    if how == "sum":
        return A + B, lambda a: sa(a) + sb(a)
    if how == "union":
        return A | B, lambda a: max(sa(a), sb(a))
    C, sc = draw(convex_part())
    return (A + B) + C, lambda a: sa(a) + sb(a) + sc(a)


@LAW
@given(convex_composite(), st.integers(-4, 4), st.integers(-4, 4))
def law_support_ends_certified(composite, i, j):
    """support(v) is an Enclosure whose lower end is a lower bound and upper end an upper bound of sigma_S(v), also
    through the set algebra (A7: a sum with a bracketed part keeps each end in its direction)."""
    S, sigma = composite
    if i == j == 0:
        reject()
    a = np.array([i, j], float)
    e = support_of(S, X2(a))
    exact = sigma(a)
    tol = 1e-12 * max(1.0, abs(exact))
    assert isinstance(e, Enclosure) and e.lower - tol <= exact <= e.upper + tol, (S, a, e, exact)


@st.composite
def subset_and_point(draw):
    kind = draw(st.sampled_from(["ball", "box", "ellipsoid", "span", "undecided", "meet", "join", "predicate"]))
    ball, box = pg.NormBall(X2, 1.0), pg.CoordinateBox(X2, [-0.5, -1.0], [0.75, 0.5])
    S = {"ball": lambda: ball, "box": lambda: box, "ellipsoid": lambda: pg.Ellipsoid(X2, 1.0, [0.25, 1.0]),
         "span": lambda: csub.Span([X2([1.0, 1.0])]),
         "undecided": lambda: PredicateSubset(X2, lambda x: pg.Truth.UNKNOWN if x.values[0] > 0 else False),
         "meet": lambda: ball & box, "join": lambda: ball | box,
         "predicate": lambda: PredicateSubset(X2, lambda x: x.values[1] >= 0) & ball}[kind]()
    return S, X2(np.array([draw(grid), draw(grid)]))


@LAW
@given(subset_and_point())
def law_subset_in_is_member_true(sp):
    """A subset of a space is a Set: x in S <=> S.member(x) is TRUE, and `in` raises on UNKNOWN (decision Q3)."""
    S, x = sp
    t = S.member(x)
    try:
        answer = x in S
    except TypeError:
        answer = None
    assert answer == (None if t is Truth.UNKNOWN else t is Truth.TRUE), (S, x, t, answer)


@st.composite
def subsets_any(draw):
    return draw(st.sampled_from([lambda: pg.NormBall(X2, 1.0), lambda: pg.CoordinateBox(X2, 0.0, 1.0),
                                 lambda: csub.Span([X2([1.0, 0.0])]), lambda: csub.Whole(X2),
                                 lambda: PredicateSubset(X2, lambda x: True),
                                 lambda: pg.NormBall(X2, 1.0) & pg.NormBall(X2, 2.0),
                                 lambda: pg.NormBall(X2, 1.0) + pg.CoordinateBox(X2, 0.0, 1.0)]))()


@LAW
@given(subsets_any())
def law_empty_subset_identity(S):
    """EmptySet(X) is the identity of | and absorbing for & on the subsets of a space, with the space as ambient."""
    E = pg.EmptySet(X2)
    assert isinstance(S, Subset) and S.ambient is X2
    assert (S | E) is S and (E | S) is S
    for e in (S & E, E & S):
        assert isinstance(e, EmptySet) and e.ambient == X2


# ------------------------------------------------------------------ measures (plan 067 P5)
@st.composite
def point_measures(draw):
    """A counting measure with positive weights, a Dirac measure, or a combination of them (a measure of finitely many
    points: its mass and integrals are exact)."""
    pts = draw(st.lists(halves, min_size=1, max_size=4, unique=True))
    kind = draw(st.sampled_from(["counting", "dirac", "combination"]))
    if kind == "counting":
        w = draw(st.lists(st.floats(1e-3, 10.0), min_size=len(pts), max_size=len(pts)))
        return pg.CountingMeasure(pts, weights=w)
    if kind == "dirac":
        return pg.DiracMeasure(pts[0])
    cs = draw(st.lists(st.floats(-5.0, 5.0), min_size=len(pts), max_size=len(pts)))
    return sum(c * pg.DiracMeasure(p) for c, p in zip(cs, pts))


@LAW
@example(pg.LebesgueMeasure("[0, 2]"))
@example(pg.CountingMeasure([0.25, 0.5], weights=[0.5, 0.5]))
@given(st.one_of(point_measures(), intervals().map(pg.LebesgueMeasure)))
def law_probability_claim_agrees_with_the_mass(mu):
    """probability TRUE iff non-negative with mass 1 (to rounding); FALSE when the mass is off 1 or mu is signed."""
    m, err = mu.mass()
    held = mu.claims.status("probability")[0]
    signed = mu.claims.status("signed")[0]
    one = abs(m - 1.0) <= err + 64 * EPS * max(1.0, abs(m))
    if held is Truth.TRUE:
        assert one and signed is Truth.FALSE, (mu, m)
    if not one or signed is Truth.TRUE:
        assert held is Truth.FALSE, (mu, m, held)


@LAW
@given(point_measures(), st.floats(-3.0, 3.0))
def law_integration_is_linear_and_one_gives_the_mass(mu, c):
    f, g = (lambda x: x ** 2), (lambda x: np.cos(3.0 * x))
    v = mu.integrate(lambda x: c * f(x) + g(x))[0]
    w = c * mu.integrate(f)[0] + mu.integrate(g)[0]
    scale = 1.0 + abs(c) * abs(mu.integrate(lambda x: np.abs(f(x)))[0]) + abs(mu.integrate(lambda x: np.abs(g(x)))[0])
    assert abs(v - w) <= 64 * EPS * scale * max(1, len(getattr(mu, "terms", ()))), (mu, v, w)
    assert mu.integrate(lambda x: np.ones_like(x))[0] == pytest.approx(mu.mass()[0], rel=1e-14, abs=1e-14)


# ------------------------------------------------------------------ states (plan 067 P4)
class _InstanceKind:
    """Not a state: the kind is an instance attribute, not declared by the class."""

    def __init__(self):
        self.space, self.kind, self.claims = X2, kinds.SET, ()


class _Sniffed:
    """Not a state: the attributes of a Gaussian, no declared kind."""
    space, covariance = X2, None

    def sample(self):
        return X2.zero


STATE_CASES = {
    "ball": (lambda: pg.NormBall(X2, 1.0), kinds.SET), "whole": (lambda: pg.NoPrior(X2), kinds.SET),
    "span": (lambda: csub.Span([X2([1.0, 0.0])]), kinds.SET), "empty": (lambda: pg.EmptySet(X2), kinds.SET),
    "predicate": (lambda: PredicateSubset(X2, lambda x: True), kinds.SET),
    "gaussian": (lambda: X2.standard_gaussian(), kinds.MEASURE), "dirac": (lambda: pg.Dirac(X2.zero), kinds.MEASURE),
    "discrete": (lambda: pg.DiscreteMeasure({0.0: 0.25, 1.0: 0.75}), kinds.MEASURE),
    "ordering": (lambda: pg.Ordering.norm(X2), kinds.ORDERING),
    "element": (lambda: X2([1.0, 2.0]), None), "family": (lambda: X2.family(np.eye(2)), None),
    "base measure": (lambda: pg.LebesgueMeasure("[0, 1]"), None), "relation": (lambda: pg.Identity(X2), None),
    "instance kind": (_InstanceKind, None), "sniffed": (_Sniffed, None)}


@LAW
@example("instance kind")
@example("sniffed")
@given(st.sampled_from(sorted(STATE_CASES)))
def law_every_state_declares_its_kind(case):
    """state_kind(obj) is the kind obj's class declares (set, measure or ordering), and anything else is refused: an
    element, a family, a base measure, a relation, an instance attribute or Gaussian-like attributes are not states."""
    make, expected = STATE_CASES[case]
    obj = make()
    try:
        got = pg.state_kind(obj)
    except TypeError:
        got = None
    assert got is expected, (case, got, expected)
    assert isinstance(obj, pg.State) == (expected is not None), case


@st.composite
def fibre_problem(draw):
    n = draw(st.integers(1, 5))
    k = draw(st.integers(1, n))
    A = draw(hnp.arrays(float, (k, n), elements=st.floats(-3, 3, allow_nan=False)))
    if np.linalg.svd(A, compute_uv=False)[-1] < 1e-2 * max(1.0, np.abs(A).max()):
        reject()                                                  # full row rank, well conditioned: d attainable
    return A, draw(hnp.arrays(float, k, elements=st.floats(-3, 3, allow_nan=False)))


@LAW
@given(fibre_problem())
def law_no_prior_is_the_whole_space_by_one_route(prob):
    """The prior none is Whole(X), and its update with exact data is the fibre G^-1(d) = G^+ d + ker G exactly, by the
    one route of a set prior (the cells B1 and B5 were two)."""
    A, dv = prob
    k, n = A.shape
    X, D = pg.Euclidean(n), pg.Euclidean(k)
    G, d = pg.Matrix(A, X, D), D(dv)
    K0, K1 = pg.update(states_base.NoPrior(X), G, d), pg.update(csub.Whole(X), G, d)
    assert type(K0) is type(K1) and K0.cell == K1.cell
    assert isinstance(K1, pg.AffineSubset) and hasattr(K1, "summary"), type(K1)
    m = K1.summary().values
    ref = np.linalg.lstsq(A, dv, rcond=None)[0]
    scale = 1.0 + np.abs(ref).max()
    assert np.abs(m - ref).max() <= 1e3 * n * EPS * scale * np.linalg.cond(A), (m, ref)
    assert np.array_equal(K0.summary().values, m)


# ------------------------------------------------------------------ spaces: each level's contract (plan 068 P1)
I01 = Interval("[0, 1]")
_HERM = np.array([[2.0, 0.5 + 0.5j], [0.5 - 0.5j, 1.0]])
SPACE_CASES = {
    "R3": (lambda: pg.Euclidean(3), lambda X: (X(np.array([1.0, -2.0, 0.5])), X(np.array([0.0, 3.0, 1.0])))),
    "R3 weighted": (lambda: pg.Euclidean(3, weights=[1.0, 2.0, 3.0]),
                    lambda X: (X(np.array([1.0, -2.0, 0.5])), X(np.array([0.0, 3.0, 1.0])))),
    "R2 covariance": (lambda: pg.Euclidean(2, covariance=np.array([[2.0, 0.5], [0.5, 1.0]])),
                      lambda X: (X(np.array([1.0, -2.0])), X(np.array([0.5, 3.0])))),
    "C2": (lambda: pg.Euclidean(2, field="C"), lambda X: (X(np.array([1.0 + 1j, -2.0])), X(np.array([0.5j, 3.0])))),
    "C2 metric": (lambda: pg.Euclidean(2, field="C", metric=_HERM),
                  lambda X: (X(np.array([1.0 + 1j, -2.0])), X(np.array([0.5j, 3.0])))),
    "L2": (lambda: pg.L2(I01), lambda X: (X(lambda t: t), X(lambda t: 1.0 - t ** 2))),
    "Lp3": (lambda: pg.Lp(I01, 3), lambda X: (X(lambda t: t), X(lambda t: 1.0 - t ** 2))),
    "Lp1": (lambda: pg.Lp(I01, 1), lambda X: (X(lambda t: t), X(lambda t: 1.0 - t ** 2))),
    "Lpinf": (lambda: pg.Lp(I01, np.inf), lambda X: (X(lambda t: t), X(lambda t: 1.0 - t ** 2))),
    "C[0,1]": (lambda: pg.Continuous("[0, 1]"), lambda X: (X(lambda t: t), X(lambda t: 1.0 - t ** 2))),
    "H1": (lambda: pg.Sobolev(I01, 1, length_scale=0.2), lambda X: (X(lambda t: t), X(lambda t: 1.0 - t ** 2))),
    "direct sum": (lambda: pg.DirectSum({"a": pg.Euclidean(2), "b": pg.L2(I01)}),
                   lambda X: (X.element({"a": np.array([1.0, 2.0]), "b": lambda t: t}),
                              X.element({"a": np.array([0.0, -1.0]), "b": lambda t: 1.0 - t}))),
    "complexified": (lambda: pg.Complexified(pg.L2(I01)),
                     lambda X: (X(lambda t: t + 1j * t ** 2), X(lambda t: 1.0 - 1j * t))),
}
_SPACES = {}


def _space_case(name):
    if name not in _SPACES:
        make, elements = SPACE_CASES[name]
        X = make()
        _SPACES[name] = (X, elements(X))
    return _SPACES[name]


@LAW
@example("Lp3")
@example("C[0,1]")
@example("L2")
@given(st.sampled_from(sorted(SPACE_CASES)))
def law_operations_exist_at_their_level(name):
    """A level's operations exist exactly at that level: every space is a Set with a field, a zero that belongs to it and
    linear combinations; a normed space has a norm, a dual (complete, with the space as its primal) and functionals; a
    Hilbert space, and only a Hilbert space, has an inner product and a Riesz map (L^p for p != 2 and C have neither, as
    attributes: there is nothing to call)."""
    X, (x, y) = _space_case(name)
    assert isinstance(X, spaces_base.VectorSpace) and isinstance(X, sets_pkg.Set), name
    assert X.field in ("R", "C"), (name, X.field)
    assert X.member(X.zero) is Truth.TRUE, name
    combo, ref = X.linear_combination([2.0, -0.5], [x, y]), 2.0 * x + (-0.5) * y
    assert X.member(combo) is Truth.TRUE, name
    assert X.norm(combo - ref) <= 1e-10 * (1.0 + X.norm(ref)), name
    assert isinstance(X, spaces_base.NormedSpace), name
    assert all(callable(getattr(X, op, None)) for op in ("norm", "dual", "functional")), name
    D = X.dual()
    assert isinstance(D, spaces_base.BanachSpace) and D.primal == X, name
    hilbert = isinstance(X, spaces_base.HilbertSpace)
    assert hasattr(X, "inner") == hilbert and hasattr(X, "riesz") == hilbert, (name, hilbert, hasattr(X, "inner"),
                                                                              hasattr(X, "riesz"))


@st.composite
def hermitian_problem(draw):
    """A Hermitian positive definite n x n matrix (condition at most 100) and three complex vectors and a scalar."""
    n = draw(st.integers(1, 4))
    re = draw(hnp.arrays(float, (n, n), elements=st.floats(-1, 1, allow_nan=False)))
    im = draw(hnp.arrays(float, (n, n), elements=st.floats(-1, 1, allow_nan=False)))
    Q, _ = np.linalg.qr(re + 1j * im + 3 * np.eye(n))
    lam = draw(hnp.arrays(float, n, elements=st.floats(0, 2)))
    W = (Q * 10.0 ** lam) @ Q.conj().T
    W = (W + W.conj().T) / 2
    vec = lambda: draw(hnp.arrays(float, n, elements=st.floats(-3, 3))) + 1j * draw(  # noqa: E731
        hnp.arrays(float, n, elements=st.floats(-3, 3)))
    a = complex(draw(st.floats(-2, 2)), draw(st.floats(-2, 2)))
    return W, vec(), vec(), a


@LAW
@example((_HERM, np.array([1.0 + 1j, -2.0]), np.array([0.5j, 3.0]), 0.3 - 1j))
@given(hermitian_problem())
def law_complex_inner_product_is_hermitian(prob):
    """Euclidean(n, field="C", metric=W) has <x, y> = y^H W x: Hermitian (<y, x> = conj <x, y>), linear in x and
    conjugate-linear in y, its real inner product Re <x, y>; Euclidean(n, field="C", covariance=W^-1) is the same
    space, its metric applied by the one metric code of R^n (no separate complex code, no dense inverse)."""
    W, xv, yv, a = prob
    n = len(xv)
    E, F = pg.Euclidean(n, field="C", metric=W), pg.Euclidean(n, field="C", covariance=np.linalg.inv(W))
    x, y = E(xv), E(yv)
    scale = (1.0 + np.abs(W).max()) * (1.0 + np.abs(xv).max()) * (1.0 + np.abs(yv).max())
    tol = 1e3 * n * EPS * scale * np.linalg.cond(W)
    ref = yv.conj() @ W @ xv
    assert abs(E.cinner(x, y) - ref) <= tol, (E.cinner(x, y), ref)
    assert abs(E.cinner(y, x) - np.conj(E.cinner(x, y))) <= tol
    assert abs(E.cinner(E(a * xv), y) - a * E.cinner(x, y)) <= (1 + abs(a)) * tol
    assert abs(E.cinner(x, E(a * yv)) - np.conj(a) * E.cinner(x, y)) <= (1 + abs(a)) * tol
    assert abs(E.inner(x, y) - E.cinner(x, y).real) <= tol
    assert abs(F.cinner(F(xv), F(yv)) - ref) <= tol, (F.cinner(F(xv), F(yv)), ref)


# ------------------------------------------------------------------ covariances (plan 068 P4)
COVARIANCE_KINDS = ("stated", "spectral", "congruence", "posterior", "factored", "low rank", "property posterior")


@st.composite
def covariance_case(draw):
    """(kind, n, metric W, a symmetric matrix S with eigenvalues of either sign, a second factor B, a vector v): the raw
    material of a covariance of each implementation of the root on a weighted R^n."""
    kind = draw(st.sampled_from(COVARIANCE_KINDS))
    n = draw(st.integers(1, 5))
    W = draw(spd(n, max_cond=100))
    Q, _ = np.linalg.qr(draw(hnp.arrays(float, (n, n), elements=st.floats(-1, 1, allow_nan=False))) + 3 * np.eye(n))
    lam = draw(hnp.arrays(float, n, elements=st.floats(-2, 2)))
    S = (Q * np.sign(lam) * 10.0 ** np.abs(lam)) @ Q.T
    B = draw(hnp.arrays(float, (draw(st.integers(1, 4)), n), elements=st.floats(-3, 3)))
    v = draw(hnp.arrays(float, n, elements=st.floats(-3, 3)))
    return kind, n, W, (S + S.T) / 2, B, v


_FACTORED_NEGATIVE = ("factored", 2, np.eye(2), np.eye(2), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([-3.0, 0.5]))


def _covariance_of(case):
    """A Covariance of the case's kind, or None where the construction refuses (an indefinite stated operator)."""
    from pygeoinf0.probability.lazy_covariance import FactoredCovariance
    from pygeoinf0.probability.lazy_covariance import LowRankCovariance
    from pygeoinf0.core.errors import Refused
    kind, n, W, S, B, v = case
    E = pg.Euclidean(n, metric=W)
    Wi = np.linalg.inv(W)
    if kind == "stated":                     # S as the Gram form <C x, y> = y^T S x: the operator W^-1 S
        try:
            return pg.Covariance(pg.Matrix(Wi @ S, domain=E, codomain=E))
        except Refused:
            return None
    if kind == "spectral":
        Om = np.linalg.cholesky(Wi)          # columns orthonormal in W's inner product
        return pg.Covariance.spectral(np.abs(np.diag(S)), E.family(Om.T))
    if kind == "congruence":
        return pg.Covariance.spectral(np.abs(np.diag(S)), E.family(np.linalg.cholesky(Wi).T)).congruence(
            pg.Matrix(B, domain=E, codomain=pg.Euclidean(B.shape[0])))
    if kind in ("posterior", "property posterior"):
        P2 = S @ S.T + np.eye(n)
        prior = pg.GaussianMeasure(E, covariance=pg.Matrix(Wi @ P2, domain=E, codomain=E))
        D = pg.Euclidean(B.shape[0])
        G = pg.Matrix(B, domain=E, codomain=D)
        if kind == "posterior":
            return prior.condition(G, D(np.zeros(B.shape[0])), D.standard_gaussian()).covariance
        P = pg.Euclidean(1)
        T = pg.Matrix(v[None, :] + 0.5, domain=E, codomain=P)
        return pg.inference.Problem(M=E, D=D, P=P, prior=prior, mdr=pg.Additive(G, D.standard_gaussian()), mpr=T,
                                    datum=D(np.zeros(B.shape[0]))).run("pli").covariance
    if kind == "factored":                   # Phi V Phi^T + diag(extra), V of either sign, extra of either sign
        return FactoredCovariance(pg.Euclidean(B.shape[0]), B[:, :n], S, v[:B.shape[0]] if n >= B.shape[0] else None)
    TT = pg.Matrix(B.T @ B)                  # low rank: a^2 T S T* - a^4 beta^T diag(w) beta
    w, a2 = np.abs(v[:1]) + 0.1, 1.5
    beta = S[:1]
    diag = np.maximum(a2 * np.diag(B.T @ B) - a2 * a2 * ((beta * beta).T @ w), 0.0)
    return LowRankCovariance(pg.Euclidean(n), TT, beta, w, a2, diag)


@LAW
@example(_FACTORED_NEGATIVE)
@given(covariance_case())
def law_covariance_claims_hold(case):
    """What a Covariance claims (theorem, derived or checked) is true of its realisation: self-adjoint and positive
    semidefinite in the space's metric, by the precision rule (core.logic.decide: a claim that the rule finds FALSE is
    a false claim; UNKNOWN is the rounding level), and trace class with a finite trace."""
    C = _covariance_of(case)
    if C is None:
        return
    from pygeoinf0.core.logic import decide
    from pygeoinf0.core.claims import ASSERTED as AS, UNKNOWN as UN
    assert isinstance(C, pg.Covariance)
    X = C.domain
    M = C.matrix()
    K = X.apply_metric(M)                                          # the Gram form y^T W C x
    size = max(float(np.abs(K).max()), 1e-300)
    sym = 0.5 * (K + K.T)
    lam = np.linalg.eigvalsh(sym)
    for key, truth in (("self-adjoint", decide(float(np.abs(K - K.T).max()), size)),
                       ("positive semidefinite", decide(max(-float(lam.min()), 0.0), max(float(np.abs(lam).max()),
                                                                                           1e-300)))):
        holds, status = C.holds(key)
        if holds is Truth.TRUE and status not in (AS, UN):
            assert truth is not Truth.FALSE, f"{C.name}: {key} claimed ({status}) and false: {C.claims}"
    holds, status = C.holds("trace class")
    assert holds is not Truth.FALSE
    if holds is Truth.TRUE and status not in (AS, UN):
        assert math.isfinite(C.trace().upper)


@LAW
@example(_FACTORED_NEGATIVE)
@given(covariance_case())
def law_covariance_trace_encloses_the_trace(case):
    """A Covariance's trace is an Enclosure that contains the trace of its realised matrix (basis independent: the sum
    of its eigenvalues), and the posterior's bound without a matrix, 0 <= tr C_post <= tr C, contains it too."""
    C = _covariance_of(case)
    if C is None:
        return
    t = float(np.trace(C.matrix()))
    T = C.trace()
    assert isinstance(T, Enclosure)
    assert T.contains(t), f"{C.name}: tr = {t!r} outside {T!r}"
    if hasattr(C, "trace_bound"):
        b = C.trace_bound()
        assert b.contains(t), f"{C.name}: tr = {t!r} outside the bound {b!r}"


_J = Interval("[0.2, 0.6]")
_K = Interval("[0.5, 1]")
FUNCTIONAL_CASES = {   # name: (space, element maker, the package's functionals on it)
    "R3": (lambda: pg.Euclidean(3, weights=[1.0, 2.0, 3.0]), lambda X, c: X(np.array([1.0, -2.0, 0.5]) + c),
           lambda X: [X.functional(np.array([1.0, 0.5, -1.0])), X.functional(np.array([0.0, 2.0, 1.0])),
                      pg.Functional.from_operator(pg.Matrix(np.array([[1.0, -1.0, 2.0]]), domain=X,
                                                            codomain=pg.Euclidean(1)))]),
    "C2": (lambda: pg.Euclidean(2, field="C"), lambda X, c: X(np.array([1.0 + 1j, -2.0]) + c),
           lambda X: [X.functional(np.array([1.0 - 1j, 0.5])), X.functional(np.array([2.0j, 1.0]))]),
    "L2": (lambda: pg.L2(I01), lambda X, c: X(lambda t: np.cos(3 * t) + c),
           lambda X: [X.functional(pg.LebesgueMeasure(_J)), X.average(_K),
                      pg.Functionals([X(lambda t: t), X(lambda t: t ** 2)])[1]]),
    "H1": (lambda: pg.Sobolev(I01, 1, length_scale=0.2), lambda X, c: X(lambda t: np.cos(3 * t) + c),
           lambda X: [X.evaluation(0.3), X.functional(pg.LebesgueMeasure(_J)), X.average(_K)]),
    "C[0,1]": (lambda: pg.Continuous("[0, 1]"), lambda X, c: X(lambda t: np.cos(3 * t) + c),
               lambda X: [X.evaluation(0.3), X.average(_J), X.integrals([0.2, 0.7])[1],
                          X.functional(pg.LebesgueMeasure(_K), scale=-2.0)]),
    "Lp3": (lambda: pg.Lp(I01, 3), lambda X, c: X(lambda t: np.cos(3 * t) + c),
            lambda X: [X.functional(pg.LebesgueMeasure(_J)), X.functional(pg.LebesgueMeasure(_K))]),
    "Lp1": (lambda: pg.Lp(I01, 1), lambda X, c: X(lambda t: np.cos(3 * t) + c),
            lambda X: [X.functional(pg.LebesgueMeasure(_J)), X.functional(pg.LebesgueMeasure(_K))]),
}
_FUNCTIONALS = {}


def _functional_case(name):
    if name not in _FUNCTIONALS:
        make, element, build = FUNCTIONAL_CASES[name]
        X = make()
        _FUNCTIONALS[name] = (X, element, build(X))
    return _FUNCTIONALS[name]


@LAW
@example("L2", 0, 1, 0.5, -2.0)
@example("C[0,1]", 1, 0, 1.0, 0.25)
@example("Lp3", 0, 1, -1.5, 2.0)
@given(st.sampled_from(sorted(FUNCTIONAL_CASES)), st.integers(0, 3), st.integers(0, 3), st.floats(-3, 3),
       st.floats(-3, 3))
def law_every_functional_is_a_functional_of_the_dual(name, i, j, a, b):
    """Every functional the package builds on a space X (X.functional, X.evaluation, X.average, G[i] of
    pg.Functionals and pg.Integrals, Functional.from_operator) is a Functional, an element of X.dual(): linear,
    |l(x)| <= ||l|| ||x||, and the dual's zero and linear combinations a l + b m are elements of X.dual() that act
    as 0 and a l(x) + b m(x)."""
    X, element, ls = _functional_case(name)
    l, m = ls[i % len(ls)], ls[j % len(ls)]
    D = X.dual()
    x, y = element(X, 0.0), element(X, 0.5)
    for f in (l, m):
        assert isinstance(f, pg.Functional) and f.space == X and D.member(f) is Truth.TRUE, (name, f)
        fx, fy = f(x), f(y)
        tol = 1e-9 * (1.0 + abs(fx) + abs(fy))
        assert abs(f(X.linear_combination([a, b], [x, y])) - (a * fx + b * fy)) <= (1 + abs(a) + abs(b)) * tol, name
        assert abs(fx) <= f.norm() * X.norm(x) * (1 + 1e-9) + 1e-12, (name, f, fx, f.norm())
    z = D.zero
    assert isinstance(z, pg.Functional) and D.member(z) is Truth.TRUE and z(x) == 0, name
    c = D.linear_combination([a, b], [l, m])
    assert isinstance(c, pg.Functional) and D.member(c) is Truth.TRUE, name
    assert abs(c(x) - (a * l(x) + b * m(x))) <= 1e-9 * (1.0 + abs(a * l(x)) + abs(b * m(x))), name


# ------------------------------------------------------------------ finite subspaces and discretisations (plan 068 P6)
def _hat01(j, n):
    return lambda t: np.maximum(0.0, 1.0 - np.abs(n * np.asarray(t, float) - j))


def _legendre01(k):
    return lambda t: np.sqrt(2 * k + 1) * np.polynomial.legendre.Legendre.basis(k)(2 * np.asarray(t, float) - 1)


_W4 = np.array([[2.0, 0.3, 0.0, 0.1], [0.3, 1.0, 0.2, 0.0], [0.0, 0.2, 3.0, 0.4], [0.1, 0.0, 0.4, 1.5]])
_G4 = np.array([[1.0, 0.0, 2.0, -1.0], [0.0, 1.0, 1.0, 0.5], [2.0, 1.0, 5.0, -1.5]]).T     # third = 2 first + second


def _span_cases():
    """name -> (build the Span, an element of its space linear in (a, b) for each k): built on demand, cached."""
    from pygeoinf0.convex.subspaces import Span
    I1, B2 = Interval("[0, 1]"), pg.Box((0, 1), (0, 1))
    return {
        "R^4 weighted, dependent generators": (
            lambda: Span([pg.Euclidean(4, metric=_W4)(c) for c in _G4.T]),
            lambda X, a, b, k: X(np.array([a, b, a + k * b, k * a - b]))),
        "R^3, generators of scales 1 and 1e-9": (
            lambda: Span([pg.Euclidean(3)(np.array([1.0, 1.0, 0.0])), pg.Euclidean(3)(np.array([0.0, 1e-9, 1e-9]))]),
            lambda X, a, b, k: X(np.array([a, b, k]))),
        "L2[0, 1], hats": (lambda: Span(F_bases().hats(I1, 9), space=pg.L2(I1)),
                           lambda X, a, b, k: X(lambda t: a * np.sin(k * np.pi * t) + b * t ** 2)),
        "L2[0, 1], Legendre": (lambda: Span(F_bases().legendre(I1, 4), space=pg.L2(I1)),
                               lambda X, a, b, k: X(lambda t: a * np.cos(k * t) + b * np.exp(t))),
        "H1 form of the square, hats": (
            lambda: Span(F_bases().hats(B2, 6), space=pg.SobolevForm(B2, 1, length_scale=0.2).space()),
            lambda X, a, b, k: X(lambda x, y: a * np.sin(k * x) * y + b * x * (1 - y))),
        "L2 of the square, 6400 local hats (sparse Gram)": (
            lambda: Span(F_bases().hats(B2, 82), space=pg.L2(B2, realisation="local")),
            lambda X, a, b, k: X(lambda x, y: a * np.sin(k * np.pi * x) * y + b * x)),
    }


def F_bases():
    import pygeoinf0.functions as F
    return F.bases


_SPANS = {}


def _span_case(name):
    """(V, element): element(X, a, b, k) = a p_k + b q for elements p_k (k = 1..5) and q made once (fits are slow;
    linear combinations of realised elements are not)."""
    if name not in _SPANS:
        build, make = _span_cases()[name]
        V = build()
        parts = {}

        def element(X, a, b, k):
            if k not in parts:
                parts[k] = make(X, 1.0, 0.0, k)
            if "q" not in parts:
                parts["q"] = make(X, 0.0, 1.0, 1)
            return parts[k] * a + parts["q"] * b
        _SPANS[name] = (V, element)
    return _SPANS[name]


_SPAN_NAMES = ["R^4 weighted, dependent generators", "R^3, generators of scales 1 and 1e-9", "L2[0, 1], hats",
               "L2[0, 1], Legendre", "H1 form of the square, hats"]
# the sparse case (6400 local hats, about 25 s an example on europa) runs as an explicit example only


@LAW
@example("R^4 weighted, dependent generators", 1.0, -2.0, 1)
@example("L2 of the square, 6400 local hats (sparse Gram)", 1.0, -0.5, 2)
@example("R^3, generators of scales 1 and 1e-9", 0.5, 2.0, 3)
@given(st.sampled_from(_SPAN_NAMES), st.floats(-3, 3).map(lambda v: round(v, 6)),
       st.floats(-3, 3).map(lambda v: round(v, 6)), st.integers(1, 4))   # no subnormal coefficients
def law_finite_subspace_projection_is_orthogonal(name, a, b, k):
    """The projection of a finite subspace V = span{v_1, ..., v_k} (pg.Span, convex/subspaces.py) is the orthogonal
    projection: idempotent, self-adjoint in the space's inner product, its range in V (member TRUE) and every generator
    fixed (range = V), the residual x - P x orthogonal to every generator; with independent generators the coefficient
    space carries the Gram, so ||S c||_M = ||c|| (dependent, badly scaled, sparse and form-space spans included)."""
    V, element = _span_case(name)
    X = V.space
    x, y = element(X, a, b, k), element(X, b, -a, k + 1)
    nx, ny = X.norm(x), X.norm(y)
    tol = 1e-9
    Px, Py = V.project(x), V.project(y)
    assert X.norm(V.project(Px) - Px) <= tol * (nx + 1e-300), (name, "idempotent")
    assert abs(X.inner(Px, y) - X.inner(x, Py)) <= tol * nx * ny + 1e-300, (name, "self-adjoint")
    assert X.norm(Px) <= nx * (1 + tol), (name, "norm <= 1")
    if X.norm(Px) > tol * nx:
        assert V.member(Px) is Truth.TRUE, (name, "range in V")
    n = len(V.vectors)
    gens = [V.vectors[i] for i in sorted({0, n // 3, (2 * n) // 3, n - 1})]   # not all 6400 of the sparse case
    for v in gens:
        nv = X.norm(v)
        if nv > 0:
            assert X.norm(V.project(v) - v) <= tol * nv, (name, "generator fixed")
        assert abs(X.inner(x - Px, v)) <= tol * nx * nv + 1e-300, (name, "residual orthogonal")
    if V.dim == len(V.vectors):
        C = V.coefficient_space
        c = C(np.linspace(-1.0, 1.0, V.dim) * a + b)
        assert abs(C.norm(c) - X.norm(V.synthesis(c))) <= tol * (C.norm(c) + 1e-300), (name, "coefficient metric")


def _l2_reference(gens, f, edges, q=12):
    """dist(f, span gens) in L2(0, 1) by an independent rule: composite Gauss-Legendre with q points on each panel of
    the given edges, the Gram and loads solved by least squares, the distance the rule's norm of f - sum c_i g_i."""
    g, w = np.polynomial.legendre.leggauss(q)
    a, b = np.asarray(edges[:-1]), np.asarray(edges[1:])
    t = (0.5 * (a + b))[:, None] + (0.5 * (b - a))[:, None] * g
    wt = ((0.5 * (b - a))[:, None] * w).ravel()
    t = t.ravel()
    A = np.array([gi(t) for gi in gens]).T * np.sqrt(wt)[:, None]
    r = np.asarray(f(t), float) * np.sqrt(wt)
    c, *_ = np.linalg.lstsq(A, r, rcond=None)
    return float(np.linalg.norm(r - A @ c))


def _discretisation_cases():
    """name -> (the Discretisation, an element and its plain function from (a, w), the independent distance)."""
    from pygeoinf0.convex.subspaces import Span
    I1 = Interval("[0, 1]")
    L = pg.L2(I1)
    hats = [_hat01(j, 8) for j in range(1, 8)]
    legs = [_legendre01(k) for k in range(5)]
    X = pg.Euclidean(4, metric=_W4)
    return {
        "L2[0, 1], hats": (lambda: pg.Discretisation(Span(F_bases().hats(I1, 9), space=L)),
                           lambda a, w: (lambda t: np.sin(w * np.pi * t) + a * t),
                           lambda fn: _l2_reference(hats, fn, np.linspace(0, 1, 65))),
        "L2[0, 1], Legendre": (lambda: pg.Discretisation(Span(F_bases().legendre(I1, 4), space=L)),
                               lambda a, w: (lambda t: np.sin(w * np.pi * t) + a * t),
                               lambda fn: _l2_reference(legs, fn, np.linspace(0, 1, 65))),
        "R^4 weighted": (lambda: pg.Discretisation(Span([X(c) for c in _G4.T[:2]])),
                         lambda a, w: np.array([a, w, a * w, 1.0 - a]),
                         _euclid_dist_two),
    }


def _euclid_dist_two(v):
    """dist(v, span of the first two generators) in the metric W: ||x||_W = ||A x||_2 with W = A^T A (Cholesky)."""
    A = np.linalg.cholesky(_W4).T
    G = _G4[:, :2]
    c, *_ = np.linalg.lstsq(A @ G, A @ v, rcond=None)
    return float(np.linalg.norm(A @ (v - G @ c)))


_DISCRETISATIONS = {}
_DISC_NAMES = ["L2[0, 1], hats", "L2[0, 1], Legendre", "R^4 weighted"]


def _discretisation_case(name):
    if name not in _DISCRETISATIONS:
        build, fn, ref = _discretisation_cases()[name]
        _DISCRETISATIONS[name] = (build(), fn, ref)
    return _DISCRETISATIONS[name]


def _checked_element(D, fn, a, w):
    plain = fn(a, w)
    return D.space(plain), plain


@LAW
@example("L2[0, 1], hats", 0.5, 1.0)
@example("R^4 weighted", -1.0, 2.0)
@given(st.sampled_from(_DISC_NAMES), st.floats(-2, 2), st.floats(0.5, 6))
def law_discretisation_claim_never_stronger_than_checked(name, a, w):
    """A Discretisation's approximation claim is never stronger than what was checked: checking elements records a
    CHECKED claim about those elements only; the claim about a class of elements stays what was stated (ASSERTED by
    D.state) or UNKNOWN, never CHECKED or stronger by a sample; and the uniform approximation of M by the span is a
    theorem, FALSE whenever the span is smaller than M (a unit element orthogonal to it exists)."""
    from pygeoinf0.core.claims import ASSERTED, CHECKED, THEOREM
    D, fn, _ = _discretisation_case(name)
    f, _ = _checked_element(D, fn, a, w)
    before = D.claims.find("approximation").status
    (e,) = D.check([f])
    assert e.status == CHECKED, (name, e.status)
    after = D.claims.find("approximation")
    assert after.status == before and RANK[after.status] >= RANK[ASSERTED], (name, after)
    chk = D.claims.find("checked approximation")
    assert chk.status == CHECKED and "element" in chk.statement, (name, chk)
    u = D.claims.find("uniform approximation")
    full = isinstance(D.space, spaces_base.Euclidean) and D.space.dim == D.subspace.dim
    assert u.status == THEOREM and u.holds is full, (name, u)


@LAW
@example("L2[0, 1], hats", 0.5, 1.0)
@example("R^4 weighted", -1.0, 2.0)
@given(st.sampled_from(_DISC_NAMES), st.floats(-2, 2), st.floats(0.5, 6))
def law_discretisation_error_is_stated_on_the_function_space(name, a, w):
    """The error a Discretisation states for an element f is dist(f, V) = ||f - P f|| in the norm of the space M, never
    a quantity in coefficient coordinates: its checked Enclosure contains the distance computed by an independent rule
    (composite Gauss-Legendre with a least-squares fit on [0, 1]; the Cholesky factor of the metric on R^4)."""
    D, fn, ref = _discretisation_case(name)
    f, plain = _checked_element(D, fn, a, w)
    (e,) = D.check([f])
    true = ref(plain)
    slack = 1e-9 * (1.0 + D.space.norm(f))
    assert e.lower - slack <= true <= e.upper + slack, (name, float(e), e.lower, e.upper, true)


# ------------------------------------------------------------------ quadrature (plan 069 P5)
QUAD_RULES = ("gauss", "gauss01", "panels", "graded_unit", "both_ends", "trapezoid", "simplex")


def _quad():
    from pygeoinf0 import quadrature
    return quadrature


@st.composite
def quadrature_case(draw):
    """A rule of pygeoinf0.quadrature with drawn parameters: (name, n, nodes, weights, pieces, k). pieces are the
    intervals on which the rule's stated degree holds in the local variable (one interval for a single rule, the panels of
    a composite or graded rule); k is the dimension of the reference simplex (pieces None)."""
    Q = _quad()
    name = draw(st.sampled_from(QUAD_RULES))
    if name == "simplex":
        k = draw(st.integers(1, 3))
        n = draw(st.integers(1, (6, 5, 4)[k - 1]))
        X, w = Q.simplex(k, n)
        return name, n, X, w, None, k
    if name == "gauss":
        n = draw(st.integers(1, 40))
        x, w = Q.gauss(n)
        return name, n, x, w, [(-1.0, 1.0)], None
    if name == "gauss01":
        n = draw(st.integers(1, 40))
        x, w = Q.gauss01(n)
        return name, n, x, w, [(0.0, 1.0)], None
    if name == "trapezoid":
        n, a = draw(st.integers(2, 300)), draw(st.floats(-2, 2))
        b = a + draw(st.floats(0.1, 4))
        x, w = Q.trapezoid(n, a, b)
        return name, n, x, w, [(a, b)], None
    n = draw(st.integers(1, 12))
    if name == "panels":
        gaps = draw(hnp.arrays(float, draw(st.integers(1, 6)), elements=st.floats(0.05, 1.0)))
        edges = draw(st.floats(-2, 2)) + np.r_[0.0, np.cumsum(gaps)]
        x, w = Q.panels(edges, n)
    elif name == "graded_unit":
        cells, depth = draw(st.integers(1, 4)), 10.0 ** -draw(st.integers(2, 14))
        x, w, h = Q.graded_unit(n, cells, depth)
        edges = Q.graded_edges(cells, depth)
        assert edges[0] == h, (name, h, edges[0])
    else:
        cells, depth = draw(st.integers(1, 6)), 10.0 ** -draw(st.integers(2, 12))
        x, w = Q.both_ends(n, cells, depth)
        edges = Q.both_ends_edges(cells, depth)
    return name, n, x, w, list(zip(edges[:-1], edges[1:])), None


def _piecewise_legendre(x, pieces, C):
    """Values at x of the piecewise polynomial with Legendre coefficients C[j] in the local variable of piece j."""
    edges = np.array([lo for lo, _ in pieces] + [pieces[-1][1]])
    j = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(pieces) - 1)
    lo, hi = edges[j], edges[j + 1]
    t = np.clip((2 * x - lo - hi) / (hi - lo), -1.0, 1.0)
    return (np.polynomial.legendre.legvander(t, C.shape[1] - 1) * C[j]).sum(axis=1)


def _node_rounding(pieces, C):
    """The error a rule commits on the piecewise polynomial C only by its nodes being rounded to doubles (|x| eps each):
    by Markov's inequality |p'| <= 2 d^2 max|p| / h on a piece of length h, so at most 2 d^2 eps max(|lo|, |hi|) sum|c|
    per piece (times 4 for the evaluation)."""
    d = C.shape[1] - 1
    return 8 * d * d * EPS * sum(max(abs(lo), abs(hi)) * np.abs(c).sum() for (lo, hi), c in zip(pieces, C))


def _simplex_monomials(k, d):
    return [a for a in np.ndindex(*(d + 1,) * k) if sum(a) <= d]


@LAW
@given(quadrature_case(), st.data())
def law_quadrature_rules_integrate_their_stated_degree(case, data):
    """Every rule of pygeoinf0.quadrature integrates the polynomials of its stated degree d = rule.exact_degree(n)
    exactly, by the precision rule (core.precision.TOL relative to the integral of |p|, beyond the rounding of the nodes
    to doubles: _node_rounding): random polynomials of degree d in the local variable of each piece (Legendre
    coefficients; the integral is the length times c_0), and on the reference simplex random combinations of the
    monomials of total degree <= d (Dirichlet: int u^a = prod a_i! / (|a| + k)!)."""
    from pygeoinf0.core.precision import TOL
    name, n, x, w, pieces, k = case
    d = getattr(_quad(), name).exact_degree(n)
    if k is None:
        C = data.draw(hnp.arrays(float, (len(pieces), d + 1), elements=st.floats(-1, 1)))
        C = np.where(np.abs(C) < 1e-30, 0.0, C)                    # subnormal products are no rule's error
        value = w @ _piecewise_legendre(x, pieces, C)
        exact = sum((hi - lo) * c[0] for (lo, hi), c in zip(pieces, C))
        scale = sum((hi - lo) * np.abs(c).sum() for (lo, hi), c in zip(pieces, C))
        floor = _node_rounding(pieces, C)
    else:
        alphas = _simplex_monomials(k, d)
        c = data.draw(hnp.arrays(float, len(alphas), elements=st.floats(-1, 1)))
        c = np.where(np.abs(c) < 1e-30, 0.0, c)
        terms = np.array([math.prod(math.factorial(e) for e in a) / math.factorial(sum(a) + k) for a in alphas])
        value = w @ sum(ci * np.prod(x ** np.array(a), axis=1) for ci, a in zip(c, alphas))
        exact, scale, floor = c @ terms, np.abs(c) @ terms, 0.0
    assert abs(value - exact) <= TOL * scale + floor, (name, n, k, value, exact, scale, floor)


@LAW
@given(quadrature_case(), st.data())
def law_quadrature_rules_are_sharp_where_they_claim(case, data):
    """No rule integrates one degree more than it states: the polynomial P_(d+1) of the local variable on one piece
    (its integral 0; Legendre, or on the simplex the Jacobi polynomial P_(d+1)^(k-1, 0)(2 u_1 - 1), orthogonal to
    the constants under the simplex's marginal (1 - u_1)^(k-1) / (k-1)!) gets a rule value above the precision rule
    times the bound sqrt(volume) ||P||_2 of its integral of |P|, beyond the rounding of the nodes (_node_rounding), on
    every piece wider than 1e-6 of its distance from 0 (narrower ones are not resolved by doubles to witness it)."""
    from scipy.special import eval_jacobi

    from pygeoinf0.core.precision import TOL
    name, n, x, w, pieces, k = case
    m = getattr(_quad(), name).exact_degree(n) + 1
    if k is None:
        wide = [i for i, (lo, hi) in enumerate(pieces) if hi - lo >= 1e-6 * max(abs(lo), abs(hi))]
        j = data.draw(st.sampled_from(wide))
        C = np.zeros((len(pieces), m + 1))
        C[j, m] = 1.0
        value = w @ _piecewise_legendre(x, pieces, C)
        lo, hi = pieces[j]
        scale = (hi - lo) / math.sqrt(2 * m + 1)                   # sqrt(h) ||P_m||_(L2 of the piece)
        floor = _node_rounding(pieces, C)
    else:
        value = w @ eval_jacobi(m, k - 1, 0, 2 * x[:, 0] - 1)
        scale, floor = math.sqrt(1 / math.factorial(k) / ((2 * m + k) * math.factorial(k - 1))), 0.0
    assert abs(value) > TOL * scale + floor, (name, n, k, m, value, scale, floor)


# ------------------------------------------------------------------ rules of regions (plan 069 P2)
@st.composite
def region_case(draw):
    """A region with charts and its exact volume and centroid (None where not closed-form): intervals, boxes, disks,
    balls, shells, cylinders, spheres, circles, tori, polylines, flat simplices and ball pieces (a ball cut by a shell
    about another centre), with drawn sizes and centres."""
    from pygeoinf0.domains.pieces import BallPiece
    size = st.floats(0.3, 3.0)
    c2 = tuple(draw(st.floats(-2, 2)) for _ in range(2))
    c3 = c2 + (draw(st.floats(-2, 2)),)
    kind = draw(st.sampled_from(("interval", "box2", "box3", "disk", "ball", "shell", "cylinder", "sphere", "circle",
                                 "torus", "polyline", "simplex", "ballpiece")))
    R = draw(size)
    if kind == "interval":
        a = draw(st.floats(-2, 2))
        return kind, pg.Interval(a, a + R), (a + R / 2,)
    if kind in ("box2", "box3", "torus"):
        sides = [(c, c + draw(size)) for c in (c3 if kind == "box3" else c2)]
        if kind == "torus":                                 # the coordinates are not periodic: no centroid check
            return kind, pg.Torus(*sides), None
        return kind, pg.Box(*sides), tuple((lo + hi) / 2 for lo, hi in sides)
    if kind == "disk":
        return kind, pg.Disk(R, c2), c2
    if kind == "circle":
        return kind, pg.Circle(R, c2), c2
    if kind == "ball":
        return kind, pg.Ball(R, c3), c3
    if kind == "sphere":
        return kind, pg.Sphere(R, c3), c3
    if kind == "shell":
        return kind, pg.Shell(R * draw(st.floats(0.1, 0.9)), R, c3), c3
    if kind == "cylinder":
        z0 = draw(st.floats(-1, 1))
        z1 = z0 + draw(size)
        return kind, pg.Cylinder(R, (z0, z1), c2), c2 + ((z0 + z1) / 2,)
    if kind == "polyline":
        P = np.array(c2) + np.cumsum(draw(hnp.arrays(float, (draw(st.integers(2, 5)), 2),
                                                     elements=st.floats(0.2, 1.0))), axis=0)
        L = np.linalg.norm(np.diff(P, axis=0), axis=1)
        return kind, pg.Polyline(P), tuple(L @ (0.5 * (P[1:] + P[:-1])) / L.sum())
    if kind == "simplex":
        skew = st.floats(-0.2, 0.2)                         # a f - b e >= 0.05: never degenerate
        V = np.array(c2) + np.array([[0.0, 0.0], [draw(size), draw(skew)], [draw(skew), draw(size)]])
        return kind, pg.Simplex(*V), tuple(V.mean(axis=0))
    d = R * draw(st.floats(0.2, 1.5))                       # ball piece: B(c, rho) cut by a shell about c - d e_x
    C = (c3[0] - d,) + c3[1:]
    inner = draw(st.floats(0.0, 0.6)) * d
    outer = d + R * draw(st.floats(0.2, 1.2))
    if not inner < outer or outer <= d - R:
        reject()
    return kind, BallPiece((c3, R), (C, inner, outer)), None


@LAW
@given(region_case(), st.integers(12, 24))
def law_region_rules_carry_the_region_measure(case, n):
    """The rule of a region (pygeoinf0.quadrature.rule, through the region's charts) carries the region's Hausdorff
    measure: its weights sum to the volume (exact for every region here) and it integrates the coordinates to the
    volume times the centroid, by the precision rule (core.precision.TOL relative to the volume times the size of the
    region), with its nodes in the region for solid regions."""
    from pygeoinf0 import quadrature
    from pygeoinf0.core.precision import TOL
    kind, S, centroid = case
    X, w = quadrature.rule(S, n)
    V = float(S.volume)
    assert X.shape == (len(w), S.cartesian_dim), (kind, X.shape)
    assert abs(w.sum() - V) <= TOL * V, (kind, n, float(w.sum()), V)
    if centroid is not None:
        scale = V * (1.0 + float(np.max(np.abs(X))))
        assert np.all(np.abs(w @ X - V * np.asarray(centroid)) <= TOL * scale), (kind, n, w @ X, V, centroid)
    if S.dim == S.cartesian_dim and kind != "ballpiece":
        assert np.all(S.contains(X)), (kind, n)


# ------------------------------------------------------------------ planted defects (the controls)
def _adjoint_forgets_domain_metric(mp):
    def bad(self):
        return linear.Matrix(self.codomain.apply_metric(self.A).T, domain=self.codomain, codomain=self.domain,
                             name=f"{self.name}*")
    mp.setattr(linear.Matrix, "_make_adjoint", bad)


def _composition_adjoint_wrong_order(mp):
    mp.setattr(linear.Composition, "_make_adjoint", lambda self: linear.Composition(self.A.adjoint, self.B.adjoint))


def _sum_adjoint_drops_coefficients(mp):
    mp.setattr(linear.Sum, "_make_adjoint", lambda self: linear.Sum([(1.0, A.adjoint) for _, A in self.terms]))


def _metric_applied_unsymmetrised(mp):
    init = metrics._Ops.__init__

    def bad(self, op, base):
        init(self, op, base)
        if self.kind == "dense":
            self.A = op.A
    mp.setattr(metrics._Ops, "__init__", bad)


def _trace_forgets_the_metric(mp):
    mp.setattr(linear.FiniteRank, "trace",
               lambda self: float(self.lam @ np.sum(np.asarray(self.generators.values) ** 2, axis=1)))


def _posterior_drops_trailing_c(mp):
    cond = gaussian.GaussianMeasure._condition

    def bad(self, G, d, noise):
        post = cond(self, G, d, noise)
        post.covariance = self.covariance - (self.covariance @ G.adjoint) @ post.solver @ G
        return post
    mp.setattr(gaussian.GaussianMeasure, "_condition", bad)


def _rule_promotes(mp):
    from pygeoinf0.operators import blocks
    for mod in (linear, blocks, metrics):
        mp.setattr(mod, "by_rule", lambda *premises: DERIVED)


def _kleene_and_is_left(mp):
    mp.setattr(Truth, "__and__", lambda a, b: a)


def _enclosure_scaled_unguarded(mp):
    def bad(self, c, reason=None):
        lo, hi = sorted((c * self.lower, c * self.upper)) if c else (0.0, 0.0)
        return Enclosure(c * float(self), lo, hi, self.status, reason or self.reason)
    mp.setattr(Enclosure, "scaled", bad)


def _enclosure_rounds_to_nearest(mp):
    mp.setattr(numbers, "_down", lambda r: float(r))
    mp.setattr(numbers, "_up", lambda r: float(r))


def _enclosure_max_takes_the_least_upper_end(mp):
    def bad(*terms):
        return Enclosure(max(float(t) for t in terms), max(t.lower for t in terms), min(t.upper for t in terms))
    mp.setattr(numbers, "maximum", bad)


def _contains_reads_undecided_as_true(mp):
    mp.setattr(sets_pkg.Set, "__contains__", lambda self, x: self.member(x) is not Truth.FALSE)


def _points_keep_the_coordinate_axis(mp):
    def bad(points, d):
        X = np.asarray(points, float)
        return X.reshape(-1, d), X.shape if d == 1 and X.ndim else X.shape[:-1]   # (..., 1) read as numbers
    mp.setattr(regions_mod, "as_points", bad)


def _meet_of_intervals_is_none_when_null(mp):
    old = intervals_mod.meet_intervals

    def bad(I, J):
        K = old(I, J)
        return K if isinstance(K, Interval) else None
    mp.setattr(intervals_mod, "meet_intervals", bad)


def _minkowski_composes_values(mp):
    def bad(self, v):                                   # the second part's value, not its upper end (A7's defect)
        return support_of(self.parts[0], v) + float(support_of(self.parts[1], v))
    mp.setattr(calgebra.Minkowski, "support", bad)


def _subset_join_with_empty_is_lazy(mp):
    mp.setattr(subsets_mod.Subset, "__or__", lambda self, other: self._join(other))


def _union_with_empty_is_lazy(mp):
    mp.setattr(Region, "__or__", lambda self, other: regions_mod.RegionUnion([self, other])
               if isinstance(other, Region) else other)


def _probability_claimed_for_any_nonnegative_measure(mp):
    mp.setattr(measures_base, "ROUNDING", 1e300)                   # every non-negative measure "has mass 1"


def _signed_integral_drops_the_coefficients(mp):
    mp.setattr(measures_base.SignedMeasure, "integrate",
               lambda self, f: (float(sum(m.integrate(f)[0] for _, m in self.terms)), 0.0))
    mp.setattr(pg.CountingMeasure, "integrate", lambda self, f: (float(np.sum(f(self.set.points))), 0.0))


def _kind_read_from_the_instance(mp):
    def bad(obj):
        k = getattr(obj, "kind", None)
        return k if isinstance(k, kinds.Kind) else None
    mp.setattr(kinds, "declared_kind", bad)
    mp.setattr(states_base, "declared_kind", bad)


def _whole_space_read_as_a_general_set(mp):
    mp.setattr(cells_backward, "_whole", lambda S: False)          # B5's general route: S ∩ G^-1(d) as a feasible set


def _inner_planted_on_a_banach_space(mp):
    mp.setattr(spaces_base.LpSpace, "inner", lambda self, f, g: 0.0, raising=False)
    mp.setattr(spaces_base.LpSpace, "riesz", lambda self, mu: None, raising=False)


def _complex_metric_applied_transposed(mp):
    apply = metrics._Ops.apply

    def bad(self, V):
        if self.kind == "dense" and np.iscomplexobj(self.A):
            V, one = metrics._cols(V)
            return metrics._out(self.A.T @ V, one)
        return apply(self, V)
    mp.setattr(metrics._Ops, "apply", bad)


def _factored_positivity_ignores_the_diagonal(mp):
    from pygeoinf0.probability import lazy_covariance
    mp.setattr(lazy_covariance, "_extra_nonnegative", lambda extra: True)


def _factored_trace_forgets_the_diagonal(mp):
    from pygeoinf0.probability import lazy_covariance
    mp.setattr(lazy_covariance.FactoredCovariance, "trace",
               lambda self: lazy_covariance._factored_trace(self.Phi, self.V, None))


# ------------------------------------------------------------------ operators (plan 068 P3)
def _unbounded_cases():
    """The closed, densely defined operators of the package (operators/unbounded.py), built on demand."""
    from pygeoinf0.operators import unbounded as ub
    return {
        "form interval": lambda: pg.SobolevForm("[0, 1]", 1, length_scale=0.1).operator(),
        "form box, order 2": lambda: pg.SobolevForm(pg.Box(Interval("[0, 1]"), Interval("[0, 2]")), 2,
                                                     length_scale=0.3).operator(),
        "form sphere": lambda: pg.SobolevForm(pg.Sphere(1.0), 1, length_scale=0.3).operator(),
        # p = 1 + x as a table (exact: plan 071 P2 states a callable p's premises CHECKED, from samples)
        "sturm-liouville": lambda: ub.SturmLiouville("[0, 2]", pg.Tabulated("[0, 2]", [0.0, 2.0], [1.0, 3.0]), 0.5, 2.0,
                                                     left="dirichlet", right=("robin", 0.5), interfaces=[1.0]),
        "radial laplacian": lambda: ub.SturmLiouville.radial_laplacian(0.0, 1.0, 2),
        "laplacian interval": lambda: pg.Laplacian(Interval("[0, 1]"), bc={"left": "dirichlet",
                                                                          "right": ("robin", 1.0)}),
        "laplacian box": lambda: pg.Laplacian(pg.Box(Interval("[0, 1]"), Interval("[0, 2]")), bc="dirichlet"),
        "laplacian ball": lambda: pg.Laplacian(pg.Ball(1.0), bc={"boundary": "dirichlet"}, centre="regular"),
        "laplacian shell": lambda: pg.Laplacian(pg.Shell(0.5, 1.0), bc={"inner": "neumann", "outer": "dirichlet"}),
        "laplacian sphere": lambda: pg.Laplacian(pg.Sphere(2.0)),
        "laplacian circle": lambda: pg.Laplacian(pg.Circle(1.5)),
    }


_UNBOUNDED_NAMES = ["form interval", "form box, order 2", "form sphere", "sturm-liouville", "radial laplacian",
                    "laplacian interval", "laplacian box", "laplacian ball", "laplacian shell", "laplacian sphere",
                    "laplacian circle"]


@LAW
@example("form interval")
@example("laplacian ball")
@given(st.sampled_from(_UNBOUNDED_NAMES))
def law_unbounded_domain_is_dense_and_not_closed(name):
    """An unbounded operator A on X (a FormOperator, a Sturm-Liouville operator, the Laplacian of a domain) is a
    single-valued linear relation whose dom is D(A): a linear subspace of the ambient X that is dense and, A being
    unbounded, not closed (a closed operator with a closed domain is bounded: the closed graph theorem). It is a
    LinearSubspace and not a ClosedSubspace, it has no orthogonal projection (NotDefined), its complement is {0}, and the
    graph-norm space is reached only by domain_space(): a different Hilbert space, or a refusal that says it is not
    built (review A4, B5)."""
    from pygeoinf0.core.claims import THEOREM
    from pygeoinf0.core.errors import NotDefined, Unsupported
    from pygeoinf0.relations.linear import LinearRelation
    A = _unbounded_cases()[name]()
    assert isinstance(A, LinearRelation), (name, type(A).__mro__)
    X = A.domain
    assert A.codomain == X and isinstance(X, spaces_base.HilbertSpace), name
    assert A.is_single_valued is Truth.TRUE, name
    assert A.claims.status("unbounded") == (Truth.TRUE, THEOREM), (name, A.claims.status("unbounded"))
    assert A.claims.status("closed graph") == (Truth.TRUE, THEOREM), name
    assert A.claims.status("total") == (Truth.FALSE, THEOREM), name
    D = A.dom
    assert isinstance(D, csub.LinearSubspace) and not isinstance(D, csub.ClosedSubspace), (name, type(D).__mro__)
    assert D.space == X, name
    assert D.claims.status("dense") == (Truth.TRUE, THEOREM), (name, D.claims.status("dense"))
    assert D.claims.status("closed") == (Truth.FALSE, THEOREM), (name, D.claims.status("closed"))
    assert isinstance(D.perp(), csub.Zero) and D.perp().space == X, name
    with pytest.raises(NotDefined):
        D.project(X.zero)
    try:
        V = A.domain_space()
    except (Unsupported, NotDefined):
        V = None
    assert V is None or (isinstance(V, spaces_base.HilbertSpace) and V != X), (name, V)


@st.composite
def error_problem(draw):
    """Square matrices A~, B~ on R^n, true perturbations dA, dB of spectral norm exactly eA, eB (the true operators are
    A = A~ - dA, B = B~ - dB), and two coefficients."""
    n = draw(st.integers(2, 4))
    mat = lambda: draw(hnp.arrays(float, (n, n), elements=st.floats(-3, 3, allow_nan=False)))  # noqa: E731
    At, Bt, RA, RB = mat() + 4 * np.eye(n), mat(), mat(), mat()
    if min(np.linalg.norm(RA, 2), np.linalg.norm(RB, 2)) < 1e-3:
        reject()
    eA, eB = draw(st.floats(1e-6, 2.0)), draw(st.floats(1e-6, 2.0))
    c1, c2 = draw(st.floats(-4, 4)), draw(st.floats(-4, 4))
    return At, Bt, eA * RA / np.linalg.norm(RA, 2), eB * RB / np.linalg.norm(RB, 2), eA, eB, c1, c2


def _with_error(M, e, X, name):
    from pygeoinf0.operators import error as operror
    A = pg.Matrix(M, X, X, name=name)
    operror.state(A, e, f"planted: ||{name}~ - {name}|| <= {e:g}")
    return A


_E1 = np.array([[1.0, 0.0], [0.0, 1.0]])


@LAW
@example((4 * np.eye(2), _E1, 0.5 * _E1, 0.25 * _E1, 0.5, 0.25, 3.0, -2.0))
@given(error_problem())
def law_operator_error_rules_bound_the_true_error(prob):
    """The stated error ||A~ - A|| <= e of an operator built from others (operators/error.py) bounds the true error:
    sums sum |c_j| e_j, compositions e_A ||B~|| + ||A~|| e_B + e_A e_B (also through a structural shortcut, a Matrix
    product), inverses n^2 e / (1 - n e) with n = ||A~^-1|| when n e < 1, and UNKNOWN (never "rounding only") when
    n e >= 1, where A may be singular."""
    from pygeoinf0.core.claims import UNKNOWN
    from pygeoinf0.solvers import stated_error
    At, Bt, dA, dB, eA, eB, c1, c2 = prob
    n = len(At)
    X = pg.Euclidean(n)
    A, B = _with_error(At, eA, X, "A"), _with_error(Bt, eB, X, "B")
    Atrue, Btrue = At - dA, Bt - dB
    slack = 1e-9 * (1 + np.abs(At).max() + np.abs(Bt).max()) ** 3
    cases = [(c1 * A + c2 * B, c1 * Atrue + c2 * Btrue), (A @ B, Atrue @ Btrue), (B @ A, Btrue @ Atrue)]
    for op, true in cases:
        st_ = stated_error(op)
        assert st_ is not None, op.name
        assert np.linalg.norm(op.matrix() - true, 2) <= st_[0] + slack, (op.name, st_, np.linalg.norm(op.matrix() - true, 2))
    inv = A.inverse(pg.Direct())
    st_ = stated_error(inv)
    assert st_ is not None, "the inverse of an operator with a stated error states one"
    ninv = 1.0 / np.linalg.svd(At, compute_uv=False)[-1]
    if ninv * eA < 1 - 1e-9:
        true = np.linalg.inv(Atrue)
        assert np.linalg.norm(np.linalg.inv(At) - true, 2) <= st_[0] + slack * ninv ** 2, (st_, ninv, eA)
    elif ninv * eA > 1 + 1e-9:
        assert math.isnan(st_[0]) and st_[1] == UNKNOWN, st_


@LAW
@given(st.integers(2, 4), st.floats(1e-6, 1.0))
def law_operator_error_unknown_where_no_rule_applies(n, e):
    """Where no rule bounds the error of an operator built from one with a stated error, the statement is UNKNOWN (a NaN
    bound with status unknown), never None ("nothing beyond rounding", review A6): a composition with a factor whose
    norm is not on record, and a pseudo-inverse (a perturbation can move eigenvalues across its threshold)."""
    from pygeoinf0.core.claims import UNKNOWN
    from pygeoinf0.solvers import stated_error
    X = pg.Euclidean(n)
    A = _with_error(np.eye(n) * 2.0, e, X, "A")
    F = pg.LinearOperator(X, X, apply=lambda x: x, adjoint=lambda x: x, name="F")    # no norm on record
    for op in (F @ A, A @ F, A.pseudo_inverse(1e-12)):
        st_ = stated_error(op)
        assert st_ is not None and math.isnan(st_[0]) and st_[1] == UNKNOWN, (op.name, st_)


def _operator_domain_claimed_closed(mp):
    from pygeoinf0.core.claims import THEOREM, Claim, Claims
    from pygeoinf0.relations import linear as rlinear
    mp.setattr(rlinear.OperatorDomain, "claims", property(lambda self: Claims([
        Claim("closed linear subspace", THEOREM, "the old claim (review A4)", key="closed"),
        Claim("dense", THEOREM, "Kato VI.2.1", key="dense")])))


def _sum_rule_drops_the_coefficients(mp):
    from pygeoinf0.operators import error as operror
    rule = operror.sum_error
    mp.setattr(operror, "sum_error", lambda terms, error: rule([(1.0, B) for _, B in terms], error))


def _no_norm_read_as_rounding_only(mp):
    from pygeoinf0.operators import error as operror
    rule = operror.compose_error

    def bad(A, B, error):
        out = rule(A, B, error)
        return None if out is not None and math.isnan(out[0]) else out
    mp.setattr(operror, "compose_error", bad)
    mp.setattr(operror, "pseudo_inverse_error", lambda A, error: None)


def _functional_returned_as_a_bare_form(mp):
    class Form:                                       # the old LinearForm: a callable with a space and a norm
        def __init__(self, r):
            self.space, self.kernel = r.space, r

        def __call__(self, x):
            return self.space.inner(self.kernel, x)

        def norm(self):
            return self.kernel.norm()

    def bad(self, spec, name=None):
        return Form(self.riesz(spec))
    mp.setattr(spaces_base.HilbertSpace, "functional", bad)
    for k in list(_FUNCTIONALS):                       # built again with the defect
        mp.delitem(_FUNCTIONALS, k)


# ------------------------------------------------------------------ the data space and blocks (plan 068 P5)
@st.composite
def euclidean_summands(draw):
    """Real Euclidean summands of every kind (a data space, the dot product, weights, a metric, a covariance), summand
    weights, and two vectors of the sum's dimension."""
    spaces = []
    for _ in range(draw(st.integers(1, 3))):
        n = draw(st.integers(1, 3))
        kind = draw(st.sampled_from(["data", "plain", "weights", "metric", "covariance"]))
        if kind == "data":
            spaces.append(pg.DataSpace(n))
        elif kind == "plain":
            spaces.append(pg.Euclidean(n))
        elif kind == "weights":
            spaces.append(pg.Euclidean(n, weights=draw(hnp.arrays(float, n, elements=st.floats(0.1, 10.0)))))
        else:
            spaces.append(pg.Euclidean(n, **{kind: draw(spd(n, max_cond=1e3))}))
    w = draw(st.one_of(st.none(), hnp.arrays(float, len(spaces), elements=st.floats(0.1, 10.0))))
    N = sum(E.dim for E in spaces)
    return spaces, w, draw(hnp.arrays(float, N, elements=finite)), draw(hnp.arrays(float, N, elements=finite))


_SUM_EXAMPLE = ([pg.Euclidean(2, weights=[1.0, 3.0]), pg.Euclidean(1)], np.array([2.0, 0.5]),
                np.array([1.0, 2.0, -1.0]), np.array([0.5, -1.0, 2.0]))


@LAW
@example(_SUM_EXAMPLE)
@given(euclidean_summands())
def law_direct_sum_of_euclideans_is_a_euclidean(prob):
    """A direct sum of Euclideans is a Euclidean (Q9): R^(n_1 + ... + n_K) whose coordinates carry block labels
    (block, coordinate), with the inner product of the direct sum, <x, y> = sum_k w_k <x_k, y_k>_k; plain data spaces
    with unit weights sum to a plain data space."""
    from pygeoinf0.spaces import DataSpace, euclidean_sum
    spaces, w, a, b = prob
    E = euclidean_sum(spaces, weights=w)
    X = pg.DirectSum(spaces, weights=w)
    assert isinstance(E, pg.Euclidean) and E.dim == len(a) and E.fields == ("block", "coordinate")
    ends = np.cumsum([0] + [D.dim for D in spaces])
    for k in range(len(spaces)):
        assert list(E.select(block=k)) == list(range(ends[k], ends[k + 1])), k
    x = X({k: a[ends[k]:ends[k + 1]] for k in range(len(spaces))})
    y = X({k: b[ends[k]:ends[k + 1]] for k in range(len(spaces))})
    lhs, rhs = E.inner(E(a), E(b)), X.inner(x, y)
    scale = 1.0 + X.norm(x) * X.norm(y)
    assert abs(lhs - rhs) <= 1e-10 * scale, (lhs, rhs)
    if all(D.is_standard for D in spaces) and (w is None or np.all(np.asarray(w) == 1.0)):
        assert isinstance(E, DataSpace) and E.is_standard


@st.composite
def column_problem(draw):
    n = draw(st.integers(1, 4))
    blocks = []
    for _ in range(draw(st.integers(1, 3))):
        m = draw(st.integers(1, 3))
        blocks.append((draw(hnp.arrays(float, (m, n), elements=finite)), draw(spd(m, max_cond=1e3))))
    N = sum(B.shape[0] for B, _ in blocks)
    return (draw(spd(n, max_cond=1e3)), blocks, draw(hnp.arrays(float, n, elements=finite)),
            draw(hnp.arrays(float, N, elements=finite)))


_COLUMN_EXAMPLE = (np.diag([1.0, 4.0]), [(np.array([[1.0, 2.0]]), np.array([[3.0]])),
                                          (np.array([[0.0, 1.0], [1.0, 1.0]]), np.diag([0.5, 2.0]))],
                   np.array([1.0, -1.0]), np.array([2.0, 1.0, -3.0]))


@LAW
@example(_COLUMN_EXAMPLE)
@given(column_problem())
def law_column_operator_is_the_stacked_matrix(prob):
    """The column operator x -> (G_k x)_k into the direct sum of the G_k's codomains (a Euclidean with block labels)
    is the stacked matrix [A_1; ...; A_K]: its matrix, its adjoint (W_X^-1 M^T W_D with W_D the block metric) and its norm
    ||G x||^2 = sum_k ||G_k x||_k^2 agree with the stacked matrix's."""
    from pygeoinf0.operators.blocks import ColumnOperator
    from scipy.linalg import block_diag
    Wx, blocks, m, d = prob
    X = pg.Euclidean(len(m), metric=Wx)
    ops = [pg.Matrix(A, X, pg.Euclidean(A.shape[0], metric=W), name=f"A{k}") for k, (A, W) in enumerate(blocks)]
    G = ColumnOperator(ops)
    M, Wd = np.vstack([A for A, _ in blocks]), block_diag(*[W for _, W in blocks])
    cond = np.linalg.cond(Wx) * np.linalg.cond(Wd)
    assert np.allclose(G.matrix(), M, rtol=0, atol=_tol(np.abs(M).max() + 1, 1.0, M.size))
    Bs = np.linalg.solve(Wx, M.T @ Wd)
    assert np.allclose(G.adjoint.matrix(), Bs, rtol=0, atol=_tol(np.abs(Bs).max() + 1, cond, M.size)), "adjoint"
    x, y = X(m), G.codomain(d)
    scale = np.abs(M) @ np.abs(m) @ np.abs(Wd) @ np.abs(d) + np.abs(m) @ np.abs(Wx) @ np.abs(Bs) @ np.abs(d)
    _adjoint_gap(G, x, y, scale, cond, M.size)
    sq = G.codomain.norm(G(x)) ** 2
    ref = sum(op.codomain.norm(op(x)) ** 2 for op in ops)
    assert abs(sq - ref) <= _tol(np.abs(m) @ np.abs(M.T) @ np.abs(Wd) @ np.abs(M) @ np.abs(m) + 1, 1.0, M.size), (sq, ref)
    assert abs(sq - (M @ m) @ Wd @ (M @ m)) <= _tol(np.abs(m) @ np.abs(M.T) @ np.abs(Wd) @ np.abs(M) @ np.abs(m) + 1,
                                                    1.0, M.size)


@st.composite
def whitening_problem(draw):
    n = draw(st.integers(1, 4))
    return n, draw(spd(n)), draw(hnp.arrays(float, n, elements=finite))


@LAW
@example((2, np.array([[4.0, 1.0], [1.0, 1.0]]), np.array([1.0, -2.0])))
@given(whitening_problem())
def law_whitening_is_an_isometry_onto_the_whitened_coordinates(prob):
    """D.whitened(C) (Q1: the noise covariance does not change D; the whitened coordinates are a named construction) is
    an isometry of D's vectors with the C^-1 inner product onto the plain coordinates R^n: ||W d||^2 = d^T C^-1 d, the
    whitened noise is N(0, I) (M C M^T = I for its matrix M), and its adjoint is its inverse (it is onto)."""
    from pygeoinf0.spaces import DataSpace
    n, C, d = prob
    D = pg.DataSpace(n)
    W = D.whitened(C)
    assert isinstance(W.codomain, DataSpace) and W.codomain.is_standard and W.codomain.dim == n
    cond = np.linalg.cond(C)
    z = W(D(d))
    q = d @ np.linalg.solve(C, d)
    assert abs(W.codomain.norm(z) ** 2 - q) <= _tol(abs(q) + np.abs(d) @ np.abs(np.linalg.inv(C)) @ np.abs(d), cond, n), \
        "not an isometry"
    Mw = W.matrix()
    assert np.allclose(Mw @ C @ Mw.T, np.eye(n), rtol=0, atol=_tol(1.0, cond, n * n)), "whitened noise is not N(0, I)"
    back = W.inv(z)
    assert np.allclose(back.values, d, rtol=0, atol=_tol(np.abs(d).max() + 1, cond, n)), "not inverted"
    assert W.adjoint is W.inv
    assert W.claims.status("isometry") == (Truth.TRUE, "theorem")


def _euclidean_sum_forgets_the_weights(mp):
    from pygeoinf0.spaces import direct_sum
    sum_metric = direct_sum._sum_metric
    mp.setattr(direct_sum, "_sum_metric", lambda spaces, w: sum_metric(spaces, np.ones(len(spaces))))


def _column_adjoint_forgets_the_block_metrics(mp):
    from pygeoinf0.operators import blocks

    def bad(self):                                    # M^T read as the adjoint in the codomain's coordinates
        return linear.Matrix(self.domain.solve_metric(self.matrix().T), domain=self.codomain, codomain=self.domain)
    mp.setattr(blocks.ColumnOperator, "_make_adjoint", bad)


def _whitening_by_the_inverse_covariance(mp):
    from pygeoinf0.operators import whitening
    mp.setattr(whitening.Whitening, "_forward", lambda self, V: self.domain.apply_metric(V))   # C^-1 d, not L^-1 d


def _span_coefficients_ignore_the_gram(mp):
    from pygeoinf0.convex import subspaces
    mp.setattr(subspaces.Span, "coefficients", lambda self, x: self.loads(x))


def _check_promotes_the_class_claim(mp):
    from pygeoinf0.core.claims import CHECKED, Claim
    from pygeoinf0.operators import galerkin
    rule = galerkin.Discretisation.check

    def bad(self, fs):
        out = rule(self, fs)
        self._approximation = Claim("||f - P f|| <= max e for every f in M", CHECKED, "promoted from a sample",
                                    key="approximation")
        return out
    mp.setattr(galerkin.Discretisation, "check", bad)             # _planted builds the cases afresh under it


def _residual_in_coefficient_coordinates(mp):
    from pygeoinf0.operators import galerkin

    def bad(self, f):                                               # the Galerkin residual b - G c: zero to rounding
        V = self.subspace
        return float(np.linalg.norm(V.loads(f) - V.gram @ V.coefficients(f)))
    mp.setattr(galerkin.Discretisation, "residual", bad)


# ------------------------------------------------------------------ space identity without realisation (plan 069 P1)
_BOX = None
_REALISED = {}


def _realised_space(kind, r):
    """L2 or L^3 of the unit square realised by r (cached: the law draws pairs of them)."""
    global _BOX
    if _BOX is None:
        _BOX = pg.Box((0, 1), (0, 1))
    key = (kind, r)
    if key not in _REALISED:
        from pygeoinf0.realisations import Cells
        real = Cells.tensor(*[np.linspace(0, 1, int(r[5:]) + 1)] * 2) if r.startswith("cells") else r
        _REALISED[key] = pg.L2(_BOX, realisation=real) if kind == "L2" else pg.Lp(_BOX, 3, realisation=real)
    return _REALISED[key]


_REALISATIONS = ["local", "tensor", "legendre", "cells2", "cells3"]


@LAW
@given(st.sampled_from(["L2", "L3"]), st.sampled_from(_REALISATIONS), st.sampled_from(_REALISATIONS))
def law_space_identity_is_independent_of_realisation(kind, a, b):
    """A function space is the same whatever realises its elements (decision Q4): X == Y and hash(X) == hash(Y) for two
    realisations of one measure and norm, X.realised(r) == X, and an element made in either is a member of both."""
    if kind == "L3" and (a.startswith("cells") or b.startswith("cells")):
        reject()                                       # the cell realisation is for L2 only (registry.choose)
    X, Y = _realised_space(kind, a), _realised_space(kind, b)
    assert X == Y and Y == X, f"{X} realised {X.realisation} != {Y} realised {Y.realisation}"
    assert hash(X) == hash(Y)
    assert X.dual() == Y.dual()
    g = Y.constant(1.0)
    assert X.member(g) is Truth.TRUE and Y.member(X.constant(2.0)) is Truth.TRUE, "an element is not in its space"


@LAW
@given(st.sampled_from(_REALISATIONS), st.sampled_from(_REALISATIONS))
def law_mixing_realisations_refuses_naming_both(a, b):
    """Elements of one space realised differently are never combined silently (decision Q4): arithmetic, the inner
    product and operator application either agree with one realisation (a == b) or raise RealisationMismatch, which is
    not a SpaceMismatch, names both realisations and lists the choices."""
    from pygeoinf0.core.errors import RealisationMismatch, SpaceMismatch
    X, Y = _realised_space("L2", a), _realised_space("L2", b)
    f, g = X.constant(2.0), Y.constant(3.0)
    I = pg.Identity(X)
    for what, fn in (("f + g", lambda: f + g), ("<f, g>", lambda: X.inner(f, g)), ("I(g)", lambda: I(g))):
        if X.realisation == Y.realisation:
            v = fn()
            if what == "<f, g>":
                assert abs(v - 6.0) <= 1e-12, (what, v)
            continue
        try:
            fn()
        except RealisationMismatch as e:
            assert not isinstance(e, SpaceMismatch)
            assert X.realisation in str(e) and Y.realisation in str(e), (what, str(e))
            assert e.needs, (what, "no choices named")
        except Exception as e:                            # anything else is not the refusal of the law
            raise AssertionError(f"{what}: {type(e).__name__}: {e}") from e
        else:
            raise AssertionError(f"{what}: realisations {X.realisation} and {Y.realisation} mixed silently")


@LAW
@given(st.sampled_from(_REALISATIONS), st.sampled_from(_REALISATIONS), st.sampled_from([0.25, 0.5, 1.0]))
def law_symbolic_representer_pairs_in_the_elements_realisation(a, b, h):
    """A symbolic representer X.riesz(mu) is a measure, not a realisation (plan 071 P3, review item 10): <X.riesz(mu), g>
    = int g dmu is read from g's own realisation, so it equals the pairing in g's space (Y.inner(Y.riesz(mu), g)) bit for
    bit whatever realises X, in both orders; never an AttributeError of X's backend reading g."""
    X, Y = _realised_space("L2", a), _realised_space("L2", b)
    S = pg.Box((0, h), (0, 0.5))
    g = Y.constant(3.0)
    want = Y.inner(Y.riesz(S), g)
    try:
        got = (X.inner(X.riesz(S), g), X.inner(g, X.riesz(S)))
    except Exception as e:                                # a crash is not the pairing of the law (asserted outside the
        got = f"{type(e).__name__}: {e}"                  # handler: one failure, whatever the crash)
    assert got == (want, want), (a, b, got, want)
    if not b.startswith("cells"):
        assert abs(want - 1.5 * h) <= 1e-12, (b, want)    # int_S 3 = 3 h / 2

def _identity_includes_the_realisation(mp):
    from pygeoinf0.spaces import base
    same = base._FunctionSpace._same_measure
    mp.setattr(base._FunctionSpace, "_same_measure",
               lambda self, other: same(self, other) and getattr(other, "realisation", None) == self.realisation)


def _realisations_read_as_the_same(mp):
    from pygeoinf0.spaces import identity
    mp.setattr(identity, "same_realisation", lambda X, Y: True)


def _symbolic_pairing_read_by_the_calling_space(mp):
    from pygeoinf0.spaces import representers
    mp.setattr(representers, "pairing_space", lambda space, e: space)

def _quadrature_rule_drops_a_node(mp):
    """Every rule of the module loses its last node (and its weight); the composite rules lose one per panel too."""
    Q = _quad()
    for name in QUAD_RULES:
        rule = getattr(Q, name)

        def bad(*args, _rule=rule, **kw):
            out = _rule(*args, **kw)
            return (out[0][:-1], out[1][:-1]) + tuple(out[2:])
        bad.exact_degree, bad.exactness = rule.exact_degree, rule.exactness
        mp.setattr(Q, name, bad)


def _quadrature_degree_understated(mp):
    """Every rule states one degree less than it reaches: the sharpness law then meets a polynomial it integrates."""
    Q = _quad()
    for name in QUAD_RULES:
        rule = getattr(Q, name)
        mp.setattr(rule, "exact_degree", lambda n, _d=rule.exact_degree: _d(n) - 1)


# ------------------------------------------------------------------ named priors (plan 069 P6, A13, E3, decision Q2)
def _prior_spaces():
    cells = pg.realisations.Cells.tensor(np.linspace(0, 1, 5), np.linspace(0, 2, 3))
    planet = pg.LayeredBall([0.3, 0.6, 1.0], names=["core", "mantle", "crust"], interfaces=["cmb", "moho"])
    return {
        "L2": lambda: pg.L2(I01), "L2 box": lambda: pg.L2(pg.Box([(0, 1), (0, 1)])), "Lp3": lambda: pg.Lp(I01, 3),
        "H1": lambda: pg.Sobolev(I01, 1, length_scale=0.2), "C[0,1]": lambda: pg.Continuous("[0, 1]"),
        "R3": lambda: pg.Euclidean(3, weights=[1.0, 2.0, 3.0]), "C2": lambda: pg.Euclidean(2, field="C"),
        "data": lambda: pg.DataSpace(2, labels=["a", "b"]),
        "direct sum": lambda: pg.DirectSum([pg.L2(I01), pg.Euclidean(2)]),
        "euclidean sum": lambda: pg.DirectSum([pg.Euclidean(2), pg.Euclidean(1)]),
        "complexified": lambda: pg.Complexified(pg.Sobolev(I01, 1, length_scale=0.2)),
        "cells complexified": lambda: pg.Complexified(pg.L2(pg.Box([(0, 1), (0, 2)]), realisation=cells)),
        "piecewise by pieces": lambda: pg.PiecewiseSpace.from_pieces(planet, {n: pg.L2(planet[n]) for n in planet.names}),
        "piecewise fields": lambda: pg.PiecewiseSpace(pg.Partition(pg.Ball(1.0), [pg.Ball(0.5), pg.Shell(0.5, 1.0)],
                                                      names=["in", "out"]), {"vp": pg.L2}),
        "dual": lambda: pg.L2(I01).dual(),
    }


PRIOR_SPACES = _prior_spaces()
_PRIOR_NAMES = ("legendre_decay", "random_features", "cylindrical", "standard_gaussian")
_PRIOR_KW = ({}, {"alpha": 2.0}, {"decay": 0.5}, {"features": 8})
_PRIOR_BUILT = {}


def _prior_space(name):
    if name not in _PRIOR_BUILT:
        _PRIOR_BUILT[name] = PRIOR_SPACES[name]()
    return _PRIOR_BUILT[name]


@LAW
@example("L2", 0)
@example("H1", 1)
@example("R3", 0)
@given(st.sampled_from(sorted(PRIOR_SPACES)), st.integers(0, len(_PRIOR_KW) - 1))
def law_no_space_builds_a_prior_unless_asked(name, k):
    """No space picks a prior (decision Q2): X.gaussian(...) refuses with PriorUnstated on every space class, naming
    the constructors; a draw X.random() exists only where one law is meant (the standard Gaussian of a
    finite-dimensional inner product, or directions of complex coordinates), else it refuses too."""
    from pygeoinf0.core.errors import PriorUnstated, Unsupported
    X = _prior_space(name)
    try:
        out = X.gaussian(**_PRIOR_KW[k])
    except PriorUnstated as e:
        assert all(n in str(e) for n in _PRIOR_NAMES) and e.needs, (name, str(e))
    except Exception as e:
        raise AssertionError(f"{name}: X.gaussian raised {type(e).__name__}: {e}, not PriorUnstated") from e
    else:
        raise AssertionError(f"{name}: X.gaussian({_PRIOR_KW[k]}) built {out!r} without being named")
    try:
        X.standard_gaussian()
        one_law = True                          # a finite-dimensional inner product: its standard Gaussian
    except PriorUnstated:
        one_law = False
    if not one_law and hasattr(X, "random") and "complex" not in name:
        try:
            out = X.random(seed=0)
        except (PriorUnstated, Unsupported):
            pass
        else:
            raise AssertionError(f"{name}: X.random() drew {out!r} from a prior nobody named")


@st.composite
def named_prior_case(draw):
    kind = draw(st.sampled_from(["legendre", "features", "standard"]))
    if kind == "legendre":
        return kind, draw(st.floats(0.2, 0.95)), draw(st.one_of(st.none(), st.integers(1, 40)))
    if kind == "features":
        return kind, draw(st.sampled_from([0.2, 0.5])), draw(st.sampled_from([3, 8])), draw(st.integers(0, 2))
    return kind, draw(hnp.arrays(float, 3, elements=st.floats(0.1, 10.0))), None


_NAMED = {}


def _named_prior(case):
    from pygeoinf0.probability.priors import legendre_decay, random_features
    kind, a, b = case[0], case[1], case[2]
    if kind == "legendre":
        X = _prior_space("L2")
        probes = [X(lambda t, j=j: np.cos(3 * j * t) + t ** j) for j in range(4)]
        return legendre_decay(X, a, modes=b), probes
    if kind == "features":
        key = case
        if key not in _NAMED:
            X = _prior_space("L2 box")
            _NAMED[key] = random_features(X, length_scale=a, features=b, seed=case[3])
        X = _NAMED[key].space
        return _NAMED[key], [X(lambda x, y, j=j: np.cos(2 * j * x) * (1 + y) ** j) for j in range(4)]
    X = pg.Euclidean(3, weights=a)
    return X.standard_gaussian(), [X(np.eye(3)[j] + 0.5) for j in range(3)]


@LAW
@example(("legendre", 0.8, None))
@example(("features", 0.2, 8, 1))
@given(named_prior_case())
def law_named_prior_claims_hold(case):
    """A named prior's covariance is a pg.Covariance whose claims (theorem, derived or checked) are true of it on the
    function space: self-adjoint and positive semidefinite as a Gram form <C u_i, u_j> over elements u_i (by the
    precision rule), and trace class with an Enclosure of its trace containing sum_j lam_j ||psi_j||^2 + the stated
    omitted part, computed from the norms of its generators in the space (not from coefficients)."""
    from pygeoinf0.core.logic import decide
    from pygeoinf0.core.claims import ASSERTED as AS, UNKNOWN as UN
    mu, probes = _named_prior(case)
    C = mu.covariance
    X = mu.space
    if case[0] == "standard":
        assert isinstance(C, pg.Identity)
        return
    assert isinstance(C, pg.Covariance), type(C)
    K = np.array([[X.inner(C(u), v) for v in probes] for u in probes])
    size = max(float(np.abs(K).max()), 1e-300)
    lam = np.linalg.eigvalsh(0.5 * (K + K.T))
    for key, truth in (("self-adjoint", decide(float(np.abs(K - K.T).max()), size)),
                       ("positive semidefinite", decide(max(-float(lam.min()), 0.0), size))):
        holds, status = C.holds(key)
        assert holds is Truth.TRUE and status not in (AS, UN), (key, C.claims)
        assert truth is not Truth.FALSE, f"{key} claimed ({status}) and false: {K}"
    holds, status = C.holds("trace class")
    assert holds is Truth.TRUE and status not in (AS, UN), C.claims
    gens, w, _ = C.finite
    known = float(np.asarray(w) @ np.asarray(X.norm(gens)) ** 2)
    T = C.trace()
    lo, hi = known * (1 - 1e-10), known * (1 + 1e-10) + (C.rest["trace"] if C.rest else 0.0)
    assert T.lower <= hi and T.upper >= lo and T.lower <= known * (1 + 1e-10), f"tr {known!r} (+ rest) vs {T!r}"


@LAW
@example(0.8, None)
@example(0.95, 3)
@given(st.floats(0.05, 0.99), st.one_of(st.none(), st.integers(1, 60)))
def law_legendre_decay_states_its_truncation(decay, modes):
    """legendre_decay(X, decay) is the prior C phi_k = decay^(2k) phi_k for every k (trace 1 / (1 - decay^2)), applied
    and sampled as its first K modes: the omitted part's stated trace and norm bound the true ones, sum_{k >= K}
    decay^(2k) = decay^(2K) / (1 - decay^2) and decay^(2K), and the trace Enclosure contains the full trace."""
    from pygeoinf0.probability.priors import legendre_decay
    C = legendre_decay(_prior_space("L2"), decay, modes=modes).covariance
    K = len(C.finite[1])
    if modes is not None:
        assert K == modes
    q = decay * decay
    tail, top = q ** K / (1 - q), q ** K
    assert C.rest is not None or tail == 0.0
    if C.rest is not None:
        assert C.rest["trace"] >= tail * (1 - 1e-12), (C.rest, tail)
        assert C.rest["norm"] >= top * (1 - 1e-12), (C.rest, top)
    T = C.trace()
    assert T.contains(1.0 / (1 - q)) or abs(T.upper - 1.0 / (1 - q)) <= 1e-12 / (1 - q), (T, 1.0 / (1 - q))


def _l2_default_prior_restored(mp):
    from pygeoinf0.probability.priors import legendre_decay, random_features

    def default(self, **kw):
        return legendre_decay(self, 0.8) if self.domain.cartesian_dim == 1 else random_features(self, None, 8)
    mp.setattr(spaces_base._LebesgueBase, "gaussian", default, raising=False)
    mp.setattr(spaces_base._LebesgueBase, "random", lambda self, size=None, seed=None, **kw:
               default(self).sample(size, seed))


def _legendre_generators_unnormalised(mp):
    from pygeoinf0.realisations import backends
    hook = backends.IntervalBackend.legendre_decay_generators

    def unnormalised(self, decay, modes):
        R, lam = hook(self, decay, modes)
        return self.lincomb([(2.0, R)]), lam
    mp.setattr(backends.IntervalBackend, "legendre_decay_generators", unnormalised)


def _legendre_tail_forgets_the_geometric_sum(mp):
    from pygeoinf0.probability import priors
    mp.setattr(priors, "_legendre_tail", lambda decay, K: (decay ** (2 * K), decay ** (2 * K)))



def _region_rule_drops_the_jacobian(mp):
    """The rules of box charts integrate dxi instead of the region's measure: the Jacobians and density are dropped."""
    from pygeoinf0 import quadrature
    chart_rule = quadrature.chart_rule
    mp.setattr(quadrature, "chart_rule", lambda chart, n=16: chart_rule(chart._replace(jacobians=None, density=None), n))


# ------------------------------------------------------------------ piecewise spaces (plan 069 P7, review B9)
_PW_CUTS = [(0.5,), (0.3, 0.7), (0.25, 0.4, 0.8)]
_PW = {}


def _pw(kind, cuts, s=1.0):
    """A partition of [0, 1] at cuts and L2 or H^s_l (l = 0.2) of the whole interval (cached)."""
    key = (kind, cuts, s)
    if key not in _PW:
        ends = (0.0,) + tuple(cuts) + (1.0,)
        P = pg.Partition("[0, 1]", [pg.Interval(a, b) for a, b in zip(ends[:-1], ends[1:])],
                         names=[f"p{k}" for k in range(len(ends) - 1)])
        _PW[key] = P, pg.L2("[0, 1]") if kind == "L2" else pg.Sobolev("[0, 1]", s, length_scale=0.2)
    return _PW[key]


_pw_coeffs = hnp.arrays(float, 4, elements=st.floats(-2, 2, allow_nan=False, allow_infinity=False))


@LAW
@example((0.3, 0.7), np.array([1.0, 1.0, -2.0, 0.5]))
@given(st.sampled_from(_PW_CUTS), _pw_coeffs)
def law_restrictions_are_an_isometry_onto_the_broken_l2_space(cuts, c):
    """X.restrictions(P) of X = L2(0, 1) maps into the broken PiecewiseSpace, which is the glued one (the seams are
    null sets); it preserves the norm, sum_k ||u|Omega_k||^2 = ||u||^2 = int p^2 (exact for a cubic p), and its glue
    is its inverse, to the rounding of the coefficients."""
    P, X = _pw("L2", cuts)
    R = X.restrictions(P)
    M = R.codomain
    assert isinstance(M, pg.PiecewiseSpace), type(M)
    assert M == pg.PiecewiseSpace(P, X, broken=True) == pg.PiecewiseSpace(P, X), "L2: glued and broken differ"
    p = np.polynomial.Polynomial(c)
    sq = (p * p).integ()
    want, scale = sq(1.0) - sq(0.0), float(np.sum(np.abs(c))) ** 2 + EPS
    u = X.element(lambda t: p(t), resolution=1.0)
    got = M.norm(R(u)) ** 2
    assert abs(got - want) <= 1e-12 * scale, (got, want)
    t = np.linspace(0, 1, 9)
    back = R.glue(R(u))(t)
    assert np.max(np.abs(back - p(t))) <= 1e-12 * math.sqrt(scale), (back, p(t))


@LAW
@given(st.sampled_from(["L2", "H"]), st.sampled_from(_PW_CUTS), st.sampled_from([1.0, 2.0]),
       hnp.arrays(float, (4, 2), elements=st.floats(-2, 2, allow_nan=False, allow_infinity=False)))
def law_broken_norm_is_the_sum_of_the_pieces_norms(kind, cuts, s, ab):
    """The norm of a broken PiecewiseSpace is the sum of its pieces' norms: ||x||^2 = sum_k ||x_k||^2_(X(Omega_k)),
    for L2 and for H^s pieces (the pieces need not match at the seams; integer orders, where the restriction space's
    norm of a formula is computed on intervals)."""
    P, X = _pw(kind, cuts, s)
    M = pg.PiecewiseSpace(P, X, broken=True)
    y = M.element({n: (lambda t, a=a, b=b: a + b * t) for n, (a, b) in zip(M.names, ab)}, resolution=1.0)
    parts = [M.summand(n).norm(M.part(y, n)) ** 2 for n in M.names]
    got = M.norm(y) ** 2
    assert abs(got - sum(parts)) <= 1e-12 * (sum(parts) + EPS), (got, parts)
    assert abs(M.inner(y, y) - sum(parts)) <= 1e-12 * (sum(parts) + EPS)


@LAW
@example((0.5,), 1.0, np.array([1.0, 1.0, -2.0, 0.5]), 1.0)
@given(st.sampled_from(_PW_CUTS), st.sampled_from([0.75, 1.0, 1.25]), _pw_coeffs,
       st.floats(0.1, 3.0) | st.floats(-3.0, -0.1))
def law_glued_traces_match_and_broken_need_not(cuts, s, c, d):
    """Elements of the glued H^s (1/2 < s < 3/2) have matching traces on every seam, by the precision rule; the broken
    space holds elements with a jump d (they are members), its seam_jumps report d, and gluing such an element is
    refused naming the trace, while gluing the pieces of one function gives that function in H^s."""
    P, H = _pw("H", cuts, s)
    Mg, Mb = pg.PiecewiseSpace(P, H), pg.PiecewiseSpace(P, H, broken=True)
    p = np.polynomial.Polynomial(c)
    f = lambda t: p(t)                                          # noqa: E731
    J = Mg.seam_jumps(Mg.split({"value": f}, resolution=1.0))
    assert J.match is Truth.TRUE, dict(J)
    y = Mb({n: (d if k == 0 else 0.0) for k, n in enumerate(Mb.names)})
    assert Mb.member(y) is Truth.TRUE
    J = Mb.seam_jumps(y)
    first = P.seams[0].name
    assert abs(J[first] - abs(d)) <= 1e-12 * abs(d) and J.match is Truth.FALSE, dict(J)
    try:
        Mb.glue(y)
    except pg.Refused as e:
        assert "trace" in str(e), str(e)
    else:
        raise AssertionError(f"a broken element with a jump {d:g} on {first} was glued into {H}")
    g = Mb.glue(Mb.split({"value": f}, resolution=1.0))
    assert g.space == H
    t = np.linspace(0, 1, 9)
    assert np.max(np.abs(g(t) - p(t))) <= 1e-9 * (float(np.sum(np.abs(c))) + EPS), (g(t), p(t))


def _restriction_halves_a_piece(mp):
    from pygeoinf0.operators import pieces
    apply = pieces.Restrict._apply
    mp.setattr(pieces.Restrict, "_apply", lambda self, u: apply(self, u) * (0.5 if self.label == "p1" else 1.0))


def _broken_norm_counts_a_piece_twice(mp):
    from pygeoinf0.spaces import piecewise
    inner = pg.DirectSum.inner

    def twice(self, f, g):
        n = self.names[0]
        return inner(self, f, g) + self.summand(n).inner(self.part(f, n), self.part(g, n))
    mp.setattr(piecewise.PiecewiseSpace, "inner", twice, raising=False)
    mp.setattr(piecewise.PiecewiseSpace, "norm", lambda self, f: math.sqrt(twice(self, f, f)), raising=False)


def _glue_skips_the_trace_check(mp):
    from pygeoinf0.spaces import gluing
    mp.setattr(gluing, "require_matching_traces", lambda *a, **k: None)


# ---------------------------------------------- plan 069 P4: piecewise constant on a Partition (review A11, B9)
_strip_cuts = st.lists(st.floats(0.05, 0.95), min_size=0, max_size=3, unique=True).map(sorted)


def _strip_partition(xs, ys):
    """A Partition of [0, 1] x [0, 2] into strips at the x cuts xs, strip k cut at its own y cuts ys[k] (not a tensor
    mesh unless the cuts agree)."""
    ex = [0.0] + [x for x in xs] + [1.0]
    pieces = {}
    for k, (a, b) in enumerate(zip(ex[:-1], ex[1:])):
        ey = [0.0] + [2 * y for y in ys[k % len(ys)]] + [2.0]
        for j, (c, d) in enumerate(zip(ey[:-1], ey[1:])):
            pieces[f"s{k}c{j}"] = pg.Box((a, b), (c, d))
    return pg.Partition(pg.Box((0, 1), (0, 2)), pieces)


@LAW
@example([0.4], [[0.25], [0.6, 0.85]], np.array([1.0, 2.0, -1.0, 3.0]), np.array([0.5, -1.0, 2.0, 0.25]))
@given(_strip_cuts.filter(lambda xs: all(b - a > 0.02 for a, b in zip([0.0] + xs, xs + [1.0]))),
       st.lists(_strip_cuts.filter(lambda ys: all(b - a > 0.02 for a, b in zip([0.0] + ys, ys + [1.0]))),
                min_size=1, max_size=3),
       hnp.arrays(float, 4, elements=st.floats(-3, 3, allow_nan=False, allow_infinity=False)),
       hnp.arrays(float, 4, elements=st.floats(-3, 3, allow_nan=False, allow_infinity=False)))
def law_piecewise_constant_on_a_partition_is_the_l2_projection(xs, ys, c, e):
    """L2(D) realised piecewise constant on a Partition P of D (realisations/piecewise_constant.py, Cells(P)) is the
    orthogonal projection Q onto the functions constant on P's pieces: the cells are P's pieces in P's order with P's
    tiling claim, their volumes add up to |D|, (Q u)_k is the average of u on piece k (exact for bilinear u: its value
    at the centre), and <u, v> realised is <Q u, Q v> = sum_k |Omega_k| (Q u)_k (Q v)_k."""
    P = _strip_partition(xs, ys)
    C = pg.realisations.Cells(P)
    assert C.n == len(P) and tuple(C.names) == tuple(P.order), "the cells are not the partition's pieces"
    assert C.claims.find("tiling").status == P.claims.find("tiling").status
    vol = np.array([P[n].volume for n in P.order])
    assert np.allclose(C.volumes, vol, rtol=1e-14, atol=0), (C.volumes, vol)
    assert abs(C.volumes.sum() - P.domain.volume) <= 1e-13 * P.domain.volume, C.volumes.sum()
    M = pg.L2(P.domain, realisation=C)
    f = lambda p: (lambda x, y: p[0] + p[1] * x + p[2] * y + p[3] * x * y)
    u, v = M(f(c)), M(f(e))
    centre = np.array([[(s.a + s.b) / 2 for s in P[n].sides] for n in P.order])
    qu, qv = f(c)(*centre.T), f(e)(*centre.T)
    scale = 1 + float(np.abs(c).sum())
    assert np.max(np.abs(u.realisation.V[:, 0] - qu)) <= 1e-13 * scale, (u.realisation.V[:, 0], qu)
    want = float(np.sum(vol * qu * qv))
    assert abs(M.inner(u, v) - want) <= 1e-13 * scale * (1 + float(np.abs(e).sum())), (M.inner(u, v), want)


def _cells_drop_a_partition_piece(mp):
    from pygeoinf0.realisations import piecewise_constant as pc
    real = pc.partition_boxes
    mp.setattr(pc, "partition_boxes", lambda P: (lambda names, boxes: (names[:-1], boxes[:-1]))(*real(P)))


# ---------------------------------------------- plan 070 P2: the data-dependent choice rules in meta/ (review D3, B14)
FEW = settings(max_examples=max(1, int(os.environ.get("PYGEOINF0_LAW_EXAMPLES", 200)) // 5), deadline=None,
               derandomize=True, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])


@st.composite
def choice_problem(draw):
    """M = R^m, D = R^n (n < m), G with full row rank, a Gaussian prior N(0, I), standard Gaussian noise, a property
    map T into R^1, and two data: d (well above the noise, so the discrepancy target n is reachable) and another."""
    n = draw(st.integers(2, 4))
    m = n + draw(st.integers(1, 3))
    A = draw(hnp.arrays(float, (n, m), elements=st.floats(-1, 1, allow_nan=False)))
    A = A + 2.0 * np.eye(n, m)
    B = draw(hnp.arrays(float, (1, m), elements=st.floats(-1, 1, allow_nan=False))) + 0.5
    x = draw(hnp.arrays(float, m, elements=st.floats(-2, 2, allow_nan=False)))
    e = draw(hnp.arrays(float, (2, n), elements=st.floats(-1, 1, allow_nan=False)))
    M, D = pg.Euclidean(m), pg.Euclidean(n)
    G, T = pg.Matrix(A, M, D, name="G"), pg.Matrix(B, M, pg.Euclidean(1), name="T")
    d1, d2 = A @ (3.0 * x + 1.0) + e[0], A @ (-2.0 * x + 1.0) + 0.5 * e[1]
    if min(d1 @ d1, d2 @ d2) <= 2.0 * n:
        reject()
    return M, D, G, T, D(d1), D(d2)


def _choice_cases(M, D, G, T):
    """The arguments of every rule of pg.meta.RULES on one problem (a rule without a case fails the law)."""
    from pygeoinf0 import meta
    return {"discrepancy": dict(method="least_squares"),
            "maximum_evidence": dict(method="pli", family=meta.scales()),
            "fit_noise": dict(method="pli", family=meta.NoiseFamily(D)),
            "knee": dict(method="sola", family=meta.damping("sola", lo=1e-3, hi=1e3), rule=meta.Knee(samples=101))}


@FEW
@given(choice_problem())
def law_every_choice_rule_records_whether_it_used_the_datum(prob):
    """Every data-dependent choice rule of pg.meta (pg.meta.RULES) takes (problem, method, hyperparameter family) and
    its result is a meta record: the claim "meta", the claim "used the datum" (holds when the datum entered the
    choice), a choice that used the datum also the claim "data-dependent rule", and its family as a named record (key
    "hyperparameter family"; review B14). A choice that changes with the datum must say that it used it."""
    from pygeoinf0 import meta
    M, D, G, T, d1, d2 = prob
    mdr = pg.Additive(G, D.standard_gaussian())
    prior = pg.GaussianMeasure(M, pg.Identity(M))
    cases = _choice_cases(M, D, G, T)
    assert set(cases) == set(meta.RULES), f"rules without a case: {sorted(set(meta.RULES) ^ set(cases))}"
    for name, rule in meta.RULES.items():
        out = []
        for d in (d1, d2):
            pb = pg.Problem(M=M, D=D, mdr=mdr, prior=prior, mpr=T, datum=d)
            try:
                out.append(rule(pb, **cases[name]))
            except pg.Refused:
                reject()
        for r in out:
            keys = {c.key: c for c in r.claims if c.key}
            assert "meta" in keys and keys["meta"].holds, (name, r.claims)
            assert "used the datum" in keys, f"{name}: no 'used the datum' claim in {r.claims}"
            assert "hyperparameter family" in keys, f"{name}: the family is not a named record in {r.claims}"
            assert set(r.values) == set(r.family.parameters), (name, r.values, r.family.parameters)
            if keys["used the datum"].holds:
                assert "data-dependent rule" in keys, (name, r.claims)
        changed = any(not np.array_equal(np.asarray(out[0].values[k], float), np.asarray(out[1].values[k], float))
                      for k in out[0].values)
        if changed:
            used = {c.key: c for c in out[0].claims if c.key}["used the datum"]
            assert used.holds, f"{name}: the choice changed with the datum but says {used.statement!r}"


def _choice_rule_drops_the_datum_claim(mp):
    from pygeoinf0 import meta
    rule = meta.RULES["discrepancy"]

    def dropped(*a, **k):
        r = rule(*a, **k)
        r.claims = type(r.claims)(c for c in r.claims if c.key != "used the datum")
        return r
    mp.setitem(meta.RULES, "discrepancy", dropped)


# ---------------------------------------------- plan 070 P5: Problem.run carries the calibration (review D9)
def _run_cases(M, D, G, T):
    """The elements and options of every method of pg.inference.METHODS on one problem (a method without a case fails
    the law): a linear point MDR, an additive Gaussian kernel, its acceptance region, a norm ball and a Gaussian
    prior."""
    K = pg.Additive(G, D.standard_gaussian())
    ball, mu = pg.NormBall(M, 100.0), pg.GaussianMeasure(M, pg.Identity(M))
    return {"minimum_norm": (dict(mdr=G), {}),
            "least_squares": (dict(mdr=G), dict(gamma=0.5)),
            "sola": (dict(mdr=K, mpr=T), dict(gamma=0.5)),
            "csola": (dict(prior=ball, mdr=K.acceptance(0.05), mpr=T), dict(gamma=0.5)),
            "ocsola": (dict(prior=ball, mdr=K, mpr=T), dict(gamma=0.5, alpha=0.05)),
            "bgp": (dict(prior=ball, mdr=K.acceptance(0.05), mpr=T), {}),
            "pli": (dict(prior=mu, mdr=K, mpr=T), {})}


_calibrations = st.sampled_from([
    lambda: None, lambda: pg.inference.Unstated(), lambda: pg.inference.Deterministic(),
    lambda: pg.inference.Frequentist(0.05, fixed_before_data=True), lambda: pg.inference.Bayesian(),
    lambda: pg.inference.Hardened(0.05, 0.1)])


@FEW
@given(choice_problem(), _calibrations)
def law_every_result_of_problem_run_carries_its_calibration(prob, calibration):
    """Every method of pg.inference.METHODS declares the calibrations it supports (Method.calibrations, a tuple of
    calibration classes; review D9). Problem.run of a method that supports the problem's calibration (None: unstated)
    returns a result whose claims contain every claim of problem.calibration_claims(); a calibration it does not
    support is refused (pg.Refused) naming every calibration it supports."""
    from pygeoinf0.inference import methods as imethods
    M, D, G, T, d, _ = prob
    cases = _run_cases(M, D, G, T)
    assert set(cases) == set(imethods.METHODS), f"methods without a case: {sorted(set(imethods.METHODS) ^ set(cases))}"
    cal = calibration()
    for name, (fields, options) in cases.items():
        pb = pg.Problem(M=M, D=D, datum=d, calibration=cal, **fields)
        supported = imethods.METHODS[name].calibrations
        assert isinstance(supported, tuple) and supported, f"{name} declares no calibrations"
        if cal is not None and not isinstance(cal, supported):
            with pytest.raises(pg.Refused) as refusal:
                pb.run(name, **options)
            for c in supported:
                assert c.__name__ in str(refusal.value), (name, c.__name__, str(refusal.value))
            continue
        out = pb.run(name, **options)
        carried = {(c.statement, c.status) for c in getattr(out, "claims", None) or ()}
        want = {(c.statement, c.status) for c in pb.calibration_claims()}
        assert want and want <= carried, f"{name}: the calibration claims {sorted(want - carried)} are not on its result"


def _run_drops_the_calibration_claims(mp):
    from pygeoinf0.inference import calibration as ical
    mp.setattr(ical, "carry", lambda result, claims: result)


# ------------------------------------------------------------------ the data-space engine (plan 070 P3, review D4, Q1)
@st.composite
def engine_problem(draw):
    n = draw(st.integers(1, 4))
    m = n + draw(st.integers(0, 2))
    A = draw(hnp.arrays(float, (n, m), elements=st.floats(-1, 1, allow_nan=False))) + 2.0 * np.eye(n, m)
    return n, m, A, draw(spd(n))


@LAW
@example((2, 3, np.array([[2.0, 0.5, 0.0], [0.0, 2.0, 1.0]]), np.array([[4.0, 1.0], [1.0, 1.0]])))
@given(engine_problem())
def law_the_engine_never_infers_a_metric(prob):
    """The data-space engine (solvers/dataspace/) takes its metric as an explicit argument: D.whitened(C), the noise
    covariance C, or D's own inner product. Given nothing, a measure or a set it refuses, naming those choices (Q1: D's
    metric is not the noise; the law lives in the relation, and the public routes read it). An explicit metric gives
    the whitened Gram's spectrum: the generalised eigenvalues of (G G^T, C), the same for D.whitened(C) and for C."""
    from pygeoinf0.solvers.dataspace import DataDecomposition
    n, m, A, C = prob
    M, D = pg.Euclidean(m), pg.DataSpace(n)
    G = pg.Matrix(A, M, D, name="G")
    implicit = (None, pg.GaussianMeasure(D, pg.Matrix(C, D, D)), pg.NormBall(D, 1.0), pg.Ellipsoid(D, 1.0, C))
    for noise in implicit:
        try:
            DataDecomposition(G, noise)
        except pg.Refused as e:
            assert {"D.whitened(C)", "the noise covariance C", "D"} <= set(e.needs), e.needs
        else:
            raise AssertionError(f"a metric inferred from {type(noise).__name__}")
    L = np.linalg.cholesky(C)
    Li = np.linalg.inv(L)
    exact = np.linalg.eigvalsh(Li @ (A @ A.T) @ Li.T)
    cond = np.linalg.cond(C)
    for metric in (D.whitened(C), C):
        lam = np.sort(DataDecomposition(G, metric).lam)
        assert np.allclose(lam, exact, rtol=0, atol=_tol(np.abs(exact).max() + 1.0, cond, n * m)), (lam, exact)


def _engine_reads_the_measure(mp):
    from pygeoinf0.solvers.dataspace import whitening
    stated = whitening.metric_of

    def inferring(noise, D, offset=None):                       # the defect: a measure's covariance read silently
        if noise is None:
            return stated(D, D, offset)
        return stated(getattr(noise, "covariance", D) if not hasattr(noise, "radius") else D, D, offset)
    mp.setattr(whitening, "metric_of", inferring)


# ------------------------------------------- the two data routes (plan 071 P4, Mac review item 4, Q1 as amended)
@st.composite
def route_problem(draw):
    """G: R^m -> R^n with full row rank, a noise covariance C (SPD, or diagonal), a noise mean or none, a property map
    into R^1, a stated gamma and a datum well above the noise (the discrepancy target n is reachable)."""
    n = draw(st.integers(2, 4))
    m = n + draw(st.integers(0, 2))
    A = draw(hnp.arrays(float, (n, m), elements=st.floats(-1, 1, allow_nan=False))) + 2.0 * np.eye(n, m)
    C = draw(spd(n, max_cond=100))
    if draw(st.booleans()):
        C = np.diag(np.diag(C))
    b = draw(st.one_of(st.none(), hnp.arrays(float, n, elements=st.floats(-1, 1, allow_nan=False))))
    B = draw(hnp.arrays(float, (1, m), elements=st.floats(-1, 1, allow_nan=False))) + 1.0
    x = draw(hnp.arrays(float, m, elements=st.floats(-2, 2, allow_nan=False)))
    gamma = 10.0 ** draw(st.floats(-2, 1))
    y = A @ (5.0 * x + 3.0)
    q = y @ np.linalg.solve(C, y)                       # chi^2 of the zero model: above the target n by a margin
    y = y * max(1.0, math.sqrt(16.0 * n / q))
    return n, m, A, C, b, B, gamma, y + (0.0 if b is None else b)


def _routes(n, m, A, C, b, B):
    """The same problem by both routes: A, a plain D with pg.Additive(G, N(b, C)); B, D = Euclidean(n, covariance=C)
    with its standard Gaussian (coordinates N(0, C)) shifted by b. (Not GaussianMeasure(D_B, C): on the weighted D that
    reads C in D's inner product, the coordinate covariance C W^-1 = C^2 for a diagonal C.)"""
    M = pg.Euclidean(m)
    T = pg.Matrix(B, M, pg.Euclidean(1), name="T")
    DA = pg.Euclidean(n)
    noise_a = pg.GaussianMeasure(DA, pg.Matrix(C, DA, DA, name="C"), None if b is None else DA(b))
    DB = pg.Euclidean(n, covariance=C)
    noise_b = DB.standard_gaussian() if b is None else DB.standard_gaussian() + DB(b)
    prior = pg.GaussianMeasure(M, pg.Identity(M))
    return [(D, pg.Additive(pg.Matrix(A, M, D, name="G"), noise), T, prior) for D, noise in ((DA, noise_a), (DB, noise_b))]


@FEW
@example((2, 2, np.array([[2.0, 0.5], [0.0, 2.0]]), np.array([[4.0, 1.0], [1.0, 1.0]]), np.array([0.5, -0.5]),
          np.array([[1.0, 2.0]]), 0.3, np.array([30.0, -20.0])))
@given(route_problem())
def law_both_data_routes_give_the_same_inference(prob):
    """Q1 as amended (2026-11-15): a plain data space with the noise stated in the relation, pg.Additive(G, N(b, C)),
    and the whitened norm, D = Euclidean(n, covariance=C) with unit noise in D's metric, are the same model. Every
    route where both apply gives the same result to rounding: the posterior (cell B10), the property posterior (pli),
    SOLA's estimate and its noise law, damped least squares (cell B13: its minimiser and the misfit in the C^-1 norm)
    and the discrepancy principle's gamma. Neither route refuses."""
    n, m, A, C, b, B, gamma, d = prob
    (DA, RA, T, prior), (DB, RB, _, _) = _routes(n, m, A, C, b, B)
    pa = pg.Problem(mdr=RA, mpr=T, prior=prior, datum=DA(d))
    pb = pg.Problem(mdr=RB, mpr=T, prior=prior, datum=DB(d))
    cond = np.linalg.cond(C) * np.linalg.cond(A @ A.T + gamma * C)
    big = 1.0 + np.abs(d).max() * (1.0 + np.abs(np.linalg.inv(C)).max())

    def same(x, y, what):
        x, y = np.asarray(x, float), np.asarray(y, float)
        assert np.allclose(x, y, rtol=0, atol=_tol(big * (1 + np.abs(y).max()), cond, n * m)), (what, x, y)
    qa, qb = pa.update(), pb.update()
    same(qa.mean.values, qb.mean.values, "posterior mean (B10)")
    la, lb = pa.run("pli"), pb.run("pli")
    same(la.mean.values, lb.mean.values, "pli mean")
    same(la.covariance.matrix(), lb.covariance.matrix(), "pli covariance")
    sa, sb = pa.run("sola", gamma=gamma), pb.run("sola", gamma=gamma)
    same(sa.estimate.values, sb.estimate.values, "sola estimate")
    same(sa.noise_law.covariance.matrix(), sb.noise_law.covariance.matrix(), "sola noise law")
    oa, ob = pa.run("least_squares", gamma=gamma), pb.run("least_squares", gamma=gamma)
    same(oa.summary().values, ob.summary().values, "damped least squares (B13)")
    x = oa.summary()
    same(oa.cost(x), ob.cost(x), "the misfit: the C^-1 norm")
    try:
        ga = pg.meta.discrepancy(pa).gamma
    except pg.Refused as e:
        raise AssertionError(f"route A refused by the discrepancy principle: {e}")
    gb = pg.meta.discrepancy(pb).gamma
    assert abs(ga - gb) <= 1e-6 * abs(gb), ("discrepancy gamma", ga, gb)


def _route_a_read_as_unit_noise(mp):
    from pygeoinf0.relations import kernels
    mp.setattr(kernels.AdditiveMarkovKernel, "standardised",            # the defect: C dropped, b kept
               lambda self: (self.effective[0], lambda d: self.datum(d) - (0.0 if self.effective[1].zero_mean
                                                                            else self.effective[1].mean)))


LAWS = {
    law_matrix_adjoint: _adjoint_forgets_domain_metric,
    law_adjoint_involution: _adjoint_forgets_domain_metric,
    law_composition_adjoint: _composition_adjoint_wrong_order,
    law_sum_adjoint: _sum_adjoint_drops_coefficients,
    law_inner_product_symmetric: _metric_applied_unsymmetrised,
    law_finite_rank_covariance: _trace_forgets_the_metric,
    law_posterior_covariance: _posterior_drops_trailing_c,
    law_claims_never_promote: _rule_promotes,
    law_truth_kleene: _kleene_and_is_left,
    law_enclosure_scaling: _enclosure_scaled_unguarded,
    law_enclosure_sum: _enclosure_rounds_to_nearest,
    law_enclosure_positive_scaling: _enclosure_rounds_to_nearest,
    law_enclosure_max: _enclosure_max_takes_the_least_upper_end,
    law_in_is_member_true: _contains_reads_undecided_as_true,
    law_one_point_convention: _points_keep_the_coordinate_axis,
    law_interval_meet_is_exact: _meet_of_intervals_is_none_when_null,
    law_empty_set_identity: _union_with_empty_is_lazy,
    law_support_ends_certified: _minkowski_composes_values,
    law_subset_in_is_member_true: _contains_reads_undecided_as_true,
    law_empty_subset_identity: _subset_join_with_empty_is_lazy,
    law_probability_claim_agrees_with_the_mass: _probability_claimed_for_any_nonnegative_measure,
    law_integration_is_linear_and_one_gives_the_mass: _signed_integral_drops_the_coefficients,
    law_every_state_declares_its_kind: _kind_read_from_the_instance,
    law_no_prior_is_the_whole_space_by_one_route: _whole_space_read_as_a_general_set,
    law_operations_exist_at_their_level: _inner_planted_on_a_banach_space,
    law_complex_inner_product_is_hermitian: _complex_metric_applied_transposed,
    law_covariance_claims_hold: _factored_positivity_ignores_the_diagonal,
    law_covariance_trace_encloses_the_trace: _factored_trace_forgets_the_diagonal,
    law_unbounded_domain_is_dense_and_not_closed: _operator_domain_claimed_closed,
    law_operator_error_rules_bound_the_true_error: _sum_rule_drops_the_coefficients,
    law_operator_error_unknown_where_no_rule_applies: _no_norm_read_as_rounding_only,
    law_every_functional_is_a_functional_of_the_dual: _functional_returned_as_a_bare_form,
    law_direct_sum_of_euclideans_is_a_euclidean: _euclidean_sum_forgets_the_weights,
    law_column_operator_is_the_stacked_matrix: _column_adjoint_forgets_the_block_metrics,
    law_whitening_is_an_isometry_onto_the_whitened_coordinates: _whitening_by_the_inverse_covariance,
    law_finite_subspace_projection_is_orthogonal: _span_coefficients_ignore_the_gram,
    law_discretisation_claim_never_stronger_than_checked: _check_promotes_the_class_claim,
    law_discretisation_error_is_stated_on_the_function_space: _residual_in_coefficient_coordinates,
    law_space_identity_is_independent_of_realisation: _identity_includes_the_realisation,
    law_mixing_realisations_refuses_naming_both: _realisations_read_as_the_same,
    law_symbolic_representer_pairs_in_the_elements_realisation: _symbolic_pairing_read_by_the_calling_space,
    law_quadrature_rules_integrate_their_stated_degree: _quadrature_rule_drops_a_node,
    law_quadrature_rules_are_sharp_where_they_claim: _quadrature_degree_understated,
    law_no_space_builds_a_prior_unless_asked: _l2_default_prior_restored,
    law_named_prior_claims_hold: _legendre_generators_unnormalised,
    law_legendre_decay_states_its_truncation: _legendre_tail_forgets_the_geometric_sum,
    law_region_rules_carry_the_region_measure: _region_rule_drops_the_jacobian,
    law_restrictions_are_an_isometry_onto_the_broken_l2_space: _restriction_halves_a_piece,
    law_broken_norm_is_the_sum_of_the_pieces_norms: _broken_norm_counts_a_piece_twice,
    law_glued_traces_match_and_broken_need_not: _glue_skips_the_trace_check,
    law_piecewise_constant_on_a_partition_is_the_l2_projection: _cells_drop_a_partition_piece,
    law_every_choice_rule_records_whether_it_used_the_datum: _choice_rule_drops_the_datum_claim,
    law_every_result_of_problem_run_carries_its_calibration: _run_drops_the_calibration_claims,
    law_the_engine_never_infers_a_metric: _engine_reads_the_measure,
    law_both_data_routes_give_the_same_inference: _route_a_read_as_unit_noise,
}
NAMES = [f.__name__ for f in LAWS]


@pytest.mark.parametrize("law", LAWS, ids=NAMES)
def test_law_holds(law):
    law()


# what a control could leave behind (plan 071 P5, review item 16): every functools cache of the package (found by
# _package_caches; test_every_package_cache_is_found keeps the walk complete), the package's module-level memos below,
# and this module's lazy case caches (its underscore dicts and _BOX)
_PACKAGE_MEMOS = (("pygeoinf0.core.sampling", "_CHARTS", dict), ("pygeoinf0.catalogue.probing", "_cache", dict),
                  ("pygeoinf0.realisations.rkhs.weighted_solids", "_CONES", lambda: None))
_CACHE_DECORATOR = re.compile(r"^\s*@(functools\.)?(lru_cache|cache)\b", re.M)


def _package_caches():
    """Every functools cache (lru_cache, cache) of the loaded pygeoinf0 modules: their functions and class attributes."""
    found = {}
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "pygeoinf0" or name.startswith("pygeoinf0.")):
            continue
        for v in list(vars(mod).values()):
            members = [getattr(a, "__func__", a) for a in vars(v).values()] if isinstance(v, type) else [v]
            for c in members:
                if callable(getattr(c, "cache_clear", None)) and callable(getattr(c, "cache_info", None)):
                    found[id(c)] = c
    return list(found.values())


def _clear_package_caches():
    for c in _package_caches():
        c.cache_clear()


@contextlib.contextmanager
def _planted(control):
    """The defect of control planted for the duration of the block, then undone, with nothing computed under it
    outliving it: the package's caches are cleared on entry (the defect meets no value cached before it) and on exit,
    its memos and this module's case caches are fresh inside the block and restored after it."""
    with pytest.MonkeyPatch.context() as mp:
        for name, value in list(globals().items()):
            if name.startswith("_") and not name.startswith("__") and type(value) is dict:
                mp.setitem(globals(), name, {})
        mp.setitem(globals(), "_BOX", None)
        for module, name, fresh in _PACKAGE_MEMOS:
            if module in sys.modules:
                mp.setattr(sys.modules[module], name, fresh())
        _clear_package_caches()
        try:
            control(mp)
            yield
        finally:
            _clear_package_caches()                       # before the patches are undone: no call comes between
    _clear_package_caches()


@pytest.mark.parametrize("law", LAWS, ids=NAMES)
def test_law_catches_its_planted_defect(law):
    """The control: with the defect planted the law must fail (an AssertionError of the law, not a crash), or the law
    is vacuous."""
    with _planted(LAWS[law]):
        with pytest.raises(AssertionError):
            law()


def test_no_control_leaves_its_defect_in_a_cache():
    """Plan 071 P5 (review item 16): a rule drawn under a control is not kept after the control. gauss01 is cached and
    calls the module's gauss, so a gauss01(7) drawn while gauss drops a node cached the 6-node rule for every later
    test of the process."""
    Q = _quad()
    with _planted(_quadrature_rule_drops_a_node):
        assert len(Q.gauss01(7)[0]) < 7                            # planted: the rule drawn under it has lost nodes
    kept = [f"{c.__module__}.{c.__qualname__}" for c in _package_caches() if c.cache_info().currsize]
    assert not kept, f"caches kept values drawn under the control: {kept}"
    t, w = Q.gauss01(7)
    assert len(t) == len(w) == 7, f"gauss01(7) kept the planted rule: {len(t)} nodes"
    assert abs(w @ t ** 13 - 1 / 14) <= 1e-15


def test_every_package_cache_is_found():
    """_package_caches finds every functools cache the package's sources declare, so _planted clears them all."""
    import importlib
    from pathlib import Path
    root = Path(pg.__file__).parent
    declared = {}
    for path in root.rglob("*.py"):
        n = len(_CACHE_DECORATOR.findall(path.read_text(encoding="utf-8")))
        if n:
            rel = path.relative_to(root.parent).with_suffix("")
            declared[".".join(p for p in rel.parts if p != "__init__")] = n
    for module in declared:
        importlib.import_module(module)
    found = {}
    for c in _package_caches():
        found[c.__module__] = found.get(c.__module__, 0) + 1
    assert found == declared, f"declared {declared}, found {found}"


def test_every_law_has_a_control():
    laws = {n for n, f in globals().items() if n.startswith("law_") and callable(f)}
    assert laws == set(NAMES), f"laws without a planted-defect control: {sorted(laws - set(NAMES))}"
