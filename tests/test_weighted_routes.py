"""Weighted measures as functionals on Sobolev spaces (EXP-031 phase W3): the same-piece, touching and generic pair
routes of realisations/rkhs/weighted.py and weighted_volume.py against the oracles of the experiment (computed there by
independent methods, digits copied from experiments/EXP-031-weighted-measures/results/*.json):

    P4  weighted segment, H^0.75 and H^1.25 in 2-D      the trace identity (Proposition 2), w^ in closed form
    P6  curve (t, 0.5 + 0.15 sin 2 pi t), w = 1 + x     Richardson-extrapolated exact polyline energies
    P2  weighted sphere in a ball, H^2 and H^1 (3-D)     Funk-Hecke
    P3  sub-box (2-D) and sub-ball (3-D)                 the package's exact polygon potential; the radial Fourier
                                                         transform
    P1  the bump average in H^2 of the unit square        the Hankel transform
plus constant weights through the new entries against the existing exact atoms, additivity over split pieces, cases
with s <= d/2, and error estimates that dominate the actual errors."""
import math
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg

warnings.simplefilter("ignore", pg.RealisationWarning)
P0, Q0 = np.array([0.1, 0.2]), np.array([0.8, 0.7])
LSEG = float(np.linalg.norm(Q0 - P0))
SEG_W = {"const": lambda x, y: 1 + 0 * x, "linear": lambda x, y: 1 + 2 * np.hypot(x - 0.1, y - 0.2) / LSEG,
         "cos": lambda x, y: np.cos(3 * math.pi * np.hypot(x - 0.1, y - 0.2) / LSEG)}
P4 = {(0.75, "const"): 3.2728930162164764, (0.75, "linear"): 14.001600874294866, (0.75, "cos"): 1.0931341589778323,
      (1.25, "const"): 1.3252088878518176, (1.25, "linear"): 5.580067903304508, (1.25, "cos"): 0.2003682814427872}


def energy(H, mu):
    k = H.riesz(mu)
    E = k.norm() ** 2
    return E, 2 * math.sqrt(E) * float(k.realisation.kx.err[0])      # the estimate of the energy from that of ||k||


@pytest.mark.parametrize("s, kind", list(P4))
def test_weighted_segments_against_the_trace_identity(s, kind):
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), s, length_scale=0.2)
    E, est = energy(H, pg.LebesgueMeasure(pg.Segment(P0, Q0)).with_density(SEG_W[kind]))
    assert E == pytest.approx(P4[(s, kind)], rel=1e-12)
    assert est >= abs(E - P4[(s, kind)]) or est < 1e-13 * E


@pytest.mark.parametrize("s", [0.75, 1.25])
def test_segment_routes_agree_with_the_exact_atoms_and_add_up(s):
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), s, length_scale=0.2)
    seg = pg.Segment(P0, Q0)
    kc, kw = H.riesz(seg), H.riesz(pg.LebesgueMeasure(seg).with_density(SEG_W["const"]))
    assert kw.norm() == pytest.approx(kc.norm(), rel=1e-13)
    assert H.inner(kc, kw) == pytest.approx(kc.norm() ** 2, rel=1e-11)                 # a constant and a weighted atom
    X = np.array([[0.3, 0.5], [0.45, 0.45], [0.9, 0.9]])                               # (0.45, 0.45) is on the segment
    assert np.allclose(kw(X[:, 0], X[:, 1]), kc(X[:, 0], X[:, 1]), rtol=1e-9, atol=0)
    M = 0.5 * (P0 + Q0)                                                                # halves sharing a vertex
    halves = [pg.LebesgueMeasure(pg.Segment(a, b)).with_density(SEG_W["linear"]) for a, b in ((P0, M), (M, Q0))]
    assert H.riesz(halves[0] + halves[1]).norm() ** 2 == pytest.approx(P4[(s, "linear")], rel=1e-12)
    right = [pg.LebesgueMeasure(pg.Segment((0.3, 0.3), q)).with_density(SEG_W["const"]) for q in ((0.7, 0.3), (0.3, 0.6))]
    exact = H.inner(H.riesz(pg.Segment((0.3, 0.3), (0.7, 0.3))), H.riesz(pg.Segment((0.3, 0.3), (0.3, 0.6))))
    assert H.inner(H.riesz(right[0]), H.riesz(right[1])) == pytest.approx(exact, rel=1e-12)   # common vertex, right angle


def test_weighted_subintervals_of_an_interval():
    H = pg.Sobolev(pg.Interval(0, 1), 0.75, length_scale=0.2)                        # s > 1/2 = d/2: points bounded
    mu = pg.LebesgueMeasure(pg.Interval(0.2, 0.6)).with_density(lambda x: 1 + 0 * x)
    assert H.riesz(mu).norm() == pytest.approx(H.riesz(pg.Interval(0.2, 0.6)).norm(), rel=1e-13)
    f = H(lambda x: np.cos(3 * x))
    nu = pg.LebesgueMeasure(pg.Interval(0.2, 0.6)).with_density(lambda x: x)
    assert H.inner(H.riesz(nu), f) == pytest.approx((np.cos(1.8) / 9 + 0.6 * np.sin(1.8) / 3)
                                                    - (np.cos(0.6) / 9 + 0.2 * np.sin(0.6) / 3), rel=1e-13)


@pytest.mark.parametrize("s, ref, tol", [(1.25, 3.382156247487753, 1e-10), (0.75, 8.224030235367914, 1e-8)])
def test_a_curve_given_by_a_callable_against_richardson_extrapolated_polylines(s, ref, tol):
    # the oracle's own accuracy: 5e-11 (s = 1.25) and 6e-9 (s = 0.75) relative (DESIGN 6.6)
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), s, length_scale=0.2)
    C = pg.Curve(lambda t: (t, 0.5 + 0.15 * np.sin(2 * np.pi * t)), 0.1, 0.9)
    E, est = energy(H, pg.LebesgueMeasure(C).with_density(lambda x, y: 1 + x))
    assert E == pytest.approx(ref, rel=tol)


@pytest.mark.parametrize("s", [0.75, 1.25])
def test_the_chart_route_on_a_circle_equals_the_package_circle(s):
    H = pg.Sobolev(pg.Box((-1, 1), (-1, 1)), s, length_scale=0.2)
    circ = pg.Circle(0.5, (0.1, 0.0))
    p = circ.parametrisation()
    C = pg.Curve(p.gamma, 0, 2 * np.pi, derivative=p.derivative, closed=True)
    kc, kw = H.riesz(circ), H.riesz(pg.LebesgueMeasure(C).with_density(1.0))
    tol = 1e-12 if s > 1 else 2e-9                                    # the package's circle is a graded quadrature
    assert kw.norm() ** 2 == pytest.approx(kc.norm() ** 2, rel=tol)
    assert abs(kw.norm() - kc.norm()) <= kw.realisation.kx.err[0] + kc.realisation.kx.err[0] + 1e-14


SPHERE_W = {"poly": lambda x, y, z: 1 + 0.5 * (z + 0.1) / 0.6 + 0.3 * (((x - 0.1) / 0.6) ** 2 - (y / 0.6) ** 2),
            "exp": lambda x, y, z: np.exp((z + 0.1) / 0.6), "sign": lambda x, y, z: (z + 0.1) / 0.6}
P2 = {(2, 0.3, "poly"): 3.5981114832222136, (2, 0.3, "sign"): 0.6076146759100987,
      (1, 0.1, "exp"): 40.717111410171796, (1, 0.1, "sign"): 7.330445913576664}


@pytest.mark.parametrize("s, l, kind", list(P2))
def test_weighted_spheres_against_funk_hecke(s, l, kind):
    H = pg.Sobolev(pg.Ball(1.0), s, length_scale=l)
    E, est = energy(H, pg.LebesgueMeasure(pg.Sphere(0.6, (0.1, 0, -0.1))).with_density(SPHERE_W[kind]))
    assert E == pytest.approx(P2[(s, l, kind)], rel=1e-12)


def test_a_constant_weight_on_a_sphere_through_the_new_route():
    H = pg.Sobolev(pg.Ball(1.0), 1, length_scale=0.1)
    S = pg.Sphere(0.6, (0.1, 0, -0.1))
    kw = H.riesz(pg.LebesgueMeasure(S).with_density(lambda x, y, z: 1 + 0 * x))
    assert kw.norm() == pytest.approx(H.riesz(S).norm(), rel=1e-13)


P3 = {(2, 0.2): (0.03070912511904142, 0.09420375250709914), (0.75, 0.2): (0.06873295275970989, 0.21140765702648562)}


@pytest.mark.parametrize("s, l", list(P3))
def test_sub_boxes_against_the_polygon_potential_route(s, l):
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), s, length_scale=l)
    sub = pg.Box((0.2, 0.6), (0.3, 0.7))
    E1, _ = energy(H, sub)
    E2, est = energy(H, pg.LebesgueMeasure(sub).with_density(lambda x, y: (1 + x) * (1 + 0.5 * y)))
    assert E1 == pytest.approx(P3[(s, l)][0], rel=1e-12) and E2 == pytest.approx(P3[(s, l)][1], rel=1e-12)
    k1 = H.riesz(sub)
    X = np.array([[0.3, 0.4], [0.7, 0.5], [0.2, 0.5]])                   # inside, outside, on an edge
    k3 = H.riesz(pg.LebesgueMeasure(sub).with_density(lambda x, y: 1 + 0 * x))    # the polar fan route
    assert np.allclose(k3(X[:, 0], X[:, 1]), k1(X[:, 0], X[:, 1]), rtol=1e-11, atol=0)


@pytest.mark.slow(reason="22 s on europa at load 26")
def test_split_boxes_add_up():
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), 2, length_scale=0.2)
    w = lambda x, y: (1 + x) * (1 + 0.5 * y)                            # noqa: E731
    parts = [pg.LebesgueMeasure(pg.Box((0.2, 0.4), (0.3, 0.7))).with_density(w),
             pg.LebesgueMeasure(pg.Box((0.4, 0.6), (0.3, 0.7))).with_density(w)]
    k = H.riesz(parts[0] + parts[1])
    assert k.norm() ** 2 == pytest.approx(P3[(2, 0.2)][1], rel=1e-9)
    assert abs(k.norm() ** 2 - P3[(2, 0.2)][1]) <= 2 * k.norm() * k.realisation.kx.err[0] + 1e-15


def test_the_bump_target_against_the_hankel_reference():
    C0, R0, Z = np.array([0.3, 0.65]), 0.15, 0.028532523625370905

    def bump(x, y):
        q = np.clip(np.hypot(x - C0[0], y - C0[1]) / R0, 0, 1)
        with np.errstate(divide="ignore", over="ignore"):
            return np.where(q < 1, np.exp(1 - 1 / np.maximum(1 - q * q, 1e-300)), 0.0)
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), 2, length_scale=0.1)
    T = pg.LebesgueMeasure(pg.Disk(R0, tuple(C0))).with_density(bump)
    assert T.mass()[0] == pytest.approx(Z, rel=1e-9)
    E, est = energy(H, T)
    assert E / Z ** 2 == pytest.approx(4.993167743973474, rel=1e-10)
    assert est / Z ** 2 >= abs(E / Z ** 2 - 4.993167743973474)


@pytest.mark.parametrize("s, l, refs", [(2, 0.3, (0.08843594539865834, 0.01764253789379371)),
                                        (1, 0.1, (0.3727920590038102, 0.09188958582684902))])
def test_sub_balls_against_the_radial_fourier_reference(s, l, refs):
    H = pg.Sobolev(pg.Ball(1.0), s, length_scale=l)
    S = pg.Ball(0.5, (0.1, 0, -0.1))
    assert energy(H, S)[0] == pytest.approx(refs[0], rel=1e-12)
    if s == 1:
        mu = pg.LebesgueMeasure(S).with_density(lambda x, y, z: 1 - ((x - 0.1) ** 2 + y ** 2 + (z + 0.1) ** 2) / 0.25)
        E, est = energy(H, mu)
        assert E == pytest.approx(refs[1], rel=1e-11) and est >= abs(E - refs[1])


def test_weighted_measures_as_data_and_values():
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), 1.5, length_scale=0.15)
    mus = [pg.LebesgueMeasure(pg.Disk(0.1, (0.3, 0.3))).with_density(lambda x, y: 1 + x),
           pg.LebesgueMeasure(pg.Segment((0.1, 0.8), (0.9, 0.6))).with_density(lambda x, y: x * y),
           pg.LebesgueMeasure(pg.Box((0.5, 0.8), (0.2, 0.5))).with_density(lambda x, y: np.exp(x)),
           pg.Segment((0.2, 0.1), (0.2, 0.9))]
    G = pg.Functionals(H.riesz(mus))
    A = (G @ G.adjoint).A if hasattr(G @ G.adjoint, "A") else None
    assert A is not None and np.allclose(A, A.T) and np.all(np.linalg.eigvalsh(A) > 0)
    f = H(lambda x, y: np.sin(2 * x) + y)
    import scipy.integrate as si
    ref = si.dblquad(lambda r, t: (1 + 0.3 + r * np.cos(t)) * (np.sin(2 * (0.3 + r * np.cos(t))) + 0.3 + r * np.sin(t))
                     * r, 0, 2 * np.pi, 0, 0.1, epsabs=1e-13)[0]
    assert G(f).values[0] == pytest.approx(ref, rel=1e-11)
    k = H.riesz(mus[0])                                       # values: polar route inside, cone outside
    x = np.array([[0.3, 0.3], [0.35, 0.28], [0.45, 0.3], [0.8, 0.8]])
    ref = [si.dblquad(lambda r, t, p=p: (1 + 0.3 + r * np.cos(t)) * r * pg.realisations.rkhs.kernels.Matern(1.5, 2, 0.15)(
        np.hypot(0.3 + r * np.cos(t) - p[0], 0.3 + r * np.sin(t) - p[1])), 0, 2 * np.pi, 0, 0.1, epsabs=1e-13)[0]
        for p in x[2:]]
    assert np.allclose(k(x[2:, 0], x[2:, 1]), ref, rtol=1e-9)


def test_weighted_patches_in_3d_through_the_new_route():
    H = pg.Sobolev(pg.Ball(1.0), 1.5, length_scale=0.3)
    P = pg.Patch((-0.3, -0.2, 0.1), (0.5, 0.1, 0.0), (0.1, 0.4, 0.2))
    kc, kw = H.riesz(P), H.riesz(pg.LebesgueMeasure(P).with_density(lambda x, y, z: 1 + 0 * x))
    assert kw.norm() == pytest.approx(kc.norm(), rel=1e-13)
    X = np.array([[0.0, 0.0, 0.3], [0.05, 0.05, 0.13], [0.6, 0.6, 0.0]])
    assert np.allclose(kw(*X.T), kc(*X.T), rtol=1e-13, atol=0)
    assert H.inner(kc, kw) == pytest.approx(kc.norm() ** 2, rel=1e-13)
    kp = H.riesz(pg.LebesgueMeasure(P).with_density(lambda x, y, z: 1 + x + y * z))     # Duffy, both levels agree
    assert 2 * kp.norm() * kp.realisation.kx.err[0] < 1e-12 * kp.norm() ** 2


@pytest.mark.parametrize("s, tol", [(2, 1e-12), (0.75, 1e-12)])
def test_simplicial_triangles_add_up_to_the_square(s, tol):
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), s, length_scale=0.2)
    w = lambda x, y: (1 + x) * (1 + 0.5 * y)                                   # noqa: E731
    sq = H.riesz(pg.LebesgueMeasure(pg.Box((0.2, 0.6), (0.3, 0.7))).with_density(w)).norm() ** 2
    tri = pg.Simplicial([[0.2, 0.3], [0.6, 0.3], [0.6, 0.7], [0.2, 0.7]], [[0, 1, 2], [0, 2, 3]])
    k = H.riesz(pg.LebesgueMeasure(tri).with_density(w))
    E = k.norm() ** 2
    assert E == pytest.approx(sq, rel=tol)                   # touching cells: the common-edge route (O1)
    assert abs(E - sq) <= 2 * k.norm() * k.realisation.kx.err[0] + 1e-13 * sq


def test_a_triangle_in_3d_against_the_exact_polygon_potential():
    H = pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.3)
    tri = pg.Simplex((0.1, 0.0, 0.2), (0.5, 0.1, 0.1), (0.2, 0.4, 0.3))
    k = H.riesz(tri)
    from pygeoinf0.realisations.rkhs.atoms import FlatWeight
    from pygeoinf0.realisations.rkhs.weighted_volume import TrianglePiece
    from pygeoinf0.realisations.rkhs.kernels import Matern
    piece = TrianglePiece(*tri.vertices, FlatWeight(constant=1.0))
    X, W = piece.formula_nodes(40)
    ref = float(W @ piece.potential(Matern(2, 3, 0.3), X, 1))           # outer Gauss x exact potential (s > d/2)
    assert k.norm() ** 2 == pytest.approx(ref, rel=1e-11)
    x = np.array([[0.3, 0.2, 0.5]])
    kw = H.riesz(pg.LebesgueMeasure(tri).with_density(lambda x, y, z: 1 + 0 * x))
    assert kw(*x.T) == pytest.approx(k(*x.T), rel=1e-12)


@pytest.mark.slow(reason="19 and 30 s on europa at load 26")
@pytest.mark.parametrize("s", [2, 0.75])
def test_four_triangles_and_split_patches_add_up(s):
    """A square as four triangles about its centre (common edges and common vertices), and a patch in 3-D as two
    patches sharing an edge, against the single-piece routes."""
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), s, length_scale=0.2)
    w = lambda x, y: (1 + x) * (1 + 0.5 * y)                                   # noqa: E731
    sq = H.riesz(pg.LebesgueMeasure(pg.Box((0.2, 0.6), (0.3, 0.7))).with_density(w)).norm() ** 2
    V = [[0.4, 0.5], [0.2, 0.3], [0.6, 0.3], [0.6, 0.7], [0.2, 0.7]]
    four = pg.Simplicial(V, [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
    assert H.riesz(pg.LebesgueMeasure(four).with_density(w)).norm() ** 2 == pytest.approx(sq, rel=1e-12)
    B = pg.Sobolev(pg.Ball(1.0), s + 0.5, length_scale=0.3)
    w3 = lambda x, y, z: 1 + x + y * z                                         # noqa: E731
    whole = B.riesz(pg.LebesgueMeasure(pg.Patch((-0.3, -0.2, 0.1), (0.5, 0.1, 0.0), (0.1, 0.4, 0.2)))
                    .with_density(w3)).norm() ** 2
    halves = [pg.LebesgueMeasure(pg.Patch(o, (0.25, 0.05, 0.0), (0.1, 0.4, 0.2))).with_density(w3)
              for o in ((-0.3, -0.2, 0.1), (-0.05, -0.15, 0.1))]
    assert B.riesz(halves[0] + halves[1]).norm() ** 2 == pytest.approx(whole, rel=1e-12)


def test_a_bent_pair_of_triangles_in_3d_is_consistent_at_two_levels():
    H = pg.Sobolev(pg.Ball(1.0), 1, length_scale=0.3)                          # s <= d/2: singular kernel
    roof = pg.Simplicial([[0, 0, 0], [0.5, 0, 0], [0.25, 0.3, 0.2], [0.25, -0.3, 0.2]], [[0, 1, 2], [0, 1, 3]])
    k = H.riesz(roof)
    assert 2 * k.norm() * k.realisation.kx.err[0] < 1e-11 * k.norm() ** 2


def test_pairings_with_formulas_carry_a_two_order_estimate():
    """EXP-031 O4: int f w dsigma for weighted pieces (Sobolev) and symbolic L2 representers at two orders, the
    difference in the error of the data."""
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), 1.5, length_scale=0.2)
    mus = [pg.LebesgueMeasure(pg.Disk(0.15, (0.4, 0.5))).with_density(lambda x, y: 1 + x),
           pg.LebesgueMeasure(pg.Segment((0.1, 0.2), (0.8, 0.7))).with_density(lambda x, y: 1 + y)]
    K = H.riesz(mus)
    v = H.inner(K, H(lambda x, y: np.exp(-((x - 0.4) ** 2 + (y - 0.5) ** 2) / 0.01)))
    e = H.backend.last_pairing_error[:, 0]          # in H^s a formula's own error is unknown (NaN) and masks it in G(f)
    assert np.all(e > 0) and np.all(e < 1e-9 * np.abs(v))
    M = pg.L2(pg.Box((0, 1), (0, 1)))
    GL = pg.Functionals(M.riesz(mus[:1]))
    dL = GL(M(lambda x, y: np.exp(-((x - 0.4) ** 2 + (y - 0.5) ** 2) / 0.01)))
    assert dL.err[0] > 0 and dL.err[0] < 1e-9 * abs(dL.values[0])
    sharp = GL(M(lambda x, y: np.exp(-((x - 0.4) ** 2 + (y - 0.5) ** 2) / 1e-5)))      # a narrow peak in the disk
    exact = 1.4 * np.pi * 1e-5                                                         # int (1 + x) e^(-r^2 / 1e-5)
    assert abs(sharp.values[0] - exact) <= sharp.err[0] + 1e-12 * exact
