"""Weighted measures on the kinds of set and space that W3 refused (EXP-031 O3): images of pieces (tori, spectral
Neumann/Dirichlet spaces, corrected interval spaces), boxes and parallelepipeds in 3-D, declared singular weights on
segments and subintervals, constant-weight shells, sphere charts covering their sphere; and the refusals that remain,
each with its reason."""
import math
import warnings

import numpy as np
import pytest
import scipy.integrate as si

import pygeoinf0 as pg

warnings.simplefilter("ignore", pg.RealisationWarning)
ONE2 = lambda x, y: 1 + 0 * x                                                  # noqa: E731  (a callable weight 1)
ONE3 = lambda x, y, z: 1 + 0 * x                                               # noqa: E731


def test_images_on_a_torus_and_in_spectral_spaces():
    H = pg.Sobolev(pg.Torus("[0, 1]", "[0, 1]"), 1.5, length_scale=0.2)
    seg = pg.Segment((0.1, 0.2), (0.8, 0.7))
    kc, kw = H.riesz(seg), H.riesz(pg.LebesgueMeasure(seg).with_density(ONE2))
    assert kw.norm() == pytest.approx(kc.norm(), rel=1e-13) and H.inner(kc, kw) == pytest.approx(kc.norm() ** 2,
                                                                                                    rel=1e-13)
    N = pg.Sobolev(pg.SobolevForm(pg.Box((0, 1), (0, 1)), 1, length_scale=0.2).operator(), 1.5)    # Neumann images
    seg = pg.Segment((0.0, 0.2), (0.6, 0.7))                                    # from a face
    assert N.riesz(pg.LebesgueMeasure(seg).with_density(ONE2)).norm() == pytest.approx(N.riesz(seg).norm(), rel=1e-13)
    k = N.riesz(pg.LebesgueMeasure(pg.Box((0.0, 0.3), (0.2, 0.5))).with_density(ONE2))       # a sub-box at a face
    f = N(lambda x, y: np.cos(np.pi * x) * np.cos(2 * np.pi * y))
    ref = si.dblquad(lambda y, x: np.cos(np.pi * x) * np.cos(2 * np.pi * y), 0, 0.3, 0.2, 0.5)[0]
    assert N.inner(k, f) == pytest.approx(ref, rel=1e-12) and k.realisation.kx.err[0] < 1e-10 * k.norm()
    I = pg.SobolevForm(pg.Interval(0, 1), 1, length_scale=0.2).space()        # integer order: boundary correction
    kc, kw = I.riesz(pg.Interval(0.2, 0.5)), I.riesz(pg.LebesgueMeasure(pg.Interval(0.2, 0.5)).with_density(
        lambda x: 1 + 0 * x))
    assert kw.norm() == pytest.approx(kc.norm(), rel=1e-13)


@pytest.mark.slow(reason="16 s on europa at load 26")
def test_boxes_in_3d_add_up_and_do_not_depend_on_orientation():
    H = pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.3)
    whole = H.riesz(pg.Box((-0.3, 0.2), (-0.2, 0.2), (-0.1, 0.25)))
    halves = [pg.Box(a, (-0.2, 0.2), (-0.1, 0.25)) for a in ((-0.3, -0.05), (-0.05, 0.2))]
    k = H.riesz(pg.LebesgueMeasure(halves[0]) + pg.LebesgueMeasure(halves[1]))
    assert k.norm() == pytest.approx(whole.norm(), rel=1e-12)
    th = 0.4
    Rm = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    rotated = pg.domains.Flat(np.array([-0.3, -0.2, -0.1]) @ Rm.T, np.diag([0.5, 0.4, 0.35]) @ Rm.T)
    assert H.riesz(rotated).norm() == pytest.approx(whole.norm(), rel=1e-12)
    kw = H.riesz(pg.LebesgueMeasure(pg.Box((-0.3, 0.2), (-0.2, 0.2), (-0.1, 0.25))).with_density(
        lambda x, y, z: 1 + x + 0.5 * y * z))
    assert 2 * kw.norm() * kw.realisation.kx.err[0] < 1e-8 * kw.norm() ** 2      # two levels: 5.5e-10 measured
    X = np.array([[0.0, 0.0, 0.0], [0.6, 0.5, 0.1]])                              # near (polar) and far points
    kc1 = H.riesz(pg.LebesgueMeasure(pg.Box((-0.3, 0.2), (-0.2, 0.2), (-0.1, 0.25))).with_density(ONE3))
    assert np.allclose(kc1(*X.T), whole(*X.T), rtol=1e-12)


def _oracle(l, a, c, J):
    """int int exp(-|x - y| / l) / (2 l) w(x) w(y), w = |x - c|^-a on J: scipy with algebraic weights (s = 1, d = 1)."""
    def P(x):
        tot = 0.0
        for y0, y1 in ((J[0], min(x, c)), (min(x, c), max(x, c)), (max(x, c), J[1])):
            if y1 <= y0:
                continue
            f = lambda y: np.exp(-abs(x - y) / l) / (2 * l)                      # noqa: E731
            if y1 == c:
                tot += si.quad(f, y0, y1, weight="alg", wvar=(0, -a), epsabs=1e-15, epsrel=1e-13)[0]
            elif y0 == c:
                tot += si.quad(f, y0, y1, weight="alg", wvar=(-a, 0), epsabs=1e-15, epsrel=1e-13)[0]
            else:
                tot += si.quad(lambda y: f(y) * abs(y - c) ** -a, y0, y1, epsabs=1e-15, epsrel=1e-13,
                               points=[x] if y0 < x < y1 else None)[0]
        return tot
    return (si.quad(P, J[0], c, weight="alg", wvar=(0, -a), epsabs=1e-15, epsrel=1e-12, limit=200)[0]
            + si.quad(P, c, J[1], weight="alg", wvar=(-a, 0), epsabs=1e-15, epsrel=1e-12, limit=200)[0])


def test_declared_singular_weights_on_subintervals_and_segments():
    l, a, c = 0.2, 0.3, 0.45
    H = pg.Sobolev(pg.Interval(0, 1), 1, length_scale=l)
    mu = pg.LebesgueMeasure(pg.Interval(0.2, 0.7)).with_density(lambda x: np.abs(x - c) ** -a, singular_at=[c],
                                                                    exponent=a)
    k = H.riesz(mu)
    E, ref = k.norm() ** 2, _oracle(l, a, c, (0.2, 0.7))
    assert E == pytest.approx(ref, rel=1e-10)
    assert 2 * k.norm() * k.realisation.kx.err[0] >= abs(E - ref)              # the floor of the omitted cells
    B = pg.Sobolev(pg.Box((0, 1), (0, 1)), 0.75, length_scale=0.2)              # s <= d/2
    seg = pg.Segment((0.1, 0.5), (0.9, 0.5))
    nu = pg.LebesgueMeasure(seg).with_density(lambda x, y: np.abs(x - 0.5) ** -a, singular_at=[(0.5, 0.5)], exponent=a)
    kb = B.riesz(nu)
    f = B(lambda x, y: np.cos(3 * x) + y)
    ref = sum(si.quad(lambda x: np.cos(3 * x) + 0.5, *J, weight="alg", wvar=wv, epsabs=1e-14)[0]
              for J, wv in (((0.1, 0.5), (0, -a)), ((0.5, 0.9), (-a, 0))))
    assert B.inner(kb, f) == pytest.approx(ref, rel=1e-10)
    with pytest.raises(pg.errors.Unsupported, match="not in L2 of the piece"):
        B.riesz(pg.LebesgueMeasure(seg).with_density(lambda x, y: np.abs(x - 0.5) ** -0.6, singular_at=[(0.5, 0.5)],
                                                     exponent=0.6))
    with pytest.raises(pg.errors.Unsupported, match="segments, polylines and subintervals only"):
        B.riesz(pg.LebesgueMeasure(pg.Disk(0.2, (0.5, 0.5))).with_density(
            lambda x, y: np.hypot(x - 0.5, y - 0.5) ** -0.3, singular_at=[(0.5, 0.5)], exponent=0.3))


def _shell_fourier(s, l, r, R, X):
    def ballhat(rho, a):
        x = rho * a
        return np.where(x < 1e-3, 4 * np.pi * a ** 3 / 3 * (1 - x * x / 10),
                        4 * np.pi * (np.sin(x) - x * np.cos(x)) / np.maximum(rho, 1e-300) ** 3)
    e = np.linspace(0, X, int(X / 0.5) + 1)
    g, gw = np.polynomial.legendre.leggauss(12)
    rho = (0.5 * (e[1:] + e[:-1])[:, None] + 0.5 * np.diff(e)[:, None] * g).ravel()
    w = (0.5 * np.diff(e)[:, None] * gw).ravel()
    m = ballhat(rho, R) - ballhat(rho, r)
    E = float(w @ (rho ** 2 * (1 + l * l * rho * rho) ** (-s) * m * m)) * 4 * np.pi / (2 * np.pi) ** 3
    return E + 4 * np.pi / (2 * np.pi) ** 3 * 8 * np.pi ** 2 * (R * R + r * r) * l ** (-2 * s) * X ** (-1 - 2 * s) / (1 + 2 * s)


@pytest.mark.parametrize("s, l", [pytest.param(2, 0.3, marks=pytest.mark.slow(reason="66 s on europa at load 26")), (1, 0.1)])
def test_constant_shells_against_the_radial_fourier_transform(s, l):
    H = pg.Sobolev(pg.Ball(1.0), s, length_scale=l)
    assert H.riesz(pg.Shell(0.3, 0.6)).norm() ** 2 == pytest.approx(_shell_fourier(s, l, 0.3, 0.6, 2000 / l), rel=1e-12)
    if s == 2:                                     # a weighted shell: ball minus ball by the round-pair route (O5)
        k = H.riesz(pg.LebesgueMeasure(pg.Shell(0.3, 0.6)).with_density(lambda x, y, z: 1 + 0 * x))
        ref = _shell_fourier(s, l, 0.3, 0.6, 2000 / l)
        assert k.norm() ** 2 == pytest.approx(ref, rel=1e-8)                   # 3-D lens orders: ~3e-9
        assert 2 * k.norm() * k.realisation.kx.err[0] >= abs(k.norm() ** 2 - ref)


def test_sphere_charts_covering_the_sphere_and_their_refusals():
    ph = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, ph, 0], [1, ph, 0], [-1, -ph, 0], [1, -ph, 0], [0, -1, ph], [0, 1, ph], [0, -1, -ph], [0, 1, -ph],
                  [ph, 0, -1], [ph, 0, 1], [-ph, 0, -1], [-ph, 0, 1]], float)
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7],
         [9, 8, 1]]
    S = pg.Sphere(0.6, (0.1, 0, -0.1))
    ico = pg.Simplicial(0.3 * V + np.array([0.1, 0, -0.1]), F, chart=S)
    H = pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.3)
    w = lambda x, y, z: 1 + z                                                  # noqa: E731
    assert H.riesz(pg.LebesgueMeasure(ico).with_density(w)).norm() == pytest.approx(
        H.riesz(pg.LebesgueMeasure(S).with_density(w)).norm(), rel=1e-14)
    with pytest.raises(pg.errors.Unsupported, match="cover"):
        H.riesz(pg.DensityMeasure(pg.LebesgueMeasure(ico), np.arange(20.0)))
    with pytest.raises(pg.errors.Unsupported, match="cover"):
        H.riesz(pg.Simplicial(0.3 * V + np.array([0.1, 0, -0.1]), F[:10], chart=S))


def test_tetrahedra_by_the_cone_decomposition_of_the_difference_body():
    from pygeoinf0.realisations.rkhs.weighted_solids import _cones, _size
    vol = sum(abs(np.linalg.det(np.array(c))) / 6 for c in _cones())
    assert vol == pytest.approx(10 / 3, rel=1e-14)                              # T - T, a cuboctahedron
    H = pg.Sobolev(pg.Ball(1.0), 1, length_scale=0.3)                          # s <= d/2
    cells = [[0, 1, 2, 6], [0, 2, 3, 6]]                                        # two Kuhn tetrahedra (congruent)
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) * 0.4 - 0.2
    a, b = (H.riesz(pg.Simplicial(V, [c])) for c in cells)
    assert a.norm() == pytest.approx(b.norm(), rel=1e-9)             # 7e-11: their reference maps differ
    assert 2 * a.norm() * a.realisation.kx.err[0] < 1e-9 * a.norm() ** 2
    assert _size(np.zeros((1, 3)))[0] == 1.0



@pytest.mark.parametrize("s", [2, 0.75])
def test_overlapping_disks_against_the_hankel_transform(s):
    from scipy import special
    l, (ca, Ra), (cb, Rb) = 0.1, ((0.3, 0.4), 0.15), ((0.45, 0.5), 0.1)
    D = math.hypot(0.15, 0.1)
    e = np.linspace(0, 4000.0, 16001)
    g, gw = np.polynomial.legendre.leggauss(12)
    rho = (0.5 * (e[1:] + e[:-1])[:, None] + 0.5 * np.diff(e)[:, None] * g).ravel()
    w = (0.5 * np.diff(e)[:, None] * gw).ravel()
    ref = float(w @ (rho * (1 + l * l * rho * rho) ** (-s) * (2 * np.pi * Ra * special.j1(rho * Ra) / rho)
                     * (2 * np.pi * Rb * special.j1(rho * Rb) / rho) * special.j0(rho * D))) / (2 * np.pi)
    H = pg.Sobolev(pg.Box((0, 1), (0, 1)), s, length_scale=l)
    ka, kb = H.riesz(pg.Disk(Ra, ca)), H.riesz(pg.Disk(Rb, cb))
    assert H.inner(ka, kb) == pytest.approx(ref, rel=1e-9)                     # 1.6e-10 (s = 2), 2e-11 (s = 0.75)


@pytest.mark.slow(reason="92 s on europa at load 26")
def test_joins_of_simplices_agree_with_the_triangle_routes_and_tile_a_cube():
    """weighted_joins.py: simplices sharing a face, edge or vertex in join coordinates; for triangles against the
    dedicated routes of weighted_touching.py, for tetrahedra by a cube tiled with six Kuhn tetrahedra (every pair
    shares an edge or a face) against the box route."""
    from pygeoinf0.realisations.rkhs.atoms import FlatWeight
    from pygeoinf0.realisations.rkhs.kernels import Matern
    from pygeoinf0.realisations.rkhs.weighted_contacts import join_energy
    from pygeoinf0.realisations.rkhs.weighted_contacts import edge_energy, vertex_energy
    from pygeoinf0.realisations.rkhs.weighted_volume import TrianglePiece
    w = FlatWeight(lambda X: (1 + X[:, 0]) * (1 + 0.5 * X[:, 1]))
    k = Matern(0.75, 2, 0.2)
    a, b = TrianglePiece([0.2, 0.3], [0.6, 0.7], [0.6, 0.3], w), TrianglePiece([0.2, 0.3], [0.6, 0.7], [0.2, 0.7], w)
    assert join_energy(k, a, b, [0, 1], [0, 1], 1) == pytest.approx(edge_energy(k, a, b, [0, 1], [0, 1], 1), rel=1e-9)
    c = np.array([0.4, 0.5])
    t0, t1 = TrianglePiece(c, [0.2, 0.3], [0.6, 0.3], w), TrianglePiece(c, [0.6, 0.7], [0.2, 0.7], w)
    assert join_energy(k, t0, t1, [0], [0], 1) == pytest.approx(vertex_energy(k, t0, t1, 0, 0, 1), rel=1e-10)
    from pygeoinf0.realisations.rkhs.weighted_solids import TetraPiece
    k3 = Matern(1, 3, 0.3)                                                    # two tetrahedra sharing a face
    one = FlatWeight(constant=1.0)
    A = TetraPiece([0, 0, 0], [0.4, 0, 0], [0.4, 0.4, 0], [0.4, 0.4, 0.4], one)
    B = TetraPiece([0, 0, 0], [0.4, 0, 0], [0.4, 0.4, 0], [0.4, 0, -0.4], one)        # the other side of z = 0
    lo, hi = (join_energy(k3, A, B, [0, 1, 2], [0, 1, 2], lv) for lv in (0, 1))
    assert abs(hi - lo) < 1e-7 * hi                                          # measured 1e-8


@pytest.mark.slow(reason="382 s on europa at load 26")
def test_a_cube_tiled_by_six_kuhn_tetrahedra():
    """Every pair of the six Kuhn tetrahedra shares an edge or a face (weighted_joins.py): their energies add up to
    the cube's (box route). Passed on europa, 2026-10-28: 1e-10 (250 s)."""
    import itertools
    H = pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.6)
    V = np.array(list(itertools.product((0, 1), repeat=3)), float) * 0.4 - 0.2
    cells = []
    for perm in itertools.permutations(range(3)):
        idx, cur = [0], np.zeros(3, int)
        for ax in perm:
            cur = cur.copy()
            cur[ax] = 1
            idx.append(int(cur[0] * 4 + cur[1] * 2 + cur[2]))
        cells.append(idx)
    kuhn = H.riesz(pg.Simplicial(V, cells))
    cube = H.riesz(pg.Box((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)))
    assert kuhn.norm() == pytest.approx(cube.norm(), rel=1e-10)


def test_integer_order_bessel_kernels_match_kv():
    """Plan 028: orders 0 and 1 of z^m K_m(z) use Cephes k0 and k1 (5x faster than kv, the H^1 kernel in 2-D): the
    same values to 4.1e-14 relative (4e-15 for z <= 50; the worst near underflow, z about 700), at 0 (K_0 infinite,
    z K_1 -> 1) and in the far tail."""
    from scipy import special
    from pygeoinf0.realisations.rkhs.kernels import _zK
    z = np.r_[0.0, 1e-300, 1e-12, np.geomspace(1e-8, 700, 2000), 746.0]
    with np.errstate(all="ignore"):
        k0, k1 = special.kv(0, z), np.where(z < 1e-300, 1.0, z * special.kv(1, z))
    a, b = _zK(0, z), _zK(1, z)
    assert np.isinf(a[0]) and b[0] == 1.0 and a[-1] == 0.0 and b[-1] == 0.0
    ok = np.isfinite(k0) & (k0 > 0)
    assert np.max(np.abs(a[ok] / k0[ok] - 1)) < 1e-13
    ok = k1 > 0
    assert np.max(np.abs(b[ok] / k1[ok] - 1)) < 1e-13
