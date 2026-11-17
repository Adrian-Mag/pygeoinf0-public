"""Spectral Sobolev spaces on rectangles and boxes (plan 006, P3): D(A^(s/2)) for A = 1 - l^2 Delta with Neumann or
Dirichlet faces, realised by signed reflection images of the Matern kernel, against independent cosine/sine series."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported, ZeroFunctional


def axis_eigen(L, lo, hi, N):
    """L2-orthonormal eigenfunctions and eigenvalues of -d^2 on [0, L] with the conditions lo, hi ('N' or 'D')."""
    m = np.arange(N)
    if lo == "N" and hi == "N":
        return (lambda x: np.where(m == 0, 1 / math.sqrt(L), math.sqrt(2 / L) * np.cos(np.pi * m * np.asarray(x)[..., None] / L)),
                (np.pi * m / L) ** 2)
    if lo == "D" and hi == "D":
        return lambda x: math.sqrt(2 / L) * np.sin(np.pi * (m + 1) * np.asarray(x)[..., None] / L), (np.pi * (m + 1) / L) ** 2
    trig = np.sin if lo == "D" else np.cos
    return lambda x: math.sqrt(2 / L) * trig(np.pi * (m + 0.5) * np.asarray(x)[..., None] / L), (np.pi * (m + 0.5) / L) ** 2


def series(s, l, sides, conds, X, Y, N=1000):
    """K(x, y) = sum (1 + l^2 (mu_m + nu_n))^-s psi_m(x1) psi_m(y1) phi_n(x2) phi_n(y2) for points X, Y (k, 2)."""
    (f, mu), (g, nu) = (axis_eigen(L, *c, N) for L, c in zip(sides, conds))
    W = (1 + l * l * (mu[:, None] + nu[None, :])) ** -s
    return np.array([(f(x[0]) * f(y[0])) @ W @ (g(x[1]) * g(y[1])) for x, y in zip(X, Y)])


def rect_space(s, l, dirichlet=(), sides=(1.0, 0.7)):
    D = pg.Box(f"[0, {sides[0]}]", f"[0, {sides[1]}]")
    return pg.Sobolev(pg.SobolevForm(D, 1, length_scale=l, dirichlet=dirichlet).operator(), s), D


@pytest.mark.parametrize("s", [2.5, 2.3])
@pytest.mark.parametrize("dirichlet, conds", [((), ("NN", "NN")), ("all", ("DD", "DD")),
                                              (("x-", "y+"), ("DN", "ND"))])
def test_rectangle_kernels_against_eigen_series(s, dirichlet, conds):
    l = 0.15
    H, _ = rect_space(s, l, dirichlet)
    P = np.array([[0.1, 0.2], [0.5, 0.35], [0.93, 0.66], [0.3, 0.05]])
    X, Y = P[[0, 1, 2, 3, 0, 1]], P[[0, 1, 2, 3, 2, 3]]
    k = H.riesz([pg.DiracMeasure(tuple(x)) for x in X])
    q = H.riesz([pg.DiracMeasure(tuple(y)) for y in Y])
    got = np.diag(H.inner(k, q))
    ref = series(s, l, (1.0, 0.7), conds, X, Y)
    assert np.allclose(got, ref, rtol=2e-6, atol=1e-7 * ref.max())
    assert np.allclose(k[4](*Y[4]), got[4], rtol=1e-12)                    # values of representers: the same kernel


def test_neumann_order_one_is_the_intrinsic_h1_form_on_a_rectangle():
    l = 0.2
    D = pg.Box("[0, 1]", "[0, 0.7]")
    a = pg.SobolevForm(D, 1, length_scale=l)
    H = a.space()
    assert H == pg.Sobolev(a.operator(), 1) != pg.Sobolev(D, 1, length_scale=l)
    r1, r2 = pg.Segment((0.1, 0.3), (0.8, 0.3)), pg.Segment((0.5, 0.05), (0.5, 0.6))     # crossing rays (s > 1/2)
    E = H.inner(H.riesz(r1), H.riesz(r2))
    N = 3000
    m = np.arange(N)
    am = np.where(m == 0, 0.7, math.sqrt(2) / (np.pi * np.maximum(m, 1)) * (np.sin(np.pi * m * 0.8) - np.sin(np.pi * m * 0.1)))
    bn = np.where(m == 0, 0.55 / math.sqrt(0.7), math.sqrt(2 / 0.7) * 0.7 / (np.pi * np.maximum(m, 1)) *
                  (np.sin(np.pi * m * 0.6 / 0.7) - np.sin(np.pi * m * 0.05 / 0.7)))
    psi = np.where(m == 0, 1.0, math.sqrt(2) * np.cos(np.pi * m * 0.5))             # psi_m(0.5) on [0, 1]
    phi = np.where(m == 0, 1 / math.sqrt(0.7), math.sqrt(2 / 0.7) * np.cos(np.pi * m * 0.3 / 0.7))
    W = 1 / (1 + l * l * ((np.pi * m)[:, None] ** 2 + (np.pi * m / 0.7)[None, :] ** 2))
    ref = (am * psi) @ W @ (phi * bn)
    assert E == pytest.approx(ref, rel=2e-5)


def test_box_kernel_with_mixed_faces_against_a_triple_series():
    l, s, N = 0.1, 3.5, 150
    D = pg.Box("[0, 1]", "[0, 0.8]", "[0, 0.6]")
    H = pg.Sobolev(pg.SobolevForm(D, 1, length_scale=l, dirichlet=("y-", "z-", "z+")).operator(), s)
    (f, mu), (g, nu), (h, rho) = (axis_eigen(L, lo, hi, N) for L, lo, hi in [(1.0, "N", "N"), (0.8, "D", "N"),
                                                                              (0.6, "D", "D")])
    W = (1 + l * l * (mu[:, None, None] + nu[None, :, None] + rho[None, None, :])) ** -s
    P = np.array([[0.0, 0.4, 0.3], [0.5, 0.8, 0.1], [0.97, 0.05, 0.55]])
    k = H.riesz([pg.DiracMeasure(tuple(p)) for p in P])
    got = H.inner(k, k)
    ref = np.array([[np.einsum("i,j,k,ijk->", f(x[0]) * f(y[0]), g(x[1]) * g(y[1]), h(x[2]) * h(y[2]), W)
                     for y in P] for x in P])
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-6 * ref.max())
    H2 = pg.Sobolev(pg.SobolevForm(D, 1, length_scale=l, dirichlet=("y-",)).operator(), 2.0)
    p, q, x = np.array([0.0, 0.2, 0.1]), np.array([0.9, 0.7, 0.5]), (0.6, 0.1, 0.45)   # a ray from a Neumann face
    E = H2.inner(H2.riesz(pg.Segment(tuple(p), tuple(q))), H2.riesz(pg.DiracMeasure(x)))
    t, w = np.polynomial.legendre.leggauss(64)
    Y = p + 0.5 * (t[:, None] + 1) * (q - p)
    kx = H2.riesz(pg.DiracMeasure(x))
    assert E == pytest.approx(0.5 * np.linalg.norm(q - p) * w @ kx(Y), rel=1e-10)    # int over the ray of K(x, .)


def test_variances_at_faces_edges_and_corners_of_a_neumann_box():
    l, s = 0.05, 2.0                                     # images 2l apart or more contribute < e^-20
    D = pg.Box("[0, 1]", "[0, 1]", "[0, 1]")
    H = pg.Sobolev(pg.SobolevForm(D, 1, length_scale=l).operator(), s)
    k0 = math.gamma(s - 1.5) / (math.gamma(s) * (4 * math.pi) ** 1.5 * l ** 3)
    v = pg.probability.cylindrical(H).variance(np.array([[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [0.0, 1.0, 0.5], [1.0, 0.0, 1.0]]))
    assert np.allclose(v / k0, [1, 2, 4, 8], rtol=1e-6)
    HD = pg.Sobolev(pg.SobolevForm(D, 1, length_scale=l, dirichlet="all").operator(), s)
    assert pg.probability.cylindrical(HD).variance(np.array([[0.0, 0.5, 0.5]]))[0] == 0.0          # zero functional on a Dirichlet face
    assert pg.probability.cylindrical(HD).variance(np.array([[0.5, 0.5, 0.5]]))[0] == pytest.approx(k0, rel=1e-6)


def test_dirichlet_faces_refusals_and_claims():
    l = 0.1
    H, D = rect_space(1.5, l, dirichlet=("x-",))
    face = pg.Segment((0.0, 0.1), (0.0, 0.6))                                       # a ray inside the Dirichlet face
    assert H.riesz(face).norm() == 0
    with pytest.raises(ZeroFunctional):
        pg.Functionals(H.riesz([face, pg.Segment((0.2, 0.1), (0.8, 0.6))]))
    assert H.riesz(pg.Segment((0.0, 0.1), (0.5, 0.6))).norm() > 0                 # a ray leaving the face is not zero
    with pytest.raises(Unsupported, match="Robin"):
        pg.Sobolev(pg.SobolevForm(D, 1, length_scale=l, robin=0.2).operator(), 1.5)
    with pytest.raises(Unsupported):
        pg.SobolevForm(D, 2, length_scale=l).space()
    assert "conditions on elements at x-: u = 0" in repr(H.claims) and "reflection images" in repr(H.claims)


def test_rectangle_inference_through_the_matrix_free_route():
    l, s = 0.08, 2.5
    H, D = rect_space(s, l, dirichlet=("y-",))
    rng = np.random.default_rng(5)
    P = rng.uniform([0.0, 0.02], [1.0, 0.7], (300, 2))
    truth = lambda x, y: np.sin(3 * x) * y                                          # noqa: E731
    sigma = 0.01
    Dd = pg.Euclidean(len(P), covariance=sigma ** 2)
    d = Dd(truth(*P.T) + sigma * rng.standard_normal(len(P)))
    out = {}
    for dense_max in (10_000, 60):
        with pg.settings.using(dense_max=dense_max, accuracy=1e-6):
            G = pg.Functionals(H.riesz([pg.DiracMeasure(tuple(p)) for p in P]), into=Dd)
            alpha = pg.solvers.damping_for_misfit(G @ G.adjoint, d, d.space.dim)
            post = pg.probability.cylindrical(H, alpha=alpha).condition(G, d)
            out[dense_max] = (alpha, post.mean(np.array([0.2, 0.5]), np.array([0.3, 0.0])))
            if dense_max == 60:
                assert G.kernels.realisation.kx.atoms.hmatrix.stats["far"] > 0
    assert out[60][0] == pytest.approx(out[10_000][0], rel=1e-5)
    assert np.allclose(out[60][1], out[10_000][1], rtol=1e-5, atol=1e-7)
    assert out[10_000][1][1] == 0.0 or abs(out[10_000][1][1]) < 1e-12               # Dirichlet face y = 0
