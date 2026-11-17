"""Plan 009 P4b: covariances that act degree by degree (probability/degrees.py). A Bessel-potential covariance on a ball
congruent by a radial multiplication (a uniformiser, a depth-dependent amplitude), transported to a decomposition of the
ball into shells (layered planets, plan 007), congruent by a number per degree and summand, with isotropic covariances
on sphere summands (boundary topographies). References: the dense application of the covariance, the undivided ball,
NumPy trapezoid sums of the closed-form eigenfunctions, kernels rescaled by hand."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported

KS = [lambda r: r * (1 - r), lambda r: r ** 2, lambda r: np.sin(3 * r) * r, lambda r: r ** 3, lambda r: r + 0.2]
LAB = [(2, 0), (2, 3), (3, 1), (3, 1), (2, 0)]


@pytest.fixture(scope="module")
def setup():
    lap = pg.Laplacian(pg.Ball(1.0), bc={"boundary": "dirichlet"})
    C = lap.bessel_potential(3.0, 2.0, degrees=[2, 3], modes=24)
    X = C.domain
    j = pg.Tabulated((0.0, 1.0), np.linspace(0, 1, 9), 1.0 + 0.5 * np.cos(np.linspace(0, 3, 9)), rule="linear",
                     name="j")
    J = pg.Multiplication(X, j)
    return lap, C, X, j, J, C.congruence(J)


def test_congruence_by_a_radial_multiplication_keeps_the_degree_structure(setup):
    lap, C, X, j, J, CJ = setup
    assert CJ.degree_parts is not None and CJ.claims.find("rotation invariant").holds
    G = pg.Functionals.separable(X, KS, LAB)
    B = G.structured_gram(CJ)
    assert B is not None and B.claims.find("rotation invariant").status == "theorem"
    dense = np.array(X.inner(J(C(J(G.kernels))), G.kernels))
    assert np.allclose(B.matrix(), dense, rtol=1e-11, atol=1e-12 * np.abs(dense).max())
    gens, lam, _ = C.finite
    Jg = J(gens)
    t = float(lam @ np.diag(np.atleast_2d(X.inner(Jg, Jg))))
    assert float(CJ.trace().lower) == pytest.approx(t, rel=1e-11)
    assert CJ.trace().upper >= t and CJ.rest is not None


def test_pointwise_variance_and_spectrum(setup):
    lap, C, X, j, J, CJ = setup
    r = np.linspace(0.05, 0.95, 7)
    ref = np.zeros(7)
    for s in (2, 3):
        sp = lap.radial(s).spectrum()
        lam = (1.0 + sp.eigenvalues(24) / 9.0) ** -1.0
        ref += (2 * s + 1) / (4 * math.pi) * (sp.eigenfunctions(24)(r) ** 2 @ lam)
    assert np.allclose(C.radial_variance(r), ref, rtol=1e-11)
    jr = np.interp(r, np.linspace(0, 1, 9), 1.0 + 0.5 * np.cos(np.linspace(0, 3, 9)))
    assert np.allclose(CJ.radial_variance(r), ref * jr ** 2, rtol=1e-11)
    ev, mult, tail, tmax = CJ.degree_spectrum()
    assert float(ev @ mult) == pytest.approx(float(CJ.trace().lower), rel=1e-11)
    assert tail == pytest.approx(CJ.rest["trace"]) and ev.max() <= 1.5 ** 2 * 1.0 + 1e-12
    ev0, _, _, _ = C.degree_spectrum()
    assert np.allclose(np.sort(ev0), np.sort(C.finite[1][np.r_[0:24, 5 * 24:6 * 24]]))


def _pieces(cut=0.4, w=0.8, wh=0.3):
    planet = pg.LayeredBall([cut, 1.0], names=("a", "b"))
    V = pg.PiecewiseSpace.lebesgue(planet, realisation="radial_harmonics")
    S2 = pg.L2(pg.Sphere(1.0))
    M = pg.DirectSum({"a": V.summand("a"), "b": V.summand("b"), "h": S2}, weights={"a": w, "b": w, "h": wh})
    M0 = pg.DirectSum({"x": pg.L2(pg.Ball(1.0), realisation="radial_harmonics"), "h": S2}, weights={"x": w, "h": wh})
    hv = [0.7, -0.2, 0.3, 1.1, 0.4]
    return M, M0, S2, hv


def test_glued_pieces_equal_the_undivided_ball_and_the_weight_convention(setup):
    lap, C, X, j, J, CJ = setup
    M, M0, S2, hv = _pieces()
    A = pg.Covariance.angular(S2, {2: 0.5, 3: 0.25})
    assert float(A.trace()) == pytest.approx(5 * 0.5 + 7 * 0.25)
    K1 = pg.Covariance.glued(M, {("a", "b"): CJ, "h": A})
    K0 = pg.Covariance.glued(M0, {"x": CJ, "h": A})
    G1 = pg.Functionals.separable(M, {"a": KS, "b": KS, "h": hv}, LAB)
    G0 = pg.Functionals.separable(M0, {"x": KS, "h": hv}, LAB)
    B1, B0 = G1.structured_gram(K1).matrix(), G0.structured_gram(K0).matrix()
    assert np.allclose(B1, B0, rtol=1e-11, atol=1e-12 * np.abs(B0).max())
    GX = pg.Functionals.separable(X, KS, LAB)
    vol = GX.structured_gram(CJ).matrix() / 0.8                          # <g, C g>_M = <k, C k>_0 / w
    GS = pg.Functionals.separable(S2, hv, LAB)
    sph = np.array(S2.inner(A(GS.kernels), GS.kernels)) / 0.3
    assert np.allclose(B0, vol + sph, rtol=1e-11, atol=1e-12 * np.abs(B0).max())
    assert float(K1.trace()) == pytest.approx(float(CJ.trace()) + float(A.trace()), rel=1e-12)
    with pytest.raises(Unsupported):
        K1(M.zero)


def test_rules_and_scalars_per_summand(setup):
    lap, C, X, j, J, CJ = setup
    M, _, S2, hv = _pieces()
    A = pg.Covariance.angular(S2, {2: 0.5, 3: 0.25})
    q = pg.realisations.Trapezoid(201, over=(0.0, 1.0))
    CJq = C.congruence(J, quadrature=q)
    K = pg.Covariance.glued(M, {("a", "b"): CJq, "h": A})
    Gq = pg.Functionals.separable(M, {"a": KS, "b": KS, "h": hv}, LAB, quadrature={"a": q, "b": q})
    B = Gq.structured_gram(K).matrix()
    x, w = np.linspace(0, 1, 201), np.full(201, 1 / 200)
    w[[0, -1]] *= 0.5
    jx = np.interp(x, np.linspace(0, 1, 9), 1.0 + 0.5 * np.cos(np.linspace(0, 3, 9)))
    ref = np.zeros((5, 5))
    for s in (2, 3):
        sp = lap.radial(s).spectrum()
        lam = (1.0 + sp.eigenvalues(24) / 9.0) ** -1.0
        Phi = sp.eigenfunctions(24)(x)
        rows = [i for i, (l, _) in enumerate(LAB) if l == s]
        Am = np.array([(w * x * x * KS[i](x) * jx) @ Phi for i in rows])
        for a_, i in enumerate(rows):
            for b_, k in enumerate(rows):
                if LAB[i] == LAB[k]:
                    ref[i, k] = (Am[a_] * lam) @ Am[b_] / 0.8 + (0.5 if s == 2 else 0.25) * hv[i] * hv[k] / 0.3
    assert np.allclose(B, ref, rtol=1e-11, atol=1e-12 * np.abs(ref).max())
    tq = sum((2 * s + 1) * float((1.0 + lap.radial(s).spectrum().eigenvalues(24) / 9.0) ** -1.0 @
                                 ((w * x * x * jx * jx) @ lap.radial(s).spectrum().eigenfunctions(24)(x) ** 2))
             for s in (2, 3))
    assert float(CJq.trace().lower) == pytest.approx(tq, rel=1e-12)
    # D = 2 I (+) 3 I (+) 5 I: the same Gram as the kernels scaled by 2 and 3 and the sphere part by 25
    D = pg.BlockDiagonal(M, [2.0 * pg.Identity(M.summand("a")), 3.0 * pg.Identity(M.summand("b")),
                             5.0 * pg.Identity(S2)])
    KD = pg.Covariance.glued(M, {("a", "b"): CJ, "h": A}).congruence(D)
    G = pg.Functionals.separable(M, {"a": KS, "b": KS, "h": hv}, LAB)
    Gs = pg.Functionals.separable(M, {"a": [lambda r, f=f: 2 * f(r) for f in KS],
                                      "b": [lambda r, f=f: 3 * f(r) for f in KS], "h": [5 * v for v in hv]}, LAB)
    K1 = pg.Covariance.glued(M, {("a", "b"): CJ, "h": A})
    BD, Bs = G.structured_gram(KD).matrix(), Gs.structured_gram(K1).matrix()
    assert np.allclose(BD, Bs, rtol=1e-11, atol=1e-12 * np.abs(Bs).max())
    ev, mult, _, _ = KD.degree_spectrum()
    assert float(ev @ mult) == pytest.approx(float(KD.trace().lower), rel=1e-10)
    half = KD._scaled(0.5, "half")
    assert np.allclose(G.structured_gram(half).matrix(), 0.5 * BD, rtol=1e-13)


def test_glued_refuses_pieces_that_do_not_tile(setup):
    lap, C, X, j, J, CJ = setup
    M, _, S2, _ = _pieces()
    with pytest.raises(ValueError):
        pg.Covariance.glued(M, {("b", "a"): CJ})
    with pytest.raises(ValueError):
        pg.Covariance.glued(M, {("a",): CJ})
    with pytest.raises(Unsupported):
        pg.Covariance.angular(X, {2: 1.0})



def test_restricting_separable_functionals_keeps_the_structure(setup):
    """Exact deletion of data rows (plan 009 G9) on separable functionals: the restricted map is again separable, so
    its structured Gram is the submatrix of the full one (no dense route)."""
    lap, C, X, j, J, CJ = setup
    M, _, S2, hv = _pieces()
    K = pg.Covariance.glued(M, {("a", "b"): CJ, "h": pg.Covariance.angular(S2, {2: 0.5, 3: 0.25})})
    G = pg.Functionals.separable(M, {"a": KS, "b": KS, "h": hv}, LAB, into=pg.Euclidean(5))
    full = G.structured_gram(K).matrix()
    idx = [4, 0, 3]
    BJ = G.select_rows(idx, pg.Euclidean(3)).structured_gram(K)
    assert BJ is not None
    assert np.allclose(BJ.matrix(), full[np.ix_(idx, idx)], rtol=1e-13, atol=1e-15 * np.abs(full).max())
    mdr = pg.Additive(G, pg.GaussianMeasure(G.codomain, pg.Diagonal(np.arange(1.0, 6.0), G.codomain)))
    GJ, noise = mdr.select([0, 3, 4]).effective
    keep = [0, 3, 4]
    assert hasattr(GJ, "structured_gram")
    assert np.allclose(GJ.structured_gram(K).matrix(), full[np.ix_(keep, keep)], rtol=1e-13)



def test_the_omitted_part_of_a_congruence_is_bounded_sharply(setup):
    """The trace of the omitted part of J R J: explicit to 4N modes, ||J||^2 times the tail beyond; it must exceed the
    true omitted trace (from a 16x longer known part) and be far below the crude ||J||^2 tr R."""
    lap, C, X, j, J, CJ = setup
    crude = J.norm_bound() ** 2 * C.rest["trace"]
    sharp = CJ.rest["trace"]
    C16 = lap.bessel_potential(3.0, 2.0, degrees=[2, 3], modes=384)
    true_omitted = float(C16.congruence(J).trace().lower) - float(CJ.trace().lower)
    assert true_omitted <= sharp < crude
    assert float(CJ.trace().upper) >= float(C16.congruence(J).trace().lower)
