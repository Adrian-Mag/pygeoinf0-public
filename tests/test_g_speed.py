"""Plan 031: G(m) by one table of kernels, direct elements of radial-harmonic spaces, a block solver."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.realisations.series import radial_direct as RHM
from pygeoinf0.realisations.series.radial_direct import harmonic_element


def problem(L=4):
    M = pg.L2(pg.Shell(1.0, 2.0), realisation="radial_harmonics")
    radial = [lambda r, k=k: np.cos(k * r) for k in range(5)]
    lab = [(l, j) for l in range(L + 1) for j in range(2 * l + 1)]
    items = [radial[i % 5] for i in range(len(lab))]
    return M, pg.Functionals.separable(M, items, lab), lab


def element(M, seed=0):
    rng = np.random.default_rng(seed)
    r = np.linspace(1.1, 1.9, 7)
    return harmonic_element(M, r, rng.standard_normal((7, 16))), r


def test_the_table_equals_one_harmonic_at_a_time():
    M, G, _ = problem()
    f, _ = element(M)
    A, B = G.kernels.realisation, M.family([f]).realisation
    assert np.abs(RHM.gram_sr_rh(A, B) - RHM._by_harmonic(A, B)).max() < 1e-13


def test_the_table_handles_many_members_and_harmonics_beyond_the_field():
    M, G, lab = problem(L=4)                                 # kernels to degree 4, fields to degree 3
    fs = M.family([element(M, s)[0] for s in range(3)])
    A, B = G.kernels.realisation, fs.realisation
    full = RHM._by_harmonic(A, B)
    assert np.abs(RHM.gram_sr_rh(A, B) - full).max() < 1e-13
    assert np.all(full[[l >= 4 for l, _ in lab]] == 0.0)
    old, RHM.GRAM_SR_RH_MAX = RHM.GRAM_SR_RH_MAX, 200       # a smaller table: split by members, then by harmonics
    try:
        assert np.abs(RHM.gram_sr_rh(A, B) - full).max() < 1e-13
        RHM.GRAM_SR_RH_MAX = 1
        assert np.abs(RHM.gram_sr_rh(A, B) - full).max() < 1e-13
    finally:
        RHM.GRAM_SR_RH_MAX = old


def test_g_of_a_direct_element_is_the_integral():
    """f = a(r) Y_21 with a linear between radii, zero outside: <cos(k r) Y_21, f> by an independent quadrature."""
    M = pg.L2(pg.Shell(1.0, 2.0), realisation="radial_harmonics")
    r = np.array([1.2, 1.5, 1.7])
    a = np.array([1.0, -2.0, 0.5])
    c = np.zeros((3, 9))
    c[:, 5] = a
    f = harmonic_element(M, r, c)
    G = pg.Functionals.separable(M, [lambda x: np.cos(3 * x)], [(2, 1)])
    g, w = np.polynomial.legendre.leggauss(40)
    ref = 0.0
    for lo, hi in zip(r[:-1], r[1:]):                        # the integrand has a kink at each radius
        x = 0.5 * (hi - lo) * g + 0.5 * (hi + lo)
        ref += 0.5 * (hi - lo) * np.sum(w * np.cos(3 * x) * np.interp(x, r, a) * x * x)
    assert G(f).values[0] == pytest.approx(ref, rel=1e-12)


def test_a_direct_element_is_the_values_and_zero_outside():
    M = pg.L2(pg.Shell(1.0, 2.0), realisation="radial_harmonics")
    c = np.zeros((2, 4))
    c[:, 0] = [1.0, 3.0]                                     # Y_00 = 1 / sqrt(4 pi)
    f = harmonic_element(M, [1.5, 1.7], c)
    y00 = 1.0 / np.sqrt(4.0 * np.pi)
    v = lambda r: float(f(np.array([r]), np.array([0.0]), np.array([0.0]))[0])
    assert v(1.6) == pytest.approx(2.0 * y00) and v(1.3) == 0.0 and v(1.9) == 0.0


def test_harmonic_element_refusals():
    M = pg.L2(pg.Shell(1.0, 2.0), realisation="radial_harmonics")
    with pytest.raises(ValueError, match="shape"):
        harmonic_element(M, [1.2, 1.3], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="ascending"):
        harmonic_element(M, [1.3, 1.2], np.zeros((2, 4)))
    with pytest.raises(ValueError, match="within"):
        harmonic_element(M, [0.5, 1.2], np.zeros((2, 4)))
    with pytest.raises(pg.core.errors.Unsupported, match="radial-harmonic"):
        harmonic_element(pg.L2(pg.Box("[0, 1]", "[0, 1]")), [0.1, 0.2], np.zeros((2, 4)))


def blocks(cond=10.0, shift=True, seed=0):
    rng = np.random.default_rng(seed)
    D = pg.Euclidean(7)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    K1 = Q @ np.diag([1.0, 0.5, 1.0 / cond]) @ Q.T
    K2 = np.array([[2.0, 0.3], [0.3, 1.0]])
    K1, K2 = 0.5 * (K1 + K1.T), 0.5 * (K2 + K2.T)
    order = rng.permutation(7)
    A = pg.BlockDiagonal(D, [K1, K2], shared=[0, 1, 1], order=order, name="A")
    return D, (A + pg.Diagonal(rng.uniform(0.1, 1.0, 7), D, name="C")) if shift else A


@pytest.mark.parametrize("shift", [True, False])
def test_block_direct_is_the_dense_solve_and_is_recommended(shift):
    D, A = blocks(shift=shift)
    I = A.inverse()
    assert isinstance(I.solver, pg.BlockDirect)
    y = D(np.arange(1.0, 8.0))
    assert I(y).values == pytest.approx(np.linalg.solve(A.matrix(), y.values), rel=1e-12)
    Y = D.family(np.eye(7))
    assert I(Y).values == pytest.approx(np.linalg.inv(A.matrix()), rel=1e-10)


def test_block_direct_forms_nothing_n_by_n(monkeypatch):
    D, A = blocks()
    monkeypatch.setattr(type(A), "matrix", lambda self: pytest.fail("the block solver formed the matrix"))
    A.inverse()(D(np.ones(7)))


def test_block_direct_gives_a_block_its_own_pseudo_inverse_on_request():
    D, A = blocks(cond=1e14, shift=False)
    P = A.pseudo_inverse(1e-12)
    x = P(D(np.arange(1.0, 8.0))).values
    M = A.matrix()
    assert np.linalg.norm(x) < 1e3                            # the near-null direction of the first block is dropped
    assert np.linalg.norm(M @ x - np.arange(1.0, 8.0)) < 20.0 and "1 directions dropped, in 1 blocks" in P.report()
    assert np.linalg.norm(A.inv(D(np.arange(1.0, 8.0))).values) > 1e10   # the inverse keeps it
