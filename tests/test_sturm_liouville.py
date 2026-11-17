"""Plan 009 P3, G5: Sturm-Liouville spectra (closed forms on balls and shells, Galerkin brackets with interfaces), tail
bounds and Bessel-potential covariances (EXP-030 DESIGN G5, NOTES.md section 5). References outside the package: a
Chebyshev collocation of v = r u (NumPy and scipy.linalg.eig), explicit eigenvalues, scipy.integrate.quad, the Green
function of kappa^2 + L_s in modified spherical Bessel functions."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import eig
from scipy.optimize import brentq
from scipy.special import spherical_in, spherical_kn

import pygeoinf0 as pg
from pygeoinf0.core.errors import Refused, Unsupported
from pygeoinf0.operators.unbounded import SturmLiouville


def cheb(N, a, b):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.r_[2.0, np.ones(N - 1), 2.0] * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    D = np.outer(c, 1 / c) / (X - X.T + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    r = a + (b - a) * (x + 1) / 2
    return r, D * 2 / (b - a)


def collocation(a, b, s, inner, outer, n, N=72):
    """Eigenvalues of -v'' + s(s+1)/r^2 v with v = r u (EXP-030 NOTES.md section 5(a)), boundary rows replaced."""
    r, D = cheb(N, a, b)                                   # r[0] = b, r[-1] = a
    A = -D @ D + np.diag(s * (s + 1) / np.where(r > 0, r, 1.0) ** 2)
    B = np.eye(N + 1)
    for row, end, cond in ((0, b, outer), (N, a, inner)):
        A[row], B[row] = 0.0, 0.0
        if end == 0 or cond[0] == "dirichlet":
            A[row, row] = 1.0
        elif row == 0:                                     # outer: v' = v (1 - beta/b)/b
            A[row] = D[row]
            A[row, row] -= (1 - cond[1] / b) / b
        else:                                              # inner: v' = v (1 + beta/a)/a
            A[row] = D[row]
            A[row, row] -= (1 + cond[1] / a) / a
    w = eig(A, B, right=False)
    w = np.sort(w[np.isfinite(w)].real)
    return w[w > -1e-9][:n]


CASES = [(0.0, 2.0, 0, None, ("dirichlet", 0.0)), (0.0, 1.0, 1, None, ("dirichlet", 0.0)),
         (0.0, 1.2, 2, None, ("neumann", 0.0)), (0.0, 1.0, 5, None, ("robin", 2.0)),
         (1.2, 3.5, 3, ("neumann", 0.0), ("dirichlet", 0.0)), (0.5, 1.0, 0, ("dirichlet", 0.0), ("robin", 0.7)),
         (0.5, 1.0, 4, ("robin", 1.5), ("neumann", 0.0))]


@pytest.mark.parametrize("a, b, s, inner, outer", CASES)
def test_closed_form_eigenvalues_against_collocation(a, b, s, inner, outer):
    op = SturmLiouville.radial_laplacian(a, b, s, inner=inner, outer=outer, centre="regular")
    sp = op.spectrum()
    lam = sp.eigenvalues(8)
    ref = collocation(a, b, s, inner or ("dirichlet", 0.0), outer, 8)
    assert np.allclose(lam, ref, rtol=1e-10, atol=1e-10)
    lo, hi = sp.brackets(8)
    assert np.all(lo <= lam) and np.all(lam <= hi) and np.all(hi - lo <= 1e-12 * np.maximum(lam, 1))
    assert np.all(sp.lower_bound(np.arange(1, 9)) <= lo * (1 + 1e-14))


def test_dirichlet_ball_degree_zero_is_n_pi_over_R_and_needs_the_centre_condition():
    with pytest.raises(Refused):
        SturmLiouville.radial_laplacian(0.0, 2.0, 0)
    op = SturmLiouville.radial_laplacian(0.0, 2.0, 0, centre="regular")
    assert np.allclose(op.spectrum().eigenvalues(6), (np.arange(1, 7) * math.pi / 2) ** 2, rtol=1e-14)
    with pytest.raises(Refused):
        pg.Laplacian(pg.Ball(1.0), bc={"boundary": "dirichlet"}).radial(0)
    assert pg.Laplacian(pg.Ball(1.0), bc={"boundary": "dirichlet"}, centre="regular").radial(0) is not None
    assert pg.Laplacian(pg.Ball(1.0), bc={"boundary": "dirichlet"}).radial(3).claims


@pytest.mark.parametrize("a, b, s, inner, outer", CASES[2:5])
def test_eigenfunctions_orthonormal_and_satisfy_the_conditions(a, b, s, inner, outer):
    op = SturmLiouville.radial_laplacian(a, b, s, inner=inner, outer=outer)
    sp = op.spectrum()
    phi = sp.eigenfunctions(5)
    G = np.array([[quad(lambda r: phi(r)[i] * phi(r)[j] * r * r, a, b, limit=400, epsabs=1e-14)[0]
                   for j in range(5)] for i in range(5)])
    assert np.abs(G - np.eye(5)).max() < 1e-12
    h = 1e-5 * (b - a)
    d = (3 * phi(b) - 4 * phi(b - h) + phi(b - 2 * h)) / (2 * h)          # second-order one-sided derivative at b
    lam = sp.eigenvalues(5)
    if outer[0] == "dirichlet":
        assert np.abs(phi(b)).max() < 1e-12
    else:
        assert np.all(np.abs(b * b * d + outer[1] * phi(b)) < 1e-6 * np.maximum(lam, 1) * np.abs(phi(b - 0.1)).max())
    P, u = sp.pl(5)
    assert u == 0 and P.err.max() < 1e-11


def test_tail_lower_bounds_dominate_the_eigenvalues():
    for a, b, s, inner, outer in CASES:
        op = SturmLiouville.radial_laplacian(a, b, s, inner=inner, outer=outer, centre="regular")
        sp = op.spectrum()
        lam = sp.eigenvalues(120)
        assert np.all(sp.lower_bound(np.arange(1, 121)) <= lam * (1 + 1e-13) + 1e-12)


def test_green_function_of_the_q2_bessel_potential_against_the_covariance():
    R, kappa = 1.0, 3.0
    C = pg.Laplacian(pg.Ball(R), bc={"boundary": "dirichlet"}).bessel_potential(kappa, 2.0, degrees=[2, 4], modes=48)
    X = C.domain
    k = lambda r: r * (1 - r) * np.exp(r)                 # noqa: E731
    from pygeoinf0.realisations.series import legendre as LG
    P, _ = LG.fit(lambda r: k(r), 0.0, R)
    F = pg.Family(X, X.backend.separable(P, [4 + 1, 16 + 3], [0, 0], L=4))
    K, E = C.form_bracket(F)
    assert abs(K[0, 1]) < 1e-15                            # different harmonics: zero (rotation invariance)
    for s, i in ((2, 0), (4, 1)):
        def G(r, rp):
            lo, hi = min(r, rp), max(r, rp)
            u2 = spherical_kn(s, kappa * hi) - spherical_kn(s, kappa * R) / spherical_in(s, kappa * R) * \
                spherical_in(s, kappa * hi)
            return (2 * kappa / math.pi) * spherical_in(s, kappa * lo) * u2
        inner = lambda r: quad(lambda rp: G(r, rp) * k(rp) * rp * rp, 0, R, points=[r], epsabs=1e-15,  # noqa: E731
                               epsrel=1e-13, limit=200)[0]
        ref = kappa ** 2 * quad(lambda r: k(r) * r * r * inner(r), 0, R, epsabs=1e-15, epsrel=1e-13, limit=200)[0]
        assert K[i, i] - 1e-12 * ref <= ref <= K[i, i] + E[i, i] + 1e-12 * ref
    assert C.claims.find("rotation invariant").status == "theorem"


def test_bessel_potential_trace_brackets_are_nested_and_refuse_q_le_1():
    Lap = pg.Laplacian(pg.Shell(1.2, 3.5), bc={"inner": "neumann", "outer": "dirichlet"})
    with pytest.raises(Refused):
        Lap.bessel_potential(1.0, 1.0, degrees=[2], modes=8)
    X = pg.L2(pg.Shell(1.2, 3.5), realisation="radial_harmonics")
    T1 = Lap.bessel_potential(2.0, 3.0, degrees=[2, 4], modes=8, space=X).trace()
    T2 = Lap.bessel_potential(2.0, 3.0, degrees=[2, 4], modes=64, space=X).trace()
    assert T1.lower <= T2.lower + 1e-12 and T2.upper <= T1.upper + 1e-12 and T2.width < 0.1 * T1.width
    C = Lap.bessel_potential(2.0, 3.0, degrees=[2], rtol=1e-3, space=X)
    T = C.trace()
    assert T.width <= 2e-3 * T.lower
    mu = pg.GaussianMeasure(X, C)
    m = mu.sample(seed=0)
    assert np.isfinite(m.norm())


def test_galerkin_brackets_contain_exact_eigenvalues():
    sp = SturmLiouville("[0, 1]", 1.0).spectrum()
    lo, up = sp.brackets(10)
    ex = (np.arange(1, 11) * math.pi) ** 2
    assert np.all(lo <= ex) and np.all(ex <= up) and np.all((up - lo) / ex < 1e-8)
    sp = SturmLiouville("[0, 1]", 1.0, q=-5.0, left="neumann", right=("robin", 2.0)).spectrum()
    lo, up = sp.brackets(5)
    ks = [brentq(lambda k: k * math.tan(k) - 2, n * math.pi + 1e-12, n * math.pi + math.pi / 2 - 1e-12, xtol=1e-15)
          for n in range(5)]
    ex = np.array(ks) ** 2 - 5.0
    assert np.all(lo <= ex) and np.all(ex <= up)
    assert np.all(sp.lower_bound(np.arange(1, 6)) <= ex)


def test_galerkin_with_an_interface_and_the_interval_covariance():
    p = pg.Tabulated("[0, 1]", [0, 0.5, 1], [1.0, 4.0], rule="constant")
    sp = SturmLiouville("[0, 1]", p, interfaces=[0.5]).spectrum()
    lo, up = sp.brackets(5)

    def char(k):                                           # sin(k x) on the left, B sin(k (1 - x)/2) on the right
        return math.cos(k / 2) * math.sin(k / 4) + 2 * math.cos(k / 4) * math.sin(k / 2)
    grid = np.linspace(0.1, 30, 3000)
    v = [char(k) for k in grid]
    roots = [brentq(char, grid[i], grid[i + 1], xtol=1e-15) for i in range(len(grid) - 1) if v[i] * v[i + 1] < 0]
    ex = np.array(roots[:5]) ** 2
    assert np.all(lo <= ex) and np.all(ex <= up)
    kappa = 4.0
    op = SturmLiouville("[0, 1]", 1.0)
    C = op.bessel_potential(kappa, 2.0, modes=30)
    exact = (kappa / math.tanh(kappa) - 1) / 2             # sum_n 1/(1 + (n pi/kappa)^2)
    T = C.trace()
    assert T.lower <= exact <= T.upper and T.status == "checked"


def test_laplacian_needs_every_condition_and_a_domain_with_a_spectrum():
    with pytest.raises(ValueError):
        pg.Laplacian(pg.Shell(1.0, 2.0), bc={"outer": "dirichlet"})
    with pytest.raises(ValueError):
        pg.Laplacian(pg.Ball(1.0), bc={"inner": "dirichlet", "boundary": "neumann"})
    with pytest.raises(Unsupported):                        # plan 068 P3: not built on a disk (was TypeError)
        pg.Laplacian(pg.Disk(1.0), bc={"boundary": "dirichlet"})
    assert "Laplacian" in pg.__all__
    Lap = pg.Laplacian(pg.Shell(1.0, 2.0), bc={"inner": ("robin", 0.5), "outer": "neumann"})
    assert Lap.claims.find("rotation invariant").status == "theorem"


def test_separable_grams_with_a_degree_wise_radial_covariance_are_structured_and_exact():
    Lap = pg.Laplacian(pg.Ball(1.0), bc={"boundary": "dirichlet"})
    C = Lap.bessel_potential(3.0, 2.0, degrees=[2, 3], modes=24).normalised(1.0)
    X = C.domain
    lab = [(2, 0), (2, 3), (3, 1), (3, 1), (2, 0)]
    ks = [lambda r: r * (1 - r), lambda r: r ** 2, lambda r: np.sin(3 * r) * r, lambda r: r ** 3, lambda r: r]
    G = pg.Functionals.separable(X, ks, lab)
    B = G.structured_gram(C)
    assert B is not None and B.claims.find("rotation invariant").status == "theorem"
    assert any(c.key == "truncation" for c in B.claims)
    dense = np.array(X.inner(C(G.kernels), G.kernels))
    assert np.allclose(B.matrix(), dense, rtol=1e-11, atol=1e-13 * np.abs(dense).max())
    assert abs(B.matrix()[0, 1]) == 0.0 and abs(B.matrix()[0, 4]) > 0
