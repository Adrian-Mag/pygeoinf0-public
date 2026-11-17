"""Complex Euclidean spaces (plan 020 P3a, EXP-035 COMPLEX.md): the inner-product convention (linear in the first
argument, conjugate-linear in the second), the complex structure, Hermitian metrics, real spaces refusing complex
values, and real models with complex data through the port: a Debye relaxation (nonlinear) and a Fourier-domain
continuation (linear), checked against the closed-form real Tikhonov solution."""
import numpy as np
import pytest

import pygeoinf0 as pg


def test_inner_product_convention_and_complex_structure():
    rng = np.random.default_rng(0)
    C = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    W = C @ C.conj().T + 4 * np.eye(4)                             # Hermitian positive definite
    for D in (pg.Euclidean(4, field="C"), pg.Euclidean(4, weights=[1, 2, 3, 4], field="C"), pg.Euclidean(4, metric=W, field="C")):
        x, y = D.random(seed=1), D.random(seed=2)
        a = 0.7 - 1.3j
        assert np.isclose(D.cinner(a * x, y), a * D.cinner(x, y))                   # linear in the first
        assert np.isclose(D.cinner(x, a * y), np.conj(a) * D.cinner(x, y))          # conjugate-linear in the second
        assert np.isclose(D.cinner(x, y), np.conj(D.cinner(y, x)))
        assert np.isclose(D.inner(x, y), D.cinner(x, y).real)
        assert np.isclose(D.norm(D.J(x)), D.norm(x))                                # J is an isometry ...
        assert np.allclose(D.J(D.J(x)).values, -x.values)                           # ... with J^2 = -I
        assert D.cinner(x, x).real > 0
    one, i = pg.Euclidean(1, field="C")(np.array([1.0])), pg.Euclidean(1, field="C")(np.array([1j]))
    assert np.isclose(pg.Euclidean(1, field="C").cinner(one, i), -1j)                   # <1, i> = 1 conj(i) = -i
    assert np.isclose(pg.Euclidean(2, covariance=np.array([4.0, 0.25]), field="C").norm(
        pg.Euclidean(2, covariance=np.array([4.0, 0.25]), field="C")(np.array([2.0, 0.5j]))), np.sqrt(2.0))


def test_real_spaces_refuse_complex_values():
    with pytest.raises(TypeError, match="imaginary parts would be lost"):
        pg.Euclidean(2)(np.array([1.0, 1j]))
    with pytest.raises(TypeError, match="complex scalar"):
        pg.Euclidean(2)(np.array([1.0, 2.0])) * 1j
    with pytest.raises(ValueError, match="Hermitian"):
        pg.Euclidean(2, metric=np.array([[1.0, 1j], [1j, 1.0]]), field="C")


def test_identity_coordinates_of_a_complex_space():
    D = pg.Euclidean(3, covariance=np.array([1.0, 4.0, 0.5]), field="C")
    C = pg.port.Coordinates.identity(D)
    v = D.random(seed=3)
    c = pg.Euclidean(3, field="C").random(seed=4)
    assert np.allclose(C.numbers(v), v.values) and np.iscomplexobj(C.numbers(v))
    for A, x, y in ((C.export, v, c), (C.imp, c, v)):
        lhs, rhs = A.codomain.inner(A(x), y), A.domain.inner(x, A.adjoint(y))
        assert abs(lhs - rhs) < 1e-13 * abs(lhs)


# ------------------------------------------------------------------- a real model with complex data (nonlinear)
W_FREQ = np.logspace(-1, 1, 7)


def debye(x):
    """d(omega) = a / (1 + i omega tau) with x = (log a, log tau): real parameters, complex data."""
    a, tau = np.exp(x)
    return a / (1 + 1j * W_FREQ * tau)


def debye_jacobian(x):
    a, tau = np.exp(x)
    d = a / (1 + 1j * W_FREQ * tau)
    return np.c_[d, -d * (1j * W_FREQ * tau) / (1 + 1j * W_FREQ * tau)]          # d/dlog a, d/dlog tau


def test_debye_relaxation_through_the_port():
    M, D = pg.Euclidean(2), pg.Euclidean(len(W_FREQ), covariance=0.01, field="C")
    G = pg.port.External(M, D, debye, jacobian=debye_jacobian, name="debye")
    H = pg.port.External(M, D, debye, jvp=lambda x, v: debye_jacobian(x) @ v,
                    vjp=lambda x, w: debye_jacobian(x).conj().T @ w, name="debye-vjp")      # J^H w; the port takes Re
    theta = M(np.array([0.2, -0.4]))
    for F in (G, H):
        res = pg.port.taylor_test(F, theta, directions=3)
        assert all(r > 1.9 for r in res.rates) and res.adjoint_gap < 1e-12, res
    y = D.random(seed=5)
    assert np.allclose(G.derivative(theta).adjoint(y).values, H.derivative(theta).adjoint(y).values)
    truth = np.array([0.5, 0.3])
    fit = pg.optimisers.gauss_newton(G, D(debye(truth)), M(np.array([0.0, 0.0])), tol=1e-12, maxiter=50)
    assert fit.converged and np.allclose(fit.theta.values, truth, atol=1e-8)


# ------------------------------------------------------------------- a second use: Fourier-domain continuation
def test_fourier_continuation_matches_the_real_tikhonov_solution():
    """A real profile m (R^N) observed as the complex spectrum of its upward continuation, d = F_h m with
    (F_h)_kj = exp(-|k| h) exp(-2 pi i k j / N): least squares on the real model must use Re(F^H W F)."""
    N, h, alpha = 32, 0.08, 1e-3
    k = np.fft.fftfreq(N, d=1.0 / N)
    F = np.exp(-np.abs(k) * h)[:, None] * np.exp(-2j * np.pi * np.outer(k, np.arange(N)) / N) / np.sqrt(N)
    sigma = np.full(N, 0.05)
    M, D = pg.Euclidean(N), pg.Euclidean(N, covariance=sigma ** 2, field="C")
    G = pg.port.External(M, D, lambda m: F @ m, jacobian=lambda m: F, name="continuation")
    rng = np.random.default_rng(6)
    truth = np.sin(2 * np.pi * np.arange(N) / N) + 0.3 * rng.standard_normal(N)
    d = F @ truth + sigma / np.sqrt(2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))   # circular noise
    res = pg.port.taylor_test(G, M(truth), directions=3)
    assert res.adjoint_gap < 1e-12
    fit = pg.optimisers.gauss_newton(G, D(d), M.zero, alpha=alpha, tol=1e-12)
    Wd = 1 / sigma ** 2
    exact = np.linalg.solve(np.real(F.conj().T @ (Wd[:, None] * F)) + alpha * np.eye(N), np.real(F.conj().T @ (Wd * d)))
    assert np.allclose(fit.theta.values, exact, atol=1e-8)


# ------------------------------------------------------------------- complex function spaces (plan 020 P3b)
def test_complexified_function_space_conventions_and_values():
    L = pg.L2(pg.LebesgueMeasure(pg.Interval("[0, 1]")))
    X = pg.Complexified(L)
    z = X(lambda t: np.exp(2j * np.pi * t))
    w = X(lambda t: t + 1j * t ** 2)
    assert np.isclose(X.norm(z) ** 2, 1.0)                                       # |exp(2 pi i t)| = 1
    exact = -1 / (2 * np.pi) - (1 / (2 * np.pi) + 1 / (2 * np.pi ** 2)) * 1j     # int exp(2 pi i t) conj(t + i t^2) dt
    assert np.isclose(X.cinner(z, w), exact, atol=1e-12)
    a = 0.3 - 1.1j
    assert np.isclose(X.cinner(X.scale(a, z), w), a * X.cinner(z, w))
    assert np.isclose(X.cinner(z, X.scale(a, w)), np.conj(a) * X.cinner(z, w))
    assert np.isclose(X.norm(X.J(z)), X.norm(z)) and X.norm(X.J(X.J(z)) + z) < 1e-14
    assert np.isclose(X.values(z, 0.25), 1j, atol=1e-12)


def test_lift_of_a_real_operator_is_complex_linear_with_its_adjoint():
    H = pg.Sobolev(pg.Interval("[0, 1]"), 1, length_scale=0.2)
    X = pg.Complexified(H)
    A = pg.Functionals(H.riesz([np.array([0.2]), np.array([0.7])]))          # point values on H^1: R^2
    AC = X.lift(A, codomain=pg.Complexified(pg.Euclidean(2)))
    z = X(lambda t: np.cos(3 * t) + 1j * t)
    K = AC.codomain
    assert K.norm(AC(X.J(z)) - K.J(AC(z))) < 1e-13                            # A_C J = J A_C
    y = K.element({"re": pg.Euclidean(2)(np.array([1.0, -2.0])), "im": pg.Euclidean(2)(np.array([0.5, 3.0]))})
    lhs, rhs = K.inner(AC(z), y), X.inner(z, AC.adjoint(y))
    assert abs(lhs - rhs) < 1e-12 * abs(lhs)
    assert np.isclose(K.cinner(AC(z), y), X.cinner(z, AC.adjoint(y)))         # the complex adjoint too
    assert AC.claims.status("complex-linear") == (pg.Truth.TRUE, "theorem")


# ---------------------------------------------------------------------------- P3b: native complex cell values
def _cell_space(nx=10, ny=5):
    cells = pg.realisations.Cells.tensor(np.linspace(0, 1, nx + 1), np.linspace(0, 2, ny + 1))
    return pg.L2(pg.Box([(0, 1), (0, 2)]), realisation=cells), cells


def test_complex_cell_values_are_native_with_the_same_interface():
    from pygeoinf0.spaces.complex import CellsComplexified
    from pygeoinf0.realisations.piecewise_constant import ComplexCellsBackend
    M, cells = _cell_space()
    C = pg.Complexified(M)
    assert isinstance(C, CellsComplexified) and isinstance(C.backend, ComplexCellsBackend)
    assert type(pg.Complexified(pg.Sobolev(pg.Interval("[0, 1]"), 1, length_scale=0.2))) is pg.Complexified
    z = C(lambda x, y: np.exp(1j * (3 * x + y)) * (1 + x))
    w = C(lambda x, y: x - 2j * y)
    assert np.iscomplexobj(z.realisation.V) and z.realisation.V.shape == (cells.n, 1)
    vol, Z, W = cells.volumes, z.realisation.V[:, 0], w.realisation.V[:, 0]
    assert np.isclose(C.cinner(z, w), np.sum(vol * Z * W.conj()), rtol=1e-14)        # conjugate-linear in the second
    assert np.isclose(C.inner(z, w), C.cinner(z, w).real, rtol=1e-14)
    assert np.isclose(C.norm(z) ** 2, np.sum(vol * np.abs(Z) ** 2), rtol=1e-14)
    a = 0.3 - 1.1j
    assert np.isclose(C.cinner(a * z, w), a * C.cinner(z, w))
    assert np.isclose(C.cinner(z, a * w), np.conj(a) * C.cinner(z, w))
    assert np.isclose(C.norm(C.J(z)), C.norm(z)) and C.norm(C.J(C.J(z)) + z) < 1e-14
    u, v = C.parts(z)
    assert u.space == M and C.norm(C((u, v)) - z) < 1e-14 and C.norm(C({"re": u, "im": v}) - z) < 1e-14
    k = cells.locate(np.array([[0.55, 1.3]]))[0]
    assert C.values(z, 0.55, 1.3) == Z[k] and z(0.55, 1.3) == Z[k]                 # complex values at points
    assert C(2 - 1j).realisation.V[0, 0] == 2 - 1j


def test_complex_cell_values_agree_with_the_complexification_by_parts():
    """The native storage and the complexification H (+) H realise the same elements and inner products."""
    from pygeoinf0.spaces.direct_sum import DirectSum
    M, _ = _cell_space()
    C = pg.Complexified(M)
    P = DirectSum.__new__(pg.Complexified)
    pg.Complexified.__init__(P, M)                                                   # the by-parts version
    f, g = (lambda x, y: np.exp(1j * (3 * x + y))), (lambda x, y: x * y + 1j * np.cos(y))
    assert np.isclose(C.cinner(C(f), C(g)), P.cinner(P(f), P(g)), rtol=1e-13)
    assert np.isclose(C.norm(C(f)), P.norm(P(f)), rtol=1e-13)


def test_real_cell_space_refuses_complex_values():
    M, _ = _cell_space()
    with pytest.raises(pg.Refused, match="Complexified"):
        _ = M(lambda x, y: np.exp(1j * x)).realisation
    assert M(lambda x, y: (x + 0j) * y).realisation.V.dtype == float                 # zero imaginary parts are fine
    with pytest.raises(TypeError, match="Complexified"):
        _ = (1j) * M(lambda x, y: x)


def test_complex_cell_coordinates_and_the_complex_linearity_check():
    M, cells = _cell_space()
    C = pg.Complexified(M)
    E = pg.port.Coordinates.cells(C).export
    assert (isinstance(E.codomain, pg.spaces.CoordinateSpace) and E.codomain.field == "C") and E.codomain.dim == cells.n
    z, c = C.random(seed=1), E.codomain.random(seed=2)
    assert np.array_equal(E(z).values, z.realisation.V[:, 0])
    assert np.isclose(E.codomain.inner(E(z), c), C.inner(z, E.adjoint(c)), rtol=1e-13)     # the real adjoint
    assert np.isclose(E.codomain.cinner(E(z), c), C.cinner(z, E.adjoint(c)), rtol=1e-13)   # and the complex one
    r = pg.port.complex_linearity(E)
    assert r.holds and max(r.defects) < 1e-14 and r.claims.status("complex-linear") == (pg.Truth.TRUE, "checked")
    L = C.lift(pg.port.Coordinates.cells(M).export, codomain=pg.Complexified(pg.Euclidean(cells.n)))
    assert pg.port.complex_linearity(L).holds
    conj = pg.LinearOperator(C, C, lambda w: C.element((C.part(w, "re"), -1.0 * C.part(w, "im"))), name="conj")
    r = pg.port.complex_linearity(conj)                                                   # real-linear, not complex-linear
    assert not r.holds and np.allclose(r.defects, 2.0)                               # conj(i z) = -i conj(z)
    with pytest.raises(pg.Refused, match="not a complex space"):
        pg.port.complex_linearity(pg.port.Coordinates.cells(M).export)


# ---------------------------------------------------------------------------- P3c: circular measures, the port
def test_circular_gaussian_on_complex_euclidean_moments_and_claims():
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    S = A @ A.conj().T / n
    X = pg.Euclidean(n, weights=np.linspace(1, 2, n), field="C")
    mu = pg.probability.circular.circular(X, covariance=S)
    assert mu.claims.status("circular") == (pg.Truth.TRUE, "theorem")
    Z = mu.sample(40000, seed=1).values.T
    assert np.abs(Z @ Z.conj().T / Z.shape[1] - S).max() < 0.05 * np.abs(S).max()        # E z z^H = S
    assert np.abs(Z @ Z.T / Z.shape[1]).max() < 0.05 * np.abs(S).max()                    # E z z^T = 0
    u = X.random(seed=4)
    w = X.apply_metric(u.values)
    assert np.isclose(mu.variance(u), 0.5 * (w.conj() @ S @ w).real, rtol=1e-12)          # C_R = S W / 2
    from pygeoinf0.probability.circular import cvariance
    assert np.isclose(cvariance(mu, u), (w.conj() @ S @ w).real, rtol=1e-12)
    assert X.standard_gaussian().claims.status("circular")[0]                                       # default: S = W^-1
    with pytest.raises(pg.Refused, match="Hermitian"):
        pg.probability.circular.circular(X, covariance=np.triu(S))


def test_complex_matrix_pushforward_and_conditioning_against_dense_formulas():
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    S = A @ A.conj().T / n
    X = pg.Euclidean(n, weights=np.linspace(1, 2, n), field="C")
    Gm = rng.standard_normal((3, n)) + 1j * rng.standard_normal((3, n))
    D = pg.Euclidean(3, field="C")
    G = pg.ComplexMatrix(Gm, X, D, name="G")
    assert G.claims.status("complex-linear") == (pg.Truth.TRUE, "theorem") and pg.port.complex_linearity(G).holds
    x, y = X.random(seed=1), D.random(seed=2)
    assert np.isclose(D.cinner(G(x), y), X.cinner(x, G.adjoint(y)), rtol=1e-13)
    mu = pg.probability.circular.circular(X, covariance=S)
    assert G(mu).claims.status("circular") == (pg.Truth.TRUE, "derived")
    noise = pg.probability.circular.circular(D, covariance=0.1)
    d = D(np.array([1 + 1j, -2j, 0.5]))
    post = mu.condition(G, d, noise=noise)
    Sd = Gm @ S @ Gm.conj().T + 0.1 * np.eye(3)
    assert np.allclose(post.mean.values, S @ Gm.conj().T @ np.linalg.solve(Sd, d.values), atol=1e-13)
    P = S - S @ Gm.conj().T @ np.linalg.solve(Sd, Gm @ S)
    u = X.random(seed=4)
    w = X.apply_metric(u.values)
    assert np.isclose(post.variance(u), 0.5 * (w.conj() @ P @ w).real, rtol=1e-10)
    assert post.claims.status("circular") == (pg.Truth.TRUE, "derived")
    Zp = post.sample(40000, seed=2).values.T - post.mean.values[:, None]                   # Matheron samples
    assert np.abs(Zp @ Zp.conj().T / Zp.shape[1] - P).max() < 0.05 * np.abs(P).max()


def _cole_cole_problem():
    nx, nz = 24, 12
    cells = pg.realisations.Cells.tensor(np.linspace(0, 240, nx + 1), np.linspace(-120, 0, nz + 1))
    C = pg.Complexified(pg.L2(pg.Box([(0, 240), (-120, 0)]), realisation=cells))
    xc, vol = 0.5 * (cells.lo + cells.hi), cells.volumes

    def sigma(m, rho0=100.0, tau=0.01, c=0.5, omega=2 * np.pi):          # Pelton / Cole-Cole conductivity at 1 Hz
        return 1.0 / (rho0 * (1 - m * (1 - 1 / (1 + (1j * omega * tau) ** c))))
    body = ((xc[:, 0] - 120) ** 2 / 40 ** 2 + (xc[:, 1] + 50) ** 2 / 20 ** 2) < 1
    rx = np.linspace(10, 230, 23)
    K = np.vstack([np.exp(-((xc[None, :, 0] - rx[:, None]) ** 2 + (xc[None, :, 1] + z0) ** 2) / (2 * s ** 2))
                   for z0, s in ((0.0, 35.0), (60.0, 45.0))]) * vol / 1e3
    return C, cells, xc, body, sigma, K


def test_cole_cole_complex_conductivity_posterior_on_a_cell_mesh():
    """Plan 020 P3c's first use: the complex conductivity perturbation of a chargeable body (Cole-Cole), circular prior
    and noise, complex-linear data; the posterior mean equals the dense complex formula and carries the circularity."""
    C, cells, xc, body, sigma, K = _cole_cole_problem()
    truth_v = sigma(np.where(body, 0.3, 0.0)) - sigma(0.0)
    E = pg.port.Coordinates.cells(C).export
    assert E.claims.status("complex-linear") == (pg.Truth.TRUE, "theorem")
    G = pg.ComplexMatrix(K, E.codomain, name="K") @ E
    assert G.claims.status("complex-linear") == (pg.Truth.TRUE, "derived") and pg.port.complex_linearity(G).holds
    dist = np.linalg.norm(xc[:, None] - xc[None], axis=2)
    S = np.abs(truth_v).max() ** 2 * np.exp(-dist / 30.0)
    prior = pg.probability.circular.circular(C, covariance=S)
    s = 0.02 * np.abs(K @ truth_v).max()
    noise = pg.probability.circular.circular(G.codomain, covariance=s ** 2)
    truth = C(lambda x, y: 0j * x)
    truth.realisation.V[:, 0] = truth_v
    d = G(truth) + noise.sample(seed=3)
    post = prior.condition(G, d, noise=noise)
    ref = S @ K.T @ np.linalg.solve(K @ S @ K.T + s ** 2 * np.eye(len(K)), d.values)
    mv = post.mean.realisation.V[:, 0]
    assert np.abs(mv - ref).max() < 1e-8 * np.abs(ref).max()
    assert post.claims.status("circular") == (pg.Truth.TRUE, "derived")
    phase = np.angle(sigma(0.0) + mv)
    assert phase[body].mean() > 3 * abs(phase[~body].mean())                        # the IP phase anomaly is found
    avg = C(lambda x, y: 0j * x)
    avg.realisation.V[:, 0] = np.where(body, 1.0, 0.0) / cells.volumes[body].sum()
    assert post.std(avg) < 0.6 * prior.std(avg)


def test_holomorphic_external_on_complex_cells_through_the_port():
    """Complex log-conductivity v (sigma* = exp v): jvp J dv, vjp J^H w; Taylor rate 2 and the adjoint identity on
    complex directions, a complex-linear derivative, a J^T vjp caught, Gauss-Newton converging on complex unknowns."""
    cells = pg.realisations.Cells.tensor(np.linspace(0, 4, 5), np.linspace(0, 2, 3))
    C = pg.Complexified(pg.L2(pg.Box([(0, 4), (0, 2)]), realisation=cells))
    rng = np.random.default_rng(1)
    K = rng.standard_normal((30, cells.n)) / 4
    D, cc = pg.Euclidean(30, field="C"), pg.port.Coordinates.cells(C)
    F = pg.port.External(C, D, lambda v: K @ np.exp(v), coords=(cc, None), jvp=lambda v, dv: K @ (np.exp(v) * dv),
                    vjp=lambda v, w: np.conj(np.exp(v)) * (K.T @ w), name="F")
    vt = cc.element(rng.standard_normal(cells.n) * 0.5 + 1j * rng.standard_normal(cells.n) * 0.3)
    t = pg.port.taylor_test(F, vt)
    assert all(abs(r - 2) < 0.1 for r in t.rates) and t.adjoint_gap < 1e-12
    assert pg.port.complex_linearity(F.derivative(vt)).holds
    wrong = pg.port.External(C, D, F.g, coords=(cc, None), jvp=lambda v, dv: K @ (np.exp(v) * dv),
                        vjp=lambda v, w: np.exp(v) * (K.T @ w), name="F (J^T)")
    assert pg.port.taylor_test(wrong, vt).claims.status("adjoint")[0] is pg.Truth.FALSE
    fit = pg.optimisers.gauss_newton(F, F(vt), cc.element(np.zeros(cells.n, complex)))
    assert fit.converged and np.abs(cc.numbers(fit.theta) - cc.numbers(vt)).max() < 1e-6
