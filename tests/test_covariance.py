"""Plan 009 P3, G6: multiplication operators and covariance algebra (EXP-030 DESIGN G6, NOTES.md section 6).
References outside the package: scipy.integrate.quad, dense NumPy traces, explicit series, the tensor realisation of L2 of
a ball as an independent route for products."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

import pygeoinf0 as pg


def q(f, a, b):
    return quad(f, a, b, epsabs=1e-14, epsrel=1e-13, limit=400)[0]


def test_names_are_public():
    assert {"Multiplication", "Covariance"} <= set(pg.__all__)


def test_multiplication_by_a_function_on_an_interval():
    M = pg.L2("[0, 2]")
    a = lambda x: 1 + 0.5 * np.sin(3 * x)                               # noqa: E731
    A = pg.Multiplication(M, M(a))
    f, g = M(np.cos), M(lambda x: np.exp(-x))
    ref = q(lambda x: a(x) * np.cos(x) * np.exp(-x), 0, 2)
    assert M.inner(A(f), g) == pytest.approx(ref, rel=1e-12)
    assert M.inner(f, A.adjoint(g)) == pytest.approx(ref, rel=1e-12)    # self-adjoint: the adjoint identity
    assert A.adjoint is A
    assert A.norm_bound() >= 1.5 - 1e-12                                # the sup of |a| is 1.5
    assert A.claims.find("bounded").status == "derived"
    for k in (1, 5, 20):                                                # ||A f|| / ||f|| <= the bound
        h = M(lambda x, k=k: np.cos(k * x) * np.exp(-30 * (x - 0.52) ** 2))
        assert A(h).norm() <= A.norm_bound() * h.norm()


def test_multiplication_by_a_table_has_the_exact_norm_and_positivity():
    M = pg.L2("[0, 2]")
    t = pg.Tabulated("[0, 2]", [0, 1, 2], [1.0, -3.0, 2.0])
    B = pg.Multiplication(M, t)
    assert B.norm_bound() == 3.0 and B.claims.find("bounded").status == "derived"
    assert B.positivity()[0] is None                                     # a changes sign: no positivity claim
    f = M(lambda x: x ** 2)
    assert M.inner(B(f), f) == pytest.approx(q(lambda x: t(x) * x ** 4, 0, 1) + q(lambda x: t(x) * x ** 4, 1, 2),
                                             rel=1e-12)
    P = pg.Multiplication(M, pg.Tabulated("[0, 2]", [0, 2], [0.5, 2.0]))
    assert P.positivity()[0] == "psd"
    # a narrow bump at the peak of |t| (x = 1) nearly attains the norm
    h = M(lambda x: np.exp(-((x - 1) / 0.01) ** 2), breaks=[1.0])
    assert B(h).norm() / h.norm() > 0.98 * 3.0


def test_callable_multipliers_need_a_declared_bound():
    M = pg.L2("[0, 1]")
    with pytest.raises(ValueError):
        pg.Multiplication(M, np.exp)
    C = pg.Multiplication(M, lambda x: np.exp(-x), bound=1.0)
    assert C.claims.find("bounded").status == "asserted"
    with pytest.raises(pg.errors.Unsupported):
        pg.Multiplication(pg.Sobolev("[0, 1]", 1, length_scale=0.1), np.exp, bound=3.0)


def test_radial_multiplication_on_a_ball_against_the_tensor_realisation():
    B = pg.Ball(1.0)
    X = pg.L2(B, realisation="radial_harmonics")
    T = pg.L2(B)                                                         # tensor realisation: independent route
    t = pg.Tabulated("[0, 1]", [0, 0.5, 1], [2.0, 1.0, 0.5])
    J = pg.Multiplication(X, t)
    u = lambda x, y, z: x + y * z                                        # noqa: E731
    v = lambda x, y, z: 1 + x * x                                        # noqa: E731
    w = lambda x, y, z: t(np.sqrt(x * x + y * y + z * z)) * u(x, y, z)   # noqa: E731
    lhs = X.inner(J(X(u)), X(v))
    rhs = T.inner(T(w, breaks=([0.5], None, None)), T(v))
    assert lhs == pytest.approx(rhs, rel=1e-9)
    c = J.claims.find("rotation invariant")
    assert c.holds and c.status == "derived"
    p = np.array([0.3, 0.2, -0.4])
    assert J(X(u))(p) == pytest.approx(t(np.linalg.norm(p)) * u(*p), abs=1e-12)


def test_non_radial_multipliers_are_not_given_the_rotation_claim():
    X = pg.L2(pg.Ball(1.0), realisation="radial_harmonics")
    A = pg.Multiplication(X, X(lambda x, y, z: 1 + x))
    c = A.claims.find("rotation invariant")
    assert c is not None and c.holds is False
    assert A.norm_bound() >= 2.0 - 1e-12
    R = pg.Multiplication(X, X(lambda x, y, z: 1 + x * x + y * y + z * z))
    assert R.claims.find("rotation invariant").holds
    assert pg.Multiplication(pg.L2("[0, 1]"), pg.L2("[0, 1]")(np.exp)).claims.find("rotation invariant") is None


def _cosine_covariance(N=40):
    M = pg.L2("[0, 1]")
    n = np.arange(1, N + 1)
    phi = M.family(lambda x, k: np.sqrt(2) * np.cos(k * np.pi * x), k=n)
    C = pg.Covariance.spectral(1.0 / n ** 2, phi, tail=1.0 / N, tail_max=1.0 / (N + 1) ** 2,
                               reason="sum_{n > N} n^-2 <= 1/N")
    return M, C


def test_spectral_trace_bracket_contains_the_series():
    _, C = _cosine_covariance()
    T = C.trace()
    assert T.contains(math.pi ** 2 / 6) and T.status == "derived"
    assert T.lower == pytest.approx(sum(1.0 / k ** 2 for k in range(1, 41)), rel=1e-14)


def test_congruence_trace_against_a_long_series():
    M, C = _cosine_covariance()
    a = pg.Tabulated("[0, 1]", [0, 1], [1.0, 2.0])
    A = pg.Multiplication(M, a)
    D = C.congruence(A)
    # tr(A C A*) = sum_n n^-2 int a^2 2 cos^2(n pi x) dx, independent reference by quadrature over 4000 terms + tail
    ints = [q(lambda x, k=k: (1 + x) ** 2 * 2 * np.cos(k * np.pi * x) ** 2, 0, 1) for k in range(1, 41)]
    known = sum(i / k ** 2 for i, k in zip(ints, range(1, 41)))
    assert D.trace().lower == pytest.approx(known, rel=1e-11)
    full = known + sum(7 / 3 / k ** 2 for k in range(41, 200000))      # int (1 + x)^2 2 cos^2 -> 7/3 for large n
    assert D.trace().contains(full)
    assert D.trace().upper <= known + 4.0 / 40 * (1 + 1e-12)             # ||A||^2 = 4 times the tail bound
    assert D.positivity()[0] == "psd" and D.is_selfadjoint()[0]


def test_congruence_and_trace_in_euclidean_space_against_numpy():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(6, 6))
    S = R @ R.T
    B = rng.normal(size=(4, 6))
    C = pg.Covariance(pg.Matrix(S))
    assert float(C.trace()) == pytest.approx(np.trace(S), rel=1e-13)
    assert float(C.congruence(pg.Matrix(B)).trace()) == pytest.approx(np.trace(B @ S @ B.T), rel=1e-13)


def test_normalised_carries_the_trace_bracket():
    _, C = _cosine_covariance()
    Cn = C.normalised(2.0)
    assert float(Cn.trace()) == pytest.approx(2.0)
    rel = Cn.scale_bracket
    exact_scale = 2.0 / (math.pi ** 2 / 6)
    used = 2.0 / float(C.trace())
    assert rel.contains(exact_scale / used)
    assert Cn.claims.find("normalisation") is not None
    Cl = C.normalised(1.0, at="lower")
    assert float(Cl.trace()) > 1.0 - 1e-15 and Cl.trace().lower == pytest.approx(1.0)


def test_variance_bracket_and_gaussian_measure_with_a_truncated_covariance():
    M, C = _cosine_covariance()
    u = M(lambda x: x)
    V = C.variance(u)
    # <C u, u> = sum_n n^-2 (int sqrt2 cos(n pi x) x dx)^2 = sum over odd n of 8 / (n^6 pi^4)
    exact = sum(8 / (k ** 6 * math.pi ** 4) for k in range(1, 20001, 2))
    assert V.contains(exact)
    mu = pg.GaussianMeasure(M, C)
    assert any(c.key == "truncation" for c in mu.claims)
    assert len(mu.sample(4, seed=0)) == 4
    nu = mu.pushforward(pg.Multiplication(M, pg.Tabulated("[0, 1]", [0, 1], [1.0, 2.0])))
    assert isinstance(nu.covariance, pg.Covariance) and nu.covariance.trace().upper < math.inf


def test_compression_by_a_band_projection_and_the_rotation_claim():
    X = pg.L2(pg.Ball(1.0), realisation="radial_harmonics")
    vol = 4 * math.pi / 3
    phi = X.family([lambda x, y, z: np.ones_like(x) / math.sqrt(vol),
                    lambda x, y, z: z / math.sqrt(4 * math.pi / 15)])
    C = pg.Covariance.spectral([3.0, 2.0], phi)
    assert float(C.trace()) == pytest.approx(5.0, rel=1e-12)
    P = pg.BandProjection(X, [0])
    R = C.compressed(P)
    assert float(R.trace()) == pytest.approx(3.0, rel=1e-10)
    assert R.claims.find("compressed") is not None
    A = pg.Covariance(pg.operators.band.AngularFunction(X, lambda l: (1.0 + l) ** -2.0))
    assert A.claims.find("rotation invariant").holds
    J = pg.Multiplication(X, pg.Tabulated("[0, 1]", [0, 1], [1.0, 0.5]))
    assert A.congruence(J).claims.find("rotation invariant").holds
    K = pg.Multiplication(X, X(lambda x, y, z: 1 + x))
    assert A.congruence(K).claims.find("rotation invariant") is None
    with pytest.raises(ValueError):
        C.compressed(J)                                                  # not an orthogonal projection
