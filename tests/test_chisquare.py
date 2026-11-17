"""Plan 009 P3, G7: weighted chi-square laws with proved brackets, hardening of Gaussian measures, the Hardened
calibration (EXP-030 DESIGN G7, NOTES.md section 7). References outside the package: scipy.stats.chi2, the
hypoexponential closed form, a convolution integral by scipy.integrate.quad, Monte Carlo draws, a long spectrum."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2

import pygeoinf0 as pg
from pygeoinf0.core.errors import Refused


def test_name_is_public():
    assert "WeightedChiSquare" in pg.__all__


def test_equal_weights_against_scipy():
    Q = pg.WeightedChiSquare([2.0, 2.0, 2.0], [1, 2, 2])                 # merged: 2 chi2_5
    assert len(Q.weights) == 1 and Q.dof == 5
    for x in (0.3, 3.0, 10.0, 30.0, 60.0):
        F = Q.cdf(x, method="davies")                                     # the series, not the closed form
        assert Q.cdf(x).contains(chi2.cdf(x / 2, 5))
        ref = chi2.cdf(x / 2, 5)
        assert abs(float(F) - ref) < 1e-12 and F.contains(ref) and F.status == "derived"
        assert F.width < 2e-12
    q = Q.quantile(0.95, method="davies")
    assert q.contains(2 * chi2.ppf(0.95, 5)) and q.width < 1e-9


def test_two_weights_of_multiplicity_two_against_the_hypoexponential_law():
    l1, l2 = 1.0, 3.0
    Q = pg.WeightedChiSquare([l1, l2], [2, 2])
    for x in (0.5, 1.0, 5.0, 20.0, 50.0):
        ref = 1 - (l1 * math.exp(-x / (2 * l1)) - l2 * math.exp(-x / (2 * l2))) / (l1 - l2)
        F = Q.cdf(x)
        assert F.contains(ref) and abs(float(F) - ref) < 1e-12


def test_two_weights_against_the_convolution_integral():
    l1, l2, x = 1.0, 0.5, 2.0
    f1 = lambda y: chi2.pdf(y / l1, 1) / l1                              # noqa: E731
    # P(l1 Z1^2 + l2 Z2^2 <= x) = int_0^x f1(y) F2(x - y) dy; y = s^2 removes the y^(-1/2) singularity
    ref = quad(lambda s: 2 * s * f1(s * s) * chi2.cdf((x - s * s) / l2, 1), 0, math.sqrt(x), epsabs=1e-14,
               epsrel=1e-13, limit=400)[0]
    Q = pg.WeightedChiSquare([l1, l2])
    F = Q.cdf(x, method="davies")
    assert F.lower - 1e-10 <= ref <= F.upper + 1e-10
    assert F.width < 2e-6                                                # r = 2: slow series, capped, stated
    P = Q.cdf(x)                                                         # auto: the polar formula
    assert abs(float(P) - ref) < 1e-12 and P.status == "checked"
    I = Q.cdf(x, method="imhof")
    assert I.lower - 1e-10 <= ref <= I.upper + 1e-10 and I.status == "checked" and abs(float(I) - ref) < 1e-6


def test_monte_carlo():
    lam = np.array([3.0, 1.0, 0.5, 0.25, 0.1])
    Q = pg.WeightedChiSquare(lam)
    rng = np.random.default_rng(7)
    n = 10_000_000
    x = 6.0
    hits = 0
    for _ in range(10):
        Z = rng.standard_normal((n // 10, 5))
        hits += int(np.count_nonzero((Z ** 2) @ lam <= x))
    p = hits / n
    F = Q.cdf(x)
    assert abs(p - float(F)) < 5 * math.sqrt(p * (1 - p) / n)
    assert F.width < 1e-11


def test_truncation_bracket_contains_the_full_spectrum_value():
    lam = 1.0 / np.arange(1, 20001) ** 2
    full = pg.WeightedChiSquare(lam[:3000], tail=1 / 3000, tail_max=lam[3000])
    known = pg.WeightedChiSquare(lam[:50], tail=float(lam[50:].sum()) + 1 / 20000, tail_max=lam[50])
    for x in (0.8, 2.0, 4.0):
        Ff, Fk = full.cdf(x), known.cdf(x)
        assert Fk.lower <= Ff.lower and Ff.upper <= Fk.upper + 1e-12
        assert Fk.upper - Fk.lower < 0.05
    qf, qk = full.quantile(0.95), known.quantile(0.95)
    assert qk.lower <= qf.lower and qf.upper <= qk.upper
    assert qk.status == "derived" and qk.upper - qk.lower < 0.05 * float(qk)
    markov = pg.WeightedChiSquare(lam[:50], tail=float(lam[50:].sum()) + 1 / 20000)
    assert markov.quantile(0.95).upper >= qk.upper                       # the Chernoff tail is sharper than Markov


def test_saddlepoint_is_labelled_and_close():
    lam = 1.0 / np.arange(1, 301) ** 1.5
    Q = pg.WeightedChiSquare(lam, np.arange(1, 301) * 2 + 1)
    d, s = Q.quantile(0.95), Q.quantile(0.95, method="saddlepoint")
    assert s.status == "asserted" and s.upper == math.inf
    assert d.status == "derived" and d.width < 1e-8 * float(d)
    assert abs(float(s) - float(d)) < 5e-3 * float(d)                    # the documented difference: small, not bounded


def test_hardening_in_euclidean_space_is_an_ellipsoid_with_its_mass():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    S = A @ A.T + 0.1 * np.eye(5)
    X = pg.Euclidean(5)
    mu = pg.GaussianMeasure(X, pg.Matrix(S))
    E = mu.harden(pg.Identity(X), 0.95)
    assert isinstance(E, pg.Ellipsoid)
    ref = pg.WeightedChiSquare(np.linalg.eigvalsh(S)).quantile(0.95)
    assert E.threshold == pytest.approx(ref.upper, rel=1e-12)
    assert E.mass.lower >= 0.95 - 1e-15 and E.claims.find("mass").status == "derived"
    Z = rng.multivariate_normal(np.zeros(5), S, size=200_000)
    inside = np.mean(np.sum(Z * Z, axis=1) <= E.threshold)
    assert abs(inside - 0.95) < 5 * math.sqrt(0.95 * 0.05 / 200_000)
    B = pg.Matrix(rng.normal(size=(2, 5)))                                 # a form into R^2: rank 2
    H = mu.harden(B, 0.9)
    assert H.threshold == pytest.approx(pg.WeightedChiSquare(np.linalg.eigvalsh(B.matrix() @ S @ B.matrix().T))
                                        .quantile(0.9).upper, rel=1e-10)


def test_hardening_on_a_function_space_with_an_omitted_part():
    M = pg.L2("[0, 1]")
    N = 30
    n = np.arange(1, N + 1)
    phi = M.family(lambda x, k: np.sqrt(2) * np.cos(k * np.pi * x), k=n)
    C = pg.Covariance.spectral(1.0 / n ** 2, phi, tail=1.0 / N, tail_max=1.0 / (N + 1) ** 2,
                               reason="sum_{n > N} n^-2 <= 1/N")
    mu = pg.GaussianMeasure(M, C)
    J = pg.Multiplication(M, pg.Tabulated("[0, 1]", [0, 1], [1.0, 2.0]))
    S = mu.harden(J, 0.9)
    law, rest, st = pg.probability.hardening.law_of_form(mu, J)
    assert rest == pytest.approx(4.0 / N) and st == "derived"
    assert S.threshold > law.quantile(0.9).upper                         # the omitted part costs a larger threshold
    assert S.claims.find("mass").status == "derived"
    m = mu.sample(seed=3)
    assert S.member(m) == pg.Truth.of(S.value(m) <= S.threshold)
    exact = pg.Covariance.spectral(1.0 / n ** 2, phi)                    # no omitted part: the known law exactly
    S0 = pg.GaussianMeasure(M, exact).harden(J, 0.9)
    assert S0.threshold == pytest.approx(law.quantile(0.9).upper, rel=1e-12)
    assert S0.mass.contains(0.9) or S0.mass.lower >= 0.9 - 1e-12


def test_cameron_martin_ball_is_refused():
    M = pg.L2("[0, 1]")
    n = np.arange(1, 11)
    phi = M.family(lambda x, k: np.sqrt(2) * np.cos(k * np.pi * x), k=n)
    C = pg.Covariance.spectral(1.0 / n ** 2, phi, tail=0.1)
    mu = pg.GaussianMeasure(M, C)
    with pytest.raises(Refused):
        mu.harden(pg.Inverse(C, pg.CG()), 0.9, kind="quadratic")


def test_hardened_calibration_and_its_coverage_by_monte_carlo():
    cal = pg.inference.Hardened(alpha=0.1, beta=0.05)
    assert cal.coverage() == pytest.approx(dict(union=0.85, joint=0.9 * 0.95))
    rng = np.random.default_rng(5)
    n, T = 3, 40_000
    S = np.diag([2.0, 1.0, 0.5])
    X = pg.Euclidean(n)
    mu = pg.GaussianMeasure(X, pg.Matrix(S))
    prior_set = mu.harden(pg.Identity(X), 0.95)
    m = rng.multivariate_normal(np.zeros(n), S, size=T)
    d = m + rng.standard_normal((T, n))                                   # noise N(0, I)
    r2 = chi2.ppf(0.9, n)                                                 # acceptance region of level 0.9 at each m
    in_prior = np.sum(m * m, axis=1) <= prior_set.threshold
    in_conf = np.sum((d - m) ** 2, axis=1) <= r2
    cover = np.mean(in_prior & in_conf)
    assert cover >= cal.coverage()["joint"] - 5 * math.sqrt(0.25 / T)
    assert cover >= cal.coverage()["union"]
    claims = cal.check(pg.Problem(M=X, prior=prior_set))
    assert claims.find("coverage").status == "theorem"
    assert claims.find("prior mass").status == "derived"
