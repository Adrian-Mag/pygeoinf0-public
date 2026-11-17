"""Plan 054: a posterior's moments carry the stated errors of the representers they are built from (probability/
input_error.py). On the unit sphere the representers of point evaluations in H^2 are realised to the harmonic degree cap
and state errors of about 1e-2 (Family.errors()); before, the posterior's variances and mean said nothing of them (the
posterior sd off by 1.3e-3 while mean_error said 4e-15, no warning; /disks/data/tmp/pygeoinf0-vs-pygeoinf2.md, B2').
Reference: the exact zonal kernel summed to degree 60000 with its tail at the diagonal (mpmath not needed: the terms
decay like l^-3 and the sum is checked against its tail)."""
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import PrecisionWarning
from pygeoinf0.probability import input_error

B, SIGMA = 0.09, 0.1           # length_scale 0.3: (1 + b l(l+1))^-s with b = 0.3^2


def _zonal(x, s, L=60000):
    """sum_l (2l+1)/(4 pi) (1 + b l(l+1))^-s P_l(x), with the tail at x = 1 added (its asymptote l^(1-2s))."""
    x = np.asarray(x, float)
    p0, p1 = np.ones_like(x), x.copy()
    tot = p0 / (4 * np.pi) + 3.0 / (4 * np.pi) * (1 + 2 * B) ** -s * p1
    for deg in range(1, L):
        p0, p1 = p1, ((2 * deg + 1) * x * p1 - deg * p0) / (deg + 1)
        n = deg + 1
        tot = tot + (2 * n + 1) / (4 * np.pi) * (1 + B * n * (n + 1)) ** -s * p1
    tot[np.isclose(x, 1.0)] += 2 / (4 * np.pi) * B ** -s * L ** (2 - 2 * s) / (2 * s - 2)
    return tot


def _points(k, seed):
    v = np.random.default_rng(seed).standard_normal((k, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


@pytest.fixture(scope="module")
def case():
    obs, test = _points(8, 5), _points(5, 6)
    d = 0.5 * obs[:, 2] + 0.2 + SIGMA * np.random.default_rng(2).standard_normal(len(obs))
    K = _zonal(np.clip(obs @ obs.T, -1, 1), 2)
    Kt = _zonal(np.clip(test @ obs.T, -1, 1), 2)
    k0 = _zonal(np.array([1.0]), 2)[0]
    S = K + SIGMA ** 2 * np.eye(len(obs))
    var = k0 - np.einsum("ij,ji->i", Kt, np.linalg.solve(S, Kt.T))
    return obs, d, test, var, k0


def _posterior(obs, d):
    H = pg.Sobolev(pg.Sphere(1.0), 2, length_scale=0.3)
    G = pg.Functionals(H.riesz(obs))
    D = G.codomain
    prior = pg.probability.cylindrical(H)
    return H, prior, prior.condition(G, D(d), noise=pg.GaussianMeasure(D, covariance=SIGMA ** 2 * pg.Identity(D)))


def test_posterior_variances_state_the_representers_errors(case):
    obs, d, test, var, _ = case
    with pytest.warns(PrecisionWarning, match="stated input error"):
        H, prior, post = _posterior(obs, d)
        v = post.variance(test)
    err = post.variance_error
    true = np.abs(v - var)
    assert np.all(true > 1e-9 * var)                       # the representers' truncation shows: not to 9 digits
    assert np.all(true <= err)                             # and the stated error holds it
    assert post.mean_error > 1e-9 * float(H.norm(post.mean))


def test_prior_variance_states_the_elements_errors(case):
    _, _, test, _, k0 = case
    mu = pg.probability.cylindrical(pg.Sobolev(pg.Sphere(1.0), 2, length_scale=0.3))
    with pytest.warns(PrecisionWarning, match="element's representers"):
        v = mu.variance(test[:2])
    true = np.abs(v - k0)
    assert np.all(true > 1e-9 * k0) and np.all(true <= mu.variance_error)


def test_control_without_the_statement_the_error_is_silent(case, monkeypatch):
    """Control: no input error stated (the former code) gives the same numbers with no warning and no error stated, while
    they are not right to 9 digits."""
    obs, d, test, var, _ = case
    monkeypatch.setattr(input_error, "operator_error", lambda G: None)
    monkeypatch.setattr(input_error, "element_errors", lambda u: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        H, prior, post = _posterior(obs, d)
        v = post.variance(test)
    assert post.variance_error is None and post.mean_error < 1e-12 * float(H.norm(post.mean))
    assert np.max(np.abs(v - var) / var) > 1e-6


def test_exact_representers_cost_and_say_nothing():
    """Exact representers (the circle's closed forms state errors 0): no statement, no warning."""
    H = pg.Sobolev(pg.Circle(1.0), 1, length_scale=0.5)
    t = np.array([0.3, 1.0, 2.0, 4.0])
    k = H.riesz(np.c_[np.cos(t), np.sin(t)])
    assert not np.any(k.errors())
    G = pg.Functionals(k)
    D = G.codomain
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        post = pg.probability.cylindrical(H).condition(G, D(np.sin(t)), noise=pg.GaussianMeasure(D, covariance=0.01 * pg.Identity(D)))
        post.variance(np.c_[np.cos([0.5, 2.5]), np.sin([0.5, 2.5])])
    assert post._input is None and post.variance_error is None


@pytest.mark.parametrize("planted", [False, True])
def test_the_forward_operators_error_reaches_an_exact_elements_variance(case, monkeypatch, planted):
    """u = 1 (realised to rounding): Var <x, u> = 4 pi - 1^T S^-1 1 exactly, wrong only through G's truncated kernels
    (1e-4 relative). Its stated error holds it. Control (planted): without the operator's statement (the former code) the
    variance states at most the element's rounding while it is wrong beyond 9 digits."""
    obs, d, _, _, _ = case
    if planted:
        monkeypatch.setattr(input_error, "operator_error", lambda G: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        H, prior, post = _posterior(obs, d)
        u = H(lambda x, y, z: 0.0 * x + 1.0, resolution=0.5)
        v = post.variance(u)
    K = _zonal(np.clip(obs @ obs.T, -1, 1), 2)
    one = np.ones(len(obs))
    true = abs(v - (4 * np.pi - one @ np.linalg.solve(K + SIGMA ** 2 * np.eye(len(obs)), one)))
    assert true > 1e-9 * v
    stated = post.variance_error or 0.0
    assert (stated < true) if planted else (true <= stated)


@pytest.mark.parametrize("planted", [False, True])
def test_small_noise_the_operators_error_is_judged_with_the_priors_c(monkeypatch, planted):
    """u = 1 at small noise (sigma = 1e-3): the variance is wrong by about 7e-4 through G's truncated kernels whatever the
    noise. d<C_post u, u> = -2 <dG C_post u, y_u> with y_u = S^-1 G C u, C the PRIOR's. Control (planted): y_u from the
    posterior's C (S^-1 G C_post u = S^-1 C_e y_u, the former code) is smaller by about sigma^2 and the statement
    understates (3e-7 against 7e-4)."""
    sigma = 1e-3
    obs = _points(8, 5)
    d = 0.5 * obs[:, 2] + 0.2 + sigma * np.random.default_rng(2).standard_normal(len(obs))
    H = pg.Sobolev(pg.Sphere(1.0), 2, length_scale=0.3)
    G = pg.Functionals(H.riesz(obs))
    D = G.codomain
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        post = pg.probability.cylindrical(H).condition(G, D(d), noise=pg.GaussianMeasure(D, covariance=sigma ** 2 * pg.Identity(D)))
        if planted:
            post._input.C = post.covariance
        u = H(lambda x, y, z: 0.0 * x + 1.0, resolution=0.5)
        v = post.variance(u)
    K = _zonal(np.clip(obs @ obs.T, -1, 1), 2)
    one = np.ones(len(obs))
    true = abs(v - (4 * np.pi - one @ np.linalg.solve(K + sigma ** 2 * np.eye(len(obs)), one)))
    assert true > 1e-4
    assert (post.variance_error < true) if planted else (true <= post.variance_error)
