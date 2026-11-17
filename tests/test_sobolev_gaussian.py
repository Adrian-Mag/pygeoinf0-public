"""The Gaussian with Cameron-Martin space H^s: exact prior and posterior variances of ray and point functionals."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg


def test_prior_and_posterior_variances_are_energies():
    D = pg.Disk(1.0)
    H = pg.Sobolev(D, 1.5, length_scale=0.2)
    rays = [pg.Segment((-1, 0), (1, 0)), pg.Segment((0, -1), (0, 1)), pg.Segment((-0.6, -0.8), (0.6, 0.8))]
    K = H.riesz(rays)
    E = H.inner(K, K)
    prior = pg.probability.cylindrical(H)
    assert prior.variance(K[0]) == pytest.approx(E[0, 0], rel=1e-13)
    sigma = 0.05
    Dd = pg.Euclidean(3, covariance=sigma ** 2)
    G = pg.Functionals(K, into=Dd)
    d = Dd(np.array([0.3, -0.1, 0.2]))
    post = prior.condition(G, d)
    p = H.riesz((0.2, 0.1))                                            # point value at (0.2, 0.1): s > d/2
    b = H.inner(K, p)
    var_ref = H.norm(p) ** 2 - b @ np.linalg.solve(E + sigma ** 2 * np.eye(3), b)
    assert post.variance(p) == pytest.approx(var_ref, rel=1e-10)
    mean_ref = b @ np.linalg.solve(E + sigma ** 2 * np.eye(3), d.values)
    assert post.mean(0.2, 0.1) == pytest.approx(mean_ref, rel=1e-10)
    with pytest.raises(pg.errors.NotDefined, match="cylindrical"):
        prior.sample()


def test_discrepancy_principle():
    H = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.2)
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, (30, 2))
    rays = [pg.Segment((math.cos(a), math.sin(a)), (math.cos(b), math.sin(b))) for a, b in ang]
    sigma = 0.01
    i = np.arange(30)
    C = sigma ** 2 * np.exp(-np.abs(i[:, None] - i[None]) / 2.0)          # correlated noise
    Dd = pg.Euclidean(30, covariance=C)
    G = pg.Functionals(H.riesz(rays), into=Dd)
    d = G(H(lambda x, y: np.exp(-((x - 0.2) ** 2 + y ** 2) / 0.1))) + Dd.standard_gaussian().sample(seed=2)
    A = G @ G.adjoint
    a = pg.solvers.damping_for_misfit(A, d, d.space.dim)
    m = G.adjoint((A + a * pg.Identity(Dd)).inv(d))
    assert (G(m) - d).norm() ** 2 == pytest.approx(30, rel=1e-6)
    assert np.all(G.kernels.errors() >= 0)


def test_scaled_prior_matches_damped_least_squares_and_vectorised_std():
    H = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.2)
    rays = [pg.Segment((-1, 0), (1, 0)), pg.Segment((0, -1), (0, 1)), pg.Segment((-0.6, -0.8), (0.6, 0.8))]
    sigma, alpha = 0.05, 0.3
    Dd = pg.Euclidean(3, covariance=sigma ** 2)
    G = pg.Functionals(H.riesz(rays), into=Dd)
    d = Dd(np.array([0.3, -0.1, 0.2]))
    post = pg.probability.cylindrical(H, alpha=alpha).condition(G, d)
    m = G.adjoint((G @ G.adjoint + alpha * pg.Identity(Dd)).inv(d))
    pts = np.array([[0.2, 0.1], [-0.5, 0.3], [0.0, 0.0]])
    assert np.allclose([post.mean(*p) for p in pts], [m(*p) for p in pts], rtol=1e-10)
    K, Pk = G.kernels, H.riesz(pts)
    E, B = H.inner(K, K), H.inner(K, Pk)
    var_ref = (H.norm(Pk) ** 2 - np.sum(B * np.linalg.solve(E + alpha * sigma ** 2 * np.eye(3), B), axis=0)) / alpha
    s = post.std(pts)
    assert s.shape == (3,) and np.allclose(s ** 2, var_ref, rtol=1e-10)
    assert post.std(pts[0]) == pytest.approx(math.sqrt(var_ref[0]), rel=1e-10)
    assert np.allclose(post.variance(rays), post.variance(K), rtol=1e-14)
    assert pg.probability.cylindrical(H, alpha=4.0).variance(Pk[0]) == pytest.approx(H.norm(Pk[0]) ** 2 / 4, rel=1e-13)
    with pytest.raises(pg.errors.NotDefined, match="cylindrical"):
        pg.probability.cylindrical(H, alpha=2.0).sample()
    assert G(pg.probability.cylindrical(H, alpha=2.0)).sample(seed=0).space == Dd          # its push-forward to R^n is sampleable


def test_review_edge_cases():
    D = pg.Disk(1.0)
    H = pg.Sobolev(D, 1.5, length_scale=0.2)
    for bad in (0.0, -1.0, np.inf, 1e-320):
        with pytest.raises(ValueError):
            pg.probability.cylindrical(H, alpha=bad)
    Dd = pg.Euclidean(2, covariance=1e-4)
    G = pg.Functionals(H.riesz([pg.Segment((-1, 0), (1, 0)), pg.Segment((0, -1), (0, 1))]), into=Dd)
    post = pg.probability.cylindrical(H, alpha=2.0).condition(G, Dd(np.array([0.1, 0.2])))
    assert post.std(np.zeros((0, 2))).shape == (0,)
    X = G(post).sample(200, seed=0)                    # the posterior's push-forward to R^n has samples
    assert X.values.shape == (200, 2)
    assert pg.Inclusion(H, pg.L2(D, tol=1e-6)).tol == 1e-6 and pg.Inclusion(H, pg.L2(D)).tol == 1e-4


def test_posterior_mean_of_a_centred_prior_is_the_damped_solution_as_an_element():
    H = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.2)
    Dd = pg.Euclidean(2, covariance=1e-4)
    G = pg.Functionals(H.riesz([pg.Segment((-1, 0), (1, 0)), pg.Segment((0, -1), (0, 1))]), into=Dd)
    d = Dd(np.array([0.1, 0.2]))
    m = G.adjoint((G @ G.adjoint + 0.5 * pg.Identity(Dd)).inv(d))
    post = pg.probability.cylindrical(H, alpha=0.5).condition(G, d)
    assert post.mean.norm() == pytest.approx(m.norm(), rel=1e-12)
    assert post.mean.distance(m) < 1e-12 * m.norm()


def _rays(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a, b = rng.normal(size=3), rng.normal(size=3)
        a, b = 0.9 * a / np.linalg.norm(a), 0.9 * b / np.linalg.norm(b)
        t = np.linspace(0, 1, int(rng.integers(3, 9)))[:, None]
        P = (1 - t) * a + t * b
        out.append(P * (1 - 0.3 * np.sin(np.pi * t)))                           # bent towards the centre
    return out


def _lincomb(c, F):
    out = c[0] * F[0]
    for i in range(1, len(c)):
        out = out + c[i] * F[i]
    return out


def test_piecewise_density_on_polylines():
    H = pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.3)
    P = _rays(1)[0]
    w = np.linspace(1.0, 2.0, len(P) - 1)
    mu = pg.LebesgueMeasure(pg.Polyline(P)).with_density(w)
    assert isinstance(mu, pg.PieceDensity) and mu(mu.set) == pytest.approx(w @ mu.set._L)
    k = H.riesz(mu)
    segs = H.riesz([pg.Segment(p, q) for p, q in zip(P[:-1], P[1:])])
    kk = _lincomb(w, segs)
    assert H.inner(k, k) == pytest.approx(H.inner(kk, kk), rel=1e-13)
    f = H(lambda x, y, z: x + y * z)
    assert H.inner(f, k) == pytest.approx(sum(wj * H.inner(f, segs[j]) for j, wj in enumerate(w)), rel=1e-12)
    with pytest.raises(ValueError):
        pg.LebesgueMeasure(pg.Polyline(P)).with_density(-w)


def test_blocked_and_parallel_energies_and_values(monkeypatch):
    from pygeoinf0.realisations.rkhs import energy, backend as rkhs
    H = pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.3)
    rays = [pg.Polyline(P) for P in _rays(40)]
    dense = H.inner(H.riesz(rays), H.riesz(rays))
    pts = np.random.default_rng(1).uniform(-0.5, 0.5, (37, 3))
    c = np.random.default_rng(2).normal(size=len(rays))
    monkeypatch.setattr(energy, "ATOM_BLOCK", 23)
    monkeypatch.setattr(rkhs, "EVAL_PAIRS", 500)
    for n in ("1", "3"):
        monkeypatch.setenv("PYGEOINF0_WORKERS", n)
        F = H.riesz(rays)
        assert np.max(np.abs(H.inner(F, F) - dense)) <= 1e-14 * np.max(np.abs(dense))
        cross = H.inner(F, H.riesz(pts))                                    # k_i(x_j), the reproducing property
        m = _lincomb(c, F)
        assert np.allclose(m(*pts.T), c @ cross, rtol=1e-12, atol=1e-14)
        assert np.allclose(F[3](*pts.T), cross[3], rtol=1e-12, atol=1e-14)
