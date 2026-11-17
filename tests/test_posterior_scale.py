"""Plan 005 P5: the posterior above settings.dense_max. The randomised Nystrom decomposition (solvers/dataspace/partial.py) with
its probabilistic bound on ||W - W_k||, the one-sided per-functional variance bound (closed forms), post.std and samples
at scale (probability/scale.py) against the exact dense posterior, and the deflated block CG."""
import sys
from pathlib import Path

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.solvers.dataspace import partial

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder  # noqa: E402


def problem(n=300, m=800, decay=1.6, seed=1, scale=30.0):
    """A data map whose whitened Gram has a slowly decaying spectrum (weighted data space: a diagonal noise)."""
    rng = np.random.default_rng(seed)
    M, D = pg.Euclidean(m), pg.Euclidean(n, weights=rng.uniform(0.5, 2, n))
    A = rng.normal(size=(n, m)) * (scale * np.arange(1, m + 1) ** -decay)[None, :]
    G = pg.Matrix(A, M, D)
    T = pg.Matrix(rng.normal(size=(30, m)), M, pg.Euclidean(30))
    return M, D, G, T, D(rng.normal(size=n))


@pytest.mark.parametrize("tail", [1.0, 0.1])
def test_randomised_posterior_variances_are_one_sided_and_bounded(tail):
    M, D, G, T, d = problem()
    C = pg.Identity(M)
    full = pg.decompose(G, None, C)
    with pg.settings.using(dense_max=100):
        part = pg.decompose(G, None, C, tail=tail)
    assert part.route == "randomised" and part.partial and part.error <= tail
    assert len(part.lam) < 300                                       # a real truncation
    assert any("Nystrom" in c.statement and "1 - 1e-10" in c.reason for c in part.claims)
    for a, nu in ((1.0, 1.0), (2.0, 0.5), (0.5, 2.0)):
        pf, pp = full.posterior(T, d, a=a, nu=nu), part.posterior(T, d, a=a, nu=nu)
        gap = pf.std ** 2 - pp.std ** 2
        assert np.all(gap >= -1e-10 * pf.std ** 2)                   # the low-rank variance never exceeds the exact
        assert np.all(gap <= pp.variance_bound * (1 + 1e-9) + 1e-14)  # and the stated bound covers the difference
        assert any("Nystrom" in c.statement for c in pp.claims)


def test_the_bound_holds_over_many_seeds():
    """The bound on ||W - W_k|| fails with probability <= 1e-10: never over 40 independent problems and probes."""
    for seed in range(40):
        M, D, G, T, d = problem(n=120, m=300, seed=100 + seed)
        C = pg.Identity(M)
        full = pg.decompose(G, None, C)
        with pg.settings.using(dense_max=50):
            part = pg.decompose(G, None, C, tail=0.3)
        Wk = part._U[0] @ np.diag(part.lam) @ part._U[0].T
        Wf = full._U[0] @ np.diag(full.lam) @ full._U[0].T
        E = np.linalg.eigvalsh(Wf - Wk)
        slack = 1e-12 * full.lam.max() * 120                         # rounding of the dense reconstruction
        assert E.min() >= -slack                                     # W_k <= W
        assert E.max() <= part.error * (1 + 1e-9) + slack


def test_full_rank_randomised_equals_the_dense_decomposition():
    M, D, G, T, d = problem(n=90, m=60)                               # rank 60: the range finder finds all of it
    C = pg.Identity(M)
    full = pg.decompose(G, None, C)
    with pg.settings.using(dense_max=30):
        part = pg.decompose(G, None, C)
    pf, pp = full.posterior(T, d), part.posterior(T, d)
    assert np.allclose(pp.std, pf.std, rtol=1e-9) and np.allclose(pp.mean, pf.mean, rtol=0, atol=1e-7 * np.abs(pf.mean).max())


def test_deflated_block_cg_solves_the_whitened_system():
    M, D, G, T, d = problem(n=200, m=500)
    with pg.settings.using(dense_max=50):
        dec = pg.decompose(G, None, pg.Identity(M), tail=1.0)
    W, _ = partial.operator(dec)
    Wd = W(np.eye(200))
    B = np.random.default_rng(3).normal(size=(200, 7))
    for rho2 in (1.0, 4.0):
        Z, its, rel = partial.pcg(W, dec._U[0], dec.lam, B, rho2=rho2, tol=1e-12)
        ref = np.linalg.solve(rho2 * Wd + np.eye(200), B)
        cond = np.linalg.cond(rho2 * Wd + np.eye(200))
        assert np.abs(Z - ref).max() <= 10 * cond * 1e-12 * np.abs(ref).max()   # residual 1e-12 times the condition
        assert 0 < its <= 40 and rel <= 1e-12                          # deflation: few iterations


@pytest.mark.parametrize("name, n", [("L1", 300), ("L2", 200)])
def test_post_std_at_scale_on_the_ladder(name, n):
    P = ladder.build(name, n)
    _, _, post = ladder.solve(P)
    u = P.measures[:40]
    exact = post.std(u)
    with pg.settings.using(dense_max=40):
        _, _, big = ladder.solve(P)
        assert big._scale is not None
        approx = big.std(u)
        claims = str(big.variance_claims)
        forced = big.std(u, exact=True)
        one = big.std(u[0])
    assert np.all(np.abs(approx - exact) <= 0.01 * exact)               # decision 3: 1 % of the posterior std
    assert "randomised decomposition" in claims and "Nystrom" in claims
    assert np.allclose(forced, exact, rtol=1e-6)
    assert one == pytest.approx(approx[0], rel=1e-12)


def test_functionals_failing_the_rule_are_computed_exactly():
    P = ladder.build("L1", 300)
    _, _, post = ladder.solve(P)
    u = P.measures[:30]
    exact = post.std(u)
    with pg.settings.using(dense_max=40):
        _, _, big = ladder.solve(P)
        big._scale.dec                                                  # build at the default accuracy
        with pg.settings.using(accuracy=1e-12):                         # every bound now fails the rule
            s = big.std(u)
        assert "by deflated block CG" in str(big.variance_claims)
    assert np.allclose(s, exact, rtol=1e-7)


def test_matheron_samples_with_block_cg_match_exact_solves():
    rng = np.random.default_rng(2)
    n, m = 160, 400
    M, D = pg.Euclidean(m), pg.Euclidean(n, weights=rng.uniform(0.5, 2, n))
    G = pg.Matrix(rng.normal(size=(n, m)) * (8.0 * np.arange(1, m + 1) ** -1.2)[None, :], M, D)
    d = D(rng.normal(size=n))
    prior = pg.GaussianMeasure(M, pg.Matrix(np.diag(np.linspace(2, 0.5, m)), M, M))
    post = prior.condition(G, d)
    with pg.settings.using(dense_max=40, accuracy=1e-6):
        big = prior.condition(G, d)
        with pg.settings.using(dense_max=40):
            big._scale._dec = pg.decompose(G, None, prior.covariance, tail=50.0)  # truncated: CG iterates
            from pygeoinf0.solvers.dataspace.partial import operator
            big._scale.W, big._scale.wt = operator(big._scale._dec)
        X = big._scale.sampler(30, np.random.default_rng(7))
    r7 = np.random.default_rng(7)
    X0, E0 = prior._centred(30, r7), D.standard_gaussian()._centred(30, r7)
    Xe = X0 - (prior.covariance @ G.adjoint)(post.solver(G(X0) + E0))
    assert np.abs(X.values - Xe.values).max() <= 1e-6 * np.abs(Xe.values).max()
    assert "deflated block CG" in str(big._scale.last_sample) and "0 iterations" not in str(big._scale.last_sample)
    draws = big.sample(400, seed=11)                                    # moments of the law (a statistical check)
    assert np.abs(draws.values.mean(axis=0) - post.mean.values).max() <= 4 * post.std(M.family([M(np.eye(m)[i]) for i in
                                                                                                 range(m)])).max() / 20


@pytest.mark.parametrize("tail", [0.5, 30.0])
@pytest.mark.parametrize("many", [False, True])
def test_family_posteriors_state_the_bound(tail, many, monkeypatch):
    """Plan 009 G4 families (maps by synthesis) and many-property maps on the randomised route: every property's
    variance one-sided and within its bound."""
    from test_families import plain, problem as family_problem
    from pygeoinf0.solvers.dataspace import factored, decomposition
    Sp, G, noise, T, d, n = family_problem(p=60)
    if many:                                                   # the plain route with > COVARIANCE_DENSE properties
        monkeypatch.setattr(factored, "COVARIANCE_DENSE", 10)
        monkeypatch.setattr(decomposition, "COVARIANCE_DENSE", 10)
        T = plain(T)
    full = pg.decompose(G, noise, pg.Identity(Sp))
    G.structured_gram = lambda S=None, summands=None: None     # hide the structure: the randomised route
    with pg.settings.using(dense_max=50):
        part = pg.decompose(G, noise, pg.Identity(Sp), tail=tail)
    assert part.route == "randomised"
    for a in (1.0, 3.0):
        pf, pp = full.posterior(T, d, a=a), part.posterior(T, d, a=a)
        vf = pf.variance if pf.covariance is None or hasattr(pf, "variance") else np.diag(pf.covariance)
        gap = vf - pp.variance
        assert np.all(gap >= -1e-10 * vf) and np.all(gap <= pp.variance_bound * (1 + 1e-9) + 1e-14)
        assert "Nystrom" in str(pp.claims)


@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_the_combined_bound_covers_a_known_product_error(eps, monkeypatch):
    """The propagation of a product error, exactly: the decomposition sees G^ = (1 + eps) G (so W^ = (1 + eps)^2 W, and
    the data image is off by eps / (1 + eps)), stated as delta = (1 + eps)^2 - 1; every exact variance (with G) must lie
    inside the combined bound, and without the engine term some would not."""
    from pygeoinf0.probability import scale
    M, D, G, T, d = problem(n=200, m=500)
    prior = pg.GaussianMeasure(M, pg.Matrix(np.diag(np.linspace(2, 0.5, 500)), M, M))
    Gh = pg.Matrix((1 + eps) * G.matrix(), M, D)
    u = M.family([M(v) for v in np.random.default_rng(4).normal(size=(30, 500))])
    exact = prior.condition(G, d).std(u)
    for delta, must in ((2 * eps + eps ** 2, True), (0.0, False)):
        monkeypatch.setattr(scale, "product_error", lambda G_, dl=delta: (dl, "test: a known product error"))
        with pg.settings.using(dense_max=50):
            big = prior.condition(Gh, d)
            s = big.std(u)
            bound = big._scale.last_bound
        inside = np.abs(exact ** 2 - s ** 2) <= bound * (1 + 1e-9) + 1e-11 * exact ** 2
        assert inside.all() if must else not inside.all()


@pytest.mark.slow(reason="L3 above dense_max through the spectral engine: about 4 minutes on a laptop")
def test_the_combined_bound_covers_a_fast_engine():
    """L3 (3-D segments): above dense_max the products come from settings.engine; every std must lie inside the
    combined bound (Nystrom plus the engine's estimated product error, propagated: probability/scale.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from posterior_scale import functionals
    P = ladder.build("L3", 120, segments=2)
    u = functionals(P, 60)
    with pg.settings.using(dense_max=40):
        G, alpha, big = ladder.solve(P)
        s = big.std(u)
        bound = big._scale.last_bound
        claim = str(big.variance_claims[-1])
    exact = pg.probability.cylindrical(P.H, alpha=alpha).condition(G, P.d).std(u)          # the same alpha, exact dense routes
    assert "engine" in claim and "estimate" in claim and "not yet measured" not in claim
    assert np.all(np.abs(exact ** 2 - s ** 2) <= bound * (1 + 1e-9) + 1e-11 * exact ** 2)   # plus rounding


def test_product_error_reads_the_spectral_plan_of_the_atoms(monkeypatch):
    """The spectral engine's estimate is stored on the plan of the atom set (Atoms.spectral.est); the europa L3 1e5
    run found product_error looking for it on the engine and falling back to settings.accuracy."""
    from types import SimpleNamespace as NS
    from pygeoinf0.probability import scale
    atoms = NS(n=500, spectral=NS(est=3.2e-4))
    engine = NS(route=lambda A: ("spectral", NS(), "test"))
    G = NS(kernels=NS(realisation=NS(kx=NS(atoms=atoms))), domain=NS(backend=NS(engine=engine)))
    with pg.settings.using(dense_max=100):
        delta, how = scale.product_error(G)
        assert delta == 3.2e-4 and "estimated relative product error" in how
        atoms.spectral = None
        delta, how = scale.product_error(G)
        assert delta == pg.settings.accuracy and "not yet measured" in how


def test_route_by_cost_when_the_spectrum_does_not_fall_below_the_tail():
    """L2: the spectrum stays above the tail; the rank is capped at about p and every functional goes to block CG."""
    P = ladder.build("L2", 300)
    _, _, post = ladder.solve(P)
    u = P.measures[:20]
    exact = post.std(u)
    with pg.settings.using(dense_max=40):
        _, _, big = ladder.solve(P)
        s = big.std(u)
        claim = str(big.variance_claims[-1])
        rank = len(big._scale.dec.lam)
    assert rank <= 64 and "rank cap" in claim and "cheaper than a larger rank" in claim
    assert np.allclose(s, exact, rtol=1e-6)
