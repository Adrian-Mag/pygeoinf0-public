"""Cells (plan 008 P3): the existing methods re-expressed as cells give the same numbers; refusals; claims gating;
the table; orderings; exact-data Gaussian conditioning; problems and methods."""
import numpy as np
import pytest

import pygeoinf0 as pg

rng = np.random.default_rng(21)


def interval_problem(n=12, weighted=True):
    H = pg.Sobolev("[0, 1]", 1, length_scale=0.1)
    D = pg.Euclidean(n, covariance=np.linspace(0.5, 2.0, n) ** 2 * 1e-4) if weighted else pg.Euclidean(n)
    G = pg.Functionals(H.riesz([pg.Interval(a, a + 0.08) for a in np.linspace(0.02, 0.9, n)]), into=D)
    truth = H(lambda x: np.sin(2 * np.pi * x) + 0.3 * x)
    d = G(truth) + D.standard_gaussian().sample(seed=3)
    T = pg.Functionals(H.riesz([pg.Interval(0.4, 0.6), pg.DiracMeasure(0.25)]))
    return H, D, G, T, d


def test_condition_is_cell_b10_with_the_same_numbers():
    import ladder
    for name in ("L1", "L2"):
        P = ladder.build(name, 60)
        G = pg.Functionals(P.H.riesz(P.measures), into=P.D)
        prior = pg.probability.cylindrical(P.H, alpha=2.0)
        post = prior.condition(G, P.d)                                  # through the cell (plan 008)
        old = prior._condition(G, P.d, P.D.standard_gaussian())                  # the pre-008 body itself
        pts = np.array([[0.3]]) if name == "L1" else np.array([[0.2, 0.3], [-0.1, 0.4]])
        assert np.array_equal(post.mean(*pts.T), old.mean(*pts.T))
        assert np.array_equal(post.std(P.measures[:5]), old.std(P.measures[:5]))
        cell = pg.update(prior, pg.Additive(G, P.D.standard_gaussian()), P.d)
        assert np.array_equal(cell.std(P.measures[:5]), old.std(P.measures[:5]))
        assert post.cell == "B10" and "independent of x" in repr(post.claims)
        assert "cylindrical" in repr(post.claims)


def test_push_of_a_measure_is_its_push_forward():
    H, D, G, T, d = interval_problem()
    mu = pg.probability.cylindrical(H, alpha=3.0)
    a, b = pg.push(mu, T), T(mu)
    assert np.array_equal(a.covariance.matrix(), b.covariance.matrix())
    assert a.cell == "F10" and "cylindrical" in repr(a.claims)


def test_ordering_summaries_are_minimum_norm_and_damped_least_squares():
    H, D, G, T, d = interval_problem()
    J = pg.Ordering.norm(H)
    m_ref = G.adjoint((G @ G.adjoint).inv(d))
    hard = pg.update(J, G, d)
    assert hard.summary().distance(m_ref) == 0.0
    assert hard.feasible.member(m_ref) is pg.Truth.TRUE and hard.cost(m_ref + H.riesz(0.5)) == np.inf
    A = G @ G.adjoint
    R = pg.Additive(G, D.standard_gaussian())
    alpha = pg.meta.discrepancy(pg.Problem(mdr=R, datum=d)).gamma         # chosen from the datum: a meta rule
    assert float(alpha) == float(pg.solvers.damping_for_misfit(A, d, D.dim))
    m_dls = G.adjoint((A + alpha * pg.Identity(D)).inv(d))
    soft = pg.update(J, R, d, rule=("soft", alpha))
    assert soft.summary().distance(m_dls) == 0.0
    assert soft.claims.find("data-dependent rule") is not None and "discrepancy" in repr(soft.claims)
    ls = pg.inference.least_squares(R, d, gamma=float(alpha))
    assert ls.summary().distance(m_dls) == 0.0
    with pytest.raises(pg.Refused, match="weight"):
        pg.update(J, pg.Additive(G, D.standard_gaussian()), d)


def test_minimum_norm_method_and_the_fibre():
    H, D, G, T, d = interval_problem()
    K = pg.update(pg.NoPrior(H), G, d)
    assert isinstance(K, pg.AffineSubset) and K.cell == "B5"            # the whole space's row (plan 067 P4; was B1)
    assert K.summary().distance(G.adjoint((G @ G.adjoint).inv(d))) == 0.0
    assert pg.inference.minimum_norm(G, d).distance(K.summary()) == 0.0
    P = pg.push(K, T)                                                  # T is not estimable: the line T m + T(ker G)
    assert P.direction.dim == 2
    assert P.interval(0) == (-np.inf, np.inf)


def test_estimability_as_a_cell():
    D = pg.Euclidean(4)
    X = pg.Euclidean(6)
    G = pg.Matrix(rng.normal(size=(4, 6)), X, D)
    Te = pg.Matrix(rng.normal(size=(1, 4)) @ G.A, X, pg.Euclidean(1))   # T = W G: estimable
    K = pg.update(pg.NoPrior(X), G, D(rng.normal(size=4)))
    Pe = pg.push(K, Te)
    assert Pe.direction.dim == 0
    lo, hi = Pe.interval(0)
    assert lo == hi == pytest.approx((Te.A @ K.summary().values)[0], rel=1e-12)


def test_exact_data_conditioning_is_the_small_noise_limit():
    H, D0, G0, T, d0 = interval_problem(weighted=False)
    mu = pg.probability.cylindrical(H, alpha=1.0)
    exact = pg.update(mu, G0, d0)
    assert exact.cell == "B8"
    u = H.riesz(pg.Interval(0.3, 0.5))
    errs = []
    for eps in (1e-2, 1e-4, 1e-6):
        noisy = mu.condition(G0, d0, pg.GaussianMeasure(D0, eps ** 2 * pg.Identity(D0)))
        errs.append(abs(H.inner(noisy.mean, u) - H.inner(exact.mean, u)))
    assert errs[2] < errs[1] < errs[0] and errs[2] < 1e-6
    D2 = pg.Euclidean(2)
    C = pg.FiniteRank(D2.family(np.array([[1.0, 0.0]])), [1.0])        # a prior concentrated on the first axis
    with pytest.raises(pg.Refused, match="support"):
        pg.update(pg.GaussianMeasure(D2, C), pg.Matrix(np.eye(2), D2, D2), D2([0.0, 1.0]))


def test_refusals_and_the_policy():
    H, D, G, T, d = interval_problem()
    L = pg.Additive(G, D.standard_gaussian())
    S = pg.NormBall(H, 1.0)
    for call, word in ((lambda: pg.update(pg.NoPrior(H), L, d), "prior"),
                       (lambda: pg.update(S, L, d), "acceptance"),
                       (lambda: pg.push(pg.NoPrior(H), L), "family"),
                       (lambda: pg.push(S, L), "family"),
                       (lambda: pg.push(pg.probability.cylindrical(H), pg.Additive(G, pg.NormBall(D, 1.0))), "random set"),
                       (lambda: pg.push(pg.Ordering.norm(H), L), "further choice")):
        with pytest.raises(pg.Refused) as e:
            call()
        assert word in str(e.value)
    M = pg.L2("[0, 1]")
    F = pg.LinearOperator(M, D, apply=lambda m: D(np.zeros(D.dim)), adjoint=lambda y: M.zero, name="F")
    F._claims.clear()                                                  # linearity not on record
    with pytest.raises(pg.HypothesisUnmet) as e:
        pg.update(pg.NoPrior(M), pg.Additive(F, pg.NormBall(D, 1.0)), d)
    assert "assert_claim" in e.value.needs[0]
    K = pg.update(pg.NoPrior(M), pg.Additive(F, pg.NormBall(D, 1.0)), d, policy="tagged")
    assert K.complete is False and K.missing


def test_the_table_lists_every_cell():
    text = pg.cells.table()
    for name in [f"F{i}" for i in range(4, 16)] + [f"B{i}" for i in range(4, 14)]:   # F1-F3, B1-B3 (the prior none)
        assert f"| {name} |" in text                                                   # are the set rows (plan 067 P4)
    assert "refused" in text and "today:" in text


def test_ordering_push_is_the_inf_projection():
    X = pg.Euclidean(5, weights=rng.uniform(1, 2, 5))
    T = pg.Matrix(rng.normal(size=(2, 5)), X, pg.Euclidean(2))
    c = X(rng.normal(size=5))
    Jp = pg.push(pg.Ordering.norm(X, centre=c), T)
    p = pg.Euclidean(2)(rng.normal(size=2))
    W = X.metric_matrix()                                              # min (1/2)||x - c||^2 with T x = p, by lstsq
    L = np.linalg.cholesky(W).T
    z = np.linalg.lstsq(T.A @ np.linalg.inv(L), p.values - T.A @ c.values, rcond=None)[0]
    assert Jp.cost(p) == pytest.approx(0.5 * z @ z, rel=1e-10)
    assert Jp.summary().values == pytest.approx(T.A @ c.values)


def test_occam_summary_of_a_hard_update():
    H, D, G, T, d = interval_problem()
    r = 0.8 * D.norm(d)
    J = pg.update(pg.Ordering.norm(H), pg.Additive(G, pg.NormBall(D, r)), d)
    m = J.summary()
    assert D.norm(G(m) - d) == pytest.approx(r, rel=1e-6)            # the misfit constraint is active
    assert J.feasible.member(m) is not pg.Truth.FALSE


def test_problem_runs_methods_when_their_objects_exist():
    H, D, G, T, d = interval_problem()
    pb = pg.Problem(M=H, D=D)
    pb = pb.with_(mdr=pg.Additive(G, D.standard_gaussian()), datum=d)
    runnable = {n for n, ok, miss, _ in pb.methods() if ok}
    assert {"least_squares", "pli"} - runnable == {"pli"} and "least_squares" in runnable
    with pytest.raises(pg.Refused, match="prior"):
        pb.run("pli")
    pb = pb.with_(P=T.codomain, mpr=T, prior=pg.probability.cylindrical(H, alpha=2.0), calibration=pg.Bayesian())
    post = pb.run("pli")
    assert np.allclose(post.covariance.matrix(), pg.push(pg.update(pb.prior, pb.mdr, d), T).covariance.matrix())
    assert "cylindrical" in repr(pb.calibration_claims())
    K = pg.Problem(M=H, D=D, mdr=G, datum=d).update()
    assert "unstated" in repr(K.claims)
