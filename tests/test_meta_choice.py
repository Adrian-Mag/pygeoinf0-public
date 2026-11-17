"""The data-dependent choice rules of pg.meta (plan 070 P2, review D3 and B14; thesis el:59, el:524-526): the
discrepancy principle, evidence maximisation, the knee and fit_noise take (problem, method, hyperparameter family),
state their family as a named record and whether they used the datum; the cells take a stated gamma; ELEMENTS
derives from Problem.FIELDS. The numbers are those of the rules before the move (the solver numerics are unchanged)."""
import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import pygeoinf0 as pg
from pygeoinf0 import meta


def small(seed=0, n=6, m=9):
    rng = np.random.default_rng(seed)
    M, D = pg.Euclidean(m), pg.Euclidean(n)
    A = rng.normal(size=(n, m))
    G = pg.Matrix(A, M, D, name="G")
    T = pg.Matrix(rng.normal(size=(1, m)), M, pg.Euclidean(1), name="T")
    d = D(A @ rng.normal(size=m) * 2.0 + rng.normal(size=n))
    prior = pg.GaussianMeasure(M, pg.Identity(M))
    return M, D, A, G, T, d, prior


def keys(rec):
    return {c.key: c for c in rec.claims if c.key}


def test_discrepancy_takes_a_problem_and_records_the_datum_and_its_family():
    M, D, A, G, T, d, prior = small()
    with pytest.raises(pg.NoiseUnstated):              # a point relation into a plain D states no noise (plan 071 P4)
        meta.discrepancy(pg.Problem(mdr=G, datum=d))
    pb = pg.Problem(mdr=pg.Additive(G, D.standard_gaussian()), datum=d)     # unit noise, stated
    ch = meta.discrepancy(pb)
    assert ch.family.name == "damping" and ch.family.parameters == ("gamma",) and ch.family.method == "least_squares"
    assert float(ch.gamma) == float(pg.solvers.damping_for_misfit(G @ G.adjoint, d, target=D.dim))
    k = keys(ch)
    assert k["meta"].holds and k["used the datum"].holds and "data-dependent rule" in k
    assert "gamma" in k["hyperparameter family"].statement and "least_squares" in k["hyperparameter family"].statement
    assert ch.gamma.claims == ch.claims                  # the chosen value carries the record into what uses it
    r = A @ A.T @ np.linalg.solve(A @ A.T + float(ch.gamma) * np.eye(D.dim), d.values) - d.values
    assert r @ r == pytest.approx(D.dim, rel=1e-9)       # chi^2 = n
    assert meta.discrepancy(pb, target=2.0).gamma != ch.gamma
    with pytest.raises(pg.Refused, match="least_squares"):
        meta.discrepancy(pb, method="sola")
    with pytest.raises(pg.Refused, match="datum"):
        meta.discrepancy(pg.Problem(mdr=G))


def test_the_cells_take_a_stated_gamma_and_carry_a_choice_s_claims():
    M, D, A, G, T, d, prior = small(1)
    R = pg.Additive(G, D.standard_gaussian())
    J = pg.Ordering.norm(M)
    with pytest.raises(pg.Refused, match="pg.meta.discrepancy"):
        pg.update(J, R, d, rule=("soft", "discrepancy"))
    pb = pg.Problem(mdr=R, datum=d)
    with pytest.raises(pg.Refused, match="stated gamma"):
        pb.run("least_squares")
    ch = meta.discrepancy(pb)
    ls = pb.run("least_squares", **ch.options())
    assert ls.claims.find("data-dependent rule") is not None and ls.claims.find("used the datum").holds
    fixed = pb.run("least_squares", gamma=float(ch.gamma))
    assert np.array_equal(fixed.summary().values, ls.summary().values)
    assert fixed.claims.find("data-dependent rule") is None


def test_a_centred_ordering_chooses_from_the_centred_datum():
    M, D, A, G, T, d, prior = small(2)
    c = M(np.linspace(-1, 1, M.dim))
    fam = meta.damping("least_squares", centre=c)
    ch = meta.discrepancy(pg.Problem(mdr=pg.Additive(G, D.standard_gaussian()), datum=d), family=fam)
    assert float(ch.gamma) == float(pg.solvers.damping_for_misfit(G @ G.adjoint, d - G(c), target=D.dim))
    assert ch.options()["centre"] is c


def test_maximum_evidence_over_the_scales_is_a_stationary_point_of_the_dense_evidence():
    M, D, A, G, T, d, prior = small(3, n=12, m=8)
    pb = pg.Problem(mdr=pg.Additive(G, D.standard_gaussian()), prior=prior, datum=d)
    ch = meta.maximum_evidence(pb)
    a, nu = ch.values["a"], ch.values["nu"]
    assert ch.family.name == "scales" and set(ch.family.parameters) == {"a", "nu"}

    def logp(a, nu):
        return multivariate_normal(np.zeros(D.dim), a * a * A @ A.T + nu * nu * np.eye(D.dim)).logpdf(d.values)
    assert ch.log_evidence == pytest.approx(logp(a, nu), rel=1e-10)
    h = 1e-5
    assert abs(logp(a * (1 + h), nu) - logp(a * (1 - h), nu)) / (2 * h) < 1e-4 * D.dim
    assert abs(logp(a, nu * (1 + h)) - logp(a, nu * (1 - h))) / (2 * h) < 1e-4 * D.dim
    assert keys(ch)["used the datum"].holds and "gamma = (nu / a)^2" in keys(ch)["hyperparameter family"].statement
    assert ch.options() == {"a": a, "nu": nu}
    fit = meta.fit_noise(pb, family=meta.NoiseFamily(D))       # one group, a free: the same maximum
    assert fit.log_evidence == pytest.approx(ch.log_evidence, rel=1e-7)


def test_fit_noise_takes_a_problem_and_its_family_is_a_named_record():
    M, D, A, G, T, d, prior = small(4, n=10, m=8)
    fam = meta.NoiseFamily(D, sigma=1.0, groups=[np.arange(5), np.arange(5, 10)])
    fit = meta.fit_noise(pg.Problem(mdr=G, prior=prior, datum=d), family=fam)
    assert set(fit.values) == set(fam.parameters) == set(fit.theta) | {"a"}
    k = keys(fit)
    assert k["used the datum"].holds and "data-dependent rule" in k
    assert "theta[0]" in k["hyperparameter family"].statement
    with pytest.raises(pg.Refused, match="family"):
        meta.fit_noise(pg.Problem(mdr=G, prior=prior, datum=d))


def test_the_knee_rule_takes_a_problem_and_does_not_use_the_datum():
    M, D, A, G, T, d, prior = small(5)
    noise = D.standard_gaussian()
    pb = pg.Problem(mdr=pg.Additive(G, noise), mpr=T, datum=d)
    k = meta.knee(pb, "sola", meta.damping("sola", lo=1e-3, hi=1e3))
    old = meta.Knee().find(meta.TradeOff(pg.decompose(G, noise), T), 1e-3, 1e3)
    assert k.gamma == old.gamma and k.values == {"gamma": k.gamma}
    kk = keys(k)
    assert kk["used the datum"].holds is False and "data-dependent rule" not in kk
    assert "[0.001, 1000]" in kk["hyperparameter family"].statement
    with pytest.raises(pg.Refused, match="lo"):
        meta.knee(pb, "sola", meta.damping("sola"))


def test_the_pli_records_name_their_hyperparameter_family():
    M, D, A, G, T, d, prior = small(6)
    dec = pg.decompose(G, D.standard_gaussian(), middle=prior.covariance)
    vs = meta.VarianceSplit(dec, T, a=2.0, nu=0.5)
    f = keys(vs)["hyperparameter family"]
    assert "a = 2" in f.statement and "nu = 0.5" in f.statement and "gamma = (nu / a)^2 = 0.0625" in f.statement
    cv = meta.Coverage(dec, T, M(np.ones(M.dim)), a=2.0, nu=0.5)
    assert "gamma = (nu / a)^2" in keys(cv)["hyperparameter family"].statement
    pc = meta.PredictiveCheck(dec, d, a=2.0, nu=0.5)
    assert "a = 2" in keys(pc)["hyperparameter family"].statement


def test_elements_derive_from_the_problem_fields():
    from pygeoinf0.inference import problem
    for f in problem.FIELDS:
        if f != "datum":
            assert meta.ELEMENTS[f] == problem.ELEMENTS[f]
    assert "datum" not in meta.ELEMENTS and meta.ELEMENTS["MDR"] == problem.ELEMENTS["mdr"]
    assert set(problem.ELEMENTS) == set(problem.FIELDS) - {"datum"}


def test_the_rules_left_their_former_homes():
    assert not hasattr(pg.solvers, "discrepancy") and not hasattr(pg.relations, "fit_noise")
    assert not hasattr(pg.relations, "NoiseFamily") and not hasattr(pg.relations, "NoiseFit")
    with pytest.raises(ImportError):
        import pygeoinf0.relations.noise  # noqa: F401
    M, D, A, G, T, d, prior = small(7)
    assert not hasattr(pg.decompose(G, D.standard_gaussian()), "discrepancy")
    assert math.isfinite(float(pg.decompose(G, D.standard_gaussian()).damping_for_misfit(d, D.dim)))
