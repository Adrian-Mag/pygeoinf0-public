"""The data space and the noise (plan 071 P4; Mac review items 4, 5, 6; the author's decision Q1 as amended 2026-11-15:
"One should be able to do either. Plain data space with noise separate, or work in whitened norm in which case one
should be able to put a covariance on the dataspace").

Route A: a plain D (Euclidean(n), DataSpace(n)) with the noise in the relation, pg.Additive(G, N(b, C)).
Route B: the whitened norm, D = Euclidean(n, covariance=C), with the standard Gaussian of D (coordinates N(0, C)).
Both work and agree to rounding (tests/test_laws.py has the law over generated problems). Neither silently assumes a
noise: a point relation into a plain D with nothing said about the noise is refused where a method needs a noise law
(pg.NoiseUnstated); into a covariance-weighted D it reads unit noise in D's metric, and says so. sola needs a stated
gamma. A covariance that is not symmetric (beyond rounding) or not positive definite is refused (pg.NotACovariance).
"""
import numpy as np
import pytest

import pygeoinf0 as pg

SIG = np.array([0.5, 1.0, 2.0, 0.3, 0.7, 1.5])


def _problem(route, dense=False, mean=False):
    rng = np.random.default_rng(3)
    n, m = len(SIG), 4
    A = rng.standard_normal((n, m))
    C = np.diag(SIG ** 2)
    if dense:
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        C = (Q * SIG ** 2) @ Q.T
        C = 0.5 * (C + C.T)
    b = rng.standard_normal(n) if mean else None
    d = A @ rng.standard_normal(m) + 0.1 * rng.standard_normal(n)
    M = pg.Euclidean(m)
    T = pg.Matrix(rng.standard_normal((1, m)), M, pg.Euclidean(1), name="T")
    if route == "A":
        D = pg.Euclidean(n)
        cov = pg.Matrix(C, D, D, name="C") if dense else pg.Diagonal(SIG ** 2, D)
        noise = pg.GaussianMeasure(D, cov, None if b is None else D(b))
    else:
        D = pg.Euclidean(n, covariance=C if dense else SIG ** 2)
        noise = D.standard_gaussian() if b is None else D.standard_gaussian() + D(b)
    G = pg.Matrix(A, M, D, name="G")
    return pg.Problem(M=M, D=D, mdr=pg.Additive(G, noise), mpr=T, prior=pg.GaussianMeasure(M, pg.Identity(M)),
                      datum=D(d))


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("mean", [False, True])
def test_the_discrepancy_principle_takes_either_route_and_they_agree(dense, mean):
    a, b = (pg.meta.discrepancy(_problem(r, dense, mean)) for r in "AB")
    assert a.gamma == pytest.approx(b.gamma, rel=1e-9)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("mean", [False, True])
def test_damped_least_squares_takes_either_route_and_they_agree(dense, mean):
    a, b = (_problem(r, dense, mean).run("least_squares", gamma=0.3) for r in "AB")
    assert np.allclose(a.summary().values, b.summary().values, rtol=1e-9, atol=1e-12)
    m = a.summary()
    assert a.cost(m) == pytest.approx(b.cost(m), rel=1e-9)          # the same misfit: the C^-1 norm


@pytest.mark.parametrize("dense", [False, True])
def test_sola_and_pli_agree_on_either_route(dense):
    pa, pb = _problem("A", dense), _problem("B", dense)
    sa, sb = pa.run("sola", gamma=0.3), pb.run("sola", gamma=0.3)
    assert np.allclose(sa.estimate.values, sb.estimate.values, rtol=1e-9)
    la, lb = pa.run("pli"), pb.run("pli")
    assert np.allclose(la.mean.values, lb.mean.values, rtol=1e-9)
    assert np.allclose(la.covariance.matrix(), lb.covariance.matrix(), rtol=1e-9)


def test_a_point_relation_into_a_plain_data_space_has_no_noise_and_refuses():
    pb = _problem("A")
    G = pb.mdr.G
    point = pg.Problem(M=pb.M, D=pb.D, mdr=G, mpr=pb.mpr, prior=pb.prior, datum=pb.datum)
    with pytest.raises(pg.NoiseUnstated) as e:
        pg.meta.discrepancy(point)
    assert any("pg.Additive" in c for c in e.value.needs) and any("covariance=" in c for c in e.value.needs)
    with pytest.raises(pg.NoiseUnstated):
        point.run("sola", gamma=0.3)
    with pytest.raises(pg.NoiseUnstated):
        pg.meta.maximum_evidence(point)
    assert np.all(np.isfinite(point.run("least_squares", gamma=0.3).summary().values))   # no noise read: kept
    stated = pg.meta.discrepancy(point, target=2.0)                  # a stated target reads no noise: kept
    assert np.isfinite(stated.gamma)


def test_a_point_relation_into_a_covariance_weighted_data_space_reads_unit_noise_in_its_metric_and_says_so():
    pb = _problem("B")
    point = pb.with_(mdr=pb.mdr.G)
    a, b = pg.meta.discrepancy(point), pg.meta.discrepancy(pb)
    assert a.gamma == pytest.approx(b.gamma, rel=1e-12)
    assert any("standard Gaussian of the data space" in c.statement for c in a.claims), a.claims
    s = point.run("sola", gamma=0.3)
    assert any("standard Gaussian of the data space" in c.statement for c in s.claims), s.claims


def test_sola_needs_a_stated_gamma():
    pb = _problem("A")
    with pytest.raises(pg.Refused) as e:
        pb.run("sola")
    assert "gamma" in str(e.value) and "pg.meta" in str(e.value)
    with pytest.raises(pg.Refused):
        pg.inference.sola(pb.mdr, pb.mpr, pb.datum)


def test_a_non_symmetric_covariance_is_refused():
    pb = _problem("A")
    G = pb.mdr.G
    C = np.diag(SIG ** 2)
    C[0, 1] = 0.1
    with pytest.raises(pg.NotACovariance, match="symmetric"):
        pg.decompose(G, C)
    D = G.codomain
    with pytest.raises(pg.NotACovariance, match="symmetric"):
        pg.decompose(G, pg.GaussianMeasure(D, pg.Matrix(C, D, D)))
    Cr = np.diag(SIG ** 2)
    Cr[0, 1] = 1e-15 * Cr.max()                                         # rounding asymmetry: within the tolerance
    assert pg.decompose(G, Cr).lam.shape == (D.dim,)


@pytest.mark.parametrize("form", ["vector", "matrix", "operator"])
def test_a_negative_variance_is_named(form):
    pb = _problem("A")
    G, D = pb.mdr.G, pb.D
    v = SIG ** 2
    v[2] = -1.0
    C = {"vector": v, "matrix": np.diag(v), "operator": pg.Diagonal(v, D)}[form]
    with pytest.raises(pg.NotACovariance, match="coordinate 2"):
        pg.decompose(G, C)


def test_a_covariance_that_is_not_positive_definite_is_refused_by_name():
    pb = _problem("A")
    G = pb.mdr.G
    C = np.diag(SIG ** 2)
    C[0, 1] = C[1, 0] = 5.0
    with pytest.raises(pg.NotACovariance, match="positive definite"):
        pg.decompose(G, C)


def test_decompose_and_condition_refuse_an_unstated_noise_on_a_plain_data_space():
    # plan 071 P8 (the author's decision on 071 P4 open decision 2): no noise and a plain D is a refusal naming the
    # routes; stated unit noise gives the former numbers; a covariance-weighted D keeps unit noise in its metric
    M, D = pg.Euclidean(3), pg.Euclidean(2)
    A = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
    G = pg.Matrix(A, M, D, name="G")
    with pytest.raises(pg.NoiseUnstated) as e:
        pg.decompose(G)
    assert any("standard_gaussian" in n for n in e.value.needs)
    prior = M.standard_gaussian()
    d = D(np.array([1.0, -1.0]))
    with pytest.raises(pg.NoiseUnstated):
        prior.condition(G, d)
    stated = prior.condition(G, d, D.standard_gaussian())
    assert np.allclose(stated.mean.values, A.T @ np.linalg.solve(A @ A.T + np.eye(2), d.values))
    assert pg.decompose(G, D.standard_gaussian()) is not None
    W = pg.Euclidean(2, covariance=np.diag([4.0, 9.0]))
    GW = pg.Matrix(A, M, W, name="G")
    assert pg.decompose(GW) is not None and M.standard_gaussian().condition(GW, W(np.array([1.0, -1.0]))) is not None
