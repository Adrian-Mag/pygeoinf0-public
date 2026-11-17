"""Direct sums (plan 007, P1): inner products by summand, representers of functionals that touch some summands,
Functionals and their adjoints, conditioning, the matrix-free route, products of Gaussian measures, block operators,
and signed measures. References are computed summand by summand or by direct quadrature, not through the sum."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.quadrature import gauss


def _setup():
    H1 = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.2)
    H2 = pg.Sobolev(pg.Sphere(1.0), 2, length_scale=0.3)
    E = pg.Euclidean(3, weights=[1.0, 2.0, 3.0])
    X = pg.DirectSum({"disk": H1, "sphere": H2, "terms": E}, weights={"disk": 2.0, "sphere": 0.5})
    return X, H1, H2, E


def _unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def _data(n, seed=0):
    """n functionals: a chord of the disk (times -1.5), a signed sum of two sphere points, coefficients on R^3; datum i
    leaves out the disk when i % 3 == 1 and the sphere when i % 3 == 2."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a, b = rng.uniform(0, 2 * np.pi, 2)
        seg = pg.Segment((math.cos(a), math.sin(a)), (math.cos(b + 2), math.sin(b + 2)))
        p, q = _unit(rng.normal(size=3)), _unit(rng.normal(size=3))
        d = {"terms": rng.normal(size=3)}
        if i % 3 != 1:
            d["disk"] = -1.5 * pg.LebesgueMeasure(seg)
        if i % 3 != 2:
            d["sphere"] = 0.7 * pg.DiracMeasure(p) - 0.2 * pg.DiracMeasure(q)
        out.append(d)
    return out


def _f(x, y):
    return np.sin(2 * x) * np.cos(3 * y) + x * y


def _g(x, y, z):
    return z + 0.5 * x * y


def _direct(d):
    """sum_k l_k(x_k) for x = (f, g, v) by quadrature and point values, without pygeoinf0's representers."""
    v = np.array([0.3, -1.0, 2.0])
    out = float(d["terms"] @ v)
    if "disk" in d:
        c, mu = d["disk"].terms[0]
        p, q = mu.set.endpoints
        t, w = gauss(64)
        P = p + 0.5 * (t[:, None] + 1) * (q - p)
        out += c * 0.5 * np.linalg.norm(q - p) * (w @ _f(*P.T))
    if "sphere" in d:
        out += sum(c * _g(*m.x0) for c, m in d["sphere"].terms)
    return out


def test_inner_product_is_the_weighted_sum():
    X, H1, H2, E = _setup()
    a, b = H1.riesz(pg.Segment((-0.5, 0), (0.5, 0.3))), H1.riesz(np.array([0.2, 0.1]))
    p = H2.riesz(np.array([0.0, 0.0, 1.0])) - 2.0 * H2.riesz(_unit([1, 1, 0]))
    x = X({"disk": a, "sphere": p, "terms": E([1.0, 2.0, 3.0])})
    y = X({"disk": b, "terms": E([0.5, 0.0, -1.0])})
    ref = 2.0 * H1.inner(a, b) + 1.0 * E.inner(E([1.0, 2.0, 3.0]), E([0.5, 0.0, -1.0]))
    assert X.inner(x, y) == pytest.approx(ref, rel=1e-13)
    assert x.norm() ** 2 == pytest.approx(2 * a.norm() ** 2 + 0.5 * p.norm() ** 2 + E([1, 2, 3]).norm() ** 2, rel=1e-13)
    assert (x["disk"] - a).norm() == pytest.approx(0, abs=1e-12 * a.norm())
    assert np.array_equal(x["terms"].values, [1, 2, 3]) and np.all(y["sphere"](np.array([0.0, 0.0, 1.0])) == 0)
    z = 2.0 * x - y
    assert z.norm() ** 2 == pytest.approx(X.inner(z, z), rel=1e-13)
    assert X.inner(z, x) == pytest.approx(2 * x.norm() ** 2 - X.inner(y, x), rel=1e-12)
    assert X == pg.DirectSum({"disk": H1, "sphere": H2, "terms": E}, weights={"disk": 2.0, "sphere": 0.5})
    assert X != pg.DirectSum({"disk": H1, "sphere": H2, "terms": E})
    with pytest.raises(AttributeError, match="evaluate a part"):
        x(np.array([0.1, 0.2]))


def test_representers_values_and_gram_by_summand():
    X, H1, H2, E = _setup()
    data = _data(9)
    K = X.riesz(data)
    G = pg.Functionals(K)
    x = X({"disk": H1(_f), "sphere": H2(_g), "terms": E([0.3, -1.0, 2.0])})
    assert np.allclose(G(x).values, [_direct(d) for d in data], rtol=1e-9, atol=1e-9)
    gram = np.zeros((9, 9))                                      # sum_k G_k / w_k, each G_k from its summand alone
    for key, w in (("disk", 2.0), ("sphere", 0.5)):
        idx = [i for i, d in enumerate(data) if key in d]
        Hk = X.summand(key)
        Fk = Hk.riesz([data[i][key] for i in idx])
        gram[np.ix_(idx, idx)] += Hk.inner(Fk, Fk) / w
    A = np.array([d["terms"] for d in data])
    gram += A @ np.linalg.solve(np.diag([1.0, 2.0, 3.0]), A.T)
    assert np.allclose(X.inner(K, K), gram, rtol=1e-12, atol=1e-14)
    assert np.allclose(G.gram(), gram, rtol=1e-12, atol=1e-14)
    assert X.part(K, "disk").norms()[1] == 0.0 and X.part(K, "sphere").norms()[2] == 0.0
    k3 = X.riesz(data[3])                                        # one dict: one element, equal to member 3
    assert (k3 - K[3]).norm() <= 1e-12 * k3.norm()


def test_adjoint_and_posterior_against_the_data_space_formulas():
    X, H1, H2, E = _setup()
    data = _data(12, seed=1)
    sigma = 0.1
    D = pg.Euclidean(12, covariance=sigma ** 2)
    G = pg.Functionals(X.riesz(data), into=D)
    rng = np.random.default_rng(2)
    x = X({"disk": H1(_f), "sphere": H2(_g), "terms": E([0.3, -1.0, 2.0])})
    y = D(rng.normal(size=12))
    assert D.inner(G(x), y) == pytest.approx(X.inner(x, G.adjoint(y)), rel=1e-10)
    d = D(G(x).values + sigma * rng.normal(size=12))
    alpha = 0.3
    post = pg.probability.cylindrical(X, alpha=alpha).condition(G, d)
    A = G.gram()
    S = A + alpha * sigma ** 2 * np.eye(12)
    assert np.allclose(G(post.mean).values, A @ np.linalg.solve(S, d.values), rtol=1e-9, atol=1e-12)
    u = {"disk": pg.DiracMeasure(np.array([0.1, -0.2])), "sphere": pg.DiracMeasure(_unit([0, 1, 1])), "terms": [1, 0, 0]}
    ru = X.riesz(u)
    b = X.inner(G.kernels, ru)
    var = (ru.norm() ** 2 - b @ np.linalg.solve(S, b)) / alpha
    assert post.variance(u) == pytest.approx(var, rel=1e-9)
    assert post.std([u, u]) == pytest.approx([math.sqrt(var)] * 2, rel=1e-9)
    prior_var = (H1.riesz(np.array([0.1, -0.2])).norm() ** 2 / 2.0 + H2.riesz(_unit([0, 1, 1])).norm() ** 2 / 0.5
                 + 1.0) / alpha                                  # a . W^-1 a = 1 for a = e_0
    assert pg.probability.cylindrical(X, alpha=alpha).variance(u) == pytest.approx(prior_var, rel=1e-10)


def test_matrix_free_route_agrees_with_the_dense_one():
    X, H1, H2, E = _setup()
    data = _data(30, seed=3)
    D = pg.Euclidean(30, covariance=0.05 ** 2)
    rng = np.random.default_rng(4)
    G = pg.Functionals(X.riesz(data), into=D)
    d = D(G(X({"disk": H1(_f), "sphere": H2(_g), "terms": E([1.0, 0, 0])})).values + 0.05 * rng.normal(size=30))
    a_dense = pg.solvers.damping_for_misfit(G @ G.adjoint, d, d.space.dim)
    pred_dense = G(pg.probability.cylindrical(X, alpha=a_dense).condition(G, d).mean).values
    with pg.settings.using(dense_max=10, accuracy=1e-8):
        G2 = pg.Functionals(X.riesz(data), into=D)
        A = G2 @ G2.adjoint
        assert not isinstance(A, pg.Matrix) and "without forming" in str(A.claims)
        a_free = pg.solvers.damping_for_misfit(A, d, d.space.dim)
        assert "Lanczos" in str(a_free.claims)
        post = pg.probability.cylindrical(X, alpha=a_free).condition(G2, d)
        pred_free = G2(post.mean).values
        assert post.std(data[:2]).shape == (2,)
    assert a_free == pytest.approx(a_dense, rel=1e-6)
    assert np.sqrt(np.mean((pred_free - pred_dense) ** 2)) <= 1e-6 * 0.05


def test_product_of_gaussian_measures():
    M = pg.L2(pg.Disk(1.0))
    E = pg.Euclidean(2, covariance=[4.0, 0.25])
    X = pg.DirectSum({"field": M, "terms": E}, weights={"terms": 3.0})
    mu = pg.GaussianMeasure.product({"field": pg.probability.random_features(M), "terms": E.standard_gaussian()}, space=X)
    v = E([1.0, -2.0])
    u = X.embed("terms", v)
    assert mu.variance(u) == pytest.approx(3.0 ** 2 * E.standard_gaussian().variance(v), rel=1e-12)
    f = M(lambda x, y: x + y ** 2)
    assert mu.variance(X.embed("field", f)) == pytest.approx(pg.probability.random_features(M).variance(f), rel=1e-12)
    S = mu.sample(4000, seed=5)
    vals = X.part(S, "terms").values
    assert np.allclose(vals.var(axis=0), [4.0, 0.25], rtol=0.1)            # coordinates N(0, C_e) of E's Gaussian
    assert "product" in str(mu.claims) and "[derived] the covariance is trace class" in str(mu.claims)


def test_block_diagonal_operators_and_projections():
    E1, E2 = pg.Euclidean(2, weights=[1.0, 3.0]), pg.Euclidean(3)
    X = pg.DirectSum([E1, E2], weights=[2.0, 0.5])
    Y = pg.DirectSum([E2, E1], weights=[1.5, 4.0])
    rng = np.random.default_rng(6)
    A = pg.BlockDiagonal(X, [pg.Matrix(rng.normal(size=(3, 2)), E1, E2), pg.Matrix(rng.normal(size=(2, 3)), E2, E1)], Y)
    x = X([E1(rng.normal(size=2)), E2(rng.normal(size=3))])
    y = Y([E2(rng.normal(size=3)), E1(rng.normal(size=2))])
    assert Y.inner(A(x), y) == pytest.approx(X.inner(x, A.adjoint(y)), rel=1e-12)
    P = X.projection(1)
    u = E2(rng.normal(size=3))
    assert E2.inner(P(x), u) == pytest.approx(X.inner(x, P.adjoint(u)), rel=1e-12)


def test_signed_measures_are_combinations_of_representers():
    H = pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.3)
    poly = pg.LebesgueMeasure(pg.Polyline([[0.5, 0, 0], [0, 0.4, 0.1], [-0.3, 0.2, 0.5]])).with_density([1.0, 2.0])
    p = pg.DiracMeasure(np.array([0.1, 0.2, 0.3]))
    nu = -2.0 * poly + 0.5 * p
    assert isinstance(nu, pg.SignedMeasure) and len(nu.terms) == 2 and "signed" in str(nu.claims)
    k = H.riesz(nu)
    ref = -2.0 * H.riesz(poly) + 0.5 * H.riesz(p)
    f = H(lambda x, y, z: np.cos(x + 2 * y) * z)
    assert H.inner(k, f) == pytest.approx(H.inner(ref, f), rel=1e-12)
    assert k.norm() == pytest.approx(ref.norm(), rel=1e-12)
    assert (k - ref).norm() <= 1e-7 * ref.norm()            # ||k - ref||^2 by cancellation: sqrt(eps) relative
    S = pg.Sobolev(pg.Sphere(2.0), 2, length_scale=0.5)
    q1, q2 = 2.0 * _unit([1, 0, 1]), 2.0 * _unit([0, 1, -1])
    ks = S.riesz(3.0 * pg.DiracMeasure(q1) - pg.DiracMeasure(q2))
    f = S(lambda x, y, z: x * z + y)
    assert S.inner(ks, f) == pytest.approx(3.0 * (q1[0] * q1[2] + q1[1]) - (q2[0] * q2[2] + q2[1]), rel=1e-6)
    assert sum([poly, p]).terms[1][1] is p
