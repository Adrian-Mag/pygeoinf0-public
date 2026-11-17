"""Spaces on a layered ball (plan 007, P3): L2 on shells, L2 of the ball as the direct sum over layers, the domain
independence of restriction energies, layered Sobolev fields (representers of rays, evaluation with sides), and the data
space of a layered model with interface and Euclidean terms through Functionals, dense and matrix-free. References are
analytic, from spaces on single layers, or from the ball's own spaces."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg


def _planet():
    return pg.LayeredBall([0.3, 0.6, 0.9, 1.0], names=["core", "lower", "upper", "crust"],
                          interfaces=["cmb", "mid", "moho"])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _f(x, y, z):
    return np.exp(0.7 * x) * (1 + y * z) + np.sin(3 * z)


def test_l2_on_a_shell():
    S = pg.Shell(0.5, 1.0)
    M = pg.L2(S)
    assert M.realisation == "tensor"
    assert M.integrate(M(lambda x, y, z: x ** 2)) == pytest.approx(4 * math.pi / 15 * (1 - 0.5 ** 5), rel=1e-12)
    assert M.constant(1.0).norm() ** 2 == pytest.approx(S.volume, rel=1e-12)


def test_lebesgue_on_the_layers_is_l2_of_the_ball():
    P = _planet()
    V = pg.PiecewiseSpace.lebesgue(P)
    x = V({n: _f for n in P.names})
    ball = pg.L2(pg.Ball(1.0))(_f)
    assert x.norm() == pytest.approx(ball.norm(), rel=1e-10)                       # L2(B) = (+)_k L2(Omega_k)
    y = V({n: (lambda X, Y, Z: X) for n in P.names})
    assert y.norm() ** 2 == pytest.approx(4 * math.pi / 15, rel=1e-12)
    assert V.inner(x, y) == pytest.approx(pg.L2(pg.Ball(1.0)).inner(ball, pg.L2(pg.Ball(1.0)).x), rel=1e-10)


def test_one_field_functional_of_a_list_is_the_functionals_into_rn():
    """M.functional([mu, ...]) (one field; plan 071 P3, review item 11): pg.Functionals into R^n, row i the functional
    of mu_i."""
    P = _planet()
    V = pg.PiecewiseSpace.lebesgue(P)
    x = V({n: _f for n in P.names})
    mus = [pg.LebesgueMeasure(pg.Ball(0.25)), pg.LebesgueMeasure(pg.Shell(0.5, 0.9))]
    T = V.functional(mus)
    assert isinstance(T, pg.Functionals) and T.codomain.dim == 2
    assert np.array_equal(T(x).values, [V.functional(m)(x) for m in mus])

def test_restriction_energies_do_not_depend_on_the_domain():
    shell, ball = pg.Sobolev(pg.Shell(0.6, 0.9), 2, length_scale=0.2), pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.2)
    rays = [pg.LebesgueMeasure(pg.Polyline([[0.85, 0, 0], [0.55, 0.45, 0.1], [0.1, 0.8, 0.2]])).with_density([1, 3]),
            pg.LebesgueMeasure(pg.Segment([0.0, -0.7, 0.3], [0.5, -0.5, 0.4])),
            pg.DiracMeasure(np.array([0.0, 0.0, -0.75]))]
    assert shell.construction.kind == "restriction" and pg.Shell(0.6, 0.9).boundary_parts == ("inner", "outer")
    Ks, Kb = shell.riesz(rays), ball.riesz(rays)
    assert np.allclose(shell.inner(Ks, Ks), ball.inner(Kb, Kb), rtol=1e-13, atol=0)


def test_a_density_on_a_one_piece_polyline_is_kept():
    H = pg.Sobolev(pg.Ball(1.0), 2, length_scale=0.3)
    p, q = np.array([0.1, -0.2, 0.3]), np.array([-0.4, 0.5, 0.2])
    mu = pg.LebesgueMeasure(pg.Polyline([p, q])).with_density([2.5])
    t, g = np.polynomial.legendre.leggauss(64)
    ref = 2.5 * 0.5 * np.linalg.norm(q - p) * (g @ _f(*(p + 0.5 * (t[:, None] + 1) * (q - p)).T))
    assert H.inner(H.riesz(mu), H(_f)) == pytest.approx(ref, rel=1e-10)
    assert H.riesz(mu).norm() == pytest.approx(2.5 * H.riesz(pg.Segment(p, q)).norm(), rel=1e-13)


def test_layered_sobolev_representers_and_gram():
    P = _planet()
    V = pg.PiecewiseSpace.sobolev(P, 2, length_scale=[0.1, 0.15, 0.2, 0.05])
    rays = [pg.LebesgueMeasure(pg.Polyline([[-0.98, 0.1, 0.0], [0.0, 0.2, 0.1], [0.7, -0.6, 0.2]])).with_density([2, 1]),
            pg.LebesgueMeasure(pg.Polyline([[0.0, 0.0, 0.99], [0.1, 0.35, 0.3], [0.6, 0.6, -0.3]])).with_density([1, 1])]
    K = V.riesz(rays)
    ref = np.zeros((2, 2))
    for name in P.names:
        H = V.summand(name)
        here = [i for i, m in enumerate(rays) if name in P.split(m)]
        F = H.riesz([P.split(rays[i])[name] for i in here])
        ref[np.ix_(here, here)] += H.inner(F, F)
    assert np.allclose(V.inner(K, K), ref, rtol=1e-13, atol=0)
    t, g = np.polynomial.legendre.leggauss(64)                 # <k_ray, f> = int f w ds, densities included
    x = V({n: _f for n in P.names})
    for i, mu in enumerate(rays):
        Q, w = mu.set._P, mu.weights
        ref_i = sum(d * 0.5 * np.linalg.norm(b - a) * (g @ _f(*(a + 0.5 * (t[:, None] + 1) * (b - a)).T))
                    for a, b, d in zip(Q[:-1], Q[1:], w))
        assert V.inner(K[i], x) == pytest.approx(ref_i, rel=1e-10)
    x = np.array([0.0, 0.25, 0.45])                              # a point in "lower"
    assert V.evaluate(K[1], x) == pytest.approx(V.summand("lower").riesz(P.split(rays[1])["lower"])(x), rel=1e-13)
    W = pg.PiecewiseSpace.sobolev(P, 2, length_scale=0.1, pieces=["upper", "crust"])
    with pytest.raises(pg.errors.SupportOutsideDomain, match="no summand"):
        W.riesz(rays[0])


def test_evaluation_with_sides():
    P = _planet()
    V = pg.PiecewiseSpace.from_pieces(P, {n: pg.L2(P[n]) for n in P.names})
    x = V({n: float(k) for k, n in enumerate(P.names)})
    assert V.evaluate(x, np.array([[0.1, 0, 0], [0, 0.7, 0], [0, 0, 0.95]])) == pytest.approx([0.0, 2.0, 3.0])
    on = np.array([0.0, 0.0, 0.9])
    assert V.evaluate(x, on, side="below") == pytest.approx(2.0) and V.evaluate(x, on, side="above") == pytest.approx(3.0)
    with pytest.raises(ValueError, match="side"):
        V.evaluate(x, on)
    W = pg.PiecewiseSpace.sobolev(P, 2, length_scale=0.2, pieces=["upper", "crust"])
    jump = W.riesz({"crust": pg.DiracMeasure(on), "upper": -1.0 * pg.DiracMeasure(on)})    # value above - below
    y = W({"upper": lambda X, Y, Z: Z, "crust": lambda X, Y, Z: 2 * Z})
    assert W.inner(jump, y) == pytest.approx(0.9, rel=1e-10)
    assert np.isnan(W.evaluate(y, np.array([0.0, 0.0, 0.2])))


def test_section_plot_of_a_layered_field():
    import matplotlib
    matplotlib.use("Agg")
    from pygeoinf0.plotting.layered import section
    P = _planet()
    V = pg.PiecewiseSpace.from_pieces(P, {n: pg.L2(P[n]) for n in P.names})
    x = V({n: float(k) for k, n in enumerate(P.names)})
    U, W, vals = section(x, normal=(0.0, 0.0, 1.0), n=41)
    r = np.hypot(U, W)
    inside = U ** 2 + W ** 2 <= 1.0
    assert np.all(np.isnan(vals[~inside]))
    expect = np.searchsorted(np.array(P.radii), r[inside] - 1e-9, side="left")      # on an interface: side "below"
    assert np.allclose(vals[inside], expect, rtol=0, atol=1e-10)
    ax = x.plot(n=41)
    assert len(ax.patches) == len(P.radii)                                           # interfaces and surface


def _moon_like(n, seed=0):
    """A model space with wave speeds in three layers, interface topography and two 'station terms', and n rays (bent
    polylines between surface points) with interface terms at their moho crossings."""
    P = _planet()
    Vp = pg.PiecewiseSpace.sobolev(P, 2, length_scale=0.15, pieces=["lower", "upper", "crust"])
    Hm = pg.Sobolev(P.interface("moho"), 2, length_scale=0.3)
    X = pg.DirectSum({"vp": Vp, "moho": Hm, "stations": pg.Euclidean(2)}, weights={"moho": 0.5, "stations": 4.0})
    rng = np.random.default_rng(seed)
    data = []
    while len(data) < n:
        a, b = _unit(rng.normal(size=3)), _unit(rng.normal(size=3))
        D = math.acos(np.clip(a @ b, -1, 1))
        if not 0.4 < D < 1.6:
            continue
        m = _unit(a + b) * 0.97 * math.cos(D / 2) ** 0.5                       # the middle vertex, above r = 0.6
        pts = np.array([a, m, b])
        cr = P.crossings(pts)
        terms = [((-1) ** j * 0.3) * pg.DiracMeasure(c.point) for j, c in enumerate(cr) if c.interface == "moho"]
        d = {"vp": -1.5 * pg.LebesgueMeasure(pg.Polyline(pts)).with_density([1.2, 0.8]),
             "stations": np.eye(2)[len(data) % 2]}
        if terms:
            d["moho"] = sum(terms[1:], terms[0])
        data.append(d)
    return X, data


def test_layered_data_space_dense_and_matrix_free():
    X, data = _moon_like(36)
    n = len(data)
    D = pg.Euclidean(n, covariance=0.1 ** 2)
    rng = np.random.default_rng(1)
    d = D(rng.normal(size=n))
    G = pg.Functionals(X.riesz(data), into=D)
    y = D(rng.normal(size=n))
    assert D.inner(G(G.adjoint(y)), y) == pytest.approx(X.norm(G.adjoint(y)) ** 2, rel=1e-10)
    a_dense = pg.solvers.damping_for_misfit(G @ G.adjoint, d, d.space.dim)
    post = pg.probability.cylindrical(X, alpha=a_dense).condition(G, d)
    pred_dense = G(post.mean).values
    pts = {"vp": np.array([[0.0, 0.0, 0.95], [0.5, 0.5, 0.2]]), "moho": 0.9 * _unit([1.0, 2.0, 2.0])}
    std_dense = post.std({"vp": pts["vp"]})
    with pg.settings.using(dense_max=12, accuracy=1e-8):
        G2 = pg.Functionals(X.riesz(data), into=D)
        a_free = pg.solvers.damping_for_misfit(G2 @ G2.adjoint, d, d.space.dim)
        assert "Lanczos" in str(a_free.claims)
        post2 = pg.probability.cylindrical(X, alpha=a_free).condition(G2, d)
        pred_free = G2(post2.mean).values
        std_free = post2.std({"vp": pts["vp"]})
    assert a_free == pytest.approx(a_dense, rel=1e-6)
    assert np.sqrt(np.mean((pred_free - pred_dense) ** 2)) <= 1e-6 * 0.1
    assert std_free == pytest.approx(std_dense, rel=1e-6)
    assert post.std({"moho": pts["moho"]}) < pg.probability.cylindrical(X, alpha=a_dense).std({"moho": pts["moho"]})
    m = post.mean
    assert np.isfinite(m["vp"]["crust"](np.array([0.0, 0.0, 0.95]))) and m["stations"].values.shape == (2,)
