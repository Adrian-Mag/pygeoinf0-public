"""Sobolev constructions on intervals (plan 006, P2): intrinsic forms with boundary data, spectral powers of their
operators, and the restriction, checked against independent references (closed forms; 30-digit values from
experiments/EXP-026-sobolev-constructions/refs_kernels_1d.py, which solves the boundary conditions in mpmath and
cross-checks them with finite elements; eigen-series), with the data-space machinery on each."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import (NormUnavailable, PointEvaluationUndefined, Unsupported, ViolatesBoundaryCondition,
                                   ZeroFunctional)

I = pg.Interval("[0, 1]")


def K(H, xs, ys):
    """The reproducing kernel by energies of Dirac representers."""
    return H.inner(H.riesz([pg.DiracMeasure(x) for x in xs]), H.riesz([pg.DiracMeasure(y) for y in ys]))


def form(l, **kw):
    return pg.SobolevForm(I, 1, length_scale=l, **kw)


def closed(name, l, X, Y, beta=None):
    lo, hi = np.minimum(X, Y), np.maximum(X, Y)
    if name == "neumann":
        return np.cosh(lo / l) * np.cosh((1 - hi) / l) / (l * np.sinh(1 / l))
    if name == "dirichlet":
        return np.sinh(lo / l) * np.sinh((1 - hi) / l) / (l * np.sinh(1 / l))
    if name == "mixed":
        return np.sinh(lo / l) * np.cosh((1 - hi) / l) / (l * np.cosh(1 / l))
    g = beta / l
    p0 = lambda t: np.cosh(t / l) + g * np.sinh(t / l)                            # noqa: E731
    return p0(lo) * p0(1 - hi) / (l * ((1 + g * g) * np.sinh(1 / l) + 2 * g * np.cosh(1 / l)))


@pytest.mark.parametrize("l", [0.05, 0.2, 1.0])
def test_order_one_kernels_are_the_closed_forms(l):
    x = np.array([0.0, 0.07, 0.5, 0.93, 1.0])
    X, Y = np.meshgrid(x, x, indexing="ij")
    xi = x[1:-1]                                     # Dirichlet: no points on the Dirichlet parts (ZeroFunctional)
    for name, H, xs in [("neumann", form(l).space(), x), ("dirichlet", form(l, dirichlet="all").space(), xi),
                        ("mixed", form(l, dirichlet="left").space(), x[1:]),
                        ("robin", form(l, robin=0.03).space(), x)]:
        Xs, Ys = np.meshgrid(xs, xs, indexing="ij")
        ref = closed(name, l, Xs, Ys, 0.03)
        assert np.allclose(K(H, xs, xs), ref, rtol=1e-12, atol=1e-14 * ref.max()), name
        k = H.riesz(pg.DiracMeasure(0.5))                                     # values of the representer: same kernel
        assert np.allclose(k(np.array([0.0, 0.3, 1.0])), closed(name, l, np.array([0.0, 0.3, 1.0]), 0.5, 0.03),
                           rtol=1e-12, atol=1e-14)
    res = pg.Sobolev(I, 1, length_scale=l)                                    # Robin with beta = l IS the restriction
    rob = form(l, robin=l).space()
    assert rob == res and hash(rob) == hash(res)
    assert np.allclose(K(rob, x, x), np.exp(-np.abs(X - Y) / l) / (2 * l), rtol=1e-12)
    assert form(l).space() == pg.Sobolev(form(l).operator(), 1) != res      # second representation theorem


# 30-digit references (EXP-026 refs_kernels_1d.py: mpmath boundary-condition solves, FEM-checked to 1e-8)
REFS = {  # (construction, l): {(x, y): G(x, y)}
    ("intrinsic2", 0.2): {(0.0, 0.0): 3.33632813462877523915699074666, (0.0, 0.3): 1.30861467124656019173164412196,
                          (0.5, 0.75): 0.980837775909690416193077139797, (0.05, 0.0): 2.92392667251058401832837471961},
    ("neumann2", 0.2): {(0.0, 0.0): 2.50249721257257207179036868514, (0.0, 0.3): 1.39971291634463170401414013735,
                        (0.5, 0.75): 0.965078290705591376768096443632, (0.05, 0.0): 2.43631376848100419938626647036},
    ("robin2", 0.2): {(0.0, 0.0): 1.89168274155308358228938016385, (0.0, 0.3): 1.15249897248133416192916262317,
                      (0.5, 0.75): 0.915819608134051251090800342551, (0.05, 0.0): 1.89688168713593154159245407031},
    ("dirichlet2", 1.0): {(0.05, 0.3): 0.00248564858261466342328084657144,
                          (0.5, 0.75): 0.0117785698324797336655454008183, (0.3, 0.3): 0.0122832480615215096528205756914},
    ("mixed2", 0.1): {(1.0, 1.0): 4.99999956715774075217190565848, (0.3, 0.3): 2.45665280315888901043060647917,
                      (0.75, 0.75): 2.60105675156967272538146113111},
}


def order_two(name, l):
    if name == "intrinsic2":
        return pg.SobolevForm(I, 2, length_scale=l).space()
    op = {"neumann2": form(l), "robin2": form(l, robin=0.03), "dirichlet2": form(l, dirichlet="all"),
          "mixed2": form(l, dirichlet="left")}[name].operator()
    return pg.Sobolev(op, 2)


@pytest.mark.parametrize("name, l", list(REFS))
def test_order_two_kernels_against_references(name, l):
    H = order_two(name, l)
    for (x, y), v in REFS[(name, l)].items():
        assert K(H, [x], [y])[0, 0] == pytest.approx(v, rel=1e-12), (x, y)
    assert np.isclose(K(pg.Sobolev(I, 2, length_scale=l), [0.2], [0.6])[0, 0],
                      (1 + 0.4 / l) * math.exp(-0.4 / l) / (4 * l), rtol=1e-13)            # restriction: Matern


def test_order_two_constructions_differ_and_satisfy_their_natural_conditions():
    l, y, h = 0.2, 0.4, 1e-3
    xs = np.array([0, h, 2 * h, 3 * h, 4 * h])                           # one-sided differences at the left end
    D1 = np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4]) / h
    D2 = np.array([35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12]) / h ** 2
    D3 = np.array([-5 / 2, 9, -12, 7, -3 / 2]) / h ** 3
    vals = {}
    for name in ("intrinsic2", "neumann2", "restriction2"):
        H = pg.Sobolev(I, 2, length_scale=l) if name == "restriction2" else order_two(name, l)
        g = H.riesz(pg.DiracMeasure(y))(xs)
        vals[name] = g
        u, u1, u2, u3 = g[0], D1 @ g, D2 @ g, D3 @ g
        if name == "intrinsic2":                                          # u'' = 0, 2 u' = l^2 u'''
            assert abs(u2) < 1e-4 * abs(u) / l ** 2 and abs(2 * u1 - l * l * u3) < 1e-3 * abs(u) / l
        elif name == "neumann2":                                          # u' = 0, u''' = 0 (even reflection)
            assert abs(u1) < 1e-6 * abs(u) / l and abs(u3) < 1e-3 * abs(u) / l ** 3
        else:                                                             # the jet of a decaying extension
            assert abs(l * l * u2 - 2 * l * u1 + u) < 1e-4 * abs(u)
    assert abs(vals["intrinsic2"][0] - vals["neumann2"][0]) > 0.01 * vals["neumann2"][0]
    assert abs(vals["intrinsic2"][0] - vals["restriction2"][0]) > 0.01 * vals["restriction2"][0]


def eigen_series(s, l, x, y, kind, N=400_000):
    n = np.arange(N)[:, None]
    if kind == "neumann":
        mu, f = (np.pi * n) ** 2, lambda t: np.where(n == 0, 1.0, math.sqrt(2) * np.cos(np.pi * n * t))   # noqa
    elif kind == "dirichlet":
        mu, f = (np.pi * (n + 1)) ** 2, lambda t: math.sqrt(2) * np.sin(np.pi * (n + 1) * t)   # noqa: E731
    else:
        mu, f = (np.pi * (n + 0.5)) ** 2, lambda t: math.sqrt(2) * np.sin(np.pi * (n + 0.5) * t)   # noqa: E731
    return np.sum((1 + l * l * mu) ** -s * f(np.asarray(x)[None]) * f(np.asarray(y)[None]), axis=0)


@pytest.mark.parametrize("kind, kw", [("neumann", {}), ("dirichlet", dict(dirichlet="all")),
                                      ("mixed", dict(dirichlet="left"))])
def test_fractional_spectral_spaces_by_images_match_eigen_series(kind, kw):
    s, l = 1.3, 0.2
    H = pg.Sobolev(form(l, **kw).operator(), s)
    x, y = np.array([0.1, 0.42, 0.9, 0.42]), np.array([0.42, 0.42, 0.97, 0.9])
    got = np.diag(K(H, x, y))
    assert np.allclose(got, eigen_series(s, l, x, y, kind), rtol=1e-7, atol=1e-9)
    J = pg.Interval(0.2, 0.5)                                           # an average: the representer's own integral
    k = H.riesz(J)
    t, w = np.polynomial.legendre.leggauss(64)
    xq = 0.35 + 0.15 * t
    assert H.norm(k) ** 2 == pytest.approx(0.15 * w @ k(xq), rel=1e-9)
    with pytest.raises(NormUnavailable):
        H(lambda x: np.sin(np.pi * x)).norm()
    if kind == "neumann":                                               # integer s: images and finite rank agree
        Hi = pg.Sobolev(form(l).operator(), 2)
        assert np.allclose(K(Hi, x, y), K(pg.Sobolev(form(l).operator(), 2.0), x, y), rtol=1e-13)


def test_identity_between_restriction_and_intrinsic_norms_and_the_equivalence_constant():
    l = 0.2
    Hres, Hint = pg.Sobolev(I, 1, length_scale=l), form(l).space()
    f = lambda x: np.exp(x) * np.cos(3 * x)                                  # noqa: E731
    nr, ni = Hres(f).norm() ** 2, Hint(f).norm() ** 2
    assert nr == pytest.approx(ni + l * (f(0.0) ** 2 + f(1.0) ** 2), rel=1e-12)
    # the intrinsic norm of a formula: int f^2 + l^2 f'^2 (closed form for cos)
    assert Hint(lambda x: np.cos(np.pi * x)).norm() ** 2 == pytest.approx(0.5 + 0.5 * (l * np.pi) ** 2, rel=1e-12)
    assert Hres.constant(1.0).norm() ** 2 == pytest.approx(1 + 2 * l, rel=1e-13)
    # sup ||u||_res^2 / ||u||_int^2 = 1 + coth(1/(2l)), attained by K_N(., 0) + K_N(., 1)
    kN = Hint.riesz([pg.DiracMeasure(0.0), pg.DiracMeasure(1.0)]).sum()
    u_res = Hres(lambda x: kN(x))
    ratio = u_res.norm() ** 2 / kN.norm() ** 2
    assert ratio == pytest.approx(1 + 1 / math.tanh(1 / (2 * l)), rel=1e-9)
    # order 2: the restriction norm of x^2 is a_2 + exterior jet energies 2 l (u^2 + l u d_n u + l^2 (d_n u)^2)
    H2 = pg.Sobolev(I, 2, length_scale=l)
    ref = 1 / 5 + 8 * l * l / 3 + 4 * l ** 4 + 2 * l * (1 + 2 * l + 4 * l * l)
    assert H2(lambda x: x * x).norm() ** 2 == pytest.approx(ref, rel=1e-12)
    # spectral Neumann s = 2: ||(1 - l^2 d^2) u||^2 on {u' = 0 at the ends}
    HN2 = pg.Sobolev(form(l).operator(), 2)
    assert HN2(lambda x: np.cos(np.pi * x)).norm() ** 2 == pytest.approx(0.5 * (1 + (l * np.pi) ** 2) ** 2, rel=1e-12)
    with pytest.raises(ViolatesBoundaryCondition):
        HN2(lambda x: x * x).norm()


def test_representers_adjoints_and_minimum_norm_solutions_differ_between_constructions():
    l = 0.15
    rng = np.random.default_rng(1)
    pts = np.sort(rng.uniform(0.05, 0.95, 6))
    measures = [pg.DiracMeasure(p) for p in pts] + [pg.LebesgueMeasure(pg.Interval(0.0, 0.2)),
                                                     pg.LebesgueMeasure(pg.Interval(0.6, 1.0))]
    d = np.array([1.0, -0.5, 0.3, 0.8, 0.0, -1.0, 0.1, 0.05])
    spaces = {"restriction": pg.Sobolev(I, 1, length_scale=l), "neumann": form(l).space(),
              "dirichlet": form(l, dirichlet="all").space(), "neumann2": pg.Sobolev(form(l).operator(), 2),
              "intrinsic2": pg.SobolevForm(I, 2, length_scale=l).space(),
              "robin1.5": pg.Sobolev(form(l, robin=0.1).operator(), 3)}
    sols = {}
    for name, H in spaces.items():
        D = pg.Euclidean(len(measures))
        G = pg.Functionals(H.riesz(measures), into=D)
        GG = G @ G.adjoint
        M = GG.matrix()
        assert np.allclose(M, M.T, rtol=1e-12) and np.linalg.eigvalsh(M).min() > 0, name
        y = D(rng.standard_normal(len(measures)))
        m1 = G.adjoint(y)
        assert G(m1) @ y == pytest.approx(m1 @ G.adjoint(y), rel=1e-12)
        m = G.adjoint(GG.inv(D(d)))                                    # the minimum-norm solution
        assert np.allclose(G(m).values, d, atol=1e-9 * np.abs(d).max())
        h = H.riesz(pg.DiracMeasure(0.5)) - G.adjoint(GG.inv(G(H.riesz(pg.DiracMeasure(0.5)))))   # G h = 0
        assert np.allclose(G(h).values, 0, atol=1e-10)
        assert (m + 0.3 * h).norm() > m.norm() and m.inner(h) == pytest.approx(0, abs=1e-10 * m.norm() * h.norm())
        # the reproducing property: <k_J, f> = int_J f, for f a representer (smooth on J) and for a formula
        t, wq = np.polynomial.legendre.leggauss(32)
        f = H.riesz(pg.DiracMeasure(0.7))
        assert H.inner(H.riesz(pg.Interval(0.1, 0.4)), f) == pytest.approx(0.15 * wq @ f(0.25 + 0.15 * t), rel=1e-12)
        if name != "robin1.5":                                            # (its elements satisfy the Robin condition)
            f = H(lambda x: np.sin(np.pi * x) ** 2) if name in ("dirichlet", "neumann2") else H(np.cos)
            assert H.inner(H.riesz(pg.Interval(0.1, 0.4)), f) == pytest.approx(0.15 * wq @ f.rep(0.25 + 0.15 * t),
                                                                              rel=1e-12)
        sols[name] = m
    xs = np.linspace(0, 1, 101)
    vals = {k: v(xs) for k, v in sols.items()}
    names = list(vals)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert np.max(np.abs(vals[names[i]] - vals[names[j]])) > 1e-3, (names[i], names[j])
    assert abs(vals["dirichlet"][0]) < 1e-12 and abs(vals["dirichlet"][-1]) < 1e-12


def test_prior_variances_at_the_ends_and_refusals():
    l = 0.1
    k0 = 1 / (2 * l)
    assert pg.probability.cylindrical(form(l).space()).variance(0.0) == pytest.approx(2 / math.tanh(1 / l) * k0, rel=1e-13)
    assert pg.probability.cylindrical(pg.Sobolev(I, 1, length_scale=l)).variance(1.0) == pytest.approx(k0, rel=1e-13)
    HD = form(l, dirichlet="all").space()
    assert pg.probability.cylindrical(HD).variance(1e-3) == pytest.approx(closed("dirichlet", l, 1e-3, 1e-3), rel=1e-10)
    assert HD.riesz(pg.DiracMeasure(0.0)).norm() == 0 and pg.probability.cylindrical(HD).std(np.array([0.0, 1.0])) == pytest.approx(0)
    with pytest.raises(ZeroFunctional):                                  # a datum where every element vanishes
        pg.Functionals(HD.riesz([pg.DiracMeasure(0.3), pg.DiracMeasure(1.0)]))
    with pytest.raises(ViolatesBoundaryCondition):
        HD(np.cos).norm()
    assert HD(lambda x: np.sin(np.pi * x)).norm() > 0
    with pytest.raises(PointEvaluationUndefined):
        pg.Sobolev(form(l, dirichlet="all").operator(), 0.5).riesz(pg.DiracMeasure(0.5))
    with pytest.raises(Unsupported, match="Robin"):
        pg.Sobolev(form(l, robin=0.5).operator(), 1.5)
    claims = repr(form(l, dirichlet="left").space().claims)
    assert "Kato" in claims and "u = 0" in claims and "Dirichlet at left" in repr(form(l, dirichlet="left").space())
    assert "rank-2 boundary correction" in claims and "condition number" in claims
    assert "reflection images" in repr(pg.Sobolev(form(l).operator(), 1.5).claims)
    bound = pg.Inclusion(form(l).space(), pg.Continuous(I)).claims
    assert any(f"{math.sqrt(2 / math.tanh(1 / l) * k0):.6g}" in c.statement for c in bound)


@pytest.mark.parametrize("make, points", [(lambda l: form(l, robin=0.05).space(), False),
                                          (lambda l: pg.Sobolev(form(l, dirichlet="left").operator(), 1.5), True)])
def test_inference_through_the_matrix_free_route(make, points):
    l = 0.1
    H = make(l)
    rng = np.random.default_rng(3)
    truth = lambda x: np.sin(2 * np.pi * x) * x                                    # noqa: E731
    t, wq = np.polynomial.legendre.leggauss(16)
    if points:                                       # point values (the fractional kernel is costly on segments)
        xs = np.sort(rng.uniform(0.01, 1.0, 400))
        measures, clean = [pg.DiracMeasure(x) for x in xs], truth(xs)
    else:                                            # 400 averages: H-matrix with far (ACA) blocks above dense_max
        w = rng.uniform(0.002, 0.01, 400)
        a = rng.uniform(0.01, 0.99 - w)
        measures = [pg.LebesgueMeasure(pg.Interval(float(x), float(x + y))) for x, y in zip(a, w)]
        clean = np.array([0.5 * (y) * wq @ truth(x + 0.5 * y * (t + 1)) for x, y in zip(a, w)])
    sigma = 0.01
    D = pg.Euclidean(len(measures), covariance=sigma ** 2)
    d = D(clean + sigma * rng.standard_normal(len(measures)))
    out = {}
    for dense_max in (10_000, 50):
        with pg.settings.using(dense_max=dense_max, accuracy=1e-6):
            G = pg.Functionals(H.riesz(measures), into=D)
            alpha = pg.solvers.damping_for_misfit(G @ G.adjoint, d, d.space.dim)
            post = pg.probability.cylindrical(H, alpha=alpha).condition(G, d)
            out[dense_max] = (alpha, post.mean(np.array([0.0, 0.3, 0.77])), post.std(np.array([0.0, 0.5, 1.0])))
            assert ("Lanczos" in str(alpha.claims)) == (dense_max == 50)
            if dense_max == 50:
                assert G.kernels.realisation.kx.atoms.hmatrix.stats["far"] > 0
    assert out[50][0] == pytest.approx(out[10_000][0], rel=1e-5)
    assert np.allclose(out[50][1], out[10_000][1], rtol=1e-5, atol=1e-6)
    assert np.allclose(out[50][2], out[10_000][2], rtol=1e-4, atol=1e-12)    # (a cancelling difference of variances)


def test_exterior_energy_forms_and_the_graph_norm_space():
    from pygeoinf0.realisations.rkhs.green import exterior_form
    l = 0.3                                                      # sympy (EXP-026 refs_facts.py), jets (u, d_n u, ...)
    assert np.allclose(exterior_form(1, l), [[l]], rtol=1e-14)
    assert np.allclose(exterior_form(2, l), [[2 * l, l ** 2], [l ** 2, 2 * l ** 3]], rtol=1e-13)
    assert np.allclose(exterior_form(3, l), [[3 * l, 3 * l ** 2, l ** 3], [3 * l ** 2, 8 * l ** 3, 3 * l ** 4],
                                             [l ** 3, 3 * l ** 4, 3 * l ** 5]], rtol=1e-12)
    assert pg.Sobolev(I, 3, length_scale=l).constant(1.0).norm() ** 2 == pytest.approx(1 + 6 * l, rel=1e-13)
    A = form(0.2, robin=0.1).operator()
    DA = A.domain_space()                                       # D(A) with ||A u||: the graph-norm space
    assert DA == pg.Sobolev(A, 2) and DA != form(0.2, robin=0.1).space()
    k = DA.riesz(pg.DiracMeasure(0.3))                          # its kernel (Green function of A^2) against its norm:
    u = DA(lambda x: k(x), breaks=[0.3])                        # ||k||^2 = K(0.3, 0.3) = ||(1 - l^2 d^2) k||^2
    assert u.norm() ** 2 == pytest.approx(k.norm() ** 2, rel=1e-9)
