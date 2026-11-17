"""Targets kept symbolic on pieces, expanded in harmonics (plan 044): against a brute quadrature, dense formulas and rotations."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.realisations.series.symbolic_harmonics import is_symbolic
from pygeoinf0.realisations.series.symbolic_harmonics import support_box
from pygeoinf0.realisations.series.radial_direct import harmonic_element

BALL = pg.Ball(1.0)
PLANET = pg.LayeredBall([0.3, 0.55, 1.0], names=["core", "mid", "top"])
L = 4
LAB = np.array([(l, j) for l in range(L + 1) for j in range(2 * l + 1)], int)


def space():
    return pg.PiecewiseSpace(PLANET.partition, {"a": pg.L2(BALL, realisation="radial_harmonics")})


def rows(M, field, measures):
    """The functionals x -> int x_field d(m|pieces), one row per measure (plan 069 P7: pg.Functionals of the parts)."""
    return pg.Functionals(M.riesz([M.parts(field, m) for m in measures]))


def field(M):
    """A band-limited field, linear in r: a table of random coefficients."""
    rng = np.random.default_rng(44)
    radii = np.array([0.0, 0.3, 0.55, 0.8, 1.0])
    c = rng.standard_normal((5, (L + 1) ** 2))
    return M.split({"a": harmonic_element(M.fields["a"], radii, c)}), harmonic_element(M.fields["a"], radii, c), radii


def bump(c, rho):
    return F.on(BALL).bump(centre=c, radius=rho).normalised()


def brute(b, v, radii, c, rho, nr=30, na=80):
    (r0, r1), (t0, t1), (p0, p1) = support_box(type("Q", (), {"set": pg.Ball(rho, centre=c)})(), (0, 0, 0), 0.0, 1.0)
    edges = np.unique(np.r_[r0, r1, [x for x in radii if r0 < x < r1]])
    x, w = np.polynomial.legendre.leggauss(nr)
    xa, wa = np.polynomial.legendre.leggauss(na)
    th, ph = t0 + 0.5 * (t1 - t0) * (xa + 1), p0 + 0.5 * (p1 - p0) * (xa + 1)
    W = 0.25 * (t1 - t0) * (p1 - p0) * np.outer(wa * np.sin(th), wa)
    T, P = np.meshgrid(th, ph, indexing="ij")
    total = 0.0
    for a, d in zip(edges[:-1], edges[1:]):
        for r, wr in zip(a + 0.5 * (d - a) * (x + 1), 0.5 * (d - a) * w):
            X = r * np.sin(T) * np.cos(P), r * np.sin(T) * np.sin(P), r * np.cos(T)
            total += wr * r * r * np.sum(W * np.asarray(b(*X), float) * np.asarray(v(*X), float))
    return total


def kernels(M, n=12):
    lab = LAB[np.random.default_rng(1).choice(len(LAB), n, replace=False)]
    k = [lambda r, p=i: (1.0 + p * r) * r for i in range(n)]
    return pg.Functionals.separable(M, {key: k for key in M.names}, lab, into=pg.Euclidean(n)), lab


CASES = [((0.0, 0.0, 0.6), 0.25), ((0.3, 0.2, 0.4), 0.2), ((0.5, -0.1, 0.1), 0.08)]


@pytest.mark.parametrize("c, rho", CASES)
def test_the_expansion_is_a_brute_quadrature_on_a_band_limited_field(c, rho):
    M = space()
    x, v, radii = field(M)
    b = bump(c, rho)
    T = rows(M, "a", [b.measure()])
    assert is_symbolic(T)
    Th = pg.harmonic_target(T, L)
    assert float(Th(x).values[0]) == pytest.approx(brute(b, v, radii, c, rho), rel=1e-9)


@pytest.mark.parametrize("c, rho", CASES)
def test_the_plain_call_on_a_realised_field_is_the_brute_quadrature(c, rho):
    """T(m), the call people make: with no degree to choose and the field's kinks between any fixed rule's nodes."""
    M = space()
    x, v, radii = field(M)
    b = bump(c, rho)
    want = brute(b, v, radii, c, rho)
    assert float(rows(M, "a", [b.measure()])(x).values[0]) == pytest.approx(want, rel=1e-9)
    assert float(M.functional("a", b.measure())(x)) == pytest.approx(want, rel=1e-9)


def test_no_warning_on_good_rules_and_a_warning_on_a_bad_one(monkeypatch):
    import warnings
    from pygeoinf0.core.errors import RealisationWarning
    import pygeoinf0.realisations.series.symbolic_harmonics as HT
    M = space()
    x, _, _ = field(M)
    b = bump((0.0, 0.0, 0.5), 0.2)                              # a sliver of it reaches the next piece
    with warnings.catch_warnings():
        warnings.simplefilter("error", RealisationWarning)
        rows(M, "a", [b.measure()])(x)
    M = space()
    x, _, _ = field(M)
    monkeypatch.setattr(HT, "PANELS", 1)
    monkeypatch.setattr(HT, "RADIAL", 5)
    with pytest.warns(RealisationWarning, match="estimated relative error"):
        rows(M, "a", [bump((0.0, 0.0, 0.5), 0.2).measure()])(x)


def test_the_norm_of_a_cut_bump_is_refined_and_is_the_integral():
    """<t, t> on the pieces against an integral in coordinates about the bump's own centre, with no warning."""
    import warnings
    from pygeoinf0.core.errors import RealisationWarning
    M = space()
    c, rho = (0.0, 0.0, 0.5), 0.2
    b = bump(c, rho)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RealisationWarning)
        n2 = float(M.norm(rows(M, "a", [b.measure()]).kernels[0])) ** 2
    x, w = np.polynomial.legendre.leggauss(120)
    s, ws = 0.5 * rho * (x + 1), 0.5 * rho * w
    xa, wa = np.polynomial.legendre.leggauss(80)
    th, ph = np.arccos(xa), np.linspace(0, 2 * math.pi, 161)[:-1]
    total = 0.0
    for si, wi in zip(s, ws):
        X = (c[0] + si * np.outer(np.sin(th), np.cos(ph)), c[1] + si * np.outer(np.sin(th), np.sin(ph)),
             c[2] + si * np.outer(np.cos(th), np.ones(160)))
        total += wi * si * si * np.sum(wa[:, None] * np.asarray(b(*X), float) ** 2) * (2 * math.pi / 160)
    assert n2 == pytest.approx(total, rel=1e-8)


def test_a_constant_gives_the_mass_and_a_bump_across_a_seam_has_parts():
    M = space()
    b = bump((0.0, 0.0, 0.5), 0.2)
    T = rows(M, "a", [b.measure()])
    assert len(M.parts("a", b.measure())) == 2
    x = M.split({"a": lambda x_, y_, z_: 0.25 + 0.0 * x_})
    assert float(pg.harmonic_target(T, L)(x).values[0]) == pytest.approx(0.25, rel=1e-9)


def test_sola_on_a_cut_bump_against_the_dense_formula():
    M = space()
    G, lab = kernels(M)
    D = G.codomain
    b = bump((0.0, 0.0, 0.5), 0.2)
    T = rows(M, "a", [b.measure()])
    y = D(np.random.default_rng(2).standard_normal(D.dim))
    noise = pg.GaussianMeasure(D, pg.Diagonal(np.full(D.dim, 0.01), D))
    res = pg.decompose(G, noise).sola(T, 0.5, y)
    Th = pg.harmonic_target(T, L)
    cross, _ = G.structured_cross(Th.tau)
    K = np.asarray(G.structured_gram().matrix(), float)
    c = cross @ Th.Phi[0]
    x = np.linalg.solve(K + 0.5 * 0.01 * np.eye(D.dim), c)
    assert float(np.ravel(res.estimate)[0]) == pytest.approx(x @ y.values, rel=1e-10)
    assert float(np.ravel(res.target)[0]) == pytest.approx(float(M.norm(T.kernels[0])) ** 2, rel=1e-12)


def test_a_rotated_family_is_the_bump_at_each_centre():
    M = space()
    x, _, _ = field(M)
    r, rho = 0.6, 0.25
    T0 = M.functional("a", bump((0.0, 0.0, r), rho).measure())
    with pytest.raises(pg.Refused, match="degree"):
        pg.RotatedFamily(T0, np.array([[10.0, 30.0]]))
    ll = np.array([[10.0, 30.0], [-40.0, 200.0], [60.0, 5.0]])
    fam = pg.RotatedFamily(T0, ll, degree=L)
    for i, (lat, lon) in enumerate(np.radians(ll)):
        c = (r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat))
        direct = pg.harmonic_target(rows(M, "a", [bump(c, rho).measure()]), L)
        assert fam(x).values[i] == pytest.approx(float(direct(x).values[0]), rel=1e-8)


def test_a_bump_off_the_axis_is_not_zonal():
    M = space()
    T0 = M.functional("a", bump((0.4, 0.1, 0.3), 0.15).measure())
    with pytest.raises(ValueError, match="not zonal"):
        pg.RotatedFamily(T0, np.array([[0.0, 0.0]]), degree=L)


def test_a_prior_needs_the_degree_and_then_is_the_dense_formula():
    M = space()
    G, lab = kernels(M)
    D = G.codomain
    degrees = list(range(L + 1))
    ball = pg.Laplacian(BALL, bc={"boundary": "dirichlet"}, centre="regular").bessel_potential(10.0, 2.0, degrees=degrees, modes=16)
    C = pg.Covariance.glued(M, {tuple(M.names): ball})
    y = D(np.random.default_rng(3).standard_normal(D.dim))
    noise = pg.GaussianMeasure(D, pg.Diagonal(np.full(D.dim, 0.01), D))
    dec = pg.decompose(G, noise, middle=C)
    T = rows(M, "a", [bump((0.0, 0.0, 0.5), 0.2).measure()])
    with pytest.raises(pg.Refused, match="degree"):
        dec.sola(T, 0.5, y)
    Th = pg.harmonic_target(T, L)
    post = dec.posterior(Th, y, 1.0, 1.0)
    cross, tt = G.structured_cross(Th.tau, C)
    K = np.asarray(G.structured_gram(C).matrix(), float)
    c, f = cross @ Th.Phi[0], Th.Phi[0]
    A = K + 0.01 * np.eye(D.dim)
    assert float(np.ravel(post.mean)[0]) == pytest.approx(c @ np.linalg.solve(A, y.values), rel=1e-9)
    assert float(np.ravel(post.variance)[0]) == pytest.approx(f @ np.asarray(tt) @ f - c @ np.linalg.solve(A, c), rel=1e-9)


def test_the_support_box_holds_the_support():
    c, rho = (0.3, 0.2, 0.4), 0.15
    (r0, r1), (t0, t1), (p0, p1) = support_box(type("Q", (), {"set": pg.Ball(rho, centre=c)})(), (0, 0, 0), 0.0, 1.0)
    rng = np.random.default_rng(4)
    u = rng.normal(size=(500, 3))
    P = np.asarray(c) + rho * rng.uniform(0, 1, (500, 1)) ** (1 / 3) * u / np.linalg.norm(u, axis=1, keepdims=True)
    r = np.linalg.norm(P, axis=1)
    th, ph = np.arccos(P[:, 2] / r), np.arctan2(P[:, 1], P[:, 0])
    ph = np.where(ph < p0, ph + 2 * math.pi, ph)
    assert np.all((r >= r0) & (r <= r1) & (th >= t0) & (th <= t1) & (ph >= p0) & (ph <= p1))


# the same standard on the other realisations (plan 044 P3): fields with kinks, against a brute quadrature split at them ----------
def kinked(x, y, z):
    return np.abs(np.sqrt(x * x + y * y + z * z) - 0.5) + z * x


def _ball_brute(b, c, rho, f, kinks=(0.5,), nr=40, na=90):
    (r0, r1), (t0, t1), (p0, p1) = support_box(type("Q", (), {"set": pg.Ball(rho, centre=c)})(), (0, 0, 0), 0.0, 1.0)
    edges = np.unique(np.r_[r0, r1, [x for x in kinks if r0 < x < r1]])
    x, w = np.polynomial.legendre.leggauss(nr)
    xa, wa = np.polynomial.legendre.leggauss(na)
    th, ph = t0 + 0.5 * (t1 - t0) * (xa + 1), p0 + 0.5 * (p1 - p0) * (xa + 1)
    W = 0.25 * (t1 - t0) * (p1 - p0) * np.outer(wa * np.sin(th), wa)
    T, P = np.meshgrid(th, ph, indexing="ij")
    total = 0.0
    for a, d in zip(edges[:-1], edges[1:]):
        for r, wr in zip(a + 0.5 * (d - a) * (x + 1), 0.5 * (d - a) * w):
            X = r * np.sin(T) * np.cos(P), r * np.sin(T) * np.sin(P), r * np.cos(T)
            total += wr * r * r * np.sum(W * np.asarray(b(*X), float) * np.asarray(f(*X), float))
    return total


@pytest.mark.parametrize("real", ["tensor", "legendre"])
@pytest.mark.parametrize("c, rho", [((0.0, 0.0, 0.55), 0.3), ((0.3, 0.2, 0.35), 0.2)])
def test_a_tensor_field_with_a_kink_on_a_ball(real, c, rho):
    X = pg.L2(BALL, realisation=real)
    u = X.element(kinked, breaks=((0.5,), None, None))
    u = 1.0 * u + 0.0 * u
    b = bump(c, rho)
    got = X.inner(X.riesz(b.measure()), u)
    assert got == pytest.approx(_ball_brute(b, c, rho, kinked), rel=1e-8)


def test_a_box_field_with_a_kink():
    B = pg.Box([(0, 1), (0, 1), (0, 1)])
    f = lambda x, y, z: np.abs(x - 0.47) + y * z
    c, rho = (0.5, 0.5, 0.5), 0.2
    for real in ("tensor", "local"):
        X = pg.L2(B, realisation=real)
        u = X.element(f, breaks=((0.47,), None, None))
        u = 1.0 * u + 0.0 * u
        g = F.on(B).bump(centre=c, radius=rho).normalised()
        x, w = np.polynomial.legendre.leggauss(30)
        edges = [0.3, 0.47, 0.7]
        xs = np.concatenate([a + 0.5 * (d - a) * (x + 1) for a, d in zip(edges[:-1], edges[1:])])
        wx = np.concatenate([0.5 * (d - a) * w for a, d in zip(edges[:-1], edges[1:])])
        ya, wa = np.polynomial.legendre.leggauss(60)
        ys, wy = 0.3 + 0.2 * (ya + 1), 0.2 * wa
        G = np.meshgrid(xs, ys, ys, indexing="ij")
        want = np.einsum("i,j,k,ijk", wx, wy, wy, np.asarray(g(*G), float) * np.asarray(f(*G), float))
        assert X.inner(X.riesz(g.measure()), u) == pytest.approx(want, rel=1e-8)


def test_cell_values_with_a_bump():
    from pygeoinf0.realisations import Cells
    n = np.linspace(0.0, 1.0, 11)
    B = pg.Box([(0, 1), (0, 1)])
    X = pg.L2(pg.LebesgueMeasure(B), realisation=Cells.tensor(n, n))
    f = lambda x, y: x + 2 * y
    u = 1.0 * X.element(f) + 0.0 * X.element(f)
    c, rho = (0.46, 0.53), 0.17
    g = F.on(B).bump(centre=c, radius=rho).normalised()
    want = 0.0
    x, w = np.polynomial.legendre.leggauss(40)
    for i in range(10):
        for j in range(10):
            xs, ys = n[i] + 0.05 * (x + 1), n[j] + 0.05 * (x + 1)
            G = np.meshgrid(xs, ys, indexing="ij")
            want += float(u.realisation.V[i + 10 * j, 0]) * np.einsum("i,j,ij", 0.05 * w, 0.05 * w, np.asarray(g(*G), float))
    assert X.inner(X.riesz(g.measure()), u) == pytest.approx(want, rel=1e-8)
