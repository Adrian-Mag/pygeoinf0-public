"""Plan 009 P2, G2: separable functionals k (x) Y_lj, Grams assembled per harmonic (shared blocks), the derivation of
rotation invariance from the factors, and per-summand SOLA components from the blocks (EXP-030 DESIGN G2, Theorem
G2.1, NOTES.md section 3). Uses outside the CMB problem: harmonic data of a field on a sphere (gravity-like
coefficients with order-dependent noise), radial kernels on the Moon's layered ball."""
import math

import numpy as np
import pytest
from scipy.special import spherical_in, sph_harm_y

import pygeoinf0 as pg
from pygeoinf0 import quadrature
from pygeoinf0.operators.band import AngularFunction


def Y(l, j, x, y, z):
    """The package's real harmonic from scipy (Condon-Shortley removed): an independent reference."""
    r = np.sqrt(x * x + y * y + z * z)
    th = np.arccos(np.clip(z / np.where(r > 0, r, 1.0), -1, 1))
    ph = np.mod(np.arctan2(y, x), 2 * np.pi)
    m = (j + 1) // 2
    v = sph_harm_y(l, m, th, ph)
    out = v.real if j == 0 else math.sqrt(2) * (v.real if j % 2 else v.imag)
    return (-1.0) ** m * out


def moon_problem(seed=1):
    """Radial kernels on a layered ball (core, mantle) and coefficients of a surface field: a CMB-like structure."""
    planet = pg.LayeredBall([0.35, 1.0], names=("core", "mantle"))
    V = pg.PiecewiseSpace.lebesgue(planet, realisation="radial_harmonics")
    M = pg.DirectSum({"core": V.summand("core"), "mantle": V.summand("mantle"), "surface": pg.L2(pg.Sphere(1.0))},
                     weights={"core": 0.6, "mantle": 0.6, "surface": 1.5})
    rng = np.random.default_rng(seed)
    core, mantle, surf, lab = [], [], [], []
    ker = {}
    for l in (1, 2, 3):
        for a in range(3):
            ker[l, a] = (pg.Tabulated("[0, 0.35]", np.linspace(0, 0.35, 5), rng.normal(size=5) * np.linspace(0, 1, 5)),
                         pg.Tabulated("[0.35, 1]", np.linspace(0.35, 1, 7), rng.normal(size=7)), float(rng.normal()))
    for l in (1, 2, 3):
        for j in range(2 * l + 1):
            for a in range(3):
                lab.append((l, j))
                c, m, s = ker[l, a]
                core.append(c)
                mantle.append(m if a != 1 else None)          # one multiplet does not see the mantle
                surf.append(s)
    G = pg.Functionals.separable(M, {"core": core, "mantle": mantle, "surface": surf}, lab)
    return M, G, lab


def test_separable_gram_equals_the_dense_gram_with_shared_blocks():
    M, G, lab = moon_problem()
    K = G.structured_gram()
    D = G.gram()
    assert np.abs(K.matrix() - D).max() <= 1e-14 * np.abs(D).max()
    assert K.n_blocks == sum(2 * l + 1 for l in (1, 2, 3)) and len(K.matrices) == 3        # one block per degree
    assert any(c.key == "block diagonal" and c.status == "derived" for c in K.claims)
    f = M({"core": lambda x, y, z: np.exp(x) * y + z, "mantle": lambda x, y, z: x * y + z, "surface": lambda x, y, z: z})
    v = G(f).values
    assert np.allclose(v, D @ np.linalg.lstsq(D, v, rcond=None)[0], atol=1e-12)   # consistent with the representers


def test_separable_gram_against_the_tensor_realisation():
    """The same functionals realised in the tensor realisation from formulas (independent route)."""
    B = pg.Ball(1.0)
    Mr, Mt = pg.L2(B, realisation="radial_harmonics"), pg.L2(B)
    ks = [lambda r: 1 + r, lambda r: np.cos(2 * r), lambda r: r ** 2 - 0.3]
    lab = [(l, j) for l in (0, 1, 2) for j in range(2 * l + 1) for _ in range(2)]
    rad = [ks[(i // 2) % 3 if (i % 2) else 0] for i in range(len(lab))]
    Gr = pg.Functionals.separable(Mr, rad, lab)
    reps = [Mt(lambda x, y, z, k=k, l=l, j=j: k(np.sqrt(x * x + y * y + z * z)) * Y(l, j, x, y, z))
            for k, (l, j) in zip(rad, lab)]
    Dt = Mt.inner(Mt.family(reps), Mt.family(reps))
    assert np.abs(Gr.structured_gram().matrix() - Dt).max() <= 1e-11 * np.abs(Dt).max()


def test_a_rotation_invariant_operator_has_the_predicted_block_form():
    """B(g_i, g_j) = int int g_i(x) exp(x . y) g_j(y) dx dy by tensor quadrature of the ball (no realisation): zero
    between different harmonics, equal across orders, and equal to Funk-Hecke's 4 pi int int k k' i_l(r rho) r^2 rho^2."""
    B = pg.Ball(1.0)
    X, W = quadrature.rule(B, 18)
    ks = [lambda r: 1 + r, lambda r: r * np.cos(r)]
    lab = [(l, j, a) for l in (0, 1, 2) for j in range(2 * l + 1) for a in range(2)]
    r = np.linalg.norm(X, axis=1)
    A = np.stack([W * ks[a](r) * Y(l, j, *X.T) for l, j, a in lab], axis=1)
    EA = np.vstack([np.exp(X[s:s + 1000] @ X.T) @ A for s in range(0, len(X), 1000)])
    Bm = A.T @ EA
    scale = np.abs(Bm).max()
    t, w = np.polynomial.legendre.leggauss(40)
    rr, ww = 0.5 * (t + 1), 0.5 * w
    for i, (l, j, a) in enumerate(lab):
        for k, (l2, j2, b) in enumerate(lab):
            if (l, j) != (l2, j2):
                assert abs(Bm[i, k]) <= 1e-12 * scale
            else:
                pred = 4 * math.pi * np.einsum("p,q,p,q,pq->", ww * rr ** 2, ww * rr ** 2, ks[a](rr), ks[b](rr),
                                               spherical_in(l, np.outer(rr, rr)))
                assert Bm[i, k] == pytest.approx(pred, rel=1e-11)


def test_structured_grams_with_rotation_invariant_middle_operators():
    M, G, lab = moon_problem()
    P = pg.BandProjection(M, [1, 3])
    A = AngularFunction(M, lambda l: 1.0 / (1.0 + l * (l + 1.0)) ** 1.5, summands=["core", "surface"])
    for S in (P, A, 2.0 * A + P, A @ P):
        K = G.structured_gram(S)
        assert K is not None and any(c.status == "theorem" and c.key == "rotation invariant" for c in K.claims)
        Dn = (G @ S @ G.adjoint).matrix()
        assert np.abs(K.matrix() - Dn).max() <= 1e-12 * max(np.abs(Dn).max(), 1e-300)
    rng = np.random.default_rng(0)
    Q = pg.FiniteRank(M.family([M({"core": lambda x, y, z: x + rng.normal()})]), [1.0])       # not invariant
    assert G.structured_gram(Q) is None


def test_harmonic_data_on_a_sphere_with_order_dependent_noise():
    """Gravity-like data: coefficients of a field on a sphere of radius R with errors depending on the order; the
    decomposition uses the structured Gram (blocks shared across orders, noise not) and agrees with the dense route."""
    R = 1.7
    S = pg.L2(pg.Sphere(R))
    lab = [(l, j) for l in range(2, 9) for j in range(2 * l + 1)]
    G = pg.Functionals.separable(S, [1.0 / R ** 2] * len(lab), lab)       # h -> h_lj
    h = S(lambda x, y, z: np.exp(x / R) * (1 + y * z / R ** 2))
    from pygeoinf0.spaces.harmonic import index
    assert np.allclose(G(h).values, h.realisation.C[index(*np.array(lab).T), 0], atol=1e-14)
    n = len(lab)
    rng = np.random.default_rng(5)
    sig = rng.uniform(0.5, 2.0, n)
    D = G.codomain
    noise = pg.GaussianMeasure(D, pg.Diagonal(sig ** 2, D))
    dec = pg.decompose(G, noise)
    assert dec.route == "blocks" and len(dec._rows) == n
    dense = pg.decompose(G, noise, gram=G.gram(), blocks=None)
    T = pg.Functionals([S(lambda x, y, z: np.exp(-((x / R - 0.3) ** 2 + (y / R) ** 2)))])
    d = D(rng.normal(size=n))
    g = np.array([0.01, 1.0])
    a, b = dec.sola(T, g, d), dense.sola(T, g, d)
    for key in ("estimate", "variance", "mismatch"):
        assert np.allclose(getattr(a, key), getattr(b, key), rtol=1e-11, atol=1e-14)


def test_per_summand_sola_components_from_the_blocks():
    M, G, lab = moon_problem()
    n = len(lab)
    rng = np.random.default_rng(2)
    D = G.codomain
    noise = pg.GaussianMeasure(D, pg.Diagonal(rng.uniform(0.5, 1.5, n), D))
    Tsep = pg.Functionals.separable(M, {"core": [pg.Tabulated("[0, 0.35]", [0, 0.35], [1.0, 0.2])] * 3,
                                        "surface": [0.7, 0.2, -0.4]}, [(1, 0), (2, 3), (3, 6)], name="T")
    plain = pg.Functionals(G.kernels, into=D)                           # the same functionals without structure
    Tplain = pg.Functionals(Tsep.kernels)
    d = D(rng.normal(size=n))
    g = np.array([0.05, 0.5])
    a = pg.decompose(G, noise).sola(Tsep, g, d, fields=True)
    b = pg.decompose(plain, noise).sola(Tplain, g, d, fields=True)
    assert set(a.components) == set(b.components)
    for k in a.components:
        for q in ("delivered", "mismatch", "target"):
            assert np.allclose(a.components[k][q], b.components[k][q], rtol=1e-10, atol=1e-13)


# --------------------------------------------------------------------- tabulated kernels times a smooth factor (P4a speed)
class _Opaque:
    """The same kernel without its exact form: radial_family then fits k (dnu/dmu) tau adaptively (the old route)."""

    def __init__(self, k):
        self.k, self.breaks = k, k.breaks

    def __call__(self, r):
        return self.k(r)


def _tapered_family(a, b, items, outer, eps_fraction=1e-3):
    from pygeoinf0.operators.separable import radial_family
    M = pg.L2(pg.Ball(b) if a == 0 else pg.Shell(a, b), realisation="radial_harmonics")
    eps2 = (eps_fraction * outer) ** 2
    taper = lambda r: np.asarray(r, float) ** 2 / (np.asarray(r, float) ** 2 + eps2)   # noqa: E731
    phi = lambda r: 1.0 / (np.asarray(r, float) ** 2 + eps2)                           # noqa: E731  (dnu/dmu) tau
    pl, claims = radial_family(a, b, items, np.ones(len(items)), M.backend.radial_measure(),
                               pg.LebesgueMeasure(pg.Interval(a, b)), taper, label="functional")
    return pl, claims, phi


def _tables(n=6, seed=3, R=6371.0, markers=(1221.5, 3480.0, 5701.0)):
    rng = np.random.default_rng(seed)
    x = np.r_[0.5, np.sort(rng.uniform(1.0, R - 1.0, 60)), R]
    ks = []
    for i in range(n):
        c, w = rng.normal(size=3), rng.uniform(0.5, 6.0, 3)
        v = sum(ci * np.sin(wi * math.pi * x / R + j) for j, (ci, wi) in enumerate(zip(c, w)))
        ks.append(pg.Tabulated((0.0, R), x, v, rule="linear", segments=markers, extrapolation="flat"))
    return ks


@pytest.mark.parametrize("a, b", [(0.0, 3480.0), (3480.0, 5701.0)])          # a ball through the taper's peak; a shell
def test_tabulated_kernels_times_a_factor_match_the_adaptive_fit(a, b):
    ks = _tables()
    exact, claims, _ = _tapered_family(a, b, ks, 6371.0)
    fitted, _, _ = _tapered_family(a, b, [_Opaque(k) for k in ks], 6371.0)
    assert any("common factor" in c.statement for c in claims)
    from pygeoinf0.realisations.series import legendre as LG
    x = np.linspace(a, b, 4001)[1:-1]
    x = x[~np.isin(x, exact.breaks)]
    u, v = LG.evaluate(exact, x), LG.evaluate(fitted, x)
    assert np.abs(u - v).max() <= 1e-11 * np.abs(v).max()


@pytest.mark.parametrize("a, b", [(0.0, 3480.0), (3480.0, 5701.0)])
def test_tapered_gram_against_scipy_quad(a, b):
    """<g_i, g_j>_(L2(r^2 dr)) of the realised representers equals int k_i k_j phi^2 r^2 dr by quad (independent)."""
    from scipy.integrate import quad
    from pygeoinf0.realisations.series.radial_harmonic import rgram       # the Gram the package uses (r^2 at the nodes)
    ks = _tables(n=3)
    pl, _, phi = _tapered_family(a, b, ks, 6371.0)
    G = rgram(pl, pl)
    eps = 6.371                                                     # the taper's scale: the integrand peaks there
    edges = [a] + sorted({float(x) for x in pl.breaks if a < x < b} | ({eps * 2.0 ** j for j in range(-8, 9)}
                                                                       if a == 0 else set())) + [b]
    for i in range(3):
        for j in range(i, 3):
            f = lambda r: ks[i](r) * ks[j](r) * phi(r) ** 2 * r * r   # noqa: E731
            ref = math.fsum(quad(f, x0, x1, epsabs=0, epsrel=1e-13, limit=200)[0] for x0, x1 in zip(edges[:-1], edges[1:]))
            assert abs(G[i, j] - ref) <= 1e-12 * math.sqrt(G[i, i] * G[j, j])


@pytest.mark.parametrize("rule", ["constant", "midpoint-linear", "linear"])
def test_every_tabulated_rule_times_a_factor(rule):
    """The general case: tables read by any rule (jumps at cell edges included), times the common factor; checked
    against the exact function between breakpoints."""
    from pygeoinf0.realisations.series import legendre as LG
    a, b = 0.0, 2.0
    edges = np.array([0.0, 0.3, 0.75, 0.9, 1.4, 2.0])
    if rule == "linear":
        k = pg.Tabulated((a, b), [0.1, 0.5, 0.9, 1.4, 2.0], [1.0, -0.5, 0.3, 0.8, 0.0], segments=[0.9])
    else:
        k = pg.Tabulated((a, b), edges, [1.0, -0.5, 0.3, 0.8, -0.2], rule=rule)
    pl, claims, phi = _tapered_family(a, b, [k], 2.0, eps_fraction=0.05)
    assert any("common factor" in c.statement for c in claims)
    x = np.linspace(0.003, 1.997, 997)
    x = x[np.min(np.abs(x[:, None] - pl.breaks[None, :]), axis=1) > 1e-9]
    assert np.allclose(LG.evaluate(pl, x)[:, 0], k(x) * phi(x), rtol=1e-12, atol=1e-13 * phi(0))


def test_the_factor_route_is_fast_and_states_its_error():
    """300 tables on a shell: the kernels are never sampled one by one (the adaptive route evaluates each at every
    Gauss point of every piece)."""
    import time
    ks = _tables(n=300)
    t0 = time.perf_counter()
    pl, claims, _ = _tapered_family(3480.0, 6371.0, ks, 6371.0)
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    _tapered_family(3480.0, 6371.0, [_Opaque(k) for k in ks[:30]], 6371.0)
    slow_per_kernel = (time.perf_counter() - t0) / 30
    assert fast < 300 * slow_per_kernel / 3
    stated = [c for c in claims if "common factor" in c.statement][0]
    assert "largest estimated error" in stated.statement and float(pl.err.max()) < 1e-10
