"""Function spaces with a density on 2-D and 3-D domains (plan 064). Stage 1: L2(c sigma) for a constant c > 0 is L2(sigma)
with every pairing times c, so on the sphere it is numerically the unweighted space with direct-sum weight c: data,
SOLA's X, adjoints, box supports, BGP brackets and oCSOLA box ends agree to rounding; functionals given as measures do
not change with c, and a pairing <B, h> / <B, B> with a fixed element B does not either. Constant densities on balls,
boxes and disks in every realisation. Stage 2: radial densities rho(r) dx on balls and shells (radial x harmonic
realisation) against an independent tensor Gauss rule: inner products, integrals, representers of measures, the
adjoint identity, separable functionals, box supports. Reprs and claims name the measure; stage 3 and the rest are
refused. Synthetic only."""
import math
import warnings

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported
from pygeoinf0.inference.methods import decomposition_of

R_CORE = 3479.95776
C_CMB = 1 / (4 * math.pi * R_CORE ** 2)
S2 = pg.Sphere(1.0)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def cmb_problem(X, nd=8, P=3, deg=10, seed=0):
    rng = np.random.default_rng(seed)
    lat = lambda n: np.degrees(np.arcsin(rng.uniform(-1, 1, n)))     # noqa: E731
    cd, ct = np.c_[lat(nd), rng.uniform(0, 360, nd)], np.c_[lat(P), rng.uniform(0, 360, P)]
    G = pg.CapAverages(X, 1500.0 / R_CORE, cd, summand="cmb", degree=deg)
    T = pg.CapAverages(X, 1000.0 / R_CORE, ct, summand="cmb", degree=deg)
    D = G.codomain
    noise = pg.GaussianMeasure(D, 0.01 * pg.Identity(D))
    H = X.summands[0]
    mt = X({"cmb": H(lambda x, y, z: 3.0 * x * y + 2.0 * z, resolution=1.0)})      # |m| <= 5 on the sphere
    return G, T, noise, mt, D


def cmb_numbers(X, P=3):
    G, T, noise, mt, D = cmb_problem(X, P=P)
    L = pg.Additive(G, noise)
    d = G(mt)
    box = pg.PointwiseBox(X, envelope={"cmb": 5.0}, rtol=1e-4)
    Xs = decomposition_of(L).sola_operator(T, 0.05)
    r = pg.inference.ocsola(box, L, T, d, X=Xs)
    Pp = T.codomain
    tf = T.adjoint(Pp.family(np.eye(P)))
    af = G.adjoint(Xs.adjoint(Pp.family(np.eye(P))))
    w = norm.ppf(0.975) * np.sqrt(np.diag(Xs(noise).covariance.matrix()))
    slab = pg.inference.slab_family(D, Xs.matrix().T, sigma=w, mass=1.0)[0]
    up = pg.FeasibleSet(box, G, slab, d).bound(tf[0])
    return dict(data=d.values, X=Xs.matrix(), ends=np.r_[r[0], r[1]], tau=np.r_[r.tau_lower, r.tau_upper],
                support_t=box.support(tf[0]), support_a=box.support(af[1]), inner=X.inner(tf[0], af[1]),
                norm=X.norm(tf[1]), bgp=(up.lower, up.upper), prior=box.support(tf[0]) + box.support(tf[0] * -1.0),
                truth=T(mt).values, noise=w)


def test_constant_density_on_the_sphere_is_the_direct_sum_weight():
    """The CMB pairing as the space's own measure, L2(S^2, dOmega / (4 pi r^2)) with weight 1, equals L2(S^2) with the
    direct-sum weight 1 / (4 pi r^2): every number of the plan 062 pipeline to rounding."""
    XW = pg.DirectSum({"cmb": pg.L2(S2)}, weights={"cmb": C_CMB})
    XD = pg.DirectSum({"cmb": pg.L2(pg.LebesgueMeasure(S2).with_density(C_CMB))}, weights={"cmb": 1.0})
    a, b = cmb_numbers(XW), cmb_numbers(XD)
    for k in a:
        assert rel(b[k], a[k]) <= 1e-12, k
    assert np.all(a["ends"][:3] <= a["truth"]) and np.all(a["truth"] <= a["ends"][3:])


def test_plain_space_with_a_constant_density():
    """L2(c sigma) without a direct sum: inner products c times, integrals c times, errors sqrt(c) times; cap averages and
    representers of measures as functionals do not change; adjoints satisfy <G u, y> = <u, G* y>."""
    c = 0.37
    M0, M = pg.L2(S2), pg.L2(pg.LebesgueMeasure(S2).with_density(c))
    f, g = (lambda x, y, z: 1 + x * y + z ** 3), (lambda x, y, z: np.exp(z) * (1 - x))
    assert M.inner(M(f, resolution=0.5), M(g, resolution=0.5)) == \
        pytest.approx(c * M0.inner(M0(f, resolution=0.5), M0(g, resolution=0.5)), rel=1e-14)
    assert M.integrate(M(f, resolution=0.5)) == pytest.approx(c * M0.integrate(M0(f, resolution=0.5)), rel=1e-14)
    assert M(f, resolution=0.5).error == pytest.approx(math.sqrt(c) * M0(f, resolution=0.5).error, rel=1e-12, abs=1e-300)
    cen = [[10.0, 20.0], [-40.0, 200.0]]
    T0, T = pg.CapAverages(M0, 0.3, cen, degree=12), pg.CapAverages(M, 0.3, cen, degree=12)
    h0, h = M0(f, resolution=0.5), M(f, resolution=0.5)
    assert rel(T(h).values, T0(h0).values) <= 1e-13                       # averages: functionals of the measure
    E = pg.Euclidean(2)
    y = E(np.array([0.7, -1.3]))
    assert E.inner(T(h), y) == pytest.approx(M.inner(h, T.adjoint(y)), rel=1e-12)
    assert rel(M.norm(T.adjoint(y)) ** 2, M0.norm(T0.adjoint(y)) ** 2 / c) <= 1e-12   # representers scale by 1/c
    cap = pg.LebesgueMeasure(S2)                                           # the whole sphere: representer 1/c
    assert M.inner(M.riesz(cap), h) == pytest.approx(M0.integrate(h0), rel=1e-13)
    assert M.average(S2)(h) == pytest.approx(M0.average(S2)(h0), rel=1e-13)
    assert M == pg.L2(pg.LebesgueMeasure(S2).with_density(float(np.float64(c))))   # constants compare by value
    assert M != M0


@pytest.mark.parametrize("c", [1.0, 0.01, 7.0])
def test_a_pairing_over_its_kernel_norm_does_not_depend_on_c(c):
    """<B, h> / <B, B> for a fixed element B (the hand-built representer B / ||B||^2 of the direct-sum route): the same
    for every c, with no extra 1 / weight."""
    M = pg.L2(pg.LebesgueMeasure(S2).with_density(c))
    B = M(lambda x, y, z: np.exp(-((z - 0.8) ** 2) / 0.1), resolution=0.3)
    h = M(lambda x, y, z: 2.0 + x - z * y, resolution=0.5)
    M1 = pg.L2(S2)
    B1, h1 = M1(lambda x, y, z: np.exp(-((z - 0.8) ** 2) / 0.1), resolution=0.3), \
        M1(lambda x, y, z: 2.0 + x - z * y, resolution=0.5)
    assert M.inner(B, h) / M.inner(B, B) == pytest.approx(M1.inner(B1, h1) / M1.inner(B1, B1), rel=1e-13)


@pytest.mark.parametrize("domain, realisation", [(pg.Ball(2.0), None), (pg.Ball(2.0), "radial_harmonics"),
                                                 (pg.Ball(2.0), "legendre"), (pg.Shell(0.5, 1.0), "radial_harmonics"),
                                                 (pg.Box((0, 1), (0, 2)), None), (pg.Box((0, 1), (0, 2)), "tensor"),
                                                 (pg.Disk(1.0), None)])
def test_constant_density_on_volumes_and_flat_domains(domain, realisation):
    """(f, g) = varpi / V int f g dx on a ball or shell (a constant density 1 / V), and on boxes and disks: inner
    products and integrals are c times the Lebesgue ones in every realisation; the average over the domain and the
    representer of a sub-domain's measure pair as functionals of the measure."""
    c = 1.0 / domain.volume
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M0 = pg.L2(domain, realisation=realisation)
        M = pg.L2(pg.LebesgueMeasure(domain).with_density(c), realisation=realisation)
        if domain.dim == 3:
            f, g = (lambda x, y, z: 1 + x * x + y * z), (lambda x, y, z: np.cos(z) + x)
        else:
            f, g = (lambda x, y: 1 + x * y), (lambda x, y: np.cos(y) + x)
        u0, v0, u, v = M0(f, resolution=0.3), M0(g, resolution=0.3), M(f, resolution=0.3), M(g, resolution=0.3)
        assert M.inner(u, v) == pytest.approx(c * M0.inner(u0, v0), rel=1e-13)
        assert M.integrate(u) == pytest.approx(c * M0.integrate(u0), rel=1e-13)
        assert M.norm(M.riesz(pg.LebesgueMeasure(domain))) ** 2 == pytest.approx(domain.volume / c, rel=1e-12)
        assert M.average(domain)(u) == pytest.approx(M0.average(domain)(u0), rel=1e-12)
        if isinstance(domain, pg.Box):
            sub = pg.Box((0.2, 0.6), (0.5, 1.5))
            assert M.inner(M.riesz(sub), u) == pytest.approx(M0.inner(M0.riesz(sub), u0), rel=1e-12)


def test_discretisation_pairings_and_gram_with_a_constant_density():
    """pg.Discretisation on L2(c dx) of a box: its Gram is c times the Lebesgue one; pairings int phi dmu are
    functionals of the measures and do not change."""
    c = 0.2
    box = pg.Box((0, 1), (0, 1))
    M0, M = pg.L2(box), pg.L2(pg.LebesgueMeasure(box).with_density(c))
    basis = [lambda x, y, a=a: np.exp(-((x - a) ** 2 + (y - 0.5) ** 2) / 0.05) for a in (0.3, 0.5, 0.7)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D0, D = pg.Discretisation(pg.Span([M0(b, resolution=0.1) for b in basis], space=M0)), \
            pg.Discretisation(pg.Span([M(b, resolution=0.1) for b in basis], space=M))
        G0, G = (np.asarray(x.toarray() if hasattr(x, "toarray") else x, float) for x in (D0.subspace.gram, D.subspace.gram))
        assert rel(G, c * G0) <= 1e-13
        mus = [pg.LebesgueMeasure(pg.Box((0.1, 0.4), (0.2, 0.9))).with_density(lambda x, y: 1 + x * y),
               pg.LebesgueMeasure(pg.Box((0.5, 0.9), (0.0, 0.5)))]
        E0, E = D0.pairings(mus).matrix(), D.pairings(mus).matrix()
        assert rel(E, E0) <= 1e-10


# ------------------------------------------------------------------------------------------------ stage 2: radial
SHELL = pg.Shell(0.5, 1.0)


def rho(r):
    return 1.0 + 2.0 * r ** 2 + np.sin(3.0 * r)


def quadrature(n=48):
    """An independent tensor Gauss rule on SHELL (r: Gauss, theta: Gauss, phi: equispaced): points, weights of dx."""
    tr, wr = leggauss(n)
    r, wr = 0.75 + 0.25 * tr, 0.25 * wr
    tt, wt = leggauss(n)
    th, wt = 0.5 * math.pi * (tt + 1), 0.5 * math.pi * wt
    ph = np.linspace(0, 2 * math.pi, 2 * n + 1)[:-1]
    R, T, P = np.meshgrid(r, th, ph, indexing="ij")
    X = (R * np.sin(T) * np.cos(P), R * np.sin(T) * np.sin(P), R * np.cos(T))
    W = np.einsum("i,j,k->ijk", wr * r * r, wt * np.sin(th), np.full(ph.size, 2 * math.pi / ph.size))
    return X, W, R


def radial_space():
    return pg.L2(pg.LebesgueMeasure(SHELL).with_density(rho, radial=True, name="rho"), realisation="radial_harmonics")


def f(x, y, z):
    return 1 + x * y + z ** 2


def g(x, y, z):
    return np.exp(z) * (1 + x)


def test_radial_density_against_independent_quadrature():
    M = radial_space()
    X, W, R = quadrature()
    u, v = M(f, resolution=0.2), M(g, resolution=0.2)
    assert M.inner(u, v) == pytest.approx(float(np.sum(W * rho(R) * f(*X) * g(*X))), rel=1e-12)
    assert M.norm(u) ** 2 == pytest.approx(float(np.sum(W * rho(R) * f(*X) ** 2)), rel=1e-12)
    assert M.integrate(u) == pytest.approx(float(np.sum(W * rho(R) * f(*X))), rel=1e-12)


def test_radial_density_representers_and_the_adjoint_identity():
    """The representer of a measure nu is (dnu/dx) / rho: <k, u> = int u dnu for Lebesgue measure on the shell and for
    the normalised measure of a small ball inside it; <G u, y> = <u, G* y> for kernels given as elements."""
    M = radial_space()
    X, W, R = quadrature()
    u = M(f, resolution=0.2)
    k = M.riesz(pg.LebesgueMeasure(SHELL))
    assert M.inner(k, u) == pytest.approx(float(np.sum(W * f(*X))), rel=1e-12)
    assert M.norm(k) ** 2 == pytest.approx(float(np.sum(W / rho(R))), rel=1e-10)
    sh = pg.LebesgueMeasure(pg.Shell(0.6, 0.8)).normalised()                 # the mean of f there: 1 + E r^2 / 3
    mean = 1 + (0.8 ** 5 - 0.6 ** 5) / 5 / ((0.8 ** 3 - 0.6 ** 3) / 3) / 3
    assert M.inner(M.riesz(sh), u) == pytest.approx(mean, rel=1e-12)
    M0 = pg.L2(SHELL, realisation="radial_harmonics")
    nu = pg.LebesgueMeasure(pg.Ball(0.2, centre=(0.0, 0.0, 0.75))).normalised()
    with warnings.catch_warnings():              # a small ball's harmonic coefficients: the route's own stated error
        warnings.simplefilter("ignore")
        a, b = M.inner(M.riesz(nu), u), M0.inner(M0.riesz(nu), M0(f, resolution=0.2))
    assert a == pytest.approx(b, rel=1e-12)                                 # the same functional with or without rho
    assert a == pytest.approx(1.0 + 0.75 ** 2 + 0.2 ** 2 / 5, rel=1e-4)
    E = pg.Euclidean(2)
    G = pg.Functionals([M(lambda x, y, z: np.exp(-((z - 0.7) ** 2) / 0.05), resolution=0.1), M(g, resolution=0.2)],
                       into=E)
    y = E(np.array([0.3, -1.2]))
    assert E.inner(G(u), y) == pytest.approx(M.inner(u, G.adjoint(y)), rel=1e-12)


def test_radial_density_separable_functionals_and_supports():
    """Separable functionals integrate against r^2 dr, whatever rho (plan 066; their structured Gram equals the plain one); a
    constant written as a radial density is the constant density; a pointwise box's support is int A |q| rho dx."""
    M = radial_space()
    X, W, R = quadrature()
    Sf = pg.Functionals.separable(M, [lambda r: r, lambda r: np.cos(r)], [(0, 0), (1, 0)], into=pg.Euclidean(2))
    assert rel(Sf.structured_gram().matrix(), Sf.gram()) <= 1e-12
    u = M(f, resolution=0.2)
    Y00 = 1 / math.sqrt(4 * math.pi)
    assert Sf(u).values[0] == pytest.approx(float(np.sum(W * R * Y00 * f(*X))), rel=1e-10)   # against r^2 dr (066)
    c = 0.3
    Mr = pg.L2(pg.LebesgueMeasure(SHELL).with_density(lambda r: np.full(np.shape(r), c), radial=True),
               realisation="radial_harmonics")
    Mc = pg.L2(pg.LebesgueMeasure(SHELL).with_density(c), realisation="radial_harmonics")
    q = (lambda x, y, z: z - 0.1 + 0.3 * x)
    ur, uc = Mr(f, resolution=0.2), Mc(f, resolution=0.2)
    assert Mr.norm(ur) == pytest.approx(Mc.norm(uc), rel=1e-13)
    br = pg.PointwiseBox(Mr, envelope=2.0, rtol=1e-5).support(Mr(q, resolution=0.2))
    bc = pg.PointwiseBox(Mc, envelope=2.0, rtol=1e-5).support(Mc(q, resolution=0.2))
    assert (br.lower, br.upper) == pytest.approx((bc.lower, bc.upper), rel=1e-12)
    b = pg.PointwiseBox(M, envelope=2.0, rtol=1e-5).support(M(q, resolution=0.2))
    lo, hi = b.lower, b.upper
    Xf, Wf, Rf = quadrature(96)
    ref = 2.0 * float(np.sum(Wf * rho(Rf) * np.abs(q(*Xf))))         # a kink: the rule converges slowly
    assert lo <= hi and abs(0.5 * (lo + hi) - ref) <= 1e-3 * ref


def test_radial_density_restricts_to_a_concentric_shell_and_balls_work():
    M = radial_space()
    inner = pg.Shell(0.5, 0.8)
    mu = M.measure.restricted(inner)
    assert mu.weight.kind == "radial"
    sub = pg.L2(mu, realisation="radial_harmonics")
    Xq, Wq, Rq = quadrature()
    u = sub(f, resolution=0.2)
    tr, wr = leggauss(48)
    r, wr = 0.65 + 0.15 * tr, 0.15 * wr
    # the mean of f over the unit sphere at radius r is 1 + r^2 / 3: int f rho dx = 4 pi int (1 + r^2/3) rho r^2 dr
    assert sub.integrate(u) == pytest.approx(4 * math.pi * float(np.sum(wr * (1 + r * r / 3) * rho(r) * r * r)),
                                             rel=1e-12)
    B = pg.L2(pg.LebesgueMeasure(pg.Ball(1.0)).with_density(lambda r: 2.0 + r, radial=True),
              realisation="radial_harmonics")
    tr, wr = leggauss(48)
    r, wr = 0.5 * (tr + 1), 0.5 * wr
    assert B.integrate(B(f, resolution=0.2)) == \
        pytest.approx(4 * math.pi * float(np.sum(wr * (1 + r * r / 3) * (2.0 + r) * r * r)), rel=1e-12)


# ------------------------------------------------------------------------------------------------ claims and refusals
def test_reprs_and_claims_name_the_measure():
    M = pg.L2(pg.LebesgueMeasure(S2).with_density(0.25))
    assert "0.25 dσ" in repr(M)
    assert any("0.25 dσ" in c.statement for c in M.claims)
    assert "rho(r) dx" in repr(radial_space())
    assert "Lebesgue" in repr(pg.L2(S2))


def test_refusals():
    with pytest.raises(Unsupported, match="plan 064"):
        pg.L2(pg.LebesgueMeasure(S2).with_density(lambda x, y, z: 1 + z * z))          # stage 3: a general density
    with pytest.raises(Unsupported, match="plan 064"):
        pg.L2(pg.LebesgueMeasure(pg.Box((0, 1), (0, 1))).with_density(lambda x, y: 1 + x))
    with pytest.raises(Unsupported, match="radial_harmonics"):
        pg.L2(pg.LebesgueMeasure(SHELL).with_density(rho, radial=True))                 # the tensor realisation
    with pytest.raises(ValueError):
        pg.L2(pg.LebesgueMeasure(S2).with_density(0.0))
    with pytest.raises(ValueError):
        pg.L2(pg.LebesgueMeasure(S2).with_density(-1.0))
    with pytest.raises(ValueError):
        pg.LebesgueMeasure(S2).with_density(lambda r: r, radial=True)                     # no radius on a sphere
    with pytest.raises(Unsupported, match="Sobolev"):
        pg.Sobolev(pg.LebesgueMeasure(S2).with_density(0.5), 1.5, length_scale=0.5)
    with pytest.raises(Unsupported, match="Lebesgue measure"):
        pg.Laplacian(SHELL, bc={"inner": "neumann", "outer": "neumann"}).bessel_potential(2.0, 2.0, [0, 1], modes=4, space=radial_space())
