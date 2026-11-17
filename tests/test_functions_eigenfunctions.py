"""The Laplacian eigenfunctions of the bank (plan 021 P1 WP-E): what the generic contract cannot see. Orthonormality and
the eigenvalue equations against independent quadrature and finite differences, the boundary conditions, the Robin
root finding, the zeros of the Bessel functions, the limits at r = 0 and the refusals."""
import itertools
import math

import numpy as np
import pytest
from scipy.special import jn_zeros, jv, spherical_jn

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.functions.families.eigenfunctions import ball_wavenumber

I01 = pg.Interval("[0, 1]")
I_SHIFTED = pg.Interval("[-2, 3]")
ALL_BCS = ("dirichlet", "neumann", "dn", "nd")
DISK_SHIFTED = pg.Disk(2.0, centre=(0.3, -0.5))
BALL_SHIFTED = pg.Ball(1.7, centre=(0.1, -0.2, 0.3))


def lo_index(bc):
    return 0 if bc == "neumann" else 1


def gl(a, b, n=80):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * (t + 1) + a, 0.5 * (b - a) * w


def interval_f(D, **kw):
    return F.interval.laplace_eigen(D, **kw)


def second_difference(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2


# --------------------------------------------------------------------------------------------- interval
@pytest.mark.parametrize("D", [I01, I_SHIFTED])
@pytest.mark.parametrize("bc", ALL_BCS)
def test_interval_orthonormal_and_eigenvalue(D, bc):
    x, w = gl(D.a, D.b)
    L = D.b - D.a
    ns = range(lo_index(bc), lo_index(bc) + 5)
    modes = [interval_f(D, n=n, bc=bc) for n in ns]
    G = np.array([[w @ (f(x) * g(x)) for g in modes] for f in modes])
    assert np.allclose(G, np.eye(len(modes)), atol=1e-12)
    for n, f in zip(ns, modes):
        k = (n if bc in ("dirichlet", "neumann") else n - 0.5) * math.pi / L
        assert f.closed["eigenvalue"][0] == pytest.approx(k * k, rel=1e-14)
        assert f.closed["l2_norm_sq"][0] == 1.0
        assert np.max(np.abs(-second_difference(f, x) - k * k * f(x))) <= 1e-4 * k * k * np.max(np.abs(f(x))) + 1e-6
        v, status, _ = f.closed["integral"]
        assert v == pytest.approx(w @ f(x), abs=1e-12) and status == "derived"


@pytest.mark.parametrize("bc, left, right", [("dirichlet", 0, 0), ("neumann", 1, 1), ("dn", 0, 1), ("nd", 1, 0)])
def test_interval_boundary_conditions(bc, left, right):
    """Value 0 (kind 0) or derivative 0 (kind 1) at each end, for n = 1..4."""
    for n in range(1, 5):
        f = interval_f(I_SHIFTED, n=n, bc=bc)
        for kind, e in zip((left, right), (I_SHIFTED.a, I_SHIFTED.b)):
            val = abs(float(f(e))) if kind == 0 else abs(float(f.gradient(e)[..., 0]))
            assert val < 1e-12


def test_interval_vanishes_at_boundary_only_for_dirichlet():
    for bc in ALL_BCS:
        assert interval_f(I01, n=2, bc=bc).regularity.vanishes_at_boundary == (1 if bc == "dirichlet" else None)
    assert interval_f(I01, n=2, bc="robin", beta=(1.0, 1.0)).regularity.vanishes_at_boundary is None
    assert interval_f(I01, n=2, bc="dirichlet").element(pg.L2(I01)) is not None


def test_interval_known_values():
    f = interval_f(I_SHIFTED, n=3, bc="dirichlet")
    assert f(-1.0) == pytest.approx(math.sqrt(2 / 5) * math.sin(3 * math.pi / 5))
    assert interval_f(I_SHIFTED, n=0, bc="neumann")(0.3) == pytest.approx(1 / math.sqrt(5))
    assert interval_f(I01, n=2, bc="nd").closed["eigenvalue"][0] == pytest.approx((1.5 * math.pi) ** 2)
    assert interval_f(I01, n=1, bc="dn").closed["integral"][0] == pytest.approx(math.sqrt(2) * 2 / math.pi)


@pytest.mark.parametrize("beta", [(0.3, 0.0), (1.5, 0.7), (4.0, 9.0), (0.0, 25.0), (1e-6, 1e-6)])
def test_interval_robin_equations_and_ordering(beta):
    D = I_SHIFTED
    L = D.b - D.a
    x, w = gl(D.a, D.b, 120)
    modes = [interval_f(D, n=n, bc="robin", beta=beta) for n in range(1, 6)]
    G = np.array([[w @ (f(x) * g(x)) for g in modes] for f in modes])
    assert np.allclose(G, np.eye(5), atol=1e-11)
    lam = [f.closed["eigenvalue"][0] for f in modes]
    assert all(a < b for a, b in itertools.pairwise(lam))
    for n, (f, lm) in enumerate(zip(modes, lam), start=1):
        assert (n - 1) * math.pi / L - 1e-12 <= math.sqrt(lm) <= n * math.pi / L + 1e-12     # Neumann <= Robin <= Dirichlet
        # outward normal derivative + beta u = 0 at each end
        ua, ub = float(f(D.a)), float(f(D.b))
        dua, dub = float(f.gradient(D.a)[..., 0]), float(f.gradient(D.b)[..., 0])
        scale = max(abs(ua), abs(ub), abs(dua), abs(dub))
        assert abs(-dua + beta[0] * ua) <= 1e-12 * scale and abs(dub + beta[1] * ub) <= 1e-12 * scale
        assert np.max(np.abs(-second_difference(f, x) - lm * f(x))) <= 1e-4 * max(lm, 1.0) * np.max(np.abs(f(x))) + 1e-6
        assert f.closed["integral"][0] == pytest.approx(w @ f(x), abs=1e-11)


def test_interval_robin_limits():
    """beta = 0 is Neumann (index n - 1), beta -> infinity Dirichlet (Dirichlet at one end: DN)."""
    x, _ = gl(0.0, 1.0, 20)
    for n in range(1, 5):
        assert np.allclose(interval_f(I01, n=n, bc="robin", beta=(0.0, 0.0))(x),
                           interval_f(I01, n=n - 1, bc="neumann")(x), atol=1e-12)
        assert np.allclose(interval_f(I01, n=n, bc="robin", beta=(1e9, 1e9))(x),
                           interval_f(I01, n=n, bc="dirichlet")(x), atol=1e-6)
        assert np.allclose(interval_f(I01, n=n, bc="robin", beta=(1e9, 0.0))(x), interval_f(I01, n=n, bc="dn")(x), atol=1e-6)


def test_interval_robin_lowest_eigenvalue_small_beta():
    """For small betas the first eigenvalue is ~ (beta_a + beta_b) / L (the Rayleigh quotient of the constant)."""
    f = interval_f(pg.Interval("[0, 2]"), n=1, bc="robin", beta=(1e-4, 3e-4))
    assert f.closed["eigenvalue"][0] == pytest.approx(4e-4 / 2.0, rel=1e-3)


def test_interval_scaling_keeps_the_eigenvalue():
    f = interval_f(I01, n=3, bc="dirichlet")
    g = f.scaled(-2.5)
    assert g.closed["eigenvalue"] == f.closed["eigenvalue"] and isinstance(g, type(f))
    assert g.closed["l2_norm_sq"][0] == pytest.approx(6.25)
    h = interval_f(I01, n=1, bc="dn").normalised()
    assert h.closed["eigenvalue"][0] == pytest.approx((math.pi / 2) ** 2) and h.closed["integral"][0] == 1.0
    x, w = gl(0.0, 1.0)
    assert w @ h(x) == pytest.approx(1.0)
    with pytest.raises(pg.Refused, match="integral 0"):
        interval_f(I01, n=2, bc="dirichlet").normalised()


@pytest.mark.parametrize("kw, msg", [
    ({"bc": "dirichlet"}, "n is required"), ({"n": 0, "bc": "dirichlet"}, "n must lie in"),
    ({"n": 0, "bc": "robin", "beta": (1.0, 1.0)}, "n must lie in"), ({"n": 1.5}, "must be an integer"),
    ({"n": True}, "must be an integer"), ({"n": 1, "bc": "periodic"}, "bc must be one of"),
    ({"n": 1, "bc": "robin"}, "beta"), ({"n": 1, "bc": "dirichlet", "beta": (1.0, 1.0)}, "beta"),
    ({"n": 1, "bc": "robin", "beta": (-1.0, 1.0)}, ">= 0"), ({"n": 1, "bc": "robin", "beta": (math.inf, 1.0)}, "finite"),
    ({"n": 1, "bc": "robin", "beta": 3.0}, "pair")])
def test_interval_refusals(kw, msg):
    with pytest.raises(pg.Refused, match=msg):
        interval_f(I01, **kw)


def test_claims_and_registry():
    f = interval_f(I01, n=2, bc="robin", beta=(1.0, 2.0))
    cl = {c.key: c for c in f.claims}
    assert cl["eigenvalue"].status == "derived" and cl["l2_norm_sq"].status == "derived"
    assert cl["regularity"].status == "theorem" and cl["regularity"].reason
    e = F.entries()
    assert e["laplace_eigen"].kinds == ("interval", "box", "disk", "ball")      # one family (plan 070 P6, review E2)
    assert not [n for n in e if n.startswith("laplace_eigen_")]
    for k in e["laplace_eigen"].kinds:
        assert 2 <= len(e["laplace_eigen"].example_params(F.geometry.standard(k)[0])) <= 4
    assert e["laplace_eigen"].references
    assert hasattr(F.on(pg.Ball(1.0)), "laplace_eigen") and hasattr(F.on(pg.Disk(1.0)), "laplace_eigen")
    with pytest.raises(pg.Refused, match="not defined on a shell"):
        F.ball.laplace_eigen(pg.Shell(0.5, 1.0), n=1, l=0, j=0)
    with pytest.raises(pg.Refused, match="on a disk takes n, m, trig, bc"):
        F.ball.laplace_eigen(pg.Disk(1.0), n=1, l=0, j=0)


# --------------------------------------------------------------------------------------------- box
def box_grid(D, n=24):
    nodes = [gl(s.a, s.b, n) for s in D.sides]
    G = np.meshgrid(*[a for a, _ in nodes], indexing="ij")
    W = np.ones_like(G[0])
    for i, (_, wi) in enumerate(nodes):
        shape = [1] * len(nodes)
        shape[i] = -1
        W = W * wi.reshape(shape)
    return [g.ravel() for g in G], W.ravel()


@pytest.mark.parametrize("D", [pg.Box((0.0, 1.0), (-1.0, 2.0)), pg.Box((0.0, 1.0), (0.0, 2.0), (-1.0, 0.5))])
def test_box_orthonormal_eigenvalue_and_integral(D):
    d = len(D.sides)
    X, w = box_grid(D)
    bc = ("dirichlet", "neumann", "nd")[:d]
    idx = list(itertools.product(*[range(lo_index(b), lo_index(b) + 2) for b in bc]))
    funcs = [F.box.laplace_eigen(D, n=i, bc=bc) for i in idx]
    V = np.array([f(*X) for f in funcs])
    assert np.allclose((V * w) @ V.T, np.eye(len(funcs)), atol=1e-11)
    for f, i in zip(funcs, idx):
        ks = [(ni if b in ("dirichlet", "neumann") else ni - 0.5) * math.pi / (s.b - s.a) for s, b, ni in zip(D.sides, bc, i)]
        assert f.closed["eigenvalue"][0] == pytest.approx(sum(k * k for k in ks), rel=1e-13)
        assert f.closed["integral"][0] == pytest.approx(w @ f(*X), abs=1e-11)
        assert f.closed["l2_norm_sq"][0] == 1.0


def test_box_is_the_product_and_solves_the_pde():
    D = pg.Box((0.0, 1.0), (0.0, 2.0))
    f = F.box.laplace_eigen(D, n=(2, 3), bc=("robin", "dn"), beta=((0.5, 1.5), None))
    fx = interval_f(pg.Interval(0.0, 1.0), n=2, bc="robin", beta=(0.5, 1.5))
    fy = interval_f(pg.Interval(0.0, 2.0), n=3, bc="dn")
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 1, 30), rng.uniform(0, 2, 30)
    assert np.allclose(f(x, y), fx(x) * fy(y), rtol=1e-13)
    assert f(np.stack([x, y], axis=-1)).shape == (30,)
    lam = f.closed["eigenvalue"][0]
    assert lam == pytest.approx(fx.closed["eigenvalue"][0] + fy.closed["eigenvalue"][0])
    h = 1e-4
    lap = (f(x + h, y) + f(x - h, y) + f(x, y + h) + f(x, y - h) - 4 * f(x, y)) / h ** 2
    assert np.max(np.abs(-lap - lam * f(x, y))) <= 1e-4 * lam * np.max(np.abs(f(x, y)))
    assert f.gradient(x, y).shape == (30, 2)


def test_box_vanishing_claim_and_refusals():
    D = pg.Box((0.0, 1.0), (0.0, 1.0))
    f = F.box.laplace_eigen(D, n=(1, 2), bc="dirichlet")
    assert f.regularity.vanishes_at_boundary == 1
    assert F.box.laplace_eigen(D, n=(1, 2), bc=("dirichlet", "neumann")).regularity.vanishes_at_boundary is None
    assert f(0.0, 0.3) == pytest.approx(0.0, abs=1e-15) and f(0.4, 1.0) == pytest.approx(0.0, abs=1e-15)
    for kw, msg in [({"n": (1,)}, "tuple of 2"), ({"n": None}, "tuple of 2"), ({"n": (1, 0)}, "n must lie in"),
                    ({"n": (1, 1), "bc": ("dirichlet",)}, "tuple of 2"), ({"n": (1, 1), "bc": "robin"}, "beta"),
                    ({"n": (1, 1), "beta": ((1.0, 1.0), None)}, "beta"), ({"n": (1, 1), "beta": (None,)}, "beta"),
                    ({"n": (1.0, 1)}, "integer")]:
        with pytest.raises(pg.Refused, match=msg):
            F.box.laplace_eigen(D, **kw)


# --------------------------------------------------------------------------------------------- disk
def polar_grid(D, nr=60, nphi=64):
    r, wr = gl(0.0, D.radius, nr)
    phi = (np.arange(nphi) + 0.5) * 2 * math.pi / nphi
    Rr, Pp = np.meshgrid(r, phi, indexing="ij")
    W = (wr * r)[:, None] * np.full(nphi, 2 * math.pi / nphi)[None, :]
    return D.centre[0] + Rr * np.cos(Pp), D.centre[1] + Rr * np.sin(Pp), W


def disk_modes(bc):
    return [{"n": n, "m": m, "trig": trig, "bc": bc} for m in range(3) for n in range(0 if (bc == "neumann" and m == 0) else 1, 3)
            for trig in (("cos",) if m == 0 else ("cos", "sin"))]


@pytest.mark.parametrize("bc", ["dirichlet", "neumann"])
def test_disk_orthonormal_eigenvalue_and_integral(bc):
    D = DISK_SHIFTED
    x, y, W = polar_grid(D)
    funcs = [F.disk.laplace_eigen(D, **kw) for kw in disk_modes(bc)]
    V = np.array([f(x, y).ravel() for f in funcs])
    assert np.allclose((V * W.ravel()) @ V.T, np.eye(len(funcs)), atol=1e-10)
    rng = np.random.default_rng(2)
    r, a = rng.uniform(0.2, 1.8, 20), rng.uniform(0, 2 * math.pi, 20)
    px, py = D.centre[0] + r * np.cos(a), D.centre[1] + r * np.sin(a)
    h = 1e-4
    for f in funcs:
        lam = f.closed["eigenvalue"][0]
        lap = (f(px + h, py) + f(px - h, py) + f(px, py + h) + f(px, py - h) - 4 * f(px, py)) / h ** 2
        assert np.max(np.abs(-lap - lam * f(px, py))) <= 1e-4 * max(lam, 1.0) * np.max(np.abs(f(x, y)))
        assert f.closed["integral"][0] == pytest.approx((W * f(x, y)).sum(), abs=1e-10)


def test_disk_boundary_conditions_and_eigenvalues():
    D = pg.Disk(1.5)
    a = np.linspace(0, 2 * math.pi, 17)
    bx, by = 1.5 * np.cos(a), 1.5 * np.sin(a)
    for kw in disk_modes("dirichlet"):
        f = F.disk.laplace_eigen(D, **kw)
        assert np.max(np.abs(f(bx, by))) < 1e-12 and f.regularity.vanishes_at_boundary == 1
        assert f.closed["eigenvalue"][0] == pytest.approx((float(jn_zeros(kw["m"], kw["n"])[-1]) / 1.5) ** 2, rel=1e-13)
    for kw in disk_modes("neumann"):
        f = F.disk.laplace_eigen(D, **kw)
        g = f.gradient(bx, by)
        assert np.max(np.abs(g[:, 0] * np.cos(a) + g[:, 1] * np.sin(a))) < 1e-11
        assert f.regularity.vanishes_at_boundary is None
    ev = lambda **kw: F.disk.laplace_eigen(D, **kw).closed["eigenvalue"][0]
    assert ev(n=1, m=0) == pytest.approx((2.404825557695773 / 1.5) ** 2)
    assert ev(n=1, m=1, bc="neumann") == pytest.approx((1.841183781340659 / 1.5) ** 2)
    assert ev(n=1, m=0, bc="neumann") == pytest.approx((3.831705970207512 / 1.5) ** 2)


def test_disk_values_at_the_centre_and_the_constant():
    D = pg.Disk(2.0, centre=(1.0, 1.0))
    c = F.disk.laplace_eigen(D, n=0, m=0, bc="neumann")
    assert c(1.0, 1.0) == pytest.approx(1 / math.sqrt(4 * math.pi)) and c.closed["eigenvalue"][0] == 0.0
    assert c.closed["integral"][0] == pytest.approx(math.sqrt(4 * math.pi))
    N = 1 / math.sqrt(4 * math.pi * jv(1, 2.404825557695773) ** 2)
    f0 = F.disk.laplace_eigen(D, n=1, m=0)
    assert f0(1.0, 1.0) == pytest.approx(N)                       # J_0(0) = 1: the limit, not 0
    assert f0(1.0 + 1e-9, 1.0) == pytest.approx(N, rel=1e-9)
    for m, trig in itertools.product((1, 2), ("cos", "sin")):
        f = F.disk.laplace_eigen(D, n=1, m=m, trig=trig)
        assert f(1.0, 1.0) == 0.0 and abs(f(1.0 + 1e-7, 1.0 + 1e-7)) < 1e-6


def test_disk_gradient_at_the_centre():
    D = pg.Disk(1.0)
    g = F.disk.laplace_eigen(D, n=1, m=1).gradient(0.0, 0.0)           # ~ N k x / 2 near the centre
    assert g[0] > 0 and abs(g[1]) < 1e-14
    s = F.disk.laplace_eigen(D, n=1, m=1, trig="sin").gradient(0.0, 0.0)
    assert s[1] > 0 and abs(s[0]) < 1e-14


@pytest.mark.parametrize("kw, msg", [
    ({"m": 0}, "n is required"), ({"n": 1}, "m is required"), ({"n": 0, "m": 0}, "n must lie in"),
    ({"n": 0, "m": 1, "bc": "neumann"}, "n must lie in"), ({"n": 1, "m": -1}, "m must lie in"),
    ({"n": 1, "m": 0, "trig": "sin"}, "needs m >= 1"), ({"n": 1, "m": 1, "trig": "tan"}, "trig must be"),
    ({"n": 1, "m": 0, "bc": "robin"}, "bc must be"), ({"n": 1.0, "m": 0}, "integer")])
def test_disk_refusals(kw, msg):
    with pytest.raises(pg.Refused, match=msg):
        F.disk.laplace_eigen(pg.Disk(1.0), **kw)


# --------------------------------------------------------------------------------------------- ball
def ball_grid(D, nr=40, nt=24, nphi=48):
    r, wr = gl(0.0, D.radius, nr)
    mu, wm = np.polynomial.legendre.leggauss(nt)
    phi = (np.arange(nphi) + 0.5) * 2 * math.pi / nphi
    Rr, Mm, Pp = np.meshgrid(r, mu, phi, indexing="ij")
    s = np.sqrt(1 - Mm ** 2)
    W = (wr * r ** 2)[:, None, None] * wm[None, :, None] * np.full(nphi, 2 * math.pi / nphi)[None, None, :]
    c = D.centre
    return c[0] + Rr * s * np.cos(Pp), c[1] + Rr * s * np.sin(Pp), c[2] + Rr * Mm, W


def ball_modes(bc, lmax=2, nmax=2):
    return [{"n": n, "l": l, "j": j, "bc": bc} for l in range(lmax + 1)
            for n in range(0 if (bc == "neumann" and l == 0) else 1, nmax + 1) for j in range(2 * l + 1)]


@pytest.mark.parametrize("bc", ["dirichlet", "neumann"])
def test_ball_orthonormal_and_integral(bc):
    D = BALL_SHIFTED
    x, y, z, W = ball_grid(D)
    funcs = [F.ball.laplace_eigen(D, **kw) for kw in ball_modes(bc)]
    V = np.array([f(x, y, z).ravel() for f in funcs])
    assert np.allclose((V * W.ravel()) @ V.T, np.eye(len(funcs)), atol=1e-9)
    for f in funcs:
        assert f.closed["integral"][0] == pytest.approx((W * f(x, y, z)).sum(), abs=1e-9)
        assert f.closed["l2_norm_sq"][0] == 1.0


@pytest.mark.parametrize("bc", ["dirichlet", "neumann"])
def test_ball_eigenvalue_equation_and_boundary(bc):
    D = BALL_SHIFTED
    centre = np.asarray(D.centre)
    rng = np.random.default_rng(3)
    u = rng.normal(size=(25, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    P = centre + u * rng.uniform(0.2, 1.5, 25)[:, None]
    h = 1e-4
    for kw in ball_modes(bc):
        f = F.ball.laplace_eigen(D, **kw)
        lam = f.closed["eigenvalue"][0]
        lap = -6 * f(*P.T)
        for i in range(3):
            E = np.zeros(3)
            E[i] = h
            lap = lap + f(*(P + E).T) + f(*(P - E).T)
        lap /= h ** 2
        umax = np.max(np.abs(f(*P.T)))
        assert np.max(np.abs(-lap - lam * f(*P.T))) <= 2e-4 * max(lam, 1.0) * umax + 1e-6
        if bc == "dirichlet":
            assert np.max(np.abs(f(*(centre + D.radius * u).T))) < 1e-11
        else:                                                    # d_r u = 0 on the sphere: a central difference in r
            dr = 1e-5
            deriv = (f(*(centre + (D.radius + dr) * u).T) - f(*(centre + (D.radius - dr) * u).T)) / (2 * dr)
            assert np.max(np.abs(deriv)) < 1e-7 * (1 + umax * math.sqrt(lam))


def test_ball_centre_limit():
    """The correct r = 0 limit: N Y_00 for l = 0 (intervalinf returned 0 there), 0 for l >= 1; no NaN at the centre."""
    D = pg.Ball(1.3, centre=(0.5, 0.5, 0.5))
    c = D.centre
    for bc in ("dirichlet", "neumann"):
        f = F.ball.laplace_eigen(D, n=1, l=0, j=0, bc=bc)
        v0 = float(f(*c))
        assert np.isfinite(v0) and v0 != 0.0
        for eps in (1e-3, 1e-6, 1e-9):
            assert float(f(c[0] + eps, c[1], c[2])) == pytest.approx(v0, rel=10 * eps ** 2 + 1e-12)
    for l in (1, 2, 3):
        for j in range(2 * l + 1):
            g = F.ball.laplace_eigen(D, n=1, l=l, j=j)
            assert float(g(*c)) == 0.0
            assert abs(float(g(c[0] + 1e-6, c[1] - 2e-6, c[2] + 1e-6))) < 1e-4
    X = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    assert np.all(np.isfinite(F.ball.laplace_eigen(D, n=2, l=2, j=4, bc="neumann")(X)))


def test_ball_l0_is_the_sinc():
    """l = 0, Dirichlet: u = sin(n pi r / R) / (r sqrt(2 pi R)), with the finite limit n pi / R / sqrt(2 pi R) at r = 0."""
    R = 2.0
    D = pg.Ball(R)
    r = np.array([0.3, 0.9, 1.7])
    for n in (1, 2, 3):
        f = F.ball.laplace_eigen(D, n=n, l=0, j=0)
        assert np.allclose(f(r, 0 * r, 0 * r), np.sin(n * math.pi * r / R) / r / math.sqrt(2 * math.pi * R), rtol=1e-12)
        assert f.closed["eigenvalue"][0] == pytest.approx((n * math.pi / R) ** 2, rel=1e-14)
        assert f(0.0, 0.0, 0.0) == pytest.approx(n * math.pi / R / math.sqrt(2 * math.pi * R), rel=1e-12)


def test_ball_constant_mode_and_zonal_harmonic():
    D = pg.Ball(2.0)
    c = F.ball.laplace_eigen(D, n=0, l=0, j=0, bc="neumann")
    vol = 4 * math.pi * 8 / 3
    assert c(0.3, 0.1, -0.2) == pytest.approx(1 / math.sqrt(vol)) and c.closed["eigenvalue"][0] == 0.0
    assert c.closed["integral"][0] == pytest.approx(math.sqrt(vol))
    f = F.ball.laplace_eigen(D, n=1, l=1, j=0)               # ~ z j_1(k r) / r: odd in z
    assert f(0.2, 0.0, 0.5) * f(0.2, 0.0, -0.5) < 0 and f(0.5, 0.0, 0.0) == pytest.approx(0.0, abs=1e-15)


def test_spherical_bessel_zeros_are_complete_and_ordered():
    dense = np.linspace(1e-4, 60, 600001)
    for l in (0, 1, 2, 5, 10):
        dz = [ball_wavenumber(1.0, l, n, "dirichlet") for n in range(1, 9)]       # views on the Laplacian's spectrum
        nz = [ball_wavenumber(1.0, l, n, "neumann") for n in range(1, 9)]
        for zeros, func in ((dz, lambda x, l=l: spherical_jn(l, x)), (nz, lambda x, l=l: spherical_jn(l, x, True))):
            z = np.array(zeros)
            assert np.all(np.diff(z) > 0) and np.max(np.abs(func(z))) < 1e-12
            v = func(dense)
            crossings = dense[:-1][np.sign(v[1:]) != np.sign(v[:-1])]
            found = crossings[crossings < z[-1] + 0.01]
            assert len(found) == 8 and np.allclose(found, z, atol=2e-4)           # none skipped: the n-th is the n-th


def test_ball_eigenvalues_known():
    D = pg.Ball(1.0)
    ev = lambda **kw: F.ball.laplace_eigen(D, **kw).closed["eigenvalue"][0]
    assert ev(n=1, l=1, j=1) == pytest.approx(4.493409457909064 ** 2)
    assert ev(n=1, l=1, j=1, bc="neumann") == pytest.approx(2.081575977801 ** 2, rel=1e-10)
    assert ev(n=1, l=0, j=0, bc="neumann") == pytest.approx(4.493409457909064 ** 2)       # j_0' = -j_1
    assert ev(n=2, l=2, j=0) == ev(n=2, l=2, j=4)                                          # 2l + 1 fold degeneracy


@pytest.mark.parametrize("kw, msg", [
    ({"n": 1, "j": 0}, "l is required"), ({"n": 1, "l": 1}, "j is required"), ({"n": 1, "l": 1, "j": 3}, "j must lie in"),
    ({"n": 1, "l": 1, "j": -1}, "j must lie in"), ({"n": 0, "l": 1, "j": 0}, "n must lie in"),
    ({"n": 0, "l": 0, "j": 0}, "n must lie in"), ({"l": 0, "j": 0}, "n is required"),
    ({"n": 1, "l": 0, "j": 0, "bc": "robin"}, "bc must be"), ({"n": 1, "l": 0.5, "j": 0}, "integer"),
    ({"n": 1, "l": 100, "j": 0}, "l must lie in")])
def test_ball_refusals(kw, msg):
    with pytest.raises(pg.Refused, match=msg):
        F.ball.laplace_eigen(pg.Ball(1.0), **kw)


# --------------------------------------------------------------------------------------------- review fixes
@pytest.mark.parametrize("beta", [(0.0, 1e4), (0.0, 1e6), (0.0, 1e8), (0.0, 1e10), (1e8, 0.0), (1e6, 1e6), (1e8, 1e8)])
@pytest.mark.parametrize("D", [I01, pg.Interval("[-0.7, 1.9]")])
def test_robin_norm_is_exact_for_large_beta(D, beta):
    """The claimed l2_norm_sq = 1 holds to rounding even where phi_b -> pi/2 (cos phi_b = k / hypot(k, beta_b) is
    passed, not recovered as sqrt(1 - sin^2), which cancelled to 5e-10 at beta_b = 1e8)."""
    x, w = gl(float(D.a), float(D.b), 300)
    for n in (1, 2):
        f = interval_f(D, n=n, bc="robin", beta=beta)
        assert w @ f(x) ** 2 == pytest.approx(1.0, abs=2e-13)
        assert f.closed["l2_norm_sq"][0] == 1.0
        assert w @ f(x) == pytest.approx(f.closed["integral"][0], abs=1e-11)


def test_robin_large_beta_b_mirrors_beta_a():
    """The two ends are treated alike: reflecting the interval swaps (beta_a, beta_b)."""
    x, _ = gl(0.0, 1.0, 200)
    f = interval_f(I01, n=2, bc="robin", beta=(0.0, 1e8))
    g = interval_f(I01, n=2, bc="robin", beta=(1e8, 0.0))
    assert f.closed["eigenvalue"][0] == pytest.approx(g.closed["eigenvalue"][0], rel=1e-13)
    assert min(np.max(np.abs(f(x) - g(1.0 - x))), np.max(np.abs(f(x) + g(1.0 - x)))) <= 1e-6  # equal up to sign


@pytest.mark.parametrize("bc", ["dirichlet", "neumann", "dn", "nd"])
def test_exact_phase_cosines_for_the_pure_conditions(bc):
    from pygeoinf0.functions.families.eigenfunctions import interval_mode
    m = interval_mode(0.0, 1.0, bc, 2)
    assert m.ca in (0.0, 1.0) and m.sa in (0.0, 1.0) and m.ca + m.sa == 1.0
    assert m.cos_phib in (0.0, 1.0) and m.sin_phib in (0.0, 1.0) and m.cos_phib + m.sin_phib == 1.0


@pytest.mark.parametrize("kw", [
    {"n": 1, "bc": ["dirichlet"]}, {"n": 1, "bc": {"a": 1}}, {"n": 1, "bc": 5}, {"n": 1, "bc": None},
    {"n": 1, "bc": "robin", "beta": "12"}, {"n": 1, "bc": "robin", "beta": b"12"}, {"n": 1, "bc": "robin", "beta": (1.0,)},
    {"n": 1, "bc": "robin", "beta": ("a", 1.0)}, {"n": 1, "bc": "robin", "beta": (1.0, 1.0, 1.0)}])
def test_interval_badly_typed_inputs_are_refused(kw):
    with pytest.raises(pg.Refused):
        interval_f(I01, **kw)


@pytest.mark.parametrize("kw", [
    {"n": (1, 1), "bc": 5}, {"n": (1, 1), "bc": (["dirichlet"], "dirichlet")}, {"n": (1, 1), "bc": None},
    {"n": (1, 1), "bc": ({"x": 1}, "neumann")}, {"n": (1, 1), "bc": "robin", "beta": ("12", "12")},
    {"n": (1, 1), "bc": "robin", "beta": 3}])
def test_box_badly_typed_inputs_are_refused(kw):
    with pytest.raises(pg.Refused):
        F.box.laplace_eigen(pg.Box((0.0, 1.0), (0.0, 1.0)), **kw)
