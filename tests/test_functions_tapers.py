"""The tapers and radial profiles of the function bank (plan 021 P1, work package A): cosine_taper, raised_cosine,
smoothstep, bspline_profile. Checks what the contract test cannot see: values against independent formulas (numpy,
scipy.special.betainc, scipy.interpolate.BSpline), the closed forms against exact quadrature for EVERY example (the
contract leaves out the C^0/C^1 ones its reference integral cannot verify), the claimed regularity numerically at each
junction, the special values, the r = 0 limits, the refusals and the claims."""
import itertools
import math

import numpy as np
import pytest
from scipy import integrate, special
from scipy.interpolate import BSpline

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.functions import geometry as geo
from pygeoinf0.functions.families import tapers
from pygeoinf0.functions.families.tapers import FULL_EXAMPLES, SMOOTHSTEP, TAPER_KINDS, cardinal_bspline
from pygeoinf0.spaces.membership import check

FAMILIES = tuple(FULL_EXAMPLES)
BSPLINE_PEAK = {1: 1.0, 2: 3 / 4, 3: 2 / 3, 4: 115 / 192, 5: 11 / 20}


# --------------------------------------------------------------------------------------- independent profiles
def profile_cosine(t, a):
    t = np.asarray(t, float)
    s = np.clip((t - a) / (1 - a), 0, 1)
    return np.where(t >= 1, 0.0, 0.5 * (1 + np.cos(math.pi * s)))


def profile_smoothstep(t, n):
    t = np.asarray(t, float)
    return np.where(t >= 1, 0.0, 1 - special.betainc(n + 1, n + 1, np.clip(t, 0, 1)))


def profile_bspline(t, p):
    n = p + 1
    B = BSpline.basis_element(np.arange(n + 1) - n / 2)
    x = np.asarray(t, float) * n / 2
    return np.where(np.abs(t) >= 1, 0.0, np.nan_to_num(B(x, extrapolate=False)) / B(0.0))


def independent(name, params):
    """(t -> profile, breakpoints in t) written independently of the module."""
    if name == "cosine_taper":
        a = params["inner"] / params["radius"]
        return (lambda t: profile_cosine(t, a)), [a]
    if name == "raised_cosine":
        return (lambda t: profile_cosine(t, 0.0)), []
    if name == "smoothstep":
        n = params["order"]
        return (lambda t: profile_smoothstep(t, n)), []
    p = params["degree"]
    return (lambda t: profile_bspline(t, p)), [1 - 2 * j / (p + 1) for j in range(1, (p + 1) // 2 + 1) if 1 - 2 * j / (p + 1) > 0]


def sphere_area(d):
    return 2 * math.pi ** (d / 2) / math.gamma(d / 2)


def exact_integral(D, name, params, power=1):
    """int over D of profile(distance)^power by scipy.integrate.quad, with the profile's breakpoints."""
    prof, br = independent(name, params)
    R, k = params["radius"], geo.kind(D)
    if k == "sphere":
        w = lambda r: 2 * math.pi * D.radius * math.sin(r / D.radius)
    elif k == "circle":
        w = lambda r: 2.0
    else:
        d = geo.ambient(D)
        w = lambda r: sphere_area(d) * r ** (d - 1)
    pts = sorted({b * R for b in br} | {0.0, R})
    total = 0.0
    for lo, hi in itertools.pairwise(pts):
        total += integrate.quad(lambda r: w(r) * float(prof(r / R)) ** power, lo, hi, epsabs=0, epsrel=1e-13,
                                limit=200)[0]
    return total


def cases(full=True):
    out = []
    for name in FAMILIES:
        for k in TAPER_KINDS:
            for i, D in enumerate(geo.standard(k)):
                for j, ex in enumerate(FULL_EXAMPLES[name](D)):
                    out.append(pytest.param(name, D, ex, id=f"{name}-{k}{i}-{j}"))
    return out


def build(name, D, params):
    return F.entries()[name].builder(D, **params)


# --------------------------------------------------------------------------------------- values and closed forms
@pytest.mark.parametrize("name, D, params", cases())
def test_values_against_independent_profile(name, D, params):
    f = build(name, D, params)
    prof, _ = independent(name, f.params)
    c, R = np.asarray(f.params["centre"]), f.params["radius"]
    X, _ = geo.quadrature(D, 10)
    d = geo.distance(D, c, *X.T)
    assert np.allclose(f(*X.T), prof(d / R), atol=1e-13)
    # points straddling the edge of the support and the centre itself
    e = np.array([0.0, 1e-9, 0.3, 0.999, 1.0, 1.001, 2.0]) * R
    P = _points_at_distance(D, c, e)
    assert np.allclose(f(*P.T), prof(e / R), atol=1e-9)
    assert np.all(f(*P.T)[-2:] == 0.0)


def _points_at_distance(D, c, dist):
    """Points of D at the given geodesic distances from c (along a fixed direction), when they exist in D."""
    k = geo.kind(D)
    dist = np.asarray(dist, float)
    if k == "sphere":
        o, Rr = np.asarray(D.centre, float), D.radius
        u = (c - o) / Rr
        v = np.cross(u, [0.0, 0.0, 1.0]) if abs(u[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
        v = v / np.linalg.norm(v)
        return o + Rr * (np.cos(dist / Rr)[:, None] * u + np.sin(dist / Rr)[:, None] * v)
    if k == "circle":
        o, Rr = np.asarray(D.centre, float), D.radius
        u = (c - o) / Rr
        v = np.array([-u[1], u[0]])
        return o + Rr * (np.cos(dist / Rr)[:, None] * u + np.sin(dist / Rr)[:, None] * v)
    e = np.zeros(len(c))
    e[0] = 1.0
    return c + dist[:, None] * e


@pytest.mark.parametrize("name, D, params", cases())
def test_closed_forms_against_exact_quadrature(name, D, params):
    f = build(name, D, params)
    v, status, _ = f.closed["integral"]
    assert v == pytest.approx(exact_integral(D, name, f.params), rel=1e-9)
    l2 = f.closed["l2_norm_sq"][0]
    assert l2 == pytest.approx(exact_integral(D, name, f.params, 2), rel=1e-9)
    assert status == "derived" and f.closed["l2_norm_sq"][1] == "derived"
    assert f.closed["max"][0] == 1.0
    assert f.integral()[0] == v
    c = np.asarray(f.params["centre"])
    assert f(*c) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("name, D, params", cases())
def test_support_peak_and_gradient(name, D, params):
    f = build(name, D, params)
    c, R = np.asarray(f.params["centre"]), f.params["radius"]
    X, _ = geo.quadrature(D, 14)
    vals = f(*X.T)
    d = geo.distance(D, c, *X.T)
    assert np.all(vals[d >= R] == 0.0) and np.all(vals >= -1e-15) and vals.max() <= 1.0 + 1e-14
    assert f.support.kind == "interior" and f.support.radius == R and tuple(f.support.centre) == tuple(c)
    assert R < geo.room(D, c) and R < geo.injectivity(D)
    if f.gradient is None:
        assert (name == "bspline_profile" and f.params["degree"] == 1) or geo.kind(D) not in tapers.FLAT
        return
    P = X[:: max(1, len(X) // 40)]
    P = P[geo.distance(D, c, *P.T) > 0]
    h = 1e-6 * R
    G = f.gradient(*P.T)
    for i in range(P.shape[1]):
        E = np.zeros(P.shape[1])
        E[i] = h
        fd = (f(*(P + E).T) - f(*(P - E).T)) / (2 * h)
        assert np.max(np.abs(fd - G[:, i])) <= 1e-5 * max(1e-12, np.max(np.abs(G))) + 1e-9
    assert np.all(np.abs(f.gradient(*c)) == 0.0)                           # the r = 0 limit: zero gradient


@pytest.mark.parametrize("name, D, params", cases())
def test_membership_follows_the_regularity_claim(name, D, params):
    f = build(name, D, params)
    assert check(f, pg.L2(D))[0] and f.element(pg.L2(D)) is not None
    smax, strict = f.regularity.sobolev_bound()
    assert (smax, strict) == (f.regularity.order + 1.5, True)
    try:
        spaces = {s: pg.Sobolev(D, s, length_scale=0.2) for s in (1, 2)}
    except Exception:                                                      # noqa: BLE001 (no Sobolev space on this kind)
        pytest.skip("no Sobolev space on this domain")
    for s, M in spaces.items():
        assert check(f, M)[0] == (s < smax)
    assert f.element(spaces[1]) is not None


@pytest.mark.parametrize("name, D, params", cases())
def test_claims(name, D, params):
    f = build(name, D, params)
    for c in f.claims:
        assert c.status in ("theorem", "derived", "checked") and (c.status != "theorem" or c.reason)
    reg = next(c for c in f.claims if c.key == "regularity")
    assert reg.status == "theorem" and ("Harris" in reg.reason or "Abramowitz" in reg.reason or "de Boor" in reg.reason)
    assert f.regularity.piecewise and f.regularity.vanishes_at_boundary == "all orders"
    g = f.normalised()
    assert g.integral()[0] == 1.0 and g.closed["integral"][0] == 1.0
    assert g(*np.asarray(f.params["centre"])) == pytest.approx(1.0 / f.integral()[0], rel=1e-12)


# --------------------------------------------------------------------------------------- regularity, numerically
def one_sided(g, x0, side, order, h, deg, n=40):
    """The `order`-th derivative of g at x0 from one side, by a Chebyshev-node polynomial fit on [x0, x0 + side h]."""
    t = 0.5 * (np.cos(np.pi * (np.arange(n) + 0.5) / n) + 1)
    x = x0 + side * h * t
    poly = np.polynomial.Polynomial.fit(x, g(x), deg)
    return poly.deriv(order)(x0)


def assert_regularity_at(g, x0, k, scale, h, deg, jumps=True):
    """Derivatives of order <= k are continuous at x0; order k + 1 jumps (unless jumps is False). The pieces on both
    sides are fitted by a polynomial of degree deg on a window of width h (exact for polynomial pieces)."""
    for j in range(1, k + 2):
        left, right = one_sided(g, x0, -1, j, h, deg), one_sided(g, x0, +1, j, h, deg)
        s = scale ** j
        if j <= k:
            assert abs(left - right) <= 1e-5 * s, (j, left, right)
        elif jumps:
            assert abs(left - right) >= 1e-3 * s, (j, left, right)


def line_profile(name, **params):
    """The function along the line through the centre of Interval [0, 1], as a function of x."""
    D = pg.Interval("[0, 1]")
    f = build(name, D, params)
    return (lambda x: f(x)), f


@pytest.mark.parametrize("inner", [0.0, 0.1, 0.25])
def test_cosine_taper_regularity(inner):
    g, f = line_profile("cosine_taper", centre=(0.5,), radius=0.4, inner=inner)
    w = 0.4 - inner
    assert f.regularity.order == 1
    kw = {"h": 0.01, "deg": 8}
    assert_regularity_at(g, 0.9, 1, math.pi / w, **kw)                    # the edge of the support
    assert_regularity_at(g, 0.1, 1, math.pi / w, **kw)
    if inner > 0:
        assert_regularity_at(g, 0.5 + inner, 1, math.pi / w, **kw)         # the end of the flat top
        assert_regularity_at(g, 0.5 - inner, 1, math.pi / w, **kw)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_smoothstep_regularity(n):
    g, f = line_profile("smoothstep", centre=(0.5,), radius=0.4, order=n)
    assert f.regularity.order == n
    assert_regularity_at(g, 0.9, n, 1 / 0.4, h=0.05, deg=2 * n + 1)
    assert_regularity_at(g, 0.1, n, 1 / 0.4, h=0.05, deg=2 * n + 1)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_bspline_regularity_at_every_knot(p):
    g, f = line_profile("bspline_profile", centre=(0.5,), radius=0.4, degree=p)
    assert f.regularity.order == p - 1
    n = p + 1
    for kx in (-n / 2 + j for j in range(1, n)):                            # the interior knots, in x = t n/2
        assert_regularity_at(g, 0.5 + 0.4 * kx * 2 / n, p - 1, n / (2 * 0.4), h=0.03, deg=p)
    for edge in (0.1, 0.9):                                                 # the edge of the support (an end knot)
        assert_regularity_at(g, edge, p - 1, n / (2 * 0.4), h=0.03, deg=p)
    # the odd degrees have a knot at the centre too (|r|^p): derivative orders < p are continuous across it, order p
    # jumps; the even degrees are a single polynomial there
    assert_regularity_at(g, 0.5, p - 1, n / (2 * 0.4), h=0.03, deg=p, jumps=p % 2 == 1)


# --------------------------------------------------------------------------------------- special values
def test_raised_cosine_special_values():
    D = pg.Interval("[0, 1]")
    f = F.interval.raised_cosine(D, centre=(0.5,), radius=0.4)
    assert f(0.5) == 1.0 and f(0.7) == pytest.approx(0.5) and f(0.9) == 0.0 and f(0.95) == 0.0
    assert f(0.6) == pytest.approx(0.5 * (1 + math.cos(math.pi / 4)))
    assert f.integral()[0] == pytest.approx(0.4, rel=1e-12)                 # int_{-R}^{R} (1 + cos(pi x/R))/2 dx = R
    assert f.closed["l2_norm_sq"][0] == pytest.approx(0.75 * 0.4, rel=1e-12)    # (1/4)(2R + R) = 3R/4
    B = pg.Ball(1.0)
    g = F.ball.raised_cosine(B, radius=0.5)
    assert g.integral()[0] == pytest.approx(2 * math.pi * 0.5 ** 3 * (1 / 3 - 2 / math.pi ** 2), rel=1e-12)
    Dk = pg.Disk(1.0)
    h = F.disk.raised_cosine(Dk, radius=0.5)
    assert h.integral()[0] == pytest.approx(math.pi * 0.25 * (0.5 - 2 / math.pi ** 2), rel=1e-12)


def test_cosine_taper_special_values():
    D = pg.Interval("[0, 1]")
    f = F.interval.cosine_taper(D, centre=(0.5,), radius=0.4, inner=0.1)
    assert f(0.5) == 1.0 and f(0.55) == 1.0 and f(0.6) == 1.0 and f(0.45) == 1.0
    assert f(0.75) == pytest.approx(0.5) and f(0.9) == 0.0 and f(0.25) == pytest.approx(0.5)
    assert f.integral()[0] == pytest.approx(2 * 0.1 + 0.3, rel=1e-12)       # flat 2 inner + the taper 2 * w / 2
    assert f.closed["l2_norm_sq"][0] == pytest.approx(2 * 0.1 + 2 * 0.3 * 3 / 8, rel=1e-12)
    # inner = 0 is the raised cosine, whatever the domain
    B = pg.Ball(1.0)
    a = F.ball.cosine_taper(B, radius=0.6, inner=0.0)
    b = F.ball.raised_cosine(B, radius=0.6)
    X, _ = geo.quadrature(B, 8)
    assert np.array_equal(a(*X.T), b(*X.T)) and a.integral()[0] == b.integral()[0]
    assert F.ball.cosine_taper(B, radius=0.6).params["inner"] == 0.3        # the default inner is radius/2


def test_cosine_taper_flat_top_in_a_ball():
    B = pg.Ball(1.0)
    f = F.ball.cosine_taper(B, radius=0.6, inner=0.2)
    X, _ = geo.quadrature(B, 12)
    r = np.linalg.norm(X, axis=1)
    assert np.all(f(*X.T)[r <= 0.2] == 1.0) and np.all(f(*X.T)[r >= 0.6] == 0.0)
    exact = 4 * math.pi * (0.2 ** 3 / 3 + integrate.quad(
        lambda s: 0.5 * (1 + math.cos(math.pi * (s - 0.2) / 0.4)) * s * s, 0.2, 0.6, epsrel=1e-13)[0])
    assert f.integral()[0] == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_smoothstep_polynomials(n):
    S = np.polynomial.Polynomial(SMOOTHSTEP[n])
    t = np.linspace(0, 1, 101)
    assert np.allclose(S(t), special.betainc(n + 1, n + 1, t), atol=1e-14)   # the regularised incomplete beta
    assert S(0.0) == 0.0 and S(1.0) == pytest.approx(1.0) and S(0.5) == pytest.approx(0.5)
    assert np.allclose(S(t) + S(1 - t), 1.0, atol=1e-14)                    # S_n(t) + S_n(1 - t) = 1
    for j in range(1, n + 1):                                               # derivatives 1..n vanish at both ends
        assert S.deriv(j)(0.0) == 0.0 and abs(S.deriv(j)(1.0)) < 1e-12
    assert abs(S.deriv(n + 1)(1.0)) > 1.0
    assert np.all(S.deriv()(t) >= -1e-14)                                   # nondecreasing, so p is nonincreasing
    assert tuple(S.coef[:2]) == (0.0, 0.0)
    assert S.coef[:3][2] == (3.0 if n == 1 else 0.0)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_smoothstep_in_a_ball_and_mass(n):
    B = pg.Ball(1.0)
    f = F.ball.smoothstep(B, radius=0.5, order=n)
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    P = np.stack([0.5 * t, 0 * t, 0 * t], axis=1)
    assert np.allclose(f(*P.T), 1 - special.betainc(n + 1, n + 1, t), atol=1e-14)
    assert f(0.25, 0.0, 0.0) == pytest.approx(0.5)
    c = np.polynomial.Polynomial(SMOOTHSTEP[n])
    poly = np.polynomial.Polynomial([1.0]) - c                              # the mass 4 pi R^3 int_0^1 p(t) t^2 dt exactly
    mass = 4 * math.pi * 0.5 ** 3 * (poly * np.polynomial.Polynomial([0, 0, 1])).integ()(1.0)
    assert f.integral()[0] == pytest.approx(mass, rel=1e-12)


def test_smoothstep_polynomial_masses_are_rational():
    # 1D, radius 1: int_{-1}^{1} (1 - S_n) = 2 (1 - int_0^1 S_n) and int_0^1 S_n = 1/2 by the symmetry S(t) + S(1 - t) = 1
    D = pg.Interval("[-2, 2]")
    for n in (1, 2, 3):
        f = F.interval.smoothstep(D, centre=(0.0,), radius=1.0, order=n)
        assert f.integral()[0] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_cardinal_bspline_properties(p):
    x = np.linspace(-(p + 1) / 2 - 1, (p + 1) / 2 + 1, 801)
    B = cardinal_bspline(x, p)
    ref = BSpline.basis_element(np.arange(p + 2) - (p + 1) / 2)
    assert np.allclose(B, np.nan_to_num(ref(x, extrapolate=False)), atol=1e-13)
    assert cardinal_bspline(0.0, p) == pytest.approx(BSPLINE_PEAK[p], rel=1e-14)
    assert np.all(B >= -1e-12) and np.allclose(B, cardinal_bspline(-x, p), atol=1e-12)
    assert np.all(B[np.abs(x) >= (p + 1) / 2] == 0.0) or np.allclose(B[np.abs(x) >= (p + 1) / 2], 0.0, atol=1e-13)
    assert integrate.quad(lambda s: float(cardinal_bspline(s, p)), -(p + 1) / 2, (p + 1) / 2, epsrel=1e-13,
                          points=[-(p + 1) / 2 + j for j in range(1, p + 1)], limit=200)[0] == pytest.approx(1.0, rel=1e-12)
    # partition of unity: sum_k B(x - k) = 1
    u = np.random.default_rng(p).uniform(-3, 3, 50)
    assert np.allclose(sum(cardinal_bspline(u - k, p) for k in range(-12, 13)), 1.0, atol=1e-13)
    # unimodal: nonincreasing on [0, inf)
    assert np.all(np.diff(B[x >= 0]) <= 1e-12)
    # the derivative formula against central differences
    if p >= 2:
        h = 1e-6
        d = cardinal_bspline(x, p, 1)
        fd = (cardinal_bspline(x + h, p) - cardinal_bspline(x - h, p)) / (2 * h)
        assert np.max(np.abs(d - fd)) < 2e-6                                       # kinks of B'' within h of a knot
    # B_p' (x) = B_{p-1}(x + 1/2) - B_{p-1}(x - 1/2)
    if p >= 2:
        assert np.allclose(cardinal_bspline(x, p, 1),
                           cardinal_bspline(x + 0.5, p - 1) - cardinal_bspline(x - 0.5, p - 1), atol=1e-12)


def test_bspline_profile_values():
    D = pg.Interval("[0, 1]")
    for p in range(1, 6):
        f = F.interval.bspline_profile(D, centre=(0.5,), radius=0.4, degree=p)
        assert f(0.5) == 1.0 and f(0.9) == 0.0 and f(0.95) == 0.0
        h = (p + 1) / 2
        x = 0.4 * 0.5                                                      # t = 1/2
        assert f(0.5 + x) == pytest.approx(float(cardinal_bspline(h * 0.5, p)) / BSPLINE_PEAK[p], rel=1e-13)
        assert f(0.5 + x) == pytest.approx(f(0.5 - x))
        # the 1D mass: int B dx = 1 and x = t h, so int p dr = 2 radius/(2 h B(0)) = radius/(h B(0))
        assert f.integral()[0] == pytest.approx(2 * 0.4 / (2 * h) / BSPLINE_PEAK[p], rel=1e-11)
    f1 = F.interval.bspline_profile(D, centre=(0.5,), radius=0.4, degree=1)
    assert f1(0.7) == pytest.approx(0.5)                                    # degree 1 is the cone 1 -
    f3 = F.interval.bspline_profile(D, centre=(0.5,), radius=0.4)           # the default degree is 3
    assert f3.params["degree"] == 3 and f3(0.5 + 0.2) == pytest.approx((1 / 6) / (2 / 3), rel=1e-13)   # B3(1) = 1/6


def test_bspline_one_sample_in_a_ball_gradient_and_cone():
    B = pg.Ball(1.0)
    cone = F.ball.bspline_profile(B, radius=0.5, degree=1)
    assert cone.gradient is None and cone.regularity.order == 0
    assert cone(0.25, 0.0, 0.0) == pytest.approx(0.5)
    assert cone.integral()[0] == pytest.approx(4 * math.pi * 0.5 ** 3 * (1 / 3 - 1 / 4), rel=1e-12)   # int (1-t) t^2
    f = F.ball.bspline_profile(B, radius=0.5, degree=3)
    assert f.gradient is not None and np.all(f.gradient(0.0, 0.0, 0.0) == 0.0)


# --------------------------------------------------------------------------------------- the r = 0 limits, every kind
@pytest.mark.parametrize("name, D, params", cases())
def test_centre_limit(name, D, params):
    f = build(name, D, params)
    c = np.asarray(f.params["centre"])
    P = _points_at_distance(D, c, np.array([0.0, 1e-12, 1e-8, 1e-5]))
    v = f(*P.T)
    assert np.all(np.isfinite(v)) and v[0] == pytest.approx(1.0, abs=1e-14)
    assert np.all(np.abs(v - 1.0) < 1e-3)                                   # no jump at the centre
    if f.gradient is not None:
        G = f.gradient(*P.T)
        r = geo.distance(D, c, *P.T)
        assert np.all(np.isfinite(G)) and np.all(np.linalg.norm(G, axis=-1) <= 20 * r / f.params["radius"] ** 2 + 1e-9)


# --------------------------------------------------------------------------------------- validation and refusals
def test_registered_and_listed():
    names = {n for n, _, _ in F.catalogue()}
    assert set(FAMILIES) <= names
    for name in FAMILIES:
        e = F.entries()[name]
        assert e.kinds == TAPER_KINDS and e.references and "bump" not in name
        assert 1 <= len(FULL_EXAMPLES[name](geo.standard("ball")[0])) <= 5
        assert len(e.example_params(geo.standard("interval")[0])) >= 1       # the interval is never limited
    B = pg.Ball(1.0)
    assert F.on(B).smoothstep(radius=0.3, order=2).family == "smoothstep"
    assert F.ball.bspline_profile(B, radius=0.3).params["degree"] == 3


@pytest.mark.parametrize("name", FAMILIES)
def test_common_refusals(name):
    B = pg.Ball(1.0)
    T = pg.Torus((0.0, 1.0), (0.0, 1.0))
    S = pg.Sphere(1.0)
    b = getattr(F.ball, name)
    for bad in (-0.1, 0.0, float("nan"), float("inf"), "0.3", True):
        with pytest.raises(pg.Refused, match="radius"):
            b(B, radius=bad)
    with pytest.raises(pg.Refused, match="leaves the domain"):
        b(B, radius=1.0)
    with pytest.raises(pg.Refused, match="leaves the domain"):
        b(B, radius=0.6, centre=(0.5, 0.0, 0.0))
    with pytest.raises(pg.Refused, match="leaves the domain"):
        getattr(F.torus, name)(T, radius=0.5)                               # half the shortest period is not inside
    with pytest.raises(pg.Refused, match="leaves the domain"):
        getattr(F.sphere, name)(S, radius=math.pi)
    with pytest.raises(pg.Refused, match="Cartesian coordinates"):
        b(B, centre=(0.0, 0.0))
    with pytest.raises(pg.Refused, match="finite"):
        b(B, centre=(0.0, float("nan"), 0.0), radius=0.2)
    with pytest.raises(pg.Refused, match="sequence"):
        b(B, centre="abc", radius=0.2)
    with pytest.raises(pg.Refused, match="does not lie on the sphere"):
        getattr(F.sphere, name)(S, centre=(0.0, 0.0, 2.0), radius=0.2)
    with pytest.raises(pg.Refused, match="outside the torus"):
        getattr(F.torus, name)(T, centre=(2.0, 0.5), radius=0.2)
    with pytest.raises(pg.Refused, match=f"{name} is not defined on a"):
        b(geo.standard("curve")[0], radius=0.2)
    with pytest.raises(TypeError):
        b(B, 0.3)                                                           # keyword-only parameters
    with pytest.raises(TypeError):
        b(B, radius=0.3, width=1.0)                                         # no unknown parameters


def test_family_specific_refusals():
    B = pg.Ball(1.0)
    for bad in (-0.1, 0.5, 0.6, float("nan"), "0.1", True):
        with pytest.raises(pg.Refused, match="inner"):
            F.ball.cosine_taper(B, radius=0.5, inner=bad)
    for bad in (0, 4, 1.5, 2.0, "2", True, None):
        with pytest.raises(pg.Refused, match="order"):
            F.ball.smoothstep(B, radius=0.5, order=bad)
    for bad in (0, 6, 2.5, 3.0, "3", False, None):
        with pytest.raises(pg.Refused, match="degree"):
            F.ball.bspline_profile(B, radius=0.5, degree=bad)
    assert F.ball.smoothstep(B, radius=0.5, order=np.int64(3)).params["order"] == 3
    assert F.ball.cosine_taper(B, radius=0.5, inner=np.float64(0.25)).params["inner"] == 0.25


def test_default_radius_and_centre():
    B = pg.Ball(1.0)
    for name in FAMILIES:
        f = getattr(F.ball, name)(B)
        assert f.params["radius"] == 0.5 and f.params["centre"] == (0.0, 0.0, 0.0)
    T = pg.Torus((0.0, 2 * math.pi), (0.0, 2 * math.pi))
    assert F.torus.raised_cosine(T).params["radius"] == pytest.approx(0.5 * math.pi)


# --------------------------------------------------------------------------------------- the geometry of the kinds
def test_geodesic_and_periodic_distance_are_used():
    S = pg.Sphere(2.0)
    f = F.sphere.raised_cosine(S, radius=1.0)
    p = np.array([0.0, 0.0, 2.0])
    q = np.array([2 * math.sin(0.5), 0.0, 2 * math.cos(0.5)])               # geodesic distance 2 * 0.5 = 1 from p
    assert f(*p) == 1.0 and f(*q) == pytest.approx(0.0, abs=1e-12)
    r = np.array([2 * math.sin(0.25), 0.0, 2 * math.cos(0.25)])
    assert f(*r) == pytest.approx(0.5, abs=1e-12)                           # t = 0.5
    # mass 2 pi R^2 int_0^{rho/R} p(R s) sin s ds, closed form via quad
    ref = integrate.quad(lambda s: 2 * math.pi * 4 * 0.5 * (1 + math.cos(math.pi * s * 2)) * math.sin(s), 0, 0.5,
                         epsrel=1e-13)[0]
    assert f.integral()[0] == pytest.approx(ref, rel=1e-12)
    T = pg.Torus((0.0, 1.0), (0.0, 2.0))
    g = F.torus.raised_cosine(T, centre=(0.05, 1.0), radius=0.3)
    assert g(0.95, 1.0) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.1 / 0.3)))      # across the period
    assert g(0.05, 1.0) == 1.0
    C = pg.Circle(1.0)
    h = F.circle.smoothstep(C, radius=1.0, order=2)
    assert h.integral()[0] == pytest.approx(1.0, rel=1e-12)                 # 2 * int_0^1 (1 - S(s)) ds = 2 * 1/2


# --------------------------------------------------------------------------------------- inner close to radius
def _kind_cases():
    out = []
    for k in TAPER_KINDS:
        for i, D in enumerate(geo.standard(k)):
            c = geo.centre(D)
            out.append(pytest.param(D, 0.4 * min(geo.room(D, c), geo.injectivity(D)), id=f"{k}{i}"))
    return out


@pytest.mark.parametrize("D, radius", _kind_cases())
@pytest.mark.parametrize("ratio", [0.5, 0.9, 0.99, 0.996, 0.999, 0.9999, 0.999999])
def test_cosine_taper_masses_for_inner_close_to_radius(D, radius, ratio):
    """Regression: the integral and the squared L2 norm were taken as a difference of two integrals from 0 of the taper's
    cosine extension, which oscillates ~ ratio/(2 (1 - ratio)) times on [0, inner] and is not resolved by 8 x 64
    nodes (errors 1e-10 .. 0.15 for ratio 0.996 .. 0.9999 on the ball)."""
    f = F.entries()["cosine_taper"].builder(D, radius=radius, inner=ratio * radius)
    for key, power in (("integral", 1), ("l2_norm_sq", 2)):
        ref = exact_integral(D, "cosine_taper", f.params, power)
        assert f.closed[key][0] == pytest.approx(ref, rel=1e-11), (key, ratio)


def test_cosine_taper_thin_taper_values_by_hand():
    """Closed forms that need no quadrature: the interval [0, 1] (centre 1/2, so the interval of radius R < 1/2 is
    inside), integral = 2 inner + 2 (R - inner), l2 = 2 inner + 2 (R - inner) 3/8 (the mean of ((1 + cos)/2)^2 is 3/8),
    for the reported case R = 0.195, and a ball against the exact flat-top plus shell."""
    I = pg.Interval("[0, 1]")
    R = 0.195
    for inner in (0.0, 0.5 * R, 0.995 * R, 0.9999 * R, 0.999999 * R):
        f = F.interval.cosine_taper(I, radius=R, inner=inner)
        assert f.closed["integral"][0] == pytest.approx(2 * inner + (R - inner), rel=1e-12)
        assert f.closed["l2_norm_sq"][0] == pytest.approx(2 * inner + 2 * (R - inner) * 3 / 8, rel=1e-12)
    # the ball, flat top plus taper in closed form: int_inner^R (1 + cos(pi s))/2 4 pi r^2 dr by quad
    B = pg.Ball(1.0)
    f = F.ball.cosine_taper(B, radius=0.4, inner=0.39996)
    top = 4 / 3 * math.pi * 0.39996 ** 3
    taper = integrate.quad(lambda r: 4 * math.pi * r * r * 0.5 * (1 + math.cos(math.pi * (r - 0.39996) / 4e-5)),
                           0.39996, 0.4, epsrel=1e-14)[0]
    assert f.closed["integral"][0] == pytest.approx(top + taper, rel=1e-12)
    # the squared L2 norm of the reported case (l2 0.26803, not 0.30952)
    g = F.ball.cosine_taper(B, radius=0.4, inner=0.9999 * 0.4)
    assert g.closed["l2_norm_sq"][0] == pytest.approx(0.26803, abs=5e-5)


# --------------------------------------------------------------------------------------- contract coverage
def test_registered_examples_are_the_full_lists_and_declare_their_kinks():
    """Since plan 021 P2 the contract integrates radial functions in polar coordinates split at Support.breaks, so
    every parameter set is registered on every kind (until then the kinked and C^0 ones were left out)."""
    reg = F.entries()
    for name, full in FULL_EXAMPLES.items():
        for k in reg[name].kinds:
            for D in geo.standard(k):
                assert reg[name].example_params(D) == full(D), (name, k)
    D = geo.standard("ball")[0]
    f = F.on(D).cosine_taper(radius=0.4, inner=0.1)
    assert f.support.breaks == (0.1,)
    assert F.on(D).raised_cosine(radius=0.4).support.breaks == ()
