"""The compactly supported radial functions of the bank (plan 021, WP-B): wendland, wu, buhmann.

What the automatic contract cannot see: the closed polynomial forms against the literature, the values at the centre
and the edge, the smoothness orders from the exact coefficients, the autoconvolution and dimension-walk definitions
against numerical integration, the exact masses, the positive definiteness claims against radial Fourier transforms, the
refusals, and the claims' statuses."""
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.functions import geometry as geo
from pygeoinf0.functions.families.wendland import (
    derivative,
    expanded,
    horner,
    wendland_profile,
    wu_profile,
)
from pygeoinf0.spaces.membership import check

INTERVAL, DISK, BALL = geo.standard("interval")[0], geo.standard("disk")[0], geo.standard("ball")[0]


def profile_values(family, D, t, **params):
    """The family's values at distances t * radius from the centre along the first axis (radius 0.4)."""
    f = getattr(F.on(D), family)(radius=0.4, **params)
    c = np.asarray(f.support.centre, float)
    P = np.tile(c, (len(t), 1))
    P[:, 0] += 0.4 * np.asarray(t)
    return f(*P.T)


def exact_poly(m, p):
    return [float(a) for a in expanded(m, list(p))]


# ----------------------------------------------------------------------------------------- Wendland's closed forms
WENDLAND_TABLE = {                  # (d, k) -> t -> value; Wendland 1995 / 2005 table, normalised to 1 at 0
    (1, 0): lambda t: 1 - t,
    (1, 1): lambda t: (1 - t) ** 3 * (3 * t + 1),
    (1, 2): lambda t: (1 - t) ** 5 * (8 * t ** 2 + 5 * t + 1),
    (1, 3): lambda t: (1 - t) ** 7 * (21 * t ** 3 + 19 * t ** 2 + 7 * t + 1),
    (3, 0): lambda t: (1 - t) ** 2,
    (3, 1): lambda t: (1 - t) ** 4 * (4 * t + 1),
    (3, 2): lambda t: (1 - t) ** 6 * (35 * t ** 2 + 18 * t + 3) / 3,
    (3, 3): lambda t: (1 - t) ** 8 * (32 * t ** 3 + 25 * t ** 2 + 8 * t + 1),
    (2, 1): lambda t: (1 - t) ** 4 * (4 * t + 1),          # d = 2 and d = 3 share l = floor(d/2) + k + 1
    (5, 1): lambda t: (1 - t) ** 5 * (5 * t + 1),
}


@pytest.mark.parametrize("d, k", sorted(WENDLAND_TABLE))
def test_wendland_matches_the_published_forms(d, k):
    t = np.linspace(0.0, 0.999, 41)
    got = profile_values("wendland", BALL, t, k=k, d=d)
    assert np.allclose(got, WENDLAND_TABLE[d, k](t), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6, 9])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_wendland_degree_smoothness_and_centre_value(d, k):
    m, p, ell = wendland_profile(d, k)
    poly = expanded(m, list(p))
    assert ell == d // 2 + k + 1 and m == ell + k
    assert len(poly) - 1 == d // 2 + 3 * k + 1                       # degree floor(d/2) + 3k + 1
    assert poly[0] == 1 and sum(poly) == 0                           # psi(0) = 1, psi(1) = 0 exactly
    odd = [i for i, a in enumerate(poly) if i % 2 == 1 and a != 0]
    assert min(odd) == 2 * k + 1                                     # the first odd power: no more than C^(2k) at 0
    assert all(poly[i] == 0 for i in range(1, 2 * k + 1, 2))
    # the contact order at the edge is l + k, at least 2k + 1
    dp = poly
    for _ in range(m):
        assert abs(sum(dp)) == 0
        dp = derivative(dp)
    assert sum(dp) != 0


def _num_I(f, t):
    return quad(lambda s: s * f(s), t, 1.0, epsabs=1e-14, epsrel=1e-14)[0]


@pytest.mark.parametrize("d, k", [(1, 1), (3, 1), (3, 2), (2, 2), (5, 1)])
def test_wendland_is_the_iterated_integral_of_the_truncated_power(d, k):
    """psi_{d,k} = I^k (1-t)_+^l by nested numerical quadrature (the definition), normalised."""
    ell = d // 2 + k + 1

    def level(j, t):
        return (1 - t) ** ell if j == 0 else _num_I(lambda s: level(j - 1, s), t)
    t = np.array([0.05, 0.3, 0.55, 0.9])
    want = np.array([level(k, x) for x in t]) / level(k, 0.0)
    assert np.allclose(profile_values("wendland", BALL, t, k=k, d=d), want, rtol=1e-9)


@pytest.mark.parametrize("name, D, params", [("wendland", BALL, {"k": 2}), ("wu", DISK, {"ell": 2, "k": 1}),
                                             ("buhmann", INTERVAL, {})])
def test_value_one_at_the_centre_zero_at_and_beyond_the_edge(name, D, params):
    f = getattr(F.on(D), name)(radius=0.3, **params)
    c = np.asarray(f.support.centre)
    assert f(*c) == pytest.approx(1.0, abs=1e-15)
    for dist in (0.3, 0.3 + 1e-9, 0.35, 0.4):
        e = np.zeros_like(c)
        e[0] = dist
        assert abs(f(*(c + e))) <= 1e-12
    assert f.closed["max"][0] == 1.0
    X, _ = geo.quadrature(D, 24)
    assert f(*X.T).max() <= 1.0 + 1e-15


# ----------------------------------------------------------------------------------------------------------- Wu
def test_wu_matches_the_published_forms():
    t = np.linspace(0.0, 0.999, 31)
    f20 = profile_values("wu", BALL, t, ell=2, k=0)
    f21 = profile_values("wu", BALL, t, ell=2, k=1)
    assert np.allclose(f20, (1 - t) ** 5 * (1 + 5 * t + 9 * t ** 2 + 5 * t ** 3 + t ** 4), rtol=1e-12, atol=1e-14)
    assert np.allclose(f21, (1 - t) ** 4 * (4 + 16 * t + 12 * t ** 2 + 3 * t ** 3) / 4, rtol=1e-12, atol=1e-14)
    assert np.allclose(profile_values("wu", BALL, t, ell=1, k=0), (1 - t) ** 3 * (1 + 3 * t + t ** 2), atol=1e-14)


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_wu_psi_l_is_the_autoconvolution(ell):
    """psi_{l,0}(r/2) = int (1-x^2)^l (1-(r-x)^2)^l dx / psi_l(0) by numerical integration."""
    r = np.array([0.2, 0.7, 1.3, 1.9])
    def conv(rr):
        return quad(lambda x: (1 - x * x) ** ell * (1 - (rr - x) ** 2) ** ell, max(-1, rr - 1), min(1, rr + 1),
                    epsabs=1e-14, epsrel=1e-13)[0]
    num = [conv(rr) for rr in r]
    g0 = quad(lambda x: (1 - x * x) ** (2 * ell), -1, 1, epsabs=1e-14)[0]
    assert np.allclose(profile_values("wu", BALL, r / 2, ell=ell, k=0), np.array(num) / g0, rtol=1e-10)


@pytest.mark.parametrize("ell, k", [(2, 1), (3, 1), (3, 2), (4, 2)])
def test_wu_k_applies_minus_one_over_t_d_dt(ell, k):
    """psi_{l,k}(t) is proportional to (-(1/t) d/dt) psi_{l,k-1}: the ratio of the exact polynomials is constant."""
    _, _, lower = wu_profile(ell, k - 1)
    _, _, upper = wu_profile(ell, k)
    d = [-a for a in derivative(list(lower))]
    shifted = d[1:]                                              # divided by t
    ratio = {Fraction(a, b) for a, b in zip(upper, shifted, strict=False) if b != 0}
    assert len(ratio) == 1 and all(upper[i] * shifted[0] == shifted[i] * upper[0] for i in range(len(shifted)))


@pytest.mark.parametrize("ell, k", [(ell, k) for ell in (1, 2, 3, 4, 6) for k in range(ell + 1)])
def test_wu_smoothness_order_from_the_coefficients(ell, k):
    m, _, poly = wu_profile(ell, k)
    assert poly[0] == 1 and sum(poly) == 0
    odd = [i for i, a in enumerate(poly) if i % 2 == 1 and a != 0]
    assert min(odd) == 2 * (ell - k) + 1                          # C^(2(l-k)) at the centre, no more
    assert m == 2 * ell + 1 - k                                   # contact order at the edge
    assert len(poly) - 1 == 4 * ell + 1 - 2 * k


# ------------------------------------------------------------------------------------------------------ Buhmann
def test_buhmann_values_and_derivatives_at_the_edge():
    t = np.array([0.0, 0.5, 1.0])
    f = profile_values("buhmann", INTERVAL, t[:2])
    assert f[0] == 1.0 and f[1] == pytest.approx(12 * 0.5 ** 4 * math.log(0.5) - 21 * 0.5 ** 4 + 4 - 3 + 1)
    x = np.array([1 - 1e-2, 1 - 1e-3])
    v = profile_values("buhmann", INTERVAL, x)
    assert np.all(v > 0) and v[1] / v[0] == pytest.approx(1e-4 ** 1, rel=0.2)      # contact of order 4 at t = 1
    xs = np.linspace(0, 0.999, 500)
    assert np.all(np.diff(profile_values("buhmann", INTERVAL, xs)) < 0)             # decreasing


# --------------------------------------------------------------------------------- exact masses and unit mass
def exact_mass(poly, ambient, radius, power=1):
    """|S^(a-1)| radius^a int_0^1 poly(t)^power t^(a-1) dt with exact rational arithmetic."""
    q = [Fraction(1)]
    for _ in range(power):
        out = [Fraction(0)] * (len(q) + len(poly) - 1)
        for i, a in enumerate(q):
            for j, b in enumerate(poly):
                out[i + j] += a * b
        q = out
    inner = sum(a / (i + ambient) for i, a in enumerate(q))
    return geo.SPHERE_AREA[ambient] * radius ** ambient * float(inner)


@pytest.mark.parametrize("D, ambient", [(INTERVAL, 1), (DISK, 2), (BALL, 3)])
@pytest.mark.parametrize("family, params", [("wendland", {"k": 0, "d": 4}), ("wendland", {"k": 3}),
                                            ("wendland", {"k": 1, "d": 1}), ("wu", {"ell": 3, "k": 2}),
                                            ("wu", {"ell": 1, "k": 1})])
def test_integral_and_l2_norm_are_the_exact_rational_masses(D, ambient, family, params):
    f = getattr(F.on(D), family)(radius=0.3, **params)
    if family == "wendland":
        m, p, _ = wendland_profile(params.get("d", ambient), params["k"])
    else:
        m, p, _ = wu_profile(params["ell"], params["k"])
    poly = expanded(m, list(p))
    assert f.integral()[0] == pytest.approx(exact_mass(poly, ambient, 0.3), rel=1e-12)
    assert f.closed["l2_norm_sq"][0] == pytest.approx(exact_mass(poly, ambient, 0.3, power=2), rel=1e-12)
    g = f.normalised()
    assert g.integral()[0] == 1.0 and g.params["multiplier"] == pytest.approx(1 / f.integral()[0])
    X, _ = geo.quadrature(D, 12)
    assert np.allclose(g(*X.T) * exact_mass(poly, ambient, 0.3), f(*X.T), rtol=1e-12, atol=1e-15)


def test_buhmann_mass_by_substitution():
    """int_0^1 phi(t) t^2 dt for the ball, by adaptive quadrature."""
    want = 4 * math.pi * 0.2 ** 3 * quad(lambda t: (12 * t ** 4 * math.log(t) - 21 * t ** 4 + 32 * t ** 3
                                                     - 12 * t ** 2 + 1) * t * t, 0, 1, epsabs=1e-14)[0]
    assert F.on(BALL).buhmann(radius=0.2).integral()[0] == pytest.approx(want, rel=1e-12)


def test_sphere_and_circle_use_the_geodesic_distance():
    S, C = geo.standard("sphere")[0], geo.standard("circle")[0]
    f = F.on(S).wendland(k=2, radius=1.2)
    north = np.asarray(f.support.centre)
    theta = np.array([0.0, 0.3, 0.9, 1.19, 1.3])
    P = np.stack([np.sin(theta), 0 * theta, np.cos(theta)], axis=-1)
    assert np.allclose(f(*P.T), WENDLAND_TABLE[3, 2](np.minimum(theta / 1.2, 1.0)) * (theta < 1.2), atol=1e-14)
    assert np.allclose(north, [0, 0, 1])
    g = F.on(C).wu(ell=2, k=1, radius=2.0)
    phi = np.array([0.0, 0.5, 1.5, 2.5])
    Q = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    c = np.asarray(g.support.centre)
    ang = np.arccos(np.clip(Q @ c, -1, 1))
    want = (1 - ang / 2) ** 4 * (4 + 16 * ang / 2 + 12 * (ang / 2) ** 2 + 3 * (ang / 2) ** 3) / 4 * (ang < 2)
    assert np.allclose(g(*Q.T), want, atol=1e-14)
    assert g.gradient is None                                    # not offered on curved kinds


# --------------------------------------------------------------------------------------- gradient at the centre
@pytest.mark.parametrize("name, params", [("wendland", {"k": 1}), ("wendland", {"k": 3}), ("wu", {"ell": 2, "k": 1}),
                                          ("buhmann", {})])
def test_gradient_vanishes_at_the_centre_and_is_finite(name, params):
    f = getattr(F.on(BALL), name)(radius=0.3, **params)
    c = np.asarray(f.support.centre)
    g = f.gradient(*c)
    assert np.all(np.isfinite(g)) and np.allclose(g, 0.0, atol=1e-15)
    P = c + np.array([[1e-9, 0, 0], [0, 0.29999, 0], [0.5, 0, 0]])
    G = f.gradient(*P.T)
    assert np.all(np.isfinite(G)) and np.allclose(G[2], 0.0)


@pytest.mark.parametrize("name, params", [("wendland", {"k": 0}), ("wu", {"ell": 2, "k": 2})])
def test_cone_profiles_have_no_gradient(name, params):
    f = getattr(F.on(BALL), name)(radius=0.3, **params)
    assert f.gradient is None and f.regularity.order == 0 and f.regularity.piecewise
    assert f.regularity.sobolev_bound() == (1.5, True)           # Lipschitz: in H^1, not H^(3/2)


# ----------------------------------------------------------------------------------------- positive definiteness
_T, _W = np.polynomial.legendre.leggauss(1500)
_R, _WR = 0.5 * (_T + 1), 0.5 * _W


def radial_fourier(profile, m, omega):
    """Fourier transform of x -> profile(|x|) on R^m at the radial frequencies omega (Hankel transform, [0, 1])."""
    nu = m / 2 - 1
    vals = profile(_R)
    out = []
    for w in omega:
        out.append(np.sum(_WR * vals * _R ** (m - 1)) if w == 0 else
                   np.sum(_WR * vals * _R ** (m / 2) * jv(nu, w * _R)) / w ** nu)
    return np.array(out)


OMEGA = np.linspace(0.0, 150.0, 451)


def pd_min(m_poly, dim):
    m, p = m_poly
    return radial_fourier(lambda r: (1 - r) ** m * horner(p, r), dim, OMEGA).min()


@pytest.mark.parametrize("d, k", [(1, 0), (1, 1), (2, 1), (3, 0), (3, 2), (4, 1), (5, 0)])
def test_wendland_transform_nonnegative_up_to_dimension_d_and_not_beyond(d, k):
    m, p, _ = wendland_profile(d, k)
    for dim in range(1, d + 1):
        assert pd_min((m, p), dim) > -1e-12, (d, k, dim)
    if (d, k) in ((1, 0), (1, 1), (3, 0), (3, 1)):
        assert pd_min((m, p), d + 2) < -1e-9                       # the dimension matters: not PD on R^(d+2)


@pytest.mark.parametrize("ell, k", [(1, 0), (2, 0), (2, 1), (3, 1), (2, 2)])
def test_wu_transform_nonnegative_up_to_dimension_2k_plus_1_and_not_beyond(ell, k):
    m, p, _ = wu_profile(ell, k)
    for dim in range(1, 2 * k + 2):
        assert pd_min((m, p), dim) > -1e-12, (ell, k, dim)
    assert pd_min((m, p), 2 * k + 3) < -1e-9


def test_buhmann_transform_nonnegative_on_r1_r2_r3_and_negative_on_r5():
    def phi(r):
        return np.where(r > 0, 12 * r ** 4 * np.log(np.where(r > 0, r, 1)) - 21 * r ** 4 + 32 * r ** 3
                        - 12 * r ** 2 + 1, 1.0)
    for dim in (1, 2, 3):
        assert radial_fourier(phi, dim, OMEGA).min() > -1e-12
    assert radial_fourier(phi, 5, OMEGA).min() < -1e-12


# ------------------------------------------------------------------------------------------------ claims
def claim(f, key):
    return next((c for c in f.claims if c.key == key), None)


@pytest.mark.parametrize("name, params, status", [("wendland", {"k": 2}, "theorem"), ("wu", {"ell": 2, "k": 1}, "theorem"),
                                                  ("buhmann", {}, "checked")])
def test_positive_definiteness_claim_statuses_and_references(name, params, status):
    f = getattr(F.on(BALL), name)(radius=0.3, **params)
    c = claim(f, "positive_definite")
    assert c is not None and c.status == status and c.reason
    if status == "theorem":
        assert "Wendland 2005" in c.reason or "Wu 1995" in c.reason
    assert claim(f, "regularity").status == "theorem" and "199" in claim(f, "regularity").reason
    assert claim(f.scaled(3.0), "positive_definite") is not None
    assert claim(f.normalised(), "positive_definite") is not None
    assert claim(f.scaled(-1.0), "positive_definite") is None
    assert "max" not in f.scaled(-1.0).closed and f.scaled(3.0).closed["max"][0] == 3.0


def test_the_claimed_dimension_follows_d_and_k():
    assert "m <= 3" in claim(F.on(BALL).wendland(k=1), "positive_definite").statement
    assert "m <= 1" in claim(F.on(BALL).wendland(k=1, d=1), "positive_definite").statement
    assert "m <= 5" in claim(F.on(BALL).wu(ell=3, k=2), "positive_definite").statement
    assert claim(F.on(INTERVAL).wendland(), "positive_definite").statement.endswith("m <= 1")


def test_no_positive_definiteness_claim_on_curved_kinds():
    for k in ("sphere", "circle"):
        D = geo.standard(k)[0]
        for name in ("wendland", "wu", "buhmann"):
            f = getattr(F.on(D), name)(radius=0.5)
            assert claim(f, "positive_definite") is None
            assert claim(f, "regularity").status == "theorem" and "normal coordinates" in claim(f, "regularity").reason


def test_regularity_orders_and_membership():
    B = BALL
    for k in range(4):
        f = F.on(B).wendland(k=k, radius=0.4)
        assert f.regularity.order == 2 * k and f.regularity.vanishes_at_boundary == "all orders"
    assert F.on(B).wu(ell=4, k=1, radius=0.4).regularity.order == 6
    assert F.on(B).buhmann(radius=0.4).regularity.order == 2
    f = F.on(B).wendland(k=1, radius=0.4)                       # C^2: H^1 and H^2 yes, H^3 no
    assert [check(f, pg.Sobolev(B, s, length_scale=0.2))[0] for s in (1, 2, 3)] == [True, True, False]
    assert f.element(pg.L2(B)) is not None
    with pytest.raises(pg.Refused, match="H"):
        f.element(pg.Sobolev(B, 3, length_scale=0.2))


# ------------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("name, params, match", [
    ("wendland", {"k": 4}, "k must be an integer in"),
    ("wendland", {"k": -1}, "k must be an integer in"),
    ("wendland", {"k": 1.5}, "k must be an integer in"),
    ("wendland", {"k": True}, "k must be an integer in"),
    ("wendland", {"d": 0}, "d must be an integer in"),
    ("wendland", {"d": 31}, "d must be an integer in"),
    ("wendland", {"d": "3"}, "d must be an integer in"),
    ("wendland", {"radius": -0.1}, "positive and finite"),
    ("wendland", {"radius": float("nan")}, "positive and finite"),
    ("wendland", {"radius": float("inf")}, "positive and finite"),
    ("wendland", {"radius": "big"}, "positive number"),
    ("wendland", {"radius": 1.0}, "leaves the domain"),
    ("wendland", {"centre": (0.9, 0, 0), "radius": 0.3}, "leaves the domain"),
    ("wendland", {"centre": (0, 0)}, "3 finite Cartesian coordinates"),
    ("wendland", {"centre": (0, 0, float("nan"))}, "3 finite Cartesian coordinates"),
    ("wu", {"ell": 0}, "ell must be an integer in"),
    ("wu", {"ell": 7}, "ell must be an integer in"),
    ("wu", {"ell": 2, "k": 3}, "k must be an integer in"),
    ("wu", {"k": -1}, "k must be an integer in"),
    ("wu", {"radius": 0.0}, "positive and finite"),
    ("buhmann", {"radius": 2.0}, "leaves the domain"),
    ("buhmann", {"radius": -1.0}, "positive and finite"),
])
def test_refusals_carry_the_reason(name, params, match):
    with pytest.raises(pg.Refused, match=match):
        getattr(F.on(BALL), name)(**params)


def test_unknown_parameters_and_wrong_kinds_are_refused():
    with pytest.raises(TypeError):
        F.on(BALL).wendland(order=2)
    with pytest.raises(pg.Refused, match="wendland is not defined on a"):
        F.ball.wendland(geo.standard("curve")[0])
    with pytest.raises(pg.Refused, match="leaves the domain"):
        F.on(geo.standard("sphere")[0]).wendland(radius=4.0)            # beyond the antipode: pi R = 3.14


def test_defaults_are_recorded_in_the_parameters():
    f = F.on(DISK).wendland()
    assert f.params["k"] == 1 and f.params["d"] == 2 and f.params["radius"] == pytest.approx(0.5)
    assert F.on(DISK).wendland(d=5).params["d"] == 5
    assert F.on(F.geometry.standard("shell")[0]).wu().params["ell"] == 2


def test_registered_with_references_and_all_kinds():
    ents = F.entries()
    for name in ("wendland", "wu", "buhmann"):
        e = ents[name]
        assert set(e.kinds) == {"interval", "box", "disk", "ball", "shell", "cylinder", "sphere", "circle", "torus"}
        assert e.references and 2 <= len(e.example_params(BALL)) <= 4
    assert any("Wendland 1995" in r for r in ents["wendland"].references)
    assert any("Wu 1995" in r for r in ents["wu"].references)
    assert any("Buhmann 1998" in r for r in ents["buhmann"].references)


# ------------------------------------------------------------------- centres on closed manifolds (review fixes)
SPHERE, CIRCLE, TORUS = (geo.standard(k)[0] for k in ("sphere", "circle", "torus"))


@pytest.mark.parametrize("name, params", [("wendland", {}), ("wu", {}), ("buhmann", {})])
@pytest.mark.parametrize("D, centre", [
    (SPHERE, (0.0, 0.0, 2.0)), (SPHERE, (0.0, 0.0, 0.5)), (SPHERE, (0.0, 0.0, 0.0)), (SPHERE, (0.0, 0.0, 1.001)),
    (CIRCLE, (3.0, 0.0)), (CIRCLE, (0.0, 0.0)), (CIRCLE, (1.0, 0.01))])
def test_a_centre_off_the_manifold_is_refused(name, params, centre, D):
    with pytest.raises(pg.Refused, match="must lie on it"):
        getattr(F.on(D), name)(centre=centre, radius=0.5, **params)


@pytest.mark.parametrize("name, params", [("wendland", {"k": 2}), ("wu", {}), ("buhmann", {})])
def test_a_centre_on_the_manifold_is_accepted_and_the_claims_hold(name, params):
    """The closed integral and l2 norm of an accepted off-axis centre agree with a high-degree sphere quadrature."""
    c = np.array([0.6, 0.0, 0.8])                                  # unit length up to rounding
    f = getattr(F.on(SPHERE), name)(centre=tuple(c * (1 + 1e-12)), radius=0.5, **params)
    assert np.linalg.norm(f.support.centre) == pytest.approx(1.0, abs=1e-15)
    X, w = geo.quadrature(SPHERE, 64)
    vals = f(*X.T)
    # the quadrature meets the compact support's kink only to ~1e-3; the defects this guards against were factors 10-30
    assert w @ vals == pytest.approx(f.integral()[0], rel=2e-3)
    assert w @ vals ** 2 == pytest.approx(f.closed["l2_norm_sq"][0], rel=2e-3)
    assert vals.max() <= 1.0 + 1e-14 and f(*c) == pytest.approx(1.0, abs=1e-12)


def test_the_circle_closed_integral_matches_quadrature_for_an_off_origin_centre():
    f = F.on(CIRCLE).buhmann(centre=(0.0, 1.0), radius=0.5)
    X, w = geo.quadrature(CIRCLE, 64)
    assert w @ f(*X.T) == pytest.approx(f.integral()[0], rel=2e-3)


# ---------------------------------------------------------------------------------------------- bool radius
@pytest.mark.parametrize("name", ["wendland", "wu", "buhmann"])
@pytest.mark.parametrize("radius", [True, False, np.True_])
def test_a_bool_radius_is_refused_as_such(name, radius):
    with pytest.raises(pg.Refused, match="not a bool"):
        getattr(F.on(BALL), name)(radius=radius)


# ---------------------------------------------------------------------------------------------------- torus
@pytest.mark.parametrize("name, params", [("wendland", {"k": 1}), ("wu", {"ell": 2, "k": 1}), ("buhmann", {})])
def test_torus_support_wraps_with_the_minimum_image_distance(name, params):
    f = getattr(F.on(TORUS), name)(centre=(0.3, 0.2), radius=1.0, **params)
    assert f(0.3, 0.2) == pytest.approx(1.0, abs=1e-14)
    L = 2 * math.pi
    assert f(0.3 - 0.5 + L, 0.2) == pytest.approx(f(0.3 - 0.5, 0.2), abs=1e-14) and f(0.3 - 0.5, 0.2) > 0
    assert f(0.3 - 1.0 + L, 0.2) == 0.0 and f(0.3, 0.2 + 1.2) == 0.0       # beyond the support, across the period
    X, w = geo.quadrature(TORUS, 64)
    assert w @ f(*X.T) == pytest.approx(f.integral()[0], rel=2e-3)
    assert claim(f, "positive_definite") is None
    assert "minimum-image" in claim(f, "regularity").reason


def test_torus_refuses_a_support_that_wraps_and_a_centre_outside_the_box():
    with pytest.raises(pg.Refused, match="leaves the domain"):
        F.on(TORUS).wendland(radius=math.pi)                      # half the shortest period is pi
    with pytest.raises(pg.Refused, match="fundamental box"):
        F.on(TORUS).wu(centre=(7.0, 1.0), radius=0.5)


def test_askey_is_credited_with_sufficiency_only():
    reason = claim(F.on(BALL).wendland(), "positive_definite").reason
    assert "sufficiency" in reason and " iff " not in reason
