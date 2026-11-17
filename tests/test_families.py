"""Plan 009 P2, G4: property families indexed by centres and radii (cap averages, rotated zonal functionals), the
closed forms over a family without p x p matrices, and BGP vectorised over directions (EXP-030 DESIGN G4). References:
direct quadrature of the weights around each centre, explicit rotations, the plain (unfactored) route of P1 and its
per-direction BGP."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.properties.families import directions
from pygeoinf0.spaces.harmonic import profile_function


def cap_reference(h, xi, profile, psi0, n=80, m=160):
    """int w(xi . eta) h(eta) d eta / int w by Gauss in the angle from xi (support split at psi0) x equispaced azimuth."""
    w0, end = profile_function(profile, psi0)
    t, wt = np.polynomial.legendre.leggauss(n)
    psi, wp = 0.5 * end * (t + 1), 0.5 * end * wt
    al = 2 * np.pi * np.arange(m) / m
    e1 = np.cross(xi, [0.3, 0.5, 0.8])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(xi, e1)
    P, A = np.meshgrid(psi, al, indexing="ij")
    pts = np.cos(P)[..., None] * xi + np.sin(P)[..., None] * (np.cos(A)[..., None] * e1 + np.sin(A)[..., None] * e2)
    W = np.broadcast_to((wp * np.sin(psi) * w0(psi))[:, None], P.shape)
    return float((W * h(pts.reshape(-1, 3)).reshape(P.shape)).sum() / W.sum())


CEN = np.array([[10.0, 20.0], [-45.0, 100.0], [80.0, -30.0], [-4.5, -157.5]])


@pytest.mark.parametrize("band", [range(0, 21), None])
def test_cap_averages_against_direct_quadrature(band):
    S = pg.L2(pg.Sphere(1.0))
    h = pg.BandProjection(S, range(0, 21))(S(lambda x, y, z: np.exp(x) * np.cos(y) + z ** 3))
    T = pg.CapAverages(S, 0.3, CEN, band=band, degree=None if band else 40)
    ref = [cap_reference(h, xi, "raised-cosine", 0.3) for xi in directions(CEN)]
    assert np.allclose(T(h).values, ref, rtol=0, atol=1e-12 * h.norm())
    assert any(c.status == "theorem" and "Funk-Hecke" in c.statement for c in T.claims)


def test_cap_averages_of_a_summand_with_several_radii_and_rotation_covariance():
    R = 2.0
    M = pg.DirectSum({"v": pg.L2(pg.Ball(1.0), realisation="radial_harmonics"), "h": pg.L2(pg.Sphere(R))},
                     weights={"v": 1.0, "h": 0.3})
    hf = lambda x, y, z: np.sin(x) + y * z / 4.0                           # noqa: E731
    x = M({"v": lambda x, y, z: x, "h": hf})
    T = pg.CapAverages(M, [0.2, 0.5], CEN, profile="gaussian", band=range(0, 25), summand="h")
    hb = pg.BandProjection(M.summand("h"), range(0, 25))(M.summand("h")(hf))
    unit = lambda p: hb(R * p)                                             # noqa: E731  the field on the unit sphere
    for i, psi0 in enumerate((0.2, 0.5)):
        ref = [cap_reference(unit, xi, "gaussian", psi0, n=120) for xi in directions(CEN)]
        assert np.allclose(T(x).values[i * len(CEN):(i + 1) * len(CEN)], ref, rtol=0, atol=1e-11)
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    xr = M({"v": lambda x, y, z: x, "h": lambda x, y, z: hf(*(sum(Q[k, i] * c for k, c in enumerate((x, y, z))) for i in range(3)))})
    T2 = T.at(directions(CEN) @ Q.T)                                       # rotated centres
    assert np.allclose(T2(xr).values, T(x).values, rtol=0, atol=1e-12)


def test_rotated_family_of_a_zonal_functional_on_a_ball():
    M = pg.L2(pg.Ball(1.0), realisation="radial_harmonics")
    t0f = lambda x, y, z: (1 + x * x + y * y + z * z) * np.exp(z)           # noqa: E731  zonal about the pole
    T = pg.RotatedFamily(pg.Functional.from_representer(M(t0f)), CEN)
    m = M(lambda x, y, z: np.cos(x + y) * (1 + z))
    vals = []
    for xi in directions(CEN):
        e3 = xi
        e1 = np.cross(e3, [0.3, 0.5, 0.8])
        e1 /= np.linalg.norm(e1)
        Rm = np.stack([e1, np.cross(e3, e1), e3], axis=1)                  # maps the pole to xi
        rep = M(lambda x, y, z, Rm=Rm: t0f(*(sum(Rm[k, i] * c for k, c in enumerate((x, y, z))) for i in range(3))))
        vals.append(M.inner(rep, m))
    assert np.allclose(T(m).values, vals, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        pg.RotatedFamily(pg.Functional.from_representer(M(lambda x, y, z: np.exp(x))), CEN)    # not zonal


def plain(T):
    """The same map without its factors (P1's generic route)."""
    return pg.LinearOperator(T.domain, T.codomain, apply=T._apply, adjoint=T.adjoint._fn, name="T_plain")


def problem(p=60, seed=2):
    S = pg.L2(pg.Sphere(1.0))
    rng = np.random.default_rng(seed)
    lab = [(l, j) for l in range(1, 11) for j in range(2 * l + 1) for _ in range(2)]
    G = pg.Functionals.separable(S, [(1.0 + 0.1 * l) * (1 + a % 2) for a, (l, j) in enumerate(lab)], lab)
    n = len(lab)
    D = G.codomain
    noise = pg.GaussianMeasure(D, pg.Diagonal(rng.uniform(0.5, 1.5, n), D))
    T = pg.CapAverages(S, 0.4, np.c_[rng.uniform(-80, 80, p), rng.uniform(0, 360, p)], band=range(1, 11))
    return S, G, noise, T, D(rng.normal(size=n)), n


def test_sola_and_pli_over_a_family_equal_the_plain_route():
    S, G, noise, T, d, n = problem()
    a, b = pg.decompose(G, noise), pg.decompose(G, noise)
    g = np.array([0.1, 1.0, 10.0])
    r1, r2 = a.sola(T, g, d), b.sola(plain(T), g, d)
    for k in ("estimate", "variance", "mismatch", "delivered", "d_variance"):
        assert np.allclose(getattr(r1, k), getattr(r2, k), rtol=1e-11, atol=1e-13)
    p1, p2 = a.posterior(T, d, a=2.0), b.posterior(plain(T), d, a=2.0)
    assert np.allclose(p1.mean, p2.mean, rtol=1e-11, atol=1e-13)
    assert np.allclose(p1.variance, np.diag(p2.covariance), rtol=1e-11, atol=1e-13)
    assert np.allclose(p1.covariance.block(np.arange(10), np.arange(5, 20)), p2.covariance[:10, 5:20], atol=1e-13)
    X = a.sola_operator(T, 1.0)
    assert np.allclose(X(d).values, r1.estimate[1], atol=1e-12)


def test_vectorised_bgp_against_the_per_direction_route():
    S, G, noise, T, d, n = problem()
    a = pg.decompose(G, noise)
    r = 1.5 * math.sqrt(n)
    iv = a.ellipsoid_intervals(T, a=5.0, r=r, datum=d)
    b = pg.decompose(G, noise)
    E = np.eye(T.codomain.dim)
    for k in range(0, T.codomain.dim, 7):
        hi = b.ellipsoid_bound(plain(T), a=5.0, r=r, datum=d, q=E[k])
        lo = b.ellipsoid_bound(plain(T), a=5.0, r=r, datum=d, q=-E[k])
        assert iv.upper[k] == pytest.approx(hi.value, rel=1e-9) and iv.lower[k] == pytest.approx(-lo.value, rel=1e-9)
    assert set(iv.status_upper) | set(iv.status_lower) == {"EXACT"}


def test_vectorised_bgp_needs_no_fallback_when_newton_converges_from_the_infeasible_side():
    rng = np.random.default_rng(0)
    S = pg.L2(pg.Sphere(1.0))
    lab = [(l, j) for l in range(2, 15, 2) for j in range(2 * l + 1) for _ in range(67)]      # ~8000 data, 119 blocks
    n = len(lab)
    G = pg.Functionals.separable(S, list((1.0 + 0.1 * l) * rng.uniform(0.5, 1.5) for l, j in lab), lab)
    t = rng.uniform(0, 1, n)
    noise = pg.GaussianMeasure(G.codomain, pg.Diagonal((0.2 + t) ** 2, G.codomain))
    truth = S(lambda x, y, z: np.sin(2 * x) * np.cos(y) + z ** 2)
    d = G.codomain(G(truth).values + (0.2 + t) * rng.normal(size=n))
    T = pg.CapAverages(S, math.radians(5.0), np.c_[np.full(48, -89.5), 4.0 * np.arange(48)], band=range(2, 15))
    a = pg.decompose(G, noise)
    iv = a.ellipsoid_intervals(T, a=5.0 * truth.norm(), r=1.5 * math.sqrt(n), datum=d)
    assert set(iv.status_upper) | set(iv.status_lower) == {"EXACT"}
    assert any(f"0 of {2 * T.codomain.dim} directions solved again" in c.statement for c in iv.claims)
    E = np.eye(T.codomain.dim)
    for k in (0, 37):
        hi = a.ellipsoid_bound(plain(T), a=5.0 * truth.norm(), r=1.5 * math.sqrt(n), datum=d, q=E[k])
        assert iv.upper[k] == pytest.approx(hi.value, rel=1e-9)


def test_methods_over_a_map_sized_family_stay_lazy():
    from pygeoinf0.solvers.dataspace.factored import COVARIANCE_DENSE
    S, G, noise, T, d, n = problem(p=COVARIANCE_DENSE + 500)
    prior = pg.GaussianMeasure(S, pg.BandProjection(S, range(0, 11)) * 4.0)
    mdr = pg.Additive(G, noise)
    P = pg.Problem(M=S, D=G.codomain, mdr=mdr, P=T.codomain, mpr=T, datum=d, prior=prior)
    post = P.run("pli")
    assert not isinstance(post.covariance, pg.Matrix) and post.variances.shape == (T.codomain.dim,)
    sub = T.at(T.centres[:30])
    ref = P.with_(mpr=plain(sub), P=sub.codomain).run("pli")
    assert np.allclose(post.mean.values[:30], ref.mean.values, rtol=1e-10, atol=1e-12)
    assert np.allclose(post.variances[:30], np.diag(ref.covariance.matrix()), rtol=1e-10, atol=1e-13)
    sola = P.run("sola", gamma=0.5)
    assert sola.estimate.shape == (T.codomain.dim,)
