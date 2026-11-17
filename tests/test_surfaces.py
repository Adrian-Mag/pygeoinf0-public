"""Area measures on flat patches as functionals on H^s of 3-D domains (bounded iff s > 1/2): exact potentials, energies
by graded quadrature with error estimates (checked against adaptive SciPy quadrature in turf/surfaces/)."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import UnboundedFunctional


def _setup(s=1.5, l=0.3):
    D = pg.Box("[0, 1]", "[0, 1]", "[0, 1]")
    return D, pg.Sobolev(D, s, length_scale=l)


def test_patch_representer_reproduces_the_surface_integral():
    D, H = _setup(s=2.0)                                                  # s > 3/2: point values are bounded too
    P = pg.Patch((0.2, 0.1, 0.5), (0.5, 0.1, 0.0), (0.0, 0.4, 0.2))
    k = H.riesz(P)
    f = H(lambda x, y, z: 1 + x * y - z ** 2)
    a, w = np.polynomial.legendre.leggauss(20)
    a, w = (a + 1) / 2, w / 2
    o, (e1, e2) = np.array(P.origin), np.array(P.edges)
    X = o + a[:, None, None] * e1 + a[None, :, None] * e2
    ref = np.einsum("ab,a,b->", 1 + X[..., 0] * X[..., 1] - X[..., 2] ** 2, w, w) * np.linalg.norm(np.cross(e1, e2))
    assert k @ f == pytest.approx(ref, rel=1e-13)
    assert k(0.4, 0.3, 0.55) == pytest.approx(H.riesz((0.4, 0.3, 0.55)) @ k, rel=1e-12)   # value = energy with a Dirac
    assert k.error < 1e-12 * k.norm()


def test_energy_is_additive_over_a_split_patch():
    D, H = _setup()
    o, e1, e2 = np.array([0.2, 0.2, 0.3]), np.array([0.4, 0.0, 0.1]), np.array([0.0, 0.3, 0.0])
    whole = H.riesz(pg.Patch(o, e1, e2))
    halves = H.riesz([pg.Patch(o, e1, 0.4 * e2), pg.Patch(o + 0.4 * e2, e1, 0.6 * e2)])
    assert (halves[0] + halves[1]).norm() == pytest.approx(whole.norm(), rel=1e-12)
    ray = pg.Segment((0.1, 0.35, 0.05), (0.9, 0.3, 0.9))                   # a ray crossing the patch
    assert H.inner(H.riesz(ray), whole) == pytest.approx(H.inner(H.riesz(ray), halves).sum(), rel=1e-12)


def test_trace_threshold_and_data_with_rays_and_an_interface():
    D, _ = _setup()
    P = pg.Patch((0.1, 0.1, 0.5), (0.8, 0.0, 0.0), (0.0, 0.8, 0.0))
    with pytest.raises(UnboundedFunctional, match="0.5"):
        pg.Sobolev(D, 0.5, length_scale=0.3).riesz(P)
    H = pg.Sobolev(D, 1.2, length_scale=0.3)                          # s in (1/2, 3/2): singular kernel, patch bounded
    rng = np.random.default_rng(1)
    rays = [pg.clip(pg.Segment(p, q), D) for p, q in zip(rng.uniform(-0.2, 1.2, (6, 3)), rng.uniform(-0.2, 1.2, (6, 3)))]
    rays = [r for r in rays if r is not None]
    K = H.riesz(rays + [P])
    Gm = H.inner(K, K)
    assert np.allclose(Gm, Gm.T, rtol=1e-12) and np.linalg.eigvalsh(Gm).min() > 0
    post = pg.probability.cylindrical(H, alpha=1.0).condition(pg.Functionals(K, into=pg.Euclidean(len(K), covariance=1e-4)),
                                           pg.Euclidean(len(K), covariance=1e-4)(np.zeros(len(K))))
    assert post.std(P) < pg.probability.cylindrical(H).std(P)


def test_patch_needs_a_3d_domain():
    H = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.2)
    with pytest.raises(pg.SpaceMismatch):
        H.riesz(pg.Patch((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0)))


def test_sphere_inside_a_ball():
    B = pg.Ball(1.0)
    H = pg.Sobolev(B, 2.0, length_scale=0.3)
    c, R = np.array([0.1, 0.0, -0.1]), 0.5
    S = pg.Sphere(R, tuple(c))
    k = H.riesz(S)
    f = H(lambda x, y, z: 1 + x ** 2)
    assert k @ f == pytest.approx(4 * math.pi * R ** 2 * (1 + c[0] ** 2) + 4 * math.pi * R ** 4 / 3, rel=1e-13)
    x0 = (0.3, 0.2, 0.1)
    assert k(*x0) == pytest.approx(H.riesz(x0) @ k, rel=1e-12)
    Kern = H.backend.kernel
    assert k.norm() ** 2 == pytest.approx(8 * math.pi ** 2 * R ** 2 * float(Kern.G(0.0, 2 * R)), rel=1e-13)
    inner = H.riesz(pg.Sphere(0.3, tuple(c)))                               # concentric: exact
    assert inner.error == 0.0 and np.isfinite(H.inner(inner, k))
    rays = [pg.clip(pg.Segment((-1.2, y, 0.05), (1.2, -y, 0.0)), B) for y in (-0.4, 0.0, 0.3)]
    P = pg.Patch((-0.4, -0.4, 0.2), (0.8, 0.0, 0.0), (0.0, 0.8, 0.1))
    K = H.riesz(rays + [S, P])
    Gm = H.inner(K, K)
    assert np.allclose(Gm, Gm.T, rtol=1e-12) and np.linalg.eigvalsh(Gm).min() > 0
    assert np.all(K.errors() < 1e-10 * K.norms())
    with pytest.raises(pg.errors.SupportOutsideDomain):
        H.riesz(pg.Sphere(0.5, (0.8, 0.0, 0.0)))


@pytest.mark.parametrize("s", [0.8, 1.5])
def test_circle_inside_a_disk(s):
    D = pg.Disk(1.0)
    H = pg.Sobolev(D, s, length_scale=0.2)                 # circles need s > 1/2 in 2-D; points only s > 1
    c, R = np.array([0.1, -0.05]), 0.45
    C = pg.Circle(R, tuple(c))
    k = H.riesz(C)
    f = H(lambda x, y: 1 + x ** 2)
    assert k @ f == pytest.approx(2 * math.pi * R * (1 + c[0] ** 2) + math.pi * R ** 3, rel=1e-13)
    if s > 1:
        x0 = (0.3, 0.2)
        assert k(*x0) == pytest.approx(H.riesz(x0) @ k, rel=1e-12)
    rays = [pg.clip(pg.Segment((-1.2, y), (1.2, -y)), D) for y in (-0.4, 0.0, 0.3)]
    K = H.riesz(rays + [C, pg.Circle(0.2, tuple(c))])
    Gm = H.inner(K, K)
    assert np.allclose(Gm, Gm.T, rtol=1e-12) and np.linalg.eigvalsh(Gm).min() > 0
    assert np.all(K.errors() < 1e-8 * K.norms())                         # the warning level (singular kernels)
    with pytest.raises(pg.errors.Unsupported):
        pg.Sobolev(pg.Ball(1.0), 2.0, length_scale=0.3).riesz(pg.Circle(0.5))


def test_sobolev_on_a_flat_patch_is_the_planar_restriction_space():
    P = pg.Patch((0.2, -0.1, 0.3), (0.8, 0.0, 0.0), (0.0, 0.6, 0.0))      # the rectangle [0.2, 1] x [-0.1, 0.5] at z = 0.3
    Bx = pg.Box("[0.2, 1]", "[-0.1, 0.5]")
    HP, HB = pg.Sobolev(P, 1.5, length_scale=0.2), pg.Sobolev(Bx, 1.5, length_scale=0.2)
    segs = [((0.3, 0.0), (0.9, 0.4)), ((0.25, 0.45), (0.95, -0.05))]
    pts = [(0.5, 0.2), (0.8, 0.1)]
    KP = HP.riesz([pg.Segment(p + (0.3,), q + (0.3,)) for p, q in segs] + [x + (0.3,) for x in pts])
    KB = HB.riesz([pg.Segment(p, q) for p, q in segs] + pts)
    assert np.allclose(HP.inner(KP, KP), HB.inner(KB, KB), rtol=1e-13, atol=0)
    assert KP[0](0.6, 0.2, 0.3) == pytest.approx(KB[0](0.6, 0.2), rel=1e-13)


def test_round_supports_are_checked_exactly():
    H = pg.Sobolev(pg.Ball(1.0), 2.0, length_scale=0.3)
    u = np.array([1.0, 0.45, 0.2]) / np.linalg.norm([1.0, 0.45, 0.2])
    with pytest.raises(pg.errors.SupportOutsideDomain):
        H.riesz(pg.Sphere(0.5, tuple(0.52 * u)))                         # reaches radius 1.02: no sample point did
    H.riesz(pg.Sphere(0.5, tuple(0.4999 * u)))
    with pytest.raises(pg.errors.SupportOutsideDomain):
        pg.Sobolev(pg.Box("[0, 1]", "[0, 1]"), 1.5, length_scale=0.2).riesz(pg.Circle(0.3, (0.5, 0.75)))


def test_integrals_over_interfaces_prior_mean_and_mixed_lists():
    D = pg.Disk(1.0)
    H = pg.Sobolev(D, 1.5, length_scale=0.2)
    C = pg.Circle(0.5)
    ray = pg.Segment((-0.9, 0.1), (0.8, -0.2))
    f = H(lambda x, y: 1 + x * y)
    assert f.integral(C) == pytest.approx(2 * math.pi * 0.5, rel=1e-13)          # int_C x y ds = 0
    k = H.riesz([ray, C])
    m = k[0] + 0.3 * k[1]
    assert m.integral(C) == pytest.approx(H.inner(m, k[1]), rel=1e-14)
    with pytest.raises(pg.errors.UnboundedFunctional):
        pg.Sobolev(D, 0.5, length_scale=0.2)(lambda x, y: x).integral(C)
    Dd = pg.Euclidean(1, covariance=1e-4)
    G = pg.Functionals(H.riesz([ray]), into=Dd)
    d = Dd(np.array([0.7]))
    mref = 0.5 * k[1]
    post = pg.probability.cylindrical(H, alpha=2.0, mean=mref).condition(G, d)
    dm = G.adjoint((G @ G.adjoint + 2.0 * pg.Identity(Dd)).inv(d - G(mref)))
    assert post.mean.distance(mref + dm) < 1e-12 * (mref + dm).norm()
    K = H.riesz([np.array([[0.1, 0.2], [0.0, -0.3]]), C, (0.2, 0.2)])
    assert len(K) == 4
    with pytest.raises(ValueError, match="p != q"):
        pg.Segment((0.1, 0.2), (0.1, 0.2))
