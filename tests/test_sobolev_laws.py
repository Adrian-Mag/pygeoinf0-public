"""Laws and edge cases of Sobolev spaces (from the adversarial code review): measure parsing, additivity and
continuity of energies, inclusion, intrinsic geometry of paths, periodic translation invariance, contracts."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import SpaceMismatch, SupportOutsideDomain, Unsupported


def E(H, a, b):
    return H.inner(H.riesz(a), H.riesz(b))


def test_measure_parsing():
    H = pg.Sobolev("[0, 1]", 1, length_scale=0.1)
    assert len(H.riesz([0.3, 0.7])) == 2                                   # points are decided by shape: on an
    assert len(H.riesz((0.3, 0.7))) == 2                                   # interval a sequence of numbers is
    assert len(H.riesz(np.linspace(0.1, 0.9, 5))) == 5                     # that many points
    assert isinstance(H.riesz(0.3), pg.Function)
    D2 = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.1)
    assert isinstance(D2.riesz((0.1, 0.2)), pg.Function) and len(D2.riesz(np.zeros((4, 2)) + 0.1)) == 4
    with pytest.raises(SpaceMismatch):
        pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.1).riesz((0.1, 0.1, 0.1))
    with pytest.raises(SupportOutsideDomain):
        pg.L2("[0, 1]").riesz(pg.Interval(0.5, 2))


@pytest.mark.parametrize("dom, s", [(pg.Disk(1.0), 0.75), (pg.Disk(1.0), 1.0), (pg.Disk(1.0), 1.5),
                                    (pg.Ball(1.0), 1.25), (pg.Ball(1.0), 2.0)])
def test_energies_are_additive_under_splitting(dom, s):
    H = pg.Sobolev(dom, s, length_scale=0.1)
    d = dom.dim
    def pt(*c):
        return np.array(c + (0.0,) * (d - len(c)), float)
    S1, S2, S = (pg.Segment(pt(-0.5, 0.0), pt(0.0, 0.0)), pg.Segment(pt(0.0, 0.0), pt(0.4, 0.0)),
                 pg.Segment(pt(-0.5, 0.0), pt(0.4, 0.0)))
    others = [pg.Segment(pt(0.1, -0.5), pt(0.1, 0.5)),                      # crossing
              pg.Segment(pt(-0.2, 0.0), pt(0.6, 0.0)),                      # collinear overlap
              pg.Segment(pt(0.4, 0.0), pt(0.5, 0.3)),                       # touching an end point
              pg.Segment(pt(0.0, 0.0), pt(1e-5, 0.0)),                      # very short, adjacent
              pg.Segment(pt(-0.3, 0.2), pt(0.2, -0.1))]
    if d == 3:
        others.append(pg.Segment(np.array([0.1, -0.5, 1e-7]), np.array([0.1, 0.5, 1e-7])))   # skew, h = 1e-7
    for T in others:
        whole, parts = E(H, S, T), E(H, S1, T) + E(H, S2, T)
        scale = math.sqrt(E(H, S, S) * E(H, T, T))
        assert abs(whole - parts) <= 2e-12 * scale, (T, whole, parts)     # rounding-level


def test_energy_is_continuous_in_the_angle():
    H = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.1)
    S = pg.Segment((-0.5, 0.0), (0.5, 0.0))
    ref = E(H, S, S)
    for k in range(2, 13):
        th = 10.0 ** -k
        T = pg.Segment((-0.5 * math.cos(th), -0.5 * math.sin(th)), (0.5 * math.cos(th), 0.5 * math.sin(th)))
        assert abs(E(H, S, T) - ref) <= 10 * th * ref + 1e-12 * ref


def test_inclusion_and_integrals():
    S = pg.Sphere(1.0)
    H = pg.Sobolev(S, 2.0, length_scale=0.2)
    f = H(lambda x, y, z: x * z + 1)
    L = pg.L2(S)
    g = pg.Inclusion(H, L)(f)
    assert g(0.6, 0.0, 0.8) == pytest.approx(1.48, rel=1e-13) and g.space == L
    with pytest.raises(SpaceMismatch):
        pg.Inclusion(L, L)
    I = pg.Sobolev("[0, 1]", 1, length_scale=0.1)
    assert I.riesz(0.3).integral() == pytest.approx(0.5 * (2 - math.exp(-3) - math.exp(-7)), rel=1e-12)


def test_paths_are_intrinsic_and_tori_periodic():
    P = pg.Polyline([(0, 0), (1, 0), (1, 1)])
    l = 0.3
    H = pg.Sobolev(P, 1, length_scale=l)                                 # arc length: kernel exp(-|s - t|/l)/(2l)
    assert E(H, (0.5, 0.0), (1.0, 0.5)) == pytest.approx(math.exp(-1 / l) / (2 * l), rel=1e-13)
    with pytest.raises(SupportOutsideDomain):
        H.riesz(pg.Segment((0, 0), (1, 1)))                               # a chord is not on the path
    T = pg.Torus(f"[0, {2 * math.pi!r}]", f"[0, {2 * math.pi!r}]")
    HT = pg.Sobolev(T, 1.5, length_scale=0.4)
    a = pg.Segment((6.0, 1.0), (6.5, 1.5))                               # across the seam
    b = pg.Segment((6.0 - 2 * math.pi, 1.0), (6.5 - 2 * math.pi, 1.5))   # the same set on the torus
    c = pg.Segment((0.3, 0.8), (0.1, 2.0))
    assert E(HT, a, c) == pytest.approx(E(HT, b, c), rel=1e-12)


def test_representer_families_obey_the_backend_laws():
    for H, items in [(pg.Sobolev("[0, 1]", 1, length_scale=0.1), [0.2, pg.Interval(0.3, 0.6), 0.9]),
                     (pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.2), [(0.1, 0.2), pg.Segment((0, -1), (0, 1))]),
                     (pg.Sobolev(pg.Circle(1.0), 1, length_scale=0.3), [(1.0, 0.0), (0.0, 1.0)]),
                     (pg.Sobolev(pg.Sphere(1.0), 2, length_scale=0.3), [(0, 0, 1.0), pg.Arc(pg.Sphere(1.0), (1, 0, 0), (0, 1, 0))])]:
        K = H.riesz(items)
        B, R = H.backend, K.realisation
        G = B.gram(R, R)
        assert np.allclose(G, G.T, rtol=1e-12, atol=1e-14 * np.abs(G).max())
        assert np.allclose(B.sqnorms(R), np.diag(G), rtol=1e-12)
        A = np.array([[1.0], [-2.0]] + [[0.5]] * (len(items) - 2))
        assert np.allclose(B.gram(B.combine(R, A), R), A.T @ G, rtol=1e-12, atol=1e-13 * np.abs(G).max())
        assert np.all(np.isfinite(B.error(R)))


def test_space_identity_and_signature():
    D = pg.Disk(1.0)
    assert pg.L2(D) == pg.L2(D, realisation="tensor")
    assert pg.Sobolev(D, 0) == pg.L2(D)
    with pytest.raises(Unsupported):
        pg.Sobolev(D, -1, length_scale=0.1)
    with pytest.raises(TypeError):
        pg.Sobolev(D, 1.5)
    assert len(pg.Sobolev(pg.Sphere(1.0), 2, length_scale=0.2).claims) >= 5


def test_torus_atoms_far_away_and_spectral_formula_norms():
    L = 2 * math.pi
    T = pg.Torus(f"[0, {L!r}]", f"[0, {L!r}]")
    H = pg.Sobolev(T, 2.0, length_scale=0.05)
    a = pg.Segment((1.0, 1.0), (1.5, 1.2))
    far = pg.Segment((1.0 + 3 * L, 1.0 - 2 * L), (1.5 + 3 * L, 1.2 - 2 * L))     # the same set, three periods out
    c = pg.Segment((1.2, 0.8), (1.3, 1.6))
    assert E(H, far, c) == pytest.approx(E(H, a, c), rel=1e-13)
    f = H(lambda x, y: np.cos(x) + np.sin(2 * y))                           # spectral: exact on tori
    l, s = 0.05, 2.0
    ref = ((1 + l ** 2) ** s + (1 + 4 * l ** 2) ** s) * 2 * math.pi ** 2
    assert f.norm() ** 2 == pytest.approx(ref, rel=1e-13)


def test_cylindrical_gaussian_claims_and_pushforward_samples():
    H = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.2)
    g = pg.probability.cylindrical(H)
    assert any("not trace class" in c.statement for c in g.claims)
    assert not any(c.status == "asserted" and "trace class" in c.statement for c in g.claims)
    G = pg.Functionals(H.riesz([pg.Segment((-1, 0), (1, 0)), pg.Segment((0, -1), (0, 1))]))
    X = G(g).sample(size=20000, seed=0).values
    assert np.allclose(np.cov(X.T), G.gram(), rtol=0.05, atol=0.02 * G.gram().max())
