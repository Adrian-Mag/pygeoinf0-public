"""Automatic latitude-longitude witness cells for box BGP on a sphere (plan 061 P3): the exact cell integrals against
harmonic_cell_integrals, the cells' data images against the operator, a synthetic problem whose answer is known
(bracketed, and the bracket narrows with K.sub), one slab of noise on a sphere by the reduced cell route, refusals.
Synthetic caps only."""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.optimisers.dual.sphere_cells import SphereGrid, is_sphere_box, sphere_cells
from pygeoinf0.core.errors import Unsupported
from pygeoinf0.realisations.bounds import sphere_l1 as S

R_CORE = 3479.95776
X = pg.DirectSum({"cmb": pg.L2(pg.Sphere(1.0))}, weights={"cmb": 1 / (4 * math.pi * R_CORE ** 2)})
CENTRES = np.array([[90.0, 0.0], [10.0, 40.0], [-30.0, 200.0]])


def caps(centres=CENTRES, space=X, radius_km=2000.0, deg=14):
    kw = dict(summand="cmb") if space is X else {}
    return pg.CapAverages(space, radius_km / R_CORE, np.asarray(centres, float), degree=deg, **kw)


def representer(T, i):
    D = T.codomain
    return T.adjoint(D(D.solve_metric(np.eye(D.dim)[i])))


def test_the_grid_integrals_equal_harmonic_cell_integrals_and_sum_to_the_sphere():
    lat, lon = np.linspace(-math.pi / 2, math.pi / 2, 9), np.linspace(0, 2 * math.pi, 17)
    g = SphereGrid(6, lat, lon)
    Y = S.harmonic_cell_integrals(6, lat, lon)
    B = np.random.default_rng(0).standard_normal((49, 3))
    V = np.random.default_rng(1).standard_normal((g.size, 2))
    assert np.allclose(g.values(B), Y @ B, atol=1e-13) and np.allclose(g.coefficients(V), Y.T @ V, atol=1e-12)
    one = np.zeros(49)
    one[0] = math.sqrt(4 * math.pi)                                    # the constant 1
    assert g.values(one).sum() == pytest.approx(4 * math.pi, rel=1e-13)


def test_cell_images_against_quadrature_and_the_constant_field():
    """H[E, i] = w int_E r_i by Gauss quadrature over a few cells (independent of SphereGrid); the all-ones cell field
    is the constant 1, whose data image is T(1)."""
    from numpy.polynomial import legendre as npleg
    from pygeoinf0.spaces.harmonic import values
    T = caps()
    box = pg.PointwiseBox(X, envelope={"cmb": 1.0})
    c = representer(T, 0)
    cp = sphere_cells(box, T, c, sub=2)
    nl = 2 * 15
    Hrows = cp.images(T)
    reps = [np.asarray(representer(T, i).realisation.parts[0][0].C, float)[:, 0] for i in range(3)]
    z, wz = npleg.leggauss(24)
    for E in (0, 7, 431, 905, cp.size - 1):
        p, q = divmod(E, 2 * nl)
        a, b = -math.pi / 2 + p * math.pi / nl, -math.pi / 2 + (p + 1) * math.pi / nl
        lo0, lo1 = q * math.pi / nl, (q + 1) * math.pi / nl
        ph, la = 0.5 * (a + b) + 0.5 * (b - a) * z, 0.5 * (lo0 + lo1) + 0.5 * (lo1 - lo0) * z
        PH, LA = np.meshgrid(ph, la, indexing="ij")
        Xp = np.stack([np.cos(PH) * np.cos(LA), np.cos(PH) * np.sin(LA), np.sin(PH)], -1).reshape(-1, 3)
        W = (np.outer(wz, wz) * np.cos(PH)).ravel() * 0.25 * (b - a) * (lo1 - lo0)
        for i in range(3):
            ref = X.weights[0] * float(W @ (values(14, Xp) @ reps[i]))
            assert Hrows[E, i] == pytest.approx(ref, rel=1e-10, abs=1e-12 * np.abs(Hrows[:, i]).max())
    one = X({"cmb": pg.L2(pg.Sphere(1.0))(lambda x, y, z: 1.0 + 0 * z, resolution=1.0)})
    assert np.allclose(cp.apply(np.ones(cp.size)), T(one).values, rtol=1e-10)
    assert np.allclose(cp.adjoint(np.eye(3)[0]), cp.t) and np.allclose(Hrows[:, 0], cp.t)   # c: datum 0's representer


def test_box_bgp_with_automatic_sphere_cells_brackets_the_known_answer():
    """sup <c, m> over |m| <= 1 with |<c, m> - d| <= w for c datum 0's cap average: U = min(d + w, sigma_B(c)).
    With the data inactive U = sigma_B(c) (the support, certified), and the witness approaches it as the cells
    shrink."""
    box = pg.PointwiseBox(X, envelope={"cmb": 1.0}, rtol=1e-7)
    T = caps()
    D = T.codomain
    c = representer(T, 0)
    sb = box.support(c)
    slo, shi = sb.lower, sb.upper
    gaps = []
    for sub in (2, 4, 8):
        K = pg.FeasibleSet(box, T, pg.NormBall(D, 1e12), D(np.zeros(3)))
        K.sub = sub
        b = K.bound(c)
        assert b.lower <= shi and b.upper >= slo and b.upper <= shi * (1 + 1e-12)
        gaps.append(b.upper - b.lower)
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 1e-3 * shi
    T1 = caps(CENTRES[:1])
    D1 = T1.codomain
    d, w = 0.5 * slo, 0.1 * slo
    exact = d + w
    for sub in (2, 4):
        K = pg.FeasibleSet(box, T1, pg.NormBall(D1, w), D1([d]))
        K.sub = sub
        b = K.bound(c)
        assert b.lower - 1e-12 <= exact <= b.upper + 1e-12 and b.upper - b.lower <= 1e-8 * exact


def test_one_slab_on_a_sphere_takes_the_reduced_cell_route():
    box = pg.PointwiseBox(X, envelope={"cmb": 1.0}, rtol=1e-7)
    T = caps(CENTRES[:1])
    D = T.codomain
    c = representer(T, 0)
    slo = box.support(c).lower
    K = pg.FeasibleSet(box, T, pg.Slab(D, [1.0], 0.1 * slo), D([0.5 * slo]))
    b = K.bound(c)
    assert b.lower - 1e-12 <= 0.6 * slo <= b.upper + 1e-12 and b.upper - b.lower <= 1e-8 * slo


def test_a_two_datum_ellipsoid_bracket_narrows_with_the_cells():
    box = pg.PointwiseBox(X, envelope={"cmb": 1.0}, rtol=1e-6)
    T = caps(CENTRES[:2])
    D = T.codomain
    c = representer(caps(np.array([[0.0, 20.0]])), 0)
    m_true = 0.4 * representer(T, 0) * (1.0 / box.support(representer(T, 0)).upper)   # small, inside the box
    d = T(m_true)
    out = []
    for sub in (2, 6):
        K = pg.FeasibleSet(box, T, pg.Ellipsoid(D, 1.0, covariance=list((0.05 * np.abs(d.values)) ** 2)), d)
        K.sub = sub
        out.append(K.bound(c))
    assert all(b.lower <= b.upper for b in out)
    assert out[1].upper - out[1].lower <= out[0].upper - out[0].lower
    assert out[1].lower >= out[0].lower - 1e-9 * abs(out[0].upper)


def test_sphere_cells_refuse_mixed_spaces_and_varying_bounds():
    Y = pg.DirectSum({"cmb": pg.L2(pg.Sphere(1.0)), "line": pg.L2(pg.Interval(0, 1))})
    assert not is_sphere_box(pg.PointwiseBox(Y, envelope={"cmb": 1.0, "line": 1.0}))
    assert is_sphere_box(pg.PointwiseBox(X, envelope={"cmb": 1.0}))
    T = caps()
    with pytest.raises(Unsupported):
        sphere_cells(pg.PointwiseBox(Y, envelope={"cmb": 1.0, "line": 1.0}), T, X.zero)
    with pytest.raises(Unsupported):
        sphere_cells(pg.PointwiseBox(X, upper={"cmb": 1.0}), T, X.zero)
    with pytest.raises(ValueError):
        sphere_cells(pg.PointwiseBox(X, envelope={"cmb": 1.0}), T, X.zero, sub=0)
