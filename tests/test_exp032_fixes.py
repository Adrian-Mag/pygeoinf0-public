"""Library findings of EXP-032 (experiments/EXP-032-least-squares-suite/REPORT.md on exp032/ls-suite, "Findings about
the library"), fixed generally: 4 (the norms of a family of symbolic representers from the diagonal only), 5 (breaks on
a periodic axis honoured: the indicator of a short arc of a circle), 6 (sub-arc functionals in Sobolev spaces on arcs
and circles, in the path's own coordinate). References: closed forms and scipy.integrate.quad of the formulas, not the
library's own quadrature, and the same functionals on an interval domain (another route of the library)."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

import pygeoinf0 as pg
from pygeoinf0.spaces import representers
from pygeoinf0.core.errors import SupportOutsideDomain
from pygeoinf0.realisations.series import tensor


# ------------------------------------------------------------------------------------- finding 4: diagonal norms

def test_functionals_norms_evaluate_no_off_diagonal_pairing(monkeypatch):
    M = pg.L2(pg.Box("[0, 1]", "[0, 1]"))
    boxes = [pg.Box((0.1 * i, 0.1 * i + 0.3), (0.2, 0.7)) for i in range(6)]      # overlapping neighbours
    calls, real = [], representers.pair

    def spy(space, mu, nu):
        calls.append(mu is nu)
        return real(space, mu, nu)
    monkeypatch.setattr(representers, "pair", spy)
    G = pg.Functionals(M.riesz(boxes))
    assert calls and all(calls) and len(calls) == len(boxes)
    np.testing.assert_allclose(G._knorm, np.sqrt([0.3 * 0.5] * 6), rtol=1e-14)
    calls.clear()
    Gr = G.gram()                                       # the Gram itself still pairs every couple
    assert not all(calls)
    np.testing.assert_allclose(np.sqrt(np.diag(Gr)), G._knorm, rtol=1e-14)


def test_norm_of_one_symbolic_representer_unchanged():
    M = pg.L2(pg.Box("[0, 1]", "[0, 2]"))
    k = M.riesz(pg.Box((0.25, 0.5), (0.0, 2.0)))
    assert abs(M.norm(k) - math.sqrt(0.5)) < 1e-15


# --------------------------------------------------------------------------- finding 5: arcs of a circle in L2

def _arc_indicator(M, a, b):
    def ind(x, y):
        return (np.mod(np.arctan2(y, x) - a, 2 * np.pi) <= b - a).astype(float)
    return M(ind, breaks=[[float(np.mod(a, 2 * np.pi)), float(np.mod(b, 2 * np.pi))]])


def _model(t):
    return np.exp(np.cos(t)) * np.cos(3 * np.sin(t))


ARCS = [(0.30, 0.35), (6.2, 0.1 + 2 * np.pi), (1.0, 3.0), (0.32, 0.4)]          # short, across phi = 0, long


def test_short_arc_indicator_on_the_circle_is_not_zero():
    M = pg.L2(pg.Circle(2.0))                                                  # measure R dphi
    for a, b in ARCS:
        k = _arc_indicator(M, a, b)
        assert abs(M.norm(k) ** 2 - 2.0 * (b - a)) < 1e-13
        assert isinstance(k.realisation.bases[0], tensor.AxisBasis)            # fitted piecewise at the breaks


def test_arc_functionals_on_the_circle_against_quadrature():
    M = pg.L2(pg.Circle(1.0))
    G = pg.Functionals([_arc_indicator(M, a, b) for a, b in ARCS])
    m = M(lambda x, y: np.exp(x) * np.cos(3 * y))
    assert isinstance(m.realisation.bases[0], tensor.FourierBasis)            # smooth formulas stay Fourier
    exact = [quad(_model, a, b, epsabs=1e-14, epsrel=1e-13)[0] for a, b in ARCS]
    np.testing.assert_allclose(G(m).values, exact, atol=1e-13)
    lengths = np.array([b - a for a, b in ARCS])
    over = [max(0.0, min(b, d) - max(a, c)) for (a, b) in ARCS for (c, d) in ARCS]
    np.testing.assert_allclose(G.gram(), np.reshape(over, (4, 4)), atol=1e-13)
    np.testing.assert_allclose(G._knorm ** 2, lengths, atol=1e-13)
    y = np.array([1.0, -2.0, 0.5, 1.0])
    s = G.adjoint(pg.Vector(G.codomain, y))
    assert abs(M.inner(m, s) - y @ exact) < 1e-12                             # mixed Fourier / piecewise pairing
    d = m - s                                                                  # mixed sum: values to rounding
    t = np.linspace(0.0, 2 * np.pi, 997)
    ind = np.array([[(np.mod(u - a, 2 * np.pi) <= b - a) for a, b in ARCS] for u in t], float)
    np.testing.assert_allclose(d(np.cos(t), np.sin(t)), _model(t) - ind @ y, atol=1e-12)
    ref = quad(lambda u: _model(u) ** 2, 0, 2 * np.pi, epsabs=1e-14, limit=200)[0] - 2 * y @ exact \
        + y @ np.reshape(over, (4, 4)) @ y
    assert abs(M.norm(d) ** 2 - ref) < 1e-11


def test_fourier_to_piecewise_transfer_is_exact_to_rounding():
    lo, hi = 0.0, 2 * np.pi
    F = tensor.FourierBasis(lo, hi, tensor.FOURIER_MAX)                      # the highest frequency, K = 256
    P = tensor.AxisBasis([lo, 0.3, 0.35, hi], 1)
    U = tensor._union_basis([F, P])
    rng = np.random.default_rng(1)
    C = rng.standard_normal((F.n, 2)) / (1 + np.arange(F.n))[:, None]
    D = tensor._transfer_apply(C, 0, F, U)
    t = rng.uniform(lo, hi, 500)
    want = F.vander(t) @ C
    p, loc, _ = U.locate(t)
    got = np.einsum("ma,man->mn", U.vander(loc), D.reshape(U.P, U.n, 2)[p])
    assert np.abs(got - want).max() < 1e-12 * np.abs(C).sum(axis=0).max()


def test_breaks_at_the_ends_of_a_periodic_axis_keep_fourier():
    M = pg.L2(pg.Circle(1.0))
    f = M(lambda x, y: x, breaks=[[0.0, 2 * np.pi]])
    assert isinstance(f.realisation.bases[0], tensor.FourierBasis)


# ------------------------------------------------------------- finding 6: sub-arc functionals in Sobolev spaces

def _on_equator(t):
    return (math.cos(t), math.sin(t), 0.0)


def _weight(x, y, *z):
    return 1.0 + 0.5 * x * y


def test_sub_arc_functionals_on_an_arc_against_an_interval_and_quadrature():
    S2 = pg.Sphere(1.0)
    H = pg.Sobolev(pg.Arc(S2, _on_equator(0.0), _on_equator(2.0)), 1, length_scale=0.3)
    HI = pg.Sobolev(pg.Interval(0.0, 2.0), 1, length_scale=0.3)              # the same space in arc length
    sub = pg.Arc(S2, _on_equator(0.3), _on_equator(0.9))
    k = H.riesz(sub)
    kI = HI.riesz(pg.Interval(0.3, 0.9))
    assert abs(H.norm(k) - HI.norm(kI)) < 1e-12 * HI.norm(kI)
    kw = H.riesz(pg.LebesgueMeasure(sub).with_density(_weight))             # a weight of the Cartesian coordinates
    kwI = HI.riesz(pg.LebesgueMeasure(pg.Interval(0.3, 0.9)).with_density(
        lambda s: _weight(np.cos(s), np.sin(s))))
    assert abs(H.inner(kw, k) - HI.inner(kwI, kI)) < 1e-9 * abs(HI.inner(kwI, kI))
    f = H(lambda x, y, z: np.exp(x) * np.cos(3 * y))
    for kk, w in ((k, lambda t: 1.0), (kw, lambda t: _weight(np.cos(t), np.sin(t)))):
        ref = quad(lambda t: w(t) * _model(t), 0.3, 0.9, epsabs=1e-14, epsrel=1e-13)[0]
        assert abs(H.inner(kk, f) - ref) < 1e-10


def test_sub_arc_functionals_on_a_circle_against_quadrature():
    C = pg.Circle(1.5)
    H = pg.Sobolev(C, 1, length_scale=0.3)
    f = H(lambda x, y: np.exp(x / 1.5) * np.cos(2 * y / 1.5))
    for a, b in [(0.3, 0.9), (6.0, 6.6), (-0.4, 0.2)]:                       # across phi = 0 both ways
        k = H.riesz(C.arc(a, b))
        ref = quad(lambda t: 1.5 * _model2(t), a, b, epsabs=1e-14, epsrel=1e-13)[0]          # ds = R dphi
        assert abs(H.inner(k, f) - ref) < 1e-10 * max(1.0, abs(ref))
        # ||k||^2 = int_arc k ds, with k's values from the potential route (not the energy route of the norm)
        val = quad(lambda t: 1.5 * float(k(1.5 * np.cos(t), 1.5 * np.sin(t))), a, b, epsabs=1e-13, epsrel=1e-12)[0]
        assert abs(H.norm(k) ** 2 - val) < 1e-8 * val
    k1, k2 = H.riesz(C.arc(6.0, 6.6)), H.riesz(C.arc(-0.4 + 2 * np.pi, 0.2 + 2 * np.pi))      # rotation invariance
    assert abs(H.norm(k1) - H.norm(k2)) < 1e-12


def _model2(t):
    return np.exp(np.cos(t)) * np.cos(2 * np.sin(t))


def test_a_set_off_the_path_is_refused():
    S2 = pg.Sphere(1.0)
    H = pg.Sobolev(pg.Arc(S2, _on_equator(0.0), _on_equator(1.0)), 1, length_scale=0.3)
    with pytest.raises(SupportOutsideDomain):
        H.riesz(pg.Arc(S2, _on_equator(0.5), _on_equator(1.5)))


# ------------------------------------------- finding 2: overlapping sub-paths, parallelogram tubes, arcs of a circle

def _clip(poly, other):
    """Sutherland-Hodgman: the part of the convex polygon poly (counter-clockwise vertices) inside the convex other."""
    out = list(poly)
    for i in range(len(other)):
        a, b = np.asarray(other[i]), np.asarray(other[(i + 1) % len(other)])

        def side(p, a=a, b=b):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        inp, out = out, []
        for j in range(len(inp)):
            p, q = np.asarray(inp[j]), np.asarray(inp[(j + 1) % len(inp)])
            if side(q) >= 0:
                if side(p) < 0:
                    out.append(p + (q - p) * side(p) / (side(p) - side(q)))
                out.append(q)
            elif side(p) >= 0:
                out.append(p + (q - p) * side(p) / (side(p) - side(q)))
    return out


def _moments(poly):
    """(area, int x dA) of a polygon (shoelace)."""
    P = np.asarray(poly)
    x, y = P[:, 0], P[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    return 0.5 * cr.sum(), (cr * (x + xn)).sum() / 6.0


def _corners(o, e1, e2):
    o, e1, e2 = (np.asarray(v, float) for v in (o, e1, e2))
    P = [o, o + e1, o + e1 + e2, o + e2]
    return P if e1[0] * e2[1] - e1[1] * e2[0] > 0 else P[::-1]


TUBES = [((0.1, 0.1), (0.8, 0.6), (-0.012, 0.016)), ((0.1, 0.8), (0.8, -0.5), (0.01, 0.016)),
         ((0.12, 0.11), (0.7, 0.55), (-0.012, 0.016)), ((0.2, 0.3), (0.5, 0.2), (-0.05, 0.1))]


def test_overlapping_parallelogram_tubes_pair_exactly():
    M = pg.L2(pg.Box("[0, 1]", "[0, 1]"))
    sets = [pg.Patch(*t) for t in TUBES] + [pg.Box("[0.2, 0.5]", "[0.1, 0.35]")]
    polys = [_corners(*t) for t in TUBES] + [[(0.2, 0.1), (0.5, 0.1), (0.5, 0.35), (0.2, 0.35)]]
    k = M.riesz(sets)
    G = M.inner(k, k)
    for i in range(len(sets)):
        for j in range(len(sets)):
            clipped = _clip(polys[i], polys[j])
            ref = _moments(clipped)[0] if len(clipped) > 2 else 0.0
            assert abs(G[i, j] - ref) < 1e-13, (i, j)
    assert G[0, 2] > 0 and G[0, 1] > 0                                         # overlapping and crossing tubes
    # a weight: int_{S cap T} x dA (x the weight of one set), against the polygon's first moment
    kw = M.riesz(pg.LebesgueMeasure(sets[0]).with_density(lambda x, y: x))
    for j in (1, 2, 4):
        ref = _moments(_clip(polys[0], polys[j]))[1]
        assert abs(M.inner(kw, k[j]) - ref) < 1e-12


def test_parallelepipeds_in_a_box_pair_exactly():
    from pygeoinf0.domains import Flat
    M = pg.L2(pg.Box("[0, 1]", "[0, 1]", "[0, 1]"))
    E = np.array([(0.6, 0.2, 0.1), (-0.05, 0.1, 0.0), (0.0, 0.02, 0.1)])
    o = np.array([0.1, 0.1, 0.1])
    shift = np.array([0.2, -0.1, 0.3])                                        # in edge coordinates
    A, B = Flat(o, E), Flat(o + shift @ E, E)
    vol = abs(np.linalg.det(E))
    ref = np.prod(1 - np.abs(shift)) * vol                                    # translated copies
    kA, kB = M.riesz(A), M.riesz(B)
    assert abs(M.inner(kA, kB) - ref) < 1e-14
    assert abs(M.inner(kA, M.riesz(pg.Box("[0, 1]", "[0, 1]", "[0, 1]"))) - vol) < 1e-14


def test_sub_segments_and_sub_polylines_pair_by_their_overlap():
    D = pg.Polyline([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    M = pg.L2(D)
    A = pg.Polyline([(0.5, 0.0), (1.0, 0.0), (1.0, 0.6)])                   # 0.5 + 0.6
    B = pg.Polyline([(0.8, 0.0), (1.0, 0.0), (1.0, 1.0), (0.7, 1.0)])       # 0.2 + 1 + 0.3
    C = pg.Segment((1.0, 0.3), (1.0, 0.9))
    k = M.riesz([A, B, C])
    np.testing.assert_allclose(M.inner(k, k), [[1.1, 0.8, 0.3], [0.8, 1.5, 0.6], [0.3, 0.6, 0.6]], atol=1e-13)


def test_arcs_of_a_circle_pair_by_their_overlap_in_l2():
    C = pg.Circle(2.0)
    M = pg.L2(C)
    arcs = [(0.1, 0.9), (0.5, 1.3), (6.0, 6.6), (-0.5, 0.2), (2.0, 3.0)]
    k = M.riesz([C.arc(a, b) for a, b in arcs])
    G = M.inner(k, k)
    for i, (a, b) in enumerate(arcs):
        for j, (c, d) in enumerate(arcs):
            ref = sum(max(0.0, min(b, d + s) - max(a, c + s)) for s in (-2 * np.pi, 0.0, 2 * np.pi))
            assert abs(G[i, j] - 2.0 * ref) < 1e-12, (i, j)
    f = M(lambda x, y: np.exp(x / 2.0))
    for i, (a, b) in enumerate(arcs):
        ref = quad(lambda t: 2.0 * np.exp(np.cos(t)), a, b, epsabs=1e-14)[0]
        assert abs(M.inner(k[i], f) - ref) < 1e-11


def test_arcs_overlapping_twice_are_refused():
    C = pg.Circle(1.0)
    M = pg.L2(C)
    k1, k2 = M.riesz(C.arc(0.0, 5.0)), M.riesz(C.arc(4.0, 7.0))
    with pytest.raises(pg.Refused):
        M.inner(k1, k2)


# ----------------------------------------------- finding 8: cells paired with formulas on tori and boxes, exactly

def _separable_reference(fs, box):
    return np.prod([quad(f, a, b, epsabs=1e-14, epsrel=1e-13)[0] for f, (a, b) in zip(fs, box)])


def test_cells_paired_with_a_formula_on_the_3d_torus_and_box(monkeypatch):
    fs_torus = [lambda x: np.exp(np.sin(2 * np.pi * x)), lambda y: np.cos(2 * np.pi * y) + 0.3,
                lambda z: 1 + 0.5 * np.sin(4 * np.pi * z)]
    fs_box = [lambda x: np.exp(x), lambda y: np.cos(3 * y), lambda z: 1 + z * z]
    cells = [((0.0, 0.0625), (0.5, 0.5625), (0.25, 0.3125)), ((0.3, 0.7), (0.1, 0.2), (0.9, 1.0)),
             ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))]
    for D, fs in ((pg.Torus("[0, 1]", "[0, 1]", "[0, 1]"), fs_torus), (pg.Box("[0, 1]", "[0, 1]", "[0, 1]"), fs_box)):
        M = pg.L2(D)
        f = M(lambda x, y, z, fs=fs: fs[0](x) * fs[1](y) * fs[2](z))
        k = M.riesz([pg.Box(*c) for c in cells])
        f.realisation
        with monkeypatch.context() as mp:                  # the exact route: the formula is not evaluated at points
            mp.setattr(tensor, "evaluate_points", None)
            got = M.inner(k, f)
        ref = [_separable_reference(fs, c) for c in cells]
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


def test_box_integral_rows_against_quadrature():
    from pygeoinf0.realisations.rkhs.box_form import axis_row
    rng = np.random.default_rng(3)
    F = tensor.FourierBasis(0.0, 2.0, 41)
    P = tensor.AxisBasis([0.0, 0.5, 1.2, 2.0], 9)
    for b in (F, P):
        c = rng.standard_normal(b.S)
        p_loc = lambda t, b=b: (lambda p, loc, _: np.einsum("ma,ma->m", b.vander(loc),        # noqa: E731
                                                           c.reshape(b.P, b.n)[p]))(*b.locate(t))
        for lo, hi in ((0.1, 1.9), (0.6, 0.9), (0.0, 2.0)):
            pts = sorted({lo, hi} | {x for x in b.breaks if lo < x < hi})
            ref = sum(quad(lambda t: float(p_loc(np.array([t]))[0]), u, v, epsabs=1e-14, limit=200)[0]
                      for u, v in zip(pts[:-1], pts[1:]))
            assert abs(axis_row(b, lo, hi) @ c - ref) < 1e-12
