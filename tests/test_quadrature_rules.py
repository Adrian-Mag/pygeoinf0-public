"""Plan 009 P4a: a fixed quadrature rule on a parent interval, restricted to the pieces of a decomposed space
(pg.realisations.Trapezoid(n, over=...)), and per-summand rules for the structured Grams of separable functionals.
Use outside the CMB problem: a layered ball whose layers cut one radial interval, integrated on one grid as a
computation on the undivided ball would be (reproducing such a computation summand by summand). References: NumPy's
trapezoid sums on the parent grid, written out here."""
import numpy as np
import pytest

import pygeoinf0 as pg


def test_the_pieces_of_a_partition_add_up_to_the_parent_rule():
    q = pg.realisations.Trapezoid(101, over=(0.0, 2.0))
    f = lambda x: np.exp(x) * np.cos(3 * x)
    x, w = np.linspace(0, 2, 101), np.full(101, 0.02)
    w[[0, -1]] *= 0.5
    parent = float(w @ f(x))
    for cuts in ([0.0, 0.73, 2.0], [0.0, 0.5, 1.37, 2.0], [0.0, 0.02, 2.0]):     # 0.5 and 0.02 are nodes
        pieces = sum(float(q.integrate(f, a, b)) for a, b in zip(cuts[:-1], cuts[1:]))
        assert pieces == pytest.approx(parent, rel=1e-14, abs=1e-14)
        n = sum(len(q.nodes(a, b)[0]) for a, b in zip(cuts[:-1], cuts[1:]))
        assert n == 101                                       # every parent node counted once (half-open pieces)
    assert np.array_equal(q.nodes(0.0, 2.0)[1], w)
    with pytest.raises(ValueError):
        q.nodes(-0.1, 1.0)
    plain = pg.realisations.Trapezoid(101)
    assert float(plain.integrate(f, 0.0, 2.0)) == pytest.approx(parent, rel=1e-15)


def test_per_summand_rules_reproduce_the_undivided_trapezoid_gram():
    """Radial kernels k_i = r^2 p_i(r) (p_i cubic) on a ball cut at r = 0.4 and 0.75 into three summands with equal
    weights (L2 of the ball, decomposed), against dr: representers p_i. The structured Gram with one parent rule
    restricted to the layers equals the trapezoid sum sum_n w_n r_n^2 p_i(r_n) p_j(r_n) / w on the parent grid; the
    exact one equals int r^2 p_i p_j dr / w (polynomial integration, NumPy)."""
    planet = pg.LayeredBall([0.4, 0.75, 1.0], names=("a", "b", "c"))
    V = pg.PiecewiseSpace.lebesgue(planet, realisation="radial_harmonics")
    wt = 0.8
    M = pg.DirectSum({k: V.summand(k) for k in ("a", "b", "c")}, weights={k: wt for k in ("a", "b", "c")})
    rng = np.random.default_rng(3)
    P = [np.polynomial.Polynomial(rng.normal(size=4)) for _ in range(4)]
    kernels = [(lambda r, p=p: r * r * p(r)) for p in P]
    lab = [(2, 0), (2, 0), (2, 3), (2, 0)]
    q = pg.realisations.Trapezoid(257, over=(0.0, 1.0))
    radial = {k: kernels for k in ("a", "b", "c")}
    dr = {k: pg.LebesgueMeasure(f"[{a}, {b}]") for k, a, b in (("a", 0, 0.4), ("b", 0.4, 0.75), ("c", 0.75, 1))}
    G = pg.Functionals.separable(M, radial, lab, against=dr, quadrature={k: q for k in ("a", "b", "c")})
    Gx = pg.Functionals.separable(M, radial, lab, against=dr)
    x, w = np.linspace(0, 1, 257), np.full(257, 1 / 256)
    w[[0, -1]] *= 0.5
    ref, exact = np.zeros((4, 4)), np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if lab[i] == lab[j]:
                ref[i, j] = (w * x * x * P[i](x) * P[j](x)).sum() / wt
                prim = (np.polynomial.Polynomial([0, 0, 1]) * P[i] * P[j]).integ()
                exact[i, j] = (prim(1.0) - prim(0.0)) / wt
    A = G.structured_gram().matrix()
    assert np.abs(A - ref).max() <= 1e-12 * np.abs(ref).max()
    assert np.abs(Gx.structured_gram().matrix() - exact).max() <= 1e-12 * np.abs(exact).max()
    claim = [c.statement for c in G.structured_gram().claims if "radial integrals" in c.statement]
    assert claim and "restricted" in claim[0]
