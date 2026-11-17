"""pg.Discretisation (plan 025 P2; operators/galerkin.py since plan 068 P6, on pg.Span(basis, space=M)): the span of a
basis as a classic discretisation, every number equal to the space's on the span.

Measured (europa, EXP-039): 1521 P1 hats of a 40 x 40 grid in L2 of the unit square with realisation="local": the
discretisation (fits, Gram, factor) 0.9 s, the Gram equal to the Kronecker product of the 1-D P1 mass matrices to 2e-15.
"""
import time
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.core.errors import Refused, Unsupported

warnings.simplefilter("ignore", pg.RealisationWarning)
B = pg.Box((0, 1), (0, 1))


def p1_mass(n):
    """The 1-D P1 mass matrix of the interior nodes of a uniform grid of n cells on [0, 1]."""
    h = 1.0 / n
    return sp.diags([np.full(n - 2, h / 6), np.full(n - 1, 4 * h / 6), np.full(n - 2, h / 6)], [-1, 0, 1]).toarray()


def hats(n):
    return [F.on(B).hat(count=(n + 1, n + 1), index=(i, j)) for i in range(1, n) for j in range(1, n)]


def f_xy(x, y):
    return np.sin(np.pi * x) * np.sin(2 * np.pi * y)


@pytest.fixture(scope="module")
def hat_case():
    M = pg.L2(B, realisation="local")
    t0 = time.perf_counter()
    D = pg.Discretisation(pg.Span(hats(40), space=M))
    return M, D, time.perf_counter() - t0


def test_the_hat_gram_is_the_closed_form_mass_matrix(hat_case):
    M, D, seconds = hat_case
    assert len(D) == 1521 and seconds < 30                          # 0.9 s on europa
    G = D.subspace.gram.toarray() if sp.issparse(D.subspace.gram) else D.subspace.gram
    Gc = np.kron(p1_mass(40), p1_mass(40))
    assert np.max(np.abs(G - Gc)) <= 1e-13 * Gc.max()
    assert D.claims.find("rank").status == "checked"
    assert "span of 1521" in repr(D)


def test_projection_adjoint_and_coefficient_norms(hat_case):
    M, D, _ = hat_case
    f = M(f_xy)
    S = D.subspace.synthesis
    c = D.subspace.coefficient_space(np.random.default_rng(0).standard_normal(len(D)))
    assert M.inner(S(c), f) == pytest.approx(D.subspace.coefficient_space.inner(c, S.adjoint(f)), rel=1e-12)
    assert D.subspace.coefficient_space.norm(c) == pytest.approx(S(c).norm(), rel=1e-12)
    assert np.allclose(D.subspace.coefficients(f), S.adjoint(f).values, rtol=1e-13, atol=0)
    Pf = D.subspace.project(f)
    assert np.allclose(D.subspace.loads(Pf), D.subspace.loads(f), rtol=1e-10, atol=1e-16)          # f - P f is orthogonal to the span
    h = 1.0 / 40
    nodal = np.array([f_xy(i * h, j * h) for i in range(1, 40) for j in range(1, 40)])
    assert D.residual(f) < (f - D.subspace.element(nodal)).norm()                         # the best approximation in the span


def test_data_matrices_are_the_functionals_of_the_basis(hat_case):
    M, D, _ = hat_case
    ms = [pg.LebesgueMeasure(pg.Box((0.2, 0.4), (0.3, 0.6))), pg.LebesgueMeasure(pg.Box((0.55, 0.9), (0.1, 0.35)))]
    T = pg.Functionals(M, M.riesz(ms))
    A = D.matrix(T)
    assert A.A.shape == (2, len(D))
    c = np.random.default_rng(1).standard_normal(len(D))
    assert np.allclose(A(D.subspace.coefficient_space(c)).values, T(D.subspace.element(c)).values, rtol=1e-12, atol=1e-15)
    assert A.domain == D.subspace.coefficient_space
    with pytest.raises(Unsupported):
        D.matrix(pg.Identity(M))


def test_pixels_give_cell_averages():
    n = 8
    M = pg.L2(B, realisation="local")
    px = [(lambda x, y: 1.0 + 0 * x, ((i / n, (i + 1) / n), (j / n, (j + 1) / n))) for i in range(n) for j in range(n)]
    D = pg.Discretisation(pg.Span([M(f, support=box) for f, box in px], space=M))
    assert np.allclose(np.diag(D.subspace.gram), 1.0 / n ** 2, rtol=1e-13) and np.count_nonzero(D.subspace.gram - np.diag(np.diag(D.subspace.gram))) == 0
    c = D.subspace.coefficients(M(lambda x, y: x * y))
    mid = (np.arange(n) + 0.5) / n
    assert np.allclose(c, np.outer(mid, mid).ravel(), rtol=1e-12)                # averages of x y on the cells


def test_spherical_harmonics_are_orthonormal_and_the_identity_is_the_identity():
    S = pg.Sphere()
    M = pg.L2(S)
    basis = [F.on(S).spherical_harmonic(degree=d, j=j) for d in range(4) for j in range(2 * d + 1)]
    D = pg.Discretisation(pg.Span(basis, space=M))
    assert len(D) == 16 and np.allclose(D.subspace.gram, np.eye(16), atol=1e-13)
    assert np.allclose(D.operator(pg.Identity(M)).A, np.eye(16), atol=1e-12)
    assert np.allclose(D.galerkin(), D.subspace.gram, atol=0)


def test_a_dependent_basis_is_refused():
    M = pg.L2(B)
    a, b = M(lambda x, y: x), M(lambda x, y: y)
    with pytest.raises(Refused, match="rank 2"):
        pg.Discretisation(pg.Span([a, b, a + 2.0 * b], space=M))


def test_the_same_functions_in_another_space():
    n = 6
    D = pg.Discretisation(pg.Span(hats(n), space=pg.L2(B, realisation="local")))
    assert np.allclose(D.gram_in(pg.L2(B, realisation="tensor")), np.kron(p1_mass(n), p1_mass(n)), rtol=1e-12, atol=1e-16)
    with pytest.raises(Unsupported):
        pg.Discretisation(pg.Span(pg.L2(B).family([lambda x, y: x, lambda x, y: y]), space=pg.L2(B))).gram_in(pg.L2(B))


def test_many_local_functions_have_a_sparse_gram():
    n = 81                                                       # 6400 hats: above the dense limit of local.py
    D = pg.Discretisation(pg.Span(hats(n), space=pg.L2(B, realisation="local")))
    assert sp.issparse(D.subspace.gram) and D.subspace.gram.nnz <= 9 * len(D)
    Gc = sp.kron(sp.csr_matrix(p1_mass(n)), sp.csr_matrix(p1_mass(n)))
    assert abs(D.subspace.gram - Gc).max() <= 1e-13 * Gc.max()
    c = D.subspace.coefficients(D.space(f_xy))
    assert np.isfinite(c).all() and D.claims.find("rank").status == "checked"


# ------------------------------------------------------------------ P3: exact pairings with local bases
def _clip_length(a, b, lo, hi):
    d, t0, t1 = b - a, 0.0, 1.0
    for k in range(2):
        if d[k] == 0:
            if not lo[k] <= a[k] <= hi[k]:
                return 0.0
            continue
        u, v = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
        t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
    return max(t1 - t0, 0.0) * np.linalg.norm(d)


def _rays(m, seed=3):
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0, 1, (m, 2)), rng.uniform(0, 1, (m, 2))
    return a, b, [pg.Segment(tuple(p), tuple(q)) for p, q in zip(a, b)]


def _pixels(M, n):
    return [M(lambda x, y: 1.0 + 0 * x, support=((i / n, (i + 1) / n), (j / n, (j + 1) / n)))
            for i in range(n) for j in range(n)]


def test_rays_and_pixels_give_the_exact_path_lengths():
    n = 16
    D = pg.Discretisation(pg.Span(_pixels(pg.L2(B, realisation="local"), n), space=pg.L2(B, realisation="local")))
    a, b, segs = _rays(200)
    E = D.pairings(segs)
    ref = np.array([[_clip_length(p, q, (i / n, j / n), ((i + 1) / n, (j + 1) / n)) for i in range(n) for j in range(n)]
                    for p, q in zip(a, b)])
    assert sp.issparse(E.A) and np.max(np.abs(E.A.toarray() - ref)) <= 1e-14
    assert E.claims[-1].key == "pairings" and E.domain == D.subspace.coefficient_space
    edge = D.pairings([pg.Segment((0.0, 0.25), (1.0, 0.25))]).A.toarray()       # in a face: one row of pixels only
    assert np.count_nonzero(edge) == n and edge.sum() == pytest.approx(1.0, rel=1e-14)


def test_rays_and_hats_are_exact():
    n = 12
    D = pg.Discretisation(pg.Span(hats(n), space=pg.L2(B, realisation="local")))
    a, b, segs = _rays(60, seed=5)
    E = D.pairings(segs).A.toarray()
    g, w = np.polynomial.legendre.leggauss(4)                  # an independent rule, split at every grid line
    idx = [(i, j) for i in range(1, n) for j in range(1, n)]
    for s_, (p, q) in enumerate(zip(a, b)):
        d = q - p
        ts = np.unique(np.clip([0, 1] + [(k / n - p[c]) / d[c] for c in range(2) for k in range(n + 1)], 0, 1))
        tt = (0.5 * (ts[1:] + ts[:-1]))[:, None] + (0.5 * np.diff(ts))[:, None] * g
        ww = (0.5 * np.diff(ts))[:, None] * w * np.linalg.norm(d)
        X, Y = p[0] + tt * d[0], p[1] + tt * d[1]
        ref = [np.sum(ww * np.maximum(0, 1 - np.abs(X * n - i)) * np.maximum(0, 1 - np.abs(Y * n - j))) for i, j in idx]
        assert np.max(np.abs(E[s_] - ref)) <= 1e-15


def test_points_and_boxes_and_the_fallback_to_the_space():
    L = pg.L2(B, realisation="local")
    D = pg.Discretisation(pg.Span(hats(8), space=L))
    boxes = [pg.LebesgueMeasure(pg.Box((0.2, 0.4), (0.3, 0.6))), pg.LebesgueMeasure(pg.Box((0.55, 0.9), (0.1, 0.35)))]
    E = D.pairings(boxes).A.toarray()
    assert np.allclose(E, D.matrix(pg.Functionals(L, L.riesz(boxes))).A, rtol=1e-13, atol=1e-16)
    pts = D.pairings([(0.3, 0.41), (0.77, 0.5)]).A.toarray()
    c = np.random.default_rng(2).standard_normal(len(D))
    f = D.subspace.element(c)
    assert np.allclose(pts @ c, [f(0.3, 0.41), f(0.77, 0.5)], rtol=1e-13)
    T = pg.Discretisation(pg.Span(hats(8), space=pg.L2(B, realisation="tensor")))                # no exact route: the space's
    assert np.allclose(T.pairings(boxes).A.toarray(), E, rtol=1e-12, atol=1e-16)
    with pytest.raises(Refused, match="local basis"):
        T.pairings([pg.Segment((0.1, 0.1), (0.9, 0.8))])                         # a ray is unbounded on L2


def test_many_rays_and_hats_in_seconds():
    D = pg.Discretisation(pg.Span(hats(40), space=pg.L2(B, realisation="local")))
    _, _, segs = _rays(4000, seed=7)
    t0 = time.perf_counter()
    E = D.pairings(segs)
    assert time.perf_counter() - t0 < 20                       # 10^4 rays x 1521 hats: 0.95 s on europa
    assert E.A.shape == (4000, 1521) and E.A.nnz < 0.2 * 4000 * 1521


# ------------------------------------------------------------------ P4: H^1 Grams in the form space
def p1_stiffness(n):
    h = 1.0 / n
    return sp.diags([np.full(n - 2, -1 / h), np.full(n - 1, 2 / h), np.full(n - 2, -1 / h)], [-1, 0, 1]).toarray()


@pytest.mark.parametrize("real, n", [("local", 20), ("tensor", 8)])
def test_the_hat_gram_in_the_form_space_is_the_fem_matrix(real, n):
    ell = 0.1
    V = pg.SobolevForm(B, 1, length_scale=ell).space(realisation=real)
    D = pg.Discretisation(pg.Span(hats(n), space=V))
    m, k = p1_mass(n), p1_stiffness(n)
    G = D.subspace.gram.toarray() if sp.issparse(D.subspace.gram) else D.subspace.gram
    assert np.max(np.abs(G - (np.kron(m, m) + ell ** 2 * (np.kron(k, m) + np.kron(m, k))))) <= 1e-13 * np.abs(G).max()


def test_galerkin_representers_converge_to_the_kernel_energy():
    # the formula inner product and the space's kernel agree: b^T G^-1 b -> ||k_mu||^2 from below, O(h^2)
    # (europa: relative gaps 6.9e-2, 1.8e-2, 4.6e-3 at n = 10, 20, 40, all nodes, natural Neumann)
    V = pg.SobolevForm(B, 1, length_scale=0.1).space(realisation="local")
    mu = pg.LebesgueMeasure(pg.Box((0.3, 0.5), (0.4, 0.7)))
    e = V.riesz(mu).norm() ** 2
    gaps = []
    for n in (10, 20):
        D = pg.Discretisation(pg.Span([F.on(B).hat(count=(n + 1, n + 1), index=(i, j)) for i in range(n + 1)
                                  for j in range(n + 1)], space=V))
        b = D.pairings([mu]).A.toarray().ravel()
        gaps.append((e - b @ D.subspace.solve(b)) / e)
    assert 0 < gaps[1] < gaps[0] < 0.1 and gaps[0] / gaps[1] > 3


def test_dirichlet_faces_and_the_restriction_space():
    W = pg.SobolevForm(B, 1, length_scale=0.1, dirichlet="all").space(realisation="local")
    pg.Discretisation(pg.Span(hats(6), space=W))                                     # interior hats vanish on every face
    with pytest.raises(Refused, match="vanishing"):                  # the bank's membership check, first
        pg.Discretisation(pg.Span([F.on(B).hat(count=(7, 7), index=(0, 3))], space=W))
    with pytest.raises(pg.core.errors.ViolatesBoundaryCondition, match="Dirichlet face"):     # the form's check
        pg.Discretisation(pg.Span([lambda x, y: x * (1 - x) * y * (1 - y), lambda x, y: 1 + x * y], space=W))
    H = pg.Sobolev(B, 1, length_scale=0.1, realisation="local")       # the restriction space: no formula norms
    with pytest.raises(pg.core.errors.NormUnavailable, match="SobolevForm"):
        pg.Discretisation(pg.Span(hats(4), space=H))


# ------------------------------------------------------------------ P5: bases from the bank
def test_bases_from_the_bank():
    L = pg.L2(B, realisation="local")
    D = pg.Discretisation(pg.Span(F.bases.hats(B, 11), space=L))
    assert len(D) == 81 and np.allclose(D.subspace.gram, np.kron(p1_mass(10), p1_mass(10)), rtol=1e-12, atol=1e-16)
    Db = pg.Discretisation(pg.Span(F.bases.hats(B, 11, boundary=True), space=L))
    f = L(lambda x, y: np.exp(x) * np.cos(3 * y))
    assert len(Db) == 121 and Db.residual(f) < 0.01 < D.residual(f)                  # boundary nodes carry f's trace
    P = pg.Discretisation(pg.Span(F.bases.pixels(B, 10), space=L))
    assert np.allclose(P.subspace.gram, 0.01 * np.eye(100), atol=1e-16)
    assert pg.Discretisation(pg.Span(F.bases.bsplines(B, 3, 8), space=L)).residual(f) < 1e-3
    leg = pg.Discretisation(pg.Span(F.bases.legendre(B, 5), space=pg.L2(B)))
    assert np.allclose(leg.subspace.gram, np.eye(36), atol=1e-13)
    assert pg.Discretisation(pg.Span(F.bases.fourier(B, 3), space=pg.L2(B))).subspace.dim == 49
    S, Dk = pg.Sphere(), pg.Disk()
    assert np.allclose(pg.Discretisation(pg.Span(F.bases.spherical_harmonics(S, 5), space=pg.L2(S))).subspace.gram, np.eye(36), atol=1e-13)
    assert np.allclose(pg.Discretisation(pg.Span(F.bases.zernike(Dk, 6), space=pg.L2(Dk))).subspace.gram, np.eye(28), atol=1e-13)
    with pytest.raises(Refused):                                                  # pixels jump: not in H^1
        pg.Discretisation(pg.Span(F.bases.pixels(B, 4), space=pg.SobolevForm(B, 1, length_scale=0.1).space()))


def test_a_box_indicator_is_realised_on_its_own_edges():
    # before plan 025 its fit crashed on open meshgrids (the mask built an array of differently shaped arrays) and its
    # breaks were the edges of its bounding ball (0.198, 0.552 for the box [0.25, 0.5]^2)
    b = F.on(B).box_indicator(lower=(0.25, 0.25), upper=(0.5, 0.5))
    assert np.allclose(b.fit_breaks(), ((0.25, 0.5), (0.25, 0.5)))
    for M in (pg.L2(B, realisation="local"), pg.L2(B)):
        assert b.element(M).norm() ** 2 == pytest.approx(0.0625, rel=1e-13)


def _bumps(m=4, seed=0):
    c = np.random.default_rng(seed).uniform(0.2, 0.8, (m, 2))
    return c, [F.on(B).bump(centre=tuple(p), radius=0.1).normalised() for p in c]


def _bump_hat_reference(c, bumps, m, q=64):
    """int hat_ij b dx by a tensor Gauss rule of q^2 points on every grid cell (the hats are bilinear there); interior
    hats of the m-cell grid in C order. q = 32 is 5e-10 off q = 64 here, so the reference is good to well below 1e-12."""
    h, (g, w) = 1.0 / m, np.polynomial.legendre.leggauss(q)
    u, wu = 0.5 * (g + 1), 0.5 * w
    ref = np.zeros((len(bumps), (m - 1) ** 2))
    for k, b in enumerate(bumps):
        lo, hi = np.floor((c[k] - 0.1) / h).astype(int), np.ceil((c[k] + 0.1) / h).astype(int)
        for a in range(lo[0], hi[0]):
            for e in range(lo[1], hi[1]):
                fv = b.fn((a + u)[:, None] * h + 0 * u, 0 * u[:, None] + (e + u) * h) * np.outer(wu, wu) * h * h
                for di in (0, 1):
                    for dj in (0, 1):
                        i, j = a + di, e + dj
                        if 1 <= i < m and 1 <= j < m:
                            ref[k, (i - 1) * (m - 1) + j - 1] += np.sum(fv * np.outer(u if di else 1 - u, u if dj else 1 - u))
    return ref


def test_functionals_of_a_family_with_a_sparse_gram(monkeypatch):
    """Plan 028: a local realisation returns Grams as scipy.sparse beyond DENSE_MAX members (81 x 81 hats, mac's
    report); Functionals of such a family and M.inner(family, function) gave a ValueError. Lowered DENSE_MAX here."""
    import pygeoinf0.realisations.series.local as lp
    M = pg.SobolevForm(B, 1, length_scale=0.1).space(realisation="local")
    L2 = pg.L2(B, realisation="local")
    c, bumps = _bumps(3)
    T = pg.Functionals(L2, [b.element(L2) for b in bumps]) @ pg.Inclusion(M, L2)
    D = pg.Discretisation(pg.Span(F.bases.hats(B, 11), space=M))
    dense = D.matrix(T).A
    fam = L2.family([h.element(L2) for h in F.bases.hats(B, 11)])
    row = L2.inner(fam, bumps[0].element(L2))
    monkeypatch.setattr(lp, "DENSE_MAX", 10)
    assert sp.issparse(L2.inner(fam, fam))
    assert np.allclose(pg.Discretisation(pg.Span(F.bases.hats(B, 11), space=M)).matrix(T).A, dense, rtol=0, atol=1e-15)
    r = L2.inner(fam, bumps[0].element(L2))
    assert isinstance(r, np.ndarray) and r.shape == (81,) and np.allclose(r, row, rtol=0, atol=1e-15)


def test_pairings_of_densities_are_exact_on_local_bases():
    """Plan 028: density measures (normalised bumps) were paired through their Riesz representers in H^1 (1e-4
    relative off, mac's report, while presented as exact); now L2 inner products of the local realisations, to 1e-12 of
    an independent rule, in the order given among other measures, and the claim says so."""
    M = pg.SobolevForm(B, 1, length_scale=0.1).space(realisation="local")
    D = pg.Discretisation(pg.Span(F.bases.hats(B, 21), space=M))
    c, bumps = _bumps(4)
    ref = _bump_hat_reference(c, bumps, 20)
    E = D.pairings([bumps[0].measure(), pg.Segment((0.1, 0.1), (0.9, 0.8)), *[b.measure() for b in bumps[1:]]])
    A = E.A.toarray()
    assert np.max(np.abs(A[[0, 2, 3, 4]] - ref)) < 1e-12
    assert np.allclose(A.sum(axis=1)[[0, 2, 3, 4]], 1.0, rtol=0, atol=1e-12)
    assert "densities: L2 inner products" in E.claims[-1].reason
    seg = D.pairings([pg.Segment((0.1, 0.1), (0.9, 0.8))]).A.toarray()
    assert np.allclose(A[1], seg[0], rtol=0, atol=1e-15)


def test_pairings_of_no_measures_are_refused_by_name():
    D = pg.Discretisation(pg.Span(hats(4), space=pg.L2(B, realisation="local")))
    for none in ([], (), np.empty((0, 2))):
        with pytest.raises(Refused, match="no measures"):
            D.pairings(none)
