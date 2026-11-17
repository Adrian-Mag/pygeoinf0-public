"""Plan 005, P4 (option C): the H-matrix of atom energies against the exact energies."""
import numpy as np
import pytest

import ladder
import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported
from pygeoinf0.realisations.rkhs.hmatrix import HMatrix


def _atoms(P):
    K = P.H.riesz(P.measures)
    return K.realisation.kx.atoms, P.H.backend.engine


@pytest.mark.parametrize("name, n, kw, far", [("L3", 300, {"segments": 8}, False), ("L2", 600, {}, False),
                                              ("L1", 600, {}, True)])
def test_products_agree_with_the_exact_energies(name, n, kw, far):
    A, engine = _atoms(ladder.build(name, n, **kw))
    E = engine.exact.energy(A, A)                                      # dense exact (n x n), cached on A
    A.self_energy = None                                              # so that apply uses the H-matrix
    with pg.settings.using(dense_max=10, compression=1e-10, engine="hmatrix"):
        V = np.random.default_rng(0).normal(size=(A.n, 3))
        Y = engine.apply(A, V)
        H = A.hmatrix
    assert H is not None and (H.stats["far"] > 0 or not far)
    err = np.linalg.norm(Y - E @ V) / np.linalg.norm(E @ V)
    assert err <= max(1e-9, 10 * H.rel_error) and H.rel_error <= 1e-8, (err, H.rel_error)
    assert "compressed" in str(H.claims)


def test_blocks_cover_every_pair_once():
    A, engine = _atoms(ladder.build("L3", 80, segments=8))
    H = HMatrix(engine.exact, A)
    N = len(A.P)
    cover = np.zeros((N, N), int)
    p = H.tree.perm
    for a, b, c, d, kind, data in H.blocks:
        cover[np.ix_(p[a:b], p[c:d])] += 1
        if a != c:
            cover[np.ix_(p[c:d], p[a:b])] += 1
    assert np.all(cover == 1)


def test_memory_budget_is_checked():
    A, engine = _atoms(ladder.build("L3", 80, segments=8))
    with pg.settings.using(memory_gb=1e-6):
        with pytest.raises(Unsupported, match="memory_gb"):
            HMatrix(engine.exact, A)


def test_memory_budget_is_checked_while_building():
    """The near field passes the check before building, the stored far blocks exceed the budget while building."""
    A, engine = _atoms(ladder.build("L1", 600))
    H = HMatrix(engine.exact, A)
    near = sum(int(H.tree.size(t)) * int(H.tree.size(s)) for t, s in H.tree.partition()[0])
    assert H.stats["entries"] > near
    gb = 8 * (near + H.stats["entries"]) / 2 / 2 ** 30
    with pg.settings.using(memory_gb=gb):
        with pytest.raises(Unsupported, match="after .* blocks"):
            HMatrix(engine.exact, A)


def test_periodic_domains_use_the_exact_engine():
    T = pg.Torus("[0, 1]", "[0, 1]")
    H = pg.Sobolev(T, 1.5, length_scale=0.1)
    pts = np.random.default_rng(0).uniform(0, 1, (30, 2))
    K = H.riesz([pg.Segment(tuple(p), tuple(p + [0.1, 0.05])) for p in pts * 0.8])
    A = K.realisation.kx.atoms
    with pg.settings.using(dense_max=5):
        y = H.backend.engine.apply(A, np.ones(A.n))
    assert A.hmatrix is None and np.all(np.isfinite(y))


def test_parallel_products_equal_serial_ones(monkeypatch):
    from pygeoinf0.realisations.rkhs import hmatrix
    A, engine = _atoms(ladder.build("L1", 600))
    H = HMatrix(engine.exact, A)
    V = np.random.default_rng(3).normal(size=(A.n, 2))
    serial = H.apply(V)
    monkeypatch.setattr(hmatrix, "PARALLEL_BLOCKS", 10)
    monkeypatch.setenv("PYGEOINF0_WORKERS", "3")
    assert np.allclose(H.apply(V), serial, rtol=1e-13, atol=1e-13 * np.abs(serial).max())


def test_compressed_dense_gram_is_an_option_with_its_error_stated():
    P = ladder.build("L3", 250, segments=6)
    exact = P.H.backend.engine.exact.energy(*(2 * [P.H.riesz(P.measures).realisation.kx.atoms]))
    with pg.settings.using(dense_gram="compressed", compression=1e-10):
        K = P.H.riesz(P.measures)
        G = pg.Functionals(K, into=P.D)
        A = G @ G.adjoint                                              # dense (n <= dense_max), from the H-matrix
        H = K.realisation.kx.atoms.hmatrix
        assert H is not None and "estimated relative error" in str(A.claims)
        Ec = P.H.backend.engine.energy(K.realisation.kx.atoms, K.realisation.kx.atoms)
        G2, alpha, post = ladder.solve(P)
    assert np.linalg.norm(Ec - exact) <= max(10 * H.rel_error, 1e-12) * np.linalg.norm(exact)
    _, alpha_exact, _ = ladder.solve(P)
    assert alpha == pytest.approx(alpha_exact, rel=1e-6)
