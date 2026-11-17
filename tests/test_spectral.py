"""Plan 005, P4 (the author's decision 5): the spectral energy engine (realisations/rkhs/spectral.py) against the exact
energies, its estimate and rate, its place behind pg.settings.engine, its fallbacks and claims, and finufft kept out of
forked workers. Short runs (europa); the tests that need finufft are skipped without it."""
import gc
import math
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

import ladder
import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported
from pygeoinf0 import execution
from pygeoinf0.realisations.rkhs import spectral as S
from pygeoinf0.realisations.rkhs import spectral_terms as ST

needs_finufft = pytest.mark.skipif(S._finufft() is None, reason="finufft is not installed (pip install finufft)")
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK_WORKERS = min(4, int(os.environ.get("PYGEOINF0_WORKERS", "1")))


def _atoms(P):
    return P.H.riesz(P.measures).realisation.kx.atoms, P.H.backend.engine


def _exact(A, engine):
    """The exact energies of A, then forgotten by A, so that products go through the engines."""
    E = engine.exact.energy(A, A)
    A.self_energy = None
    return E


def _rel(Y, Z):
    return float(np.max(np.linalg.norm(Y - Z, axis=0) / np.linalg.norm(Z, axis=0)))


def _discrepancy_coefficients(E, d, sigma):
    """c = (E + lam I)^-1 d with ||lam (E + lam I)^-1 d||^2 = n sigma^2 (the posterior mean's data coefficients)."""
    w, U = np.linalg.eigh(E)
    b = U.T @ d
    lo, hi = 1e-14 * w.max(), 1e3 * w.max()
    for _ in range(200):
        lam = math.sqrt(lo * hi)
        lo, hi = (lam, hi) if np.sum((lam / (w + lam) * b) ** 2) < len(d) * sigma ** 2 else (lo, lam)
    return U @ (b / (w + lam))


@needs_finufft
@pytest.mark.parametrize("name, n, kw", [("L3", 60, {"segments": 8}), ("L2", 60, {}), ("fan", 60, {"segments": 8})])
def test_products_agree_with_the_exact_energies(name, n, kw):
    """Below the claimed estimate and within the target, in products and in predicted data (decision 3)."""
    P = ladder.build(name, n, **kw)
    A, engine = _atoms(P)
    E = _exact(A, engine)
    eng = S.SpectralEngine(engine.exact)
    V = np.random.default_rng(1).standard_normal((A.n, 3))
    err = _rel(eng.apply(A, V), E @ V)
    plan = A.spectral
    assert err <= plan.est <= plan.tol and err <= eng.estimate(A, V), (err, plan.est, plan.tol)
    claims = eng.claims(A)
    assert [c.status for c in claims] == ["derived", "checked"] and f"{plan.est:.1e}" in claims[1].statement
    c = _discrepancy_coefficients(E, np.asarray(P.d.values, float), P.sigma)
    D = (eng.apply(A, c) - E @ c) / P.sigma                                    # predicted-data errors in noise std
    rms, top = eng.estimate_data(A, c, P.sigma)
    assert np.sqrt(np.mean(D ** 2)) <= min(rms, pg.settings.accuracy) and np.max(np.abs(D)) <= top, (D, rms, top)


@needs_finufft
@pytest.mark.parametrize("name, kw", [("L3", {"segments": 8}), ("L2", {})])
def test_the_rate_of_corrected_products(name, kw):
    """The derived rate p = 2s + 2 - d (3 for L3, s = 2, d = 3, and for L2, s = 3/2, d = 2) against the errors at
    l Xi = 8, 8 sqrt 2, 16: the observed rate over the doubling is near p (measured 3.8 and 3.1 at n = 60), and the
    two-resolution estimate with p is above the actual error at every resolution."""
    A, engine = _atoms(ladder.build(name, 40, **kw))
    E = _exact(A, engine)
    V = np.random.default_rng(2).standard_normal((A.n, 2))
    errs, ests = [], []
    for lxi in (8.0, 8 * math.sqrt(2), 16.0):
        eng = S.SpectralEngine(engine.exact, lxi=lxi)
        errs.append(_rel(eng.apply(A, V), E @ V))
        ests.append(eng.estimate(A, V))
    p = S.rate(engine.kernel)
    assert p == 3 and math.log2(errs[0] / errs[2]) >= 0.8 * p, errs
    assert all(e <= s for e, s in zip(errs, ests)), (errs, ests)


@needs_finufft
def test_the_ladder_through_the_settings():
    """The unchanged user code (ladder.solve) with pg.settings.using(engine="spectral", dense_max=...) against the
    dense route: alpha, predicted data, and the posterior mean at one fixed set of probe points (as
    benchmarks/check_matrix_free.py), with the claims naming the engine and its estimate."""
    P = ladder.build("L3", 200, segments=8)
    X = np.random.default_rng(11).uniform(-0.9, 0.9, (400, 3)) / math.sqrt(3)
    out = {}
    for route, kw in (("dense", {}), ("spectral", dict(engine="spectral", dense_max=100))):
        with pg.settings.using(**kw):
            G, alpha, post = ladder.solve(P)
            out[route] = (float(alpha), np.asarray(post.mean(*X.T), float).ravel(), np.asarray(G(post.mean).values))
            claims = str((G @ G.adjoint).claims)
    assert "spectral energies" in claims and "estimated relative error" in claims
    (a0, m0, d0), (a1, m1, d1) = out["dense"], out["spectral"]
    std = math.sqrt(P.H.backend.kernel.k0 / a0)                               # prior point std
    assert a1 == pytest.approx(a0, rel=1e-3)
    assert np.sqrt(np.mean((d1 - d0) ** 2)) <= 0.03 * P.sigma                 # CG (0.01 sigma) and engine together
    assert np.max(np.abs(m1 - m0)) <= 0.01 * std


@needs_finufft
def test_direct_transforms_agree_with_finufft(monkeypatch):
    A, engine = _atoms(ladder.build("L3", 6, segments=4))
    eng = S.SpectralEngine(engine.exact, lxi=4.0, eps=1e-12, reach_tol=1e-4)
    V = np.random.default_rng(3).standard_normal((A.n, 2))
    y1 = eng.apply(A, V)
    assert "finufft" in str(eng.claims(A))
    A.spectral = None
    monkeypatch.setattr(S, "_finufft", lambda: None)
    y2 = eng.apply(A, V)
    assert "direct sums (finufft not installed)" in str(eng.claims(A))
    assert np.max(np.abs(y1 - y2)) <= 1e-9 * np.max(np.abs(y1))


def test_without_finufft_tiny_problems_use_direct_sums_and_larger_ones_are_refused(monkeypatch):
    monkeypatch.setattr(S, "_finufft", lambda: None)
    A, engine = _atoms(ladder.build("L3", 6, segments=4))
    E = _exact(A, engine)
    eng = S.SpectralEngine(engine.exact, lxi=4.0)
    V = np.random.default_rng(4).standard_normal((A.n, 2))
    assert _rel(eng.apply(A, V), E @ V) <= 1e-2
    assert "direct sums (finufft not installed)" in str(eng.claims(A))
    B, engine = _atoms(ladder.build("L3", 300, segments=8))
    with pytest.raises(Unsupported, match="pip install finufft"):
        S.SpectralEngine(engine.exact).apply(B, np.ones(B.n))


def test_flat_torus_with_unequal_periods():
    """Periods per axis (no cubic cell), lattice images in the corrections (segments across the cell's faces)."""
    H = pg.Sobolev(pg.Torus("[0, 1]", "[0, 2]"), 1.5, length_scale=0.1)
    rng = np.random.default_rng(5)
    p = rng.uniform((0.0, 0.0), (1.0, 2.0), (40, 2))
    q = p + 0.25 * rng.normal(size=(40, 2)) / math.sqrt(2)
    p[:4], q[:4] = [(0.9, 0.5), (0.05, 0.4), (0.5, 1.9), (0.4, 0.1)], [(1.1, 0.55), (0.05, 0.7), (0.55, 2.15),
                                                                      (0.7, 0.1)]   # across the faces, and crossing
    A = H.riesz([pg.Segment(tuple(a), tuple(b)) for a, b in zip(p, q)]).realisation.kx.atoms   # their images
    engine = H.backend.engine
    E = _exact(A, engine)
    eng = S.SpectralEngine(engine.exact)
    V = rng.standard_normal((A.n, 2))
    err = _rel(eng.apply(A, V), E @ V)
    plan = A.spectral
    assert np.any(plan.pairs[2] != 0) and plan.used.M[0] != plan.used.M[1]
    assert err <= plan.est <= plan.tol, (err, plan.est)
    assert "no aliasing" in str(eng.claims(A))


def test_the_plan_lives_on_the_atom_set():
    """Caveat 1: no cache keyed by id(A) (ids are reused after garbage collection); the plan is A.spectral."""
    P = ladder.build("L3", 6, segments=4)
    A, engine = _atoms(P)
    eng = S.SpectralEngine(engine.exact, lxi=4.0)
    assert A.spectral is None
    eng.apply(A, np.ones(A.n))
    plan = A.spectral
    eng.apply(A, np.ones(A.n))
    assert A.spectral is plan and not any(isinstance(v, dict) for v in vars(eng).values())
    with pg.settings.using(accuracy=0.001):                      # another target: a new plan, the same pairs
        eng.apply(A, np.ones(A.n))
        assert A.spectral is not plan and A.spectral.pairs is plan.pairs
    del A, plan
    gc.collect()
    B, _ = _atoms(P)
    assert B.spectral is None


@pytest.mark.parametrize("case", ["points", "spheres", "patches", "circles", "mixed", "interval", "reflections"])
def test_other_measures_fall_back_and_say_so(case):
    rng = np.random.default_rng(6)
    ball = pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.2)
    pts = [pg.DiracMeasure(tuple(x)) for x in rng.uniform(-0.4, 0.4, (4, 3))]
    H, measures = {
        "points": lambda: (ball, pts),
        "spheres": lambda: (ball, [pg.Sphere(0.1, tuple(c)) for c in rng.uniform(-0.4, 0.4, (3, 3))]),
        "patches": lambda: (ball, [pg.Patch(tuple(o), (0.2, 0.0, 0.0), (0.0, 0.2, 0.1))
                                   for o in rng.uniform(-0.4, 0.2, (3, 3))]),
        "circles": lambda: (pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.2),
                            [pg.Circle(0.2, tuple(c)) for c in rng.uniform(-0.3, 0.3, (3, 2))]),
        "mixed": lambda: (ball, [pg.Segment((-0.5, 0.0, 0.0), (0.5, 0.1, 0.0)), pg.Segment((0, -0.5, 0), (0, 0.5, 0.2))]
                          + pts),
        "interval": lambda: (pg.Sobolev("[0, 1]", 1, length_scale=0.1),
                             [pg.LebesgueMeasure(pg.Interval(a, a + 0.1)) for a in (0.1, 0.3, 0.5, 0.7)]),
        "reflections": lambda: (pg.Sobolev(pg.SobolevForm(pg.Box("[0, 1]", "[0, 0.7]"), 1, length_scale=0.15)
                                           .operator(), 2.5),
                                [pg.Segment((0.1, 0.1), (0.9, 0.6)), pg.Segment((0.2, 0.6), (0.8, 0.05)),
                                 pg.DiracMeasure((0.5, 0.3))]),    # segments alone: the torus of copies (plan 006)
    }[case]()
    A = H.riesz(measures).realisation.kx.atoms
    engine = H.backend.engine
    E = _exact(A, engine)
    V = rng.standard_normal((A.n, 2))
    with pg.settings.using(engine="spectral", dense_max=1):
        name, _, why = engine.route(A)
        Y = engine.apply(A, V)
        G = pg.Functionals(H.riesz(measures))
        claims = str((G @ G.adjoint).claims)
    assert name != "spectral" and "does not apply" in why and "does not apply" in claims, why
    assert A.spectral is None and _rel(Y, E @ V) <= 1e-8


def test_the_default_route_is_auto():
    """settings.engine = "auto" (the author's decision of 2026-10-27): spectral for segments in 3-D when finufft is
    installed; the H-matrix in 2-D and without finufft, with its usual claim; "hmatrix" and "exact" as named."""
    assert pg.settings.engine == "auto" and pg.settings.collinear is None
    assert "engine='auto', collinear=None" in repr(pg.settings)
    P = ladder.build("L3", 60, segments=8)
    A, engine = _atoms(P)
    E = _exact(A, engine)
    V = np.random.default_rng(7).standard_normal((A.n, 2))
    with pg.settings.using(dense_max=10):
        name, _, why = engine.route(A)
        assert name == ("spectral" if S._finufft() is not None else "hmatrix"), why
    with pg.settings.using(dense_max=10, engine="hmatrix"):
        assert engine.route(A)[0] == "hmatrix" and engine.describe(A) is None
        Y = engine.apply(A, V)
        G = pg.Functionals(P.H.riesz(P.measures), into=P.D)
        claims = str((G @ G.adjoint).claims)
    assert A.hmatrix is not None and _rel(Y, E @ V) <= 1e-7
    assert "H-matrix (near field exact" in claims and "settings.compression" in claims
    with pytest.raises(ValueError, match="settings.engine"):
        pg.settings.engine = "fmm"
    with pg.settings.using(engine="exact", dense_max=10):
        assert engine.route(A)[0] == "exact" and _rel(engine.apply(A, V), E @ V) <= 1e-12
    with pg.settings.using(dense_max=10, accuracy=1e-6):        # beyond the spectral engine's reach: the H-matrix
        name, _, why = engine.route(A)
        assert name == "hmatrix" and ("l Xi = 159" in why or "finufft" in why), why
    B, engine2 = _atoms(ladder.build("L2", 60))                  # 2-D: the H-matrix, and the claim is its usual one
    with pg.settings.using(dense_max=10):
        name, _, why = engine2.route(B)
        assert name == "hmatrix" and "2-D" in why and engine2.describe(B) is None


def test_auto_without_finufft_is_the_hmatrix(monkeypatch):
    monkeypatch.setattr(S, "_finufft", lambda: None)
    A, engine = _atoms(ladder.build("L3", 60, segments=8))
    with pg.settings.using(dense_max=10):
        name, _, why = engine.route(A)
    assert name == "hmatrix" and "finufft is not installed" in why


def test_the_memory_refusal_names_settings_collinear():
    A, engine = _atoms(ladder.build("L3", 30, segments=8))
    with pytest.raises(Unsupported, match="pg.settings.collinear = 0.3"):
        ST.close_pairs(A.P, A.P + A.E1, S.RCORR * engine.kernel.l, np.zeros((1, 3)), memory_gb=1e-6)


def test_unreachable_targets_are_refused():
    A, engine = _atoms(ladder.build("L3", 6, segments=4))
    with pg.settings.using(accuracy=1e-9):
        with pytest.raises(Unsupported, match="does not reach"):
            S.SpectralEngine(engine.exact).apply(A, np.ones(A.n))
    assert A.spectral is None


def test_transforms_refuse_to_run_in_a_forked_worker(monkeypatch):
    monkeypatch.setattr(execution, "_WORKER", True)
    with pytest.raises(Unsupported, match="parent process"):
        S._Transforms(np.zeros((3, 2)), (4, 4), 1e-6, 1)


def _pairs(a, b, k=None):
    return set(zip(a.tolist(), b.tolist()))


def test_close_pairs_in_blocks_and_chunks_are_the_brute_force_pairs(monkeypatch):
    """The search in blocks of whole segments and chunks of few sample pairs (as at large n) finds exactly the pairs of
    the distances of all segment pairs; with collinear = 0.3 the pairs of one measure and the pairs of different
    measures that meet (distance 0) or are at sin(angle) < 0.3, and no others."""
    A, engine = _atoms(ladder.build("L3", 30, segments=8))
    P, Qe, rc, zero = A.P, A.P + A.E1, 1.5 * engine.kernel.l, np.zeros((1, 3))
    a, b = np.triu_indices(len(P))
    dist = ST.distance(P[a], Qe[a], P[b], Qe[b])
    near = dist < rc
    touch = _pairs(a[dist <= ST.TOUCH * rc], b[dist <= ST.TOUCH * rc])
    brute = _pairs(a[near], b[near]) | {(i, i) for i in range(len(P))}
    full = _pairs(*ST.close_pairs(P, Qe, rc, zero))
    monkeypatch.setattr(ST, "SAMPLES", 37)
    monkeypatch.setattr(ST, "CHUNK", 50)
    assert _pairs(*ST.close_pairs(P, Qe, rc, zero)) == full == brute
    sel = _pairs(*ST.close_pairs(P, Qe, rc, zero, A.owner, 0.3))
    o = A.owner
    s2 = {(i, j): float(ST.sine2((Qe - P)[[i]], (Qe - P)[[j]])[0]) for i, j in full}
    cross = {x for x in full if o[x[0]] != o[x[1]]}
    assert len(cross) > 20 and len(full - sel) > 0
    assert sel == (full - cross) | {x for x in cross if s2[x] < 0.09 or x in touch}


def _brute_triples(P, Qe, rc, shifts):
    """Every (a, b, k), a <= b, with segment b translated by shifts[k] closer than rc, and the self pairs (a, a, 0)."""
    a, b = np.triu_indices(len(P))
    out = {(i, i, 0) for i in range(len(P))}
    for k, s in enumerate(shifts):
        near = ST.distance(P[a], Qe[a], P[b] + s, Qe[b] + s) < rc
        out |= set(zip(a[near].tolist(), b[near].tolist(), [k] * int(near.sum())))
    return out


@pytest.mark.parametrize("workers", [1, FORK_WORKERS])
def test_close_pairs_of_dense_clusters_and_torus_images_are_the_brute_force_pairs(monkeypatch, workers):
    """The search in spatial blocks (plan 005: the faster pair search) on sets where blocks are split and cells are
    small (clusters of short and long segments, a fan from one point, CHUNK small), and with the lattice shifts of a
    torus, against the distances of all segment pairs: the same pairs, each once."""
    rng = np.random.default_rng(5)
    P = np.vstack([rng.uniform(0, 1, (150, 3)), 0.5 + 0.02 * rng.standard_normal((150, 3)), np.full((60, 3), 0.3)])
    E = rng.standard_normal((360, 3)) * np.r_[np.full(150, 0.08), np.full(150, 0.01), np.full(60, 0.3)][:, None]
    Qe, rc = P + E, 0.03
    shifts = np.array([[0.0, 0.0, 0.0]] + [list(s) for s in np.eye(3)] + [list(-s) for s in np.eye(3)])
    monkeypatch.setattr(ST, "CHUNK", 400)
    with pg.settings.using(workers=workers):
        for sh in (shifts[:1], shifts):
            got = ST.close_pairs(P, Qe, rc, sh)
            triples = list(zip(*[x.tolist() for x in got]))
            assert len(triples) == len(set(triples)) and set(triples) == _brute_triples(P, Qe, rc, sh)
            assert np.all(np.diff((got[0] * len(P) + got[1]) * len(sh) + got[2]) > 0)        # sorted, as before


def test_close_pairs_beyond_the_memory_budget_are_refused():
    A, engine = _atoms(ladder.build("L3", 30, segments=8))
    with pytest.raises(Unsupported, match="settings.memory_gb = 1e-06: .* pairs .* grows like n\\^2"):
        ST.close_pairs(A.P, A.P + A.E1, S.RCORR * engine.kernel.l, np.zeros((1, 3)), memory_gb=1e-6)


@needs_finufft
@pytest.mark.parametrize("name", ["L3", "fan"])
def test_collinear_corrections_keep_the_estimate_above_the_error(name):
    """settings.collinear / SpectralEngine(collinear=): fewer corrected pairs between rays; the two-resolution estimate stays above the
    actual error, and the claims say which pairs were corrected."""
    A, engine = _atoms(ladder.build(name, 60, segments=8))
    E = _exact(A, engine)
    V = np.random.default_rng(8).standard_normal((A.n, 2))
    full = S.SpectralEngine(engine.exact, lxi=10.0)
    full.apply(A, V)
    n_full = A.spectral.stats["cross"]
    assert n_full > 0
    A.spectral = None
    eng = S.SpectralEngine(engine.exact, lxi=10.0, collinear=0.3)
    err = _rel(eng.apply(A, V), E @ V)
    assert A.spectral.stats["cross"] < n_full and err <= eng.estimate(A, V), (err, eng.estimate(A, V))
    assert "only those meeting or at sin(angle) < 0.3" in str(eng.claims(A))
    A.spectral = None
    with pg.settings.using(collinear=0.3):                     # the setting: the same selection, a plan keyed by it
        S.SpectralEngine(engine.exact, lxi=10.0).apply(A, V)
        assert A.spectral.smax == 0.3 and A.spectral.stats["cross"] < n_full
    with pytest.raises(ValueError, match="settings.collinear"):
        pg.settings.collinear = 2.0


FORK_SCRIPT = """
import sys
sys.path.insert(0, "tests")
import numpy as np
import ladder
import pygeoinf0 as pg
from pygeoinf0.realisations.rkhs import spectral as S, spectral_terms as ST
pg.settings.workers = WORKERS
done = []
for seed in (0, 1, 2, 3):
    P = ladder.build("L3", 40, seed=seed, segments=8)
    A = P.H.riesz(P.measures).realisation.kx.atoms
    eng = S.SpectralEngine(P.H.backend.engine.exact, lxi=6.0)
    y = eng.apply(A, np.ones(A.n))          # pair search and corrections in the fork pool, transforms here
    assert np.all(np.isfinite(y)) and len(ST._blocks(A.spectral.stats["pairs"])) > 1
    assert A.spectral.used.far["tr"].nthreads == WORKERS
    done.append((A, eng))
for A, eng in done:                         # finufft again in the parent, after every pool
    assert np.all(np.isfinite(eng.apply(A, np.ones(A.n))))
print("ok")
"""


@needs_finufft
@pytest.mark.skipif(FORK_WORKERS < 2, reason="needs worker processes: PYGEOINF0_WORKERS=2..4 (europa, slot free)")
def test_corrections_in_forked_workers_alternate_with_finufft_in_the_parent():
    """Caveat 6: finufft bundles libgomp, and a forked child of a process that started OpenMP threads can hang if it
    uses OpenMP. Corrections (fork pool) and products (finufft, threads = workers) alternate over four atom sets; the
    run must finish (timeout 300 s; the process group is killed on a hang)."""
    p = subprocess.Popen([sys.executable, "-c", FORK_SCRIPT.replace("WORKERS", str(FORK_WORKERS))], cwd=SRC,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        pytest.fail("the fork pool and finufft hung (timeout 300 s)")
    assert p.returncode == 0 and out.strip().endswith("ok"), err[-2000:]
