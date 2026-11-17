"""The vectorised BGP (optimisers/dual/spectral.py; plan 009 P4a speed): the joint Newton route on the dual g(mu, omega)
against the per-direction solver (optimisers/dual/spectral.SpectralBall.bound) and the bracketing route, on problems whose
whitened eigenvalues span 1e-6 to 5e9 (the CMB problem's range); every bound stays EXACT (gap <= TOL)."""
import math

import numpy as np
import pytest
from scipy.stats import chi2

from pygeoinf0.core.precision import TOL
from pygeoinf0.optimisers.dual import SpectralBall
from pygeoinf0.optimisers.dual import spectral as SM


class _Dec:
    """The parts of a data-space decomposition that SpectralBall reads."""

    def __init__(self, lam, delta):
        self.lam, self._delta = lam, delta

    def _coef(self, e):
        return self._delta

    def positive(self, rcond):
        return self.lam > rcond * self.lam.max()

    def zero_count(self, rcond):
        pos = self.positive(rcond)
        return int((~pos).sum()), int((pos & ~self.positive(np.finfo(float).eps * len(self.lam))).sum()), len(pos)

    def describe(self):
        return "test spectrum"


def problem(n=400, lam=(1e-6, 5e9), nu=2.0, a=1.0, a_true=0.03, seed=0):
    rng = np.random.default_rng(seed)
    lam = np.exp(rng.uniform(math.log(lam[0]), math.log(lam[1]), n))
    y = rng.normal(size=n)
    y *= a_true / np.linalg.norm(y)
    delta = np.sqrt(lam) * y + nu * rng.normal(size=n)
    return SpectralBall(_Dec(lam, delta), None, a, nu * math.sqrt(chi2.ppf(0.95, df=n)))


def directions(sp, J=150, perp=0.05, seed=1):
    rng = np.random.default_rng(seed)
    n = len(sp.s)
    B = np.zeros((len(sp.pos), J))
    B[sp.pos] = sp.s[:, None] * rng.normal(size=(n, J)) / math.sqrt(n)
    a2 = ((B[sp.pos] / sp.s[:, None]) ** 2).sum(axis=0)
    return B, a2 * (1.0 + perp), rng.normal(size=J)


def one_by_one(sp, B, cc, const):
    return [sp.bound(B[:, j], float(cc[j]), float(const[j])) for j in range(B.shape[1])]


@pytest.mark.parametrize("perp", [0.05, 100.0])                    # small and dominant unseen parts
@pytest.mark.parametrize("a, nu", [(1.0, 2.0), (2.0, 1.0)])
def test_joint_route_equals_the_per_direction_solver(perp, a, nu):
    sp = problem(a=a, nu=nu)
    B, cc, const = directions(sp, perp=perp)
    value, lower, upper, status, redo = SM.bounds(sp, B, cc, const)
    assert redo == 0 and np.all(status == "EXACT")
    assert np.all(upper - lower <= TOL * np.maximum(1.0, np.abs(value)))
    ref = one_by_one(sp, B, cc, const)
    assert all(b.status == "EXACT" for b in ref)
    rv = np.array([b.value for b in ref])
    assert np.abs(value - rv).max() <= 2 * TOL * np.maximum(1.0, np.abs(rv)).max()


def test_joint_and_bracketing_routes_agree():
    sp = problem(seed=3)
    B, cc, _ = directions(sp, J=40, seed=4)
    a_pos = B[sp.pos] / sp.s[:, None]
    ab = np.vstack([a_pos, np.sqrt(np.maximum(cc - (a_pos ** 2).sum(axis=0), 0.0))[None, :]])
    yb, ob = SM._outer(sp, ab)
    _, mb = SM._solve(sp, ab, ob)
    y, om, mu, conv = SM._joint(sp, ab, np.full(40, np.median(mb) * 3.0), np.full(40, np.median(ob) / 5.0))
    assert conv.all()
    assert np.allclose(om, ob, rtol=1e-9) and np.allclose(mu, mb, rtol=1e-9)
    assert np.allclose((ab * y).sum(axis=0), (ab * yb).sum(axis=0), rtol=1e-12)


def test_unconverged_columns_take_the_bracketing_route(monkeypatch):
    sp = problem(seed=5)
    B, cc, const = directions(sp, J=30, seed=6)
    monkeypatch.setattr(SM, "NEWTON", 0)                          # no joint steps at all: every column falls back
    joint = SM._joint
    monkeypatch.setattr(SM, "_joint", lambda sp_, ab, mu, om, steps=0: joint(sp_, ab, mu, om, steps=0))
    value, lower, upper, status, redo = SM.bounds(sp, B, cc, const)
    assert redo == 0 and np.all(status == "EXACT")
    rv = np.array([b.value for b in one_by_one(sp, B, cc, const)])
    assert np.abs(value - rv).max() <= 2 * TOL * np.maximum(1.0, np.abs(rv)).max()


def test_prior_decided_zero_and_empty_cases():
    sp = problem(seed=7, a=1.0)
    B, cc, const = directions(sp, J=6, seed=8)
    B[:, 0] = 0.0                                                   # the zero direction
    cc[0] = 0.0
    B[:, 1] = 0.0                                                   # a direction the data do not see (only a_perp)
    cc[1] = 4.0
    value, lower, upper, status, redo = SM.bounds(sp, B, cc, const)
    assert np.all(status == "EXACT")
    ref = np.array([b.value for b in one_by_one(sp, B, cc, const)])
    assert value[0] == pytest.approx(const[0])
    assert np.abs(value - ref).max() <= 2 * TOL * np.maximum(1.0, np.abs(ref)).max()
    assert value[1] < const[1] + 2.0 * sp.r                          # part of the prior's budget fits the data
    far = problem(seed=7, a=1e-6)                                   # a prior too small for the data: empty
    v2, _, _, st2, _ = SM.bounds(far, B, cc, const)
    assert all(s == "EMPTY" for s in st2)


def test_chunks_start_from_their_predecessors(monkeypatch):
    sp = problem(seed=9)
    B, cc, const = directions(sp, J=3 * SM.CHUNK + 5, seed=10)
    starts = []
    joint = SM._joint
    monkeypatch.setattr(SM, "_joint", lambda sp_, ab, mu, om, steps=SM.NEWTON: (starts.append((mu[0], om[0])),
                                                                               joint(sp_, ab, mu, om, steps))[1])
    value, lower, upper, status, redo = SM.bounds(sp, B, cc, const)
    assert redo == 0 and np.all(status == "EXACT") and len(starts) == 4
    assert all(abs(math.log(s[1] / starts[0][1])) < 1.0 for s in starts[1:])   # neighbouring multipliers


def test_worker_spans_give_the_serial_bounds():
    import pygeoinf0 as pg
    sp = problem(seed=11)
    B, cc, const = directions(sp, J=12 * SM.CHUNK + 9, seed=12)
    serial = SM.bounds(sp, B, cc, const)
    with pg.settings.using(workers=3):
        spans = SM.bounds(sp, B, cc, const)
    assert spans[4] == 0 and np.all(spans[3] == "EXACT")
    for k in range(3):                                            # value, lower, upper: same certified bounds
        assert np.allclose(spans[k], serial[k], rtol=2 * TOL, atol=2 * TOL)


@pytest.mark.parametrize("workers", [1, 3])
def test_one_pool_serves_every_batch_and_sign(monkeypatch, workers):
    """factored.intervals' pattern: batches of directions, both signs, through one Directions pool, bit for bit the
    separate bounds calls (one pool forked per call), with a single fork of worker processes."""
    import multiprocessing as mp
    import pygeoinf0 as pg
    sp = problem(seed=13)
    B, cc, const = directions(sp, J=3 * (2 * SM.SPAN * SM.CHUNK) + 77, seed=14)
    batch = 2 * SM.SPAN * SM.CHUNK + 40                            # several spans per batch, a short last batch
    with pg.settings.using(workers=workers):
        ref = []
        for s in range(0, B.shape[1], batch):
            sl = slice(s, s + batch)
            ref.append((SM.bounds(sp, B[:, sl], cc[sl], const[sl]), SM.bounds(sp, -B[:, sl], cc[sl], -const[sl])))
        forks = []
        real = mp.get_context
        monkeypatch.setattr(mp, "get_context", lambda m=None: (forks.append(m), real(m))[1])
        with SM.Directions(sp, B.shape[0], batch) as pool:
            new = [SM.signed_bounds(sp, B[:, sl], cc[sl], const[sl], directions=pool)
                   for sl in (slice(s, s + batch) for s in range(0, B.shape[1], batch))]
    assert forks == ([] if workers == 1 else ["fork"])
    for (rp, rm), (np_, nm) in zip(ref, new):
        for r, n in ((rp, np_), (rm, nm)):
            assert r[4] == n[4]
            for k in range(4):
                assert np.array_equal(r[k], n[k])


def test_intervals_serial_bypass_and_pool_route_agree(monkeypatch):
    """factored.intervals: with one worker the serial route (no Directions pool, no shared buffer), with several the
    pool route; the two give the same bounds and statuses bit for bit, over several batches and both signs."""
    import pygeoinf0 as pg
    from pygeoinf0.solvers.dataspace import factored
    from test_families import problem as family_problem
    Sp, G, noise, T, d, n = family_problem(p=300)
    monkeypatch.setattr(factored, "BATCH", 128)                   # several batches, a short last one
    dec = pg.decompose(G, noise)
    a, r = 5.0, 1.5 * np.sqrt(n)
    made = []
    real = SM.Directions
    monkeypatch.setattr(SM, "Directions", lambda *x, **k: (made.append(1), real(*x, **k))[1])
    with pg.settings.using(workers=1):
        serial = dec.ellipsoid_intervals(T, a=a, r=r, datum=d)
    assert made == []                                              # the bypass: no pool, no shared buffer
    with pg.settings.using(workers=3):
        pooled = dec.ellipsoid_intervals(T, a=a, r=r, datum=d)
    assert made == [1]
    for k in ("lower", "upper"):
        assert np.array_equal(getattr(serial, k), getattr(pooled, k))
    for k in ("status_lower", "status_upper"):
        assert list(getattr(serial, k)) == list(getattr(pooled, k))
    assert set(serial.status_upper) | set(serial.status_lower) == {"EXACT"}
