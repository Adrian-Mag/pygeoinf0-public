"""The scale ladder of plan 005: four reference problems at any number of data n, and the one user code that solves
them all (solve). Problems are deterministic (seeded); data are synthetic, d = int truth dmu_i + noise, with the noise
standard deviation a fixed fraction of the rms of the clean data; the truth is a numpy function of the coordinates.

    L1 interval   H^1([0, 1]), l = 0.1             averages over random subintervals
    L2 disk       H^{3/2}(unit disk), l = 0.1      straight rays (chords between boundary points)
    L3 ball       H^2(unit ball), l = 0.08         curved rays: polylines of k segments bottoming inside the ball, with
                                                   a positive density per segment (a travel-time weight, PieceDensity)
    L4 sphere     H^2(unit sphere), l = 0.2        great-circle arcs
    fan           as L3, with shared endpoints     rays between 20 sources and 50 stations: rays leaving one source are
                                                   nearly collinear near it (EXP-025's bench3d.fan; the spectral engine
                                                   needs its corrections between measures there)

    P = ladder.build("L3", 1000)
    G, alpha, post = ladder.solve(P)              # the same lines at every n (plan 005, invariant 1)
"""
import collections
import math

import numpy as np

import pygeoinf0 as pg
from pygeoinf0 import quadrature
from pygeoinf0.quadrature import gauss

Problem = collections.namedtuple("Problem", "name n H measures D d sigma truth")
NOISE = 0.05                    # noise standard deviation / rms of the clean data


def solve(P):
    """The user code of the ladder: build the data operator, choose alpha by the discrepancy principle, condition the
    prior. Nothing here depends on the size of the problem."""
    G = pg.Functionals(P.H.riesz(P.measures), into=P.D)
    alpha = pg.meta.discrepancy(pg.Problem(mdr=G, datum=P.d)).gamma
    post = pg.probability.cylindrical(P.H, alpha=alpha).condition(G, P.d)
    return G, alpha, post


def build(name, n, seed=0, **kw):
    return {"L1": interval, "L2": disk, "L3": ball, "L4": sphere, "fan": fan}[name](n, seed=seed, **kw)


# ------------------------------------------------------------------------------------------ the four problems

def interval(n, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.02, 0.1, n)
    a = rng.uniform(0, 1 - w)
    J = [pg.Interval(float(x), float(x + y)) for x, y in zip(a, w)]

    def truth(x):
        return np.sin(2 * np.pi * x) + 0.5 * np.cos(5 * np.pi * x)
    clean = np.array([_segment_integral(truth, np.array([j.a]), np.array([j.b])) for j in J])
    return _finish("L1", pg.Sobolev("[0, 1]", 1, length_scale=0.1), [pg.LebesgueMeasure(j) for j in J], clean, truth,
                   rng)


def disk(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    b = a + rng.uniform(0.3, 2 * np.pi - 0.3, n)
    P, Q = np.c_[np.cos(a), np.sin(a)], np.c_[np.cos(b), np.sin(b)]

    def truth(x, y):
        return np.exp(-((x - 0.3) ** 2 + y ** 2) / 0.05) - 0.7 * np.exp(-((x + 0.4) ** 2 + (y - 0.3) ** 2) / 0.02)
    clean = np.array([_segment_integral(truth, p, q) for p, q in zip(P, Q)])
    rays = [pg.Segment(tuple(p), tuple(q)) for p, q in zip(P, Q)]
    return _finish("L2", pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.1), rays, clean, truth, rng)


def _ball_truth(x, y, z):
    return (np.exp(-((x - 0.3) ** 2 + y ** 2 + (z - 0.2) ** 2) / 0.04)
            - 0.8 * np.exp(-((x + 0.2) ** 2 + (y + 0.4) ** 2 + z ** 2) / 0.02) + 0.3 * z * (x * x + y * y + z * z))


def _curved_ray(p, u, D, segments):
    """The polyline from p (unit) over the angular distance D towards u (unit, orthogonal to p), bottoming inside the
    ball, with a positive weight per segment (slowness-like), and its clean datum."""
    th = np.linspace(0, D, segments + 1)
    rmin = 1 - 0.6 * (1 - math.cos(D / 2))                          # shallower than the chord, inside the ball
    r = 1 - (1 - rmin) * np.sin(np.pi * th / D)
    X = r[:, None] * (np.cos(th)[:, None] * p + np.sin(th)[:, None] * u)
    X[[0, -1]] /= np.linalg.norm(X[[0, -1]], axis=1)[:, None]       # the ends exactly on the sphere
    mid = 0.5 * np.linalg.norm(X[1:] + X[:-1], axis=1)
    w = 1.0 / (1.0 + mid) ** 2
    clean = sum(wj * _segment_integral(_ball_truth, X[j], X[j + 1]) for j, wj in enumerate(w))
    return pg.LebesgueMeasure(pg.Polyline(X)).with_density(w), clean


def ball(n, seed=0, segments=8):
    rng = np.random.default_rng(seed)
    measures, clean = [], []
    for _ in range(n):
        p = _unit(rng.normal(size=3))
        v = rng.normal(size=3)
        u = _unit(v - (v @ p) * p)                                  # a unit vector orthogonal to p
        D = rng.uniform(np.radians(30), np.radians(150))            # angular distance between the two ends
        mu, c = _curved_ray(p, u, D, segments)
        measures.append(mu)
        clean.append(c)
    return _finish("L3", pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.08), measures, np.array(clean),
                   _ball_truth, rng)


def fan(n, seed=0, segments=8, sources=20, stations=50):
    rng = np.random.default_rng(seed)
    src = [_unit(rng.normal(size=3)) for _ in range(sources)]
    sta = [_unit(rng.normal(size=3)) for _ in range(stations)]
    measures, clean = [], []
    while len(measures) < n:
        p, q = src[rng.integers(sources)], sta[rng.integers(stations)]
        D = math.acos(np.clip(p @ q, -1, 1))
        if not np.radians(30) < D < np.radians(150):
            continue
        mu, c = _curved_ray(p, _unit(q - (q @ p) * p), D, segments)
        measures.append(mu)
        clean.append(c)
    return _finish("fan", pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=0.08), measures, np.array(clean),
                   _ball_truth, rng)


def sphere(n, seed=0):
    rng = np.random.default_rng(seed)
    S = pg.Sphere(1.0)

    def truth(x, y, z):
        return z + 0.5 * x * y + np.exp(-((x - 0.6) ** 2 + (y - 0.6) ** 2 + (z - 0.5) ** 2) / 0.05)
    arcs, clean = [], []
    while len(arcs) < n:
        p, q = _unit(rng.normal(size=3)), _unit(rng.normal(size=3))
        if not 0.2 < math.acos(np.clip(p @ q, -1, 1)) < 2.8:
            continue
        A = pg.Arc(S, tuple(p), tuple(q))
        X, wq = quadrature.rule(A, 32)
        arcs.append(A)
        clean.append(wq @ truth(*X.T))
    return _finish("L4", pg.Sobolev(S, 2, length_scale=0.2), arcs, np.array(clean), truth, rng)


# ------------------------------------------------------------------------------------------ helpers

def _unit(v):
    return v / np.linalg.norm(v)


def _segment_integral(f, p, q, n=32):
    """int f ds over the straight segment p -> q (arc length), by n-point Gauss-Legendre."""
    t, w = gauss(n)
    X = p + 0.5 * (t[:, None] + 1) * (q - p)
    return 0.5 * np.linalg.norm(q - p) * (w @ f(*X.T))


def _finish(name, H, measures, clean, truth, rng):
    sigma = NOISE * float(np.sqrt(np.mean(clean ** 2)))
    D = pg.Euclidean(len(measures), covariance=sigma ** 2)
    return Problem(name, len(measures), H, measures, D, D(clean + sigma * rng.standard_normal(len(clean))), sigma, truth)
