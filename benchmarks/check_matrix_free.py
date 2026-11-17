"""The dense and the matrix-free routes on one ladder problem: alpha from each route's discrepancy principle, and the
posterior mean at probe points (in units of the prior point std) with each route's own alpha and with a common alpha,
so that a difference of alpha is told apart from a difference of the solve (plan 005, P2 check).

    PYTHONPATH=.:tests python benchmarks/check_matrix_free.py L3 100 --segments 8 --dense-max 50
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import ladder  # noqa: E402

import pygeoinf0 as pg  # noqa: E402


def route(P, X, dense_max, alpha=None):
    with pg.settings.using(dense_max=dense_max):
        G = pg.Functionals(P.H.riesz(P.measures), into=P.D)
        A = G @ G.adjoint
        a = pg.solvers.damping_for_misfit(A, P.d, P.D.dim) if alpha is None else alpha
        post = pg.probability.cylindrical(P.H, alpha=float(a)).condition(G, P.d)
        mean = np.asarray(post.mean(*X.T), float).ravel()
        pred = np.asarray(G(post.mean), float).ravel()
        r = float(np.sum((pred - np.asarray(P.d, float)) ** 2)) / P.sigma ** 2 / P.n
        print(f"  dense_max {dense_max:6d}: alpha {float(a):.8g}, chi^2/n {r:.6f}", flush=True)
        solver = getattr(post, "solver", None)
        for c in (solver.claims if solver is not None else []):
            print("   ", c, flush=True)
    return float(a), mean, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("n", type=int)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--dense-max", type=int, default=50)
    a = ap.parse_args()
    P = ladder.build(a.problem, a.n, **({"segments": a.segments} if a.problem == "L3" else {}))
    rng = np.random.default_rng(11)
    dim = P.H.domain.cartesian_dim
    X = rng.uniform(-0.9, 0.9, (400, dim)) / math.sqrt(dim)
    print("own alpha per route", flush=True)
    ad, md, pd = route(P, X, 10 ** 6)
    am, mm, pm = route(P, X, a.dense_max)
    print("common alpha (the dense route's)", flush=True)
    _, mc, pc = route(P, X, a.dense_max, alpha=ad)
    k0 = P.H.backend.kernel.k0
    for label, alpha, m, p in (("own alpha", am, mm, pm), ("common alpha", ad, mc, pc)):
        std = math.sqrt(k0 / ad)                         # prior point std of the dense route's prior
        print(f"{label:13s}: max |mean - dense mean| = {np.max(np.abs(m - md)):.3e} ({np.max(np.abs(m - md)) / std:.3f}"
              f" prior std); max |mean| = {np.max(np.abs(md)):.3e}; max |pred - dense pred| / sigma = "
              f"{np.max(np.abs(p - pd)) / P.sigma:.3e}; alpha ratio {alpha / ad:.6f}", flush=True)


if __name__ == "__main__":
    main()
