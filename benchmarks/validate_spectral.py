"""The spectral energy engine against the exact route on one ladder problem (plan 005 P4, the author's decision 5): the
measurements that decide whether settings.engine = "spectral" may become a size-based default. Run on europa.

    PYGEOINF0_WORKERS=30 PYTHONPATH=.:tests python benchmarks/validate_spectral.py L3 3000 --segments 8
    ... --no-solve          (products only: no end-to-end comparison)
    ... --collinear none 0.3   (stages 2 and 3 per value of pg.settings.collinear against one dense route: none corrects
                                every pair closer than r_corr, 0.3 those between rays only at sin(angle) < 0.3)

1. dense      the unchanged user code (ladder.solve) on the dense route: exact energies E, alpha, the posterior mean at
              400 fixed probe points (as benchmarks/check_matrix_free.py), predicted data
2. products   the spectral engine on the same atoms: setup (pairs, corrections, transforms; stats), time per product,
              relative errors against E on probe vectors next to the engine's estimate, and the predicted-data errors
              (E_a - E) c / sigma at the discrepancy coefficients c, rms and max, next to settings.accuracy and the
              engine's own two-resolution estimates
   rates      the actual errors at l Xi, l Xi / sqrt 2 and l Xi / 2: the observed convergence rate next to the derived
              p = 2s + 2 - d that the estimate assumes, and the rate from the differences alone (a third resolution)
3. solve      ladder.solve with settings.using(engine="spectral", dense_max=n - 1) (matrix-free: Lanczos, CG): alpha,
              predicted data and the posterior mean compared with the dense route at the same probe points

Every stage prints its wall time and peak memory (this process; the forked workers separately); the last line is a
JSON record of every number.
"""
import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import ladder  # noqa: E402

import pygeoinf0 as pg  # noqa: E402
from pygeoinf0.realisations.rkhs import spectral as S  # noqa: E402


def peak():
    """(this process, the largest worker) peak resident memory in GB."""
    return tuple(resource.getrusage(w).ru_maxrss / 2 ** 20 for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))


def stage(rec, name, t0):
    rec[f"t_{name}"], (rec[f"gb_{name}"], rec[f"gb_workers_{name}"]) = time.time() - t0, peak()
    print(f"{name:10s} {rec[f't_{name}']:9.1f} s   peak {rec[f'gb_{name}']:6.2f} GB (workers {rec[f'gb_workers_{name}']:.2f})",
          flush=True)


def coefficients(E, d, sigma):
    """c = (E + lam I)^-1 d at the discrepancy lam (||lam (E + lam I)^-1 d||^2 = n sigma^2)."""
    w, U = np.linalg.eigh(E)
    b = U.T @ d
    lo, hi = 1e-14 * w.max(), 1e3 * w.max()
    for _ in range(200):
        lam = math.sqrt(lo * hi)
        lo, hi = (lam, hi) if np.sum((lam / (w + lam) * b) ** 2) < len(d) * sigma ** 2 else (lo, lam)
    return U @ (b / (w + lam))


def solve(P, X):
    G, alpha, post = ladder.solve(P)
    return G, float(alpha), np.asarray(post.mean(*X.T), float).ravel(), np.asarray(G(post.mean).values, float)


def products(rec, P, A, E, c):
    """The spectral engine on the atoms A of the dense route, against their exact energies E; c: the discrepancy
    coefficients."""
    engine = P.H.backend.engine
    A.self_energy = None
    with pg.settings.using(engine="spectral"):
        name, spec, why = engine.route(A)
        if name != "spectral":
            raise SystemExit(f"the spectral engine does not apply: {why}")
        t0 = time.time()
        plan = spec.plan(A)
        stage(rec, "setup", t0)
        print(plan.claims(), {k: v for k, v in plan.stats.items()}, sep="\n", flush=True)
        V = np.random.default_rng(5).standard_normal((A.n, 2))
        t0 = time.time()
        Y = spec.apply(A, V)
        stage(rec, "products", t0)
        rec["t_per_product"] = rec["t_products"] / V.shape[1]
        EV = E @ V
        rec["rel"] = float(np.max(np.linalg.norm(Y - EV, axis=0) / np.linalg.norm(EV, axis=0)))
        rec["rel_estimate"], rec["rel_claimed"], rec["target"] = spec.estimate(A, V), plan.est, plan.tol
        D = (spec.apply(A, c) - E @ c) / P.sigma
        rec["data_rms"], rec["data_max"] = float(np.sqrt(np.mean(D ** 2))), float(np.max(np.abs(D)))
        rec["data_rms_estimate"], rec["data_max_estimate"] = spec.estimate_data(A, c, P.sigma)
        rec["accuracy"] = pg.settings.accuracy
        t0 = time.time()                                      # the rate: actual errors at l Xi, l Xi / sqrt 2, l Xi / 2
        levels = [plan.used] + [plan.level(plan.used.lxi / f) for f in (S.SQ2, 2.0)]
        Ys = [plan.product(L, V) for L in levels]
        for L in levels[1:]:
            plan._drop(L)
        errs = [float(np.max(np.linalg.norm(Yl - EV, axis=0) / np.linalg.norm(EV, axis=0))) for Yl in Ys]
        diffs = [float(np.max(np.linalg.norm(Ys[i] - Ys[i + 1], axis=0))) for i in (0, 1)]
        rec["rate_derived"] = S.rate(plan.kernel)
        rec["rel_levels"] = errs
        rec["rate_actual"] = [math.log(errs[i + 1] / errs[i]) / math.log(S.SQ2) for i in (0, 1)]
        rec["rate_differences"] = math.log(diffs[1] / diffs[0]) / math.log(S.SQ2)
        stage(rec, "rates", t0)
        print(f"rates: derived p = {rec['rate_derived']:g}; actual {rec['rate_actual'][0]:.2f} (l Xi / sqrt 2 -> l Xi), "
              f"{rec['rate_actual'][1]:.2f} (l Xi / 2 -> l Xi / sqrt 2); from differences alone "
              f"{rec['rate_differences']:.2f}; errors {', '.join(f'{e:.2e}' for e in errs)}", flush=True)
        rec["stats"] = {k: (list(map(int, v)) if isinstance(v, tuple) else v) for k, v in plan.stats.items()}
    print(f"products: relative error {rec['rel']:.2e} (estimate {rec['rel_estimate']:.2e}, claimed {rec['rel_claimed']:.2e}, "
          f"target {rec['target']:.1e}); predicted data error rms {rec['data_rms']:.2e} sigma, max {rec['data_max']:.2e} "
          f"sigma (estimates {rec['data_rms_estimate']:.2e}, {rec['data_max_estimate']:.2e}; accuracy "
          f"{rec['accuracy']:g})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("n", type=int)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--no-solve", action="store_true")
    ap.add_argument("--collinear", nargs="+", default=["none"])
    a = ap.parse_args()
    print(pg.settings, flush=True)
    P = ladder.build(a.problem, a.n, **({"segments": a.segments} if a.problem in ("L3", "fan") else {}))
    rec = dict(problem=a.problem, n=P.n, segments=a.segments, settings=repr(pg.settings))
    dim = P.H.domain.cartesian_dim
    X = np.random.default_rng(11).uniform(-0.9, 0.9, (400, dim)) / math.sqrt(dim)
    t0 = time.time()
    G, a0, m0, d0 = solve(P, X)
    stage(rec, "dense", t0)
    A = G.kernels.realisation.kx.atoms
    E = P.H.backend.engine.energy(A, A)                       # cached by the dense route
    t0 = time.time()
    c = coefficients(E, np.asarray(P.d.values, float), P.sigma)
    stage(rec, "eigh", t0)
    for v in a.collinear:
        pg.settings.collinear = None if v == "none" else float(v)
        r = rec if len(a.collinear) == 1 else rec.setdefault("collinear", {}).setdefault(v, {})
        r["collinear"] = pg.settings.collinear
        print(f"--- pg.settings.collinear = {pg.settings.collinear}", flush=True)
        products(r, P, A, E, c)
        if a.no_solve:
            continue
        t0 = time.time()
        with pg.settings.using(engine="spectral", dense_max=P.n - 1):
            G1, a1, m1, d1 = solve(P, X)
            print((G1 @ G1.adjoint).claims, flush=True)
        stage(r, "solve", t0)
        std = math.sqrt(P.H.backend.kernel.k0 / a0)
        r.update(alpha_rel=abs(a1 / a0 - 1), pred_rms=float(np.sqrt(np.mean((d1 - d0) ** 2)) / P.sigma),
                 pred_max=float(np.max(np.abs(d1 - d0)) / P.sigma), mean_max=float(np.max(np.abs(m1 - m0)) / std))
        print(f"solve vs dense: alpha {r['alpha_rel']:.2e} relative; predicted data rms {r['pred_rms']:.2e} sigma, "
              f"max {r['pred_max']:.2e} sigma; posterior mean max {r['mean_max']:.2e} prior point std", flush=True)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
