"""Plan 005 P6: examples/global_tomography.py at 10^4, 10^5 (10^6) data, costs per stage and the posterior std at scale.

    N_DATA=100000 PYGEOINF0_WORKERS=12 python benchmarks/global_scale.py OUT.json [--full] [--points 200] [--check 20]
        [--collinear 0.3]

At 1e5 rays the spectral engine's exact corrections on all close segment pairs exceed settings.memory_gb (about 1e9
pairs, refused); --collinear sets pg.settings.collinear (only pairs of different rays that meet or are nearly collinear
are corrected; the others are left to the far field, with the engine's estimate), as the 1e5 scale validation did.

Runs the example's own code (read from examples/global_tomography.py, not copied) in two parts: the inverse problem
(rays, synthetic data, representers, discrepancy, posterior mean; the example's stage timings), then
- the posterior standard deviation of dVp at `points` points of the example's sections through the user call
  post.std(points) (P5's route above settings.dense_max, exact below), its claims, and, above dense_max, the std of
  `check` of them by the exact route (deflated block CG on the same operator) for comparison;
- with --full, the example's sections and plots unchanged; without, the evaluation of the solution and of the total
  sensitivity timed at 1000 of the section points and extrapolated linearly to all of them (the cost is points x
  segments; stated as an extrapolation).
The posterior is the example's own (GaussianMeasure.condition is wrapped to keep it; nothing is solved twice).
"""
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(HERE, "examples", "global_tomography.py")
MARK = "# ---------------------------------------------------------------- sections and maps"
VALS = "vals = {}"


def main():
    out = sys.argv[1]
    full = "--full" in sys.argv
    p = int(sys.argv[sys.argv.index("--points") + 1]) if "--points" in sys.argv else 200
    check = int(sys.argv[sys.argv.index("--check") + 1]) if "--check" in sys.argv else 20
    import pygeoinf0 as pg
    from pygeoinf0.probability.gaussian import GaussianMeasure
    if "--collinear" in sys.argv:                               # an optional setting (invariant 1): which close pairs
        pg.settings.collinear = float(sys.argv[sys.argv.index("--collinear") + 1])   # the spectral engine corrects
    posts, cond = [], GaussianMeasure.condition

    def keep(self, *a, **k):
        post = cond(self, *a, **k)
        posts.append(post)
        return post
    GaussianMeasure.condition = keep
    src = open(EXAMPLE).read()
    head, tail = src.split(MARK)
    ns = sys.modules["__main__"].__dict__                       # the example's pool pickles its functions by name
    exec(compile(head, EXAMPLE, "exec"), ns)
    post, n, costs, lap = posts[-1], ns["n"], ns["costs"], ns["lap"]
    report = {"n_data": n, "segments": int(len(ns["W"])), "workers": pg.settings.workers,
              "dense_max": pg.settings.dense_max, "engine": pg.settings.engine, "collinear": pg.settings.collinear, "alpha": float(ns["alpha"]),
              "chi2_per_datum": float(ns["chi2"] / n), "solver_claims": str(getattr(post.solver, "claims", ""))[:600]}
    defs, _ = tail.split(VALS)
    exec(compile(defs, EXAMPLE, "exec"), ns)                   # plane, depth_map, SECTIONS, total (cheap)
    rng = np.random.default_rng(5)
    allpts = np.vstack([pts for pts, _ in ns["SECTIONS"].values()])
    pts = allpts[rng.choice(len(allpts), p, replace=False)]
    lap("(P6) re-entry")
    s = post.std(pts)
    lap("(P6) posterior std at points")
    claims = getattr(post, "variance_claims", None)
    report["std"] = {"points": p, "min": float(s.min()), "median": float(np.median(s)), "max": float(s.max()),
                     "claims": [str(c) for c in claims][:6] if isinstance(claims, list) else str(claims)}
    sc = getattr(post, "_scale", None)
    if sc is not None and n > pg.settings.dense_max:
        report["std"].update(rank=int(len(sc.dec.lam)), norm_E=float(sc.dec.error),
                             max_std_bound_over_std=float(np.max((np.sqrt(s ** 2 + sc.last_bound) - s) / s)))
        ref = post.std(pts[:check], exact=True)
        lap("(P6) exact std check (block CG)")
        report["std"]["max_rel_vs_exact_cg"] = float(np.max(np.abs(s[:check] - ref) / ref))
        report["std"]["inside_bound"] = int(np.sum(np.abs(ref ** 2 - s[:check] ** 2) <= sc.last_bound[:check] * (1 + 1e-9)
                                                   + 1e-11 * ref ** 2))
    if full:
        exec(compile(VALS + tail.split(VALS, 1)[1], EXAMPLE, "exec"), ns)
    else:
        sub = allpts[rng.choice(len(allpts), 1000, replace=False)]
        t0 = time.time()
        ns["m"](*sub.T), ns["total"](*sub.T)
        t = time.time() - t0
        report["evaluation_extrapolated"] = {"points_timed": 1000, "seconds": t, "section_points": len(allpts),
                                             "estimated_seconds_all_sections": t * len(allpts) / 1000}
        lap("(P6) evaluation sample")
    report["costs"] = costs
    report["max_rss_gb"] = dict(main=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                                workers=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20)
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
