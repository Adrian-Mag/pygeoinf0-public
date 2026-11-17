"""Stage timings of the ladder problems (plan 005, invariant 5): median wall seconds over repeats, as JSON on stdout.

    python benchmarks/run.py                       # default cases, 3 repeats
    python benchmarks/run.py --cases L1/100 L3/300 --repeat 5
    OPENBLAS_NUM_THREADS=8 python benchmarks/run.py --check baseline.json   # compare with a baseline (recorded with
                                                   # 8 BLAS threads): exit 1 if a stage is more than 20 %
                                                   # (and 0.05 s) slower

Stages: riesz (representers), gram (pg.Functionals with its Gram), normal (G @ G.adjoint), discrepancy, condition
(posterior mean), std (posterior standard deviations of 50 point values, or of 50 data functionals where points are not
bounded), evaluate (the mean at 2000 points). Every repeat builds fresh objects from the same problem, so no cache is
shared between repeats. Run on europa (timings on the laptop are neither safe nor meaningful).
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import ladder  # noqa: E402

import pygeoinf0 as pg  # noqa: E402

DEFAULT = ["L1/100", "L1/1000", "L1/4000", "L2/100", "L2/1000", "L3/100", "L3/300", "L4/100", "L4/300"]
SLOWER, FLOOR = 1.2, 0.05


def points(P, m, seed=1):
    rng = np.random.default_rng(seed)
    if P.name == "L1":
        return rng.uniform(0, 1, m)
    if P.name == "L4":
        X = rng.normal(size=(m, 3))
        return X / np.linalg.norm(X, axis=1)[:, None]
    d = 2 if P.name == "L2" else 3
    X = rng.normal(size=(m, d))
    return 0.9 * X / np.linalg.norm(X, axis=1)[:, None] * rng.uniform(0, 1, (m, 1)) ** (1 / d)


def run_once(P):
    t, out = time.perf_counter(), {}

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        out[name] = now - t
        t = now
    K = P.H.riesz(P.measures)
    lap("riesz")
    G = pg.Functionals(K, into=P.D)
    G.gram()
    lap("gram")
    A = G @ G.adjoint
    lap("normal")
    alpha = pg.solvers.damping_for_misfit(A, P.d, P.D.dim)
    lap("discrepancy")
    post = pg.probability.cylindrical(P.H, alpha=alpha).condition(G, P.d)
    m = post.mean
    lap("condition")
    X = points(P, 2000)
    post.std(X[:50] if P.H.dual().integrals_bounded(0) else P.measures[:50])
    lap("std")
    m(X) if X.ndim == 1 else m(*X.T)
    lap("evaluate")
    out["total"] = sum(out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=DEFAULT)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--check")
    a = ap.parse_args()
    cases = {}
    for case in a.cases:
        name, n = case.split("/")
        P = ladder.build(name, int(n))
        runs = [run_once(P) for _ in range(a.repeat)]
        cases[case] = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        print(f"{case:10s} " + " ".join(f"{k} {v:.3f}" for k, v in cases[case].items()), file=sys.stderr, flush=True)
    meta = dict(git=os.environ.get("GIT_SHA", "?"), host=platform.node(), python=platform.python_version(),
                numpy=np.__version__, load=os.getloadavg(), workers=os.environ.get("PYGEOINF0_WORKERS", "1"),
                blas_threads=os.environ.get("OPENBLAS_NUM_THREADS", "?"), repeat=a.repeat,
                date=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(dict(meta=meta, cases=cases), indent=1))
    if a.check:
        baseline = json.load(open(a.check))
        base, bmeta = baseline["cases"], baseline.get("meta", {})
        for key in ("host", "workers", "blas_threads"):    # dense LAPACK stages scale with the BLAS threads
            if key in bmeta and str(bmeta[key]) != str(meta[key]):
                print(f"WARNING: {key} = {meta[key]} here, {bmeta[key]} in the baseline: timings are not comparable",
                      file=sys.stderr)
        bad = [(c, k, base[c][k], v) for c, r in cases.items() if c in base for k, v in r.items()
               if v > SLOWER * base[c][k] and v - base[c][k] > FLOOR]
        for c, k, b, v in bad:
            print(f"SLOWER {c} {k}: {v:.3f} s against {b:.3f} s", file=sys.stderr)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
