"""Where the time and memory go after the Gram matrix (plan 005, P0): one ladder problem, each stage timed and
profiled (cProfile, top functions by own time), with the resident memory after it.

    PYGEOINF0_WORKERS=30 PYTHONPATH=. python benchmarks/profile_stages.py L3 10000 --segments 4
"""
import argparse
import cProfile
import io
import os
import pstats
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import ladder  # noqa: E402

import pygeoinf0 as pg  # noqa: E402


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 2 ** 20
    return float("nan")


def stage(name, fn, top=12):
    pr = cProfile.Profile()
    t = time.perf_counter()
    pr.enable()
    out = fn()
    pr.disable()
    dt = time.perf_counter() - t
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top)
    lines = [ln for ln in s.getvalue().splitlines() if ln.strip()][3:3 + top + 2]
    print(f"\n=== {name}: {dt:.2f} s, resident {rss_gb():.2f} GB (peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f} GB)")
    print("\n".join(lines), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("n", type=int)
    ap.add_argument("--segments", type=int, default=8)
    a = ap.parse_args()
    kw = {"segments": a.segments} if a.problem == "L3" else {}
    P = stage("build problem", lambda: ladder.build(a.problem, a.n, **kw))
    K = stage("riesz", lambda: P.H.riesz(P.measures))
    G = stage("Functionals (norms: the Gram energies)", lambda: pg.Functionals(K, into=P.D))
    stage("G.gram()", G.gram)
    A = stage("G @ G.adjoint", lambda: G @ G.adjoint)
    alpha = stage("discrepancy", lambda: pg.solvers.damping_for_misfit(A, P.d, P.D.dim))
    post = stage("condition", lambda: pg.probability.cylindrical(P.H, alpha=alpha).condition(G, P.d))
    m = stage("post.mean", lambda: post.mean)
    stage("residual G(m) - d", lambda: P.D.norm(G(m) - P.d))
    stage("post.std of 20 data functionals", lambda: post.std(P.measures[:20]))


if __name__ == "__main__":
    main()
