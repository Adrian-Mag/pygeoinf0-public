"""Plan 005 P5 acceptance: posterior standard deviations at scale against the exact route (plan 005, P5: "agreement with
exact variances at 10^3-10^4 within the estimate; claims state the approximation").

    python benchmarks/posterior_scale.py [--sizes 1000 10000] [--problems L1 L2 L3 L4] [--functionals 200] [--out FILE]

For each ladder problem (tests/ladder.py) and size n: the user code ladder.solve at the default settings (n <=
settings.dense_max: exact dense routes) gives the reference std of p functionals (data functionals, and point
evaluations where the space has them); the same user code with settings.dense_max = n // 10 forces the matrix-free
route and the randomised posterior (probability/scale.py). Reported: the largest error relative to the exact std,
whether every std lies within its stated (combined) bound against the exact posterior with the same alpha, the rank, ||W - W_k||, how many functionals took the exact route, and
timings. Results as JSON lines (and to --out).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
import pygeoinf0 as pg  # noqa: E402
import ladder  # noqa: E402


def functionals(P, p, seed=3):
    """p functionals: data functionals, and point evaluations where the space has bounded point values."""
    rng = np.random.default_rng(seed)
    if P.H.dual().integrals_bounded(0):
        if P.name == "L1":
            return rng.uniform(0, 1, p)
        if P.name == "L4":
            X = rng.normal(size=(p, 3))
            return X / np.linalg.norm(X, axis=1)[:, None]
        d = 2 if P.name == "L2" else 3
        X = rng.normal(size=(p, d))
        return 0.9 * X / np.linalg.norm(X, axis=1)[:, None] * rng.uniform(0, 1, (p, 1)) ** (1 / d)
    return P.measures[:p]


LARGE = 20000           # above this n no dense reference: the low-rank std against exact block CG on a subset


def run_large(name, n, p, sub=20):
    """n above LARGE: the user code above dense_max only; the at-scale std of p functionals against exact deflated
    block CG on the same computed operator for `sub` of them (the Nystrom part checked; the engine's is stated)."""
    t0 = time.perf_counter()
    P = ladder.build(name, n)
    t_build = time.perf_counter() - t0
    u = functionals(P, p)
    t0 = time.perf_counter()
    _, alpha, big = ladder.solve(P)
    t_solve = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = big.std(u)
    t_std = time.perf_counter() - t0
    sc = big._scale
    bound, claims = sc.last_bound, str(big.variance_claims[-1])
    t0 = time.perf_counter()
    ref = big.std(u[:sub], exact=True)
    t_ref = time.perf_counter() - t0
    rel = np.abs(s[:sub] - ref) / ref
    return {"problem": name, "n": n, "functionals": len(s), "reference": f"exact block CG on {sub} (same operator)",
            "max_rel_std_error_vs_cg": float(rel.max()), "inside_combined_bound": int(np.sum(
                np.abs(ref ** 2 - s[:sub] ** 2) <= bound[:sub] * (1 + 1e-9) + 1e-11 * ref ** 2)),
            "max_std_bound_over_std": float(np.max((np.sqrt(s ** 2 + bound) - s) / s)), "rank": int(len(sc.dec.lam)),
            "norm_E": float(sc.dec.error), "cg": "deflated block CG" in claims, "claim": claims[:400],
            "seconds": {"build": t_build, "solve_matrix_free": t_solve, "std_at_scale": t_std, "cg_reference": t_ref}}


def run(name, n, p):
    if n > LARGE:
        return run_large(name, n, p)
    t0 = time.perf_counter()
    P = ladder.build(name, n)
    t_build = time.perf_counter() - t0
    u = functionals(P, p)
    t0 = time.perf_counter()
    _, alpha, post = ladder.solve(P)
    exact = post.std(u)
    t_exact = time.perf_counter() - t0
    with pg.settings.using(dense_max=max(n // 10, 50)):
        t0 = time.perf_counter()
        _, alpha2, big = ladder.solve(P)
        t_solve = time.perf_counter() - t0
        t0 = time.perf_counter()
        approx = big.std(u)
        t_std = time.perf_counter() - t0
        sc = big._scale
        dec = sc.dec
        cross, tt = sc._pairings(P.H.riesz(u) if not isinstance(u, list) else P.H.riesz(u))
        lam, C, _ = dec._spec(cross)
        C = C[:, None] if C.ndim == 1 else C
        var_k = tt - np.einsum("kp,k->p", C * C, 1.0 / (lam + 1.0))
        bound = dec.error * np.einsum("kp,k->p", C * C, 1.0 / (lam + 1.0) ** 2)
        claims = str(big.variance_claims[-1])
        total = sc.last_bound
    # the reference with the same alpha: each route chooses alpha by the discrepancy principle (eigh or Lanczos) and the
    # two differ slightly, which would mix a different posterior into the comparison
    t0 = time.perf_counter()
    G = pg.Functionals(P.H.riesz(P.measures), into=P.D)
    exact = pg.probability.cylindrical(P.H, alpha=alpha2).condition(G, P.d).std(u)
    t_ref = time.perf_counter() - t0
    rel = np.abs(approx - exact) / exact
    inside = np.abs(exact ** 2 - approx ** 2) <= total * (1 + 1e-9) + 1e-11 * exact ** 2   # the combined bound (+ rounding)
    return {"problem": name, "n": n, "functionals": len(exact), "alpha_rel": float(abs(alpha2 / alpha - 1)),
            "max_rel_std_error": float(rel.max()), "rule": pg.settings.accuracy, "within_rule": bool(rel.max() <= pg.settings.accuracy),
            "inside_combined_bound": int(inside.sum()), "max_std_bound_over_std": float(np.max(
                (np.sqrt(approx ** 2 + total) - approx) / approx)), "rank": int(len(dec.lam)), "norm_E": float(dec.error),
            "cg": "deflated block CG" in claims, "claim": claims[:300],
            "seconds": {"build": t_build, "exact_solve_and_std": t_exact, "exact_same_alpha": t_ref, "matrix_free_solve": t_solve, "std_at_scale": t_std}}


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    a.add_argument("--problems", nargs="+", default=["L1", "L2", "L3", "L4"])
    a.add_argument("--functionals", type=int, default=200)
    a.add_argument("--out")
    o = a.parse_args()
    for n in o.sizes:
        for name in o.problems:
            try:
                row = run(name, n, o.functionals)
            except Exception as exc:  # noqa: BLE001  (a benchmark row records the failure and goes on)
                row = {"problem": name, "n": n, "error": f"{type(exc).__name__}: {exc}"}
            line = json.dumps(row, default=float)
            print(line, flush=True)
            if o.out:
                with open(o.out, "a") as f:
                    f.write(line + "\n")


if __name__ == "__main__":
    main()
