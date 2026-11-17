"""One rung of the scale ladder end to end with the unchanged user code (ladder.solve), each stage timed, with the
routes' claims and the resident memory (plan 005, P4/P6).

    PYGEOINF0_WORKERS=30 PYTHONPATH=.:tests python benchmarks/rung.py L3 30000 --segments 27
    ... --engine spectral      (pg.settings.engine for the products above dense_max; default: the setting's default)
    ... --collinear 0.3        (pg.settings.collinear: between rays correct only the pairs that meet or are at
                                sin(angle) < 0.3)

With the spectral engine the plan's counts and timings (pairs, corrections, transforms) and claims are printed, and
the engine's two-resolution estimate of the predicted-data error at the posterior mean (rms and max, noise std).
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import ladder  # noqa: E402

import pygeoinf0 as pg  # noqa: E402


def gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def gb_workers():
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("n", type=int)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--engine", choices=("hmatrix", "spectral", "exact"), default=None)
    ap.add_argument("--collinear", type=float, default=None)
    a = ap.parse_args()
    if a.engine is not None:
        pg.settings.engine = a.engine
    if a.collinear is not None:
        pg.settings.collinear = a.collinear
    t0 = t = time.time()

    def lap(name):
        nonlocal t
        print(f"{name:34s} {time.time() - t:9.1f} s   peak {gb():6.2f} GB (workers {gb_workers():.2f})", flush=True)
        t = time.time()
    print(pg.settings, flush=True)
    P = ladder.build(a.problem, a.n, **({"segments": a.segments} if a.problem in ("L3", "fan") else {}))
    lap(f"build problem (n = {P.n})")
    K = P.H.riesz(P.measures)
    lap(f"riesz ({len(K.realisation.kx.atoms.P)} atoms)")
    G = pg.Functionals(K, into=P.D)
    lap("Functionals")
    A = G @ G.adjoint
    lap("G @ G.adjoint")
    print(A.claims, flush=True)
    alpha = pg.solvers.damping_for_misfit(A, P.d, P.D.dim)
    lap("discrepancy (incl. any compression)")
    print(alpha.claims, flush=True)
    atoms = K.realisation.kx.atoms
    if atoms.hmatrix is not None:
        print(atoms.hmatrix.stats, atoms.hmatrix.claims, sep="\n", flush=True)
    if atoms.spectral is not None:
        print(atoms.spectral.stats, atoms.spectral.claims(), sep="\n", flush=True)
    post = pg.probability.cylindrical(P.H, alpha=alpha).condition(G, P.d)
    m = post.mean
    lap("condition (posterior mean)")
    print(post.solver.claims, flush=True)
    r = G(m) - P.d
    chi2 = P.D.norm(r) ** 2
    lap("G(m) - d")
    print(f"alpha = {float(alpha):.4e}; chi^2 / n = {chi2 / P.n:.4f} (target 1, spread sqrt(2/n) = {np.sqrt(2 / P.n):.4f})")
    kx = m.realisation.kx
    if atoms.spectral is not None and kx is not None and kx.atoms is atoms:
        rms, top = atoms.spectral.estimate_data(np.asarray(kx.C, float)[:, 0], P.sigma)
        lap("estimate of the data error")
        print(f"spectral engine: predicted-data error at the posterior mean ~ {rms:.2e} noise std (rms), {top:.2e} "
              f"(max); settings.accuracy = {pg.settings.accuracy:g}", flush=True)
    print(f"total {time.time() - t0:.1f} s, peak {gb():.2f} GB (workers {gb_workers():.2f})", flush=True)


if __name__ == "__main__":
    main()
