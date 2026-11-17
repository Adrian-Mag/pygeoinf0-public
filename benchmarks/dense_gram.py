"""The dense Gram G G* of a ladder problem under pg.settings.dense_gram = "exact" and "compressed" (plan 005, P4): the
wall time of each route, and the compressed Gram's actual error against the exact one next to its stated estimate.

    PYGEOINF0_WORKERS=30 PYTHONPATH=.:tests python benchmarks/dense_gram.py L3 3000 --segments 27
    ... --routes compressed          (one route only: no comparison)
    ... --engine spectral            (also the products E V of that matrix-free engine against the exact Gram: time,
                                      memory, relative error next to the engine's estimate)

Each route builds the problem afresh, so no energies are shared between them. Run on europa.
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


def gram(problem, n, segments, route):
    P = ladder.build(problem, n, **({"segments": segments} if problem in ("L3", "fan") else {}))
    K = P.H.riesz(P.measures)
    t = time.time()
    with pg.settings.using(dense_gram=route):
        G = pg.Functionals(K, into=P.D)
        A = G @ G.adjoint
        M = A.matrix()
    wall = time.time() - t
    atoms = K.realisation.kx.atoms
    print(f"{route:10s} n = {P.n}, {len(atoms.P)} atoms: {wall:9.1f} s, peak {gb():6.2f} GB", flush=True)
    print(A.claims, flush=True)
    if atoms.hmatrix is not None:
        print(atoms.hmatrix.stats, atoms.hmatrix.claims, sep="\n", flush=True)
    return M, wall, P, atoms


def engine_products(P, atoms, engine):
    """E V by the matrix-free engine settings.engine = engine on the atoms of the exact route, against their energies."""
    backend = P.H.backend.engine
    E = backend.energy(atoms, atoms)                                # cached by the exact route
    atoms.self_energy = None
    V = np.random.default_rng(5).standard_normal((atoms.n, 2))
    with pg.settings.using(engine=engine):
        name, eng, why = backend.route(atoms)
        t = time.time()
        Y = eng.apply(atoms, V)                                     # the first product includes the setup
        setup = time.time() - t
        t = time.time()
        Y = eng.apply(atoms, V)
        wall = (time.time() - t) / V.shape[1]
        rel = np.max(np.linalg.norm(Y - E @ V, axis=0) / np.linalg.norm(E @ V, axis=0))
        print(f"{name:10s} products ({why}): setup {setup:.1f} s, {wall:.2f} s per product, peak {gb():6.2f} GB; "
              f"relative error {rel:.2e}", flush=True)
        if atoms.spectral is not None:
            print(atoms.spectral.stats, atoms.spectral.claims(), sep="\n", flush=True)
            print(f"two-resolution estimate on these vectors {eng.estimate(atoms, V):.2e}", flush=True)
        elif atoms.hmatrix is not None:
            print(atoms.hmatrix.stats, atoms.hmatrix.claims, sep="\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("n", type=int)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--routes", nargs="+", default=["exact", "compressed"])
    ap.add_argument("--engine", choices=("hmatrix", "spectral", "exact"), default=None)
    a = ap.parse_args()
    print(pg.settings, flush=True)
    routes = a.routes if a.engine is None or "exact" in a.routes else ["exact"] + a.routes
    out = {r: gram(a.problem, a.n, a.segments, r) for r in routes}
    if a.engine is not None:
        engine_products(*out["exact"][2:], a.engine)
    if len(out) == 2:
        (E, te, *_), (C, tc, *_) = out["exact"], out["compressed"]
        rel = np.linalg.norm(C - E) / np.linalg.norm(E)
        d = np.sqrt(np.diag(E))
        print(f"speed-up {te / tc:.2f}; actual relative Frobenius error {rel:.2e}; "
              f"max |C - E| / sqrt(E_ii E_jj) {np.max(np.abs(C - E) / np.outer(d, d)):.2e}", flush=True)


if __name__ == "__main__":
    main()
