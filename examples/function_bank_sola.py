"""SOLA with averaging kernels from the function bank (plan 021 P2).

One data set, four target shapes at the same place: the bank supplies the kernels (unit mass, with their support and
regularity as claims), f.measure(M.measure) turns each into the functional m -> int f m dmu, and the data-space
decomposition gives SOLA's estimate, noise std and delivered mismatch for a whole array of gamma.

Part 2 does the same on a disk whose space carries a weighted measure: the target is normalised against that measure,
so it averages with respect to it.

    python examples/function_bank_sola.py
"""
import numpy as np

import pygeoinf0 as pg
import pygeoinf0.functions as F


def _reg(t):
    o = t.regularity.order
    return "C^inf" if o == np.inf else "jumps" if o < 0 else f"C^{int(o)}"


def interval():
    I = pg.Interval("[0, 1]")
    M = pg.Sobolev(I, 1, length_scale=0.1)
    D = pg.Euclidean(12, covariance=0.02 ** 2)
    G = pg.Functionals(M.riesz([pg.Interval(a, a + 0.08) for a in np.linspace(0, 0.92, 12)]) * 12.5, into=D)
    truth = M(lambda x: np.sin(2 * np.pi * x) + x, resolution=0.1)        # its narrowest feature (claims: met)
    d = G(truth) + D.standard_gaussian().sample(seed=1)

    on = F.on(I)
    targets = {"box": on.plateau(centre=0.5, half_width=0.1).normalised(),
               "smooth box": on.plateau(centre=0.5, half_width=0.08, edge=0.04).normalised(),
               "gaussian": on.gaussian(centre=0.5, width=0.05).normalised(),
               "bump": on.bump(centre=0.5, radius=0.12).normalised()}
    for t in targets.values():                     # each kernel is in L2, so m -> int t m is bounded on H^1 too
        assert t.element(pg.L2(I)) is not None
    T = pg.Functionals(M.riesz([t.measure(M.measure) for t in targets.values()]))
    true = T(truth).values

    dec = pg.decompose(G, D.standard_gaussian())
    gammas = np.geomspace(1e-1, 1e5, 7)               # gamma weighs the noise covariance (0.02^2) against G G*
    res = dec.sola(T, gammas, d)
    print(f"{'target':>11} {'kernel':>7}  true    " + "  ".join(f"gamma={g:<8.0e}" for g in gammas))
    for i, (name, t) in enumerate(targets.items()):
        est = "  ".join(f"{e:+.3f} ±{s:.3f}" for e, s in zip(res.estimate[:, i], res.std[:, i]))
        print(f"{name:>11} {_reg(t):>7}  {true[i]:.3f}  {est}")
    print("delivered mismatch ||t - G* X*||^2 (H^1 dual norm):")
    for i, name in enumerate(targets):
        print(f"{name:>11} {'':>7}         " + "  ".join(f"{m:<15.3g}" for m in res.mismatch[:, i]))


def weighted_interval():
    """L2(r^2 dr) on [0, 1] (the radial part of a ball; weighted spaces exist on intervals, EXP-031): a bump
    normalised against r^2 dr averages with respect to it."""
    I = pg.Interval("[0, 1]")
    w = pg.LebesgueMeasure(I).with_density(lambda r: r ** 2, name="r^2")
    M = pg.L2(w)
    t = F.on(I).bump(centre=0.6, radius=0.15).normalised(against=w)
    T = pg.Functionals(M.riesz([t.measure(w)]))
    mass, status, _ = t.closed["mass"]
    print(f"\nL2(r^2 dr): int t r^2 dr = {mass:g} ({status}); <t, 1> = {T(M(lambda r: 1 + 0 * r, resolution=1.0)).values[0]:.12f}, "
          f"<t, r> = {T(M(lambda r: r, resolution=1.0)).values[0]:.6f} (the r^2-weighted mean radius near 0.6)")


if __name__ == "__main__":
    interval()
    weighted_interval()
