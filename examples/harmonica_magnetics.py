"""Induced magnetics over a buried body with the pygeoinf0-harmonica plugin (plan 020 P4; plugins/README.md).

The unknown is the susceptibility (SI) in L2 of a box below the surface, realised on 2048 prisms; the data are the
total-field anomaly at 225 points 20 m above the surface (Harmonica's closed-form prism fields; inducing field inc 60,
dec 10, 50 000 nT). The kernel decays like depth^-3, so a prior with the same variance at every depth puts the
recovered body at the top of the box. Li & Oldenburg's depth weighting appears here as what it is in a Bayesian
reading: a prior whose standard deviation grows with depth, (z0 + depth)^(beta / 2) with beta = 3. Both priors are
Gaussian measures on L2(box); the posteriors are exact (linear map, Gaussian noise).

What it shows (europa, 2026-10-31): depth weighting moves the recovered susceptibility from 175 m to 267 m mean depth
(body at 225 m). The body's average susceptibility is barely resolved by either (0.004 and 0.014 against 0.05: the
non-uniqueness of potential fields), and the posterior std says so (the truth 2.2 sd away) only because the prior's
scale makes the body a plausible draw; with a prior sd of 0.02 the same means come with stds of 0.008 and 0.007 and
the truth 5.6 and 5.2 sd away. The posterior mean barely depends on the prior's scale here (small noise); its std
does.

    python examples/harmonica_magnetics.py        (needs plugins/pygeoinf0-harmonica on the path)
"""
import time

import numpy as np

import pygeoinf0 as pg
import pygeoinf0_harmonica as ph


def main():
    e = n = np.linspace(-400.0, 400.0, 17)
    u = np.linspace(-400.0, 0.0, 9)
    M, C, P = ph.prism_space(e, n, u)
    xc = 0.5 * (P[:, 0::2] + P[:, 1::2])
    depth = -xc[:, 2]
    body = (np.abs(xc[:, 0] - 50) < 100) & (np.abs(xc[:, 1]) < 100) & (depth > 150) & (depth < 300)
    truth = C.element(np.where(body, 0.05, 0.0))

    g = np.linspace(-350.0, 350.0, 15)
    X = np.column_stack([*(a.ravel() for a in np.meshgrid(g, g)), np.full(g.size ** 2, 20.0)])
    inducing = (60.0, 10.0, 50_000.0)
    t = time.time()
    G = ph.magnetic_operator(C, P, X, inducing)
    print(f"{G.codomain.dim} data, {C.dim} prisms: sensitivities in {time.time() - t:.2f} s")
    print(G.claims.find("realisation"))

    R = C.export.codomain
    s = 0.02 * np.abs(G(truth).values).max()
    noise = pg.GaussianMeasure(G.codomain, s ** 2 * pg.Identity(G.codomain))
    d = G(truth) + noise.sample(seed=1)
    corr = np.exp(-np.linalg.norm(xc[:, None] - xc[None], axis=2) / 80.0)
    vol = np.prod(P[:, 1::2] - P[:, 0::2], axis=1)
    avg = C.imp(R(np.where(body, 1.0, 0.0) / vol[body].sum()))          # the average over the body, a covector

    for label, sd in (("flat prior    ", np.full(C.dim, 0.05)),
                      ("depth-weighted", 0.05 * ((25.0 + depth) / (25.0 + depth.mean())) ** 1.5)):
        prior = pg.GaussianMeasure(R, pg.Matrix(sd[:, None] * corr * sd[None], R, R)).pushforward(C.imp)
        t = time.time()
        post = prior.condition(G, d, noise=noise)
        m = C.numbers(post.mean)
        w = np.clip(m, 0, None)
        a, sa = M.inner(post.mean, avg), post.std(avg)
        print(f"{label}: mean depth of the recovered susceptibility {w @ depth / w.sum():5.0f} m (body 225 m); "
              f"body average {a:.4f} +- {sa:.4f} (true 0.0500, {(0.05 - a) / sa:.1f} sd away); "
              f"{time.time() - t:.1f} s")


if __name__ == "__main__":
    main()
