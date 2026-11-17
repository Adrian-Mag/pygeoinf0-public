"""Complex conductivity of a chargeable body on a cell mesh (plan 020 P3c; EXP-035 COMPLEX.md).

The unknown is the complex conductivity perturbation per cell at 1 Hz; a chargeable body follows the Pelton
(Cole-Cole) model, the background does not. Data are complex-linear averages of it (real sensitivity kernels at two
depths below 23 surface receivers). With a circular Gaussian prior and circular noise, the posterior is circular and
its mean is the complex formula S K^H (K S K^H + s^2 I)^-1 d; the induced-polarisation phase of the body appears in it.

    python examples/cole_cole_cells.py
"""
import numpy as np

import pygeoinf0 as pg


def sigma(m, rho0=100.0, tau=0.01, c=0.5, omega=2 * np.pi):
    """Pelton / Cole-Cole complex conductivity at angular frequency omega (chargeability m)."""
    return 1.0 / (rho0 * (1 - m * (1 - 1 / (1 + (1j * omega * tau) ** c))))


def main():
    cells = pg.realisations.Cells.tensor(np.linspace(0, 240, 25), np.linspace(-120, 0, 13))
    C = pg.Complexified(pg.L2(pg.Box([(0, 240), (-120, 0)]), realisation=cells))     # native complex cell values
    xc, vol = 0.5 * (cells.lo + cells.hi), cells.volumes
    body = ((xc[:, 0] - 120) ** 2 / 40 ** 2 + (xc[:, 1] + 50) ** 2 / 20 ** 2) < 1
    cc = pg.port.Coordinates.cells(C)
    truth = cc.element(sigma(np.where(body, 0.3, 0.0)) - sigma(0.0))

    rx = np.linspace(10, 230, 23)
    K = np.vstack([np.exp(-((xc[None, :, 0] - rx[:, None]) ** 2 + (xc[None, :, 1] + z0) ** 2) / (2 * s ** 2))
                   for z0, s in ((0.0, 35.0), (60.0, 45.0))]) * vol / 1e3
    G = pg.ComplexMatrix(K, cc.export.codomain, name="K") @ cc.export
    print(G.claims.find("complex-linear"))
    print(pg.port.complex_linearity(G).claims)

    scale = np.abs(cc.numbers(truth)).max()
    dist = np.linalg.norm(xc[:, None] - xc[None], axis=2)
    prior = pg.probability.circular.circular(C, covariance=scale ** 2 * np.exp(-dist / 30.0))
    s = 0.02 * np.abs(G(truth).values).max()
    noise = pg.probability.circular.circular(G.codomain, covariance=s ** 2)
    d = G(truth) + noise.sample(seed=3)
    post = prior.condition(G, d, noise=noise)
    print(post.claims.find("circular"))

    phase = lambda v: 1e3 * np.angle(sigma(0.0) + v)           # milliradians
    t, m = cc.numbers(truth), cc.numbers(post.mean)
    print(f"IP phase (mrad)  body: true {phase(t[body]).mean():5.1f}, posterior mean {phase(m[body]).mean():5.1f}"
          f"   background: true {phase(t[~body]).mean():4.1f}, posterior mean {phase(m[~body]).mean():4.1f}")
    avg = cc.element(np.where(body, 1.0, 0.0) / vol[body].sum() + 0j)
    print(f"std of Re(average over the body): prior {prior.std(avg):.2e}, posterior {post.std(avg):.2e}")


if __name__ == "__main__":
    main()
