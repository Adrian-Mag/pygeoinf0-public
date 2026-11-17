"""The Fourier basis on angular axes: exact analysis and mass, agreement with Legendre, continuity across the seam."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.realisations.series import tensor as T


def test_fourier_basis_is_exact_for_trigonometric_polynomials():
    b = T.FourierBasis(0.0, 2 * np.pi, 9)                      # K = 4
    t = b.nodes()
    v = 0.3 + np.cos(t) - 2 * np.sin(3 * t) + 0.5 * np.cos(4 * t)
    c = b.analyse(v[:, None], 0)[:, 0]
    want = np.zeros(9)
    want[[0, 1, 6, 7]] = [0.3, 1.0, -2.0, 0.5]
    assert np.allclose(c, want, atol=1e-14)
    x = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    G = (b.vander(x) * (2 * np.pi / 400)).T @ b.vander(x)      # periodic rectangle rule: exact for degree < 400
    assert np.allclose(G, b.mass_blocks("1")[0], atol=1e-12)


@pytest.mark.parametrize("make, fun", [
    (lambda: pg.Disk(1.0), lambda x, y: np.exp(-((x - 0.3) ** 2 + y ** 2) / 0.2) * np.cos(3 * y)),
    (lambda: pg.Cylinder(1.0, z=(0, 2)), lambda x, y, z: np.exp(-((x - 0.3) ** 2 + y ** 2) / 0.2) * (1 + z ** 2)),
    (lambda: pg.Ball(1.0), lambda x, y, z: np.exp(-((x - 0.3) ** 2 + y ** 2 + (z - 0.2) ** 2) / 0.2)),
])
def test_fourier_and_legendre_agree(make, fun):
    out = {}
    for basis in ("legendre", "fourier"):
        M = pg.L2(make(), realisation="legendre" if basis == "legendre" else "tensor")
        f = M(fun)
        g = M(lambda *X: 1.0 + X[0] * X[1])
        out[basis] = (f.norm(), f @ g, f.integral(), (f * g).integral(), f.realisation.size)
    a, b = out["legendre"], out["fourier"]
    assert np.allclose(a[:4], b[:4], rtol=1e-10, atol=1e-13)
    assert b[4] < a[4]                                          # fewer coefficients for the same accuracy


def test_fourier_realisation_is_continuous_across_the_seam():
    M = pg.L2(pg.Disk(1.0))
    f = M(lambda x, y: np.exp(-((x - 0.6) ** 2 + y ** 2) / 0.05))   # centred on phi = 0
    r = np.linspace(0.1, 0.95, 7)
    eps = 1e-9
    below = f(r * np.cos(-eps), r * np.sin(-eps))
    above = f(r * np.cos(eps), r * np.sin(eps))
    assert np.allclose(below, above, rtol=0, atol=1e-8)
    assert "Fourier" in f.info()


def test_product_kernels_project_onto_a_fourier_model():
    M = pg.L2(pg.Disk(1.0))
    K = M.product_family([lambda r, rc: np.exp(-((r - rc) / 0.1) ** 2),
                          lambda phi, pc: np.exp((np.cos(phi - pc) - 1) / 0.2 ** 2)],
                         rc=np.array([0.3, 0.6, 0.8]), pc=np.array([0.0, 2.0, 5.5]))
    m = M(lambda x, y: np.sin(3 * x) * np.cos(2 * y) + 0.5 * y)
    fast = pg.Functionals(K)(m).values
    slow = np.array([k @ m for k in K])                          # each kernel realised as a tensor function
    assert np.allclose(fast, slow, rtol=1e-10, atol=1e-13)
