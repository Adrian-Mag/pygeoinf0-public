"""Sobolev spaces on the sphere: the spectral metric, point and arc representers against independent references."""
import math

import numpy as np
import pytest
from scipy import integrate, special

import pygeoinf0 as pg


def zonal_kernel(gamma, s, l, R, N=4000):
    """sum_n (2n+1)/(4 pi R^2) (1 + l^2 n(n+1)/R^2)^-s P_n(cos gamma): independent of pygeoinf0's harmonics."""
    n = np.arange(N + 1)
    c = (2 * n + 1) / (4 * math.pi * R ** 2) * (1 + l ** 2 * n * (n + 1) / R ** 2) ** -s
    return np.polynomial.legendre.legval(np.cos(gamma), c)


def test_spectral_metric_and_point_kernel():
    R, l, s = 2.0, 0.3, 2.0
    H = pg.Sobolev(pg.Sphere(R), s, length_scale=l)
    z = H(lambda x, y, z: z)                                       # degree 1: ||z||^2 = (1 + 2 l^2/R^2)^s ||z||_L2^2
    assert z.norm() ** 2 == pytest.approx((1 + 2 * l ** 2 / R ** 2) ** s * 4 * math.pi * R ** 4 / 3, rel=1e-13)
    P = np.array([[0, 0, R], [R * math.sin(1.0), 0, R * math.cos(1.0)], [0, R * math.sin(2.5), R * math.cos(2.5)]])
    k = H.riesz([pg.DiracMeasure(p) for p in P])
    K = H.inner(k, k)
    G = np.arccos(np.clip(P @ P.T / R ** 2, -1, 1))
    ref = zonal_kernel(G, s, l, R, N=200000)
    e = k.realisation.err                                         # estimated H^s truncation error of each representer
    gap = np.abs(ref - K)                                         # truncation error of the Gram matrix
    assert np.all(gap <= 1.5 * np.sqrt(np.outer(e, e)) * 1.0 + 1e-15)   # Cauchy-Schwarz with the estimated tails
    assert np.all(np.diag(gap) >= 0.3 * e ** 2)                   # and the estimates are not wildly pessimistic
    f = H(lambda x, y, z: x * z + y)
    assert H.evaluation(P[1])(f) == pytest.approx(f(*P[1]), rel=1e-12)


def test_arc_integrals_reproduce():
    S = pg.Sphere(1.0)
    H = pg.Sobolev(S, 1.5, length_scale=0.2)
    arc = pg.Arc(S, (1, 0, 0), (0, 1, 0))                          # quarter of the equator
    k = H.riesz(arc)
    f = H(lambda x, y, z: x ** 2 + 0.5 * z + y)
    ref = integrate.quad(lambda t: math.cos(t) ** 2 + math.sin(t), 0, math.pi / 2)[0]
    assert k @ f == pytest.approx(ref, rel=1e-12)                  # <k_arc, f> = int_arc f ds, exact for formulas
    from pygeoinf0.core.errors import UnboundedFunctional
    with pytest.raises(UnboundedFunctional):
        pg.Sobolev(S, 0.5, length_scale=0.2).riesz(arc)
