"""Independent contracts for the SOLA notebook experiment, not a package SOLA API change."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "discretisation_sola"
sys.path.insert(0, str(EXAMPLE))
spec = importlib.util.spec_from_file_location("sola_demo", EXAMPLE / "sola_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_pixel_ray_matrix_and_disk_mass():
    p, q = np.array([[0., .25], [.25, 0.]]), np.array([[1., .25], [.25, 1.]])
    G = demo.ray_matrix(p, q, 2).toarray()
    assert_allclose(G, [[.5, .5, 0, 0], [.5, 0, .5, 0]], atol=1e-15)
    for n in (8, 17, 40):
        t = demo.disk_pixel_density(n)
        assert_allclose(t.sum() / n**2, 1., rtol=0, atol=2e-10)


def test_disk_quadrature_is_the_same_physical_average():
    p, q, w = demo.disk_chords(24)
    vals = demo.P.ray_integrals(p, q, lambda x, y: x*x+y*y)
    cx, cy = demo.CENTRE
    assert_allclose(w @ vals, cx*cx+cy*cy+demo.RADIUS**2/2, rtol=1e-12, atol=1e-13)
    assert_allclose(w @ np.linalg.norm(q-p, axis=1), 1., atol=1e-13)


@pytest.mark.parametrize("constrained", [False, True])
def test_sola_weights_against_independent_kkt(constrained):
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 7))
    K, b = A@A.T, rng.normal(size=4)
    v, tau = rng.normal(size=4), .3
    x = demo.sola_weights(K, b, tau, v if constrained else None)
    H = K+tau*np.eye(4)
    if constrained:
        M = np.block([[H, v[:, None]], [v[None, :], np.zeros((1, 1))]])
        reference = np.linalg.solve(M, np.r_[b, 1.])[:4]
        assert_allclose(v@x, 1., atol=1e-13)
    else:
        reference = np.linalg.solve(H, b)
    assert_allclose(x, reference, rtol=2e-12, atol=1e-13)


def test_constraint_infeasibility_and_invalid_gram():
    with pytest.raises(ValueError, match="response"):
        demo.sola_weights(np.eye(2), np.ones(2), .1, np.zeros(2))
    with pytest.raises(ValueError):
        demo.sola_weights(np.eye(2), np.ones(2), -1.)
    with pytest.raises(ValueError):
        demo.sola_weights(np.array([[np.nan]]), np.ones(1), .1)


def test_noise_and_target_error_are_different():
    rng = np.random.default_rng(9)
    x, d0, eta = rng.normal(size=(3, 5))
    target = .4
    z = demo.estimator_diagnostics(x, d0, d0+eta, target, .01)
    assert_allclose(z["estimate"]-target, z["target_bias"]+z["noise_draw"], atol=1e-14)
    assert_allclose(z["std"], .01*np.linalg.norm(x), atol=1e-15)


def test_pixel_objective_uses_area_and_not_the_coefficient_dot_product():
    rng = np.random.default_rng(7)
    n = 9
    p, q = rng.random((6, 2)), rng.random((6, 2))
    G = demo.ray_matrix(p, q, n)
    t = demo.disk_pixel_density(n)
    K = (G@G.T).toarray()*n*n
    b = np.asarray(G@t)
    x = demo.sola_weights(K, b, .3, np.linalg.norm(q-p, axis=1))
    density = np.asarray(G.T@x)*n*n
    explicit = np.sum((density-t)**2)/n**2
    via_gram = x@K@x-2*x@b+t@t/n**2
    assert_allclose(via_gram, explicit, rtol=2e-12, atol=1e-12)
