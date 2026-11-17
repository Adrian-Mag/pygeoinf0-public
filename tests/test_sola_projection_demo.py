"""Contracts for Hilbert projections and the unconstrained pixel limit."""
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'/'discretisation_sola'))
from sola_projection import projected_gram
from sola_demo import ray_matrix, sola_weights


def test_projection_matches_ambient_orthogonal_projector_and_changes_of_basis():
    rng = np.random.default_rng(42)
    A, t, Phi = rng.normal(size=(5, 9)), rng.normal(size=9), rng.normal(size=(9, 4))
    Q = np.linalg.qr(Phi)[0]
    expected = (A@Q@Q.T@A.T, A@Q@Q.T@t, t@Q@Q.T@t)
    for basis in (Phi, Phi@np.diag([.01, .3, 2., 20.])):
        result = projected_gram(basis.T@basis, A@basis, basis.T@t)
        for actual, correct in zip(result, expected):
            assert_allclose(actual, correct, rtol=3e-11, atol=2e-12)


def test_nested_projections_and_exact_data_span():
    rng = np.random.default_rng(4)
    A, t = rng.normal(size=(3, 9)), rng.normal(size=9)
    Phi = np.linalg.qr(np.column_stack([A.T, t]))[0]
    previous = np.zeros((3, 3))
    for k in (1, 2, 3, 4):
        Z = Phi[:, :k]
        K, b, tt = projected_gram(Z.T@Z, A@Z, Z.T@t)
        assert np.linalg.eigvalsh(K-previous).min() > -2e-12
        previous = K
        if k >= 3:
            assert_allclose(K, A@A.T, rtol=2e-12, atol=2e-12)
            assert_allclose(b, A@t, rtol=2e-12, atol=2e-12)
        assert tt <= t@t+2e-12


def test_invalid_or_dependent_trial_basis_is_not_regularized_silently():
    with pytest.raises(np.linalg.LinAlgError):
        projected_gram(np.ones((2, 2)), np.ones((3, 2)), np.ones(2))
    with pytest.raises(ValueError):
        projected_gram(np.eye(2), np.ones((3, 3)), np.ones(2))


def test_parallel_ray_example_has_exact_vanishing_weights_and_density():
    # Distinct full horizontal rays: for fine enough pixels, K_h=n I exactly.
    p = np.array([[0., .13], [0., .47], [0., .83]])
    q = p+np.array([1., 0.])
    tau = .7
    for n in (8, 16, 32, 64):
        G = ray_matrix(p, q, n)
        K = (G@G.T).toarray()*n*n
        b = np.asarray(G@np.ones(n*n))
        x = sola_weights(K, b, tau)
        assert_allclose(x, np.ones(3)/(n+tau), rtol=2e-13, atol=1e-15)
        density = np.asarray(G.T@x)*n*n
        assert_allclose(density@density/n**2, 3*n/(n+tau)**2, rtol=2e-13)
