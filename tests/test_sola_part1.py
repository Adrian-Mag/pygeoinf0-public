"""Independent contracts for the constrained, common-target SOLA tutorial."""
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest

EXAMPLE=Path(__file__).parents[1]/'examples'/'discretisation_sola'
sys.path.insert(0,str(EXAMPLE/'part1'))
sys.path.insert(0,str(EXAMPLE))
import study as tutorial
import sola_demo as S


def test_batched_constrained_weights_match_independent_kkt():
    rng=np.random.default_rng(42)
    E=rng.normal(size=(5,9)); K=E@E.T
    B=rng.normal(size=(3,5)); v=rng.normal(size=5); tau=.7
    X=tutorial.constrained_sola(K,B,tau,v)
    kkt=np.block([[K+tau*np.eye(5),v[:,None]],[v[None,:],np.zeros((1,1))]])
    expected=np.linalg.solve(kkt,np.vstack([B.T,np.ones((1,3))]))[:5].T
    assert_allclose(X,expected,rtol=2e-12,atol=2e-13)
    assert_allclose(X@v,np.ones(3),rtol=0,atol=2e-13)


def test_galerkin_pairings_match_ambient_projection_and_basis_invariance():
    rng=np.random.default_rng(7)
    G=rng.normal(size=(4,11)); T=rng.normal(size=(3,11))
    Phi=rng.normal(size=(11,6)); Q=np.linalg.qr(Phi)[0]
    for basis in (Phi,Phi@np.diag([.02,.1,.4,2.,5.,20.])):
        K,B,tt=tutorial.galerkin_pairings(basis.T@basis,G@basis,T@basis)
        assert_allclose(K,G@Q@Q.T@G.T,rtol=2e-11,atol=2e-12)
        assert_allclose(B,T@Q@Q.T@G.T,rtol=2e-11,atol=2e-12)
        assert_allclose(tt,np.sum((T@Q)**2,axis=1),rtol=2e-11,atol=2e-12)


def test_parallel_rays_normalized_weights_forget_fixed_targets():
    # For separated horizontal unit rays K_h=h^-1 I, v=1.
    B=np.array([[2.,0.,.3],[0.,1.,2.]])
    p=np.array([[0.,.13],[0.,.47],[0.,.83]]);q=p+[1.,0.]
    for n in (8,32,128):
        E=S.ray_matrix(p,q,n);K=(E@E.T).toarray()*n*n
        X=tutorial.constrained_sola(K,B,.2,np.ones(3))
        expected=np.ones((2,3))/3+(B-B.mean(axis=1,keepdims=True))/(n+.2)
        assert_allclose(X,expected,rtol=2e-12,atol=2e-13)


def test_generalized_disk_targets_preserve_mass_and_polynomial_averages():
    centre=(.7,.3);radius=.15
    density=S.disk_pixel_density(17,centre=centre,radius=radius)
    assert_allclose(density.sum()/17**2,1.,rtol=0,atol=2e-10)
    p,q,w=S.disk_chords(24,centre=centre,radius=radius)
    actual=w@S.P.ray_integrals(p,q,lambda x,y:x*x+y*y)
    assert_allclose(actual,centre[0]**2+centre[1]**2+radius**2/2,rtol=2e-12,atol=1e-13)


def test_physical_response_does_not_require_a_constant_in_trial_space():
    # One trial direction in R^3, with the physical constant outside its span.
    G=np.array([[1.,2.,0.],[0.,1.,1.]])
    Phi=np.array([[1.],[0.],[0.]])
    T=np.array([[0.,0.,1.]])
    K,B,_=tutorial.galerkin_pairings(Phi.T@Phi,G@Phi,T@Phi)
    v=G@np.ones(3)
    X=tutorial.constrained_sola(K,B,.5,v)
    assert_allclose(X@v,[1.],rtol=0,atol=1e-14)
    assert not np.allclose(X@(G@Phi@Phi.T@np.ones(3)),1.)


def test_rejects_incompatible_or_infeasible_finite_problems():
    with pytest.raises(ValueError):tutorial.constrained_sola(np.eye(2),np.ones((1,3)),.1,np.ones(2))
    with pytest.raises(ValueError):tutorial.constrained_sola(np.eye(2),np.ones((1,2)),.1,np.zeros(2))
    with pytest.raises(np.linalg.LinAlgError):tutorial.galerkin_pairings(np.ones((2,2)),np.ones((3,2)),np.ones((1,2)))


def test_native_operator_route_matches_exact_data_representer_span():
    ex=tutorial.Experiment(centres=((.3,.65),))
    for name in ('p','q','d0','d','v'):
        setattr(ex,name,getattr(ex,name)[:3])
    native=tutorial.native_system(ex,nodes=16,target_nodes=12)
    # The finite span of all data representers must reproduce K and B exactly.
    K,B,_=tutorial.galerkin_pairings(native['K'],native['K'],native['B'])
    assert_allclose(K,native['K'],rtol=3e-12,atol=3e-13)
    assert_allclose(B,native['B'],rtol=3e-12,atol=3e-13)
    X=tutorial.constrained_sola(K,B,.2,ex.v)
    assert_allclose(X@ex.v,[1.],rtol=0,atol=3e-13)
