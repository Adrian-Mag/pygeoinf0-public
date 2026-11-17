"""Physical integration contracts for the presentation's common-probe maps."""
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest

EXAMPLE=Path(__file__).parents[1]/'examples'/'discretisation_sola'
sys.path.insert(0,str(EXAMPLE/'part1'))
sys.path.insert(0,str(EXAMPLE))
import study as F
import probe_views as V


def test_unit_peak_profile_is_the_normalized_H2_reproducing_kernel():
    ex=F.Experiment()
    H,_,_=F.sobolev_objects(ex,12)
    centre=np.array([.3,.65])
    points=np.array([centre,centre+[.1,0],centre+[0,.1],centre+[.2,0]])
    k=H.riesz(centre)
    expected=k(points[:,0],points[:,1])/H.inner(k,k)
    actual=V.probe_values(points,centre,ex.length_scale)
    assert_allclose(actual,expected,rtol=2e-13,atol=2e-14)
    assert actual[0]==1
    assert_allclose(actual[1],actual[2],rtol=0,atol=2e-15)
    assert 0<actual[3]<actual[1]<1


def test_response_maps_equal_direct_ray_and_disk_integrals_and_weighted_data():
    ex=F.Experiment(centres=((.3,.65),(.7,.3)))
    for name in ('p','q','v','d','d0'):setattr(ex,name,getattr(ex,name)[:4])
    points=np.array([[.55,.55],[.15,.2]])
    result=V.probe_responses(ex,points,block=1)
    again=V.probe_responses(ex,points,block=2)
    assert_allclose(result['ray'],again['ray'],rtol=2e-13,atol=2e-14)
    assert_allclose(result['target'],again['target'],rtol=2e-13,atol=2e-14)
    weights=np.array([.6,-.4,.7,.2]);weights/=weights@ex.v
    for j,z in enumerate(points):
        def model(x,y):
            xy=np.stack(np.broadcast_arrays(x,y),axis=-1)
            return V.probe_values(xy,z,ex.length_scale)
        direct=F.S.P.ray_integrals(ex.p,ex.q,model,nodes=240)
        assert_allclose(result['ray'][:,j],direct,rtol=2e-7,atol=2e-9)
        assert_allclose(weights@result['ray'][:,j],weights@direct,rtol=2e-7,atol=2e-9)
        for a,c in enumerate(ex.centres):
            p,q,w=F.S.disk_chords(64,c,ex.radius)
            target=w@F.S.P.ray_integrals(p,q,model,nodes=160)
            assert_allclose(result['target'][a,j],target,rtol=2e-7,atol=2e-9)


def test_probe_edge_cases_are_explicit():
    for ell in (0.,-1.,np.nan):
        with pytest.raises(ValueError):V.probe_values([[.5,.5]],[.5,.5],ell)
    with pytest.raises(ValueError):V.probe_values([[np.nan,.5]],[.5,.5],.1)
    with pytest.raises(ValueError):V.probe_values([[.5,.5]],[np.inf,.5],.1)
    ex=F.Experiment(centres=((.3,.65),))
    for points in (np.ones((2,3)),[[np.nan,0]],[[1.1,.2]]):
        with pytest.raises(ValueError):V.probe_responses(ex,points)
    with pytest.raises(ValueError):V.probe_responses(ex,[[.5,.5]],block=0)
    empty=V.probe_responses(ex,np.empty((0,2)))
    assert empty['ray'].shape==(len(ex.p),0)
    assert empty['target'].shape==(1,0)
