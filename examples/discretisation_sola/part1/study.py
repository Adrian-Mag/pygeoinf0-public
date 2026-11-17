"""Shared implementation for presentation part 1; finite realizations stay explicit.

The pixel method includes its L2 mass. The Galerkin and native methods use the
SAME restriction H2 space and constrained objective. Kernel translates are one
dense conforming trial basis, not a claim about all finite-element algorithms.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import scipy.linalg as la

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sola_demo as S

CENTRES=((.3,.65),(.7,.3),(.5,.5))
RADIUS=S.RADIUS
LABELS=('positive anomaly','negative anomaly','central disk')
PIXELS=(20,40,80,160,320,640,1280,2560)
TRIALS=(3,5,9,17,33,65)


def constrained_sola(K,B,tau,v):
    r"""Minimize each $x^T(K+\tau I)x-2b^Tx$ with $v^Tx=1$.

    B has targets in rows. Return X with one data transformation per row. The
    common Cholesky factor is reused for all right-hand sides and normalization.
    """
    K,B,v=np.asarray(K,float),np.atleast_2d(np.asarray(B,float)),np.asarray(v,float)
    n=len(v)
    if K.shape!=(n,n) or B.ndim!=2 or B.shape[1]!=n or v.shape!=(n,):
        raise ValueError('incompatible data-space matrix shapes')
    if tau<0 or not np.isfinite(tau):raise ValueError('finite nonnegative penalty required')
    factor=la.cho_factor((K+K.T)/2+tau*np.eye(n),lower=True)
    Z=la.cho_solve(factor,np.column_stack([B.T,v]))
    W,a=Z[:,:-1],Z[:,-1]
    den=float(v@a)
    if den<=0 or not np.isfinite(den):raise ValueError('constant response is infeasible')
    return (W+a[:,None]*((1-v@W)/den)[None,:]).T


def galerkin_pairings(M,E,U):
    r"""Return $E M^{-1}E^T$, $U M^{-1}E^T$, diag$(U M^{-1}U^T)$.

    No full continuum ray/ray Gram is assembled here. M is the actual Hilbert
    Gram of the finite trial basis, E and U are data and target evaluations.
    """
    M,E,U=np.asarray(M,float),np.asarray(E,float),np.atleast_2d(np.asarray(U,float))
    if M.ndim!=2 or M.shape[0]!=M.shape[1] or E.shape[1]!=len(M) or U.shape[1]!=len(M):
        raise ValueError('incompatible trial-space shapes')
    factor=la.cho_factor(M,lower=True)
    Z=la.cho_solve(factor,np.column_stack([E.T,U.T]))
    K=E@Z[:,:len(E)]
    B=U@Z[:,:len(E)]
    tt=np.einsum('ij,ji->i',U,Z[:,len(E):])
    return (K+K.T)/2,B,tt


@dataclass
class Experiment:
    centres: tuple=CENTRES
    radius: float=RADIUS
    lam: float=S.LAMBDA
    length_scale: float=.1

    def __post_init__(self):
        self.src,self.stations,self.p,self.q=S.P.geometry()
        self.d0,self.d=S.P.data()
        self.v=la.norm(self.q-self.p,axis=1)
        self.truth=np.array([w@S.P.ray_integrals(a,b,S.P.true_model,nodes=100)
                            for c in self.centres for a,b,w in [S.disk_chords(64,c,self.radius)]])

    @property
    def target_l2_norm2(self):return 1/(np.pi*self.radius**2)

    def measures(self,target_nodes=32):
        import pygeoinf0 as pg
        rays=[pg.Segment(a,b) for a,b in zip(self.p,self.q)]
        targets=[]
        for c in self.centres:
            a,b,w=S.disk_chords(target_nodes,c,self.radius)
            targets.append(pg.SignedMeasure([(float(wi),pg.LebesgueMeasure(pg.Segment(ai,bi)))
                                           for ai,bi,wi in zip(a,b,w)]))
        return rays,targets


def diagnostics(X,ex):
    """Report the original continuum-ray delivery, rather than a pixel truth fit."""
    delivered=X@ex.d0;estimate=X@ex.d
    return dict(estimate=estimate.tolist(),delivered_truth=delivered.tolist(),
                target_truth=ex.truth.tolist(),target_bias=(delivered-ex.truth).tolist(),
                noise_draw=(estimate-delivered).tolist(),std=(S.P.SIGMA*la.norm(X,axis=1)).tolist(),
                constant_response=(X@ex.v).tolist(),weight_norm=la.norm(X,axis=1).tolist(),
                weight_separation=float(la.norm(X[0]-X[1])) if len(X)>1 else 0.)


def pixel_system(ex,n):
    """Exact ray/cell integrals, diagonal L2 mass elimination and all fixed targets."""
    start=time.perf_counter()
    E=S.ray_matrix(ex.p,ex.q,n)
    assembly=time.perf_counter()-start
    start=time.perf_counter()
    targets=np.array([S.disk_pixel_density(n,c,ex.radius) for c in ex.centres])
    target_time=time.perf_counter()-start
    start=time.perf_counter()
    K=(E@E.T).toarray()*n*n  # E M_h^-1 E^T, M_h=h^2 I; no huge dense mass matrix.
    B=np.asarray(E@targets.T).T
    gram_time=time.perf_counter()-start
    memory=E.data.nbytes+E.indices.nbytes+E.indptr.nbytes+targets.nbytes+K.nbytes+B.nbytes
    return dict(K=K,B=B,tt=np.full(len(ex.centres),ex.target_l2_norm2),E=E,
                target_density=targets,dimension=n*n,n=n,h=1/n,
                timings=dict(data_assembly=assembly,target_assembly=target_time,
                             metric_and_gram=gram_time),matrix_bytes=int(memory))


def sobolev_objects(ex,target_nodes=32):
    import pygeoinf0 as pg
    H=pg.Sobolev(pg.Box((0,1),(0,1)),2,length_scale=ex.length_scale)
    rays,targets=ex.measures(target_nodes)
    return H,H.riesz(rays),H.riesz(targets)


def native_system(ex,nodes=24,target_nodes=32):
    """Native G G* and T G*; never introduces a finite model trial basis."""
    import pygeoinf0 as pg
    start=time.perf_counter()
    with S.quadrature_nodes(nodes):
        H,F,T=sobolev_objects(ex,target_nodes)
        G=pg.Functionals(F,name='G');Q=pg.Functionals(T,name='T')
    setup=time.perf_counter()-start
    start=time.perf_counter()
    with S.quadrature_nodes(nodes):
        K=np.asarray((G@G.adjoint).matrix())
    gram_time=time.perf_counter()-start
    start=time.perf_counter()
    with S.quadrature_nodes(nodes):
        B=np.asarray((Q@G.adjoint).matrix())
        tt=np.diag(Q.gram()).copy()
    target_time=time.perf_counter()-start
    return dict(H=H,F=F,T=T,G=G,Q=Q,K=K,B=B,tt=tt,nodes=nodes,target_nodes=target_nodes,
                dimension=None,h=None,timings=dict(data_assembly=setup,
                  target_assembly=target_time,metric_and_gram=gram_time),
                matrix_bytes=int(K.nbytes+B.nbytes+tt.nbytes))


def galerkin_system(ex,n,nodes=24,target_nodes=32):
    """Fully finite constrained objective from conforming H2 kernel translates.

    The exact physical response v=G1 is prescribed independently of trial-space
    representation. Also return its projected counterpart to expose the difference.
    """
    start=time.perf_counter()
    H,F,T=sobolev_objects(ex,target_nodes)
    z=np.linspace(0,1,n);xx,yy=np.meshgrid(z,z)
    Phi=H.riesz(np.column_stack([xx.ravel(),yy.ravel()]))
    with S.quadrature_nodes(nodes):
        M=np.asarray(H.inner(Phi,Phi))
        E=np.asarray(H.inner(F,Phi))
    assembly=time.perf_counter()-start
    start=time.perf_counter()
    with S.quadrature_nodes(nodes):U=np.asarray(H.inner(T,Phi))
    target_time=time.perf_counter()-start
    start=time.perf_counter()
    # <k_z,1>_H=1 is known although the restriction-space norm of 1 is not needed.
    # Add that pairing as one RHS so the diagnostic reuses the same mass factor.
    K,B_aug,tt_aug=galerkin_pairings(M,E,np.vstack([U,np.ones(n*n)]))
    B,tt,projected_v=B_aug[:-1],tt_aug[:-1],B_aug[-1]
    gram_time=time.perf_counter()-start
    return dict(K=K,B=B,tt=tt,projected_v=projected_v,n=n,h=1/(n-1),dimension=n*n,
                timings=dict(data_assembly=assembly,target_assembly=target_time,
                             metric_and_gram=gram_time),
                matrix_bytes=int(M.nbytes+E.nbytes+U.nbytes+K.nbytes+B.nbytes))


def solve_system(system,ex,tau):
    start=time.perf_counter()
    X=constrained_sola(system['K'],system['B'],tau,ex.v)
    timings=dict(system['timings'],solve=time.perf_counter()-start)
    return X,dict(timings=timings,total_seconds=sum(timings.values()),
                  matrix_bytes=system['matrix_bytes'],**diagnostics(X,ex))


def native_reference(ex,nodes=24,target_nodes=32):
    system=native_system(ex,nodes,target_nodes)
    # Stationary restriction-space kernel and congruent interior disks: same norm.
    if not np.allclose(system['tt'],system['tt'][0],rtol=2e-10,atol=2e-12):
        raise ArithmeticError('congruent targets unexpectedly have unequal dual norms')
    tau=ex.lam*system['tt'][0]
    X,row=solve_system(system,ex,tau)
    return system,X,tau,row


def compare_to_reference(X,ref,Xref):
    delta=X-Xref
    return dict(weight_error=float(la.norm(delta)/la.norm(Xref)),
                delivered_functional_error=float(np.sqrt(max(np.trace(delta@ref['K']@delta.T),0)
                         /np.sum(ref['tt']))),
                mismatch=np.sqrt(np.maximum(ref['tt']-2*np.sum(X*ref['B'],axis=1)
                         +np.einsum('ij,jk,ik->i',X,ref['K'],X),0)/ref['tt']).tolist())


def scalar_result(row):
    return {k:v for k,v in row.items() if k not in ('X','K','B','E','target_density','H','F','T','G','Q')}


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(path):return json.loads(Path(path).read_text())
