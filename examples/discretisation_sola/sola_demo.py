"""Numerics for the two SOLA discretisation notebooks; no package API is changed.

The physical target is always the mean over one fixed disk. Pixel SOLA uses the
area-weighted L2 pairing. Continuum SOLA uses the dual pairing induced by H^2.
Both deliver the original ray functional ``sum_i x_i integral_ray_i m``.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import scipy.linalg as la
from scipy.integrate import quad
import scipy.sparse as sp

import problem as P

CENTRE, RADIUS = (.3, .65), .12
AREA = np.pi*RADIUS**2
LAMBDA = .1


def ray_matrix(p, q, n):
    """Exact ray lengths in piecewise-constant pixels; columns ordered y then x."""
    rows, cols, vals = [], [], []
    grid = np.linspace(0, 1, n+1)
    for i, (a, b) in enumerate(zip(p, q)):
        dx, dy = b-a
        ts = [0., 1.]
        if dx != 0:
            ts.extend((grid-a[0])/dx)
        if dy != 0:
            ts.extend((grid-a[1])/dy)
        ts = np.unique(np.clip(ts, 0, 1))
        mid = (ts[:-1]+ts[1:])/2
        ix = np.clip((a[0]+mid*dx)*n, 0, n-1).astype(int)
        iy = np.clip((a[1]+mid*dy)*n, 0, n-1).astype(int)
        lengths = np.diff(ts)*np.hypot(dx, dy)
        keep = lengths > 0
        rows.extend([i]*int(keep.sum()))
        cols.extend((iy[keep]*n+ix[keep]).tolist())
        vals.extend(lengths[keep].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(p), n*n))


def disk_pixel_density(n, centre=CENTRE, radius=RADIUS):
    """L2 projection of the fixed normalized disk density, using disk/cell areas.

    The represented target has exact unit mass to numerical integration accuracy;
    the physical disk is not shrunk with the mesh. No centre-in-disk approximation.
    """
    cx, cy = centre
    area = np.pi*radius**2
    out = np.zeros((n, n))
    ix0, ix1 = max(0, int((cx-radius)*n)), min(n, int((cx+radius)*n)+1)
    iy0, iy1 = max(0, int((cy-radius)*n)), min(n, int((cy+radius)*n)+1)
    for iy in range(iy0, iy1):
        y0, y1 = iy/n-cy, (iy+1)/n-cy
        for ix in range(ix0, ix1):
            x0, x1 = ix/n-cx, (ix+1)/n-cx
            if max(abs(x0), abs(x1))**2+max(abs(y0), abs(y1))**2 <= radius**2:
                out[iy, ix] = 1/area
                continue
            xmin = 0. if x0 <= 0 <= x1 else min(abs(x0), abs(x1))
            ymin = 0. if y0 <= 0 <= y1 else min(abs(y0), abs(y1))
            if xmin*xmin+ymin*ymin >= radius**2:
                continue
            lo, hi = max(x0, -radius), min(x1, radius)
            if hi <= lo:
                continue
            def width(x):
                s = np.sqrt(max(radius**2-x*x, 0.))
                return max(0., min(y1, s)-max(y0, -s))
            cuts = []
            for y in (y0, y1):
                if abs(y) < radius:
                    z = np.sqrt(radius**2-y*y)
                    cuts.extend([a for a in (-z, z) if lo < a < hi])
            overlap = quad(width, lo, hi, points=sorted(set(cuts)), epsabs=1e-14, epsrel=2e-12)[0]
            out[iy, ix] = overlap*n*n/area
    return out.ravel()


def disk_chords(nodes=32, centre=CENTRE, radius=RADIUS):
    """A convergent area quadrature: horizontal chord integrals with y=cy+r sin(theta).

    This approximates the SAME disk functional in the Sobolev dual; it is not a
    change of target. ``w @ ray_integrals(p,q,f)`` approximates its area mean.
    """
    z, w = np.polynomial.legendre.leggauss(nodes)
    theta, w = z*np.pi/2, w*np.pi/2
    half, y = radius*np.cos(theta), centre[1]+radius*np.sin(theta)
    p = np.column_stack([centre[0]-half, y])
    q = np.column_stack([centre[0]+half, y])
    return p, q, w*radius*np.cos(theta)/(np.pi*radius**2)


def target_truth(nodes=64):
    p, q, w = disk_chords(nodes)
    return float(w@P.ray_integrals(p, q, P.true_model, nodes=100))


def sola_weights(K, b, tau, response=None):
    r"""Minimize $x^T Kx-2b^Tx+\tau x^Tx$, optionally with $v^Tx=1$.

    A Cholesky solve, followed by the finite-rank response correction from the
    thesis's constrained SOLA derivation. No eigenvalue clipping or ridge tuning.
    ``tau=lambda * ||target||^2`` corresponds to the normalized objective used here.
    """
    K, b = np.asarray(K, float), np.asarray(b, float)
    if tau < 0 or not np.isfinite(tau) or not np.isfinite(K).all() or not np.isfinite(b).all():
        raise ValueError("finite inputs and nonnegative tau are required")
    H = (K+K.T)/2+tau*np.eye(len(b))
    factor = la.cho_factor(H, lower=True, check_finite=True)
    x = la.cho_solve(factor, b)
    if response is not None:
        v = np.asarray(response, float)
        a = la.cho_solve(factor, v)
        den = float(v@a)
        if den <= 0 or not np.isfinite(den):
            raise ValueError("unit response is infeasible for this response vector")
        x += a*((1-v@x)/den)
    return x


def estimator_diagnostics(x, d_clean, d, target, sigma):
    """Noise propagation and the exact transformed MDR for this known truth."""
    delivered, estimate = float(x@d_clean), float(x@d)
    return dict(delivered_truth=delivered, estimate=estimate, target_truth=float(target),
                target_bias=delivered-target, noise_draw=estimate-delivered,
                std=float(sigma*np.linalg.norm(x)), weight_norm=float(np.linalg.norm(x)))


def pixel_study(grids=(20, 40, 80, 160, 320, 640, 1280, 2560), lam=LAMBDA):
    """Refine pixels with fixed physical target and fixed dimensionless trade-off."""
    _, _, p, q = P.geometry()
    d0, d = P.data()
    v, target = np.linalg.norm(q-p, axis=1), target_truth()
    tnorm2, rows = 1/AREA, []       # full continuum L2 target norm, fixed on every mesh
    for n in grids:
        start = time.perf_counter()
        G, t = ray_matrix(p, q, n), disk_pixel_density(n)
        K, b = (G@G.T).toarray()*n*n, np.asarray(G@t)
        x = sola_weights(K, b, lam*tnorm2, v)
        xu = sola_weights(K, b, lam*tnorm2)
        density = np.asarray(G.T@x)*n*n
        xc = (np.arange(n)+.5)/n
        X, Y = np.meshgrid(xc, xc)
        dp = G@P.true_model(X, Y).ravel()
        row = dict(n=n, x=x, xu=xu, density=density.reshape(n,n),
                   target_density=t.reshape(n,n),
                   kernel_norm=float(np.sqrt(x@K@x)),
                   relative_mismatch=float(np.sqrt(max(tnorm2-2*x@b+x@K@x, 0)/tnorm2)),
                   projected_relative_mismatch=float(np.sqrt(np.mean((density-t)**2)/tnorm2)),
                   mass=float(v@x), pixel_forward_discrepancy=float(x@(dp-d0)),
                   target_mass=float(t.sum()/n**2), seconds=time.perf_counter()-start,
                   unconstrained=estimator_diagnostics(xu,d0,d,target,P.SIGMA),
                   **estimator_diagnostics(x,d0,d,target,P.SIGMA))
        rows.append(row)
        print(f"pixels {n:4d}: estimate={row['estimate']:.6f}, delivered={row['delivered_truth']:.6f}, "
              f"std={row['std']:.4g}, kernel L2 norm={row['kernel_norm']:.4g}, {row['seconds']:.1f}s", flush=True)
    return rows


@contextmanager
def quadrature_nodes(nodes):
    from pygeoinf0.realisations.rkhs import kernels
    old = kernels.NODES
    kernels.NODES = nodes
    try:
        yield
    finally:
        kernels.NODES = old


def sobolev_gram(nodes, target_nodes=32):
    """H^2 representers of the original rays and of a chord quadrature of the disk."""
    import pygeoinf0 as pg
    _, _, p, q = P.geometry()
    a,b,w = disk_chords(target_nodes)
    mu = pg.SignedMeasure([(float(wi),pg.LebesgueMeasure(pg.Segment(ai,bi))) for ai,bi,wi in zip(a,b,w)])
    H = pg.Sobolev(pg.Box((0,1),(0,1)), 2, length_scale=.1)
    with quadrature_nodes(nodes):
        F = H.riesz([pg.Segment(ai,bi) for ai,bi in zip(p,q)]+[mu])
        J = np.asarray(H.inner(F,F))
    return dict(H=H,F=F,K=J[:-1,:-1],b=J[:-1,-1],tt=float(J[-1,-1]),nodes=nodes,target_nodes=target_nodes)


def sobolev_study(levels=(4,6,8,10,16,24), target_nodes=32, lam=LAMBDA):
    """Refine numerical integrals of one fixed continuum problem; reference is not a proof of accuracy."""
    _,_,p,q = P.geometry()
    d0,d = P.data()
    v,target = np.linalg.norm(q-p,axis=1),target_truth()
    ref = sobolev_gram(max(levels),target_nodes)
    tnorm2 = ref['tt']
    reference_x = sola_weights(ref['K'],ref['b'],lam*tnorm2,v)
    rows=[]
    for k in levels:
        start=time.perf_counter()
        g = ref if k==max(levels) else sobolev_gram(k,target_nodes)
        lo=float(la.eigvalsh(g['K'],subset_by_index=[0,0])[0])
        knorm=float(np.linalg.norm(ref['K'],2)) if not rows else rows[0]['reference_K_norm']
        row=dict(nodes=k,gram_error=float(np.linalg.norm(g['K']-ref['K'])/np.linalg.norm(ref['K'])),
                 min_eigenvalue=lo,reference_K_norm=knorm,geometry=g)
        if lo < -1e-10*knorm:
            row.update(status='underresolved: negative Gram eigenvalue',seconds=time.perf_counter()-start)
        else:
            x=sola_weights(g['K'],g['b'],lam*tnorm2,v)
            norm2=float(x@g['K']@x)
            row.update(status='computed',x=x,x_error=float(np.linalg.norm(x-reference_x)/np.linalg.norm(reference_x)),
                       mass=float(v@x),kernel_norm=float(np.sqrt(max(norm2,0))),
                       relative_mismatch=float(np.sqrt(max(g['tt']-2*x@g['b']+norm2,0)/tnorm2)),
                       seconds=time.perf_counter()-start,
                       **estimator_diagnostics(x,d0,d,target,P.SIGMA))
        rows.append(row)
        print(f"Sobolev nodes {k:2d}: Gram rel.error={row['gram_error']:.2e}, {row['status']}",flush=True)
    ref.update(x=reference_x,tnorm2=tnorm2)
    return rows,ref


def sobolev_fields(ref,n=64):
    """Values of smooth Riesz functions; not Lebesgue averaging-density kernels."""
    import pygeoinf0 as pg
    H,F=ref['H'],ref['F']
    G=pg.Functionals(F[:-1])
    r=G.adjoint(G.codomain(ref['x']))
    t=F[-1]
    z=(np.arange(n)+.5)/n
    X,Y=np.meshgrid(z,z)
    with quadrature_nodes(ref['nodes']):
        rv,tv=r(X,Y),t(X,Y)
    return X,Y,rv,tv,r,t


def json_rows(rows):
    """Scalar audit without copying large realisations into JSON."""
    return [{k:v for k,v in r.items() if k not in ('x','xu','density','target_density','geometry')}
            for r in rows]


def save_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
