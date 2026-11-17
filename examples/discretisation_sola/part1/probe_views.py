"""Physical measures and responses to a common unit-peak test model.

These are untimed display diagnostics. They do not change the SOLA objective,
its weights, or its physical observations. Matplotlib is optional until plotting.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.special import k1

import study as F


def probe_values(points, centre, length_scale):
    r"""Evaluate $q_z(y)=\rho K_1(\rho)$, $\rho=|y-z|/\ell$, with $q_z(z)=1$.

    This is the unit-peak reproducing kernel of the restriction H2 space in 2D.
    It is a known test model, not an assumption about the unknown model.
    """
    if not np.isfinite(length_scale) or length_scale<=0:
        raise ValueError('positive finite probe length scale required')
    points,centre=np.asarray(points,float),np.asarray(centre,float)
    if points.shape[-1:]!=(2,) or centre.shape!=(2,):
        raise ValueError('two-dimensional points and centre required')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(centre)):
        raise ValueError('finite probe coordinates required')
    rho=np.linalg.norm(points-centre,axis=-1)/length_scale
    out=np.ones_like(rho)
    positive=rho>0
    out[positive]=rho[positive]*k1(rho[positive])
    return out


def probe_responses(ex, points, *, nodes=24, target_nodes=32, block=64):
    r"""Compute $G_iq_z$ and $T_aq_z$ on the original rays and disk targets.

    Reproduction gives $G_iq_z=g_i(z)/K(z,z)$ and $T_aq_z=t_a(z)/K(z,z)$.
    Fresh small point families bound temporary pairing arrays. Physical disk
    integration uses the same chord quadrature as the original study.
    """
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.all(np.isfinite(points)):
        raise ValueError('finite (n,2) probe centres required')
    if np.any((points<0)|(points>1)):
        raise ValueError('probe centres must lie in the unit box')
    if not isinstance(block,(int,np.integer)) or block<1:
        raise ValueError('positive integer block size required')
    H,rays,targets=F.sobolev_objects(ex,target_nodes)
    anchor=H.riesz(np.array([.5,.5]))
    peak=H.inner(anchor,anchor)  # constant diagonal of this restricted stationary kernel
    result=dict(points=points,ray=np.empty((len(ex.p),len(points))),
                target=np.empty((len(ex.centres),len(points))),peak=float(peak))
    with F.S.quadrature_nodes(nodes):
        for start in range(0,len(points),block):
            stop=min(start+block,len(points))
            probes=H.riesz(points[start:stop])
            result['ray'][:,start:stop]=H.inner(rays,probes)/peak
            result['target'][:,start:stop]=H.inner(targets,probes)/peak
    return result


def response_grid(ex, output_dir, n=60):
    """Reuse a shared, input/source-bound response grid across untimed plots."""
    output_dir=Path(output_dir);output_dir.mkdir(exist_ok=True,parents=True)
    z=(np.arange(n)+.5)/n;xx,yy=np.meshgrid(z,z)
    points=np.column_stack([xx.ravel(),yy.ravel()])
    key=dict(n=n,centres=ex.centres,radius=ex.radius,length_scale=ex.length_scale,
             p=ex.p.tolist(),q=ex.q.tolist(),ray_nodes=24,target_chords=32,
             source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    signature=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    path=output_dir/'probe_responses.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as stored:
            if str(stored['signature'])==signature:
                return {k:stored[k].copy() for k in ('points','ray','target','peak')}
    result=probe_responses(ex,points)
    np.savez_compressed(path,**result,signature=signature)
    return result


def plot_probe(ex, response):
    """Show the actual unit-peak test model and its horizontal profile."""
    import matplotlib.pyplot as plt
    points=response['points'];n=int(np.sqrt(len(points)));centre=ex.centres[0]
    values=probe_values(points,centre,ex.length_scale).reshape(n,n)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    im=ax[0].imshow(values,origin='lower',extent=[0,1,0,1],vmin=0,vmax=1,cmap='magma')
    ax[0].plot(*centre,'w+',ms=10);ax[0].set_title('The known unit-peak test anomaly')
    fig.colorbar(im,ax=ax[0],shrink=.8,label='model value')
    x=np.unique(np.r_[np.linspace(0,1,300),centre[0]])
    line=np.column_stack([x,np.full_like(x,centre[1])])
    ax[1].plot(x,probe_values(line,centre,ex.length_scale));ax[1].axvline(centre[0],color='.5',ls=':')
    ax[1].set(xlabel='model coordinate y_x',ylabel='unit-peak model value',ylim=(0,1.05),
              title=f'Horizontal profile; chosen length {ex.length_scale:g}')
    ax[0].set(xlabel='model coordinate y_x',ylabel='model coordinate y_y');ax[0].grid(False)
    return fig


def _map(ax, values, response, centre, radius, title, *, limit=None, sequential=False):
    import matplotlib.pyplot as plt
    n=int(np.sqrt(len(response['points'])))
    if limit is None:limit=max(float(np.max(np.abs(values))),1e-15)
    im=ax.imshow(np.asarray(values).reshape(n,n),origin='lower',extent=[0,1,0,1],
                 cmap='viridis' if sequential else 'RdBu_r',vmin=0 if sequential else -limit,vmax=limit)
    ax.add_patch(plt.Circle(centre,radius,fill=False,color='k',lw=1))
    ax.set_title(title,fontsize=10);ax.set(xlabel='probe centre z_x',ylabel='probe centre z_y')
    ax.grid(False)
    return im


def plot_physical_and_responses(ex, response, weights, title, target=0):
    """Show exact supports and their continuum responses, including signed rays."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    weights=np.asarray(weights,float);centre=ex.centres[target]
    near=np.argmin(np.linalg.norm(response['points']-centre,axis=1))
    ray_index=int(np.argmax(response['ray'][:,near]))
    delivered=weights@response['ray'];wanted=response['target'][target]
    fig,ax=plt.subplots(2,3,figsize=(13,7.6),constrained_layout=True)
    fig.suptitle(title,fontsize=13)
    a,b=ex.p[ray_index],ex.q[ray_index]
    ax[0,0].plot([a[0],b[0]],[a[1],b[1]],color='#2563eb',lw=2.5)
    ax[0,0].scatter([a[0],b[0]],[a[1],b[1]],s=15,color='k')
    ax[0,0].set_title(f'Exact sensitivity: ray {ray_index}',fontsize=10)
    ax[0,0].text(.03,.03,f'Integral along the segment\nMeasure mass = length {ex.v[ray_index]:.3f}',
                 transform=ax[0,0].transAxes,fontsize=9)
    ax[0,1].add_patch(plt.Circle(centre,ex.radius,facecolor='#fecaca',edgecolor='#b91c1c',lw=2))
    ax[0,1].set_title('Exact target: uniform disk average',fontsize=10)
    ax[0,1].text(.03,.03,f'Area density {ex.target_l2_norm2:.3f} inside disk\nTotal mass = 1',
                 transform=ax[0,1].transAxes,fontsize=9)
    largest=max(float(np.max(np.abs(weights))),1e-15)
    collection=LineCollection(np.stack([ex.p,ex.q],axis=1),cmap='RdBu_r',
        norm=Normalize(-largest,largest),linewidths=.15+1.8*np.sqrt(np.abs(weights)/largest),alpha=.7)
    collection.set_array(weights);ax[0,2].add_collection(collection)
    ax[0,2].set_title('Delivered measure: signed weighted rays',fontsize=10)
    ax[0,2].text(.03,.03,f'All {len(weights)} exact segments\nTotal signed mass = {weights@ex.v:.6f}',
                 transform=ax[0,2].transAxes,fontsize=9,
                 bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    fig.colorbar(collection,ax=ax[0,2],shrink=.7,label='signed ray weight')
    for a in ax[0]:
        a.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel='physical x',ylabel='physical y');a.grid(False)
    ax[0,0].add_patch(plt.Circle(centre,ex.radius,fill=False,color='.4',ls=':'))
    ax[0,2].add_patch(plt.Circle(centre,ex.radius,fill=False,color='k'))
    im=_map(ax[1,0],response['ray'][ray_index],response,centre,ex.radius,
            'Response: ray integral of the moving probe',sequential=True)
    fig.colorbar(im,ax=ax[1,0],shrink=.7,label='ray integral (model x length)')
    limit=max(float(np.max(np.abs(delivered))),float(np.max(np.abs(wanted))))
    for a,values,label in ((ax[1,1],wanted,'Response: disk average of the moving probe'),
                           (ax[1,2],delivered,'Response: weighted ray data for the probe')):
        im=_map(a,values,response,centre,ex.radius,label,limit=limit)
    fig.colorbar(im,ax=ax[1,1:].tolist(),shrink=.7,label='target / delivered response (model units)')
    return fig


def plot_response_evolution(ex, response, weights, labels, title, *, reference=None, target=0):
    """Compare actual continuum-ray questions on one fixed family of probes."""
    import matplotlib.pyplot as plt
    panels=[(label,np.asarray(w)@response['ray']) for label,w in zip(labels,weights)]
    if reference is not None:panels.append(('Native continuum response',reference@response['ray']))
    panels.append(('Requested uniform-disk response',response['target'][target]))
    limit=max(float(np.max(np.abs(v))) for _,v in panels)
    fig,axes=plt.subplots(1,len(panels),figsize=(3.1*len(panels),3.7),constrained_layout=True)
    fig.suptitle(title,fontsize=12)
    for a,(label,values) in zip(np.atleast_1d(axes),panels):
        im=_map(a,values,response,ex.centres[target],ex.radius,label,limit=limit)
    fig.colorbar(im,ax=np.atleast_1d(axes).tolist(),shrink=.7,label='response to unit-peak anomaly (model units)')
    return fig


def plot_three_targets(ex, response, weights):
    """Display requested and delivered response maps for all physical targets."""
    import matplotlib.pyplot as plt
    delivered=np.asarray(weights)@response['ray']
    limit=max(float(np.max(np.abs(delivered))),float(np.max(np.abs(response['target']))))
    fig,ax=plt.subplots(2,len(ex.centres),figsize=(12,7),constrained_layout=True,squeeze=False)
    for j,(centre,label) in enumerate(zip(ex.centres,F.LABELS)):
        for i,(values,prefix) in enumerate(((response['target'][j],'Requested'),(delivered[j],'Delivered'))):
            im=_map(ax[i,j],values,response,centre,ex.radius,f'{prefix}: {label}',limit=limit)
    fig.colorbar(im,ax=ax.ravel().tolist(),shrink=.7,label='response to the same unit-peak anomaly')
    fig.suptitle('Three physical questions, compared with one known moving test anomaly')
    return fig
