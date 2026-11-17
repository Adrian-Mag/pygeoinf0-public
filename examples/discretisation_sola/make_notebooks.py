"""Generate the requested pair of notebooks; execute with run_notebooks.py."""
from pathlib import Path
import nbformat as nbf

HERE=Path(__file__).resolve().parent
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell

common=r'''%matplotlib inline
import json, time, socket, hashlib, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sola_demo as S
P = S.P
OUT = Path('output'); OUT.mkdir(exist_ok=True)
src, sta, p, q = P.geometry()
d_clean, d = P.data()
truth = S.target_truth()
print('machine:', socket.gethostname())
print('Python:', sys.executable)
print('code commit:', subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip())
print('problem.py SHA256:', hashlib.sha256(Path('problem.py').read_bytes()).hexdigest())
print(f'{len(d)} original rays, noise sigma={P.SIGMA}; fixed disk mean={truth:.12f}')
print(f'disk centre={S.CENTRE}, radius={S.RADIUS}; fixed dimensionless lambda={S.LAMBDA}')
'''

nb1=nbf.v4.new_notebook(cells=[
md(r'''# SOLA on pixels: what changes when the pixels shrink?

This is the **same experiment** as the existing discretisation demo: the same
600 rays, smooth truth and one noisy data vector, copied unchanged in `problem.py`.
We now estimate one property with SOLA rather than reconstruct a model.

The requested property is fixed: the mean over the disk centred at $(0.3,0.65)$,
radius $0.12$. The main calculation constrains the delivered functional to return
one on a constant model. An unconstrained calculation is included for comparison.
We do not tune weights on the realised observations.

The question is whether refining a pixel representation preserves the selected
question and its interpretation. No outcome is assumed in advance.'''),
code(common),
code(r'''z=np.linspace(0,1,200); X,Y=np.meshgrid(z,z)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for a,b in zip(p,q): axs[0].plot([a[0],b[0]],[a[1],b[1]],color='.4',lw=.3,alpha=.2)
axs[0].scatter(*src.T,s=15,marker='*'); axs[0].scatter(*sta.T,s=15,marker='^')
axs[0].set_title('The original 600 rays')
im=axs[1].imshow(P.true_model(X,Y),origin='lower',extent=[0,1,0,1],cmap='RdBu_r')
axs[1].add_patch(plt.Circle(S.CENTRE,S.RADIUS,fill=False,color='k'))
axs[1].set_title('Known truth and the fixed target disk')
for ax in axs: ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1)
fig.colorbar(im,ax=axs[1]); plt.show()'''),
md(r'''## The pixel SOLA problem

Each entry of $G_h$ is the exact length of a ray in a pixel. Pixel area is $a=h^2$.
The sensitivity density is $G_{ij}/a$, and the delivered density is
$r_h=G_h^T x/a$. The target $t_h$ uses disk/pixel overlap areas, not a
centre-in-disk mask. Its integral is one on every mesh.

The data Gram and target/data pairings are
$$K_h=G_hG_h^T/a,\qquad b_h=G_ht_h.$$
We minimize
$$\frac{\|t_h-r_h\|_{L^2}^2}{\|t\|_{L^2}^2}
 +\lambda\frac{\sigma^2\|x\|^2}{\sigma_{\rm ref}^2},
 \qquad v^Tx=1,$$
where $v_i$ is the full ray length, $\sigma_{\rm ref}=\sigma=0.01$,
$\|t\|_{L^2}^2=1/(\pi R^2)$ and $\lambda=0.1$ are held fixed.
Using the full target norm rather than its projected norm only adds a constant
to the fixed-$\lambda$ objective. The solve is
$$H_h=K_h+\lambda\|t\|^2 I,\quad x_0=H_h^{-1}b_h,\quad
x=x_0+H_h^{-1}v\frac{1-v^Tx_0}{v^TH_h^{-1}v}.$$
`sola_demo.py` implements this correction; tests compare it against an independent
KKT system. There is no clipping of eigenvalues or data-misfit tuning.'''),
code(r'''start=time.perf_counter()
rows=S.pixel_study()
S.save_json(OUT/'pixel_results.json',dict(lambda_value=S.LAMBDA,rows=S.json_rows(rows)))
np.savez_compressed(OUT/'pixel_weights.npz',**{f'n{r["n"]}':r['x'] for r in rows})
print(f'Completed in {time.perf_counter()-start:.1f}s')
print('\n n     mass      target mass   estimate   delivered truth   sigma      target bias   ||r_h||L2   relative mismatch')
for r in rows:
 print(f"{r['n']:4d} {r['mass']:10.7f} {r['target_mass']:10.7f} {r['estimate']:10.5f} {r['delivered_truth']:12.5f} "
       f"{r['std']:10.4g} {r['target_bias']:11.5f} {r['kernel_norm']:11.4g} {r['relative_mismatch']:10.4g}")'''),
md(r'''## Spatial interpretation on each mesh

These are **Lebesgue density kernels for the pixel model**. The target is a disk
density; the delivered kernel is built from the pixels touched by weighted rays.
The display is clipped to one common colour range so that changing peak amplitudes
does not hide the spatial support. The printed norms and peak labels remain unclipped.'''),
code(r'''chosen=[rows[i] for i in (0,2,4,len(rows)-1)]
fig,axs=plt.subplots(1,len(chosen),figsize=(15,3.8),constrained_layout=True)
lim=2/S.AREA
for ax,r in zip(axs,chosen):
 im=ax.imshow(r['density'],origin='lower',extent=[0,1,0,1],cmap='RdBu_r',vmin=-lim,vmax=lim)
 ax.add_patch(plt.Circle(S.CENTRE,S.RADIUS,fill=False,color='k',lw=1))
 ax.set_title(f"{r['n']} x {r['n']}\nmax |density|={np.abs(r['density']).max():.1f}")
 ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im,ax=axs,shrink=.8,extend='both',label='pixel density (colour-clipped)'); plt.show()'''),
md(r'''## Weights, uncertainty and kernel mismatch

Each output is a noisy observation of its own delivered property:
$$x_h^Td=x_h^TGm+x_h^T\eta.$$
The physical delivered truth below is evaluated from the original accurately
integrated ray data, not from a pixel approximation of the truth. Thus
$x_h^Td- Tm=(x_h^TGm-Tm)+x_h^T\eta$ is checked against the actual continuum truth.'''),
code(r'''ns=np.array([r['n'] for r in rows]); xf=rows[-1]['x']
fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
plots=[('weight norm','weight_norm'),('propagated noise sigma','std'),
       ('pixel delivered-kernel L2 norm','kernel_norm'),('relative L2 target mismatch','relative_mismatch')]
for ax,(label,key) in zip(axs.ravel()[:4],plots):
 ax.loglog(ns,[r[key] for r in rows],'o-'); ax.set_title(label)
axs[1,1].semilogx(ns,[np.linalg.norm(r['x']-xf)/np.linalg.norm(xf) for r in rows],'o-')
axs[1,1].set_title('relative weight change from finest mesh')
axs[1,2].semilogx(ns,[r['target_bias'] for r in rows],'o-',label='delivered truth minus target')
axs[1,2].axhline(0,color='k',lw=.7); axs[1,2].set_title('target error for this one known model')
for ax in axs.ravel(): ax.set_xlabel('pixels per side'); ax.grid(alpha=.2)
plt.show()
fig,ax=plt.subplots(figsize=(8,4))
ax.errorbar(ns,[r['estimate'] for r in rows],yerr=[2*r['std'] for r in rows],fmt='o-',label='estimate +/- 2 propagated sigma')
ax.plot(ns,[r['delivered_truth'] for r in rows],'s--',label='actual delivered truth')
ax.axhline(truth,color='k',label='requested disk mean'); ax.set_xscale('log'); ax.legend(); plt.show()
for r in rows:
 np.testing.assert_allclose(r['estimate']-truth,r['target_bias']+r['noise_draw'],atol=2e-13)
print('Transformed-MDR identity checked on every mesh.')
print('pixel-forward error in the delivered value:',[float(f"{r['pixel_forward_discrepancy']:.3g}") for r in rows])'''),
md(r'''## Does the normalization matter?

Yes. The preceding experiment imposes unit response. Without it, a different
sequence of estimators is obtained. The following table uses the same physical
target and trade-off but omits the constraint; it should not be confused with
the constrained results.'''),
code(r'''print(' n      unconstrained estimate   std        weight norm     constrained estimate')
for r in rows:
 u=r['unconstrained']
 print(f"{r['n']:4d} {u['estimate']:18.6f} {u['std']:10.4g} {u['weight_norm']:12.5g} {r['estimate']:18.6f}")
fig,ax=plt.subplots(figsize=(8,4))
ax.semilogx(ns,[r['estimate'] for r in rows],'o-',label='unit response')
ax.semilogx(ns,[r['unconstrained']['estimate'] for r in rows],'s-',label='no constraint')
ax.axhline(truth,color='k',label='target truth'); ax.legend(); plt.show()'''),
md(r'''## What this can establish

On a fixed pixel space this is a valid finite-dimensional SOLA problem. The
observed weight and kernel changes show what the chosen question does under
refinement. A rising density-kernel norm is different from divergence of the
finite vector of data weights or from an invalid noise-propagation identity.

Zero-width ray integrals have no bounded extension to L2 of the square. Their
pixel density kernels become concentrated in thin strips. Unit integral alone
does not provide an L2 limit. A finite refinement study illustrates this mechanism
but does not prove an asymptotic limit for all ray geometries.

This is distinct from what happens to a least-squares model. Such models can
converge in L2 to zero while their ray predictions do not tend to zero, precisely
because ray integration is not continuous in L2. Thus "the limit is not in L2"
is not the general explanation. Here we inspect delivered sensitivity densities,
data weights and functional values as three separate objects.

The covariance refers to the delivered property. Neither unit response nor a small
noise standard deviation bounds error in the requested disk mean without further
information about the model. One truth/noise draw is not a coverage experiment.

The companion notebook changes the kernel-matching geometry to H2, on which the
original ray functionals are bounded. It is not simply a more accurate evaluation
of this pixel/L2 criterion.''')])

nb2=nbf.v4.new_notebook(cells=[
md(r'''# SOLA on H2 functions: refine the numerics of a fixed problem

The rays, truth, noisy observations and physical disk-average target are identical
to the pixel notebook. The model space is the restriction Sobolev space
$H^2([0,1]^2)$ with length scale $0.1$. Ray integration is a bounded functional
there. No model norm bound or probability distribution is asserted.

The inner product specifies how functionals are compared. This changes the
selection of X; the transformed observation equation remains exact for the
selected weights. The experiment refines quadrature, not the model space.'''),
code(common+r'''import pygeoinf0 as pg
print('pygeoinf0:',pg.__version__,pg.__file__)
print('settings:',pg.settings)
'''),
md(r'''## Physical functionals and their Riesz representers

For a ray datum, $G_i(m)=\int_{\ell_i}m\,ds=\langle g_i,m\rangle_H$.
For the requested property, $T(m)=\int_{\rm disk}m\,dA/(\pi R^2)
=\langle t,m\rangle_H$.

We compute $K_{ij}=\langle g_i,g_j\rangle_H$ and $b_i=\langle t,g_i\rangle_H$.
The same constrained SOLA formula as in the pixel notebook then applies.
The dimensionless trade-off is fixed at 0.1; the accurately evaluated reference
$\|t\|_H^2$ is held fixed across quadrature levels.

Disk area integration is represented by Gauss quadrature over horizontal chords
with $y=c_y+R\sin\theta$. This is a numerical approximation of one fixed disk
functional. We check this approximation separately from ray-pair integration.

The smooth functions $g_i,t,r=\sum_i x_i g_i$ are **Riesz representers**. Their
pairing is the H2 inner product: their plotted values are not spatial weights
to integrate against the model using ordinary area measure.'''),
code(r'''start=time.perf_counter()
rows,reference=S.sobolev_study()
S.save_json(OUT/'sobolev_results.json',dict(lambda_value=S.LAMBDA,rows=S.json_rows(rows),
          target_norm_squared=reference['tt'],target_chords=reference['target_nodes']))
np.savez_compressed(OUT/'sobolev_weights.npz',**{f"nodes{r['nodes']}":r['x'] for r in rows if 'x' in r})
print(f'Completed in {time.perf_counter()-start:.1f}s')
print('\n nodes   Gram rel.error  smallest eigenvalue   relative X change     estimate     std     relative mismatch')
for r in rows:
 if 'x' in r:
  print(f"{r['nodes']:5d} {r['gram_error']:13.3e} {r['min_eigenvalue']:19.3e} {r['x_error']:18.3e} "
        f"{r['estimate']:12.6f} {r['std']:9.3g} {r['relative_mismatch']:12.5g}")
 else: print(f"{r['nodes']:5d} {r['gram_error']:13.3e} {r['min_eigenvalue']:19.3e}   {r['status']}")'''),
md(r'''## Quadrature convergence

A Gram matrix must be positive semidefinite. A substantially negative eigenvalue
flags under-resolved pair integrals; those rows are not presented as valid SOLA
solutions. Agreement with the finest setting is a numerical check, not a theorem
that the reference is exact.'''),
code(r'''good=[r for r in rows if 'x' in r]
fig,axs=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
axs[0].semilogy([r['nodes'] for r in rows],[max(r['gram_error'],1e-17) for r in rows],'o-')
axs[0].set_title('Gram change from 24-node reference')
axs[1].semilogy([r['nodes'] for r in good],[max(r['x_error'],1e-17) for r in good],'o-')
axs[1].set_title('relative weight change')
axs[2].plot([r['nodes'] for r in good],[r['estimate'] for r in good],'o-',label='estimate')
axs[2].plot([r['nodes'] for r in good],[r['delivered_truth'] for r in good],'s--',label='delivered truth')
axs[2].axhline(truth,color='k',label='target truth'); axs[2].legend(fontsize=8)
for ax in axs: ax.set_xlabel('Gauss nodes per integration piece'); ax.grid(alpha=.2)
plt.show()'''),
md(r'''## The same disk target at finer target quadrature

The primary convergence study used 32 target chords. Recompute only the target
pairings with 64 chords, keeping the ray-pair quadrature at 24. This isolates
target integration error from sensitivity-kernel integration error.'''),
code(r'''fine_target=S.sobolev_gram(24,target_nodes=64)
xf=S.sola_weights(fine_target['K'],fine_target['b'],S.LAMBDA*reference['tt'],np.linalg.norm(q-p,axis=1))
target_check=dict(relative_X_change=float(np.linalg.norm(xf-reference['x'])/np.linalg.norm(reference['x'])),
                 relative_target_norm_change=float(abs(fine_target['tt']-reference['tt'])/reference['tt']),
                 target_truth_change=abs(S.target_truth(64)-S.target_truth(32)))
print(json.dumps(target_check,indent=2)); S.save_json(OUT/'target_quadrature_check.json',target_check)
assert target_check['relative_X_change'] < 2e-3, 'Target quadrature needs further refinement.'
assert target_check['target_truth_change'] < 1e-9'''),
md(r'''## Smooth Riesz functions and the underlying ray functional

Left and middle: target and delivered Riesz representers in H2, on the same
colour scale. Right: the original weighted rays. These are two representations
of functionals, not interchangeable spatial density pictures. In particular,
$\langle r,m\rangle_H=\sum_i x_i\int_{\ell_i}m\,ds$; it is generally not
$\int r m\,dA$. The model is not reconstructed in this experiment.'''),
code(r'''start=time.perf_counter()
X,Y,rv,tv,rfun,tfun=S.sobolev_fields(reference,n=64)
fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
lim=max(abs(rv).max(),abs(tv).max())
for ax,V,title in zip(axs[:2],[tv,rv],['target Riesz representer','delivered Riesz representer']):
 im=ax.imshow(V,origin='lower',extent=[0,1,0,1],cmap='RdBu_r',vmin=-lim,vmax=lim)
 ax.add_patch(plt.Circle(S.CENTRE,S.RADIUS,fill=False,color='k')); ax.set_title(title)
fig.colorbar(im,ax=axs[:2],shrink=.8)
lc=LineCollection(np.stack([p,q],axis=1),array=reference['x'],cmap='RdBu_r',linewidths=.7)
axs[2].add_collection(lc); axs[2].set_title('delivered functional: weighted original rays')
fig.colorbar(lc,ax=axs[2],shrink=.8,label='ray weight')
for ax in axs: ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
plt.show(); print(f'Field evaluation: {time.perf_counter()-start:.1f}s')
with S.quadrature_nodes(24):
 alone=float(np.asarray(rfun(np.array([.3]),np.array([.65]))).ravel()[0])
 two=np.asarray(rfun(np.array([.3,.4]),np.array([.65,.5]))).ravel()[0]
np.testing.assert_allclose(alone,two,rtol=2e-12,atol=1e-10)
print('Drawing-grid independence checked at a fixed physical point.')'''),
md(r'''## What the estimate establishes

The same noisy data are used throughout. The target bias below is known only
because this is a synthetic experiment. It is not included in the propagated
noise standard deviation.'''),
code(r'''last=next(r for r in rows if r['nodes']==24)
for key in ['mass','estimate','delivered_truth','target_truth','target_bias','noise_draw','std','relative_mismatch']:
 print(f'{key:22s} {last[key]:.10g}')
np.testing.assert_allclose(last['estimate']-truth,last['target_bias']+last['noise_draw'],atol=2e-13)
if (OUT/'pixel_results.json').exists():
 pix=json.loads((OUT/'pixel_results.json').read_text())['rows']
 print('\n                       estimate    delivered truth    noise sigma    target bias')
 for label,z in [('pixels, finest',pix[-1]),('Sobolev, converged',last)]:
  print(f"{label:22s} {z['estimate']:10.6f} {z['delivered_truth']:16.6f} {z['std']:14.6g} {z['target_bias']:13.6f}")
 print('These use different mismatch geometries: this is not an accuracy ranking of one common estimator.')'''),
md(r'''## Interpretation and limits

When the weight vector stabilizes as quadrature is refined, the selected
transformation of the original MDR stabilizes. Its delivered property still
differs from the disk mean; convergence of numerical calculation does not remove
non-identifiability or turn propagated noise into a target-error bound.

The Sobolev metric provides a defined dual norm for the original ray functionals.
No norm bound or prior distribution is used. The choice of space and inner product
is explicit, and changing it changes which reachable question is judged closest.

Pixel-first L2 and continuum H2 therefore need not produce the same X. A growing
pixel density norm does not imply that every finite data combination is invalid:
its exact interpretation is always through its original weighted-ray MDR. It does
mean that such pixel kernels should not be presented as converging L2 sensitivity
representers without evidence and a compatible continuum formulation.

This notebook checks one target, one ray geometry and one fixed trade-off. It
does not prove that any target can be recovered, establish coverage after
data-dependent tuning, or show that pixel discretisations of an explicitly posed
Sobolev problem would fail. Those would approximate the Sobolev problem instead.''')])

for nb,name in [(nb1,'3_pixels_sola.ipynb'),(nb2,'4_sobolev_sola.ipynb')]:
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            cell.source=cell.source.replace('\\\\','\\')
    nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
    nbf.write(nb,HERE/name)
    print('wrote',name)
