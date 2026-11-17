"""Generate only the four part-1 notebooks; preserves earlier executed notebooks."""
from pathlib import Path
import nbformat as nbf

HERE=Path(__file__).resolve().parent
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
common=r'''%matplotlib inline
from pathlib import Path
import hashlib, json, os, socket, subprocess, sys, time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import study as F
S=F.S
ex=F.Experiment(length_scale=.1)  # shared by both H2 routes; edit consistently in all notebooks
OUT=Path('output');OUT.mkdir(exist_ok=True)
COLORS=('#2563eb','#dc2626','#7c3aed')
plt.rcParams.update({'figure.dpi':110,'axes.grid':True,'grid.alpha':.2,'font.size':11})
print('host:',socket.gethostname(),'Python:',sys.executable)
print('source commit:',subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip())
print('original problem SHA256:',hashlib.sha256(Path(S.P.__file__).read_bytes()).hexdigest())
print('600 original rays; one fixed noise draw; sigma =',S.P.SIGMA)
for label,c,t in zip(F.LABELS,ex.centres,ex.truth):print(label,c,'radius',ex.radius,'true disk mean',f'{t:.9f}')
print('Fixed normalized trade-off lambda =',ex.lam,'; H2 length scale =',ex.length_scale)
import platform, scipy, pygeoinf0 as pg
from threadpoolctl import threadpool_info
runtime=dict(host=socket.gethostname(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
             platform=platform.platform(),cpu_count=os.cpu_count(),package_source=pg.__file__,
             threadpools=threadpool_info(),targets=ex.centres,radius=ex.radius,lambda_value=ex.lam,length_scale=ex.length_scale,
             study_sha256=hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest(),
             helper_sha256=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
             problem_sha256=hashlib.sha256(Path(S.P.__file__).read_bytes()).hexdigest())
F.save(OUT/'runtime.json',runtime)
print('BLAS thread counts:',[v['num_threads'] for v in runtime['threadpools']])
'''

nb1=nbf.v4.new_notebook(cells=[
md(r'''# 1 · Classic constrained SOLA, starting with pixels

This is the first inspectable part of the proposed presentation. We keep the
original **unit box, 600 straight rays, smooth truth and noisy data**. We ask for
three fixed disk means (radius 0.12), including the original positive-anomaly disk.

We begin exactly as a classical discrete SOLA application would: parameterize
immediately by pixel values, form the forward matrix, use the **correct mass
matrix**, and solve constrained SOLA. Then refine only the numerical pixels.

The physical target disks, rays, noise covariance and trade-off stay fixed.
This tests numerical refinement; it does not shrink the requested averaging region.'''),
code(common),
code(r'''z=np.linspace(0,1,220); xx,yy=np.meshgrid(z,z)
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for a,b in zip(ex.p,ex.q):ax[0].plot(*np.column_stack([a,b]),color='.35',lw=.35,alpha=.15)
ax[0].scatter(*ex.src.T,s=18,marker='*',color='k');ax[0].scatter(*ex.stations.T,s=22,marker='^',color='k')
im=ax[1].imshow(S.P.true_model(xx,yy),origin='lower',extent=[0,1,0,1],cmap='RdBu_r')
for c,color,label in zip(ex.centres,COLORS,F.LABELS):
    for a in ax:a.add_patch(plt.Circle(c,ex.radius,fill=False,color=color,lw=2))
    ax[1].plot([],[],color=color,label=label)
for a in ax:a.set_aspect('equal');a.set_xlim(0,1);a.set_ylim(0,1);a.grid(False)
ax[0].set_title('The same observations throughout');ax[1].set_title('Known truth and fixed physical targets')
ax[1].legend(fontsize=8);fig.colorbar(im,ax=ax[1],shrink=.8);plt.show()
'''),
md(r'''## Pixel coefficients are not an orthonormal basis

Let $m_h=\sum_j c_j\mathbf1_{C_j}$ on square pixels of side $h$.
The observation matrix contains exact ray lengths:
$$E_{ij}=\int_{\Gamma_i}\mathbf1_{C_j}\,ds.$$
The $L^2$ coefficient metric is $M_h=h^2I$, so
$$G_h^*=M_h^{-1}E^T,\qquad K_h=E M_h^{-1}E^T.$$
A transpose alone would be the adjoint in the **Euclidean coefficient metric**.
We retain the cell areas here. For the disk target density $t$, its coefficients
$t_h$ are exact cell averages from disk/cell overlap areas, and
$$b_h=E t_h.$$

The next cell verifies the weighted adjoint identity on an actual mesh.'''),
code(r'''n=20;h=1/n
E=S.ray_matrix(ex.p,ex.q,n)
rng=np.random.default_rng(42);c=rng.normal(size=n*n);y=rng.normal(size=len(ex.d))
Gstar_y=np.asarray(E.T@y)/h**2
lhs=(E@c)@y;rhs=h**2*c@Gstar_y
print('data pairing:',lhs,'model pairing:',rhs,'difference:',abs(lhs-rhs))
assert np.allclose(lhs,rhs,rtol=2e-12,atol=2e-12)
print('Model dimension:',n*n,'Data dimension:',len(ex.d))
print('The huge diagonal mass matrix is applied by scaling, never stored densely.')
'''),
md(r'''## The constrained SOLA solve

For each target, minimize
$$\frac{\|t-r_h\|_{L^2}^2}{\|t\|_{L^2}^2}
+\lambda\frac{x^TC_dx}{\sigma_{\rm ref}^2},\qquad v^Tx=1,$$
where $r_h=\sum_i x_i g_{i,h}$, $v_i=G_i1=L_i$,
$C_d=\sigma^2I$, $\sigma_{\rm ref}=\sigma$ and $\lambda=0.1$.
Thus the fixed matrix penalty is $\tau I$, with $\tau=\lambda\|t\|_{L^2}^2$.
We retain the **full physical target norm**; projecting the target adds only an
$x$-independent constant at this fixed penalty.

Set $H_h=K_h+\tau I$. The exact response correction is
$$x_0=H_h^{-1}b_h,\qquad
x=x_0+H_h^{-1}v\frac{1-v^Tx_0}{v^TH_h^{-1}v}.$$
There is one common Cholesky factor and four right-hand sides: three targets and
$v$. No weights are tuned on the realized data. The helper batches this formula.'''),
code(r'''tau=ex.lam*ex.target_l2_norm2
system=F.pixel_system(ex,20)
K=system['K'];b=system['B'][0];v=ex.v
factor=la.cho_factor(K+tau*np.eye(len(v)),lower=True)
x0=la.cho_solve(factor,b);a=la.cho_solve(factor,v)
x=x0+a*(1-v@x0)/(v@a)
X=F.constrained_sola(K,system['B'],tau,v)
assert np.allclose(x,X[0],rtol=2e-12,atol=2e-13)
print('Physical constant responses:',X@v)
print('Estimates:',X@ex.d,'requested means:',ex.truth)
'''),
md(r'''## Refine the pixels, not the experiment

We recompute $E_h$, its mass-corrected Gram and the target projections. We evaluate
**actual delivered truth** as $X_h d_{\rm clean}$ using accurately integrated
original rays. This avoids conflating a pixel approximation of the truth with
what the linear data transformation actually measures.

For every target,
$$X_hd-Tm=(X_hGm-Tm)+X_h\eta.$$
The standard deviation below describes $X_h\eta$, not the target discrepancy.'''),
code(r'''rows=[];weights=[];pictures={}
max_n=int(os.environ.get('SOLA_PART1_MAX_PIXEL','2560'))
for n in F.PIXELS:
    if n>max_n:continue
    system=F.pixel_system(ex,n);X,row=F.solve_system(system,ex,tau)
    kernel2=np.einsum('ij,jk,ik->i',X,system['K'],X)
    mismatch2=ex.target_l2_norm2-2*np.sum(X*system['B'],axis=1)+kernel2
    row.update(n=n,h=1/n,dimension=n*n,kernel_norm=np.sqrt(kernel2).tolist(),
               relative_mismatch=np.sqrt(np.maximum(mismatch2,0)/ex.target_l2_norm2).tolist())
    rows.append(row);weights.append(X)
    if n in (20,80,320,max_n):
        pictures[n]=np.asarray(system['E'].T@X[0]).reshape(n,n)*n*n
    print(f"{n:4d}: estimates={np.round(row['estimate'],5)}, "
          f"separation ||x1-x2||={row['weight_separation']:.5g}, norm r1={row['kernel_norm'][0]:.5g}")
F.save(OUT/'pixels.json',dict(rows=rows,lambda_value=ex.lam,tau=tau))
np.savez_compressed(OUT/'pixel_weights.npz',weights=weights,grids=[r['n'] for r in rows])
assert max(np.max(np.abs(np.array(r['constant_response'])-1)) for r in rows)<2e-10
'''),
code(r'''ns=np.array([r['n'] for r in rows])
fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for k,(color,label) in enumerate(zip(COLORS,F.LABELS)):
    ax[0].semilogx(ns,[r['estimate'][k] for r in rows],'o-',color=color,label=label)
    ax[0].semilogx(ns,[r['delivered_truth'][k] for r in rows],':',color=color)
    ax[0].axhline(ex.truth[k],color=color,ls='--',alpha=.65)
    ax[1].loglog(ns,[r['std'][k] for r in rows],'o-',color=color)
    ax[2].semilogx(ns,[r['relative_mismatch'][k] for r in rows],'o-',color=color)
ax[0].legend(fontsize=8);ax[0].set_title('Estimate; dotted delivered truth; dashed target')
ax[1].set_title('Propagated noise decreases');ax[2].set_title('Relative L2 target mismatch')
for a in ax:a.set_xlabel('pixels per side (physical targets fixed)')
plt.show()
fig,ax=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
ax[0].loglog(ns,[r['weight_separation'] for r in rows],'o-');ax[0].set_ylabel(r'$\|x_h(q_1)-x_h(q_2)\|$')
for k,color in enumerate(COLORS):ax[1].loglog(ns,[r['kernel_norm'][k] for r in rows],'o-',color=color)
ax[1].set_ylabel(r'$\|r_h\|_{L^2}$ (can turn upward late)')
for a in ax:a.set_xlabel('pixels per side')
plt.show()
'''),
code(r'''fig,axes=plt.subplots(1,len(pictures),figsize=(14,3.7),constrained_layout=True)
lim=2*ex.target_l2_norm2
for ax,(n,density) in zip(np.atleast_1d(axes),pictures.items()):
    im=ax.imshow(density,origin='lower',extent=[0,1,0,1],cmap='RdBu_r',vmin=-lim,vmax=lim)
    ax.add_patch(plt.Circle(ex.centres[0],ex.radius,fill=False,color='k'))
    ax.set_title(f"{n} x {n}\npeak |density|={np.max(np.abs(density)):.1f}")
    ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
fig.colorbar(im,ax=np.atleast_1d(axes),shrink=.7,extend='both',label='cell density; common clipped colour scale')
plt.show()
'''),
md(r'''## Why the correct mass matrix does not fix this continuum problem

Zero-width ray integration has no bounded extension to $L^2$ of area. The pixel
representers $g_{i,h}$ exist on each finite space; continuum $L^2$ representers
$g_i$ do not. For fixed generic rays, diagonal entries of $K_h$ grow like $1/h$.

Unit response prevents the zero functional, but does not preserve localization.
The density has unit mass on ray-touched cells of total area $O(h)$, forcing its
$L^2$ norm eventually to grow at least as $h^{-1/2}$. If
$K_h=a_h A+o(a_h)$ with $a_h\to\infty$ and $A>0$, bounded fixed target pairings
become negligible and
$$x_h(q)\longrightarrow\frac{A^{-1}v}{v^TA^{-1}v},$$
independent of target location. The common limit, if it exists, is a normalized
ray statistic. A displayed constant map would not establish a constant Earth.
Near-parallel geometry can delay this regime; the plotted meshes need not yet
reach it. No divergence of the data weights is asserted.

The transformed MDR remains valid. The pixel density's continuum volume-average
interpretation and the criterion's continuum limit are what need attention.
Next we fix a Hilbert formulation where the rays really are bounded.''')])

nb2=nbf.v4.new_notebook(cells=[
md(r'''# 2 · Discretize a well-defined Sobolev SOLA problem

Keep the same 600 rays, three physical targets, noise draw and response constraint.
Define the continuum space first, then approximate its objective with conforming
finite trial spaces. This is still a **fully finite quadratic problem** at each level.

We use $H=H^2_\ell(\Omega)$, $\ell=0.1$, with the restriction/Bessel-potential norm
implemented by `pg.Sobolev`. This is not the intrinsic derivative norm with
arbitrarily chosen boundary conditions. The original smooth truth belongs to H2.
Ray integrals and disk means are bounded on this space; no prior radius or
probability law is asserted.'''),
code(common),
md(r'''## A conforming finite model realization

Use nested dyadic point grids $Z_h$ and trial functions $\phi_j=k_{z_j}$, the
H2 Riesz representers of point evaluation. These are trial centres, not extra data.
Their union is dense: a continuous H2 function orthogonal to every $k_z$ vanishes
on a dense set and therefore everywhere.

The finite matrices are
$$M_{jk}=\langle\phi_j,\phi_k\rangle_H,\quad
E_{ij}=G_i\phi_j,\quad U_{aj}=T_a\phi_j.$$
Thus
$$\boxed{K_h=E M^{-1}E^T,\qquad B_h=U M^{-1}E^T.}$$
The mass matrix is now a **full Sobolev Gram matrix**. Ordinary discontinuous
pixels would not be conforming H2 functions. The finite builder evaluates these
matrices without assembling the continuum ray/ray Gram.

The code below makes one small realization explicitly; the refinement loop uses
exactly this algebra.'''),
code(r'''import pygeoinf0 as pg
H,F_ray,F_target=F.sobolev_objects(ex,32)
z=np.linspace(0,1,5);xx,yy=np.meshgrid(z,z)
Phi=H.riesz(np.column_stack([xx.ravel(),yy.ravel()]))
with S.quadrature_nodes(24):
    M=np.asarray(H.inner(Phi,Phi))
    E=np.asarray(H.inner(F_ray,Phi))
    U=np.asarray(H.inner(F_target,Phi))
factor=la.cho_factor(M,lower=True)
Z=la.cho_solve(factor,E.T)
K_h=E@Z;B_h=U@Z
print('M:',M.shape,'E:',E.shape,'U:',U.shape,'K_h:',K_h.shape)
print('No continuum G G* is required by this finite assembly.')
'''),
md(r'''## Keep physical normalization and penalty fixed

The constraint is the exact physical response $v_i=G_i1=L_i$, known analytically.
It is an explicit constraint on the **data weights**, even though the trial space
need not contain the constant function exactly. Do not replace it silently with
$G_iP_h1$: that would change the constrained problem at each level.

The continuum normalized penalty $\tau=\lambda\|t_H\|_H^2$ is also kept fixed;
using $\|P_ht_H\|$ to rescale it on every mesh would change the objective.
A continuum-reference calculation is made **only to assess accuracy** and fix
that target norm, not to supply $K_h$ or $B_h$ to the finite builder.'''),
code(r'''ref,Xref,tau,reference=F.native_reference(ex,24,32)
print('Fixed tau:',tau,'reference estimates:',reference['estimate'])
print('Reference cost is excluded from each finite method timing.')
X=F.constrained_sola(K_h,B_h,tau,ex.v)
print('Exact physical constant response:',X@ex.v)
projected_v=E@la.cho_solve(factor,np.ones(len(M)))
print('Projected trial-space response:',X@projected_v)
assert np.allclose(X@ex.v,1.,rtol=0,atol=2e-12)
'''),
md(r'''## Refine nested trial spaces

$P_h$ is the H-orthogonal projector onto the trial space. Boundedness gives
$$K_h=GP_hG^*,\quad B_h=TP_hG^*.$$
Approximation of the data representers implies $K_h\to GG^*$, $B_h\to TG^*$.
The constrained minimizer converges as well: the positive fixed penalty gives
uniform coercivity, and $v$ is the same nonzero vector on every level.

We report both weight error and an actual functional error in the **common H2
geometry**, using $(X_h-X)K(X_h-X)^T$. This does not confuse Riesz functions with
ordinary area-density kernels.'''),
code(r'''rows=[];weights=[]
max_n=int(os.environ.get('SOLA_PART1_MAX_TRIAL','65'))
for n in F.TRIALS:
    if n>max_n:continue
    system=F.galerkin_system(ex,n,24,32)
    X,row=F.solve_system(system,ex,tau)
    row.update(n=n,h=1/(n-1),dimension=n*n,**F.compare_to_reference(X,ref,Xref))
    row['gram_error']=float(la.norm(system['K']-ref['K'])/la.norm(ref['K']))
    row['projected_response_error']=float(np.max(np.abs(X@system['projected_v']-1)))
    rows.append(row);weights.append(X)
    print(f"{n:2d} x {n:2d}: relative X error={row['weight_error']:.3e}, "
          f"functional error={row['delivered_functional_error']:.3e}, estimates={np.round(row['estimate'],6)}")
F.save(OUT/'galerkin.json',dict(rows=rows,reference=reference,tau=tau))
np.savez_compressed(OUT/'galerkin_weights.npz',weights=weights,reference=Xref,grids=[r['n'] for r in rows])
assert rows[-1]['weight_error']<rows[0]['weight_error']
assert max(np.max(np.abs(np.array(r['constant_response'])-1)) for r in rows)<2e-10
'''),
code(r'''h=np.array([r['h'] for r in rows])
fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for key,label in [('gram_error','K'),('weight_error','X'),('delivered_functional_error','delivered functional')]:
    ax[0].loglog(h,[r[key] for r in rows],'o-',label=label)
ax[0].legend(fontsize=8);ax[0].set_title('Approach to the continuum calculation')
for k,(color,label) in enumerate(zip(COLORS,F.LABELS)):
    ax[1].semilogx(h,[r['estimate'][k] for r in rows],'o-',color=color,label=label)
    ax[1].axhline(reference['estimate'][k],color=color,ls='--')
    ax[1].axhline(ex.truth[k],color=color,ls=':',alpha=.6)
ax[1].legend(fontsize=8);ax[1].set_title('Dashed continuum estimate; dotted target truth')
ax[2].loglog(h,[r['projected_response_error'] for r in rows],'o-')
ax[2].set_title('Projected normalization error\n(actual response is one throughout)')
for a in ax:a.invert_xaxis();a.set_xlabel('trial-centre spacing h (finer to the right)')
plt.show()
'''),
md(r'''## Watch the finite resolving representer converge

For the first disk, plot the actual **finite-space** resolving representer
$$r_h=P_hG^*x_h=\sum_j c_{h,j}\phi_j,
\qquad c_h=M_h^{-1}E_h^Tx_h.$$
This includes both the changing SOLA weights and the trial-space projection.
Plotting only $G^*x_h$ would omit the latter.

All panels use the same evaluation grid and colour scale, with the requested
disk outlined. The last panels show the native limit $G^*x$ and the target Riesz
representer $t_H$. These are **H2 Riesz functions, not area-density kernels**:
their interpretation is through the H2 inner product. Their displayed area
integrals need not be one. The physical response $x_h^TG1$ remains one, while
$\langle r_h,1\rangle_H=x_h^TGP_h1$ is the projected response reported above.

The extra assembly below is an untimed plotting diagnostic. Small blocks bound
temporary kernel arrays; these computations do not change the benchmark.'''),
code(r'''plot_n=60
z=(np.arange(plot_n)+.5)/plot_n
xx,yy=np.meshgrid(z,z)
plot_points=np.column_stack([xx.ravel(),yy.ravel()])
selected=[(r['n'],W[0]) for r,W in zip(rows,weights)
          if r['n'] in {3,9,17,rows[-1]['n']}]
pictures={};field_errors={}
block=64
native_norm2=float(Xref[0]@ref['K']@Xref[0])
for n,x_h in selected:
    z_h=np.linspace(0,1,n);xh,yh=np.meshgrid(z_h,z_h)
    trial_points=np.column_stack([xh.ravel(),yh.ravel()])
    Phi_h=H.riesz(trial_points)
    m=n*n
    M_plot=np.empty((m,m),order='F');E_plot=np.empty((len(ex.d),m))
    with S.quadrature_nodes(24):
        for j in range(0,m,block):
            stop=min(j+block,m)
            # Fresh families keep the atom sets small; slicing a Family may
            # retain the full atom set and merely select coefficient columns.
            trial_block=H.riesz(trial_points[j:stop])
            M_plot[:,j:stop]=H.inner(Phi_h,trial_block)
            E_plot[:,j:stop]=H.inner(F_ray,trial_block)
    rhs=E_plot.T@x_h
    # Overwrite this diagnostic mass matrix to avoid a second full matrix copy.
    factor_plot=la.cho_factor(M_plot,lower=True,overwrite_a=True,check_finite=False)
    c_h=la.cho_solve(factor_plot,rhs,check_finite=False)
    error2=float(c_h@rhs-2*c_h@(E_plot.T@Xref[0])+native_norm2)
    field_errors[n]=float(np.sqrt(max(error2,0)/native_norm2))
    values=np.empty(len(plot_points))
    for j in range(0,len(values),block):
        stop=min(j+block,len(values))
        points=H.riesz(plot_points[j:stop])
        values[j:stop]=np.asarray(H.inner(points,Phi_h))@c_h
    pictures[n]=values.reshape(plot_n,plot_n)
    del M_plot,E_plot,factor_plot,Phi_h
    print(f'{n} x {n}: relative H2 representer error = {field_errors[n]:.3e}')

native_values=np.empty(len(plot_points));target_values=np.empty(len(plot_points))
with S.quadrature_nodes(24):
    for j in range(0,len(plot_points),block):
        stop=min(j+block,len(plot_points));points=H.riesz(plot_points[j:stop])
        native_values[j:stop]=np.asarray(H.inner(points,F_ray))@Xref[0]
        target_values[j:stop]=np.asarray(H.inner(points,F_target[0])).reshape(-1)
native_values=native_values.reshape(plot_n,plot_n)
target_values=target_values.reshape(plot_n,plot_n)
panels=[(f'{n} x {n} trial centres\nrelative H2 error {field_errors[n]:.2e}',v)
        for n,v in pictures.items()]
panels += [('Native continuum resolving representer',native_values),
           ('Uniform-disk target: H2 representer',target_values)]
lim=max(float(np.max(np.abs(v))) for _,v in panels)
fig,axes=plt.subplots(1,len(panels),figsize=(3.2*len(panels),3.7),constrained_layout=True)
for ax,(title,values) in zip(np.atleast_1d(axes),panels):
    im=ax.imshow(values,origin='lower',extent=[0,1,0,1],cmap='RdBu_r',vmin=-lim,vmax=lim)
    ax.add_patch(plt.Circle(ex.centres[0],ex.radius,fill=False,color='k'))
    ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
fig.colorbar(im,ax=np.atleast_1d(axes),shrink=.7,label='H2 Riesz function; common colour scale')
plt.show()
F.save(OUT/'galerkin_fields.json',dict(levels=list(pictures),relative_H2_errors=field_errors,
       target_index=0,plot_grid=plot_n,interpretation='P_h G* x_h, paired in H2; not area density',
       plot_host=socket.gethostname(),threads=threadpool_info()))
np.savez_compressed(OUT/'galerkin_fields.npz',levels=list(pictures),fields=np.array(list(pictures.values())),
                    native=native_values,target=target_values,points=plot_points)
assert field_errors[selected[-1][0]]<field_errors[selected[0][0]]
assert all(np.all(np.isfinite(v)) for _,v in panels)
'''),
md(r'''## What this establishes

Discretizing first works when it is a consistent approximation of the declared
continuum formulation. There is no need to avoid finite matrices. The choice of
H2 and its metric fixes the SOLA comparison geometry; it does not assert an upper
bound on the unknown model norm. The limiting question remains $XG$, not exactly T.

This particular finite realization uses **dense kernel-translate trial matrices**.
Sparse finite elements or other conforming bases may have different costs. The
next notebook computes the same continuum objective directly in data space.
The fixed 24-node/32-chord integration error remains separate from trial error.''')])

nb3=nbf.v4.new_notebook(cells=[
md(r'''# 3 · Native continuum Sobolev SOLA in pygeoinf0

Use the **same H2 space, three targets, exact constant response and normalized
penalty** as notebook 2. Define G and T as bounded functionals on H, then assemble
$GG^*$ and $TG^*$ directly. No model trial dimension is introduced.

“Continuum” means the mathematical formulation and its data/target representation.
Numerical kernel pairings still need integration and linear solves. Here refinement
means more accurate pairings, not more model coefficients.'''),
code(common),
md(r'''## Operators first

Riesz representation gives $g_i=\mathcal R_H^{-1}G_i$ and
$t_a=\mathcal R_H^{-1}T_a$. Then
$$K_{ij}=\langle g_i,g_j\rangle_H,\qquad
B_{ai}=\langle t_a,g_i\rangle_H.$$
In `pygeoinf0`, the direct construction is:'''),
code(r'''import pygeoinf0 as pg
H,F_ray,F_target=F.sobolev_objects(ex,32)
with S.quadrature_nodes(24):
    G=pg.Functionals(F_ray,name='G')
    T=pg.Functionals(F_target,name='T')
    K=(G@G.adjoint).matrix()
    B=(T@G.adjoint).matrix()
    tt=np.diag(T.gram())
tau=ex.lam*tt[0]
X=F.constrained_sola(K,B,tau,ex.v)
print('Data-space Gram:',K.shape,'target/data pairing:',B.shape,'X:',X.shape)
print('Constant responses:',X@ex.v)
print('Estimates:',X@ex.d,'actual delivered truths:',X@ex.d0,'target truths:',ex.truth)
# A = X G is an actual operator on the continuum H2 space.
A=pg.Matrix(X,domain=G.codomain,codomain=T.codomain)@G
print('Delivered operator:',A)
'''),
md(r'''## Refine integration and check target quadrature separately

The ray-pair quadrature is varied at fixed 32-chord disk quadrature. We then check
32 versus 64 disk chords separately, with the penalty held fixed. Agreement with
another integration rule is a numerical sensitivity check, not a rigorous bound.

The same positive penalty and response constraint are retained. No clipping of
negative Gram eigenvalues is used to manufacture a continuum Gram.'''),
code(r'''ref,Xref,tau,reference=F.native_reference(ex,24,32)
rows=[];weights=[]
for nodes in (10,16,24):
    system=F.native_system(ex,nodes,32)
    low=float(la.eigvalsh(system['K'],subset_by_index=[0,0])[0])
    if low < -1e-10*la.norm(ref['K'],2):raise ArithmeticError('under-resolved Gram')
    X,row=F.solve_system(system,ex,tau)
    row.update(nodes=nodes,min_eigenvalue=low,**F.compare_to_reference(X,ref,Xref))
    row['gram_error']=float(la.norm(system['K']-ref['K'])/la.norm(ref['K']))
    rows.append(row);weights.append(X)
    print('nodes',nodes,'relative X error',row['weight_error'],'estimates',row['estimate'])
fine=F.native_system(ex,24,64)
X64=F.constrained_sola(fine['K'],fine['B'],tau,ex.v)
target_check=dict(target_chords=(32,64),penalty_fixed=tau,
                  relative_X_change=float(la.norm(X64-Xref)/la.norm(Xref)))
print('Separate target quadrature check:',target_check)
F.save(OUT/'continuum.json',dict(rows=rows,reference=reference,tau=tau,target_check=target_check))
np.savez_compressed(OUT/'continuum_reference.npz',K=ref['K'],B=ref['B'],tt=ref['tt'],X=Xref)
assert rows[1]['weight_error']<1e-7
assert target_check['relative_X_change']<1e-3
'''),
code(r'''nodes=np.array([r['nodes'] for r in rows])
fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
for key,label in [('gram_error','K'),('weight_error','X')]:
    ax[0].semilogy(nodes,np.maximum([r[key] for r in rows],1e-16),'o-',label=label)
ax[0].legend();ax[0].set_xlabel('ray quadrature nodes');ax[0].set_title('Reference: 24 nodes / 32 chords')
for k,(color,label) in enumerate(zip(COLORS,F.LABELS)):
    ax[1].plot(nodes,[r['estimate'][k] for r in rows],'o-',color=color,label=label)
    ax[1].axhline(ex.truth[k],color=color,ls=':')
ax[1].legend(fontsize=8);ax[1].set_xlabel('ray quadrature nodes');ax[1].set_title('Target mismatch survives convergence')
plt.show()
'''),
md(r'''## Riesz functions are not Lebesgue averaging-density kernels

The delivered representer is $r_H=G^*x$, so the delivered property is
$\langle r_H,m\rangle_H=\sum_i x_i\int_{\Gamma_i}m\,ds$.
A smooth plot of $r_H$ does **not** mean that the property is $\int r_Hm\,dA$.
The physical observation kernels remain ray measures; the Sobolev metric makes
the functionals bounded and supplies their Riesz representatives.'''),
code(r'''r=ref['G'].adjoint(ref['G'].codomain(Xref[0]));t=ref['T'][0]
z=(np.arange(60)+.5)/60;xx,yy=np.meshgrid(z,z)
with S.quadrature_nodes(24):rv,tv=r(xx,yy),t(xx,yy)
fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for a,v,title in zip(ax,[tv,rv],['Target Riesz function','Delivered Riesz function']):
    im=a.imshow(v,origin='lower',extent=[0,1,0,1],cmap='viridis')
    a.add_patch(plt.Circle(ex.centres[0],ex.radius,fill=False,color='white',lw=1))
    a.set_title(title+' (H2 pairing)');a.grid(False);fig.colorbar(im,ax=a,shrink=.8)
plt.show()
for k,label in enumerate(F.LABELS):
    print(label,'target:',ex.truth[k],'delivered:',reference['delivered_truth'][k],
          'estimate:',reference['estimate'][k],'noise sigma:',reference['std'][k])
'''),
md(r'''## The two Sobolev routes solve the same limiting problem

For $P_h$ the conforming H-orthogonal trial projector,
$$K_h=GP_hG^*\to GG^*,\qquad B_h=TP_hG^*\to TG^*.$$
The finite Galerkin weights approach the native continuum weights. Numerical
quadrature must also converge; the two limits are separate.

Both calculations select a delivered question $XG$. Neither has proved exact
recovery of the physical target T or a target-error bound without model information.
The advantage established here is a well-defined, stable question-selection
problem under refinement. Notebook 4 measures the cost of these realizations.''')])

nb4=nbf.v4.new_notebook(cells=[
md(r'''# 4 · Costs and the accuracy of the delivered question

Compare three actual implementations on the **same 600 rays and three targets**:

1. Pixels with the correct L2 mass matrix (the continuum-inconsistent baseline).
2. A fully finite, conforming H2 Galerkin problem in dense kernel translates.
3. Native H2 data-space Gram assembly.

The last two approximate the **same objective**. Pixel/L2 changes the comparison
geometry, so a fast pixel solve is not an accuracy-equivalent substitute. The
comparison does not benchmark every possible finite-element or iterative method.

We measure fresh assembly, metric elimination/Gram formation and factor/solve
costs. Imports and the common synthetic data generation are excluded for all.
Reference and accuracy diagnostics are outside the timed regions. No conclusion
about a winner is assumed in advance.'''),
code(common),
md(r'''## Method and timing controls

One process, one BLAS thread, fresh problem objects for each repetition. Native
Gram caching within a single solve is allowed; no previous method's assembled
Gram is supplied to another timed method. One common data-space factor handles
all three targets plus normalization. We report all raw repetitions and medians.

“Matrix payload” sums named major algebra arrays. It excludes workspaces,
allocator behavior, kernel objects and cached copies; **it is not peak RAM**.
The native formulation still has an N-by-N Gram here, while the dense Galerkin
realization also assembles a trial-by-trial Sobolev mass matrix.'''),
code(r'''repeats=int(os.environ.get('SOLA_PART1_BENCH_REPEATS','3'))
ref,Xref,tau,reference=F.native_reference(ex,24,32)
records=[]
settings=[('pixels',n) for n in (80,320,1280)]
settings += [('H2 Galerkin',n) for n in (17,33,65) if n<=int(os.environ.get('SOLA_PART1_MAX_TRIAL','65'))]
settings += [('H2 native',n) for n in (10,16,24)]
# A small untimed BLAS warm-up is shared by every method.
la.cho_solve(la.cho_factor(np.eye(5)),np.ones(5))
for method,level in settings:
    for repeat in range(repeats):
        if method=='pixels':
            system=F.pixel_system(ex,level);penalty=ex.lam*ex.target_l2_norm2
        elif method=='H2 Galerkin':
            system=F.galerkin_system(ex,level,24,32);penalty=tau
        else:
            system=F.native_system(ex,level,32);penalty=tau
        X,row=F.solve_system(system,ex,penalty)
        row.update(method=method,level=level,repeat=repeat,
                   dimension=system['dimension'],**F.compare_to_reference(X,ref,Xref))
        records.append(row)
        print(method,level,'repeat',repeat,'seconds',round(row['total_seconds'],4),
              'relative distance to H2 reference',f"{row['weight_error']:.3e}",flush=True)
        del system
summary=[]
for method,level in settings:
    selected=[r for r in records if r['method']==method and r['level']==level]
    row=dict(method=method,level=level,dimension=selected[0]['dimension'],
             median_seconds=float(np.median([r['total_seconds'] for r in selected])),
             min_seconds=float(min(r['total_seconds'] for r in selected)),
             max_seconds=float(max(r['total_seconds'] for r in selected)),
             matrix_payload_MiB=selected[0]['matrix_bytes']/1024**2,
             weight_error=selected[0]['weight_error'],
             delivered_functional_error=selected[0]['delivered_functional_error'],
             timings={k:float(np.median([r['timings'][k] for r in selected])) for k in selected[0]['timings']})
    summary.append(row)
F.save(OUT/'benchmark.json',dict(records=records,summary=summary,repeats=repeats,
       host=socket.gethostname(),threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
       reference=reference,accuracy_metric='relative data weights and H2 delivered-functional distance'))
import csv
with (OUT/'benchmark.csv').open('w',newline='') as f:
    names=[k for k in summary[0] if k!='timings'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
    w.writerows({k:r[k] for k in names} for r in summary)
print('\n method       level    seconds     matrix MiB    weight error')
for r in summary:print(f"{r['method']:12s} {r['level']:5d} {r['median_seconds']:11.4f} "
                      f"{r['matrix_payload_MiB']:12.2f} {r['weight_error']:14.4e}")
'''),
code(r'''fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
colours={'pixels':'#b45309','H2 Galerkin':'#2563eb','H2 native':'#059669'}
for method in colours:
    selected=[r for r in summary if r['method']==method]
    ax[0].loglog([r['level'] for r in selected],[r['median_seconds'] for r in selected],'o-',label=method,color=colours[method])
    ax[1].loglog([r['median_seconds'] for r in selected],np.maximum([r['delivered_functional_error'] for r in selected],1e-16),'o-',color=colours[method])
    ax[2].loglog([r['level'] for r in selected],[r['matrix_payload_MiB'] for r in selected],'o-',color=colours[method])
ax[0].legend(fontsize=8);ax[0].set_ylabel('median total seconds')
ax[1].set_xlabel('median total seconds');ax[1].set_ylabel('H2 delivered-question distance')
ax[1].set_title('Pixels: diagnostic distance, not same-objective error')
ax[2].set_ylabel('named matrix payload MiB (not peak RAM)')
for a in (ax[0],ax[2]):a.set_xlabel('resolution level (method-specific)')
plt.show()
fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
labels=[f"{r['method']}\n{r['level']}" for r in summary];bottom=np.zeros(len(summary))
for key,label in [('data_assembly','data / trial assembly'),('target_assembly','target evaluations'),('metric_and_gram','metric elimination / Gram'),('solve','factor and solve')]:
    val=np.array([r['timings'][key] for r in summary]);ax.bar(labels,val,bottom=bottom,label=label);bottom+=val
ax.set_ylabel('median stage seconds');ax.legend(fontsize=8);plt.show()
'''),
md(r'''## Compare costs at a declared accuracy

For the two H2 methods, use a relative weight-error threshold of $10^{-3}$ against
the fixed 24-node/32-chord reference. This threshold is a numerical study setting,
not an uncertainty interval. The separate target-quadrature check in notebook 3
bounds what can reasonably be inferred from that reference agreement.

For pixels, the reference distance describes how its selected question differs
from the chosen H2 question. It is **not** a pixel discretization error estimate.'''),
code(r'''tolerance=1e-3
for method in ('H2 Galerkin','H2 native'):
    eligible=[r for r in summary if r['method']==method and r['weight_error']<=tolerance]
    if eligible:
        best=min(eligible,key=lambda r:r['median_seconds'])
        print(method,': cheapest measured setting meeting tolerance:',best['level'],
              'median seconds:',best['median_seconds'],'relative error:',best['weight_error'])
    else:print(method,': no measured setting met tolerance; do not invent an equal-accuracy speedup.')
if (OUT/'pixels.json').exists() and (OUT/'galerkin.json').exists():
    px=F.load(OUT/'pixels.json')['rows'][-1];ga=F.load(OUT/'galerkin.json')['rows'][-1]
    print('\nFinest pixel estimates:',px['estimate'])
    print('Finest Galerkin estimates:',ga['estimate'])
print('Native continuum estimates:',reference['estimate'])
print('Requested target truths:',ex.truth)
'''),
md(r'''## End of part 1

- The pixel example uses its mass matrix correctly. Its ray/L2 continuum
  interpretation still fails under refinement.
- A declared Sobolev space makes the observations and the matching criterion
  well-defined. Conforming finite discretization and native continuum assembly
  approach the same question.
- Actual costs depend on representation, accuracy, data count, number of targets
  and implementation. This benchmark is evidence for these settings and this dense
  trial basis, not a universal performance ordering.
- Numerical convergence does not remove target mismatch or assert prior bounds.

This notebook sequence is the reviewable first part. Later presentation sections
and the Slidev deck are deliberately left for the author's next instructions.''')])

# Physical views and common-probe responses are untimed diagnostics. Keep each
# section reproducible from the preceding notebook cells, independently of caches.
probe_intro=r'''
The top row shows the physical measures: an exact ray integral, a uniform disk
average, and the signed weighted rays selected by SOLA. A ray has zero area but
nonzero measure mass. The disk has density $1/(\pi R^2)$ inside its boundary.
The delivered measure has mass $\sum_i x_iL_i=1$ and retains its ray support.

For the bottom row, place the **same known unit-peak smooth test anomaly** at each
centre $z$ and ask what these physical measurements return:
$$G_iq_z,\qquad T_aq_z,\qquad (XG)_a q_z.$$
The test shape and its horizontal profile are shown explicitly. Its chosen
length is 0.1 throughout; it is neither a physical ray width nor information
about the unknown model. It has decaying tails, rather than compact support.
The heatmaps are **responses to test models**, not area-density representations
of the ray measures. Ray responses have units of model times length; target and
delivered averages have model units, with a shared colour scale for that pair.
'''

pixel_views=[md(r'''## Physical questions and responses to a known moving anomaly
'''+probe_intro+r'''
This display probe does not alter the pixel/L2 matching metric or its SOLA weights.
We evaluate the actual transformed continuum-ray data for every test model,
rather than substituting a pixel approximation of that model.'''),
code(r'''# physical-probe: pixels
import probe_views as V
probes=V.response_grid(ex,OUT)
V.plot_probe(ex,probes);plt.show()
V.plot_physical_and_responses(ex,probes,weights[-1][0],
    f"Pixel SOLA ({rows[-1]['n']} per axis): measures and physical responses");plt.show()
indices=[i for i,r in enumerate(rows) if r['n'] in {rows[0]['n'],80,320,rows[-1]['n']}]
V.plot_response_evolution(ex,probes,[weights[i][0] for i in indices],
    [f"{rows[i]['n']} x {rows[i]['n']} pixels" for i in indices],
    'Same test anomaly: how the actual pixel-SOLA questions change');plt.show()
print('All actual constant responses:',weights[-1]@ex.v)
''')]

galerkin_views=[md(r'''## Physical questions and full continuum response maps
'''+probe_intro+r'''
For this H2 space, the test model is the unit-peak reproducing kernel
$$q_z(y)=\frac{K_H(y,z)}{K_H(z,z)},\qquad
(G_iq_z,T_aq_z,(X_hG)_a q_z)
=\frac{(g_i(z),t_a(z),(G^*x_{h,a})(z))}{K_H(z,z)}.$$
The smooth maps therefore have a direct physical reading without changing the
observations: they predict measurements of this specified test model.

The evolution below is the **full continuum response** $X_hGq_z$. The following
projected-field section instead shows $P_hG^*x_h$, whose rescaled values are
$X_hG(P_hq_z)$. Those agree on trial models but generally differ on arbitrary
continuum test models. Keeping both views exposes the trial approximation.'''),
code(r'''# physical-probe: galerkin
import probe_views as V
probes=V.response_grid(ex,OUT)
V.plot_probe(ex,probes);plt.show()
V.plot_physical_and_responses(ex,probes,weights[-1][0],
    f"H2 Galerkin ({rows[-1]['n']} per axis): actual measures and responses");plt.show()
indices=[i for i,r in enumerate(rows) if r['n'] in {3,9,17,rows[-1]['n']}]
V.plot_response_evolution(ex,probes,[weights[i][0] for i in indices],
    [f"{rows[i]['n']} x {rows[i]['n']} trial centres" for i in indices],
    'Full continuum responses X_h G q_z (trial projection shown separately below)',
    reference=Xref[0]);plt.show()
print('Actual responses use G q_z, not G P_h q_z.')
''')]

native_views=[md(r'''## Read the Riesz plots as responses to test anomalies
'''+probe_intro+r'''
Here the common test anomaly is $q_z=K_H(\cdot,z)/K_H(z,z)$. Reproduction gives
$$r_F(z)=F(K_H(\cdot,z)),\qquad F(q_z)=r_F(z)/K_H(z,z).$$
Thus a Riesz image, rescaled by the constant kernel peak, already is a physical
response map for this family of known models. It is not a blurred observation.
The same statement applies to sensitivity, target and resolving functionals.
The requested and delivered maps below use the same probe for all three disks;
any discrepancy is an actual target discrepancy for that test model, rather than
a universal error bound for an unrestricted unknown model.'''),
code(r'''# physical-probe: native
import probe_views as V
probes=V.response_grid(ex,OUT)
V.plot_probe(ex,probes);plt.show()
V.plot_physical_and_responses(ex,probes,Xref[0],
    'Native H2 SOLA: exact measures and their common-probe responses');plt.show()
V.plot_three_targets(ex,probes,Xref);plt.show()
print('Kernel peak used for unit-peak normalization:',float(probes['peak']))
print('Unit response to constant models:',Xref@ex.v)
''')]

comparison_views=[md(r'''## Compare the actual questions with a common physical test

Use the same moving, unit-peak anomaly as in notebooks 1-3. Each map evaluates
the actual transformed ray data $XGq_z$; none is a pixel density or projected
Galerkin field. Physical rays and target disks remain exactly the same.

For $q_z=K_H(\cdot,z)/K_H(z,z)$ these are rescaled continuum Riesz functions,
but the interpretation is simply the predicted response to a known test model.
The display probe is shared across all methods, including pixel/L2; it does not
change their different matching objectives. This figure is outside all timings.
It uses the finest saved pixel and Galerkin weights from notebooks 1 and 2;
the plotted pixel level may differ from the benchmark's timed settings.'''),
code(r'''# physical-probe: comparison
import probe_views as V
probes=V.response_grid(ex,OUT)
with np.load(OUT/'pixel_weights.npz') as saved:
    px_level=int(saved['grids'][-1]);px_weights=saved['weights'][-1]
with np.load(OUT/'galerkin_weights.npz') as saved:
    ga_level=int(saved['grids'][-1]);ga_weights=saved['weights'][-1]
V.plot_response_evolution(ex,probes,[px_weights[0],ga_weights[0],Xref[0]],
    [f'Pixels {px_level} x {px_level}',f'H2 Galerkin {ga_level} x {ga_level}','Native H2'],
    'One physical test family: the actual delivered questions versus the disk target');plt.show()
print('Responses to constant models:')
for label,W in [('pixels',px_weights),('H2 Galerkin',ga_weights),('native',Xref)]:
    print(label,W@ex.v)
''')]

for nb,heading,cells in [(nb1,'## Why the correct mass matrix',pixel_views),
                        (nb2,'## Watch the finite resolving',galerkin_views),
                        (nb3,'## The two Sobolev routes',native_views),
                        (nb4,'## End of part 1',comparison_views)]:
    position=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith(heading))
    nb.cells[position:position]=cells

for name,nb in [('01_classic_pixels.ipynb',nb1),('02_sobolev_galerkin.ipynb',nb2),
                ('03_sobolev_continuum.ipynb',nb3),('04_costs_and_comparison.ipynb',nb4)]:
    nb.metadata.update(kernelspec={'display_name':'Python 3','language':'python','name':'python3'})
    nbf.write(nb,HERE/name)
    print(name)
