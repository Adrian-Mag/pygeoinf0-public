"""Generate the conforming projection follow-up; leaves the original pair alone."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook(cells=[
md(r'''# Unconstrained SOLA: pixel collapse and Sobolev projection convergence

This follows notebooks 3 and 4 on their **same 600 rays, data, fixed disk mean and
dimensionless trade-off**. Here all solves are **unconstrained**. The complete
derivations and assumptions are in [unconstrained_limits.md](unconstrained_limits.md).

For bounded $G,T$ on a Hilbert space $H$, a conforming subspace with orthogonal
projector $P_h$ gives
$$K_h=GP_hG^*,\quad b_h=GP_h t_H,\quad x_h=(K_h+\tau I)^{-1}b_h.$$
If the subspaces approximate the data representers, then $x_h\to x$.
Here $t_H$ denotes the target's Riesz representer in $H$.

The zero-width rays are **not bounded on $L^2$**. Their pixel representers instead
satisfy $(K_h)_{ii}=\Theta(h^{-1})$ and $(K_h)_{ij}=O(1)$ for distinct fixed
non-overlapping rays. A bounded fixed target has $b_h=O(1)$, so $x_h=O(h)\to0$.
The corresponding pixel densities also tend to zero in $L^2$. This differs from
the normalized pixel kernels of notebook 3, whose norms must eventually diverge.'''),
code(r'''%matplotlib inline
import hashlib, json, os, socket, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sola_demo as S
import sola_projection as Q
OUT=Path('output'); OUT.mkdir(exist_ok=True)
_,_,p,q=S.P.geometry()
d0,d=S.P.data(); truth=S.target_truth()
print('machine:',socket.gethostname(),'Python:',sys.executable)
print('execution commit:',subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip())
print('problem SHA256:',hashlib.sha256(Path('problem.py').read_bytes()).hexdigest())
print('true disk mean:',truth)
'''),
md(r'''## Pixel $L^2$: a closed-form solve with a degenerate limit

For cell area $h^2$, ray crossing matrix $E_h$ and projected disk density $t_h$,
$$K_h=h^{-2}E_h E_h^T,\qquad b_h=E_h t_h.$$
Keep $\tau=\lambda\|t\|_{L^2}^2$ fixed. No response constraint is imposed.
We calculate the full target mismatch, including the target component outside
the pixel space, and propagate noise for the actual delivered ray combination.
The theorem concerns the limit with the number and geometry of rays fixed;
near-parallel rays can delay its asymptotic regime.'''),
code(r'''pixels=[]; pixel_weights=[]
tt=1/S.AREA; tau=S.LAMBDA*tt
for n in (20,40,80,160,320,640,1280,2560):
    E=S.ray_matrix(p,q,n); t=S.disk_pixel_density(n)
    K=(E@E.T).toarray()*n*n; b=np.asarray(E@t)
    x=S.sola_weights(K,b,tau)
    r2=float(x@K@x)
    row=dict(n=n,h=1/n,kernel_norm=np.sqrt(r2),
             relative_mismatch=np.sqrt(max(tt-2*x@b+r2,0)/tt),
             **S.estimator_diagnostics(x,d0,d,truth,S.P.SIGMA))
    pixels.append(row); pixel_weights.append(x)
    print(f"{n:4d}: ||x||={row['weight_norm']:.6f}, ||r_h||={row['kernel_norm']:.6f}, "
          f"estimate={row['estimate']:.6f}, mismatch={row['relative_mismatch']:.6f}")
S.save_json(OUT/'unconstrained_pixels.json',pixels)
np.savez_compressed(OUT/'unconstrained_pixel_weights.npz',weights=pixel_weights)
'''),
code(r'''h=np.array([r['h'] for r in pixels]); wn=np.array([r['weight_norm'] for r in pixels])
fig,ax=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
ax[0].loglog(h,wn,'o-',label=r'$\|x_h\|$')
ax[0].loglog(h,wn[-1]*h/h[-1],'--',label=r'reference slope $h$')
ax[0].set_ylabel('weight norm'); ax[0].legend()
ax[1].semilogx(h,[r['estimate'] for r in pixels],'o-',label='estimate')
ax[1].semilogx(h,[r['delivered_truth'] for r in pixels],'x--',label='actual delivered truth')
ax[1].axhline(truth,color='k',ls=':',label='target truth'); ax[1].axhline(0,color='.6',lw=.7)
ax[1].legend(); ax[1].set_ylabel('property')
ax[2].semilogx(h,[r['relative_mismatch'] for r in pixels],'o-')
ax[2].axhline(1,color='k',ls=':'); ax[2].set_ylabel(r'$\|t-r_h\|/\|t\|$')
for a in ax: a.invert_xaxis(); a.set_xlabel('h (finer to the right)'); a.grid(alpha=.25)
plt.show()
'''),
md(r'''## Sobolev $H^2$: refine actual conforming trial spaces

Notebook 4 refined numerical integration of continuum representer pairings.
Here we keep that quadrature fixed and refine **nested finite subspaces of $H^2$**:
$$V_h=\operatorname{span}\{k_z:z\in Z_h\},$$
where $Z_h$ are nested dyadic point grids and $k_z$ are point-evaluation Riesz
representers. These points specify trial functions; the observations remain the
original rays. The union of these spaces is dense because $H^2$ functions are
continuous and a function orthogonal to all $k_z$ vanishes on a dense set.

For trial Gram $M_{jk}=\langle k_{z_j},k_{z_k}\rangle_H$, data evaluation matrix
$E_{ij}=G_i k_{z_j}$ and target vector $u_j=T k_{z_j}$,
$$K_h=E M^{-1}E^T,\quad b_h=E M^{-1}u.$$
We use Cholesky solves, retain the mass matrix and keep the **continuum** penalty
$\tau=\lambda\|t_H\|_H^2$ fixed. No nugget or eigenvalue clipping is introduced.
Compare to $x=(K+\tau I)^{-1}b$, the full continuum-representer calculation.
The target is approximated by 32 disk chords and ray pairings by 24-node quadrature;
these fixed numerical errors are separate from trial-space truncation.'''),
code(r'''max_n=int(os.environ.get('SOLA_PROJECTION_MAX_N','65'))
levels=tuple(n for n in (3,5,9,17,33,65) if n<=max_n)
print('Trial grid sizes:',levels)
rows,weights,reference,xref=Q.projection_study(levels=levels)
S.save_json(OUT/'sobolev_projection.json',dict(rows=rows,reference=reference,max_trial_n=max_n))
np.savez_compressed(OUT/'sobolev_projection_weights.npz',weights=weights,reference=xref)
print('\nContinuum reference:',json.dumps(reference,indent=2))
print('\n n  dimension      relative K       relative b       relative X       estimate')
for r in rows:
    print(f"{r['n']:2d} {r['dimension']:8d} {r['gram_error']:16.7e} {r['cross_error']:16.7e} "
          f"{r['x_error']:16.7e} {r['estimate']:14.9f}")
assert rows[-1]['x_error'] < rows[0]['x_error']
assert rows[-1]['gram_error'] < rows[0]['gram_error']
'''),
code(r'''h=np.array([r['h'] for r in rows])
fig,ax=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
for key,label in [('gram_error','K'),('cross_error','b'),('x_error','X')]:
    ax[0].loglog(h,[r[key] for r in rows],'o-',label=label)
ax[0].legend(); ax[0].set_ylabel('relative error to continuum calculation')
ax[1].semilogx(h,[r['estimate'] for r in rows],'o-',label='estimate')
ax[1].semilogx(h,[r['delivered_truth'] for r in rows],'x--',label='delivered truth')
ax[1].axhline(reference['estimate'],ls='--',color='.5',label='continuum estimate')
ax[1].axhline(truth,color='k',ls=':',label='target truth'); ax[1].legend(fontsize=8)
ax[1].set_ylabel('property')
ax[2].semilogx(h,[r['std'] for r in rows],'o-')
ax[2].axhline(reference['std'],ls='--',color='k'); ax[2].set_ylabel('propagated noise std')
for a in ax: a.invert_xaxis(); a.set_xlabel('h (finer to the right)'); a.grid(alpha=.25)
plt.show()
'''),
md(r'''## What converges, and what does not

For fixed positive $R=\gamma C_d$, the exact identity
$$X_h-X=\big[(B_h-B)-X(K_h-K)\big](K_h+R)^{-1}$$
proves Sobolev convergence. The inverse is uniformly bounded and projection
residuals tend to zero. The delivered question $X_hG$ converges to $XG$; its
mismatch with $T$ can persist. No prior radius or distribution has been asserted.

For pixels, unbounded ray representer norms drive the unconstrained solution
towards the zero question, despite diminishing noise. With unit response instead,
the density has unit mass on ray-touched cells of total area $O(h)$, so its $L^2$
norm must grow at least as $h^{-1/2}$. The two limits are mathematically different.

These are theorems under the stated assumptions; the finite grids illustrate
them. Quadrature, conditioning, and target mismatch remain distinct error sources.
The previous 32/64-chord check measures target quadrature sensitivity; agreement
within a single fixed quadrature must not be interpreted as an exact disk result.''')
])
nb.metadata.update(kernelspec={'display_name':'Python 3','language':'python','name':'python3'})
nbf.write(nb,HERE/'5_unconstrained_limits.ipynb')
