"""Build tomography.ipynb: Sobolev spaces, rays as sets and measures, ray tomography in 2-D, 3-D and on the sphere."""
import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))  # noqa: E731
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))  # noqa: E731

md("""
# Ray tomography in Sobolev spaces

A ray integral m -> ∫_γ m ds integrates over a set of measure zero, so it is **not** a bounded functional on L²: an element of
L² is a class of functions equal almost everywhere. It is bounded on the Sobolev space H^s_l when s > (d − 1)/2 (trace
theorem); its Riesz representer is then a function on the whole domain, smooth away from the ray, with a kink across it.

pygeoinf0 defines H^s_l(Ω) on a domain Ω ⊂ ℝ^d as the restriction space {U|_Ω : U ∈ H^s_l(ℝ^d)} with the quotient norm (McLean
2000, ch. 3): no boundary conditions, no padding. Its kernel is the Matérn kernel, so Gram matrices of rays are exact
double integrals of a closed-form kernel. On the sphere H^s is the spectral (Laplace–Beltrami) space.

Executed on europa.
""")
code("""
import math, time
import numpy as np
import matplotlib.pyplot as plt
import pygeoinf0 as pg
%matplotlib inline
plt.rcParams["figure.dpi"] = 100
""")

md("""
## 1. Rays are sets and measures; L² refuses them
""")
code("""
D = pg.Disk(1.0)
ray = pg.Segment((-0.8, -0.6), (0.6, 0.8))
print(ray, "| length", ray.volume, "| its measure:", pg.LebesgueMeasure(ray))
try:
    pg.L2(D).riesz(ray)
except pg.errors.UnboundedFunctional as e:
    print("L2 refuses:", e)
""")
code("""
for s in (0.5, 1.0, 1.5):
    H = pg.Sobolev(D, s, length_scale=0.1)
    print(f"H^{s}: points bounded {bool(H.dual().integrals_bounded(0))}, rays bounded {bool(H.dual().integrals_bounded(1))}")
H = pg.Sobolev(D, 1.5, length_scale=0.1)
print(H); print(H.claims)
""")

md("""
## 2. A representer: the smoothed ray

The representer k_γ = ∫_γ K(·, y) ds(y) of the ray integral in H^{3/2}_l(disk) (exponential kernel, l = 0.1): a function
on the whole disk, concentrated within a few l of the ray, with a kink across it. ⟨k_γ, f⟩ = ∫_γ f ds for every f.
""")
code("""
k = H.riesz(ray)
k.plot(n=160, title="representer of the ray integral in H^{3/2}");
f = H(lambda x, y: np.cos(2 * x) + x * y)
t, w = np.polynomial.legendre.leggauss(40)                 # Gauss on [-1, 1], mapped to the ray
P = np.array([-0.8, -0.6]) + np.outer((t + 1) / 2, [1.4, 1.4])
print("<k, f> =", k @ f, " | Gauss quadrature of f along the ray:", w @ (np.cos(2 * P[:, 0]) + P[:, 0] * P[:, 1]) / 2 * ray.volume)
print("||k||^2 =", k.norm() ** 2, "= int int K ds ds (the energy of the ray measure)")
""")

md("""
## 3. Two-dimensional tomography: 24 sources × 24 receivers around a disk
""")
code("""
th_s = np.linspace(0, 2 * np.pi, 24, endpoint=False)
th_r = th_s + np.pi / 24
S = np.stack([np.cos(th_s), np.sin(th_s)], 1); Rc = np.stack([np.cos(th_r), np.sin(th_r)], 1)
rays = [pg.Segment(s, r) for s in S for r in Rc if np.linalg.norm(s - r) > 0.5]
m_true = lambda x, y: np.exp(-((x - 0.3) ** 2 + (y + 0.2) ** 2) / 0.05) - 0.6 * np.exp(-((x + 0.4) ** 2 + (y - 0.35) ** 2) / 0.03)
t0 = time.perf_counter()
K = H.riesz(rays)
n = len(rays)
d_clean = pg.Functionals(K)(H(m_true)).values           # exact ray integrals of the formula (reproducing property)
sigma = 0.01 * np.abs(d_clean).max()
Dd = pg.Euclidean(n, covariance=sigma ** 2)
G = pg.Functionals(K, into=Dd)
d = Dd(d_clean) + Dd.standard_gaussian().sample(seed=1)
A = G @ G.adjoint
T_gram = time.perf_counter() - t0
alpha = pg.meta.discrepancy(pg.Problem(mdr=G, datum=d)).gamma                   # chi^2(m_alpha) = n
t0 = time.perf_counter()
m = G.adjoint((A + alpha * pg.Identity(Dd)).inv(d))
T_solve = time.perf_counter() - t0
print(f"{n} rays: representers and Gram {T_gram:.2f} s, solve {1e3 * T_solve:.0f} ms; alpha = {alpha:.3g}; "
      f"chi2/n = {(G(m) - d).norm() ** 2 / n:.3f}; ||m||_H = {m.norm():.4f}")
""")
code("""
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
H(m_true).plot(ax=axs[0], n=120, title="m_true")
for r in rays[::7]:
    p, q = r.endpoints; axs[1].plot([p[0], q[0]], [p[1], q[1]], lw=0.4, color="0.4")
axs[1].set_aspect("equal"); axs[1].set_title(f"every 7th of {n} rays")
m.plot(ax=axs[2], n=120, title="damped least squares in H^{3/2}")
fig.tight_layout()
""")
code("""
mL = pg.Inclusion(H, pg.L2(D, tol=1e-5))(m)               # into L2: to compare with the truth as functions
tL = pg.L2(D, tol=1e-5)(m_true)
print(f"L2 error of the solution: {(mL - tL).norm() / tL.norm():.3f} (relative); estimated realisation error {mL.error:.1e}")
""")

md("""
### Bayesian view: the Gaussian with Cameron–Martin space H^s

`pg.probability.cylindrical(H, alpha=α)` is the cylindrical Gaussian N(0, I/α) whose Cameron–Martin space is H^s (on a domain of ℝ^d
with s > d/2 it is the Matérn process). It has no samples in H^s, but every bounded functional has an exact variance,
Var ∫x dμ = ‖k_μ‖²/α, and conditioning on ray data d = Gx + e (e the standard Gaussian of the data space, here
N(0, σ²I)) gives a posterior whose mean is the damped least-squares solution above and whose point variances are exact:
(k(0) − b_x^T (E + σ²α I)^{-1} b_x)/α, with b_x the ray–point energies and E the Gram matrix.
""")
code("""
post = pg.probability.cylindrical(H, alpha=alpha).condition(G, d)
print("posterior mean = damped least squares:", abs(post.mean(0.1, 0.2) - m(0.1, 0.2)))
g = np.linspace(-0.97, 0.97, 91)
X, Y = np.meshgrid(g, g); inside = X ** 2 + Y ** 2 < 0.97 ** 2
pts = np.stack([X[inside], Y[inside]], 1)
t0 = time.perf_counter()
std = np.full(X.shape, np.nan); std[inside] = post.std(pts)          # exact, vectorised over the points
print(f"{len(pts)} posterior standard deviations in {time.perf_counter() - t0:.1f} s")
B = H.inner(K, H.riesz(pts[:1])); E = G.gram()                         # one of them by hand
v = (H.norm(H.riesz(pts[0])) ** 2 - B[:, 0] @ np.linalg.solve(E + sigma ** 2 * alpha * np.eye(n), B[:, 0])) / alpha
print("by hand:", math.sqrt(v), " package:", std[inside][0])
fig, ax = plt.subplots(figsize=(5.4, 4.4))
im = ax.pcolormesh(X, Y, std, shading="auto", cmap="viridis"); ax.set_aspect("equal")
ax.set_title("posterior standard deviation of point values (prior std %.2f)"
             % pg.probability.cylindrical(H, alpha=alpha).std(pts[0]), fontsize=9)
plt.colorbar(im, ax=ax);
""")
code("""
C = pg.Circle(0.5)                                  # an interface: its mean is a bounded functional (s > 1/2)
print(f"mean over the circle r = 0.5: true {H(m_true).integral(C) / C.volume:.4f}, "
      f"estimate {m.integral(C) / C.volume:.4f} +- {post.std(C) / C.volume:.4f} "
      f"(prior std {pg.probability.cylindrical(H, alpha=alpha).std(C) / C.volume:.4f})")
""")

md("""
## 4. Three dimensions: rays through a ball
""")
code("""
B = pg.Ball(1.0)
H3 = pg.Sobolev(B, 2.0, length_scale=0.15)              # exponential kernel in 3-D (nu = 1/2); rays need s > 1
rng = np.random.default_rng(0)
def sphere_pts(k):
    v = rng.standard_normal((k, 3)); return v / np.linalg.norm(v, axis=1)[:, None]
P3, Q3 = sphere_pts(400), sphere_pts(400)
rays3 = [pg.Segment(p, q) for p, q in zip(P3, Q3) if np.linalg.norm(p - q) > 0.3]
m3 = lambda x, y, z: np.exp(-((x - 0.3) ** 2 + y ** 2 + (z - 0.2) ** 2) / 0.08)
t0 = time.perf_counter()
K3 = H3.riesz(rays3); n3 = len(rays3)
d3 = pg.Functionals(K3)(H3(m3)).values
s3 = 0.01 * np.abs(d3).max(); D3 = pg.Euclidean(n3, covariance=s3 ** 2)
G3 = pg.Functionals(K3, into=D3)
dd3 = D3(d3) + D3.standard_gaussian().sample(seed=2)
A3 = G3 @ G3.adjoint
mm3 = G3.adjoint((A3 + pg.meta.discrepancy(pg.Problem(mdr=G3, datum=dd3)).gamma * pg.Identity(D3)).inv(dd3))
print(f"{n3} rays in a ball: {time.perf_counter() - t0:.2f} s; chi2/n = {(G3(mm3) - dd3).norm() ** 2 / n3:.2f}")
g = np.linspace(-1, 1, 101); X, Y = np.meshgrid(g, g)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, f_, ttl in ((axs[0], lambda x, y: m3(x, y, 0.2 + 0 * x), "m_true, z = 0.2"), (axs[1], lambda x, y: mm3(x, y, 0.2 + 0 * x), "solution, z = 0.2")):
    V = f_(X, Y); V = np.where(X ** 2 + Y ** 2 <= 1 - 0.04, V, np.nan)
    im = ax.pcolormesh(X, Y, V, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1); ax.set_aspect("equal"); ax.set_title(ttl)
fig.colorbar(im, ax=axs);
""")
md("""
### Interfaces: what do the rays say about a buried sphere?

A sphere inside the ball is a set, its area measure a functional: u ↦ ∫_S u dA is bounded on H^s(B) for s > 1/2 (trace
theorem). Its representer has a closed-form potential, so the prior and posterior uncertainty of the mean of the model
over the sphere are exact numbers (and patches work the same way).
""")
code("""
core = pg.Sphere(0.5)
a3 = pg.meta.discrepancy(pg.Problem(mdr=G3, datum=dd3)).gamma
prior3 = pg.probability.cylindrical(H3, alpha=a3)
post3 = prior3.condition(G3, dd3)
true, est = H3(m3).integral(core) / core.volume, post3.mean.integral(core) / core.volume
print(f"mean of m over the sphere r = 0.5: true {true:.4f}, estimate {est:.4f} "
      f"(the damped solution: {mm3.integral(core) / core.volume:.4f})")
sd0, sd = prior3.std(core) / core.volume, post3.std(core) / core.volume
print(f"its standard deviation: prior {sd0:.4f}, posterior {sd:.4f}; actual error / posterior std = {(est - true) / sd:.1f}")
""")
md("""
The actual error is several posterior standard deviations. That is not a numerical error: the true model (a unit bump)
is not a plausible draw from N(0, I/α) with α chosen by the discrepancy principle, so the posterior spread measures the
uncertainty under that prior, not the bias of the regularisation. A prior meant to be believed needs its scale chosen
for the model class, not for the fit.
""")

md("""
## 5. On the sphere: surface-wave tomography along great circles

H^s on the sphere is spectral (the metric (1 + l² n(n+1)/R²)^s on spherical harmonics). A path measure is not
band-limited, so representers are truncated; pygeoinf0 chooses the degree and reports the truncation error of each representer.
""")
code("""
Sph = pg.Sphere(1.0)
Hs = pg.Sobolev(Sph, 2.0, length_scale=0.2)
def unit(k):
    v = rng.standard_normal((k, 3)); return v / np.linalg.norm(v, axis=1)[:, None]
A_, B_ = unit(300), unit(300)
arcs = [pg.Arc(Sph, a, b) for a, b in zip(A_, B_) if 0.2 < math.acos(np.clip(a @ b, -1, 1)) < math.pi - 0.2]
ms = lambda x, y, z: z + 0.5 * x * y + np.exp(4 * (0.6 * x + 0.8 * z - 1))
t0 = time.perf_counter()
Ks = Hs.riesz(arcs); ns = len(arcs)
ds = pg.Functionals(Ks)(Hs(ms)).values
ss = 0.01 * np.abs(ds).max(); Ds = pg.Euclidean(ns, covariance=ss ** 2)
Gs = pg.Functionals(Ks, into=Ds)
dds = Ds(ds) + Ds.standard_gaussian().sample(seed=3)
As = Gs @ Gs.adjoint
msol = Gs.adjoint((As + pg.meta.discrepancy(pg.Problem(mdr=Gs, datum=dds)).gamma * pg.Identity(Ds)).inv(dds))
print(f"{ns} arcs: {time.perf_counter() - t0:.2f} s; representers {Hs.backend.summary(Ks.realisation)}, "
      f"largest relative truncation error {np.max(Ks.errors() / Ks.norms()):.1e}")
fig, axs = plt.subplots(1, 2, figsize=(14, 3.6))
Hs(ms).plot(ax=axs[0], title="m_true"); msol.plot(ax=axs[1], title="damped least squares in H^2(S^2)");
""")
md("""
## 6. What is exact, what is estimated

- Domains of ℝ^d (interval, box, disk, cylinder, ball): Gram matrices of rays and points are computed from exact reductions
  to one-dimensional analytic integrals (closed-form radial tail of the Matérn kernel), agreeing with mpmath to about
  1e-14; no truncation parameter (rays within 1e-8 radians of parallel: about 1e-8).
- Tori and circles: the same, with periodic images (Poisson summation equals the spectral definition).
- Sphere: spectral truncation with an estimated error (algebraic convergence for paths).
- Surfaces in 3-D domains (flat patches, spheres): their potentials are exact (closed form for spheres); energies with
  rays and other surfaces are graded Gauss quadratures with error estimates (1e-11 or better against SciPy).
- Norms of formulas (functions given by expressions) in a restriction space are the minimal-extension norm: not computed
  (exact on tori and circles); inner products of formulas with representers are ∫_γ f ds, by Gauss quadrature along the
  support (rounding-level for smooth formulas). Comparisons as functions go through pg.Inclusion.

The comparison with pygeoinf, pygeoinf2, pygeoinf3D and intervalinf is in turf/sob/ (README section).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "inferences", "display_name": "Python (inferences)", "language": "python"}
nbf.write(nb, "tomography.ipynb")
print("wrote tomography.ipynb")
