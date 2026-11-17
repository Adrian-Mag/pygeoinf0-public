"""The same data under three model-data relations (plan 008): exact data, bounded noise and Gaussian noise, with the
same model space, data and property, and what each relation (with its prior) lets us say about the property.

    model space   M = H^1_l([0, 1]), l = 0.1 (the restriction Sobolev space)
    data          averages of the model over 24 subintervals, noise standard deviation sigma (D carries it: the
                  metric of D is the inverse noise covariance, so D's standard Gaussian is the noise)
    property      the average of the model over [0.45, 0.55]
    relations     pg.Exact(G): d = G m;  pg.Additive(G, ball): d in G m + B (B the 95 % chi-square ball of the noise,
                  a bounded-noise reading); pg.Additive(G, D.standard_gaussian()): d ~ N(G m, sigma^2 I)
    priors        none; the set ||m|| <= 3 (stated); the cylindrical Gaussian N(0, I) of M (stated)

Outputs (cells of pygeoinf0.cells): the fibre G^-1(d) and its minimum-norm summary (the property is not estimable, so
the fibre leaves it unbounded); the BGP feasible property interval (deterministic, given the two sets); the same
geometry from the acceptance region of the Gaussian noise (confidence-set inversion, coverage >= 95 % for every model
in the set); the PLI posterior (credible interval, Bayesian under the stated prior); SOLA (an estimate of the delivered
property, not the target) and cSOLA (its outer set for the target under the two sets).

Run on europa (single-threaded is enough): RELATIONS_OUT=dir python relations_three_noises.py -> dir/*.png, values.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402

import pygeoinf0 as pg  # noqa: E402

OUT = os.path.abspath(os.environ.get("RELATIONS_OUT", "relations"))
os.makedirs(OUT, exist_ok=True)

M = pg.Sobolev("[0, 1]", 1, length_scale=0.1)
n, sigma = 24, 0.02
D = pg.Euclidean(n, covariance=sigma ** 2)
J = [pg.Interval(a, a + 1.0 / 30) for a in np.linspace(0.0, 1 - 1.0 / 30, n)]
G = pg.Functionals(M.riesz(J) * 30.0, into=D)                                # averages over the subintervals
T = pg.Functionals(M.riesz([pg.Interval(0.45, 0.55)]) * 10.0)                # the average over [0.45, 0.55]
truth = M(lambda x: np.sin(2 * np.pi * x) + 0.8 * np.exp(-((x - 0.5) / 0.08) ** 2), resolution=0.08)   # the bump's width
d = G(truth) + D.standard_gaussian().sample(seed=1)
p_true = float(T(truth).values[0])

exact = pg.Exact(G)
bounded = pg.Additive(G, pg.NormBall(D, np.sqrt(chi2.ppf(0.95, n))))           # fixed before the data
gaussian = pg.Additive(G, D.standard_gaussian())
prior_set, prior_mu = pg.NormBall(M, 3.0), pg.probability.cylindrical(M, alpha=1.0)

K0 = pg.update(pg.NoPrior(M), exact, d)                                         # B5 (the whole space): the fibre
m_dagger = K0.summary()
P0 = pg.push(K0, T)                                                             # F7: an affine set in P
K1 = pg.update(prior_set, bounded, d)                                           # B6: the feasible set
bgp = pg.push(K1, T).bounds(0)                                                  # F7 + the dual: BGP
csi = pg.push(pg.update(prior_set, gaussian.acceptance(0.05), d), T).interval(0)  # confidence-set inversion
post = pg.update(prior_mu, gaussian, d)                                         # B10: the posterior
pli = pg.push(post, T)                                                          # F10: PLI
pm, ps = float(pli.mean.values[0]), float(np.sqrt(pli.covariance.matrix()[0, 0]))
so = pg.inference.sola(gaussian, T, d, gamma=1.0)
s_est, s_sd = float(so.estimate.values[0]), float(np.sqrt(so.noise_law.covariance.matrix()[0, 0]))
cs = pg.inference.csola(prior_set, bounded, T, d, X=so.X).interval(0)
try:
    pg.update(prior_set, gaussian, d)
except pg.Refused as e:
    refusal = str(e)

values = {
    "property (truth)": p_true,
    "exact: minimum-norm summary": float(T(m_dagger).values[0]),
    "exact: the fibre's property set": [str(x) for x in P0.interval(0)],
    "bounded (BGP)": [bgp[0].value, bgp[1].value], "BGP status": [bgp[0].status, bgp[1].status],
    "Gaussian acceptance region (confidence-set inversion)": list(csi),
    "PLI 95% credible": [pm - 1.96 * ps, pm + 1.96 * ps],
    "SOLA delivered estimate +- 1.96 sd": [s_est - 1.96 * s_sd, s_est + 1.96 * s_sd],
    "cSOLA outer set": list(cs),
    "set prior with a Gaussian kernel": refusal,
}
with open(os.path.join(OUT, "values.json"), "w") as f:
    json.dump(values, f, indent=1)
for k, v in values.items():
    print(f"{k:55s} {v}")

# figure 1: the model space -------------------------------------------------------------------------------------
x = np.linspace(0, 1, 401)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, truth(x), "k", lw=2, label="true model")
ax.plot(x, m_dagger(x), "C0", label="exact data: minimum-norm summary of the fibre")
mean, sd = post.mean(x), post.std(x)
ax.plot(x, mean, "C3", label="Gaussian noise: posterior mean")
ax.fill_between(x, mean - 2 * sd, mean + 2 * sd, color="C3", alpha=0.2, label="posterior ± 2 sd (cylindrical prior)")
for i, (j, y) in enumerate(zip(J, d.values)):
    ax.plot([j.a, j.b], [y, y], "C2", lw=2, label="data: averages over subintervals" if i == 0 else None)
ax.axvspan(0.45, 0.55, color="0.85", zorder=0, label="property: the average over [0.45, 0.55]")
ax.set(xlabel="x", ylabel="m(x)", title="One data set, three model-data relations")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "model_space.png"), dpi=150)

# figure 2: the property ------------------------------------------------------------------------------------------
rows = [("exact data, prior none: the fibre (T is not estimable)", (-np.inf, np.inf), "C0"),
        ("bounded noise, ||m|| <= 3: BGP (deterministic)", (bgp[0].value, bgp[1].value), "C1"),
        ("Gaussian acceptance region, ||m|| <= 3: CSI (coverage >= 95%)", csi, "C4"),
        ("cSOLA outer set (same sets)", cs, "C5"),
        ("Gaussian noise, Gaussian prior: PLI 95% credible", (pm - 1.96 * ps, pm + 1.96 * ps), "C3"),
        ("SOLA: the DELIVERED property ± 1.96 sd", (s_est - 1.96 * s_sd, s_est + 1.96 * s_sd), "C2")]
fig, ax = plt.subplots(figsize=(8, 3.6))
lo_all = min(r[1][0] for r in rows if np.isfinite(r[1][0]))
hi_all = max(r[1][1] for r in rows if np.isfinite(r[1][1]))
pad = 0.3 * (hi_all - lo_all)
for i, (label, (a, b), c) in enumerate(rows):
    a, b = (a if np.isfinite(a) else lo_all - pad), (b if np.isfinite(b) else hi_all + pad)
    ax.plot([a, b], [i, i], color=c, lw=5, solid_capstyle="butt")
    ax.text(lo_all - pad, i + 0.28, label, fontsize=7)
ax.axvline(p_true, color="k", lw=1.5, label="true property")
ax.set(yticks=[], xlabel="average of m over [0.45, 0.55]", xlim=(lo_all - pad, hi_all + pad), ylim=(-0.6, len(rows)))
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "property.png"), dpi=150)
print("figures in", OUT)
