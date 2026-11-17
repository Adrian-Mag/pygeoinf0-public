"""Linearised P-wave travel-time tomography of the whole Earth: SensRay (PREM, TauP rays) and pygeoinf0.

    unknown   dVp in H^2_l(B), the Sobolev space of the unit ball B (lengths in Earth radii R), l = 500 km
    data      dt_i = -int_ray_i dVp / Vp^2 ds   (rays traced in PREM, i = 1..10000; P, PcP, PKP, PKIKP)
    solution  the regularised minimum-norm model  m = G* (G G* + alpha C_e)^-1 d,  alpha from the discrepancy principle

Each ray is a polyline (TauP's path, kept every <= 250 km and at every PREM discontinuity) carrying the measure
w_j ds on its j-th segment, w_j = the mean of 1/Vp^2 along that part of the TauP path; dt_i = -R <k_i, dVp>, where
k_i = int K(., y) dmu_i(y) is the Riesz representer of the ray's measure in H (the Matern kernel K, exact energies).
The synthetic data come from a known dVp by Gauss quadrature along the same polylines, plus noise of 0.3 s.

Rays come through the obspy plugin (plugins/pygeoinf0-obspy, plan 019): its functions are this example's former
ray code, unchanged. Run on europa (one environment with obspy, sensray, matplotlib, pygeoinf0 and pygeoinf0-obspy):
    PYGEOINF0_WORKERS=30 python global_tomography.py      ->  global_tomography/*.png, costs.json
"""
import json
import os
import resource
import time
from multiprocessing import get_context

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sensray import PlanetModel

import pygeoinf0 as pg
import pygeoinf0_obspy as po                     # the obspy plugin (plugins/pygeoinf0-obspy): TauP paths

R, L_SEG, SIGMA, ELL = 6371.0, 250.0, 0.3, 500.0           # km, km, s, km
N_DATA = int(os.environ.get("N_DATA", 10000))
WORKERS = int(os.environ.get("PYGEOINF0_WORKERS", 1))               # pygeoinf0 reads the same variable
OUT = os.path.abspath("global_tomography")                    # figures and costs.json, in the working directory
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)
costs, _t = {}, [time.time(), sum(os.times()[:4])]


def lap(name):
    """Record the wall and CPU seconds (this process and its workers) since the last lap."""
    now, cpu = time.time(), sum(os.times()[:4])
    costs[name] = dict(wall_s=round(now - _t[0], 2), cpu_s=round(cpu - _t[1], 1))
    print(f"{name:32s} {now - _t[0]:9.1f} s wall {cpu - _t[1]:10.1f} s cpu", flush=True)
    _t[:] = [now, cpu]


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


# ---------------------------------------------------------------- reference model, events and stations
prem = PlanetModel.from_standard_model("prem")
taup = prem.taupy_model
DISC = np.unique(np.concatenate([[l["depth"][0], l["depth"][-1]] for l in prem.layers.values()]))
R_CMB, R_ICB = 3480.0 / R, 1221.5 / R
PHASES = {"P": (10, 98), "PcP": (25, 75), "PKP": (145, 178), "PKIKP": (115, 178)}     # distance windows, degrees

BELTS = [[(55, -160), (50, 160), (35, 142), (0, 125), (-10, 120), (-20, 170), (-40, 178)],     # subduction
         [(-30, -72), (0, -80), (15, -95), (40, -125), (55, -160)],                            # subduction
         [(35, -5), (38, 25), (35, 55), (30, 80), (28, 95)],                                   # collision
         [(60, -30), (30, -42), (0, -18), (-50, -12)]]                                         # mid-ocean ridge
CONTINENTS = [(45, -100, 25), (50, 20, 18), (50, 90, 25), (0, 20, 25), (-25, 135, 15), (-15, -60, 20)]


def events(n):
    out = []
    for _ in range(n):
        b = rng.integers(len(BELTS))
        k = rng.integers(len(BELTS[b]) - 1)
        p = (1 - (u := rng.random())) * xyz(*BELTS[b][k]) + u * xyz(*BELTS[b][k + 1])
        p = p / np.linalg.norm(p) + rng.normal(0, np.radians(1.5), 3)
        z = rng.choice([rng.uniform(10, 60), rng.uniform(70, 300), rng.uniform(350, 650)],
                       p=[0.6, 0.25, 0.15] if b < 2 else [0.8, 0.2, 0.0] if b == 2 else [1.0, 0.0, 0.0])
        out.append((p / np.linalg.norm(p), z if b < 3 else rng.uniform(8, 15)))
    return out


def stations(n_land, n_ocean):
    land, sea = [], []
    while len(land) < n_land or len(sea) < n_ocean:
        p = rng.normal(size=3)
        p /= np.linalg.norm(p)
        on = any(p @ xyz(a, b) > np.cos(np.radians(r)) for a, b, r in CONTINENTS)
        (land if on else sea).append(p)
    return np.array(land[:n_land] + sea[:n_ocean])


EV, ST = events(max(120, N_DATA // 300)), stations(230, 40)     # ~1.7 arrivals per pair: enough for N_DATA


def rays(pair):
    """The arrivals of the selected phases between event e and station s: (e, s, phase, TauP time, polyline,
    segment densities, travel time of the TauP path in PREM recomputed from the path, for the check)."""
    e, s = pair
    src, z = EV[e]
    rec = ST[s]
    D = np.arccos(np.clip(src @ rec, -1, 1))
    names = [ph for ph, (a, b) in PHASES.items() if a <= np.degrees(D) <= b]
    out, seen = [], set()
    arrs, D, t = po.arrivals(taup, src, z, rec, names)
    for arr in arrs:
        key = (arr.name, round(arr.time, 2))
        if key in seen:
            continue
        seen.add(key)
        X, dep = po.path_points(arr, src, t, R)
        keep = po.thin_by_length(X, dep, L_SEG, R, DISC)
        w, t_path = po.mean_slowness2(X, dep, keep, R, lambda d: prem.get_property_at_depth("vp", d))
        out.append((e, s, arr.name, arr.time, X[keep], w, t_path))
    return out


with get_context("fork").Pool(WORKERS) as pool:
    cand = [r for rs in pool.imap(rays, [(e, s) for e in range(len(EV)) for s in range(len(ST))], 64) for r in rs]
lap("rays: TauP + polylines")
data = [cand[i] for i in np.sort(rng.choice(len(cand), N_DATA, replace=False))]
n = len(data)
check = np.array([abs(r[6] - r[3]) for r in data])
print(f"{len(cand)} candidate arrivals; {n} selected; |T(path) - T(TauP)|: median {np.median(check):.3f} s, "
      f"max {check.max():.3f} s; {sum(len(r[5]) for r in data)} segments", flush=True)


# ---------------------------------------------------------------- the true model dVp (km/s)
def dvp(x, y, z):
    X = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
    r = np.linalg.norm(X, axis=-1)
    depth = R * (1 - r)
    out = np.zeros(r.shape)
    a = up = xyz(0, 125)                                 # slab: trench at (0 N, 125 E), strike N, dipping west 55 deg
    strike = np.array([0.0, 0.0, 1.0])
    down = np.cos(np.radians(55)) * np.cross(up, strike) - np.sin(np.radians(55)) * up
    rel = X - a
    dist = R * np.abs(rel @ np.cross(strike, down))
    along = R * rel @ strike
    dip = R * rel @ down                                 # down-dip distance from the trench, km
    out += 0.35 * np.exp(-(dist / 100) ** 2 - (along / 1500) ** 2) * (dip > 0) * (dip < 1000) \
        / (1 + np.exp((depth - 670) / 40))
    c = xyz(0, -30)                                      # plume under (0, 30 W), from the CMB to the surface
    h = R * np.linalg.norm(X - (X @ c)[..., None] * c, axis=-1)
    out -= 0.2 * np.exp(-(h / 250) ** 2) * (X @ c > 0) / (1 + np.exp((R_CMB - r) * R / 30))
    for lat, lon in ((0, 15), (-5, -160)):               # two slow piles above the CMB
        ang = np.degrees(np.arccos(np.clip((X / np.maximum(r, 1e-12)[..., None]) @ xyz(lat, lon), -1, 1)))
        hh = (r - R_CMB) * R
        out -= 0.2 * np.exp(-(ang / 25) ** 4) * (hh > 0) / (1 + np.exp((hh - 700) / 80))
    out += 0.05 * (1 + np.tanh(R * X[..., 1] / 300)) / 2 / (1 + np.exp((r - R_ICB) * R / 30))  # inner core, east fast
    return out


P0 = np.vstack([r[4][:-1] for r in data])
P1 = np.vstack([r[4][1:] for r in data])
W = np.concatenate([r[5] for r in data])
owner = np.repeat(np.arange(n), [len(r[5]) for r in data])
tq, gq = np.polynomial.legendre.leggauss(16)
Xq = P0[:, None] + 0.5 * (tq[None, :, None] + 1) * (P1 - P0)[:, None]
seg_int = np.linalg.norm(P1 - P0, axis=1) * (dvp(*Xq.transpose(2, 0, 1)) @ gq) / 2
d_true = -R * np.bincount(owner, W * seg_int, minlength=n)
d_obs = d_true + SIGMA * rng.normal(size=n)
lap("synthetic data")


# ---------------------------------------------------------------- the inverse problem in pygeoinf0
H = pg.Sobolev(pg.Ball(radius=1.0), 2, length_scale=ELL / R)
K = H.riesz([pg.LebesgueMeasure(pg.Polyline(r[4])).with_density(r[5]) for r in data])
lap("Riesz representers (atoms)")
D = pg.Euclidean(n, covariance=SIGMA ** 2)
G = pg.Functionals(-R * K, into=D)                    # dVp -> travel-time anomalies (s)
d = D(d_obs)
alpha = pg.meta.discrepancy(pg.Problem(mdr=G, datum=d)).gamma   # by size (pg.settings): eigh of the exact Gram up to
lap("Gram and discrepancy principle")                 # dense_max data, Lanczos on matrix-free products above
m = pg.probability.cylindrical(H, alpha=alpha).condition(G, d).mean
lap("posterior mean")                                 # Cholesky up to dense_max, CG above
pred = G(m).values
chi2 = float(np.sum((pred - d_obs) ** 2) / SIGMA ** 2)
print(f"alpha = {alpha:.3e}, chi^2/n = {chi2 / n:.3f}, rms(d) = {np.std(d_obs):.3f} s", flush=True)


# ---------------------------------------------------------------- sections and maps
def plane(e1, e2, N=201):
    u = np.linspace(-1, 1, N)
    U, V = np.meshgrid(u, u)
    inside = U ** 2 + V ** 2 < 1
    return U[inside][:, None] * e1 + V[inside][:, None] * e2, inside


def depth_map(depth, step=2.0):
    lon, lat = np.meshgrid(np.arange(-180, 180 + step, step), np.arange(-90, 90 + step, step))
    return ((1 - depth / R) * xyz(lat, lon)).reshape(-1, 3), lon.shape


SECTIONS = {"equatorial plane": plane(xyz(0, 0), xyz(0, 90)),
            "meridian 15 E / 165 W": plane(xyz(0, 15), xyz(90, 0)),
            "depth 200 km": depth_map(200.0),
            "depth 2700 km": depth_map(2700.0)}
total = R * K.sum()                                   # sum_i |g_i|: every travel-time representer is <= 0
vals = {}
for name, (pts, shape) in SECTIONS.items():
    vals[name] = dict(true=dvp(*pts.T), solution=np.ravel(m(*pts.T)), total=np.ravel(total(*pts.T)))
lap("evaluate model and total sensitivity")


def show(ax, name, v, cmap, lim=None, title=""):
    pts, shape = SECTIONS[name]
    if name.startswith("depth"):
        im = ax.imshow(v.reshape(shape), origin="lower", extent=(-180, 180, -90, 90), cmap=cmap,
                       vmin=-lim if lim else None, vmax=lim)
        ax.set(xlabel="longitude", ylabel="latitude")
    else:
        img = np.full(shape.shape, np.nan)
        img[shape] = v
        im = ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, vmin=-lim if lim else None, vmax=lim)
        for rr in (R_CMB, R_ICB, 1.0):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, lw=0.6, color="0.3"))
        ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{title} {name}", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.75)


# representers of four rays, each on its own great-circle plane
dist = np.array([np.degrees(np.arccos(np.clip(EV[r[0]][0] @ ST[r[1]], -1, 1))) for r in data])
picks = [min((i for i, r in enumerate(data) if r[2] == ph), key=lambda i: abs(dist[i] - target))
         for ph, target in (("P", 70), ("PcP", 50), ("PKP", 160), ("PKIKP", 150))]
fig, axs = plt.subplots(2, 2, figsize=(10, 9.5))
for ax, i in zip(axs.flat, picks):
    e, s, ph = data[i][:3]
    e1 = EV[e][0]
    e2 = ST[s] - (ST[s] @ e1) * e1
    e2 /= np.linalg.norm(e2)
    pts, inside = plane(e1, e2, 241)
    img = np.full(inside.shape, np.nan)
    img[inside] = -R * np.ravel(K[i](*pts.T))
    im = ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), cmap="Blues_r")
    ax.plot(data[i][4] @ e1, data[i][4] @ e2, color="#d95f02", lw=0.8)
    for rr in (R_CMB, R_ICB, 1.0):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, lw=0.6, color="0.3"))
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{ph}, {dist[i]:.0f} deg, source {EV[e][1]:.0f} km: "
                 "g_i(x) = dt_i of the bump K(., x)", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.75, label="s")
fig.suptitle("Riesz representers of travel times in H^2 (l = 500 km); ray in orange")
fig.savefig(f"{OUT}/representers.png", dpi=150, bbox_inches="tight")

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for ax, name in zip(axs.flat, SECTIONS):
    show(ax, name, vals[name]["total"], "Blues", title="total sensitivity sum |g_i|,")
fig.suptitle(f"Total sensitivity of {n} travel times (Riesz representers)")
fig.savefig(f"{OUT}/total_sensitivity.png", dpi=150, bbox_inches="tight")

fig, axs = plt.subplots(4, 2, figsize=(11, 19))
for row, name in zip(axs, SECTIONS):
    lim = np.max(np.abs(vals[name]["true"])) or 0.05
    show(row[0], name, vals[name]["true"], "RdBu", lim, "true dVp (km/s),")
    show(row[1], name, vals[name]["solution"], "RdBu", lim, "regularised minimum norm,")
fig.suptitle(f"dVp: truth and solution (alpha = {alpha:.2e}, chi^2/n = {chi2 / n:.2f})")
fig.savefig(f"{OUT}/models.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 5))
for p, z in EV:
    ax.plot(np.degrees(np.arctan2(p[1], p[0])), np.degrees(np.arcsin(p[2])), "o", ms=3, color="#d95f02")
ax.plot(np.degrees(np.arctan2(ST[:, 1], ST[:, 0])), np.degrees(np.arcsin(ST[:, 2])), "^", ms=4, color="#1b9e77")
ax.set(xlim=(-180, 180), ylim=(-90, 90), title=f"{len(EV)} events (circles), {len(ST)} stations (triangles)")
fig.savefig(f"{OUT}/geometry.png", dpi=120, bbox_inches="tight")
lap("plots")

phases = {ph: sum(r[2] == ph for r in data) for ph in sorted({r[2] for r in data})}
costs.update(n_data=n, phases=phases, events=len(EV), stations=len(ST), candidate_arrivals=len(cand),
             segments=int(len(W)), workers=WORKERS, alpha=float(alpha), chi2_per_datum=chi2 / n,
             taup_time_check_s=dict(median=float(np.median(check)), max=float(check.max())),
             relative_model_misfit={k: float(np.linalg.norm(v["solution"] - v["true"]) / np.linalg.norm(v["true"]))
                                    for k, v in vals.items()},
             max_rss_gb=dict(main=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                             workers=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20))
json.dump(costs, open(f"{OUT}/costs.json", "w"), indent=1)
print(json.dumps(costs, indent=1))
