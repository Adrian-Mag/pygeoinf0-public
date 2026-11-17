"""Linearised body-wave tomography of a layered Moon: SensRay (weber_core, TauP rays) and pygeoinf0.

    planet    the ball of lunar radius R cut at the inner-core, core-mantle, partial-melt and Moho radii (pg.LayeredBall)
    unknowns  dVp and dVs in the partial-melt layer, the mantle and the crust: H^2_l on each layer, independent across
              interfaces (pg.PiecewiseSpace.sobolev); outward topography of the Moho and of the core-mantle boundary (km, H^2 on spheres)
    data      P, S (first arrivals), PcP and ScS from four Apollo-like stations and synthetic moonquakes and impacts:
                  dt_i = -R sum_layers int dv / v^2 ds  +  sum over Moho and CMB crossings of c h(x),
              c = sum over the two leg ends at the crossing of (+eta if the leg is below, -eta if above),
              eta = sqrt(v^-2 - p^2 / r^2) of that leg (experiments/EXP-027-layered-moon/NOTES.md, checked there against
              TauP and exact travel times)
    solution  the same lines as the ladder: G = pg.Functionals(X.riesz(data), into=D); alpha by the discrepancy
              principle; pg.probability.cylindrical(X, alpha).condition(G, d): posterior mean and pointwise standard deviations

Each ray is TauP's path as a polyline (lengths in units of R) with vertices at every discontinuity of weber_core: its
velocities are constant in each layer, so rays are straight chords there and the polyline is exact once collinear
points are dropped; each segment carries 1/v^2 of its leg's wave type. Synthetic data come from a known model by Gauss
quadrature along the same polylines and the true topography at the crossing points (not through pygeoinf0), plus noise.

Rays come through the obspy plugin (plugins/pygeoinf0-obspy, plan 019), whose functions are this example's former
ray code, unchanged. Run on europa (obspy, sensray, pygeoinf0 and pygeoinf0-obspy importable):  python moon_tomography.py   ->  $MOON_OUT/*.png, costs.json
"""
import json
import os
import sys
import resource
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sensray import PlanetModel

import pygeoinf0 as pg
import pygeoinf0_obspy as po                     # the obspy plugin (plugins/pygeoinf0-obspy): TauP paths

N_DATA = int(os.environ.get("N_DATA", 1000))
OUT = os.path.abspath(os.environ.get("MOON_OUT", "moon_synthetic"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2027)
costs, _t = {}, [time.time(), sum(os.times()[:4])]


def lap(name):
    now, cpu = time.time(), sum(os.times()[:4])
    costs[name] = dict(wall_s=round(now - _t[0], 2), cpu_s=round(cpu - _t[1], 1))
    print(f"{name:34s} {now - _t[0]:9.1f} s wall {cpu - _t[1]:10.1f} s cpu", flush=True)
    _t[:] = [now, cpu]


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack(np.broadcast_arrays(np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)), axis=-1)


# ---------------------------------------------------------------- reference model, planet, stations and events
moon = PlanetModel.from_standard_model("weber_core")
R = float(moon.radius)                                         # 1738 km
taup = moon.taupy_model
DISC = np.unique(np.concatenate([layer["depth"] for layer in moon.layers.values()]))     # every .nd depth
Z_MOHO, Z_PML, Z_CMB, Z_ICB = 38.0, 1258.0, 1408.0, 1498.0
planet = pg.LayeredBall([1 - Z_ICB / R, 1 - Z_CMB / R, 1 - Z_PML / R, 1 - Z_MOHO / R, 1.0],
                        names=["inner core", "outer core", "pml", "mantle", "crust"],
                        interfaces=["icb", "cmb", "pml top", "moho"])
SOLID = ["pml", "mantle", "crust"]
STATIONS = {"S12": (-3.01, -23.42), "S14": (-3.64, -17.48), "S15": (26.13, 3.63), "S16": (-8.97, 15.50)}
ST = np.array([xyz(*v) for v in STATIONS.values()])
PHASES = {"P": ("p", "P"), "S": ("s", "S"), "PcP": ("PcP",), "ScS": ("ScS",)}       # first arrival of each kind
WAVE = {"P": "vp", "S": "vs", "PcP": "vp", "ScS": "vs"}


def events():
    out = []
    for _ in range(50):                                        # deep moonquake nests, nearside
        out.append(("deep", xyz(rng.uniform(-45, 45), rng.uniform(-60, 60)), rng.uniform(750, 1100)))
    for _ in range(30):                                        # shallow moonquakes
        out.append(("shallow", xyz(np.degrees(np.arcsin(rng.uniform(-1, 1))), rng.uniform(-180, 180)),
                    rng.uniform(5, 100)))
    for _ in range(40):                                        # impacts, at the surface
        out.append(("impact", xyz(np.degrees(np.arcsin(rng.uniform(-1, 1))), rng.uniform(-180, 180)), 0.0))
    return out


EV = events()


def rays(e, s):
    """The selected arrivals between event e and station s: dicts with the phase kind, TauP's time and ray parameter,
    the polyline (units of R), the 1/v^2 of each segment and the path time recomputed from the polyline."""
    kind, src, z = EV[e]
    rec = ST[s]
    arrs, D, t = po.arrivals(taup, src, z, rec, [p for ps in PHASES.values() for p in ps])
    out = []
    for name, members in PHASES.items():
        cand = [a for a in arrs if a.name in members]
        if not cand:
            continue
        arr = min(cand, key=lambda a: a.time)
        X, dep = po.path_points(arr, src, t, R)
        X = X[po.thin_by_bends(X, dep, DISC)]                  # vertices at discontinuities, and where the path bends
        w, t_path = po.midpoint_slowness2(X, R, lambda r, wave=WAVE[name]: moon.get_property_at_radius(wave, r))
        out.append(dict(event=e, station=s, phase=name, taup_name=arr.name, time=arr.time, p=arr.ray_param, X=X,
                        w=w, t_path=t_path, distance=float(np.degrees(D))))
    return out


cand = [r for e in range(len(EV)) for s in range(len(ST)) for r in rays(e, s)]
lap("rays: TauP + polylines")
data = [cand[i] for i in np.sort(rng.choice(len(cand), min(N_DATA, len(cand)), replace=False))]
n = len(data)
path_check = np.array([abs(r["t_path"] - r["time"]) for r in data])
print(f"{len(cand)} candidate arrivals; {n} selected; |T(polyline) - T(TauP)| max {path_check.max():.2e} s; "
      f"{sum(len(r['w']) for r in data)} segments", flush=True)


# ---------------------------------------------------------------- interface terms at the crossings
def coefficients(r):
    """[(interface, point, c)] for the Moho and CMB crossings of a ray: c = sum over the leg ends of (+eta below,
    -eta above), eta from TauP's ray parameter and the leg's speed; and the same eta from the polyline's incidence."""
    out, check = [], 0.0
    X = planet.cut(r["X"])[0]                                  # the vertices crossings() refers to
    for c in planet.crossings(r["X"]):
        if c.interface not in ("moho", "cmb"):
            continue
        rk = R * np.linalg.norm(c.point)
        total = 0.0
        for side, cos, j in zip(c.sides, c.cosines, (c.vertex - 1, c.vertex)):
            if side is None:
                continue
            v = float(moon.get_property_at_radius(WAVE[r["phase"]], R * np.linalg.norm(0.5 * (X[j] + X[j + 1]))))
            eta = np.sqrt(max(1 / v ** 2 - (r["p"] / rk) ** 2, 0.0))
            check = max(check, abs(eta - cos / v))
            total += eta if side == "below" else -eta
        out.append((c.interface, c.point, total))
    return out, check


terms, eta_check = [], 0.0
for r in data:
    t, e = coefficients(r)
    terms.append(t)
    eta_check = max(eta_check, e)
print(f"interface terms: {sum(len(t) for t in terms)} crossings; |eta(p) - cos(i)/v| max {eta_check:.1e} s/km", flush=True)


# ---------------------------------------------------------------- the true model (km/s and km), units of R
def bump(X, lat, lon, depth, width_km):
    c = (1 - depth / R) * xyz(lat, lon)
    return np.exp(-np.sum((X - c) ** 2, axis=-1) * R ** 2 / (2 * width_km ** 2))


def true_dv(layer, wave, X):
    X = np.asarray(X, float)
    up = X @ xyz(10, -20) / np.maximum(np.linalg.norm(X, axis=-1), 1e-12)
    if layer == "crust":
        dvp = -0.35 / (1 + np.exp(-(up - 0.7) / 0.08)) + 0.15 / (1 + np.exp((up + 0.3) / 0.15))   # slow maria, fast farside
        return dvp if wave == "vp" else 0.6 * dvp
    if layer == "mantle":
        dvp = -0.25 * bump(X, 0, -10, 450, 220) + 0.2 * bump(X, 30, 25, 900, 180)
        return dvp if wave == "vp" else 0.5 * dvp - 0.15 * bump(X, -25, -30, 700, 200)
    return (0.1 if wave == "vp" else -0.1) * X[..., 0] / max(1 - Z_PML / R, 1e-12)  # pml: degree 1


def true_h(interface, X):
    u = np.asarray(X, float) / np.linalg.norm(X, axis=-1, keepdims=True)
    if interface == "moho":
        return 10.0 * np.exp(-np.sum((u - xyz(5, -15)) ** 2, axis=-1) / (2 * 0.45 ** 2)) - 4.0
    z = u @ xyz(30, 60)
    return 8.0 * (1.5 * z ** 2 - 0.5)                                              # cmb: degree 2


tq, gq = np.polynomial.legendre.leggauss(32)
parts = {k: np.zeros(n) for k in ("vp", "vs", "moho", "cmb")}       # the clean data, summand by summand
for i, r in enumerate(data):
    P, wr = r["X"], r["w"]
    lay = planet.layer_of(0.5 * (P[1:] + P[:-1]))
    for j in range(len(wr)):
        name = planet.names[lay[j]]
        if name not in SOLID:
            continue
        Q = P[j] + 0.5 * (tq[:, None] + 1) * (P[j + 1] - P[j])
        parts[WAVE[r["phase"]]][i] -= R * wr[j] * 0.5 * np.linalg.norm(P[j + 1] - P[j]) * (gq @ true_dv(name, WAVE[r["phase"]], Q))
    for k, x, c in terms[i]:
        parts[k][i] += c * true_h(k, x)
clean = sum(parts.values())
sigma = np.array([0.5 if WAVE[r["phase"]] == "vp" else 1.0 for r in data])       # s: picking errors, P and S
d_obs = clean + sigma * rng.normal(size=n)
lap("synthetic data")


# ---------------------------------------------------------------- the inverse problem in pygeoinf0
ELL = {"pml": 150 / R, "mantle": 250 / R, "crust": 120 / R}
SIG = {"vp": 0.2, "vs": 0.12, "moho": 10.0, "cmb": 15.0}            # prior pointwise std at alpha = 1 (km/s, km)


def scale(H, x0, s):
    """The weight that gives point values in H the prior standard deviation s at alpha = 1: ||k_x0||^2 / s^2."""
    return H.riesz(np.asarray(x0, float)).norm() ** 2 / s ** 2


inner = {"pml": 0.5 * (planet.radii[1] + planet.radii[2]), "mantle": 0.7, "crust": 1 - 0.5 * Z_MOHO / R}
spaces = {}
for wave in ("vp", "vs"):
    V = pg.PiecewiseSpace.sobolev(planet, 2, length_scale=ELL, pieces=SOLID)
    w = {k: scale(V.summand(k), [inner[k], 0, 0], SIG[wave]) for k in SOLID}
    spaces[wave] = pg.PiecewiseSpace.sobolev(planet, 2, length_scale=ELL, pieces=SOLID, weights=w)
spaces["moho"] = pg.Sobolev(planet.interface("moho"), 2, length_scale=200 / R)
spaces["cmb"] = pg.Sobolev(planet.interface("cmb"), 2, length_scale=150 / R)
weights = {"moho": scale(spaces["moho"], [planet.radii[3], 0, 0], SIG["moho"]),
           "cmb": scale(spaces["cmb"], [planet.radii[1], 0, 0], SIG["cmb"])}
X = pg.DirectSum(spaces, weights=weights)
functionals = []
for r, t in zip(data, terms):
    f = {WAVE[r["phase"]]: -R * pg.LebesgueMeasure(pg.Polyline(r["X"])).with_density(r["w"])}
    for k in ("moho", "cmb"):
        here = [c * pg.DiracMeasure(x) for kk, x, c in t if kk == k]
        if here:
            f[k] = sum(here[1:], here[0])
    functionals.append(f)
D = pg.Euclidean(n, covariance=sigma ** 2)
G = pg.Functionals(X.riesz(functionals), into=D)          # dv, h -> travel-time anomalies (s)
lap("Riesz representers (split at interfaces)")
G.gram()
d = D(d_obs)
lap("Gram matrix (exact energies)")
alpha = pg.meta.discrepancy(pg.Problem(mdr=G, datum=d)).gamma
lap("discrepancy principle (eigh)")
post = pg.probability.cylindrical(X, alpha=alpha).condition(G, d)
m = post.mean
lap("posterior mean (Cholesky)")
x_true = X({**{w: {k: (lambda x, y, z, k=k, w=w: true_dv(k, w, np.stack(np.broadcast_arrays(x, y, z), axis=-1)))
                   for k in SOLID} for w in ("vp", "vs")},
            **{k: (lambda x, y, z, k=k: true_h(k, np.stack(np.broadcast_arrays(x, y, z), axis=-1))) for k in ("moho", "cmb")}})
forward_parts = {k: float(np.max(np.abs(G(X.embed(k, x_true[k])).values - parts[k]))) for k in parts}
forward_check = float(np.max(np.abs(G(x_true).values - clean)))       # G on the truth (formulas) vs the quadrature data
print(f"max |G(true model) - synthetic clean data| = {forward_check:.2e} s (noise 0.5 / 1.0 s); by summand "
      + ", ".join(f"{k} {v:.1e}" for k, v in forward_parts.items()), flush=True)
lap("forward check (G on the true model)")
pred = G(m).values
chi2 = float(np.sum((pred - d_obs) ** 2 / sigma ** 2))
chi2_stated = float(np.sum((G(pg.probability.cylindrical(X, alpha=1.0).condition(G, d).mean).values - d_obs) ** 2 / sigma ** 2))
print(f"alpha = {alpha:.3e}, chi^2/n = {chi2 / n:.3f}, rms(d) = {np.std(d_obs):.3f} s; at the stated prior "
      f"(alpha = 1) chi^2/n = {chi2_stated / n:.3f}", flush=True)


# ---------------------------------------------------------------- sections and maps
def plane(lon, N):
    """Points of the meridian plane through longitudes lon and lon + 180 (x: equatorial, y: polar), inside the ball."""
    e1, e2 = xyz(0, lon), np.array([0.0, 0.0, 1.0])
    u = np.linspace(-1, 1, N)
    U, W = np.meshgrid(u, u)
    inside = U ** 2 + W ** 2 < 1
    return U[inside][:, None] * e1 + W[inside][:, None] * e2, inside


def interface_map(name, step):
    lon, lat = np.meshgrid(np.arange(-180, 180 + step, step), np.arange(-90, 90 + step, step))
    return planet.radii[planet.interface_index(name)] * xyz(lat, lon).reshape(-1, 3), lon.shape


PLANE = plane(-10, 121)
SPARSE = plane(-10, 41)
sec = {}
for wave in ("vp", "vs"):
    pts, inside = PLANE
    k = planet.layer_of(pts, side="below")
    truth = np.full(len(pts), np.nan)
    for name in SOLID:
        sel = k == planet.index(name)
        truth[sel] = true_dv(name, wave, pts[sel])
    sec[wave] = dict(true=truth, mean=spaces[wave].evaluate(m[wave], pts, side="below"))
lap("evaluate the mean on sections")
for wave in ("vp", "vs"):
    pts, _ = SPARSE
    solid = np.isin(planet.layer_of(pts, side="below"), [planet.index(s) for s in SOLID])
    std = np.full(len(pts), np.nan)
    std[solid] = post.std({wave: pts[solid]})
    sec[wave]["std"] = std
    sec[wave]["prior std"] = SIG[wave] / np.sqrt(alpha)
lap("posterior std on sections")
maps = {}
for name in ("moho", "cmb"):
    pts, shape = interface_map(name, 3.0)
    maps[name] = dict(true=true_h(name, pts), mean=np.ravel(m[name](pts)))
    spts, sshape = interface_map(name, 9.0)
    maps[name]["std"], maps[name]["sshape"] = post.std({name: spts}), sshape
    maps[name]["shape"] = shape
lap("interface maps (mean and std)")


def show_section(ax, vals, inside, cmap, lim, title):
    img = np.full(inside.shape, np.nan)
    img[inside] = vals
    im = ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, vmin=-lim if lim else None, vmax=lim)
    for rr in planet.radii:
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, lw=0.5, color="0.3"))
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.75)


fig, axs = plt.subplots(2, 3, figsize=(13, 8.5))
for row, wave in zip(axs, ("vp", "vs")):
    lim = float(np.nanmax(np.abs(sec[wave]["true"]))) or 0.1
    show_section(row[0], sec[wave]["true"], PLANE[1], "RdBu", lim, f"true d{wave} (km/s)")
    show_section(row[1], sec[wave]["mean"], PLANE[1], "RdBu", lim, f"posterior mean d{wave}")
    show_section(row[2], sec[wave]["std"], SPARSE[1], "viridis", None,
                 f"posterior std d{wave} (prior {sec[wave]['prior std']:.3f})")
fig.suptitle(f"Meridian section 10 W / 170 E; alpha = {alpha:.2e}, chi^2/n = {chi2 / n:.2f}, n = {n}")
fig.savefig(f"{OUT}/sections.png", dpi=140, bbox_inches="tight")

fig, axs = plt.subplots(2, 3, figsize=(15, 7))
for row, name in zip(axs, ("moho", "cmb")):
    M = maps[name]
    lim = float(np.abs(M["true"]).max())
    for ax, key, cm, sh in ((row[0], "true", "RdBu", M["shape"]), (row[1], "mean", "RdBu", M["shape"]),
                            (row[2], "std", "viridis", M["sshape"])):
        im = ax.imshow(M[key].reshape(sh), origin="lower", extent=(-180, 180, -90, 90), cmap=cm,
                       vmin=-lim if key != "std" else None, vmax=lim if key != "std" else None)
        pts = np.array([x for t in terms for kk, x, _ in t if kk == name])
        if len(pts) and key != "true":
            ax.plot(np.degrees(np.arctan2(pts[:, 1], pts[:, 0])), np.degrees(np.arcsin(pts[:, 2] / np.linalg.norm(pts, axis=1))),
                    ".", ms=1.5, color="k")
        ax.set_title(f"{name} topography (km, outward): {key}", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
fig.suptitle("Interfaces; dots: crossing and reflection points of the rays")
fig.savefig(f"{OUT}/interfaces.png", dpi=140, bbox_inches="tight")


def depth_map(depth, step):
    lon, lat = np.meshgrid(np.arange(-180, 180 + step, step), np.arange(-90, 90 + step, step))
    return (1 - depth / R) * xyz(lat, lon).reshape(-1, 3), lon.shape


crust = {}
fig, axs = plt.subplots(2, 3, figsize=(15, 7))
for row, wave in zip(axs, ("vp", "vs")):
    pts, shape = depth_map(0.5 * Z_MOHO, 3.0)
    spts, sshape = depth_map(0.5 * Z_MOHO, 10.0)
    crust[wave] = dict(true=true_dv("crust", wave, pts), mean=spaces[wave].evaluate(m[wave], pts),
                       std=post.std({wave: spts}))
    lim = float(np.abs(crust[wave]["true"]).max())
    for ax, key, cm, sh in ((row[0], "true", "RdBu", shape), (row[1], "mean", "RdBu", shape),
                            (row[2], "std", "viridis", sshape)):
        im = ax.imshow(crust[wave][key].reshape(sh), origin="lower", extent=(-180, 180, -90, 90), cmap=cm,
                       vmin=-lim if key != "std" else None, vmax=lim if key != "std" else None)
        ax.set_title(f"d{wave} at {0.5 * Z_MOHO:g} km depth (crust): {key}", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
    row[0].plot(np.degrees(np.arctan2(ST[:, 1], ST[:, 0])), np.degrees(np.arcsin(ST[:, 2])), "^", ms=6, color="k")
fig.suptitle(f"Crust (prior std {SIG['vp'] / np.sqrt(alpha):.3f} / {SIG['vs'] / np.sqrt(alpha):.3f} km/s at alpha); "
             "triangles: stations")
fig.savefig(f"{OUT}/crust.png", dpi=140, bbox_inches="tight")
lap("crust maps (mean and std)")

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for kind, mk, col in (("deep", "o", "#7570b3"), ("shallow", "s", "#d95f02"), ("impact", "x", "#666666")):
    P = np.array([p for k, p, _ in EV if k == kind])
    axs[0].plot(np.degrees(np.arctan2(P[:, 1], P[:, 0])), np.degrees(np.arcsin(P[:, 2])), mk, ms=4, color=col, label=kind)
axs[0].plot(np.degrees(np.arctan2(ST[:, 1], ST[:, 0])), np.degrees(np.arcsin(ST[:, 2])), "^", ms=8, color="#1b9e77",
            label="stations")
axs[0].set(xlim=(-180, 180), ylim=(-90, 90), title="events and Apollo-like stations")
axs[0].legend(fontsize=8)
for r in data[::4]:
    e1 = EV[r["event"]][1]
    e2 = ST[r["station"]] - (ST[r["station"]] @ e1) * e1
    e2 /= np.linalg.norm(e2)
    axs[1].plot(r["X"] @ e1, r["X"] @ e2, lw=0.4, color={"P": "#1f78b4", "S": "#e31a1c", "PcP": "#33a02c",
                                                         "ScS": "#ff7f00"}[r["phase"]])
for rr in planet.radii:
    axs[1].add_patch(plt.Circle((0, 0), rr, fill=False, lw=0.6, color="0.3"))
axs[1].set(aspect="equal", xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), title="every 4th ray in its own plane (P, S, PcP, ScS)")
fig.savefig(f"{OUT}/geometry.png", dpi=130, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist((pred - d_obs) / sigma, bins=40, color="0.5")
ax.set(xlabel="(predicted - observed) / sigma", title=f"data fit: chi^2/n = {chi2 / n:.2f}")
fig.savefig(f"{OUT}/data_fit.png", dpi=130, bbox_inches="tight")
lap("plots")


def misfit(true, est):
    ok = np.isfinite(true) & np.isfinite(est)
    return float(np.linalg.norm(est[ok] - true[ok]) / np.linalg.norm(true[ok]))


phases = {k: sum(r["phase"] == k for r in data) for k in PHASES}
costs.update(n_data=n, phases=phases, events=len(EV), stations=len(ST), candidate_arrivals=len(cand),
             segments=int(sum(len(r["w"]) for r in data)), interface_terms=int(sum(len(t) for t in terms)),
             summands=[str(k) for k in X.names], alpha=float(alpha), chi2_per_datum=chi2 / n,
             path_time_check_s=float(path_check.max()), eta_check_s_per_km=float(eta_check),
             forward_check_s=forward_check, forward_check_by_summand_s=forward_parts,
             chi2_per_datum_at_alpha_1=chi2_stated / n,
             truth_rms_over_prior_std={**{f"d{w} section": float(np.sqrt(np.nanmean(sec[w]["true"] ** 2)) / SIG[w])
                                          for w in ("vp", "vs")},
                                       **{f"d{w} crust map": float(np.sqrt(np.mean(crust[w]["true"] ** 2)) / SIG[w])
                                          for w in ("vp", "vs")},
                                       **{k: float(np.sqrt(np.mean(maps[k]["true"] ** 2)) / SIG[k]) for k in ("moho", "cmb")}},
             relative_misfit={**{f"d{w} section": misfit(sec[w]["true"], sec[w]["mean"]) for w in ("vp", "vs")},
                              **{f"d{w} crust map": misfit(crust[w]["true"], crust[w]["mean"]) for w in ("vp", "vs")},
                              **{f"{k} map": misfit(maps[k]["true"], maps[k]["mean"]) for k in ("moho", "cmb")}},
             workers=pg.settings.workers, threads=os.environ.get("OPENBLAS_NUM_THREADS"),
             max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20))
json.dump(costs, open(f"{OUT}/costs.json", "w"), indent=1)
print(json.dumps(costs, indent=1))
