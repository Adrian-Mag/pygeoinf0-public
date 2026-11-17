"""Linearised P and S travel-time tomography of the Moon from the Apollo picks: real data, SensRay and pygeoinf0.

    catalogue  Nunn_catalog_picks.csv (the author's analysis_of_lunar_data): P and S arrival times at Apollo 12, 14, 15
               and 16 of artificial impacts (known sources), meteoroid impacts, shallow and deep moonquakes (stacked
               nests), compiled by Nunn et al. (2020, pp. 23-26), with event locations and origin times t_0 (estimated
               for natural events); LanderLocs.csv for the ALSEP positions
    reference  a 1-D model of SensRay (default M1: Garcia et al. 2019, Table 11, the seismic-only inversion of this data
               set). The planet's interfaces are the model's discontinuities (a depth with a jump in vp or vs); for M1
               the only one is the core-mantle boundary (1340 km), so the crust is part of the mantle layer
    unknowns   dVp and dVs in every layer above the CMB (H^2_l restriction spaces, pg.PiecewiseSpace.sobolev); CMB topography (H^2 on the
               CMB sphere: direct P and S see it only by grazing, so it keeps its prior); P and S station terms as
               zero-mean contrasts over the four stations (Garcia et al. 2019, p. 35), a Euclidean summand
    nuisance   origin times of the natural events: free, a span of per-event indicators in the data space (shared by the
               event's P and S picks); the posterior is conditioned on the quotient datum (plan 008, EXP-028 DESIGN 1.7).
               Artificial impacts keep their known times
    noise      per pick: the catalogue's spread across studies, at least the class value of Nunn et al. (2020, p. 26)
    solution   alpha by the discrepancy principle on the quotient; posterior mean and pointwise standard deviations

Each ray is TauP's path in the reference model, simplified (Douglas-Peucker) to a polyline that stays within EPS_KM of
it, with vertices at every discontinuity; each segment carries the path average of 1/v^2 (s^2/km^2) over its part of
the path. Run on europa (experiments/EXP-029-moon-real-data/run_europa.sh): python moon_real_data.py -> $MOON_OUT/.
"""
import csv
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

MODEL = os.environ.get("MOON_MODEL", "M1")
CATS = os.environ.get("CATALOGUE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments",
                                                   "EXP-029-moon-real-data", "data"))
OUT = os.path.abspath(os.environ.get("MOON_OUT", "moon_real"))
N_PICKS = int(os.environ.get("N_PICKS", 0))           # 0: every usable pick; otherwise a random subset (smoke runs)
GROUPS = os.environ.get("EVENT_GROUPS", "artificial,M,SH,A").split(",")   # W and AW: provenance unknown (NOTES R1)
EPS_KM = 0.5                                           # polylines within 0.5 km of TauP's paths
STATIONS = ["Apollo 12", "Apollo 14", "Apollo 15", "Apollo 16"]
CLASS_STD = {"artificial": (1.3, 3.7), "meteoroid": (4.0, 18.2), "shallow": (3.0, 13.2), "deep": (10.1, 3.7)}
SIG = {"vp": 0.2, "vs": 0.12, "cmb": 10.0, "station": 2.3}     # prior pointwise std at alpha = 1 (km/s, km, s)
LOCATIONS = os.environ.get("LOCATIONS", "1") == "1"   # linearised hypocentre shifts of natural events (a summand)
SIG_LOC = {"deep": (30.0, 30.0), "shallow": (30.0, 50.0), "meteoroid": (30.0, 0.0)}   # km: horizontal, depth (0: fixed)
ELL_KM = 300.0
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2029)
costs, _t = {}, [time.time(), sum(os.times()[:4])]


def lap(name):
    now, cpu = time.time(), sum(os.times()[:4])
    costs[name] = dict(wall_s=round(now - _t[0], 2), cpu_s=round(cpu - _t[1], 1))
    print(f"{name:34s} {now - _t[0]:9.1f} s wall {cpu - _t[1]:10.1f} s cpu", flush=True)
    _t[:] = [now, cpu]


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack(np.broadcast_arrays(np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)), axis=-1)


def group(name):
    """The catalogue group of an event, by its name: artificial (S-IVB, LM), M, SH, A, and W, AW (t_0 = 0, no spread)."""
    if "S-IVB" in name or "LM" in name:
        return "artificial"
    for g in ("AW", "SH", "A", "M", "W"):
        if name.startswith(g):
            return g
    return "other"


def kind(name):
    if "S-IVB" in name or "LM" in name:
        return "artificial"
    if name.startswith("SH"):
        return "shallow"
    if name.startswith("A"):                           # A (Lognonne et al. 2003 nests) and AW: deep moonquakes
        return "deep"
    return "meteoroid"                                 # M (meteoroid impacts) and W (surface events: NOTES.md R1)


# ---------------------------------------------------------------- catalogue
lander = {r["Lander"]: xyz(float(r["Latitude"]), float(r["Longitude"]))
          for r in csv.DictReader(open(f"{CATS}/LanderLocs.csv", encoding="utf-8-sig")) if r["Element"] == "ALSEP"}
rows = list(csv.DictReader(open(f"{CATS}/Nunn_catalog_picks.csv", encoding="utf-8-sig")))
EVENTS = {}
for r in rows:
    EVENTS.setdefault(r["Event Name"], dict(pos=xyz(float(r["Lat"]), float(r["Lon"])), depth=float(r["Depth"]),
                                            t0=float(r["t_0"]), kind=kind(r["Event Name"])))
rows = [r for r in rows if group(r["Event Name"]) in GROUPS]
if N_PICKS:
    rows = [rows[i] for i in np.sort(rng.choice(len(rows), min(N_PICKS, len(rows)), replace=False))]

# ---------------------------------------------------------------- reference model and planet
moon = PlanetModel.from_standard_model(MODEL)
R = float(moon.radius)
taup = moon.taupy_model
dep = np.concatenate([layer["depth"] for layer in moon.layers.values()])
vps = np.concatenate([layer["vp"] for layer in moon.layers.values()])
vss = np.concatenate([layer["vs"] for layer in moon.layers.values()])
JUMPS = sorted({float(dep[i]) for i in range(1, len(dep))
                if dep[i] == dep[i - 1] and (vps[i] != vps[i - 1] or vss[i] != vss[i - 1]) and 0 < dep[i] < R})
Z_CMB = float(moon.layers["outer-core"]["depth"][0]) if "outer-core" in moon.layers else None
tops = [0.0] + JUMPS
names = [f"{z:g}-{b:g} km" for z, b in zip(tops, JUMPS + [R])][::-1]                  # by depth range, centre first
planet = pg.LayeredBall([1 - z / R for z in JUMPS[::-1]] + [1.0], names=names,
                        interfaces=["cmb" if z == Z_CMB else f"{z:g} km" for z in JUMPS[::-1]])
SOLID = [f"{z:g}-{b:g} km" for z, b in zip(tops, JUMPS + [R]) if Z_CMB is None or z < Z_CMB][::-1]   # above the CMB
print(f"{MODEL}: R = {R} km; discontinuities {JUMPS} km; layers {planet.names}", flush=True)


# ---------------------------------------------------------------- rays
def simplify(X, forced, eps):
    """Indices of a polyline through X that stays within eps of every point of X: Douglas-Peucker between forced vertices."""
    keep = set(int(i) for i in forced)
    stack = [(int(a), int(b)) for a, b in zip(forced[:-1], forced[1:])]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        u = X[j] - X[i]
        dist = np.linalg.norm(np.cross(X[i + 1:j] - X[i], u), axis=1) / np.linalg.norm(u)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep.add(m)
            stack += [(i, m), (m, j)]
    return np.array(sorted(keep))


def ray(r):
    """The first arrival of the pick's phase in the reference model as a weighted polyline (units of R), or None."""
    ev, wave = EVENTS[r["Event Name"]], "vp" if r["Phase"] == "P" else "vs"
    src, rec = ev["pos"], lander[r["Station"]]
    D = np.arccos(np.clip(src @ rec, -1, 1))
    arrs = [a for a in taup.get_ray_paths(ev["depth"], np.degrees(D), [r["Phase"], r["Phase"].lower()])
            if a.name in (r["Phase"], r["Phase"].lower())]
    if not arrs:
        return None
    arr = min(arrs, key=lambda a: a.time)
    t = (rec - np.cos(D) * src) / np.sin(D)
    th, z = arr.path["dist"], arr.path["depth"]
    X = (1 - z / R)[:, None] * (np.cos(th)[:, None] * src + np.sin(th)[:, None] * t)
    ok = np.r_[True, np.linalg.norm(np.diff(X, axis=0), axis=1) > 1e-12]
    X, z = X[ok], z[ok]
    ds = np.linalg.norm(np.diff(X, axis=0), axis=1) * R
    v = moon.get_property_at_radius(wave, R * np.linalg.norm(0.5 * (X[1:] + X[:-1]), axis=1))
    p_src = (X[1] - X[0]) / (np.linalg.norm(X[1] - X[0]) * v[0])     # slowness vector leaving the source (s/km)
    forced = np.unique(np.r_[0, np.flatnonzero(np.min(np.abs(z[:, None] - np.array(JUMPS + [-1.0])), axis=1) < 1e-6),
                             len(X) - 1])
    keep = simplify(X, forced, EPS_KM / R)
    w = np.add.reduceat(ds / v ** 2, keep[:-1]) / (np.linalg.norm(np.diff(X[keep], axis=0), axis=1) * R)
    return dict(X=X[keep], w=w, p=arr.ray_param, T=arr.time, p_src=p_src, t_path=float(np.sum(ds / v)), wave=wave,
                station=STATIONS.index(r["Station"]), event=r["Event Name"], phase=r["Phase"],
                obs=float(r["median arrival time"]) - ev["t0"], std=float(r["std dev arrival time"]), name=arr.name)


data, dropped = [], []
for r in rows:
    q = ray(r)
    (data if q is not None else dropped).append(q if q is not None else (r["Event Name"], r["Station"], r["Phase"]))
n = len(data)
path_check = max(abs(q["t_path"] - q["T"]) for q in data)
print(f"{len(rows)} picks; {n} with a TauP arrival in {MODEL}, {len(dropped)} without; {sum(len(q['w']) for q in data)} "
      f"segments; |T(path) - T(TauP)| max {path_check:.1e} s", flush=True)
lap("rays: TauP + polylines")


def cmb_terms(q):
    """c h(x) terms at CMB vertices of a ray: c = sum over the leg ends of (+eta below, -eta above)."""
    if "cmb" not in planet.interface_names:
        return []
    X, out = planet.cut(q["X"])[0], []
    for c in planet.crossings(q["X"]):
        if c.interface != "cmb":
            continue
        rk, total = R * np.linalg.norm(c.point), 0.0
        for side, j in zip(c.sides, (c.vertex - 1, c.vertex)):
            if side is not None:
                v = float(moon.get_property_at_radius(q["wave"], R * np.linalg.norm(0.5 * (X[j] + X[j + 1]))))
                eta = np.sqrt(max(1 / v ** 2 - (q["p"] / rk) ** 2, 0.0))
                total += eta if side == "below" else -eta
        out.append((c.point, total))
    return out


# ---------------------------------------------------------------- data, noise, nuisance
d_obs = np.array([q["obs"] - q["T"] for q in data])                 # observed minus reference travel time (s)
sigma = np.array([max(q["std"], CLASS_STD[EVENTS[q["event"]]["kind"]][q["phase"] == "S"]) for q in data])
natural = sorted({q["event"] for q in data if EVENTS[q["event"]]["kind"] != "artificial"})
demeaned = d_obs.copy()                                             # minus the sigma-weighted mean of the event's picks
for e in natural:
    s = np.array([q["event"] == e for q in data])
    demeaned[s] -= np.sum(d_obs[s] / sigma[s] ** 2) / np.sum(1 / sigma[s] ** 2)
with open(f"{OUT}/residuals.csv", "w") as fh:
    fh.write("event,kind,station,phase,taup_phase,observed_s,reference_s,residual_s,demeaned_s,sigma_s,catalogue_std_s\n")
    for q, r0, r1, s in zip(data, d_obs, demeaned, sigma):
        fh.write(f"{q['event']},{EVENTS[q['event']]['kind']},{STATIONS[q['station']]},{q['phase']},{q['name']},"
                 f"{q['obs']:.3f},{q['T']:.3f},{r0:.3f},{r1:.3f},{s:.3f},{q['std']:.3f}\n")
if os.environ.get("STOP_AFTER_RESIDUALS"):
    raise SystemExit(0)
D = pg.Euclidean(n, covariance=sigma ** 2)
E = pg.Span([D(np.array([q["event"] == e for q in data], float)) for e in natural])   # free origin times
B = np.linalg.qr(np.c_[np.ones(4), np.eye(4)[:, :3]])[0][:, 1:]                    # zero-sum station contrasts (4 x 3)


def axes(e):
    """(east, north, up) at the source of event e, with the prior std of the shift along each (km; 0: fixed)."""
    up = EVENTS[e]["pos"]
    east = np.cross([0.0, 0.0, 1.0], up)
    east = east / np.linalg.norm(east) if np.linalg.norm(east) > 1e-12 else np.array([1.0, 0.0, 0.0])
    h, z = SIG_LOC[EVENTS[e]["kind"]]
    return [(a, sg) for a, sg in ((east, h), (np.cross(up, east), h), (up, z)) if sg > 0]


SLOTS, n_loc = {}, 0                                                # the summand coordinates of each event's shift
for e in (natural if LOCATIONS else []):
    SLOTS[e] = n_loc
    n_loc += len(axes(e))

# ---------------------------------------------------------------- model space and the data operator
ELL = ELL_KM / R


def scale(H, x0, s):
    return H.riesz(np.asarray(x0, float)).norm() ** 2 / s ** 2


spaces = {}
for wave in ("vp", "vs"):
    V = pg.PiecewiseSpace.sobolev(planet, 2, length_scale=ELL, pieces=SOLID)
    mids = {k: 0.5 * (planet.radii[planet.index(k) - 1] + planet.radii[planet.index(k)]) for k in SOLID}
    spaces[wave] = pg.PiecewiseSpace.sobolev(planet, 2, length_scale=ELL, pieces=SOLID,
                                      weights={k: scale(V.summand(k), [mids[k], 0, 0], SIG[wave]) for k in SOLID})
weights = {"stations": 0.75 / SIG["station"] ** 2}                   # rows of B have squared norm 3/4
if Z_CMB is not None:
    spaces["cmb"] = pg.Sobolev(planet.interface("cmb"), 2, length_scale=ELL)
    weights["cmb"] = scale(spaces["cmb"], [1 - Z_CMB / R, 0, 0], SIG["cmb"])
spaces["stations"] = pg.Euclidean(6)
if n_loc:
    spaces["sources"] = pg.Euclidean(n_loc)                         # shifts in units of their prior std (weight 1)
X = pg.DirectSum(spaces, weights=weights)
functionals, n_cmb = [], 0
for q in data:
    a = np.zeros(6)
    a[(0 if q["wave"] == "vp" else 3):][:3] = B[q["station"]]
    f = {q["wave"]: -R * pg.LebesgueMeasure(pg.Polyline(q["X"])).with_density(q["w"]), "stations": a}
    if q["event"] in SLOTS:                                         # dT = -p_src . dx (moving the source along the ray)
        b = np.zeros(n_loc)
        for j, (ax, sg) in enumerate(axes(q["event"])):
            b[SLOTS[q["event"]] + j] = -sg * float(q["p_src"] @ ax)
        f["sources"] = b
    terms = cmb_terms(q)
    n_cmb += len(terms)
    if terms:
        f["cmb"] = sum((c * pg.DiracMeasure(x) for x, c in terms[1:]), terms[0][1] * pg.DiracMeasure(terms[0][0]))
    functionals.append(f)
G = pg.Functionals(X.riesz(functionals), into=D)
lap("Riesz representers")
G.gram()
lap("Gram matrix (exact energies)")
L = pg.Additive(G, D.standard_gaussian(), nuisance=E)                         # free origin times: the kernel on D / E
Gq, _ = L.effective
d = D(d_obs)
dq = L.datum(d)
alpha = pg.meta.discrepancy(pg.Problem(mdr=Gq, datum=dq)).gamma
lap("discrepancy principle on the quotient")
post = pg.update(pg.probability.cylindrical(X, alpha=alpha), L, d)
m = post.mean
dof = dq.space.dim
chi2 = float(dq.space.norm(dq - Gq(m)) ** 2)
chi2_prior = float(dq.space.norm(dq) ** 2)
chi2_stated = float(dq.space.norm(dq - Gq(pg.update(pg.probability.cylindrical(X, alpha=1.0), L, d).mean)) ** 2)
print(f"n = {n}, free origin times {len(natural)}, quotient dimension {dof}; alpha = {alpha:.3e}; chi^2/dof: "
      f"{chi2_prior / dof:.2f} (reference model) -> {chi2 / dof:.3f}; at the stated prior (alpha = 1) "
      f"{chi2_stated / dof:.3f}; CMB terms {n_cmb}", flush=True)
lap("posterior (conditioned on the quotient)")


# ---------------------------------------------------------------- maps, sections, station terms
def depth_map(depth, step):
    lon, lat = np.meshgrid(np.arange(-180, 180 + step, step), np.arange(-90, 90 + step, step))
    return (1 - depth / R) * xyz(lat, lon).reshape(-1, 3), lon.shape


MAPS = {}
for depth in (100.0, 600.0):
    pts, shape = depth_map(depth, 4.0)
    spts, sshape = depth_map(depth, 12.0)
    for wave in ("vp", "vs"):
        MAPS[(wave, depth)] = dict(mean=spaces[wave].evaluate(m[wave], pts).reshape(shape),
                                   std=post.std({wave: spts}).reshape(sshape))
lap("depth maps (mean and std)")
st_mean = np.r_[B @ m["stations"].values[:3], B @ m["stations"].values[3:]]
st_std = post.std([{"stations": np.r_[B[s], np.zeros(3)]} for s in range(4)] +
                  [{"stations": np.r_[np.zeros(3), B[s]]} for s in range(4)])
cmb_std = post.std({"cmb": (1 - Z_CMB / R) * xyz(0.0, 0.0)}) if Z_CMB is not None else None
shifts = {}
if n_loc:
    c = m["sources"].values
    for e in SLOTS:
        ax = axes(e)
        shifts[e] = float(np.linalg.norm([c[SLOTS[e] + j] * sg for j, (_, sg) in enumerate(ax)]))
cmb_prior = pg.probability.cylindrical(X, alpha=alpha).std({"cmb": (1 - Z_CMB / R) * xyz(0.0, 0.0)}) if Z_CMB is not None else None

fig, axs = plt.subplots(2, 4, figsize=(20, 7.5))
for col, (wave, depth) in enumerate([(w, z) for z in (100.0, 600.0) for w in ("vp", "vs")]):
    M = MAPS[(wave, depth)]
    lim = float(np.abs(M["mean"]).max()) or 1e-3
    for row, key, cm in ((0, "mean", "RdBu"), (1, "std", "viridis")):
        im = axs[row, col].imshow(M[key], origin="lower", extent=(-180, 180, -90, 90), cmap=cm,
                                  vmin=-lim if key == "mean" else None, vmax=lim if key == "mean" else None)
        axs[row, col].set_title(f"d{wave} at {depth:g} km: posterior {key} (km/s)", fontsize=9)
        plt.colorbar(im, ax=axs[row, col], shrink=0.7)
    S = np.array(list(lander[s] for s in STATIONS))
    axs[0, col].plot(np.degrees(np.arctan2(S[:, 1], S[:, 0])), np.degrees(np.arcsin(S[:, 2])), "^", color="k", ms=6)
fig.suptitle(f"{MODEL} + dv: n = {n} picks, alpha = {alpha:.2e}, chi^2/dof = {chi2 / dof:.2f} "
             f"(prior std {SIG['vp'] / np.sqrt(alpha):.3f} / {SIG['vs'] / np.sqrt(alpha):.3f} km/s)")
fig.savefig(f"{OUT}/maps.png", dpi=130, bbox_inches="tight")

fig, axs = plt.subplots(1, 3, figsize=(17, 5))
colors = {"artificial": "#1b9e77", "meteoroid": "#666666", "shallow": "#d95f02", "deep": "#7570b3"}
for k, c in colors.items():
    P = np.array([e["pos"] for name, e in EVENTS.items() if e["kind"] == k and group(name) in GROUPS])
    axs[0].plot(np.degrees(np.arctan2(P[:, 1], P[:, 0])), np.degrees(np.arcsin(P[:, 2])), "o", ms=3, color=c, label=k)
axs[0].plot(np.degrees(np.arctan2(S[:, 1], S[:, 0])), np.degrees(np.arcsin(S[:, 2])), "^", ms=8, color="k", label="ALSEP")
axs[0].set(xlim=(-180, 180), ylim=(-90, 90), title="events and stations")
axs[0].legend(fontsize=7)
resid0 = (dq.values) / 1.0
resid1 = (dq - Gq(m)).values
axs[1].hist([resid0, resid1], bins=40, label=["reference model", "posterior mean"], color=["0.6", "#1f78b4"])
axs[1].set(title="whitened residuals in the quotient (origin times removed)", xlabel="residual / sigma")
axs[1].legend()
labels = [f"{s[-2:]} {ph}" for ph in "PS" for s in STATIONS]
axs[2].errorbar(np.arange(8), st_mean, yerr=st_std, fmt="o")
axs[2].axhline(0, color="0.5", lw=0.6)
axs[2].set(xticks=np.arange(8), xticklabels=labels, ylabel="s", title="station terms (zero-mean contrasts), mean +- std")
fig.savefig(f"{OUT}/data_and_terms.png", dpi=130, bbox_inches="tight")
lap("plots")

kinds = {k: sum(e["kind"] == k and group(name) in GROUPS for name, e in EVENTS.items()) for k in CLASS_STD}
used = {k: sum(EVENTS[q["event"]]["kind"] == k for q in data) for k in CLASS_STD}
costs.update(model=MODEL, event_groups=GROUPS, discontinuities_km=JUMPS, layers=list(planet.names), picks=len(rows), used=n,
             dropped_no_arrival=len(dropped), events_by_kind=kinds, picks_by_kind=used,
             phases={ph: sum(q["phase"] == ph for q in data) for ph in "PS"}, free_origin_times=len(natural),
             quotient_dim=dof, segments=int(sum(len(q["w"]) for q in data)), cmb_terms=n_cmb, alpha=float(alpha),
             chi2_per_dof=chi2 / dof, chi2_per_dof_reference=chi2_prior / dof, chi2_per_dof_at_alpha_1=chi2_stated / dof,
             station_terms_s=dict(zip(labels, np.round(st_mean, 3).tolist())),
             station_terms_std_s=dict(zip(labels, np.round(st_std, 3).tolist())),
             locations=LOCATIONS, location_prior_km=SIG_LOC, location_parameters=n_loc,
             source_shift_km={k: dict(median=float(np.median([v for e, v in shifts.items() if EVENTS[e]["kind"] == k])),
                                      max=float(max(v for e, v in shifts.items() if EVENTS[e]["kind"] == k)))
                              for k in SIG_LOC if any(EVENTS[e]["kind"] == k for e in shifts)},
             cmb_std_km=cmb_std, cmb_prior_std_km=cmb_prior, path_time_check_s=path_check, eps_km=EPS_KM,
             sigma_s=dict(min=float(sigma.min()), median=float(np.median(sigma)), max=float(sigma.max())),
             max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20))
json.dump(costs, open(f"{OUT}/costs.json", "w"), indent=1, default=float)
print(json.dumps(costs, indent=1, default=float))
