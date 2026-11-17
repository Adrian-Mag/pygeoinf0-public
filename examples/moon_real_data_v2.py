"""The second real-data Moon inversion (plan 007 P5, the author's decisions of 2026-10-27 on RESEARCH-P5): Apollo P and
S picks, SensRay and pygeoinf0, with per-class noise, per-event location priors, LRO impact sites and the Weber picks.

What changed against moon_real_data.py (run 1, kept unchanged so that it stays reproducible):
    impacts    artificial-impact coordinates and times of the LRO crater identifications (ImpactParameters.csv of the
               Nunn et al. 2020 supplement; Plescia et al. 2016, Stooke 2017); 16 S-IVB (31 km from the Apollo-era
               site, its time poorly known before LRO, Nunn et al. 2020 p. 20) gets a free origin time
    locations  natural events keep Lognonne et al. 2003's set (that of the picks); the prior std of each event's shift
               is the spread of its published locations (data/location_priors.csv, made by
               experiments/EXP-029-moon-real-data/prepare_location_priors.py, whose docstring states the rule); events
               without three determinations take the class value (deep 34 km per axis, meteoroid impacts 35 km
               horizontal, shallow 50 km horizontal and 50 km depth)
    noise      sigma0 = max(Lognonne 2003 per-pick bound 1/3/10 s, spread across studies) where Lognonne 2003 has the
               pick (PDS lognonne_2003_catalog.csv), else max(spread across studies, class value of Nunn et al. 2020
               p. 26) as in run 1; then one factor per class (event class x phase; a class of fewer than MIN_CLASS
               picks shares its event class's factor) from the residuals of the posterior at the stated prior
               (alpha = 1): f_c^2 <- f_c^2 sum_c (r_i / sigma_i)^2 / sum_c rho_i, rho_i = sigma_i^2 P_ii the redundancy
               number (the expected value of (r_i / sigma_i)^2), iterated to convergence (Helmert / Foerstner variance
               components), with f_c >= FACTOR_FLOOR = 1: inflation only (decision 4; the sources say the stated errors
               are too small, and free factors below 1 put single picks at 0.25 s); the free estimates are reported;
               then a gross-error screen: the pick with the largest standardised residual
               |r_i| / (sigma_i sqrt(rho_i)) > SCREEN is removed and the factors re-estimated, until none is left;
               the screen rescales each class robustly at the current solution (robust_scale: the median instead of
               the mean of (r_i / sigma_i)^2 / rho_i, not below the floor), since with the classical factors a few gross errors inflate
               their class's factor and hide (meteoroid S: 3 picks 107-172 s late gave 2.5), and is then confirmed
               with the final factors alone; picks with rho_i < RHO_MIN are not tested (a gross error there
               is not detectable: the smallest detectable one grows as 1 / sqrt(rho_i), Baarda 1968)
    alpha      the main posterior is at the stated prior (alpha = 1); the discrepancy-principle alpha and chi^2 for a
               few alphas are reported as sensitivities
    Weber      the PDS table of Weber et al. 2011 S picks (relative times, one arbitrary origin per cluster; A10 = A24):
               VARIANT=nunn (main) predicts the intra-cluster S differences from the Nunn-only posterior, including the
               posterior location terms of the same nest, and compares (noise WEBER_SIGMA = 3.7 s; the 0.1/0.5 weights
               of the masters file are not used); VARIANT=merged adds them as data (own free time per cluster, the
               nest's location terms shared, its own noise class with factor >= 1)
    CMB        the CMB summand is kept and is prior-only: no P or S ray of these data touches the CMB in M1
    REFERENCE_TEST=M2   no inversion: reference residuals by class at M1 and at the model named (a TauP pass)

Run on europa (experiments/EXP-029-moon-real-data/run_europa.sh): python moon_real_data_v2.py -> $MOON_OUT/.
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
from scipy.stats import chi2 as chi2_law
from sensray import PlanetModel

import pygeoinf0 as pg

MODEL = os.environ.get("MOON_MODEL", "M1")
CATS = os.environ.get("CATALOGUE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments",
                                                   "EXP-029-moon-real-data", "data"))
DATA = os.environ.get("MOON_DATA", CATS)            # EXP-029's data/ (PDS, Zenodo files and location_priors.csv)
OUT = os.path.abspath(os.environ.get("MOON_OUT", "moon_real_v2"))
VARIANT = os.environ.get("VARIANT", "nunn")          # nunn (main, with the Weber check) or merged (Weber picks as data)
REFERENCE_TEST = os.environ.get("REFERENCE_TEST", "")
FREE_T0 = [e for e in os.environ.get("FREE_T0", "16 S-IVB").split(",") if e]   # artificial impacts with a free time
GROUPS = ["artificial", "M", "SH", "A"]               # W and AW stay out (decision 1)
EPS_KM = 0.5
STATIONS = ["Apollo 12", "Apollo 14", "Apollo 15", "Apollo 16"]
CLASS_STD = {"artificial": (1.3, 3.7), "meteoroid": (4.0, 18.2), "shallow": (3.0, 13.2), "deep": (10.1, 3.7)}
LG03_BOUND = {"0": 1.0, "1": 3.0, "2": 10.0}         # Lognonne 2003 error codes: error smaller than 1, 3, 10 s
WEBER_SIGMA = 3.7                                     # s: the stacked deep S spread (Nunn et al. 2020 p. 26)
MIN_CLASS = 10
FACTOR_FLOOR = float(os.environ.get("FACTOR_FLOOR", 1.0))   # 1: inflate only (decision 4); 0: free estimates
SCREEN = 4.0
RHO_MIN = 0.1                                         # picks with smaller redundancy are not tested (uncontrolled)
SIG = {"vp": 0.2, "vs": 0.12, "cmb": 10.0, "station": 2.3}     # prior pointwise std at alpha = 1 (km/s, km, s)
ELL_KM = 300.0
os.makedirs(OUT, exist_ok=True)
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
    if "S-IVB" in name or "LM" in name:
        return "artificial"
    for g in ("AW", "SH", "A", "M", "W"):
        if name.startswith(g):
            return g
    return "other"


def kind(name):
    if "S-IVB" in name or "LM" in name:
        return "artificial"
    return "shallow" if name.startswith("SH") else "deep" if name.startswith("A") else "meteoroid"


def angle_km(a, b, r):
    return float(np.arccos(np.clip(a @ b, -1, 1)) * r)


# ---------------------------------------------------------------- catalogues
lander = {r["Lander"]: xyz(float(r["Latitude"]), float(r["Longitude"]))
          for r in csv.DictReader(open(f"{CATS}/LanderLocs.csv", encoding="utf-8-sig")) if r["Element"] == "ALSEP"}
rows = [r for r in csv.DictReader(open(f"{CATS}/Nunn_catalog_picks.csv", encoding="utf-8-sig"))
        if group(r["Event Name"]) in GROUPS]
EVENTS = {}
for r in rows:
    EVENTS.setdefault(r["Event Name"], dict(lat=float(r["Lat"]), lon=float(r["Lon"]), depth=float(r["Depth"]),
                                            t0=float(r["t_0"]), kind=kind(r["Event Name"])))
for e in EVENTS.values():
    e["pos"] = xyz(e["lat"], e["lon"])

# LRO impact sites and times (decision 3): "A13/S-IVB" -> "13 S-IVB"
impacts = {}
for r in csv.DictReader(open(f"{DATA}/ImpactParameters.csv", encoding="utf-8-sig")):
    name = r["Impact"].strip()
    if "/" in name and r["Latitude"].strip() not in ("", "-"):
        a, b = name.split("/")
        impacts[f"{a[1:]} {b}"] = (float(r["Latitude"]), float(r["Longitude"]), float(r["UST (time recorded on Earth)"][17:]))
impact_moves = {}
for name, e in EVENTS.items():
    if e["kind"] == "artificial":
        lat, lon, sec = impacts[name]
        new = xyz(lat, lon)
        impact_moves[name] = dict(shift_km=round(angle_km(e["pos"], new, 1737.1), 2), dt0_s=round(sec - e["t0"], 2))
        e.update(lat=lat, lon=lon, pos=new, t0=sec)

# per-event location priors (decision 3; rule in prepare_location_priors.py)
LOC_PRIOR = {r["event"]: (float(r["sigma_h_km"]), float(r["sigma_z_km"]), r["rule_h"], r["rule_z"])
             for r in csv.DictReader(open(f"{DATA}/location_priors.csv"))}

# Lognonne et al. 2003 per-pick error codes (PDS); rows matched to events by name (impacts) or coordinates
LG03 = {}
for r in csv.DictReader(open(f"{DATA}/lognonne_2003_catalog.csv", encoding="utf-8-sig")):
    if r["Type"] in EVENTS:
        names = [r["Type"]]
    else:
        names = [n for n, e in EVENTS.items() if e["kind"] != "artificial" and n.startswith(r["Type"][:1])
                 and abs(e["lat"] - float(r["Lat"])) < 0.05 and abs(e["lon"] - float(r["Long"])) < 0.05]
    for n in names:
        for s in STATIONS:
            for ph in "PS":
                t, c = r[f"{s[-2:]}{ph}"].strip(), r[f"{s[-2:]}{ph}_EC"].strip()
                if t and c in LG03_BOUND:
                    LG03[(n, s, ph)] = (LG03_BOUND[c], float(t) - float(r["Seconds"]))

# Weber et al. 2011 S picks (PDS), clusters of our nests only (A10 = A24), at least two picks
WEBER = []
for r in csv.DictReader(open(f"{DATA}/weber_2011_dmq_s_picks.csv", encoding="utf-8-sig")):
    nest = f"A{int(r['A#'][1:]):02d}"
    nest = "A24" if nest == "A10" else nest
    picks = [(s, float(r[f"S{s[-2:]}"])) for s in STATIONS if r[f"S{s[-2:]}"].strip()]
    if nest in EVENTS and len(picks) >= 2:
        WEBER += [dict(cluster=r["A#"], nest=nest, station=s, time=t) for s, t in picks]

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
names = [f"{z:g}-{b:g} km" for z, b in zip(tops, JUMPS + [R])][::-1]
planet = pg.LayeredBall([1 - z / R for z in JUMPS[::-1]] + [1.0], names=names,
                        interfaces=["cmb" if z == Z_CMB else f"{z:g} km" for z in JUMPS[::-1]])
SOLID = [f"{z:g}-{b:g} km" for z, b in zip(tops, JUMPS + [R]) if Z_CMB is None or z < Z_CMB][::-1]
print(f"{MODEL}: R = {R} km; discontinuities {JUMPS} km; layers {planet.names}", flush=True)


# ---------------------------------------------------------------- picks
def sigma0(name, station, phase, spread):
    """(sigma0, rule): the Lognonne 2003 bound where it has the pick, else the class value (both at least the spread)."""
    if (name, station, phase) in LG03:
        return max(LG03[(name, station, phase)][0], spread), "lg03"
    return max(spread, CLASS_STD[EVENTS[name]["kind"]][phase == "S"]), "class"


PICKS = []
for r in rows:
    n, s, ph = r["Event Name"], r["Station"], r["Phase"]
    s0, rule = sigma0(n, s, ph, float(r["std dev arrival time"]))
    lg = LG03.get((n, s, ph))
    PICKS.append(dict(source="nunn", event=n, station=s, phase=ph, obs=float(r["median arrival time"]) -
                      EVENTS[n]["t0"], spread=float(r["std dev arrival time"]), sigma0=s0, rule=rule,
                      lg03_minus_nunn=(lg[1] - (float(r["median arrival time"]) - float(r["t_0"]))) if lg else np.nan,
                      cls=f"{EVENTS[n]['kind']} {ph}", nuis=n if (EVENTS[n]["kind"] != "artificial" or n in FREE_T0)
                      else None))
for w in WEBER:
    PICKS.append(dict(source="weber", event=w["nest"], station=w["station"], phase="S", obs=w["time"], spread=0.0,
                      sigma0=WEBER_SIGMA, rule="weber", lg03_minus_nunn=np.nan, cls="weber S",
                      nuis=f"weber {w['cluster']}"))
count = {}
for p in PICKS:
    count[p["cls"]] = count.get(p["cls"], 0) + 1
pooled = {EVENTS[p["event"]]["kind"] for p in PICKS if p["source"] == "nunn" and count[p["cls"]] < MIN_CLASS}
for p in PICKS:                                       # small classes share their event class's factor
    if p["source"] == "nunn" and EVENTS[p["event"]]["kind"] in pooled:
        p["cls"] = f"{EVENTS[p['event']]['kind']} P+S"
print(f"{sum(p['source'] == 'nunn' for p in PICKS)} Nunn picks ({sum(p['rule'] == 'lg03' for p in PICKS)} with a "
      f"Lognonne 2003 code), {len(WEBER)} Weber picks in {len({w['cluster'] for w in WEBER})} clusters of our nests; "
      f"impacts moved to LRO sites: {impact_moves}", flush=True)


# ---------------------------------------------------------------- rays
def simplify(X, forced, eps):
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


def first_time(model, ev, station, phase, extended):
    """The first-arrival time of the phase in a TauP model (with Pn, Pg / Sn, Sg if extended), or nan."""
    D = np.degrees(np.arccos(np.clip(ev["pos"] @ lander[station], -1, 1)))
    ph = [phase, phase.lower()] + ([phase + "n", phase + "g"] if extended else [])
    arrs = [a for a in model.get_travel_times(ev["depth"], D, ph) if a.name in ph]
    return min((a.time for a in arrs), default=np.nan), D


def ray(p):
    """The first P or S arrival of the pick in the reference model as a weighted polyline (units of R), or None."""
    ev, wave = EVENTS[p["event"]], "vp" if p["phase"] == "P" else "vs"
    src, rec = ev["pos"], lander[p["station"]]
    D = np.arccos(np.clip(src @ rec, -1, 1))
    arrs = [a for a in taup.get_ray_paths(ev["depth"], np.degrees(D), [p["phase"], p["phase"].lower()])
            if a.name in (p["phase"], p["phase"].lower())]
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
    p_src = (X[1] - X[0]) / (np.linalg.norm(X[1] - X[0]) * v[0])
    forced = np.unique(np.r_[0, np.flatnonzero(np.min(np.abs(z[:, None] - np.array(JUMPS + [-1.0])), axis=1) < 1e-6),
                             len(X) - 1])
    keep = simplify(X, forced, EPS_KM / R)
    w = np.add.reduceat(ds / v ** 2, keep[:-1]) / (np.linalg.norm(np.diff(X[keep], axis=0), axis=1) * R)
    return dict(X=X[keep], w=w, p=arr.ray_param, T=arr.time, p_src=p_src, t_path=float(np.sum(ds / v)), wave=wave,
                station_index=STATIONS.index(p["station"]), name=arr.name, delta=float(np.degrees(D)))


if REFERENCE_TEST:
    # decision 5: reference residuals by class at M1 and at REFERENCE_TEST, first arrivals (P, p, Pn, Pg; S likewise),
    # natural events and free-time impacts demeaned per event (sigma0-weighted), other impacts at their LRO times
    other = PlanetModel.from_standard_model(REFERENCE_TEST).taupy_model
    out = []
    for p in (q for q in PICKS if q["source"] == "nunn"):
        ev = EVENTS[p["event"]]
        t1, D = first_time(taup, ev, p["station"], p["phase"], True)
        t2, _ = first_time(other, ev, p["station"], p["phase"], True)
        out.append(dict(p, delta=D, r1=p["obs"] - t1, r2=p["obs"] - t2))
    for key in ("r1", "r2"):
        for e in {q["nuis"] for q in out if q["nuis"]}:
            s = [q for q in out if q["nuis"] == e and np.isfinite(q[key])]
            if s:
                mean = sum(q[key] / q["sigma0"] ** 2 for q in s) / sum(1 / q["sigma0"] ** 2 for q in s)
                for q in s:
                    q[key] -= mean
    table = []
    sets = [(c, lambda q, c=c: f"{EVENTS[q['event']]['kind']} {q['phase']}" == c)
            for c in sorted({f"{EVENTS[q['event']]['kind']} {q['phase']}" for q in out})]
    sets.append(("artificial P, delta < 10 deg", lambda q: EVENTS[q["event"]]["kind"] == "artificial"
                 and q["phase"] == "P" and q["delta"] < 10))
    for c, sel in sets:
        s = [q for q in out if sel(q) and np.isfinite(q["r1"]) and np.isfinite(q["r2"])]
        row = dict(cls=c, n=len(s))
        for key, mname in (("r1", MODEL), ("r2", REFERENCE_TEST)):
            r = np.array([q[key] for q in s])
            z = r / np.array([q["sigma0"] for q in s])
            row.update({f"{mname} mean_s": float(r.mean()), f"{mname} rms_s": float(np.sqrt(np.mean(r ** 2))),
                        f"{mname} mean_z2": float(np.mean(z ** 2))})
        table.append(row)
        print(row, flush=True)
    with open(f"{OUT}/reference_test.csv", "w") as fh:
        fh.write(f"event,kind,station,phase,delta_deg,observed_s,sigma0_s,residual_{MODEL}_s,residual_{REFERENCE_TEST}_s\n")
        for q in out:
            fh.write(f"{q['event']},{EVENTS[q['event']]['kind']},{q['station']},{q['phase']},{q['delta']:.3f},"
                     f"{q['obs']:.3f},{q['sigma0']:.3f},{q['r1']:.3f},{q['r2']:.3f}\n")
    json.dump(table, open(f"{OUT}/reference_test.json", "w"), indent=1)
    raise SystemExit(0)

data, dropped = [], []
for p in PICKS:
    q = ray(p)
    if q is None:
        dropped.append((p["event"], p["station"], p["phase"], p["source"]))
    else:
        data.append(dict(p, **q))
n_all = len(data)
path_check = max(abs(q["t_path"] - q["T"]) for q in data)
print(f"{n_all} picks with a TauP arrival, {len(dropped)} without {dropped}; {sum(len(q['w']) for q in data)} segments; "
      f"|T(path) - T(TauP)| max {path_check:.1e} s", flush=True)
lap("rays: TauP + polylines")


def cmb_terms(q):
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


# ---------------------------------------------------------------- unknowns and functionals (all picks)
natural = sorted({e for e, v in EVENTS.items() if v["kind"] != "artificial"})
B = np.linalg.qr(np.c_[np.ones(4), np.eye(4)[:, :3]])[0][:, 1:]                    # zero-sum station contrasts (4 x 3)


def axes(e):
    """(east, north, up) at the source of event e with the prior std of the shift along each (km; 0: fixed)."""
    up = EVENTS[e]["pos"]
    east = np.cross([0.0, 0.0, 1.0], up)
    east = east / np.linalg.norm(east) if np.linalg.norm(east) > 1e-12 else np.array([1.0, 0.0, 0.0])
    h, z = LOC_PRIOR[e][:2]
    return [(a, sg) for a, sg in ((east, h), (np.cross(up, east), h), (up, z)) if sg > 0]


SLOTS, n_loc = {}, 0
for e in natural:
    SLOTS[e] = n_loc
    n_loc += len(axes(e))

ELL = ELL_KM / R


def scale(H, x0, s):
    return H.riesz(np.asarray(x0, float)).norm() ** 2 / s ** 2


spaces = {}
for wave in ("vp", "vs"):
    V = pg.PiecewiseSpace.sobolev(planet, 2, length_scale=ELL, pieces=SOLID)
    mids = {k: 0.5 * (planet.radii[planet.index(k) - 1] + planet.radii[planet.index(k)]) for k in SOLID}
    spaces[wave] = pg.PiecewiseSpace.sobolev(planet, 2, length_scale=ELL, pieces=SOLID,
                                      weights={k: scale(V.summand(k), [mids[k], 0, 0], SIG[wave]) for k in SOLID})
weights = {"stations": 0.75 / SIG["station"] ** 2}
if Z_CMB is not None:
    spaces["cmb"] = pg.Sobolev(planet.interface("cmb"), 2, length_scale=ELL)
    weights["cmb"] = scale(spaces["cmb"], [1 - Z_CMB / R, 0, 0], SIG["cmb"])
spaces["stations"] = pg.Euclidean(6)
spaces["sources"] = pg.Euclidean(n_loc)                             # shifts in units of their prior std (weight 1)
X = pg.DirectSum(spaces, weights=weights)
functionals, n_cmb = [], 0
for q in data:
    a = np.zeros(6)
    a[(0 if q["wave"] == "vp" else 3):][:3] = B[q["station_index"]]
    f = {q["wave"]: -R * pg.LebesgueMeasure(pg.Polyline(q["X"])).with_density(q["w"]), "stations": a}
    if q["event"] in SLOTS:                                         # dT = -p_src . dx (Weber picks: the nest's terms)
        b = np.zeros(n_loc)
        for j, (ax, sg) in enumerate(axes(q["event"])):
            b[SLOTS[q["event"]] + j] = -sg * float(q["p_src"] @ ax)
        f["sources"] = b
    terms = cmb_terms(q)
    n_cmb += len(terms)
    if terms:
        f["cmb"] = sum((c * pg.DiracMeasure(x) for x, c in terms[1:]), terms[0][1] * pg.DiracMeasure(terms[0][0]))
    functionals.append(f)
G_all = pg.Functionals(X.riesz(functionals), into=pg.Euclidean(n_all))
K = np.asarray(G_all.gram())                                        # prior covariance of the data functionals (alpha=1)
lap("Gram matrix of all picks (exact energies)")

y_all = np.array([q["obs"] - q["T"] for q in data])                # observed minus reference travel time (s)
sig0 = np.array([q["sigma0"] for q in data])
CLASSES = sorted({q["cls"] for q in data})
cls = np.array([CLASSES.index(q["cls"]) for q in data])
NUIS = sorted({q["nuis"] for q in data if q["nuis"]})
Emat = np.array([[q["nuis"] == k for k in NUIS] for q in data], float)
nunn = np.array([q["source"] == "nunn" for q in data])


def gls(idx, sig, alpha=1.0):
    """The posterior at prior alpha with free nuisance times, in data coordinates: residual r = Sigma P y, redundancy
    numbers rho = sigma^2 diag P, chi^2 = r' Sigma^-1 r (the quotient's), P = W - W E (E' W E)^-1 E' W,
    W = (K / alpha + Sigma)^-1."""
    E = Emat[idx][:, Emat[idx].any(axis=0)]
    W = np.linalg.inv(K[np.ix_(idx, idx)] / alpha + np.diag(sig ** 2))
    WE = W @ E
    P = W - WE @ np.linalg.solve(E.T @ WE, WE.T)
    P = 0.5 * (P + P.T)
    r = sig ** 2 * (P @ y_all[idx])
    return dict(P=P, r=r, rho=sig ** 2 * np.diag(P), chi2=float(np.sum(r ** 2 / sig ** 2)), dof=len(idx) - E.shape[1])


def factors(idx, floor=None, tol=1e-5):
    """Per-class noise factors at the stated prior (module docstring), iterated to convergence; at least floor
    (default FACTOR_FLOOR; the Weber class at least 1 in any case)."""
    floor = FACTOR_FLOOR if floor is None else floor
    f = np.ones(len(CLASSES))
    for it in range(500):
        sig = sig0[idx] * f[cls[idx]]
        g = gls(idx, sig)
        new = f.copy()
        for c in set(cls[idx]):
            m = cls[idx] == c
            new[c] = f[c] * np.sqrt(np.sum((g["r"][m] / sig[m]) ** 2) / max(np.sum(g["rho"][m]), 1e-12))
            new[c] = max(new[c], 1.0 if CLASSES[c] == "weber S" else floor)
        done = np.max(np.abs(new / f - 1)) < tol
        f = new
        if done:
            break
    sig = sig0[idx] * f[cls[idx]]
    return f, sig, gls(idx, sig), it + 1


def robust_scale(idx, sig, g, f):
    """Per pick, its class's robust rescaling at the current solution: sqrt(median of r_i^2 / (sigma_i^2 rho_i) over
    the class's tested picks / 0.45494, the median of chi^2_1), so that the rescaled factor is at least the floor
    (the screen tests against the noise model in use, not below it). One step, not iterated: iterating a median-based
    variance component can collapse to zero for a class of low redundancy (it did for shallow P)."""
    s = np.ones(len(idx))
    for c in set(cls[idx]):
        m = cls[idx] == c
        k = m & (g["rho"] >= RHO_MIN)
        if k.sum() >= 3:
            low = (1.0 if CLASSES[c] == "weber S" else FACTOR_FLOOR) / f[c]
            s[m] = max(np.sqrt(np.median(g["r"][k] ** 2 / (sig[k] ** 2 * g["rho"][k])) / 0.45494), low)
    return s


use = np.flatnonzero(nunn if VARIANT == "nunn" else np.ones(n_all, bool))
f_first, sig_first, g_first, it_first = factors(use)
s_first = dict(zip(cls[use], robust_scale(use, sig_first, g_first, f_first)))
removed = []
for robust in (True, False):          # screen with the robust rescaling, then confirm with the final factors alone
    while True:
        f, sig, g, iters = factors(use)
        s = robust_scale(use, sig, g, f) if robust else np.ones(len(use))
        ok = g["rho"] >= RHO_MIN
        wstd = np.where(ok, g["r"] / (s * sig * np.sqrt(np.where(ok, g["rho"], 1.0))), 0.0)
        j = int(np.argmax(np.abs(wstd)))
        if abs(wstd[j]) <= SCREEN:
            break
        q = data[use[j]]
        removed.append(dict(event=q["event"], station=q["station"], phase=q["phase"], source=q["source"],
                            cluster=q["nuis"] if q["source"] == "weber" else "", residual_s=round(float(g["r"][j]), 2),
                            sigma_s=round(float(sig[j]), 2), robust_sigma_s=round(float(s[j] * sig[j]), 2),
                            redundancy=round(float(g["rho"][j]), 3), standardised=round(float(wstd[j]), 2),
                            step="robust" if robust else "final"))
        print(f"screen: removed {removed[-1]}", flush=True)
        use = np.delete(use, j)
FACTORS = {c: dict(before_screen=round(float(f_first[i]), 3), robust_before_screen=round(float(f_first[i] *
                   s_first[i]), 3), final=round(float(f[i]), 3), n=int(np.sum(cls[use] == i)),
                   redundancy=round(float(np.sum(g["rho"][cls[use] == i])), 1))
           for i, c in enumerate(CLASSES) if np.any(cls[use] == i)}
f_free, sig_free, g_free, _ = factors(use, floor=0.0)            # the unconstrained estimates, for the record


def alpha_for(idx, sig, target):
    """The discrepancy alpha (chi^2 = target) in data coordinates, by bisection in log alpha."""
    lo, hi = -4.0, 4.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if gls(idx, sig, 10 ** mid)["chi2"] > target else (mid, hi)
    return 10 ** lo


for i, c in enumerate(CLASSES):
    if c in FACTORS:
        FACTORS[c].update(unconstrained=round(float(f_free[i]), 3))
FREE = dict(chi2_per_dof_at_alpha_1=g_free["chi2"] / g_free["dof"], alpha_discrepancy=alpha_for(use, sig_free,
            g_free["dof"]), sigma_min_s=float(sig_free.min()))
print(f"noise factors ({it_first} / {iters} iterations; floor {FACTOR_FLOOR}): {FACTORS}; unconstrained: {FREE}",
      flush=True)
print(f"numpy check at alpha = 1: chi^2/dof {g['chi2'] / g['dof']:.3f} (dof {g['dof']}); sum rho {g['rho'].sum():.1f}",
      flush=True)
lap("noise factors and screen")

# ---------------------------------------------------------------- the posterior (pygeoinf0) at the final noise
sel = list(use)
n = len(sel)
D = pg.Euclidean(n, covariance=sig ** 2)
E = pg.Span([D(col) for col in Emat[sel][:, Emat[sel].any(axis=0)].T])
G = pg.Functionals(X.riesz([functionals[i] for i in sel]), into=D)
G.gram()
L = pg.Additive(G, D.standard_gaussian(), nuisance=E)
Gq, _ = L.effective
d = D(y_all[sel])
dq = L.datum(d)
dof = dq.space.dim
post = pg.update(pg.probability.cylindrical(X, alpha=1.0), L, d)
m = post.mean


def chi2_at(alpha):
    mean = m if alpha == 1.0 else pg.update(pg.probability.cylindrical(X, alpha=alpha), L, d).mean
    return float(dq.space.norm(dq - Gq(mean)) ** 2)


chi2 = chi2_at(1.0)
chi2_ref = float(dq.space.norm(dq) ** 2)
try:
    alpha_dp = float(pg.meta.discrepancy(pg.Problem(mdr=Gq, datum=dq)).gamma)
except ValueError:                                                  # the reference model already has chi^2 < dof
    alpha_dp = float("inf")
ladder = {f"{a:.3g}": round(chi2_at(float(a)) / dof, 4) for a in (0.1, 0.3, 1.0, 3.0, 10.0, 100.0, alpha_dp)
          if np.isfinite(a)}
print(f"n = {n} ({sum(data[i]['source'] == 'weber' for i in sel)} Weber), free times {E.dim}, quotient dimension {dof}; "
      f"alpha = 1 (stated prior): chi^2/dof {chi2_ref / dof:.3f} (reference) -> {chi2 / dof:.3f} (numpy "
      f"{g['chi2'] / g['dof']:.3f}); discrepancy alpha {alpha_dp:.3g} (inf: none, the reference model's chi^2 is "
      f"already below the quotient dimension); ladder {ladder}; CMB terms {n_cmb}", flush=True)
lap("posterior (pygeoinf0, quotient)")

# ---------------------------------------------------------------- the Weber check (prediction from the Nunn posterior)
weber = np.flatnonzero(~nunn)
check = {}
if VARIANT == "nunn" and len(weber):
    P = g["P"]
    mu = K[np.ix_(weber, use)] @ (P @ y_all[use])                   # posterior mean of the Weber functionals
    mu_pkg = pg.Functionals(X.riesz([functionals[i] for i in weber]))(m).values
    C = K[np.ix_(weber, weber)] - K[np.ix_(weber, use)] @ P @ K[np.ix_(use, weber)]
    clusters = sorted({data[i]["nuis"] for i in weber})
    def compare(keep, pred, cov):
        """Observed minus predicted, one free time per cluster removed (Qk: centring within the kept picks of each
        cluster); its predictive covariance Qk (cov + sigma_w^2 I) Qk; z standardised per pick; chi^2 on the
        sum (k_c - 1) differences."""
        Qk = np.zeros((len(keep), len(keep)))
        for c in clusters:
            s = np.array([data[weber[i]]["nuis"] == c for i in keep])
            if s.sum() > 1:
                Qk[np.ix_(s, s)] = np.eye(s.sum()) - 1.0 / s.sum()
        r = Qk @ (y_all[weber[keep]] - pred[keep])
        S = Qk @ (cov[np.ix_(keep, keep)] + WEBER_SIGMA ** 2 * np.eye(len(keep))) @ Qk
        k = int(round(np.trace(Qk)))
        z = np.where(np.diag(S) > 1e-9, r / np.sqrt(np.maximum(np.diag(S), 1e-9)), 0.0)
        c2 = float(r @ np.linalg.pinv(S, rcond=1e-10, hermitian=True) @ r)
        return dict(picks=int(np.sum(np.diag(Qk) > 0)), differences=k, rms_s=float(np.sqrt(np.sum(r ** 2) / k)),
                    rms_sigma=float(np.sqrt(np.sum(z ** 2) / np.sum(np.diag(Qk) > 0))), chi2_per_dof=c2 / k,
                    p_value=float(chi2_law.sf(c2, k)), z=z, r=r)

    allk = np.arange(len(weber))
    post_check = compare(allk, mu, C)
    flagged = [int(i) for i in allk if abs(post_check["z"][i]) > SCREEN]
    keep = np.array([i for i in allk if i not in flagged])
    check = dict(posterior=post_check, reference=compare(allk, np.zeros(len(weber)), np.zeros_like(C)),
                 prior=compare(allk, np.zeros(len(weber)), K[np.ix_(weber, weber)]),
                 flagged=[dict(cluster=data[weber[i]]["nuis"], station=data[weber[i]]["station"],
                               z=round(float(post_check["z"][i]), 2)) for i in flagged],
                 posterior_screened=compare(keep, mu, C) if flagged else None,
                 reference_screened=compare(keep, np.zeros(len(weber)), np.zeros_like(C)) if flagged else None,
                 package_minus_numpy_s=float(np.max(np.abs(mu_pkg - mu))))
    for k, v in check.items():
        if isinstance(v, dict):
            print(f"Weber check, {k}: " + ", ".join(f"{a} {b:.3g}" for a, b in v.items() if np.isscalar(b)), flush=True)
    print(f"Weber check: flagged (|z| > {SCREEN}) {check['flagged']}; package - numpy {check['package_minus_numpy_s']:.1e} s",
          flush=True)
    with open(f"{OUT}/weber_check.csv", "w") as fh:
        fh.write("cluster,nest,station,observed_relative_s,reference_s,predicted_dT_s,predicted_std_s,"
                 "difference_residual_s,z\n")
        for j, i in enumerate(weber):
            q = data[i]
            fh.write(f"{q['nuis'][6:]},{q['event']},{q['station']},{q['obs']:.2f},{q['T']:.3f},{mu[j]:.3f},"
                     f"{np.sqrt(max(C[j, j], 0)):.3f},{post_check['r'][j]:.3f},{post_check['z'][j]:.3f}\n")
    lap("Weber check")


# ---------------------------------------------------------------- maps, station terms, locations
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
np.savez_compressed(f"{OUT}/maps.npz", **{f"{w}_{z:g}_{k}": v[k] for (w, z), v in MAPS.items() for k in ("mean", "std")})
lap("depth maps (mean and std)")
st_mean = np.r_[B @ m["stations"].values[:3], B @ m["stations"].values[3:]]
st_std = post.std([{"stations": np.r_[B[s], np.zeros(3)]} for s in range(4)] +
                  [{"stations": np.r_[np.zeros(3), B[s]]} for s in range(4)])
cmb_std = post.std({"cmb": (1 - Z_CMB / R) * xyz(0.0, 0.0)}) if Z_CMB is not None else None
cmb_prior = pg.probability.cylindrical(X, alpha=1.0).std({"cmb": (1 - Z_CMB / R) * xyz(0.0, 0.0)}) if Z_CMB is not None else None
c = m["sources"].values
c_std = post.std([{"sources": np.eye(n_loc)[j]} for j in range(n_loc)])
picks_of = {e: sum(data[i]["event"] == e for i in sel) for e in SLOTS}
LOC = {}
for e in SLOTS:
    ax = axes(e)
    cj = c[SLOTS[e]:SLOTS[e] + len(ax)]
    LOC[e] = dict(kind=EVENTS[e]["kind"], picks=picks_of[e], prior_km=[round(s, 1) for _, s in ax],
                  shift_km=round(float(np.linalg.norm([cj[j] * s for j, (_, s) in enumerate(ax)])), 1),
                  shift_in_prior_std=[round(float(v), 2) for v in cj],
                  post_std_in_prior_std=[round(float(v), 2) for v in c_std[SLOTS[e]:SLOTS[e] + len(ax)]])
beyond = {e: v for e, v in LOC.items() if max(abs(x) for x in v["shift_in_prior_std"]) > 1}
print(f"location shifts beyond one prior std on an axis: {beyond}", flush=True)

# ---------------------------------------------------------------- figures
lander_ll = np.array([[np.degrees(np.arcsin(lander[s][2])), np.degrees(np.arctan2(lander[s][1], lander[s][0]))]
                      for s in STATIONS])
fig, axs = plt.subplots(2, 4, figsize=(20, 7.5))
for col, (wave, depth) in enumerate([(w, z) for z in (100.0, 600.0) for w in ("vp", "vs")]):
    M = MAPS[(wave, depth)]
    lim = float(np.abs(M["mean"]).max()) or 1e-3
    for row, key, cm in ((0, "mean", "RdBu"), (1, "std", "viridis")):
        im = axs[row, col].imshow(M[key], origin="lower", extent=(-180, 180, -90, 90), cmap=cm,
                                  vmin=-lim if key == "mean" else None, vmax=lim if key == "mean" else None)
        axs[row, col].set_title(f"d{wave} at {depth:g} km: posterior {key} (km/s)", fontsize=9)
        plt.colorbar(im, ax=axs[row, col], shrink=0.7)
    axs[0, col].plot(lander_ll[:, 1], lander_ll[:, 0], "^", color="k", ms=6)
fig.suptitle(f"{MODEL} + dv ({VARIANT}): n = {n} picks, stated prior (alpha = 1), chi^2/dof = {chi2 / dof:.2f}; "
             f"per-class noise factors; discrepancy alpha {alpha_dp:.2g}")
fig.savefig(f"{OUT}/maps.png", dpi=130, bbox_inches="tight")

fig, axs = plt.subplots(1, 4, figsize=(23, 5))
colors = {"artificial": "#1b9e77", "meteoroid": "#666666", "shallow": "#d95f02", "deep": "#7570b3"}
for k, col in colors.items():
    for e, v in EVENTS.items():
        if v["kind"] != k:
            continue
        axs[0].plot(v["lon"], v["lat"], "o", ms=3, color=col)
        if e in LOC:                                               # posterior horizontal shift (x5)
            ax = axes(e)
            sh = sum(c[SLOTS[e] + j] * s * a for j, (a, s) in enumerate(ax) if j < 2)
            new = v["pos"] * (R - v["depth"]) + 5 * sh
            la, lo = np.degrees(np.arcsin(new[2] / np.linalg.norm(new))), np.degrees(np.arctan2(new[1], new[0]))
            axs[0].plot([v["lon"], lo], [v["lat"], la], "-", color=col, lw=0.8)
    axs[0].plot([], [], "o", color=col, label=k)
axs[0].plot(lander_ll[:, 1], lander_ll[:, 0], "^", ms=8, color="k", label="ALSEP")
axs[0].set(xlim=(-180, 180), ylim=(-90, 90), title="events, stations and posterior location shifts (x5)")
axs[0].legend(fontsize=7)
axs[1].hist([g_first["r"] / sig_first, g["r"] / sig], bins=40, label=["before the screen", "final"],
            color=["0.6", "#1f78b4"])
axs[1].set(title="residual / sigma at the stated prior", xlabel="r / sigma")
axs[1].legend()
labels = [f"{s[-2:]} {ph}" for ph in "PS" for s in STATIONS]
axs[2].errorbar(np.arange(8), st_mean, yerr=st_std, fmt="o")
axs[2].axhline(0, color="0.5", lw=0.6)
axs[2].set(xticks=np.arange(8), xticklabels=labels, ylabel="s", title="station terms (zero-mean contrasts), mean +- std")
fc = list(FACTORS)
axs[3].bar(np.arange(len(fc)), [FACTORS[k]["final"] for k in fc], color="#1f78b4")
axs[3].axhline(1, color="0.5", lw=0.6)
axs[3].set(xticks=np.arange(len(fc)), ylabel="factor", title="noise factor per class (empirical Bayes, alpha = 1)")
axs[3].set_xticklabels([f"{k}\n(n={FACTORS[k]['n']})" for k in fc], fontsize=7, rotation=30)
fig.savefig(f"{OUT}/data_and_terms.png", dpi=130, bbox_inches="tight")

if check:
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    r0 = check["reference"]["r"]
    r1 = check["posterior"]["r"]
    xs = np.arange(len(weber))
    axs[0].plot(xs, r0, "o", color="0.6", label=f"reference {MODEL}: rms {check['reference']['rms_s']:.1f} s")
    axs[0].plot(xs, r1, "o", color="#1f78b4", label=f"Nunn posterior: rms {check['posterior']['rms_s']:.1f} s")
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels([f"{data[i]['nuis'][6:]} {data[i]['station'][-2:]}" for i in weber], rotation=90, fontsize=7)
    axs[0].axhline(0, color="0.5", lw=0.6)
    axs[0].set(ylabel="observed - predicted, cluster mean removed (s)", title="Weber et al. 2011 S picks (PDS)")
    axs[0].legend(fontsize=8)
    axs[1].hist(check["posterior"]["z"], bins=20, color="#1f78b4")
    axs[1].set(title=f"standardised differences (rms {check['posterior']['rms_sigma']:.2f}, "
                     f"chi^2/dof {check['posterior']['chi2_per_dof']:.2f})", xlabel="z")
    fig.savefig(f"{OUT}/weber_check.png", dpi=130, bbox_inches="tight")
lap("plots")

with open(f"{OUT}/residuals.csv", "w") as fh:
    fh.write("source,event,kind,class,station,phase,taup_phase,delta_deg,observed_s,reference_s,residual_ref_s,spread_s,"
             "sigma_rule,sigma0_s,factor,sigma_s,lg03_minus_nunn_s,used,posterior_residual_s,redundancy\n")
    pos = {i: j for j, i in enumerate(sel)}
    for i, q in enumerate(data):
        j = pos.get(i)
        fh.write(f"{q['source']},{q['event']},{EVENTS[q['event']]['kind']},{q['cls']},{q['station']},{q['phase']},"
                 f"{q['name']},{q['delta']:.3f},{q['obs']:.3f},{q['T']:.3f},{y_all[i]:.3f},{q['spread']:.3f},"
                 f"{q['rule']},{q['sigma0']:.3f},{f[cls[i]]:.3f},{sig0[i] * f[cls[i]]:.3f},{q['lg03_minus_nunn']:.2f},"
                 f"{int(j is not None)},{'' if j is None else round(float(g['r'][j]), 3)},"
                 f"{'' if j is None else round(float(g['rho'][j]), 4)}\n")

claims = [
    "noise: sigma0 = max(Lognonne 2003 bound 1/3/10 s, spread across studies) where LG03 has the pick, else "
    "max(spread, class value of Nunn et al. 2020 p. 26); one factor per class (event class x phase) estimated at the "
    "stated prior from redundancy numbers (variance components), iterated to convergence, at least "
    f"{FACTOR_FLOOR:g} (inflation only); classes of fewer than "
    f"{MIN_CLASS} picks share their event class's factor; gross errors with |standardised residual| > {SCREEN} removed "
    "one at a time, standardised with robust (median) class factors so that gross errors cannot hide by inflating their "
    f"class, then confirmed with the final factors; picks with redundancy below {RHO_MIN} are not tested" + ("; Weber picks sigma0 = 3.7 s with factor >= 1" if VARIANT == "merged" else ""),
    "location prior per event: the spread of its published locations (Garcia et al. 2019 compilation; rule in "
    "prepare_location_priors.py), else the class value (deep 34 km per axis, meteoroid impacts 35 km horizontal, "
    "shallow 50 km horizontal, 50 km depth); linearised dT = -p_src . dx",
    f"artificial impacts at the LRO sites and times (ImpactParameters.csv); free origin time for {FREE_T0}",
    f"CMB topography is prior-only: {n_cmb} CMB terms (no P or S ray of these data touches the CMB in {MODEL}); its "
    f"posterior std {cmb_std:.2f} km equals its prior std {cmb_prior:.2f} km" if Z_CMB is not None else "no CMB",
    "the main posterior is at the stated prior (alpha = 1); the discrepancy alpha is a reported sensitivity",
]
summary = dict(
    variant=VARIANT, model=MODEL, discontinuities_km=JUMPS, layers=list(planet.names), claims=claims,
    picks=dict(nunn=int(nunn.sum()), weber=int((~nunn).sum()), dropped_no_arrival=dropped,
               used=n, used_weber=int(sum(data[i]["source"] == "weber" for i in sel)),
               lg03_coded=int(sum(q["rule"] == "lg03" for q in data if q["source"] == "nunn")),
               removed_by_screen=removed, by_class={k: v["n"] for k, v in FACTORS.items()},
               phases={ph: sum(data[i]["phase"] == ph for i in sel) for ph in "PS"}),
    impact_moves=impact_moves, free_origin_times=E.dim, quotient_dim=dof, segments=int(sum(len(q["w"]) for q in data)),
    noise_factors=FACTORS, factor_floor=FACTOR_FLOOR, unconstrained_factors_summary=FREE, sigma_s=dict(min=float(sig.min()), median=float(np.median(sig)), max=float(sig.max())),
    chi2_per_dof_reference=chi2_ref / dof, chi2_per_dof_at_alpha_1=chi2 / dof, alpha_discrepancy=float(alpha_dp),
    chi2_per_dof_ladder=ladder, cmb_terms=n_cmb, cmb_std_km=cmb_std, cmb_prior_std_km=cmb_prior,
    station_terms_s=dict(zip(labels, np.round(st_mean, 3).tolist())),
    station_terms_std_s=dict(zip(labels, np.round(st_std, 3).tolist())),
    max_abs_mean_km_s={f"{w} {z:g} km": float(np.abs(v["mean"]).max()) for (w, z), v in MAPS.items()},
    min_std_over_prior={f"{w} {z:g} km": float(v["std"].min() / SIG[w]) for (w, z), v in MAPS.items()},
    location_shift_km={k: dict(median=float(np.median([v["shift_km"] for v in LOC.values() if v["kind"] == k])),
                               max=float(max(v["shift_km"] for v in LOC.values() if v["kind"] == k)))
                       for k in ("deep", "shallow", "meteoroid")},
    locations_beyond_prior=beyond, locations=LOC,
    weber_check={k: ({a: b for a, b in v.items() if np.isscalar(b)} if isinstance(v, dict) else v)
                 for k, v in check.items()},
    path_time_check_s=path_check, eps_km=EPS_KM, timings=costs,
    max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20))
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k not in ("locations",)}, indent=1, default=float))
