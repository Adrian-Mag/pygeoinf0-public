"""Where a fit sampled its callable, and a check of a fit at a declared resolution (plan 054, narrow bumps, option C; the
statements are in core/fits.py).

    Recorder(fun, backend, family)   the callable, unchanged, remembering the points the fit sampled (natural coordinates)
    spacing(rec, backend, breaks, N) per column, the largest spacing s of its samples along a natural axis, in lengths
    lattice_misses(...)              where a fit differs from its callable on a lattice of spacing w / 2
    with_breaks(breaks, cuts, d)     breaks with cuts added per natural axis, keeping NaturalBreaks' hints
    localised(values)                whether sampled members (columns) are negligible on most of the domain (moved from
                                     realisations/backends._localised, plan 069 P2: core/fits.py asks it)

Lengths: a natural coordinate's spacing times the largest |d x / d xi_k| over the domain (an angle times the largest
radius it is taken at), measured once per domain by differences of to_cartesian. A call of the fit is a product grid in
natural coordinates (a tensor, harmonic or radial grid; on an interval any set of points) or it is not used: calls on
scattered points (cell rules) state nothing here, and their backend states its own (Recorder.stated).
"""
import math

import numpy as np

_CHARTS = {}
MISS = 1e-8                     # a lattice point is missed above MISS x max|f| (legendre's scout criterion)
CHUNK = 1 << 21                 # lattice points per evaluation


def _interval(D):
    return hasattr(D, "a") and hasattr(D, "b") and not getattr(D, "reference", ())


def chart(B, breaks=None):
    """(ranges, to_natural, to_cartesian, scales, periodic) of the backend's domain, or None if it has no chart. ranges:
    the backend's sampling_box(breaks) (a declared support) or the reference box."""
    D = B.domain
    if _interval(D):
        ident = lambda *x: x                                            # noqa: E731
        return [(float(D.a), float(D.b))], ident, ident, [1.0], [False]
    ref = tuple(getattr(D, "reference", ()) or ())
    if not ref or not hasattr(D, "to_cartesian") or not hasattr(D, "from_cartesian"):
        return None
    key = id(D)
    if key not in _CHARTS or _CHARTS[key][0] is not D:
        _CHARTS[key] = (D, _metric(D, ref))
    scales, periodic = _CHARTS[key][1]
    if scales is None:
        return None
    box = getattr(B, "sampling_box", lambda b: None)(breaks)
    ranges = [tuple(map(float, r)) for r in (box or ref)]
    return ranges, D.from_cartesian, D.to_cartesian, scales, periodic


def _metric(D, ref):
    """(largest |d x / d xi_k| per axis, periodic per axis), by central differences on a 5^d grid of the reference box."""
    d = len(ref)
    try:
        g = [lo + (hi - lo) * np.array([0.0, 0.25, 0.5, 0.75, 1.0]) for lo, hi in ref]
        P = [a.ravel() for a in np.meshgrid(*g, indexing="ij")]
        scales, periodic = [], []
        for k, (lo, hi) in enumerate(ref):
            h = 1e-6 * (hi - lo)
            up = [p if j != k else np.minimum(p + h, hi) for j, p in enumerate(P)]
            dn = [p if j != k else np.maximum(p - h, lo) for j, p in enumerate(P)]
            Xu, Xd = (np.array(D.to_cartesian(*q), float) for q in (up, dn))
            dist = np.sqrt(((Xu - Xd) ** 2).sum(axis=0)) / (up[k] - dn[k])
            scales.append(float(np.nanmax(dist)))
            a = [p if j != k else np.full_like(p, lo) for j, p in enumerate(P)]
            b = [p if j != k else np.full_like(p, hi) for j, p in enumerate(P)]
            Xa, Xb = (np.array(D.to_cartesian(*q), float) for q in (a, b))
            periodic.append(bool(np.allclose(Xa, Xb, atol=1e-12 * max(1.0, float(np.abs(Xa).max())))))
        return (scales if all(math.isfinite(s) and s > 0 for s in scales) else None), periodic
    except Exception:                                                    # noqa: BLE001  (no chart: nothing stated)
        return None, [False] * d


class Recorder:
    """fun, unchanged, remembering the natural-coordinate nodes of each call that is a product grid (per natural axis),
    grouped by the columns it was asked for (families) and by its nodes on the other axes. Recording never changes a
    fit: a call that cannot be read is not recorded."""

    def __init__(self, fun, B, family=False):
        self.fun, self.family, self.groups, self.stated = fun, family, {}, None
        self.chart = chart(B)

    def __call__(self, *args):
        if self.chart is not None:
            try:
                self._note(args[:-1] if self.family else args, args[-1] if self.family else None)
            except Exception:                                            # noqa: BLE001
                pass
        return self.fun(*args)

    def __getattr__(self, name):                                        # a bank function's attributes stay visible
        fun = self.__dict__.get("fun")
        if fun is None:
            raise AttributeError(name)
        return getattr(fun, name)

    def _note(self, X, idx):
        ranges, to_nat, _, _, periodic = self.chart
        d = len(ranges)
        X = [np.asarray(x, float) for x in X]
        S = np.broadcast_shapes(*[x.shape for x in X])
        if len(S) != d or min(S, default=0) == 0:
            return
        nodes = []
        for k in range(d):
            first = tuple(slice(None) if j == k else 0 for j in range(d))
            last = tuple(slice(None) if j == k else -1 for j in range(d))
            a = [np.broadcast_to(x, S)[first] for x in X]
            b = [np.broadcast_to(x, S)[last] for x in X]
            na = [_wrap(np.asarray(t, float).ravel(), r, p) for t, r, p in zip(to_nat(*a), ranges, periodic)]
            nb = [_wrap(np.asarray(t, float).ravel(), r, p) for t, r, p in zip(to_nat(*b), ranges, periodic)]
            for j in range(d):
                if j != k and np.ptp(na[j]) > 1e-9 * (ranges[j][1] - ranges[j][0]):
                    return
            if not np.allclose(na[k], nb[k], atol=1e-9 * (ranges[k][1] - ranges[k][0])):
                return
            nodes.append(na[k])
        cols = None if idx is None else np.atleast_1d(np.asarray(idx)).astype(int)
        ck = None if cols is None else (int(cols[0]), int(cols[-1]), len(cols), int(cols.sum()))
        others = tuple(np.round(t, 13).tobytes() for t in nodes[1:])
        g = self.groups.setdefault((ck, others), [cols, [], nodes[1:]])
        g[1].append(nodes[0])


def _wrap(t, rng, periodic):
    lo, hi = rng
    t = np.asarray(t, float)
    return lo + np.mod(t - lo, hi - lo) if periodic else t


def _gap(t, lo, hi, periodic):
    t = np.unique(np.clip(t[np.isfinite(t)], lo, hi))
    if t.size == 0:
        return hi - lo
    inner = float(np.diff(t).max()) if t.size > 1 else 0.0
    a, b = float(t[0] - lo), float(hi - t[-1])
    return max(inner, a + b) if periodic else max(inner, a, b)


def spacing(rec, B, breaks, N):
    """Per column, the largest spacing s (in lengths) of the samples the fit took along any natural axis: the least
    over the recorded groups that hold the column (a group's own grid; their union is at least as fine). inf: nothing
    readable was recorded (s is then not known). A backend that samples on scattered points states its own
    (rec.stated: cells' rules)."""
    if rec is not None and rec.__dict__.get("stated") is not None:            # a backend's own statement (cells)
        return np.full(N, float(rec.stated))
    out = np.full(N, np.inf)
    ch = chart(B, breaks)
    if ch is None or rec is None:
        return out
    ranges, _, _, scales, periodic = ch
    for cols, firsts, rest in rec.groups.values():
        axes = [np.concatenate(firsts)] + list(rest)
        s = max(_gap(t, lo, hi, p) * sc for t, (lo, hi), sc, p in zip(axes, ranges, scales, periodic))
        if cols is None:
            out = np.minimum(out, s)
        else:
            c = cols[(cols >= 0) & (cols < N)]
            out[c] = np.minimum(out[c], s)
    return out


def lattice_misses(B, fun, R, N, w, breaks, family=False):
    """(points, cuts): the lattice of spacing <= w / 2 (lengths) on each natural axis, and per axis the coordinates
    where the realisation R differs from fun by more than MISS x max|f| (cuts: one lattice cell beyond each run of
    missed nodes; around the worst one where most of an axis is missed), or None for an axis with no miss. None if the
    domain has no chart."""
    ch = chart(B, breaks)
    if ch is None:
        return None
    ranges, _, to_cart, scales, _ = ch
    axes = [np.linspace(lo, hi, int(math.ceil((hi - lo) * sc / (0.5 * w))) + 1) for (lo, hi), sc in zip(ranges, scales)]
    d = len(axes)
    worst = [np.zeros(len(a)) for a in axes]
    vmax = 0.0
    rows = max(1, CHUNK // max(1, int(np.prod([len(a) for a in axes[1:]])) * max(1, N)))
    cols = np.arange(N)
    for s in range(0, len(axes[0]), rows):
        part = [axes[0][s:s + rows]] + axes[1:]
        G = np.meshgrid(*part, indexing="ij", sparse=True)
        X = to_cart(*G)
        with np.errstate(all="ignore"):
            V = np.asarray(fun(*X, cols) if family else fun(*X), float)
        V = np.broadcast_to(V, tuple(len(p) for p in part) + ((N,) if V.ndim == d + 1 or N > 1 else ())).reshape(
            tuple(len(p) for p in part) + (N,))
        F = np.asarray(B.grid_values(R, part), float).reshape(V.shape)
        ok = np.isfinite(V) & np.isfinite(F)
        vmax = max(vmax, float(np.abs(np.where(ok, V, 0.0)).max(initial=0.0)),
                   float(np.abs(np.where(ok, F, 0.0)).max(initial=0.0)))
        diff = np.where(ok, np.abs(V - F), 0.0).max(axis=-1)
        for k in range(d):
            m = diff.max(axis=tuple(j for j in range(d) if j != k)) if d > 1 else diff
            if k == 0:
                worst[0][s:s + len(m)] = np.maximum(worst[0][s:s + len(m)], m)
            else:
                worst[k] = np.maximum(worst[k], m)
    thr = max(MISS, 10 * float(getattr(B, "tol", 0.0))) * max(vmax, 1e-300)
    cuts = []
    for a, m in zip(axes, worst):
        bad = np.nonzero(m > thr)[0]
        if not len(bad):
            cuts.append(None)
            continue
        if len(bad) > len(a) // 2:                    # missed along most of the axis: cut around the worst miss
            j = int(np.argmax(m))
            bad = np.arange(max(j - 1, 0), min(j + 2, len(a)))
        runs = np.split(bad, np.nonzero(np.diff(bad) > 1)[0] + 1)
        cuts.append(sorted({float(a[max(r[0] - 1, 0)]) for r in runs} | {float(a[min(r[-1] + 1, len(a) - 1)]) for r in runs}))
    return axes, cuts


def with_breaks(breaks, cuts, d):
    """breaks (a list on an interval, one entry per natural axis otherwise; NaturalBreaks keep their hints) with the cuts
    added."""
    if d == 1 and (breaks is None or np.ndim(breaks) == 1 and not isinstance(breaks, tuple) or np.isscalar(breaks)):
        old = [] if breaks is None else list(np.ravel(breaks))
        return sorted(set(old) | set(cuts[0] or ()))
    old = list(breaks) if breaks is not None else [None] * d
    axes = tuple(None if (o is None and c is None) else sorted(set(np.ravel(o if o is not None else []).tolist())
                                                               | set(c or ())) for o, c in zip(old, cuts))
    if hasattr(breaks, "__dict__"):
        out = type(breaks).__new__(type(breaks), axes)
        out.__dict__.update(breaks.__dict__)
        return out
    return axes


def localised(values):
    """True when sampled members (columns of values) are negligible (below 1e-12 of their maximum) on most of the
    domain: on average live at fewer than 35 % of the samples (the choice of a sparse family)."""
    v = np.abs(np.asarray(values, float))
    live = (v > 1e-12 * np.maximum(v.max(axis=0), 1e-300)).mean(axis=0)
    return bool(live.mean() < 0.35)

