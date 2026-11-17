"""Plots on curves (values against the natural coordinate: angle on a circle, arc length on segments, polylines,
arcs), on the sphere (a longitude-latitude map) and on flat patches (a map in their own coordinates)."""
import numpy as np


def _check(f, V):
    if not np.isfinite(V).any():
        raise ValueError(f"{f.name} has no finite values: its realisation contains NaN or inf, so an input that built it "
                         "was not finite (for example infinite data weights from a zero noise level)")


def _curve_values(f, n):
    D = f.space.domain
    lo, hi = D.reference[0]
    s = np.linspace(lo, hi, n)
    return s, f.on_grid(s)


def _sphere_values(f, n):
    lon = np.linspace(0, 2 * np.pi, 2 * n)
    colat = np.linspace(0, np.pi, n)
    P, T = np.meshgrid(lon, colat)
    return np.degrees(P), 90 - np.degrees(T), f(*f.space.domain.from_angles(T, P))


def plot_function(f, ax=None, *, n=None, cmap="RdBu_r", symmetric=None, colorbar=True, title=None, **kw):
    import matplotlib.pyplot as plt
    from ..domains import Sphere
    D = f.space.domain
    if D.dim == 1:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3))
        s, V = _curve_values(f, n or 1001)
        _check(f, V)
        ax.plot(s, V, label=kw.pop("label", f.name), **kw)
        ax.set_xlabel(D.internal_names[0]); ax.set_xlim(s[0], s[-1])
        ax.set_title(title or f"{f.name} ∈ {f.space}", fontsize=10)
        return ax
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.8))
    if not isinstance(D, Sphere):                     # a patch: a map in its own coordinates
        (a0, a1), (b0, b1) = D.reference
        s, t = np.linspace(a0, a1, n or 151), np.linspace(b0, b1, n or 151)
        V = f.on_grid(s, t)
        _check(f, V)
        S, Tt = np.meshgrid(s, t, indexing="ij")
        im = ax.pcolormesh(S, Tt, V, shading="auto", cmap=cmap, **kw)
        ax.set_xlabel(D.internal_names[0]); ax.set_ylabel(D.internal_names[1]); ax.set_aspect("equal")
        if colorbar:
            plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title or f"{f.name} ∈ {f.space}", fontsize=10)
        return ax
    X, Y, V = _sphere_values(f, n or 181)
    _check(f, V)
    finite = V[np.isfinite(V)]
    if symmetric is None:
        symmetric = finite.min() < 0 < finite.max()
    vmax = np.abs(finite).max()
    lim = dict(vmin=-vmax, vmax=vmax) if symmetric else {}
    im = ax.pcolormesh(X, Y, V, shading="auto", cmap=cmap, **lim, **kw)
    ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90); ax.set_aspect("equal")
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title or f"{f.name} ∈ {f.space}", fontsize=10)
    return ax


def plot_family(F, ax=None, max_lines=12, **kw):
    import matplotlib.pyplot as plt
    if F.space.domain.dim == 1:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3))
        for i in np.unique(np.linspace(0, len(F) - 1, min(len(F), max_lines)).astype(int)):
            plot_function(F[i], ax=ax, linewidth=0.8, title=f"family {F.name} ({len(F)} members)", **kw)
        return ax
    return plot_function(F[0], ax=ax, **kw)
