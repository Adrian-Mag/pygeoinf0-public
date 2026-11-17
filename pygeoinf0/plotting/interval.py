"""Plots of elements. matplotlib is imported only when a plot is asked for."""
import numpy as np

from ..realisations.series import legendre as L


def _axes(ax, figsize=(6.4, 3.4)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_function(f, ax=None, *, given=False, averages=None, show_class=False, label=None, **kw):
    """Plot the canonical representative of f, piece by piece (a jump is drawn as a jump, not as a steep line).

    given=True        plot the representative you supplied instead (f.rep)
    averages=k        also draw the averages of f over k equal cells: these depend only on the class
    show_class=True   also mark a few points where other representatives of the same class differ
    """
    new = ax is None
    ax = _axes(ax)
    pl = f.realisation
    br = pl.breaks
    width = br[-1] - br[0]
    color = kw.pop("color", None)
    lab = f.name if label is None else label
    for i in range(pl.npieces):
        x0, x1 = br[i], br[i + 1]
        m = int(max(24, np.ceil(1200 * (x1 - x0) / width), 6 * pl.coefs[i].shape[0]))
        t = np.linspace(-1.0, 1.0, m)
        x = x0 + 0.5 * (x1 - x0) * (t + 1.0)
        if given:
            y = f.rep(x[1:-1]) if m > 2 else f.rep(x)
            x = x[1:-1] if m > 2 else x
        else:
            y = L.polyval(t, pl.coefs[i])[:, 0]
        line = ax.plot(x, y, color=color, label=lab if i == 0 else None, **kw)
        color = line[0].get_color()
    if averages:
        k = int(averages)
        edges = np.linspace(br[0], br[-1], k + 1)
        avg = np.array([f.integral((edges[j], edges[j + 1])) for j in range(k)])
        mass = np.array([f.space.measure_of_cell(edges[j], edges[j + 1]) for j in range(k)])
        ax.stairs(avg / mass, edges, baseline=None, color="darkorange", linewidth=1.4, label=f"averages over {k} cells")
    if show_class:
        rng = np.random.default_rng(0)
        xs = np.sort(rng.uniform(br[0], br[-1], 9))
        ys = f(xs)
        span = np.ptp(f(np.linspace(br[0], br[-1], 200))) or 1.0
        alt = ys + rng.normal(0.0, 0.35 * span, xs.size)
        ax.scatter(xs, alt, s=26, facecolors="none", edgecolors="crimson", zorder=3,
                   label="values of other representatives (same element)")
        for x_, y0, y1 in zip(xs, ys, alt):
            ax.plot([x_, x_], [y0, y1], color="crimson", alpha=0.25, linewidth=0.8)
    ax.set_xlabel("x")
    if new:
        ax.set_title(f"{f.name} ∈ {f.space}", fontsize=10)
    if averages or show_class:
        ax.legend(fontsize=8)
    return ax


def plot_family(F, ax=None, max_lines=12, **kw):
    ax = _axes(ax)
    n = min(len(F), max_lines)
    idx = np.unique(np.linspace(0, len(F) - 1, n).astype(int))
    for i in idx:
        plot_function(F[i], ax=ax, label=f"{F.name}[{i}]" if n <= 6 else None, linewidth=1.0, **kw)
    ax.set_title(f"{F.name}: {len(idx)} of {len(F)} elements of {F.space}", fontsize=10)
    if n <= 6:
        ax.legend(fontsize=8)
    return ax


def plot_vector(v, ax=None, **kw):
    n = v.space.dim
    if n == 2:
        ax = _axes(ax, figsize=(3.6, 3.6))
        ax.quiver([0], [0], [v.values[0]], [v.values[1]], angles="xy", scale_units="xy", scale=1, **kw)
        r = max(1e-12, float(np.abs(v.values).max())) * 1.2
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.8", linewidth=0.8)
        ax.axvline(0, color="0.8", linewidth=0.8)
    elif n == 1:
        ax = _axes(ax, figsize=(5, 1.4))
        x = float(v.values[0])
        r = max(1.0, abs(x)) * 1.2
        ax.axhline(0, color="0.5", linewidth=1)
        ax.plot([x], [0], "o", **kw)
        ax.annotate(f"{x:.4g}", (x, 0), textcoords="offset points", xytext=(0, 8), ha="center")
        ax.set_xlim(-r, r)
        ax.set_yticks([])
    else:
        ax = _axes(ax)
        ax.stem(np.arange(n), v.values, basefmt="0.7", **kw)
        ax.set_xlabel("component i")
    ax.set_title(f"{v.name} ∈ {v.space}", fontsize=10)
    return ax


def plot_vector_family(V, ax=None, max_lines=8, **kw):
    ax = _axes(ax)
    n = min(len(V), max_lines)
    for i in range(n):
        ax.plot(np.arange(V.space.dim), V.values[i], marker=".", linewidth=0.8, **kw)
    ax.set_xlabel("component i")
    ax.set_title(f"{V.name}: {n} of {len(V)} vectors of {V.space}", fontsize=10)
    return ax
