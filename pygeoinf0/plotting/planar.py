"""Plots of elements on 2-D domains (Box, Disk): a colour map of the canonical representative, NaN outside."""
import numpy as np


def _grid(dom, n):
    (x0, x1), (y0, y1) = dom.bounding_box
    pad = 0.02 * max(x1 - x0, y1 - y0)
    x = np.linspace(x0 - pad, x1 + pad, n)
    y = np.linspace(y0 - pad, y1 + pad, int(n * (y1 - y0 + 2 * pad) / (x1 - x0 + 2 * pad)) + 1)
    return np.meshgrid(x, y, indexing="xy")


def _boundary(ax, dom):
    from ..domains import Disk
    if isinstance(dom, Disk):
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(dom.centre[0] + dom.radius * np.cos(t), dom.centre[1] + dom.radius * np.sin(t), color="0.3", lw=0.8)
    else:
        (x0, x1), (y0, y1) = dom.bounding_box
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="0.3", lw=0.8)


def plot_function(f, ax=None, *, n=300, cmap="RdBu_r", symmetric=None, contours=0, colorbar=True, title=None, **kw):
    import matplotlib.pyplot as plt
    new = ax is None
    if new:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    dom = f.space.domain
    X, Y = _grid(dom, n)
    V = f(X, Y)
    finite = V[np.isfinite(V)]
    if finite.size == 0:
        raise ValueError(f"{f.name} has no finite values on {dom}: its realisation contains NaN or inf, so an input that "
                         "built it was not finite (for example infinite data weights from a zero noise level)")
    if symmetric is None:
        symmetric = finite.size > 0 and finite.min() < 0 < finite.max()
    vmax = np.abs(finite).max() if finite.size else 1.0
    lim = dict(vmin=-vmax, vmax=vmax) if symmetric else {}
    im = ax.pcolormesh(X, Y, np.ma.masked_invalid(V), shading="auto", cmap=cmap, **lim, **kw)
    if contours:
        ax.contour(X, Y, np.ma.masked_invalid(V), levels=contours, colors="k", linewidths=0.4, alpha=0.6)
    _boundary(ax, dom)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title or f"{f.name} ∈ {f.space}", fontsize=10)
    return ax


def plot_family(F, ax=None, max_lines=6, **kw):
    import matplotlib.pyplot as plt
    k = min(len(F), max_lines)
    idx = np.unique(np.linspace(0, len(F) - 1, k).astype(int))
    fig, axs = plt.subplots(1, len(idx), figsize=(3.2 * len(idx), 3.0), squeeze=False)
    for a, i in zip(axs[0], idx):
        plot_function(F[i], ax=a, colorbar=False, title=f"{F.name}[{i}]", **kw)
    fig.tight_layout()
    return axs[0]
