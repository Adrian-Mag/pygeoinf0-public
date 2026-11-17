"""Plots of elements on 3-D domains (Box, Cylinder, Ball).

Static (matplotlib): three orthogonal slices through a point (the centre by default), one colour scale.
Interactive (plotly, interactive=True): a slice plane you move with a slider along x, y or z, like scrolling
through a CT scan; it also works in exported HTML.
"""
import numpy as np


def _centre(dom):
    return tuple(0.5 * (lo + hi) for lo, hi in dom.bounding_box)


def _slice(f, axis, value, n):
    bb = f.space.domain.bounding_box
    others = [k for k in range(3) if k != axis]
    u = np.linspace(*bb[others[0]], n)
    v = np.linspace(*bb[others[1]], int(n * (bb[others[1]][1] - bb[others[1]][0]) / (bb[others[0]][1] - bb[others[0]][0])) + 1)
    U, V = np.meshgrid(u, v, indexing="xy")
    P = [None, None, None]
    P[axis] = np.full(U.shape, value)
    P[others[0]], P[others[1]] = U, V
    return U, V, f(*P), others


def plot_function(f, ax=None, *, at=None, n=160, cmap="RdBu_r", interactive=False, axis="z", **kw):
    if interactive:
        return interactive_slices(f, axis=axis, n=min(n, 64), cmap=cmap)
    import matplotlib.pyplot as plt
    dom = f.space.domain
    c = _centre(dom) if at is None else tuple(at)
    slices = [_slice(f, k, c[k], n) for k in range(3)]
    vals = np.concatenate([s[2][np.isfinite(s[2])] for s in slices])
    if vals.size == 0:
        raise ValueError(f"{f.name} has no finite values on {dom}: its realisation contains NaN or inf, so an input that "
                         "built it was not finite (for example infinite data weights from a zero noise level)")
    vmax = np.abs(vals).max() if vals.size else 1.0
    sym = vals.size and vals.min() < 0 < vals.max()
    lim = dict(vmin=-vmax, vmax=vmax) if sym else dict(vmin=vals.min(), vmax=vals.max())
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    names = "xyz"
    for a, (U, V, W, others), k in zip(axs, slices, range(3)):
        im = a.pcolormesh(U, V, np.ma.masked_invalid(W), shading="auto", cmap=cmap, **lim, **kw)
        a.set_aspect("equal")
        a.set_xlabel(names[others[0]])
        a.set_ylabel(names[others[1]])
        a.set_title(f"{names[k]} = {c[k]:.3g}", fontsize=10)
    fig.colorbar(im, ax=axs, shrink=0.85)
    fig.suptitle(f"{f.name} ∈ {f.space}", fontsize=11)
    return axs


def interactive_slices(f, axis="z", n=64, steps=17, cmap="RdBu_r"):
    """A plotly figure: a heat map of f on the plane {axis = value}, with a slider over value."""
    import plotly.graph_objects as go
    k = "xyz".index(axis)
    bb = f.space.domain.bounding_box
    values = np.linspace(bb[k][0], bb[k][1], steps)
    data = [_slice(f, k, v, n) for v in values]
    finite = np.concatenate([d[2][np.isfinite(d[2])] for d in data])
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    U, V, _, others = data[0]
    names = "xyz"
    rnd = lambda A: np.where(np.isfinite(A), np.round(A, 4 - int(np.floor(np.log10(max(vmax, 1e-300))))), np.nan)
    frames = [go.Frame(data=[go.Heatmap(x=np.round(U[0], 4), y=np.round(V[:, 0], 4), z=rnd(d[2]), zmin=-vmax, zmax=vmax,
                                        colorscale=cmap)],
                       name=f"{v:.3g}") for v, d in zip(values, data)]
    fig = go.Figure(data=frames[steps // 2].data, frames=frames)
    fig.update_layout(
        title=f"{f.name}: slices {axis} = const", width=560, height=560,
        xaxis=dict(title=names[others[0]], scaleanchor="y"), yaxis=dict(title=names[others[1]]),
        sliders=[dict(active=steps // 2, currentvalue=dict(prefix=f"{axis} = "),
                      steps=[dict(method="animate", label=fr.name,
                                  args=[[fr.name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                         transition=dict(duration=0))]) for fr in frames])])
    return fig


def plot_family(F, ax=None, max_lines=3, **kw):
    out = []
    for i in np.unique(np.linspace(0, len(F) - 1, min(len(F), max_lines)).astype(int)):
        out.append(plot_function(F[i], **kw))
    return out
