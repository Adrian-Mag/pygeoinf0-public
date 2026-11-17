"""Plots of a field on the concentric pieces of a ball (a pg.PiecewiseSpace of one field, e.g. on a LayeredBall's
partition): the plane through the centre with a given normal, the seams and the surface drawn as circles. A grid point on an interface takes the value from side= (default "below": a
plotting choice; the field has two values there). Interface fields are elements of Sobolev spaces on spheres and plot
as maps (manifold.py)."""
import numpy as np


def round_pieces(space):
    """(the domain, its centre, the radii of the seams and of the surface) for a PiecewiseSpace of one field on
    concentric balls and shells, else None."""
    P = getattr(space, "partition", None)
    if P is None or getattr(P, "kind", None) != "round" or not getattr(space, "single", False):
        return None
    D = P.domain
    outer = float(D.radius if hasattr(D, "radius") else D.outer)
    return D, np.asarray(D.centre, float), [float(s.at) for s in P.seams] + [outer]


def section(f, normal=(0.0, 0.0, 1.0), n=241, side="below"):
    """(U, W, values) of the field on the n x n grid of the central plane with the given normal (NaN outside)."""
    V = f.space
    _, centre, radii = round_pieces(V)
    nrm = np.asarray(normal, float) / np.linalg.norm(normal)
    e1 = np.cross(nrm, [1.0, 0.0, 0.0] if abs(nrm[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    R = radii[-1]
    U, W = np.meshgrid(np.linspace(-R, R, n), np.linspace(-R, R, n))
    inside = U ** 2 + W ** 2 <= R ** 2
    pts = centre + U[inside][:, None] * e1 + W[inside][:, None] * e2
    vals = np.full(U.shape, np.nan)
    vals[inside] = V.evaluate(f, pts, side=side)
    return U, W, vals


def plot_function(f, ax=None, *, normal=(0.0, 0.0, 1.0), n=241, side="below", cmap="RdBu_r", **kw):
    import matplotlib.pyplot as plt
    U, W, vals = section(f, normal, n, side)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 5))
    finite = vals[np.isfinite(vals)]
    lim = float(np.abs(finite).max()) if finite.size else 1.0
    im = ax.pcolormesh(U, W, np.ma.masked_invalid(vals), shading="auto", cmap=cmap, vmin=-lim, vmax=lim, **kw)
    for r in round_pieces(f.space)[2]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.6, color="0.3"))
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{f.name}: section normal to {tuple(round(float(c), 3) for c in normal)}", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax


def plot_family(F, ax=None, max_lines=3, **kw):
    return [plot_function(F[int(i)], **kw) for i in np.unique(np.linspace(0, len(F) - 1, min(len(F), max_lines)).astype(int))]
