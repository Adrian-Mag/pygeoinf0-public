"""Plots of sets in R^1, R^2 and R^3 (plan 058). matplotlib is imported only when a plot is asked for.

A Polytope is drawn as itself (its vertices: the exact intersection of all its half-spaces, convex/polytope.py) and
labelled with what it is of its set (exact, outer, inner). Any other set with a support function is drawn as its
outer polytope from `directions` supports (certified upper ends for a feasible set or image), labelled outer: it
contains the set, and shrinks onto it as the directions grow. In R^1 a set is its interval.
"""
import numpy as np


def directions_for(n, k):
    """k unit directions of R^n: +-1 (n = 1), equally spaced angles (n = 2), a Fibonacci sphere (n = 3)."""
    if n == 1:
        return np.array([[1.0], [-1.0]])
    if n == 2:
        a = 2 * np.pi * np.arange(k) / k
        return np.c_[np.cos(a), np.sin(a)]
    i = np.arange(k) + 0.5
    z = 1 - 2 * i / k
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def as_polytope(S, directions=180):
    """S itself if a Polytope, else its outer Polytope from `directions` (a count or rows of directions)."""
    from ..convex.sets import Polytope
    if isinstance(S, Polytope):
        return S
    Q = directions_for(S.space.dim, int(directions)) if np.ndim(directions) == 0 else directions
    return S.polytope(Q)


def _label(P, label):
    if label is not None:
        return label
    return P.name if P.approx == "exact" else f"{P.of.name} ({P.approx})"


def plot_set(S, ax=None, directions=180, label=None, **kw):
    """Draw S (module docstring); kw go to matplotlib (fill / bar / Poly3DCollection). Returns the axes."""
    import matplotlib.pyplot as plt
    n = S.space.dim
    if n not in (1, 2, 3):
        raise ValueError(f"sets in R^{n} are not drawn: project onto one, two or three properties first")
    P = as_polytope(S, directions)
    V = P.vertices()
    lab = _label(P, label)
    kw.setdefault("alpha", 0.35)
    if n == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from scipy.spatial import ConvexHull
        if ax is None:
            ax = plt.figure().add_subplot(projection="3d")
        H = ConvexHull(V)
        ax.add_collection3d(Poly3DCollection([V[s] for s in H.simplices], label=lab, **kw))
        lo, hi = V.min(axis=0), V.max(axis=0)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
        return ax
    if ax is None:
        _, ax = plt.subplots()
    if n == 1:
        a, b = float(V[0, 0]), float(V[1, 0])
        ax.barh(0.0, b - a, left=a, height=0.4, label=lab, **kw)
        ax.set_yticks([])
        return ax
    ax.fill(V[:, 0], V[:, 1], label=lab, **kw)
    ax.plot(np.r_[V[:, 0], V[0, 0]], np.r_[V[:, 1], V[0, 1]], color=kw.get("color", "k"), lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    return ax
